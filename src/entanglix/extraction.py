"""Peak extraction from spectroscopy maps under declared settings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths

from entanglix.measurements import Dataset
from entanglix.models import StrictModel

# The median absolute deviation of Gaussian noise times this factor is its standard deviation.
MAD_SCALE = 1.4826
UNIFORM_RTOL = 1e-6


class ExtractionError(ValueError):
    """A map cannot be extracted under the declared settings."""


class ExtractionSettings(StrictModel):
    background_window_ghz: float = Field(gt=0)
    threshold: float = Field(gt=0)
    min_separation_ghz: float = Field(gt=0)
    fit_half_window_ghz: float = Field(gt=0)
    linewidth_bounds_ghz: list[float] = Field(min_length=2, max_length=2)
    line_shape: Literal["lorentzian", "gaussian"]

    @model_validator(mode="after")
    def check_values(self) -> ExtractionSettings:
        values = [
            self.background_window_ghz,
            self.threshold,
            self.min_separation_ghz,
            self.fit_half_window_ghz,
            *self.linewidth_bounds_ghz,
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("extraction settings must be finite")
        lower, upper = self.linewidth_bounds_ghz
        if not 0 < lower < upper:
            raise ValueError("linewidth_bounds_ghz requires 0 < lower < upper")
        return self


@dataclass(frozen=True)
class Extraction:
    """Extracted peaks by column, failures, the projection used, and each column's noise."""

    projection: tuple[float, float]
    points: dict[str, np.ndarray]
    failures: tuple[dict, ...]
    noise: np.ndarray


def _shape(kind: str, detuning: np.ndarray, width: float) -> np.ndarray:
    """Unit-height line with half width at half maximum ``width``."""
    if kind == "lorentzian":
        return width**2 / (detuning**2 + width**2)
    return np.exp(-math.log(2) * detuning**2 / width**2)


def frequency_step(dataset: Dataset) -> float:
    """The map's frequency spacing; extraction needs a uniform grid of at least 3 points."""
    if dataset.spec.map is None:
        raise ExtractionError(f"dataset {dataset.spec.id!r} has no map to extract")
    frequency = dataset.arrays["map.frequency"]
    spacing = np.diff(frequency)
    step = float(spacing.mean()) if spacing.size else math.nan
    if frequency.size < 3 or np.max(np.abs(spacing - step)) > UNIFORM_RTOL * step:
        raise ExtractionError("extraction needs a uniform frequency grid of at least 3 points")
    return step


def _residual(dataset: Dataset, settings: ExtractionSettings):
    """Remove each column's running-median background along frequency."""
    step = frequency_step(dataset)
    frequency = dataset.arrays["map.frequency"]
    signal = np.atleast_2d(dataset.arrays["map.signal"]).astype(np.complex128)
    size = (1, max(3, 2 * round(settings.background_window_ghz / (2 * step)) + 1))
    background = median_filter(signal.real, size=size, mode="nearest")
    background = background + 1j * median_filter(signal.imag, size=size, mode="nearest")
    return frequency, step, signal - background


def _principal(residual: np.ndarray) -> tuple[float, float]:
    pairs = np.stack([residual.real.ravel(), residual.imag.ravel()])
    axis = np.linalg.eigh(pairs @ pairs.T)[1][:, -1]
    projected = residual.real * axis[0] + residual.imag * axis[1]
    if abs(projected.min()) > projected.max():
        axis = -axis
    return float(axis[0]), float(axis[1])


def learn_projection(dataset: Dataset, settings: ExtractionSettings) -> tuple[float, float]:
    """Learn the I/Q direction of a map's lines, signed so that lines are positive."""
    return _principal(_residual(dataset, settings)[2])


def _groups(positions: np.ndarray, separation: float) -> list[list[int]]:
    groups = [[0]]
    for index in range(1, positions.size):
        if positions[index] - positions[index - 1] < separation:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _fit(row, frequency, found, starts, target, half, noise, settings) -> dict:
    """Fit the target line with every detected line in its window and a constant offset.

    A fit that cannot report the centre's uncertainty returns only a failure ``detail``.
    """
    low, high = max(found[target] - half, 0), min(found[target] + half + 1, row.size)
    members = [index for index, pixel in enumerate(found) if low <= pixel < high]
    window, values = frequency[low:high], row[low:high]
    narrow, wide = settings.linewidth_bounds_ghz
    start, lower, upper = [], [], []
    for index in members:
        start += [frequency[found[index]], starts[index], row[found[index]]]
        lower += [window[0], narrow, 0.0]
        upper += [window[-1], wide, np.inf]

    def residual(q):
        model = np.full(window.size, q[-1])
        for centre, width, amplitude in q[:-1].reshape(-1, 3):
            model += amplitude * _shape(settings.line_shape, window - centre, width)
        return (model - values) / noise

    result = least_squares(residual, start + [0.0], bounds=(lower + [-np.inf], upper + [np.inf]))
    # Rank and covariance come from the Jacobian itself, with its columns scaled to unit norm;
    # a pseudoinverse would drop undetermined directions and report finite uncertainties.
    norms = np.linalg.norm(result.jac, axis=0)
    _, singular, vt = np.linalg.svd(result.jac / np.where(norms > 0, norms, 1), full_matrices=False)
    tolerance = singular[0] * max(result.jac.shape) * np.finfo(float).eps
    if np.count_nonzero(singular > tolerance) < result.jac.shape[1]:
        return {"detail": "rank-deficient: the window does not determine every line parameter"}
    if not result.success:
        return {"detail": "the optimizer did not converge"}
    j = 3 * members.index(target)
    sigma = math.sqrt(((vt.T / singular**2) @ vt)[j, j]) / norms[j]
    if not math.isfinite(sigma) or sigma <= 0:
        return {"detail": "non-finite centre uncertainty"}
    centre, width, amplitude = (float(value) for value in result.x[j : j + 3])
    return {
        "centre": centre,
        "centre_uncertainty": sigma,
        "linewidth": width,
        "amplitude": amplitude,
    }


def extract(
    dataset: Dataset,
    settings: ExtractionSettings,
    projection: tuple[float, float] | None = None,
) -> Extraction:
    """Extract each column's peaks; without a projection, learn it from this map."""
    frequency, step, residual = _residual(dataset, settings)
    if projection is None:
        projection = _principal(residual)
    elif len(projection) != 2 or not math.isclose(math.hypot(*projection), 1, abs_tol=1e-9):
        raise ExtractionError("projection must be a unit I/Q direction")
    projected = residual.real * projection[0] + residual.imag * projection[1]
    bias = dataset.arrays.get("map.bias")
    half = max(2, round(settings.fit_half_window_ghz / step))
    narrow, wide = settings.linewidth_bounds_ghz
    rows = {
        name: [] for name in ("column", "centre", "centre_uncertainty", "linewidth", "amplitude")
    }
    failures, noises = [], []
    for column, row in enumerate(projected):
        location = (
            {"column": column} if bias is None else {"column": column, "bias": float(bias[column])}
        )
        noise = MAD_SCALE * float(np.median(np.abs(row - np.median(row))))
        noises.append(noise)
        if not noise > 0:
            failures.append({**location, "reason": "no noise estimate"})
            continue
        level = settings.threshold * noise
        found, _ = find_peaks(row, height=level, prominence=level)
        if not found.size:
            failures.append({**location, "reason": "no peak"})
            continue
        starts = np.clip(peak_widths(row, found, rel_height=0.5)[0] * step / 2, narrow, wide)
        positions = frequency[found]
        for group in _groups(positions, settings.min_separation_ghz):
            if len(group) > 1:
                # Maxima closer than the declared separation cannot be told apart.
                failures.append(
                    {
                        **location,
                        "reason": "unresolved overlap",
                        "frequencies_ghz": [float(positions[index]) for index in group],
                    }
                )
                continue
            peak = _fit(row, frequency, found, starts, group[0], half, noise, settings)
            if "detail" in peak:
                failures.append(
                    {
                        **location,
                        "reason": "line fit failed",
                        "frequency_ghz": float(positions[group[0]]),
                        "detail": peak["detail"],
                    }
                )
                continue
            rows["column"].append(column)
            for name, value in peak.items():
                rows[name].append(value)
    points = {
        name: np.asarray(values, dtype=np.int64 if name == "column" else np.float64)
        for name, values in rows.items()
    }
    if bias is not None:
        points["bias"] = bias[points["column"]]
    return Extraction(projection, points, tuple(failures), np.asarray(noises))
