"""Synthetic round trip for Block 10 spectroscopy calibration; CPU only, no runner jobs.

Two-tone maps of a two-junction transmon are generated from known parameters. Peaks are
extracted with declared, frozen settings, matched one to one with a declared candidate family
and fitted by bounded weighted least squares; one case is evaluated on a withheld bias range.
Map cases first fit each column's strongest peak as a declared anchor transition.
The forward model is a charge-basis NumPy Hamiltonian; the results record its agreement with
``entanglix.models.load_system`` on the same circuit. Stage 1 is a small slice that measures
cost and runs every other case once; stage 2 is the declared full run. The tracked results keep
settings, provenance, summaries, failure counts and representative failures; ``--full-output``
writes every realisation elsewhere.

OPENBLAS_NUM_THREADS=1 .venv-runner/bin/python tests/probes/s2_calibration.py --stage 2 \
  --output tests/probes/s2_calibration_results.json \
  --full-output ~/Projects/entanglix-research/s2-probe/stage2_full_results.json
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.constants import e, h
from scipy.ndimage import median_filter
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.signal import find_peaks

E2H = e**2 / (2 * h * 1e-15) / 1e9  # GHz·fF, as Entanglix converts capacitance
ECJ = 4.0
NCUT = 20
FREQ = np.round(np.arange(2.0, 7.0, 0.001), 6)  # GHz, 1 MHz pixels
HWHM = 0.002
NOISE = 0.02  # per quadrature, against an f01 amplitude of 1
BACKGROUND = (0.5 + 0.3j, 0.02)  # constant and slope per GHz
PHASE = 0.7
VISIBLE = {(0, 1, 1): 1.0, (1, 2, 1): 0.3, (0, 2, 2): 0.2}
FAMILY = tuple((i, f, k) for i in (0, 1) for f in (1, 2, 3) if f > i for k in (1, 2))
F01 = ((0, 1, 1),)
ANCHOR = (0, 1, 1)  # declared: each column's strongest peak is this transition
EXTRACTION = {
    "median_window_px": 101,
    "threshold_noise": 6.0,
    "min_separation_px": 10,
    "fit_half_window_px": 20,
    "hwhm_range_ghz": [2e-4, 0.02],
}
MATCHING = {
    "phase1_range_ghz": 0.3,
    "cutoff_sigma": 5.0,
    "resolution_linewidths": 2.0,
    "phase1_limit": 20,
    "phase2_limit": 20,
}
RANK_RTOL, NULL_TOL, MERGE_TOL, STEP = 1e-6, 1e-3, 1e-3, 1e-5
COUNTS = {
    1: {"A": (10, 4), "B": (5, 1), "C": (1, 1), "D": (1, 1), "E": (1, 2)},
    2: {"A": (100, 4), "B": (20, 1), "C": (5, 2), "D": (10, 4), "E": (20, 8)},
}
SPEC = """format_version = 1
id = "squid_transmon"
[units]
energy = "GHz"
time = "ns"
[model]
kind = "circuit"
nodes = [0, 1]
ground = 0
offset_charges = { "1" = 0.0 }
[[model.branches]]
id = "J1"
kind = "JJ"
nodes = [0, 1]
josephson_energy = { value = 12.0, unit = "GHz" }
charging_energy = { value = 4.0, unit = "GHz" }
[[model.branches]]
id = "J2"
kind = "JJ"
nodes = [0, 1]
josephson_energy = { value = 8.0, unit = "GHz" }
charging_energy = { value = 4.0, unit = "GHz" }
[[model.branches]]
id = "shunt"
kind = "C"
nodes = [0, 1]
capacitance = { value = 70.0, unit = "fF" }
[[model.external_fluxes]]
branch = "J2"
value = FLUX
unit = "Phi0"
[truncation]
cutoffs = { theta1 = 30 }
retained_levels = 6
"""


def charge_levels(ej, ec, levels=4):
    n = np.arange(-NCUT, NCUT + 1)
    index = np.arange(n.size)
    matrix = np.zeros((ej.size, n.size, n.size))
    matrix[:, index, index] = 4 * ec * n**2
    matrix[:, index[:-1], index[1:]] = matrix[:, index[1:], index[:-1]] = -ej[:, None] / 2
    values = np.linalg.eigvalsh(matrix)[:, :levels]
    return values - values[:, :1]


class Squid:
    names = ("EJ1", "EJ2", "Csh", "V0", "P")
    truth = np.array([12.0, 8.0, 70.0, 0.1, 1.0])
    scales = np.array([1.0, 1.0, 10.0, 0.01, 0.01])
    lower = np.array([2.0, 2.0, 30.0, -0.5, 0.5])
    upper = np.array([25.0, 25.0, 120.0, 0.5, 2.0])

    @staticmethod
    def levels(theta, bias):
        ej1, ej2, csh, offset, period = theta
        flux = (bias - offset) / period
        ej = np.sqrt(np.maximum(ej1**2 + ej2**2 + 2 * ej1 * ej2 * np.cos(2 * np.pi * flux), 0))
        return charge_levels(ej, E2H / (2 * E2H / ECJ + csh))

    @staticmethod
    def conventions(theta):
        theta, applied = theta.copy(), []
        if theta[0] < theta[1]:
            theta[[0, 1]] = theta[[1, 0]]
            applied.append("junction exchange")
        wrapped = (theta[3] + theta[4] / 2) % theta[4] - theta[4] / 2
        if not np.isclose(wrapped, theta[3]):
            theta[3] = wrapped
            applied.append("offset alias")
        return theta, applied


class Transmon:
    names = ("EJ", "EC")
    truth = np.array([20.0, 0.243])
    scales = np.array([1.0, 0.01])
    lower = np.array([5.0, 0.1])
    upper = np.array([40.0, 0.5])

    @staticmethod
    def levels(theta, bias):
        return charge_levels(np.full(bias.size, theta[0]), theta[1])

    @staticmethod
    def conventions(theta):
        return theta.copy(), []


def predict(model, theta, bias, family):
    levels = model.levels(theta, bias)
    return np.stack([(levels[:, f] - levels[:, i]) / k for i, f, k in family], axis=1)


def profile(detuning, shape):
    if shape == "lorentzian":
        return HWHM**2 / (detuning**2 + HWHM**2)
    width = 2 * HWHM / np.sqrt(8 * np.log(2))  # same full width at half maximum
    return np.exp(-(detuning**2) / (2 * width**2))


def make_map(rng, bias, visible=VISIBLE, shape="lorentzian", extra=()):
    centres = predict(Squid, Squid.truth, bias, tuple(visible))
    clean = np.zeros((bias.size, FREQ.size))
    for column in range(bias.size):
        for centre, amplitude in zip(centres[column], visible.values(), strict=True):
            clean[column] += amplitude * profile(FREQ - centre, shape)
        for frequency, amplitude in extra:
            clean[column] += amplitude * profile(FREQ - frequency, shape)
    background = BACKGROUND[0] + BACKGROUND[1] * (FREQ - FREQ[0])
    noise = NOISE * (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
    return np.exp(1j * PHASE) * clean + background + noise


def fit_peak(row, target, found, noise):
    half = EXTRACTION["fit_half_window_px"]
    low, high = max(target - half, 0), min(target + half + 1, row.size)
    members = [p for p in found if low <= p < high]
    window, values = FREQ[low:high], row[low:high]
    narrow, wide = EXTRACTION["hwhm_range_ghz"]
    start, lower, upper = [], [], []
    for p in members:
        start += [FREQ[p], HWHM, row[p]]
        lower += [window[0], narrow, 0.0]
        upper += [window[-1], wide, np.inf]

    def residual(q):
        shape = np.full(window.size, q[-1])
        for centre, width, amplitude in q[:-1].reshape(-1, 3):
            shape += amplitude * width**2 / ((window - centre) ** 2 + width**2)
        return (shape - values) / noise

    result = least_squares(residual, start + [0.0], bounds=(lower + [-np.inf], upper + [np.inf]))
    j = 3 * members.index(target)
    sigma = np.sqrt(np.linalg.pinv(result.jac.T @ result.jac)[j, j])
    if not result.success or not np.isfinite(sigma) or sigma <= 0:
        return None
    centre, width, amplitude = (float(v) for v in result.x[j : j + 3])
    return {"centre": centre, "sigma": float(sigma), "hwhm": width, "amplitude": amplitude}


def extract(signal, axis=None):
    """Frozen extraction; ``axis`` is learned from training data and reused for holdout."""
    window = (1, EXTRACTION["median_window_px"])
    smooth = median_filter(signal.real, size=window, mode="nearest")
    smooth = smooth + 1j * median_filter(signal.imag, size=window, mode="nearest")
    residual = signal - smooth
    if axis is None:
        pairs = np.stack([residual.real.ravel(), residual.imag.ravel()])
        axis = np.linalg.eigh(pairs @ pairs.T)[1][:, -1]
        projected = residual.real * axis[0] + residual.imag * axis[1]
        if abs(projected.min()) > projected.max():
            axis = -axis
    projected = residual.real * axis[0] + residual.imag * axis[1]
    columns, failures = [], []
    for column, row in enumerate(projected):
        noise = 1.4826 * np.median(np.abs(row - np.median(row)))
        found, _ = find_peaks(
            row,
            height=EXTRACTION["threshold_noise"] * noise,
            distance=EXTRACTION["min_separation_px"],
        )
        if not found.size:
            failures.append({"column": column, "reason": "no peak"})
        peaks = []
        for p in found:
            peak = fit_peak(row, p, list(found), noise)
            if peak is None:
                failures.append({"column": column, "reason": "line fit failed"})
            else:
                peaks.append(peak)
        columns.append(peaks)
    return columns, failures, axis


def points(rng, bias, sigma, model=Squid, truth=None):
    centres = predict(model, model.truth if truth is None else truth, bias, F01)[:, 0]
    noisy = centres + rng.normal(0, sigma, bias.size)
    return [[{"centre": float(c), "sigma": sigma, "hwhm": HWHM, "amplitude": 1.0}] for c in noisy]


def assign(peaks, predicted, cutoff, extra=None):
    """One-to-one matching per column; ``cutoff(peak)`` is the standardised unmatched cost.

    Unresolved and unmatched peaks both cost ``cutoff(peak)**2``. The returned score ranks
    solutions; it carries no monotonicity guarantee across matching and fitting.
    ``extra`` adds a prediction variance per column and line, as in holdout evaluation.
    """
    labels, objective = [], 0.0
    resolution = MATCHING["resolution_linewidths"]
    extra = np.zeros_like(predicted) if extra is None else extra
    for column, column_peaks in enumerate(peaks):
        lines = predicted[column]
        row, candidates = [None] * len(column_peaks), []
        for i, peak in enumerate(column_peaks):
            if np.sum(np.abs(peak["centre"] - lines) <= resolution * peak["hwhm"]) >= 2:
                row[i] = ("unresolved",)
                objective += cutoff(peak) ** 2
            else:
                candidates.append(i)
        if candidates:
            chosen = [column_peaks[i] for i in candidates]
            distance2 = np.array(
                [(p["centre"] - lines) ** 2 / (p["sigma"] ** 2 + extra[column]) for p in chosen]
            )
            unmatched = np.array([cutoff(p) ** 2 for p in chosen])
            dummy = np.full((len(chosen), len(chosen)), np.inf)
            np.fill_diagonal(dummy, unmatched)
            for r, c in zip(*linear_sum_assignment(np.hstack([distance2, dummy])), strict=True):
                if c < lines.size:
                    row[candidates[r]] = ("assigned", int(c))
                    objective += distance2[r, c]
                else:
                    row[candidates[r]] = ("unexplained",)
                    objective += unmatched[r]
        labels.append(row)
    return labels, float(objective)


def phase_cutoff(phase):
    if phase == 1:
        return lambda peak: MATCHING["phase1_range_ghz"] / peak["sigma"]
    return lambda peak: MATCHING["cutoff_sigma"]


def assigned_rows(peaks, labels):
    return [
        (column, peak["centre"], peak["sigma"], label[1])
        for column, (column_peaks, column_labels) in enumerate(zip(peaks, labels, strict=True))
        for peak, label in zip(column_peaks, column_labels, strict=True)
        if label[0] == "assigned"
    ]


def residual_function(model, data, labels, family):
    rows = assigned_rows(data["peaks"], labels)
    columns = np.array([r[0] for r in rows], dtype=int)
    centres = np.array([r[1] for r in rows])
    sigmas = np.array([r[2] for r in rows])
    lines = np.array([r[3] for r in rows], dtype=int)

    def residual(theta):
        return (centres - predict(model, theta, data["bias"], family)[columns, lines]) / sigmas

    return residual, len(rows)


def solve(model, data, family, start, anchor=None):
    """Optional anchor phase on each column's strongest peak, then phases 1 and 2."""
    theta = np.clip(np.asarray(start, float), model.lower + 1e-9, model.upper - 1e-9)
    record = {"cycles": [0, 0], "nonconverged_phase": None, "active_bounds": []}
    if anchor is not None:
        strongest = [[max(c, key=lambda q: q["amplitude"])] if c else [] for c in data["peaks"]]
        first, inner = solve(model, {"bias": data["bias"], "peaks": strongest}, (anchor,), theta)
        record["anchor"] = {
            "cycles": inner["cycles"],
            "nonconverged_phase": inner["nonconverged_phase"],
            "labels": label_counts(first["labels"], (anchor,)) if first else inner["last_labels"],
        }
        if first is None:
            record["no_assigned_peaks"] = True
            record["status"] = "anchor initialization failed under the declared search"
            return None, record
        theta = np.clip(first["raw"], model.lower + 1e-9, model.upper - 1e-9)
    result = None
    for phase in (1, 2):
        cutoff = phase_cutoff(phase)
        limit = MATCHING[f"phase{phase}_limit"]
        labels, _ = assign(data["peaks"], predict(model, theta, data["bias"], family), cutoff)
        for _ in range(limit):
            residual, count = residual_function(model, data, labels, family)
            if count == 0:
                record["no_assigned_peaks"] = True
                record["status"] = f"no assigned peaks in phase {phase} under the declared search"
                record["last_labels"] = label_counts(labels, family)
                return None, record
            result = least_squares(
                residual, theta, bounds=(model.lower, model.upper), x_scale=model.scales
            )
            theta = result.x
            record["cycles"][phase - 1] += 1
            new, _ = assign(data["peaks"], predict(model, theta, data["bias"], family), cutoff)
            if new == labels:
                break
            labels = new
        else:
            record["nonconverged_phase"] = phase
    record["active_bounds"] = [model.names[k] for k in np.flatnonzero(result.active_mask)]
    labels, objective = assign(
        data["peaks"], predict(model, theta, data["bias"], family), phase_cutoff(2)
    )
    mapped, applied = model.conventions(theta)
    solution = {
        "theta": mapped,
        "raw": theta,
        "labels": labels,
        "objective": objective,
        "conventions": applied,
    }
    return solution, record


def scaled_jacobian(function, model, theta):
    columns = []
    for k, scale in enumerate(model.scales):
        step = np.zeros_like(theta)
        step[k] = STEP * scale
        columns.append((function(theta + step) - function(theta - step)) / (2 * STEP))
    return np.stack(columns, axis=-1)


def diagnose(model, data, solution, family):
    residual, count = residual_function(model, data, solution["labels"], family)
    theta = solution["theta"]
    chi2 = float(np.sum(residual(theta) ** 2))
    jacobian = scaled_jacobian(residual, model, theta)
    p = theta.size
    singular, right = np.linalg.svd(jacobian, full_matrices=True)[1:]
    rank = int(np.sum(singular > RANK_RTOL * singular[0])) if singular.size else 0
    null = right[rank:]
    cov_scaled = right[:rank].T @ np.diag(singular[:rank] ** -2) @ right[:rank]
    determined = [bool(np.linalg.norm(null[:, k]) <= NULL_TOL) for k in range(p)]
    sigma = [
        float(np.sqrt(cov_scaled[k, k]) * model.scales[k]) if determined[k] else None
        for k in range(p)
    ]
    known = [k for k in range(p) if determined[k]]
    block = cov_scaled[np.ix_(known, known)]
    norms = np.sqrt(np.diag(block))
    correlation = (block / np.outer(norms, norms)).round(4).tolist() if known else []
    width = model.upper - model.lower
    return (
        {
            "assigned": count,
            "chi2": chi2,
            "reduced_chi2": chi2 / (count - p) if count > p else None,
            "singular_values": singular.tolist(),
            "rank": rank,
            "undetermined_combinations": null.round(4).tolist(),
            "numerically_determined": dict(zip(model.names, determined, strict=True)),
            "sigma": dict(zip(model.names, sigma, strict=True)),
            "sigma_over_prior_width": {
                name: s / width[k] if s is not None else None
                for k, (name, s) in enumerate(zip(model.names, sigma, strict=True))
            },
            "correlation": {"parameters": [model.names[k] for k in known], "matrix": correlation},
            "outcome": "solution" if count >= p and rank == p else "insufficient evidence",
        },
        cov_scaled,
        null,
    )


def label_counts(labels, family):
    counts = {}
    for row in labels:
        for label in row:
            key = "{}{}/{}".format(*family[label[1]]) if label[0] == "assigned" else label[0]
            counts[key] = counts.get(key, 0) + 1
    return counts


def missing_lines(model, data, labels, theta, family):
    predicted = predict(model, theta, data["bias"], family)
    taken = {
        (c, label[1]) for c, row in enumerate(labels) for label in row if label[0] == "assigned"
    }
    inside = (predicted >= FREQ[0]) & (predicted <= FREQ[-1])
    return int(sum(1 for c, j in zip(*np.nonzero(inside), strict=True) if (c, j) not in taken))


def run_starts(model, data, family, starts, anchor=None):
    found, records, clock = [], [], time.perf_counter()
    for start in starts:
        solution, record = solve(model, data, family, start, anchor)
        records.append(record)
        if solution is not None and record["nonconverged_phase"] is None:
            found.append(solution)
    distinct = []
    for solution in sorted(found, key=lambda s: s["objective"]):
        if not any(
            s["labels"] == solution["labels"]
            and np.max(np.abs(s["theta"] - solution["theta"]) / model.scales) <= MERGE_TOL
            for s in distinct
        ):
            distinct.append(solution)
    summaries = []
    for solution in distinct:
        diagnosis, cov, null = diagnose(model, data, solution, family)
        solution.update(diagnosis=diagnosis, cov=cov, null=null)
        summaries.append(
            {
                "theta": dict(zip(model.names, solution["theta"].tolist(), strict=True)),
                "objective": solution["objective"],
                "conventions": solution["conventions"],
                "labels": label_counts(solution["labels"], family),
                "missing_lines": missing_lines(
                    model, data, solution["labels"], solution["theta"], family
                ),
                **diagnosis,
            }
        )
    if len(distinct) > 1:
        # Largest per-parameter |difference| from the best solution, divided by the two
        # solutions' combined marginal sigma; a local, marginal measure only.
        best = distinct[0]
        for solution, summary in zip(distinct[1:], summaries[1:], strict=True):
            spread = [
                abs(solution["theta"][k] - best["theta"][k])
                / np.hypot(summary["sigma"][name], summaries[0]["sigma"][name])
                for k, name in enumerate(model.names)
                if summary["sigma"][name] and summaries[0]["sigma"][name]
            ]
            summary["largest_marginal_separation"] = float(max(spread)) if spread else None
    return {
        "starts": len(starts),
        "nonconverged": sum(r["nonconverged_phase"] is not None for r in records),
        "no_assigned_peaks": sum(bool(r.get("no_assigned_peaks")) for r in records),
        "statuses": [r.get("status") for r in records],
        "cycles": [r["cycles"] for r in records],
        "active_bounds": [r["active_bounds"] for r in records],
        "anchor": [r.get("anchor") for r in records],
        "solutions": summaries,
        "seconds": time.perf_counter() - clock,
    }, distinct


def covered(model, solution, factor):
    sigma = solution["diagnosis"]["sigma"]
    return {
        name: sigma[name] is not None
        and bool(abs(solution["theta"][k] - model.truth[k]) <= factor * sigma[name])
        for k, name in enumerate(model.names)
    }


def holdout(model, data, solution, family, shift=None):
    """Match and score withheld peaks against the prediction and its propagated uncertainty."""
    theta = solution["theta"] if shift is None else solution["theta"] + shift
    predicted = predict(model, theta, data["bias"], family)
    gradient = scaled_jacobian(lambda t: predict(model, t, data["bias"], family), model, theta)
    extra = np.einsum("cfp,pq,cfq->cf", gradient, solution["cov"], gradient)
    if solution["null"].size:
        leak = np.linalg.norm(gradient @ solution["null"].T, axis=-1)
        undetermined_lines = leak > NULL_TOL * np.linalg.norm(gradient, axis=-1)
    else:
        undetermined_lines = np.zeros(predicted.shape, dtype=bool)
    labels, _ = assign(data["peaks"], predicted, phase_cutoff(2), extra)
    total = sum(len(row) for row in data["peaks"])
    z, undetermined = [], 0
    for column, peak_centre, sigma, line in assigned_rows(data["peaks"], labels):
        if undetermined_lines[column, line]:
            undetermined += 1
            continue
        spread = np.sqrt(sigma**2 + extra[column, line])
        z.append((peak_centre - predicted[column, line]) / spread)
    z = np.abs(np.array(z))
    counts = label_counts(labels, family)
    scored = int(z.size)
    return (
        {
            "extracted": total,
            "assigned": scored + undetermined,  # includes prediction_not_determined
            "prediction_not_determined": undetermined,
            "unresolved": counts.get("unresolved", 0),
            "unexplained": counts.get("unexplained", 0),
            "failures": len(data["failures"]),
            "within": {k: int(np.sum(z <= k)) for k in (1, 2)},
            "coverage_over_extracted": {
                k: float(np.sum(z <= k) / total) if total else None for k in (1, 2)
            },
            "coverage_over_scored": {
                k: float(np.sum(z <= k) / scored) if scored else None for k in (1, 2)
            },
            "mean_z2_over_scored": float(np.mean(z**2)) if scored else None,
        },
        gradient,
        labels,
    )


def displaced(direction, factor, model=Squid):
    signs = np.array(direction, float)
    start = model.truth * (1 + factor * signs)
    if model is Squid:
        start[3] = model.truth[3] + factor * signs[3] * model.truth[4]
    return start


PATTERNS = ((1, 1, 1, 1, 1), (-1, -1, -1, -1, -1), (1, -1, 1, -1, 1), (-1, 1, -1, 1, -1))
BIAS = np.linspace(-0.25, 0.65, 41)
TRAIN = BIAS <= BIAS[0] + 0.7 * (BIAS[-1] - BIAS[0])


def map_dataset(signal, bias, axis=None):
    peaks, failures, axis = extract(signal, axis)
    return {"bias": bias, "peaks": peaks, "failures": failures}, axis


def case_a(rng, realisations, starts):
    rows, kept = [], []
    for _ in range(realisations):
        clock = time.perf_counter()
        signal = make_map(rng, BIAS)
        training, axis = map_dataset(signal[TRAIN], BIAS[TRAIN])
        withheld, _ = map_dataset(signal[~TRAIN], BIAS[~TRAIN], axis)
        extract_seconds = time.perf_counter() - clock
        search, distinct = run_starts(
            Squid, training, FAMILY, [displaced(p, 0.03) for p in PATTERNS[:starts]], ANCHOR
        )
        row = {"extract_seconds": extract_seconds, "failures": len(training["failures"]), **search}
        if distinct:
            best = distinct[0]
            row["covered_1sigma"] = covered(Squid, best, 1)
            row["covered_2sigma"] = covered(Squid, best, 2)
            row["holdout"], gradient, labels = holdout(Squid, withheld, best, FAMILY)
            kept.append((withheld, best, gradient, labels))
        rows.append(row)
    return {"realisations": rows, "summary": coverage_summary(Squid, rows)}, kept


def pool_holdout(held):
    """Pool holdout counts over realisations that returned a fit."""
    keys = ("extracted", "assigned", "prediction_not_determined", "unresolved", "unexplained")
    pooled = {key: sum(h[key] for h in held) for key in keys}
    within = {k: sum(h["within"][k] for h in held) for k in (1, 2)}
    scored = pooled["assigned"] - pooled["prediction_not_determined"]
    return {
        "realisations": len(held),
        **pooled,
        "within": within,
        "per_realisation_mean_z2_over_scored": stats(h["mean_z2_over_scored"] for h in held),
        "coverage_over_extracted": {
            k: within[k] / pooled["extracted"] if pooled["extracted"] else None for k in (1, 2)
        },
        "coverage_over_scored": {k: within[k] / scored if scored else None for k in (1, 2)},
    }


def stats(values):
    values = np.array([v for v in values if v is not None], float)
    if not values.size:
        return None
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def coverage_summary(model, rows):
    """Coverage of the true values by the best fit, at two denominators, and fit statistics."""
    fitted = [r for r in rows if "covered_1sigma" in r]
    best = [r["solutions"][0] for r in rows if r["solutions"]]
    outcomes = [b["outcome"] for b in best]
    ranks = [b["rank"] for b in best]
    statuses = [s or "fit returned" for r in rows for s in r["statuses"]]
    others = [s for r in rows for s in r["solutions"][1:]]
    summary = {
        "realisations": len(rows),
        "with_fit": len(fitted),
        "without_fit": len(rows) - len(fitted),
        "best_outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))},
        "best_rank_counts": {str(k): ranks.count(k) for k in sorted(set(ranks))},
        "best_reduced_chi2": stats(b["reduced_chi2"] for b in best),
        "best_sigma": {n: stats(b["sigma"][n] for b in best) for n in model.names},
        "best_sigma_over_prior_width": {
            n: stats(b["sigma_over_prior_width"][n] for b in best) for n in model.names
        },
        "best_unexplained": stats(b["labels"].get("unexplained", 0) for b in best),
        "best_unresolved": stats(b["labels"].get("unresolved", 0) for b in best),
        "starts": sum(r["starts"] for r in rows),
        "start_statuses": {s: statuses.count(s) for s in sorted(set(statuses))},
        "starts_ending_on_a_bound": sum(bool(b) for r in rows for b in r["active_bounds"]),
        "other_solutions_largest_marginal_separation": stats(
            s.get("largest_marginal_separation") for s in others
        ),
    }
    for factor in (1, 2):
        key = f"covered_{factor}sigma"
        hits = {name: sum(r[key][name] for r in fitted) for name in model.names}
        summary[f"coverage_{factor}sigma_given_fit"] = {
            name: hits[name] / len(fitted) if fitted else None for name in model.names
        }
        summary[f"coverage_{factor}sigma_over_all_realisations"] = {
            name: hits[name] / len(rows) if rows else None for name in model.names
        }
    held = [r["holdout"] for r in fitted if "holdout" in r]
    if held:
        summary["holdout_pooled_given_fit"] = pool_holdout(held)
    summary["multiple_solutions"] = sum(len(r["solutions"]) > 1 for r in rows)
    summary["nonconverged_starts"] = sum(r["nonconverged"] for r in rows)
    return summary


def case_b(rng, realisations):
    out = {}
    for factor in (0.05, 0.10, 0.20):
        start = displaced((1, -1, 1, 1, -1), factor)
        initial = predict(Squid, start, BIAS, F01) - predict(Squid, Squid.truth, BIAS, F01)
        rows = []
        for _ in range(realisations):
            data, _ = map_dataset(make_map(rng, BIAS), BIAS)
            search, distinct = run_starts(Squid, data, FAMILY, [start], ANCHOR)
            if distinct:
                cover = covered(Squid, distinct[0], 5)
                search["reached_truth_within_5sigma"] = all(cover.values())
                search["covered_1sigma"] = covered(Squid, distinct[0], 1)
                search["covered_2sigma"] = covered(Squid, distinct[0], 2)
            rows.append(search)
        out[f"{factor:g}"] = {
            "initial_f01_error_ghz": {
                "median": float(np.median(np.abs(initial))),
                "max": float(np.max(np.abs(initial))),
            },
            "reached_truth": sum(bool(r.get("reached_truth_within_5sigma")) for r in rows),
            "summary": coverage_summary(Squid, rows),
            "realisations": rows,
        }
    return out


def case_c(rng, realisations, starts):
    bias = np.array([0.0])
    rows = []
    for _ in range(realisations):
        data = {"bias": bias, "peaks": points(rng, bias, 1e-4, Transmon), "failures": []}
        options = [Transmon.truth * [1.1, 0.9], Transmon.truth * [0.9, 1.1]][:starts]
        rows.append(run_starts(Transmon, data, F01, options)[0])
    return rows


def case_d(rng, realisations, starts):
    cells = {}
    for span in (0.9, 0.5, 0.2):
        bias = np.linspace(-0.25, -0.25 + span * Squid.truth[4], 41)
        for sigma in (1e-4, 1e-3, 1e-2):
            rows = []
            for _ in range(realisations):
                data = {"bias": bias, "peaks": points(rng, bias, sigma), "failures": []}
                search, distinct = run_starts(
                    Squid, data, F01, [displaced(p, 0.03) for p in PATTERNS[:starts]]
                )
                if distinct:
                    search["covered_1sigma"] = covered(Squid, distinct[0], 1)
                    search["covered_2sigma"] = covered(Squid, distinct[0], 2)
                rows.append(search)
            cells[f"span {span:g}, sigma {sigma * 1e3:g} MHz"] = {
                "summary": coverage_summary(Squid, rows),
                "realisations": rows,
            }
    return cells


def case_e(rng, realisations, starts):
    crossing = float(predict(Squid, Squid.truth, np.array([0.3]), ((1, 2, 1),))[0, 0])
    stresses = {
        "E1 line-shape mismatch": {"shape": "gaussian"},
        "E2 defect crossing f12": {"extra": ((crossing, 0.3),)},
        "E3 unmodelled line": {"extra": ((6.5, 0.3),)},
        "E4 one/two-photon ambiguity": {"visible": {(0, 1, 1): 1.0, (0, 2, 2): 0.2}},
        "E5 defect stronger than f01": {"extra": ((4.5, 1.5),)},
    }
    out = {}
    for name, options in stresses.items():
        if name.startswith("E4"):
            draws = np.random.default_rng(4).uniform(Squid.lower, Squid.upper, (starts, 5))
            options_starts = list(draws)
        else:
            options_starts = [displaced(p, f) for f in (0.03, 0.10) for p in PATTERNS][:starts]
        rows = []
        for _ in range(realisations):
            data, _ = map_dataset(make_map(rng, BIAS, **options), BIAS)
            search, distinct = run_starts(Squid, data, FAMILY, options_starts, ANCHOR)
            if distinct:
                search["covered_1sigma"] = covered(Squid, distinct[0], 1)
                search["covered_2sigma"] = covered(Squid, distinct[0], 2)
            search["failures"] = len(data["failures"])
            rows.append(search)
        out[name] = {"summary": coverage_summary(Squid, rows), "realisations": rows}
    out["E2 defect crossing f12"]["defect_ghz"] = crossing
    return out


def case_f(kept):
    """Displace EJ1 so the median withheld line moves by 2 and 20 median prediction sigmas."""
    out = {"shift_2_median_sigma_pred": [], "shift_20_median_sigma_pred": []}
    for withheld, best, gradient, labels in kept:
        rows = assigned_rows(withheld["peaks"], labels)
        if not rows:
            continue
        slope = np.median([abs(gradient[c, j, 0]) for c, _, _, j in rows])  # per unit scale
        spread = np.median(
            [np.sqrt(s**2 + gradient[c, j] @ best["cov"] @ gradient[c, j]) for c, _, s, j in rows]
        )
        for multiple in (2, 20):
            shift = np.zeros(5)
            shift[0] = multiple * spread / slope * Squid.scales[0]
            key = f"shift_{multiple}_median_sigma_pred"
            out[key].append(holdout(Squid, withheld, best, FAMILY, shift)[0])
    return out


def failed(rows, reached=None, keep=3):
    """Count realisations without a returned solution and keep a few representatives."""
    bad = [
        (i, row)
        for i, row in enumerate(rows)
        if not row["solutions"]
        or row["solutions"][0]["outcome"] != "solution"
        or (reached is not None and not row.get(reached))
    ]
    return {
        "count": len(bad),
        "of": len(rows),
        "representatives": [
            {
                "realisation": i,
                **{k: row[k] for k in ("statuses", "anchor", "cycles", "nonconverged")},
                "solutions": row["solutions"],
            }
            for i, row in bad[:keep]
        ],
    }


def compact(result):
    """The tracked results: settings, provenance, summaries, failure counts and examples."""
    keep = ("stage", "provenance", "settings", "model_check", "seconds")
    out = {key: result[key] for key in keep}
    out["A"] = {
        "summary": result["A"]["summary"],
        "failed": failed(result["A"]["realisations"]),
    }
    out["B"] = {
        factor: {
            "initial_f01_error_ghz": b["initial_f01_error_ghz"],
            "reached_truth": b["reached_truth"],
            "summary": b["summary"],
            "failed": failed(b["realisations"], "reached_truth_within_5sigma"),
        }
        for factor, b in result["B"].items()
    }
    out["C"] = {"summary": coverage_summary(Transmon, result["C"]), "realisations": result["C"]}
    out["D"] = {
        cell: {"summary": v["summary"], "example": v["realisations"][0]}
        for cell, v in result["D"].items()
    }
    out["E"] = {
        name: {
            "summary": v["summary"],
            "failed": failed(v["realisations"]),
            **({"defect_ghz": v["defect_ghz"]} if "defect_ghz" in v else {}),
        }
        for name, v in result["E"].items()
    }
    out["F"] = {shift: pool_holdout(held) for shift, held in result["F"].items()}
    return out


def provenance():
    repository = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.check_output(["git", *a], cwd=repository, text=True)  # noqa: E731
    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "source_revision": git("rev-parse", "HEAD").strip(),
        "source_dirty": bool(git("status", "--porcelain", "--untracked-files=no").strip()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "scipy", "scqubits")},
        "environment": {"OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
    }


def model_check():
    from entanglix.models import load_system

    worst = 0.0
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "squid.toml"
        for flux in (0.0, 0.2, 0.4):
            path.write_text(SPEC.replace("FLUX", repr(flux)))
            values = np.linalg.eigvalsh(load_system(path).hamiltonian_ghz)[:4]
            mine = Squid.levels(np.array([12.0, 8.0, 70.0, 0.0, 1.0]), np.array([flux]))[0]
            worst = max(worst, float(np.max(np.abs((values - values[0]) - mine))))
    clock = time.perf_counter()
    for _ in range(100):
        Squid.levels(Squid.truth, BIAS)
    return {
        "max_level_difference_ghz": worst,
        "forward_ms_per_41_points": (time.perf_counter() - clock) * 10,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, help="every realisation, kept outside Git")
    args = parser.parse_args()
    counts = COUNTS[args.stage]
    rng = np.random.default_rng(20261020 + args.stage)
    clock = time.perf_counter()
    timings = {}
    result = {
        "stage": args.stage,
        "provenance": provenance(),
        "settings": {
            "truth": dict(zip(Squid.names, Squid.truth.tolist(), strict=True)),
            "ecj_ghz": ECJ,
            "charge_cutoff": NCUT,
            "bias_v": [BIAS[0], BIAS[-1], BIAS.size],
            "holdout": "contiguous upper 30 % of the bias range",
            "window_ghz": [FREQ[0], FREQ[-1]],
            "pixel_mhz": 1,
            "hwhm_ghz": HWHM,
            "noise": NOISE,
            "visible_lines": {"{}{}/{}".format(*k): v for k, v in VISIBLE.items()},
            "family": ["{}{}/{}".format(*k) for k in FAMILY],
            "anchor": "{}{}/{} as each column's largest-amplitude peak (map cases)".format(*ANCHOR),
            "extraction": EXTRACTION,
            "matching": MATCHING,
            "rank_rtol": RANK_RTOL,
            "null_tolerance": NULL_TOL,
            "merge_tolerance": MERGE_TOL,
            "scales": dict(zip(Squid.names, Squid.scales.tolist(), strict=True)),
            "bounds": [Squid.lower.tolist(), Squid.upper.tolist()],
            "counts": counts,
            "seed": 20261020 + args.stage,
        },
        "model_check": model_check(),
    }
    for name, run in (
        ("A", lambda: case_a(rng, *counts["A"])),
        ("B", lambda: case_b(rng, counts["B"][0])),
        ("C", lambda: case_c(rng, *counts["C"])),
        ("D", lambda: case_d(rng, *counts["D"])),
        ("E", lambda: case_e(rng, *counts["E"])),
    ):
        start = time.perf_counter()
        output = run()
        if name == "A":
            output, kept = output
            result["F"] = case_f(kept)
        result[name] = output
        timings[name] = time.perf_counter() - start
        print(f"case {name}: {timings[name]:.1f} s", flush=True)
    result["seconds"] = {**timings, "total": time.perf_counter() - clock}
    write = lambda path, value: path.write_text(  # noqa: E731
        json.dumps(value, indent=2, allow_nan=False, default=float) + "\n"
    )
    if args.full_output:
        args.full_output.parent.mkdir(parents=True, exist_ok=True)
        write(args.full_output, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, compact(result))


if __name__ == "__main__":
    main()
