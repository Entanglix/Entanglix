"""Assignment, fitting and diagnostics for calibration from spectroscopy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tomli_w
from scipy.optimize import least_squares, linear_sum_assignment

from entanglix.models import SpecError, load_system

# The calibration protocol's own tolerances; every fit records the values it used.
RANK_RTOL = 1e-6
UNDETERMINED_TOLERANCE = 1e-3
JACOBIAN_STEP = 1e-5
# A convention is kept only where every predicted line moves by at most this fraction of the
# largest predicted line.
EQUIVALENCE_RTOL = 1e-8
UNEXPLAINED, UNRESOLVED = -1, -2
ANCHOR_FAILED = "anchor initialization failed under the declared search"


@dataclass(frozen=True)
class Member:
    """A candidate transition between dressed levels by energy order, or bare product states."""

    label: str
    initial: int | tuple[int, ...]
    final: int | tuple[int, ...]
    photons: int


@dataclass(frozen=True)
class Peaks:
    """Peaks by bias column; ``line`` holds supplied member indices, or -1 when unlabelled."""

    column: np.ndarray
    centre: np.ndarray
    uncertainty: np.ndarray
    linewidth: np.ndarray
    amplitude: np.ndarray
    line: np.ndarray

    def take(self, rows) -> Peaks:
        return Peaks(*(getattr(self, name)[rows] for name in self.__dataclass_fields__))


class Spectrum:
    """Each column's candidate transition frequencies, by full construction of the system.

    ``theta`` holds the fitted system parameters in order, then the axis map's offset and
    period when ``flux`` names the parameter a bias reaches. A column without a bias keeps
    the system's own flux value.
    """

    def __init__(self, path: Path, document: dict, members, fitted, bias, flux=None):
        self.path, self.document, self.members = path, document, tuple(members)
        self.fitted, self.bias, self.flux = tuple(fitted), np.asarray(bias, float), flux
        self.own_flux = document["parameters"][flux]["value"] if flux is not None else None
        self.bare = sorted(
            {state for m in self.members for state in (m.initial, m.final) if type(state) is tuple}
        )

    def _lines(self, system) -> np.ndarray:
        if self.bare:
            energies, vectors = np.linalg.eigh(system.hamiltonian_ghz)
            rows = [np.ravel_multi_index(state, system.dimensions) for state in self.bare]
            # Each declared bare state takes the dressed state it overlaps most, one to one.
            states, dressed = linear_sum_assignment(abs(vectors[rows]) ** 2, maximize=True)
            index = {self.bare[s]: d for s, d in zip(states, dressed, strict=True)}
        else:
            energies, index = np.linalg.eigvalsh(system.hamiltonian_ghz), {}
        levels = [
            (index.get(m.initial, m.initial), index.get(m.final, m.final)) for m in self.members
        ]
        return np.array(
            [(energies[f] - energies[i]) / m.photons for (i, f), m in zip(levels, self.members)]
        )

    def __call__(self, theta, columns) -> np.ndarray:
        """Lines of the requested columns; other columns are NaN."""
        predicted = np.full((self.bias.size, len(self.members)), np.nan)
        parameters = self.document["parameters"]
        for name, value in zip(self.fitted, theta, strict=False):
            parameters[name]["value"] = float(value)
        built = {}
        for column in columns:
            flux = None
            if self.flux is not None:
                bias = self.bias[column]
                offset, period = theta[-2:]
                flux = self.own_flux if math.isnan(bias) else float((bias - offset) / period)
                parameters[self.flux]["value"] = flux
            if flux not in built:
                content = tomli_w.dumps(self.document).encode()
                built[flux] = self._lines(load_system(self.path, content=content))
            predicted[column] = built[flux]
        return predicted


@dataclass(frozen=True)
class Matching:
    range_ghz: float
    cutoff: float
    resolution: float
    range_iterations: int
    cutoff_iterations: int
    anchor: int | None = None


@dataclass(frozen=True)
class Problem:
    """Everything a search needs: the forward model, the peaks, bounds, scales and settings.

    ``matching`` is None when every peak carries a supplied assignment. ``windows`` gives each
    column's measured frequency range, or None for points.
    """

    spectrum: Spectrum
    peaks: Peaks
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    scales: np.ndarray
    merge_tolerance: float
    matching: Matching | None = None
    exchange: tuple[tuple[int, int], ...] = ()
    offset_window: float | None = None
    windows: tuple = field(default=())


def assign(peaks: Peaks, predicted, candidates, unmatched, resolution):
    """Match each column's peaks one to one with candidate lines.

    Matching a peak costs its squared standardised distance; an unmatched peak, or one within
    ``resolution`` linewidths of two candidates, costs ``unmatched``. Returns each peak's member
    index, UNEXPLAINED or UNRESOLVED, and the total cost, which ranks assignments and solutions
    without any monotonicity guarantee.
    """
    lines = np.full(peaks.centre.size, UNEXPLAINED)
    score = 0.0
    candidates = np.asarray(candidates)
    for column in np.unique(peaks.column):
        rows = np.flatnonzero(peaks.column == column)
        frequencies = predicted[column, candidates]
        offsets = peaks.centre[rows, None] - frequencies[None, :]
        unresolved = (np.abs(offsets) <= resolution * peaks.linewidth[rows, None]).sum(axis=1) >= 2
        lines[rows[unresolved]] = UNRESOLVED
        score += float(unmatched[rows[unresolved]].sum())
        offsets, rows = offsets[~unresolved], rows[~unresolved]
        if not rows.size:
            continue
        distance = (offsets / peaks.uncertainty[rows, None]) ** 2
        dummy = np.full((rows.size, rows.size), np.inf)
        np.fill_diagonal(dummy, unmatched[rows])
        for r, c in zip(*linear_sum_assignment(np.hstack([distance, dummy])), strict=True):
            if c < candidates.size:
                lines[rows[r]] = candidates[c]
                score += float(distance[r, c])
            else:
                score += float(unmatched[rows[r]])
    return lines, score


def residual_function(problem: Problem, peaks: Peaks, lines):
    """Standardised residuals of the assigned peaks as a function of the parameters."""
    rows = np.flatnonzero(lines >= 0)
    columns = np.unique(peaks.column[rows])

    def residual(theta):
        predicted = problem.spectrum(theta, columns)
        return (
            peaks.centre[rows] - predicted[peaks.column[rows], lines[rows]]
        ) / peaks.uncertainty[rows]

    return residual


def _fit(problem: Problem, peaks: Peaks, lines, theta):
    return least_squares(
        residual_function(problem, peaks, lines),
        theta,
        bounds=(problem.lower, problem.upper),
        x_scale=problem.scales,
    )


def _counts(lines) -> dict[str, int]:
    return {
        "assigned": int(np.sum(lines >= 0)),
        "unexplained": int(np.sum(lines == UNEXPLAINED)),
        "unresolved": int(np.sum(lines == UNRESOLVED)),
    }


def _phases(problem: Problem, peaks: Peaks, theta, candidates, cycles, prefix=""):
    """Match within the range, then within the cutoff, refitting until the assignment holds.

    Returns the last fit, the assignment, its cost and a failure, which is None on success.
    """
    matching = problem.matching
    columns = np.unique(peaks.column)
    fit, lines, score = None, np.full(peaks.centre.size, UNEXPLAINED), math.inf
    for phase, unmatched, limit in (
        ("range", (matching.range_ghz / peaks.uncertainty) ** 2, matching.range_iterations),
        ("cutoff", np.full(peaks.centre.size, matching.cutoff**2), matching.cutoff_iterations),
    ):
        name = f"{prefix}{phase}"
        lines, score = assign(
            peaks, problem.spectrum(theta, columns), candidates, unmatched, matching.resolution
        )
        for _ in range(limit):
            if not np.any(lines >= 0):
                return fit, lines, score, f"no assigned peaks in the {name} phase"
            fit = _fit(problem, peaks, lines, theta)
            cycles[name] = cycles.get(name, 0) + 1
            if not fit.success:
                return fit, lines, score, f"the optimizer did not converge: {fit.message}"
            theta = fit.x
            new, score = assign(
                peaks, problem.spectrum(theta, columns), candidates, unmatched, matching.resolution
            )
            if np.array_equal(new, lines):
                break
            lines = new
        else:
            return fit, lines, score, f"non-converged: the {name} phase reached its iteration limit"
    return fit, lines, score, None


def _cost(problem: Problem, predicted, lines):
    """Supplied labels keep their assignment and cost χ²; otherwise peaks are matched afresh."""
    peaks = problem.peaks
    if problem.matching is None:
        rows = np.flatnonzero(lines >= 0)
        z = (peaks.centre[rows] - predicted[peaks.column[rows], lines[rows]]) / peaks.uncertainty[
            rows
        ]
        return lines, float(z @ z)
    return assign(
        peaks,
        predicted,
        range(len(problem.spectrum.members)),
        np.full(peaks.centre.size, problem.matching.cutoff**2),
        problem.matching.resolution,
    )


def apply_conventions(problem: Problem, theta, lines, score):
    """Map a converged point by each declared convention that keeps every predicted line.

    A convention is kept only where the model, at the mapped values, reproduces every candidate
    line of every column with peaks within EQUIVALENCE_RTOL of the largest line; the assignment
    and cost are then recomputed at the published values. Returns the values, assignment and
    cost, the equivalents removed, and the conventions not applied with their largest change.
    """
    columns = np.unique(problem.peaks.column)
    reference = problem.spectrum(theta, columns)
    tolerance = EQUIVALENCE_RTOL * np.max(np.abs(reference[columns]))
    mapped, predicted, removed, refused = np.array(theta, dtype=float), None, [], []
    steps = [*problem.exchange, *(["offset"] if problem.offset_window is not None else [])]
    for step in steps:
        candidate = mapped.copy()
        if step == "offset":
            shift = math.floor(candidate[-2] / candidate[-1] - problem.offset_window)
            if not shift:
                continue
            candidate[-2] -= shift * candidate[-1]
            label = f"offset alias: shifted by {-shift} period(s)"
        else:
            a, b = step
            if candidate[a] >= candidate[b]:
                continue
            candidate[[a, b]] = candidate[[b, a]]
            label = f"exchange {problem.names[a]}/{problem.names[b]}"
        trial = problem.spectrum(candidate, columns)
        change = float(np.max(np.abs(trial[columns] - reference[columns])))
        if change <= tolerance:
            mapped, predicted = candidate, trial
            removed.append(label)
        else:
            refused.append({"convention": label, "largest_line_change_ghz": change})
    if predicted is not None:
        lines, score = _cost(problem, predicted, lines)
    return mapped, lines, score, removed, refused


def search(problem: Problem, start):
    """One start: the optional anchor phase, then range and cutoff matching, or a labelled fit."""
    record = {"cycles": {}}
    peaks, theta = problem.peaks, np.asarray(start, dtype=float)
    try:
        if problem.matching is None:
            lines, score = peaks.line, None
            fit = _fit(problem, peaks, lines, theta)
            record["cycles"]["fit"] = 1
            failure = None if fit.success else f"the optimizer did not converge: {fit.message}"
        else:
            if problem.matching.anchor is not None:
                # The anchor transition is taken to be each column's largest-amplitude peak.
                strongest = [
                    rows[np.argmax(peaks.amplitude[rows])]
                    for rows in (np.flatnonzero(peaks.column == c) for c in np.unique(peaks.column))
                ]
                fit, lines, _, failure = _phases(
                    problem,
                    peaks.take(strongest),
                    theta,
                    [problem.matching.anchor],
                    record["cycles"],
                    "anchor ",
                )
                record["anchor"] = {"columns": len(strongest), **_counts(lines)}
                if failure is not None:
                    record["status"] = (
                        ANCHOR_FAILED if failure.startswith("no assigned") else failure
                    )
                    return None, record
                theta = fit.x
            fit, lines, score, failure = _phases(
                problem, peaks, theta, range(len(problem.spectrum.members)), record["cycles"]
            )
    except SpecError as exc:
        record["status"] = f"construction failed: {exc}"
        return None, record
    if failure is not None:
        record["status"] = failure
        return None, record
    if score is None:
        score = float(np.sum(fit.fun**2))
    try:
        values, lines, score, removed, refused = apply_conventions(problem, fit.x, lines, score)
    except SpecError as exc:
        record["status"] = f"construction failed: {exc}"
        return None, record
    # A parameter ends on a bound within one diagnostic step of it, where differences turn
    # one-sided.
    room = np.minimum(fit.x - problem.lower, problem.upper - fit.x)
    steps = diagnostic_steps(problem.scales, problem.lower, problem.upper)
    record.update(
        status="solution",
        bounds_hit=[problem.names[k] for k in np.flatnonzero(room <= steps)],
        equivalents_removed=removed,
        conventions_not_applied=refused,
    )
    return {"theta": values, "lines": lines, "objective": score}, record


def diagnostic_steps(scales, lower, upper, step=JACOBIAN_STEP):
    """Finite-difference steps: ``step`` times the smaller of each scale and its bounds' width.

    A step is then at most ``step`` of its interval, so differences always fit strictly inside.
    """
    return step * np.minimum(scales, np.asarray(upper, float) - np.asarray(lower, float))


def _scaled_derivative(residual, theta, values, k, scale, d, lower, upper):
    """The residuals' derivative along parameter ``k`` per unit scale, from steps of ``d``.

    Central differences where both neighbours lie strictly inside the bounds; otherwise
    second-order one-sided differences toward the farther bound, which has room for both
    steps. The bounds themselves are never sampled. A value outside its bounds, such as an
    offset wrapped by a convention, uses central differences.
    """
    down, up = theta[k] - lower, upper - theta[k]
    if (down > d and up > d) or down < 0 or up < 0:
        stencil = ((-d, -0.5), (d, 0.5))
    elif up >= down:
        stencil = ((0.0, -1.5), (d, 2.0), (2 * d, -0.5))
    else:
        stencil = ((0.0, 1.5), (-d, -2.0), (-2 * d, 0.5))
    total = np.zeros_like(values)
    for offset, weight in stencil:
        shifted = theta.copy()
        shifted[k] += offset
        total += weight * (values if offset == 0 else residual(shifted))
    return scale * total / d


def diagnose(
    residual,
    theta,
    scales,
    lower,
    upper,
    names,
    *,
    rank_rtol=RANK_RTOL,
    undetermined_tolerance=UNDETERMINED_TOLERANCE,
    step=JACOBIAN_STEP,
) -> dict:
    """Fit statistics, rank, undetermined combinations, covariance and correlations at ``theta``.

    The Jacobian is taken in coordinates scaled by the declared scales, by finite differences
    sampled strictly inside the bounds. Covariance is unscaled and lies along the determined directions
    only; a parameter whose component along an undetermined combination exceeds
    ``undetermined_tolerance`` has no uncertainty.
    """
    theta, scales, lower, upper = (
        np.asarray(value, dtype=float) for value in (theta, scales, lower, upper)
    )
    width = upper - lower
    values = residual(theta)
    chi2, count, p = float(values @ values), values.size, theta.size
    steps = diagnostic_steps(scales, lower, upper, step)
    columns = [
        _scaled_derivative(residual, theta, values, k, scales[k], steps[k], lower[k], upper[k])
        for k in range(p)
    ]
    singular, right = np.linalg.svd(np.stack(columns, axis=-1), full_matrices=True)[1:]
    rank = int(np.sum(singular > rank_rtol * singular[0])) if singular.size else 0
    undetermined = right[rank:]
    scaled = right[:rank].T @ np.diag(singular[:rank] ** -2.0) @ right[:rank]
    known = [k for k in range(p) if np.linalg.norm(undetermined[:, k]) <= undetermined_tolerance]
    covariance = scaled[np.ix_(known, known)] * np.outer(scales[known], scales[known])
    sigma = np.sqrt(np.diag(covariance))
    statistics = {"chi2": chi2, "assigned": count, "fitted": p}
    if count > p:
        statistics["reduced_chi2"] = chi2 / (count - p)
    determined = [names[k] for k in known]
    return {
        "outcome": "solution" if count >= p and rank == p else "insufficient evidence",
        "statistics": statistics,
        "rank": rank,
        "singular_values": singular.tolist(),
        "undetermined_combinations": undetermined.tolist(),
        "determined": determined,
        "uncertainty": dict(zip(determined, sigma.tolist(), strict=True)),
        "uncertainty_over_prior_width": {
            names[k]: float(s / width[k]) for k, s in zip(known, sigma, strict=True)
        },
        "covariance": {"parameters": determined, "matrix": covariance.tolist()},
        "correlation": {
            "parameters": determined,
            "matrix": (covariance / np.outer(sigma, sigma)).tolist(),
        },
    }


def calibrate_problem(problem: Problem, starts) -> dict:
    """Search from every start, merge equal solutions, and report each distinct one."""
    records, found = [], []
    for index, start in enumerate(starts):
        solution, record = search(problem, start)
        records.append(record)
        if solution is not None:
            found.append((index, solution))
    distinct = []
    for index, solution in sorted(found, key=lambda item: item[1]["objective"]):
        for number, other in enumerate(distinct):
            if (
                np.array_equal(other["lines"], solution["lines"])
                and np.max(np.abs(other["theta"] - solution["theta"]) / problem.scales)
                <= problem.merge_tolerance
            ):
                other["starts"].append(index)
                records[index]["solution"] = number
                break
        else:
            records[index]["solution"] = len(distinct)
            distinct.append({**solution, "starts": [index]})
    reports = []
    for solution in distinct:
        report = diagnose(
            residual_function(problem, problem.peaks, solution["lines"]),
            solution["theta"],
            problem.scales,
            problem.lower,
            problem.upper,
            problem.names,
        )
        windowed = [c for c, window in enumerate(problem.windows) if window is not None]
        predicted = problem.spectrum(solution["theta"], windowed)
        taken = {(c, j) for c, j in zip(problem.peaks.column, solution["lines"]) if j >= 0}
        missing = [
            {"column": c, "line": problem.spectrum.members[j].label}
            for c in windowed
            for j in range(len(problem.spectrum.members))
            if problem.windows[c][0] <= predicted[c, j] <= problem.windows[c][1]
            and (c, j) not in taken
        ]
        removed = sorted(
            {item for i in solution["starts"] for item in records[i]["equivalents_removed"]}
        )
        refused = sorted(
            {
                item["convention"]
                for i in solution["starts"]
                for item in records[i]["conventions_not_applied"]
            }
        )
        bounds = sorted({item for i in solution["starts"] for item in records[i]["bounds_hit"]})
        reports.append(
            {
                "outcome": report.pop("outcome"),
                "values": dict(zip(problem.names, solution["theta"].tolist(), strict=True)),
                "objective": solution["objective"],
                "starts": sorted(solution["starts"]),
                "equivalents_removed": removed,
                "conventions_not_applied": refused,
                "bounds_hit": bounds,
                "counts": {**_counts(solution["lines"]), "missing": len(missing)},
                "lines": {
                    member.label: int(np.sum(solution["lines"] == j))
                    for j, member in enumerate(problem.spectrum.members)
                    if np.any(solution["lines"] == j)
                },
                **report,
                "assignment": solution["lines"],
                "missing": missing,
            }
        )
    for report in reports[1:]:
        # Largest per-parameter difference from the best solution over the two solutions'
        # combined marginal uncertainty: a local, marginal measure only.
        best = reports[0]
        spread = [
            abs(report["values"][name] - best["values"][name])
            / math.hypot(report["uncertainty"][name], best["uncertainty"][name])
            for name in problem.names
            if name in report["uncertainty"] and name in best["uncertainty"]
        ]
        if spread:
            report["largest_marginal_separation"] = max(spread)
    if not reports:
        statuses = {record["status"] for record in records}
        outcome = ANCHOR_FAILED if statuses == {ANCHOR_FAILED} else "no fit"
    elif all(report["outcome"] == "insufficient evidence" for report in reports):
        outcome = "insufficient evidence"
    else:
        outcome = "several distinct solutions" if len(reports) > 1 else "solution"
    return {"outcome": outcome, "starts": records, "solutions": reports}
