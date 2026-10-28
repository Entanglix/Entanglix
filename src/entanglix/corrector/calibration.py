"""Requested calibration checks: holdout prediction, identifiability and refinement."""

import functools
import tempfile
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from entanglix.calibration import (
    UNEXPLAINED,
    UNRESOLVED,
    assign,
    decompose,
    diagnose,
    residual_function,
    scaled_jacobian,
)
from entanglix.corrector import (
    CHECK_DEFINITIONS,
    load_policy,
    relevant_checks,
    validate_refinement,
)
from entanglix.corrector.checks import check, inapplicable
from entanglix.corrector.reference import write_refined_system
from entanglix.extraction import frequency_step
from entanglix.measurements import load_dataset
from entanglix.models import load_system
from entanglix.protocols.calibrate import (
    chosen_solution,
    dataset_peaks,
    merge_peaks,
    prepare,
    spectrum_for,
)

SCOPES = {
    "holdout": (
        "withheld peaks under the source's fitted parameters and recorded procedure, with "
        "nothing refitted or learned; each prediction's uncertainty combines the centre "
        "uncertainty with the recomputed covariance, propagated linearly"
    ),
    "identifiability": (
        "the source's training peaks and final assignment at the fitted parameters; a local "
        "linear approximation, conditional on the assignment and the centre uncertainties"
    ),
    "refinement": (
        "the source's assigned training peaks at the fitted parameters, predicted at the "
        "refined truncation and standardised by each peak's centre uncertainty"
    ),
}
NO_FIT = "the source run returned no fitted parameters"
NO_REFINEMENT = "no refinement declared; a check request may supply one"


def _overlap(first, second) -> bool:
    return any(max(a, c) < min(b, d) for a, b in first for c, d in second)


def _relation(holdout, training) -> str | None:
    """How a holdout dataset shares bias points with a training dataset, if it does."""
    if holdout.sha256 == training.sha256:
        return "is"
    part, other = holdout.spec.split, training.spec.split
    if other is not None and other.parent_sha256 == holdout.sha256:
        return "is the parent of"
    if part is not None and part.parent_sha256 == training.sha256:
        return "is a split part of"
    if (
        part is not None
        and other is not None
        and part.parent_sha256 == other.parent_sha256
        and (_overlap(part.columns, other.columns) or _overlap(part.points, other.points))
    ):
        return "overlaps"
    return None


def holdout_form(loaded, dataset) -> tuple[str | None, str | None]:
    """The procedure the source recorded for a holdout dataset, or why none applies.

    Returns the form, "map", "labelled points" or "unlabelled points", or the reason the
    holdout check is inapplicable. Refuses a dataset that shares bias points with a fitting
    input, a partly labelled dataset, labels outside the family, and data the recorded
    procedure cannot read.
    """
    for name, training in loaded.datasets.items():
        relation = _relation(dataset, training)
        if relation is not None:
            raise ValueError(f"the holdout dataset {relation} the fitting input {name!r}")
    labels = dataset.arrays.get("points.assignment")
    given = 0 if labels is None else int(np.count_nonzero(labels != ""))
    if labels is not None and 0 < given < labels.size:
        raise ValueError(
            f"partly labelled holdout dataset: {given} of {labels.size} points carry an assignment"
        )
    spec = loaded.spec
    if spec.extraction is not None and dataset.spec.map is not None:
        try:
            frequency_step(dataset)
        except ValueError as exc:
            raise ValueError(f"holdout map: {exc}") from exc
        form = "map"
    elif dataset.spec.points is None:
        return None, "the source recorded no extraction settings, so a holdout map is not read"
    elif given:
        unknown = sorted(set(labels.tolist()) - {member.label for member in loaded.members})
        if unknown:
            raise ValueError(
                f"holdout label {unknown[0]!r} is not a member of the candidate family"
            )
        form = "labelled points"
    elif spec.matching is None:
        return None, (
            "the source fitted supplied labels and recorded no matching settings "
            "for unlabelled points"
        )
    else:
        if "points.linewidth" not in dataset.arrays:
            raise ValueError("unlabelled holdout points need each point's linewidth")
        form = "unlabelled points"
    unit = dataset.spec.bias_unit
    if unit is not None:
        if spec.axis_map is None:
            raise ValueError("a holdout dataset with a bias axis needs the source's axis map")
        if unit not in {training.spec.bias_unit for training in loaded.datasets.values()}:
            raise ValueError(f"the holdout bias unit {unit!r} differs from the training datasets'")
    return form, None


def calibration_request(
    loaded, result, checks, *, holdout=None, solution=None, expect_coverage=None
) -> tuple[dict, dict[str, bytes]]:
    """Validate a calibration check request; returns its request fields and files to freeze."""
    number = chosen_solution(result, solution, "check")
    fields = {} if number is None else {"solution": number}
    if holdout is None:
        if expect_coverage is not None:
            raise ValueError("a coverage expectation needs a holdout dataset")
        return fields, {}
    if "holdout" not in checks:
        raise ValueError("a holdout dataset is used only by the holdout check")
    holdout = Path(holdout)
    dataset = load_dataset(holdout)
    holdout_form(loaded, dataset)
    name = f"holdout/{holdout.name}"
    files = {
        name: dataset.files[holdout.resolve().name],
        f"holdout/{dataset.spec.array_file}": dataset.files[dataset.spec.array_file],
    }
    fields["holdout"] = {"dataset": name, "sha256": dataset.sha256}
    if expect_coverage is not None:
        if (
            len(expect_coverage) != 2
            or expect_coverage[0] not in (1, 2)
            or not 0 < expect_coverage[1] <= 1
        ):
            raise ValueError("a coverage expectation needs within 1 or 2 and a minimum in (0, 1]")
        within, minimum = expect_coverage
        fields["holdout"]["expectation"] = {"within": int(within), "minimum": float(minimum)}
    return fields, files


def calibration_listing(loaded, result=None, policy=None) -> dict:
    """The checks a calibration can take, at launch and preview or, with its result, once run.

    Availability only annotates the listing; it never refuses the calibration.
    """
    policy = policy or load_policy()
    count = None if result is None else len(result["solutions"])
    forms = [
        *(["map"] if loaded.spec.extraction is not None else []),
        "labelled points",
        *(["unlabelled points"] if loaded.spec.matching is not None else []),
    ]
    entries = []
    for name in relevant_checks("calibrate"):
        reason = (
            NO_FIT
            if count == 0
            else refinement_reason(loaded, loaded.spec.refinement, policy)
            if name == "refinement"
            else ""
        )
        entries.append(
            {
                "name": name,
                "error_mode": CHECK_DEFINITIONS[name],
                "available": not reason,
                **({"reason": reason} if reason else {}),
            }
        )
    listing = {"checks": entries, "holdout_forms": forms}
    return listing if count is None else {**listing, "solutions": count}


def refinement_reason(loaded, declaration, policy) -> str:
    """Why a refinement declaration cannot be checked, or an empty string."""
    if not declaration:
        return NO_REFINEMENT
    try:
        validate_refinement(
            SimpleNamespace(refinement=declaration),
            loaded.document,
            loaded.system_path.parent,
            policy,
        )
    except ValueError as exc:
        return str(exc)
    return ""


def _training(loaded, source, result, number):
    """The source's training problem, its final assignment, the fitted values and residual.

    Peaks are rebuilt from the frozen training data and must equal the source's record.
    """
    problem, record = prepare(loaded)
    artifact = tomllib.loads((source / result["artifacts"]["assignment"]["path"]).read_text())
    peaks, recorded = problem.peaks, artifact["peaks"]
    if (
        [item["column"] for item in recorded] != peaks.column.tolist()
        or [item["centre_ghz"] for item in recorded] != peaks.centre.tolist()
        or [item["uncertainty_ghz"] for item in recorded] != peaks.uncertainty.tolist()
    ):
        raise ValueError("training peaks from the frozen data differ from the source's record")
    index = {member.label: j for j, member in enumerate(loaded.members)}
    index.update(unexplained=UNEXPLAINED, unresolved=UNRESOLVED)
    lines = np.array(
        [index[label] for label in artifact["solutions"][number]["assignment"]], dtype=int
    )
    values = result["solutions"][number]["values"]
    theta = np.array([values[name] for name in loaded.names], dtype=float)
    return problem, record, lines, theta, residual_function(problem, peaks, lines)


def _agreement(report, outcome, fit, tolerance) -> dict:
    """Differences between the recomputed diagnostics and the fit's own report."""
    names = [name for name in report["determined"] if name in fit["uncertainty"]]
    uncertainty = max(
        (abs(report["uncertainty"][n] / fit["uncertainty"][n] - 1) for n in names), default=0.0
    )
    correlation = 0.0
    if names:
        mine = [report["correlation"]["parameters"].index(n) for n in names]
        theirs = [fit["correlation"]["parameters"].index(n) for n in names]
        correlation = float(
            np.max(
                np.abs(
                    np.array(report["correlation"]["matrix"])[np.ix_(mine, mine)]
                    - np.array(fit["correlation"]["matrix"])[np.ix_(theirs, theirs)]
                )
            )
        )
    chi2 = abs(report["statistics"]["chi2"] - fit["statistics"]["chi2"])
    agreement = {
        "outcome_matches": outcome == fit["outcome"],
        "rank_matches": report["rank"] == fit["rank"],
        "determined_matches": report["determined"] == fit["determined"],
        "largest_uncertainty_difference": uncertainty,
        "largest_correlation_difference": correlation,
        "chi2_difference": chi2,
    }
    within = (
        uncertainty <= tolerance
        and correlation <= tolerance
        and chi2 <= tolerance * max(1.0, fit["statistics"]["chi2"])
    )
    if agreement["rank_matches"] and report["undetermined_combinations"]:
        mine = np.array(report["undetermined_combinations"])
        theirs = np.array(fit["undetermined_combinations"])
        # The sine of the largest angle between the two undetermined subspaces.
        difference = float(np.linalg.norm(mine.T @ mine - theirs.T @ theirs, 2))
        agreement["undetermined_difference"] = difference
        within = within and difference <= tolerance
    matches = agreement["outcome_matches"] and agreement["rank_matches"]
    agreement["agrees"] = bool(matches and agreement["determined_matches"] and within)
    return agreement


def _identifiability(result, number, training, policy) -> dict:
    problem, _, _, theta, residual = training
    limits = policy.thresholds["calibration"]
    report = diagnose(
        residual,
        theta,
        problem.scales,
        problem.lower,
        problem.upper,
        problem.names,
        rank_rtol=limits["rank_rtol"],
        undetermined_tolerance=limits["undetermined_tolerance"],
        step=limits["jacobian_step"],
    )
    outcome = report.pop("outcome")
    fit = result["solutions"][number]
    return {
        "name": "identifiability",
        "scope": SCOPES["identifiability"],
        "solution": number,
        "tolerances": {"check": dict(limits), "fit": result["tolerances"]},
        "values": {
            **report,
            "agreement": _agreement(report, outcome, fit, limits["agreement_rtol"]),
        },
    }


def _holdout(loaded, run, request, number, training, policy) -> dict:
    scope = SCOPES["holdout"]
    entry = request.get("holdout")
    if entry is None:
        return inapplicable(
            "holdout", "no holdout dataset supplied with the check request", scope=scope
        )
    dataset = load_dataset(run / "spec" / entry["dataset"])
    if dataset.sha256 != entry["sha256"]:
        raise ValueError("frozen holdout dataset digest mismatch")
    form, reason = holdout_form(loaded, dataset)
    if reason is not None:
        return inapplicable("holdout", reason, scope=scope)
    projection = None
    if form == "map":
        projections = {tuple(item["projection"]) for item in training()[1]["datasets"].values()}
        if len(projections) != 1:
            return inapplicable(
                "holdout",
                "the source learned a different projection for each training map; "
                "a holdout map needs one",
                scope=scope,
            )
        (projection,) = projections
    values = holdout_values(loaded, entry["dataset"], dataset, form, training(), policy, projection)
    item = {
        "name": "holdout",
        "scope": scope,
        "solution": number,
        "dataset": {"file": entry["dataset"], "id": dataset.spec.id, "sha256": dataset.sha256},
        "values": values,
    }
    expectation = entry.get("expectation")
    if expectation is None:
        return item
    coverage = values.get("coverage_over_peaks", {}).get(f"within_{expectation['within']}", 0.0)
    return item | check(
        "holdout",
        coverage,
        expectation["minimum"],
        scope=scope,
        minimum=True,
        failure_meaning="declared expectation violated",
        expectation=expectation,
    )


def holdout_values(loaded, name, dataset, form, training, policy, projection=None) -> dict:
    """Counts, coverages and per-peak predictions of a holdout dataset at the fitted values.

    Maps are extracted under the source's settings and ``projection``; unlabelled peaks are
    matched one to one with the source's cutoff and resolution, in units of each line's
    prediction uncertainty. Nothing is refitted or learned.
    """
    problem, _, _, theta, residual = training
    labels = {member.label: j for j, member in enumerate(loaded.members)}
    extraction = loaded.spec.extraction if form == "map" else None
    columns, _, arrays, failures, _ = dataset_peaks(name, dataset, extraction, labels, projection)
    peaks, count = merge_peaks([arrays]), len(loaded.members)
    limits = policy.thresholds["calibration"]
    spectrum, every = spectrum_for(loaded, columns), range(len(columns))
    # Nothing is refitted: the covariance comes from the training residuals at the fitted
    # values, and each line's gradient carries it to the holdout prediction.
    bounds = (problem.scales, problem.lower, problem.upper)
    _, jacobian = scaled_jacobian(residual, theta, *bounds, limits["jacobian_step"])
    _, _, covariance, undetermined = decompose(jacobian, limits["rank_rtol"])
    predicted, gradient = scaled_jacobian(
        lambda values: spectrum(values, every).ravel(), theta, *bounds, limits["jacobian_step"]
    )
    variance = np.einsum("ij,jk,ik->i", gradient, covariance, gradient).reshape(-1, count)
    # A prediction that moves along an undetermined combination has no finite uncertainty.
    along = np.linalg.norm(gradient @ undetermined.T, axis=1)
    determined = along <= limits["undetermined_tolerance"] * np.linalg.norm(gradient, axis=1)
    determined, predicted = determined.reshape(-1, count), predicted.reshape(-1, count)
    if form == "labelled points":
        lines = peaks.line
    else:
        matching = loaded.spec.matching
        lines, _ = assign(
            peaks,
            predicted,
            range(count),
            np.full(peaks.centre.size, matching.cutoff**2),
            matching.resolution,
            variance=variance,
        )
    assigned = np.flatnonzero(lines >= 0)
    column, line = peaks.column[assigned], lines[assigned]
    known = determined[column, line]
    spread = np.sqrt(peaks.uncertainty[assigned] ** 2 + variance[column, line])
    standardised = (peaks.centre[assigned] - predicted[column, line]) / spread
    scored, total = standardised[known], peaks.centre.size
    within = {k: int(np.sum(np.abs(scored) <= k)) for k in (1, 2)}
    counts = {
        "peaks": total,
        "assigned": int(assigned.size),
        "not_determined": int(np.sum(~known)),
        "unresolved": int(np.sum(lines == UNRESOLVED)),
        "unexplained": int(np.sum(lines == UNEXPLAINED)),
        "scored": int(scored.size),
        **({"extraction_failures": len(failures)} if form == "map" else {}),
    }
    values = {"form": form, "counts": counts}
    if total:
        values["coverage_over_peaks"] = {f"within_{k}": within[k] / total for k in (1, 2)}
    if scored.size:
        values["coverage_over_scored"] = {f"within_{k}": within[k] / scored.size for k in (1, 2)}
        values["mean_squared_standardised_residual"] = float(np.mean(scored**2))
    rows = []
    position = {int(i): n for n, i in enumerate(assigned)}
    for i in range(total):
        row = {
            "column": int(peaks.column[i]),
            "centre_ghz": float(peaks.centre[i]),
            "uncertainty_ghz": float(peaks.uncertainty[i]),
        }
        if i in position:
            n = position[i]
            row.update(
                line=loaded.members[line[n]].label,
                predicted_ghz=float(predicted[column[n], line[n]]),
                determined=bool(known[n]),
            )
            if known[n]:
                row.update(
                    prediction_uncertainty_ghz=float(spread[n]),
                    standardised_residual=float(standardised[n]),
                )
        else:
            row["line"] = "unexplained" if lines[i] == UNEXPLAINED else "unresolved"
        rows.append(row)
    values.update(columns=columns, peaks=rows)
    return values


def _refinement(loaded, number, declaration, training, policy) -> dict:
    scope = SCOPES["refinement"]
    reason = refinement_reason(loaded, declaration, policy)
    if reason:
        return inapplicable("refinement", reason, scope=scope)
    problem, record, lines, theta, _ = training()
    peaks = problem.peaks
    rows = np.flatnonzero(lines >= 0)
    column, line = peaks.column[rows], lines[rows]
    columns = np.unique(column)
    with tempfile.TemporaryDirectory(prefix="entanglix-calibration-refinement-") as directory:
        path = write_refined_system(loaded.inputs, declaration, Path(directory))
        document = tomllib.loads(path.read_text())
        refined = spectrum_for(loaded, record["columns"], document, path)(theta, columns)
        dimensions = list(load_system(path).dimensions)
    original = problem.spectrum(theta, columns)
    shift = refined[column, line] - original[column, line]
    standardised = shift / peaks.uncertainty[rows]
    before = (peaks.centre[rows] - original[column, line]) / peaks.uncertainty[rows]
    after = before - standardised
    values = {
        "dimensions": list(loaded.system.dimensions),
        "refined_dimensions": dimensions,
        "assigned": int(rows.size),
        "largest_standardised_shift": float(np.max(np.abs(standardised), initial=0.0)),
        "largest_shift_ghz": float(np.max(np.abs(shift), initial=0.0)),
        "lines": {
            loaded.members[j].label: float(np.max(np.abs(standardised[line == j])))
            for j in np.unique(line)
        },
        "chi2": float(before @ before),
        "refined_chi2": float(after @ after),
    }
    return {
        "name": "refinement",
        "scope": scope,
        "solution": number,
        "refinement": {name: step.model_dump() for name, step in declaration.items()},
        "values": values,
    }


def calibration_checks(loaded, run, source, request, declaration, policy) -> list[dict]:
    """The requested checks on the named solution, from the frozen inputs and request."""
    from entanglix.runner import _script_result

    result = _script_result(source)
    number = request.get("solution")
    training = functools.cache(lambda: _training(loaded, source, result, number))
    outcomes = []
    for name in request["checks"]:
        if name not in SCOPES:
            item = inapplicable(name, "not applicable to calibrate", scope="source calibration")
        elif number is None:
            item = inapplicable(name, NO_FIT, scope=SCOPES[name])
        elif name == "holdout":
            item = _holdout(loaded, run, request, number, training, policy)
        elif name == "identifiability":
            item = _identifiability(result, number, training(), policy)
        else:
            item = _refinement(loaded, number, declaration, training, policy)
        outcomes.append(item)
    return outcomes
