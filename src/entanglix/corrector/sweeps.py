"""Requested checks and explicit coverage for retained serial grids."""

from dataclasses import replace

import numpy as np

from entanglix.corrector import CHECK_DEFINITIONS, load_policy, select_evidence
from entanglix.corrector.checks import check, inapplicable, series_checks
from entanglix.corrector.dynamics import check_listing
from entanglix.runner import _load_record


def sweep_listing(loaded):
    policy = load_policy()
    listing = check_listing(loaded.points[0], policy=policy)
    corners = int(np.prod([1 if size == 1 else 2 for size in loaded.manifest["shape"]]))
    for entry in listing["checks"]:
        if entry["name"] in {"segments", "spectral_refinement"}:
            entry["selection"] = (
                "Selected after completion: corners, optional metric extrema and steepest change; "
                "warnings when diagnostics are requested"
            )
            entry["maximum_points"] = policy.thresholds["evidence"]["max_points"]
            if corners > entry["maximum_points"]:
                entry.update(
                    available=False,
                    reason="grid corners exceed the verification evidence-point bound",
                )
    if loaded.definition.expectation:
        listing["checks"].append(
            {
                "name": "declared_expectation",
                "error_mode": CHECK_DEFINITIONS["declared_expectation"],
                "available": True,
            }
        )
    listing["execution"].update(point_count=len(loaded.points), serial=True)
    if loaded.definition.timeout_seconds:
        listing["execution"]["timeout_seconds"] = loaded.definition.timeout_seconds
    listing["grid"] = {
        "shape": loaded.manifest["shape"],
        "axes": [axis.model_dump() for axis in loaded.definition.axes],
    }
    return listing


def _point_output(source, index):
    directory = source / "artifacts/points" / f"{index:06d}"
    result = _load_record(directory / "result.toml")
    with np.load(directory / "output.npz", allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    return result, arrays


def sweep_checks(loaded, source, request, declaration, policy):
    from entanglix.corrector.dynamics import diagnostics, refined_segments
    from entanglix.corrector.verification import integrity, spectral_refinement

    shape = tuple(loaded.manifest["shape"])
    with np.load(source / "artifacts/output.npz", allow_pickle=False) as archive:
        grids = {name: np.array(archive[name]) for name in archive.files}
    metric = grids[loaded.definition.evidence_metric] if loaded.definition.evidence_metric else None
    if metric is not None and (metric.shape != shape or not np.isfinite(metric).all()):
        raise ValueError("stored sweep metric does not match the declared grid")
    effective = [
        replace(point, definition=point.definition.model_copy(update={"refinement": declaration}))
        for point in loaded.points
    ]
    diagnostic_rows = []
    warnings = None
    if "diagnostics" in request["checks"] and loaded.manifest["kind"] == "lindblad":
        warning_values = []
        for index, point in enumerate(effective):
            _, arrays = _point_output(source, index)
            item = diagnostics(point, arrays, policy)
            diagnostic_rows.append({"index": index, **item})
            warning_values.append(
                max(
                    (
                        w["population"] if w["triggered"] else 0
                        for w in item["values"]["top_level_population"]
                    ),
                    default=0,
                )
            )
        warnings = np.array(warning_values).reshape(shape)
    selection, reason = None, ""
    if set(request["checks"]) & {"segments", "spectral_refinement"}:
        try:
            selection = select_evidence(shape, metric=metric, warnings=warnings, policy=policy)
        except ValueError as exc:
            reason = str(exc)
    outcomes = []
    for name in request["checks"]:
        dynamics = loaded.manifest["kind"] == "lindblad"
        if (name in {"integrity", "segments", "diagnostics"} and not dynamics) or (
            name == "spectral_refinement" and dynamics
        ):
            item = inapplicable(
                name, f"not applicable to {loaded.manifest['kind']}", scope="sweep computation"
            )
        elif name == "declared_expectation":
            expected = loaded.definition.expectation
            if expected is None:
                item = inapplicable(name, "no expectation declared in the sweep spec", scope="grid")
            else:
                with np.load(source / "artifacts/output.npz", allow_pickle=False) as archive:
                    values = np.array(archive[expected.metric])
                axis = next(
                    i for i, value in enumerate(loaded.definition.axes) if value.id == expected.axis
                )
                if values.shape != shape:
                    raise ValueError("expectation metric does not match the declared grid")
                grid_axis = loaded.definition.axes[axis].values
                order = np.argsort(grid_axis)
                lines = np.moveaxis(values, axis, -1).reshape(-1, shape[axis])[:, order]
                checks = [
                    dict(
                        series_checks(
                            loaded.manifest["kind"],
                            line,
                            monotonic_direction=expected.direction,
                            policy=policy,
                        )[0],
                        line_index=index,
                    )
                    for index, line in enumerate(lines)
                ]
                item = {
                    "name": name,
                    "outcome": "fail" if any(c["outcome"] == "fail" for c in checks) else "pass",
                    "scope": "all grid lines along the declared axis, ordered by its numeric value",
                    "metric": expected.metric,
                    "axis": expected.axis,
                    "direction": expected.direction,
                    "checks": checks,
                }
        elif name == "diagnostics":
            item = {
                "name": name,
                "scope": "every retained point; quantities are descriptive only",
                "points": diagnostic_rows,
            }
        elif name in {"segments", "spectral_refinement"} and reason:
            item = inapplicable(name, reason, scope="bounded grid evidence")
        else:
            indices = range(len(effective)) if name == "integrity" else selection.points
            rows = []
            for index in indices:
                result, arrays = _point_output(source, index)
                point = effective[index]
                if result["index"] != index or result["plan_sha256"] != loaded.points[index].digest:
                    raise ValueError("point result does not match its frozen plan")
                item = (
                    integrity(point, arrays, result, policy)
                    if name == "integrity"
                    else refined_segments(point, arrays, policy)
                    if name == "segments"
                    else spectral_refinement(point, arrays, policy)
                )
                if name == "integrity":
                    coordinate = tuple(loaded.manifest["points"][index]["coordinate"])
                    aggregate = _load_record(source / "result.toml")["metrics"]
                    for metric_name in point.definition.metrics:
                        values = grids[metric_name]
                        reported = np.asarray(aggregate[metric_name])
                        residual = (
                            float("inf")
                            if values.shape != shape or reported.shape != shape
                            else max(
                                abs(result["metrics"][metric_name] - values[coordinate]),
                                abs(reported[coordinate] - values[coordinate]),
                            )
                        )
                        item["checks"].append(
                            check(
                                "grid_metric_consistency",
                                residual,
                                policy.thresholds["refinement"]["observable_atol"],
                                scope="point result, aggregate array and reported metric grid",
                                metric=metric_name,
                            )
                        )
                    if any(check["outcome"] == "fail" for check in item["checks"]):
                        item["outcome"] = "fail"
                rows.append(
                    {
                        "index": index,
                        "coordinate": loaded.manifest["points"][index]["coordinate"],
                        **item,
                    }
                )
            item = {
                "name": name,
                "outcome": "fail"
                if any(row["outcome"] == "fail" for row in rows)
                else "inapplicable"
                if any(row["outcome"] == "inapplicable" for row in rows)
                else "pass",
                "scope": "all retained points"
                if name == "integrity"
                else "selected points only; no convergence claim at unsampled points",
                "points": rows,
            }
        checked = {
            row["index"] for row in item.get("points", []) if row.get("outcome") != "inapplicable"
        }
        if name == "declared_expectation" and item.get("outcome") in {"pass", "fail"}:
            checked = set(range(len(effective)))
        item["coverage"] = {
            "point_count": len(effective),
            "checked_points": sorted(checked),
            "uncovered_points": [i for i in range(len(effective)) if i not in checked],
            "selection_reasons": {str(i): list(selection.reasons[i]) for i in selection.points}
            if selection and name in {"segments", "spectral_refinement"}
            else {},
            "uncovered_warning_points": [
                int(i) for i in np.flatnonzero(warnings.ravel()) if i not in checked
            ]
            if warnings is not None and name in {"segments", "spectral_refinement"}
            else [],
        }
        outcomes.append(item)
    return outcomes
