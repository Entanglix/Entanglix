"""Numerical residuals and scoped reports, shared with engine acceptance tests."""

import math

import numpy as np

from entanglix.corrector import load_policy


def series_checks(
    point_kind,
    metric,
    *,
    monotonic_direction=None,
    policy=None,
):
    """Check an explicitly declared sweep expectation from trusted metric code."""
    from entanglix.corrector import relevant_checks

    relevant_checks(point_kind)
    policy = policy or load_policy()
    values = np.asarray(metric)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("series metric must be a finite one-dimensional array")
    tolerance = policy.thresholds["refinement"][
        "observable_atol" if point_kind == "lindblad" else "excitation_convergence_atol_ghz"
    ]
    scope = "all grid points; declared metric"
    checks = []
    if monotonic_direction is None:
        checks.append(
            inapplicable(
                "declared_monotonicity",
                "no expected direction declared",
                scope=scope,
            )
        )
    elif monotonic_direction in {"increasing", "decreasing"}:
        difference = np.diff(values) * (1 if monotonic_direction == "increasing" else -1)
        checks.append(
            check(
                "declared_monotonicity",
                float(max(0, -difference.min(initial=0))),
                tolerance,
                scope=scope,
                failure_meaning="declared expectation violated",
            )
        )
    else:
        raise ValueError("unknown declared monotonic direction")
    return checks


def physical_residuals(states):
    states = np.asarray(states)
    if states.ndim != 3 or states.shape[1] != states.shape[2] or not np.isfinite(states).all():
        raise ValueError("stored states must be a finite sequence of square density matrices")
    return {
        "trace_error": float(np.max(abs(np.trace(states, axis1=1, axis2=2) - 1))),
        "hermiticity_error": float(
            np.max(np.linalg.norm(states - states.conj().transpose(0, 2, 1), axis=(1, 2)))
        ),
        "minimum_eigenvalue": float(
            np.linalg.eigvalsh((states + states.conj().transpose(0, 2, 1)) / 2).min()
        ),
    }


def check(name, residual, tolerance, *, scope, minimum=False, **details):
    passed = residual >= tolerance if minimum else residual <= tolerance
    return {
        "name": name,
        "scope": scope,
        "outcome": "pass" if passed else "fail",
        "residual": float(residual),
        "tolerance": float(tolerance),
        **details,
    }


def inapplicable(name, reason, *, scope):
    return {"name": name, "scope": scope, "outcome": "inapplicable", "reason": reason}


def output_checks(states, dimensions, *, policy=None):
    policy = policy or load_policy()
    states = np.asarray(states)
    if states.shape[1:] != (math.prod(dimensions),) * 2:
        raise ValueError("stored state dimensions do not match the validated model")
    residuals = physical_residuals(states)
    limits = policy.thresholds["output"]
    scope = "every stored state at every output time"
    checks = [
        check("trace", residuals["trace_error"], limits["trace_atol"], scope=scope),
        check(
            "hermiticity", residuals["hermiticity_error"], limits["hermiticity_atol"], scope=scope
        ),
        check(
            "positivity",
            residuals["minimum_eigenvalue"],
            limits["minimum_eigenvalue"],
            scope=scope,
            minimum=True,
        ),
    ]
    return checks, top_level_populations(states, dimensions, policy=policy)


def top_level_populations(states, dimensions, *, policy):
    limits = policy.thresholds["output"]
    scope = "every stored state at every output time"
    populations = np.diagonal(states, axis1=1, axis2=2).real.reshape((len(states), *dimensions))
    warnings = []
    for axis in range(len(dimensions)):
        top = np.take(populations, dimensions[axis] - 1, axis=axis + 1)
        severity = float(np.max(top.reshape(len(states), -1).sum(axis=1)))
        warnings.append(
            {
                "name": "top_level_population",
                "subsystem_index": axis,
                "scope": scope,
                "population": severity,
                "threshold": limits["top_level_population_warning"],
                "triggered": severity > limits["top_level_population_warning"],
            }
        )
    return warnings


def coverage_report(selection, shape):
    selected = set(selection.points)
    return {
        "point_count": math.prod(shape),
        "evidence_points": list(selection.points),
        "output_checked_points": [
            index for index in range(math.prod(shape)) if index not in selected
        ],
        "selection_reasons": {
            str(index): list(reasons) for index, reasons in selection.reasons.items()
        },
        "uncovered_points": list(selection.uncovered),
        "complete": not selection.uncovered,
    }
