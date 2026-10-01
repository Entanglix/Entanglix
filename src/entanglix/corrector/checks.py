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
    """Stack checks called by trusted metric code, never by candidate exemptions.

    Monotonic direction is supplied by the protocol's metric implementation, not
    read from a system or result. Protocol compilation supplies this in sweeps.
    """
    from entanglix.corrector import required_checks

    required_checks("parameter_series", point_kind=point_kind)
    policy = policy or load_policy()
    values = np.asarray(metric)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("series metric must be a finite one-dimensional array")
    tolerance = policy.thresholds["reference"][
        "observable_atol" if point_kind == "lindblad" else "excitation_convergence_atol_ghz"
    ]
    scope = "all grid points; declared metric"
    checks = [
        inapplicable(
            "seed_reproducibility", "deterministic computation has no random seed", scope=scope
        )
    ]
    if monotonic_direction is None:
        checks.append(
            inapplicable(
                "declared_monotonicity",
                "trusted metric definition predicts no monotonicity",
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
            )
        )
    else:
        raise ValueError("unknown trusted monotonic direction")
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
    return checks, warnings


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


def series_verdict(output_reports, evidence_reports, selection, *, point_count):
    """A bounded selection cannot silently certify missing or failed evidence."""
    passed = (
        len(output_reports) == point_count
        and not selection.uncovered
        and all(
            report and all(item["outcome"] in {"pass", "inapplicable"} for item in report)
            for report in output_reports
        )
    )
    passed &= all(
        bool(evidence_reports.get(index))
        and all(item["outcome"] in {"pass", "inapplicable"} for item in evidence_reports[index])
        for index in selection.points
    )
    return "VERIFIED" if passed else "UNVERIFIED"


def spectral_checks(system, energies, *, policy=None):
    policy = policy or load_policy()
    limits = policy.thresholds["reference"]
    declared = np.linalg.eigvalsh(system.hamiltonian_ghz)
    checks = [
        check(
            "spectral_consistency",
            float(np.max(abs(energies - declared[: len(energies)]))),
            limits["spectral_atol_ghz"],
            scope="every reported energy; consistency with the declared Hamiltonian",
            reference_engine="numpy-eigvalsh",
            reference_dimension=len(declared),
        )
    ]
    branches = system.provenance.get("converted_branches", {})
    capacitors = [
        branch["supplied"] for branch in branches.values() if branch["supplied"]["kind"] == "C"
    ]
    inductors = [
        branch["supplied"] for branch in branches.values() if branch["supplied"]["kind"] == "L"
    ]
    junctions = [branch for branch in branches.values() if branch["supplied"]["kind"] == "JJ"]
    if (
        len(system.provenance.get("transformation", [])) == 1
        and len(junctions) == 1
        and len(inductors) <= 1
        and len(branches) == len(capacitors) + len(inductors) + 1
    ):
        import scqubits as scq

        ec = 1 / sum(
            1 / branch["charging_energy_ghz"]
            for branch in branches.values()
            if "charging_energy_ghz" in branch
        )
        ej = junctions[0]["josephson_energy_ghz"]
        cutoff = system.provenance["cutoffs"]["theta1"]
        if inductors:
            el = next(
                branch["inductive_energy_ghz"]
                for branch in branches.values()
                if branch["supplied"]["kind"] == "L"
            )
            flux = system.provenance["oriented_loops"][0]["value"]
            reference = scq.Fluxonium(EJ=ej, EC=ec, EL=el, flux=flux, cutoff=cutoff)
            engine = "scqubits-fluxonium"
        else:
            reference = scq.Transmon(
                EJ=ej, EC=ec, ng=system.provenance["coordinate_offsets"][0], ncut=cutoff
            )
            engine = "scqubits-transmon"
        expected = reference.eigenvals(evals_count=len(energies))
        checks.append(
            check(
                "spectral_reference",
                float(np.max(abs(energies - energies[0] - expected + expected[0]))),
                limits["spectral_atol_ghz"],
                scope="every excitation energy; independent single-mode constructor",
                reference_engine=engine,
                reference_dimension=len(expected),
            )
        )
    if (
        len(system.dimensions) == 1
        and len(branches) == 2
        and len(capacitors) == len(inductors) == 1
        and "capacitance" in capacitors[0]
        and "inductance" in inductors[0]
    ):

        def value(quantity):
            return (
                system.provenance["parameters"][quantity["parameter"]]["value"]
                if "parameter" in quantity
                else quantity["value"]
            )

        capacitance = value(capacitors[0]["capacitance"]) * 1e-15
        inductance = value(inductors[0]["inductance"]) * 1e-9
        frequency = 1 / (2 * np.pi * np.sqrt(capacitance * inductance)) / 1e9
        expected = np.arange(len(energies)) * frequency
        checks.append(
            check(
                "spectral_reference",
                float(np.max(abs(energies - energies[0] - expected))),
                limits["spectral_atol_ghz"],
                scope="every LC excitation energy; supplied C and L",
                reference_engine="analytic-LC",
                reference_dimension=len(expected),
            )
        )
    if not any(item["name"] == "spectral_reference" for item in checks):
        checks.append(
            inapplicable(
                "spectral_reference",
                "no registered constructor or analytic reference for this model",
                scope="model spectrum",
            )
        )
    return checks
