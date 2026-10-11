"""Independent replay and declared constraints of neutral control candidates."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from entanglix.corrector import CHECK_DEFINITIONS, load_policy, relevant_checks
from entanglix.corrector.checks import check, inapplicable
from entanglix.engines import QuTiPIntegrator
from entanglix.evolution import EvolutionInput
from entanglix.models import ConstructedSystem
from entanglix.protocols.control import load_control
from entanglix.solvers import validate_problem

CONTROL_LIMIT = (
    "Replay tests the declared closed control model, not device-model or rotating-wave accuracy."
)


def load_candidate(root: Path):
    loaded = load_control(root / "control.toml")
    problem = root / "problem.toml"
    if problem.read_bytes() != loaded.problem_bytes:
        raise ValueError("frozen ProblemSpec disagrees with the declared control model")
    return replace(
        loaded,
        inputs={**loaded.inputs, "problem.toml": problem},
        input_digests={**loaded.input_digests, "problem.toml": loaded.digest},
    )


def subset_reason(loaded):
    try:
        validate_problem(loaded.problem_spec, {}, "control checks")
    except ValueError as exc:
        return str(exc)
    return ""


def control_listing(loaded, *, policy=None):
    policy = policy or load_policy()
    reason = subset_reason(loaded)
    entries = []
    for name in relevant_checks("control"):
        unavailable = reason or (
            "no fidelity target declared"
            if name == "target" and loaded.definition.problem.final_fidelity is None
            else ""
        )
        entries.append(
            {
                "name": name,
                "error_mode": CHECK_DEFINITIONS[name],
                "available": not bool(unavailable),
                **({"reason": unavailable} if unavailable else {}),
                **({"engine": "qutip-cpu"} if name != "constraints" else {}),
            }
        )
    return {"checks": entries, "limit": CONTROL_LIMIT}


def pulse_reason(loaded, pulse):
    n = loaded.definition.problem.N
    times, amplitudes = pulse.times_ns, pulse.amplitudes_ghz
    if pulse.kind != "zero_order":
        return "only zero_order pulses are covered"
    if times.shape != (n,) or amplitudes.shape != (len(loaded.definition.channels), n):
        return "neutral pulse shape does not match the declared channels and knot count"
    if not np.isfinite(times).all() or not np.isfinite(amplitudes).all():
        return "neutral pulse contains non-finite values"
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        return "neutral pulse times must start at zero and increase"
    return ""


def constraints(loaded, pulse, policy):
    scope = "returned knots, including terminal amplitudes and differences"
    reason = pulse_reason(loaded, pulse)
    if reason:
        return {
            "name": "constraints",
            "outcome": "fail",
            "scope": scope,
            "checks": [check("pulse_structure", 1, 0, scope=scope, diagnostic=reason)],
        }
    problem = loaded.definition.problem
    limits = policy.thresholds["control"]
    amplitudes, times = pulse.amplitudes_ghz, pulse.times_ns
    bounds = np.array([channel.amplitude_bound_ghz for channel in loaded.definition.channels])
    maxima = np.max(abs(amplitudes), axis=1)
    steps = np.diff(times)
    step_lower, step_upper = loaded.problem_spec["problem"]["free_dt"]
    duration_lower, duration_upper = (problem.N - 1) * np.array([step_lower, step_upper])
    # Subtracting knot times introduces roundoff at the scale of the whole pulse.
    time_slack = 32 * np.finfo(float).eps * max(abs(times[-1]), duration_upper)
    items = [
        check(
            "amplitude",
            max(0, float(np.max(maxima - bounds))),
            limits["amplitude_atol_ghz"],
            scope=scope,
            maxima_ghz=maxima.tolist(),
            bounds_ghz=bounds.tolist(),
        ),
        check(
            "duration",
            max(0, float(duration_lower - times[-1]), float(times[-1] - duration_upper)),
            time_slack,
            scope="duration allowed by the frozen time-step bracket",
            duration_ns=float(times[-1]),
            declared_ns=loaded.definition.pulse.T,
            bounds_ns=[float(duration_lower), float(duration_upper)],
        ),
        check(
            "equal_steps",
            max(0, float(np.max(step_lower - steps)), float(np.max(steps - step_upper))),
            time_slack,
            scope="all propagation intervals within the frozen time-step bracket",
            bounds_ns=[step_lower, step_upper],
        ),
    ]
    derivatives = np.diff(amplitudes, axis=1) / steps
    for name, values, tolerance in (
        ("du", derivatives, limits["du_atol_ghz_per_ns"]),
        (
            "ddu",
            np.diff(derivatives, axis=1) / ((steps[:-1] + steps[1:]) / 2),
            limits["ddu_atol_ghz_per_ns2"],
        ),
    ):
        bound = getattr(problem, name + "_bound")
        if bound is not None:
            maximum = float(np.max(abs(values), initial=0))
            items.append(
                check(
                    name,
                    max(0, maximum - bound),
                    tolerance,
                    scope=scope,
                    maximum=maximum,
                    bound=bound,
                )
            )
    return {
        "name": "constraints",
        "outcome": "fail" if any(item["outcome"] == "fail" for item in items) else "pass",
        "scope": scope,
        "checks": items,
        "uncovered_constraints": [],
    }


def replay_values(loaded, pulse, policy):
    """QuTiP evolves an entangled input to retain relative column phases and leakage."""
    indices = np.array(loaded.definition.goal.subspace[0]) - 1
    d = len(indices)
    n = len(loaded.system.hamiltonian_ghz)
    identity = np.eye(d)
    model = ConstructedSystem(
        id=loaded.system.id,
        hamiltonian_ghz=np.kron(
            loaded.system.hamiltonian_ghz - loaded.listing["energy_origin_ghz"] * np.eye(n),
            identity,
        ),
        operators={},
        dissipators={},
        dimensions=(n, d),
        basis="declared-control-frame",
        provenance={},
        drives={
            name: np.kron(operator, identity) for name, operator in loaded.system.drives.items()
        },
    )
    initial = np.zeros((n, d), dtype=complex)
    initial[indices, np.arange(d)] = 1 / np.sqrt(d)
    target = np.zeros_like(initial)
    target[indices] = loaded.definition.goal.operator() / np.sqrt(d)
    limits = policy.thresholds["control"]
    trace = QuTiPIntegrator().evolve(
        model,
        EvolutionInput(
            initial.ravel(),
            pulse.times_ns,
            drive_amplitudes_ghz={
                channel.id: pulse.amplitudes_ghz[index, :-1]
                for index, channel in enumerate(loaded.definition.channels)
            },
            rtol=limits["rtol"],
            atol=limits["atol"],
            max_steps=limits["max_steps"],
        ),
    )
    rho = trace.states[-1]
    survival = float(np.diag(rho).real[np.repeat(np.isin(np.arange(n), indices), d)].sum())
    overlap = float((target.ravel().conj() @ rho @ target.ravel()).real)
    return {
        "fidelity": (survival + d * overlap) / (d + 1),
        "leakage": 1 - survival,
        "engine": trace.engine,
        "method": trace.method,
        "rtol": limits["rtol"],
        "atol": limits["atol"],
    }


def control_checks(loaded, pulse, result, names, policy):
    reason = subset_reason(loaded)
    outcomes, values = [], None
    for name in names:
        scope = "returned neutral pulse in the declared control model"
        if name not in relevant_checks("control") or reason:
            item = inapplicable(name, reason or "not applicable to control", scope=scope)
        elif name == "constraints":
            item = constraints(loaded, pulse, policy)
        elif name == "target" and loaded.definition.problem.final_fidelity is None:
            item = inapplicable(name, "no fidelity target declared", scope=scope)
        elif pulse_reason(loaded, pulse):
            item = inapplicable(name, pulse_reason(loaded, pulse), scope=scope)
        else:
            if values is None:
                values = replay_values(loaded, pulse, policy)
            reported = result.get("metrics", {}).get("solver_reported_fidelity")
            if name == "replay" and (
                type(reported) not in (int, float) or not np.isfinite(reported)
            ):
                item = inapplicable(name, "no finite solver-reported fidelity", scope=scope)
            elif name == "replay":
                item = check(
                    name,
                    abs(values["fidelity"] - reported),
                    policy.thresholds["control"]["fidelity_atol"],
                    scope=scope,
                    solver_reported_fidelity=reported,
                    **values,
                )
            else:
                item = check(
                    name,
                    values["fidelity"],
                    loaded.definition.problem.final_fidelity,
                    minimum=True,
                    scope=scope,
                    failure_meaning="unmet declared target",
                    **values,
                )
        outcomes.append(item)
    return outcomes
