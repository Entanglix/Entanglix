"""Independent replay and declared constraints of neutral control candidates."""

import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tomli_w

from entanglix.corrector import (
    CHECK_DEFINITIONS,
    load_policy,
    relevant_checks,
    validate_refinement,
)
from entanglix.corrector.checks import check, inapplicable
from entanglix.engines import QuTiPIntegrator
from entanglix.evolution import EvolutionInput
from entanglix.models import ConstructedSystem
from entanglix.protocols.control import load_control, override_system
from entanglix.solvers import validate_problem

CONTROL_LIMIT = (
    "Replay tests the declared closed control model, not device-model or rotating-wave accuracy."
)
UNCERTAINTY = ("uncertainty", "uncertainty_refinement")
UNCERTAINTY_SCOPE = (
    "returned neutral pulse on every declared evaluation sample, with the nominal model "
    "reported separately; closed-system evaluations under the rotating-wave approximation"
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


def uncertainty_reason(loaded, name, declaration, policy):
    evaluation = loaded.definition.evaluation
    if evaluation is None:
        return "no evaluation table declared"
    if loaded.system.charge_basis_vectors is None:
        return "state correspondence is supported only for a single periodic circuit coordinate"
    definition = loaded.definition
    if any(
        evaluation.quantities[target] == "calibrated"
        for sample in evaluation.samples
        for target in sample
    ) and (
        not definition.frame.transitions
        or any(channel.follows is None for channel in definition.channels)
    ):
        return "calibrated samples need a named frame transition with every carrier following it"
    if name == "uncertainty_refinement":
        if not declaration:
            return "no refinement declared; a check request may supply one"
        system = loaded.inputs["system.toml"]
        try:
            validate_refinement(
                SimpleNamespace(refinement=declaration),
                tomllib.loads(system.read_text()),
                system.parent,
                policy,
            )
        except ValueError as exc:
            return str(exc)
        for subsystem, numbers in definition.frame.excitation_numbers.items():
            if subsystem in declaration and numbers != list(range(len(numbers))):
                return (
                    "refinement extends only the canonical frame excitation numbers 0, 1, ...; "
                    f"{subsystem} declares {numbers}"
                )
    return ""


def control_listing(loaded, *, policy=None):
    policy = policy or load_policy()
    reason = subset_reason(loaded)
    evaluation = loaded.definition.evaluation
    entries = []
    for name in relevant_checks("control"):
        if name in UNCERTAINTY and evaluation is None:
            continue
        unavailable = reason or (
            "no fidelity target declared"
            if name == "target" and loaded.definition.problem.final_fidelity is None
            else uncertainty_reason(loaded, name, evaluation.refinement, policy)
            if name in UNCERTAINTY
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


def _sampled(loaded, changes, directory, frequencies=None, refinement=None):
    """Build the control model with sampled changes at the given frame and carriers.

    Without frequencies, the named transition and its followers are resolved on this
    model, which then serves as a sample's calibration model.
    """
    document = loaded.definition.model_dump(exclude_none=True)
    # The returned amplitudes are evaluated as they are; nothing is renormalised.
    document.pop("rabi_bounds", None)
    document["parameters"] = {
        **document["parameters"],
        **{
            target.removeprefix("parameters."): value
            for target, value in changes.items()
            if target.startswith("parameters.")
        },
    }
    frame = document["frame"]
    if frequencies is None:
        for name in frame["transitions"]:
            frame["frequencies_ghz"].pop(name)
        for channel in document["channels"]:
            if "follows" in channel:
                del channel["frequency_ghz"]
    else:
        frame["frequencies_ghz"], carriers = frequencies
        frame["transitions"] = {}
        for channel in document["channels"]:
            channel.pop("follows", None)
            channel["frequency_ghz"] = carriers[channel["id"]]
    system = override_system(
        loaded.inputs["system.toml"].read_bytes(),
        {target: value for target, value in changes.items() if target.startswith("model.")},
    )
    if refinement:
        data = tomllib.loads(system.decode("utf-8"))
        step = refinement[data["id"]]
        data["truncation"]["retained_levels"] += step.retained_levels_step
        for name, increment in step.cutoff_steps.items():
            data["truncation"]["cutoffs"][name] += increment
        system = tomli_w.dumps(data).encode("utf-8")
        numbers = frame.get("excitation_numbers", {})
        if data["id"] in numbers:
            # Only the canonical sequence is refined; it extends with the retained levels.
            numbers[data["id"]] = list(range(data["truncation"]["retained_levels"]))
    directory.mkdir()
    (directory / "system.toml").write_bytes(system)
    (directory / "control.toml").write_text(tomli_w.dumps(document))
    return load_control(directory / "control.toml")


def _frequencies(loaded):
    return (
        dict(loaded.definition.frame.frequencies_ghz),
        {channel.id: channel.frequency_ghz for channel in loaded.definition.channels},
    )


def _evaluate(sample, nominal, logical, pulse, policy):
    """Replay on the sample's own eigenstates, phase-aligned to the nominal ones by overlap."""
    vectors = sample.system.charge_basis_vectors
    rows = max(len(vectors), len(nominal))

    def padded(columns):
        # Charge bases of different cutoffs share their central charge state.
        result = np.zeros((rows, columns.shape[1]), dtype=complex)
        start = (rows - len(columns)) // 2
        result[start : start + len(columns)] = columns
        return result

    common = min(vectors.shape[1], nominal.shape[1])
    overlaps = np.einsum(
        "ij,ij->j", padded(nominal)[:, :common].conj(), padded(vectors)[:, :common]
    )
    phases = np.ones(vectors.shape[1], dtype=complex)
    phases[:common] = np.exp(-1j * np.angle(overlaps))
    aligned = replace(
        sample,
        system=replace(
            sample.system,
            drives={
                name: phases.conj()[:, None] * operator * phases[None, :]
                for name, operator in sample.system.drives.items()
            },
        ),
    )
    values = replay_values(aligned, pulse, policy)
    magnitudes = abs(overlaps)
    minimum = policy.thresholds["uncertainty"]["min_state_overlap"]
    return {
        "fidelity": values["fidelity"],
        "leakage": values["leakage"],
        "state_overlaps": magnitudes.tolist(),
        "ambiguous_correspondence": bool(np.any(magnitudes[logical] < minimum)),
        "dimensions": list(sample.system.dimensions),
        "integration": {key: values[key] for key in ("engine", "method", "rtol", "atol")},
    }


def sampled_values(loaded, pulse, policy, refinement=None):
    """Evaluate the nominal model and every sample; refine each at its resolved drive."""
    evaluation = loaded.definition.evaluation
    nominal = loaded.system.charge_basis_vectors
    logical = np.array(loaded.definition.goal.subspace[0]) - 1
    rows = []
    with tempfile.TemporaryDirectory(prefix="entanglix-uncertainty-") as scratch:
        for index, changes in enumerate([{}, *evaluation.samples]):
            calibrated = {
                target: value
                for target, value in changes.items()
                if evaluation.quantities[target] == "calibrated"
            }
            calibration = (
                _sampled(loaded, calibrated, Path(scratch) / f"calibration-{index}")
                if calibrated
                else loaded
            )
            frequencies = _frequencies(calibration)
            identity = {
                "values": changes,
                "calibrated": sorted(calibrated),
                "frame_frequencies_ghz": frequencies[0],
                "carrier_frequencies_ghz": frequencies[1],
            }
            row = {
                **identity,
                **_evaluate(
                    _sampled(loaded, changes, Path(scratch) / f"sample-{index}", frequencies),
                    nominal,
                    logical,
                    pulse,
                    policy,
                ),
            }
            if refinement:
                refined = identity | _evaluate(
                    _sampled(
                        loaded, changes, Path(scratch) / f"refined-{index}", frequencies, refinement
                    ),
                    nominal,
                    logical,
                    pulse,
                    policy,
                )
                refined["fidelity_change"] = refined["fidelity"] - row["fidelity"]
                refined["leakage_change"] = refined["leakage"] - row["leakage"]
                row["refined"] = refined
            rows.append(row)
    return rows


def _minimum(rows):
    index = min(range(len(rows)), key=lambda position: rows[position]["fidelity"])
    return index, rows[index]["fidelity"]


def _ambiguous(rows):
    return [position for position, row in enumerate(rows) if row["ambiguous_correspondence"]]


def uncertainty_items(name, rows, declaration):
    """Report declared samples, indexed in table order; the nominal row is not one of them."""
    nominal, *declared = rows
    coverage = (
        f"all {len(declared)} declared samples; the nominal model is reported separately "
        "and is not part of the minimum"
    )
    samples = [{key: value for key, value in row.items() if key != "refined"} for row in declared]
    index, fidelity = _minimum(samples)
    if name == "uncertainty":
        values = {
            "coverage": coverage,
            "nominal": {key: value for key, value in nominal.items() if key != "refined"},
            "samples": samples,
            "minimum_fidelity": fidelity,
            "minimum_sample": index,
            "ambiguous_samples": _ambiguous(samples),
        }
        return {"name": name, "scope": UNCERTAINTY_SCOPE, "engine": "qutip-cpu", "values": values}
    refined = [row["refined"] for row in declared]
    refined_index, refined_fidelity = _minimum(refined)
    values = {
        "coverage": coverage,
        "nominal": nominal["refined"],
        "samples": refined,
        "minimum_fidelity": refined_fidelity,
        "minimum_sample": refined_index,
        "unrefined_minimum_fidelity": fidelity,
        "unrefined_minimum_sample": index,
        "minimum_change": refined_fidelity - fidelity,
        "largest_sample_change": max(abs(row["fidelity_change"]) for row in refined),
        "ambiguous_samples": _ambiguous(refined),
    }
    return {
        "name": name,
        "scope": UNCERTAINTY_SCOPE + "; each sample keeps its resolved drive at the refinement",
        "engine": "qutip-cpu",
        "refinement": {key: step.model_dump() for key, step in declaration.items()},
        "values": values,
    }


def control_checks(loaded, pulse, result, names, policy, declaration=None):
    reason = subset_reason(loaded)
    evaluation = loaded.definition.evaluation
    if declaration is None:
        declaration = evaluation.refinement if evaluation is not None else {}
    outcomes, values, rows = [], None, None
    for name in names:
        scope = "returned neutral pulse in the declared control model"
        if name not in relevant_checks("control") or reason:
            item = inapplicable(name, reason or "not applicable to control", scope=scope)
        elif name in UNCERTAINTY:
            unavailable = uncertainty_reason(loaded, name, declaration, policy) or pulse_reason(
                loaded, pulse
            )
            if unavailable:
                item = inapplicable(name, unavailable, scope=UNCERTAINTY_SCOPE)
            else:
                if rows is None:
                    refine = "uncertainty_refinement" in names and not uncertainty_reason(
                        loaded, "uncertainty_refinement", declaration, policy
                    )
                    rows = sampled_values(
                        loaded, pulse, policy, refinement=declaration if refine else None
                    )
                item = uncertainty_items(name, rows, declaration)
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
