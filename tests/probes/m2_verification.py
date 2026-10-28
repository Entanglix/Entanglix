"""Measure candidate verification checks; no production policy or verdict changes.

Run with CPU JAX and one BLAS thread. Existing 5B summaries and 5C retained
trajectories supply the comparison baselines; their full references are not rerun.
The 5C cases require the git-ignored research_runs/ source and reference records
named by m2_simulation_cpu.json and m2_simulation_cuda.json.
Segment subprocesses have a hard wall timeout, including imports and compilation.
Their cold-process costs are not intrinsic or warmed solver timings. Propagator
costs include full-step and half-step exponentials with one BLAS thread.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import qutip as qt
import tomli_w
from m2_feasibility import ROOT, construct, spectrum
from scipy.linalg import expm

sys.path.insert(0, str(ROOT / "tests"))
from engine_cases import exact_states  # noqa: E402

from entanglix.corrector.checks import output_checks
from entanglix.corrector.reference import embed_state
from entanglix.engines import Carrier, EvolutionInput, QuTiPIntegrator, RotatingFrame
from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator
from entanglix.engines.frames import matrix, prepare_dynamics
from entanglix.evolution import normalize_input
from entanglix.models import ConstructedSystem, load_system
from entanglix.protocols.plans import load_plan

SEGMENT_TIMEOUT_SECONDS = 12.0
NUMERICAL_TARGET = 1e-6
MODEL_SECONDS = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(function, *args):
    started = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - started


def trace_norm(array):
    return np.linalg.svd(array, compute_uv=False).sum(axis=-1)


def density(state):
    return np.outer(state, state.conj()) if state.ndim == 1 else state


def embedding(system, refined):
    indices = np.ravel_multi_index(
        np.unravel_index(np.arange(np.prod(system.dimensions)), system.dimensions),
        refined.dimensions,
    )
    phases = embed_state(system, refined, np.ones(len(indices)))[indices]
    result = np.zeros((np.prod(refined.dimensions), len(indices)), dtype=complex)
    result[indices, np.arange(len(indices))] = phases
    return result


def metrics(system, states):
    """Probe formulas, including coherence; no new public metric kind is implied."""
    states = np.asarray(states)
    dimensions = system.dimensions
    if len(dimensions) == 2:
        nq, nr = dimensions
        reduced = np.trace(states.reshape(-1, nq, nr, nq, nr), axis1=2, axis2=4)
        return {
            "coherence": 2 * abs(reduced[:, 0, 1]),
            "coherence_real": 2 * reduced[:, 0, 1].real,
            "excited": reduced[:, 1, 1].real,
            "photons": np.einsum("tij,ji->t", states, system.operators["r.number"]).real,
        }
    result = {
        "excited": states[:, 1, 1].real,
        "coherence": 2 * abs(states[:, 0, 1]),
        "coherence_real": 2 * states[:, 0, 1].real,
        "number": np.einsum("tii,i->t", states, np.arange(dimensions[0], dtype=float)).real,
    }
    if dimensions[0] > 2:
        result["level2"] = states[:, 2, 2].real
    return result


def residuals(system, states, refined, target):
    original, expected = metrics(system, states), metrics(refined, target)
    return {
        name: {
            "endpoint": float(abs(values[-1] - expected[name][-1])),
            "trajectory": float(np.max(abs(values - expected[name]))),
        }
        for name, values in original.items()
    }


def metric_conversion(system, state_error, trace_error):
    # Center each real linear observable. The trace correction keeps the tighter
    # half-range factor valid when stored states have numerical trace drift.
    axis = system.subsystem_ids.index("r") if "r" in system.subsystem_ids else 0
    number_max = system.dimensions[axis] - 1
    result = {
        "excited": 0.5 * state_error + 0.5 * trace_error,
        "photons" if len(system.dimensions) == 2 else "number": (
            0.5 * number_max * (state_error + trace_error)
        ),
        # Triangle inequality for the probe's nonlinear 2|rho_01| formula.
        "coherence": 2 * state_error,
        "coherence_real": state_error,
    }
    if len(system.dimensions) == 1 and system.dimensions[0] > 2:
        result["level2"] = 0.5 * (state_error + trace_error)
    return result


def generator(system, prepared, amplitudes):
    hamiltonian = matrix(prepared.hamiltonian).copy()
    for name, amplitude in amplitudes.items():
        hamiltonian += amplitude * matrix(prepared.drives[name])
    collapse = [qt.Qobj(matrix(terms)) for terms in prepared.collapse]
    return qt.liouvillian(qt.Qobj(2 * np.pi * hamiltonian), collapse).full()


def propagator_check(system, request, stored):
    started = time.perf_counter()
    normalized = normalize_input(system, request)
    prepared = prepare_dynamics(system, normalized)
    active_carriers = normalized.frame is None and any(
        np.any(normalized.drive_amplitudes_ghz.get(name, 0)) and carrier.frequency_ghz != 0
        for name, carrier in normalized.carriers.items()
    )
    if active_carriers:
        return {"applicable": False, "reason": "continuous carrier in the lab frame"}
    states = np.asarray(stored)
    if prepared.frame_levels_ghz is not None:
        phase = np.exp(2j * np.pi * normalized.times_ns[:, None] * prepared.frame_levels_ghz)
        states = phase[:, :, None] * states * phase.conj()[:, None, :]
    vectors = states.transpose(0, 2, 1).reshape(len(states), -1)
    groups = {}
    for index in range(len(states) - 1):
        amplitudes = tuple(sorted(normalized.amplitudes_for_interval(index).items()))
        groups.setdefault(amplitudes, []).append(index)
    if len(groups) > 4:
        return {"applicable": False, "reason": "more than four distinct generators"}
    duration = float(normalized.times_ns[1] - normalized.times_ns[0])
    defects, evaluation_disagreements = [], []
    full_step_seconds, half_step_seconds = 0.0, 0.0
    for amplitudes, indices in groups.items():
        liouvillian = generator(system, prepared, dict(amplitudes))
        began = time.perf_counter()
        full = expm(duration * liouvillian)
        full_step_seconds += time.perf_counter() - began
        began = time.perf_counter()
        half = expm(duration * liouvillian / 2)
        half_step_seconds += time.perf_counter() - began
        predicted = vectors[indices] @ full.T
        second = (vectors[indices] @ half.T) @ half.T
        shape = (len(indices), states.shape[1], states.shape[2])
        defects.extend(trace_norm((vectors[np.array(indices) + 1] - predicted).reshape(shape)))
        evaluation_disagreements.extend(trace_norm((predicted - second).reshape(shape)))
    initial = float(trace_norm(states[0] - prepared.initial_density(normalized)))
    total = initial + float(np.sum(defects))
    floating = float(np.sum(evaluation_disagreements))
    trace_error = float(np.max(abs(np.trace(states, axis1=1, axis2=2) - 1)))
    return {
        "applicable": True,
        "scope": "same declared generator, all stored intervals; exact-arithmetic bound",
        "dimension": states.shape[1],
        "distinct_generators": len(groups),
        "initial_state_term": initial,
        "maximum_defect": float(max(defects)),
        "state_error_bound": total,
        "floating_point_disagreement_estimate": floating,
        "floating_point_caveat": "full step versus two half steps is not a certified roundoff bound",
        "metric_error_bound": metric_conversion(system, total, trace_error),
        "metric_floating_estimate": metric_conversion(system, floating, 0),
        "wall_seconds": time.perf_counter() - started,
        "exponential_seconds": full_step_seconds + half_step_seconds,
        "full_step_exponential_seconds": full_step_seconds,
        "half_step_exponential_seconds": half_step_seconds,
        "cost_scope": "one BLAS thread; generator construction, full-step expm, a second half-step expm and two-half-step products for the floating-point estimate, all interval products and trace norms",
    }


def static_screen(system, refined, request):
    started = time.perf_counter()
    energies, vectors = np.linalg.eigh(system.hamiltonian_ghz)
    target_energies, target_vectors = np.linalg.eigh(refined.hamiltonian_ghz)
    mapping = embedding(system, refined)
    overlaps = abs(target_vectors.conj().T @ mapping @ vectors) ** 2
    matched = overlaps.argmax(axis=0)
    weights = np.einsum("ij,ji->i", vectors.conj().T @ density(request.initial_state), vectors).real
    occupied = np.flatnonzero(weights > 1e-5)
    ground = int(matched[0])
    shifts = abs(
        energies[occupied]
        - energies[0]
        - target_energies[matched[occupied]]
        + target_energies[ground]
    )
    outside = 1 - np.sum(abs(mapping.conj().T @ target_vectors[:, matched[occupied]]) ** 2, axis=0)
    driven = any(np.any(value) for value in request.drive_amplitudes_ghz.values())
    report = {
        "wall_seconds": time.perf_counter() - started,
        "scope": "initially occupied undriven Hamiltonian eigenstates",
        "occupied_weight_threshold": 1e-5,
        "minimum_assignment_overlap": float(overlaps[matched[occupied], occupied].min()),
        "maximum_excitation_shift_ghz": float(shifts.max()),
        "maximum_outside_retained_weight": float(max(0, outside.max())),
        "dynamics_accuracy_estimate": None,
        "unresolved": "static eigensystem alone does not establish open/driven dynamics convergence",
        "driven": driven,
    }
    if request.frame is not None:
        normalized = normalize_input(system, request)
        prepared = prepare_dynamics(system, normalized)
        # Undo the frame-energy subtraction to compare physical full and RWA H0.
        rwa = matrix(prepared.hamiltonian) + np.diag(prepared.frame_levels_ghz)
        rwa_energies, rwa_vectors = np.linalg.eigh(rwa)
        overlap = abs(vectors.conj().T @ rwa_vectors) ** 2
        matched_rwa = overlap.argmax(axis=1)
        report["rwa_excitation_shift_ghz"] = float(
            abs(
                energies[occupied]
                - energies[0]
                - rwa_energies[matched_rwa[occupied]]
                + rwa_energies[matched_rwa[0]]
            ).max()
        )
        if (
            not driven
            and len(system.dimensions) == 2
            and density(request.initial_state)[system.dimensions[1], system.dimensions[1]].real
            > 0.99
        ):
            # Narrow Purcell candidate, not a general decay-rate estimator.
            excited = system.dimensions[1]
            index = int(abs(vectors[excited]).argmax())
            counterpart = int(matched_rwa[index])
            photons = system.operators["r.number"]
            full_n = float(np.vdot(vectors[:, index], photons @ vectors[:, index]).real)
            rwa_n = float(
                np.vdot(rwa_vectors[:, counterpart], photons @ rwa_vectors[:, counterpart]).real
            )
            rate = next(iter(system.dissipators.values()))[1]
            duration = float(request.times_ns[-1] - request.times_ns[0])
            target_index = int(matched[index])
            target_photons = refined.operators["r.number"]
            refined_n = float(
                np.vdot(
                    target_vectors[:, target_index],
                    target_photons @ target_vectors[:, target_index],
                ).real
            )
            report["purcell_candidate"] = {
                "assumptions": "undriven weak-loss qubit-like dressed excitation, resonator loss only",
                "full_photon_admixture": full_n,
                "rwa_photon_admixture": rwa_n,
                "rate_ratio": full_n / rwa_n,
                "endpoint_estimate": abs(
                    np.exp(-rate * full_n * duration) - np.exp(-rate * rwa_n * duration)
                ),
                "claim": "model-specific estimate, not a bound",
                "refined_full_photon_admixture": refined_n,
                "truncation_endpoint_estimate": abs(
                    np.exp(-rate * full_n * duration) - np.exp(-rate * refined_n * duration)
                ),
            }
    report["refined_model_construction_seconds"] = (
        MODEL_SECONDS.get(id(refined), 0.0) if refined is not system else 0.0
    )
    report["wall_seconds"] = (
        time.perf_counter() - started + report["refined_model_construction_seconds"]
    )
    return report


def segment_worker(connection, system, request, engine):
    try:
        adapter = DynamiqsIntegrator("cpu") if engine == "dynamiqs" else QuTiPIntegrator()
        trace, seconds = timed(adapter.evolve, system, request)
        connection.send({"states": np.asarray(trace.states), "solve_seconds": seconds})
    except Exception as exc:
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def segment_checks(
    system,
    refined,
    request,
    states,
    engine,
    *,
    transition_aware=False,
    same_frame=False,
    timeout_seconds=SEGMENT_TIMEOUT_SECONDS,
):
    count = len(request.times_ns) - 1
    width = min(2, count)
    starts = {0, (count - width) // 2, count - width}
    if transition_aware:
        for value in request.drive_amplitudes_ghz.values():
            if np.ndim(value):
                starts.update(int(index) for index in np.flatnonzero(np.diff(value)) + 1)
        starts = {min(max(0, index - 1), count - width) for index in starts}
    mapping = embedding(system, refined)
    reports = []
    for first in sorted(starts):
        last = first + width
        initial = mapping @ states[first] @ mapping.conj().T
        amplitudes = {
            name: value[first:last] if np.ndim(value) else value
            for name, value in request.drive_amplitudes_ghz.items()
        }
        segment = replace(
            request,
            initial_state=initial,
            times_ns=request.times_ns[first : last + 1],
            frame=request.frame if same_frame else None,
            observables={},
            drive_amplitudes_ghz=amplitudes,
            rtol=1e-11,
            atol=1e-13,
            max_steps=2_000_000,
        )
        parent, child = mp.get_context("spawn").Pipe(duplex=False)
        process = mp.get_context("spawn").Process(
            target=segment_worker, args=(child, refined, segment, engine)
        )
        began = time.perf_counter()
        process.start()
        child.close()
        try:
            payload = parent.recv() if parent.poll(timeout_seconds) else None
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            parent.close()
        report = {
            "start_ns": float(request.times_ns[first]),
            "stop_ns": float(request.times_ns[last]),
            "wall_seconds": time.perf_counter() - began,
            "hard_timeout_seconds": timeout_seconds,
            "engine": engine,
            "outcome": "allowance_exhausted"
            if payload is None
            else "error"
            if "error" in payload
            else "measured",
        }
        if payload is not None and "states" in payload:
            report["solve_seconds"] = payload["solve_seconds"]
            report["startup_and_transport_seconds"] = (
                report["wall_seconds"] - payload["solve_seconds"]
            )
            report["local_metric_residuals"] = residuals(
                system, states[first : last + 1], refined, payload["states"]
            )
        elif payload:
            report["error"] = payload["error"]
        reports.append(report)
    covered, completed = set(), set()
    for first, report in zip(sorted(starts), reports):
        covered.update(range(first, first + width))
        if report["outcome"] == "measured":
            completed.update(range(first, first + width))
    return {
        "scope": "local restarts from source states; no full-window bound",
        "reference_frame": "same rotating-rwa" if same_frame else "lab",
        "cost_scope": "fresh spawn per segment; wall time includes process startup, imports, serialization and cleanup; solve_seconds includes first-use JAX/dynamiqs compilation and is not a warmed intrinsic solver cost",
        "selection": "start/middle/end plus drive transitions"
        if transition_aware
        else "start/middle/end",
        "attempted_window_fraction": len(covered) / count,
        "completed_window_fraction": len(completed) / count,
        "wall_seconds": sum(item["wall_seconds"] for item in reports),
        "segments": reports,
    }


def evaluate(
    name, system, refined, request, states, source_seconds, baseline, *, engine="dynamiqs"
):
    checks, integrity_seconds = timed(output_checks, states, system.dimensions)
    print(f"{name}: integrity, static screen, propagator", flush=True)
    static = static_screen(system, refined, request)
    propagator = propagator_check(system, request, states)
    print(f"{name}: independent local segments", flush=True)
    segments = segment_checks(system, refined, request, states, engine)
    same_frame_segments = None
    if name.startswith("5b-readout"):
        print(f"{name}: same-frame refined local segments", flush=True)
        same_frame_segments = segment_checks(
            system, refined, request, states, engine, same_frame=True
        )
    report = {
        "name": name,
        "dimensions": list(system.dimensions),
        "reference_dimensions": list(refined.dimensions),
        "duration_ns": float(request.times_ns[-1] - request.times_ns[0]),
        "source_wall_seconds": source_seconds,
        "source_cost_kind": "retained job"
        if name.startswith("5c-")
        else "regenerated engine solve",
        "source_parameters": {
            "rtol": request.rtol,
            "atol": request.atol,
            "max_steps": request.max_steps,
            "frame": "lab" if request.frame is None else "rotating-rwa",
        },
        "baseline": baseline,
        "integrity": {
            "wall_seconds": integrity_seconds,
            "checks": checks[0],
            "warnings": checks[1],
        },
        "static": static,
        "propagator": propagator,
        "segments": segments,
    }
    costs = {
        "static_only": static["wall_seconds"],
        "integrity_static_segments": integrity_seconds
        + static["wall_seconds"]
        + segments["wall_seconds"],
    }
    if same_frame_segments is not None:
        report["same_frame_refined_segments"] = same_frame_segments
        costs["integrity_static_same_frame_segments"] = (
            integrity_seconds + static["wall_seconds"] + same_frame_segments["wall_seconds"]
        )
    if propagator["applicable"]:
        costs["integrity_static_propagator"] = (
            integrity_seconds + static["wall_seconds"] + propagator["wall_seconds"]
        )
    report["verification_cost"] = {
        candidate: {
            "seconds": seconds,
            "source_ratio": seconds / source_seconds,
            "fits_source_fractions": {
                str(fraction): seconds <= fraction * source_seconds
                for fraction in (0.05, 0.1, 0.25)
            },
        }
        for candidate, seconds in costs.items()
    }
    return report


def model(document):
    with tempfile.TemporaryDirectory(prefix="entanglix-verification-model-") as directory:
        path = Path(directory) / "system.toml"
        path.write_text(tomli_w.dumps(document))
        system, seconds = timed(load_system, path)
        MODEL_SECONDS[id(system)] = seconds
        return system


def readout(ej, dimensions):
    (system, document), seconds = timed(construct, ej, dimensions)
    MODEL_SECONDS[id(system)] = seconds
    return system, document


def oscillator(levels, *, composite=False):
    if not composite:
        lower = qt.destroy(levels).full()
        number = np.diag(np.arange(levels))
        return ConstructedSystem(
            id="oscillator_probe",
            hamiltonian_ghz=0.2 * number,
            operators={"number": number, "position": lower + lower.conj().T, "lower": lower},
            dissipators={"loss": (lower, 0.01)},
            dimensions=(levels,),
            basis="fock",
            provenance={},
            drives={"drive": lower + lower.conj().T},
        )
    leaf = {
        "id": "r",
        "kind": "oscillator",
        "frequency": {"value": 0.2, "unit": "GHz"},
        "retained_levels": levels,
    }
    document = {
        "format_version": 1,
        "id": "oscillator_probe",
        "units": {"energy": "GHz", "time": "ns"},
    }
    document["model"] = {
        "kind": "composite",
        "subsystems": [leaf, {**leaf, "id": "s", "frequency": {"value": 0.3, "unit": "GHz"}}],
    }
    document["drives"] = [{"id": "drive", "factors": ["r.position"]}]
    return model(document)


def readout_cases():
    baseline_path = ROOT / "tests/probes/m2_frames_results.json"
    for entry in json.loads(baseline_path.read_text())["cases"]:
        system, document = readout(entry["ej_ghz"], (5, 6))
        refined, _ = readout(entry["ej_ghz"], (6, 8))
        ket = (
            np.eye(30)[6]
            if entry["kind"] == "purcell"
            else (np.eye(30)[0] + np.eye(30)[6]) / np.sqrt(2)
        )
        frequency = spectrum(system, document)["drive_frequency_ghz"]
        request = EvolutionInput(
            ket,
            np.arange(entry["duration_ns"] + 1, dtype=float),
            drive_amplitudes_ghz={"readout": entry["amplitude_ghz"]},
            carriers={"readout": Carrier(frequency)},
            frame=RotatingFrame({"q": frequency, "r": frequency}, rwa=True),
            max_steps=2_000_000,
        )
        trace, seconds = timed(QuTiPIntegrator().evolve, system, request)
        baseline = {
            "kind": "retained full-window independent summary, not trajectory",
            "artifact": "tests/probes/m2_frames_results.json",
            "sha256": digest(baseline_path),
            "trajectory_residuals": entry["maximum_observable_residuals"]["rotating-production"],
            "reference_wall_seconds": entry["wall_seconds"]["lab-reference"],
            "limitation": "baseline aggregate cannot supply arbitrary endpoint errors or segment states",
        }
        if entry["kind"] == "purcell":
            baseline["reported_metric"] = "excited population"
            baseline["trajectory_population_error"] = baseline["trajectory_residuals"]["excited"]
            baseline["metric_limitation"] = (
                "retained summary supplies a trajectory maximum, not endpoint population error; coherence is uninformative for this excited initial state"
            )
        else:
            baseline["reported_metric"] = "endpoint coherence"
            baseline["endpoint_coherence_error"] = entry["endpoint_coherence_error"]
        name = f"5b-{entry['kind']}-ej{entry['ej_ghz']}-a{entry['amplitude_ghz']}"
        yield name, system, refined, request, np.asarray(trace.states), seconds, baseline


def retained_cases():
    for device in ("cpu", "cuda"):
        summary_path = ROOT / f"tests/probes/m2_simulation_{device}.json"
        record = json.loads(summary_path.read_text())["records"][0]
        source = ROOT / "research_runs/readout" / record["run_id"]
        loaded = load_plan(source / "spec/system.toml", source / "spec/plan.toml")
        reference = source.parent / record["report"]["reference"]["run_id"]
        began = time.perf_counter()
        target = load_plan(reference / "spec/system.toml", reference / "spec/plan.toml")
        MODEL_SECONDS[id(target.system)] = time.perf_counter() - began
        for path, expected in (
            (
                source / "artifacts/output.npz",
                tomllib.loads((source / "result.toml").read_text())["artifacts"]["output"][
                    "sha256"
                ],
            ),
            (reference / "artifacts/output.npz", record["report"]["reference"]["artifact_sha256"]),
        ):
            if digest(path) != expected:
                raise ValueError(f"retained baseline digest mismatch: {path}")
        with np.load(source / "artifacts/output.npz") as arrays:
            states = arrays["states"]
        with np.load(reference / "artifacts/output.npz") as arrays:
            expected = arrays["states"]
        baseline = {
            "kind": "retained full-window independent trajectory",
            "summary": str(summary_path.relative_to(ROOT)),
            "summary_sha256": digest(summary_path),
            "source_run_id": source.name,
            "reference_run_id": reference.name,
            "reference_engine": target.definition.engine,
            "residuals": residuals(loaded.system, states, target.system, expected),
        }
        yield (
            f"5c-purcell-{device}",
            loaded.system,
            target.system,
            loaded.request,
            states,
            record["usage"]["wall_seconds"],
            baseline,
            target.definition.engine,
        )


def added_cases():
    two = load_system(ROOT / "examples/two-level/system.toml")
    coherent = np.array([1, 1j]) / np.sqrt(2)
    yield (
        "4b-loose-tolerance",
        two,
        two,
        EvolutionInput(coherent, np.arange(11.0), rtol=0.1, atol=0.1),
        None,
    )
    source, refined = oscillator(2, composite=True), oscillator(3, composite=True)
    yield (
        "4b-driven-truncation",
        source,
        refined,
        EvolutionInput(np.eye(4)[0], np.arange(3.0), drive_amplitudes_ghz={"drive": 0.15}),
        None,
    )
    source, refined = oscillator(4), oscillator(8)
    yield (
        "strong-drive",
        source,
        refined,
        EvolutionInput(
            np.eye(4)[0],
            np.linspace(0, 10, 21),
            drive_amplitudes_ghz={"drive": 0.15},
            carriers={"drive": Carrier(0.2)},
        ),
        None,
    )
    yield (
        "top-fock-false-alarm",
        source,
        refined,
        EvolutionInput(np.eye(4)[-1], np.linspace(0, 100, 21)),
        None,
    )
    closed = replace(
        two,
        dissipators={},
        dissipator_factors={},
        hamiltonian_ghz=np.diag([0.0, 0.02]),
        hamiltonian_terms=(),
    )
    initial = 0.5 * np.eye(2) / 2 + 0.5 * density(coherent)
    yield (
        "slow-phase-loose-tolerance",
        closed,
        closed,
        EvolutionInput(initial, np.linspace(0, 10000, 1001), rtol=1e-3, atol=1e-5),
        None,
    )
    amplitudes = np.zeros(100)
    amplitudes[47] = 0.04
    declared = EvolutionInput(
        np.eye(2)[0], np.arange(101.0), drive_amplitudes_ghz={"x_drive": amplitudes}
    )
    faulty = amplitudes.copy()
    faulty[47] = 0.05
    yield (
        "delayed-pulse-local-fault",
        closed,
        closed,
        declared,
        replace(declared, drive_amplitudes_ghz={"x_drive": faulty}),
    )

    def transmon(levels):
        import scqubits as scq

        qubit = scq.Transmon(EJ=15.0, EC=0.25, ng=0.0, ncut=30)
        energies, vectors = qubit.eigensys(evals_count=levels)
        charge = vectors.conj().T @ qubit.n_operator() @ vectors
        return ConstructedSystem(
            id="two_photon",
            hamiltonian_ghz=np.diag(energies - energies[0]),
            operators={"charge": charge},
            dissipators={},
            dimensions=(levels,),
            basis="isolated_eigenstates",
            provenance={},
            drives={"drive": charge},
            subsystem_ids=("q",),
        )

    source, refined = transmon(3), transmon(6)
    frequency = float((source.hamiltonian_ghz[2, 2] - source.hamiltonian_ghz[0, 0]).real / 2)
    request = EvolutionInput(
        np.eye(3)[0],
        np.linspace(0, 100, 101),
        drive_amplitudes_ghz={"drive": 0.03},
        carriers={"drive": Carrier(frequency)},
        frame=RotatingFrame({"q": frequency}, rwa=True),
    )
    yield "two-photon-transmon", source, refined, request, None


def benchmark():
    result = {"propagators": [], "integrity": []}
    for dimensions in ((2, 3), (3, 4), (4, 5), (5, 6), (6, 8)):
        system, document = readout(15, dimensions)
        frequency = spectrum(system, document)["drive_frequency_ghz"]
        request = EvolutionInput(
            np.eye(np.prod(dimensions))[dimensions[1]],
            np.arange(11.0),
            frame=RotatingFrame({"q": frequency, "r": frequency}, rwa=True),
        )
        trace = QuTiPIntegrator().evolve(system, request)
        report = propagator_check(system, request, trace.states)
        result["propagators"].append(report)
        print(
            f"propagator dimension {np.prod(dimensions)}: {report['wall_seconds']:.3f}s", flush=True
        )
    rng = np.random.default_rng(17)
    for dimension in (30, 48, 120, 240):
        count = 21
        vectors = rng.normal(size=(count, dimension)) + 1j * rng.normal(size=(count, dimension))
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        states = vectors[:, :, None] * vectors.conj()[:, None, :] / 2 + np.eye(dimension)[None] / (
            2 * dimension
        )
        _, seconds = timed(output_checks, states, (dimension,))
        result["integrity"].append(
            {
                "dimension": dimension,
                "stored_states": count,
                "wall_seconds": seconds,
                "distribution": "seeded mixed states, not diagonal shortcuts",
            }
        )
    return result


def segment_signal(report, metric, quantity="trajectory"):
    values = [
        item["local_metric_residuals"][metric][quantity]
        for item in report["segments"]
        if item["outcome"] == "measured"
    ]
    return max(values) if values else None


def assessment(result):
    cases = {case["name"]: case for case in result["cases"]}
    detection = []
    for name, metric in (
        ("4b-loose-tolerance", "coherence"),
        ("4b-driven-truncation", "photons"),
        ("strong-drive", "number"),
        ("slow-phase-loose-tolerance", "coherence_real"),
        ("delayed-pulse-local-fault", "excited"),
        ("two-photon-transmon", "level2"),
    ):
        case = cases[name]
        propagation = case["propagator"].get("metric_error_bound", {}).get(metric)
        segment = segment_signal(case["segments"], metric)
        detection.append(
            {
                "case": name,
                "metric": metric,
                "target": NUMERICAL_TARGET,
                "measured_endpoint_error": case["baseline"]["residuals"][metric]["endpoint"],
                "propagator_numerical_signal": propagation,
                "segment_local_signal": segment,
                "segment_missed_failure": segment is not None and segment <= NUMERICAL_TARGET,
                "static_cleared_dynamics": False,
            }
        )
    false_alarm = cases["top-fock-false-alarm"]
    delayed = cases["delayed-pulse-local-fault"]
    extrapolations = []
    direct_thresholds = []
    same_frame_comparisons = []
    for case in result["cases"]:
        if not case["name"].startswith("5b-readout"):
            continue
        # Deliberately test a simple extrapolation hypothesis. Local micromotion
        # can make it grossly conservative; it is not an accepted error rule.
        estimate = max(
            item["local_metric_residuals"]["coherence"]["endpoint"]
            * case["duration_ns"]
            / (item["stop_ns"] - item["start_ns"])
            for item in case["segments"]["segments"]
            if item["outcome"] == "measured"
        )
        measured = case["baseline"]["endpoint_coherence_error"]
        local = segment_signal(case["segments"], "coherence", "endpoint")
        local_trajectory = segment_signal(case["segments"], "coherence")
        direct_thresholds.append(
            {
                "case": case["name"],
                "metric": "endpoint coherence",
                "local_endpoint_signal": local,
                "local_trajectory_signal": local_trajectory,
                "full_window_endpoint_error": measured,
                "accepted_endpoint_target": 5e-4,
                "numerical_threshold": NUMERICAL_TARGET,
                "false_alarm_at_numerical_threshold": local_trajectory > NUMERICAL_TARGET
                and measured <= 5e-4,
                "false_alarm_at_endpoint_target": local > 5e-4 >= measured,
            }
        )
        same_frame_comparisons.append(
            {
                "case": case["name"],
                "same_frame_local_endpoint_residuals": {
                    metric: segment_signal(case["same_frame_refined_segments"], metric, "endpoint")
                    for metric in ("coherence", "excited", "photons")
                },
                "lab_frame_local_endpoint_coherence_residual": local,
                "same_frame_wall_seconds": case["same_frame_refined_segments"]["wall_seconds"],
                "lab_frame_wall_seconds": case["segments"]["wall_seconds"],
            }
        )
        extrapolations.append(
            {
                "case": case["name"],
                "metric": "endpoint coherence",
                "target": 5e-4,
                "hypothesis": "largest local endpoint defect divided by segment duration times full duration",
                "estimate": estimate,
                "measured": measured,
                "false_alarm": estimate > 5e-4 >= measured,
            }
        )
    checks = {
        "loose_tolerance_propagator_detected": cases["4b-loose-tolerance"]["propagator"][
            "metric_error_bound"
        ]["coherence"]
        > NUMERICAL_TARGET,
        "slow_error_integrity_passes": all(
            item["outcome"] == "pass"
            for item in cases["slow-phase-loose-tolerance"]["integrity"]["checks"]
        ),
        "slow_error_propagator_detected": cases["slow-phase-loose-tolerance"]["propagator"][
            "metric_error_bound"
        ]["coherence_real"]
        > NUMERICAL_TARGET,
        "truncation_segments_detected": segment_signal(
            cases["4b-driven-truncation"]["segments"], "photons"
        )
        > NUMERICAL_TARGET,
        "strong_drive_segments_detected": segment_signal(
            cases["strong-drive"]["segments"], "number"
        )
        > NUMERICAL_TARGET,
        "two_photon_case_populates_level2": cases["two-photon-transmon"][
            "maximum_level2_population"
        ]
        > 0.1,
        "two_photon_segments_detected": segment_signal(
            cases["two-photon-transmon"]["segments"], "level2"
        )
        > NUMERICAL_TARGET,
        "top_fock_diagnostic_triggers": false_alarm["integrity"]["warnings"][0]["triggered"],
        "top_fock_no_accuracy_failure": false_alarm["analytic_number_residual"] < NUMERICAL_TARGET
        and segment_signal(false_alarm["segments"], "number") < NUMERICAL_TARGET,
        "delayed_sparse_segments_miss": segment_signal(delayed["segments"], "excited")
        < NUMERICAL_TARGET,
        "delayed_transition_segments_detect": segment_signal(
            delayed["transition_aware_segments"], "excited"
        )
        > NUMERICAL_TARGET,
        "delayed_propagator_detects": delayed["propagator"]["metric_error_bound"]["excited"]
        > NUMERICAL_TARGET,
        "spectral_small_cutoff_detected": result["spectral_failure"]["residual_ghz"]
        > result["spectral_failure"]["target_ghz"],
        "budget_exhaustion_returns_no_segment_evidence": all(
            item["outcome"] == "allowance_exhausted"
            for item in result["timeout_control"]["segments"]
        ),
        "initial_state_term_measured": result["initial_state_control"]["initial_state_term"] > 1.99,
        "same_frame_readout_segments_completed": all(
            item["outcome"] == "measured"
            for case in result["cases"]
            if case["name"].startswith("5b-readout")
            for item in case["same_frame_refined_segments"]["segments"]
        ),
    }
    return {
        "regression_checks": checks,
        "pass": all(checks.values()),
        "detection": detection,
        "direct_lab_segment_thresholding": {
            "comparisons": direct_thresholds,
            "false_alarms_at_numerical_threshold": sum(
                item["false_alarm_at_numerical_threshold"] for item in direct_thresholds
            ),
            "false_alarms_at_endpoint_target": sum(
                item["false_alarm_at_endpoint_target"] for item in direct_thresholds
            ),
            "scope": "six RWA readout cases; false alarm means rejecting a case whose retained full-window endpoint error meets the accepted 5e-4 target; local endpoints and the reported full-window endpoint have different scopes",
            "claim": "lab-frame segments include RWA differences and cannot be thresholded as pure numerical error at 1e-6; even the same endpoint target does not translate into a local restart rule",
        },
        "same_frame_refined_segments": {
            "comparisons": same_frame_comparisons,
            "scope": "independent tighter refined RWA solves restarted from source lab states at absolute times; local truncation/numerical evidence with the same approximation, not an RWA check or a full-window truncation estimate",
        },
        "segment_extrapolation": {
            "comparisons": extrapolations,
            "false_alarms": sum(item["false_alarm"] for item in extrapolations),
            "claim": "rejected candidate estimate; sampled restarts do not establish global metric accuracy",
        },
        "candidate_assessment": {
            "propagator": "detects numerical/local-pulse errors in every covered interval; cost depends on dimension and generator count",
            "segments": "detect the sampled truncation/drive failures; sparse restarts can miss a persistent error introduced elsewhere",
            "static": "Purcell rate estimate is promising within its narrow assumptions; no generic driven/truncation metric estimate was established",
            "general_replacement_ready": False,
        },
        "policy_candidates": {
            "propagator_dimension_options": {
                str(seconds): [
                    item["dimension"]
                    for item in result["benchmarks"]["propagators"]
                    if item["wall_seconds"] <= seconds
                ]
                for seconds in (0.1, 1.0, 3.0)
            },
            "caveat": "measured options, not accepted caps or default accuracy targets; full-window estimates from sampled segments remain unvalidated",
        },
    }


def study(output):
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("set JAX_PLATFORMS=cpu for the CPU-only segment probe")
    result = {
        "complete": False,
        "kind": "feasibility measurements, not production verdicts",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("numpy", "scipy", "qutip", "scqubits", "dynamiqs", "jax")
            },
            "lock_sha256": digest(ROOT / "uv.lock"),
            "threads": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in [
                Path(__file__).resolve(),
                ROOT / "tests/probes/m2_feasibility.py",
                ROOT / "tests/engine_cases.py",
                ROOT / "src/entanglix/evolution.py",
                ROOT / "src/entanglix/models.py",
                ROOT / "src/entanglix/corrector/reference.py",
                ROOT / "src/entanglix/corrector/checks.py",
                ROOT / "src/entanglix/protocols/plans.py",
                *sorted((ROOT / "src/entanglix/engines").glob("*.py")),
            ]
        },
        "candidate_targets": {
            "numerical": NUMERICAL_TARGET,
            "readout_endpoint_coherence": 5e-4,
            "purcell_endpoint_population": 0.005,
        },
        "cases": [],
    }

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    for args in readout_cases():
        result["cases"].append(evaluate(*args))
        save()
    for name, system, refined, request, states, seconds, baseline, engine in retained_cases():
        result["cases"].append(
            evaluate(name, system, refined, request, states, seconds, baseline, engine=engine)
        )
        save()
    for name, system, refined, request, faulty in added_cases():
        trace, seconds = timed(QuTiPIntegrator().evolve, system, faulty or request)
        states = np.asarray(trace.states)
        reference_request = replace(
            request,
            initial_state=embed_state(system, refined, request.initial_state),
            frame=None,
            rtol=1e-11,
            atol=1e-13,
            max_steps=2_000_000,
        )
        if any(np.any(request.drive_amplitudes_ghz.get(key, 0)) for key in request.carriers):
            expected, reference_seconds = timed(
                DynamiqsIntegrator("cpu").evolve, refined, reference_request
            )
            target = np.asarray(expected.states)
            reference_kind = "new full-window independent dynamiqs trajectory"
        else:
            target, reference_seconds = timed(exact_states, refined, reference_request)
            reference_kind = "existing M1 exponential helper; independent generator construction, shared scipy.expm"
        baseline = {
            "kind": reference_kind,
            "wall_seconds": reference_seconds,
            "residuals": residuals(system, states, refined, target),
        }
        case = evaluate(name, system, refined, request, states, seconds, baseline)
        if faulty is not None:
            case["fault_injection"] = (
                "one declared pulse 0.04 GHz executed as 0.05 GHz, all later evolution correct"
            )
            case["transition_aware_segments"] = segment_checks(
                system, refined, request, states, "dynamiqs", transition_aware=True
            )
        if name == "two-photon-transmon":
            case["maximum_level2_population"] = float(np.asarray(trace.states)[:, 2, 2].real.max())
        if name == "top-fock-false-alarm":
            analytic = (system.dimensions[0] - 1) * np.exp(-0.01 * request.times_ns)
            case["analytic_number_residual"] = float(
                np.max(abs(metrics(system, states)["number"] - analytic))
            )
        result["cases"].append(case)
        save()
    document = tomllib.loads((ROOT / "examples/fluxonium/system.toml").read_text())
    document["truncation"]["cutoffs"]["theta1"] = 20
    source = model(document)
    document["truncation"]["cutoffs"]["theta1"] = 40
    refined = model(document)
    eigenvalues, seconds = timed(np.linalg.eigvalsh, source.hamiltonian_ghz)
    target, refined_seconds = timed(np.linalg.eigvalsh, refined.hamiltonian_ghz)
    result["spectral_failure"] = {
        "case": "4b-fluxonium-small-cutoff",
        "reported_levels": 4,
        "residual_ghz": float(
            np.max(abs(eigenvalues[:4] - eigenvalues[0] - target[:4] + target[0]))
        ),
        "target_ghz": 1e-5,
        "eigensolve_seconds": seconds + refined_seconds,
    }
    result["benchmarks"] = benchmark()
    system = oscillator(4)
    request = EvolutionInput(np.eye(4)[-1], np.linspace(0, 100, 21))
    trace = QuTiPIntegrator().evolve(system, request)
    result["timeout_control"] = segment_checks(
        system,
        system,
        request,
        np.asarray(trace.states),
        "dynamiqs",
        timeout_seconds=0.01,
    )
    result["initial_state_control"] = propagator_check(
        system,
        replace(request, initial_state=np.eye(4)[0]),
        trace.states,
    )
    if any(digest(ROOT / path) != expected for path, expected in result["source_sha256"].items()):
        raise RuntimeError("probe source changed during measurements")
    result["assessment"] = assessment(result)
    result["pass"] = result["assessment"]["pass"]
    result["complete"] = True
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    study(parser.parse_args().output)
