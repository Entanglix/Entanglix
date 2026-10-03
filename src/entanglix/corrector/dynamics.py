"""Check discovery and scoped, same-frame convergence comparisons."""

import math
import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np

from entanglix.corrector import CHECK_DEFINITIONS, load_policy, relevant_checks, validate_refinement
from entanglix.corrector.checks import check, inapplicable, output_checks, top_level_populations
from entanglix.corrector.metrics import metric_tolerance, metric_values
from entanglix.corrector.reference import embedding_matrix, prepare_refinement
from entanglix.engines.frames import matrix, prepare_dynamics
from entanglix.evolution import EvolutionInputError, normalize_input

WHOLE_WINDOW_LIMIT = "No built-in check measures whole-window error or the accuracy of a rotating-wave approximation."


def refinement_reason(loaded, policy):
    try:
        validate_refinement(
            loaded.definition,
            tomllib.loads(loaded.inputs["system.toml"].read_text()),
            loaded.inputs["system.toml"].parent,
            policy,
        )
    except ValueError as exc:
        return str(exc)
    return ""


def refined_dimensions(loaded):
    return [
        count
        + (
            loaded.definition.refinement[name].retained_levels_step
            if name in loaded.definition.refinement
            else 0
        )
        for name, count in zip(loaded.system.subsystem_ids, loaded.system.dimensions)
    ]


def segment_windows(request, policy):
    count = len(request.times_ns) - 1
    width = min(count, policy.thresholds["segments"]["segment_intervals"])
    windows = list(
        dict.fromkeys((start, start + width) for start in (0, (count - width) // 2, count - width))
    )
    changes = sorted(
        {
            int(index) + 1
            for value in request.drive_amplitudes_ghz.values()
            if np.ndim(value)
            for index in np.flatnonzero(np.diff(value))
        }
    )
    candidates = list(
        dict.fromkeys(
            (max(0, min(index - 1, count - width)), max(0, min(index - 1, count - width)) + width)
            for index in changes
        )
    )
    candidates = [window for window in candidates if window not in windows]
    slots = max(0, policy.thresholds["segments"]["max_segments"] - len(windows))
    if candidates and slots:
        indices = (
            np.linspace(0, len(candidates) - 1, min(slots, len(candidates))).round().astype(int)
        )
        windows.extend(candidates[index] for index in indices)
    return windows[: policy.thresholds["segments"]["max_segments"]], changes


def segment_coverage(request, windows, changes):
    intervals = {i for first, last in windows for i in range(first, last)}
    uncovered = [
        float(request.times_ns[i]) for i in changes if not any(a <= i < b for a, b in windows)
    ]
    return {
        "covered_fraction": len(intervals) / (len(request.times_ns) - 1),
        "uncovered_drive_changes": {
            "count": len(uncovered),
            **({"first_ns": uncovered[0], "last_ns": uncovered[-1]} if uncovered else {}),
        },
        "scope": "local agreement over sampled windows only; no extrapolation",
    }


def check_listing(loaded, *, policy=None, timeout_seconds=None, selection=None):
    policy = policy or load_policy()
    definition = loaded.definition
    reason = refinement_reason(loaded, policy)
    dimensions = refined_dimensions(loaded)
    engine = definition.engine if definition.device == "cpu" else "qutip-cpu"
    entries = []
    for name in relevant_checks(definition.kind):
        item = {"name": name, "error_mode": CHECK_DEFINITIONS[name], "available": True}
        if name in {"segments", "spectral_refinement"}:
            problem = reason or (
                "no refinable subsystem"
                if name == "spectral_refinement" and not definition.refinement
                else "no reported observables for a local comparison"
                if name == "segments" and not loaded.request.observables
                else ""
            )
            item.update(
                engine=engine,
                source_dimensions=list(loaded.system.dimensions),
                refined_dimensions=dimensions,
                refined_dimension=math.prod(dimensions),
            )
            if problem:
                item.update(available=False, reason=problem)
            if name == "segments":
                windows, changes = segment_windows(loaded.request, policy)
                item["windows"] = [
                    {
                        "start_ns": float(loaded.request.times_ns[a]),
                        "end_ns": float(loaded.request.times_ns[b]),
                    }
                    for a, b in windows
                ]
                item["coverage"] = segment_coverage(loaded.request, windows, changes)
                item["tolerances"] = {
                    "observables": {
                        name: definition.segment_tolerances.get(
                            name, policy.thresholds["refinement"]["observable_atol"]
                        )
                        for name in loaded.request.observables
                    },
                    "metrics": {
                        name: metric_tolerance(metric, policy)
                        for name, metric in definition.metrics.items()
                    },
                }
        if name == "diagnostics" and reason:
            item["refined_cutoff_note"] = reason
        entries.append(item)
    return {
        "model": {
            "system_id": loaded.system.id,
            "subsystem_ids": list(loaded.system.subsystem_ids),
            "dimensions": list(loaded.system.dimensions),
            "dimension": math.prod(loaded.system.dimensions),
        },
        "execution": {
            "engine": definition.engine,
            "device": definition.device,
            "output_count": len(loaded.request.times_ns) if loaded.request else definition.levels,
            "selection": selection or {},
            **({"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}),
        },
        "checks": entries,
        **({"limit": WHOLE_WINDOW_LIMIT} if loaded.request is not None else {}),
    }


def refined_segments(loaded, arrays, policy):
    request = loaded.request
    reason = refinement_reason(loaded, policy)
    if reason or not request.observables:
        return inapplicable(
            "segments",
            reason or "no reported observables for a local comparison",
            scope="local windows",
        )
    windows, changes = segment_windows(request, policy)
    entries = []
    with tempfile.TemporaryDirectory(prefix="entanglix-segments-") as directory:
        refined = prepare_refinement(loaded, Path(directory), policy)
        mapping = embedding_matrix(loaded.system, refined.system)
        if refined.definition.engine == "dynamiqs":
            from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

            integrator = DynamiqsIntegrator(device_kind="cpu")
        else:
            from entanglix.engines import QuTiPIntegrator

            integrator = QuTiPIntegrator()
        for first, last in windows:
            segment = replace(
                refined.request,
                initial_state=mapping @ arrays["states"][first] @ mapping.conj().T,
                times_ns=request.times_ns[first : last + 1],
                drive_amplitudes_ghz={
                    name: value[first:last] if np.ndim(value) else value
                    for name, value in request.drive_amplitudes_ghz.items()
                },
            )
            details = {
                "start_ns": float(segment.times_ns[0]),
                "end_ns": float(segment.times_ns[-1]),
                "engine": refined.definition.engine,
                "dimensions": list(refined.system.dimensions),
                "rtol": segment.rtol,
                "atol": segment.atol,
                "max_steps": segment.max_steps,
            }
            try:
                normalize_input(refined.system, segment)
            except EvolutionInputError as exc:
                entries.append(
                    {
                        **details,
                        "checks": [
                            check(
                                "restart_state",
                                1,
                                0,
                                scope="stored state used to restart this window",
                                diagnostic=str(exc),
                            )
                        ],
                    }
                )
                continue
            trace = integrator.evolve(refined.system, segment)
            comparisons, _ = output_checks(
                np.asarray(trace.states), refined.system.dimensions, policy=policy
            )
            for name in request.observables:
                comparisons.append(
                    check(
                        "observable",
                        float(
                            np.max(
                                abs(
                                    arrays[f"observable.{name}"][first : last + 1]
                                    - trace.expectations[name]
                                )
                            )
                        ),
                        loaded.definition.segment_tolerances.get(
                            name, policy.thresholds["refinement"]["observable_atol"]
                        ),
                        scope="local trajectory",
                        observable=name,
                    )
                )
            for name, definition in loaded.definition.metrics.items():
                comparisons.append(
                    check(
                        "metric",
                        abs(
                            metric_values(
                                {name: definition},
                                {
                                    f"observable.{definition.observable}": arrays[
                                        f"observable.{definition.observable}"
                                    ][first : last + 1]
                                },
                            )[name]
                            - metric_values(
                                {name: definition},
                                {
                                    f"observable.{definition.observable}": trace.expectations[
                                        definition.observable
                                    ]
                                },
                            )[name]
                        ),
                        metric_tolerance(definition, policy),
                        scope="local endpoint",
                        metric=name,
                    )
                )
            entries.append(
                {
                    **details,
                    "checks": comparisons,
                }
            )
    return {
        "name": "segments",
        "outcome": "fail"
        if any(item["outcome"] == "fail" for entry in entries for item in entry["checks"])
        else "pass",
        "scope": "local convergence comparison in the source frame; inherits accumulated error",
        "segments": entries,
        "coverage": segment_coverage(request, windows, changes),
    }


def diagnostics(loaded, arrays, policy):
    warnings = top_level_populations(arrays["states"], loaded.system.dimensions, policy=policy)
    values = {"top_level_population": warnings}
    reason = refinement_reason(loaded, policy)
    if reason:
        values["refined_cutoff_omitted"] = reason
    else:
        with tempfile.TemporaryDirectory(prefix="entanglix-diagnostics-") as directory:
            refined = prepare_refinement(loaded, Path(directory), policy)
            mapping = embedding_matrix(loaded.system, refined.system)
            old = np.linalg.eigvalsh(loaded.system.hamiltonian_ghz)
            energies, vectors = np.linalg.eigh(refined.system.hamiltonian_ghz)
            values["cutoff_shifts_ghz"] = (
                energies[: len(old)] - energies[0] - old + old[0]
            ).tolist()
            values["outside_weights"] = (
                1 - np.sum(abs(mapping.conj().T @ vectors[:, : len(old)]) ** 2, axis=0)
            ).tolist()
    prepared = prepare_dynamics(loaded.system, normalize_input(loaded.system, loaded.request))
    if prepared.frame_levels_ghz is not None:
        full = np.linalg.eigvalsh(loaded.system.hamiltonian_ghz)
        projected = matrix(prepared.hamiltonian) + np.diag(prepared.frame_levels_ghz)
        rwa = np.linalg.eigvalsh(projected)
        values["full_against_rwa_shifts_ghz"] = (full - full[0] - rwa + rwa[0]).tolist()
    return {
        "name": "diagnostics",
        "scope": "top-level populations at stored times; ordered static shifts and weights, not dynamics error estimates",
        "values": values,
    }
