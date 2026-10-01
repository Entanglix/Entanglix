"""Full-window readout/Purcell acceptance using the shared frame and carrier interface."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from m2_feasibility import ROOT, construct, environment, spectrum

from entanglix.corrector.checks import physical_residuals
from entanglix.corrector.reference import embed_state
from entanglix.engines import Carrier, EvolutionInput, QuTiPIntegrator, RotatingFrame
from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator


def quantities(states, dimensions, photons):
    nq, nr = dimensions
    states = np.asarray(states)
    reduced = np.trace(states.reshape(-1, nq, nr, nq, nr), axis1=2, axis2=4)
    return {
        "coherence": 2 * abs(reduced[:, 0, 1]),
        "excited": reduced[:, 1, 1].real,
        "photons": np.einsum("tij,ji->t", states, photons).real,
    }


def study(output):
    result = {
        "complete": False,
        "environment": environment(),
        "source_sha256": {
            str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "src/entanglix/evolution.py",
                "src/entanglix/engines/frames.py",
                "src/entanglix/engines/qutip_cpu.py",
                "src/entanglix/engines/dynamiqs_jax.py",
                "tests/probes/m2_frames.py",
            )
        },
        "numerical_observable_atol": 1e-6,
        "rwa_endpoint_coherence_atol": 5e-4,
        "rwa_trajectory_diagnostic_atol": {"coherence": 0.003, "excited": 0.005, "photons": 0.008},
        "cases": [],
    }
    cases = [
        (ej, amplitude, "readout")
        for ej, amplitude in (
            (12.0, 0.0),
            (12.0, 0.001),
            (12.0, 0.002),
            (13.5, 0.002),
            (15.0, 0.0),
            (15.0, 0.002),
        )
    ]
    cases.append((15.0, 0.0, "purcell"))
    for ej, amplitude, kind in cases:
        source, document = construct(ej, (5, 6))
        refined, _ = construct(ej, (6, 8))
        frequency = spectrum(source, document)["drive_frequency_ghz"]
        duration = 1600 if kind == "purcell" else 100
        ket = np.eye(30)[6] if kind == "purcell" else (np.eye(30)[0] + np.eye(30)[6]) / np.sqrt(2)
        request = EvolutionInput(
            ket,
            np.arange(duration + 1, dtype=float),
            drive_amplitudes_ghz={"readout": amplitude},
            carriers={"readout": Carrier(frequency)},
            max_steps=2_000_000,
        )
        from dataclasses import replace

        observations, timings, physics, top_population = {}, {}, {}, {}
        for name, model, settings, engine in (
            ("lab-production", source, request, QuTiPIntegrator()),
            (
                "rotating-production",
                source,
                replace(request, frame=RotatingFrame({"q": frequency, "r": frequency}, rwa=True)),
                QuTiPIntegrator(),
            ),
            (
                "lab-reference",
                refined,
                replace(
                    request, initial_state=embed_state(source, refined, ket), rtol=1e-11, atol=1e-13
                ),
                DynamiqsIntegrator("cpu"),
            ),
        ):
            started = time.perf_counter()
            trace = engine.evolve(model, settings)
            timings[name] = time.perf_counter() - started
            observations[name] = quantities(
                trace.states, model.dimensions, model.operators["r.number"]
            )
            physics[name] = physical_residuals(np.asarray(trace.states))
            nq, nr = model.dimensions
            populations = np.diagonal(np.asarray(trace.states), axis1=1, axis2=2).real.reshape(
                -1, nq, nr
            )
            top_population[name] = {
                "q": float(populations[:, -1, :].sum(axis=1).max()),
                "r": float(populations[:, :, -1].sum(axis=1).max()),
            }
            print(f"{kind} EJ={ej} A={amplitude} {name}: {timings[name]:.2f}s", flush=True)
        residuals = {
            name: {
                observable: float(
                    np.max(abs(values[observable] - observations["lab-reference"][observable]))
                )
                for observable in values
            }
            for name, values in observations.items()
            if name != "lab-reference"
        }
        endpoint = abs(
            observations["rotating-production"]["coherence"][-1]
            - observations["lab-reference"]["coherence"][-1]
        )
        entry = {
            "kind": kind,
            "ej_ghz": ej,
            "amplitude_ghz": amplitude,
            "duration_ns": duration,
            "frequency_ghz": frequency,
            "dimensions": [5, 6],
            "reference_dimensions": [6, 8],
            "production_rtol": request.rtol,
            "production_atol": request.atol,
            "reference_rtol": 1e-11,
            "reference_atol": 1e-13,
            "max_steps": request.max_steps,
            "production_engine": "qutip-cpu",
            "reference_engine": "dynamiqs",
            "reference_device": "cpu",
            "maximum_observable_residuals": residuals,
            "endpoint_coherence_error": float(endpoint),
            "wall_seconds": timings,
            "physicality": physics,
            "top_level_population": top_population,
            "numerical_pass": all(value <= 1e-6 for value in residuals["lab-production"].values()),
            "frame_diagnostics_pass": all(
                value <= result["rwa_trajectory_diagnostic_atol"][name]
                for name, value in residuals["rotating-production"].items()
            ),
            "endpoint_pass": float(endpoint) <= 5e-4,
            "physicality_pass": all(
                p["trace_error"] <= 1e-8
                and p["hermiticity_error"] <= 1e-8
                and p["minimum_eigenvalue"] >= -1e-8
                for p in physics.values()
            ),
        }
        result["cases"].append(entry)
        output.write_text(json.dumps(result, indent=2) + "\n")
    result["complete"] = True
    result["pass"] = all(
        all(
            case[key]
            for key in (
                "numerical_pass",
                "frame_diagnostics_pass",
                "endpoint_pass",
                "physicality_pass",
            )
        )
        for case in result["cases"]
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result["pass"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if study(parser.parse_args().output) else 1)
