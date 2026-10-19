"""CPU timings of the previous production GRAPE loss and its matrix-exponential replacement.

Run with the existing environment: .venv-runner/bin/python tests/probes/grape_propagator.py
Each case/loss uses a fresh process. Raw measurements, input copies and the previous
production source stay in --workspace. --rebuild only summarizes those measurements.
No package sync, Julia, GPU, agent session or harness job is used by this probe.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_REVISION = "dd9b69fd2b92d72082c427cf70528b7ec1a2f3d9"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(workspace):
    import tomli_w

    from entanglix.protocols.control import load_control

    previous = workspace / "previous_grape.py"
    previous.write_bytes(
        subprocess.check_output(
            ["git", "show", f"{PREVIOUS_REVISION}:src/entanglix/solvers/grape.py"], cwd=ROOT
        )
    )
    inputs = []
    for name, levels, intervals in (
        ("transmon-3-20", 3, 20),
        ("transmon-3-40", 3, 40),
        ("transmon-6-20", 6, 20),
        ("transmon-10-20", 10, 20),
        ("qubit-resonator-30-20", None, 20),
    ):
        directory = workspace / "inputs" / name
        directory.mkdir(parents=True, exist_ok=True)
        definition = tomllib.loads((ROOT / "examples/transmon-control/control.toml").read_text())
        definition["problem"]["N"] = intervals + 1
        source = ROOT / "examples" / ("qubit-resonator" if levels is None else "transmon-control")
        system = tomllib.loads((source / "system.toml").read_text())
        if levels is None:
            shutil.copyfile(source / "qubit.toml", directory / "qubit.toml")
            definition["frame"]["frequencies_ghz"] = {
                "q": definition["channels"][0]["frequency_ghz"],
                "r": 6.2,
            }
            definition["goal"]["subspace"] = [[1, 6]]
        else:
            system["truncation"]["retained_levels"] = levels
        (directory / "system.toml").write_text(tomli_w.dumps(system))
        control = directory / "control.toml"
        control.write_text(tomli_w.dumps(definition))
        loaded = load_control(control)
        problem = directory / "problem.toml"
        problem.write_bytes(loaded.problem_bytes)
        inputs.append(
            {
                "name": name,
                "control": str(control),
                "problem": str(problem),
                "problem_sha256": sha(problem),
                "source_input_sha256": loaded.input_digests,
                "dimension": len(loaded.problem_spec["system"]["H_drift"]),
                "intervals": intervals,
                "options": definition["solver"]["options"],
            }
        )
    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": sha(Path(__file__)),
        "source_sha256": {
            name: sha(ROOT / name)
            for name in (
                "src/entanglix/solvers/grape.py",
                "src/entanglix/engines/dynamiqs_jax.py",
                "src/entanglix/protocols/control.py",
                "uv.lock",
            )
        },
        "previous": {
            "revision": PREVIOUS_REVISION,
            "source": str(previous),
            "sha256": sha(previous),
            "kind": "previous production loss, including the unchanged engine wrapper",
        },
        "inputs": inputs,
    }


def worker(request):
    import jax
    import numpy as np

    from entanglix.solvers import grape

    module = grape
    if request["loss"] == "previous":
        spec = importlib.util.spec_from_file_location("previous_grape", request["previous"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    problem_path = Path(request["case"]["problem"])
    problem = tomllib.loads(problem_path.read_text())
    bounds = np.asarray(problem["system"]["params"]["drive_bounds"])
    shape = (len(bounds), problem["problem"]["N"])
    upper = np.repeat(bounds[:, None], shape[1], axis=1)
    upper[:, [0, -1]] = 0
    initial = np.random.default_rng(7).uniform(-0.1, 0.1, shape) * upper
    dt = problem["pulse"]["T"] / (shape[1] - 1)
    for order, field in ((1, "du_bound"), (2, "ddu_bound")):
        if field in problem["problem"]:
            maximum = np.max(abs(np.diff(initial, n=order, axis=1) / dt**order))
            if maximum:
                initial *= min(1, problem["problem"][field] / maximum)
    started = time.perf_counter()
    function, times = module.objective(problem)
    setup = time.perf_counter() - started
    started = time.perf_counter()
    lowered = function.lower(initial)
    lowering = time.perf_counter() - started
    started = time.perf_counter()
    compiled = lowered.compile()
    compilation = time.perf_counter() - started
    started = time.perf_counter()
    value, gradient = jax.block_until_ready(compiled(initial))
    execution = time.perf_counter() - started
    warm = []
    for _ in range(5):
        started = time.perf_counter()
        jax.block_until_ready(compiled(initial))
        warm.append(time.perf_counter() - started)
    samples = []
    rng = np.random.default_rng(20261018)
    for _ in range(3):
        controls = rng.uniform(-0.8, 0.8, shape) * upper
        sample_value, sample_gradient = jax.block_until_ready(compiled(controls))
        samples.append(
            {"value": float(sample_value), "gradient": np.asarray(sample_gradient).tolist()}
        )
    # Use the backend's actual initialization, constraints, settings and termination.
    # Only replace its objective factory with the already compiled callable.
    module.objective = lambda _: (compiled, times)
    started = time.perf_counter()
    solution = module.GrapeBackend().solve(problem_path, request["case"]["options"])
    optimization = time.perf_counter() - started
    return {
        "loss": request["loss"],
        "versions": {name: version(name) for name in ("jax", "jaxlib", "dynamiqs", "scipy")},
        "devices": sorted({d.platform for item in (value, gradient) for d in item.devices()}),
        "value_dtype": str(value.dtype),
        "gradient_dtype": str(gradient.dtype),
        "setup_seconds": setup,
        "lowering_seconds": lowering,
        "compile_seconds": compilation,
        "first_execution_seconds": execution,
        "cold_first_call_seconds": lowering + compilation + execution,
        "warm_call_median_seconds": statistics.median(warm),
        "warm_optimization_seconds": optimization,
        "fidelity": solution.solver_reported_fidelity,
        "termination": solution.termination,
        "parity_samples": samples,
    }


def summarize(workspace, output):
    import numpy as np

    metadata = json.loads((workspace / "metadata.json").read_text())
    cases = []
    for case in metadata["inputs"]:
        measurements = {
            loss: json.loads(
                (workspace / "measurements" / f"{case['name']}-{loss}.json").read_text()
            )
            for loss in ("previous", "expm")
        }
        parity = []
        for old, new in zip(
            measurements["previous"]["parity_samples"],
            measurements["expm"]["parity_samples"],
            strict=True,
        ):
            difference = abs(np.array(old["gradient"]) - new["gradient"])
            parity.append(
                {
                    "value_absolute_difference": abs(old["value"] - new["value"]),
                    "free_knot_gradient_maximum_difference": float(difference[:, 1:-1].max()),
                    "first_knot_gradient_maximum_difference": float(difference[:, 0].max()),
                }
            )
        for item in measurements.values():
            del item["parity_samples"]
        cases.append(
            {
                **case,
                "measurements": measurements,
                "parity": parity,
                "new_loss_slower_in": [
                    field
                    for field in (
                        "compile_seconds",
                        "cold_first_call_seconds",
                        "warm_call_median_seconds",
                        "warm_optimization_seconds",
                    )
                    if measurements["expm"][field] > measurements["previous"][field]
                ],
            }
        )
    save(
        output,
        {
            **{k: v for k, v in metadata.items() if k != "inputs"},
            "workspace": str(workspace),
            "timing_note": "Serial CPU measurements, one BLAS thread, fresh process per input/loss, no persistent JAX cache. Cold first call is measured as explicit lowering + compilation + synchronized first execution; objective setup is separate. Warm optimization uses the precompiled callable with unchanged backend initialization, SLSQP settings and constraints. Timings exclude process/import startup and parity checks. Compilation is counted once.",
            "comparison_note": "Numerical parity and solver-reported SLSQP outcomes, not an optimizer quality ranking. First-knot gradients of the previous loss can disagree despite that knot being fixed by the bounds. The composite uses the existing RWA export and omits dissipators; this is a computational example, not a coupled-device accuracy claim.",
            "cases": cases,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "Projects/entanglix-research/grape-propagator",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tests/probes/grape_propagator_results.json"
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        request = json.loads(args.worker.read_text())
        save(Path(request["output"]), worker(request))
        return
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if not args.rebuild:
        metadata = prepare(workspace)
        save(workspace / "metadata.json", metadata)
        environment = {
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
        }
        for case in metadata["inputs"]:
            for loss in ("previous", "expm"):
                name = f"{case['name']}-{loss}"
                result = workspace / "measurements" / f"{name}.json"
                request = workspace / "requests" / f"{name}.json"
                save(
                    request,
                    {
                        "loss": loss,
                        "case": case,
                        "previous": metadata["previous"]["source"],
                        "output": str(result),
                    },
                )
                print(f"measuring {name}", flush=True)
                with (request.parent / f"{name}.log").open("w") as log:
                    subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker", str(request)],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
    summarize(workspace, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
