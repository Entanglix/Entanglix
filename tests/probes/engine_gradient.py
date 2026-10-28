"""Bounded CPU measurements of the shared dynamiqs engine and a probe-only scan.

Run with .venv-runner/bin/python tests/probes/engine_gradient.py. Uses the existing
runner environment and grape-propagator composite input, without syncing packages.
Raw inputs, gradients, states, step counts and logs stay in --workspace. --rebuild
only summarizes them. No production code, GPU, agent session or package changes.
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = {
    "JAX_PLATFORMS": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "JAX_ENABLE_COMPILATION_CACHE": "false",
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(workspace):
    import tomllib

    import numpy as np
    import tomli_w

    from entanglix.models import load_system

    shutil.copyfile(Path(__file__), workspace / "measurement-driver.py")
    paths = {}
    for levels in (3, 6, 10):
        directory = workspace / "inputs" / str(levels)
        directory.mkdir(parents=True, exist_ok=True)
        definition = tomllib.loads((ROOT / "examples/transmon-control/system.toml").read_text())
        definition["truncation"]["retained_levels"] = levels
        path = directory / "system.toml"
        path.write_text(tomli_w.dumps(definition))
        system = load_system(path)
        # Explicit arrays preserve this circuit's eigenbasis and charge matrix, while
        # declaring an adjacent-level lowering operator for the open-system cases.
        lowering = np.diag(np.sqrt(np.arange(1, levels)), 1)
        np.savez(
            directory / "operators.npz",
            H=system.hamiltonian_ghz,
            charge=system.drives["charge_drive"],
            lower=lowering,
        )
        explicit = {
            "format_version": 1,
            "id": system.id,
            "units": {"energy": "GHz", "time": "ns"},
            "model": {
                "kind": "operators",
                "dimensions": [levels],
                "basis": "isolated_eigenstates",
                "array_file": "operators.npz",
                "hamiltonian": "H",
                "operators": {"charge": "charge", "lower": "lower"},
            },
            "drives": [{"id": "charge_drive", "factors": ["charge"]}],
            "dissipators": [
                {"id": "relaxation", "factors": ["lower"], "rate": {"value": 0.03, "unit": "1/ns"}}
            ],
        }
        path = directory / "open.toml"
        path.write_text(tomli_w.dumps(explicit))
        paths[str(levels)] = str(path)
    composite = workspace / "inputs/composite"
    composite.mkdir(parents=True, exist_ok=True)
    original = workspace.parent / "grape-propagator/inputs/qubit-resonator-30-20"
    for name in ("system.toml", "qubit.toml", "control.toml"):
        shutil.copyfile(original / name, composite / name)
    paths["30"] = str(composite / "system.toml")
    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": sha(Path(__file__)),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                ROOT / "src/entanglix/engines/dynamiqs_jax.py",
                ROOT / "src/entanglix/engines/frames.py",
                ROOT / "src/entanglix/engines/qutip_cpu.py",
                ROOT / "uv.lock",
            )
        },
        "inputs": paths,
        "input_sha256": {
            str(p.relative_to(workspace)): sha(p)
            for p in (workspace / "inputs").rglob("*")
            if p.is_file()
        },
        "environment": SETTINGS,
    }


def case_input(case):
    from dataclasses import replace

    import numpy as np

    from entanglix.evolution import Carrier, EvolutionInput, RotatingFrame
    from entanglix.models import load_system

    system = load_system(case["system"])
    if not case["open"]:
        system = replace(system, dissipators={}, dissipator_factors={})
    n = case["intervals"]
    values = 0.004 + 0.002 * np.cos(np.arange(20) * 0.7)
    if case["pulse"] == "zero_first":
        values[0] = 0
    elif case["pulse"] == "zero_interior":
        values[10] = 0
    elif case["pulse"] == "all_zero":
        values[:] = 0
    values = np.repeat(values, n // 20) if n >= 20 else values[:n]
    if case.get("scalar"):
        values = np.array(0.004)
    state = np.zeros(np.prod(system.dimensions), dtype=complex)
    if len(system.dimensions) == 1:
        indices = [0, 1] if case["initial"] == "superposition" else [1, 2]
        state[indices] = [1, np.exp(1j * np.pi / 4)]
        frequencies = {system.id: 5.482065688844835}
        drive = "charge_drive"
    else:
        state[[1, system.dimensions[1] + 1]] = [1, np.exp(1j * np.pi / 4)]
        frequencies = {"q": 5.482065688844835, "r": 6.2}
        drive = next(iter(system.drives))
    state /= np.linalg.norm(state)
    request = EvolutionInput(
        state,
        np.linspace(0, 20, n + 1),
        drive_amplitudes_ghz={drive: values},
        carriers={drive: Carrier(5.482065688844835)},
        frame=RotatingFrame(frequencies, rwa=True) if case["frame"] == "rotating" else None,
    )
    return system, request, values, drive


def probe_scan(system, request, amplitudes, drive, method, gradient):
    """Only the piecewise branch changes; scalar controls keep one whole solve."""
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp
    import numpy as np

    from entanglix.engines.frames import matrix, prepare_dynamics
    from entanglix.evolution import normalize_input

    normalized = normalize_input(system, request)
    prepared = prepare_dynamics(system, normalized)
    state = jnp.asarray(prepared.initial_density(normalized))
    base = system.hamiltonian_ghz if request.frame is None else matrix(prepared.hamiltonian)
    h0 = dq.asqarray(2 * np.pi * base, dims=system.dimensions)
    operator = dq.asqarray(2 * np.pi * matrix(prepared.drives[drive]), dims=system.dimensions)
    collapse = [dq.asqarray(matrix(terms), dims=system.dimensions) for terms in prepared.collapse]
    carrier = request.carriers[drive] if request.frame is None else None

    def solve(raw, start, end, value):
        h = h0 + (
            value * operator
            if carrier is None
            else dq.modulated(
                lambda t: (
                    value * jnp.cos(2 * jnp.pi * carrier.frequency_ghz * t + carrier.phase_rad)
                ),
                operator,
            )
        )
        solved = dq.mesolve(
            h,
            collapse,
            dq.asqarray(raw, dims=system.dimensions),
            jnp.stack([start, end]),
            method=method,
            gradient=gradient,
            progress_meter=False,
        )
        raw = solved.states[-1].to_jax()
        count = solved.infos.nsteps if solved.infos is not None else jnp.array(-1)
        return raw, (raw, count)

    if amplitudes.ndim:

        def step(raw, values):
            start, end, value = values
            return solve(raw, start, end, value)

        _, (states, counts) = jax.lax.scan(
            step,
            state,
            (jnp.asarray(request.times_ns[:-1]), jnp.asarray(request.times_ns[1:]), amplitudes),
        )
        states = jnp.concatenate([state[None], states])
    else:
        h = h0 + (
            amplitudes * operator
            if carrier is None
            else dq.modulated(
                lambda t: (
                    amplitudes * jnp.cos(2 * jnp.pi * carrier.frequency_ghz * t + carrier.phase_rad)
                ),
                operator,
            )
        )
        solved = dq.mesolve(
            h,
            collapse,
            dq.asqarray(state, dims=system.dimensions),
            request.times_ns,
            method=method,
            gradient=gradient,
            progress_meter=False,
        )
        states = solved.states.to_jax()
        counts = jnp.asarray([solved.infos.nsteps]) if solved.infos is not None else jnp.array([-1])
    return prepared.lab_states(states, request.times_ns, xp=jnp), counts


def references(system, request, controls, drive, coordinates):
    from dataclasses import replace

    import numpy as np
    from scipy.linalg import expm

    from entanglix.engines import QuTiPIntegrator
    from entanglix.engines.frames import matrix, prepare_dynamics
    from entanglix.evolution import normalize_input

    reference = replace(request, rtol=1e-12, atol=1e-14)
    normalized = normalize_input(system, reference)
    prepared = prepare_dynamics(system, normalized)

    def values(amplitudes, exact=False):
        if not exact:
            states = (
                QuTiPIntegrator()
                .evolve(system, replace(reference, drive_amplitudes_ghz={drive: amplitudes}))
                .states
            )
        else:
            state = prepared.initial_density(normalized)
            base = 2 * np.pi * matrix(prepared.hamiltonian)
            operator = 2 * np.pi * matrix(prepared.drives[drive])
            collapse = [matrix(t) for t in prepared.collapse]
            d = len(state)
            identity = np.eye(d)
            states = [state]
            for amplitude in np.broadcast_to(amplitudes, (len(request.times_ns) - 1,)):
                h = base + amplitude * operator
                if collapse:
                    generator = -1j * (np.kron(identity, h) - np.kron(h.T, identity))
                    for c in collapse:
                        cc = c.conj().T @ c
                        generator += np.kron(c.conj(), c) - 0.5 * (
                            np.kron(identity, cc) + np.kron(cc.T, identity)
                        )
                    state = (
                        expm((request.times_ns[1] - request.times_ns[0]) * generator)
                        @ state.ravel(order="F")
                    ).reshape((d, d), order="F")
                else:
                    u = expm(-1j * (request.times_ns[1] - request.times_ns[0]) * h)
                    state = u @ state @ u.conj().T
                states.append(state)
            states = prepared.lab_states(states, request.times_ns)
        return np.real(np.diag(states[-1]))[[1, 2]], np.asarray(states)

    baseline, states = values(controls)
    finite = {}
    exact_finite = {}
    for epsilon in (1e-5, 1e-6):
        columns = []
        exact_columns = []
        for index in coordinates:
            displacement = np.zeros_like(controls)
            displacement.flat[index] = epsilon
            columns.append(
                (values(controls + displacement)[0] - values(controls - displacement)[0])
                / (2 * epsilon)
            )
            if request.frame is not None:
                exact_columns.append(
                    (
                        values(controls + displacement, True)[0]
                        - values(controls - displacement, True)[0]
                    )
                    / (2 * epsilon)
                )
        finite[str(epsilon)] = np.array(columns).T.tolist()
        if exact_columns:
            exact_finite[str(epsilon)] = np.array(exact_columns).T.tolist()
    return {
        "values": baseline.tolist(),
        "states": states.tolist(),
        "finite_differences": finite,
        "expm_finite_differences": exact_finite,
    }


def worker(config):
    from dataclasses import replace
    from importlib.metadata import version

    import dynamiqs as dq
    import jax
    import jax.numpy as jnp
    import numpy as np

    from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

    jax.config.update("jax_enable_x64", True)
    dq.set_precision("double")
    system, request, controls, drive = case_input(config)
    if config.get("tight"):
        request = replace(request, rtol=1e-12, atol=1e-14)
    method = (
        dq.method.Expm()
        if config.get("method") == "expm"
        else dq.method.Tsit5(rtol=request.rtol, atol=request.atol, max_steps=request.max_steps)
    )
    gradient = {
        "direct": dq.gradient.Direct,
        "forward": dq.gradient.Forward,
        "checkpoint": dq.gradient.BackwardCheckpointed,
    }.get(config.get("gradient"), lambda: None)()
    original = dq.mesolve
    counts = []
    accepted = []

    def capture(*args, **kwargs):
        kwargs.update(method=method, gradient=gradient)
        solved = original(*args, **kwargs)
        counts.append(solved.infos.nsteps if solved.infos is not None else jnp.array(-1))
        accepted.append(
            solved.infos.naccepted
            if solved.infos is not None and hasattr(solved.infos, "naccepted")
            else jnp.array(-1)
        )
        return solved

    dq.mesolve = capture
    engine = DynamiqsIntegrator(device_kind="cpu")

    def trajectory(amplitudes):
        counts.clear()
        accepted.clear()
        if config.get("variant") == "scan":
            return probe_scan(system, request, amplitudes, drive, method, gradient)
        result = engine.evolve_jax(system, request, drive_amplitudes_ghz={drive: amplitudes})
        return result.states, jnp.stack(counts)

    result = {
        "case": config,
        "worker_sha256": sha(Path(__file__)),
        "versions": {
            name: version(name) for name in ("jax", "dynamiqs", "diffrax", "qutip", "scipy")
        },
        "controls": controls.tolist(),
        "initial_state_real": request.initial_state.real.tolist(),
        "initial_state_imag": request.initial_state.imag.tolist(),
        "scalar_preserves_whole_window_solve": bool(config.get("scalar")),
    }
    if config.get("kind") == "plain":
        elapsed = []
        for _ in range(3):
            started = time.perf_counter()
            if config["variant"] == "engine":
                counts.clear()
                states = np.asarray(engine.evolve(system, request).states)
                steps = jnp.stack(counts)
            else:
                states, steps = jax.block_until_ready(trajectory(jnp.asarray(controls)))
                states = np.asarray(states)
            elapsed.append(time.perf_counter() - started)
        result.update(
            cold_evolve_seconds=elapsed[0],
            warm_evolve_seconds=elapsed[1:],
            steps=np.asarray(steps).tolist(),
        )
        return result

    def objective(amplitudes, index):
        states, steps = trajectory(amplitudes)
        return jnp.real(states[-1, index, index]), (states, steps)

    if config.get("gradient") == "forward":

        def forward(amplitudes, index):
            values, aux = objective(amplitudes, index)
            derivative = jax.jacfwd(lambda a: objective(a, index)[0])(amplitudes)
            return (values, aux), derivative

        function = jax.jit(forward)
    else:
        function = jax.jit(jax.value_and_grad(objective, has_aux=True))
    amplitudes = jnp.asarray(controls)
    started = time.perf_counter()
    lowered = function.lower(amplitudes, jnp.array(1))
    lowering = time.perf_counter() - started
    started = time.perf_counter()
    compiled = lowered.compile()
    compilation = time.perf_counter() - started
    started = time.perf_counter()
    (value, (states, steps)), derivative = jax.block_until_ready(compiled(amplitudes, jnp.array(1)))
    execution = time.perf_counter() - started
    (second, _), second_derivative = jax.block_until_ready(compiled(amplitudes, jnp.array(2)))
    warm = []
    for _ in range(3):
        started = time.perf_counter()
        jax.block_until_ready(compiled(amplitudes, jnp.array(1)))
        warm.append(time.perf_counter() - started)
    result.update(
        lowering_seconds=lowering,
        compile_seconds=compilation,
        cold_call_seconds=lowering + compilation + execution,
        warm_call_median_seconds=statistics.median(warm),
        values=[float(value), float(second)],
        gradients=np.stack([derivative, second_derivative]).reshape(2, -1).tolist(),
        steps=np.asarray(steps).tolist(),
        states_real=np.asarray(states).real.tolist(),
        states_imag=np.asarray(states).imag.tolist(),
        devices=sorted({d.platform for d in states.devices()}),
        dtype=str(states.dtype),
    )
    if config.get("reference"):
        coordinates = list(range(controls.size))
        if config["frame"] == "lab" and not config.get("scalar"):
            coordinates = sorted({0, len(controls) // 2, 1, len(controls) - 2})
        reference = references(system, request, controls, drive, coordinates)
        reference_states = np.asarray(reference.pop("states"))
        result.update(
            reference=reference,
            coordinates_checked=coordinates,
            maximum_state_difference=float(np.max(abs(np.asarray(states) - reference_states))),
        )
    # An ordinary per-interval replay exposes the adaptive solver's accepted count.
    if config.get("reference"):
        trajectory(amplitudes)
        result["accepted_steps"] = [int(x) for x in accepted]
    return result


def evaluate(config, workspace, deadline):
    name = config["name"]
    request = workspace / "requests" / f"{name}.json"
    output = workspace / "measurements" / f"{name}.json"
    save(request, {**config, "output": str(output)})
    environment = {**os.environ, **SETTINGS}
    if config.get("kind") == "plain":
        environment["JAX_LOG_COMPILES"] = "1"
    timeout = min(
        100 if config.get("method") != "expm" or config["dimension"] < 30 else 75,
        deadline - time.monotonic(),
    )
    if timeout < 5:
        return None
    print(f"measuring {name} (limit {timeout:.0f}s)", flush=True)
    started = time.monotonic()
    with (request.parent / f"{name}.log").open("w") as log:
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(request)],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            if completed.returncode:
                save(
                    output,
                    {
                        "case": config,
                        "error": (request.parent / f"{name}.log").read_text()[-7000:],
                        "exit_code": completed.returncode,
                    },
                )
        except subprocess.TimeoutExpired:
            save(
                output,
                {
                    "case": config,
                    "error": f"Worker exceeded {timeout:.1f} second bound",
                    "timed_out": True,
                },
            )
    item = json.loads(output.read_text())
    item["process_seconds"] = time.monotonic() - started
    if config.get("kind") == "plain":
        lines = (request.parent / f"{name}.log").read_text().splitlines()
        item["jax_compile_log_counts"] = {
            token: sum(token in line for line in lines)
            for token in ("Compiling jit(_mesolve)", "Finished XLA compilation")
        }
    save(output, item)
    return item


def comparison(item):
    import numpy as np

    reference = item["reference"]
    coordinates = item["coordinates_checked"]
    analytic = np.asarray(item["gradients"])[:, coordinates]
    coarse = np.asarray(reference["finite_differences"]["1e-05"])
    fine = np.asarray(reference["finite_differences"]["1e-06"])
    error = abs(analytic - fine)
    spread = abs(fine - coarse)
    exact = reference["expm_finite_differences"]
    exact_fine = np.asarray(exact["1e-06"]) if exact else None
    return {
        "maximum_gradient_error": float(error.max()),
        "intervals_exceeding_1e_minus_7": sorted(
            {coordinates[i] for _, i in np.argwhere(error > 1e-7)}
        ),
        "finite_difference_spread": float(spread.max()),
        "reference_resolves_1e_minus_7": bool(
            spread.max() < 1e-7 and (not exact or np.max(abs(fine - exact_fine)) < 1e-7)
        ),
        "maximum_value_difference": float(
            np.max(abs(np.asarray(item["values"]) - reference["values"]))
        ),
        "expm_gradient_error": float(np.max(abs(analytic - exact_fine))) if exact else None,
        "qutip_expm_fd_difference": float(np.max(abs(fine - exact_fine))) if exact else None,
        "maximum_state_difference": item["maximum_state_difference"],
    }


def summarize(workspace, output):
    import numpy as np

    metadata = json.loads((workspace / "metadata.json").read_text())
    raw = [json.loads(p.read_text()) for p in sorted((workspace / "measurements").glob("*.json"))]
    by_name = {x["case"]["name"]: x for x in raw}
    measurements = []
    for item in raw:
        trimmed = {
            k: v
            for k, v in item.items()
            if k
            not in {
                "states_real",
                "states_imag",
                "gradients",
                "controls",
                "reference",
                "initial_state_real",
                "initial_state_imag",
            }
        }
        if item.get("reference"):
            trimmed["comparison"] = comparison(item)
        counterpart = by_name.get(item["case"].get("compare_to"))
        if counterpart and "error" not in item and "error" not in counterpart:

            def states(x):
                return np.asarray(x["states_real"]) + 1j * np.asarray(x["states_imag"])

            if "states_real" in item:
                trimmed["parity"] = {
                    "maximum_state_difference": float(
                        np.max(abs(states(item) - states(counterpart)))
                    ),
                    "maximum_gradient_difference": float(
                        np.max(abs(np.asarray(item["gradients"]) - counterpart["gradients"]))
                    ),
                }
            if counterpart.get("reference") and "gradients" in item:
                inherited = {
                    **item,
                    "reference": counterpart["reference"],
                    "coordinates_checked": counterpart["coordinates_checked"],
                    "maximum_state_difference": None,
                }
                trimmed["reference_comparison"] = comparison(inherited)
        measurements.append(trimmed)
    summary = {
        "cases_with_resolved_gradient_errors": sum(
            x.get("comparison", {}).get("reference_resolves_1e_minus_7", False)
            and x["comparison"]["maximum_gradient_error"] > 1e-7
            for x in measurements
        ),
        "errors_or_timeouts": sum("error" in x for x in measurements),
        "step_count_note": "Accepted step counts give average interval step size, not the largest actual step. A correlation alone does not prove the mechanism.",
        "method_note": "Forward uses jacfwd; default BackwardCheckpointed is a repeat of the existing adjoint. Expm is measured through mesolve, which exponentiates a D²-dimensional Liouvillian even with no collapse operators.",
        "scope_note": "Scan changes only the interval-valued branch and retains one mesolve and max_steps per interval. Scalar controls retain one whole-window solve. All timings synchronize outputs. Double precision is initialized before tracing both variants. Plain-call compile logs are retained.",
    }
    reproduction = [x for x in measurements if "comparison" in x]
    remedies = [x for x in measurements if "reference_comparison" in x and x["case"].get("suffix")]
    scans = [x for x in measurements if x["case"]["variant"] == "scan" and "parity" in x]
    missing = sorted(
        p.stem for p in (workspace / "requests").glob("*.json") if p.stem not in by_name
    )
    answers = {
        "1_reproduction": {
            "cases": len(reproduction),
            "resolved_disagreements_above_1e_minus_7": summary[
                "cases_with_resolved_gradient_errors"
            ],
            "maximum_gradient_error": max(
                (x["comparison"]["maximum_gradient_error"] for x in reproduction), default=None
            ),
            "unresolved_reference_cases": [
                x["case"]["name"]
                for x in reproduction
                if not x["comparison"]["reference_resolves_1e_minus_7"]
            ],
            "scope": "Both populations are real scalar objectives differentiated with value_and_grad. Initial states have equal populations and relative phase pi/4 on levels (0,1) or (1,2); open transmons use adjacent-level relaxation at 0.03/ns. Lab finite differences inspect four declared coordinates; rotating cases inspect every coordinate.",
        },
        "2_step_counts": [
            {
                "case": x["case"]["name"],
                "failing_coordinates": x["comparison"]["intervals_exceeding_1e_minus_7"],
                "step_counts_at_those_coordinates": [
                    x["steps"][i] for i in x["comparison"]["intervals_exceeding_1e_minus_7"]
                ],
                "accepted_counts": x.get("accepted_steps"),
            }
            for x in reproduction
            if x["comparison"]["maximum_gradient_error"] > 1e-7
        ],
        "3_remedies": [
            {
                "case": x["case"]["name"],
                "maximum_gradient_error": x["reference_comparison"]["maximum_gradient_error"],
                "compile_seconds": x["compile_seconds"],
                "warm_call_median_seconds": x["warm_call_median_seconds"],
            }
            for x in remedies
        ],
        "4_scan": {
            "parity_cases": len(scans),
            "maximum_state_difference": max(
                (x["parity"]["maximum_state_difference"] for x in scans), default=None
            ),
            "maximum_gradient_difference": max(
                (x["parity"]["maximum_gradient_difference"] for x in scans), default=None
            ),
            "restart": "A separate dq.mesolve call in each scan iteration starts from the preceding density matrix and receives the original per-solve max_steps. Scalar controls retain a single whole-window solve.",
        },
        "not_measured": missing,
    }
    save(
        output,
        {
            **metadata,
            "summary_script_sha256": sha(Path(__file__)),
            "measurement_driver": str(workspace / "measurement-driver.py"),
            "worker_provenance_note": "The script changed from 082efd69 to bb25789d during the run; the 21 earliest measurements and three failed workers carry no worker_sha256. The edit changed only the qubit-resonator initial state, the uncompiled evolve() path and summary code, and every measurement it affects was requested after it.",
            "workspace": str(workspace),
            "summary": summary,
            "answers": answers,
            "measurements": measurements,
            "downloads": [],
            "runtime_sessions": 0,
            "gpu_jobs": 0,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "Projects/entanglix-research/engine-gradient",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tests/probes/engine_gradient_results.json"
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        config = json.loads(args.worker.read_text())
        try:
            result = worker(config)
        except Exception:
            result = {"case": config, "error": traceback.format_exc()}
        save(Path(config["output"]), result)
        return
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if not args.rebuild:
        metadata = prepare(workspace)
        save(workspace / "metadata.json", metadata)
        deadline = time.monotonic() + 1800
        base = {
            "system": metadata["inputs"]["3"],
            "dimension": 3,
            "intervals": 20,
            "initial": "superposition",
            "frame": "rotating",
            "open": False,
            "pulse": "zero_first",
            "variant": "engine",
        }
        cases = []
        for frame, opened in (("rotating", False), ("rotating", True), ("lab", False)):
            for initial in ("superposition", "upper_coherence"):
                for pulse in ("zero_first", "zero_interior", "all_zero", "nonzero"):
                    name = f"{frame}-{'open' if opened else 'closed'}-{initial}-{pulse}-20"
                    cases.append(
                        {
                            **base,
                            "name": name,
                            "frame": frame,
                            "open": opened,
                            "initial": initial,
                            "pulse": pulse,
                            "reference": True,
                        }
                    )
        cases.extend(
            {**c, "name": c["name"][:-2] + "40", "intervals": 40}
            for c in list(cases)
            if c["frame"] == "rotating" and c["initial"] == "superposition"
        )
        completed = []
        for case in cases:
            item = evaluate(case, workspace, deadline)
            if item:
                completed.append(item)
        representatives = []
        for frame, opened in (("rotating", False), ("rotating", True), ("lab", False)):
            failures = [
                x
                for x in completed
                if "reference" in x
                and x["case"]["frame"] == frame
                and x["case"]["open"] == opened
                and comparison(x)["maximum_gradient_error"] > 1e-7
            ]
            if failures:
                representatives.append(
                    max(failures, key=lambda x: comparison(x)["maximum_gradient_error"])["case"]
                )
        for case in representatives:
            variants = [
                {"tight": True, "suffix": "tight"},
                {"gradient": "direct", "suffix": "direct"},
                {"gradient": "forward", "suffix": "forward"},
                {"gradient": "checkpoint", "suffix": "checkpoint-control"},
            ]
            if case["frame"] == "rotating":
                variants.append({"method": "expm", "suffix": "expm"})
            for change in variants:
                evaluate(
                    {
                        **case,
                        **change,
                        "name": case["name"] + "-" + change["suffix"],
                        "reference": False,
                        "compare_to": case["name"],
                    },
                    workspace,
                    deadline,
                )
        for levels in (3, 6, 10, 30):
            for opened in (False, True):
                for method in ("tsit5", "expm"):
                    evaluate(
                        {
                            **base,
                            "name": f"scaling-{levels}-{opened}-{method}",
                            "system": metadata["inputs"][str(levels)],
                            "dimension": levels,
                            "open": opened,
                            "method": method,
                        },
                        workspace,
                        deadline,
                    )
        for frame in ("rotating", "lab"):
            for opened in (False, True):
                for scalar in (False, True):
                    for intervals in (20, 40) if frame == "rotating" and not scalar else (20,):
                        name = f"scan-parity-{frame}-{opened}-{scalar}-{intervals}"
                        for variant in ("engine", "scan"):
                            evaluate(
                                {
                                    **base,
                                    "name": name + "-" + variant,
                                    "frame": frame,
                                    "open": opened,
                                    "scalar": scalar,
                                    "intervals": intervals,
                                    "variant": variant,
                                    **(
                                        {"compare_to": name + "-engine"}
                                        if variant == "scan"
                                        else {}
                                    ),
                                },
                                workspace,
                                deadline,
                            )
        for frame in ("rotating", "lab"):
            for intervals in (20, 200):
                for variant in ("engine", "scan"):
                    evaluate(
                        {
                            **base,
                            "name": f"plain-{frame}-{intervals}-{variant}",
                            "frame": frame,
                            "intervals": intervals,
                            "variant": variant,
                            "kind": "plain",
                        },
                        workspace,
                        deadline,
                    )
        metadata["elapsed_measurement_seconds"] = 1800 - (deadline - time.monotonic())
        metadata["budget_exhausted"] = time.monotonic() >= deadline - 5
        save(workspace / "metadata.json", metadata)
    summarize(workspace, args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
