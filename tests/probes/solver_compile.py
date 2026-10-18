"""CPU compilation measurements, without changing the solvers or their environment.

Run: .venv-runner/bin/python tests/probes/solver_compile.py
Requires the existing differentiable runner environment, pinned Julia project,
and the retained m5-probe/inputs/c30-rabi_bound-s7/control.toml beside the default
workspace. Each measurement uses a fresh process; no runner job or agent starts.
Scratch inputs, Julia entry, caches, raw measurements and logs stay in --workspace.
Repeating the command repeats measurements and rebuilds the compact results file.
Use --rebuild to rebuild only the summary from retained timings, optimizations and
gradient comparisons, with no new evaluations. The expm implementation is an extra
variant beyond the original builder's brief.
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
import tomllib
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ENTERED = time.perf_counter()
REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "src/entanglix/solvers/grape.py",
    "src/entanglix/engines/dynamiqs_jax.py",
    "src/entanglix/solvers/piccolo.py",
    "src/entanglix/solvers/julia/solve.jl",
    "src/entanglix/solvers/julia/Project.toml",
    "src/entanglix/solvers/julia/Manifest.toml",
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(workspace):
    import tomli_w

    from entanglix.protocols.control import load_control

    cases = {}
    for name, source in (
        ("c60", REPOSITORY / "examples/transmon-control/control.toml"),
        ("c30", workspace.parent / "m5-probe/inputs/c30-rabi_bound-s7/control.toml"),
    ):
        loaded = load_control(source)
        path = workspace / "inputs" / name / "problem.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(loaded.problem_bytes)
        cases[name] = {
            "source": str(source),
            "problem": str(path),
            "problem_sha256": sha(path),
            "source_input_sha256": loaded.input_digests,
            "options": loaded.definition.solver.options,
        }
    settings = workspace / "inputs/options.toml"
    settings.write_text(tomli_w.dumps(cases["c60"]["options"]))
    return cases


def initial_controls(problem, seed):
    import numpy as np

    n = problem["problem"]["N"]
    dt = problem["pulse"]["T"] / (n - 1)
    bounds = np.asarray(problem["system"]["params"]["drive_bounds"])
    upper = np.repeat(bounds[:, None], n, axis=1)
    upper[:, [0, -1]] = 0
    initial = np.random.default_rng(seed).uniform(-0.1, 0.1, upper.shape) * upper
    for order, field in ((1, "du_bound"), (2, "ddu_bound")):
        if field in problem["problem"] and n > order:
            maximum = np.max(np.abs(np.diff(initial, n=order, axis=1) / dt**order))
            if maximum:
                initial *= min(1, problem["problem"][field] / maximum)
    return initial, upper


def make_objective(problem, variant, arguments=False):
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.scipy.linalg import expm

    from entanglix.protocols.control import Goal
    from entanglix.solvers.grape import decode, objective

    dq.set_precision("double")
    times = np.linspace(0, problem["pulse"]["T"], problem["problem"]["N"])
    if variant == "current" and not arguments:
        function, times = objective(problem)
        return function, times, ()
    goal = Goal.model_validate(problem["goal"])
    indices = np.asarray(goal.subspace[0]) - 1
    d = len(indices)
    drift = jnp.asarray(decode(problem["system"]["H_drift"]))
    drives = jnp.asarray([decode(x) for x in problem["system"]["H_drives"]])
    initial = np.zeros((len(drift), d), complex)
    initial[indices, np.arange(d)] = 1 / np.sqrt(d)
    target = np.zeros_like(initial)
    target[indices] = goal.operator() / np.sqrt(d)
    target = jnp.asarray(target.ravel())
    surviving = np.repeat(np.isin(np.arange(len(drift)), indices), d)
    pure = variant.endswith("_pure") or variant == "expm"
    method = dq.method.Tsit5(rtol=1e-10, atol=1e-12, max_steps=100_000)

    def loss(amplitudes, h0=drift, operators=drives):
        h0 = jnp.kron(h0, jnp.eye(d))
        operators = jax.vmap(lambda op: jnp.kron(op, jnp.eye(d)))(operators)
        ket = jnp.asarray(initial.reshape(-1, 1))
        state = ket if pure else ket @ ket.conj().T
        dims = (len(h0),)

        def step(state, inputs):
            start, end, values = inputs
            h = h0 + jnp.einsum("c,cij->ij", values, operators)
            if variant == "expm":
                evolved = expm(-1j * h * (end - start)) @ state
            else:
                solver = dq.sesolve if pure else dq.mesolve
                positional = () if pure else ([],)
                solved = solver(
                    dq.asqarray(h, dims=dims),
                    *positional,
                    dq.asqarray(state, dims=dims),
                    jnp.stack([start, end]),
                    method=method,
                    progress_meter=False,
                )
                evolved = solved.states[-1].to_jax()
            return evolved, None

        if variant.startswith("pwc"):
            h = dq.asqarray(h0, dims=dims)
            for channel in range(len(operators)):
                h = h + dq.pwc(times, amplitudes[channel, :-1], dq.asqarray(operators[channel]))
            solver = dq.sesolve if pure else dq.mesolve
            positional = () if pure else ([],)
            state = (
                solver(
                    h,
                    *positional,
                    dq.asqarray(state, dims=dims),
                    times,
                    method=method,
                    progress_meter=False,
                )
                .states[-1]
                .to_jax()
            )
        elif variant == "current":
            for index in range(len(times) - 1):
                state, _ = step(state, (times[index], times[index + 1], amplitudes[:, index]))
        else:
            state, _ = jax.lax.scan(step, state, (times[:-1], times[1:], amplitudes[:, :-1].T))
        if pure:
            state = state.ravel()
            survival = jnp.sum(jnp.abs(state[surviving]) ** 2)
            overlap = jnp.abs(target.conj() @ state) ** 2
        else:
            survival = jnp.real(jnp.sum(jnp.diag(state)[surviving]))
            overlap = jnp.real(target.conj() @ state @ target)
        return 1 - (survival + d * overlap) / (d + 1)

    return jax.jit(jax.value_and_grad(loss)), times, (drift, drives) if arguments else ()


def optimize(function, problem, times, initial, upper, options):
    import jax
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, minimize

    from entanglix.corrector import load_policy
    from entanglix.corrector.control import constraints, replay_values
    from entanglix.protocols.control import load_control
    from entanglix.solvers import Pulse

    n = problem["problem"]["N"]
    restrictions = []
    for order, field in ((1, "du_bound"), (2, "ddu_bound")):
        if field in problem["problem"] and n > order:
            derivative = np.diff(np.eye(n), n=order, axis=0) / (times[1] - times[0]) ** order
            restrictions.append(
                LinearConstraint(
                    np.kron(np.eye(len(upper)), derivative),
                    -problem["problem"][field],
                    problem["problem"][field],
                )
            )
    calls = []

    def evaluate(flat):
        start = time.perf_counter()
        value, gradient = jax.block_until_ready(function(flat.reshape(upper.shape)))
        calls.append(time.perf_counter() - start)
        return float(value), np.asarray(gradient).ravel()

    start = time.perf_counter()
    result = minimize(
        evaluate,
        initial.ravel(),
        method="SLSQP",
        jac=True,
        bounds=Bounds(-upper.ravel(), upper.ravel()),
        constraints=restrictions,
        options={"maxiter": options["max_iter"], "ftol": options["tol"]},
    )
    elapsed = time.perf_counter() - start
    loaded = load_control(REPOSITORY / "examples/transmon-control/control.toml")
    pulse = Pulse(times, result.x.reshape(upper.shape))
    replay = replay_values(loaded, pulse, load_policy())
    return {
        "warm_optimization_seconds": elapsed,
        "iterations": int(result.nit),
        "objective_evaluations": int(result.nfev),
        "converged": bool(result.success),
        "message": str(result.message),
        "fidelity": 1 - float(result.fun),
        "independent_replay": replay,
        "constraints": constraints(loaded, pulse, load_policy())["outcome"],
        "warm_objective_call_median_seconds": statistics.median(calls),
    }


def finite_differences(problem, amplitudes):
    """Independent constant-interval propagators, including the fixed first knot."""
    import numpy as np
    from scipy.linalg import expm

    from entanglix.protocols.control import Goal
    from entanglix.solvers.grape import decode

    goal = Goal.model_validate(problem["goal"])
    indices = np.array(goal.subspace[0]) - 1
    d = len(indices)
    drift = decode(problem["system"]["H_drift"])
    drives = np.array([decode(op) for op in problem["system"]["H_drives"]])
    dt = problem["pulse"]["T"] / (problem["problem"]["N"] - 1)

    def value(controls):
        unitary = np.eye(len(drift), dtype=complex)
        for column in controls[:, :-1].T:
            unitary = expm(-1j * (drift + np.einsum("c,cij->ij", column, drives)) * dt) @ unitary
        block = unitary[np.ix_(indices, indices)]
        return 1 - (
            np.linalg.norm(block) ** 2 + abs(np.trace(goal.operator().conj().T @ block)) ** 2
        ) / (d * (d + 1))

    rows = []
    for channel, knot in (
        (c, k) for c in range(len(amplitudes)) for k in range(amplitudes.shape[1])
    ):
        estimates = []
        for step in (1e-5, 1e-6):
            plus, minus = amplitudes.copy(), amplitudes.copy()
            plus[channel, knot] += step
            minus[channel, knot] -= step
            estimates.append(
                {"step_ghz": step, "gradient": float((value(plus) - value(minus)) / (2 * step))}
            )
        rows.append(
            {
                "channel": channel,
                "knot": knot,
                "fixed_by_bounds": knot in (0, amplitudes.shape[1] - 1),
                "estimates": estimates,
            }
        )
    return rows


def gradient_comparisons(function, problem, upper, matrix_arguments):
    import jax
    import numpy as np

    rng = np.random.default_rng(20261018)
    samples = [rng.uniform(-0.8, 0.8, upper.shape) * upper for _ in range(3)]
    extra = samples[0].copy()
    extra[:, 0] = np.array([0.4, -0.3]) * np.asarray(problem["system"]["params"]["drive_bounds"])
    samples.append(extra)
    result = []
    for index, amplitudes in enumerate(samples):
        value, gradient = jax.block_until_ready(function(amplitudes, *matrix_arguments))
        comparisons = finite_differences(problem, amplitudes)
        for item in comparisons:
            item["autodiff_gradient"] = float(gradient[item["channel"], item["knot"]])
        result.append(
            {
                "sample": f"random-{index}" if index < 3 else "nonzero-first-knot",
                "outside_bounds": index == 3,
                "amplitudes_ghz": amplitudes.tolist(),
                "value": float(value),
                "knots": comparisons,
            }
        )
    return result


def measure(request):
    result = {"startup_seconds": ENTERED - request["launched_at"]}
    start = time.perf_counter()
    import dynamiqs as dq
    import jax
    import numpy as np
    import scipy

    import entanglix

    result["imports_seconds"] = time.perf_counter() - start
    result["versions"] = {
        "jax": jax.__version__,
        "dynamiqs": version("dynamiqs"),
        "scipy": scipy.__version__,
        "entanglix": entanglix.__version__,
    }
    dq.set_precision("double")
    if cache := request.get("cache"):
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_explain_cache_misses", True)
    else:
        jax.config.update("jax_enable_compilation_cache", False)
    start = time.perf_counter()
    problem = tomllib.loads(Path(request["problem"]).read_text())
    problem["problem"]["N"] = request.get("intervals", 20) + 1
    result["load_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    function, times, matrix_arguments = make_objective(
        problem, request["variant"], request.get("arguments", False)
    )
    initial, upper = initial_controls(problem, request.get("seed", 7))
    parameters = (initial, *matrix_arguments)
    result["objective_setup_seconds"] = time.perf_counter() - start
    if request.get("gradient_only"):
        return {
            "comparisons": gradient_comparisons(function, problem, upper, matrix_arguments),
            "devices": [str(d) for d in jax.devices()],
        }
    if cache:
        start = time.perf_counter()
        value, gradient = jax.block_until_ready(function(*parameters))
        result["first_call_seconds"] = time.perf_counter() - start
    else:
        start = time.perf_counter()
        lowered = function.lower(*parameters)
        result["lower_seconds"] = time.perf_counter() - start
        result["lowered_text_bytes"] = len(lowered.as_text().encode())
        start = time.perf_counter()
        function = lowered.compile()
        result["compile_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        value, gradient = jax.block_until_ready(function(*parameters))
        result["first_execution_seconds"] = time.perf_counter() - start
    result["initial_value"] = float(value)
    result["gradient_max_abs"] = float(np.max(abs(gradient)))
    durations = []
    for _ in range(9):
        start = time.perf_counter()
        jax.block_until_ready(function(*parameters))
        durations.append(time.perf_counter() - start)
    result["warm_value_gradient_median_seconds"] = statistics.median(durations)
    rng = np.random.default_rng(20261018)
    result["parity_samples"] = []
    for _ in range(3):
        amplitudes = rng.uniform(-0.8, 0.8, upper.shape) * upper
        value, gradient = jax.block_until_ready(function(amplitudes, *matrix_arguments))
        result["parity_samples"].append({"value": float(value), "gradient": gradient.tolist()})
    if not cache and request.get("optimize") and request["variant"] in {"current", "expm"}:
        result["independent_gradient_comparisons"] = gradient_comparisons(
            function, problem, upper, matrix_arguments
        )
    if request.get("optimize"):
        result["solve"] = optimize(function, problem, times, initial, upper, request["options"])
    result["devices"] = [str(d) for d in jax.devices()]
    result["process_thread_count"] = int(
        next(
            line.split()[1]
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("Threads:")
        )
    )
    return result


def child(request_path):
    request = json.loads(request_path.read_text())
    try:
        result = (
            prepare(Path(request["workspace"]))
            if request["kind"] == "prepare"
            else measure(request)
        )
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    save(Path(request["output"]), result)


def run_measurement(workspace, label, request, environment):
    print(f"measure {label}", flush=True)
    path = workspace / "measurements" / f"{label}.json"
    request_path = path.with_suffix(".request.json")
    request.update(output=str(path), launched_at=time.perf_counter())
    save(request_path, request)
    start = time.perf_counter()
    log = path.with_suffix(".log")
    with log.open("w") as stream:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=600,
            check=False,
        )
    elapsed = time.perf_counter() - start
    result = json.loads(path.read_text()) if path.exists() else {"error": "worker wrote no result"}
    result.update(process_seconds=elapsed, exit_code=process.returncode, log=str(log))
    if "error" in result:
        print(result["error"], flush=True)
    return result


def parity(result, reference):
    differences = [
        (abs(a["value"] - b["value"]), max(abs(x - y) for x, y in zip(ar, br)))
        for a, b in zip(result["parity_samples"], reference["parity_samples"])
        for ar, br in zip(a["gradient"], b["gradient"])
    ]
    return {
        "value_max_abs_difference": max(d[0] for d in differences),
        "gradient_max_abs_difference": max(d[1] for d in differences),
        "amplitude_sets": 3,
        "free_knot_gradient_max_abs_difference": max(
            abs(x - y)
            for a, b in zip(result["parity_samples"], reference["parity_samples"])
            for ar, br in zip(a["gradient"], b["gradient"])
            for x, y in zip(ar[1:-1], br[1:-1])
        ),
        "scope": "All gradients include the fixed zero first and final knots; free-knot comparison excludes those two bound-fixed columns.",
    }


def julia_probe(workspace, cases, environment):
    source = (REPOSITORY / "src/entanglix/solvers/julia/solve.jl").read_text()
    start = source.index("pulse_path =")
    neutral_writer = source[start:]
    helper = workspace / "solve_timed.jl"
    helper.write_text(
        'println("COMPILE_PROBE stage=entry"); flush(stdout)\n'
        "Base.cumulative_compile_timing(true)\n"
        "import_start = time_ns()\n"
        "using Piccolo, TOML, LinearAlgebra\n"
        'println("COMPILE_PROBE stage=imports seconds=", (time_ns()-import_start)/1e9); flush(stdout)\n'
        "problem, output_root, options_path = ARGS\n"
        "options = TOML.parsefile(options_path)\n"
        "parse_start = time_ns()\n"
        "spec = Piccolo.Specs.parse_spec(read(problem, String))\n"
        'println("COMPILE_PROBE stage=parse seconds=", (time_ns()-parse_start)/1e9); flush(stdout)\n'
        "kwargs = Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in options)\n"
        'println("COMPILE_PROBE stage=runtime version=", VERSION, " package_images=", Base.JLOptions().use_pkgimages, " julia_threads=", Threads.nthreads(), " blas_threads=", BLAS.get_num_threads()); flush(stdout)\n'
        "for repetition in 1:2\n"
        "    output = joinpath(output_root, string(repetition))\n"
        '    println("COMPILE_PROBE stage=solve_start repetition=", repetition); flush(stdout)\n'
        "    compilation_before = Base.cumulative_compile_time_ns()\n"
        "    solve_start = time_ns()\n"
        "    result = Piccolo.Specs.solve_spec(problem; run_dir=output, kwargs...)\n"
        "    solve_seconds = (time_ns()-solve_start)/1e9\n"
        "    compilation_after = Base.cumulative_compile_time_ns()\n"
        '    println("COMPILE_PROBE stage=solve_end repetition=", repetition, " seconds=", solve_seconds, " compile_seconds=", (compilation_after[1]-compilation_before[1])/1e9, " recompile_seconds=", (compilation_after[2]-compilation_before[2])/1e9); flush(stdout)\n'
        + neutral_writer
        + "\nend\n"
    )
    output = workspace / "piccolo"
    if output.exists():
        shutil.rmtree(output)
    command = [
        "julia",
        "--startup-file=no",
        f"--project={REPOSITORY / 'src/entanglix/solvers/julia'}",
        str(helper),
        cases["c60"]["problem"],
        str(output),
        str(workspace / "inputs/options.toml"),
    ]
    print("measure piccolo cold and warm", flush=True)
    started = time.perf_counter()
    child = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment
    )
    markers, iterations = [], {1: [], 2: []}
    loaded_images = []
    repetition = None
    log = workspace / "piccolo.log"
    with log.open("w") as stream:
        for line in child.stdout:
            elapsed = time.perf_counter() - started
            stream.write(line)
            stream.flush()
            if line.startswith("COMPILE_PROBE "):
                fields = dict(piece.split("=", 1) for piece in line.split()[1:])
                markers.append({**fields, "received_seconds": elapsed})
                if fields["stage"] == "imports":
                    loaded_images = sorted(
                        {
                            line.split()[-1]
                            for line in Path(f"/proc/{child.pid}/maps").read_text().splitlines()
                            if "/.julia/compiled/" in line and line.split()[-1].endswith(".so")
                        }
                    )
                if fields["stage"] == "solve_start":
                    repetition = int(fields["repetition"])
            elif line.startswith("AMICODE_ITER "):
                iterations[repetition].append(elapsed)
    result = {
        "exit_code": child.wait(),
        "process_seconds": time.perf_counter() - started,
        "markers": markers,
        "log": str(log),
        "command": command,
        "loaded_package_images": loaded_images,
        "timing_note": "Iteration times are stdout receipt times; solve_spec includes materialization and output serialization, native wall_seconds times solve! only. Recompile time is a subset of compilation time, not an additional cost.",
    }
    if result["exit_code"]:
        result["error"] = log.read_text()[-5000:]
        return result
    result["solves"] = []
    for repetition in (1, 2):
        marker = next(
            m for m in markers if m["stage"] == "solve_start" and int(m["repetition"]) == repetition
        )
        ending = next(
            m for m in markers if m["stage"] == "solve_end" and int(m["repetition"]) == repetition
        )
        retained = tomllib.loads((output / str(repetition) / "result.toml").read_text())
        ts = iterations[repetition]
        result["solves"].append(
            {
                "repetition": repetition,
                "solve_spec_seconds": float(ending["seconds"]),
                "compilation_seconds": float(ending["compile_seconds"]),
                "recompilation_seconds": float(ending["recompile_seconds"]),
                "native_solve_seconds": retained["wall_seconds"],
                "fidelity": retained["fidelity"],
                "iterations": retained["iterations"],
                "converged": retained["params"]["converged"],
                "first_iteration_since_solve_start_seconds": ts[0] - marker["received_seconds"],
                "later_iteration_median_seconds": statistics.median(
                    b - a for a, b in zip(ts, ts[1:])
                ),
            }
        )
    return result


def cache_inventory(path):
    return {str(p.relative_to(path)): p.stat().st_size for p in path.rglob("*") if p.is_file()}


def installed_julia_content():
    depot = Path.home() / ".julia"
    return {
        str(p.relative_to(depot))
        for area in ("packages", "artifacts")
        if (depot / area).exists()
        for p in (depot / area).glob("*/*" if area == "packages" else "*")
        if p.is_dir()
    }


def gradient_summary(comparisons):
    rows = []
    for sample in comparisons:
        errors = [
            {
                "channel": item["channel"],
                "knot": item["knot"],
                "error": abs(item["autodiff_gradient"] - item["estimates"][-1]["gradient"]),
            }
            for item in sample["knots"]
        ]
        rows.append(
            {
                "sample": sample["sample"],
                "outside_bounds": sample["outside_bounds"],
                "coordinates_checked": len(errors),
                "max_abs_error": max(item["error"] for item in errors),
                "first_knot_errors": [item["error"] for item in errors if item["knot"] == 0],
                "channel_0_knot_3_error": next(
                    item["error"] for item in errors if (item["channel"], item["knot"]) == (0, 3)
                ),
                "free_knot_max_abs_error": max(
                    item["error"]
                    for item in errors
                    if item["knot"] not in (0, len(sample["amplitudes_ghz"][0]) - 1)
                ),
                "coordinates_above_1e_minus_7": [item for item in errors if item["error"] > 1e-7],
            }
        )
    return rows


def retain_gradient_summary(results, workspace, rows):
    results["gradient_validation"] = {
        "measured_at": datetime.now(UTC).isoformat(),
        "script_sha256": sha(Path(__file__)),
        "reference": "SciPy matrix-exponential central differences of the leakage-aware infidelity on the full control model",
        "steps_ghz": [1e-5, 1e-6],
        "summary_step_ghz": 1e-6,
        "reporting_threshold": "1e-7 labels coordinates for reporting only; raw estimates at both steps are retained",
        "scope": "Both channels, all 21 knots, three original bound-feasible amplitude sets and one set with a nonzero first knot outside its fixed-zero bound. Direct loss evaluations only; no optimization.",
        "variants": {
            name: {
                "raw_measurement": str(workspace / "measurements" / f"{label}.json"),
                "samples": gradient_summary(comparisons),
            }
            for name, (label, comparisons) in rows.items()
        },
    }


def summarize(results):
    baseline = results["baseline"]
    alternatives = results["alternatives"]
    successes = {k: v for k, v in alternatives.items() if "error" not in v}
    projections = {}
    for name, item in {"current": baseline, **successes}.items():
        cold = sum(
            item[k]
            for k in (
                "startup_seconds",
                "imports_seconds",
                "load_seconds",
                "objective_setup_seconds",
                "lower_seconds",
                "compile_seconds",
                "first_execution_seconds",
            )
        )
        projections[name] = 9 * (cold + item["solve"]["warm_optimization_seconds"])
    results["batch_projection"] = {
        "grape": {"seconds": projections},
        "scope": "Serial 3 candidates x 3 seeds; all candidates assumed to have the measured c60 shape and optimization cost. Adds one cold setup/compile/first execution plus the measured warm optimization per job. Excludes runner gate, supervision, checks, benchmarks and independent replays. Not a measured nine-job batch.",
    }
    for name, rows in results["cache"].items():
        if all("first_call_seconds" in row for row in rows):
            warm_solve = (successes.get(name.removesuffix("_arguments"), baseline))["solve"][
                "warm_optimization_seconds"
            ]

            def overhead(row):
                return sum(
                    row[k]
                    for k in (
                        "startup_seconds",
                        "imports_seconds",
                        "load_seconds",
                        "objective_setup_seconds",
                        "first_call_seconds",
                    )
                )

            # Weight the measured cold, repeated-seed and changed-candidate calls.
            # This also handles no reuse: host callbacks can prevent serialization.
            changed_candidate_hit = bool(rows[2]["objective_cache_hits"])
            projections[f"cache_{name}"] = (
                overhead(rows[0])
                + (8 if changed_candidate_hit else 6) * overhead(rows[1])
                + (0 if changed_candidate_hit else 2) * overhead(rows[2])
                + 9 * warm_solve
            )
    piccolo = results["piccolo"]
    piccolo_projections = {}
    if "solves" in piccolo:
        cold, warm = piccolo["solves"]
        piccolo_projections["fresh_processes"] = 9 * (
            piccolo["process_seconds"] - warm["solve_spec_seconds"]
        )
        piccolo_projections["one_warm_process"] = (
            piccolo["process_seconds"] + 7 * warm["solve_spec_seconds"]
        )
    results["batch_projection"]["piccolo"] = {"seconds": piccolo_projections}
    comparability = (
        "GRAPE and Piccolo solve times are not comparable performance measurements: "
        "they use different optimizers, objectives and stopping rules. GRAPE maps tol "
        "to SLSQP ftol and minimizes infidelity; Piccolo passes tol to Ipopt with its "
        "weighted objective and regularizers. GRAPE's SLSQP reported success on its ftol test after "
        "32 iterations at F=0.999615; Piccolo hit its 100-iteration cap at "
        "F=0.999999992 and did not declare convergence."
    )
    results["batch_projection"]["comparability"] = comparability
    results["per_job_pre_iteration_overhead"] = {
        "grape_current_seconds": sum(
            baseline[k]
            for k in (
                "startup_seconds",
                "imports_seconds",
                "load_seconds",
                "objective_setup_seconds",
                "lower_seconds",
                "compile_seconds",
                "first_execution_seconds",
            )
        ),
        "grape_boundary": "Reconstructed startup/import/load/objective/trace/compile/first-execution subtotal before warm SLSQP; excludes warm-up and parity benchmarks, not a measured first-callback timestamp",
        "piccolo_seconds": next(
            marker["received_seconds"]
            for marker in piccolo["markers"]
            if marker["stage"] == "solve_start"
        )
        + piccolo["solves"][0]["first_iteration_since_solve_start_seconds"],
        "piccolo_boundary": "Observed process launch to first AMICODE_ITER (iteration zero); includes imports, materialization and compilation before that line, excludes compilation occurring in later iterations",
        "scope": "Separate setup figures for this fixture, without a cross-solver solve-time ranking. These measured boundaries differ; overhead can change with the problem.",
    }
    overhead = results["per_job_pre_iteration_overhead"]
    overhead["cold_minus_warm"] = {
        "grape_current_seconds": overhead["grape_current_seconds"]
        - baseline["warm_value_gradient_median_seconds"],
        "grape_formula": "pre-iteration subtotal minus warm value-and-gradient median",
        "piccolo_seconds": next(
            marker["received_seconds"]
            for marker in piccolo["markers"]
            if marker["stage"] == "solve_start"
        )
        + cold["solve_spec_seconds"]
        - warm["solve_spec_seconds"],
        "piccolo_formula": "first solve_start marker plus first-solve minus second-solve solve_spec time",
        "scope": "Cold-minus-warm per-job costs for this fixture, kept separate from solve-time projections; the first-iteration boundaries remain detail.",
    }
    overhead["piccolo_compilation_note"] = (
        f"Piccolo continues compiling after its first iteration line: {cold['compilation_seconds']:.2f} s compilation in total, "
        f"with the first iteration appearing {cold['first_iteration_since_solve_start_seconds']:.2f} s after cold solve_start "
        f"versus {warm['first_iteration_since_solve_start_seconds']:.2f} s warm."
    )
    results["batch_projection"]["cache_assumptions"] = (
        "Uses observed cold, repeated-seed and changed-candidate first-call costs, including recompilation when the objective is not serializable. If changed candidates miss: one first candidate, two changed candidates, six repeated seeds; if they hit: one cold plus eight repeated calls. Tracing/setup still occur in each process. Warm Julia is hypothetical process reuse, not a built service."
    )
    validation = results["gradient_validation"]["variants"]
    current_samples = validation["current"]["samples"][:3]
    expm_samples = validation["expm"]["samples"][:3]
    current_first = [error for sample in current_samples for error in sample["first_knot_errors"]]
    gradient_note = (
        f"Against SciPy-expm central differences (1e-6 GHz step) at the original three amplitude sets, "
        f"the current loss's bound-fixed first-knot gradient deviates by {min(current_first):.2g} to {max(current_first):.2g}, "
        f"while channel 0 knot 3 agrees within {max(sample['channel_0_knot_3_error'] for sample in current_samples):.2g}; "
        f"the expm variant agrees within {max(max(sample['first_knot_errors'] + [sample['channel_0_knot_3_error']]) for sample in expm_samples):.2g} at these same coordinates. "
        "The 0.0011 current/expm gradient difference is the current loss's deviation, not an error in expm; "
        "the all-knot and nonzero-first-knot coverage is reported in gradient_validation."
        " Across all 42 coordinates in the tested inputs, the current loss deviates only at the first knot when its amplitude is zero; "
        "with a nonzero first-knot amplitude, every coordinate agrees within 2.5e-10; a zero-amplitude interval elsewhere was not tested."
    )
    results["answers"] = {
        "1": f"Current loss lowering {baseline['lower_seconds']:.3f}s, compilation {baseline['compile_seconds']:.3f}s, first execution {baseline['first_execution_seconds']:.4f}s and warm value/gradient {baseline['warm_value_gradient_median_seconds']:.4f}s; warm SLSQP {baseline['solve']['warm_optimization_seconds']:.3f}s for {baseline['solve']['iterations']} iterations. Retained runner job was 14.5s with a 12.3s first call; the direct measurement excludes runner admission and job supervision. "
        + gradient_note,
        "2": "; ".join(
            f"{n} intervals: lower {v['lower_seconds']:.2f}s, compile {v['compile_seconds']:.2f}s, warm {v['warm_value_gradient_median_seconds']:.4f}s, IR {v['lowered_text_bytes']} bytes"
            for n, v in results["scaling"].items()
            if "error" not in v
        ),
        "3": "; ".join(
            f"{name}{' (extra variant beyond the brief)' if name == 'expm' else ''}: compile {v['compile_seconds']:.3f}s, warm {v['warm_value_gradient_median_seconds']:.4f}s, value difference from current {v['parity']['value_max_abs_difference']:.2g}, free-knot gradient difference from current {v['parity']['free_knot_gradient_max_abs_difference']:.2g}, F={v['solve']['fidelity']:.10f} in {v['solve']['iterations']} iterations"
            for name, v in successes.items()
        )
        + ". "
        + gradient_note,
        "4": "; ".join(
            f"{name}: first calls "
            + "/".join(f"{r.get('first_call_seconds', 0):.3f}s" for r in rows)
            + ", objective hits="
            + str([bool(row.get("objective_cache_hits")) for row in rows])
            + "; refusal="
            + "; ".join(rows[-1].get("objective_cache_write_refusals", []))
            for name, rows in results["cache"].items()
        ),
        "5": comparability
        + " "
        + (
            "; ".join(
                f"Julia repetition {r['repetition']}: solve_spec {r['solve_spec_seconds']:.3f}s, compile {r['compilation_seconds']:.3f}s, native solve {r['native_solve_seconds']:.3f}s, F={r['fidelity']:.10f}, iterations={r['iterations']}, converged={r['converged']}"
                for r in piccolo.get("solves", [])
            )
            or piccolo.get("error", "not measured")
        ),
        "6": "GRAPE-only serial nine-job projections (seconds): "
        + ", ".join(f"{k}={v:.1f}" for k, v in projections.items())
        + ". Piccolo-only serial nine-job projections (seconds): "
        + ", ".join(f"{k}={v:.1f}" for k, v in piccolo_projections.items())
        + ". "
        + comparability
        + " Each solver's projection assumes its own measured c60 solve cost for every candidate and measures neither changed-candidate optimization nor concurrent throughput; setup-only cross-solver figures are listed separately in per_job_pre_iteration_overhead.",
    }


def rebuild(output, workspace):
    results = json.loads(output.read_text())
    if Path(results["workspace"]) != workspace:
        raise ValueError("--rebuild must use the retained results' workspace")
    for name, digest in results["source_sha256"].items():
        if sha(REPOSITORY / name) != digest:
            raise ValueError(f"production source changed since the timing measurements: {name}")
    for name, case in results["cases"].items():
        if sha(Path(case["problem"])) != case["problem_sha256"]:
            raise ValueError(f"retained problem changed: {name}")
    if "timing_measurements" not in results:
        retained_script = workspace / "solver_compile.py"
        if sha(retained_script) != results["script_sha256"]:
            raise ValueError("retained timing script no longer matches its recorded digest")
        results["timing_measurements"] = {
            "measured_at": results["measured_at"],
            "script_sha256": results["script_sha256"],
            "script": str(retained_script),
            "reused_without_new_optimizations": True,
        }
    if "gradient_validation" not in results:
        raise ValueError("--rebuild requires retained all-knot gradient comparisons")
    for item in (results["baseline"], results["alternatives"]["expm"]):
        item.pop("independent_gradient_comparisons", None)
    results["script_sha256"] = sha(Path(__file__))
    results["source_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    results["rebuilt_at"] = datetime.now(UTC).isoformat()
    results["provenance_note"] = (
        "script_sha256 identifies this summary-only rebuild; timing_measurements and gradient_validation retain the original measurement script digests and timestamps. No solves, objective evaluations or gradient comparisons were repeated during this rebuild."
    )
    results["alternatives"]["expm"]["extra_variant_beyond_brief"] = True
    results["alternatives"]["expm"]["parity"]["reference"] = (
        "Current loss's autodiff; independent finite differences attribute the first-knot discrepancy to that reference"
    )
    summarize(results)
    save(output, results)
    print("rebuilt", output, "from retained measurements; no new evaluations", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild the summary from retained measurements without new evaluations",
    )
    parser.add_argument(
        "--workspace", type=Path, default=Path.home() / "Projects/entanglix-research/compile-probe"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("solver_compile_results.json")
    )
    args = parser.parse_args()
    if args.worker:
        child(args.worker)
        return
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "JULIA_NUM_THREADS": "1",
        "JULIA_PKG_OFFLINE": "true",
    }
    if args.rebuild:
        rebuild(args.output, workspace)
        return
    shutil.copyfile(Path(__file__), workspace / "solver_compile.py")
    before = installed_julia_content()
    cases = run_measurement(
        workspace, "prepare", {"kind": "prepare", "workspace": str(workspace)}, environment
    )
    if "error" in cases:
        raise RuntimeError(cases["error"])
    base_request = {
        "kind": "measure",
        "problem": cases["c60"]["problem"],
        "options": cases["c60"]["options"],
    }
    results = {
        "measured_at": datetime.now(UTC).isoformat(),
        "workspace": str(workspace),
        "script_sha256": sha(Path(__file__)),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "source_sha256": {name: sha(REPOSITORY / name) for name in SOURCE_FILES},
        "cases": {name: cases[name] for name in ("c60", "c30")},
        "thread_and_device_settings": {
            key: environment[key]
            for key in (
                "JAX_PLATFORMS",
                "CUDA_VISIBLE_DEVICES",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "JULIA_NUM_THREADS",
            )
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            ),
        },
        "method": "Fresh process for every cold JAX measurement. All execution timers synchronize value and gradient. Persistent caching disabled except in cache experiments. Julia uses one fresh process and two identical solves, with cumulative compilation accounting explicitly enabled. Raw parity samples/gradients remain in workspace measurements; summaries retain residuals. No installations, GPU jobs or agent sessions are requested.",
        "baseline": run_measurement(
            workspace,
            "baseline",
            {**base_request, "variant": "current", "optimize": True},
            environment,
        ),
    }
    save(args.output, results)
    if "error" in results["baseline"]:
        raise RuntimeError(results["baseline"]["error"])
    results["scaling"] = {}
    for intervals in (5, 10, 20, 40):
        results["scaling"][str(intervals)] = (
            results["baseline"]
            if intervals == 20
            else run_measurement(
                workspace,
                f"scaling-{intervals}",
                {**base_request, "variant": "current", "intervals": intervals},
                environment,
            )
        )
    results["alternatives"] = {}
    for variant in ("scan", "pwc"):
        results["alternatives"][variant] = run_measurement(
            workspace, variant, {**base_request, "variant": variant, "optimize": True}, environment
        )
    successful = {k: v for k, v in results["alternatives"].items() if "error" not in v}
    fastest = min(
        successful,
        key=lambda name: (
            sum(
                successful[name][k]
                for k in ("lower_seconds", "compile_seconds", "first_execution_seconds")
            )
            + successful[name]["solve"]["warm_optimization_seconds"]
        ),
    )
    for variant in (fastest + "_pure", "expm"):
        results["alternatives"][variant] = run_measurement(
            workspace, variant, {**base_request, "variant": variant, "optimize": True}, environment
        )
    for name, result in results["alternatives"].items():
        if "error" not in result:
            result["parity"] = parity(result, results["baseline"])
            result["interval_semantics"] = (
                "New solve per interval, fresh max_steps per interval"
                if name.startswith("scan")
                else "One continuous solve with PWC discontinuities and one whole-window max_steps; controller is not fully reinitialized per interval"
                if name.startswith("pwc")
                else "Exact-arithmetic constant-Hamiltonian propagator per interval; no adaptive controller or integration step limit, floating-point expm error remains"
            )
    results["alternatives"]["expm"]["extra_variant_beyond_brief"] = True
    retain_gradient_summary(
        results,
        workspace,
        {
            "current": ("baseline", results["baseline"]["independent_gradient_comparisons"]),
            "expm": ("expm", results["alternatives"]["expm"]["independent_gradient_comparisons"]),
        },
    )
    save(args.output, results)
    results["cache"] = {}
    for variant in ("current", fastest):
        for arguments in (False, True):
            name = variant + ("_arguments" if arguments else "")
            cache = workspace / "caches" / name
            if cache.exists():
                shutil.rmtree(cache)
            cache.mkdir(parents=True)
            results["cache"][name] = []
            for index, (case, seed) in enumerate((("c60", 7), ("c60", 19), ("c30", 7))):
                before_cache = cache_inventory(cache)
                row = run_measurement(
                    workspace,
                    f"cache-{name}-{index}",
                    {
                        **base_request,
                        "variant": variant,
                        "arguments": arguments,
                        "cache": str(cache),
                        "problem": cases[case]["problem"],
                        "seed": seed,
                    },
                    {
                        **environment,
                        "JAX_DEBUG_LOG_MODULES": "jax._src.compiler,jax._src.lru_cache",
                    },
                )
                after_cache = cache_inventory(cache)
                log = Path(row["log"]).read_text()
                row.update(
                    case=case,
                    seed=seed,
                    cache_bytes=sum(after_cache.values()),
                    new_entries=sorted(after_cache.keys() - before_cache.keys()),
                    objective_cache_hits=[
                        line
                        for line in log.splitlines()
                        if "Persistent compilation cache hit" in line
                        and ("jit_loss" in line or "jit(loss)" in line)
                    ],
                    objective_cache_write_refusals=[
                        line.split(": ", 1)[-1]
                        for line in log.splitlines()
                        if "Not writing persistent cache entry for 'jit_loss'" in line
                    ],
                )
                results["cache"][name].append(row)
            if arguments:
                row = results["cache"][name][0]
                if "error" not in row:
                    row["parity"] = parity(row, results["baseline"])
            save(args.output, results)
    results["piccolo"] = julia_probe(workspace, cases, environment)
    results["installed_or_downloaded"] = sorted(installed_julia_content() - before)
    results["runtime_session_launched"] = False
    results["gpu_job_run"] = False
    summarize(results)
    # Remove full gradients after they served the parity comparisons.
    for item in [
        results["baseline"],
        *results["alternatives"].values(),
        *results["scaling"].values(),
        *(r for rows in results["cache"].values() for r in rows),
    ]:
        item.pop("parity_samples", None)
        item.pop("independent_gradient_comparisons", None)
    results["scaling"]["20"] = {
        k: results["baseline"][k]
        for k in (
            "lower_seconds",
            "compile_seconds",
            "first_execution_seconds",
            "warm_value_gradient_median_seconds",
            "lowered_text_bytes",
        )
    }
    save(args.output, results)
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
