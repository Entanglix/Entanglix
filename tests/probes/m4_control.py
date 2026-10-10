"""CPU-only control-boundary probe; outputs stay in the external research workspace.

Run with the differentiable extra in an isolated Python environment. The script
builds a pinned amicode clone and instantiates its public Julia project before
attempting any solves. It stops if that environment cannot load on this Julia.
"""

import argparse
import copy
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AMICODE_COMMIT = "cb717f57cb63fa292917188c8ca32f50ddcca539"
CHILDREN = set()

JULIA_HELPER = r"""
using Piccolo, JLD2, TOML, LinearAlgebra
import Pkg
mode = ARGS[1]
if mode == "metadata"
    packages = Dict(d.name => string(d.version) for d in values(Pkg.dependencies()) if d.version !== nothing)
    s = Piccolo.Specs.parse_spec(read(ARGS[2], String))
    defaults = Dict("backend"=>string(s.solver.backend), "tol"=>s.solver.tol === nothing ? "unspecified" : s.solver.tol,
                    "max_iter"=>s.solver.max_iter)
    ipopt = IpoptOptions()
    TOML.print(stdout, Dict("packages"=>Dict(k=>packages[k] for k in ["Piccolo","Ipopt","JLD2","DirectTrajOpt"]),
                           "parsed_solver"=>defaults, "ipopt_default_tol"=>ipopt.tol,
                           "piccolo_source"=>dirname(pathof(Piccolo))))
elseif mode == "pulse"
    pulse = load_pulse(ARGS[2])
    times = collect(pulse.controls.t)
    controls = hcat([pulse((times[i]+times[i+1])/2) for i in 1:length(times)-1]...)
    TOML.print(stdout, Dict("times_ns"=>times, "controls_ghz"=>[collect(row) for row in eachrow(controls)],
                           "at_knots_ghz"=>[collect(row) for row in eachrow(hcat([pulse(t) for t in times]...))],
                           "pulse_type"=>string(typeof(pulse)), "initial_value"=>pulse(0.0),
                           "final_value"=>pulse(times[end])))
elseif mode == "direct"
    mkpath(ARGS[3])
    Piccolo.Specs.solve_spec(ARGS[2]; run_dir=ARGS[3])
end
"""


def probe_env():
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JULIA_PKG_PRECOMPILE_AUTO": "0",
        "JULIA_NUM_PRECOMPILE_TASKS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


def descendants(pid):
    found = set()
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                parent = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
                if parent == pid:
                    found.add(int(entry.name))
            except (FileNotFoundError, ProcessLookupError):
                pass
    return found


def terminate(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspected(text, *sources):
    return {
        "basis": "source_inspection",
        "text": text,
        "sources": [
            {"path": str(path), "line": line, "sha256": digest(path)} for path, line in sources
        ],
    }


def depot_inventory():
    depot = Path.home() / ".julia"
    return {
        area: sorted(str(p.relative_to(depot)) for p in (depot / area).glob("*/*"))
        for area in ("packages", "artifacts", "compiled")
    }


def command(args, work, *, cwd=None, timeout=600):
    log = work / "logs" / f"{len(list((work / 'logs').glob('*'))):03d}.log"
    clock = time.monotonic()
    with log.open("w") as stream:
        child = subprocess.Popen(
            args,
            cwd=cwd,
            env=probe_env(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            children = descendants(child.pid)
            terminate(child.pid)
            for pid in children:
                terminate(pid)
            code = "timeout"
    result = {
        "command": list(map(str, args)),
        "exit_code": code,
        "wall_seconds": time.monotonic() - clock,
        "log": str(log),
    }
    print(f"{args[0]}: {code}, {result['wall_seconds']:.1f}s, {log}", flush=True)
    return result


def setup(work, result):
    clone = work / "amicode"
    stages = result["setup_commands"] = []
    if not clone.exists():
        stages.append(
            command(
                [
                    "git",
                    "clone",
                    "--no-hardlinks",
                    str(Path.home() / "Projects/amicode"),
                    str(clone),
                ],
                work,
            )
        )
        if stages[-1]["exit_code"] != 0:
            return False
        stages.append(command(["git", "checkout", "--detach", AMICODE_COMMIT], work, cwd=clone))
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip()
    if actual != AMICODE_COMMIT:
        raise ValueError(f"probe clone must be at {AMICODE_COMMIT}; got {actual}")
    result["amicode_commit"] = actual
    for args in (["node", "--version"], ["pnpm", "--version"], ["julia", "--version"]):
        result.setdefault("versions", {})[args[0]] = subprocess.check_output(
            args, text=True
        ).strip()
    stages.append(
        command(
            ["pnpm", "--filter", "@amicode/amico-run...", "install", "--frozen-lockfile"],
            work,
            cwd=clone,
        )
    )
    if stages[-1]["exit_code"] != 0:
        return False
    stages.append(
        command(["pnpm", "--filter", "@amicode/amico-run...", "run", "build"], work, cwd=clone)
    )
    if stages[-1]["exit_code"] != 0:
        return False
    project = clone / "packages/extension/julia"
    result["julia_project"] = str(project)
    result["manifest_before_sha256"] = digest(project / "Manifest.toml")
    stages.append(
        command(
            [
                "julia",
                f"--project={project}",
                "--startup-file=no",
                "-e",
                "using Pkg; Pkg.instantiate()",
            ],
            work,
        )
    )
    result["manifest_sha256"] = digest(project / "Manifest.toml")
    result["julia_manifest_warnings"] = {
        "basis": "execution_log",
        "log": stages[-1]["log"],
        "resolved_julia_version": tomllib.loads((project / "Manifest.toml").read_text())[
            "julia_version"
        ],
        "text": Path(stages[-1]["log"]).read_text().strip(),
    }
    if stages[-1]["exit_code"] != 0:
        return False
    stages.append(
        command(
            [
                "julia",
                f"--project={project}",
                "--startup-file=no",
                "-e",
                'using Piccolo; println("PICCOLO_READY")',
            ],
            work,
            timeout=1200,
        )
    )
    return stages[-1]["exit_code"] == 0


def julia(work, mode, *args):
    project = work / "amicode/packages/extension/julia"
    return command(
        [
            "julia",
            f"--project={project}",
            "--startup-file=no",
            str(work / "helper.jl"),
            mode,
            *map(str, args),
        ],
        work,
    )


def export_system(work):
    import tomli_w

    from entanglix.models import ConstructedSystem, load_system

    source = ROOT / "examples/junction-shunt/system.toml"
    document = tomllib.loads(source.read_text())
    document["truncation"]["retained_levels"] = 3
    model_path = work / "system.toml"
    model_path.write_text(tomli_w.dumps(document))
    model = load_system(model_path)
    energies = np.diag(model.hamiltonian_ghz).real
    frequency = energies[1] - energies[0]
    charge = model.operators["charge"]
    lowering = np.diag(np.diag(charge, 1), 1) / abs(charge[0, 1])
    drift = np.diag(energies - energies[0] - frequency * np.arange(3))
    drives = {"x": (lowering + lowering.conj().T) / 2, "y": 1j * (lowering - lowering.conj().T) / 2}
    system = ConstructedSystem(
        "probe-transmon",
        drift,
        {},
        {},
        (3,),
        "isolated_eigenstates",
        {"source": str(source)},
        drives=drives,
    )
    raw = {
        "kind": "raw",
        "params": {"drive_bounds": [0.06, 0.06]},
        "H_drift": np.stack([2 * np.pi * drift.real, 2 * np.pi * drift.imag], axis=-1).tolist(),
        "H_drives": [
            np.stack([2 * np.pi * h.real, 2 * np.pi * h.imag], axis=-1).tolist()
            for h in drives.values()
        ],
    }
    information = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": digest(source),
        "retained_levels": 3,
        "energies_ghz": energies.tolist(),
        "frame_frequency_ghz": float(frequency),
        "drive_order": list(drives),
        "basis": "lowest isolated circuit eigenstates, increasing energy",
        "dissipators_omitted": list(model.dissipators),
        "approximation": "resonant RWA: keep adjacent charge transitions, remove diagonal and other sectors; subtract f01*n and E0",
        "units": "T in ns; u in GHz; raw drift and drive matrices multiplied by 2*pi, so i*dU/dt=(H_drift+sum(u*H_drive))*U",
        "drive_definition": "lowering=adjacent upper diagonal of charge/abs(n01); x=(lowering+adjoint)/2; y=i*(lowering-adjoint)/2",
        "encoding": "JSON rows, then columns, then [real, imaginary]; H_drives ordered x,y",
    }
    return system, raw, information


def problem(system):
    return {
        "schema_version": 1,
        "kind": "control",
        "system": system,
        "goal": {"kind": "unitary", "gate": "X", "subspace": [[1, 2]]},
        "pulse": {"kind": "zero_order", "T": 20.0, "init": "random", "seed": 7},
        "problem": {
            "template": "SmoothPulseProblem",
            "N": 21,
            "Q": 1000.0,
            "R": 0.0001,
            "du_bound": 0.1,
            "ddu_bound": 0.1,
            "free_dt": False,
            "free_phase": False,
            "options": {"timesteps_all_equal": True},
        },
        "solver": {"backend": "ipopt", "tol": 1e-8, "max_iter": 100},
    }


def validate(work, path):
    js = (
        'import Ajv from "ajv"; import {readFileSync} from "node:fs"; '
        "const a=new Ajv({strict:false,allErrors:true}); "
        'const v=a.compile(JSON.parse(readFileSync(process.argv[1],"utf8"))); '
        'const ok=v(JSON.parse(readFileSync(process.argv[2],"utf8"))); '
        "console.log(JSON.stringify({ok,errors:v.errors})); process.exit(ok?0:64);"
    )
    schema = work / "amicode/packages/schema/schemas/problemspec.oss.schema.json"
    outcome = command(
        ["node", "--input-type=module", "-e", js, str(schema), str(path)],
        work,
        cwd=work / "amicode/packages/schema",
    )
    outcome["validation"] = json.loads(Path(outcome["log"]).read_text())
    return outcome


def launch_args(work, path, label):
    spec = work / f"{label}-solvespec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "4",
                "lab_id": "m4-probe",
                "problem_spec": str(path),
                "executor": "local",
            }
        )
    )
    return [
        str(work / "amicode/packages/amico-run/launcher/amico-run"),
        "--spec",
        str(spec),
        "--project",
        str(work / "amicode/packages/extension/julia"),
        "--runs-root",
        str(work / "runs" / label),
    ]


def result_summary(record):
    return {
        **{
            name: record[name]
            for name in ["schema_version", "fidelity", "iterations", "wall_seconds"]
        },
        "params": {"converged": record["params"]["converged"]},
    }


def launch(work, path, label):
    before = {p for p in (work / "runs" / label).glob("*/run.toml") if not p.parent.is_symlink()}
    outcome = command(launch_args(work, path, label), work, timeout=180)
    runs = [
        p
        for p in (work / "runs" / label).glob("*/run.toml")
        if p not in before and not p.parent.is_symlink()
    ]
    if runs:
        run = runs[-1].parent
        outcome["run"] = str(run)
        outcome["files"] = sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_file())
        for file in ["run.toml", "result.toml", "FINISHED"]:
            if (run / file).exists():
                record = tomllib.loads((run / file).read_text())
                if file == "result.toml":
                    record = result_summary(record)
                elif file == "run.toml":
                    record = {
                        name: value
                        for name, value in record.items()
                        if name in ["run_id", "script_path", "orchestrator", "julia", "created_at"]
                    }
                outcome[file] = record
    if outcome["exit_code"] != 0:
        error = Path(outcome["log"]).read_text().split("Stacktrace:")[0].strip()
        if error.startswith("ERROR: SpecValidationError:"):
            # Retain public diagnostics here; the exact upstream report stays in the log.
            errors = json.loads(error.split("\n", 1)[1])["errors"]
            outcome["error"] = [{k: e[k] for k in ("path", "msg", "got") if k in e} for e in errors]
        else:
            outcome["error"] = error
    return outcome


def fidelity(propagator):
    block = propagator[:2, :2]
    gate = np.array([[0, 1], [1, 0]])
    overlap = float(abs(np.trace(gate.conj().T @ block)) ** 2 / 4)
    leakage = float(1 - np.trace(block.conj().T @ block).real / 2)
    return {"F_P": overlap, "L": leakage, "F_avg": ((1 - leakage) + 2 * overlap) / 3}


def cardinal_states():
    eye = np.eye(3, dtype=complex)
    return [
        eye[0],
        eye[1],
        (eye[0] + eye[1]) / np.sqrt(2),
        (eye[0] - eye[1]) / np.sqrt(2),
        (eye[0] + 1j * eye[1]) / np.sqrt(2),
        (eye[0] - 1j * eye[1]) / np.sqrt(2),
    ]


def rollout(system, pulse):
    import qutip as qt

    from entanglix.engines import EvolutionInput, QuTiPIntegrator

    times, controls = np.asarray(pulse["times_ns"]), np.asarray(pulse["controls_ghz"])
    propagator = qt.qeye(3)
    clock = time.monotonic()
    for index in range(len(times) - 1):
        h = system.hamiltonian_ghz.astype(complex)
        for channel, name in enumerate(system.drives):
            h += controls[channel, index] * system.drives[name]
        segment = qt.sesolve(
            2 * np.pi * qt.Qobj(h),
            propagator,
            times[index : index + 2],
            options={
                "method": "vern9",
                "rtol": 1e-10,
                "atol": 1e-12,
                "normalize_output": False,
                "nsteps": 100_000,
            },
        )
        propagator = segment.states[-1]
    result = fidelity(propagator.full())
    gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    values = []
    for initial in cardinal_states():
        trace = QuTiPIntegrator().evolve(
            system,
            EvolutionInput(initial, times, drive_amplitudes_ghz=dict(zip(system.drives, controls))),
        )
        target = gate @ initial
        values.append(float(np.vdot(target, trace.states[-1] @ target).real))
    result.update(
        {
            "six_state_average": float(np.mean(values)),
            "density_propagator_difference": float(abs(np.mean(values) - result["F_avg"])),
            "wall_seconds": time.monotonic() - clock,
            "rtol": 1e-10,
            "atol": 1e-12,
        }
    )
    return result


def gradient(work):
    import jax
    import jax.numpy as jnp

    from entanglix.engines import EvolutionInput, QuTiPIntegrator
    from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

    jax.config.update("jax_enable_x64", True)
    system, _, _ = export_system(work)
    pulse = json.loads((work / "pulse.json").read_text())
    times = np.asarray(pulse["times_ns"])
    controls = np.asarray(pulse["controls_ghz"])
    # Evaluate away from a stationary optimum so the finite-difference check is informative.
    controls = controls * 0.9
    engine = DynamiqsIntegrator("cpu")
    gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    states = cardinal_states()
    requests = [EvolutionInput(initial, times) for initial in states]

    def objective(amplitudes):
        values = []
        for initial, request in zip(states, requests):
            rho = engine.evolve_jax(
                system, request, drive_amplitudes_ghz=dict(zip(system.drives, amplitudes))
            ).states[-1]
            target = jnp.asarray(gate @ initial)
            values.append(jnp.real(jnp.vdot(target, rho @ target)))
        return 1 - jnp.mean(jnp.stack(values))

    value_and_gradient = jax.value_and_grad(objective)
    amplitudes = jnp.asarray(controls)
    amplitudes.block_until_ready()
    clock = time.monotonic()
    value, derivative = value_and_gradient(amplitudes)
    value.block_until_ready()
    derivative.block_until_ready()
    seconds = time.monotonic() - clock
    clock = time.monotonic()
    warm_value, warm_derivative = value_and_gradient(amplitudes)
    warm_value.block_until_ready()
    warm_derivative.block_until_ready()
    warm_seconds = time.monotonic() - clock
    warm_difference = float(np.max(abs(np.asarray(warm_derivative) - np.asarray(derivative))))
    derivative = np.asarray(derivative)

    def cpu_objective(amplitudes):
        values = []
        for initial in states:
            trace = QuTiPIntegrator().evolve(
                system,
                EvolutionInput(
                    initial, times, drive_amplitudes_ghz=dict(zip(system.drives, amplitudes))
                ),
            )
            target = gate @ initial
            values.append(np.vdot(target, trace.states[-1] @ target).real)
        return float(1 - np.mean(values))

    probes = []
    # Three largest derivatives, including both channels when they have nonzero sensitivity.
    flat = np.argsort(abs(derivative).ravel())[::-1]
    indices = [np.unravel_index(flat[0], derivative.shape)]
    other = 1 - indices[0][0]
    indices.append((other, int(np.argmax(abs(derivative[other])))))
    indices.append(
        next(
            np.unravel_index(i, derivative.shape)
            for i in flat
            if np.unravel_index(i, derivative.shape) not in indices
        )
    )
    for index in indices:
        for epsilon in [1e-5, 1e-6]:
            displacement = np.zeros_like(controls)
            displacement[index] = epsilon
            finite = (
                cpu_objective(controls + displacement) - cpu_objective(controls - displacement)
            ) / (2 * epsilon)
            probes.append(
                {
                    "channel": list(system.drives)[index[0]],
                    "interval": int(index[1]),
                    "epsilon_ghz": epsilon,
                    "autodiff": float(derivative[index]),
                    "finite_difference": finite,
                    "absolute_difference": abs(finite - derivative[index]),
                }
            )
    data = {
        "available": True,
        "device": str(jax.devices("cpu")[0]),
        "objective": "1-F_avg, six cardinal states",
        "control_scaling_from_solved_pulse": 0.9,
        "infidelity": float(value),
        "value_and_gradient_seconds_including_compilation": seconds,
        "warm_value_and_gradient_seconds": warm_seconds,
        "warm_gradient_maximum_difference": warm_difference,
        "warm_objective_difference": float(abs(warm_value - value)),
        "timing_scope": "same six-state objective and transformed function; warm evaluation of the current implementation, not an optimisation-loop benchmark",
        "gradient_shape": list(derivative.shape),
        "gradient_unit": "infidelity/GHz",
        "finite_differences": probes,
        "maximum_difference": float(max(p["absolute_difference"] for p in probes)),
    }
    (work / "gradient.json").write_text(json.dumps(data, indent=2) + "\n")


def cancellation(work, path, label, sig, *, during_iterations=False):
    before = {p for p in (work / "runs" / label).glob("*/run.toml") if not p.parent.is_symlink()}
    args = launch_args(work, path, label)
    log = work / f"{label}.log"
    juliapid = None
    with log.open("w") as stream:
        runner = subprocess.Popen(
            args, env=probe_env(), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        CHILDREN.add(runner.pid)
        try:
            deadline = time.monotonic() + 60
            iteration_line = None
            while time.monotonic() < deadline:
                for pid in descendants(runner.pid):
                    try:
                        if Path(f"/proc/{pid}/exe").resolve().name == "julia":
                            juliapid = pid
                    except FileNotFoundError:
                        pass
                if during_iterations:
                    iteration_line = next(
                        (
                            line
                            for line in log.read_text().splitlines()
                            if line.startswith("AMICODE_ITER ")
                        ),
                        None,
                    )
                if (
                    juliapid is not None and (not during_iterations or iteration_line is not None)
                ) or runner.poll() is not None:
                    break
                time.sleep(0.01)
            if juliapid is None:
                raise RuntimeError(f"no Julia child started in {label}; see {log}")
            CHILDREN.add(juliapid)
            if during_iterations and iteration_line is None:
                raise RuntimeError(f"no solver iteration observed in {label}; see {log}")
            clock = time.monotonic()
            os.killpg(runner.pid, sig)
            code = runner.wait(timeout=15)
            time.sleep(0.2)
            alive = Path(f"/proc/{juliapid}").exists()
            result = {
                "signal": signal.Signals(sig).name,
                "signal_point": "after first observed solver iteration"
                if during_iterations
                else "Julia executable observed during startup",
                "runner_exit_code": code,
                "elapsed_after_signal_seconds": time.monotonic() - clock,
                "runner_process_group": runner.pid,
                "julia_process_group": juliapid,
                "julia_survived_before_cleanup": alive,
                "log": str(log),
            }
            if during_iterations:
                result["first_observed_iteration"] = iteration_line
            manifests = [
                p
                for p in (work / "runs" / label).glob("*/run.toml")
                if p not in before and not p.parent.is_symlink()
            ]
            if manifests:
                run = manifests[-1].parent
                result["run"] = str(run)
                result["FINISHED"] = (
                    tomllib.loads((run / "FINISHED").read_text())
                    if (run / "FINISHED").exists()
                    else None
                )
            return result
        finally:
            terminate(runner.pid)
            CHILDREN.discard(runner.pid)
            if juliapid is not None:
                terminate(juliapid)
                CHILDREN.discard(juliapid)
                if Path(f"/proc/{juliapid}").exists():
                    raise RuntimeError(f"Julia process {juliapid} remains after cleanup")


def study(work, result):
    import tomli_w

    (work / "helper.jl").write_text(JULIA_HELPER)
    system, raw, result["export"] = export_system(work)
    specs = {
        "template": problem(
            {
                "kind": "template",
                "template": "TransmonSystem",
                "params": {
                    "levels": 3,
                    "drive_bounds": [0.03, 0.03],
                    "ω": result["export"]["frame_frequency_ghz"],
                    "δ": float(-system.hamiltonian_ghz[2, 2].real),
                },
            }
        ),
        "raw": problem(raw),
    }
    result["inputs"] = {}
    result["probe_fidelity_target"] = 0.999
    result["target_note"] = (
        "diagnostic target for this probe; optimisation still targets the declared X gate, with a 100-iteration cap"
    )
    result["solves"] = []
    for kind, definition in specs.items():
        path = work / f"{kind}.json"
        path.write_text(json.dumps(definition, indent=2) + "\n")
        result["inputs"][kind] = {"path": str(path), "sha256": digest(path)}
        validated = validate(work, path)
        if not validated["validation"]["ok"]:
            raise ValueError(f"probe spec did not validate: {validated}")
        if kind == "template":
            result["json_transport"] = launch(work, path, "json-transport")
        path = work / f"{kind}.toml"
        path.write_text(tomli_w.dumps(definition))
        result["inputs"][kind].update({"launched_path": str(path), "launched_sha256": digest(path)})
        for repetition in range(2):
            item = launch(work, path, f"{kind}-{repetition}")
            result["solves"].append({"kind": kind, "repetition": repetition, **item})
            if item["exit_code"] != 0:
                raise RuntimeError(f"{kind} solve failed; see {item['log']}")
    metadata = julia(work, "metadata", work / "raw.toml")
    result["julia_metadata"] = tomllib.loads(Path(metadata["log"]).read_text())
    run = Path(result["solves"][-1]["run"])
    pulse_path = next(run.glob("*.jld2"))
    extraction = julia(work, "pulse", pulse_path)
    if extraction["exit_code"] != 0:
        raise RuntimeError(f"pulse extraction failed: {extraction['log']}")
    pulse = tomllib.loads(Path(extraction["log"]).read_text())
    (work / "pulse.json").write_text(json.dumps(pulse, indent=2) + "\n")
    result["pulse"] = {
        "path": str(work / "pulse.json"),
        "sha256": digest(work / "pulse.json"),
        "type": pulse["pulse_type"],
        "times_ns": pulse["times_ns"],
        "endpoint_values": [pulse["initial_value"], pulse["final_value"]],
        "interpolation": "sample interval midpoints from loaded ZeroOrderPulse; final knot not an interval",
        "declared_duration_ns": 20.0,
        "actual_duration_ns": pulse["times_ns"][-1],
        "duration_difference_ns": pulse["times_ns"][-1] - 20.0,
    }
    declared_pulse = {**pulse, "times_ns": np.linspace(0, 20.0, len(pulse["times_ns"])).tolist()}
    result["pulse"]["declared_grid_rollout"] = rollout(system, declared_pulse)
    bounded = copy.deepcopy(specs["raw"])
    bounded["problem"]["free_dt"] = [1.0 - 1e-7, 1.0 + 1e-7]
    bounded_json = work / "bounded-duration.json"
    bounded_json.write_text(json.dumps(bounded, indent=2) + "\n")
    bounded_toml = bounded_json.with_suffix(".toml")
    bounded_toml.write_text(tomli_w.dumps(bounded))
    bounded_check = validate(work, bounded_json)
    if not bounded_check["validation"]["ok"]:
        raise ValueError(f"bounded-duration spec did not validate: {bounded_check}")
    bounded_run = launch(work, bounded_toml, "bounded-duration")
    if bounded_run["exit_code"] != 0:
        raise RuntimeError(f"bounded-duration solve failed: {bounded_run['log']}")
    bounded_extraction = julia(work, "pulse", next(Path(bounded_run["run"]).glob("*.jld2")))
    if bounded_extraction["exit_code"] != 0:
        raise RuntimeError(f"bounded-duration pulse extraction failed: {bounded_extraction['log']}")
    bounded_pulse = tomllib.loads(Path(bounded_extraction["log"]).read_text())
    bounded_times = np.asarray(bounded_pulse["times_ns"])
    bounded_steps = np.diff(bounded_times)
    result["duration_bounds"] = {
        "input": {
            "path": str(bounded_json),
            "sha256": digest(bounded_json),
            "launched_path": str(bounded_toml),
            "launched_sha256": digest(bounded_toml),
        },
        "validation": bounded_check,
        "launch": bounded_run,
        "pulse_extraction": bounded_extraction,
        "requested_timestep_bounds_ns": bounded["problem"]["free_dt"],
        "actual_duration_ns": float(bounded_times[-1]),
        "duration_difference_ns": float(bounded_times[-1] - 20.0),
        "actual_timestep_range_ns": [float(min(bounded_steps)), float(max(bounded_steps))],
        "maximum_timestep_bound_violation_ns": float(
            max(
                0,
                min(bounded["problem"]["free_dt"]) - min(bounded_steps),
                max(bounded_steps) - max(bounded["problem"]["free_dt"]),
            )
        ),
        "rollout": rollout(system, bounded_pulse),
        "declared_grid_rollout": rollout(
            system, {**bounded_pulse, "times_ns": np.linspace(0, 20.0, len(bounded_times)).tolist()}
        ),
    }
    result["rollout"] = rollout(system, pulse)
    result["rollout"]["piccolo_F_avg"] = result["solves"][-1]["result.toml"]["fidelity"]
    result["rollout"]["reported_difference"] = abs(
        result["rollout"]["F_avg"] - result["rollout"]["piccolo_F_avg"]
    )
    result["direct"] = julia(work, "direct", work / "raw.toml", work / "direct")
    direct_result = tomllib.loads((work / "direct/result.toml").read_text())
    result["direct"]["result"] = result_summary(direct_result)
    result["schema_variants"] = []
    for name in ["omitted", "explicit", "other_backend", "option"]:
        definition = copy.deepcopy(specs["raw"])
        if name == "omitted":
            del definition["solver"]
        elif name == "other_backend":
            definition["solver"]["backend"] = "grape"
        elif name == "option":
            definition["problem"]["options"]["bound_state"] = False
        path = work / f"variant-{name}.json"
        path.write_text(json.dumps(definition, indent=2) + "\n")
        toml_path = path.with_suffix(".toml")
        toml_path.write_text(tomli_w.dumps(definition))
        result["schema_variants"].append(
            {
                "name": name,
                "input": {
                    "path": str(path),
                    "sha256": digest(path),
                    "launched_path": str(toml_path),
                    "launched_sha256": digest(toml_path),
                },
                "validation": validate(work, path),
                "launch": launch(work, toml_path, f"variant-{name}"),
            }
        )
    omitted = julia(work, "metadata", work / "variant-omitted.toml")
    result["omitted_solver_metadata"] = tomllib.loads(Path(omitted["log"]).read_text())
    result["cancellation"] = [
        cancellation(work, work / "raw.toml", f"cancel-{sig.name}", sig)
        for sig in [signal.SIGTERM, signal.SIGKILL]
    ]
    result["cancellation"].append(
        cancellation(
            work,
            work / "raw.toml",
            "cancel-SIGTERM-iterations",
            signal.SIGTERM,
            during_iterations=True,
        )
    )
    result["gradient_command"] = command(
        [sys.executable, str(Path(__file__)), "--gradient-worker", str(work)], work, timeout=600
    )
    if result["gradient_command"]["exit_code"] == 0:
        result["gradient"] = json.loads((work / "gradient.json").read_text())
    else:
        result["gradient"] = {
            "available": False,
            "error": Path(result["gradient_command"]["log"]).read_text(),
        }
    piccolo = Path(result["julia_metadata"]["piccolo_source"])
    runner_source = work / "amicode/packages/amico-run/src"
    result["settings_observations"] = {
        "file_max_iter": inspected(
            "solve_spec forwards the parsed iteration cap; the parser defaults it to 500",
            (piccolo / "specs/run.jl", 186),
            (piccolo / "specs/parse.jl", 437),
        ),
        "file_tol": inspected(
            "solver.tol is parsed and retained; solve_spec does not forward it to solve!",
            (piccolo / "specs/parse.jl", 438),
            (piccolo / "specs/run.jl", 188),
        ),
        "direct_api": inspected(
            "solve_spec accepts max_iter and solver keyword overrides; amico-run's invocation supplies neither",
            (piccolo / "specs/run.jl", 103),
            (runner_source / "local_executor.ts", 36),
            (runner_source / "launch.ts", 48),
        ),
        "default_record": inspected(
            "params.spec retains the canonical spec; effective solver tol is not separately recorded",
            (piccolo / "specs/run.jl", 222),
        ),
        "template": inspected(
            "SmoothPulseProblem has u/du/ddu regularizers and derivative constraints; only free_dt=[lo,hi] forwards timestep bounds, while the template defaults to no bounds",
            (piccolo / "specs/materialize.jl", 275),
            (piccolo / "control/templates/smooth_pulse_problem.jl", 86),
        ),
        "converged": inspected(
            "params.converged is the heuristic iterations < max_iter; Ipopt's termination status is not exposed here",
            (piccolo / "specs/run.jl", 199),
        ),
    }
    result["wrapper_additions"] = {
        "observed": {
            "basis": "measurement",
            "text": "run.toml, FINISHED, index, latest, captured run.log, independent Julia process group and signal forwarding",
            "evidence": ["solves", "cancellation"],
        },
        "telemetry": inspected(
            "local parsing of AMICODE_ITER and DONE lines into events; this parser performs no network calls",
            (runner_source / "telemetry.ts", 3),
            (runner_source / "local_executor.ts", 366),
        ),
        "direct": inspected(
            "Julia solve_spec writes result.toml and the JLD2 pulse, without the wrapper's manifest or FINISHED",
            (piccolo / "specs/run.jl", 204),
            (piccolo / "specs/run.jl", 222),
        ),
        "entry": inspected(
            "v4 SolveSpec routes its ProblemSpec to Piccolo.Specs.solve_spec",
            (runner_source / "local_executor.ts", 36),
            (runner_source / "launch.ts", 207),
        ),
    }
    result["complete"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path.home() / "Projects/entanglix-research/m4-probe"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("m4_control_results.json")
    )
    args = parser.parse_args()
    work = args.workspace.resolve()
    os.environ.update(probe_env())
    # Adopt orphaned Julia grandchildren in the signal experiment so they can be reaped.
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("cannot enable Linux child subreaper for cancellation cleanup")
    (work / "logs").mkdir(parents=True, exist_ok=True)
    before = depot_inventory()
    result = {
        "complete": False,
        "workspace": str(work),
        "cpu_only": True,
        "python_environment": sys.prefix,
        "python_packages": {
            name: version(name) for name in ["numpy", "qutip", "scqubits", "jax", "dynamiqs"]
        },
        "source_sha256": {
            str(path): digest(ROOT / path)
            for path in [
                Path("uv.lock"),
                Path("src/entanglix/models.py"),
                Path("src/entanglix/evolution.py"),
                Path("src/entanglix/engines/frames.py"),
                Path("src/entanglix/engines/qutip_cpu.py"),
                Path("src/entanglix/engines/dynamiqs_jax.py"),
            ]
        },
    }
    try:
        result["setup_ready"] = setup(work, result)
        if not result["setup_ready"]:
            result["stopped_reason"] = (
                "Build or Julia compatibility failed; remaining items were not attempted."
            )
        else:
            study(work, result)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for pid in CHILDREN:
            terminate(pid)
        # The subreaper also adopts Julia's short-lived workers; settle those before exit.
        for pid in descendants(os.getpid()):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        result["cleanup_remaining_children"] = sorted(descendants(os.getpid()))
        after = depot_inventory()
        result["depot_additions"] = {
            area: sorted(set(after[area]) - set(before[area])) for area in before
        }
        result["probe_sha256"] = digest(Path(__file__))
        stages = result.get("setup_commands", [])
        result["network_steps"] = {
            "pnpm_install": {
                "basis": "execution_log",
                "logs": [
                    c["log"]
                    for c in stages
                    if c["command"][0] == "pnpm" and "install" in c["command"]
                ],
                "text": "cache and download evidence is reported by pnpm in the installation log; no inferred download count",
            },
            "julia_instantiate": {
                "basis": "execution_log_and_inventory",
                "logs": [
                    c["log"] for c in stages if "using Pkg; Pkg.instantiate()" in c["command"]
                ],
                "new_packages": result["depot_additions"]["packages"],
                "new_artifacts": result["depot_additions"]["artifacts"],
            },
            "local_clone": {
                "basis": "executed_command",
                "commands": [c for c in stages if c["command"][0] == "git"],
                "text": "filesystem clone when absent; an existing pinned clone is reused",
            },
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--gradient-worker":
        os.environ.update(probe_env())
        gradient(Path(sys.argv[2]))
    else:
        main()
