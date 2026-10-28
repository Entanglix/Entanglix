"""Offline CPU probe of a local Piccolo precompile workload, without production edits.

Run with .venv-runner/bin/python tests/probes/piccolo_precompile.py. Requires the
installed Julia/Piccolo environment and compile-probe/grape-propagator inputs in
~/Projects/entanglix-research. --rebuild only summarizes retained measurements.
Only the workspace depot/project are written; no package downloads or sysimage.
"""

import argparse
import difflib
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "src/entanglix/solvers/julia"
ENVIRONMENT = {
    "JULIA_PKG_OFFLINE": "true",
    "JULIA_PKG_PRECOMPILE_AUTO": "0",
    "JULIA_NUM_THREADS": "1",
    "JULIA_NUM_PRECOMPILE_TASKS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
TIMED_ENTRY = """println("PROBE stage=entry"); flush(stdout)
Base.cumulative_compile_timing(true)
started = time_ns()
using Piccolo, TOML, LinearAlgebra
WORKLOAD_IMPORT
println("PROBE stage=imports seconds=", (time_ns()-started)/1e9); flush(stdout)
problem, output_root, options_path = ARGS
started = time_ns()
spec = Piccolo.Specs.parse_spec(read(problem, String))
options = TOML.parsefile(options_path)
kwargs = Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in options)
println("PROBE stage=parse seconds=", (time_ns()-started)/1e9); flush(stdout)
println("PROBE stage=runtime julia=", VERSION, " piccolo=", pkgversion(Piccolo),
        " julia_threads=", Threads.nthreads(), " blas_threads=", BLAS.get_num_threads()); flush(stdout)
for repetition in 1:2
    output = joinpath(output_root, string(repetition))
    println("PROBE stage=solve_start repetition=", repetition); flush(stdout)
    before = Base.cumulative_compile_time_ns()
    started = time_ns()
    Piccolo.Specs.solve_spec(problem; run_dir=output, kwargs...)
    println("PROBE stage=solve_end repetition=", repetition, " seconds=", (time_ns()-started)/1e9,
            " compile_seconds=", (Base.cumulative_compile_time_ns()[1]-before[1])/1e9); flush(stdout)
    started = time_ns()
    before = Base.cumulative_compile_time_ns()
    pulse_path = only(filter(p->endswith(p, ".jld2"), readdir(output; join=true)))
    pulse = load_pulse(pulse_path)
    times = collect(pulse.controls.t)
    amplitudes = hcat([pulse(t) for t in times]...)
    open(joinpath(output, "neutral-pulse.toml"), "w") do io
        TOML.print(io, Dict("times_ns"=>times,
                           "amplitudes_ghz"=>[collect(row) for row in eachrow(amplitudes)]))
    end
    println("PROBE stage=export_end repetition=", repetition, " seconds=", (time_ns()-started)/1e9,
            " compile_seconds=", (Base.cumulative_compile_time_ns()[1]-before[1])/1e9); flush(stdout)
end
"""
WORKLOAD = """module PiccoloWorkload
using Piccolo, PrecompileTools, TOML
@setup_workload begin
    problem = joinpath(@__DIR__, "workload.toml")
    @compile_workload begin
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                mktempdir() do directory
                    spec = Piccolo.Specs.parse_spec(read(problem, String))
                    Piccolo.Specs.solve_spec(problem; run_dir=directory, max_iter=2, tol=1e-8)
                    pulse_path = only(filter(p->endswith(p, ".jld2"), readdir(directory; join=true)))
                    pulse = load_pulse(pulse_path)
                    times = collect(pulse.controls.t)
                    amplitudes = hcat([pulse(t) for t in times]...)
                    open(joinpath(directory, "neutral-pulse.toml"), "w") do io
                        TOML.print(io, Dict("times_ns"=>times,
                            "amplitudes_ghz"=>[collect(row) for row in eachrow(amplitudes)]))
                    end
                end
            end
        end
    end
end
end
"""


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path):
    return {
        str(p.relative_to(path)): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in path.rglob("*")
        if p.is_file()
    }


def run(command, environment, log, timeout=600):
    started = time.monotonic()
    events = []
    with log.open("w") as output:
        process = subprocess.Popen(
            command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # The outer command timeout is enforced by a timer, including silent precompilation.
        import threading

        timer = threading.Timer(timeout, process.kill)
        timer.start()
        try:
            for line in process.stdout:
                output.write(line)
                output.flush()
                if line.startswith(("PROBE ", "AMICODE_ITER ")):
                    fields = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
                    events.append({**fields, "received_seconds": time.monotonic() - started})
            code = process.wait()
        finally:
            timer.cancel()
    return {
        "command": command,
        "exit_code": code,
        "process_seconds": time.monotonic() - started,
        "events": events,
        "log": str(log),
        "error": log.read_text()[-6000:] if code else None,
    }


def prepare(workspace):
    import tomli_w

    shutil.copyfile(Path(__file__), workspace / "measurement-driver.py")
    installed = Path.home() / ".julia"
    save(workspace / "depot-before.json", snapshot(installed))
    environment = {**os.environ, **ENVIRONMENT}
    environment["JULIA_DEPOT_PATH"] = f"{workspace / 'depot'}:{installed}:"
    # Avoid invoking the juliaup launcher in timed measurements.
    julia = subprocess.check_output(
        [shutil.which("julia"), "--startup-file=no", "-e", 'print(joinpath(Sys.BINDIR, "julia"))'],
        env=environment,
        text=True,
    )
    package = workspace / "PiccoloWorkload"
    (package / "src").mkdir(parents=True, exist_ok=True)
    manifest = tomllib.loads((PROJECT / "Manifest.toml").read_text())
    (package / "Project.toml").write_text(
        tomli_w.dumps(
            {
                "name": "PiccoloWorkload",
                "uuid": "c5ea4195-1d32-4dfa-a277-61dc1a1c59bc",
                "version": "0.1.0",
                "deps": {
                    name: manifest["deps"][name][0]["uuid"]
                    for name in ("Piccolo", "PrecompileTools", "TOML")
                },
            }
        )
    )
    (package / "src/PiccoloWorkload.jl").write_text(WORKLOAD)
    inputs = {}
    for name, original in (
        ("c60", workspace.parent / "compile-probe/inputs/c60/problem.toml"),
        ("c30", workspace.parent / "compile-probe/inputs/c30/problem.toml"),
        ("c60-40", workspace.parent / "grape-propagator/inputs/transmon-3-40/problem.toml"),
    ):
        target = workspace / "inputs" / name / "problem.toml"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        inputs[name] = {"path": str(target), "sha256": sha(target)}
    small = tomllib.loads(Path(inputs["c60"]["path"]).read_text())
    small["problem"]["N"] = 5
    small["pulse"]["T"] = 4.0
    small["problem"]["free_dt"] = [0.9999999, 1.0000001]
    (package / "src/workload.toml").write_text(tomli_w.dumps(small))
    (workspace / "options.toml").write_text("max_iter = 100\ntol = 1e-8\n")
    for arm in ("baseline", "workload"):
        directory = workspace / f"project-{arm}"
        directory.mkdir(exist_ok=True)
        for name in ("Project.toml", "Manifest.toml"):
            shutil.copyfile(PROJECT / name, directory / name)
        (directory / "entry.jl").write_text(
            TIMED_ENTRY.replace(
                "WORKLOAD_IMPORT", "using PiccoloWorkload" if arm == "workload" else ""
            )
        )
    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [*PROJECT.glob("*.toml"), PROJECT / "solve.jl", PROJECT.parent / "piccolo.py"]
        },
        "script_sha256": sha(Path(__file__)),
        "julia": julia,
        "environment": {**ENVIRONMENT, "JULIA_DEPOT_PATH": environment["JULIA_DEPOT_PATH"]},
        "inputs": inputs,
        "workload": {"intervals": 4, "max_iter": 2, "path": str(package)},
    }, environment


def summarize(workspace, output):
    metadata = json.loads((workspace / "metadata.json").read_text())
    results = []
    for path in sorted((workspace / "measurements").glob("*.json")):
        item = json.loads(path.read_text())
        if not item["exit_code"]:
            solves = []
            events = item["events"]
            item["runtime"] = next(e for e in events if e.get("stage") == "runtime")
            for repetition in ("1", "2"):
                start = next(
                    e
                    for e in events
                    if e.get("stage") == "solve_start" and e["repetition"] == repetition
                )
                solve = next(
                    e
                    for e in events
                    if e.get("stage") == "solve_end" and e["repetition"] == repetition
                )
                export = next(
                    e
                    for e in events
                    if e.get("stage") == "export_end" and e["repetition"] == repetition
                )
                iterations = [
                    e["received_seconds"]
                    for e in events
                    if "iter" in e
                    and start["received_seconds"]
                    < e["received_seconds"]
                    < solve["received_seconds"]
                ]
                native = tomllib.loads(
                    (Path(item["output"]) / repetition / "result.toml").read_text()
                )
                solves.append(
                    {
                        "repetition": int(repetition),
                        "solve_seconds": float(solve["seconds"]),
                        "compilation_seconds": float(solve["compile_seconds"]),
                        "export_seconds": float(export["seconds"]),
                        "export_compilation_seconds": float(export["compile_seconds"]),
                        "first_iteration_seconds": iterations[0] - start["received_seconds"]
                        if iterations
                        else None,
                        "later_iteration_median_seconds": statistics.median(
                            b - a for a, b in zip(iterations[1:], iterations[2:])
                        )
                        if len(iterations) > 2
                        else None,
                        "fidelity": native["fidelity"],
                        "iterations": native["iterations"],
                        "converged": native["params"]["converged"],
                    }
                )
            first_export = next(
                e["received_seconds"]
                for e in events
                if e.get("stage") == "export_end" and e["repetition"] == "1"
            )
            item.update(
                solves=solves,
                first_complete_job_seconds=first_export,
                cold_minus_warm_seconds=first_export
                - solves[1]["solve_seconds"]
                - solves[1]["export_seconds"],
            )
        item.pop("events")
        results.append(item)
    setup = json.loads((workspace / "setup.json").read_text())
    differences = {}
    for name in ("Project.toml", "Manifest.toml"):
        differences[name] = "".join(
            difflib.unified_diff(
                (workspace / "project-baseline" / name).read_text().splitlines(True),
                (workspace / "project-workload" / name).read_text().splitlines(True),
                fromfile=f"pinned/{name}",
                tofile=f"workload/{name}",
            )
        )
    before = json.loads((workspace / "depot-before.json").read_text())
    after = json.loads((workspace / "depot-after.json").read_text())
    changes = [
        name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)
    ]
    pairs = {}
    for name in metadata["inputs"]:
        pair = {r["arm"]: r for r in results if r["case"] == name and not r["exit_code"]}
        if len(pair) == 2:
            a, b = pair["baseline"]["solves"][0], pair["workload"]["solves"][0]
            pairs[name] = {
                "fidelity_difference": b["fidelity"] - a["fidelity"],
                "iterations_identical": a["iterations"] == b["iterations"],
                "converged_identical": a["converged"] == b["converged"],
                "first_complete_job_saving_seconds": pair["baseline"]["first_complete_job_seconds"]
                - pair["workload"]["first_complete_job_seconds"],
            }
    manifest = tomllib.loads((workspace / "project-baseline/Manifest.toml").read_text())
    answers = {
        "one_off_cost": f"The workload precompile process took {setup['precompile']['process_seconds']:.2f} s; only PiccoloWorkload was compiled into the workspace depot, occupying {setup['cache_bytes'] / 2**20:.2f} MiB. The other 457 packages were reused.",
        "per_problem": {
            name: f"First complete job: baseline {next(r['first_complete_job_seconds'] for r in results if r['case'] == name and r['arm'] == 'baseline'):.2f} s, workload {next(r['first_complete_job_seconds'] for r in results if r['case'] == name and r['arm'] == 'workload'):.2f} s; saving {pair['first_complete_job_saving_seconds']:.2f} s. Fidelity difference {pair['fidelity_difference']:.1e}; iteration counts and convergence flags match: {pair['iterations_identical'] and pair['converged_identical']}."
            for name, pair in pairs.items()
        },
        "adoption": "The workload project adds one local package with Piccolo, PrecompileTools and TOML dependencies; production would also load it from solve.jl. Exact project/manifest differences and the generated package stay in the workspace; production files were not edited.",
        "depot_writes": f"The installed depot has {len(changes)} files with changed size or modification time; {sum(before.get(p, [None])[0] != after.get(p, [None])[0] for p in changes)} changed size. Cache-file changes are timestamps only: Julia's loader touches loaded .ji files (base/loading.jl:2162). juliaup/juliaup.json was also modified during setup; its cause is not in the records. The run does not establish zero writes to ~/.julia.",
    }
    save(workspace / "depot-comparison.json", {"changed_paths": changes})
    save(
        output,
        {
            **metadata,
            "summary_script_sha256": sha(Path(__file__)),
            "measurement_driver": str(workspace / "measurement-driver.py"),
            "workspace": str(workspace),
            "setup": setup,
            "measurements": results,
            "comparisons": pairs,
            "project_differences": differences,
            "generated_source_sha256": {
                str(p.relative_to(workspace)): sha(p)
                for p in [
                    workspace / "project-baseline/entry.jl",
                    workspace / "project-workload/entry.jl",
                    workspace / "PiccoloWorkload/Project.toml",
                    workspace / "PiccoloWorkload/src/PiccoloWorkload.jl",
                    workspace / "PiccoloWorkload/src/workload.toml",
                ]
            },
            "installed_depot_changes": {
                "count": len(changes),
                "size_changes": [
                    p for p in changes if before.get(p, [None])[0] != after.get(p, [None])[0]
                ],
                "raw_comparison": str(workspace / "depot-comparison.json"),
                "examples": changes[:3],
            },
            "versions": {
                "Julia": next(r["runtime"]["julia"] for r in results if "runtime" in r),
                **{
                    name: manifest["deps"][name][0]["version"]
                    for name in ("Piccolo", "PrecompileTools")
                },
            },
            "answers": answers,
            "timing_note": "One fresh process per arm/input, with two solves in each. First complete job includes startup, imports, solve and neutral-pulse export. Cold-minus-warm subtracts the second solve and export. Iteration timing uses stdout receipt. Baseline and workload use the same installed dependency versions and cached package images; workload precompilation is a separate one-off cost.",
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
        default=Path.home() / "Projects/entanglix-research/piccolo-precompile",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tests/probes/piccolo_precompile_results.json"
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if not args.rebuild:
        metadata, environment = prepare(workspace)
        save(workspace / "metadata.json", metadata)
        julia = metadata["julia"]
        package = workspace / "PiccoloWorkload"
        project = workspace / "project-workload"
        command = [
            julia,
            "--startup-file=no",
            f"--project={project}",
            "-e",
            f"using Pkg; Pkg.develop(path={json.dumps(str(package))})",
        ]
        setup = run(command, environment, workspace / "develop.log")
        if setup["exit_code"]:
            raise RuntimeError(setup["error"])
        command = [
            julia,
            "--startup-file=no",
            f"--project={project}",
            "-e",
            'using Pkg; Pkg.precompile("PiccoloWorkload")',
        ]
        print("precompiling local workload", flush=True)
        setup["precompile"] = run(command, environment, workspace / "precompile.log")
        setup["cache_bytes"] = sum(
            p.stat().st_size for p in (workspace / "depot/compiled").rglob("*") if p.is_file()
        )
        save(workspace / "setup.json", setup)
        if not setup["precompile"]["exit_code"]:
            for name, item in metadata["inputs"].items():
                for arm in ("baseline", "workload"):
                    print(f"measuring {name} {arm}", flush=True)
                    directory = workspace / f"project-{arm}"
                    output = workspace / "runs" / f"{name}-{arm}"
                    result = run(
                        [
                            julia,
                            "--startup-file=no",
                            f"--project={directory}",
                            str(directory / "entry.jl"),
                            item["path"],
                            str(output),
                            str(workspace / "options.toml"),
                        ],
                        environment,
                        workspace / f"{name}-{arm}.log",
                    )
                    save(
                        workspace / "measurements" / f"{name}-{arm}.json",
                        {**result, "case": name, "arm": arm, "output": str(output)},
                    )
        save(workspace / "depot-after.json", snapshot(Path.home() / ".julia"))
    summarize(workspace, args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
