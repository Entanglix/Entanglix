"""Run the pinned Piccolo project in the runner job's existing process group."""

import hashlib
import shutil
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import tomli_w

from entanglix.solvers import Pulse, Solution, validate_problem

PROJECT = Path(__file__).with_name("julia")


class PiccoloBackend:
    def capabilities(self):
        return {
            "gradients": True,
            "constraints": ["amplitude", "du", "ddu"],
            "gpu": False,
            "problem_kinds": ["unitary"],
        }

    def validate(self, problem_spec, options):
        validate_problem(problem_spec, options, "piccolo")
        executable = shutil.which("julia")
        if executable is None:
            raise ValueError("piccolo requires Julia on PATH and its pinned project instantiated")
        version = subprocess.check_output(
            [executable, "--startup-file=no", "--version"], text=True
        ).strip()
        return {
            "julia_version": version,
            "project_sha256": hashlib.sha256((PROJECT / "Project.toml").read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256((PROJECT / "Manifest.toml").read_bytes()).hexdigest(),
        }

    def solve(self, problem_spec: Path, options: dict):
        run = problem_spec.parent.parent
        native = run / "artifacts/piccolo"
        native.mkdir()
        settings = native / "options.toml"
        transport_options = dict(options)
        if "tol" in transport_options:
            transport_options["tol"] = float(transport_options["tol"])
        settings.write_text(tomli_w.dumps(transport_options))
        iterations = []
        child = subprocess.Popen(
            [
                "julia",
                "--startup-file=no",
                f"--project={PROJECT}",
                str(PROJECT / "solve.jl"),
                str(problem_spec),
                str(native),
                str(settings),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in child.stdout:
            print(line, end="", flush=True)
            if line.startswith("AMICODE_ITER "):
                fields = dict(part.split("=", 1) for part in line.split()[1:])
                iterations.append({key: float(value) for key, value in fields.items()})
        if child.wait() != 0:
            raise ValueError(f"Piccolo exited with code {child.returncode}; see run.log")
        result = tomllib.loads((native / "result.toml").read_text())
        neutral = tomllib.loads((native / "neutral-pulse.toml").read_text())
        trace = {
            "declared_options": options,
            "effective_options": neutral["effective_options"],
            "initialization": neutral["initialization"],
            "package_versions": result["params"]["versions"],
            "iterations": iterations,
            "ignored_guidance_fields": neutral["ignored_guidance_fields"],
            "native_artifacts": {
                "native_result": "artifacts/piccolo/result.toml",
                "native_pulse": str(next(native.glob("pulse-*.jld2")).relative_to(run)),
            },
        }
        return Solution(
            Pulse(np.asarray(neutral["times_ns"]), np.asarray(neutral["amplitudes_ghz"])),
            iterations[-1]["obj"],
            {
                "converged": result["params"]["converged"],
                "iterations": result["iterations"],
                "convergence_rule": "Piccolo's heuristic: iterations < iteration cap; not an Ipopt status.",
            },
            trace,
            solver_reported_fidelity=result["fidelity"],
        )
