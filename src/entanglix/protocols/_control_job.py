"""A solver candidate, using the common launch gate and inherited start barrier."""

import os
from pathlib import Path

import numpy as np

from entanglix.runner import _digest, _load_record, publish_toml
from entanglix.solvers import select_solver


def execute(problem: Path):
    run = Path(os.environ["ENTANGLIX_RUN_DIR"])
    launch = _load_record(run / "run.toml")["launch"]
    if _digest(problem.read_bytes()) != launch["problem_sha256"]:
        raise ValueError("frozen ProblemSpec digest changed")
    solver = _load_record(run / "spec/control.toml")["solver"]
    backend, selection = select_solver(solver["backend"], solver.get("fallback"))
    if selection != launch["selection"]:
        raise ValueError("solver installation changed after launch validation")
    if preflight := getattr(backend, "validate", None):
        runtime = preflight(_load_record(problem), solver.get("options", {}))
        expected = _load_record(run / "run.toml")["environment"].get("solver_runtime")
        if runtime != expected:
            raise ValueError("solver runtime changed after launch validation")
    solution = backend.solve(problem, solver.get("options", {}))
    output = run / "artifacts/pulse.npz"
    np.savez(
        output,
        kind=solution.pulse.kind,
        times_ns=solution.pulse.times_ns,
        amplitudes_ghz=solution.pulse.amplitudes_ghz,
    )
    trace = run / "artifacts/solver-trace.toml"
    publish_toml(trace, solution.trace)
    publish_toml(
        run / "result.toml",
        {
            "kind": "control",
            "engine": selection["actual"],
            "device": "cpu",
            "solver": selection,
            "problem_sha256": launch["problem_sha256"],
            "objective": solution.objective,
            "metrics": (
                {"solver_reported_fidelity": solution.solver_reported_fidelity}
                if solution.solver_reported_fidelity is not None
                else {}
            ),
            "termination": solution.termination,
            "ignored_guidance_fields": solution.trace.get("ignored_guidance_fields", []),
            "artifacts": {
                name: {"path": str(path.relative_to(run)), "sha256": _digest(path.read_bytes())}
                for name, path in (
                    ("pulse", output),
                    ("solver_trace", trace),
                    *(
                        (name, run / path)
                        for name, path in solution.trace.get("native_artifacts", {}).items()
                    ),
                )
            },
        },
    )


if __name__ == "__main__":
    from entanglix.protocols._simulate_job import main

    raise SystemExit(main("control"))
