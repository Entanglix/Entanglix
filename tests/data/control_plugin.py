"""Test-only entry-point backend returning an analytic two-level X pulse."""

import os
import time
import tomllib
from pathlib import Path

import numpy as np

from entanglix.solvers import Pulse, Solution, validate_problem


class Backend:
    def capabilities(self):
        return {"problem_kinds": ["unitary"], "constraints": ["amplitude", "du", "ddu"]}

    def validate(self, problem, options):
        validate_problem(problem, {}, "fixture")
        if options.keys() - {"delay_seconds"}:
            raise ValueError("unknown fixture option")

    def solve(self, problem_spec, options):
        problem = tomllib.loads(problem_spec.read_text())
        assert problem["pulse"]["T"] == 10 and problem["problem"]["N"] == 5
        (Path(os.environ["ENTANGLIX_RUN_DIR"]) / "artifacts/plugin-started").touch()
        time.sleep(options.get("delay_seconds", 0))
        # The two middle intervals each rotate by pi/4 around X.
        pulse = Pulse(np.linspace(0, 10, 5), np.array([[0, 0.1, 0.1, 0, 0], [0, 0, 0, 0, 0]]))
        return Solution(pulse, 0.0, {"converged": True}, {}, solver_reported_fidelity=1.0)
