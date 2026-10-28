"""Test-only entry-point backend that declares starting-pulse support and returns its start."""

import tomllib

import numpy as np

from entanglix.solvers import Pulse, Solution, validate_problem

# The preflight refuses a start whose first channel takes this amplitude at the second knot.
REFUSED_GHZ = 0.0625


class Backend:
    def capabilities(self):
        return {
            "problem_kinds": ["unitary"],
            "constraints": ["amplitude", "du", "ddu"],
            "initial_pulse": True,
        }

    def validate(self, problem, options, initial_pulse=None):
        validate_problem(problem, {}, "starting")
        if initial_pulse is not None and initial_pulse.amplitudes_ghz[0, 1] == REFUSED_GHZ:
            raise ValueError("the starting fixture refuses this pulse")

    def solve(self, problem_spec, options, initial_pulse=None):
        problem = tomllib.loads(problem_spec.read_text())
        times = np.linspace(0, problem["pulse"]["T"], problem["problem"]["N"])
        channels = len(problem["system"]["H_drives"])
        # The start is returned on this solver's grid, so the result shows what the job delivered.
        amplitudes = (
            initial_pulse.amplitudes_ghz
            if initial_pulse is not None
            else np.zeros((channels, len(times)))
        )
        return Solution(Pulse(times, amplitudes), 0.0, {"converged": True}, {})
