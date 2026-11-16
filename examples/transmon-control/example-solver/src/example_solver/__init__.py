"""Example solver plugin: SciPy's SLSQP behind Entanglix's solver interface.

This module is the adapter. SciPy stands in for your own solver: the adapter reads the frozen
problem file, calls the library, and returns the pulse.
"""

import tomllib
from importlib.metadata import version

import numpy as np
from scipy.linalg import expm
from scipy.optimize import Bounds, LinearConstraint, minimize

from entanglix.protocols.control import Goal
from entanglix.solvers import Pulse, Solution, validate_problem


def matrix(pairs):
    """The problem file stores each complex entry as a [real, imaginary] pair."""
    array = np.asarray(pairs, dtype=float)
    return array[..., 0] + 1j * array[..., 1]


def infidelity(amplitudes, drift, drives, target, indices, dt):
    """One minus the leakage-aware average gate fidelity of a piecewise-constant pulse."""
    # One propagator per interval; the last knot is a terminal value and drives nothing.
    steps = expm(-1j * dt * (drift + np.einsum("ck,cij->kij", amplitudes[:, :-1], drives)))
    columns = np.eye(len(drift), dtype=complex)[:, indices]
    for step in steps:
        columns = step @ columns
    block = columns[indices]
    survival = np.sum(np.abs(block) ** 2)
    overlap = abs(np.trace(target.conj().T @ block)) ** 2
    return 1 - (survival + overlap) / (len(indices) * (len(indices) + 1))


class Backend:
    def capabilities(self):
        return {
            # SLSQP differences the objective here; this adapter supplies no gradient.
            "gradients": False,
            "constraints": ["amplitude", "du", "ddu"],
            "gpu": False,
            "problem_kinds": ["unitary"],
        }

    def validate(self, problem, options):
        """Refuse unsupported input, before admission and again in the job."""
        # The fixed-step unitary problems the built-in solvers accept, with max_iter and tol.
        validate_problem(problem, options, "example-solver")
        if problem["pulse"].get("init") == "zeros":
            raise ValueError("example-solver starts from a random pulse, not pulse.init = zeros")

    def solve(self, problem_spec, options):
        problem = tomllib.loads(problem_spec.read_text())
        system, settings = problem["system"], problem["problem"]
        goal = Goal.model_validate(problem["goal"])
        indices = np.asarray(goal.subspace[0]) - 1
        drift, drives = matrix(system["H_drift"]), matrix(system["H_drives"])
        knots = settings["N"]
        times = np.linspace(0, problem["pulse"]["T"], knots)
        dt = times[1] - times[0]
        upper = np.repeat(np.asarray(system["params"]["drive_bounds"])[:, None], knots, axis=1)
        # Bounds on the first and second differences of each channel, terminal knot included.
        constraints = [
            LinearConstraint(
                np.kron(np.eye(len(upper)), np.diff(np.eye(knots), n=order, axis=0) / dt**order),
                -settings[field],
                settings[field],
            )
            for order, field in ((1, "du_bound"), (2, "ddu_bound"))
            if field in settings
        ]
        seed = problem["pulse"].get("seed", 0)
        start = np.random.default_rng(seed).uniform(-0.1, 0.1, upper.shape) * upper
        result = minimize(
            lambda flat: infidelity(
                flat.reshape(upper.shape), drift, drives, goal.operator(), indices, dt
            ),
            start.ravel(),
            method="SLSQP",
            bounds=Bounds(-upper.ravel(), upper.ravel()),
            constraints=constraints,
            options={"maxiter": options.get("max_iter", 100), "ftol": options.get("tol", 1e-6)},
        )
        return Solution(
            Pulse(times, result.x.reshape(upper.shape)),
            float(result.fun),
            {
                "converged": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
            },
            {
                "effective_seed": seed,
                "method": "SLSQP with finite-difference gradients",
                "optimizer_version": version("scipy"),
                "ignored_guidance_fields": [
                    f"problem.{field}"
                    for field in ("Q", "R", "R_u", "R_du", "R_ddu")
                    if field in settings
                ],
            },
            solver_reported_fidelity=1 - float(result.fun),
        )
