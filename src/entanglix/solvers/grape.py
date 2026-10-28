"""JAX matrix-exponential gradients, with SciPy's constrained optimizer."""

import secrets
import time
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from entanglix.protocols.control import Goal
from entanglix.solvers import Pulse, Solution, validate_problem


def decode(value):
    array = np.asarray(value, dtype=float)
    return array[..., 0] + 1j * array[..., 1]


def bounds(problem_spec):
    """Each knot's amplitude bound; the first and last knots are fixed to zero."""
    drive_bounds = np.asarray(problem_spec["system"]["params"]["drive_bounds"], dtype=float)
    upper = np.repeat(drive_bounds[:, None], problem_spec["problem"]["N"], axis=1)
    upper[:, [0, -1]] = 0
    return upper


def validate(problem_spec, options, initial_pulse=None):
    """Refuse unsupported semantics, and a start SLSQP would clip, before admission."""
    try:
        version("jax")
    except PackageNotFoundError as exc:
        raise ValueError(
            "grape requires the differentiable extra in the runner environment"
        ) from exc
    validate_problem(problem_spec, options, "grape")
    if initial_pulse is not None:
        upper = bounds(problem_spec)
        amplitudes = np.asarray(initial_pulse.amplitudes_ghz, dtype=float)
        if amplitudes.shape != upper.shape:
            raise ValueError("grape needs one starting amplitude per channel and knot")
        excess = float(np.max(np.abs(amplitudes) - upper))
        if excess > 0:
            raise ValueError(
                f"grape refuses a starting pulse outside its bounds, which include zero first "
                f"and last knots, rather than clip it: exceeded by {excess:.3g} GHz"
            )


def objective(problem_spec):
    """Propagate subspace columns on CPU for the leakage-aware average fidelity."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import expm
    from jax.sharding import SingleDeviceSharding

    jax.config.update("jax_enable_x64", True)
    device = jax.devices("cpu")[0]
    placement = SingleDeviceSharding(device)
    goal = Goal.model_validate(problem_spec["goal"])
    indices = np.asarray(goal.subspace[0]) - 1
    dimension = len(indices)
    drift = jax.device_put(decode(problem_spec["system"]["H_drift"]), device)
    drives = jax.device_put(
        np.asarray([decode(operator) for operator in problem_spec["system"]["H_drives"]]),
        device,
    )
    initial = jax.device_put(np.eye(len(drift), dtype=complex)[:, indices], device)
    target = jax.device_put(goal.operator(), device)
    times = np.linspace(0, problem_spec["pulse"]["T"], problem_spec["problem"]["N"])
    dt = problem_spec["pulse"]["T"] / (len(times) - 1)

    def loss(amplitudes):
        def step(columns, values):
            hamiltonian = drift + jnp.einsum("c,cij->ij", values.astype(drives.dtype), drives)
            return expm(-1j * dt * hamiltonian) @ columns, None

        columns, _ = jax.lax.scan(step, initial, amplitudes[:, :-1].T)
        block = columns[indices]
        survival = jnp.sum(jnp.abs(block) ** 2)
        overlap = jnp.abs(jnp.trace(target.conj().T @ block)) ** 2
        return 1 - (survival + overlap) / (dimension * (dimension + 1))

    return jax.jit(
        jax.value_and_grad(loss), in_shardings=placement, out_shardings=(placement, placement)
    ), times


class GrapeBackend:
    def validate(self, problem_spec, options, initial_pulse=None):
        validate(problem_spec, options, initial_pulse)

    def capabilities(self):
        return {
            "gradients": True,
            "constraints": ["amplitude", "du", "ddu"],
            "gpu": False,
            "problem_kinds": ["unitary"],
            "initial_pulse": True,
        }

    def solve(self, problem_spec: Path, options: dict, initial_pulse=None):
        problem_spec = tomllib.loads(problem_spec.read_text())
        validate(problem_spec, options, initial_pulse)
        value_and_grad, times = objective(problem_spec)
        problem, pulse = problem_spec["problem"], problem_spec["pulse"]
        n = problem["N"]
        upper = bounds(problem_spec)
        shape = upper.shape
        dt = times[1] - times[0]
        constraints = []
        for order, field in ((1, "du_bound"), (2, "ddu_bound")):
            if field in problem and n > order:
                derivative = np.diff(np.eye(n), n=order, axis=0) / dt**order
                constraints.append(
                    LinearConstraint(
                        np.kron(np.eye(shape[0]), derivative), -problem[field], problem[field]
                    )
                )
        initial = np.zeros(shape)
        effective_seed = None
        if initial_pulse is not None:
            # The supplied amplitudes, knot for knot, on this solver's own grid.
            initial = np.array(initial_pulse.amplitudes_ghz, dtype=float)
        elif pulse.get("init", "default") in {"default", "random"}:
            effective_seed = pulse.get("seed")
            if effective_seed is None:
                effective_seed = secrets.randbits(63)
            initial = np.random.default_rng(effective_seed).uniform(-0.1, 0.1, shape) * upper
            for order, field in ((1, "du_bound"), (2, "ddu_bound")):
                if field in problem and n > order:
                    maximum = np.max(np.abs(np.diff(initial, n=order, axis=1) / dt**order))
                    if maximum:
                        initial *= min(1, problem[field] / maximum)
        calls, iterations = [], []

        def evaluate(flat):
            started = time.monotonic()
            value, gradient = value_and_grad(np.asarray(flat, dtype=float).reshape(shape))
            value, gradient = float(value), np.asarray(gradient).ravel()
            calls.append({"objective": value, "seconds": time.monotonic() - started})
            return value, gradient

        last_iteration = time.monotonic()

        def callback(flat):
            nonlocal last_iteration
            now = time.monotonic()
            iterations.append(
                {"seconds": now - last_iteration, "objective": calls[-1]["objective"]}
            )
            last_iteration = now

        settings = {
            key: options[source]
            for source, key in (("max_iter", "maxiter"), ("tol", "ftol"))
            if source in options
        }
        result = minimize(
            evaluate,
            initial.ravel(),
            method="SLSQP",
            jac=True,
            bounds=Bounds(-upper.ravel(), upper.ravel()),
            constraints=constraints,
            callback=callback,
            options=settings,
        )
        ignored = [
            f"problem.{field}" for field in ("Q", "R", "R_u", "R_du", "R_ddu") if field in problem
        ]
        if pulse.get("init") == "zeros" and "seed" in pulse:
            ignored.append("pulse.seed")
        return Solution(
            Pulse(times, result.x.reshape(shape)),
            float(result.fun),
            {
                "converged": bool(result.success),
                **({"status": int(result.status)} if "status" in result else {}),
                "message": str(result.message),
                "iterations": int(result.get("nit", 0)),
                "objective_evaluations": int(result.get("nfev", len(calls))),
            },
            {
                **({"effective_seed": effective_seed} if effective_seed is not None else {}),
                "objective_calls": calls,
                "iterations": iterations,
                "ignored_guidance_fields": ignored,
                "declared_options": options,
                "upstream_defaults": {
                    key: value
                    for key, value in {"max_iter": 100, "tol": 1e-6}.items()
                    if key not in options
                },
                "method": "SLSQP",
                "optimizer_version": version("scipy"),
                "gradient_backend_version": version("jax"),
                "propagator": "jax.scipy.linalg.expm per constant interval",
            },
            solver_reported_fidelity=1 - float(result.fun),
        )
