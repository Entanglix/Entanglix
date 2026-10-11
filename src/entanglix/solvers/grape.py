"""JAX gradients through the existing engine, with SciPy's constrained optimizer."""

import secrets
import time
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from entanglix.models import ConstructedSystem
from entanglix.protocols.control import Goal
from entanglix.solvers import Pulse, Solution, validate_problem


def decode(value):
    array = np.asarray(value, dtype=float)
    return array[..., 0] + 1j * array[..., 1]


def validate(problem_spec, options):
    """Refuse unsupported semantics before an attempt is admitted."""
    for package in ("jax", "dynamiqs"):
        try:
            version(package)
        except PackageNotFoundError as exc:
            raise ValueError(
                "grape requires the differentiable extra in the runner environment"
            ) from exc
    validate_problem(problem_spec, options, "grape")


def objective(problem_spec):
    """One passive ancilla evaluates the leakage-aware average gate fidelity."""
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp

    from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator
    from entanglix.evolution import EvolutionInput

    dq.set_precision("double")
    goal = Goal.model_validate(problem_spec["goal"])
    indices = np.asarray(goal.subspace[0]) - 1
    dimension = len(indices)
    drift = decode(problem_spec["system"]["H_drift"])
    identity = np.eye(dimension)
    model = ConstructedSystem(
        id="control-objective",
        hamiltonian_ghz=np.kron(drift, identity) / (2 * np.pi),
        operators={},
        dissipators={},
        dimensions=(len(drift) * dimension,),
        basis="declared-control-frame",
        provenance={},
        drives={
            str(index): np.kron(decode(operator), identity) / (2 * np.pi)
            for index, operator in enumerate(problem_spec["system"]["H_drives"])
        },
    )
    initial = np.zeros((len(drift), dimension), dtype=complex)
    initial[indices, np.arange(dimension)] = 1 / np.sqrt(dimension)
    target = np.zeros_like(initial)
    target[indices] = goal.operator() / np.sqrt(dimension)
    target = jnp.asarray(target.ravel())
    surviving = np.repeat(np.isin(np.arange(len(drift)), indices), dimension)
    times = np.linspace(0, problem_spec["pulse"]["T"], problem_spec["problem"]["N"])
    request = EvolutionInput(initial.ravel(), times)
    engine = DynamiqsIntegrator(device_kind="cpu")

    def loss(amplitudes):
        rho = engine.evolve_jax(
            model,
            request,
            drive_amplitudes_ghz={name: amplitudes[int(name), :-1] for name in model.drives},
        ).states[-1]
        survival = jnp.real(jnp.sum(jnp.diag(rho) * surviving))
        overlap = jnp.real(target.conj() @ rho @ target)
        return 1 - (survival + dimension * overlap) / (dimension + 1)

    return jax.jit(jax.value_and_grad(loss)), times


class GrapeBackend:
    def validate(self, problem_spec, options):
        validate(problem_spec, options)

    def capabilities(self):
        return {
            "gradients": True,
            "constraints": ["amplitude", "du", "ddu"],
            "gpu": False,
            "problem_kinds": ["unitary"],
        }

    def solve(self, problem_spec: Path, options: dict):
        problem_spec = tomllib.loads(problem_spec.read_text())
        validate(problem_spec, options)
        value_and_grad, times = objective(problem_spec)
        problem, pulse = problem_spec["problem"], problem_spec["pulse"]
        n = problem["N"]
        bounds = np.asarray(problem_spec["system"]["params"]["drive_bounds"], dtype=float)
        shape = (len(bounds), n)
        upper = np.repeat(bounds[:, None], n, axis=1)
        upper[:, [0, -1]] = 0
        dt = times[1] - times[0]
        constraints = []
        for order, field in ((1, "du_bound"), (2, "ddu_bound")):
            if field in problem and n > order:
                derivative = np.diff(np.eye(n), n=order, axis=0) / dt**order
                constraints.append(
                    LinearConstraint(
                        np.kron(np.eye(len(bounds)), derivative), -problem[field], problem[field]
                    )
                )
        initial = np.zeros(shape)
        effective_seed = None
        if pulse.get("init", "default") in {"default", "random"}:
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
                "engine": "dynamiqs",
                "engine_version": version("dynamiqs"),
            },
            solver_reported_fidelity=1 - float(result.fun),
        )
