"""Solver discovery and the solver-independent pulse returned by a backend."""

from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Pulse:
    """N knots, channel-major amplitudes; the last knot is a terminal value."""

    times_ns: np.ndarray
    amplitudes_ghz: np.ndarray
    kind: str = "zero_order"


@dataclass(frozen=True)
class Solution:
    """A solver-native objective and a separately reported fidelity, if available."""

    pulse: Pulse
    objective: float
    termination: dict
    trace: dict
    solver_reported_fidelity: float | None = None


class SolverBackend(Protocol):
    def capabilities(self) -> dict: ...

    def solve(self, problem_spec: Path, options: dict) -> Solution: ...


def select_solver(name: str, fallback: str | None = None):
    """Use a fallback only when the requested entry point is absent."""
    entries = {}
    for entry in entry_points(group="entanglix.solvers"):
        if entry.name in entries:
            raise ValueError(f"duplicate solver entry point: {entry.name}")
        entries[entry.name] = entry
    actual = name if name in entries else fallback
    if actual not in entries:
        raise ValueError(
            f"solver {name!r} is unavailable; installed solvers: {sorted(entries)}"
            + (f"; fallback {fallback!r} is also unavailable" if fallback else "")
        )
    entry = entries[actual]
    backend = entry.load()()
    return backend, {
        "requested": name,
        "actual": actual,
        "version": entry.dist.version,
        "fallback_used": actual != name,
    }


def validate_problem(problem_spec, options, backend):
    """The shared fixed-step unitary subset and convergence-option syntax."""
    allowed = {
        "schema_version": None,
        "kind": None,
        "system": {"kind", "params", "H_drift", "H_drives"},
        "goal": {"kind", "gate", "matrix", "subspace"},
        "pulse": {"kind", "T", "init", "seed"},
        "problem": {
            "template",
            "N",
            "Q",
            "R",
            "R_u",
            "R_du",
            "R_ddu",
            "du_bound",
            "ddu_bound",
            "final_fidelity",
            "free_dt",
            "free_phase",
            "options",
        },
    }
    for name, value in problem_spec.items():
        if name not in allowed:
            raise ValueError(f"{backend} does not support {name}")
        if allowed[name] is not None:
            for key in value:
                if key not in allowed[name]:
                    raise ValueError(f"{backend} does not support {name}.{key}")
    if problem_spec["schema_version"] != 1 or problem_spec["kind"] != "control":
        raise ValueError(f"{backend} requires schema_version = 1 and kind = control")
    for name, expected in (
        ("system.kind", "raw"),
        ("goal.kind", "unitary"),
        ("pulse.kind", "zero_order"),
        ("problem.template", "SmoothPulseProblem"),
    ):
        section, key = name.split(".")
        if problem_spec[section][key] != expected:
            raise ValueError(f"{backend} does not support {name} = {problem_spec[section][key]!r}")
    problem = problem_spec["problem"]
    if problem.get("free_phase", False):
        raise ValueError(f"{backend} does not support problem.free_phase")
    if problem.get("options", {}) != {"timesteps_all_equal": True}:
        raise ValueError(f"{backend} requires problem.options.timesteps_all_equal = true only")
    dt = problem_spec["pulse"]["T"] / (problem["N"] - 1)
    bracket = problem["free_dt"]
    if (
        not isinstance(bracket, list)
        or len(bracket) != 2
        or not bracket[0] <= dt <= bracket[1]
        or not np.allclose(bracket, dt, rtol=1e-7, atol=1e-15)
    ):
        raise ValueError(f"{backend} does not support variable problem.free_dt")
    if set(problem_spec["system"]["params"]) != {"drive_bounds"}:
        raise ValueError(f"{backend} supports only system.params.drive_bounds")
    unknown = options.keys() - {"max_iter", "tol"}
    if unknown:
        raise ValueError(f"{backend} does not support solver.options.{sorted(unknown)[0]}")
    if "max_iter" in options and (type(options["max_iter"]) is not int or options["max_iter"] < 1):
        raise ValueError("solver.options.max_iter must be a positive integer")
    if "tol" in options and (
        type(options["tol"]) not in (int, float)
        or not np.isfinite(options["tol"])
        or options["tol"] <= 0
    ):
        raise ValueError("solver.options.tol must be finite and positive")
