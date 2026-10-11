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
