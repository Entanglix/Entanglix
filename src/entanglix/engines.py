"""Reference-sized deterministic evolution through QuTiP on the CPU."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import qutip as qt

from entanglix.models import ConstructedSystem


class EvolutionInputError(ValueError):
    """Invalid initial state, observable, time grid, or solver tolerance."""


@dataclass(frozen=True)
class EvolutionResult:
    times_ns: np.ndarray
    states: tuple[np.ndarray, ...]
    expectations: dict[str, np.ndarray]


def evolve_cpu(
    system: ConstructedSystem,
    initial_state: np.ndarray,
    times_ns: np.ndarray,
    *,
    observables: dict[str, np.ndarray] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> EvolutionResult:
    """Evolve a small model with a static Hamiltonian and optional collapse terms."""
    times = np.asarray(times_ns, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.all(np.isfinite(times))
        or not np.all(np.diff(times) > 0)
        or not np.allclose(np.diff(times), times[1] - times[0], rtol=1e-10, atol=1e-12)
    ):
        raise EvolutionInputError("times_ns must be a finite, increasing uniform grid")
    if not (math.isfinite(rtol) and math.isfinite(atol) and rtol > 0 and atol > 0):
        raise EvolutionInputError("solver tolerances must be positive and finite")
    dimension = int(np.prod(system.dimensions))
    initial = np.asarray(initial_state, dtype=np.complex128)
    if not np.all(np.isfinite(initial)):
        raise EvolutionInputError("initial state must contain finite values")
    if initial.shape == (dimension,):
        if not np.isclose(np.vdot(initial, initial), 1, rtol=0, atol=1e-10):
            raise EvolutionInputError("initial ket must have unit norm")
        initial = np.outer(initial, initial.conj())
    elif initial.shape == (dimension, dimension):
        if not np.allclose(initial, initial.conj().T, atol=1e-10):
            raise EvolutionInputError("initial density matrix must be Hermitian")
        if not np.isclose(np.trace(initial), 1, rtol=0, atol=1e-10):
            raise EvolutionInputError("initial density matrix must have unit trace")
        if np.linalg.eigvalsh(initial).min() < -1e-10:
            raise EvolutionInputError("initial density matrix must be positive semidefinite")
    else:
        raise EvolutionInputError(
            f"initial state must be a ket ({dimension},) or density matrix ({dimension}, {dimension})"
        )
    qdims = [list(system.dimensions), list(system.dimensions)]
    q_observables = {}
    for name, array in (observables or {}).items():
        data = np.asarray(array, dtype=np.complex128)
        if data.shape != (dimension, dimension) or not np.all(np.isfinite(data)):
            raise EvolutionInputError(
                f"observable {name!r} must be a finite ({dimension}, {dimension}) matrix"
            )
        q_observables[name] = qt.Qobj(data, dims=qdims)
    collapse = [
        qt.Qobj(math.sqrt(rate) * operator, dims=qdims)
        for operator, rate in system.dissipators.values()
        if rate > 0
    ]
    result = qt.mesolve(
        2 * np.pi * system.qutip_hamiltonian(),
        qt.Qobj(initial, dims=qdims),
        times,
        c_ops=collapse,
        e_ops=q_observables,
        options={
            "method": "adams",
            "rtol": rtol,
            "atol": atol,
            "store_states": True,
            "normalize_output": False,
        },
    )
    return EvolutionResult(
        times_ns=times,
        states=tuple(state.full() for state in result.states),
        expectations={name: np.asarray(value) for name, value in zip(q_observables, result.expect)},
    )
