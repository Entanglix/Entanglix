"""QuTiP CPU reference evolution and the shared engine interface."""

from __future__ import annotations

import math

import numpy as np
import qutip as qt

from entanglix.evolution import (
    EngineCapabilities,
    EvolutionInput,
    EvolutionInputError,
    EvolutionResult,
    Integrator,
    expectations_from_states,
    normalize_input,
)
from entanglix.models import ConstructedSystem


class QuTiPIntegrator:
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities("qutip-cpu", "cpu", True, False)

    def evolve(self, system: ConstructedSystem, request: EvolutionInput) -> EvolutionResult:
        normalized = normalize_input(system, request)
        dimensions = list(system.dimensions)
        qdims = [dimensions, dimensions]
        state = qt.Qobj(normalized.initial_density, dims=qdims)
        # Sparse operators keep reference-size CPU solves practical without changing vern9.
        collapse = [
            qt.Qobj(math.sqrt(rate) * operator, dims=qdims, dtype="CSR")
            for operator, rate in system.dissipators.values()
            if rate > 0
        ]

        def hamiltonian(amplitudes: dict[str, float]) -> qt.Qobj:
            matrix = np.array(system.hamiltonian_ghz, dtype=np.complex128, copy=True)
            for name, amplitude in amplitudes.items():
                matrix += amplitude * system.drives[name]
            return qt.Qobj(2 * np.pi * matrix, dims=qdims, dtype="CSR")

        options = {
            "method": "vern9",
            "rtol": normalized.rtol,
            "atol": normalized.atol,
            "nsteps": normalized.max_steps,
            "store_states": True,
            "normalize_output": False,
        }
        if normalized.piecewise:
            # Restart at each boundary; the value on [t_i, t_{i+1}) is index i.
            states = [normalized.initial_density]
            for index in range(len(normalized.times_ns) - 1):
                segment = qt.mesolve(
                    hamiltonian(normalized.amplitudes_for_interval(index)),
                    state,
                    normalized.times_ns[index : index + 2],
                    c_ops=collapse,
                    options=options,
                )
                state = segment.states[-1]
                states.append(state.full())
        else:
            solved = qt.mesolve(
                hamiltonian(normalized.amplitudes_for_interval(0)),
                state,
                normalized.times_ns,
                c_ops=collapse,
                options=options,
            )
            states = [item.full() for item in solved.states]
        state_arrays = tuple(states)
        return EvolutionResult(
            times_ns=normalized.times_ns,
            states=state_arrays,
            expectations=expectations_from_states(state_arrays, normalized.observables),
            dimensions=system.dimensions,
            engine="qutip-cpu",
            method="vern9",
            dtype="complex128",
            device="cpu",
            diagnostics={
                "segments": len(normalized.times_ns) - 1 if normalized.piecewise else 1,
                "max_steps": normalized.max_steps,
                "step_limit_scope": "per_output_interval",
                "operator_storage": "CSR",
            },
        )


def evolve_cpu(
    system: ConstructedSystem,
    initial_state: np.ndarray,
    times_ns: np.ndarray,
    *,
    observables: dict[str, np.ndarray] | None = None,
    drive_amplitudes_ghz: dict[str, float | np.ndarray] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_steps: int = 100_000,
) -> EvolutionResult:
    """Evolve a small model on the CPU with constant or interval-valued drives."""
    return QuTiPIntegrator().evolve(
        system,
        EvolutionInput(
            initial_state,
            times_ns,
            observables or {},
            drive_amplitudes_ghz or {},
            rtol,
            atol,
            max_steps,
        ),
    )


__all__ = [
    "EngineCapabilities",
    "EvolutionInput",
    "EvolutionInputError",
    "EvolutionResult",
    "Integrator",
    "QuTiPIntegrator",
    "evolve_cpu",
]
