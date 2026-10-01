"""QuTiP CPU reference evolution."""

from __future__ import annotations

import numpy as np
import qutip as qt

from entanglix.engines.frames import matrix, prepare_dynamics
from entanglix.evolution import (
    EngineCapabilities,
    EvolutionInput,
    EvolutionResult,
    expectations_from_states,
    normalize_input,
)
from entanglix.models import ConstructedSystem


class QuTiPIntegrator:
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities("qutip-cpu", "cpu", True, False)

    def evolve(self, system: ConstructedSystem, request: EvolutionInput) -> EvolutionResult:
        normalized = normalize_input(system, request)
        prepared = prepare_dynamics(system, normalized)
        dimensions = list(system.dimensions)
        qdims = [dimensions, dimensions]
        state = qt.Qobj(prepared.initial_density(normalized), dims=qdims)
        # Sparse operators keep reference-size CPU solves practical without changing vern9.
        collapse = [qt.Qobj(matrix(terms), dims=qdims, dtype="CSR") for terms in prepared.collapse]
        base = system.hamiltonian_ghz if normalized.frame is None else matrix(prepared.hamiltonian)
        base_hamiltonian = qt.Qobj(2 * np.pi * base, dims=qdims, dtype="CSR")
        drives = {
            name: qt.Qobj(2 * np.pi * matrix(terms), dims=qdims, dtype="CSR")
            for name, terms in prepared.drives.items()
        }

        def hamiltonian(amplitudes: dict[str, float]):
            timed = [base_hamiltonian]
            for name, amplitude in amplitudes.items():
                operator = drives[name]
                carrier = normalized.carriers.get(name) if normalized.frame is None else None
                if carrier is None:
                    timed[0] += amplitude * operator
                else:

                    def coefficient(t, amplitude=amplitude, carrier=carrier):
                        return amplitude * np.cos(
                            2 * np.pi * carrier.frequency_ghz * t + carrier.phase_rad
                        )

                    timed.append([operator, coefficient])
            return qt.QobjEvo(timed)

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
            states = [prepared.initial_density(normalized)]
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
        state_arrays = tuple(prepared.lab_states(states, normalized.times_ns))
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
                **prepared.diagnostics,
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
    carriers=None,
    frame=None,
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
            carriers or {},
            frame,
        ),
    )
