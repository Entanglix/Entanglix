"""cuDensityMat evolution through qutip-cuquantum; imported only when requested."""

from __future__ import annotations

import numpy as np

from entanglix.engines.frames import prepare_dynamics
from entanglix.evolution import (
    EngineCapabilities,
    EvolutionInput,
    EvolutionInputError,
    EvolutionResult,
    expectations_from_states,
    normalize_input,
)
from entanglix.models import ConstructedSystem


def local_storage(matrix: np.ndarray) -> str:
    """Choose Dense for small/dense local operators, Dia for narrow larger ones."""
    dimension = matrix.shape[0]
    if dimension <= 16:
        return "Dense"
    occupied = sum(np.any(np.diag(matrix, offset)) for offset in range(1 - dimension, dimension))
    return "Dia" if occupied <= 16 else "Dense"


class CuQuantumIntegrator:
    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities("cudensitymat", "cuda", True, False)

    def evolve(self, system: ConstructedSystem, request: EvolutionInput) -> EvolutionResult:
        normalized = normalize_input(system, request)
        prepared = prepare_dynamics(system, normalized)
        if not system.hamiltonian_terms:
            raise EvolutionInputError("cuDensityMat requires factorized Hamiltonian terms")
        try:
            import cupy as cp
            import qutip as qt
            import qutip_cuquantum as qcu
            from cuquantum.densitymat import WorkStream
        except ImportError as exc:
            raise RuntimeError("install the cuquantum extra to use cuDensityMat") from exc

        dimensions = list(system.dimensions)
        qdims = [dimensions, dimensions]
        device = cp.cuda.Device(self.device_id)
        storage_used = []

        with device, qcu.CuQuantumBackend(WorkStream(device_id=self.device_id)):
            properties = cp.cuda.runtime.getDeviceProperties(self.device_id)

            def tensor_operator(factors: tuple[np.ndarray, ...]):
                locals_ = []
                for factor in factors:
                    selected = local_storage(factor)
                    local = qt.Qobj(factor, dtype=selected).to("CuOperator")
                    input_data = type(local.data.terms[0].prod_terms[0].operator).__name__
                    converted = local.data.to_OperatorTerm()
                    actual = type(converted.terms[0][0]).__name__
                    storage_used.append(
                        {
                            "selected": selected,
                            "qutip_data": input_data,
                            "cudensitymat_data": actual,
                            "dimension": factor.shape[0],
                        }
                    )
                    locals_.append(local)
                return qt.tensor(*locals_)

            def terms_operator(terms):
                values = [term.coefficient_ghz * tensor_operator(term.factors) for term in terms]
                return sum(values[1:], values[0])

            base_hamiltonian = terms_operator(prepared.hamiltonian)
            drives = {name: terms_operator(terms) for name, terms in prepared.drives.items()}
            collapse = [terms_operator(terms) for terms in prepared.collapse]

            def hamiltonian(amplitudes: dict[str, float]):
                timed = [2 * np.pi * base_hamiltonian]
                for name, amplitude in amplitudes.items():
                    carrier = normalized.carriers.get(name) if normalized.frame is None else None
                    if carrier is None:
                        timed[0] += 2 * np.pi * amplitude * drives[name]
                    else:

                        def coefficient(t, amplitude=amplitude, carrier=carrier):
                            return amplitude * np.cos(
                                2 * np.pi * carrier.frequency_ghz * t + carrier.phase_rad
                            )

                        timed.append([2 * np.pi * drives[name], coefficient])
                return qt.QobjEvo(timed)

            initial = qt.Qobj(
                qcu.CuState(
                    cp.asarray(prepared.initial_density(normalized)),
                    hilbert_dims=tuple(system.dimensions),
                ),
                dims=qdims,
            )
            options = {
                "method": "CuVern7",
                "rtol": normalized.rtol,
                "atol": normalized.atol,
                "nsteps": normalized.max_steps,
                "store_states": True,
                "normalize_output": False,
                "progress_bar": "",
            }

            def to_host(item):
                if not isinstance(item.data, qcu.CuState):
                    raise RuntimeError("cuDensityMat returned a non-GPU state")
                data = item.data.to_cupy()
                if data.device.id != self.device_id or str(data.dtype) != "complex128":
                    raise RuntimeError("cuDensityMat state is on the wrong device or dtype")
                return cp.asnumpy(data).reshape(normalized.initial_density.shape, order="F")

            if normalized.piecewise:
                state = initial
                states = [prepared.initial_density(normalized)]
                for index in range(len(normalized.times_ns) - 1):
                    solved = qt.mesolve(
                        hamiltonian(normalized.amplitudes_for_interval(index)),
                        state,
                        normalized.times_ns[index : index + 2],
                        c_ops=collapse,
                        options=options,
                    )
                    state = solved.states[-1]
                    states.append(to_host(state))
            else:
                solved = qt.mesolve(
                    hamiltonian(normalized.amplitudes_for_interval(0)),
                    initial,
                    normalized.times_ns,
                    c_ops=collapse,
                    options=options,
                )
                states = [to_host(item) for item in solved.states]
            cp.cuda.get_current_stream().synchronize()
            state_arrays = tuple(prepared.lab_states(states, normalized.times_ns))
            return EvolutionResult(
                times_ns=normalized.times_ns,
                states=state_arrays,
                expectations=expectations_from_states(state_arrays, normalized.observables),
                dimensions=system.dimensions,
                engine="cudensitymat",
                method="CuVern7",
                dtype="complex128",
                device=f"cuda:{self.device_id}",
                diagnostics={
                    "segments": len(normalized.times_ns) - 1 if normalized.piecewise else 1,
                    "max_steps": normalized.max_steps,
                    "step_limit_scope": "per_output_interval",
                    "local_storage": storage_used,
                    "state_storage": "CuState",
                    "gpu_name": properties["name"].decode(),
                    "compute_capability": f"{properties['major']}.{properties['minor']}",
                    **prepared.diagnostics,
                },
            )
