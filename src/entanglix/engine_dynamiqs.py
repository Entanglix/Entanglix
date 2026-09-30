"""dynamiqs/JAX evolution; optional imports stay inside the adapter."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from entanglix.evolution import (
    EngineCapabilities,
    EvolutionInput,
    EvolutionResult,
    expectations_from_states,
    normalize_input,
)
from entanglix.models import ConstructedSystem


class DynamiqsIntegrator:
    def __init__(self, device_kind: Literal["cpu", "gpu"] = "gpu", device_id: int = 0) -> None:
        self.device_kind = device_kind
        self.device_id = device_id

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            "dynamiqs", "cuda" if self.device_kind == "gpu" else "cpu", True, False
        )

    def evolve(self, system: ConstructedSystem, request: EvolutionInput) -> EvolutionResult:
        normalized = normalize_input(system, request)
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            import dynamiqs as dq
        except ImportError as exc:
            raise RuntimeError("install the differentiable extra to use dynamiqs") from exc

        dq.set_precision("double")
        try:
            device = jax.devices(self.device_kind)[self.device_id]
        except (IndexError, RuntimeError) as exc:
            raise RuntimeError(
                f"no {self.device_kind} device {self.device_id} is available"
            ) from exc

        with jax.default_device(device):
            state = dq.asqarray(normalized.initial_density, dims=system.dimensions)
            collapse = [
                dq.asqarray(math.sqrt(rate) * operator, dims=system.dimensions)
                for operator, rate in system.dissipators.values()
                if rate > 0
            ]

            def hamiltonian(amplitudes: dict[str, float]):
                matrix = np.array(system.hamiltonian_ghz, dtype=np.complex128, copy=True)
                for name, amplitude in amplitudes.items():
                    matrix += amplitude * system.drives[name]
                return dq.asqarray(2 * np.pi * matrix, dims=system.dimensions)

            method = dq.method.Tsit5(
                atol=normalized.atol, rtol=normalized.rtol, max_steps=normalized.max_steps
            )
            if normalized.piecewise:
                states = [normalized.initial_density]
                for index in range(len(normalized.times_ns) - 1):
                    solved = dq.mesolve(
                        hamiltonian(normalized.amplitudes_for_interval(index)),
                        collapse,
                        state,
                        normalized.times_ns[index : index + 2],
                        method=method,
                        progress_meter=False,
                    )
                    state = solved.states[-1]
                    states.append(np.asarray(state.to_jax()))
            else:
                solved = dq.mesolve(
                    hamiltonian(normalized.amplitudes_for_interval(0)),
                    collapse,
                    state,
                    normalized.times_ns,
                    method=method,
                    progress_meter=False,
                )
                states = [np.asarray(item) for item in solved.states.to_jax()]
            raw = solved.states.to_jax().block_until_ready()
            if not all(item.platform == self.device_kind for item in raw.devices()):
                raise RuntimeError("dynamiqs returned states on the wrong device")
            if str(raw.dtype) != "complex128":
                raise RuntimeError("dynamiqs returned states with the wrong dtype")
        state_arrays = tuple(states)
        return EvolutionResult(
            times_ns=normalized.times_ns,
            states=state_arrays,
            expectations=expectations_from_states(state_arrays, normalized.observables),
            dimensions=system.dimensions,
            engine="dynamiqs",
            method="Tsit5",
            dtype="complex128",
            device=str(device),
            diagnostics={
                "segments": len(normalized.times_ns) - 1 if normalized.piecewise else 1,
                "max_steps": normalized.max_steps,
                "step_limit_scope": "per_solve",
                "device_kind": device.device_kind,
            },
        )
