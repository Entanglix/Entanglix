"""dynamiqs/JAX evolution; optional imports stay inside the adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from entanglix.engines.frames import matrix, prepare_dynamics
from entanglix.evolution import (
    EngineCapabilities,
    EvolutionInput,
    EvolutionInputError,
    EvolutionResult,
    normalize_input,
)
from entanglix.models import ConstructedSystem


@dataclass(frozen=True)
class JaxEvolutionResult:
    """Device arrays that preserve differentiation through drive amplitudes."""

    times_ns: np.ndarray
    states: Any
    expectations: dict[str, Any]
    dimensions: tuple[int, ...]
    device: str
    diagnostics: dict[str, object]


class DynamiqsIntegrator:
    def __init__(self, device_kind: Literal["cpu", "gpu"] = "gpu", device_id: int = 0) -> None:
        self.device_kind = device_kind
        self.device_id = device_id

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            "dynamiqs", "cuda" if self.device_kind == "gpu" else "cpu", True, True
        )

    def evolve(self, system: ConstructedSystem, request: EvolutionInput) -> EvolutionResult:
        result = self.evolve_jax(system, request)
        return EvolutionResult(
            times_ns=result.times_ns,
            states=tuple(np.asarray(result.states)),
            expectations={
                name: np.real_if_close(np.asarray(v)) for name, v in result.expectations.items()
            },
            dimensions=result.dimensions,
            engine="dynamiqs",
            method="Tsit5",
            dtype="complex128",
            device=result.device,
            diagnostics=result.diagnostics,
        )

    def evolve_jax(
        self,
        system: ConstructedSystem,
        request: EvolutionInput,
        *,
        drive_amplitudes_ghz: dict[str, Any] | None = None,
    ) -> JaxEvolutionResult:
        """Evolve with JAX controls; model construction and request validation stay outside autodiff."""
        normalized = normalize_input(system, request)
        prepared = prepare_dynamics(system, normalized)
        try:
            import jax
            import jax.numpy as jnp

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
            controls = {}
            for name, amplitude in {
                **normalized.drive_amplitudes_ghz,
                **(drive_amplitudes_ghz or {}),
            }.items():
                if name not in system.drives:
                    raise EvolutionInputError(
                        f"unknown drive {name!r}; available: {sorted(system.drives)}"
                    )
                value = jnp.asarray(amplitude)
                if value.dtype.kind not in "iuf" or value.shape not in (
                    (),
                    (len(normalized.times_ns) - 1,),
                ):
                    raise EvolutionInputError(
                        f"drive {name!r} must be real and scalar or one value per output interval"
                    )
                if not isinstance(value, jax.core.Tracer) and not np.all(
                    np.isfinite(np.asarray(value))
                ):
                    raise EvolutionInputError(f"drive {name!r} amplitude must be finite")
                controls[name] = value.astype(jnp.float64)
            piecewise = any(value.ndim == 1 for value in controls.values())
            state = dq.asqarray(prepared.initial_density(normalized), dims=system.dimensions)
            collapse = [
                dq.asqarray(matrix(terms), dims=system.dimensions) for terms in prepared.collapse
            ]
            base = (
                system.hamiltonian_ghz if normalized.frame is None else matrix(prepared.hamiltonian)
            )
            base_hamiltonian = dq.asqarray(2 * np.pi * base, dims=system.dimensions)
            drives = {
                name: dq.asqarray(2 * np.pi * matrix(terms), dims=system.dimensions)
                for name, terms in prepared.drives.items()
            }

            def hamiltonian(index: int):
                total = base_hamiltonian
                for name, amplitude in controls.items():
                    value = amplitude[index] if amplitude.ndim else amplitude
                    operator = drives[name]
                    carrier = normalized.carriers.get(name) if normalized.frame is None else None
                    if carrier is None:
                        total = total + value * operator
                    else:

                        def coefficient(t, value=value, carrier=carrier):
                            return value * jnp.cos(
                                2 * jnp.pi * carrier.frequency_ghz * t + carrier.phase_rad
                            )

                        total = total + dq.modulated(coefficient, operator)
                return total

            method = dq.method.Tsit5(
                atol=normalized.atol, rtol=normalized.rtol, max_steps=normalized.max_steps
            )
            if piecewise:
                states = [state.to_jax()[None]]
                for index in range(len(normalized.times_ns) - 1):
                    solved = dq.mesolve(
                        hamiltonian(index),
                        collapse,
                        state,
                        normalized.times_ns[index : index + 2],
                        method=method,
                        progress_meter=False,
                    )
                    state = solved.states[-1]
                    states.append(state.to_jax()[None])
                raw = jnp.concatenate(states)
            else:
                solved = dq.mesolve(
                    hamiltonian(0),
                    collapse,
                    state,
                    normalized.times_ns,
                    method=method,
                    progress_meter=False,
                )
                raw = solved.states.to_jax()
            if str(raw.dtype) != "complex128":
                raise RuntimeError("dynamiqs returned states with the wrong dtype")
            if not isinstance(raw, jax.core.Tracer):
                raw.block_until_ready()
                if any(item != device for item in raw.devices()):
                    raise RuntimeError("dynamiqs returned states on the wrong device")
            raw = prepared.lab_states(raw, normalized.times_ns, xp=jnp)
            expectations = {
                name: jnp.einsum("ij,tji->t", jnp.asarray(operator), raw)
                for name, operator in normalized.observables.items()
            }
        return JaxEvolutionResult(
            times_ns=normalized.times_ns,
            states=raw,
            expectations=expectations,
            dimensions=system.dimensions,
            device=str(device),
            diagnostics={
                "segments": len(normalized.times_ns) - 1 if piecewise else 1,
                "max_steps": normalized.max_steps,
                "step_limit_scope": "per_solve",
                "device_kind": device.device_kind,
                **prepared.diagnostics,
            },
        )
