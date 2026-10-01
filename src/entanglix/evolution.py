"""Shared inputs and results for reference-size evolution engines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal, Protocol

import numpy as np

from entanglix.models import ConstructedSystem


class EvolutionInputError(ValueError):
    """An evolution input is invalid for the constructed system."""


@dataclass(frozen=True)
class Carrier:
    frequency_ghz: float
    phase_rad: float = 0.0


@dataclass(frozen=True)
class RotatingFrame:
    frequencies_ghz: dict[str, float]
    rwa: bool = False
    excitation_numbers: dict[str, list[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvolutionInput:
    initial_state: np.ndarray
    times_ns: np.ndarray
    observables: dict[str, np.ndarray] = field(default_factory=dict)
    drive_amplitudes_ghz: dict[str, float | np.ndarray] = field(default_factory=dict)
    rtol: float = 1e-10
    atol: float = 1e-12
    max_steps: int = 100_000
    carriers: dict[str, Carrier] = field(default_factory=dict)
    frame: RotatingFrame | None = None


@dataclass(frozen=True)
class EngineCapabilities:
    name: str
    device_kind: Literal["cpu", "cuda"]
    open_system: bool
    differentiable: bool
    batched: bool = False


@dataclass(frozen=True)
class EvolutionResult:
    times_ns: np.ndarray
    states: tuple[np.ndarray, ...]
    expectations: dict[str, np.ndarray]
    dimensions: tuple[int, ...]
    engine: str
    method: str
    dtype: str
    device: str
    diagnostics: dict[str, object] = field(default_factory=dict)


class Integrator(Protocol):
    def capabilities(self) -> EngineCapabilities: ...

    def evolve(self, system: ConstructedSystem, request: EvolutionInput) -> EvolutionResult: ...


@dataclass(frozen=True)
class NormalizedInput:
    initial_density: np.ndarray
    times_ns: np.ndarray
    observables: dict[str, np.ndarray]
    drive_amplitudes_ghz: dict[str, float | np.ndarray]
    rtol: float
    atol: float
    max_steps: int
    carriers: dict[str, Carrier]
    frame: RotatingFrame | None

    @property
    def piecewise(self) -> bool:
        return any(isinstance(value, np.ndarray) for value in self.drive_amplitudes_ghz.values())

    def amplitudes_for_interval(self, index: int) -> dict[str, float]:
        return {
            name: float(value[index]) if isinstance(value, np.ndarray) else value
            for name, value in self.drive_amplitudes_ghz.items()
        }


def normalize_input(system: ConstructedSystem, request: EvolutionInput) -> NormalizedInput:
    if (
        isinstance(request.max_steps, bool)
        or not isinstance(request.max_steps, Integral)
        or request.max_steps <= 0
    ):
        raise EvolutionInputError("max_steps must be a positive integer")
    try:
        times = np.asarray(request.times_ns, dtype=float)
    except (TypeError, ValueError) as exc:
        raise EvolutionInputError("times_ns must be a finite, increasing uniform grid") from exc
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.all(np.isfinite(times))
        or not np.all(np.diff(times) > 0)
        or not np.allclose(np.diff(times), times[1] - times[0], rtol=1e-10, atol=1e-12)
    ):
        raise EvolutionInputError("times_ns must be a finite, increasing uniform grid")
    if not (
        math.isfinite(request.rtol)
        and math.isfinite(request.atol)
        and request.rtol > 0
        and request.atol > 0
    ):
        raise EvolutionInputError("solver tolerances must be positive and finite")
    dimension = math.prod(system.dimensions)
    try:
        initial = np.asarray(request.initial_state, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise EvolutionInputError("initial state must contain finite numeric values") from exc
    if not np.all(np.isfinite(initial)):
        raise EvolutionInputError("initial state must contain finite values")
    if initial.shape == (dimension,):
        if not np.isclose(np.vdot(initial, initial), 1, rtol=0, atol=1e-10):
            raise EvolutionInputError("initial ket must have unit norm")
        initial = np.outer(initial, initial.conj())
    elif initial.shape == (dimension, dimension):
        if not np.allclose(initial, initial.conj().T, rtol=0, atol=1e-10):
            raise EvolutionInputError("initial density matrix must be Hermitian")
        if not np.isclose(np.trace(initial), 1, rtol=0, atol=1e-10):
            raise EvolutionInputError("initial density matrix must have unit trace")
        if np.linalg.eigvalsh(initial).min() < -1e-10:
            raise EvolutionInputError("initial density matrix must be positive semidefinite")
    else:
        raise EvolutionInputError(
            f"initial state must be a ket ({dimension},) or density matrix ({dimension}, {dimension})"
        )
    observables = {}
    for name, array in request.observables.items():
        try:
            data = np.asarray(array, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise EvolutionInputError(f"observable {name!r} must be numeric") from exc
        if data.shape != (dimension, dimension) or not np.all(np.isfinite(data)):
            raise EvolutionInputError(
                f"observable {name!r} must be a finite ({dimension}, {dimension}) matrix"
            )
        observables[name] = data
    amplitudes = {}
    for name, amplitude in request.drive_amplitudes_ghz.items():
        if name not in system.drives:
            raise EvolutionInputError(f"unknown drive {name!r}; available: {sorted(system.drives)}")
        try:
            value = np.asarray(amplitude, dtype=float)
        except (TypeError, ValueError) as exc:
            raise EvolutionInputError(
                f"drive {name!r} amplitude must be a finite scalar or one value per output interval"
            ) from exc
        if not np.all(np.isfinite(value)):
            raise EvolutionInputError(f"drive {name!r} amplitude must be finite")
        if value.ndim == 0:
            amplitudes[name] = float(value)
        elif value.shape == (len(times) - 1,):
            amplitudes[name] = value
        else:
            raise EvolutionInputError(
                f"drive {name!r} amplitude must be a finite scalar or one value per output interval"
            )
    for name, carrier in request.carriers.items():
        if name not in system.drives:
            raise EvolutionInputError(f"unknown carrier drive {name!r}")
        if not (
            math.isfinite(carrier.frequency_ghz)
            and carrier.frequency_ghz > 0
            and math.isfinite(carrier.phase_rad)
        ):
            raise EvolutionInputError("carrier frequency must be positive and phase finite")
    return NormalizedInput(
        initial,
        times,
        observables,
        amplitudes,
        request.rtol,
        request.atol,
        int(request.max_steps),
        request.carriers,
        request.frame,
    )


def expectations_from_states(
    states: tuple[np.ndarray, ...], observables: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        name: np.real_if_close(np.asarray([np.trace(operator @ state) for state in states]))
        for name, operator in observables.items()
    }
