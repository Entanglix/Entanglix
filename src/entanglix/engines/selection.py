"""Versioned reference-size selection policy; the runner supplies availability and limits."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from numbers import Integral

import numpy as np

from entanglix.evolution import EvolutionInput, normalize_input
from entanglix.models import ConstructedSystem


class EngineSelectionError(ValueError):
    """No declared engine satisfies the requested capabilities and limits."""


def selection_table() -> dict:
    return tomllib.loads(files("entanglix.engines").joinpath("selection.toml").read_text())


@dataclass(frozen=True)
class EngineLimits:
    host_memory_bytes: int
    device_memory_bytes: int

    def __post_init__(self):
        for name in ("host_memory_bytes", "device_memory_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise EngineSelectionError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class EngineSelection:
    engine: str
    table_id: str
    estimated_host_bytes: int
    estimated_device_bytes: int
    fallback_from: str | None = None
    fallback_reason: str | None = None


def select_engine(
    system: ConstructedSystem,
    request: EvolutionInput,
    *,
    limits: EngineLimits,
    available_engines: set[str],
    needs_gradients: bool = False,
    batched: bool = False,
    fallbacks: tuple[str, ...] = (),
) -> EngineSelection:
    table = selection_table()
    engines = table["engines"]
    if not isinstance(needs_gradients, bool) or not isinstance(batched, bool):
        raise EngineSelectionError("needs_gradients and batched must be booleans")
    unknown = (set(fallbacks) | available_engines) - engines.keys()
    if unknown:
        raise EngineSelectionError(f"unknown engines: {sorted(unknown)}")
    dimension = math.prod(system.dimensions)
    primary = "dynamiqs" if needs_gradients else "qutip-cpu"
    matrices = [
        system.hamiltonian_ghz,
        *system.operators.values(),
        *system.drives.values(),
        *(operator for operator, _ in system.dissipators.values()),
    ]
    factors = [factor for term in system.hamiltonian_terms for factor in term.factors]
    factors += [
        factor
        for mapping in (system.operator_factors, system.drive_factors, system.dissipator_factors)
        for values in mapping.values()
        for factor in values
    ]
    model_bytes = sum(array.nbytes for array in matrices + factors)
    state_bytes = 16 * dimension**2
    reasons = []
    for name in dict.fromkeys((primary, *fallbacks)):
        row = engines[name]
        if name not in available_engines:
            reasons.append(f"{name}: unavailable")
            continue
        if (
            (needs_gradients and not row["differentiable"])
            or (batched and not row["batched"])
            or (any(rate > 0 for _, rate in system.dissipators.values()) and not row["open_system"])
        ):
            reasons.append(f"{name}: requested capabilities are unsupported")
            continue
        if dimension > row["max_dimension"]:
            reasons.append(
                f"{name}: dimension {dimension} exceeds validated cap {row['max_dimension']}"
            )
            continue
        # Reject unsupported sizes before materializing the initial density matrix.
        normalized = normalize_input(system, request)
        input_bytes = sum(array.nbytes for array in normalized.observables.values())
        input_bytes += normalized.initial_density.nbytes + normalized.times_ns.nbytes
        input_bytes += sum(
            np.asarray(value).nbytes for value in normalized.drive_amplitudes_ghz.values()
        )
        output_bytes = (state_bytes + 16 * len(normalized.observables)) * len(normalized.times_ns)
        host_bytes = model_bytes + input_bytes + output_bytes
        device_bytes = host_bytes if row["device_kind"] == "cuda" else 0
        if host_bytes > limits.host_memory_bytes or device_bytes > limits.device_memory_bytes:
            reasons.append(
                f"{name}: estimated host/device bytes {host_bytes}/{device_bytes} exceed configured limits {limits.host_memory_bytes}/{limits.device_memory_bytes}"
            )
            continue
        return EngineSelection(
            name,
            table["table_id"],
            host_bytes,
            device_bytes,
            primary if name != primary else None,
            "; ".join(reasons) if name != primary else None,
        )
    raise EngineSelectionError("; ".join(reasons))
