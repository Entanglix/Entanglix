"""Versioned selection by declared needs and available capabilities."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib.resources import files

from entanglix.evolution import EvolutionInput, normalize_input
from entanglix.models import ConstructedSystem


class EngineSelectionError(ValueError):
    """No declared engine satisfies the requested capabilities."""


def selection_table() -> dict:
    return tomllib.loads(files("entanglix.engines").joinpath("selection.toml").read_text())


@dataclass(frozen=True)
class EngineSelection:
    engine: str
    table_id: str
    fallback_from: str | None = None
    fallback_reason: str | None = None


def select_engine(
    system: ConstructedSystem,
    request: EvolutionInput,
    *,
    available_engines: set[str],
    needs_gradients: bool = False,
    device: str | None = None,
    batched: bool = False,
    fallbacks: tuple[str, ...] = (),
) -> EngineSelection:
    table = selection_table()
    engines = table["engines"]
    if not isinstance(needs_gradients, bool) or not isinstance(batched, bool):
        raise EngineSelectionError("needs_gradients and batched must be booleans")
    if device not in {None, "cpu", "cuda"}:
        raise EngineSelectionError("device must be cpu or cuda")
    unknown = (set(fallbacks) | available_engines) - engines.keys()
    if unknown:
        raise EngineSelectionError(f"unknown engines: {sorted(unknown)}")
    primary = "dynamiqs" if needs_gradients else "cudensitymat" if device == "cuda" else "qutip-cpu"
    reasons = []
    for name in dict.fromkeys((primary, *fallbacks)):
        row = engines[name]
        if name not in available_engines:
            reasons.append(f"{name}: unavailable")
            continue
        if device is not None and name != "dynamiqs" and row["device_kind"] != device:
            reasons.append(f"{name}: requested device {device} is unsupported")
            continue
        if (
            (needs_gradients and not row["differentiable"])
            or (batched and not row["batched"])
            or (any(rate > 0 for _, rate in system.dissipators.values()) and not row["open_system"])
        ):
            reasons.append(f"{name}: requested capabilities are unsupported")
            continue
        normalize_input(system, request)
        return EngineSelection(
            name,
            table["table_id"],
            primary if name != primary else None,
            "; ".join(reasons) if name != primary else None,
        )
    raise EngineSelectionError("; ".join(reasons))
