"""Engine choice by source run, spec, user default, or declared needs; plugins by entry point."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution, entry_points
from importlib.resources import files

from entanglix.evolution import EvolutionInput, normalize_input
from entanglix.models import ConstructedSystem

# Built-in engines and the distributions that provide them.
BUILT_IN = {"qutip-cpu": "qutip", "cudensitymat": "qutip-cuquantum", "dynamiqs": "dynamiqs"}


class EngineSelectionError(ValueError):
    """The chosen engine is missing or cannot meet the declared needs."""


def selection_table() -> dict:
    return tomllib.loads(files("entanglix.engines").joinpath("selection.toml").read_text())


@dataclass(frozen=True)
class EngineSelection:
    engine: str
    table_id: str
    source: str
    distribution: str | None = None
    version: str | None = None
    record_sha256: str | None = None


def _identity(dist) -> dict:
    from entanglix.runner import record_digest

    return {
        "distribution": dist.metadata["Name"],
        "version": dist.version,
        "record_sha256": record_digest(dist),
    }


def engine_plugins() -> dict:
    entries = {}
    for entry in entry_points(group="entanglix.integrators"):
        if entry.name in entries or entry.name in BUILT_IN:
            raise EngineSelectionError(f"duplicate engine entry point: {entry.name}")
        entries[entry.name] = entry
    return entries


def load_engine(name: str):
    """Load a plugin engine with the identity of the distribution that provides it."""
    entries = engine_plugins()
    if name not in entries:
        raise EngineSelectionError(
            f"engine {name!r} is not installed; installed plugin engines: {sorted(entries)}"
        )
    entry = entries[name]
    return entry.load()(), _identity(entry.dist)


def select_engine(
    system: ConstructedSystem,
    request: EvolutionInput,
    *,
    available_engines: set[str],
    needs_gradients: bool = False,
    device: str | None = None,
    batched: bool = False,
    engine: str | None = None,
    pinned: str | None = None,
    defaults: dict[str, str] | None = None,
) -> EngineSelection:
    """A pinned or named engine, else the user's default, else the built-in rule; no fallback."""
    table = selection_table()
    engines = table["engines"]
    if not isinstance(needs_gradients, bool) or not isinstance(batched, bool):
        raise EngineSelectionError("needs_gradients and batched must be booleans")
    if device not in {None, "cpu", "cuda"}:
        raise EngineSelectionError("device must be cpu or cuda")
    unknown = available_engines - engines.keys()
    if unknown:
        raise EngineSelectionError(f"unknown engines: {sorted(unknown)}")
    defaults = defaults or {}
    if pinned is not None:
        name, source = pinned, "source_run"
    elif engine is not None:
        name, source = engine, "spec"
    elif (device or "cpu") in defaults:
        name, source = defaults[device or "cpu"], "default"
    else:
        name = (
            "dynamiqs" if needs_gradients else "cudensitymat" if device == "cuda" else "qutip-cpu"
        )
        source = "rule"
    if name in engines:
        if name not in available_engines:
            raise EngineSelectionError(f"engine {name} ({source}) is not installed")
        row = engines[name]
        try:
            identity = _identity(distribution(BUILT_IN[name]))
        except PackageNotFoundError:
            identity = {}  # A caller may declare availability without the package installed.
    else:
        integrator, identity = load_engine(name)
        capabilities = integrator.capabilities()
        # Plugin engines serve runs without gradients or batching.
        row = {
            "device_kind": capabilities.device_kind,
            "open_system": capabilities.open_system,
            "differentiable": False,
            "batched": False,
        }
    unmet = []
    if device is not None and name != "dynamiqs" and row["device_kind"] != device:
        unmet.append(f"device {device}")
    if needs_gradients and not row["differentiable"]:
        unmet.append("gradients")
    if batched and not row["batched"]:
        unmet.append("batching")
    if any(rate > 0 for _, rate in system.dissipators.values()) and not row["open_system"]:
        unmet.append("open-system evolution")
    if unmet:
        raise EngineSelectionError(
            f"engine {name} ({source}) does not support {', '.join(unmet)}; "
            "no other engine replaces it"
        )
    normalize_input(system, request)
    return EngineSelection(name, table["table_id"], source, **identity)
