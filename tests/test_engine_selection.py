"""Selection refuses unavailable, oversized, and incompatible requests explicitly."""

from pathlib import Path

import numpy as np
import pytest

from entanglix.engines import EvolutionInput, QuTiPIntegrator
from entanglix.engines.cudensitymat import CuQuantumIntegrator
from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator
from entanglix.engines.selection import (
    EngineSelectionError,
    select_engine,
    selection_table,
)
from entanglix.models import load_system

ROOT = Path(__file__).resolve().parents[1] / "examples"


def inputs(name):
    system = load_system(ROOT / name / "system.toml")
    initial = np.zeros(np.prod(system.dimensions))
    initial[0] = 1
    return system, EvolutionInput(initial, np.array([0.0, 1.0]))


def test_policy_agrees_with_implemented_capabilities():
    table = selection_table()
    assert table["format_version"] == 1
    for engine in (QuTiPIntegrator(), CuQuantumIntegrator(), DynamiqsIntegrator()):
        caps = engine.capabilities()
        row = table["engines"][caps.name]
        for field in ("device_kind", "open_system", "differentiable", "batched"):
            assert row[field] == getattr(caps, field)


@pytest.mark.parametrize(
    "system_name, gradients, selected",
    [
        ("two-level", False, "qutip-cpu"),
        ("qubit-resonator", False, "qutip-cpu"),
        ("two-level", True, "dynamiqs"),
        ("qubit-resonator", True, "dynamiqs"),
    ],
)
def test_selection_follows_declared_needs(system_name, gradients, selected):
    system, request = inputs(system_name)
    result = select_engine(
        system,
        request,
        available_engines={"qutip-cpu", "cudensitymat", "dynamiqs"},
        needs_gradients=gradients,
    )
    assert result.engine == selected and result.table_id == "declared-needs-engines-v3"
    assert result.fallback_from is None


def test_missing_cpu_requires_explicit_compatible_gpu_fallback():
    system, request = inputs("qubit-resonator")
    with pytest.raises(EngineSelectionError, match="unavailable"):
        select_engine(system, request, available_engines={"cudensitymat"})
    result = select_engine(
        system,
        request,
        available_engines={"cudensitymat"},
        fallbacks=("cudensitymat",),
    )
    assert result.engine == "cudensitymat" and result.fallback_from == "qutip-cpu"
    assert "unavailable" in result.fallback_reason
    with pytest.raises(EngineSelectionError, match="capabilities"):
        select_engine(
            system,
            request,
            available_engines={"qutip-cpu"},
            needs_gradients=True,
            fallbacks=("qutip-cpu",),
        )


@pytest.mark.parametrize(
    "device, gradients, engine",
    [
        ("cpu", False, "qutip-cpu"),
        ("cuda", False, "cudensitymat"),
        ("cpu", True, "dynamiqs"),
        ("cuda", True, "dynamiqs"),
    ],
)
def test_explicit_device_dispatch_and_storage(device, gradients, engine):
    system, request = inputs("two-level")
    selection = select_engine(
        system,
        request,
        available_engines={"qutip-cpu", "dynamiqs", "cudensitymat"},
        device=device,
        needs_gradients=gradients,
    )
    assert selection.engine == engine


def test_large_model_dispatch_and_unsupported_batch():
    system, request = inputs("three-subsystem")
    selection = select_engine(system, request, available_engines={"qutip-cpu"})
    assert selection.engine == "qutip-cpu"
    with pytest.raises(EngineSelectionError, match="capabilities"):
        select_engine(system, request, available_engines={"qutip-cpu"}, batched=True)
