"""Selection refuses unavailable, oversized, and incompatible requests explicitly."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from entanglix.engine_cuquantum import CuQuantumIntegrator
from entanglix.engine_dynamiqs import DynamiqsIntegrator
from entanglix.engine_selection import (
    EngineLimits,
    EngineSelectionError,
    select_engine,
    selection_table,
)
from entanglix.engines import EvolutionInput, QuTiPIntegrator
from entanglix.models import load_system

ROOT = Path(__file__).resolve().parents[1] / "examples"
LIMITS = EngineLimits(1 << 30, 1 << 30)


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
        limits=LIMITS,
        available_engines={"qutip-cpu", "cudensitymat", "dynamiqs"},
        needs_gradients=gradients,
    )
    assert result.engine == selected and result.table_id == "reference-engines-v1"
    assert result.fallback_from is None
    assert 0 < result.estimated_host_bytes <= LIMITS.host_memory_bytes
    assert result.estimated_device_bytes <= LIMITS.device_memory_bytes


def test_missing_cpu_requires_explicit_compatible_gpu_fallback():
    system, request = inputs("qubit-resonator")
    with pytest.raises(EngineSelectionError, match="unavailable"):
        select_engine(system, request, limits=LIMITS, available_engines={"cudensitymat"})
    result = select_engine(
        system,
        request,
        limits=LIMITS,
        available_engines={"cudensitymat"},
        fallbacks=("cudensitymat",),
    )
    assert result.engine == "cudensitymat" and result.fallback_from == "qutip-cpu"
    assert "unavailable" in result.fallback_reason
    with pytest.raises(EngineSelectionError, match="capabilities"):
        select_engine(
            system,
            request,
            limits=LIMITS,
            available_engines={"qutip-cpu"},
            needs_gradients=True,
            fallbacks=("qutip-cpu",),
        )


def test_size_memory_and_batch_requirements_cannot_be_waived():
    system, request = inputs("three-subsystem")
    with pytest.raises(EngineSelectionError, match="validated cap"):
        select_engine(
            system,
            request,
            limits=LIMITS,
            available_engines={"qutip-cpu", "cudensitymat"},
            fallbacks=("cudensitymat",),
        )
    system, request = inputs("two-level")
    with pytest.raises(EngineSelectionError, match="capabilities"):
        select_engine(system, request, limits=LIMITS, available_engines={"qutip-cpu"}, batched=True)
    with pytest.raises(EngineSelectionError, match="estimated host/device bytes"):
        select_engine(system, request, limits=EngineLimits(1, 1), available_engines={"qutip-cpu"})
    with pytest.raises(EngineSelectionError, match="capabilities"):
        select_engine(
            system,
            request,
            limits=EngineLimits(1 << 30, 1),
            available_engines={"dynamiqs", "qutip-cpu"},
            needs_gradients=True,
            fallbacks=("qutip-cpu",),
        )
    short = select_engine(
        system, request, limits=LIMITS, available_engines={"dynamiqs"}, needs_gradients=True
    )
    long = select_engine(
        system,
        replace(request, times_ns=np.linspace(0, 1, 101)),
        limits=LIMITS,
        available_engines={"dynamiqs"},
        needs_gradients=True,
    )
    assert long.estimated_host_bytes > short.estimated_host_bytes
    assert long.estimated_device_bytes > short.estimated_device_bytes


def test_long_output_grid_exceeds_known_array_storage_budget():
    system, request = inputs("two-level")
    limits = EngineLimits(1 << 20, 1 << 20)
    select_engine(system, request, limits=limits, available_engines={"qutip-cpu"})
    with pytest.raises(EngineSelectionError, match="estimated host/device bytes"):
        select_engine(
            system,
            replace(request, times_ns=np.linspace(0, 1, 100_001)),
            limits=limits,
            available_engines={"qutip-cpu"},
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_limits_require_explicit_positive_integer_bytes(value):
    with pytest.raises(EngineSelectionError, match="positive integer"):
        EngineLimits(value, 1 << 30)
