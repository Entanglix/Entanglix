"""dynamiqs adapter evidence on a visible CUDA device."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from entanglix.engine_dynamiqs import DynamiqsIntegrator
from entanglix.engines import EvolutionInput, QuTiPIntegrator
from entanglix.models import load_system

ROOT = Path(__file__).resolve().parents[1] / "examples"


def test_dynamiqs_cpu_matches_reference():
    if importlib.util.find_spec("dynamiqs") is None:
        pytest.skip("install the differentiable extra for the CPU JAX check")
    system = load_system(ROOT / "two-level/system.toml")
    request = EvolutionInput(np.array([0, 1]), np.linspace(0, 10, 5), {"excited": np.diag([0, 1])})
    reference = QuTiPIntegrator().evolve(system, request)
    result = DynamiqsIntegrator(device_kind="cpu").evolve(system, request)
    assert result.device.startswith("TFRT_CPU")
    assert result.dtype == "complex128"
    assert result.diagnostics["max_steps"] == request.max_steps
    assert result.diagnostics["step_limit_scope"] == "per_solve"
    assert all(np.linalg.norm(a - b) < 1e-6 for a, b in zip(result.states, reference.states))
    # JAX wraps the solver's step-limit error in its runtime error type.
    limited = EvolutionInput(np.array([1, 1j]) / np.sqrt(2), np.array([0.0, 1.0]), max_steps=1)
    with pytest.raises(Exception, match="maximum number of solver steps"):
        DynamiqsIntegrator(device_kind="cpu").evolve(system, limited)


@pytest.mark.gpu
def test_dynamiqs_matches_cpu_for_open_piecewise_and_composite():
    pytest.importorskip("jax")
    pytest.importorskip("dynamiqs")
    gpu = DynamiqsIntegrator()
    cpu = QuTiPIntegrator()
    assert gpu.capabilities().open_system and gpu.capabilities().device_kind == "cuda"
    system = load_system(ROOT / "two-level/system.toml")
    requests = (
        EvolutionInput(np.array([0, 1]), np.linspace(0, 10, 5), {"excited": np.diag([0, 1])}),
        EvolutionInput(
            np.array([1, 0]),
            np.array([0.0, 1.0, 2.0]),
            {"excited": np.diag([0, 1])},
            {"x_drive": np.array([0.125, 0.0])},
        ),
    )
    for request in requests:
        reference = cpu.evolve(system, request)
        result = gpu.evolve(system, request)
        assert result.engine == "dynamiqs" and result.device.startswith("cuda:")
        assert result.dtype == "complex128"
        assert "RTX 5090" in result.diagnostics["device_kind"]
        assert all(
            np.linalg.norm(actual - expected) < 1e-6
            for actual, expected in zip(result.states, reference.states)
        )
    closed = replace(system, dissipators={}, dissipator_factors={})
    coherent = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    request = EvolutionInput(coherent, np.linspace(0, 2, 5))
    reference = cpu.evolve(closed, request)
    result = gpu.evolve(closed, request)
    assert all(np.linalg.norm(a - b) < 1e-6 for a, b in zip(result.states, reference.states))
    composite = load_system(ROOT / "qubit-resonator/system.toml")
    initial = np.zeros(30)
    initial[1] = 1
    request = EvolutionInput(initial, np.array([0.0, 0.1]))
    reference = cpu.evolve(composite, request)
    result = gpu.evolve(composite, request)
    assert all(np.linalg.norm(a - b) < 1e-6 for a, b in zip(result.states, reference.states))


@pytest.mark.gpu
def test_dynamiqs_composite_matches_exact_liouvillian(composite_evolution_reference):
    pytest.importorskip("jax")
    pytest.importorskip("dynamiqs")
    system, request, expected = composite_evolution_reference
    result = DynamiqsIntegrator().evolve(system, request)
    residual = max(np.linalg.norm(a - b) for a, b in zip(result.states, expected))
    assert residual < 1e-7, residual
    assert result.diagnostics["max_steps"] == request.max_steps
    assert result.diagnostics["step_limit_scope"] == "per_solve"
