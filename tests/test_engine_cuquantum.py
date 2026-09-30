"""cuDensityMat adapter evidence on a visible CUDA device."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from qutip.solver.integrator import IntegratorException

from entanglix.engines import EvolutionInput, QuTiPIntegrator
from entanglix.engines.cudensitymat import CuQuantumIntegrator
from entanglix.models import ConstructedSystem, TensorTerm, load_system

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1] / "examples"


def test_cudensitymat_matches_cpu_for_open_piecewise_and_composite():
    pytest.importorskip("cupy")
    pytest.importorskip("qutip_cuquantum")
    gpu = CuQuantumIntegrator()
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
        assert result.engine == "cudensitymat" and result.device == "cuda:0"
        assert result.dtype == "complex128" and result.diagnostics["state_storage"] == "CuState"
        assert "RTX 5090" in result.diagnostics["gpu_name"]
        assert result.diagnostics["compute_capability"] == "12.0"
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
    assert len(result.diagnostics["local_storage"]) >= 2 * len(composite.hamiltonian_terms)


def test_cudensitymat_composite_matches_exact_liouvillian(composite_evolution_reference):
    pytest.importorskip("cupy")
    pytest.importorskip("qutip_cuquantum")
    system, request, expected = composite_evolution_reference
    result = CuQuantumIntegrator().evolve(system, request)
    residual = max(np.linalg.norm(a - b) for a, b in zip(result.states, expected))
    assert residual < 1e-7, residual
    assert result.diagnostics["max_steps"] == request.max_steps
    assert result.diagnostics["step_limit_scope"] == "per_output_interval"


def test_cudensitymat_applies_step_limit():
    pytest.importorskip("cupy")
    pytest.importorskip("qutip_cuquantum")
    system = load_system(ROOT / "two-level/system.toml")
    request = EvolutionInput(np.array([1, 1j]) / np.sqrt(2), np.array([0.0, 1.0]), max_steps=1)
    with pytest.raises(IntegratorException, match="nsteps"):
        CuQuantumIntegrator().evolve(system, request)


def test_cudensitymat_storage_boundaries_on_gpu(tmp_path):
    pytest.importorskip("cupy")
    pytest.importorskip("qutip_cuquantum")
    source = tmp_path / "system.toml"
    for levels, selected, converted in (
        (16, "Dense", "DenseOperator"),
        (17, "Dia", "MultidiagonalOperator"),
    ):
        source.write_text(
            (ROOT / "lc-circuit/system.toml")
            .read_text()
            .replace("theta1 = 30", "theta1 = 40")
            .replace("retained_levels = 6", f"retained_levels = {levels}")
        )
        system = load_system(source)
        initial = np.exp(1j * np.arange(levels)) / np.sqrt(levels)
        request = EvolutionInput(initial, np.array([0.0, 0.1]))
        result = CuQuantumIntegrator().evolve(system, request)
        reference = QuTiPIntegrator().evolve(system, request)
        assert all(np.linalg.norm(a - b) < 1e-7 for a, b in zip(result.states, reference.states))
        assert any(
            entry["selected"] == selected and entry["cudensitymat_data"] == converted
            for entry in result.diagnostics["local_storage"]
        ), result.diagnostics["local_storage"]
    dense = np.diag(np.linspace(0, 0.2, 20)) + np.ones((20, 20)) * 0.001
    wide = ConstructedSystem(
        id="wide_reference",
        hamiltonian_ghz=dense,
        operators={},
        dissipators={},
        dimensions=(20,),
        basis="declared",
        provenance={},
        hamiltonian_terms=(TensorTerm(1, (dense,)),),
    )
    initial = np.eye(20)[0]
    result = CuQuantumIntegrator().evolve(wide, EvolutionInput(initial, np.array([0.0, 0.01])))
    reference = QuTiPIntegrator().evolve(wide, EvolutionInput(initial, np.array([0.0, 0.01])))
    assert all(np.linalg.norm(a - b) < 1e-7 for a, b in zip(result.states, reference.states))
    assert result.diagnostics["local_storage"] == [
        {
            "selected": "Dense",
            "qutip_data": "Dense",
            "cudensitymat_data": "DenseOperator",
            "dimension": 20,
        }
    ]
    banded = np.diag(np.linspace(0, 0.2, 17)) + 0.01 * (np.eye(17, k=1) + np.eye(17, k=-1))
    narrow = replace(
        wide,
        hamiltonian_ghz=banded,
        dimensions=(17,),
        hamiltonian_terms=(TensorTerm(1, (banded,)),),
    )
    initial = np.exp(1j * np.arange(17)) / np.sqrt(17)
    request = EvolutionInput(initial, np.linspace(0, 2, 5))
    result = CuQuantumIntegrator().evolve(narrow, request)
    reference = QuTiPIntegrator().evolve(narrow, request)
    assert result.diagnostics["local_storage"][0]["cudensitymat_data"] == "MultidiagonalOperator"
    assert all(np.linalg.norm(a - b) < 1e-7 for a, b in zip(result.states, reference.states))
