"""Shared evolution conventions and CPU-checkable GPU storage selection."""

from pathlib import Path

import numpy as np
import pytest
from qutip.solver.integrator import IntegratorException

from entanglix.engine_cuquantum import local_storage
from entanglix.engines import EvolutionInput, EvolutionInputError, QuTiPIntegrator
from entanglix.models import load_system

TWO_LEVEL = Path(__file__).resolve().parents[1] / "examples/two-level/system.toml"


def test_integrator_contract_and_interval_boundary():
    system = load_system(TWO_LEVEL)
    engine = QuTiPIntegrator()
    assert engine.capabilities().open_system
    assert engine.capabilities().device_kind == "cpu"
    assert not engine.capabilities().batched
    request = EvolutionInput(
        np.array([1, 0]),
        np.array([0.0, 1.0, 2.0]),
        {"excited": np.diag([0, 1])},
        {"x_drive": np.array([0.125, 0.0])},
    )
    result = engine.evolve(system, request)
    assert result.dimensions == (2,)
    assert (result.engine, result.method, result.dtype, result.device) == (
        "qutip-cpu",
        "vern9",
        "complex128",
        "cpu",
    )
    assert result.diagnostics["segments"] == 2
    assert result.diagnostics["max_steps"] == request.max_steps
    assert result.diagnostics["step_limit_scope"] == "per_output_interval"
    assert len(result.states) == len(request.times_ns)
    np.testing.assert_allclose(
        result.expectations["excited"].real[2],
        result.expectations["excited"].real[1] * np.exp(-0.01),
        atol=2e-8,
    )


def test_shared_input_rejects_bad_interval_values_and_nonpositive_states():
    system = load_system(TWO_LEVEL)
    engine = QuTiPIntegrator()
    times = np.array([0.0, 1.0, 2.0])
    with pytest.raises(EvolutionInputError, match="one value per output interval"):
        engine.evolve(
            system, EvolutionInput(np.array([1, 0]), times, drive_amplitudes_ghz={"x_drive": [0.1]})
        )
    with pytest.raises(EvolutionInputError, match="positive semidefinite"):
        engine.evolve(system, EvolutionInput(np.diag([1.1, -0.1]), times))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "10", None])
def test_shared_input_rejects_invalid_step_limit(limit):
    system = load_system(TWO_LEVEL)
    with pytest.raises(EvolutionInputError, match="max_steps must be a positive integer"):
        QuTiPIntegrator().evolve(
            system, EvolutionInput(np.array([1, 0]), np.array([0.0, 1.0]), max_steps=limit)
        )


def test_cpu_step_limit_is_applied():
    system = load_system(TWO_LEVEL)
    with pytest.raises(IntegratorException, match="step"):
        QuTiPIntegrator().evolve(
            system,
            EvolutionInput(np.array([1, 1j]) / np.sqrt(2), np.array([0.0, 1.0]), max_steps=1),
        )


def test_cpu_composite_matches_exact_liouvillian(composite_evolution_reference):
    system, request, expected = composite_evolution_reference
    result = QuTiPIntegrator().evolve(system, request)
    residual = max(np.linalg.norm(a - b) for a, b in zip(result.states, expected))
    assert residual < 1e-7, residual
    assert result.diagnostics["max_steps"] == request.max_steps


def test_local_storage_rule_at_sixteen_diagonals():
    assert local_storage(np.eye(16)) == "Dense"
    assert local_storage(np.eye(17)) == "Dia"
    narrow = np.eye(20)
    for offset in range(1, 8):
        narrow += np.eye(20, k=offset) + np.eye(20, k=-offset)
    assert local_storage(narrow) == "Dia"  # 15 occupied diagonals.
    wide = narrow + np.eye(20, k=8) + np.eye(20, k=-8)
    assert local_storage(wide) == "Dense"  # 17 occupied diagonals.
