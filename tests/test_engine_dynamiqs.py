"""dynamiqs adapter evidence on a visible CUDA device."""

import importlib.util
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from entanglix.engines import Carrier, EvolutionInput, QuTiPIntegrator
from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator
from entanglix.models import TensorTerm, load_system

ROOT = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("opened", [False, True])
@pytest.mark.parametrize("piecewise", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_dynamiqs_mixed_controls_and_carriers_reuse_compilation(
    opened, piecewise, device, caplog, record_property
):
    jax = pytest.importorskip("jax")
    system = load_system(ROOT / "two-level/system.toml")
    z = np.diag([1.0, -1.0]).astype(complex)
    system = replace(
        system,
        drives={**system.drives, "z_drive": z},
        drive_factors={**system.drive_factors, "z_drive": (z,)},
        dissipators={"relaxation": (system.operators["lower"], 0.1)} if opened else {},
        dissipator_factors=system.dissipator_factors if opened else {},
    )
    values = np.array([0.08, -0.03, 0.06]) if piecewise else 0.08
    request = EvolutionInput(
        np.array([1, 1j]) / np.sqrt(2),
        np.linspace(0.17, 2.17, 4),
        {"excited": np.diag([0.0, 1.0])},
        {"x_drive": values, "z_drive": 0.017},
        carriers={"x_drive": Carrier(0.27, 0.31)},
    )
    engine = DynamiqsIntegrator(device)
    actual = engine.evolve(system, request)
    reference = QuTiPIntegrator().evolve(system, request)
    assert max(np.linalg.norm(a - b) for a, b in zip(actual.states, reference.states)) < 1e-7
    np.testing.assert_allclose(
        actual.expectations["excited"], reference.expectations["excited"], atol=1e-7
    )
    # Change numeric values and dictionary order while preserving the array shapes.
    repeated = replace(
        request,
        drive_amplitudes_ghz={"z_drive": 0.018, "x_drive": values * 0.9},
        carriers={"x_drive": Carrier(0.28, 0.32)},
    )
    with caplog.at_level(logging.WARNING), jax.log_compiles(True):
        engine.evolve(system, repeated)
        engine.evolve_jax(system, repeated).states.block_until_ready()
    assert not [record.message for record in caplog.records if "Compiling " in record.message]
    record_property("device", actual.device)
    record_property("device_kind", actual.diagnostics["device_kind"])


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_dynamiqs_jitted_carrier_gradient_with_traced_scalar_control(device):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    system = load_system(ROOT / "two-level/system.toml")
    z = np.diag([1.0, -1.0]).astype(complex)
    system = replace(
        system,
        drives={**system.drives, "z_drive": z},
        drive_factors={**system.drive_factors, "z_drive": (z,)},
    )
    request = EvolutionInput(
        np.array([1, 1j]) / np.sqrt(2),
        np.linspace(0.13, 1.13, 4),
        {"excited": np.diag([0.0, 1.0])},
        carriers={"x_drive": Carrier(0.37, 0.29)},
    )
    engine = DynamiqsIntegrator(device)

    def objective(intervals, scalar):
        return (
            engine.evolve_jax(
                system, request, drive_amplitudes_ghz={"x_drive": intervals, "z_drive": scalar}
            )
            .expectations["excited"][-1]
            .real
        )

    controls = np.array([0.08, 0.11, -0.04])
    value, (gradient, scalar_gradient) = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        jnp.asarray(controls), jnp.asarray(0.02)
    )
    flat = np.r_[controls, 0.02]
    expected = []
    for index in range(len(flat)):
        displacement = np.zeros_like(flat)
        displacement[index] = 1e-5
        values = []
        for candidate in (flat + displacement, flat - displacement):
            result = QuTiPIntegrator().evolve(
                system,
                replace(
                    request,
                    drive_amplitudes_ghz={"x_drive": candidate[:3], "z_drive": candidate[3]},
                ),
            )
            values.append(result.expectations["excited"][-1].real)
        expected.append((values[0] - values[1]) / 2e-5)
    np.testing.assert_allclose(
        np.r_[np.asarray(gradient), float(scalar_gradient)], expected, rtol=1e-4, atol=1e-6
    )
    assert np.isfinite(float(value))


@pytest.mark.parametrize("piecewise", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_drive_gradient_through_evolution(device, piecewise, record_property):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    system = load_system(ROOT / "two-level/system.toml")
    zero = np.zeros((2, 2), dtype=complex)
    system = replace(
        system,
        hamiltonian_ghz=zero,
        hamiltonian_terms=(TensorTerm(1, (zero,)),),
        dissipators={"relaxation": (system.operators["lower"], 0.3)} if piecewise else {},
        dissipator_factors=system.dissipator_factors if piecewise else {},
    )
    request = EvolutionInput(np.array([1, 0]), np.linspace(0, 1, 3), {"excited": np.diag([0, 1])})
    engine = DynamiqsIntegrator(device_kind=device)

    def objective(amplitude):
        return (
            engine.evolve_jax(system, request, drive_amplitudes_ghz={"x_drive": amplitude})
            .expectations["excited"][-1]
            .real
        )

    amplitude = jnp.array([0.08, 0.13]) if piecewise else jnp.array(0.1)
    value, derivative = jax.value_and_grad(objective)(amplitude)
    derivative = np.asarray(derivative)
    assert np.all(np.abs(derivative) > 1e-3)
    errors = []
    for epsilon in (1e-3, 1e-4):
        finite = []
        for index in range(amplitude.size):
            displacement = np.zeros(amplitude.shape)
            displacement.flat[index] = epsilon
            values = []
            for control in (
                np.asarray(amplitude) + displacement,
                np.asarray(amplitude) - displacement,
            ):
                result = QuTiPIntegrator().evolve(
                    system, replace(request, drive_amplitudes_ghz={"x_drive": control})
                )
                values.append(result.expectations["excited"][-1].real)
            finite.append((values[0] - values[1]) / (2 * epsilon))
        finite = np.asarray(finite).reshape(amplitude.shape)
        np.testing.assert_allclose(derivative, finite, rtol=1e-4, atol=1e-6)
        errors.append(float(np.max(np.abs(derivative - finite))))
    if not piecewise:
        analytic = 2 * np.pi * np.sin(4 * np.pi * float(amplitude))
        np.testing.assert_allclose(derivative, analytic, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(value, np.sin(2 * np.pi * float(amplitude)) ** 2, atol=1e-8)
    assert engine.capabilities().differentiable
    record_property("gradient_fd_max_error", max(errors))
    record_property("gradient_units", "expectation/GHz")
    record_property("device", "cuda" if device == "gpu" else "cpu")
    record_property("gradient_dtype", str(derivative.dtype))


def test_differentiable_controls_validate_names_shapes_and_values():
    pytest.importorskip("dynamiqs")
    from entanglix.evolution import EvolutionInputError

    system = load_system(ROOT / "two-level/system.toml")
    request = EvolutionInput(np.array([1, 0]), np.array([0.0, 1.0, 2.0]))
    engine = DynamiqsIntegrator(device_kind="cpu")
    for controls, message in (
        ({"unknown": 0.1}, "unknown drive"),
        ({"x_drive": np.ones(3)}, "scalar or one value per output interval"),
        ({"x_drive": np.nan}, "finite"),
    ):
        with pytest.raises(EvolutionInputError, match=message):
            engine.evolve_jax(system, request, drive_amplitudes_ghz=controls)


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
