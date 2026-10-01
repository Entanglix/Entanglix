"""Continuous carriers and numeric RWA: independent references on CPU and CUDA."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from entanglix.corrector.checks import physical_residuals
from entanglix.engines import (
    Carrier,
    EvolutionInput,
    EvolutionInputError,
    QuTiPIntegrator,
    RotatingFrame,
)
from entanglix.engines.frames import matrix, prepare_dynamics
from entanglix.evolution import normalize_input
from entanglix.models import TensorTerm, load_system

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(
    params=[
        "qutip",
        "dynamiqs-cpu",
        pytest.param("dynamiqs-gpu", marks=pytest.mark.gpu),
        pytest.param("cudensitymat", marks=pytest.mark.gpu),
    ]
)
def integrator(request):
    if request.param == "qutip":
        return QuTiPIntegrator()
    if request.param.startswith("dynamiqs"):
        pytest.importorskip("dynamiqs")
        from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

        return DynamiqsIntegrator("cpu" if request.param.endswith("cpu") else "gpu")
    pytest.importorskip("cupy")
    from entanglix.engines.cudensitymat import CuQuantumIntegrator

    return CuQuantumIntegrator()


def assert_physical(states):
    residuals = physical_residuals(np.asarray(states))
    assert residuals["trace_error"] < 1e-8
    assert residuals["hermiticity_error"] < 1e-8
    assert residuals["minimum_eigenvalue"] >= -1e-8


@pytest.mark.parametrize("piecewise", [False, True])
def test_carrier_matches_analytic_commuting_drive(integrator, piecewise, record_property):
    system = load_system(ROOT / "examples/two-level/system.toml")
    zero = np.zeros((2, 2), dtype=complex)
    system = replace(
        system,
        hamiltonian_ghz=zero,
        hamiltonian_terms=(TensorTerm(1, (zero,)),),
        dissipators={},
        dissipator_factors={},
    )
    times = np.linspace(0.17, 4.17, 5)
    amplitudes = np.array([0.06, -0.04, 0.02, 0.08]) if piecewise else 0.05
    carrier = Carrier(0.73, 0.47)
    ket = np.array([1, 1j]) / np.sqrt(2)
    request = EvolutionInput(
        ket, times, drive_amplitudes_ghz={"x_drive": amplitudes}, carriers={"x_drive": carrier}
    )
    result = integrator.evolve(system, request)
    angle = 0.0
    reference = [np.outer(ket, ket.conj())]
    for index, (start, end) in enumerate(zip(times[:-1], times[1:])):
        amplitude = amplitudes[index] if piecewise else amplitudes
        angle += (
            amplitude
            / carrier.frequency_ghz
            * (
                np.sin(2 * np.pi * carrier.frequency_ghz * end + carrier.phase_rad)
                - np.sin(2 * np.pi * carrier.frequency_ghz * start + carrier.phase_rad)
            )
        )
        evolved = expm(-1j * angle * system.drives["x_drive"]) @ ket
        reference.append(np.outer(evolved, evolved.conj()))
    error = max(np.linalg.norm(a - b) for a, b in zip(result.states, reference))
    assert error < 1e-7
    assert_physical(result.states)
    record_property("state_error", error)
    record_property("device", result.device)


def test_lossy_carrier_matches_independent_ode(integrator, record_property):
    system = load_system(ROOT / "examples/lossy-oscillator/system.toml")
    ket = np.eye(6)[1].astype(complex)
    ket = (ket + 1j * np.eye(6)[0]) / np.sqrt(2)
    times = np.linspace(0, 4, 5)
    carrier = Carrier(0.2, -0.37)
    request = EvolutionInput(
        ket,
        times,
        drive_amplitudes_ghz={"position_drive": 0.02},
        carriers={"position_drive": carrier},
        rtol=1e-9,
        atol=1e-11,
    )

    def derivative(t, vector):
        rho = vector.reshape(6, 6)
        h = (
            2
            * np.pi
            * (
                system.hamiltonian_ghz
                + 0.02
                * np.cos(2 * np.pi * carrier.frequency_ghz * t + carrier.phase_rad)
                * system.drives["position_drive"]
            )
        )
        value = -1j * (h @ rho - rho @ h)
        for operator, rate in system.dissipators.values():
            squared = operator.conj().T @ operator
            value += rate * (
                operator @ rho @ operator.conj().T - (squared @ rho + rho @ squared) / 2
            )
        return value.ravel()

    reference = solve_ivp(
        derivative,
        (0, 4),
        np.outer(ket, ket.conj()).ravel(),
        t_eval=times,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert reference.success
    result = integrator.evolve(system, request)
    error = max(
        np.linalg.norm(a - b) for a, b in zip(result.states, reference.y.T.reshape(-1, 6, 6))
    )
    assert error < 1e-7
    refined = integrator.evolve(system, replace(request, rtol=1e-10, atol=1e-12))
    refinement = max(np.linalg.norm(a - b) for a, b in zip(result.states, refined.states))
    assert refinement < 1e-7
    assert_physical(result.states)
    record_property("state_error", error)
    record_property("refinement_error", refinement)
    record_property("device", result.device)


@pytest.mark.parametrize("piecewise", [False, True])
def test_rotating_drive_phase_and_lab_state_mapping(integrator, piecewise):
    system = load_system(ROOT / "examples/two-level/system.toml")
    system = replace(system, dissipators={}, dissipator_factors={})
    frequency, phase = 0.2, 0.61
    times = np.linspace(0.23, 5.23, 3)
    ket = np.array([1, 1j]) / np.sqrt(2)
    controls = np.array([0.015, -0.008]) if piecewise else 0.015
    frame = RotatingFrame({system.id: frequency}, rwa=True, excitation_numbers={system.id: [0, 1]})
    request = EvolutionInput(
        ket,
        times,
        drive_amplitudes_ghz={"x_drive": controls},
        carriers={"x_drive": Carrier(frequency, phase)},
        frame=frame,
    )
    rotating_drive = np.array([[0, np.exp(1j * phase)], [np.exp(-1j * phase), 0]]) / 2
    initial_phase = np.diag(np.exp(2j * np.pi * frequency * times[0] * np.arange(2)))
    vector = initial_phase @ ket
    expected = [np.outer(ket, ket.conj())]
    for index, duration in enumerate(np.diff(times)):
        amplitude = controls[index] if piecewise else controls
        vector = expm(-2j * np.pi * duration * amplitude * rotating_drive) @ vector
        lab = np.exp(-2j * np.pi * frequency * times[index + 1] * np.arange(2)) * vector
        expected.append(np.outer(lab, lab.conj()))
    result = integrator.evolve(system, request)
    assert max(np.linalg.norm(a - b) for a, b in zip(result.states, expected)) < 1e-7
    assert_physical(result.states)
    assert result.diagnostics["frame"]["returned_state_basis"] == "lab"


def test_factorized_composite_rwa_matches_direct_numeric_projection(integrator):
    system = load_system(ROOT / "examples/readout/system.toml")
    frequency = 7.002
    numbers = (np.arange(5)[:, None] + np.arange(6)[None, :]).ravel()
    ket = np.eye(30)[0] + 1j * np.eye(30)[6]
    ket = ket / np.sqrt(2)
    times = np.linspace(0, 0.2, 3)
    request = EvolutionInput(
        ket,
        times,
        drive_amplitudes_ghz={"readout": 0.002},
        carriers={"readout": Carrier(frequency)},
        frame=RotatingFrame({"q": frequency, "r": frequency}, rwa=True),
    )
    mask = numbers[:, None] == numbers[None, :]
    projected = np.where(mask, system.hamiltonian_ghz, 0) - np.diag(frequency * numbers)
    prepared = prepare_dynamics(system, normalize_input(system, request))
    np.testing.assert_allclose(matrix(prepared.hamiltonian), projected, rtol=0, atol=1e-13)
    assert all(len(term.factors) == 2 for term in prepared.hamiltonian)
    result = integrator.evolve(system, request)
    reference = QuTiPIntegrator().evolve(system, request)
    assert max(np.linalg.norm(a - b) for a, b in zip(result.states, reference.states)) < 1e-7
    assert_physical(result.states)


@pytest.mark.parametrize("phase", [0.0, 0.47, 1.2])
def test_removed_drive_norm_counts_only_dropped_sectors(phase):
    system = load_system(ROOT / "examples/two-level/system.toml")
    request = EvolutionInput(
        np.array([1, 0]),
        np.array([0.0, 1.0]),
        carriers={"x_drive": Carrier(0.2, phase)},
        frame=RotatingFrame({system.id: 0.2}, rwa=True, excitation_numbers={system.id: [0, 1]}),
    )
    prepared = prepare_dynamics(system, normalize_input(system, request))
    assert prepared.diagnostics["frame"]["removed_drive_frobenius"]["x_drive"] == 0
    mixed = system.drives["x_drive"] + np.diag([1.0, -1.0])
    system = replace(system, drives={"x_drive": mixed}, drive_factors={"x_drive": (mixed,)})
    prepared = prepare_dynamics(system, normalize_input(system, request))
    np.testing.assert_allclose(
        prepared.diagnostics["frame"]["removed_drive_frobenius"]["x_drive"], np.sqrt(2)
    )


def test_frame_and_carrier_diagnostics():
    system = load_system(ROOT / "examples/two-level/system.toml")
    request = EvolutionInput(np.array([1, 0]), np.array([0.0, 1.0]))
    invalid = [
        (replace(request, carriers={"missing": Carrier(0.2)}), "unknown carrier"),
        (replace(request, carriers={"x_drive": Carrier(0)}), "frequency"),
        (replace(request, frame=RotatingFrame({system.id: 0.2})), "rwa=True"),
        (replace(request, frame=RotatingFrame({"missing": 0.2}, rwa=True)), "subsystems"),
        (replace(request, frame=RotatingFrame({system.id: 0.2}, rwa=True)), "excitation_numbers"),
        (
            replace(
                request,
                frame=RotatingFrame(
                    {system.id: 0.2}, rwa=True, excitation_numbers={system.id: [0]}
                ),
            ),
            "integers",
        ),
    ]
    for bad, message in invalid:
        with pytest.raises(EvolutionInputError, match=message):
            QuTiPIntegrator().evolve(system, bad)
    overflow = replace(
        request,
        frame=RotatingFrame({system.id: 1e308}, rwa=True, excitation_numbers={system.id: [0, 2]}),
    )
    with (
        np.errstate(over="ignore"),
        pytest.raises(EvolutionInputError, match="total frame frequencies"),
    ):
        QuTiPIntegrator().evolve(system, overflow)
    mixed = replace(
        system,
        dissipators={"mixed": (system.drives["x_drive"], 0.01)},
        dissipator_factors={"mixed": system.drive_factors["x_drive"]},
    )
    with pytest.raises(EvolutionInputError, match="one excitation-frequency shift"):
        QuTiPIntegrator().evolve(
            mixed,
            replace(
                request,
                frame=RotatingFrame(
                    {system.id: 0.2}, rwa=True, excitation_numbers={system.id: [0, 1]}
                ),
            ),
        )


@pytest.mark.parametrize("frame_kind", ["lab", "rotating"])
@pytest.mark.parametrize("piecewise", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_dynamiqs_carrier_gradient(frame_kind, piecewise, device, record_property):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

    jax.config.update("jax_enable_x64", True)
    system = load_system(ROOT / "examples/two-level/system.toml")
    request = EvolutionInput(
        np.array([1, 0]),
        np.linspace(0, 3, 3),
        observables={"population": np.diag([0, 1])},
        carriers={"x_drive": Carrier(0.2, 0.3)},
        frame=RotatingFrame({system.id: 0.2}, rwa=True, excitation_numbers={system.id: [0, 1]})
        if frame_kind == "rotating"
        else None,
    )
    engine = DynamiqsIntegrator(device)

    def objective(amplitude):
        return (
            engine.evolve_jax(system, request, drive_amplitudes_ghz={"x_drive": amplitude})
            .expectations["population"][-1]
            .real
        )

    control = jnp.array([0.02, 0.03]) if piecewise else jnp.array(0.02)
    gradient = np.asarray(jax.grad(objective)(control))
    assert np.all(abs(gradient) > 1e-3)
    for epsilon in (1e-3, 1e-4):
        finite = []
        for index in range(control.size):
            shift = np.zeros(control.shape)
            shift.flat[index] = epsilon
            values = [
                QuTiPIntegrator()
                .evolve(
                    system,
                    replace(
                        request,
                        drive_amplitudes_ghz={"x_drive": np.asarray(control) + sign * shift},
                    ),
                )
                .expectations["population"][-1]
                .real
                for sign in (1, -1)
            ]
            finite.append((values[0] - values[1]) / (2 * epsilon))
        np.testing.assert_allclose(
            gradient,
            np.asarray(finite).reshape(control.shape),
            rtol=1e-3 if epsilon == 1e-3 else 1e-4,
            atol=1e-6,
        )
    record_property(
        "gradient_fd_error",
        float(np.max(abs(gradient - np.asarray(finite).reshape(control.shape)))),
    )
    record_property("device", device)
