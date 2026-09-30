"""Independent exponential references and populated dynamics fixtures."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from entanglix.evolution import EvolutionInput
from entanglix.models import ConstructedSystem, load_system

ROOT = Path(__file__).resolve().parents[1] / "examples"


def exact_states(system, request):
    """Column-major Lindblad exponential, without calling any integration adapter."""
    ket = request.initial_state
    state = np.outer(ket, ket.conj()) if ket.ndim == 1 else ket
    identity = np.eye(state.shape[0])
    dissipative = np.zeros((state.size, state.size), dtype=complex)
    for operator, rate in system.dissipators.values():
        squared = operator.conj().T @ operator
        dissipative += rate * (
            np.kron(operator.conj(), operator)
            - 0.5 * np.kron(identity, squared)
            - 0.5 * np.kron(squared.T, identity)
        )
    propagators = {}
    states = [state]
    for index, duration in enumerate(np.diff(request.times_ns)):
        amplitudes = tuple(
            (name, float(value[index]) if np.ndim(value) else float(value))
            for name, value in request.drive_amplitudes_ghz.items()
        )
        key = (float(duration), amplitudes)
        if key not in propagators:
            h = np.array(system.hamiltonian_ghz, copy=True, dtype=complex)
            for name, amplitude in amplitudes:
                h += amplitude * system.drives[name]
            h *= 2 * np.pi
            generator = -1j * (np.kron(identity, h) - np.kron(h.T, identity)) + dissipative
            propagators[key] = expm(duration * generator)
        state = (propagators[key] @ state.ravel(order="F")).reshape(state.shape, order="F")
        states.append(state)
    return tuple(states)


@dataclass(frozen=True)
class DynamicsCase:
    name: str
    system: ConstructedSystem
    request: EvolutionInput
    expected: tuple[np.ndarray, ...]
    closed_expected: tuple[np.ndarray, ...] | None
    analytic_population: np.ndarray | None = None


def dynamics_cases():
    two = load_system(ROOT / "two-level/system.toml")
    oscillator = load_system(ROOT / "lossy-oscillator/system.toml")
    composite = load_system(ROOT / "qubit-resonator/system.toml")
    photon = np.eye(6)[1]
    coupled_photon = np.eye(30)[1]
    coherent = np.array([1, 1j]) / np.sqrt(2)
    population = np.diag([0.0, 1.0])
    specs = [
        (
            "two-level-loss",
            two,
            EvolutionInput(np.array([0, 1]), np.linspace(0, 100, 11), {"population": population}),
            np.exp(-0.01 * np.linspace(0, 100, 11)),
        ),
        (
            "two-level-coherence",
            replace(two, dissipators={}, dissipator_factors={}),
            EvolutionInput(coherent, np.linspace(0, 2, 5), {"lower": two.operators["lower"]}),
            None,
        ),
        (
            "two-level-drive",
            two,
            EvolutionInput(
                np.array([0, 1]),
                np.linspace(0, 10, 5),
                {"population": population},
                {"x_drive": np.array([0.04, 0.08, -0.02, 0.06])},
            ),
            None,
        ),
        (
            "oscillator-loss",
            oscillator,
            EvolutionInput(
                photon, np.linspace(0, 50, 11), {"population": oscillator.operators["number"]}
            ),
            np.exp(-0.01 * np.linspace(0, 50, 11)),
        ),
        (
            "oscillator-drive",
            oscillator,
            EvolutionInput(
                photon,
                np.linspace(0, 10, 5),
                {"population": oscillator.operators["number"]},
                {"position_drive": np.array([0.02, 0.04, -0.02, 0.0])},
            ),
            None,
        ),
        (
            "composite-loss",
            composite,
            EvolutionInput(
                coupled_photon,
                np.linspace(0, 10, 11),
                {"population": composite.operators["r.number"]},
            ),
            None,
        ),
        (
            "composite-drive",
            composite,
            EvolutionInput(
                coupled_photon,
                np.linspace(0, 10, 5),
                {"population": composite.operators["r.number"]},
                {"charge_drive": np.array([0.02, -0.01, 0.03, 0.0])},
            ),
            None,
        ),
    ]
    cases = {}
    for name, system, request, analytic in specs:
        request = replace(request, rtol=1e-9, atol=1e-11, max_steps=200_000)
        closed = replace(system, dissipators={}, dissipator_factors={})
        cases[name] = DynamicsCase(
            name,
            system,
            request,
            exact_states(system, request),
            exact_states(closed, request) if system.dissipators else None,
            analytic,
        )
    return cases


def physical_residuals(states):
    return {
        "trace_error": float(max(abs(np.trace(state) - 1) for state in states)),
        "hermiticity_error": float(max(np.linalg.norm(state - state.conj().T) for state in states)),
        "minimum_eigenvalue": float(
            min(np.linalg.eigvalsh((state + state.conj().T) / 2).min() for state in states)
        ),
    }
