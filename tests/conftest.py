"""GPU test selection stays usable on CPU; required GPU runs cannot silently skip."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def composite_evolution_reference():
    """Exact Lindblad exponential for complex coherences and 10 ns of evolution."""
    import numpy as np
    from scipy.linalg import expm

    from entanglix.evolution import EvolutionInput
    from entanglix.models import load_system

    system = load_system(
        Path(__file__).resolve().parents[1] / "examples/qubit-resonator/system.toml"
    )
    rng = np.random.default_rng(42)
    ket = rng.normal(size=30) + 1j * rng.normal(size=30)
    ket /= np.linalg.norm(ket)
    request = EvolutionInput(ket, np.arange(11, dtype=float))
    h = 2 * np.pi * system.hamiltonian_ghz
    identity = np.eye(30)
    generator = -1j * (np.kron(identity, h) - np.kron(h.T, identity))
    for operator, rate in system.dissipators.values():
        squared = operator.conj().T @ operator
        generator += rate * (
            np.kron(operator.conj(), operator)
            - 0.5 * np.kron(identity, squared)
            - 0.5 * np.kron(squared.T, identity)
        )
    # Column-major vectorization; one exact 1 ns propagator gives every output.
    propagator = expm(generator)
    state = np.outer(ket, ket.conj())
    states = [state]
    for _ in request.times_ns[1:]:
        state = (propagator @ state.ravel(order="F")).reshape((30, 30), order="F")
        states.append(state)
    return system, request, tuple(states)


def gpu_availability() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "No NVIDIA GPU is visible to the test process."
    return bool(result.stdout.strip()), "No NVIDIA GPU is visible to the test process."


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--require-gpu", action="store_true", help="Fail if GPU tests cannot execute.")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if not gpu_items:
        if config.getoption("--require-gpu"):
            raise pytest.UsageError("--require-gpu selected no GPU tests.")
        return
    available, reason = gpu_availability()
    if not available:
        if config.getoption("--require-gpu"):
            raise pytest.UsageError(reason)
        skip = pytest.mark.skip(reason=reason)
        for item in gpu_items:
            item.add_marker(skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if item.config.getoption("--require-gpu") and item.get_closest_marker("gpu") and report.skipped:
        report.outcome = "failed"
        report.longrepr = "A required GPU test skipped instead of executing."
