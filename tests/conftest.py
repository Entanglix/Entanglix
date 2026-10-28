"""GPU test selection stays usable on CPU; required GPU runs cannot silently skip."""

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import tomli_w


def _runner_repository(tmp_path, extras):
    """Configure a separate exact CPU runner environment without changing the checkout."""
    root = Path(__file__).resolve().parents[1]
    repository = tmp_path / "checkout"
    repository.mkdir(parents=True)
    shutil.copytree(root / "src", repository / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("uv.lock", ".gitignore", "LICENSE"):
        shutil.copyfile(root / name, repository / name)
    config = tomllib.loads((root / "pyproject.toml").read_text())
    config["tool"].setdefault("entanglix", {})["runner"] = {"extras": extras}
    (repository / "pyproject.toml").write_text(tomli_w.dumps(config))
    for arguments in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True)
    return repository


@pytest.fixture
def runner_repository(tmp_path, monkeypatch):
    """A runner checkout with no extras, independent of developer configuration."""
    monkeypatch.delenv("ENTANGLIX_RUNNER_EXTRAS", raising=False)
    return _runner_repository(tmp_path / "cpu-runner", [])


@pytest.fixture
def differentiable_repository(tmp_path, monkeypatch):
    pytest.importorskip("dynamiqs")
    monkeypatch.delenv("ENTANGLIX_RUNNER_EXTRAS", raising=False)
    return _runner_repository(tmp_path / "differentiable-runner", ["differentiable"])


@pytest.fixture(scope="session")
def acceptance_cases():
    from engine_cases import dynamics_cases

    return dynamics_cases()


@pytest.fixture(scope="session")
def composite_evolution_reference():
    """Exact Lindblad exponential for complex coherences and 10 ns of evolution."""
    import numpy as np
    from engine_cases import exact_states

    from entanglix.evolution import EvolutionInput
    from entanglix.models import load_system

    system = load_system(
        Path(__file__).resolve().parents[1] / "examples/qubit-resonator/system.toml"
    )
    rng = np.random.default_rng(42)
    ket = rng.normal(size=30) + 1j * rng.normal(size=30)
    ket /= np.linalg.norm(ket)
    request = EvolutionInput(ket, np.arange(11, dtype=float))
    return system, request, exact_states(system, request)


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
