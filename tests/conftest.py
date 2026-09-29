"""GPU test selection stays usable on CPU; required GPU runs cannot silently skip."""

import subprocess

import pytest


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
