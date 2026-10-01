"""Repository import policy: reviewed public dependencies and entry-point plugins."""

import subprocess
import tomllib
from pathlib import Path

import pytest

from entanglix.imports import forbidden_imports as scan_imports

ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text())
ALLOWED = set(CONFIG["tool"]["entanglix"]["import-boundary"]["allowed-third-party"])


def forbidden_imports(source: str) -> list[str]:
    return scan_imports(source, ALLOWED)


def test_repository_import_boundary():
    failures = []
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py"],
        cwd=ROOT,
        text=True,
    ).split("\0")
    for name in filter(None, names):
        relative = Path(name)
        path = ROOT / relative
        if relative.parts[0] == "tests" or not path.is_file():
            continue
        forbidden = forbidden_imports(path.read_text())
        if forbidden:
            failures.append(f"{relative}: {forbidden}")
    assert not failures, "Direct imports require a reviewed public dependency:\n" + "\n".join(
        failures
    )


@pytest.mark.parametrize(
    "source",
    [
        "import private_example_solver",
        "from kubernetes import client",
        "import importlib as loader; loader.import_module('private_example_solver')",
        "from importlib import import_module as load; load('pyslurm')",
        "__import__('private_example_solver')",
        "import importlib; importlib.import_module(module_name)",
    ],
)
def test_private_scheduler_and_unreviewed_dynamic_imports_are_rejected(source):
    assert forbidden_imports(source)


def test_open_foundation_imports_are_allowed():
    assert forbidden_imports("import argparse; from entanglix.cli import main") == []
