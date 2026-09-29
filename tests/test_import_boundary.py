"""Repository import policy: reviewed public dependencies and entry-point plugins."""

import ast
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text())
ALLOWED = (
    sys.stdlib_module_names
    | {"entanglix"}
    | set(CONFIG["tool"]["entanglix"]["import-boundary"]["allowed-third-party"])
)


def forbidden_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    imports = []
    dynamic_calls = {"__import__", "importlib.import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
            for alias in node.names:
                if alias.name == "importlib":
                    dynamic_calls.add(f"{alias.asname or alias.name}.import_module")
        elif isinstance(node, ast.ImportFrom):
            if not node.level and node.module:
                imports.append(node.module)
            if node.module == "importlib":
                dynamic_calls.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in dynamic_calls:
            if (
                not node.args
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
            ):
                imports.append("<unreviewed dynamic import>")
            else:
                imports.append(node.args[0].value)
    return [name for name in imports if name.split(".")[0] not in ALLOWED]


def test_repository_import_boundary():
    failures = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"tests", ".venv", ".git", ".uv-cache", "dist", "build"}:
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
