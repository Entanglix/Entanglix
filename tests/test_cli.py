import subprocess
import sys

import pytest

from entanglix import __version__
from entanglix.cli import main


@pytest.mark.parametrize("invocation", [["entanglix"], [sys.executable, "-m", "entanglix"]])
def test_installed_entrypoints(invocation, tmp_path):
    result = subprocess.run(
        [*invocation, "--version"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"entanglix {__version__}"


def test_import_without_loading_gpu_engines(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, entanglix, entanglix.investigations; "
            "assert not {'jax', 'cupy', 'qutip', 'scqubits'} & set(sys.modules)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_help_and_required_protocol_input():
    help_result = subprocess.run(["entanglix", "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "--version" in help_result.stdout
    assert " rerun " in help_result.stdout and "rerun-" not in help_result.stdout
    result = subprocess.run(["entanglix", "control"], capture_output=True, text=True)
    assert result.returncode == 64
    assert result.stdout == ""
    assert "required: protocol" in result.stderr


@pytest.mark.parametrize(
    "command,target",
    [
        (["validate", "unused.toml"], "entanglix.models.load_system"),
        (["capabilities"], "entanglix.runner.list_capabilities"),
    ],
)
def test_cli_allocation_failure_has_diagnostic(command, target, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(target, fail)
    assert main(command) == 64
    captured = capsys.readouterr()
    assert "out of memory: MemoryError" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "command,target,expected",
    [
        (
            ["validate", "unused.toml"],
            "entanglix.models.load_system",
            "entanglix: invalid system: units: Field required\n",
        ),
        (
            ["capabilities"],
            "entanglix.runner.list_capabilities",
            "entanglix: units: Field required\n",
        ),
    ],
)
def test_cli_ordinary_errors_keep_plain_messages(command, target, expected, monkeypatch, capsys):
    from entanglix.models import SpecError

    def fail(*args, **kwargs):
        raise SpecError("units: Field required")

    monkeypatch.setattr(target, fail)
    assert main(command) == 64
    assert capsys.readouterr().err == expected
