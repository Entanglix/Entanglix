import subprocess
import sys

import pytest

from entanglix import __version__


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
            "import sys, entanglix; "
            "assert not {'jax', 'cupy', 'qutip', 'scqubits'} & set(sys.modules)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_help_and_unavailable_protocols():
    help_result = subprocess.run(["entanglix", "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "--version" in help_result.stdout
    result = subprocess.run(["entanglix", "control"], capture_output=True, text=True)
    assert result.returncode == 64
    assert result.stdout == ""
    assert "unrecognized arguments" in result.stderr
