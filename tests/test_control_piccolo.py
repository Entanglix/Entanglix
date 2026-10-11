"""Opt-in real Julia solves: argument forwarding, bounds, and process-group stops."""

import os
import shutil
import signal
import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from test_control import control_files as control_files

from entanglix.protocols.control import control, load_control
from entanglix.runner import _cpu_job_context, read_run

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.skipif(
        os.environ.get("ENTANGLIX_TEST_JULIA") != "1",
        reason="set ENTANGLIX_TEST_JULIA=1 for pinned Julia workstation tests",
    ),
]


@pytest.fixture
def piccolo_input(tmp_path):
    assert shutil.which("julia"), "ENTANGLIX_TEST_JULIA=1 requires Julia on PATH"
    shutil.copyfile(ROOT / "examples/transmon-control/system.toml", tmp_path / "system.toml")
    path = tmp_path / "control.toml"
    path.write_bytes((ROOT / "examples/transmon-control/piccolo.toml").read_bytes())
    return path, tomllib.loads(path.read_text())


@pytest.mark.parametrize("options", [{"max_iter": 1, "tol": 1}, {"max_iter": 100, "tol": 1e6}])
def test_piccolo_cap_and_tolerance_are_arguments(
    piccolo_input, runner_repository, tmp_path, options
):
    path, definition = piccolo_input
    if options["max_iter"] == 1:
        scale = float(1 / np.sqrt(2))
        definition["goal"].pop("gate")
        definition["goal"]["matrix"] = [[scale, [0, scale]], [scale, [0, -scale]]]
        path.write_text(tomli_w.dumps(definition))
    original_digest = load_control(path).digest
    definition["solver"]["options"] = options
    path.write_text(tomli_w.dumps(definition))
    result = control(path, repository=runner_repository, runs_root=tmp_path / "runs")
    record = read_run(Path(result["path"]))
    assert record["status"] == "completed", record["log_tail"]
    assert record["result"]["problem_sha256"] == original_digest
    trace = tomllib.loads((Path(result["path"]) / "artifacts/solver-trace.toml").read_text())
    assert trace["declared_options"] == trace["effective_options"] == options
    iterations = record["result"]["termination"]["iterations"]
    if options["max_iter"] == 1:
        assert iterations == 1
        assert not record["result"]["termination"]["converged"]
    else:
        assert iterations < 100
    assert type(trace["effective_options"]["tol"]) is float
    assert record["manifest"]["environment"]["solver_runtime"]["julia_version"]
    assert record["result"]["artifacts"]["native_pulse"]["path"].endswith(".jld2")


def test_piccolo_active_derivative_bound_and_neutral_knots(
    control_files, runner_repository, tmp_path
):
    path, definition = control_files
    definition["solver"] = {"backend": "piccolo", "options": {"max_iter": 100, "tol": 1e-8}}
    definition["pulse"].update(T=10, init="random")
    definition["problem"].update(du_bound=0.01, ddu_bound=0.003)
    path.write_text(tomli_w.dumps(definition))
    result = control(path, repository=runner_repository, runs_root=tmp_path / "runs")
    record = read_run(Path(result["path"]))
    assert record["status"] == "completed", record["log_tail"]
    with np.load(Path(result["path"]) / "artifacts/pulse.npz") as pulse:
        t, u = pulse["times_ns"], pulse["amplitudes_ghz"]
        assert u.shape == (2, 5)
        assert t[-1] == pytest.approx(10, abs=1e-6)
        np.testing.assert_allclose(np.diff(t), np.diff(t)[0], atol=1e-9)
        assert np.max(abs(u)) <= 0.1 + 1e-7
        dt = np.diff(t)[0]
        assert np.max(abs(np.diff(u, axis=1) / dt)) <= 0.01 + 1e-7
        assert np.max(abs(np.diff(u, n=2, axis=1) / dt**2)) == pytest.approx(0.003, abs=1e-7)
        np.testing.assert_allclose(u[:, [0, -1]], 0, atol=1e-9)


def test_piccolo_stop_during_iterations_leaves_no_group(piccolo_input, runner_repository, tmp_path):
    path, definition = piccolo_input
    definition["solver"]["options"] = {"max_iter": 10000, "tol": 1e-14}
    path.write_text(tomli_w.dumps(definition))
    command, environment, _ = _cpu_job_context(runner_repository, "entanglix")
    runs = tmp_path / "runs"
    supervisor = subprocess.Popen(
        [*command, "control", str(path), "--runs-root", str(runs)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            manifests = list(runs.glob("*/*/run.toml"))
            if manifests:
                run = manifests[0].parent
                if (run / "run.log").exists() and "AMICODE_ITER " in (run / "run.log").read_text():
                    break
            assert supervisor.poll() is None, supervisor.communicate()
            time.sleep(0.01)
        else:
            pytest.fail("Piccolo did not reach its iteration callback")
        supervisor.send_signal(signal.SIGTERM)
        stdout, stderr = supervisor.communicate(timeout=15)
        assert supervisor.returncode == 130, (stdout, stderr)
        record = read_run(run)
        assert record["status"] == "aborted"
        with pytest.raises(ProcessLookupError):
            os.killpg(record["start"]["process_group"], 0)
    finally:
        if supervisor.poll() is None:
            supervisor.send_signal(signal.SIGTERM)
            supervisor.communicate(timeout=15)
