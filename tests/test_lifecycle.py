"""Local cancellation and inherited GPU ownership, using CPU processes and real signals."""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from entanglix.runner import (
    CodeSnapshot,
    _execute_job,
    abort_run,
    acquire_gpu,
    capture_environment,
    create_run,
    publish_toml,
    read_run,
    record_finish,
    record_reconciliation,
    record_start,
    rerun_script,
    run_script,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "GPU-test-device"


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    pytest.fail("process did not reach its expected state")


@pytest.fixture
def cpu_job(tmp_path, runner_repository):
    script = tmp_path / "job.py"
    runs = tmp_path / "runs"
    children = []

    def launch(source, *options):
        script.write_text(source)
        child = subprocess.Popen(
            [
                "entanglix",
                "run-script",
                str(script),
                "--system",
                str(ROOT / "examples/two-level/system.toml"),
                "--runs-root",
                str(runs),
                *options,
            ],
            env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(child)
        return child

    yield launch, runs
    for child in children:
        attempts = list(runs.glob("*/*/run.toml"))
        for manifest in attempts:
            run = manifest.parent
            try:
                record = read_run(run)
                if record["status"] == "active" and "start" in record:
                    os.killpg(record["start"]["process_group"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


def active_attempt(runs):
    for path in runs.glob("*/*/artifacts/ready"):
        return path.parent.parent


RESISTANT_LOOP = """import os, signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])
Path('descendant.pid').write_text(str(child.pid))
Path('ready').write_text('ready')
while True:
    time.sleep(0.01)
"""


LOOP = """import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path('descendant.pid').write_text(str(child.pid))
Path('ready').write_text('ready')
while True:
    time.sleep(0.01)
"""


def test_timeout_kills_whole_group_and_records_abort(cpu_job):
    launch, runs = cpu_job
    supervisor = launch(RESISTANT_LOOP, "--timeout", "1.5")
    run = wait_for(lambda: active_attempt(runs))
    stdout, stderr = supervisor.communicate(timeout=10)
    assert supervisor.returncode == 130, stderr
    assert json.loads(stdout)["path"] == str(run)
    record = read_run(run)
    assert record["status"] == "aborted"
    assert record["finished"]["usage"]["wall_seconds"] >= 1.5
    assert "timeout; terminating" in record["log_tail"]
    with pytest.raises(ProcessLookupError):
        os.killpg(record["start"]["process_group"], 0)


def test_abort_targets_only_named_attempt(cpu_job):
    launch, runs = cpu_job
    first = launch(LOOP)
    first_run = wait_for(lambda: active_attempt(runs))
    # Keep a separate process group running to guard against broad cancellation.
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        request = subprocess.run(
            ["entanglix", "abort-run", str(first_run)], capture_output=True, text=True, timeout=5
        )
        assert request.returncode == 0, request.stderr
        response = json.loads(request.stdout)
        stdout, stderr = first.communicate(timeout=5)
        assert response["abort_requested"]
        assert first.returncode == 130, stderr
        assert json.loads(stdout)["status"] == "aborted"
        assert read_run(first_run)["abort_request"]["requested_at"]
        assert other.poll() is None
    finally:
        other.kill()
        other.wait(timeout=5)


def test_abort_of_responsive_job_is_aborted_not_signal_failure(cpu_job):
    launch, runs = cpu_job
    supervisor = launch("""import time
from pathlib import Path
Path('ready').write_text('ready')
time.sleep(30)
""")
    run = wait_for(lambda: active_attempt(runs))
    abort_run(run)
    _, stderr = supervisor.communicate(timeout=5)
    assert supervisor.returncode == 130, stderr
    assert read_run(run)["status"] == "aborted"


def test_supervisor_completes_abort_after_requester_loss(cpu_job):
    launch, runs = cpu_job
    supervisor = launch(LOOP)
    run = wait_for(lambda: active_attempt(runs))
    # The requesting client died after publishing, before delivering any signal.
    publish_toml(run / "runner/abort.toml", {"requested_at": datetime.now(UTC)})
    _, stderr = supervisor.communicate(timeout=5)
    assert supervisor.returncode == 130, stderr
    record = read_run(run)
    assert record["status"] == "aborted"
    assert "abort; terminating" in record["log_tail"]
    with pytest.raises(ProcessLookupError):
        os.killpg(record["start"]["process_group"], 0)


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_interrupting_supervisor_aborts_job_and_descendants(cpu_job, stop_signal):
    launch, runs = cpu_job
    supervisor = launch(LOOP)
    run = wait_for(lambda: active_attempt(runs))
    supervisor.send_signal(stop_signal)
    _, stderr = supervisor.communicate(timeout=5)
    assert supervisor.returncode == 130, stderr
    record = read_run(run)
    assert record["status"] == "aborted"
    assert "interrupt; terminating" in record["log_tail"]
    with pytest.raises(ProcessLookupError):
        os.killpg(record["start"]["process_group"], 0)


def test_runner_loss_keeps_job_active_until_abort_and_reconciliation(cpu_job):
    launch, runs = cpu_job
    supervisor = launch(LOOP)
    run = wait_for(lambda: active_attempt(runs))
    supervisor.kill()
    supervisor.communicate(timeout=5)
    assert read_run(run)["status"] == "active"
    with pytest.raises(ValueError, match="still owned"):
        record_reconciliation(run, last_observed_state="runner died")
    abort_run(run)
    wait_for(lambda: read_run(run)["status"] == "interrupted")
    record_reconciliation(run, last_observed_state="runner died; job group terminated")
    assert read_run(run)["reconciliation"]["usage"]["uncertain"]
    assert not (run / "FINISHED").exists()


def test_rerun_uses_frozen_inputs_and_checks_digests(tmp_path, runner_repository):
    script = tmp_path / "job.py"
    data = tmp_path / "input.txt"
    data.write_text("retained")
    script.write_text("""import os
from pathlib import Path
from entanglix.runner import publish_toml
run = Path(os.environ['ENTANGLIX_RUN_DIR'])
publish_toml(run / 'result.toml', {'engine': 'numpy', 'metrics': {'value': (run / 'spec/input.txt').read_text()}}, replace=True)
""")
    first = run_script(
        ROOT / "examples/two-level/system.toml",
        script,
        inputs={"input.txt": data},
        runs_root=tmp_path / "runs",
        repository=runner_repository,
    )
    run = Path(first["path"])
    script.write_text("raise RuntimeError('changed original')")
    data.write_text("changed")
    response = subprocess.run(
        ["entanglix", "rerun-script", str(run)],
        capture_output=True,
        text=True,
        timeout=15,
        env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
    )
    assert response.returncode == 0, response.stderr
    replay = json.loads(response.stdout)
    record = read_run(Path(replay["path"]))
    assert record["result"]["metrics"]["value"] == "retained"
    assert record["manifest"]["parent_run_ids"] == [run.name]
    assert Path(replay["path"]) != run
    frozen = run / "spec/input.txt"
    frozen.chmod(0o644)
    frozen.write_text("corrupted")
    with pytest.raises(ValueError, match="digest mismatch"):
        rerun_script(run, repository=runner_repository)


@pytest.fixture
def allocation(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    script = tmp_path / "script.py"
    script.write_text("""import time
from pathlib import Path
Path('ready').write_text('ready')
time.sleep(30)
""")
    system = ROOT / "examples/two-level/system.toml"

    def create(root):
        return create_run(
            system_id="fixture",
            protocol="script",
            inputs={"system.toml": system, "script.py": script},
            environment=capture_environment(ROOT / "uv.lock"),
            code=CodeSnapshot("test", b"", {}),
            engine="authored-script",
            device="cuda",
            needs={"device": "cuda"},
            runs_root=root,
        )

    return create


def test_gpu_lock_survives_runner_death_until_reconciled(allocation, tmp_path):
    owner = allocation(tmp_path / "first-root")
    run = owner.path
    # A real supervisor uses the local executor. No CUDA packages/device are needed.
    source = """import json, os
from pathlib import Path
from entanglix.runner import RunHandle, _execute_job, acquire_gpu
run = Path(os.environ['TEST_RUN'])
owner = RunHandle(run, int(os.environ['TEST_OWNER_FD']))
with acquire_gpu('GPU-test-device') as gpu:
    gpu.bind_run(owner)
    _execute_job(owner, [os.environ['TEST_PYTHON'], '-m', 'entanglix._script_job'], os.environ.copy(), timeout_seconds=None, gpu=gpu)
"""
    supervisor = subprocess.Popen(
        [sys.executable, "-c", source],
        pass_fds=(owner.fileno(),),
        env=os.environ
        | {
            "TEST_RUN": str(run),
            "TEST_OWNER_FD": str(owner.fileno()),
            "TEST_PYTHON": sys.executable,
            "ENTANGLIX_RUN_DIR": str(run),
        },
    )
    owner.close()
    second = allocation(tmp_path / "different-root")
    try:
        wait_for(lambda: (run / "artifacts/ready").exists())
        with pytest.raises(ValueError, match="still owned"):
            acquire_gpu(DEVICE)
        supervisor.kill()
        supervisor.wait(timeout=5)
        assert read_run(run)["status"] == "active"
        with pytest.raises(ValueError, match="still owned"):
            acquire_gpu(DEVICE)
        abort_run(run)
        wait_for(lambda: read_run(run)["status"] == "interrupted")
        with pytest.raises(ValueError, match="requires reconciliation"):
            acquire_gpu(DEVICE)
        record_reconciliation(run, last_observed_state="runner killed; group gone")
        with acquire_gpu(DEVICE) as gpu:
            gpu.bind_run(second)
            assert read_run(run)["reconciliation"]["usage"]["uncertain"]
        assert not (run / "FINISHED").exists()
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.wait(timeout=5)
        record = read_run(run)
        if record["status"] == "active" and "start" in record:
            try:
                os.killpg(record["start"]["process_group"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        second.close()


def test_gpu_remains_owned_after_job_exit_until_finish(allocation, tmp_path):
    owner = allocation(tmp_path / "one")
    with owner, acquire_gpu(DEVICE) as gpu:
        gpu.bind_run(owner)
        child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        record_start(owner, process_group=child.pid, gpu_devices=(DEVICE,))
        gpu.bind_process(child.pid)
        child.wait(timeout=5)
        with pytest.raises(ValueError, match="still owned"):
            acquire_gpu(DEVICE)
        with pytest.raises(ValueError, match="still owned"):
            record_reconciliation(owner.path, last_observed_state="job exited; runner paused")
        record_finish(owner, status="completed", exit_code=0, wall_seconds=0.1, gpu_seconds=0.1)
    with allocation(tmp_path / "three") as third, acquire_gpu(DEVICE) as gpu:
        gpu.bind_run(third)


def test_executor_accounts_allocation_and_releases_finished_gpu(allocation, tmp_path):
    owner = allocation(tmp_path / "one")
    script = owner.path / "spec/script.py"
    script.chmod(0o644)
    script.write_text("""import os
from pathlib import Path
from entanglix.runner import publish_toml
publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', {'engine': 'fixture', 'metrics': {'value': 1}}, replace=True)
""")
    with owner, acquire_gpu(DEVICE) as gpu:
        gpu.bind_run(owner)
        result = _execute_job(
            owner,
            [sys.executable, "-m", "entanglix._script_job"],
            os.environ | {"ENTANGLIX_RUN_DIR": str(owner.path)},
            timeout_seconds=None,
            gpu=gpu,
        )
        assert result["status"] == "completed"
        usage = read_run(owner.path)["finished"]["usage"]
        assert usage["gpu_seconds"] >= usage["wall_seconds"] > 0
    assert not gpu.record_path.exists()


def test_failed_spawn_records_allocated_gpu_time_without_job_start(allocation, tmp_path):
    with allocation(tmp_path / "one") as owner, acquire_gpu(DEVICE) as gpu:
        gpu.bind_run(owner)
        result = _execute_job(
            owner,
            [str(tmp_path / "missing-python")],
            os.environ.copy(),
            timeout_seconds=None,
            gpu=gpu,
        )
        record = read_run(owner.path)
        assert result["exit_code"] == 64
        assert record["status"] == "failed" and "start" not in record
        assert record["finished"]["usage"]["wall_seconds"] == 0
        assert record["finished"]["usage"]["gpu_seconds"] > 0
    assert not gpu.record_path.exists()


def test_unstarted_attempt_is_queryable_and_gpu_needs_reconciliation(allocation, tmp_path):
    owner = allocation(tmp_path / "one")
    with acquire_gpu(DEVICE) as gpu:
        gpu.bind_run(owner)
        assert read_run(owner.path)["status"] == "active"
    owner.close()
    assert read_run(owner.path)["status"] == "interrupted"
    with allocation(tmp_path / "two") as second:
        with pytest.raises(ValueError, match="requires reconciliation"):
            acquire_gpu(DEVICE)
        response = subprocess.run(
            ["entanglix", "reconcile-run", str(owner.path), "--state", "runner died before start"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert response.returncode == 0, response.stderr
        assert json.loads(response.stdout)["reconciliation"]["usage"] == {"uncertain": True}
        with acquire_gpu(DEVICE) as gpu:
            gpu.bind_run(second)


def test_unknown_group_liveness_blocks_gpu_reuse(allocation, tmp_path, monkeypatch):
    owner = allocation(tmp_path / "one")
    with acquire_gpu(DEVICE) as gpu:
        gpu.bind_run(owner)
        gpu.bind_process(12345)
    owner.close()

    retained = gpu.record_path.read_bytes()

    def unknown(group, sig):
        assert (group, sig) == (12345, 0)
        raise PermissionError("process state unavailable")

    monkeypatch.setattr("entanglix.runner.os.killpg", unknown)
    with pytest.raises(PermissionError, match="state unavailable"):
        acquire_gpu(DEVICE)
    assert gpu.record_path.read_bytes() == retained


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_rejected_without_attempt(tmp_path, timeout):
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="finite and positive"):
        run_script(
            ROOT / "examples/two-level/system.toml",
            ROOT / "examples/recorded_decay.py",
            runs_root=root,
            timeout_seconds=timeout,
        )
    assert not root.exists()


def test_live_group_blocks_gpu_reuse_even_after_descriptors_close(allocation, tmp_path):
    owner = allocation(tmp_path / "one")
    gpu = acquire_gpu(DEVICE)
    gpu.bind_run(owner)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    record_start(owner, process_group=child.pid, gpu_devices=(DEVICE,))
    gpu.bind_process(child.pid)
    gpu.close()
    owner.close()
    try:
        with pytest.raises(ValueError, match="live recorded process group"):
            acquire_gpu(DEVICE)
    finally:
        child.kill()
        child.wait(timeout=5)
