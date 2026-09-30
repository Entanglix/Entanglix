"""Run-record integrity, frozen reproduction inputs, and CPU-safe provenance."""

import hashlib
import os
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from entanglix.runner import (
    CodeSnapshot,
    RunCreationError,
    capture_code,
    capture_environment,
    create_run,
    publish_toml,
    record_finish,
    record_reconciliation,
    record_start,
)


def read(path):
    return tomllib.loads(path.read_text())


@pytest.fixture
def run_kwargs(tmp_path):
    system = tmp_path / "system.toml"
    system.write_text('id = "fixture"\n')
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    return dict(
        runs_root=tmp_path / "research_runs",
        system_id="fixture",
        protocol="script",
        inputs={"system.toml": system},
        environment=capture_environment(lock),
        code=CodeSnapshot("abc123", b"", {}),
        engine="qutip",
        device="cpu",
        needs={"gradients": False},
    )


def test_frozen_inputs_and_manifest_first(run_kwargs, monkeypatch):
    import entanglix.runner as runner

    publish = runner.publish_toml
    seen = []

    def inspect(path, record, **kwargs):
        seen.append(path.name)
        assert not list(path.parent.iterdir())
        publish(path, record, **kwargs)

    monkeypatch.setattr(runner, "publish_toml", inspect)
    owner = create_run(**run_kwargs, parent_run_ids=("previous",))
    run = owner.path
    manifest = read(run / "run.toml")
    assert seen == ["run.toml"]
    assert manifest["run_id"] == run.name
    assert manifest["parent_run_ids"] == ["previous"]
    assert manifest["env_hash"] == manifest["environment"]["env_hash"]
    original = (run / "spec/system.toml").read_bytes()
    run_kwargs["inputs"]["system.toml"].write_text("changed")
    assert (run / "spec/system.toml").read_bytes() == original
    assert manifest["spec_sha256"]["system.toml"] == hashlib.sha256(original).hexdigest()
    assert (run / "run.toml").stat().st_mode & 0o222 == 0
    assert not (run / "FINISHED").exists()


def test_concurrent_publication_and_attempts(run_kwargs, tmp_path):
    target = tmp_path / "event.toml"

    def publish(value):
        try:
            publish_toml(target, {"value": value})
            return value
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        winners = [value for value in pool.map(publish, range(4)) if value is not None]
        runs = list(pool.map(lambda _: create_run(**run_kwargs), range(4)))
    assert winners == [read(target)["value"]]
    assert len({owner.path for owner in runs}) == 4
    assert not list(tmp_path.glob(".event.toml-*"))


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("failed", 64), ("aborted", 130)])
def test_terminal_records(run_kwargs, status, exit_code):
    owner = create_run(**run_kwargs)
    run = owner.path
    record_start(owner, process_group=123, gpu_devices=("GPU-fixture",))
    with pytest.raises(FileExistsError):
        record_start(owner, process_group=456)
    record_finish(owner, status=status, exit_code=exit_code, wall_seconds=2, gpu_seconds=2)
    terminal = read(run / "FINISHED")
    assert terminal["status"] == status
    assert terminal["exit_code"] == exit_code
    assert terminal["usage"] == {"wall_seconds": 2, "gpu_seconds": 2, "uncertain": False}
    with pytest.raises(FileExistsError):
        record_finish(owner, status=status, exit_code=exit_code, wall_seconds=2, gpu_seconds=2)
    owner.close()
    with pytest.raises(ValueError, match="finished"):
        record_reconciliation(run, last_observed_state="exited")


def test_reconciliation_cannot_fabricate_completion(run_kwargs):
    owner = create_run(**run_kwargs)
    run = owner.path
    record_start(owner, process_group=2_000_000_000)
    publish_toml(run / "result.toml", {"metric": 0.5}, replace=True)
    owner.close()
    record_reconciliation(run, last_observed_state="exited")
    record = read(run / "runner/reconciled.toml")
    assert set(record) == {
        "record_version",
        "observed_at",
        "last_observed_state",
        "started_at",
        "process_group",
        "gpu_devices",
        "usage",
    }
    assert record["usage"] == {"uncertain": True}
    assert record["process_group"] == 2_000_000_000
    assert not (run / "FINISHED").exists()
    with pytest.raises(ValueError, match="closed"):
        record_finish(owner, status="completed", exit_code=0, wall_seconds=2, gpu_seconds=0)


def test_invalid_inputs_do_not_create_attempt(run_kwargs):
    for override in (
        {"system_id": "../escape"},
        {"inputs": {}},
        {"inputs": {"system.toml": Path("/missing-input")}},
        {
            "inputs": {
                "system.toml": run_kwargs["inputs"]["system.toml"],
                "../outside": run_kwargs["inputs"]["system.toml"],
            }
        },
    ):
        with pytest.raises((ValueError, FileNotFoundError)):
            create_run(**(run_kwargs | override))
    assert not run_kwargs["runs_root"].exists()


def test_result_replacement_is_atomic_and_serialization_failure_keeps_previous(tmp_path):
    path = tmp_path / "result.toml"
    publish_toml(path, {"metric": 1}, replace=True)
    with pytest.raises(TypeError):
        publish_toml(path, {"metric": None}, replace=True)
    assert read(path) == {"metric": 1}
    publish_toml(path, {"metric": 2}, replace=True)
    assert read(path) == {"metric": 2}


def test_environment_fingerprint_and_cpu_only_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("entanglix.runner.platform.system", lambda: "Linux")
    monkeypatch.setattr("entanglix.runner.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("entanglix.runner.platform.libc_ver", lambda: ("glibc", "2.41"))
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    a = capture_environment(lock, extras=("cuquantum", "differentiable-cuda12"))
    assert a == capture_environment(lock, extras=("differentiable-cuda12", "cuquantum"))
    assert a["gpu"]["driver"] == "unobserved"
    assert a["platform"] == {"system": "Linux", "machine": "x86_64", "libc": ["glibc", "2.41"]}
    monkeypatch.setattr("entanglix.runner.platform.release", lambda: "different-kernel")
    assert a == capture_environment(lock, extras=("cuquantum", "differentiable-cuda12"))
    with pytest.raises(ValueError, match="unknown GPU"):
        capture_environment(lock, gpu={"drvier": "595"})
    assert a["env_hash"] != capture_environment(lock)["env_hash"]
    assert a["env_hash"] != capture_environment(lock, gpu={"driver": "595"})["env_hash"]
    lock.write_text("version = 2\n")
    assert a["env_hash"] != capture_environment(lock)["env_hash"]
    script = (
        "import sys; from pathlib import Path; "
        "from entanglix.runner import capture_environment; "
        f"capture_environment(Path({str(lock)!r})); "
        "assert not {'jax', 'cupy', 'cuquantum', 'qutip', 'scqubits'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def repository(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args], check=True, capture_output=True
        )

    git("init")
    (repository / "tracked.bin").write_bytes(b"\0original")
    git("add", "tracked.bin")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    return repository, git


@pytest.mark.parametrize("config", [None, ("color.diff", "always"), ("diff.noprefix", "true")])
def test_dirty_code_is_reconstructable(run_kwargs, repository, monkeypatch, config):
    repository, git = repository
    if config is not None:
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", config[0])
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", config[1])
    assert not capture_code(repository).provenance()["dirty"]
    (repository / "tracked.bin").write_bytes(b"\0staged")
    git("add", "tracked.bin")
    (repository / "tracked.bin").write_bytes(b"\0unstaged")
    (repository / "new.py").write_text("value = 42\n")
    snapshot = capture_code(repository)
    assert b"\x1b[" not in snapshot.patch
    owner = create_run(**(run_kwargs | {"code": snapshot}))
    run = owner.path
    assert read(run / "run.toml")["code"]["dirty"]
    git("restore", "--source=HEAD", "--staged", "--worktree", "tracked.bin")
    git("apply", str(run / "spec/code/changes.patch"))
    assert (repository / "tracked.bin").read_bytes() == b"\0unstaged"
    assert (run / "spec/code/untracked/new.py").read_text() == "value = 42\n"


@pytest.mark.parametrize("names", [("a", "a/b"), ("system.toml", "./system.toml"), ("bad\udcff",)])
def test_path_conflicts_and_non_utf8_names_reject_before_creation(run_kwargs, names):
    source = run_kwargs["inputs"]["system.toml"]
    inputs = {"system.toml": source, **{name: source for name in names}}
    with pytest.raises(ValueError):
        create_run(**(run_kwargs | {"inputs": inputs}))
    assert not run_kwargs["runs_root"].exists()


def test_non_utf8_record_rejects_before_creation(run_kwargs):
    with pytest.raises(UnicodeEncodeError):
        create_run(**(run_kwargs | {"protocol": "bad\udcff"}))
    assert not run_kwargs["runs_root"].exists()


def test_copy_failure_records_failed_attempt_and_exposes_path(run_kwargs, monkeypatch):
    write = Path.write_bytes

    def fail_copy(path, data):
        if path.name == "system.toml" and path.parent.name == "spec":
            raise OSError("injected copy failure")
        return write(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_copy)
    with pytest.raises(RunCreationError, match="injected copy failure") as error:
        create_run(**run_kwargs)
    run = error.value.run
    assert read(run / "FINISHED")["status"] == "failed"
    assert read(run / "FINISHED")["exit_code"] == 64
    assert not (run / "runner/started.toml").exists()


def test_creation_failure_keeps_path_when_terminal_write_also_fails(run_kwargs, monkeypatch):
    monkeypatch.setattr(
        Path, "write_bytes", lambda *_: (_ for _ in ()).throw(OSError("copy failure"))
    )
    monkeypatch.setattr(
        "entanglix.runner.record_finish",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("storage failure")),
    )
    with pytest.raises(RunCreationError) as error:
        create_run(**run_kwargs)
    run = error.value.run
    assert (run / "run.toml").exists()
    assert not (run / "FINISHED").exists()
    assert "storage failure" in error.value.__notes__[0]
    record_reconciliation(run, last_observed_state="not_started")
    assert read(run / "runner/reconciled.toml")["usage"]["uncertain"]


def test_pre_start_reconciliation(run_kwargs):
    owner = create_run(**run_kwargs)
    run = owner.path
    with pytest.raises(ValueError, match="still owned"):
        record_reconciliation(run, last_observed_state="not_started")
    owner.close()
    record_reconciliation(run, last_observed_state="not_started")
    record = read(run / "runner/reconciled.toml")
    assert "started_at" not in record and "process_group" not in record
    assert not (run / "FINISHED").exists()
    with pytest.raises(ValueError, match="closed"):
        record_start(owner, process_group=123)


@pytest.mark.parametrize(
    "values",
    [
        {"status": "completed", "exit_code": 1},
        {"status": "aborted", "exit_code": 0},
        {"status": "failed", "exit_code": 0},
        {"status": "unknown"},
        {"exit_code": -9},
        {"exit_code": 256},
        {"wall_seconds": -1},
        {"wall_seconds": float("nan")},
        {"gpu_seconds": float("inf")},
    ],
)
def test_invalid_terminal_fields_leave_no_terminal(run_kwargs, values):
    owner = create_run(**run_kwargs)
    run = owner.path
    record_start(owner, process_group=123)
    fields = dict(status="completed", exit_code=0, wall_seconds=1, gpu_seconds=0)
    with pytest.raises(ValueError):
        record_finish(owner, **(fields | values))
    assert not (run / "FINISHED").exists()


@pytest.mark.parametrize("status,code", [("completed", 0), ("failed", 137), ("aborted", 130)])
def test_job_outcomes_need_start_record(run_kwargs, status, code):
    owner = create_run(**run_kwargs)
    with pytest.raises(ValueError, match="start record"):
        record_finish(owner, status=status, exit_code=code, wall_seconds=0, gpu_seconds=0)


def test_signal_failure_records_shell_exit_code(run_kwargs):
    owner = create_run(**run_kwargs)
    run = owner.path
    record_start(owner, process_group=123)
    record_finish(owner, status="failed", exit_code=137, wall_seconds=1, gpu_seconds=0)
    assert read(run / "FINISHED")["exit_code"] == 137


@pytest.mark.parametrize("group", [0, -1])
def test_invalid_process_group(run_kwargs, group):
    owner = create_run(**run_kwargs)
    run = owner.path
    with pytest.raises(ValueError):
        record_start(owner, process_group=group)
    assert not (run / "runner/started.toml").exists()


def test_reserved_code_path(run_kwargs):
    inputs = run_kwargs["inputs"] | {"code/file.py": run_kwargs["inputs"]["system.toml"]}
    with pytest.raises(ValueError, match="reserved"):
        create_run(**(run_kwargs | {"inputs": inputs}))
    assert not run_kwargs["runs_root"].exists()


def test_untracked_symlink_is_rejected(repository):
    root, _ = repository
    (root / "link").symlink_to(root / "tracked.bin")
    with pytest.raises(ValueError, match="untracked symlink"):
        capture_code(root)


def test_snapshot_size_limit(repository):
    root, _ = repository
    (root / "large.bin").write_bytes(b"x" * 32)
    with pytest.raises(ValueError, match="max_snapshot_bytes"):
        capture_code(root, max_snapshot_bytes=16)
    assert capture_code(root, max_snapshot_bytes=32).untracked["large.bin"] == b"x" * 32


@pytest.fixture
def child_process():
    """A real process that signals readiness and exits when its input is closed."""
    children = []

    def start(*, descriptors=()):
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
            pass_fds=descriptors,
        )
        children.append(child)
        assert child.stdout.readline().strip() == "ready"
        return child

    yield start
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()


def test_owner_keeps_attempt_active_after_job_exit(run_kwargs, child_process):
    owner = create_run(**run_kwargs)
    child = child_process()
    record_start(owner, process_group=child.pid)
    child.stdin.close()
    assert child.wait(timeout=5) == 0
    with pytest.raises(ValueError, match="still owned"):
        record_reconciliation(owner.path, last_observed_state="exited")
    record_finish(owner, status="completed", exit_code=0, wall_seconds=1, gpu_seconds=0)
    owner.close()
    assert read(owner.path / "FINISHED")["status"] == "completed"


def test_inherited_attempt_lock_outlives_runner_handle(run_kwargs, child_process):
    owner = create_run(**run_kwargs)
    child = child_process(descriptors=(owner.fileno(),))
    record_start(owner, process_group=child.pid)
    owner.close()
    with pytest.raises(ValueError, match="still owned"):
        record_reconciliation(owner.path, last_observed_state="exited")
    child.stdin.close()
    child.wait(timeout=5)
    record_reconciliation(owner.path, last_observed_state="exited")
    assert (owner.path / "runner/reconciled.toml").exists()
    assert not (owner.path / "FINISHED").exists()


def test_live_group_blocks_reconciliation_even_when_lock_is_free(run_kwargs, child_process):
    owner = create_run(**run_kwargs)
    child = child_process()
    record_start(owner, process_group=child.pid)
    owner.close()
    with pytest.raises(ValueError, match="process group is still alive"):
        record_reconciliation(owner.path, last_observed_state="exited")
    assert not (owner.path / "runner/reconciled.toml").exists()
    child.stdin.close()
    child.wait(timeout=5)
    record_reconciliation(owner.path, last_observed_state="exited")
    assert read(owner.path / "runner/reconciled.toml")["process_group"] == child.pid


def test_unknown_group_state_blocks_reconciliation(run_kwargs, monkeypatch):
    owner = create_run(**run_kwargs)
    record_start(owner, process_group=123)
    owner.close()

    def deny(*_):
        raise PermissionError("not permitted")

    monkeypatch.setattr(os, "killpg", deny)
    with pytest.raises(ValueError, match="cannot confirm"):
        record_reconciliation(owner.path, last_observed_state="unknown")
    assert not (owner.path / "runner/reconciled.toml").exists()


def test_ownership_precedes_manifest_publication(run_kwargs, monkeypatch):
    import entanglix.runner as runner

    publish = runner.publish_toml

    def inspect(path, record, **kwargs):
        if path.name == "run.toml":
            with pytest.raises(ValueError, match="still owned"):
                record_reconciliation(path.parent, last_observed_state="not_started")
        publish(path, record, **kwargs)

    monkeypatch.setattr(runner, "publish_toml", inspect)
    create_run(**run_kwargs)


def test_pre_manifest_failure_removes_unadmitted_directory(run_kwargs, monkeypatch):
    def fail(*_a, **_k):
        raise OSError("manifest failure")

    monkeypatch.setattr("entanglix.runner.publish_toml", fail)
    with pytest.raises(RunCreationError) as error:
        create_run(**run_kwargs)
    assert not error.value.run.exists()
    assert not list((run_kwargs["runs_root"] / "fixture").iterdir())
