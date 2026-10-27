"""Device ownership before CUDA validation, admission, and manifest publication."""

import json
import os
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_sweep import sweep_files

from entanglix.investigations import (
    create_investigation,
    read_investigation,
    record_user_decision,
)
from entanglix.protocols.simulate import compile_simulation, simulate
from entanglix.protocols.sweep import compile_sweep, sweep
from entanglix.runner import (
    _digest,
    _load_record,
    acquire_gpu,
    publish_toml,
    record_finish,
    select_gpu,
)

DEVICE = "GPU-test-device"


@pytest.fixture
def launch_context(tmp_path, runner_repository, monkeypatch):
    """Use real plans and admission, replacing only CUDA observations and execution."""
    launcher = import_module("entanglix.protocols.simulate")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(launcher, "select_gpu", lambda: DEVICE)
    monkeypatch.setattr(launcher, "version", lambda _: "installed")
    monkeypatch.setattr(launcher, "_cpu_job_context", lambda *_: (["gate"], {}, ["cuquantum"]))
    grid, _ = sweep_files(tmp_path, device="cuda", axes=1)
    inputs = ("system.toml", "simulate.toml")
    terms = {
        "format_version": 1,
        "question": "Test admission on the configured device.",
        "assumptions": [],
        "scope": {
            "operations": ["simulate", "sweep"],
            "bases": [
                {
                    "name": operation,
                    "inputs": {name: _digest((tmp_path / name).read_bytes()) for name in inputs},
                    "variations": [{"target": "parameters.f0", "values": [0.2, 0.3]}],
                }
                for operation in ("simulate", "sweep")
            ],
        },
        "evidence_requirements": {
            "requirements": ["Keep device ownership continuous."],
            "checks": {},
        },
        "checkpoint": {"finished_attempts": 1},
        "review_criteria": [],
        "stop_criteria": [],
        "limits": {"job_count": 1},
    }
    runs = tmp_path / "runs"
    investigation = Path(create_investigation(terms, runs_root=runs)["path"])
    record_user_decision(investigation, "start", revision=1)
    calls = []
    real_run = subprocess.run

    def assert_owned():
        with pytest.raises(ValueError, match="still owned"):
            acquire_gpu(DEVICE)

    def gate(command, **kwargs):
        if command[0] != "gate":
            return real_run(command, **kwargs)
        assert_owned()
        assert not list(runs.glob("*/*/run.toml"))
        env = kwargs["env"]
        assert env["CUDA_VISIBLE_DEVICES"] == DEVICE
        assert (env["JAX_PLATFORMS"], env["XLA_PYTHON_CLIENT_PREALLOCATE"]) == ("cuda,cpu", "false")
        assert kwargs["pass_fds"] == (int(env["ENTANGLIX_GPU_LOCK_FD"]),)
        os.fstat(kwargs["pass_fds"][0])
        request = json.loads(kwargs["input"])
        compile = (
            compile_sweep if Path(request["protocol"]).name == "sweep.toml" else compile_simulation
        )
        with monkeypatch.context() as patch:
            patch.setattr(launcher, "check_gpu_environment", lambda *_, **__: {"gpu": {}})
            patch.setenv("CUDA_VISIBLE_DEVICES", DEVICE)
            patch.setenv("ENTANGLIX_GPU_LOCK_FD", env["ENTANGLIX_GPU_LOCK_FD"])
            compiled = compile(
                Path(request["protocol"]),
                Path(request["directory"]),
                ["cuquantum"],
                runner_repository,
                retain_base=request["retain_base"],
            )
        calls.append("gate")
        return SimpleNamespace(returncode=0, stdout=json.dumps(compiled))

    def execute(owner, command, environment, *, gpu, **kwargs):
        assert_owned()
        assert _load_record(gpu.record_path)["run"] == str(owner.path)
        assert environment["ENTANGLIX_GPU_LOCK_FD"] == str(gpu.fileno())
        assert environment["JAX_PLATFORMS"] == "cuda,cpu"
        calls.append("execute")
        record_finish(
            owner,
            status="failed",
            exit_code=64,
            wall_seconds=0,
            gpu_seconds=time.monotonic() - gpu.allocated_at,
        )
        return {"path": str(owner.path), "status": "failed", "exit_code": 64}

    monkeypatch.setattr(launcher.subprocess, "run", gate)
    monkeypatch.setattr(launcher, "_execute_job", execute)

    def launch(operation="simulate", linked=False, **kwargs):
        function = sweep if operation == "sweep" else simulate
        return function(
            grid if operation == "sweep" else tmp_path / "simulate.toml",
            repository=runner_repository,
            runs_root=runs,
            **({"investigation_id": investigation.name, "base": operation} if linked else {}),
            **kwargs,
        )

    return launch, runs, investigation, calls


@pytest.mark.parametrize("operation", ["simulate", "sweep"])
@pytest.mark.parametrize("linked", [False, True])
@pytest.mark.parametrize("busy", ["owned", "live_group", "unreconciled"])
def test_busy_device_refuses_without_attempt_or_allowance(
    launch_context,
    tmp_path,
    operation,
    linked,
    busy,
):
    launch, runs, investigation, calls = launch_context
    child = None
    lease = acquire_gpu(DEVICE)
    try:
        if busy != "owned":
            previous = tmp_path / "previous"
            previous.mkdir()
            record = {"device": DEVICE, "run": str(previous)}
            if busy == "live_group":
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
                )
                record["process_group"] = child.pid
            publish_toml(lease.record_path, record)
            lease.close()
        message = {
            "owned": "still owned",
            "live_group": "live recorded",
            "unreconciled": "reconciliation",
        }[busy]
        with pytest.raises(ValueError, match=message):
            launch(operation, linked)
        assert not calls
        assert not list(runs.glob("*/*/run.toml"))
        record = read_investigation(investigation)
        assert record["usage"]["jobs"] == 0
        assert record["checkpoint"]["finished_attempts"] == 0
    finally:
        lease.close()
        if child is not None:
            child.kill()
            child.wait()


@pytest.mark.parametrize("operation", ["simulate", "sweep"])
def test_lock_spans_gate_admission_and_execution(launch_context, operation):
    launch, _, investigation, calls = launch_context
    result = launch(operation, linked=True)
    assert calls == ["gate", "execute"]
    assert (Path(result["path"]) / "FINISHED").exists()
    assert read_investigation(investigation)["usage"]["jobs"] == 1
    with acquire_gpu(DEVICE) as lease:
        assert not lease.record_path.exists()


@pytest.mark.parametrize("failure", ["gate", "admission", "manifest", "binding", "inputs"])
def test_launch_failures_release_device(launch_context, monkeypatch, failure):
    launcher = import_module("entanglix.protocols.simulate")
    runner = import_module("entanglix.runner")
    launch, runs, investigation, calls = launch_context
    if failure == "gate":
        monkeypatch.setattr(
            launcher.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=64, stderr="gate failed", stdout=""
            ),
        )
    elif failure == "admission":
        from entanglix.investigations import stop_investigation

        stop_investigation(investigation, reason="review_handoff")
    elif failure == "manifest":
        publish = runner.publish_toml

        def fail_manifest(path, *args, **kwargs):
            if path.name == "run.toml":
                raise OSError("manifest failed")
            return publish(path, *args, **kwargs)

        monkeypatch.setattr(runner, "publish_toml", fail_manifest)
    elif failure == "binding":

        def fail_binding(*_):
            raise OSError("allocation record failed")

        monkeypatch.setattr(runner.GpuLease, "bind_run", fail_binding)
    else:
        write = Path.write_bytes

        def fail_inputs(destination, data):
            if "spec" in destination.parts:
                raise OSError("copy failed")
            return write(destination, data)

        monkeypatch.setattr(Path, "write_bytes", fail_inputs)
    with pytest.raises((ValueError, runner.RunCreationError)):
        launch(linked=True)
    attempts = list(runs.glob("*/*/run.toml"))
    assert len(attempts) == (1 if failure in {"inputs", "binding"} else 0)
    if attempts:
        finished = _load_record(attempts[0].parent / "FINISHED")
        assert finished["status"] == "failed"
        assert finished["usage"]["gpu_seconds"] > 0
    else:
        assert not list(runs.glob("*/r*"))
    with acquire_gpu(DEVICE):
        pass


@pytest.mark.parametrize("operation", ["simulate", "sweep"])
def test_dry_run_checks_under_ownership_without_admission(launch_context, operation):
    launch, runs, investigation, calls = launch_context
    assert launch(operation, linked=True, dry_run=True)["exit_code"] == 0
    assert calls == ["gate"]
    assert not list(runs.glob("*/*/run.toml"))
    assert read_investigation(investigation)["usage"]["jobs"] == 0
    with acquire_gpu(DEVICE) as lease:
        assert not lease.record_path.exists()


def test_gate_child_retains_device_until_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    gpu = acquire_gpu(DEVICE)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,sys; os.fstat(int(sys.argv[1])); print('ready',flush=True); sys.stdin.read()",
            str(gpu.fileno()),
        ],
        pass_fds=(gpu.fileno(),),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        gpu.close()
        with pytest.raises(ValueError, match="still owned"):
            acquire_gpu(DEVICE)
        child.communicate(timeout=10)
        with acquire_gpu(DEVICE) as lease:
            assert not lease.record_path.exists()
    finally:
        gpu.close()
        if child.poll() is None:
            child.kill()
        child.communicate()


@pytest.mark.parametrize(
    "visible,devices",
    [
        (None, [DEVICE]),
        ("0", [DEVICE]),
        (DEVICE, [DEVICE]),
        (None, [DEVICE, "GPU-other"]),
        ("", []),
    ],
)
def test_device_selection_requires_one_canonical_device(monkeypatch, visible, devices):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    def query(command, **kwargs):
        if visible is not None:
            assert f"--id={visible}" in command
        return SimpleNamespace(stdout="\n".join(devices))

    monkeypatch.setattr("entanglix.runner.subprocess.run", query)
    if len(devices) == 1:
        assert select_gpu() == DEVICE
    else:
        with pytest.raises(ValueError, match="exactly one"):
            select_gpu()


@pytest.mark.parametrize(
    "operation,fault",
    [
        (operation, fault)
        for operation in ("simulate", "sweep")
        for fault in ("missing", "malformed", "invalid_device")
    ]
    + [("sweep", "missing_base")],
)
def test_protocol_read_errors_are_launch_refusals(tmp_path, runner_repository, operation, fault):
    grid, _ = sweep_files(tmp_path, axes=1)
    protocol = grid if operation == "sweep" else tmp_path / "simulate.toml"
    if fault == "missing":
        protocol.unlink()
        diagnostic = "No such file"
    elif fault == "malformed":
        protocol.write_text("[broken")
        diagnostic = "Expected"
    elif fault == "invalid_device":
        base = tmp_path / "simulate.toml"
        base.write_text(base.read_text().replace('device = "cpu"', 'device = "unknown"'))
        diagnostic = "device"
    else:
        (tmp_path / "simulate.toml").unlink()
        diagnostic = "No such file"
    function = sweep if operation == "sweep" else simulate
    runs = tmp_path / "runs"
    with pytest.raises(ValueError, match=f"{operation} rejected:") as error:
        function(protocol, repository=runner_repository, runs_root=runs)
    assert diagnostic in str(error.value)
    assert not list(runs.glob("*/*/run.toml"))
