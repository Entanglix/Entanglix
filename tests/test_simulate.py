"""Public protocol validation, metric checks, replay, and scientific GPU lifecycle."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import tomli_w

from entanglix.corrector.verification import check_run
from entanglix.protocols.plans import load_plan
from entanglix.protocols.simulate import compile_simulation, rerun_simulation, simulate
from entanglix.runner import (
    _cpu_job_context,
    _load_record,
    abort_run,
    acquire_gpu,
    publish_toml,
    read_run,
    record_reconciliation,
)

ROOT = Path(__file__).resolve().parents[1]


def protocol_files(root, *, kind="lindblad", device="cpu"):
    root.mkdir(exist_ok=True)
    document = {
        "format_version": 1,
        "id": "oscillator",
        "units": {"energy": "GHz", "time": "ns"},
        "model": {
            "kind": "composite",
            "subsystems": [
                {
                    "id": "r",
                    "kind": "oscillator",
                    "frequency": {"value": 0.2, "unit": "GHz"},
                    "retained_levels": 3,
                },
                {
                    "id": "s",
                    "kind": "oscillator",
                    "frequency": {"value": 0.5, "unit": "GHz"},
                    "retained_levels": 2,
                },
            ],
        },
        "dissipators": [
            {"id": "loss", "factors": ["r.annihilation"], "rate": {"value": 0.1, "unit": "1/ns"}}
        ],
    }
    (root / "system.toml").write_text(tomli_w.dumps(document))
    spec = {
        "format_version": 1,
        "system": "system.toml",
        "kind": kind,
        "needs": {"device": device},
        "refinement": {"r": {"retained_levels_step": 1}, "s": {"retained_levels_step": 1}},
    }
    if kind == "spectrum":
        spec["levels"] = 2
    else:
        spec["evolution"] = {
            "initial_state": [1, 0],
            "times_ns": {"start": 0, "stop": 1, "count": 3},
            "observables": {
                "number": "r.number",
                "excited": {"kind": "population", "subsystem": "r", "level": 1},
            },
        }
        spec["metrics"] = {"excited_population": {"kind": "endpoint_real", "observable": "excited"}}
    path = root / "simulate.toml"
    path.write_text(tomli_w.dumps(spec))
    return path, spec


def test_spectral_simulation_cli_and_frozen_replay(tmp_path, runner_repository):
    protocol, _ = protocol_files(tmp_path, kind="spectrum")
    command = subprocess.run(
        [
            sys.executable,
            "-m",
            "entanglix",
            "simulate",
            str(protocol),
            "--runs-root",
            str(tmp_path / "runs"),
        ],
        capture_output=True,
        text=True,
        env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
    )
    assert command.returncode == 0, command.stderr
    result = json.loads(command.stdout)
    first = Path(result["path"])
    record = read_run(first)
    assert record["verification"] == []
    assert (first / "spec/simulate.toml").exists()
    assert (first / "artifacts/observables.png").stat().st_size > 100
    (tmp_path / "system.toml").unlink()
    protocol.unlink()
    replay = rerun_simulation(first, repository=runner_repository)
    record = read_run(Path(replay["path"]))
    assert record["manifest"]["parent_run_ids"] == [first.name]
    assert record["manifest"]["env_hash"] == read_run(first)["manifest"]["env_hash"]
    assert record["verification"] == []
    with (
        np.load(first / "artifacts/output.npz") as old,
        np.load(Path(replay["path"]) / "artifacts/output.npz") as new,
    ):
        np.testing.assert_array_equal(old["energies_ghz"], new["energies_ghz"])


def test_replay_reports_changed_code_and_environment(tmp_path, runner_repository):
    path, _ = protocol_files(tmp_path, kind="spectrum")
    first = simulate(path, runs_root=tmp_path / "runs", repository=runner_repository)
    run = Path(first["path"])
    (runner_repository / "notes.md").write_text("Documentation edit.\n")
    with (runner_repository / "uv.lock").open("a") as stream:
        stream.write("\n# Changed lockfile identity.\n")
    for commit in (False, True):
        if commit:
            subprocess.run(["git", "-C", str(runner_repository), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(runner_repository),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "docs",
                ],
                check=True,
            )
        replay = rerun_simulation(run, repository=runner_repository)
        assert replay["exit_code"] == 0
        assert replay["replay"] == {"code_matches": False, "environment_matches": False}
        repeated = read_run(Path(replay["path"]))
        assert repeated["manifest"]["parent_run_ids"] == [run.name]
        assert repeated["verification"] == []
        with (
            np.load(run / "artifacts/output.npz") as old,
            np.load(Path(replay["path"]) / "artifacts/output.npz") as new,
        ):
            np.testing.assert_array_equal(old["energies_ghz"], new["energies_ghz"])


@pytest.mark.parametrize(
    "bad",
    [
        "basis",
        "path",
        "metric",
        "segment_zero",
        "segment_infinite",
        "segment_name",
        "metric_tolerance",
        "gpu_uuid",
        "timeout",
        "parameter",
    ],
)
def test_invalid_protocol_rejects_before_admission(tmp_path, bad, runner_repository):
    path, spec = protocol_files(tmp_path)
    if bad == "basis":
        spec["evolution"]["initial_state"] = [3, 0]
    elif bad == "path":
        spec["system"] = "../system.toml"
    elif bad == "metric":
        spec["metrics"]["excited_population"]["kind"] = "fidelity"
    elif bad == "segment_zero":
        spec["segment_tolerances"] = {"excited": 0}
    elif bad == "segment_infinite":
        spec["segment_tolerances"] = {"excited": float("inf")}
    elif bad == "segment_name":
        spec["segment_tolerances"] = {"unknown": 0.001}
    elif bad == "metric_tolerance":
        spec["metrics"]["excited_population"]["segment_tolerance"] = -0.001
    elif bad == "gpu_uuid":
        spec["needs"]["gpu_uuid"] = "GPU-machine-specific"
    elif bad == "timeout":
        spec["timeout_seconds"] = 0
    else:
        spec["parameters"] = {"unknown": 1}
    path.write_text(tomli_w.dumps(spec))
    with pytest.raises(ValueError) as error:
        simulate(path, runs_root=tmp_path / "runs", repository=runner_repository)
    if bad.startswith("segment_"):
        expected = (
            "segment_tolerances must name declared observables"
            if bad == "segment_name"
            else "segment_tolerances must be finite and positive"
        )
        assert expected in str(error.value)
    elif bad == "metric_tolerance":
        assert "segment_tolerance" in str(error.value)
        assert "greater than 0" in str(error.value)
    assert not (tmp_path / "runs").exists()


def test_runner_local_extras_apply_to_every_job(tmp_path, monkeypatch, runner_repository):
    monkeypatch.setenv("ENTANGLIX_RUNNER_EXTRAS", "differentiable-cuda12,cuquantum,cuquantum")
    command, environment, extras = _cpu_job_context(runner_repository, "entanglix._script_job")
    assert set(extras) == {"differentiable-cuda12", "cuquantum"}
    assert command.count("--extra") == 2
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(runner_repository / ".venv-runner")
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    monkeypatch.setenv("ENTANGLIX_RUNNER_EXTRAS", "cuquantum,differentiable-cuda12")
    assert _cpu_job_context(runner_repository, "entanglix._script_job")[2] == extras


def test_dynamiqs_simulation_verifies_metrics_and_replays(tmp_path, differentiable_repository):
    path, _ = protocol_files(tmp_path)
    first = simulate(path, runs_root=tmp_path / "runs", repository=differentiable_repository)
    run = Path(first["path"])
    record = read_run(run)
    assert record["verification"] == []
    assert record["result"]["metrics"]["excited_population"] == pytest.approx(
        np.exp(-0.1), abs=1e-9
    )
    replay = rerun_simulation(run, repository=differentiable_repository)
    assert read_run(Path(replay["path"]))["verification"] == []
    result = record["result"]
    result["metrics"]["excited_population"] = 1.0
    publish_toml(run / "result.toml", result, replace=True)
    checking = check_run(run, ("integrity",), repository=differentiable_repository)
    assert read_run(Path(checking["path"]))["result"]["report"]["checks"][0]["outcome"] == "fail"
    state = run / "spec/initial-state.npz"
    state.chmod(0o644)
    np.savez(state, initial_state=np.array([1, 0, 0]))
    with pytest.raises(ValueError, match="digest mismatch"):
        rerun_simulation(run, repository=differentiable_repository)


@pytest.mark.parametrize("metric_bound", [None, 5e-4])
def test_dynamiqs_segment_tolerances_do_not_relax_physicality(
    tmp_path, differentiable_repository, metric_bound
):
    path, spec = protocol_files(tmp_path)
    spec["evolution"]["frame"] = {"frequencies_ghz": {"r": 0.2}, "rwa": True}
    spec["segment_tolerances"] = {"excited": 0.003}
    if metric_bound:
        spec["metrics"]["excited_population"]["segment_tolerance"] = metric_bound
    path.write_text(tomli_w.dumps(spec))
    job = simulate(path, runs_root=tmp_path / "runs", repository=differentiable_repository)
    checking = check_run(
        Path(job["path"]), ("integrity", "segments"), repository=differentiable_repository
    )
    report = read_run(Path(checking["path"]))["result"]["report"]
    checks = report["checks"][1]["segments"][0]["checks"]
    assert any(item["name"] == "observable" and item.get("tolerance") == 0.003 for item in checks)
    endpoint = next(item for item in checks if item["name"] == "metric")
    assert endpoint["tolerance"] == pytest.approx(metric_bound or 1e-6)
    assert report["checks"][0]["checks"][0]["tolerance"] == 1e-8


def test_complex_array_state_and_parameter_override_compile(tmp_path):
    path, spec = protocol_files(tmp_path)
    system = _load_record(tmp_path / "system.toml")
    system["parameters"] = {"frequency": {"value": 0.2, "unit": "GHz"}}
    system["model"]["subsystems"][0]["frequency"] = {"parameter": "frequency"}
    (tmp_path / "system.toml").write_text(tomli_w.dumps(system))
    state = np.array([1, 1j, 0, 0, 0, 0]) / np.sqrt(2)
    np.savez(tmp_path / "state.npz", psi=state)
    spec["parameters"] = {"frequency": 0.3}
    spec["evolution"].update(array_file="state.npz", initial_state="psi", times_ns=[0.2, 0.4])
    path.write_text(tomli_w.dumps(spec))
    target = tmp_path / "compiled"
    target.mkdir()
    compile_simulation(path, target, [], ROOT)
    plan = load_plan(target / "system.toml", target / "plan.toml")
    np.testing.assert_array_equal(plan.request.initial_state, state)
    assert plan.system.hamiltonian_ghz[2, 2] == pytest.approx(0.3)


def test_gpu_autoselection_reports_missing_cupy_extra(tmp_path, monkeypatch):
    path, _ = protocol_files(tmp_path, device="cuda")
    target = tmp_path / "compiled"
    target.mkdir()
    monkeypatch.setattr("entanglix.protocols.simulate.version", lambda package: "installed")
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    with acquire_gpu("GPU-test-device") as gpu:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu.device)
        monkeypatch.setenv("ENTANGLIX_GPU_LOCK_FD", str(gpu.fileno()))
        with pytest.raises(ValueError, match="cuquantum extra"):
            compile_simulation(path, target, [], ROOT)


@pytest.fixture
def gpu_repository(tmp_path, monkeypatch):
    pytest.importorskip("cupy")
    from conftest import _runner_repository

    monkeypatch.delenv("ENTANGLIX_RUNNER_EXTRAS", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return _runner_repository(tmp_path, ["cuquantum", "differentiable-cuda12"])


@pytest.mark.gpu
def test_gpu_simulation_verifies_after_execution_and_replays(tmp_path, gpu_repository):
    path, _ = protocol_files(tmp_path, device="cuda")
    job = simulate(path, runs_root=tmp_path / "runs", repository=gpu_repository)
    run = Path(job["path"])
    record = read_run(run)
    assert job["exit_code"] == 0, record["log_tail"]
    assert record["result"]["engine"] == "cudensitymat"
    assert record["result"]["device"] == "cuda"
    assert "gpu_uuid" not in _load_record(run / "spec/simulate.toml")["needs"]
    assert record["start"]["gpu_devices"]
    assert record["finished"]["usage"]["gpu_seconds"] > 0
    assert record["verification"] == []
    checking = check_run(run, ("integrity", "segments"), repository=gpu_repository)
    verification = read_run(Path(checking["path"]))
    assert all(item["outcome"] == "pass" for item in verification["result"]["report"]["checks"])
    assert record["finished"]["ended_at"] <= verification["start"]["started_at"]
    assert record["manifest"]["environment"]["gpu"]["cuda"] != "unobserved"
    replay = rerun_simulation(run, repository=gpu_repository)
    assert read_run(Path(replay["path"]))["verification"] == []


def wait_for(predicate, timeout=45):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    pytest.fail("simulation did not reach the expected state")


@pytest.mark.gpu
@pytest.mark.parametrize("stop", ["abort", "loss", "timeout"])
def test_gpu_simulation_lifecycle_and_interrupted_replay(tmp_path, gpu_repository, stop):
    path, spec = protocol_files(tmp_path, device="cuda")
    if stop == "timeout":
        spec["timeout_seconds"] = 8
        path.write_text(tomli_w.dumps(spec))
    worker = gpu_repository / "src/entanglix/protocols/_simulate_job.py"
    worker.write_text(
        worker.read_text().replace(
            "execute(Path(sys.argv[2]))",
            "execute(Path(sys.argv[2]))\n            from entanglix.runner import _load_record\n            run = Path(os.environ['ENTANGLIX_RUN_DIR'])\n            if not _load_record(run / 'run.toml')['parent_run_ids']:\n                import time\n                (run / 'artifacts/ready').touch()\n                time.sleep(30)",
        )
    )
    runs = tmp_path / "runs"
    environment = os.environ | {"PYTHONPATH": str(gpu_repository / "src")}
    supervisor = subprocess.Popen(
        [sys.executable, "-m", "entanglix", "simulate", str(path), "--runs-root", str(runs)],
        env=environment,
        cwd=gpu_repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    run = None
    try:
        run = wait_for(
            lambda: next((p.parent.parent for p in runs.glob("*/*/artifacts/ready")), None)
        )
        group = read_run(run)["start"]["process_group"]
        if stop == "loss":
            supervisor.kill()
            supervisor.communicate(timeout=10)
            assert read_run(run)["status"] == "active"
            with pytest.raises(ValueError, match="still owned"):
                record_reconciliation(run, last_observed_state="runner killed")
            with pytest.raises(ValueError, match="still owned"):
                simulate(path, runs_root=runs, repository=gpu_repository)
            assert len(list(runs.glob("*/*/run.toml"))) == 1
            abort_run(run)
            wait_for(lambda: read_run(run)["status"] == "interrupted")
            with pytest.raises(ValueError, match="requires reconciliation"):
                simulate(path, runs_root=runs, repository=gpu_repository)
            assert len(list(runs.glob("*/*/run.toml"))) == 1
            record_reconciliation(run, last_observed_state="runner lost; job terminated")
            assert not (run / "FINISHED").exists()
            assert read_run(run)["reconciliation"]["usage"]["uncertain"]
        else:
            if stop == "abort":
                abort_run(run)
            _, stderr = supervisor.communicate(timeout=20)
            assert supervisor.returncode == 130, stderr
            assert read_run(run)["status"] == "aborted"
        with pytest.raises(ProcessLookupError):
            os.killpg(group, 0)
        assert read_run(run)["verification"] == []
        replay = rerun_simulation(run, repository=gpu_repository)
        assert replay["exit_code"] == 0, read_run(Path(replay["path"]))["log_tail"]
        assert read_run(Path(replay["path"]))["verification"] == []
    finally:
        if run is not None and read_run(run)["status"] == "active":
            os.killpg(read_run(run)["start"]["process_group"], signal.SIGKILL)
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.communicate(timeout=10)


@pytest.mark.parametrize("declaration", ["missing", "below_policy"])
def test_launch_dry_run_and_inspection_list_unavailable_checks(
    tmp_path, runner_repository, capsys, declaration
):
    import shutil

    path, spec = protocol_files(tmp_path)
    shutil.copyfile(ROOT / "examples/readout/system.toml", tmp_path / "system.toml")
    if declaration == "missing":
        spec.pop("refinement")
    else:
        spec["refinement"] = {
            "q": {"retained_levels_step": 1, "cutoff_steps": {"cutoff": 1}},
            "r": {"retained_levels_step": 2},
        }
    path.write_text(tomli_w.dumps(spec))
    runs = tmp_path / "runs"
    dry = simulate(path, dry_run=True, runs_root=runs, repository=runner_repository)["preview"]
    assert not runs.exists()
    assert dry["model"]["dimensions"] == [5, 6] and dry["execution"]["output_count"] == 3
    segments = next(check for check in dry["checks"] if check["name"] == "segments")
    assert not segments["available"] and segments["reason"]
    job = simulate(path, runs_root=runs, repository=runner_repository)
    emitted = json.loads(capsys.readouterr().err)["preview"]
    run = Path(job["path"])
    record = read_run(run)
    assert job["exit_code"] == 0 and record["verification"] == []
    listing = check_run(run)
    for field in ("model", "checks", "limit"):
        assert emitted[field] == dry[field] == listing[field] == record["check_listing"][field]
    checked = check_run(run, ("segments", "diagnostics"), repository=runner_repository)
    assert checked["exit_code"] == 0
    reports = read_run(Path(checked["path"]))["result"]["report"]["checks"]
    assert reports[0]["outcome"] == "inapplicable" and reports[0]["reason"]
    assert "outcome" not in reports[1] and reports[1]["values"]["refined_cutoff_omitted"]
