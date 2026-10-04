"""Scope admission, conservative resource accounting, and linked executor evidence."""

import copy
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path

import pytest
import tomli_w

from entanglix.corrector.verification import check_run
from entanglix.investigations import (
    create_investigation,
    read_investigation,
    record_finding,
    record_user_decision,
    revise_investigation,
)
from entanglix.protocols.simulate import compile_simulation, rerun_simulation, simulate
from entanglix.protocols.sweep import rerun_sweep, sweep
from entanglix.runner import (
    CodeSnapshot,
    _digest,
    _execute_job,
    _load_record,
    capture_environment,
    create_run,
    read_run,
    record_finish,
    record_reconciliation,
    record_start,
    rerun_script,
    run_script,
)


@pytest.fixture
def study(tmp_path):
    system = {
        "format_version": 1,
        "id": "study",
        "units": {"energy": "GHz", "time": "ns"},
        "parameters": {"frequency": {"value": 0.2, "unit": "GHz"}},
        "model": {
            "kind": "composite",
            "subsystems": [
                {
                    "id": "r",
                    "kind": "oscillator",
                    "frequency": {"parameter": "frequency"},
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
        "drives": [{"id": "pulse", "factors": ["r.position"]}],
    }
    spec = {
        "format_version": 1,
        "system": "system.toml",
        "parameters": {"frequency": 0.2},
        "evolution": {
            "initial_state": [0, 0],
            "times_ns": [0.0, 0.1],
            "drive_amplitudes_ghz": {"pulse": 0.0},
            "observables": {"n": "r.number"},
        },
        "metrics": {"population": {"kind": "endpoint_real", "observable": "n"}},
        "refinement": {"r": {"retained_levels_step": 1}, "s": {"retained_levels_step": 1}},
    }
    (tmp_path / "system.toml").write_text(tomli_w.dumps(system))
    protocol = tmp_path / "simulate.toml"
    protocol.write_text(tomli_w.dumps(spec))
    terms = {
        "format_version": 1,
        "question": "Characterize pulse amplitude.",
        "assumptions": [],
        "scope": {
            "operations": ["simulate", "sweep", "check"],
            "bases": [
                {
                    "name": "seed",
                    "inputs": {
                        name: _digest((tmp_path / name).read_bytes())
                        for name in ("system.toml", "simulate.toml")
                    },
                    "variations": [
                        {"target": "parameters.frequency", "values": [0.2, 0.3]},
                        {"target": "evolution.drive_amplitudes_ghz.pulse", "range": [0.0, 0.02]},
                    ],
                }
            ],
        },
        "evidence_requirements": {
            "requirements": ["Report local evidence."],
            "checks": {"seed": ["integrity", "segments", "diagnostics"]},
        },
        "checkpoint": {"finished_attempts": 100},
        "review_criteria": [],
        "stop_criteria": [],
        "limits": {},
    }
    runs = tmp_path / "runs"
    investigation = create_investigation(terms, runs_root=runs)
    path = Path(investigation["path"])
    record_user_decision(path, "start", revision=1)
    return protocol, spec, terms, runs, path


def launch(study, repository, **kwargs):
    protocol, _, _, runs, path = study
    return simulate(
        protocol,
        runs_root=runs,
        repository=repository,
        investigation_id=path.name,
        base="seed",
        **kwargs,
    )


def assessment(runs):
    return {
        "kind": "assessment",
        "runs": runs,
        "learned": "Computed the requested points.",
        "unresolved": [],
        "decision": "continue",
        "rationale": "Continue inside the scope.",
        "next_experiment": "Compare the next amplitude.",
    }


def test_simulation_checks_findings_and_replay_use_current_revision(study, runner_repository):
    values = {"parameters.frequency": 0.3, "evolution.drive_amplitudes_ghz.pulse": 0.01}
    first = launch(study, runner_repository, values=values)
    run = Path(first["path"])
    assert first["exit_code"] == 0, read_run(run)["log_tail"]
    manifest = read_run(run)["manifest"]
    assert manifest["investigation_revision"] == 1 and manifest["investigation_base"] == "seed"
    assert _load_record(run / "spec/simulate.toml")["parameters"]["frequency"] == 0.3
    assert (run / "spec/base/simulate.toml").read_bytes() == study[0].read_bytes()
    before = record_finding(study[4], assessment([run.name]))
    assert before["evidence"][0]["requested_checks_not_run"] == [
        "diagnostics",
        "integrity",
        "segments",
    ]
    assert before["evidence"][0]["relevant_checks_not_run"] == [
        item["name"] for item in sorted(check_run(run)["checks"], key=lambda item: item["name"])
    ]
    checked = check_run(run, ("integrity", "diagnostics"), repository=runner_repository)
    assert checked["exit_code"] == 0
    assert read_run(Path(checked["path"]))["manifest"]["investigation_id"] == study[4].name
    after = record_finding(study[4], assessment([run.name, Path(checked["path"]).name]))
    assert after["usage"]["total"]["jobs"] == 2
    assert after["usage"]["total"]["evaluated_points"] == 1
    assert after["usage"]["since_assessment"]["jobs"] == 1
    assert after["evidence"][0]["requested_checks_not_run"] == ["segments"]
    assert after["evidence"][0]["relevant_checks_not_run"] == ["segments"]
    assert {item["name"] for item in after["evidence"][0]["checks"]} == {"integrity", "diagnostics"}
    assert "check_listing" not in after["evidence"][0]
    assert {item["name"] for item in after["evidence"][1]["checks"]} == {"integrity", "diagnostics"}
    assert after["evidence"][1]["requested_checks_not_run"] == []
    terms = copy.deepcopy(study[2])
    terms["limits"] = {"job_count": 5}
    terms["evidence_requirements"]["checks"]["seed"] = ["integrity"]
    revise_investigation(study[4], terms)
    record_user_decision(study[4], "approve-revision", revision=2)
    current = record_finding(study[4], assessment([run.name]))
    assert current["evidence"][0]["requested_checks_not_run"] == []
    assert current["evidence"][0]["relevant_checks_not_run"] == ["segments"]
    replay = rerun_simulation(run, repository=runner_repository)
    assert replay["exit_code"] == 0
    assert read_run(Path(replay["path"]))["manifest"]["investigation_revision"] == 2
    assert read_run(run)["manifest"]["investigation_limits"] == {}
    terms["scope"]["bases"][0]["variations"][0]["values"] = [0.2]
    revise_investigation(study[4], terms)
    record_user_decision(study[4], "approve-revision", revision=3)
    with pytest.raises(ValueError, match="permitted set"):
        rerun_simulation(run, repository=runner_repository)


@pytest.mark.parametrize(
    "values,reason",
    [
        ({"parameters.frequency": 0.4}, "permitted set"),
        ({"evolution.drive_amplitudes_ghz.pulse": 0.03}, "permitted range"),
        ({"evolution.times_ns": 2}, "unsupported sweep target"),
    ],
)
def test_out_of_scope_values_create_no_attempt(study, runner_repository, values, reason):
    with pytest.raises(ValueError, match=reason):
        launch(study, runner_repository, values=values)
    assert not list(study[3].glob("*/*/run.toml"))


def test_base_digests_operations_and_dry_run_are_admission_gates(study, runner_repository):
    dry = launch(study, runner_repository, dry_run=True)
    assert dry["exit_code"] == 0 and not list(study[3].glob("*/*/run.toml"))
    terms = copy.deepcopy(study[2])
    terms["scope"]["operations"] = ["check"]
    revise_investigation(study[4], terms)
    record_user_decision(study[4], "approve-revision", revision=2)
    with pytest.raises(ValueError, match="operation outside"):
        launch(study, runner_repository, dry_run=True)
    terms["scope"]["operations"] = ["simulate"]
    revise_investigation(study[4], terms)
    record_user_decision(study[4], "approve-revision", revision=3)
    study[0].write_text(study[0].read_text() + "\n# Changed base\n")
    with pytest.raises(ValueError, match="base input digests"):
        launch(study, runner_repository)
    assert not list(study[3].glob("*/*/run.toml"))


def test_sweep_validates_each_point_and_replays_base(study, runner_repository):
    protocol = study[0].parent / "sweep.toml"
    document = {
        "format_version": 1,
        "base": "simulate.toml",
        "axes": [
            {"id": "frequency", "target": "parameters.frequency", "values": [0.2, 0.3]},
            {
                "id": "amplitude",
                "target": "evolution.drive_amplitudes_ghz.pulse",
                "values": [0.0, 0.01],
            },
        ],
    }
    protocol.write_text(tomli_w.dumps(document))
    result = sweep(
        protocol,
        runs_root=study[3],
        repository=runner_repository,
        investigation_id=study[4].name,
        base="seed",
    )
    assert result["exit_code"] == 0, read_run(Path(result["path"]))["log_tail"]
    assert read_investigation(study[4])["usage"]["evaluated_points"] == 4
    checked = check_run(Path(result["path"]), ("integrity",), repository=runner_repository)
    assert checked["exit_code"] == 0
    repeated = rerun_sweep(Path(result["path"]), repository=runner_repository)
    assert repeated["exit_code"] == 0
    document["axes"][1]["values"][-1] = 0.03
    protocol.write_text(tomli_w.dumps(document))
    with pytest.raises(ValueError, match="permitted range"):
        sweep(
            protocol,
            runs_root=study[3],
            repository=runner_repository,
            investigation_id=study[4].name,
            base="seed",
        )
    assert read_investigation(study[4])["usage"]["jobs"] == 3


def test_linked_values_cannot_move_independently(study, runner_repository):
    terms = study[2]
    terms["scope"]["bases"][0]["variations"] = [
        {
            "target": "parameters.frequency",
            "values": [0.2, 0.3],
            "linked_values": {"evolution.drive_amplitudes_ghz.pulse": [0.0, 0.01]},
        }
    ]
    revise_investigation(study[4], terms)
    record_user_decision(study[4], "approve-revision", revision=2)
    with pytest.raises(ValueError, match="linked value"):
        launch(study, runner_repository, values={"parameters.frequency": 0.3})
    good = launch(
        study,
        runner_repository,
        values={"parameters.frequency": 0.3, "evolution.drive_amplitudes_ghz.pulse": 0.01},
    )
    assert good["exit_code"] == 0


def test_compilation_uses_the_captured_base_bytes(study, runner_repository, tmp_path, monkeypatch):
    module = import_module("entanglix.protocols.simulate")
    collect = module._system_inputs
    original = study[0].read_bytes()

    def change_protocol_after_read(*args, **kwargs):
        result = collect(*args, **kwargs)
        study[0].write_bytes(original + b"\n# Changed after capture\n")
        return result

    monkeypatch.setattr(module, "_system_inputs", change_protocol_after_read)
    destination = tmp_path / "compiled"
    destination.mkdir()
    gate = compile_simulation(study[0], destination, [], runner_repository, retain_base=True)
    assert gate["base_inputs"]["simulate.toml"] == _digest(original)
    assert (destination / "base/simulate.toml").read_bytes() == original


def test_investigation_cli_varies_the_declared_base(study, runner_repository):
    command = [
        sys.executable,
        "-m",
        "entanglix",
        "simulate",
        str(study[0]),
        "--runs-root",
        str(study[3]),
        "--investigation",
        study[4].name,
        "--base",
        "seed",
        "--vary",
        "evolution.drive_amplitudes_ghz.pulse=0.01",
    ]
    completed = subprocess.run(
        command,
        env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    run = Path(json.loads(completed.stdout)["path"])
    assert read_run(run)["manifest"]["launch"]["varied_values"] == {
        "evolution.drive_amplitudes_ghz.pulse": 0.01
    }


def test_base_names_are_fixed_and_mismatch_diagnostics_name_files(study, runner_repository):
    protocol, _, terms, _, path = study
    renamed = protocol.with_name("readout.toml")
    protocol.rename(renamed)
    options = dict(
        runs_root=study[3],
        repository=runner_repository,
        investigation_id=path.name,
        base="seed",
        dry_run=True,
    )
    assert simulate(renamed, **options)["exit_code"] == 0
    inputs = terms["scope"]["bases"][0]["inputs"]
    inputs["readout.toml"] = inputs.pop("simulate.toml")
    revise_investigation(path, terms)
    record_user_decision(path, "approve-revision", revision=2)
    with pytest.raises(
        ValueError, match="missing names: readout.toml; unexpected names: simulate.toml"
    ):
        simulate(renamed, **options)
    inputs["simulate.toml"] = inputs.pop("readout.toml")
    inputs["system.toml"] = "0" * 64
    revise_investigation(path, terms)
    record_user_decision(path, "approve-revision", revision=3)
    with pytest.raises(ValueError, match="changed digests: system.toml"):
        simulate(renamed, **options)


def test_check_admission_obeys_current_operation_and_job_limit(study, runner_repository):
    first = launch(study, runner_repository)
    source = Path(first["path"])
    terms = copy.deepcopy(study[2])
    terms["limits"] = {"job_count": 1}
    revise_investigation(study[4], terms)
    record_user_decision(study[4], "approve-revision", revision=2)
    with pytest.raises(ValueError, match="job_count limit"):
        check_run(source, ("integrity",), repository=runner_repository)
    terms["limits"] = {}
    terms["scope"]["operations"] = ["simulate"]
    revise_investigation(study[4], terms)
    record_user_decision(study[4], "approve-revision", revision=3)
    with pytest.raises(ValueError, match="operation outside"):
        check_run(source, ("integrity",), repository=runner_repository)
    assert read_investigation(study[4])["usage"]["jobs"] == 1
    terms["scope"]["operations"] = ["check"]
    terms["scope"]["bases"][0]["name"] = "new-base"
    terms["evidence_requirements"]["checks"] = {"new-base": ["integrity"]}
    revise_investigation(study[4], terms)
    record_user_decision(study[4], "approve-revision", revision=4)
    checked = check_run(source, ("integrity",), repository=runner_repository)
    assert checked["exit_code"] == 0
    assert read_run(Path(checked["path"]))["manifest"]["investigation_revision"] == 4


@pytest.fixture
def admission(tmp_path):
    system = tmp_path / "system.toml"
    system.write_text('id = "fixture"\n')
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    terms = {
        "format_version": 1,
        "question": "Exercise accounting.",
        "assumptions": [],
        "scope": {
            "operations": ["run_script"],
            "bases": [{"name": "exact", "inputs": {"system.toml": _digest(system.read_bytes())}}],
        },
        "evidence_requirements": {"requirements": ["Record usage."]},
        "checkpoint": {"finished_attempts": 100},
        "review_criteria": [],
        "stop_criteria": [],
        "limits": {},
    }
    root = tmp_path / "runs"
    path = Path(create_investigation(terms, runs_root=root)["path"])
    record_user_decision(path, "start", revision=1)
    kwargs = dict(
        system_id="fixture",
        protocol="script",
        inputs={"system.toml": system},
        environment=capture_environment(lock),
        code=CodeSnapshot("fixture", b"", {}),
        engine="authored-script",
        device="cpu",
        needs={},
        runs_root=root,
        investigation_id=path.name,
        base="exact",
    )
    return kwargs, terms, path


def approve(admission, limits):
    kwargs, terms, path = admission
    terms = copy.deepcopy(terms)
    terms["limits"] = limits
    revision = revise_investigation(path, terms)["proposed_revision"]
    record_user_decision(path, "approve-revision", revision=revision)
    return kwargs


def _competing_admission(kwargs):
    try:
        with create_run(**kwargs) as owner:
            return str(owner.path)
    except ValueError as exc:
        return str(exc)


@pytest.mark.parametrize("limits", [{"job_count": 1}, {"job_seconds": 1.0}])
def test_concurrent_process_admission_is_atomic(admission, limits):
    kwargs = approve(admission, limits) | {"launch": {"timeout_seconds": 1.0}}
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_competing_admission, [kwargs, kwargs]))
    assert sum("limit exhausted" in value for value in results) == 1
    assert read_investigation(admission[2])["usage"]["jobs"] == 1


def test_uncertain_job_usage_never_returns_allowance(admission):
    kwargs = approve(admission, {"job_seconds": 1.0}) | {
        "launch": {"timeout_seconds": 1.0},
    }
    with create_run(**kwargs) as owner:
        run = owner.path
    record_reconciliation(run, last_observed_state="runner lost before start")
    assert read_investigation(admission[2])["usage"]["committed_job_seconds"] == 1.0
    with pytest.raises(ValueError, match="job_seconds limit"):
        create_run(**kwargs)
    assert read_investigation(admission[2])["usage"]["jobs"] == 1


def test_confirmed_usage_replaces_commitment(admission):
    kwargs = approve(admission, {"job_seconds": 1.0})
    kwargs = kwargs | {"launch": {"timeout_seconds": 1.0}}
    with create_run(**kwargs) as owner:
        record_start(owner, process_group=2_000_000_000)
        record_finish(owner, status="failed", exit_code=1, wall_seconds=0.2, gpu_seconds=0)
    with create_run(**(kwargs | {"launch": {"timeout_seconds": 0.8}})):
        pass
    usage = read_investigation(admission[2])["usage"]
    assert usage["job_seconds"] == 0.2 and usage["committed_job_seconds"] == 0.8
    assert usage["gpu_seconds"] == 0
    with pytest.raises(ValueError, match="job_seconds limit"):
        create_run(**kwargs)


def test_draft_cannot_admit_a_job(admission):
    kwargs, terms, _ = admission
    draft = create_investigation(terms, runs_root=kwargs["runs_root"])
    with pytest.raises(ValueError, match="must be started"):
        create_run(**(kwargs | {"investigation_id": draft["investigation_id"]}))
    assert read_investigation(Path(draft["path"]))["usage"]["jobs"] == 0


def test_new_limit_cannot_ignore_earlier_unbounded_cpu_attempt(admission):
    with create_run(**admission[0]) as owner:
        earlier = owner.path.name
    kwargs = approve(admission, {"job_seconds": 1.0}) | {"launch": {"timeout_seconds": 0.1}}
    with pytest.raises(
        ValueError, match=f"cannot bound unsettled job usage from earlier run: {earlier}"
    ):
        create_run(**kwargs)


def test_job_without_timeout_uses_remaining_allowance_and_is_ended(admission, tmp_path):
    kwargs = approve(admission, {"job_seconds": 0.5})
    with create_run(**(kwargs | {"launch": {"timeout_seconds": 0.2}})) as first:
        record_start(first, process_group=2_000_000_000)
        record_finish(first, status="failed", exit_code=1, wall_seconds=0.1, gpu_seconds=0)
    worker = tmp_path / "worker.py"
    worker.write_text("import sys, time\nassert sys.stdin.buffer.read(1) == b'1'\ntime.sleep(20)\n")
    with create_run(**kwargs) as owner:
        commitment = _load_record(owner.path / "run.toml")["investigation_commitment"]
        assert commitment == {"job_seconds": 0.4, "unbounded": False}
        with pytest.raises(ValueError, match="job_seconds limit exhausted"):
            create_run(**kwargs)
        result = _execute_job(
            owner, [sys.executable, str(worker)], os.environ.copy(), timeout_seconds=None
        )
        finished = _load_record(owner.path / "FINISHED")
        assert result["status"] == "aborted" and result["exit_code"] == 130
        assert 0.4 <= finished["usage"]["wall_seconds"] < 3
        assert finished["usage"]["job_seconds_overrun"] == pytest.approx(
            finished["usage"]["wall_seconds"] - 0.4
        )
        assert "job_seconds limit; terminating" in (owner.path / "run.log").read_text()
    assert read_investigation(admission[2])["usage"]["gpu_seconds"] == 0
    with pytest.raises(ValueError, match="job_seconds limit exhausted"):
        create_run(**kwargs)


def test_deadline_ends_job_and_refuses_new_admission(admission, tmp_path):
    kwargs = approve(admission, {"deadline": datetime.now(UTC) + timedelta(seconds=0.5)})
    worker = tmp_path / "worker.py"
    worker.write_text("import sys, time\nassert sys.stdin.buffer.read(1) == b'1'\ntime.sleep(20)\n")
    with create_run(**kwargs) as owner:
        result = _execute_job(
            owner, [sys.executable, str(worker)], os.environ.copy(), timeout_seconds=None
        )
        record = read_run(owner.path)
    assert result["status"] == "aborted" and result["exit_code"] == 130
    assert record["finished"]["usage"]["wall_seconds"] < 3
    assert "deadline" in record["log_tail"]
    assert record["finished"]["usage"]["deadline_overrun_seconds"] >= 0
    with pytest.raises(ValueError, match="deadline has passed"):
        create_run(**kwargs)


def test_admitted_deadline_survives_approval_and_cancels_delayed_start(admission, tmp_path):
    kwargs = approve(admission, {"deadline": datetime.now(UTC) + timedelta(seconds=0.2)})
    with create_run(**kwargs) as owner:
        approve(admission, {"deadline": datetime.now(UTC) + timedelta(days=1)})
        time.sleep(0.25)
        result = _execute_job(
            owner, [sys.executable, "must-not-launch.py"], os.environ.copy(), timeout_seconds=None
        )
        assert result["status"] == "aborted"
        assert not (owner.path / "runner/started.toml").exists()
        assert _load_record(owner.path / "FINISHED")["usage"]["wall_seconds"] == 0
    with create_run(**kwargs) as later:
        assert _load_record(later.path / "run.toml")["investigation_revision"] == 3


def test_interrupt_before_start_is_a_usage_failure(admission, tmp_path, monkeypatch):
    runner = import_module("entanglix.runner")

    def interrupt_start(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "record_start", interrupt_start)
    worker = tmp_path / "worker.py"
    worker.write_text("import sys\nassert sys.stdin.buffer.read(1) == b'1'\n")
    kwargs = admission[0] | {"investigation_id": None, "base": None}
    with create_run(**kwargs) as owner:
        result = _execute_job(
            owner, [sys.executable, str(worker)], os.environ.copy(), timeout_seconds=None
        )
        assert result["status"] == "failed" and result["exit_code"] == 64
        assert not (owner.path / "runner/started.toml").exists()
        assert _load_record(owner.path / "FINISHED")["usage"]["wall_seconds"] == 0


def test_authored_script_requires_exact_bundle_and_replays(study, runner_repository):
    protocol, _, terms, root, path = study
    script = protocol.parent / "script.py"
    script.write_text(
        "import os\nfrom pathlib import Path\nfrom entanglix.runner import publish_toml\npublish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', {'engine': 'authored', 'metrics': {}}, replace=True)\n"
    )
    terms["scope"]["operations"] = ["run_script"]
    terms["evidence_requirements"]["checks"] = {}
    terms["scope"]["bases"] = [
        {
            "name": "authored",
            "inputs": {
                name: _digest((protocol.parent / name).read_bytes())
                for name in ("system.toml", "script.py")
            },
        }
    ]
    revise_investigation(path, terms)
    record_user_decision(path, "approve-revision", revision=2)
    first = run_script(
        protocol.parent / "system.toml",
        script,
        runs_root=root,
        repository=runner_repository,
        investigation_id=path.name,
        base="authored",
    )
    assert first["exit_code"] == 0
    replay = rerun_script(Path(first["path"]), repository=runner_repository)
    assert replay["exit_code"] == 0
    script.write_text(script.read_text() + "\nprint('changed')\n")
    with pytest.raises(ValueError, match="base input digests"):
        run_script(
            protocol.parent / "system.toml",
            script,
            runs_root=root,
            repository=runner_repository,
            investigation_id=path.name,
            base="authored",
        )


@pytest.mark.gpu
def test_workstation_gpu_usage_refuses_next_job(study, tmp_path, monkeypatch, record_property):
    pytest.importorskip("cupy")
    from conftest import _runner_repository

    monkeypatch.delenv("ENTANGLIX_RUNNER_EXTRAS", raising=False)
    repository = _runner_repository(tmp_path / "gpu-runner", ["cuquantum", "differentiable-cuda12"])
    protocol, spec, terms, root, path = study
    spec["needs"] = {"device": "cuda"}
    spec["timeout_seconds"] = 20.0
    protocol.write_text(tomli_w.dumps(spec))
    terms["scope"]["bases"][0]["inputs"]["simulate.toml"] = _digest(protocol.read_bytes())
    terms["limits"] = {"job_seconds": 20.0}
    revise_investigation(path, terms)
    record_user_decision(path, "approve-revision", revision=2)
    first = launch(study, repository)
    assert first["exit_code"] == 0, read_run(Path(first["path"]))["log_tail"]
    record = read_run(Path(first["path"]))
    assert record["result"]["engine"] == "cudensitymat" and record["result"]["device"] == "cuda"
    usage = read_investigation(path)["usage"]
    assert 0 < usage["gpu_seconds"] < 20.0 and usage["committed_job_seconds"] == 0
    assert usage["job_seconds"] == record["finished"]["usage"]["wall_seconds"]
    assert record["manifest"]["investigation_commitment"]["job_seconds"] == 20.0
    with pytest.raises(ValueError, match="job_seconds limit"):
        launch(study, repository)
    record_property("investigation_gpu_usage", json.dumps(usage))
    record_property(
        "gpu_environment",
        json.dumps(read_run(Path(first["path"]))["manifest"]["environment"]["gpu"]),
    )
