"""Exact control bases, real plugin discovery, linked checks/reruns, and lifecycle."""

import copy
import json
import os
import shutil
import subprocess
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import tomli_w
from test_control import control_files as control_files

from entanglix.corrector.verification import check_run
from entanglix.investigations import (
    create_investigation,
    read_investigation,
    record_finding,
    record_user_decision,
    revise_investigation,
    stop_investigation,
)
from entanglix.mcp_views import attempt_data
from entanglix.protocols.control import control, load_control, rerun_control
from entanglix.runner import _cpu_job_context, read_run


@pytest.fixture
def plugin_repository(runner_repository):
    # This is installed as a real distribution entry point, not mocked discovery.
    shutil.copyfile(
        Path(__file__).parent / "data/control_plugin.py",
        runner_repository / "src/entanglix/_fixture_control.py",
    )
    project = runner_repository / "pyproject.toml"
    config = tomllib.loads(project.read_text())
    config["project"]["entry-points"]["entanglix.solvers"]["fixture"] = (
        "entanglix._fixture_control:Backend"
    )
    project.write_text(tomli_w.dumps(config))
    return runner_repository


@pytest.fixture
def control_study(control_files, plugin_repository, tmp_path):
    protocol, definition = control_files
    # Both aliases and comments must survive an investigation rerun.
    (tmp_path / "system.toml").rename(tmp_path / "device.toml")
    definition.update(system="device.toml", solver={"backend": "fixture"})
    definition["pulse"]["T"] = 10.0
    definition["problem"]["final_fidelity"] = 0.999
    protocol = tmp_path / "gate.toml"
    protocol.write_text("# Authored base, not canonical TOML\n" + tomli_w.dumps(definition))
    terms = {
        "format_version": 1,
        "question": "Establish the declared X gate and report its requested checks.",
        "assumptions": ["Closed rotating-frame model."],
        "scope": {
            "operations": ["control", "check"],
            "bases": [{"name": "gate", "inputs": load_control(protocol).input_digests}],
        },
        "evidence_requirements": {
            "requirements": ["Report fidelity, constraints and leakage."],
            "checks": {"gate": ["replay", "target", "constraints"]},
        },
        "checkpoint": {"finished_attempts": 100},
        "review_criteria": [],
        "stop_criteria": [],
        "limits": {},
    }
    root = tmp_path / "runs"
    path = Path(create_investigation(terms, runs_root=root)["path"])
    record_user_decision(path, "start", revision=1)
    return protocol, terms, root, path, plugin_repository


def launch(study, **kwargs):
    protocol, _, root, path, repository = study
    return control(
        protocol,
        runs_root=root,
        investigation_id=path.name,
        base="gate",
        repository=repository,
        **kwargs,
    )


def approve(study, **changes):
    _, terms, _, path, _ = study
    revision = revise_investigation(path, copy.deepcopy(terms) | changes)["proposed_revision"]
    record_user_decision(path, "approve-revision", revision=revision)


def finding(runs):
    return {
        "kind": "assessment",
        "runs": runs,
        "learned": "Candidate and check results retained.",
        "unresolved": [],
        "decision": "continue",
        "rationale": "Proceed within scope.",
        "next_experiment": "Inspect the next candidate.",
    }


def test_control_plugin_candidate_checks_and_finding_usage(control_study):
    protocol, _, _, path, repository = control_study
    assert launch(control_study, dry_run=True)["preview"]["solver_selection"]["actual"] == "fixture"
    assert read_investigation(path)["usage"]["jobs"] == 0
    job = launch(control_study)
    run = Path(job["path"])
    source = read_run(run)
    assert source["status"] == "completed", source["log_tail"]
    assert source["result"]["solver"]["actual"] == "fixture"
    assert source["verification"] == []
    assert (run / "spec/base/control.toml").read_bytes() == protocol.read_bytes()
    assert (run / "spec/base/device.toml").read_bytes() == protocol.with_name(
        "device.toml"
    ).read_bytes()
    assert tomllib.loads((run / "spec/control.toml").read_text())["system"] == "system.toml"
    before = record_finding(path, finding([run.name]))["evidence"][0]
    assert (
        before["requested_checks_not_run"]
        == before["relevant_checks_not_run"]
        == ["constraints", "replay", "target"]
    )
    checked = check_run(run, ("replay", "target", "constraints"), repository=repository)
    assert checked["exit_code"] == 0
    report = read_run(Path(checked["path"]))
    assert report["manifest"]["investigation_id"] == path.name
    assert all(item["outcome"] == "pass" for item in report["result"]["report"]["checks"])
    after = record_finding(path, finding([run.name]))["evidence"][0]
    assert after["relevant_checks_not_run"] == after["requested_checks_not_run"] == []
    assert {item["name"] for item in after["checks"]} == {"replay", "target", "constraints"}
    columns, rows = attempt_data(read_investigation(path))
    control_row = next(
        dict(zip(columns, row, strict=True))
        for row in rows
        if row[columns.index("protocol")] == "control"
    )
    assert control_row["checks"] == [
        [
            Path(checked["path"]).name,
            True,
            [[item["name"], "pass"] for item in report["result"]["report"]["checks"]],
            None,
        ]
    ]
    usage = read_investigation(path)["usage"]
    assert usage["jobs"] == 2 and usage["job_seconds"] > 0 and usage["gpu_seconds"] == 0


@pytest.mark.parametrize("change", ["inputs", "operation", "job_count"])
def test_control_scope_and_limits_refuse_before_admission(control_study, change):
    _, terms, root, path, _ = control_study
    replacement = copy.deepcopy(terms)
    if change == "inputs":
        replacement["scope"]["bases"][0]["inputs"]["control.toml"] = "0" * 64
        message = "changed digests: control.toml"
    elif change == "operation":
        replacement["scope"]["operations"] = ["check"]
        message = "operation outside investigation scope: control"
    else:
        replacement["limits"] = {"job_count": 1}
        assert launch(control_study)["exit_code"] == 0
        message = "job_count limit exhausted"
    approve(control_study, **replacement)
    count = len(list(root.glob("*/*/run.toml")))
    for dry_run in (True, False):
        with pytest.raises(ValueError, match=message):
            launch(control_study, dry_run=dry_run)
    assert len(list(root.glob("*/*/run.toml"))) == count
    assert read_investigation(path)["usage"]["jobs"] == count


def test_control_cli_and_rerun_keep_original_base_and_current_revision(control_study):
    protocol, terms, root, path, repository = control_study
    command, environment, _ = _cpu_job_context(repository, "entanglix")
    first = subprocess.run(
        [
            *command,
            "control",
            str(protocol),
            "--runs-root",
            str(root),
            "--investigation",
            path.name,
            "--base",
            "gate",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert first.returncode == 0, first.stderr
    run = Path(json.loads(first.stdout)["path"])
    original_digest = read_run(run)["result"]["problem_sha256"]
    # The current files are irrelevant: replay uses frozen original base bytes.
    protocol.unlink()
    protocol.with_name("device.toml").unlink()
    changed = copy.deepcopy(terms)
    changed["scope"]["operations"] = ["check"]
    approve(control_study, **changed)
    with pytest.raises(ValueError, match="operation outside investigation scope"):
        rerun_control(run, repository=repository)
    with pytest.raises(ValueError, match="retain their runs root"):
        rerun_control(run, repository=repository, runs_root=root / "other")
    approve(control_study)
    replay = subprocess.run(
        [*command, "rerun-control", str(run), "--detach"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stderr
    admitted = json.loads(replay.stdout)
    repeated = read_run(Path(admitted["path"]), wait_seconds=30)
    assert repeated["status"] == "completed", repeated.get("log_tail")
    manifest = repeated["manifest"]
    assert manifest["parent_run_ids"] == [run.name]
    assert manifest["investigation_id"] == path.name and manifest["investigation_revision"] == 3
    assert repeated["result"]["problem_sha256"] == original_digest
    assert Path(repeated["path"]).name != run.name
    assert read_investigation(path)["usage"]["jobs"] == 2


@pytest.mark.parametrize("reason", ["stop", "deadline", "job_seconds"])
def test_control_execution_obeys_investigation_lifecycle(control_study, reason):
    protocol, _, root, path, repository = control_study
    definition = tomllib.loads(protocol.read_text())
    definition["solver"]["options"] = {"delay_seconds": 60}
    protocol.write_text(tomli_w.dumps(definition))
    # Sync and preflight before setting a deadline, without consuming allowance.
    control(protocol, repository=repository, dry_run=True)
    terms = copy.deepcopy(control_study[1])
    terms["scope"]["bases"][0]["inputs"] = load_control(protocol).input_digests
    if reason == "deadline":
        terms["limits"] = {"deadline": datetime.now(UTC) + timedelta(seconds=10)}
    elif reason == "job_seconds":
        terms["limits"] = {"job_seconds": 0.5}
    approve(control_study, **terms)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(launch, control_study)
        try:
            if reason == "stop":
                until = time.monotonic() + 30
                while time.monotonic() < until:
                    started = list(root.glob("*/*/artifacts/plugin-started"))
                    if started:
                        break
                    assert not result.done(), result.result() if result.done() else None
                    time.sleep(0.02)
                else:
                    pytest.fail("control plugin did not start")
                assert stop_investigation(path, "user_cancellation")["cancellation_errors"] == []
            job = result.result(timeout=30)
        finally:
            if not result.done():
                stop_investigation(path, "user_cancellation")
    record = read_run(Path(job["path"]))
    assert record["status"] == "aborted", record["log_tail"]
    assert job["exit_code"] == 130
    with pytest.raises(ProcessLookupError):
        os.killpg(record["start"]["process_group"], 0)
    usage = read_investigation(path)["usage"]
    assert usage["jobs"] == 1 and usage["job_seconds"] > 0 and usage["gpu_seconds"] == 0
    if reason == "job_seconds":
        assert "job_seconds" in record["log_tail"]
    if reason == "stop":
        record_user_decision(path, "resume")
        assert read_investigation(path)["state"] == "active"


def test_standalone_control_rerun_and_frozen_input_tampering(control_study):
    protocol, _, root, _, repository = control_study
    original = control(protocol, repository=repository, runs_root=root)
    run = Path(original["path"])
    replay = rerun_control(run, repository=repository)
    repeated = read_run(Path(replay["path"]))
    assert repeated["status"] == "completed", repeated["log_tail"]
    assert repeated["manifest"]["parent_run_ids"] == [run.name]
    assert repeated["result"]["problem_sha256"] == read_run(run)["result"]["problem_sha256"]
    frozen = run / "spec/problem.toml"
    frozen.chmod(0o644)
    frozen.write_text(frozen.read_text() + "\n# tampered\n")
    with pytest.raises(ValueError, match="digest"):
        rerun_control(run, repository=repository)
