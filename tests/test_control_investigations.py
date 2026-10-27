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
from test_control import transmon_control as transmon_control

from entanglix.corrector.verification import check_run
from entanglix.investigations import (
    Variation,
    create_investigation,
    read_investigation,
    record_finding,
    record_user_decision,
    revise_investigation,
    stop_investigation,
)
from entanglix.mcp_views import attempt_data
from entanglix.protocols.control import control, load_control
from entanglix.runner import _cpu_job_context, _load_record, read_run, rerun


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


SEED = 2**53 + 1  # Not representable as a float.


@pytest.fixture
def varied_study(control_study):
    protocol, terms, root, path, repository = control_study
    # Control omits dissipators, so the parameter changes the frozen system only.
    # The system's name must not be mistaken for simulation defaults at admission.
    system = protocol.with_name("simulate.toml")
    protocol.with_name("device.toml").rename(system)
    document = tomllib.loads(system.read_text())
    document["parameters"] = {"gamma": {"value": 0.01, "unit": "1/ns"}}
    document["dissipators"][0]["rate"] = {"parameter": "gamma"}
    system.write_text(tomli_w.dumps(document))
    definition = tomllib.loads(protocol.read_text())
    definition.update(system="simulate.toml", parameters={"gamma": 0.01})
    protocol.write_text(tomli_w.dumps(definition))
    terms = copy.deepcopy(terms)
    terms["scope"]["bases"][0].update(
        inputs=load_control(protocol).input_digests,
        variations=[
            {"target": "parameters.gamma", "range": [0.0, 0.05]},
            {"target": "pulse.seed", "values": [7, 8, SEED]},
        ],
    )
    approve(control_study, **terms)
    return protocol, terms, root, path, repository


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


def attempt_row(path, run):
    columns, rows = attempt_data(read_investigation(path))
    return next(
        dict(zip(columns, row, strict=True))
        for row in rows
        if Path(row[columns.index("path")]).name == run.name
    )


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
        rerun(run, repository=repository)
    with pytest.raises(ValueError, match="retain their runs root"):
        rerun(run, repository=repository, runs_root=root / "other")
    approve(control_study)
    replay = subprocess.run(
        [*command, "rerun", str(run), "--detach"],
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


@pytest.mark.parametrize(
    ("variation", "message"),
    [
        ({"target": "pulse.seed", "range": [0, 3]}, "pulse.seed varies by values"),
        ({"target": "pulse.seed", "values": [1, 2.5]}, "integers from 0 to 2\\*\\*63 - 1"),
        ({"target": "pulse.seed", "values": [2**63]}, "integers from 0 to 2\\*\\*63 - 1"),
        (
            {"target": "parameters.g", "values": [0.1], "linked_values": {"pulse.seed": [-1]}},
            "integers from 0 to 2\\*\\*63 - 1",
        ),
    ],
)
def test_seed_variations_declare_nonnegative_integer_values(variation, message):
    with pytest.raises(ValueError, match=message):
        Variation.model_validate(variation)


def test_control_varies_parameters_and_seeds_inside_one_base(varied_study):
    protocol, terms, root, path, repository = varied_study
    command, environment, _ = _cpu_job_context(repository, "entanglix")
    first = subprocess.run(
        [
            *command, "control", str(protocol), "--runs-root", str(root),
            "--investigation", path.name, "--base", "gate",
            "--vary", "parameters.gamma=0.02", "--vary", f"pulse.seed={SEED}",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )  # fmt: skip
    assert first.returncode == 0, first.stderr
    run = Path(json.loads(first.stdout)["path"])
    record = read_run(run)
    assert record["status"] == "completed", record["log_tail"]
    manifest = record["manifest"]
    assert manifest["investigation_revision"] == 2
    assert manifest["launch"]["varied_values"] == {"parameters.gamma": 0.02, "pulse.seed": SEED}
    frozen = tomllib.loads((run / "spec/control.toml").read_text())
    assert frozen["parameters"] == {"gamma": 0.02} and frozen["pulse"]["seed"] == SEED
    assert tomllib.loads((run / "spec/problem.toml").read_text())["pulse"]["seed"] == SEED
    system = tomllib.loads((run / "spec/system.toml").read_text())
    assert system["parameters"]["gamma"]["value"] == 0.02
    row = attempt_row(path, run)
    assert row["values.parameters.gamma"] == 0.02 and row["values.pulse.seed"] == SEED
    assert row["seed"] == SEED and row["transition"] is None
    assert row["termination"] == {"converged": True}
    assert (run / "spec/base/control.toml").read_bytes() == protocol.read_bytes()
    # An omitted value keeps the base's own and still passes the scope check.
    nominal = launch(varied_study, values={"pulse.seed": 7})
    assert read_run(Path(nominal["path"]))["status"] == "completed"
    count = len(list(root.glob("*/*/run.toml")))
    for values, message in (
        ({"parameters.gamma": 0.1}, "outside permitted range: parameters.gamma"),
        ({"pulse.seed": 10}, "outside permitted set: pulse.seed"),
    ):
        with pytest.raises(ValueError, match=message):
            launch(varied_study, values=values)
    assert len(attempt_data(read_investigation(path))[1]) == 2
    # Solver settings and every other input stay exact.
    definition = tomllib.loads(protocol.read_text())
    definition["solver"]["options"] = {"delay_seconds": 0}
    changed = protocol.with_name("changed.toml")
    changed.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match="changed digests: control.toml"):
        control(
            changed,
            runs_root=root,
            investigation_id=path.name,
            base="gate",
            repository=repository,
            values={"pulse.seed": 8},
        )
    with pytest.raises(ValueError, match="require an investigation"):
        control(protocol, runs_root=root, repository=repository, values={"pulse.seed": 8})
    assert len(list(root.glob("*/*/run.toml"))) == count
    replay = rerun(run, repository=repository)
    repeated = Path(replay["path"])
    assert read_run(repeated)["status"] == "completed"
    assert _load_record(repeated / "run.toml")["launch"]["varied_values"] == {
        "parameters.gamma": 0.02,
        "pulse.seed": SEED,
    }
    for name in ("control.toml", "system.toml", "problem.toml"):
        assert (repeated / "spec" / name).read_bytes() == (run / "spec" / name).read_bytes()
    narrowed = copy.deepcopy(terms)
    narrowed["scope"]["bases"][0]["variations"][1]["values"] = [7, 8]
    approve(varied_study, **narrowed)
    with pytest.raises(ValueError, match="outside permitted set: pulse.seed"):
        rerun(run, repository=repository)
    assert len(list(root.glob("*/*/run.toml"))) == count + 1


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
    row = attempt_row(path, Path(job["path"]))
    assert row["status"] == "aborted" and row["seed"] == 7 and row["termination"] is None
    with pytest.raises(ProcessLookupError):
        os.killpg(record["start"]["process_group"], 0)
    usage = read_investigation(path)["usage"]
    assert usage["jobs"] == 1 and usage["job_seconds"] > 0 and usage["gpu_seconds"] == 0
    if reason == "job_seconds":
        assert "job_seconds" in record["log_tail"]
    if reason == "stop":
        record_user_decision(path, "resume")
        assert read_investigation(path)["state"] == "active"


def test_attempt_table_shows_the_derived_transition(control_study, transmon_control):
    _, terms, root, path, repository = control_study
    protocol, definition = transmon_control
    definition["solver"] = {"backend": "fixture"}
    definition["pulse"]["T"], definition["problem"]["N"] = 10.0, 5
    protocol.write_text(tomli_w.dumps(definition))
    loaded = load_control(protocol)
    scope = copy.deepcopy(terms["scope"])
    scope["bases"][0]["inputs"] = loaded.input_digests
    approve(control_study, scope=scope)
    job = control(
        protocol, runs_root=root, investigation_id=path.name, base="gate", repository=repository
    )
    assert read_run(Path(job["path"]))["status"] == "completed"
    row = attempt_row(path, Path(job["path"]))
    frequency = loaded.listing["resolved"]["transition_frequencies_ghz"]["junction_with_shunt"]
    assert row["transition"] == {
        "junction_with_shunt": {"levels": [0, 1], "frequency_ghz": frequency}
    }
    assert row["seed"] == 7


def test_standalone_control_rerun_and_frozen_input_tampering(control_study):
    protocol, _, root, _, repository = control_study
    original = control(protocol, repository=repository, runs_root=root)
    run = Path(original["path"])
    replay = rerun(run, repository=repository)
    repeated = read_run(Path(replay["path"]))
    assert repeated["status"] == "completed", repeated["log_tail"]
    assert repeated["manifest"]["parent_run_ids"] == [run.name]
    assert repeated["result"]["problem_sha256"] == read_run(run)["result"]["problem_sha256"]
    frozen = run / "spec/problem.toml"
    frozen.chmod(0o644)
    frozen.write_text(frozen.read_text() + "\n# tampered\n")
    with pytest.raises(ValueError, match="digest"):
        rerun(run, repository=repository)


def test_a_system_named_like_the_control_input_is_refused_for_an_investigation(control_study):
    protocol, terms, root, path, _ = control_study
    # The system takes the name the runner gives the control input in a base.
    protocol.with_name("device.toml").rename(protocol.with_name("control.toml"))
    text = protocol.read_text()
    assert text.count('system = "device.toml"') == 1
    protocol.write_text(text.replace('system = "device.toml"', 'system = "control.toml"'))
    # The approved base matches these inputs exactly, so only the collision refuses them.
    scope = copy.deepcopy(terms["scope"])
    scope["bases"][0]["inputs"] = load_control(protocol).input_digests
    approve(control_study, scope=scope)
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="the system file 'control.toml' would replace"):
            launch(control_study, dry_run=dry_run)
    assert list(root.glob("*/*/run.toml")) == []
    assert read_investigation(path)["usage"]["jobs"] == 0
