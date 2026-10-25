"""Operator promotion binds source evidence, respects its coverage, and publishes once."""

import json
import os
import pwd
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from test_calibration_checks import run_checks
from test_catalog import CODE, ROOT, completed, control_run, simulation, sweep_run, tree
from test_control import control_files as control_files
from test_control_checks import candidate as candidate
from test_control_checks import retained_candidate
from test_proposals import oscillator_run

from entanglix import catalog
from entanglix.catalog import list_catalog, promote, read_entry, register_result
from entanglix.cli import main
from entanglix.runner import (
    _digest,
    _load_record,
    capture_environment,
    create_run,
    publish_toml,
    record_finish,
    record_start,
)


def verification(source, items, *, status="completed", matches=True, solution=None):
    """A linked synthetic report, read through the real verification-history binding checks."""
    result = _load_record(source / "result.toml")
    artifact = (
        "pulse"
        if result["kind"] == "control"
        else "assignment"
        if result["kind"] == "calibrate"
        else "output"
    )
    request = {"checks": [item["name"] for item in items]}
    if solution is not None:
        request["solution"] = solution
    binding = {
        "source_result_sha256": _digest((source / "result.toml").read_bytes())
        if matches
        else "earlier-result",
        "source_artifact_sha256": result["artifacts"][artifact]["sha256"],
        "policy_version": "retained-test-policy",
        "request_file": "check-request.toml",
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / binding["request_file"]
        publish_toml(path, request)
        with create_run(
            system_id=source.parent.name,
            protocol="verification",
            inputs={"system.toml": source / "spec/system.toml", path.name: path},
            environment=capture_environment(ROOT / "uv.lock"),
            code=CODE,
            engine="corrector",
            device="cpu",
            needs={},
            runs_root=source.parent.parent,
            parent_run_ids=(source.name,),
            launch=binding,
        ) as owner:
            record_start(owner, process_group=os.getpgrp(), gpu_devices=[])
            if status == "completed":
                publish_toml(
                    owner.path / "result.toml",
                    {
                        "engine": "corrector",
                        "metrics": {},
                        **binding,
                        "report": {"requested_checks": request["checks"], "checks": items},
                    },
                )
            if status != "active":
                record_finish(
                    owner,
                    status=status,
                    exit_code=0 if status == "completed" else 64,
                    wall_seconds=0,
                    gpu_seconds=0,
                )
    return owner.path


def outcome(name, value="pass", **fields):
    return {"name": name, "outcome": value, "scope": "declared computation", **fields}


def test_unchecked_promotion_copies_design_and_retains_immutable_staging(tmp_path):
    run = simulation(tmp_path / "source")
    staged = register_result(run, "device", claim="a reusable oscillator")
    before = tree(Path(staged["path"]))
    roots = tmp_path / "runs"
    entry = promote(staged["staged_id"], roots)
    assert entry["staged_id"] == staged["staged_id"]
    assert entry["entry_id"] == "e" + staged["staged_id"][1:]
    assert entry["checks"] == [] and entry["checks_not_run"] == ["spectral_refinement"]
    assert entry["unusable_evidence"] == []
    assert entry["promoter"] == {
        "kind": "operator-cli",
        "os_user": pwd.getpwuid(os.getuid()).pw_name,
    }
    assert tree(Path(staged["path"])) == before
    device = roots / entry["device"]["path"] / "system.toml"
    assert device.read_bytes() == (Path(staged["path"]) / "device/system.toml").read_bytes()
    read = read_entry(entry["entry_id"], roots)
    assert read["status"] == "promoted" and read["promoter"] == entry["promoter"]
    assert read["lineage"]["source_run"]["run_id"] == run.name
    assert {item["status"] for item in list_catalog(roots)} == {"staged", "promoted"}
    assert all(
        path.stat().st_mode & 0o777 == 0o444
        for path in Path(entry["path"]).rglob("*")
        if path.is_file()
    )
    with pytest.raises(ValueError, match="already promoted"):
        promote(staged["staged_id"], roots)


def test_script_device_has_no_builtin_checks_and_remains_readable_without_its_source(tmp_path):
    gate = {
        "system_id": "two_level_reference",
        "inputs": {
            name: ROOT / "examples/two-level" / name for name in ("system.toml", "operators.npz")
        },
    }
    root = tmp_path / "runs"
    run = completed(root, gate, "script", {}, kind="script")
    staged = register_result(run, "device", claim="authored model")
    entry = promote(staged["staged_id"], root)
    assert entry["checks"] == entry["checks_not_run"] == entry["unusable_evidence"] == []
    shutil.rmtree(run)
    assert read_entry(entry["entry_id"], root)["lineage"]["source_run"]["status"] == "unavailable"
    assert (root / entry["device"]["path"] / "operators.npz").is_file()


@pytest.mark.parametrize(
    "name", ["integrity", "segments", "spectral_refinement", "replay", "constraints"]
)
def test_every_covering_blocking_failure_refuses_without_publishing(name, tmp_path, control_files):
    if name in {"replay", "constraints"}:
        run = control_run(control_files, tmp_path)
    elif name in {"integrity", "segments"}:
        run = sweep_run(tmp_path / "source")
    else:
        run = simulation(tmp_path / "source")
    fields = {"point": 0} if name in {"integrity", "segments"} else {}
    staged = register_result(run, "device", claim="candidate", **fields)
    item = outcome(name, "fail")
    if fields:
        item.update(
            coverage={"checked_points": [0]}, points=[{"index": 0, **outcome(name, "fail")}]
        )
    verification(run, [item])
    with pytest.raises(ValueError, match=f"{name} failed"):
        promote(staged["staged_id"], tmp_path / "runs")
    assert not (tmp_path / "runs/_catalog/entries").exists()


def test_a_later_pass_does_not_erase_a_failure(tmp_path):
    run = simulation(tmp_path / "source")
    staged = register_result(run, "device", claim="candidate")
    failed = verification(run, [outcome("spectral_refinement", "fail")])
    verification(run, [outcome("spectral_refinement")])
    with pytest.raises(ValueError, match=failed.name):
        promote(staged["staged_id"], tmp_path / "runs")


def test_sweep_uses_the_selected_point_outcome_and_leaves_unsampled_checks_missing(tmp_path):
    run = sweep_run(tmp_path / "source", expectation=True)
    verification(
        run,
        [
            outcome(
                "integrity",
                "fail",
                coverage={"checked_points": [0, 1]},
                points=[
                    {"index": 0, **outcome("integrity", "fail")},
                    {"index": 1, **outcome("integrity")},
                ],
            ),
            outcome(
                "segments",
                coverage={"checked_points": [0]},
                points=[{"index": 0, **outcome("segments")}],
            ),
            outcome(
                "declared_expectation",
                "fail",
                coverage={"checked_points": [0, 1]},
                failure_meaning="declared expectation violated",
            ),
        ],
    )
    staged = register_result(run, "device", claim="second point", point=1)
    entry = promote(staged["staged_id"], tmp_path / "runs")
    assert [(item["name"], item["outcome"]) for item in entry["checks"]] == [
        ("integrity", "pass"),
        ("declared_expectation", "fail"),
    ]
    assert "segments" in entry["checks_not_run"]


@pytest.mark.parametrize("expectation", [False, True])
def test_missing_sweep_checks_match_the_declared_expectation(tmp_path, expectation):
    run = sweep_run(tmp_path / "source", expectation=expectation)
    staged = register_result(run, "device", claim="second point", point=1)
    entry = promote(staged["staged_id"], run.parent.parent)
    expected = ["integrity", "segments", "diagnostics"]
    if expectation:
        expected.append("declared_expectation")
    assert entry["checks_not_run"] == expected


def test_missed_target_and_descriptive_checks_are_recorded(control_files, tmp_path):
    run = control_run(control_files, tmp_path)
    checked = verification(
        run,
        [
            outcome("target", "fail", failure_meaning="unmet declared target"),
            {
                "name": "uncertainty",
                "scope": "declared samples",
                "values": {"minimum_fidelity": 0.5},
            },
        ],
    )
    staged = register_result(run, "device_and_pulse", claim="useful starting pulse")
    entry = promote(staged["staged_id"], tmp_path / "runs")
    assert [item["outcome"] for item in entry["checks"]] == ["fail", "descriptive"]
    assert all(item["run"].endswith(checked.name) for item in entry["checks"])
    assert all(item["policy_version"] == "retained-test-policy" for item in entry["checks"])
    assert all(
        set(item) == {"name", "outcome", "run", "policy_version"} for item in entry["checks"]
    )
    assert (Path(entry["path"]) / "pulse.npz").read_bytes() == (
        Path(staged["path"]) / "pulse.npz"
    ).read_bytes()


def test_calibration_only_counts_its_selected_solution_and_allows_missed_coverage(tmp_path):
    run = oscillator_run(tmp_path, ["qa"], [5.0005])
    verification(run, [outcome("holdout", "fail", solution=1)], solution=1)
    checked = verification(
        run,
        [outcome("holdout", "fail", solution=0, failure_meaning="declared expectation violated")],
        solution=0,
    )
    staged = register_result(run, "device", claim="fitted oscillator", solution=0)
    entry = promote(staged["staged_id"], run.parent.parent)
    assert len(entry["checks"]) == 1
    assert entry["checks"][0]["run"].endswith(checked.name)
    assert entry["checks"][0]["outcome"] == "fail"


@pytest.mark.parametrize("protocol", ["sweep", "calibrate", "control"])
def test_inapplicable_requests_keep_their_outcome(tmp_path, control_files, protocol):
    if protocol == "sweep":
        run = sweep_run(tmp_path / "source")
        selection, name = {"point": 1}, "segments"
    elif protocol == "calibrate":
        run = oscillator_run(tmp_path, ["qa"], [5.0005])
        selection, name = {"solution": 0}, "holdout"
    else:
        run = control_run(control_files, tmp_path)
        selection, name = {}, "target"
    verification(
        run,
        [outcome(name, "inapplicable", reason="no prerequisite declared")],
        solution=selection.get("solution"),
    )
    staged = register_result(run, "device", claim="candidate", **selection)
    entry = promote(staged["staged_id"], run.parent.parent)
    assert len(entry["checks"]) == 1
    assert entry["checks"][0]["outcome"] == "inapplicable"
    assert name not in entry["checks_not_run"]


@pytest.mark.parametrize("state", ["failed", "active", "mismatched", "invalid"])
def test_unusable_attempt_is_retained_without_claiming_its_check_never_ran(tmp_path, state):
    run = simulation(tmp_path / "source")
    staged = register_result(run, "device", claim="candidate")
    checked = verification(
        run,
        [outcome("spectral_refinement", "fail")],
        status=state if state in {"failed", "active"} else "completed",
        matches=state != "mismatched",
    )
    if state == "invalid":
        result = _load_record(checked / "result.toml")
        result["report"]["checks"][0]["name"] = "wrong-report"
        publish_toml(checked / "result.toml", result, replace=True)
    entry = promote(staged["staged_id"], tmp_path / "runs")
    assert entry["checks"] == [] and entry["checks_not_run"] == []
    assert len(entry["unusable_evidence"]) == 1
    assert entry["unusable_evidence"][0]["run"].endswith(checked.name)
    assert entry["unusable_evidence"][0]["requested_checks"] == ["spectral_refinement"]


@pytest.mark.parametrize("changed", ["result", "artifact", "input", "removed"])
def test_promotion_refuses_changed_or_missing_source(tmp_path, changed):
    run = simulation(tmp_path / "source")
    staged = register_result(run, "device", claim="candidate")
    if changed == "removed":
        shutil.rmtree(run)
    else:
        path = {
            "result": run / "result.toml",
            "artifact": run / "artifacts/output.npz",
            "input": run / "spec/system.toml",
        }[changed]
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed|digest mismatch|unavailable"):
        promote(staged["staged_id"], tmp_path / "runs")
    assert not (tmp_path / "runs/_catalog/entries").exists()


def test_concurrent_promotions_publish_exactly_one_entry(tmp_path, monkeypatch):
    run = simulation(tmp_path / "source")
    staged = register_result(run, "device", claim="candidate")
    barrier = Barrier(2)
    publish = catalog._publish

    def racing_publish(*args, **kwargs):
        barrier.wait(timeout=10)
        return publish(*args, **kwargs)

    monkeypatch.setattr(catalog, "_publish", racing_publish)

    def attempt():
        try:
            return promote(staged["staged_id"], tmp_path / "runs")
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sum(isinstance(item, dict) for item in results) == 1
    assert "already promoted" in next(item for item in results if isinstance(item, str))
    assert len(list((tmp_path / "runs/_catalog/entries").iterdir())) == 1
    assert not list((tmp_path / "runs/_catalog").glob(".publish-*"))


def test_operator_cli_promotes_reads_and_refuses_a_second_promotion(tmp_path, capsys):
    run = simulation(tmp_path / "source")
    staged = register_result(run, "device", claim="candidate")
    args = [staged["staged_id"], "--runs-root", str(tmp_path / "runs")]
    assert main(["promote", *args]) == 0
    entry = json.loads(capsys.readouterr().out)
    assert main(["read-entry", entry["entry_id"], *args[1:]]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "promoted"
    assert main(["promote", *args]) == 64
    assert "already promoted" in capsys.readouterr().err


def test_promotion_reads_real_control_checks_without_running_them_again(candidate, tmp_path):
    run = retained_candidate(candidate, tmp_path)
    checked, _ = run_checks(run, "replay", "target", "constraints")
    staged = register_result(run, "device_and_pulse", claim="independently checked X gate")
    before = sorted(run.parent.iterdir())
    entry = promote(staged["staged_id"], run.parent.parent)
    assert sorted(run.parent.iterdir()) == before
    assert [item["outcome"] for item in entry["checks"]] == ["pass", "pass", "pass"]
    assert all(item["run"].endswith(checked.name) for item in entry["checks"])
