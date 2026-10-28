"""Calibration jobs inside an investigation: exact bases, scope refusals, reruns and findings."""

import tomllib
from pathlib import Path

import pytest
import tomli_w
from test_calibrate import write_spec
from test_calibration_checks import split_points

from entanglix.corrector.verification import check_run
from entanglix.investigations import (
    create_investigation,
    read_investigation,
    record_finding,
    record_user_decision,
)
from entanglix.mcp_views import attempt_data
from entanglix.protocols.calibrate import calibrate, load_calibration
from entanglix.runner import read_run, rerun


def started(protocol, root):
    """A started investigation whose one base is exactly the given calibration."""
    terms = {
        "format_version": 1,
        "question": "Fit the device parameters and evaluate them on withheld data.",
        "assumptions": ["Labelled transitions follow the declared family."],
        "scope": {
            "operations": ["calibrate", "check"],
            "bases": [{"name": "fit", "inputs": load_calibration(protocol).input_digests}],
        },
        "evidence_requirements": {
            "requirements": ["Report holdout coverage and identifiability."],
            "checks": {"fit": ["holdout", "identifiability"]},
        },
        "checkpoint": {"finished_attempts": 100},
        "review_criteria": [],
        "stop_criteria": [],
        "limits": {},
    }
    path = Path(create_investigation(terms, runs_root=root)["path"])
    record_user_decision(path, "start", revision=1)
    return path


@pytest.fixture
def calibration_study(runner_repository, tmp_path):
    """A started investigation whose base is an authored calibration over a training split."""
    split_points(tmp_path, 22)
    spec = write_spec(tmp_path, ["training.toml"])
    # The base's authored names and bytes must survive an investigation rerun.
    (tmp_path / "system.toml").rename(tmp_path / "device.toml")
    document = tomli_w.dumps({**tomllib.loads(spec.read_text()), "system": "device.toml"})
    spec.unlink()
    protocol = tmp_path / "fit.toml"
    protocol.write_text("# Authored base, not canonical TOML\n" + document)
    root = tmp_path / "runs"
    return protocol, root, started(protocol, root), runner_repository


def launch(study, **options):
    protocol, root, path, repository = study
    return calibrate(
        protocol,
        runs_root=root,
        repository=repository,
        investigation_id=path.name,
        base="fit",
        **options,
    )


def test_an_exact_base_admits_calibrate_and_reruns_keep_membership(calibration_study):
    protocol, root, path, repository = calibration_study
    assert launch(calibration_study, dry_run=True)["preview"]["protocol"] == "calibrate"
    assert read_investigation(path)["usage"]["jobs"] == 0
    run = Path(launch(calibration_study)["path"])
    source = read_run(run)
    assert source["status"] == "completed", source["log_tail"]
    assert source["manifest"]["investigation_id"] == path.name
    assert source["manifest"]["investigation_base"] == "fit"
    assert (run / "spec/base/calibrate.toml").read_bytes() == protocol.read_bytes()
    assert (run / "spec/base/device.toml").read_bytes() == (
        protocol.parent / "device.toml"
    ).read_bytes()
    # The fitting job holds the training split only.
    assert "holdout.toml" not in source["manifest"]["spec_sha256"]

    replay = rerun(run, repository=repository)
    repeated = read_run(Path(replay["path"]))
    assert repeated["status"] == "completed", repeated["log_tail"]
    assert repeated["manifest"]["parent_run_ids"] == [run.name]
    assert repeated["manifest"]["investigation_id"] == path.name
    assert repeated["result"] == source["result"]

    checked = check_run(
        run, ("holdout",), repository=repository, holdout=protocol.parent / "holdout.toml"
    )
    assert read_run(Path(checked["path"]))["manifest"]["investigation_id"] == path.name
    row = next(
        dict(zip(columns, row, strict=True))
        for columns, rows in [attempt_data(read_investigation(path))]
        for row in rows
        if Path(row[columns.index("path")]).name == run.name
    )
    assert row["outcome"] == "solution"
    assert row["checks"][0][:3] == [Path(checked["path"]).name, True, [["holdout", None, 0]]]
    event = record_finding(
        path,
        {
            "kind": "assessment",
            "runs": [run.name, Path(checked["path"]).name],
            "learned": "The fit and its holdout check are retained.",
            "unresolved": [],
            "decision": "continue",
            "rationale": "Identifiability is still requested.",
            "next_experiment": "Request identifiability.",
        },
    )
    evidence = {item["run_id"]: item for item in event["evidence"]}[run.name]
    assert [(check["name"], check["solution"]) for check in evidence["checks"]] == [("holdout", 0)]
    assert evidence["requested_checks_not_run"] == ["identifiability"]
    assert evidence["relevant_checks_not_run"] == ["identifiability", "refinement"]
    assert read_investigation(path)["usage"]["jobs"] == 3


def test_changed_inputs_and_unknown_bases_are_refused_without_an_attempt(calibration_study):
    protocol, root, path, _ = calibration_study
    with pytest.raises(ValueError, match="unknown investigation base: missing"):
        calibrate(
            protocol,
            runs_root=root,
            repository=calibration_study[3],
            investigation_id=path.name,
            base="missing",
        )
    training = protocol.parent / "training.toml"
    training.write_text(training.read_text() + "\n")
    with pytest.raises(
        ValueError, match="base input digests do not match: fit; changed digests: training.toml"
    ):
        launch(calibration_study)
    with pytest.raises(ValueError, match="base input digests do not match"):
        launch(calibration_study, dry_run=True)
    assert list(root.glob("*/*/run.toml")) == []
    assert read_investigation(path)["usage"]["jobs"] == 0


def test_a_system_named_like_the_spec_is_refused_for_an_investigation(calibration_study, tmp_path):
    _, root, _, repository = calibration_study
    directory = tmp_path / "renamed"
    directory.mkdir()
    split_points(directory, 22)
    spec = write_spec(directory, ["training.toml"])
    document = tomllib.loads(spec.read_text())
    # The system takes the name the runner gives the calibration spec in a base.
    (directory / "system.toml").rename(spec)
    protocol = directory / "fit.toml"
    protocol.write_text(tomli_w.dumps({**document, "system": "calibrate.toml"}))
    # A standalone launch retains no base, so the name does not collide.
    preview = calibrate(protocol, runs_root=root, repository=repository, dry_run=True)
    assert preview["preview"]["protocol"] == "calibrate"
    # The approved base matches these inputs exactly, so only the collision refuses them.
    exact = started(protocol, root)
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="the system file 'calibrate.toml' would replace"):
            calibrate(
                protocol,
                runs_root=root,
                repository=repository,
                investigation_id=exact.name,
                base="fit",
                dry_run=dry_run,
            )
    assert list(root.glob("*/*/run.toml")) == []
    assert read_investigation(exact)["usage"]["jobs"] == 0
