"""User-owned terms, retained findings, and concurrent investigation publications."""

import copy
import hashlib
import json
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
import tomli_w

from entanglix import investigations
from entanglix.corrector import CHECK_DEFINITIONS
from entanglix.investigations import (
    create_investigation,
    list_investigations,
    read_investigation,
    record_finding,
    record_user_decision,
    revise_investigation,
)
from entanglix.runner import list_capabilities, publish_toml


@pytest.fixture
def terms():
    return {
        "format_version": 1,
        "question": "Which amplitudes bracket the declared coherence loss?",
        "assumptions": ["Coherence decreases along the sampled line."],
        "scope": {
            "operations": ["simulate", "check"],
            "bases": [
                {
                    "name": "rwa",
                    "inputs": {"system.toml": hashlib.sha256(b"model").hexdigest()},
                    "variations": [
                        {
                            "target": "evolution.drive_amplitudes_ghz.readout",
                            "range": [0.0, 0.002],
                        }
                    ],
                }
            ],
        },
        "evidence_requirements": {
            "requirements": ["Report the bracket and its limitations."],
            "checks": {"rwa": ["integrity", "segments"]},
        },
        "checkpoint": {"finished_attempts": 10},
        "review_points": {"first-results": 20},
        "review_criteria": ["A wider amplitude range needs review."],
        "stop_criteria": ["Stop when the allowed method cannot establish the claim."],
        "limits": {},
    }


def finding(**changes):
    return {
        "kind": "assessment",
        "runs": [],
        "learned": "The base and evidence plan are recorded.",
        "unresolved": ["No computation is linked yet."],
        "decision": "continue",
        "rationale": "The first experiment remains in scope.",
        "next_experiment": "Run the seed grid.",
        **changes,
    }


def started(terms, tmp_path):
    path = Path(create_investigation(terms, runs_root=tmp_path)["path"])
    record_user_decision(path, "start", revision=1)
    return path


def test_draft_start_proposal_and_approval_preserve_history(terms, tmp_path):
    draft = create_investigation(terms, runs_root=tmp_path)
    path = Path(draft["path"])
    original = (path / "revisions/0001.toml").read_bytes()
    assert draft["state"] == "draft"
    assert draft["active_revision"] is None
    assert draft["events"] == []
    with pytest.raises(ValueError, match="before its start"):
        record_finding(path, finding())
    with pytest.raises(ValueError, match="must be started"):
        record_user_decision(path, "approve-revision", revision=1)
    active = record_user_decision(path, "start", revision=1, note="Proceed with these terms.")
    assert active["active_revision"] == 1
    source = active["events"][0]["source"]
    assert source["kind"] == "operator-cli" and source["os_user"]
    decision_bytes = (path / "events/00000001.toml").read_bytes()
    changed = copy.deepcopy(terms)
    changed["evidence_requirements"]["requirements"] = ["A different evidence requirement."]
    changed["limits"] = {"job_count": 60}
    assert revise_investigation(path, changed)["proposed_revision"] == 2
    assert read_investigation(path)["active_revision"] == 1
    first = record_finding(path, finding())
    assert first["revision"] == 1
    approved = record_user_decision(path, "approve-revision", revision=2)
    assert approved["active_revision"] == 2
    approved_terms = approved["revisions"][1]["terms"]
    assert approved_terms["evidence_requirements"] == changed["evidence_requirements"]
    assert approved_terms["limits"] == {"job_count": 60}
    assert record_finding(path, finding())["revision"] == 2
    assert (path / "revisions/0001.toml").read_bytes() == original
    assert (path / "events/00000001.toml").read_bytes() == decision_bytes
    assert list_investigations(tmp_path)[0]["active_revision"] == 2
    with pytest.raises(ValueError, match="already started"):
        record_user_decision(path, "start", revision=2)
    with pytest.raises(ValueError, match="already taken effect"):
        record_user_decision(path, "approve-revision", revision=1)


@pytest.mark.parametrize(
    "section,value,diagnostic",
    [
        ("approved", True, "Extra inputs"),
        ("checkpoint", {}, "exactly one"),
        ("checkpoint", {"finished_attempts": True}, "valid integer"),
        ("checkpoint", {"finished_attempts": 0}, "greater than 0"),
        ("limits", {"cpu_seconds": 10}, "Extra inputs"),
        ("limits", {"gpu_seconds": float("inf")}, "finite number"),
        ("limits", {"deadline": "2026-10-04T12:00:00"}, "UTC offset"),
        ("limits", {"deadline": 100}, "absolute timestamp"),
        ("limits", {"deadline": "1700000000"}, "absolute timestamp"),
    ],
)
def test_invalid_terms_do_not_create_a_draft(terms, tmp_path, section, value, diagnostic):
    terms[section] = value
    with pytest.raises(ValueError, match=diagnostic):
        create_investigation(terms, runs_root=tmp_path)
    assert not (tmp_path / "_investigations").exists()


@pytest.mark.parametrize(
    "deadline", ["2026-10-04T12:00:00Z", datetime(2026, 10, 4, 12, tzinfo=UTC)]
)
def test_absolute_deadline_round_trips(terms, tmp_path, deadline):
    terms["limits"] = {"deadline": deadline}
    draft = create_investigation(terms, runs_root=tmp_path)
    assert draft["revisions"][0]["terms"]["limits"]["deadline"] == datetime(
        2026, 10, 4, 12, tzinfo=UTC
    )


def test_operator_decision_without_passwd_entry(terms, tmp_path, monkeypatch):
    path = Path(create_investigation(terms, runs_root=tmp_path)["path"])

    def missing_user(uid):
        raise KeyError(uid)

    monkeypatch.setattr(investigations.pwd, "getpwuid", missing_user)
    record = record_user_decision(path, "start", revision=1)
    assert record["events"][0]["source"]["os_user"] == str(investigations.os.getuid())


def test_base_input_conflicts_are_rejected(terms, tmp_path):
    inputs = terms["scope"]["bases"][0]["inputs"]
    inputs.update({"a": inputs["system.toml"], "a/b": inputs["system.toml"]})
    with pytest.raises(ValueError, match="file/directory conflicts"):
        create_investigation(terms, runs_root=tmp_path)
    assert not (tmp_path / "_investigations").exists()


def test_checkpoint_and_limits_are_explicit(terms, tmp_path):
    del terms["checkpoint"]
    with pytest.raises(ValueError, match="checkpoint"):
        create_investigation(terms, runs_root=tmp_path)
    terms["checkpoint"] = {"elapsed_seconds": 60.0}
    del terms["limits"]
    with pytest.raises(ValueError, match="limits"):
        create_investigation(terms, runs_root=tmp_path)
    terms["limits"] = {}
    draft = create_investigation(terms, runs_root=tmp_path)
    assert draft["revisions"][0]["terms"]["limits"] == {}


@pytest.mark.parametrize("target", ["model.cutoff", "parameters.", "evolution.frame.other.q"])
def test_invalid_scope_targets_are_rejected(terms, tmp_path, target):
    terms["scope"]["bases"][0]["variations"][0]["target"] = target
    with pytest.raises(ValueError, match="sweep target syntax"):
        create_investigation(terms, runs_root=tmp_path)


def test_linked_parameter_values_and_named_checks(terms, tmp_path):
    variations = terms["scope"]["bases"][0]["variations"]
    variations.append(
        {
            "target": "parameters.EJ",
            "values": [12.0, 15.0],
            "linked_values": {"evolution.carriers.readout.frequency_ghz": [7.001, 7.003]},
        }
    )
    create_investigation(terms, runs_root=tmp_path)
    variations[-1]["linked_values"]["evolution.carriers.readout.frequency_ghz"] = [7.0]
    with pytest.raises(ValueError, match="one value per entry"):
        create_investigation(terms, runs_root=tmp_path)
    variations.pop()
    terms["evidence_requirements"]["checks"] = {"missing": ["integrity"]}
    with pytest.raises(ValueError, match="unknown bases"):
        create_investigation(terms, runs_root=tmp_path)


def test_requested_check_names_follow_corrector_definitions(terms, tmp_path):
    terms["evidence_requirements"]["checks"] = {"rwa": list(CHECK_DEFINITIONS)}
    create_investigation(terms, runs_root=tmp_path)
    terms["evidence_requirements"]["checks"]["rwa"].append("missing")
    with pytest.raises(ValueError, match="unknown requested checks"):
        create_investigation(terms, runs_root=tmp_path)


def test_finding_membership_comes_from_manifests(terms, tmp_path):
    path = started(terms, tmp_path)
    run = tmp_path / "system/r-example"
    run.mkdir(parents=True)
    publish_toml(
        run / "run.toml",
        {
            "record_version": 1,
            "run_id": run.name,
            "protocol": "script",
            "investigation_id": path.name,
        },
    )
    event = record_finding(path, finding(runs=[run.name]))
    assert event["finding"]["runs"] == [run.name]
    assert "evidence" not in event
    for supplied in ("revision", "evidence", "usage"):
        with pytest.raises(ValueError, match="Extra inputs"):
            record_finding(path, finding(**{supplied: {}}))
    for name in ("missing", "../system/r-example"):
        with pytest.raises(ValueError, match="unknown run references"):
            record_finding(path, finding(runs=[name]))
    foreign = tmp_path / "system/r-foreign"
    foreign.mkdir()
    publish_toml(
        foreign / "run.toml",
        {
            "record_version": 1,
            "run_id": foreign.name,
            "protocol": "script",
            "investigation_id": "another-investigation",
        },
    )
    with pytest.raises(ValueError, match="does not belong"):
        record_finding(path, finding(runs=[foreign.name]))
    assert len(read_investigation(path)["events"]) == 2


def _concurrent_append(arguments):
    path, terms, index = arguments
    proposed = revise_investigation(path, {**terms, "question": f"Proposal {index}"})
    event = record_finding(path, finding(learned=f"Finding {index}"))
    return proposed["proposed_revision"], event["event_id"]


def test_concurrent_processes_preserve_every_revision_and_event(terms, tmp_path):
    path = started(terms, tmp_path)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        outputs = list(pool.map(_concurrent_append, [(path, terms, i) for i in range(12)]))
    record = read_investigation(path)
    assert sorted(revision for revision, _ in outputs) == list(range(2, 14))
    assert len({event for _, event in outputs}) == 12
    assert len(record["revisions"]) == 13
    assert len(record["events"]) == 13
    assert record["active_revision"] == 1
    assert {event["finding"]["learned"] for event in record["events"][1:]} == {
        f"Finding {i}" for i in range(12)
    }


def _try_start(path):
    try:
        record_user_decision(path, "start", revision=1)
    except ValueError as exc:
        return str(exc)
    return "started"


def test_only_one_concurrent_start_becomes_effective(terms, tmp_path):
    path = Path(create_investigation(terms, runs_root=tmp_path)["path"])
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        outcomes = list(pool.map(_try_start, [path, path]))
    assert sorted(outcomes) == ["investigation has already started", "started"]
    assert len(read_investigation(path)["events"]) == 1


def test_cli_workflow_and_new_process_read(terms, tmp_path):
    spec = tmp_path / "investigation.toml"
    spec.write_text(tomli_w.dumps(terms))

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "entanglix", *map(str, args)], capture_output=True, text=True
        )

    draft = cli("create-investigation", spec, "--runs-root", tmp_path)
    assert draft.returncode == 0, draft.stderr
    path = json.loads(draft.stdout)["path"]
    assert cli("investigation-decision", path, "start", "--revision", 1).returncode == 0
    assert cli("revise-investigation", path, spec).returncode == 0
    assert cli("investigation-decision", path, "approve-revision", "--revision", 2).returncode == 0
    authored = tmp_path / "finding.toml"
    authored.write_text(tomli_w.dumps(finding()))
    assert cli("record-finding", path, authored).returncode == 0
    record = cli("read-investigation", path)
    assert record.returncode == 0, record.stderr
    assert json.loads(record.stdout)["active_revision"] == 2
    assert len(json.loads(record.stdout)["events"]) == 3
    listing = cli("list-investigations", "--runs-root", tmp_path)
    assert json.loads(listing.stdout)[0]["investigation_id"] == Path(path).name
    failed = cli("investigation-decision", path, "approve-revision", "--revision", 99)
    assert failed.returncode == 64
    assert "unknown revision: 99" in failed.stderr and "Traceback" not in failed.stderr
    capabilities = list_capabilities()
    assert "record_finding" in capabilities["operations"]
    assert "record_user_decision" not in capabilities["operations"]
