"""The example client waits for approval and recovers retained work across sessions."""

import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from entanglix.investigations import (
    read_investigation,
    record_user_decision,
    revise_investigation,
    stop_investigation,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/readout/investigate.py"
MODULE = runpy.run_path(str(EXAMPLE))
Client = MODULE["Client"]
draft = MODULE["draft"]


@pytest.mark.parametrize("case", ["normal", "early-stop"])
def test_draft_has_five_bases_and_cannot_self_start(tmp_path, case):
    workspace = tmp_path / "client"
    record = draft(workspace, runs_root=tmp_path / "runs", case=case, checkpoint=10)
    assert record["state"] == "draft"
    terms = record["revisions"][0]["terms"]
    assert terms["limits"] == {}
    assert f"{MODULE['WIDTH']:g} GHz" in terms["question"]
    requirements = " ".join(terms["evidence_requirements"]["requirements"])
    assert (f"{MODULE['TIGHT_WIDTH']:g} GHz" in requirements) == (case == "early-stop")
    assert len(terms["scope"]["bases"]) == 5
    assert "rwa" not in terms["evidence_requirements"]["checks"]  # requests are conditional on role
    for base in terms["scope"]["bases"]:
        if base["name"].startswith("refined-"):
            assert [item["target"] for item in base["variations"]] == [MODULE["AMPLITUDE"]]
    with pytest.raises(RuntimeError, match="User decision needed: state=draft"):
        Client(workspace).run()
    assert read_investigation(Path(record["path"]))["events"] == []
    assert not list((tmp_path / "runs").glob("*/*/run.toml"))


@pytest.mark.parametrize("state", ["stopped", "review"])
def test_run_command_reports_needed_decision_without_traceback(tmp_path, state):
    workspace = tmp_path / "client"
    record = draft(workspace, runs_root=tmp_path / "runs", case="normal", checkpoint=10)
    investigation = Path(record["path"])
    record_user_decision(investigation, "start", revision=1)
    if state == "stopped":
        stop_investigation(investigation, "user_cancellation")
    else:
        Client(workspace).finding(
            {"note": "A decision is needed before proceeding."},
            decision="review",
            review="precision",
        )
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "run", str(workspace)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert not result.stdout
    assert result.stderr == (
        f"User decision needed: state={'stopped' if state == 'stopped' else 'active'}, "
        f"reviews={[] if state == 'stopped' else ['precision']}; "
        "continue with the same run command after the operator decision.\n"
    )
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("stopped", [False, True])
def test_run_command_rechecks_state_after_admission_error(tmp_path, monkeypatch, capsys, stopped):
    workspace = tmp_path / "client"
    record = draft(workspace, runs_root=tmp_path / "runs", case="normal", checkpoint=10)
    investigation = Path(record["path"])
    record_user_decision(investigation, "start", revision=1)

    def refused(self):
        if stopped:
            stop_investigation(self.path, "user_cancellation")
        raise ValueError("investigation is stopped" if stopped else "unrelated failure")

    monkeypatch.setattr(Client, "run", refused)
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE), "run", str(workspace)])
    if stopped:
        with pytest.raises(SystemExit) as error:
            MODULE["main"]()
        assert error.value.code == 1
        assert "state=stopped, reviews=[]" in capsys.readouterr().err
    else:
        with pytest.raises(ValueError, match="unrelated failure"):
            MODULE["main"]()
        assert not capsys.readouterr().err


@pytest.mark.parametrize("during_job", [False, True])
def test_new_session_recovers_seed_and_checked_evidence(tmp_path, runner_repository, during_job):
    workspace = tmp_path / "client"
    record = draft(workspace, runs_root=tmp_path / "runs", case="normal", checkpoint=10)
    investigation = Path(record["path"])
    record_user_decision(investigation, "start", revision=1)
    ready = tmp_path / "ready"
    code = (
        "import runpy,time; from pathlib import Path; "
        f"client=runpy.run_path({str(EXAMPLE)!r})['Client'](Path({str(workspace)!r}), "
        f"repository=Path({str(runner_repository)!r})); client.seed(); "
        f"Path({str(ready)!r}).write_text('ready'); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:

        def is_ready():
            if during_job:
                return list((tmp_path / "runs").glob("*/*/runner/started.toml"))
            return ready.exists()

        deadline = time.monotonic() + 90
        while not is_ready() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.05)
        if not is_ready():
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"seed session failed: {stdout.decode()} {stderr.decode()}")
        os.kill(process.pid, signal.SIGINT if during_job else signal.SIGKILL)
        process.communicate(timeout=10)
        before = read_investigation(investigation)["usage"]
        client = Client(workspace, repository=runner_repository)
        failed_id = next(iter(client.failed.values())).name if during_job else None
        client.seed()
        after = read_investigation(investigation)["usage"]
        if during_job:
            assert after["jobs"] == 3  # aborted sweep, linked retry, checks
            assert client.records[client.seed_run.name][1]["manifest"]["parent_run_ids"] == [
                failed_id
            ]
            assert not after["unsettled_jobs"]
        else:
            assert after == before
            assert after["jobs"] == 2
            assert after["evaluated_points"] == 9
        client.seed()
        assert read_investigation(investigation)["usage"] == after
        assert len(client.samples) == 9
        client.finding({"seed": client.seed_run.name, "samples": list(client.samples.values())})
        assert read_investigation(investigation)["events"][-1]["finding"]["runs"]
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


@pytest.mark.parametrize("no_crossing_ej", [12.0, 15.0, None])
def test_policy_confirms_normal_results_before_precision_handoff(
    monkeypatch, tmp_path, no_crossing_ej
):
    """A known nonlinear response exposes the difference between width and confirmation."""

    class ResponseClient(Client):
        def __init__(self):
            self.case = "early-stop"
            self.path = tmp_path
            self.samples, self.records = {}, {}
            self.comparisons, self.findings = [], []

        def before_job(self):
            pass

        def seed(self):
            for ej in MODULE["FREQUENCIES"]:
                for amplitude in (0.0, 0.001, 0.002):
                    self.solve(ej, amplitude, "rwa", "seed")

        def checked(self, source, names):
            pass

        def solve(self, ej, amplitude, profile, parent):
            coefficient = 1000 if ej == no_crossing_ej else {12.0: 3000, 13.5: 5000, 15.0: 6000}[ej]
            loss = coefficient * amplitude**2
            if profile != "rwa" and amplitude:
                loss += 0.00025
            if profile == "lab_refined" and amplitude:
                loss += 1e-7
            sample = {
                "amplitude_ghz": amplitude,
                "coherence": 1 - loss,
                "source_run_id": f"{ej}-{amplitude}-{profile}",
            }
            self.samples[ej, amplitude, profile] = sample
            self.records[sample["source_run_id"]] = (tmp_path, {})
            return sample

        def compare(self, ej, bracket, stage):
            self.comparisons.append((ej, stage))
            return super().compare(ej, bracket, stage)

        def finding(self, learned, **kwargs):
            self.findings.append((learned, kwargs))

    from datetime import UTC, datetime

    monkeypatch.setitem(
        Client.run.__globals__,
        "read_investigation",
        lambda _: {
            "events": [{"kind": "decision", "action": "start", "recorded_at": datetime.now(UTC)}],
            "usage": {"jobs": 41},
        },
    )
    stops = []
    monkeypatch.setitem(
        Client.run.__globals__, "stop_investigation", lambda _, reason, **kw: stops.append(reason)
    )
    client = ResponseClient()
    result = client.run()
    crossings = [ej for ej in MODULE["FREQUENCIES"] if ej != no_crossing_ej]
    assert client.comparisons == [*((ej, "normal") for ej in crossings), (crossings[0], "tighter")]
    assert all(item["confirmed"] for item in result["brackets"][:-1])
    assert not result["brackets"][-1]["confirmed"]
    assert result["unattempted"] == [{"ej_ghz": ej, "stage": "tighter"} for ej in crossings[1:]]
    assert [item["ej_ghz"] for item in result["no_crossing"]] == (
        [] if no_crossing_ej is None else [no_crossing_ej]
    )
    wider_scope_notes = [
        note
        for note in client.findings[-1][1]["unresolved"]
        if "crossing needs wider scope" in note
    ]
    assert wider_scope_notes == (
        [] if no_crossing_ej is None else [f"EJ={no_crossing_ej:g} crossing needs wider scope."]
    )
    assert client.findings[-1][1]["review"] == "precision"
    assert stops == ["review_handoff"]


def test_confirmed_interpretation_is_reused_in_a_new_session(tmp_path):
    workspace = tmp_path / "client"
    record = draft(workspace, runs_root=tmp_path / "runs", case="normal", checkpoint=10)
    investigation = Path(record["path"])
    record_user_decision(investigation, "start", revision=1)
    learned = {"ej_ghz": 13.5, "stage": "normal", "confirmed": True}
    first = Client(workspace).finding(learned, kind="interpretation")
    before = read_investigation(investigation)["events"]
    repeated = Client(workspace).finding(learned, kind="interpretation")
    assert repeated == first
    assert read_investigation(investigation)["events"] == before
    # An approved revision gets its own interpretation even when the values agree.
    terms = record["revisions"][0]["terms"]
    terms["question"] += " Report the retained result under the renewed terms."
    revise_investigation(investigation, terms)
    record_user_decision(investigation, "approve-revision", revision=2)
    renewed = Client(workspace).finding(learned, kind="interpretation")
    assert renewed["revision"] == 2
    assert renewed["event_id"] != first["event_id"]


@pytest.mark.parametrize("no_crossing", [[], [{"ej_ghz": 15.0}]])
def test_report_next_experiment_follows_observed_no_crossing(tmp_path, no_crossing):
    workspace = tmp_path / "client"
    record = draft(workspace, runs_root=tmp_path / "runs", case="normal", checkpoint=10)
    record_user_decision(Path(record["path"]), "start", revision=1)
    finding = Client(workspace).finding(
        {"no_crossing": no_crossing}, kind="report", decision="stop"
    )
    assert finding["finding"]["next_experiment"] == (
        "Ask for a wider range if crossings are needed at EJ=15 GHz."
        if no_crossing
        else "No further experiment is required by this policy."
    )
