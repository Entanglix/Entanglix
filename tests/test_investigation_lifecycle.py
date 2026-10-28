"""Record-derived checkpoints, operator reviews, and permanent attempt cancellation."""

import copy
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from test_investigation_admission import admission as admission
from test_investigation_admission import approve, assessment, launch
from test_investigation_admission import study as study

from entanglix import investigations
from entanglix.corrector.verification import check_run
from entanglix.investigations import (
    list_investigations,
    read_investigation,
    record_finding,
    record_user_decision,
    revise_investigation,
    stop_investigation,
)
from entanglix.runner import (
    _execute_job,
    _load_record,
    abort_run,
    create_run,
    list_capabilities,
    read_run,
    record_finish,
    record_reconciliation,
    record_start,
)


def revise(admission, **changes):
    terms = copy.deepcopy(admission[1]) | changes
    revision = revise_investigation(admission[2], terms)["proposed_revision"]
    return record_user_decision(admission[2], "approve-revision", revision=revision)


def finish(kwargs, status="failed", code=64):
    with create_run(**kwargs) as owner:
        if status == "completed":
            record_start(owner, process_group=2_000_000_000)
        record_finish(owner, status=status, exit_code=code, wall_seconds=0, gpu_seconds=0)
        return owner.path


@pytest.mark.parametrize("status,code", [("completed", 0), ("failed", 64), ("aborted", 130)])
def test_finished_outcomes_gate_until_assessment(admission, status, code):
    kwargs, _, path = admission
    revise(admission, checkpoint={"finished_attempts": 1})
    run = finish(kwargs, status, code)
    record = read_investigation(path)
    assert record["checkpoint"]["due"] and record["checkpoint"]["finished_attempts"] == 1
    with pytest.raises(ValueError, match="checkpoint is due"):
        create_run(**kwargs)
    record_finding(path, assessment([run.name]) | {"kind": "interpretation"})
    with pytest.raises(ValueError, match="checkpoint is due"):
        create_run(**kwargs)
    record_finding(path, assessment([run.name]))
    assert not read_investigation(path)["checkpoint"]["due"]
    with create_run(**kwargs):
        pass
    assert read_investigation(path)["usage"]["jobs"] == 2


def test_reconciliation_counts_once_and_preserves_uncertain_usage(admission):
    kwargs, _, path = admission
    revise(
        admission,
        checkpoint={"finished_attempts": 2},
        review_points={"recovery": 2},
        limits={"job_seconds": 1.0},
    )
    finish(kwargs)
    with create_run(**(kwargs | {"launch": {"timeout_seconds": 0.25}})) as interrupted:
        lost = interrupted.path
    before = read_investigation(path)
    assert before["checkpoint"]["finished_attempts"] == 1 and not before["checkpoint"]["due"]
    assert before["usage"]["committed_job_seconds"] == 0.25
    record_reconciliation(lost, last_observed_state="runner lost before start")
    after = read_investigation(path)
    assert after["checkpoint"]["finished_attempts"] == 2 and after["checkpoint"]["due"]
    assert after["usage"] == before["usage"]
    assert after["usage"]["unsettled_jobs"] == 1
    assert not (lost / "FINISHED").exists()
    assert read_investigation(path)["checkpoint"]["finished_attempts"] == 2
    assert after["open_reviews"] == ["recovery"]
    with pytest.raises(ValueError, match="open reviews"):
        create_run(**kwargs)
    released = record_user_decision(path, "release-review", review_name="recovery")
    assert released["events"][-1]["finished_attempts"] == 2
    with pytest.raises(ValueError, match="checkpoint is due"):
        create_run(**kwargs)
    finding = record_finding(path, assessment([lost.name]))
    assert finding["finished_attempts"] == 2
    checkpoint = read_investigation(path)["checkpoint"]
    assert checkpoint["since_assessment"] == 0 and not checkpoint["due"]
    assert read_investigation(path)["usage"] == before["usage"]


def test_real_verification_job_counts_toward_checkpoint(study, runner_repository):
    terms = copy.deepcopy(study[2])
    terms["checkpoint"] = {"finished_attempts": 2}
    revision = revise_investigation(study[4], terms)["proposed_revision"]
    record_user_decision(study[4], "approve-revision", revision=revision)
    source = launch(study, runner_repository)
    assert source["exit_code"] == 0
    checked = check_run(Path(source["path"]), ("integrity",), repository=runner_repository)
    assert checked["exit_code"] == 0
    assert read_investigation(study[4])["checkpoint"]["finished_attempts"] == 2
    with pytest.raises(ValueError, match="checkpoint is due"):
        launch(study, runner_repository, dry_run=True)
    record_finding(study[4], assessment([Path(source["path"]).name]))
    assert launch(study, runner_repository, dry_run=True)["exit_code"] == 0


def test_elapsed_checkpoint_is_reset_only_by_assessment(admission, monkeypatch):
    kwargs, _, path = admission
    revise(admission, checkpoint={"elapsed_seconds": 10.0})
    start = read_investigation(path)["events"][0]["recorded_at"]

    class Clock(datetime):
        current = start + timedelta(seconds=11)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(investigations, "datetime", Clock)
    with pytest.raises(ValueError, match="checkpoint is due"):
        create_run(**kwargs)
    record_finding(path, assessment([]) | {"kind": "report"})
    assert read_investigation(path)["checkpoint"]["due"]
    record_finding(path, assessment([]))
    with create_run(**kwargs):
        pass
    Clock.current += timedelta(seconds=11)
    assert read_investigation(path)["checkpoint"]["due"]


def test_named_reviews_block_until_operator_release_and_can_reopen(admission):
    kwargs, _, path = admission
    record_finding(path, assessment([]) | {"decision": "review", "review_name": "strategy"})
    assert read_investigation(path)["open_reviews"] == ["strategy"]
    with pytest.raises(ValueError, match="open reviews"):
        create_run(**kwargs)
    record_finding(path, assessment([]))
    assert read_investigation(path)["open_reviews"] == ["strategy"]
    revise(admission, limits={"job_count": 10})
    assert read_investigation(path)["open_reviews"] == ["strategy"]
    with pytest.raises(ValueError, match="review is not open"):
        record_user_decision(path, "release-review", review_name="other")
    record_user_decision(path, "release-review", review_name="strategy")
    with create_run(**kwargs):
        pass
    record_finding(path, assessment([]) | {"decision": "review", "review_name": "strategy"})
    assert read_investigation(path)["open_reviews"] == ["strategy"]


def test_declared_review_uses_total_finished_count_and_survives_assessments(admission):
    kwargs, _, path = admission
    revise(admission, checkpoint={"finished_attempts": 1}, review_points={"results": 2})
    with pytest.raises(ValueError, match="review is not open"):
        record_user_decision(path, "release-review", review_name="results")
    finish(kwargs)
    record_finding(path, assessment([]))
    finish(kwargs)
    assert read_investigation(path)["open_reviews"] == ["results"]
    record_finding(path, assessment([]))
    with pytest.raises(ValueError, match="open reviews"):
        create_run(**kwargs)
    # Approving a revision that removes the trigger does not release an open review.
    revise(admission, checkpoint={"finished_attempts": 1}, review_points={})
    assert read_investigation(path)["open_reviews"] == ["results"]
    record_user_decision(path, "release-review", review_name="results")
    finish(kwargs)
    record_finding(path, assessment([]))
    assert read_investigation(path)["open_reviews"] == []
    with create_run(**kwargs):
        pass


@pytest.mark.parametrize("resume", [False, True])
def test_cancelled_admitted_run_never_starts_even_after_resume(admission, tmp_path, resume):
    kwargs, _, path = admission
    with create_run(**kwargs) as owner:
        stopped = stop_investigation(path, "user_cancellation")
        assert stopped["stop"]["runs"] == [owner.path.name]
        assert list_investigations(kwargs["runs_root"])[0]["state"] == "stopped"
        with pytest.raises(ValueError, match="investigation is stopped"):
            create_run(**kwargs)
        if resume:
            record_user_decision(path, "resume")
        result = _execute_job(
            owner,
            [sys.executable, str(tmp_path / "must-not-launch.py")],
            os.environ.copy(),
            timeout_seconds=None,
        )
        assert result["status"] == "aborted" and result["exit_code"] == 130
        assert not (owner.path / "runner/started.toml").exists()
        assert _load_record(owner.path / "FINISHED")["usage"]["wall_seconds"] == 0
    if resume:
        with create_run(**kwargs) as later:
            assert later.path != owner.path
        assert read_investigation(path)["usage"]["jobs"] == 2


def test_review_does_not_cancel_an_already_admitted_run(admission, tmp_path):
    kwargs, _, path = admission
    worker = tmp_path / "worker.py"
    worker.write_text("import sys\nassert sys.stdin.buffer.read(1) == b'1'\n")
    with create_run(**kwargs) as owner:
        record_finding(path, assessment([]) | {"decision": "review", "review_name": "strategy"})
        with pytest.raises(ValueError, match="open reviews"):
            create_run(**kwargs)
        result = _execute_job(
            owner,
            [sys.executable, str(worker)],
            os.environ.copy(),
            timeout_seconds=None,
            validate_result=lambda run: {},
        )
        assert result["status"] == "completed"


def test_stop_reports_indeterminate_job_liveness(admission, monkeypatch):
    with create_run(**admission[0]) as owner:
        monkeypatch.setattr(investigations, "_read_attempt", lambda run: {"status": "unknown"})
        result = stop_investigation(admission[2], "user_cancellation")
        assert result["cancellation_errors"] == [
            {"run_id": owner.path.name, "diagnostic": "cannot confirm job process-group liveness"}
        ]
        assert read_investigation(admission[2])["state"] == "stopped"


@pytest.mark.parametrize("stage", ["read", "abort"])
def test_stop_distinguishes_invalid_reads_from_completion_races(admission, monkeypatch, stage):
    import entanglix.runner as runner

    def inspect(run):
        if stage == "read":
            raise ValueError("invalid attempt record")
        return {"status": "active", "start": {}}

    def raced_abort(run):
        raise ValueError("a differently worded completion race")

    monkeypatch.setattr(investigations, "_read_attempt", inspect)
    monkeypatch.setattr(runner, "abort_run", raced_abort)
    with create_run(**admission[0]) as owner:
        result = stop_investigation(admission[2], "user_cancellation")
        assert result["cancellation_errors"] == (
            [{"run_id": owner.path.name, "diagnostic": "invalid attempt record"}]
            if stage == "read"
            else []
        )


def test_resume_requires_releases_and_preserves_usage_and_checkpoint(admission):
    kwargs = approve(admission, {"job_count": 4})
    path = admission[2]
    revise(admission, checkpoint={"finished_attempts": 1}, limits={"job_count": 4})
    finish(kwargs)
    record_finding(
        path, assessment([]) | {"kind": "report", "decision": "review", "review_name": "strategy"}
    )
    stop_investigation(path, "review_handoff", note="Need a different method.")
    with pytest.raises(ValueError, match="release open reviews"):
        record_user_decision(path, "resume")
    record_user_decision(path, "release-review", review_name="strategy")
    record_user_decision(path, "resume")
    with pytest.raises(ValueError, match="checkpoint is due"):
        create_run(**kwargs)
    record_finding(path, assessment([]))
    finish(kwargs)
    record = read_investigation(path)
    assert record["usage"]["jobs"] == 2
    # A new process sees the same state, gate and accounting; it does not replay jobs.
    child = subprocess.run(
        [sys.executable, "-m", "entanglix", "read-investigation", str(path)],
        text=True,
        capture_output=True,
    )
    assert child.returncode == 0, child.stderr
    loaded = json.loads(child.stdout)
    assert loaded["state"] == "active" and loaded["checkpoint"]["due"]
    assert loaded["usage"] == record["usage"]


def test_stop_kills_running_group_and_retains_usage(admission, tmp_path):
    kwargs, _, path = admission
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\nassert sys.stdin.buffer.read(1) == b'1'\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\nPath('ready').touch()\ntime.sleep(30)\n"
    )
    with (
        create_run(**kwargs) as delayed,
        create_run(**kwargs) as owner,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        job = pool.submit(
            _execute_job,
            owner,
            [sys.executable, str(worker)],
            os.environ.copy(),
            timeout_seconds=None,
        )
        try:
            until = time.monotonic() + 10
            while not (owner.path / "artifacts/ready").exists():
                assert time.monotonic() < until
                time.sleep(0.01)
            stopped = stop_investigation(path, "insufficient_progress", note="Stop this method.")
            assert not stopped["cancellation_errors"]
            assert stopped["stop"]["runs"] == [delayed.path.name, owner.path.name]
            assert job.result(timeout=10)["status"] == "aborted"
            record = read_run(owner.path)
            assert record["finished"]["usage"]["wall_seconds"] > 0
            assert (
                read_investigation(path)["usage"]["job_seconds"]
                == record["finished"]["usage"]["wall_seconds"]
            )
        finally:
            if not job.done():
                try:
                    abort_run(owner.path)
                except ValueError:
                    pass  # Cleanup can race with job completion.


def test_start_publication_and_stop_share_the_lock(admission, tmp_path, monkeypatch):
    import entanglix.runner as runner

    original = runner.record_start
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    stop_entered = threading.Event()

    def delayed_start(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        original(*args, **kwargs)

    monkeypatch.setattr(runner, "record_start", delayed_start)
    worker = tmp_path / "worker.py"
    worker.write_text("import sys, time\nassert sys.stdin.buffer.read(1) == b'1'\ntime.sleep(30)\n")
    with create_run(**admission[0]) as owner, ThreadPoolExecutor(max_workers=2) as pool:
        job = pool.submit(
            _execute_job,
            owner,
            [sys.executable, str(worker)],
            os.environ.copy(),
            timeout_seconds=None,
        )
        assert entered.wait(10)

        def stop():
            stop_entered.set()
            try:
                return stop_investigation(admission[2], "user_cancellation")
            finally:
                stopped.set()

        cancellation = pool.submit(stop)
        try:
            assert stop_entered.wait(10)
            assert not stopped.wait(0.05)
        finally:
            release.set()
        assert cancellation.result(timeout=10)["stop"]["runs"] == [owner.path.name]
        assert job.result(timeout=10)["status"] == "aborted"


def test_cli_stop_resume_and_operator_review_release(admission):
    kwargs, _, path = admission
    record_finding(path, assessment([]) | {"decision": "review", "review_name": "strategy"})

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "entanglix", *map(str, args)], text=True, capture_output=True
        )

    assert cli("stop-investigation", path, "not-a-reason").returncode == 64
    stopped = cli("stop-investigation", path, "review_handoff", "--note", "Need review.")
    assert stopped.returncode == 0, stopped.stderr
    assert cli("investigation-decision", path, "resume").returncode == 64
    released = cli("investigation-decision", path, "release-review", "--review", "strategy")
    assert released.returncode == 0, released.stderr
    assert cli("investigation-decision", path, "resume").returncode == 0
    assert cli("investigation-decision", path, "resume").returncode == 64
    assert (
        cli("investigation-decision", path, "release-review", "--review", "strategy").returncode
        == 64
    )
    with create_run(**kwargs):
        pass
    operations = list_capabilities()["operations"]
    assert "stop_investigation" in operations and "record_user_decision" not in operations
