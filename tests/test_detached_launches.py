"""Real supervisor handoff, launch recovery, and waits independent of jobs."""

import copy
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import tomli_w
from test_investigation_admission import study as study
from test_simulate import gpu_repository as gpu_repository
from test_simulate import protocol_files

from entanglix.investigations import (
    read_investigation,
    record_user_decision,
    revise_investigation,
)
from entanglix.launches import launch_job, prepare_launch, read_launch, start_supervisor
from entanglix.runner import (
    _load_record,
    abort_run,
    acquire_gpu,
    list_runs,
    read_run,
    record_reconciliation,
)

ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, seconds=30):
    expires = time.monotonic() + seconds
    while time.monotonic() < expires:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("process did not reach the expected state")


def arguments(study, repository, linked):
    protocol, _, _, runs, investigation = study
    return {
        "protocol": protocol,
        "runs_root": runs,
        "repository": repository,
        **({"investigation_id": investigation.name, "base": "seed"} if linked else {}),
    }


def slow_worker(repository, seconds=1):
    path = repository / "src/entanglix/protocols/_simulate_job.py"
    text = path.read_text()
    path.write_text(
        text.replace(
            "execute(Path(sys.argv[2]))",
            f"import time\n            time.sleep({seconds})\n            execute(Path(sys.argv[2]))",
        )
    )


@pytest.mark.parametrize("linked", [False, True])
def test_pending_launch_and_server_lost_before_supervisor(study, runner_repository, linked):
    args = arguments(study, runner_repository, linked)
    owner = prepare_launch("simulate", args, runs_root=study[3])
    try:
        assert read_run(owner.path)["status"] == "pending"
        assert list_runs(study[3])[0]["launch_id"] == owner.path.name
        pending = read_investigation(study[4])["unresolved_launches"]
        assert len(pending) == int(linked)
    finally:
        owner.close()  # the same ownership loss as a server dying before spawn
    assert read_run(owner.path, wait_seconds=100)["status"] == "refused"
    assert list_runs(study[3]) == []
    assert read_investigation(study[4])["usage"]["jobs"] == 0


@pytest.mark.parametrize("linked", [False, True])
def test_lost_admission_response_is_recovered_without_resubmission(
    study,
    runner_repository,
    linked,
    record_property,
):
    slow_worker(runner_repository)
    args = arguments(study, runner_repository, linked)
    start = time.monotonic()
    with prepare_launch("simulate", args, runs_root=study[3]) as owner:
        process = start_supervisor(
            owner, environment=os.environ | {"PYTHONPATH": str(runner_repository / "src")}
        )
        path = owner.path
    try:
        launch = wait_for(lambda: r if (r := read_launch(path))["status"] == "admitted" else None)
        record_property("cpu_launch_to_manifest_seconds", time.monotonic() - start)
        run = Path(launch["path"])
        assert launch["run_id"] == run.name
        assert _load_record(run / "run.toml")["launch_id"] == path.name
        record = read_run(run, wait_seconds=30)
        assert record["status"] == "completed", record["log_tail"]
        assert read_launch(path)["run_id"] == run.name
        assert len(list_runs(study[3])) == 1
        if linked:
            attempt = read_investigation(study[4])["attempts"][0]
            assert attempt["base"] == "seed"
            assert attempt["varied_values"] == [{}]
            assert attempt["revision"] == 1
    finally:
        process.wait(timeout=30)


@pytest.mark.parametrize("linked", [False, True])
@pytest.mark.parametrize("published", [False, True])
def test_supervisor_lost_at_manifest_publication_recovers_admission(
    study,
    runner_repository,
    linked,
    published,
):
    # Kill on either side of manifest publication, after the intent link. No production failpoint.
    worker = runner_repository / "src/entanglix/runner.py"
    statement = 'publish_toml(run / "run.toml", record)'
    kill = "os.kill(os.getpid(), signal.SIGKILL)"
    worker.write_text(
        worker.read_text().replace(
            statement,
            f"{statement}\n        {kill}" if published else f"{kill}\n        {statement}",
        )
    )
    with prepare_launch(
        "simulate", arguments(study, runner_repository, linked), runs_root=study[3]
    ) as owner:
        path = owner.path
        process = start_supervisor(
            owner, environment=os.environ | {"PYTHONPATH": str(runner_repository / "src")}
        )
    process.wait(timeout=30)
    assert process.returncode == -signal.SIGKILL
    assert (path / "attempt.toml").exists()
    result = read_launch(path)
    if not published:
        assert result["status"] == "refused"
        assert list_runs(study[3]) == []
        assert read_investigation(study[4])["usage"]["jobs"] == 0
        return
    assert result["status"] == "admitted"
    run = Path(result["path"])
    assert read_run(run, wait_seconds=30)["status"] == "interrupted"
    record_reconciliation(run, last_observed_state="supervisor lost immediately after manifest")
    assert read_launch(path)["run_id"] == run.name
    assert len(list_runs(study[3])) == 1


def test_wait_expiry_abort_and_interruption_do_not_hold_attempt(study, runner_repository):
    slow_worker(runner_repository, 30)
    launch = launch_job(
        "simulate",
        arguments(study, runner_repository, False),
        runs_root=study[3],
        environment=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
    )
    run = Path(launch["path"])
    wait_for(lambda: (run / "runner/started.toml").exists())
    try:
        started = time.monotonic()
        assert read_run(run, wait_seconds=0.05)["status"] == "active"
        assert time.monotonic() - started >= 0.05
        # Instrument only the reader's copy to prove it reached the blocking wait.
        marker = runner_repository / "reader-waiting"
        reader_module = runner_repository / "src/entanglix/runner.py"
        text = reader_module.read_text()
        text = text.replace(
            "\n        time.sleep(min(0.25, max(0, expires - time.monotonic())))",
            f"\n        Path({str(marker)!r}).touch()\n        time.sleep(min(0.25, max(0, expires - time.monotonic())))",
        )
        reader_module.write_text(text)
        reader = subprocess.Popen(
            [sys.executable, "-m", "entanglix", "read-run", str(run), "--wait", "100"],
            env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            wait_for(marker.exists)
            assert reader.poll() is None
            reader.send_signal(signal.SIGINT)
            output, error = reader.communicate(timeout=10)
            assert reader.returncode == 130
            assert not output
            assert b"interrupted" in error and b"Traceback" not in error
        finally:
            if reader.poll() is None:
                reader.kill()
            reader.communicate(timeout=10)
        assert read_run(run)["status"] == "active"
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(read_run, run, wait_seconds=30)
            assert not pending.done()
            abort_run(run)
            assert pending.result(timeout=10)["status"] == "aborted"
    finally:
        if read_run(run)["status"] == "active":
            abort_run(run)
        read_run(run, wait_seconds=10)


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf")])
def test_invalid_wait_is_refused(study, seconds):
    with pytest.raises(ValueError, match="wait_seconds"):
        read_run(study[3] / "absent", wait_seconds=seconds)


@pytest.mark.parametrize("failure", ["scope", "limit", "approval", "dry_run"])
def test_launch_refusals_do_not_admit(study, runner_repository, failure):
    args = arguments(study, runner_repository, True)
    if failure == "scope":
        args["values"] = {"parameters.frequency": 99}
    elif failure == "limit":
        terms = copy.deepcopy(study[2])
        terms["limits"] = {"deadline": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()}
        revision = revise_investigation(study[4], terms)["proposed_revision"]
        record_user_decision(study[4], "approve-revision", revision=revision)
    elif failure == "approval":
        args["approved"] = True
    else:
        args["dry_run"] = True
    with pytest.raises((ValueError, TypeError)):
        launch_job("simulate", args, runs_root=study[3])
    assert read_investigation(study[4])["usage"]["jobs"] == 0
    for request in (study[3] / "_launches").glob("*/request.toml"):
        assert read_launch(request.parent)["status"] == "refused"


def test_cli_detached_admission_and_uncapped_wait(study, runner_repository):
    protocol, _, _, runs, _ = study
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "entanglix",
            "simulate",
            str(protocol),
            "--runs-root",
            str(runs),
            "--detach",
        ],
        capture_output=True,
        text=True,
        env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
    )
    assert cli.returncode == 0, cli.stderr
    launch = json.loads(cli.stdout)
    assert launch["status"] == "admitted"
    finished = subprocess.run(
        [sys.executable, "-m", "entanglix", "read-run", launch["path"], "--wait", "100"],
        capture_output=True,
        text=True,
    )
    assert finished.returncode == 0, finished.stderr
    assert json.loads(finished.stdout)["status"] == "completed"


@pytest.mark.gpu
@pytest.mark.parametrize("phase", ["before_compute", "during_compute"])
def test_detached_cuda_deadline_and_continuous_ownership(
    tmp_path, gpu_repository, phase, record_property
):
    from entanglix.investigations import create_investigation
    from entanglix.runner import _digest

    protocol, _ = protocol_files(tmp_path, device="cuda")
    if phase == "before_compute":
        slow_worker(gpu_repository, 120)
    else:
        worker = gpu_repository / "src/entanglix/engines/cudensitymat.py"
        worker.write_text(
            worker.read_text().replace(
                "return EvolutionResult(",
                "import time\n            print('evolving', flush=True)\n            time.sleep(120)\n            return EvolutionResult(",
            )
        )
    # The source is tiny; a delay inside the admitted job exposes both phases.
    terms = {
        "format_version": 1,
        "question": "Deadline after connection loss",
        "assumptions": [],
        "scope": {
            "operations": ["simulate"],
            "bases": [
                {
                    "name": "cuda",
                    "inputs": {
                        name: _digest((tmp_path / name).read_bytes())
                        for name in ("system.toml", "simulate.toml")
                    },
                }
            ],
        },
        "evidence_requirements": {"requirements": ["Retain deadline and GPU ownership evidence."]},
        "checkpoint": {"finished_attempts": 10},
        "review_criteria": [],
        "stop_criteria": [],
        "limits": {"deadline": (datetime.now(UTC) + timedelta(seconds=60)).isoformat()},
    }
    runs = tmp_path / "runs"
    investigation = Path(create_investigation(terms, runs_root=runs)["path"])
    record_user_decision(investigation, "start", revision=1)
    args = {
        "protocol": protocol,
        "runs_root": runs,
        "repository": gpu_repository,
        "investigation_id": investigation.name,
        "base": "cuda",
    }
    encoded = {key: str(value) if isinstance(value, Path) else value for key, value in args.items()}
    start = time.monotonic()
    client = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import json,sys\nfrom pathlib import Path\nfrom entanglix.launches import launch_job\n"
            "print(json.dumps(launch_job('simulate',json.loads(sys.argv[1]),runs_root=Path(sys.argv[2]))),flush=True)\nimport time;time.sleep(120)\n",
            json.dumps(encoded),
            str(runs),
        ],
        env=os.environ | {"PYTHONPATH": str(gpu_repository / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = client.stdout.readline()
    assert line, client.communicate(timeout=10)[1]
    launch = json.loads(line)
    record_property("cuda_launch_to_manifest_seconds", time.monotonic() - start)
    run = Path(launch["path"])
    wait_for(lambda: (run / "runner/started.toml").exists())
    device = _load_record(run / "runner/started.toml")["gpu_devices"][0]
    try:
        if phase == "during_compute":
            wait_for(lambda: "evolving" in (run / "run.log").read_text())
        client.kill()
        client.communicate(timeout=10)
        with pytest.raises(ValueError, match="still owned"):
            acquire_gpu(device)
        # Deadline terminates the supervisor-owned job after the caller has returned.
        record = read_run(run, wait_seconds=70)
        assert record["status"] == "aborted", record["log_tail"]
        assert "deadline" in record["log_tail"]
        assert record["finished"]["usage"]["gpu_seconds"] > 0
        with acquire_gpu(device):
            pass
    finally:
        if client.poll() is None:
            client.kill()
        client.communicate(timeout=10)
        if read_run(run)["status"] == "active":
            abort_run(run)


@pytest.mark.parametrize("linked", [False, True])
def test_requesting_process_exits_before_admission_job_still_finishes(
    study, runner_repository, linked
):
    args = arguments(study, runner_repository, linked)
    # Delay the gate to make the actual requesting-process loss precede admission.
    gate = runner_repository / "src/entanglix/protocols/_simulate_job.py"
    gate.write_text(
        gate.read_text().replace(
            'if sys.argv[1] == "check":',
            'if sys.argv[1] == "check":\n            import time\n            time.sleep(0.5)',
        )
    )
    encoded = {key: str(value) if isinstance(value, Path) else value for key, value in args.items()}
    client = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys\nfrom pathlib import Path\n"
            "from entanglix.launches import prepare_launch,start_supervisor\n"
            "with prepare_launch('simulate',json.loads(sys.argv[1]),runs_root=Path(sys.argv[2])) as owner:\n"
            "    start_supervisor(owner)\n    print(owner.path,flush=True)\n",
            json.dumps(encoded),
            str(study[3]),
        ],
        env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
        capture_output=True,
        text=True,
    )
    assert client.returncode == 0, client.stderr
    path = Path(client.stdout.strip())
    launch = read_run(path, wait_seconds=30)
    assert launch["status"] == "admitted"
    record = read_run(Path(launch["path"]), wait_seconds=30)
    assert record["status"] == "completed", record["log_tail"]
    assert len(list_runs(study[3])) == 1


def test_detached_check_and_rerun_keep_links(study, runner_repository):
    args = arguments(study, runner_repository, True)
    runs = study[3]
    first = launch_job("simulate", args, runs_root=runs)
    run = Path(first["path"])
    assert read_run(run, wait_seconds=30)["status"] == "completed"
    checking = launch_job(
        "check",
        {"source": run, "checks": ["integrity"], "repository": runner_repository},
        runs_root=runs,
    )
    check = Path(checking["path"])
    report = read_run(check, wait_seconds=30)
    assert report["status"] == "completed", report["log_tail"]
    assert report["manifest"]["parent_run_ids"] == [run.name]
    assert read_launch(Path(checking["launch_path"]))["investigation_id"] == study[4].name
    replay = launch_job("rerun", {"run": run, "repository": runner_repository}, runs_root=runs)
    repeated = read_run(Path(replay["path"]), wait_seconds=30)
    assert repeated["status"] == "completed", repeated["log_tail"]
    assert repeated["manifest"]["parent_run_ids"] == [run.name]
    assert read_investigation(study[4])["usage"]["jobs"] == 3


def test_failed_supervisor_spawn_recovers_refusal(study, runner_repository, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("supervisor unavailable")

    monkeypatch.setattr("entanglix.launches.start_supervisor", fail)
    with pytest.raises(OSError, match="supervisor unavailable"):
        launch_job("simulate", arguments(study, runner_repository, False), runs_root=study[3])
    request = next((study[3] / "_launches").glob("*/request.toml"))
    assert read_launch(request.parent)["reason"] == "supervisor unavailable"
    assert list_runs(study[3]) == []


@pytest.mark.parametrize("operation", ["run_script", "sweep"])
def test_detached_script_and_sweep_replay(study, runner_repository, operation):
    protocol, _, _, runs, _ = study
    if operation == "run_script":
        script = protocol.parent / "script.py"
        script.write_text(
            "import os\nfrom pathlib import Path\nfrom entanglix.runner import publish_toml\n"
            "publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR'])/'result.toml', "
            "{'engine':'authored','metrics':{'value':1}}, replace=True)\n"
        )
        args = {"system": protocol.parent / "system.toml", "script": script, "inputs": None}
        replay_operation = "rerun"
    else:
        grid = protocol.parent / "sweep.toml"
        grid.write_text(
            tomli_w.dumps(
                {
                    "format_version": 1,
                    "base": "simulate.toml",
                    "axes": [
                        {"id": "frequency", "target": "parameters.frequency", "values": [0.2, 0.3]}
                    ],
                }
            )
        )
        args = {"protocol": grid}
        replay_operation = "rerun"
    first = launch_job(
        operation, args | {"repository": runner_repository, "runs_root": runs}, runs_root=runs
    )
    run = Path(first["path"])
    original = read_run(run, wait_seconds=30)
    assert original["status"] == "completed", original["log_tail"]
    replay = launch_job(
        replay_operation, {"run": run, "repository": runner_repository}, runs_root=runs
    )
    record = read_run(Path(replay["path"]), wait_seconds=30)
    assert record["status"] == "completed", record["log_tail"]
    assert record["manifest"]["parent_run_ids"] == [run.name]
    assert record["result"]["metrics"] == original["result"]["metrics"]


@pytest.mark.parametrize("owned", [False, True])
def test_concurrent_launch_readers_do_not_look_like_owner(
    study, runner_repository, monkeypatch, owned
):
    from threading import Barrier

    owner = prepare_launch(
        "simulate", arguments(study, runner_repository, False), runs_root=study[3]
    )
    if not owned:
        owner.close()
    barrier = Barrier(2)
    exists = Path.exists

    def overlap(path):
        if path == owner.path / "attempt.toml":
            barrier.wait(timeout=5)
        return exists(path)

    monkeypatch.setattr(Path, "exists", overlap)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(read_launch, [owner.path] * 80))
        assert {item["status"] for item in results} == {"pending" if owned else "refused"}
    finally:
        owner.close()


def test_recovery_does_not_scan_unrelated_or_removed_attempts(
    study, runner_repository, monkeypatch
):
    import shutil

    root = study[3]
    launch = launch_job("simulate", arguments(study, runner_repository, False), runs_root=root)
    run = Path(launch["path"])
    assert read_run(run, wait_seconds=30)["status"] == "completed"
    # An authored file with this name must not change the attempt's type.
    (run / "request.toml").write_text('user_data = "ordinary result"\n')
    assert read_run(run)["status"] == "completed"
    bad = root / "bad" / "r-invalid"
    bad.mkdir(parents=True)
    (bad / "run.toml").write_text("invalid TOML [")
    with prepare_launch(
        "simulate", arguments(study, runner_repository, False), runs_root=root
    ) as owner:
        refused = owner.path
    # Refused and linked launches should never enumerate the runs root.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "glob", lambda *_: pytest.fail("launch recovery scanned the root"))
        assert read_launch(refused)["status"] == "refused"
        assert read_launch(Path(launch["launch_path"]))["status"] == "admitted"
    statuses = {item["status"] for item in list_runs(root)}
    assert statuses == {"completed", "invalid"}
    shutil.rmtree(run)
    assert read_launch(Path(launch["launch_path"]))["status"] == "refused"
    assert [item["status"] for item in list_runs(root)] == ["invalid"]


def test_ctrl_c_while_detached_admission_is_pending_keeps_supervisor(study, runner_repository):
    protocol, _, _, root, _ = study
    marker = runner_repository / "gate-waiting"
    worker = runner_repository / "src/entanglix/protocols/_simulate_job.py"
    worker.write_text(
        worker.read_text().replace(
            'if sys.argv[1] == "check":',
            f'if sys.argv[1] == "check":\n            Path({str(marker)!r}).touch()\n            import time\n            time.sleep(1)',
        )
    )
    client = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "entanglix",
            "simulate",
            str(protocol),
            "--runs-root",
            str(root),
            "--detach",
        ],
        env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for(marker.exists)
        request = next((root / "_launches").glob("*/request.toml"))
        assert read_launch(request.parent)["status"] == "pending"
        client.send_signal(signal.SIGINT)
        output, error = client.communicate(timeout=10)
        assert client.returncode == 130
        assert not output
        assert b"interrupted" in error and b"Traceback" not in error
        launch = read_run(request.parent, wait_seconds=30)
        assert launch["status"] == "admitted"
        record = read_run(Path(launch["path"]), wait_seconds=30)
        assert record["status"] == "completed", record["log_tail"]
        assert len(list_runs(root)) == 1
    finally:
        if client.poll() is None:
            client.kill()
        client.communicate(timeout=10)
