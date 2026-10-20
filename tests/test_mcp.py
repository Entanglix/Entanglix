"""Scripted stdio clients exercise transport, audit, and independent supervision."""

import asyncio
import copy
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import tomli_w

pytest.importorskip("mcp")
from mcp import Client, ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Implementation
from test_detached_launches import slow_worker, wait_for
from test_investigation_admission import study as study
from test_run_records import run_kwargs as run_kwargs
from test_simulate import gpu_repository as gpu_repository

from entanglix.investigations import read_investigation
from entanglix.launches import read_launch
from entanglix.mcp import READS, Configuration, build_server, supplied_guidance
from entanglix.mcp_views import METRIC_LIMIT, investigation_view, run_view
from entanglix.runner import (
    _load_record,
    list_capabilities,
    list_runs,
    read_run,
    record_reconciliation,
)

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def client(repository, runs, state, *, device="", extras=(), cwd=None):
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "entanglix.mcp",
            "--runs-root",
            str(runs),
            "--device",
            device,
            *(["--state-directory", str(state)] if state is not None else []),
            "--runner-extras",
            *extras,
        ],
        env=os.environ | {"PYTHONPATH": str(repository / "src"), "CUDA_VISIBLE_DEVICES": ""},
        cwd=cwd,
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(
            reader, writer, client_info=Implementation(name="scripted-test", version="1")
        ) as session:
            await session.initialize()
            yield session


def content(result):
    assert not result.is_error, result
    assert result.structured_content is None
    assert len(result.content) == 1 and result.content[0].type == "text"
    return json.loads(result.content[0].text)


async def call(session, name, arguments=None, **kwargs):
    return content(await session.call_tool(name, arguments or {}, **kwargs))


async def cli_result(repository, *arguments):
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "entanglix", *arguments],
        env=os.environ | {"PYTHONPATH": str(repository / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_tool_parity_annotations_and_immutable_audit(study, runner_repository, tmp_path):
    protocol, _, terms, runs, investigation = study

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            tools = (await session.list_tools()).tools
            assert {tool.name for tool in tools} == set(list_capabilities()["operations"])
            for tool in tools:
                assert tool.annotations.read_only_hint == (tool.name in READS)
            assert (await call(session, "list_runs"))["rows"] == []
            capabilities = await call(session, "list_capabilities")
            assert capabilities == list_capabilities(runner_repository)
            assert capabilities == json.loads(await cli_result(runner_repository, "capabilities"))
            validation = await call(
                session, "validate_system", {"system": str(protocol.parent / "system.toml")}
            )
            assert validation["dimensions"] == [3, 2]
            cli = await cli_result(
                runner_repository, "validate", str(protocol.parent / "system.toml")
            )
            assert f"dimensions {tuple(validation['dimensions'])}" in cli
            for name, meaning in validation["operators"].items():
                assert f"{name}: {meaning}" in cli
            proposed = copy.deepcopy(terms)
            proposed["limits"] = {"job_count": 1}
            await call(
                session,
                "revise_investigation",
                {"investigation": str(investigation), "spec": proposed},
                meta={"model": "test-model", "turn_id": "turn-1"},
            )
            assert read_investigation(investigation)["active_revision"] == 1
            session_file = next((runs / "_audit/sessions").glob("*.toml"))
            connection_bytes = session_file.read_bytes()
            for name, args in (
                ("investigation_decision", {"action": "start"}),
                ("simulate", {"protocol": str(protocol), "approval": True}),
                ("create_investigation", {"spec": terms | {"approved": True}}),
            ):
                try:
                    result = await session.call_tool(name, args)
                    assert result.is_error
                except Exception as exc:
                    assert "unsupported" in str(exc) or "unknown" in str(exc).lower()
            assert (await call(session, "list_runs"))["rows"] == json.loads(
                await cli_result(runner_repository, "list-runs", "--runs-root", str(runs))
            )
            listed = (await call(session, "list_investigations"))["result"]
            cli_listed = json.loads(
                await cli_result(runner_repository, "list-investigations", "--runs-root", str(runs))
            )
            assert all(Path(item["path"]).is_absolute() for item in cli_listed)
            assert listed == [
                item | {"path": str(Path(item["path"]).relative_to(runs))} for item in cli_listed
            ]
            assert listed[0]["path"] == str(investigation.relative_to(runs))
            inspected = await call(
                session, "read_investigation", {"investigation": listed[0]["path"]}
            )
            cli = json.loads(
                await cli_result(runner_repository, "read-investigation", str(investigation))
            )
            # Live elapsed time advances between requests; all retained data must match.
            assert inspected["checkpoint"].pop("elapsed_seconds") <= cli["checkpoint"].pop(
                "elapsed_seconds"
            )
            assert {key: inspected[key] for key in cli if key in inspected} == {
                key: value for key, value in cli.items() if key in inspected
            }
            assert session_file.read_bytes() == connection_bytes

    asyncio.run(exercise())
    sessions = list((runs / "_audit/sessions").glob("*.toml"))
    assert len(sessions) == 1
    connection = _load_record(sessions[0])
    assert connection["client"] == {
        "source": "client-reported",
        "name": "scripted-test",
        "version": "1",
    }
    assert (
        connection["model"]
        == connection["loaded_skills"]
        == connection["transcript_location"]
        == "unknown"
    )
    assert connection["configuration"]["runs_root"] == str(runs.resolve())
    assert connection["guidance_status"] == "none supplied"
    requests = [_load_record(p) for p in (runs / "_audit/calls").glob("*/request.toml")]
    assert len(requests) == 10
    revision = next(item for item in requests if item["tool"] == "revise_investigation")
    assert json.loads(revision["client_metadata"]["json"])["model"] == "test-model"
    assert revision["client_metadata"]["source"] == "client-reported"
    assert all(item["session_id"] == connection["session_id"] for item in requests)
    assert all("outcome" not in item and "result_sha256" not in item for item in requests)
    outcomes = [_load_record(p) for p in (runs / "_audit/calls").glob("*/outcome.toml")]
    assert len(outcomes) == len(requests)
    assert sum(item["outcome"] == "error" for item in outcomes) == 3
    assert all(len(item["result_sha256"]) == 64 for item in outcomes)
    assert any(
        item.get("links", {}).get("proposed_revision") == 2
        and item["links"]["investigation_id"] == investigation.name
        for item in outcomes
    )
    assert not list_runs(runs)


def test_workspace_relative_record_path_resolves_inside_runs_root(
    study, runner_repository, tmp_path
):
    _, _, _, runs, investigation = study
    workspace = runs.parent
    given = str(Path(runs.name) / investigation.relative_to(runs))
    missing = given + "-missing"
    (workspace / "outside-record").mkdir()

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state", cwd=workspace) as session:
            record = await call(session, "read_investigation", {"investigation": given})
            assert record["investigation_id"] == investigation.name
            result = await session.call_tool("read_investigation", {"investigation": missing})
            assert result.is_error
            message = result.content[0].text
            assert missing in message
            assert "does not exist" in message
            assert "relative record paths resolve from the runs root" in message
            assert str(runs.resolve()) in message
            assert "then from the server's working directory" in message
            assert str(workspace.resolve()) in message
            assert "[Errno 2]" not in message
            outside = await session.call_tool(
                "read_investigation", {"investigation": "outside-record"}
            )
            assert outside.is_error
            assert "outside configured runs root" in outside.content[0].text

    asyncio.run(exercise())


def test_first_request_without_initialize_records_session(runner_repository, tmp_path):
    runs = tmp_path / "runs"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "entanglix.mcp", "--runs-root", str(runs), "--device", "", "--runner-extras"],
        env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
    )

    async def exercise():
        async with Client(
            params,
            mode="2026-07-28",
            client_info=Implementation(name="modern-scripted-test", version="1"),
        ) as session:
            await call(session, "list_capabilities")

    asyncio.run(exercise())
    sessions = list((runs / "_audit/sessions").glob("*.toml"))
    assert len(sessions) == 1
    connection = _load_record(sessions[0])
    assert connection["client"] == {
        "source": "client-reported",
        "name": "modern-scripted-test",
        "version": "1",
    }
    request = _load_record(next((runs / "_audit/calls").glob("*/request.toml")))
    assert request["session_id"] == connection["session_id"]
    assert (
        json.loads(request["client_metadata"]["json"])["io.modelcontextprotocol/protocolVersion"]
        == "2026-07-28"
    )


def test_investigation_views_and_finding_ack_over_stdio(study, runner_repository, tmp_path):
    from test_investigation_admission import assessment

    from entanglix.investigations import record_finding, revise_investigation

    protocol, _, terms, runs, investigation = study
    older = record_finding(investigation, assessment([]) | {"learned": "EARLIER-TEXT" * 10000})
    revise_investigation(investigation, terms | {"question": "Proposed question"})

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            latest = assessment([]) | {"learned": "LATEST-TEXT" * 10000}
            ack = await call(
                session, "record_finding", {"investigation": str(investigation), "finding": latest}
            )
            assert "finding" not in ack and "LATEST-TEXT" not in json.dumps(ack)
            assert ack["revision"] == 1 and ack["checkpoint"]["since_assessment"] == 0
            brief = await call(session, "read_investigation", {"investigation": str(investigation)})
            assert "EARLIER-TEXT" not in json.dumps(brief)
            assert brief["latest_finding"]["finding"]["learned"] == latest["learned"]
            one = await call(
                session,
                "read_investigation",
                {"investigation": str(investigation), "event_id": older["event_id"]},
            )
            assert one["event"]["finding"]["learned"] == "EARLIER-TEXT" * 10000
            assert brief["revisions"][1]["terms"]["question"] == "Proposed question"
            indexed = await call(
                session,
                "read_investigation",
                {"investigation": str(investigation), "view": "events"},
            )
            assert indexed["rows"][-1][0] == ack["event_id"]
            assert (
                await call(
                    session,
                    "read_investigation",
                    {"investigation": str(investigation), "view": "events", "after": 1},
                )
            )["rows"] == indexed["rows"][1:]
            for args in (
                {"view": "attempts", "event_id": older["event_id"]},
                {"view": "unknown"},
                {"after": 1},
                {"event_id": "../not-an-event"},
            ):
                result = await session.call_tool(
                    "read_investigation", {"investigation": str(investigation)} | args
                )
                assert result.is_error
            launch = await call(
                session,
                "simulate",
                {"protocol": str(protocol), "investigation_id": investigation.name, "base": "seed"},
            )
            run = launch["path"]
            assert (await call(session, "read_run", {"run": run, "wait_seconds": 30}))[
                "status"
            ] == "completed"
            checked = await call(session, "check", {"run": run, "checks": ["integrity"]})
            check_brief = await call(
                session, "read_run", {"run": checked["path"], "wait_seconds": 30}
            )
            check_full = await call(session, "read_run", {"run": checked["path"], "detail": True})
            assert "checks" not in check_brief["result"]["report"]["checks"][0]
            assert check_full["result"]["report"]["checks"][0]["checks"]
            attempts = await call(
                session,
                "read_investigation",
                {"investigation": str(investigation), "view": "attempts"},
            )
            rows = [dict(zip(attempts["columns"], row)) for row in attempts["rows"]]
            entry = next(row for row in rows if Path(row["path"]).name == launch["run_id"])
            assert (
                entry["metrics.population"]
                == read_run(Path(run))["result"]["metrics"]["population"]
            )
            assert entry["checks"][0][:3] == [checked["run_id"], True, [["integrity", "pass"]]]
            check_entry = next(row for row in rows if Path(row["path"]).name == checked["run_id"])
            assert check_entry["parents"] == [launch["run_id"]]
            assert check_entry["status"] == "completed"
            assert check_entry["checks"] == [["integrity", "pass"]]
            assert check_brief["manifest"]["detail_omitted"] == [
                "code",
                "environment",
                "spec_sha256",
            ]
            assert "spec_sha256" in check_full["manifest"]
            listed = await call(session, "list_runs", {"after": 1})
            assert listed["total"] == 2 and len(listed["rows"]) == 1
            assert listed["rows"][0][0] == check_entry["path"]

    asyncio.run(exercise())
    outcomes = [_load_record(path) for path in (runs / "_audit/calls").glob("*/outcome.toml")]
    assert any(item.get("links", {}).get("investigation_revision") == 1 for item in outcomes)
    assert any(item.get("links", {}).get("run_id") for item in outcomes)


def test_oversized_metrics_and_stale_checks_remain_explicit_in_summary():
    original = {
        "manifest": {
            "spec_sha256": {"system.toml": "digest"},
            "code": {"revision": "commit"},
            "environment": {"python": "3.13"},
        },
        "log_tail": "progress\n" * 900 + "fatal error: allocation failed\n",
        "result": {"metrics": {"large": list(range(METRIC_LIMIT)), "endpoint": 0.4}},
        "verification": [
            {
                "run_id": "check",
                "status": "completed",
                "policy_version": "old",
                "source_matches": False,
                "checks": [],
            }
        ],
    }
    summary = run_view(original)
    assert summary["result"]["metrics"]["large"]["omitted"]
    assert summary["result"]["metrics"]["endpoint"] == 0.4
    assert summary["verification"][0]["source_matches"] is False
    assert isinstance(original["result"]["metrics"]["large"], list)
    assert summary["log_tail"] == original["log_tail"]
    assert summary["manifest"] == {"detail_omitted": ["code", "environment", "spec_sha256"]}
    assert "code" in original["manifest"]


def test_investigation_status_keeps_effective_terms_in_full(study):
    record = read_investigation(study[-1])
    terms = record["revisions"][0]["terms"]
    terms["question"] = "Full question. " * 1000
    terms["evidence_requirements"]["requirements"] = ["Full evidence requirement. " * 1000]
    summary = investigation_view(record)
    assert summary["revisions"][0]["terms"] == terms


def test_sweep_summary_keeps_errors_without_inventing_status():
    record = {
        "sweep_progress": {
            "completed_points": [{"index": 0, "values": {"x": 1}, "error": "solver failed"}]
        }
    }
    point = run_view(record)["sweep_progress"]["completed_points"][0]
    assert point["error"] == "solver failed"
    assert "status" not in point


@pytest.mark.parametrize("report", ["Authored interpretation", {"checks": "Authored data"}])
def test_authored_reports_are_not_treated_as_corrector_reports(report):
    record = {"manifest": {"protocol": "script"}, "result": {"report": report}}
    assert run_view(record)["result"]["report"] == report


def test_launch_wait_cancel_and_concurrent_abort(
    study, runner_repository, tmp_path, record_property
):
    protocol, _, _, runs, investigation = study
    slow_worker(runner_repository, 120)

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            start = time.monotonic()
            launch = await call(
                session,
                "simulate",
                {"protocol": str(protocol), "investigation_id": investigation.name, "base": "seed"},
            )
            record_property("cpu_mcp_launch_to_manifest_seconds", time.monotonic() - start)
            run = Path(launch["path"])
            await asyncio.to_thread(wait_for, lambda: (run / "runner/started.toml").exists())
            assert (await call(session, "read_run", {"run": str(run), "wait_seconds": 0.05}))[
                "status"
            ] == "active"
            waiter = asyncio.create_task(
                call(session, "read_run", {"run": str(run), "wait_seconds": 1000})
            )
            await asyncio.sleep(0.1)
            assert not waiter.done()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert (await call(session, "read_run", {"run": str(run)}))["status"] == "active"
            waiter = asyncio.create_task(
                call(session, "read_run", {"run": str(run), "wait_seconds": 30})
            )
            await asyncio.sleep(0.1)
            await call(session, "abort_run", {"run": str(run)})
            assert (await waiter)["status"] == "aborted"
            return run

    run = asyncio.run(exercise())
    manifest = _load_record(run / "run.toml")
    audit = runs / "_audit/calls" / manifest["launch_id"]
    request, outcome = _load_record(audit / "request.toml"), _load_record(audit / "outcome.toml")
    assert request["timestamp"] <= manifest["created_at"]
    assert outcome["links"]["run_id"] == run.name
    assert outcome["links"]["investigation_revision"] == 1
    assert outcome["links"]["status"] == "admitted"
    assert any(
        _load_record(p)["outcome"] == "interrupted"
        for p in (runs / "_audit/calls").glob("*/outcome.toml")
    )


@pytest.mark.parametrize("linked", [False, True])
@pytest.mark.parametrize("phase", ["before_admission", "after_admission"])
def test_disconnect_during_admission_then_recover(
    study, runner_repository, tmp_path, linked, phase
):
    protocol, _, _, runs, investigation = study
    if phase == "before_admission":
        worker = runner_repository / "src/entanglix/_supervisor.py"
        text = worker.read_text()
        marker = 'request = _load_record(path / "request.toml")'
        assert marker in text
        worker.write_text(
            text.replace(marker, marker + "\n        import time\n        time.sleep(1)")
        )
    else:
        # Hold the response after admission; losing it must not resubmit the job.
        worker = runner_repository / "src/entanglix/mcp.py"
        marker = "                    return result\n                await asyncio.sleep(0.01)"
        assert marker in worker.read_text()
        worker.write_text(
            worker.read_text().replace(
                marker, "                    await asyncio.sleep(120)\n" + marker
            )
        )

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            arguments = {
                "protocol": str(protocol),
                **({"investigation_id": investigation.name, "base": "seed"} if linked else {}),
            }
            task = asyncio.create_task(call(session, "simulate", arguments))
            await asyncio.to_thread(
                wait_for,
                lambda: (
                    list((runs / "_launches").glob("*/supervisor.log"))
                    if phase == "before_admission"
                    else list(runs.glob("*/r*/run.toml"))
                ),
            )
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(exercise())
    path = next((runs / "_launches").glob("*"))
    launch = wait_for(lambda: item if (item := read_launch(path))["status"] != "pending" else None)
    assert launch["status"] == "admitted", launch
    run = Path(launch["path"])
    assert read_run(run, wait_seconds=30)["status"] == "completed"
    assert len(list_runs(runs)) == 1

    async def recover():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            assert (await call(session, "read_run", {"run": str(path)}))["run_id"] == run.name

    asyncio.run(recover())
    assert len(list((runs / "_audit/sessions").glob("*.toml"))) == 2


@pytest.mark.parametrize("linked", [False, True])
def test_server_lost_after_audit_before_supervisor(study, runner_repository, tmp_path, linked):
    protocol, _, _, runs, investigation = study
    server = runner_repository / "src/entanglix/mcp.py"
    statement = "result = await call_next(ctx)"
    assert statement in server.read_text()
    server.write_text(
        server.read_text().replace(statement, "os.kill(os.getpid(), 9)\n            " + statement)
    )

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            await session.call_tool(
                "simulate",
                {
                    "protocol": str(protocol),
                    **({"investigation_id": investigation.name, "base": "seed"} if linked else {}),
                },
            )

    with pytest.raises(BaseException):
        asyncio.run(exercise())
    request = next((runs / "_audit/calls").glob("*/request.toml"))
    path = Path(_load_record(request)["launch_path"])
    assert read_run(path)["status"] == "refused"
    assert list_runs(runs) == []
    assert read_investigation(investigation)["usage"]["jobs"] == 0
    assert not (request.parent / "outcome.toml").exists()


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("linked", [False, True])
def test_supervisor_lost_near_manifest_preserves_audit_link(
    study, runner_repository, tmp_path, published, linked
):
    protocol, _, _, runs, investigation = study
    worker = runner_repository / "src/entanglix/runner.py"
    statement = 'publish_toml(run / "run.toml", record)'
    kill = "os.kill(os.getpid(), signal.SIGKILL)"
    worker.write_text(
        worker.read_text().replace(
            statement,
            statement + "\n        " + kill if published else kill + "\n        " + statement,
        )
    )

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            return await session.call_tool(
                "simulate",
                {
                    "protocol": str(protocol),
                    **({"investigation_id": investigation.name, "base": "seed"} if linked else {}),
                },
            )

    result = asyncio.run(exercise())
    assert result.is_error == (not published)
    path = next((runs / "_launches").glob("*"))
    launch = read_launch(path)
    if published:
        run = Path(launch["path"])
        assert read_run(run)["status"] == "interrupted"
        record_reconciliation(run, last_observed_state="supervisor lost")
        assert _load_record(run / "run.toml")["launch_id"] == path.name
    else:
        assert launch["status"] == "refused"
    assert (
        _load_record(runs / "_audit/calls" / path.name / "outcome.toml")["links"]["status"]
        == launch["status"]
    )


def test_wait_cap_and_guidance(tmp_path):
    # Shorten the cap in the test server; exercise expiry on an owned pending launch.
    import entanglix.mcp as transport
    from entanglix.launches import reserve_launch

    (tmp_path / "docs").mkdir()
    (tmp_path / "skills/simulation").mkdir(parents=True)
    (tmp_path / "docs/research-agent.md").write_text(
        "---\nname: conduct\ndescription: research conduct\nversion: 1\n---\nFollow the user terms.\n"
    )
    (tmp_path / "skills/simulation/SKILL.md").write_text(
        "---\nname: simulation\ndescription: procedure\nversion: 2\n---\nprocedure\n"
    )
    instructions, metadata = supplied_guidance(tmp_path)
    assert instructions == "Follow the user terms."
    assert [item["version"] for item in metadata] == ["1", "2"]
    original = transport.WAIT_CAP
    transport.WAIT_CAP = 0.05
    try:
        with reserve_launch(tmp_path / "runs", "simulate") as owner:
            server = build_server(
                Configuration(tmp_path / "runs", "", (), tmp_path / "state", tmp_path)
            )

            async def exercise():
                start = time.monotonic()
                result = await server.call_tool(
                    "read_run", {"run": str(owner.path), "wait_seconds": 1000}
                )
                assert time.monotonic() - start < 1
                return result

            # Tool manager output includes both content and structured result.
            result = asyncio.run(exercise())
            assert "pending" in str(result)
    finally:
        transport.WAIT_CAP = original


def test_transport_checks_replay_and_admission_refusals(study, runner_repository, tmp_path):
    from entanglix.investigations import record_user_decision, revise_investigation

    protocol, _, terms, runs, investigation = study
    capped = copy.deepcopy(terms)
    capped["limits"] = {"job_count": 2}
    revise_investigation(investigation, capped)
    record_user_decision(investigation, "approve-revision", revision=2)

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            # A proposed limit change has no effect until a human decision.
            proposed = copy.deepcopy(terms)
            proposed["limits"] = {"job_count": 100}
            await call(
                session,
                "revise_investigation",
                {"investigation": str(investigation), "spec": proposed},
            )
            bad = await session.call_tool(
                "simulate",
                {
                    "protocol": str(protocol),
                    "investigation_id": investigation.name,
                    "base": "seed",
                    "values": {"evolution.drive_amplitudes_ghz.pulse": 10},
                },
            )
            assert bad.is_error
            launch = await call(
                session,
                "simulate",
                {"protocol": str(protocol), "investigation_id": investigation.name, "base": "seed"},
            )
            run = launch["path"]
            record = await call(
                session, "read_run", {"run": run, "wait_seconds": 30, "detail": True}
            )
            assert record["status"] == "completed"
            assert record == json.loads(await cli_result(runner_repository, "read-run", run))
            checked = await call(session, "check", {"run": run, "checks": ["integrity"]})
            record = await call(session, "read_run", {"run": checked["path"], "wait_seconds": 30})
            assert record["status"] == "completed", record["log_tail"]
            assert read_run(Path(run))["verification"][0]["checks"][0]["outcome"] == "pass"
            rejected = await session.call_tool("rerun_simulation", {"run": run})
            assert rejected.is_error
            assert "job_count" in rejected.content[0].text
            # A standalone replay records new lineage, using the same backing operation.
            solo = await call(session, "simulate", {"protocol": str(protocol)})
            assert (await call(session, "read_run", {"run": solo["path"], "wait_seconds": 30}))[
                "status"
            ] == "completed"
            replay = await call(session, "rerun_simulation", {"run": solo["path"]})
            assert (await call(session, "read_run", {"run": replay["path"], "wait_seconds": 30}))[
                "manifest"
            ]["parent_run_ids"] == [solo["run_id"]]

    asyncio.run(exercise())
    assert read_investigation(investigation)["usage"]["jobs"] == 2


def test_job_tools_record_parent_lineage_as_the_cli(study, runner_repository, tmp_path):
    protocol, _, _, runs, investigation = study
    system = protocol.parent / "system.toml"
    script = protocol.parent / "script.py"
    script.write_text(
        "import os\nfrom pathlib import Path\nfrom entanglix.runner import publish_toml\n"
        "publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR'])/'result.toml', "
        "{'engine':'authored','metrics':{'value':1}}, replace=True)\n"
    )
    grid = protocol.parent / "sweep.toml"
    grid.write_text(
        tomli_w.dumps(
            {
                "format_version": 1,
                "base": "simulate.toml",
                "axes": [{"id": "frequency", "target": "parameters.frequency", "values": [0.2]}],
            }
        )
    )
    requests = {
        "simulate": ({"protocol": str(protocol)}, ["simulate", str(protocol)]),
        "sweep": ({"protocol": str(grid)}, ["sweep", str(grid)]),
        "run_script": (
            {"system": str(system), "script": str(script)},
            ["run-script", str(script), "--system", str(system)],
        ),
    }

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            first = await call(session, "simulate", {"protocol": str(protocol)})
            second = json.loads(
                await cli_result(
                    runner_repository, "simulate", str(protocol), "--runs-root", str(runs)
                )
            )
            parents = [first["run_id"], Path(second["path"]).name]
            flags = [item for parent in parents for item in ("--parent-run", parent)]
            for name, (arguments, command) in requests.items():
                launch = await call(session, name, arguments | {"parent_run_ids": parents})
                record = await call(
                    session, "read_run", {"run": launch["path"], "wait_seconds": 30, "detail": True}
                )
                assert record["status"] == "completed", record["log_tail"]
                cli = json.loads(
                    await cli_result(runner_repository, *command, "--runs-root", str(runs), *flags)
                )
                assert (
                    record["manifest"]["parent_run_ids"]
                    == read_run(Path(cli["path"]))["manifest"]["parent_run_ids"]
                    == parents
                ), name
            count = len(list(runs.glob("*/*/run.toml")))
            refused = await session.call_tool(
                "simulate",
                {
                    "protocol": str(protocol),
                    "investigation_id": investigation.name,
                    "base": "missing",
                    "parent_run_ids": parents,
                },
            )
            assert refused.is_error
            assert len(list(runs.glob("*/*/run.toml"))) == count

    asyncio.run(exercise())


@pytest.mark.gpu
@pytest.mark.parametrize("phase", ["before_compute", "during_compute"])
def test_mcp_disconnect_preserves_cuda_deadline_and_ownership(
    study,
    gpu_repository,
    tmp_path,
    phase,
    record_property,
):
    from datetime import UTC, datetime, timedelta

    from test_simulate import protocol_files

    from entanglix.investigations import create_investigation, record_user_decision
    from entanglix.runner import _digest, abort_run, acquire_gpu, select_gpu

    protocol, _ = protocol_files(tmp_path, device="cuda")
    _, _, terms, runs, _ = study
    terms = copy.deepcopy(terms)
    terms["scope"] = {
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
    }
    terms["evidence_requirements"]["checks"] = {}
    terms["limits"] = {"deadline": (datetime.now(UTC) + timedelta(seconds=40)).isoformat()}
    investigation = Path(create_investigation(terms, runs_root=runs)["path"])
    record_user_decision(investigation, "start", revision=1)
    device = select_gpu()
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

    async def exercise():
        async with client(
            gpu_repository,
            runs,
            tmp_path / "state",
            device=device,
            extras=("cuquantum", "differentiable-cuda12"),
        ) as session:
            start = time.monotonic()
            launch = await call(
                session,
                "simulate",
                {"protocol": str(protocol), "investigation_id": investigation.name, "base": "cuda"},
            )
            record_property("cuda_mcp_launch_to_manifest_seconds", time.monotonic() - start)
            run = Path(launch["path"])
            await asyncio.to_thread(wait_for, lambda: (run / "runner/started.toml").exists())
            if phase == "during_compute":
                await asyncio.to_thread(
                    wait_for, lambda: "evolving" in (run / "run.log").read_text()
                )
            return run

    run = asyncio.run(exercise())  # stdio connection is gone; supervisor remains
    # Use the exact state directory from the server's explicit configuration.
    try:
        with pytest.raises(ValueError, match="still owned"):
            acquire_gpu(device)
        record = read_run(run, wait_seconds=50)
        assert record["status"] == "aborted", record["log_tail"]
        assert "deadline" in record["log_tail"]
        assert record["finished"]["usage"]["gpu_seconds"] > 0
        with acquire_gpu(device):
            pass
        audit = _load_record(
            runs / "_audit/calls" / record["manifest"]["launch_id"] / "outcome.toml"
        )
        assert audit["links"]["run_id"] == run.name
        assert audit["links"]["investigation_revision"] == 1
    finally:
        if read_run(run)["status"] == "active":
            abort_run(run)


@pytest.mark.parametrize("spelling", ["absolute", "relative", "symlink"])
def test_record_paths_stay_within_configured_runs_root(
    study,
    run_kwargs,
    runner_repository,
    tmp_path,
    spelling,
):
    from entanglix.investigations import create_investigation, record_user_decision
    from entanglix.runner import create_run

    _, _, terms, runs, _ = study
    foreign = tmp_path / "foreign"
    with create_run(**(run_kwargs | {"runs_root": foreign})) as owner:
        run = owner.path
    investigation = Path(create_investigation(terms, runs_root=foreign)["path"])
    record_user_decision(investigation, "start", revision=1)
    (runs / "foreign-link").symlink_to(foreign, target_is_directory=True)
    before = {
        str(p.relative_to(foreign)): p.read_bytes() for p in foreign.rglob("*") if p.is_file()
    }

    def address(path):
        if spelling == "absolute":
            return str(path)
        if spelling == "relative":
            return str(Path("..") / "foreign" / path.relative_to(foreign))
        return str(runs / "foreign-link" / path.relative_to(foreign))

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            for name, arguments in (
                ("read_run", {"run": address(run)}),
                ("reconcile_run", {"run": address(run), "last_observed_state": "interrupted"}),
                ("abort_run", {"run": address(run)}),
                ("rerun_script", {"run": address(run)}),
                ("rerun_simulation", {"run": address(run)}),
                ("rerun_sweep", {"run": address(run)}),
                ("rerun_control", {"run": address(run)}),
                ("check", {"run": address(run), "checks": ["integrity"]}),
                ("read_investigation", {"investigation": address(investigation)}),
                ("revise_investigation", {"investigation": address(investigation), "spec": terms}),
                ("record_finding", {"investigation": address(investigation), "finding": {}}),
                (
                    "stop_investigation",
                    {"investigation": address(investigation), "reason": "user_stop"},
                ),
            ):
                result = await session.call_tool(name, arguments)
                assert result.is_error, name
                assert "outside configured runs root" in result.content[0].text, result

    asyncio.run(exercise())
    assert before == {
        str(p.relative_to(foreign)): p.read_bytes() for p in foreign.rglob("*") if p.is_file()
    }
    assert not list_runs(runs)


def test_mcp_wait_ends_at_interruption_and_concurrent_reads_agree(
    study,
    run_kwargs,
    runner_repository,
    tmp_path,
):
    from entanglix.runner import create_run

    runs = study[3]
    owner = create_run(**(run_kwargs | {"runs_root": runs}))
    run = owner.path
    module = runner_repository / "src/entanglix/mcp.py"
    marker = tmp_path / "waiting"
    statement = "await asyncio.sleep(min(remaining, 0.25))"
    module.write_text(
        module.read_text().replace(
            statement,
            f"Path({str(marker)!r}).touch()\n            {statement}",
        )
    )

    async def exercise():
        async with client(runner_repository, runs, tmp_path / "state") as session:
            # Relative paths use the server's configured root, not its process cwd.
            args = {"run": str(run.relative_to(runs)), "wait_seconds": 30}
            waiter = asyncio.create_task(call(session, "read_run", args))
            await asyncio.to_thread(wait_for, marker.exists)
            assert not waiter.done()
            owner.close()  # runner lost before starting a job: no terminal record
            assert (await asyncio.wait_for(waiter, timeout=3))["status"] == "interrupted"
            for _ in range(40):
                reads = await asyncio.gather(
                    *(call(session, "read_run", {"run": str(run)}) for _ in range(4)),
                    call(session, "list_runs"),
                )
                assert {item["status"] for item in reads[:4]} == {"interrupted"}
                assert reads[4]["rows"][0][reads[4]["columns"].index("status")] == "interrupted"
            assert await call(session, "read_run", {"run": str(run), "detail": True}) == json.loads(
                await cli_result(runner_repository, "read-run", str(run))
            )

    try:
        asyncio.run(exercise())
    finally:
        owner.close()


@pytest.mark.parametrize("custom_state", [False, True])
def test_optional_state_directory_uses_runner_default(
    runner_repository,
    tmp_path,
    monkeypatch,
    custom_state,
):
    from entanglix.runner import _state_directory, acquire_gpu

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    if custom_state:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "custom-state"))
    else:
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    runs = tmp_path / "runs"

    async def exercise():
        async with client(runner_repository, runs, None) as session:
            await call(session, "list_capabilities")

    asyncio.run(exercise())
    record = _load_record(next((runs / "_audit/sessions").glob("*.toml")))
    device = "GPU-state-default-test"
    with acquire_gpu(device) as lease:
        assert Path(record["configuration"]["gpu_lock_directory"]) == lease.record_path.parent
    assert record["configuration"]["state_directory"] == str(_state_directory())
