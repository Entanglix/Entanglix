"""Scripted stdio control jobs share CLI results, admission and linked reruns."""

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from test_control import control_files as control_files
from test_control_investigations import SEED, approve
from test_control_investigations import control_study as control_study
from test_control_investigations import plugin_repository as plugin_repository
from test_control_investigations import varied_study as varied_study
from test_mcp import call, cli_result, client

from entanglix.investigations import read_investigation
from entanglix.runner import _load_record, read_run


def test_mcp_control_cli_results_checks_and_linked_rerun(control_study, tmp_path):
    protocol, _, root, investigation, repository = control_study

    async def exercise():
        cli = json.loads(
            await cli_result(
                repository, "control", str(protocol), "--runs-root", str(root),
                "--investigation", investigation.name, "--base", "gate",
            )
        )  # fmt: skip
        assert cli["exit_code"] == 0
        candidate = read_run(Path(cli["path"]))
        async with client(repository, root, tmp_path / "state") as session:
            launch = await call(
                session,
                "control",
                {"protocol": str(protocol), "investigation_id": investigation.name, "base": "gate"},
            )
            run = launch["path"]
            record = await call(
                session, "read_run", {"run": run, "wait_seconds": 30, "detail": True}
            )
            assert record["status"] == "completed", record["log_tail"]
            assert record == json.loads(await cli_result(repository, "read-run", run))
            assert record["result"] == candidate["result"]
            assert record["verification"] == []
            checked = await call(
                session, "check", {"run": run, "checks": ["replay", "target", "constraints"]}
            )
            report = await call(session, "read_run", {"run": checked["path"], "wait_seconds": 30})
            assert report["status"] == "completed", report["log_tail"]
            assert all(c["outcome"] == "pass" for c in report["result"]["report"]["checks"])
            table = await call(
                session,
                "read_investigation",
                {"investigation": str(investigation), "view": "attempts"},
            )
            row = next(
                dict(zip(table["columns"], row, strict=True))
                for row in table["rows"]
                if row[table["columns"].index("path")] == str(Path(run).relative_to(root))
            )
            assert row["checks"][0][:2] == [checked["run_id"], True]
            replay = await call(session, "rerun_control", {"run": run})
            repeated = await call(session, "read_run", {"run": replay["path"], "wait_seconds": 30})
            assert repeated["status"] == "completed", repeated["log_tail"]
            assert repeated["manifest"]["parent_run_ids"] == [launch["run_id"]]
            assert repeated["manifest"]["investigation_id"] == investigation.name
            assert repeated["result"] == record["result"]
            assert repeated["verification"] == []
            return [("control", launch), ("check", checked), ("rerun_control", replay)]

    jobs = asyncio.run(exercise())
    assert read_investigation(investigation)["usage"]["jobs"] == 4
    for name, job in jobs:
        audit = root / "_audit/calls" / job["launch_id"]
        request = _load_record(audit / "request.toml")
        outcome = _load_record(audit / "outcome.toml")
        assert request["tool"] == name and request["permission_class"] == "Controlled job"
        assert outcome["links"]["run_id"] == job["run_id"]
        assert outcome["links"]["investigation_id"] == investigation.name


def test_mcp_control_refusals_leave_no_attempt_or_allowance(control_study, tmp_path):
    protocol, _, root, investigation, repository = control_study
    approve(control_study, limits={"job_count": 1})

    async def exercise():
        with pytest.raises(subprocess.CalledProcessError) as cli:
            await cli_result(
                repository, "control", str(protocol), "--runs-root", str(root),
                "--investigation", investigation.name, "--base", "missing",
            )  # fmt: skip
        assert "unknown investigation base: missing" in cli.value.stderr
        async with client(repository, root, tmp_path / "state") as session:
            args = {
                "protocol": str(protocol),
                "investigation_id": investigation.name,
                "base": "gate",
            }
            for request, message in (
                (args | {"base": "missing"}, "unknown investigation base: missing"),
                (
                    args | {"values": {"pulse.seed": 8}},
                    "varied values name targets outside the investigation scope",
                ),
            ):
                refused = await session.call_tool("control", request)
                assert refused.is_error and message in refused.content[0].text
                assert list(root.glob("*/*/run.toml")) == []
                assert read_investigation(investigation)["usage"]["jobs"] == 0
            job = await call(session, "control", args)
            completed = await call(session, "read_run", {"run": job["path"], "wait_seconds": 30})
            assert completed["status"] == "completed", completed["log_tail"]
            count = list(root.glob("*/*/run.toml"))
            for name, request in (("control", args), ("rerun_control", {"run": job["path"]})):
                refused = await session.call_tool(name, request)
                assert refused.is_error and "job_count limit exhausted" in refused.content[0].text
                assert list(root.glob("*/*/run.toml")) == count

    asyncio.run(exercise())
    assert read_investigation(investigation)["usage"]["jobs"] == 1


def test_mcp_control_values_vary_inside_the_base(varied_study, tmp_path):
    protocol, _, root, investigation, repository = varied_study

    async def exercise():
        async with client(repository, root, tmp_path / "state") as session:
            job = await call(
                session,
                "control",
                {
                    "protocol": str(protocol),
                    "investigation_id": investigation.name,
                    "base": "gate",
                    "values": {"parameters.gamma": 0.03, "pulse.seed": SEED},
                },
            )
            record = await call(
                session, "read_run", {"run": job["path"], "wait_seconds": 30, "detail": True}
            )
            assert record["status"] == "completed", record["log_tail"]
            return Path(record["path"]), record["manifest"]

    run, manifest = asyncio.run(exercise())
    assert manifest["launch"]["varied_values"] == {"parameters.gamma": 0.03, "pulse.seed": SEED}
    assert _load_record(run / "spec/control.toml")["pulse"]["seed"] == SEED
    assert _load_record(run / "spec/system.toml")["parameters"]["gamma"]["value"] == 0.03


def test_mcp_control_records_parent_lineage_as_the_cli(control_study, tmp_path):
    protocol, _, root, _, repository = control_study
    parents = ["r-screen-a", "r-screen-b"]

    async def exercise():
        cli = json.loads(
            await cli_result(
                repository, "control", str(protocol), "--runs-root", str(root),
                "--parent-run", parents[0], "--parent-run", parents[1],
            )
        )  # fmt: skip
        async with client(repository, root, tmp_path / "state") as session:
            launch = await call(
                session, "control", {"protocol": str(protocol), "parent_run_ids": parents}
            )
            record = await call(
                session, "read_run", {"run": launch["path"], "wait_seconds": 30, "detail": True}
            )
            assert record["status"] == "completed", record["log_tail"]
            return cli, record

    cli, record = asyncio.run(exercise())
    assert (
        record["manifest"]["parent_run_ids"]
        == read_run(Path(cli["path"]))["manifest"]["parent_run_ids"]
        == parents
    )
