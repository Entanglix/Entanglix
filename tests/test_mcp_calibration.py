"""Scripted stdio calibration: calibrate, check and propose match the CLI; refusals leave nothing."""

import asyncio
import json
import tomllib
from pathlib import Path

import pytest
import tomli_w

pytest.importorskip("mcp")
from test_calibrate import write_spec
from test_calibration_checks import split_points
from test_mcp import call, cli_result, client

from entanglix.runner import _load_record, list_capabilities, read_run


def comparable(proposal):
    """A proposal's content apart from its own path, time, source run and check attempt IDs."""
    content = {
        key: value
        for key, value in proposal.items()
        if key not in {"path", "created_at", "source_run_id", "verification"}
    }
    checks = [attempt.get("checks") for attempt in proposal["verification"]]
    return content, checks


def test_mcp_calibration_matches_the_cli_and_stages_only_proposals(runner_repository, tmp_path):
    split_points(tmp_path, 22)
    protocol = write_spec(tmp_path, ["training.toml"])
    holdout = tmp_path / "holdout.toml"
    root = tmp_path / "runs"
    repository = runner_repository
    request = ["holdout", "identifiability"]

    async def exercise():
        cli = json.loads(
            await cli_result(repository, "calibrate", str(protocol), "--runs-root", str(root))
        )
        assert cli["exit_code"] == 0
        source = Path(cli["path"])
        cli_check = json.loads(
            await cli_result(
                repository, "check", str(source), *request,
                "--holdout", str(holdout), "--expect-coverage", "2", "0.9",
            )
        )  # fmt: skip
        cli_proposal = json.loads(
            await cli_result(repository, "propose-device-update", str(source))
        )
        async with client(repository, root, tmp_path / "state") as session:
            names = {tool.name for tool in (await session.list_tools()).tools}
            assert {"calibrate", "rerun_calibration", "propose_device_update"} <= names
            # Device-model write-back is absent from the research surface.
            assert names == set(list_capabilities()["operations"])
            assert not {name for name in names if "apply" in name}

            launch = await call(session, "calibrate", {"protocol": str(protocol)})
            run = launch["path"]
            record = await call(
                session, "read_run", {"run": run, "wait_seconds": 30, "detail": True}
            )
            assert record["status"] == "completed", record["log_tail"]
            assert record == json.loads(await cli_result(repository, "read-run", run))
            assert record["result"] == read_run(source)["result"]
            assert "holdout.toml" not in record["manifest"]["spec_sha256"]

            checked = await call(
                session,
                "check",
                {
                    "run": run,
                    "checks": request,
                    "holdout": str(holdout),
                    "expect_coverage": [2, 0.9],
                },
            )
            report = await call(
                session, "read_run", {"run": checked["path"], "wait_seconds": 30, "detail": True}
            )
            assert report["status"] == "completed", report["log_tail"]
            checks = report["result"]["report"]["checks"]
            assert checks == read_run(Path(cli_check["path"]))["result"]["report"]["checks"]
            assert checks[0]["outcome"] == "pass"
            assert (Path(checked["path"]) / "spec/holdout/holdout.toml").is_file()

            proposal = await call(session, "propose_device_update", {"run": run})
            assert Path(proposal["path"]) == Path(run) / "proposals/1.toml"
            assert comparable(proposal) == comparable(cli_proposal)
            assert proposal["verification"][0]["checks"] == [
                {"name": "holdout", "solution": 0, "outcome": "pass"},
                {"name": "identifiability", "solution": 0, "outcome": "descriptive"},
            ]

            replay = await call(session, "rerun_calibration", {"run": run})
            repeated = await call(
                session, "read_run", {"run": replay["path"], "wait_seconds": 30, "detail": True}
            )
            assert repeated["status"] == "completed", repeated["log_tail"]
            assert repeated["manifest"]["parent_run_ids"] == [launch["run_id"]]
            assert repeated["result"] == record["result"]

            # Refused requests leave no attempt and no proposal.
            attempts = sorted(root.glob("*/*/run.toml"))
            bad = tomllib.loads(protocol.read_text())
            bad["parameters"] = {"EX": {"scale": 0.1}}
            (tmp_path / "bad.toml").write_text(tomli_w.dumps(bad))
            for name, arguments, message in (
                (
                    "calibrate",
                    {"protocol": str(tmp_path / "bad.toml")},
                    "'EX' is not in the system",
                ),
                (
                    "check",
                    {"run": run, "checks": ["holdout"], "holdout": str(tmp_path / "training.toml")},
                    "the fitting input 'training.toml'",
                ),
                (
                    "propose_device_update",
                    {"run": checked["path"]},
                    "needs a completed calibrate run",
                ),
            ):
                refused = await session.call_tool(name, arguments)
                assert refused.is_error and message in refused.content[0].text, name
                assert sorted(root.glob("*/*/run.toml")) == attempts
            assert not (Path(checked["path"]) / "proposals").exists()
            return [("calibrate", launch), ("check", checked), ("rerun_calibration", replay)]

    jobs = asyncio.run(exercise())
    for name, job in jobs:
        audit = root / "_audit/calls" / job["launch_id"]
        assert _load_record(audit / "request.toml")["tool"] == name
        assert _load_record(audit / "request.toml")["permission_class"] == "Controlled job"
        assert _load_record(audit / "outcome.toml")["links"]["run_id"] == job["run_id"]
    writes = [
        _load_record(path)
        for path in (root / "_audit/calls").glob("*/request.toml")
        if _load_record(path)["tool"] == "propose_device_update"
    ]
    assert {item["permission_class"] for item in writes} == {"Controlled write"}
