"""Catalog tools share CLI records, retain permission classes, and never promote."""

import asyncio
import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from test_catalog import control_run, tree
from test_control import control_files as control_files
from test_mcp import call, cli_result, client

from entanglix.catalog import promote
from entanglix.runner import _load_record


def staged_content(record):
    """Compare separately staged copies apart from their identities and creation times."""
    content = copy.deepcopy(record)
    for name in ("path", "staged_id", "created_at"):
        content.pop(name)
    for name in ("device", "pulse"):
        if name in content:
            content[name].pop("path")
    return content


def test_catalog_staging_and_reads_match_cli(control_files, runner_repository, tmp_path):
    source = control_run(control_files, tmp_path)
    root = source.parent.parent
    before = tree(source)
    claim = "a two-level device and its pulse"

    async def exercise():
        cli = json.loads(
            await cli_result(
                runner_repository, "register-result", str(source),
                "--design", "device_and_pulse", "--claim", claim,
            )
        )  # fmt: skip
        async with client(runner_repository, root, tmp_path / "state") as session:
            tools = {tool.name: tool for tool in (await session.list_tools()).tools}
            assert {"register_result", "list_catalog", "read_entry"} <= tools.keys()
            assert "promote" not in tools
            assert not tools["register_result"].annotations.read_only_hint
            assert tools["list_catalog"].annotations.read_only_hint
            assert tools["read_entry"].annotations.read_only_hint

            staged = await call(
                session,
                "register_result",
                {
                    "run": str(source.relative_to(root)),
                    "design": "device_and_pulse",
                    "claim": claim,
                },
            )
            assert staged_content(staged) == staged_content(cli)
            assert not Path(staged["path"]).is_absolute()
            assert root / staged["path"] == root / "_catalog/staging" / staged["staged_id"]
            for name in staged["device"]["files"]:
                assert (root / staged["device"]["path"] / name).read_bytes() == (
                    root / cli["device"]["path"] / name
                ).read_bytes()
            assert (root / staged["pulse"]["path"]).read_bytes() == (
                root / cli["pulse"]["path"]
            ).read_bytes()

            # The fixture performs the operator action outside the research tools.
            entry = promote(staged["staged_id"], root)
            for identifier in (cli["staged_id"], staged["staged_id"], entry["entry_id"]):
                read = await call(session, "read_entry", {"identifier": identifier})
                assert read == json.loads(
                    await cli_result(
                        runner_repository, "read-entry", identifier, "--runs-root", str(root)
                    )
                )
            listed = await call(session, "list_catalog")
            assert listed["result"] == json.loads(
                await cli_result(runner_repository, "list-catalog", "--runs-root", str(root))
            )
            assert {row["status"] for row in listed["result"]} == {"staged", "promoted"}

    asyncio.run(exercise())
    assert tree(source) == before
    for path in (root / "_audit/calls").glob("*/request.toml"):
        request = _load_record(path)
        expected = "Controlled write" if request["tool"] == "register_result" else "Read-only"
        assert request["permission_class"] == expected
        outcome = _load_record(path.with_name("outcome.toml"))
        assert outcome["outcome"] == "returned"
        if request["tool"] == "register_result":
            assert outcome["links"]["run_id"] == source.name


def test_catalog_refusals_publish_no_candidate(control_files, runner_repository, tmp_path):
    source = control_run(control_files, tmp_path / "inside")
    root = source.parent.parent
    outside = control_run(control_files, tmp_path / "outside")
    nested = control_run(control_files, root / "nested")
    before = tree(source)

    async def exercise():
        async with client(runner_repository, root, tmp_path / "state") as session:
            arguments = {"run": str(source), "design": "device", "claim": "candidate"}
            for changes, message in (
                ({"point": 0}, "point selects a design of a sweep run"),
                ({"claim": " "}, "claim"),
                ({"run": str(outside)}, "outside configured runs root"),
                ({"run": str(nested)}, "catalog source must be at"),
                ({"approval": True}, "unsupported request fields"),
                ({"runs_root": str(outside.parent.parent)}, "unsupported request fields"),
            ):
                refused = await session.call_tool("register_result", arguments | changes)
                assert refused.is_error and message in refused.content[0].text
                assert not (root / "_catalog").exists()
                assert not (outside.parent.parent / "_catalog").exists()
                assert not (nested.parent.parent / "_catalog").exists()
            refused = await session.call_tool("promote", {"identifier": "not-an-entry"})
            assert refused.is_error
            assert (await call(session, "list_catalog"))["result"] == []
            assert not (root / "_catalog").exists()

    asyncio.run(exercise())
    assert tree(source) == before
