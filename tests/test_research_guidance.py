"""Shipped guidance reaches initialization and immutable connection provenance."""

import asyncio
import re
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from entanglix.mcp import supplied_guidance
from entanglix.runner import _load_record

ROOT = Path(__file__).resolve().parents[1]


def test_skills_name_existing_operations_references():
    for skill in (ROOT / "skills").glob("*/SKILL.md"):
        named = re.findall(r"`(reference/operations/[^`]+\.md)`", skill.read_text())
        assert named, skill
        assert all((ROOT / "docs/operations" / Path(path).name).is_file() for path in named)


@pytest.mark.parametrize("mode", ["instruction-only", "runtime-configured"])
def test_shipped_guidance_and_declared_restrictions_reach_connection(tmp_path, mode):
    import sys

    expected, metadata = supplied_guidance(ROOT)
    assert {item["name"] for item in metadata} == {
        "entanglix-research-conduct",
        "system-modeler",
        "simulate",
        "design-sweep",
        "control",
        "calibrate",
        "gpu-budget",
    }
    assert all(re.fullmatch(r"\d+\.\d+\.\d+", item["version"]) for item in metadata)
    # Procedures named by the actual conduct remain available as plain files.
    procedures = set(re.findall(r"`(skills/[^`]+/SKILL.md)`", expected))
    assert procedures == {
        f"skills/{name}/SKILL.md"
        for name in (
            "system-modeler",
            "simulate",
            "design-sweep",
            "control",
            "calibrate",
            "gpu-budget",
        )
    }
    for path in procedures:
        assert (ROOT / path).is_file()

    async def exercise():
        args = [
            "-m", "entanglix.mcp", "--runs-root", str(tmp_path / "runs"),
            "--device", "", "--runner-extras", "--command-restriction", mode,
        ]  # fmt: skip
        async with stdio_client(StdioServerParameters(command=sys.executable, args=args)) as (
            reader,
            writer,
        ):
            async with ClientSession(reader, writer) as session:
                initialized = await session.initialize()
                assert initialized.instructions == expected
                await session.call_tool("list_runs", {})

    asyncio.run(exercise())
    audit = _load_record(next((tmp_path / "runs/_audit/sessions").glob("*.toml")))
    assert audit["supplied_guidance"] == metadata
    assert audit["guidance_status"] == "supplied"
    assert audit["loaded_skills"] == "unknown"
    assert audit["command_restriction"] == {"source": "operator-declared", "mode": mode}
