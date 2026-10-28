"""Project setup: verbatim copies, fixed runtime templates, conflicts, and refresh."""

import json
import re
import shutil
import tomllib
from pathlib import Path

import pytest
import tomli_w

from entanglix import installation
from entanglix.project_setup import REFERENCES, setup_project

REPOSITORY = Path(__file__).resolve().parents[1]
BLOCKED = [
    "Bash(entanglix investigation-decision:*)",
    "Bash(entanglix apply-device-update:*)",
    "Bash(entanglix promote:*)",
]


def installed(root: Path, version="0.1.0", components=("base",)) -> Path:
    """An installed version with the shipped guidance and its records."""
    shutil.copytree(REPOSITORY / "skills", root / "skills")
    (root / "docs/operations").mkdir(parents=True)
    shutil.copyfile(REPOSITORY / "docs/research-agent.md", root / "docs/research-agent.md")
    for name in REFERENCES:
        shutil.copyfile(
            REPOSITORY / f"docs/operations/{name}.md", root / f"docs/operations/{name}.md"
        )
    (root / ".installation").mkdir()
    record = {"version": version, "uv": "/uv", "cache_dir": "/cache"}
    (root / ".installation/installation.toml").write_text(tomli_w.dumps(record))
    state = {"components": list(components), "runner_extras": ["differentiable"]}
    (root / ".installation/components.toml").write_text(tomli_w.dumps(state))
    return root


def project_record(directory: Path) -> dict:
    return tomllib.loads((directory / ".entanglix/project.toml").read_text())


@pytest.fixture(autouse=True)
def _no_state_override(monkeypatch):
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr(installation, "gpus", lambda: [])


def test_claude_and_codex_receive_copies_and_their_configuration(tmp_path):
    root = installed(tmp_path / "0.1.0")
    directory = tmp_path / "project"
    result = setup_project(directory, runtimes=("codex", "claude"), root=root)
    conduct = (root / "docs/research-agent.md").read_bytes()
    assert (directory / "CLAUDE.md").read_bytes() == conduct
    assert (directory / "AGENTS.md").read_bytes() == conduct
    for skill in (root / "skills").glob("*/SKILL.md"):
        assert (directory / skill.relative_to(root)).read_bytes() == skill.read_bytes()
    for name in REFERENCES:
        copied = directory / f"reference/operations/{name}.md"
        assert copied.read_bytes() == (root / f"docs/operations/{name}.md").read_bytes()
    arguments = [
        "-m",
        "entanglix.mcp",
        "--runs-root",
        str(directory / "runs"),
        "--device",
        "",
        "--max-wait",
        "3600",
    ]
    claude = json.loads((directory / ".mcp.json").read_text())["mcpServers"]["entanglix"]
    assert claude == {
        "command": str(root / ".venv/bin/python"),
        "args": arguments,
        "timeout": 3660000,
    }
    settings = json.loads((directory / ".claude/settings.json").read_text())
    assert settings == {"permissions": {"deny": BLOCKED}}
    codex = tomllib.loads((directory / ".codex/config.toml").read_text())["mcp_servers"]
    assert codex["entanglix"]["args"] == arguments
    assert codex["entanglix"]["tool_timeout_sec"] == 3660
    rules = (directory / ".codex/rules/operator.rules").read_text()
    assert 'decision="forbidden"' in rules and '"apply-device-update"' in rules
    assert (directory / ".agents/skills").readlink() == Path("../skills")
    assert (directory / "runs").is_dir()
    record = project_record(directory)
    assert record["installation_version"] == "0.1.0"
    assert record["runtimes"] == ["claude", "codex"] and record["device"] == ""
    assert record["links"] == {".agents/skills": "../skills"}
    assert any('CUDA_VISIBLE_DEVICES="" claude' in line for line in result["next"])
    assert any("close it first" in line for line in result["next"])


def test_on_request_lets_the_runtime_ask_before_each_decision_command(tmp_path):
    root = installed(tmp_path / "0.1.0")
    directory = tmp_path / "project"
    setup_project(directory, runtimes=("claude", "codex"), privilege="on-request", root=root)
    settings = json.loads((directory / ".claude/settings.json").read_text())
    assert settings == {"permissions": {"ask": BLOCKED}}
    assert 'decision="prompt"' in (directory / ".codex/rules/operator.rules").read_text()


def test_another_runtime_receives_copies_and_printed_entries(tmp_path):
    root = installed(tmp_path / "0.1.0")
    directory = tmp_path / "project"
    result = setup_project(directory, runtimes=("other",), root=root)
    assert (directory / "AGENTS.md").is_file()
    for name in (".mcp.json", ".claude", ".codex", ".agents"):
        assert not (directory / name).exists()
    printed = "\n".join(result["next"])
    assert str(root / ".venv/bin/python") in printed and "entanglix promote" in printed
    assert "claimed only for configurations a retained test covers" in printed


def test_a_file_setup_did_not_write_stops_it_before_any_write(tmp_path):
    root = installed(tmp_path / "0.1.0")
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / ".mcp.json").write_text('{"mcpServers": {}}\n')
    with pytest.raises(
        ValueError, match=r"\.mcp\.json exists and was not written by setup"
    ) as raised:
        setup_project(directory, runtimes=("claude",), root=root)
    assert "the MCP server entry is" in str(raised.value)
    assert sorted(path.name for path in directory.iterdir()) == [".mcp.json"]


@pytest.mark.parametrize("name", [".codex", "runs", ".entanglix", "skills", ".agents"])
def test_a_file_where_setup_needs_a_directory_stops_it_before_any_write(tmp_path, name):
    root = installed(tmp_path / "0.1.0")
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / name).write_text("mine\n")
    with pytest.raises(ValueError, match=rf"{re.escape(name)} exists and is not a directory"):
        setup_project(directory, runtimes=("claude", "codex"), root=root)
    assert [path.name for path in directory.iterdir()] == [name]
    # Once the file is moved, the first setup succeeds: nothing is left to conflict.
    (directory / name).unlink()
    setup_project(directory, runtimes=("claude", "codex"), root=root)


def test_a_second_setup_needs_refresh(tmp_path):
    root = installed(tmp_path / "0.1.0")
    setup_project(tmp_path / "project", runtimes=("claude",), root=root)
    with pytest.raises(ValueError, match="already set up; refresh it with --refresh"):
        setup_project(tmp_path / "project", runtimes=("claude",), root=root)


def test_refresh_refuses_an_edited_file_and_changes_nothing(tmp_path):
    root = installed(tmp_path / "0.1.0")
    directory = tmp_path / "project"
    setup_project(directory, runtimes=("claude",), root=root)
    (directory / "CLAUDE.md").write_text("my own notes\n")
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    newer = installed(tmp_path / "0.2.0", version="0.2.0")
    with pytest.raises(ValueError, match="CLAUDE.md was edited since setup wrote it"):
        setup_project(directory, refresh=True, root=newer)
    after = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    assert after == before and project_record(directory)["installation_version"] == "0.1.0"


def test_refresh_moves_the_project_to_the_newer_installation(tmp_path):
    root = installed(tmp_path / "0.1.0")
    directory = tmp_path / "project"
    setup_project(directory, runtimes=("claude", "codex"), root=root)
    (directory / "notes.md").write_text("mine\n")
    newer = installed(tmp_path / "0.2.0", version="0.2.0")
    skill = next((newer / "skills").glob("*/SKILL.md"))
    skill.write_text(skill.read_text() + "\nA newer procedure line.\n")
    (directory / ".mcp.json").unlink()  # A missing file is restored, not a conflict.
    setup_project(directory, refresh=True, root=newer)
    record = project_record(directory)
    assert record["installation_version"] == "0.2.0" and record["runtimes"] == ["claude", "codex"]
    assert (directory / skill.relative_to(newer)).read_text() == skill.read_text()
    entry = json.loads((directory / ".mcp.json").read_text())["mcpServers"]["entanglix"]
    assert entry["command"] == str(newer / ".venv/bin/python")
    assert (directory / "notes.md").read_text() == "mine\n"


def test_refresh_can_change_the_runtimes(tmp_path):
    root = installed(tmp_path / "0.1.0")
    directory = tmp_path / "project"
    setup_project(directory, runtimes=("claude", "codex"), root=root)
    setup_project(directory, runtimes=("claude",), refresh=True, root=root)
    record = project_record(directory)
    assert record["runtimes"] == ["claude"] and record["links"] == {}
    for name in (
        "AGENTS.md",
        ".codex/config.toml",
        ".codex/rules/operator.rules",
        ".agents/skills",
    ):
        assert not (directory / name).exists() and not (directory / name).is_symlink()
    assert (directory / "CLAUDE.md").is_file() and (directory / ".mcp.json").is_file()
    setup_project(directory, runtimes=("codex", "claude"), refresh=True, root=root)
    assert project_record(directory)["runtimes"] == ["claude", "codex"]
    assert (directory / ".agents/skills").readlink() == Path("../skills")


def test_auto_device_computes_on_a_gpu_only_with_the_gpu_component(tmp_path, monkeypatch):
    one = [{"uuid": "GPU-aaaa", "name": "Display"}]
    monkeypatch.setattr(installation, "gpus", lambda: one)
    base = installed(tmp_path / "base")
    setup_project(tmp_path / "base-project", runtimes=("claude",), root=base)
    assert project_record(tmp_path / "base-project")["device"] == ""
    gpu = installed(tmp_path / "gpu", components=("base", "gpu"))
    setup_project(tmp_path / "gpu-project", runtimes=("claude",), root=gpu)
    assert project_record(tmp_path / "gpu-project")["device"] == "GPU-aaaa"
    setup_project(tmp_path / "cpu-project", runtimes=("claude",), device="cpu", root=gpu)
    assert project_record(tmp_path / "cpu-project")["device"] == ""
    monkeypatch.setattr(installation, "gpus", lambda: [*one, {"uuid": "GPU-bbbb", "name": "B"}])
    with pytest.raises(ValueError, match="several GPUs found; choose one with --device UUID"):
        setup_project(tmp_path / "two-project", runtimes=("claude",), root=gpu)
    assert not (tmp_path / "two-project").exists()


def test_a_custom_state_directory_reaches_the_server_entry(tmp_path, monkeypatch):
    root = installed(tmp_path / "0.1.0")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    setup_project(tmp_path / "project", runtimes=("claude",), root=root)
    entry = json.loads((tmp_path / "project/.mcp.json").read_text())["mcpServers"]["entanglix"]
    position = entry["args"].index("--state-directory")
    assert entry["args"][position + 1] == str(tmp_path / "state")


def test_setup_requires_an_installation_and_a_runtime(tmp_path):
    with pytest.raises(ValueError, match="prepares projects for an installed Entanglix"):
        setup_project(tmp_path / "project", runtimes=("claude",), root=tmp_path)
    root = installed(tmp_path / "0.1.0")
    with pytest.raises(ValueError, match="--runtime names at least one"):
        setup_project(tmp_path / "project", root=root)
    with pytest.raises(ValueError, match="unknown runtimes"):
        setup_project(tmp_path / "project", runtimes=("cursor",), root=root)
