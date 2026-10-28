"""Research project setup from fixed templates and verbatim copies of the shipped guidance."""

import hashlib
import json
import os
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import tomli_w

from entanglix import installation

RUNTIMES = ("claude", "codex", "other")
PRIVILEGES = ("none", "on-request")
REFERENCES = (
    "system-spec",
    "protocols",
    "investigations",
    "runs",
    "control",
    "calibration",
    "catalog",
)
DECISION_COMMANDS = ("investigation-decision", "apply-device-update", "promote")
PROJECT_RECORD = Path(".entanglix/project.toml")
MAX_WAIT = 3600


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def server_entry(root: Path, directory: Path, device: str) -> tuple[str, list[str]]:
    """The installation's absolute server command; the extras set is resolved per launch."""
    arguments = ["-m", "entanglix.mcp", "--runs-root", str(directory / "runs"), "--device", device]
    if "XDG_STATE_HOME" in os.environ:
        arguments += ["--state-directory", str(Path(os.environ["XDG_STATE_HOME"]).resolve())]
    arguments += ["--max-wait", str(MAX_WAIT)]
    return str(root / ".venv/bin/python"), arguments


def _files(root: Path, directory: Path, runtimes, device, privilege) -> tuple[dict, dict]:
    """Every file and link setup owns, by project-relative path."""
    conduct = (root / "docs/research-agent.md").read_bytes()
    files = {}
    links = {}
    if "claude" in runtimes:
        files["CLAUDE.md"] = conduct
    if "codex" in runtimes or "other" in runtimes:
        files["AGENTS.md"] = conduct
    for path in sorted((root / "skills").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[str(Path("skills") / path.relative_to(root / "skills"))] = path.read_bytes()
    for name in REFERENCES:
        files[f"reference/operations/{name}.md"] = (
            root / f"docs/operations/{name}.md"
        ).read_bytes()
    command, arguments = server_entry(root, directory, device)
    blocked = [f"Bash(entanglix {name}:*)" for name in DECISION_COMMANDS]
    if "claude" in runtimes:
        entry = {"command": command, "args": arguments, "timeout": (MAX_WAIT + 60) * 1000}
        files[".mcp.json"] = (
            json.dumps({"mcpServers": {"entanglix": entry}}, indent=2) + "\n"
        ).encode()
        permission = "deny" if privilege == "none" else "ask"
        settings = {"permissions": {permission: blocked}}
        files[".claude/settings.json"] = (json.dumps(settings, indent=2) + "\n").encode()
    if "codex" in runtimes:
        config = {
            "mcp_servers": {
                "entanglix": {
                    "command": command,
                    "args": arguments,
                    "startup_timeout_sec": 60,
                    "tool_timeout_sec": MAX_WAIT + 60,
                    "default_tools_approval_mode": "approve",
                }
            }
        }
        files[".codex/config.toml"] = tomli_w.dumps(config).encode()
        decision, justification = (
            ("forbidden", "Human action: the operator runs this in their own terminal.")
            if privilege == "none"
            else ("prompt", "Human action: run only the exact command the user asked for.")
        )
        files[".codex/rules/operator.rules"] = (
            "prefix_rule(\n"
            f'    pattern=["entanglix", {json.dumps(list(DECISION_COMMANDS))}],\n'
            f'    decision="{decision}",\n'
            f'    justification="{justification}",\n'
            ")\n"
        ).encode()
        links[".agents/skills"] = "../skills"
    return files, links


def _resolve_device(root: Path, device: str) -> str:
    """auto computes on a GPU only with the GPU component and exactly one device."""
    if device == "cpu":
        return ""
    if device != "auto":
        return device
    if "gpu" not in installation.components(root)["components"]:
        return ""
    devices = installation.gpus()
    if len(devices) > 1:
        listed = ", ".join(f"{gpu['uuid']} ({gpu['name']})" for gpu in devices)
        raise ValueError(f"several GPUs found; choose one with --device UUID: {listed}")
    return devices[0]["uuid"] if devices else ""


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _current(path: Path):
    if path.is_symlink():
        return ("link", os.readlink(path))
    if path.is_file():
        return ("file", _digest(path.read_bytes()))
    return None if not path.exists() else ("other", None)


def setup_project(
    directory: Path,
    *,
    runtimes: tuple[str, ...] | None = None,
    device: str | None = None,
    privilege: str | None = None,
    refresh: bool = False,
    root: Path = installation.ROOT,
) -> dict:
    """Write a project's guidance copies and runtime configuration, all or nothing."""
    installed = installation.record(root)
    if installed is None:
        raise ValueError(
            "setup-project prepares projects for an installed Entanglix; a development "
            "checkout follows the manual workspace steps in docs/operations/mcp.md"
        )
    directory = directory.expanduser().resolve()
    record_path = directory / PROJECT_RECORD
    previous = None
    if record_path.exists():
        if not refresh:
            raise ValueError(f"{directory} is already set up; refresh it with --refresh")
        with record_path.open("rb") as stream:
            previous = tomllib.load(stream)
        runtimes = runtimes or tuple(previous["runtimes"])
        privilege = privilege or previous["agent_command_privilege"]
        device = device if device is not None else previous["device"]
    elif refresh:
        raise ValueError(f"{directory} has no Entanglix project setup to refresh")
    if not runtimes:
        raise ValueError("--runtime names at least one of claude, codex and other")
    unknown = set(runtimes) - set(RUNTIMES)
    if unknown:
        raise ValueError(f"unknown runtimes: {sorted(unknown)}; choose from {list(RUNTIMES)}")
    privilege = privilege or "none"
    if privilege not in PRIVILEGES:
        raise ValueError(f"--agent-command-privilege is one of {list(PRIVILEGES)}")
    runtimes = tuple(sorted(set(runtimes)))
    device = _resolve_device(root, "auto" if device is None else device)
    files, links = _files(root, directory, runtimes, device, privilege)
    owned = {**(previous or {}).get("files", {}), **(previous or {}).get("links", {})}
    recorded_links = (previous or {}).get("links", {})
    conflicts = []
    for name in sorted({*files, *links, *owned}):
        state = _current(directory / name)
        if state is None:
            continue
        if name not in owned:
            conflicts.append(f"{name} exists and was not written by setup")
        elif (name in recorded_links and state != ("link", recorded_links[name])) or (
            name not in recorded_links and state != ("file", owned[name])
        ):
            conflicts.append(f"{name} was edited since setup wrote it")
    # The directories setup writes into are checked before any write too.
    required = {"runs"}
    for name in [*files, *links, str(PROJECT_RECORD)]:
        required.update(str(parent) for parent in Path(name).parents if parent != Path("."))
    for name in sorted(required):
        path = directory / name
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            conflicts.append(f"{name} exists and is not a directory")
    if conflicts:
        lines = [f"setup-project changed nothing in {directory}:", *conflicts]
        command, arguments = server_entry(root, directory, device)
        lines.append(
            "Restore or move those files and run it again, or add the entries by hand; "
            f"the MCP server entry is: {json.dumps({'command': command, 'args': arguments})}"
        )
        raise ValueError("\n".join(lines))
    for name in sorted(owned.keys() - files.keys() - links.keys()):
        (directory / name).unlink(missing_ok=True)
    for name, content in files.items():
        _write(directory / name, content)
    for name, target in links.items():
        path = directory / name
        if not path.is_symlink():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
    (directory / "runs").mkdir(exist_ok=True)
    project = {
        "record_version": 1,
        "installation_version": installed["version"],
        "runtimes": list(runtimes),
        "device": device,
        "agent_command_privilege": privilege,
        "written_at": datetime.now(UTC),
        "files": {name: _digest(content) for name, content in files.items()},
        "links": links,
    }
    _write(record_path, tomli_w.dumps(project).encode())
    return {
        "directory": str(directory),
        "installation_version": installed["version"],
        "runtimes": list(runtimes),
        "device": device or "cpu",
        "agent_command_privilege": privilege,
        "next": next_steps(root, directory, runtimes, device),
    }


def next_steps(root: Path, directory: Path, runtimes, device: str) -> list[str]:
    """How to start the session that loads these files; setup starts no runtime."""
    lines = [
        "Start a fresh research session in this directory. A session already open here has "
        "not loaded these files and can still see CUDA: close it first."
    ]
    for runtime in runtimes:
        if runtime == "other":
            command, arguments = server_entry(root, directory, device)
            blocked = ", ".join(f"entanglix {name}" for name in DECISION_COMMANDS)
            lines += [
                f'Other runtime: start it in {directory} with CUDA_VISIBLE_DEVICES="", supply '
                "AGENTS.md as its instructions, and add this MCP server entry: "
                + json.dumps({"command": command, "args": arguments}),
                f"Configure it to block {blocked}, and keep its tool timeout above "
                f"{MAX_WAIT} seconds.",
            ]
        else:
            lines.append(f'cd {directory} && CUDA_VISIBLE_DEVICES="" {runtime}')
    lines.append(
        "These files configure the runtime; refusing or asking before the decision commands "
        "is claimed only for configurations a retained test covers."
    )
    return lines
