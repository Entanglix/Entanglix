"""Optional stdio transport over the CLI's backing operations."""

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import secrets
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from entanglix import __version__, investigations, runner
from entanglix.launches import (
    prepare_launch,
    read_launch,
    reserve_launch,
    start_supervisor,
    validate_wait,
)
from entanglix.mcp_views import finding_ack, investigation_view, run_view, runs_view

JOBS = {
    "control",
    "rerun_control",
    "simulate",
    "sweep",
    "run_script",
    "rerun_simulation",
    "rerun_sweep",
    "rerun_script",
    "check",
}
READS = {
    "validate_system",
    "list_capabilities",
    "list_runs",
    "read_run",
    "list_investigations",
    "read_investigation",
}
WRITES = {"create_investigation", "revise_investigation", "record_finding", "reconcile_run"}
WAIT_CAP = 30.0


def _json(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _plain(value):
    return json.loads(_json(value))


def supplied_guidance(repository):
    """Read only explicitly versioned conduct/skills; drafts are not supplied."""
    instructions = []
    supplied = []
    paths = [
        repository / "docs/research-agent.md",
        *sorted((repository / "skills").glob("*/SKILL.md")),
    ]
    for path in paths:
        if not path.is_file():
            continue
        content = path.read_text()
        if not content.startswith("---\n"):
            continue
        header, separator, body = content[4:].partition("\n---\n")
        if not separator:
            raise ValueError(f"guidance has unterminated metadata: {path}")
        metadata = dict(line.split(":", 1) for line in header.splitlines() if ":" in line)
        metadata = {key.strip(): value.strip().strip("\"'") for key, value in metadata.items()}
        if not all(metadata.get(key) for key in ("name", "description", "version")):
            raise ValueError(f"guidance needs name, description and version: {path}")
        supplied.append({key: metadata[key] for key in ("name", "description", "version")})
        if path.name == "research-agent.md":
            instructions.append(body.strip())
    return "\n\n".join(instructions) or None, supplied


@dataclass(frozen=True)
class Configuration:
    runs_root: Path
    device: str
    runner_extras: tuple[str, ...]
    state_directory: Path = field(default_factory=runner._state_directory)
    repository: Path = Path(__file__).resolve().parents[2]
    command_restriction: str = "instruction-only"

    def environment(self):
        return os.environ | {
            "CUDA_VISIBLE_DEVICES": self.device,
            "ENTANGLIX_RUNNER_EXTRAS": ",".join(self.runner_extras),
            "XDG_STATE_HOME": str(self.state_directory.resolve()),
        }


@dataclass
class Call:
    path: Path
    owner: runner.RunHandle | None = None


current_call: ContextVar[Call] = ContextVar("mcp_call")


def build_server(config: Configuration):
    root = config.runs_root.resolve()
    repository = config.repository.resolve()
    session_id = f"s{secrets.token_hex(16)}"
    session_recorded = False
    audit = root / "_audit"
    instructions, supplied = supplied_guidance(repository)
    handlers = {}

    def research_path(value):
        path = Path(value)
        resolved = (path if path.is_absolute() else root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ToolError(f"research record path is outside configured runs root {root}: {value}")
        return resolved

    async def audit_call(ctx, call_next):
        nonlocal session_recorded
        if not session_recorded:
            client = (
                (ctx.params or {}).get("clientInfo", {})
                if ctx.method == "initialize"
                else (ctx.meta or {}).get("io.modelcontextprotocol/clientInfo", {})
            )
            publish = audit / "sessions" / f"{session_id}.toml"
            publish.parent.mkdir(parents=True, exist_ok=True)
            runner.publish_toml(
                publish,
                {
                    "record_version": 1,
                    "session_id": session_id,
                    "timestamp": datetime.now(UTC),
                    "client": {
                        "source": "client-reported",
                        "name": client.get("name", "unknown"),
                        "version": client.get("version", "unknown"),
                    },
                    "configuration": {
                        "source": "server-observed",
                        "runs_root": str(root),
                        "device": config.device,
                        "runner_extras": list(config.runner_extras),
                        "state_directory": str(config.state_directory.resolve()),
                        "gpu_lock_directory": str(
                            config.state_directory.resolve() / "entanglix/gpu"
                        ),
                    },
                    "supplied_guidance": supplied,
                    "guidance_status": "supplied" if supplied else "none supplied",
                    "command_restriction": {
                        "source": "operator-declared",
                        "mode": config.command_restriction,
                    },
                    "loaded_skills": "unknown",
                    "model": "unknown",
                    "transcript_location": "unknown",
                },
            )
            session_recorded = True
        if ctx.method != "tools/call":
            return await call_next(ctx)
        params = ctx.params or {}
        name = params.get("name", "unknown")
        if not isinstance(name, str):
            name = "invalid tool name"
        raw_arguments = params.get("arguments") or {}
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
        owner = None
        if name in JOBS:
            investigation_id = arguments.get("investigation_id")
            if isinstance(arguments.get("run"), str):
                try:
                    investigation_id = runner._load_record(
                        research_path(arguments["run"]) / "run.toml"
                    ).get("investigation_id")
                except (OSError, ValueError, ToolError):
                    pass
            owner = reserve_launch(
                root,
                name,
                investigation_id=investigation_id if isinstance(investigation_id, str) else None,
            )
        call_id = owner.path.name if owner else f"c{secrets.token_hex(16)}"
        path = audit / "calls" / call_id
        try:
            path.mkdir(parents=True)
        except BaseException:
            if owner is not None:
                owner.close()
            raise
        call = Call(path, owner)
        token = current_call.set(call)
        permission = (
            "Read-only"
            if name in READS
            else "Controlled write"
            if name in WRITES
            else "Controlled job"
            if name in JOBS or name in {"abort_run", "stop_investigation"}
            else "Unavailable"
        )
        outcome = "interrupted"
        result = None
        try:
            runner.publish_toml(
                path / "request.toml",
                {
                    "record_version": 1,
                    "timestamp": datetime.now(UTC),
                    "session_id": session_id,
                    "tool": name,
                    "permission_class": permission,
                    "arguments_sha256": _hash(raw_arguments),
                    "client_metadata": {
                        "source": "client-reported",
                        "json": _json(ctx.meta) if ctx.meta else "{}",
                    },
                    **({"launch_path": str(owner.path)} if owner else {}),
                },
            )
            if name in handlers:
                unknown = set(arguments) - set(inspect.signature(handlers[name]).parameters)
                if unknown:
                    result = CallToolResult(
                        is_error=True,
                        content=[
                            TextContent(
                                type="text",
                                text=f"unsupported request fields: {', '.join(sorted(unknown))}",
                            )
                        ],
                    )
                    outcome = "error"
                    return result
            result = await call_next(ctx)
            outcome = "error" if _plain(result).get("isError", False) else "returned"
            return result
        except Exception as exc:
            outcome = "error"
            result = {"error": str(exc)}
            raise
        finally:
            if call.owner is not None:
                call.owner.close()
            binding = read_launch(owner.path) if owner else {}
            data = {}
            if result is not None and not _plain(result).get("isError", False):
                content = _plain(result).get("content", [])
                if content and content[0].get("type") == "text":
                    data = json.loads(content[0]["text"])
            manifest = data.get("manifest", {})
            links = {
                key: value
                for key, value in {
                    "investigation_id": data.get(
                        "investigation_id", manifest.get("investigation_id")
                    ),
                    "investigation_revision": data.get(
                        "revision",
                        data.get("active_revision", manifest.get("investigation_revision")),
                    ),
                    "proposed_revision": data.get("proposed_revision"),
                    "run_id": manifest.get("run_id", data.get("run_id")),
                }.items()
                if value is not None
            }
            for field, key in (("investigation", "investigation_id"), ("run", "run_id")):
                if isinstance(arguments.get(field), str):
                    try:
                        links.setdefault(
                            "source_run_id" if owner and field == "run" else key,
                            research_path(arguments[field]).name,
                        )
                    except ToolError:
                        pass
            links.update({key: value for key, value in binding.items() if value is not None})
            runner.publish_toml(
                path / "outcome.toml",
                {
                    "record_version": 1,
                    "timestamp": datetime.now(UTC),
                    "outcome": outcome,
                    "result_sha256": _hash(result) if result is not None else "unknown",
                    **({"links": links} if links else {}),
                },
            )
            current_call.reset(token)

    server = MCPServer(
        "entanglix", version=__version__, instructions=instructions, middleware=[audit_call]
    )

    async def backing(function, *args, **kwargs):
        try:
            return _plain(await asyncio.to_thread(function, *args, **kwargs))
        except Exception as exc:
            raise ToolError(runner.failure_message(exc)) from exc

    async def job(name, **arguments):
        call = current_call.get()
        try:
            for field in ("run", "source"):
                if arguments.get(field) is not None:
                    arguments[field] = research_path(arguments[field])
            owner = prepare_launch(
                name,
                arguments
                | {"repository": repository, **({"runs_root": root} if name != "check" else {})},
                runs_root=root,
                owner=call.owner,
            )
            process = start_supervisor(owner, environment=config.environment())
            owner.close()
            call.owner = None
            while True:
                result = await asyncio.to_thread(read_launch, owner.path)
                if result["status"] != "pending":
                    if process.poll() is not None:
                        process.wait()
                    if result["status"] == "refused":
                        raise ToolError(f"launch {result['launch_id']} refused: {result['reason']}")
                    return result
                await asyncio.sleep(0.01)
        except Exception as exc:
            raise ToolError(runner.failure_message(exc)) from exc

    def tool(function):
        name = function.__name__
        handlers[name] = function

        @wraps(function)
        async def encoded(**arguments):
            data = await function(**arguments)
            if isinstance(data, list):
                data = {"result": data}
            return CallToolResult(content=[TextContent(type="text", text=_json(data))])

        server.add_tool(
            encoded,
            structured_output=False,
            annotations=ToolAnnotations(readOnlyHint=name in READS),
        )
        return function

    @tool
    async def validate_system(system: str) -> dict[str, object]:
        """Validate a system spec and report its operators and provenance."""
        from entanglix.cli import _operator_meaning
        from entanglix.models import load_system

        def validate():
            loaded = load_system(Path(system))
            return {
                "id": loaded.id,
                "basis": loaded.basis,
                "dimensions": loaded.dimensions,
                "operators": {
                    name: _operator_meaning(name, loaded.provenance)
                    for name in sorted(loaded.operators)
                },
                "provenance": loaded.provenance,
            }

        return await backing(validate)

    @tool
    async def list_capabilities() -> dict[str, object]:
        """Report the backing operations and packages installed in the runner environment."""
        return await backing(runner.list_capabilities, repository)

    @tool
    async def list_runs(after: int = 0) -> dict[str, object]:
        """Read a 50-row attempt/launch table; follow next_after with after."""
        records = await backing(runner.list_runs, root)
        return await backing(runs_view, records, root, after)

    @tool
    async def read_run(
        run: str, wait_seconds: float = 0, detail: bool = False
    ) -> dict[str, object]:
        """Read a compact attempt or launch; detail=True returns the full record. Wait up to 30 seconds."""
        try:
            validate_wait(wait_seconds)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        expires = asyncio.get_running_loop().time() + min(wait_seconds, WAIT_CAP)
        path = research_path(run)
        while True:
            record = await backing(
                read_launch if path.parent.name == "_launches" else runner._read_attempt, path
            )
            if record["status"] not in {"active", "pending"}:
                break
            remaining = expires - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(remaining, 0.25))
        record = await backing(runner.read_run, path)
        return record if detail else run_view(record)

    @tool
    async def list_investigations() -> list[dict[str, object]]:
        """List investigation states in the configured runs root."""
        return await backing(investigations.list_investigations, root)

    @tool
    async def read_investigation(
        investigation: str,
        view: str = "status",
        after: int = 0,
        event_id: str | None = None,
    ) -> dict[str, object]:
        """Read status, or a paged attempts/events table; event_id returns one full event."""
        record = await backing(investigations.read_investigation, research_path(investigation))
        return await backing(investigation_view, record, view=view, after=after, event_id=event_id)

    @tool
    async def simulate(
        protocol: str,
        investigation_id: str | None = None,
        base: str | None = None,
        values: dict[str, float] | None = None,
    ) -> dict[str, object]:
        """Admit a simulation and return its ID while its supervisor executes it."""
        return await job(
            "simulate",
            protocol=protocol,
            investigation_id=investigation_id,
            base=base,
            values=values,
        )

    @tool
    async def sweep(
        protocol: str, investigation_id: str | None = None, base: str | None = None
    ) -> dict[str, object]:
        """Admit a sweep and return its ID while its supervisor executes it."""
        return await job("sweep", protocol=protocol, investigation_id=investigation_id, base=base)

    @tool
    async def control(
        protocol: str, investigation_id: str | None = None, base: str | None = None
    ) -> dict[str, object]:
        """Admit a CPU control solve against an exact approved base when investigating."""
        return await job("control", protocol=protocol, investigation_id=investigation_id, base=base)

    @tool
    async def rerun_control(run: str) -> dict[str, object]:
        """Re-solve frozen control inputs as a new linked attempt under current terms."""
        return await job("rerun_control", run=run)

    @tool
    async def run_script(
        system: str,
        script: str,
        inputs: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        investigation_id: str | None = None,
        base: str | None = None,
    ) -> dict[str, object]:
        """Admit an authored CPU script through the runner's validation and import scan."""
        return await job(
            "run_script",
            system=system,
            script=script,
            inputs=inputs,
            timeout_seconds=timeout_seconds,
            investigation_id=investigation_id,
            base=base,
        )

    @tool
    async def rerun_script(run: str, timeout_seconds: float | None = None) -> dict[str, object]:
        """Replay frozen script inputs as a new linked attempt under current terms."""
        return await job("rerun_script", run=run, timeout_seconds=timeout_seconds)

    @tool
    async def rerun_simulation(run: str) -> dict[str, object]:
        """Replay frozen simulation inputs as a new linked attempt under current terms."""
        return await job("rerun_simulation", run=run)

    @tool
    async def rerun_sweep(run: str) -> dict[str, object]:
        """Replay a frozen grid as a new linked attempt under current terms."""
        return await job("rerun_sweep", run=run)

    @tool
    async def check(
        run: str,
        checks: list[str],
        timeout_seconds: float | None = None,
        refinement: str | None = None,
    ) -> dict[str, object]:
        """Admit explicitly named checks as a linked verification job."""
        return await job(
            "check",
            source=run,
            checks=checks,
            timeout_seconds=timeout_seconds,
            refinement=refinement,
        )

    @tool
    async def abort_run(run: str) -> dict[str, object]:
        """Stop a running attempt's process group."""
        return await backing(runner.abort_run, research_path(run))

    @tool
    async def reconcile_run(run: str, last_observed_state: str) -> dict[str, object]:
        """Record uncertain usage only after ownership and process-group checks permit it."""
        await backing(
            runner.record_reconciliation,
            research_path(run),
            last_observed_state=last_observed_state,
        )
        return await backing(runner.read_run, research_path(run))

    @tool
    async def create_investigation(spec: dict) -> dict[str, object]:
        """Draft terms; only the operator CLI can make a revision effective."""
        return await backing(investigations.create_investigation, spec, runs_root=root)

    @tool
    async def revise_investigation(investigation: str, spec: dict) -> dict[str, object]:
        """Propose terms; this never approves them or changes the effective revision."""
        return await backing(
            investigations.revise_investigation, research_path(investigation), spec
        )

    @tool
    async def record_finding(investigation: str, finding: dict) -> dict[str, object]:
        """Append interpretation; usage and check status come from retained records."""
        path = research_path(investigation)
        event = await backing(investigations.record_finding, path, finding)
        record = await backing(investigations.read_investigation, path)
        return finding_ack(event, record)

    @tool
    async def stop_investigation(
        investigation: str, reason: str, note: str = ""
    ) -> dict[str, object]:
        """Close admission and cancel outstanding attempts, retaining their usage."""
        return await backing(
            investigations.stop_investigation, research_path(investigation), reason, note=note
        )

    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Entanglix stdio MCP server")
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument(
        "--device", required=True, help="physical device UUID/index, or empty for CPU only"
    )
    parser.add_argument("--runner-extras", nargs="*", required=True)
    parser.add_argument(
        "--state-directory",
        type=Path,
        help="runner state directory (default: XDG_STATE_HOME or ~/.local/state)",
    )
    parser.add_argument(
        "--command-restriction",
        choices=("instruction-only", "runtime-configured"),
        default="instruction-only",
        help="operator-declared shell restriction; not an observed enforcement guarantee",
    )
    args = parser.parse_args(argv)
    config = Configuration(
        args.runs_root,
        args.device,
        tuple(args.runner_extras),
        args.state_directory or runner._state_directory(),
        command_restriction=args.command_restriction,
    )
    build_server(config).run(transport="stdio")


if __name__ == "__main__":
    main()
