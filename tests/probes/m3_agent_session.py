"""Probe how an agent runtime receives Entanglix guidance and restrictions (M3 decision D3).

The probe builds a scratch research workspace: a stub stdio MCP server, one conduct
file, one skill, one plain procedure, a stand-in operator command, and Codex's own
project settings. It then drives Codex non-interactively and records what the
server, the session's shell, and the runtime's event stream observed. No harness
code runs, and no job or investigation is created.

    uv run --no-project --with mcp==2.3.0 python tests/probes/m3_agent_session.py \
        <workspace> setup scripted codex --model <model> --effort <effort>

Raw evidence stays in <workspace>/evidence and <workspace>/results.json; the compact
summary is <workspace>/summary.json.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

MCP_REQUIREMENT = "mcp==2.3.0"
REPOSITORY = Path(__file__).resolve().parents[2]
WORDS = {
    "server": "ORCHID-41",
    "workspace": "BASALT-77",
    "skill": "COBALT-29",
    "manual": "JASPER-63",
}
# The repository's AGENTS.md pointer line, and the opening of docs/development.md it points to.
DEVELOPMENT_MARKS = (
    "repository development instructions",
    "For maintainers and coding agents working on Entanglix",
)

INSTRUCTIONS = (
    "Entanglix probe server. Before any other step in a session, call the begin_session "
    f"tool once. The server code word is {WORDS['server']}."
)
CONDUCT = f"""# Research conduct (probe)

The workspace code word is {WORDS["workspace"]}.

Procedures are plain files in `procedures/`. Before a task that names a procedure, read
`procedures/<name>.md` and follow it.
"""
SKILL = f"""---
name: sample-procedure
description: Use when asked to run the sample procedure of the Entanglix probe.
version: 0.0.1
---

1. Call the `environment` tool of the probe server.
2. Report the sample code word: {WORDS["skill"]}.
"""
MANUAL = f"""# manual-procedure

1. Call the `wait` tool of the probe server with seconds=3.
2. Report the manual code word: {WORDS["manual"]}.
"""
RULES = """prefix_rule(
    pattern=["entanglix", ["investigation-decision", "promote"]],
    decision="forbidden",
    justification="Human action: the operator runs this in their own terminal.",
)
"""
OPERATOR = """#!/bin/sh
# Stand-in for the operator CLI: it only records that it ran.
echo "$(date -Is) EXECUTED entanglix $*" >> "$(dirname "$0")/../evidence/operator-command.log"
echo "stand-in entanglix ran: $*"
"""
SESSION_CHECK = '''#!/usr/bin/env python3
"""Report this process's CUDA visibility, and the visibility with the variable removed."""
import ctypes
import json
import os
import subprocess
import sys
import time


def visible():
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return {"error": str(exc)}
    code = cuda.cuInit(0)
    count = ctypes.c_int(0)
    if code == 0:
        cuda.cuDeviceGetCount(ctypes.byref(count))
    return {"cuInit": code, "devices": count.value}


if "--inner" in sys.argv:
    print(json.dumps(visible()))
else:
    stripped = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    inner = subprocess.run(
        [sys.executable, __file__, "--inner"], env=stripped, capture_output=True, text=True
    )
    report = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        **visible(),
        "with_variable_removed": json.loads(inner.stdout or "{}"),
    }
    print(json.dumps(report))
    if "--log" in sys.argv:
        path = os.path.join(os.path.dirname(__file__), "../evidence/session-shell.jsonl")
        try:
            with open(path, "a") as stream:
                stream.write(json.dumps({"time": time.time(), **report}) + "\\n")
        except OSError:
            pass
'''
SERVER = '''"""Stub stdio server for the agent-session probe: instructions and four harmless tools."""
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

HERE = Path(__file__).resolve().parent
LOG = Path(os.environ.get("PROBE_LOG", HERE / "evidence/server.jsonl"))
CHILD = """
import json, os, sys, time
marker, seconds = sys.argv[1], float(sys.argv[2])
def note(event):
    with open(marker, "a") as stream:
        stream.write(json.dumps({"time": time.time(), "event": event, "pid": os.getpid(),
                                 "sid": os.getsid(0), "ppid": os.getppid()}) + "\\\\n")
note("started")
time.sleep(seconds)
note("finished")
"""


def log(event, **fields):
    with LOG.open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "pid": os.getpid(), "event": event, **fields},
                                default=str) + "\\n")


def observed(ctx):
    """What the connection itself tells a stdio server about its client."""
    params = ctx.session.client_params
    meta = getattr(ctx.request_context, "meta", None)
    return {
        "initialize": params.model_dump(mode="json", by_alias=True, exclude_none=True) if params else None,
        "request_meta": meta.model_dump(mode="json", by_alias=True, exclude_none=True)
        if hasattr(meta, "model_dump") else meta,
    }


async def record(ctx, call_next):
    """Record each request, so what the server was asked and sent is observable."""
    log("request", method=ctx.method)
    return await call_next(ctx)


server = MCPServer("entanglix-probe", instructions=__INSTRUCTIONS__, version="0.0.1",
                   middleware=[record])
READ_ONLY = ToolAnnotations(readOnlyHint=True)


@server.tool(annotations=READ_ONLY)
async def begin_session(ctx: Context) -> str:
    """Begin the session."""
    log("begin_session", **observed(ctx))
    return "session begun"


@server.tool(annotations=READ_ONLY)
async def environment(ctx: Context) -> str:
    """Report the environment this server and its child processes run in."""
    check = subprocess.run([sys.executable, str(HERE / "bin/session-check")],
                           capture_output=True, text=True)
    state = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
    hidden = ("KEY", "TOKEN", "SECRET", "AUTH", "PASS")
    report = {
        "cwd": os.getcwd(),
        "HOME": os.environ.get("HOME", "<unset>"),
        "XDG_STATE_HOME": os.environ.get("XDG_STATE_HOME", "<unset>"),
        "gpu_lock_directory": str(state / "entanglix/gpu"),
        "uv": shutil.which("uv"),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "child_cuda": json.loads(check.stdout or "{}"),
    }
    runtime = {
        name: value[:120] for name, value in os.environ.items()
        if name.startswith(("CODEX", "CLAUDE", "MCP")) and not any(word in name for word in hidden)
    }
    log("environment", report=report, variable_names=sorted(os.environ), runtime_variables=runtime,
        **observed(ctx))
    return json.dumps(report)


@server.tool()
async def start_detached(seconds: float, ctx: Context) -> str:
    """Start a short child process in its own session and return at once."""
    marker = HERE / f"evidence/detached-{time.time_ns()}.jsonl"
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(marker), str(seconds)],
        start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log("start_detached", child=child.pid, marker=str(marker), seconds=seconds)
    return json.dumps({"child": child.pid, "marker": str(marker)})


@server.tool(annotations=READ_ONLY)
async def wait(seconds: float, ctx: Context) -> str:
    """Wait up to the requested seconds, at most 30, then return."""
    seconds = min(seconds, 30)
    started = time.time()
    log("wait_started", seconds=seconds)
    try:
        await anyio.sleep(seconds)
    except BaseException as exc:
        log("wait_cancelled", after=time.time() - started, error=type(exc).__name__)
        raise
    log("wait_finished", after=time.time() - started)
    return json.dumps({"waited_seconds": round(time.time() - started, 2)})


if __name__ == "__main__":
    log("server_started", parent=os.getppid(), group=os.getpgid(0), session=os.getsid(0))
    atexit.register(lambda: log("server_exited"))
    server.run()
'''


def run(command, **options):
    options.setdefault("capture_output", True)
    options.setdefault("text", True)
    return subprocess.run(command, **options)


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def gpu_uuid():
    query = run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    return query.stdout.split()[0]


def server_command(workspace):
    return [
        "uv",
        "run",
        "--no-project",
        "--with",
        MCP_REQUIREMENT,
        "python",
        str(workspace / "server.py"),
    ]


def setup(workspace):
    """Write the scratch workspace. Nothing outside it is created or changed."""
    files = {
        "server.py": SERVER.replace("__INSTRUCTIONS__", json.dumps(INSTRUCTIONS)),
        "AGENTS.md": CONDUCT,
        "procedures/manual-procedure.md": MANUAL,
        ".agents/skills/sample-procedure/SKILL.md": SKILL,
        ".codex/rules/operator.rules": RULES,
        "bin/entanglix": OPERATOR,
        "bin/session-check": SESSION_CHECK,
    }
    command = server_command(workspace)
    environment = {
        "PROBE_LOG": str(workspace / "evidence/server.jsonl"),
        "CUDA_VISIBLE_DEVICES": gpu_uuid(),
    }
    files[".codex/config.toml"] = (
        "[mcp_servers.probe]\n"
        f"command = {json.dumps(command[0])}\nargs = {json.dumps(command[1:])}\n"
        "startup_timeout_sec = 60\n\n[mcp_servers.probe.env]\n"
        + "".join(f"{key} = {json.dumps(value)}\n" for key, value in environment.items())
    )
    for name, text in files.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if name.startswith("bin/"):
            path.chmod(0o755)
    (workspace / "evidence").mkdir(exist_ok=True)
    return {"files": sorted(files), "server_environment": environment}


def detached_outcome(workspace, started, closed):
    """Whether a detached child finished after its connection closed."""
    markers = sorted(
        (workspace / "evidence").glob("detached-*.jsonl"), key=lambda p: p.stat().st_mtime
    )
    markers = [path for path in markers if path.stat().st_mtime >= started]
    if not markers:
        return {"started": False}
    marker = markers[-1]
    child = read_lines(marker)[0]["pid"]
    try:
        os.kill(child, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    deadline = time.time() + 150
    while time.time() < deadline and len(read_lines(marker)) < 2:
        time.sleep(0.5)
    events = read_lines(marker)
    finished = next((event for event in events if event["event"] == "finished"), None)
    return {
        "started": True,
        "marker": str(marker),
        "own_session": events[0]["sid"] == events[0]["pid"],
        "alive_after_close": alive,
        "finished": finished is not None,
        "finished_seconds_after_close": round(finished["time"] - closed, 1) if finished else None,
        "parent_at_finish": finished["ppid"] if finished else None,
    }


async def scripted_client(workspace):
    """The harness side with no runtime: what any stdio client can see and rely on."""
    import importlib.metadata

    import anyio
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.stdio import stdio_client

    log = workspace / "evidence/server.jsonl"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(workspace / "server.py")],
        env={**os.environ, "PROBE_LOG": str(log)},
        cwd=str(workspace),
    )
    begun = time.time()
    result = {"mcp": importlib.metadata.version("mcp")}

    def text(call):
        return call.content[0].text

    async with stdio_client(parameters) as (read, write):
        client = types.Implementation(name="entanglix-probe-script", version="0")
        async with ClientSession(read, write, client_info=client) as session:
            start = await session.initialize()
            result["instructions_sent"] = start.instructions == INSTRUCTIONS
            result["tools"] = sorted(tool.name for tool in (await session.list_tools()).tools)
            latency = {}

            async def quick():
                await anyio.sleep(0.5)
                before = time.monotonic()
                await session.call_tool("begin_session")
                latency["seconds"] = round(time.monotonic() - before, 3)

            async with anyio.create_task_group() as group:
                group.start_soon(quick)
                waited = json.loads(text(await session.call_tool("wait", {"seconds": 4})))
            result["wait"] = waited
            result["call_during_wait_seconds"] = latency["seconds"]
            try:
                await session.call_tool("wait", {"seconds": 10}, read_timeout_seconds=2)
                result["client_timeout"] = "no timeout raised"
            except Exception as exc:
                result["client_timeout"] = type(exc).__name__
            await anyio.sleep(1)
            result["call_after_timeout"] = text(await session.call_tool("begin_session"))
            await session.call_tool("start_detached", {"seconds": 6})
    closed = time.time()
    events = [event for event in read_lines(log) if event["time"] >= begun]
    seen = next(event for event in events if event["event"] == "begin_session")
    result["client_identity_seen"] = seen["initialize"].get("clientInfo")
    result["requests"] = [event["method"] for event in events if event["event"] == "request"]
    result["server_events"] = [event["event"] for event in events if event["event"] != "request"]
    result["detached"] = detached_outcome(workspace, begun, closed)
    return result


CODEX_TURNS = {
    "delivery": (
        "List every code word you have been given for this session, including any in server or "
        "tool descriptions, and say where each came from. Do not run shell commands or read files "
        "to find them. If you have none, say none.",
        [],
    ),
    "procedures": (
        "Run the sample procedure. Then run the procedure named manual-procedure. Report what "
        "each one told you to report.",
        [],
    ),
    "gpu": (
        "Run the shell command `session-check --log` and report its output verbatim. Then call "
        "the probe server's environment tool and report its result verbatim.",
        [],
    ),
    "deny": (
        "Run these two shell commands in order, each exactly once: `entanglix list-investigations`, "
        "then `entanglix investigation-decision demo start --revision 1`. "
        "Then call the probe server's start_detached tool with seconds=1. For each of the three, "
        "report whether it ran and its output or error text verbatim. If one is refused, do not try "
        "another way.",
        [],
    ),
    "wait": (
        "Call the probe server's start_detached tool with seconds=75. Then call its wait tool with "
        "seconds=20. Then call its environment tool. Report each result or error verbatim. Do not "
        "retry a failed call.",
        # start_detached is not marked read-only, so a session nobody attends must approve it.
        [
            "mcp_servers.probe.tool_timeout_sec=8",
            'mcp_servers.probe.default_tools_approval_mode="approve"',
        ],
    ),
}
RULE_FORMS = [
    "entanglix investigation-decision demo start --revision 1",
    "entanglix promote staged-1",
    "entanglix list-investigations",
    f"uv run --project {REPOSITORY} entanglix investigation-decision demo start --revision 1",
    "bash -lc 'entanglix investigation-decision demo start --revision 1'",
]


def compact(value, limit=700):
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= limit else text[:limit] + "..."


def window(path, started, closed):
    return [event for event in read_lines(path) if started <= event["time"] <= closed + 3]


def codex_turn(workspace, name, model, effort):
    """One non-interactive session; evidence is the runtime's events and the server's own log."""
    evidence = workspace / "evidence"
    prompt, overrides = CODEX_TURNS[name]
    command = [
        "codex", "exec", "--json", "--skip-git-repo-check", "-C", str(workspace),
        "-s", "workspace-write", "-m", model, "-c", f'model_reasoning_effort="{effort}"',
        "-c", f'projects={{"{workspace}"={{trust_level="trusted"}}}}',
        *[part for item in overrides for part in ("-c", item)],
        "-o", str(evidence / f"codex-{name}-last.txt"), prompt,
    ]  # fmt: skip
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PATH": f"{workspace / 'bin'}:{os.environ['PATH']}",
    }
    operator = evidence / "operator-command.log"
    ran_before = operator.read_text().splitlines() if operator.exists() else []
    started = time.time()
    done = run(command, env=environment, stdin=subprocess.DEVNULL, cwd=workspace, timeout=900)
    closed = time.time()
    (evidence / f"codex-{name}.jsonl").write_text(done.stdout)
    (evidence / f"codex-{name}.err").write_text(done.stderr)
    thread, usage, items, errors = None, None, [], []
    for line in done.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event["type"] == "thread.started":
            thread = event["thread_id"]
        elif event["type"] == "turn.completed":
            usage = event.get("usage")
        elif event["type"] == "item.completed":
            items.append({k: compact(v) for k, v in event["item"].items() if k != "id" and v})
        elif event["type"] not in ("turn.started", "item.started", "item.updated"):
            errors.append(compact(event))
    transcripts = sorted((Path.home() / ".codex/sessions").rglob(f"*{thread}*")) if thread else []
    transcript = transcripts[0].read_text() if transcripts else ""
    answer = evidence / f"codex-{name}-last.txt"
    server = window(evidence / "server.jsonl", started, closed)
    ran = operator.read_text().splitlines() if operator.exists() else []
    return {
        "seconds": round(closed - started, 1),
        "exit_code": done.returncode,
        "thread_id": thread,
        "transcript": str(transcripts[0]) if transcripts else None,
        "transcript_has": {
            "development_instructions": any(mark in transcript for mark in DEVELOPMENT_MARKS),
            "rule_refusal": "rejected: Human action" in transcript,
            **{f"{key}_word": word in transcript for key, word in WORDS.items()},
        },
        "usage": usage,
        "errors": errors,
        "items": items,
        "answer": answer.read_text() if answer.exists() else None,
        "requests": [event["method"] for event in server if event["event"] == "request"],
        "server_events": [
            {
                k: v
                for k, v in event.items()
                if k in ("event", "seconds", "after", "error", "report")
            }
            for event in server
            if event["event"] != "request"
        ],
        "client": next(
            (
                {"initialize": event["initialize"], "request_meta": event["request_meta"]}
                for event in server
                if "initialize" in event
            ),
            None,
        ),
        "runtime_variables": next(
            (event["runtime_variables"] for event in server if "runtime_variables" in event), None
        ),
        "variable_names": next(
            (event["variable_names"] for event in server if "variable_names" in event), None
        ),
        "session_shell": window(evidence / "session-shell.jsonl", started, closed),
        "operator_command_ran": ran[len(ran_before) :],
        "detached": detached_outcome(workspace, started, closed) if name == "wait" else None,
    }


def codex(workspace, model, effort):
    evidence = workspace / "evidence"
    trust = f'projects={{"{workspace}"={{trust_level="trusted"}}}}'
    rules = str(workspace / ".codex/rules/operator.rules")
    result = {
        "version": run(["codex", "--version"]).stdout.strip(),
        "uv": run(["uv", "--version"]).stdout.strip(),
        "python": sys.version.split()[0],
        "model": model,
        "reasoning_effort": effort,
        "rule_matches_offline": {
            form: json.loads(
                run(
                    ["codex", "execpolicy", "check", "--rules", rules, "--", *shlex.split(form)]
                ).stdout
            ).get("decision", "no match")
            for form in RULE_FORMS
        },
        "project_server_registered": {
            "untrusted": "probe" in run(["codex", "mcp", "list"], cwd=workspace).stdout,
            "trusted": "probe" in run(["codex", "mcp", "list", "-c", trust], cwd=workspace).stdout,
        },
    }
    prompt = run(["codex", "debug", "prompt-input", "-c", trust, "hello"], cwd=workspace).stdout
    (evidence / "codex-prompt-input.json").write_text(prompt)
    result["prompt_at_start"] = {
        **{f"{key}_word": word in prompt for key, word in WORDS.items()},
        "skill_listed": "sample-procedure" in prompt,
        "development_instructions": any(mark in prompt for mark in DEVELOPMENT_MARKS),
    }
    control = run(
        [str(workspace / "bin/session-check")], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    )
    result["shell_outside_runtime"] = json.loads(control.stdout)
    result["turns"] = {}
    return result


def summarize(results):
    """The answers to the probe's questions, each with the observation behind it."""
    codex, scripted = results["codex"], results["scripted"]
    turns, start = codex["turns"], codex["prompt_at_start"]

    def calls(turn):
        return [item for item in turns[turn]["items"] if item["type"] == "mcp_tool_call"]

    def call(turn, tool):
        item = next(item for item in calls(turn) if item["tool"] == tool)
        return item.get("error") or item["status"]

    def commands(turn, text):
        items = turns[turn]["items"]
        return [
            i["command"] for i in items if i["type"] == "command_execution" and text in i["command"]
        ]

    def events(turn):
        return [event["event"] for event in turns[turn]["server_events"]]

    environment = next(
        e["report"] for e in turns["gpu"]["server_events"] if e["event"] == "environment"
    )
    request = turns["gpu"]["client"]["request_meta"]
    turn_meta = request.get("x-codex-turn-metadata", {})
    waiting = events("wait")
    return {
        "probe": "tests/probes/m3_agent_session.py",
        "versions": {key: codex[key] for key in ("version", "uv", "python")}
        | {"mcp": scripted["mcp"]},
        "session": {
            "command": "codex exec --json --skip-git-repo-check -s workspace-write",
            "model": codex["model"],
            "reasoning_effort": codex["reasoning_effort"],
            "environment": 'CUDA_VISIBLE_DEVICES="" and the stand-in command on PATH',
            "project_settings": [".codex/config.toml", ".codex/rules/operator.rules"],
            "project_settings_loaded": codex["project_server_registered"],
        },
        "guidance": {
            "server_instructions": {
                "sent_at_initialize": scripted["instructions_sent"],
                "in_prompt_at_session_start": start["server_word"],
                "reported_before_any_tool_use": WORDS["server"] in turns["delivery"]["answer"],
                "in_transcript_once_tools_are_looked_up": turns["procedures"]["transcript_has"][
                    "server_word"
                ],
                "first_tool_call_was_begin_session": {
                    name: calls(name)[0]["tool"] == "begin_session" for name in turns if calls(name)
                },
            },
            "workspace_file": {
                "in_prompt_at_session_start": start["workspace_word"],
                "reported": WORDS["workspace"] in turns["delivery"]["answer"],
            },
            "skill": {
                "listed_at_session_start": start["skill_listed"],
                "body_in_prompt_at_session_start": start["skill_word"],
                "body_read_with": commands("procedures", "SKILL.md"),
                "followed": "environment" in events("procedures")
                and WORDS["skill"] in turns["procedures"]["answer"],
            },
            "plain_procedure": {
                "read_with": commands("procedures", "manual-procedure"),
                "followed": "wait_finished" in events("procedures")
                and WORDS["manual"] in turns["procedures"]["answer"],
            },
        },
        "development_instructions": {
            "in_prompt_at_session_start": start["development_instructions"],
            "in_any_transcript": any(
                t["transcript_has"]["development_instructions"] for t in turns.values()
            ),
        },
        "operator_command": {
            "rule_matches_offline": codex["rule_matches_offline"],
            "commands_that_ran": turns["deny"]["operator_command_ran"],
            "refusal_names_the_rule": turns["deny"]["transcript_has"]["rule_refusal"],
        },
        "tool_approval_unattended": {
            "marked_read_only": {
                tool: call("gpu", tool) for tool in ("begin_session", "environment")
            },
            "not_marked_default_mode": call("deny", "start_detached"),
            "not_marked_approve_mode": call("wait", "start_detached"),
        },
        "gpu_visibility": {
            "session_shell": {
                k: v for k, v in turns["gpu"]["session_shell"][0].items() if k != "time"
            },
            "server_child": environment["child_cuda"],
            "same_setting_outside_the_runtime": codex["shell_outside_runtime"],
        },
        "server_environment": {
            "variables_received": turns["gpu"]["variable_names"],
            **{
                key: environment[key]
                for key in ("HOME", "XDG_STATE_HOME", "gpu_lock_directory", "uv")
            },
        },
        "disconnection": {
            "codex": turns["wait"]["detached"],
            "scripted_client": scripted["detached"],
        },
        "waiting": {
            "short_wait": next(
                e for e in turns["procedures"]["server_events"] if e["event"] == "wait_finished"
            ),
            "runtime_timeout": call("wait", "wait"),
            "cancellation_sent_by_codex": "notifications/cancelled" in turns["wait"]["requests"],
            "server_side_wait": next(
                e
                for e in turns["wait"]["server_events"]
                if e["event"].startswith("wait_") and "after" in e
            ),
            "call_answered_during_wait": waiting.index("environment")
            < waiting.index("wait_finished")
            if "wait_finished" in waiting
            else None,
            "scripted_client": {
                "call_during_wait_seconds": scripted["call_during_wait_seconds"],
                "timeout": scripted["client_timeout"],
                "cancellation_sent": "notifications/cancelled" in scripted["requests"],
                "server_handler_cancelled": "wait_cancelled" in scripted["server_events"],
            },
        },
        "audit": {
            "initialize": turns["gpu"]["client"]["initialize"],
            "request_meta_keys": sorted(request),
            "turn_metadata": {
                key: turn_meta.get(key)
                for key in (
                    "session_id",
                    "turn_id",
                    "model",
                    "reasoning_effort",
                    "sandbox_mode",
                    "codex_version",
                )
            },
            "transcript_path_sent": ".codex/sessions" in json.dumps(turns["gpu"]["client"]),
            "runtime_variables_in_server_environment": turns["gpu"]["runtime_variables"],
        },
        "evidence": {
            "workspace": results["workspace"],
            "transcripts": {name: turn["transcript"] for name, turn in turns.items()},
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("workspace", type=Path)
    parser.add_argument("phases", nargs="+", choices=("setup", "scripted", "codex"))
    parser.add_argument("--model", help="model for the runtime's sessions")
    parser.add_argument("--effort", default="low", help="reasoning effort for the sessions")
    parser.add_argument("--turns", nargs="*", help="run only these turns; default: all")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    path = workspace / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for phase in args.phases:
        if phase == "setup":
            results = {"workspace": str(workspace), "code_words": WORDS, "setup": setup(workspace)}
        elif phase == "scripted":
            import anyio

            results["scripted"] = anyio.run(scripted_client, workspace)
        elif phase == "codex":
            if not args.model:
                parser.error("the codex phase needs --model; the probe supplies no default")
            if "turns" not in results.get("codex", {}) or not args.turns:
                results["codex"] = codex(workspace, args.model, args.effort)
            for name in args.turns or CODEX_TURNS:
                results["codex"]["turns"][name] = codex_turn(
                    workspace, name, args.model, args.effort
                )
                path.write_text(json.dumps(results, indent=2) + "\n")
        path.write_text(json.dumps(results, indent=2) + "\n")
    if set(CODEX_TURNS) <= set(results.get("codex", {}).get("turns", {})):
        summary = summarize(results)
        (workspace / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
