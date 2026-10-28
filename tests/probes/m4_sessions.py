"""Rebuild the recorded live control sessions; never launch a job or runtime.

Each workspace holds one started investigation with a `grape` and a `piccolo` base
and one headless Claude Code or Codex session asked to solve, check and report.
The summary is read from the retained runs, MCP audit and session output.
"""

import argparse
import collections
import hashlib
import json
import re
import tomllib
from pathlib import Path

from entanglix.investigations import read_investigation
from entanglix.runner import read_run

MCP = "mcp__entanglix__"


def tool_calls(workspace):
    """Yield (tool, arguments, failed) for both runtimes' JSON event streams."""
    names = {}
    for line in (workspace / "evidence/session.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        content = event.get("message", {}).get("content")
        for part in content if isinstance(content, list) else []:
            if part["type"] == "tool_use":
                names[part["id"]] = part["name"].removeprefix(MCP)
                yield names[part["id"]], part["input"], False
            elif part["type"] == "tool_result" and part.get("is_error"):
                yield names[part["tool_use_id"]], {}, True
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "mcp_tool_call":
            failed = bool(item.get("error")) or item.get("status") == "failed"
            yield item["tool"], item.get("arguments") or {}, False
            if failed:
                yield item["tool"], {}, True
        elif event.get("type") == "item.completed" and item.get("type") == "command_execution":
            yield "shell", {"command": item.get("command", "")}, False


def runtime(operator, workspace):
    phase = json.loads((operator / "session-phase.json").read_text())
    if "runtime_usage" in phase:
        usage = phase["runtime_usage"]
        return {
            "runtime": phase["runtime"],
            "model": phase["model"],
            "effort": phase["effort"],
            "elapsed_seconds": phase["elapsed_seconds"],
            "model_requests": usage["model_requests"],
            "input_tokens": usage["input_tokens"],
            "cached_input_tokens": usage["cached_input_tokens"],
            "output_tokens": usage["output_tokens"],
            "windows_used_percent": usage["windows_used_percent"],
            "conduct_in_initial_context": phase["conduct_in_initial_context"],
            "development_instructions_present": phase["development_instructions_present"],
        }
    transcript = (workspace / "evidence/session-transcript.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in transcript]
    first = next(index for index, row in enumerate(rows) if row.get("type") == "assistant")
    usage, windows = phase["usage"], phase["windows_used_percent"]
    cached = usage["cache_read_input_tokens"]
    return {
        "runtime": phase["runtime"],
        "model": phase["model"],
        "elapsed_seconds": phase["elapsed_seconds"],
        "model_requests": len(
            {
                row.get("requestId") or row["message"]["id"]
                for row in rows
                if row.get("type") == "assistant"
            }
        ),
        "input_tokens": usage["input_tokens"] + usage["cache_creation_input_tokens"] + cached,
        "cached_input_tokens": cached,
        "output_tokens": usage["output_tokens"],
        "windows_used_percent": {
            label: {"first": windows[0][name], "last": windows[-1][name]}
            for label, name in (("five_hour", "five_hour"), ("weekly", "seven_day"))
        },
        "conduct_in_initial_context": "Research-agent conduct" in json.dumps(rows[:first]),
        "development_instructions_present": "docs/development.md" in json.dumps(rows[:first]),
    }


def report(root):
    operator, workspace = root / "operator", root / "workspace"
    before = json.loads((operator / "before-session.json").read_text())
    record = read_investigation(Path(before["investigation"]))
    runs = {}
    for attempt in record["attempts"]:
        run = read_run(Path(attempt["path"]))
        runs[run["manifest"]["run_id"]] = run
    guidance = "AGENTS.md" if (workspace / "AGENTS.md").is_file() else "CLAUDE.md"
    version = re.compile(r"^version: (\S+)", re.M)
    calls = list(tool_calls(workspace))
    final = next(e for e in reversed(record["events"]) if e["kind"] == "finding")
    candidates = []
    for claim in json.loads((workspace / "evidence/claims.json").read_text())["candidates"]:
        source, check = runs[claim["run_id"]], runs[claim["check_run_id"]]
        checks = {item["name"]: item for item in check["result"]["report"]["checks"]}
        recorded = {
            "base": source["manifest"]["investigation_base"],
            "solver": source["result"]["solver"]["actual"],
            "converged": source["result"]["termination"]["converged"],
            "solver_reported_fidelity": source["result"]["metrics"]["solver_reported_fidelity"],
            "replayed_fidelity": checks["replay"]["fidelity"],
            "leakage": checks["replay"]["leakage"],
            "checks": {name: item["outcome"] for name, item in checks.items()},
        }
        candidates.append(
            recorded
            | {
                "run_id": claim["run_id"],
                "check_run_id": claim["check_run_id"],
                "problem_sha256": source["result"]["problem_sha256"],
                "check_links_source": check["manifest"]["parent_run_ids"] == [claim["run_id"]],
                "agent_claim_equals_record": all(claim[key] == recorded[key] for key in recorded),
            }
        )
    audit = workspace / "runs/_audit"
    sessions = sorted((audit / "sessions").glob("*.toml")) if (audit / "sessions").is_dir() else []
    requests = [
        tomllib.loads(path.read_text()) for path in (audit / "calls").glob("*/request.toml")
    ]
    entry = tomllib.loads(sessions[0].read_text()) if len(sessions) == 1 else {}
    stop = next((e for e in reversed(record["events"]) if e["kind"] == "stop"), {})
    return {
        "workspace": str(root),
        "conduct_version": version.search((workspace / guidance).read_text())[1],
        "control_skill_version": version.search(
            (workspace / "skills/control/SKILL.md").read_text()
        )[1],
        "session": runtime(operator, workspace)
        | {
            "tool_calls": dict(
                collections.Counter(name for name, _, failed in calls if not failed)
            ),
            "failed_calls": [name for name, _, failed in calls if failed],
            "run_reads_with_detail": sum(
                name == "read_run" and bool(arguments.get("detail")) for name, arguments, _ in calls
            ),
        },
        "connection_audit": {
            "entries": len(sessions),
            "client": entry.get("client"),
            "command_restriction": entry.get("command_restriction", {}).get("mode"),
            "call_audits": len(requests),
            "calls_name_the_entry": bool(entry)
            and {item["session_id"] for item in requests} == {entry["session_id"]},
        },
        "state": record["state"],
        "stop_reason": stop.get("reason"),
        "usage": record["usage"],
        "attempts": [
            {
                "run_id": run_id,
                "protocol": run["manifest"]["protocol"],
                "base": run["manifest"]["investigation_base"],
                "status": run["status"],
            }
            for run_id, run in runs.items()
        ],
        "final_finding": {
            "kind": final["finding"]["kind"],
            "runs_cited": len(set(final["finding"]["runs"]) & runs.keys()),
            "attempts": len(runs),
            "requested_checks_not_run": sorted(
                {
                    name
                    for item in final.get("evidence", [])
                    for name in item.get("requested_checks_not_run", [])
                }
            ),
        },
        "candidates": candidates,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "summary_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sessions": [report(path.expanduser().resolve()) for path in args.workspace],
    }
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
