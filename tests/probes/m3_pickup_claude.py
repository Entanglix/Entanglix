"""Rebuild the recorded Claude Code pick-up evidence; never launch a job or runtime.

Each workspace holds an investigation whose scripted client was killed during a
solve, then one headless Claude Code session asked only to continue it. The
summary is read from the retained runs, MCP launch audit and session output.
"""

import argparse
import collections
import json
import re
import runpy
import tomllib
from pathlib import Path

from entanglix import runner
from entanglix.investigations import read_investigation

EVALUATION = runpy.run_path(str(Path(__file__).with_name("m3_evaluation.py")))
MCP = "mcp__entanglix__"


def report(root):
    operator, workspace = root / "operator", root / "normal"
    before = json.loads((operator / "before-pickup.json").read_text())
    events = [
        json.loads(line)
        for line in (workspace / "evidence/pickup.jsonl").read_text().splitlines()
        if line.strip()
    ]
    start = next(item for item in events if item.get("subtype") == "init")
    end = next(item for item in reversed(events) if item["type"] == "result")
    windows = [
        {
            name: round(100 * window["utilization"])
            for name, window in item["rate_limit_info"]["unifiedWindows"].items()
        }
        for item in events
        if item["type"] == "rate_limit_event"
    ]
    calls, tools, refused, rejected = {}, collections.Counter(), [], []
    for item in events:
        content = item.get("message", {}).get("content")
        for part in content if isinstance(content, list) else []:
            if part["type"] == "tool_use":
                calls[part["id"]] = part
                tools[part["name"].removeprefix(MCP)] += 1
            elif part["type"] == "tool_result":
                call = calls[part["tool_use_id"]]
                name, text = call["name"].removeprefix(MCP), json.dumps(part.get("content"))
                if size := re.search(r"result \(([\d,]+) characters", text):
                    rejected.append(
                        {
                            "tool": name,
                            "arguments": {
                                key: value
                                for key, value in call["input"].items()
                                if key != "investigation"
                            },
                            "characters": int(size[1].replace(",", "")),
                        }
                    )
                elif part.get("is_error"):
                    refused.append({"tool": name, "message": text.split("refused: ")[-1][:90]})
    records = {
        path.parent.name: runner.read_run(path.parent)
        for path in sorted((workspace / "runs").glob("*/*/run.toml"))
    }
    investigation = read_investigation(Path(before["investigation"]))
    earlier = {item["run_id"] for item in before["attempts"]}
    later = [run for run in records if run not in earlier]
    cut = before["cut_run"]["run_id"]
    samples, points, checks = {}, collections.Counter(), collections.Counter()
    for run, record in records.items():
        manifest = record["manifest"]
        if manifest["protocol"] == "verification":
            names = tuple(item["name"] for item in record["result"]["report"]["checks"])
            checks[manifest["parent_run_ids"][0], names] += 1
        elif record["status"] == "completed":
            varied = manifest["launch"].get("varied_values", {})
            points[manifest["investigation_base"], json.dumps(varied, sort_keys=True)] += 1
            metric = record["result"]["metrics"]["coherence"]
            if manifest["protocol"] == "sweep":
                for row, ej in enumerate(EVALUATION["CLIENT"]["FREQUENCIES"]):
                    for column, amplitude in enumerate((0.0, 0.001, 0.002)):
                        samples[ej, amplitude, "rwa"] = {
                            "coherence": metric[row][column],
                            "source_run_id": run,
                        }
            else:
                spec = tomllib.loads((Path(record["path"]) / "spec/simulate.toml").read_text())
                base = manifest["investigation_base"]
                samples[
                    spec["parameters"]["EJ"],
                    spec["evolution"]["drive_amplitudes_ghz"]["readout"],
                    "lab_refined" if base.startswith("refined-") else base,
                ] = {"coherence": metric, "source_run_id": run}
    claims = json.loads((workspace / "evidence/claims.json").read_text())
    brackets = []
    for claim in claims["brackets"]:
        item = EVALUATION["confirmation"](
            samples, claim["ej_ghz"], claim["lower_amplitude_ghz"], claim["upper_amplitude_ghz"]
        )
        brackets.append(
            {
                "ej_ghz": item["ej_ghz"],
                "stage": claim["stage"],
                "lower": claim["lower_amplitude_ghz"],
                "upper": claim["upper_amplitude_ghz"],
                "width_ghz": item["width_ghz"],
                "margins": [endpoint["margin"] for endpoint in item["endpoints"]],
                "refinement_changes": [e["refinement_change"] for e in item["endpoints"]],
                "confirmed": item["confirmed"],
                "claim_agrees": item["confirmed"] == claim["confirmed"],
            }
        )
    normal = [item for item in brackets if item["stage"] == "normal"]
    reports = [
        event["finding"]
        for event in investigation["events"]
        if event["kind"] == "finding" and event["finding"]["kind"] == "report"
    ]
    links = EVALUATION["launch_links"](workspace)
    reruns = [
        run
        for run in later
        if cut in records[run]["manifest"].get("parent_run_ids", [])
        and records[run]["manifest"]["protocol"] == "simulate"
    ]
    transcript = (workspace / "evidence/pickup-transcript.jsonl").read_text().splitlines()
    conduct = [
        line for line in (workspace / "CLAUDE.md").read_text().splitlines() if len(line) > 40
    ]
    initial = transcript[
        : next(i for i, line in enumerate(transcript) if json.loads(line)["type"] == "assistant")
    ]
    criteria = {
        "interrupted_attempt_reconciled": "reconciliation" in records[cut],
        "rerun_linked_to_interrupted_attempt": bool(reruns),
        "no_repeated_point_or_check": max([*points.values(), *checks.values()]) == 1,
        "report_cites_every_attempt": bool(reports) and set(records) <= set(reports[-1]["runs"]),
        "normal_brackets_confirmed": {item["ej_ghz"] for item in normal} == {13.5, 15.0}
        and all(item["confirmed"] and item["claim_agrees"] for item in normal),
        "stopped": investigation["state"] == "stopped",
        "later_jobs_launched_through_mcp": set(later) <= set(links),
    }
    return {
        "workspace": str(root),
        "runtime": f"Claude Code {start['claude_code_version']}",
        "model": start["model"],
        "conduct_version": re.search(
            r"^version: (\S+)", (workspace / "CLAUDE.md").read_text(), re.M
        )[1],
        "command_restriction": "instruction-only; the session had no shell tool",
        # Counted, not asserted whole: the transcript escapes some punctuation.
        "conduct_lines_before_first_reply": [
            sum(
                any(line in entry or json.dumps(line)[1:-1] in entry for entry in initial)
                for line in conduct
            ),
            len(conduct),
        ],
        "development_instructions_present": any(
            mark in "\n".join(transcript) for mark in EVALUATION["GUIDANCE"]["DEVELOPMENT_MARKERS"]
        ),
        "before_pickup": {
            key: before[key]
            for key in ("state", "usage", "checkpoint", "cut_run", "client_exit_code")
        },
        "session": {
            "is_error": end["is_error"],
            "model_requests": end["num_turns"],
            "elapsed_seconds": end["duration_ms"] / 1000,
            **{
                key: end["usage"][key]
                for key in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                    "output_tokens",
                )
            },
            "windows_used_percent": {"first": windows[0], "last": windows[-1]},
            "tool_calls": dict(sorted(tools.items())),
            "refused_calls": refused,
            "results_rejected_by_runtime": rejected,
        },
        "state": investigation["state"],
        "open_reviews": investigation["open_reviews"],
        "stop_reason": next(
            (e["reason"] for e in reversed(investigation["events"]) if e["kind"] == "stop"), None
        ),
        "usage": investigation["usage"],
        "jobs_after_pickup": len(later),
        "launch_tools_after_pickup": dict(collections.Counter(links[run] for run in later)),
        "repeated_points": [list(key) for key, count in points.items() if count > 1],
        "repeated_checks": [
            {"source_run_id": source, "checks": list(names)}
            for (source, names), count in checks.items()
            if count > 1
        ],
        "report_runs_cited": len(set(reports[-1]["runs"]) & set(records)) if reports else 0,
        "attempts": len(records),
        "brackets": brackets,
        "stages_beyond_normal_terms": sorted({item["stage"] for item in brackets} - {"normal"}),
        "criteria": criteria,
        "pickup_criteria_met": all(criteria.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    EVALUATION["dump"](
        args.output,
        {
            "summary_builder_sha256": EVALUATION["GUIDANCE"]["digest"](Path(__file__).resolve()),
            "runs": [report(path.expanduser().resolve()) for path in args.workspace],
        },
    )


if __name__ == "__main__":
    main()
