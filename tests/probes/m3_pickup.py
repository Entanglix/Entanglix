"""Rebuild the recorded M3 pick-up evidence; never launch a job or runtime.

The retained operator scripts used m3_evaluation.workspace and session. This
summary reuses its record reader, scientific claim checks and runtime-usage parser.
"""

import argparse
import collections
import json
import runpy
import tomllib
from datetime import datetime
from pathlib import Path

EVALUATION = runpy.run_path(str(Path(__file__).with_name("m3_evaluation.py")))


def report(root):
    operator, workspace = root / "operator", root / "normal"
    before = json.loads((operator / "before-pickup.json").read_text())
    phase = json.loads((operator / "pickup-phase.json").read_text())
    phase["runtime_usage"] = EVALUATION["runtime_usage"](phase["transcript"])
    earlier = {item["run_id"] for item in before["attempts"]}
    summary = EVALUATION["summarize"](workspace, "normal", [phase], scripted_runs=earlier)
    client = EVALUATION["CLIENT"]["Client"](workspace)
    later = {run: record for run, (_, record) in client.records.items() if run not in earlier}
    links = EVALUATION["launch_links"](workspace)
    points, checks = collections.Counter(), collections.Counter()
    for run, (_, record) in client.records.items():
        manifest = record["manifest"]
        if manifest["protocol"] in {"simulate", "sweep"}:
            varied = manifest["launch"]["varied_values"]
            for values in [varied] if isinstance(varied, dict) else varied:
                points[manifest["investigation_base"], json.dumps(values, sort_keys=True)] += 1
        else:
            checks[
                tuple(manifest["parent_run_ids"]),
                tuple(sorted(item["name"] for item in record["result"]["report"]["checks"])),
            ] += 1
    first = min(later.values(), key=lambda record: record["start"]["started_at"])
    first_requested_at = min(
        tomllib.loads((path.parent / "request.toml").read_text())["timestamp"]
        for path in (workspace / "runs/_audit/calls").glob("*/outcome.toml")
        if tomllib.loads(path.read_text()).get("links", {}).get("run_id")
        == first["manifest"]["run_id"]
    )
    events = [json.loads(line) for line in Path(phase["transcript"]).read_text().splitlines()]
    before_first_job = [
        event for event in events if datetime.fromisoformat(event["timestamp"]) < first_requested_at
    ]
    initial_windows = [
        event["payload"]["rate_limits"]["primary"]["used_percent"]
        for event in before_first_job
        if event.get("payload", {}).get("type") == "token_count"
        and (event["payload"].get("rate_limits") or {}).get("primary")
    ]
    first_source = first["manifest"]["parent_run_ids"]
    missing_check_first = (
        first["manifest"]["protocol"] == "verification"
        and len(first_source) == 1
        and first_source[0] in earlier
        and client.records[first_source[0]][1]["manifest"]["investigation_base"].startswith(
            "refined-"
        )
        and not any(
            record["manifest"]["protocol"] == "verification"
            and first_source[0] in record["manifest"]["parent_run_ids"]
            for run, (_, record) in client.records.items()
            if run in earlier
        )
    )
    retained = json.loads((operator / "pickup-result.json").read_text())
    # Compare the rebuilt measurements with the original report, including all
    # claim source IDs and residuals, without copying its conclusions.
    matches = (
        summary["usage"] == retained["usage"]
        and summary["claims"] == retained["claims"]
        and summary["report"] == retained["report"]
        and len(later) == retained["jobs_after_pickup"]
        and phase["runtime_usage"]["input_tokens"]
        == retained["tokens"]["total_token_usage"]["input_tokens"]
        and phase["runtime_usage"]["cached_input_tokens"]
        == retained["tokens"]["total_token_usage"]["cached_input_tokens"]
        and phase["runtime_usage"]["output_tokens"]
        == retained["tokens"]["total_token_usage"]["output_tokens"]
        and phase["runtime_usage"]["model_requests"] == retained["model_requests"]
        and all(
            phase["runtime_usage"]["windows_used_percent"][name]
            == {
                "first": retained["window_percent"][0][name],
                "last": retained["window_percent"][-1][name],
            }
            for name in ("five_hour", "weekly")
        )
    )
    repeated_points = {str(key): count for key, count in points.items() if count > 1}
    repeated_checks = {str(key): count for key, count in checks.items() if count > 1}
    unlinked = sorted(set(later) - set(links))
    brackets = [
        {key: claim[key] for key in ("ej_ghz", "stage", "width_ghz", "confirmed")}
        | {
            "lower": claim["endpoints"][0]["amplitude_ghz"],
            "upper": claim["endpoints"][1]["amplitude_ghz"],
        }
        for claim in summary["claims"]
    ]
    baseline_matches = brackets == retained["baseline_brackets"]
    result = {
        "workspace": str(root),
        "before_pickup": {
            key: before[key]
            for key in (
                "state",
                "usage",
                "checkpoint",
                "client_exit_code",
                "terms_equal_evaluation",
            )
        },
        "phase": phase,
        "summary": summary,
        "jobs_after_pickup": len(later),
        "launch_tools_after_pickup": dict(
            collections.Counter(links[run] for run in later if run in links)
        ),
        "repeated_points": repeated_points,
        "repeated_checks": repeated_checks,
        "later_attempts_without_launch_link": unlinked,
        "missing_refined_check_issued_first": missing_check_first,
        "before_first_job_request": {
            "model_requests": sum(
                event["type"] == "token_usage_record" for event in before_first_job
            ),
            "five_hour_window_used_percent": {
                "first": initial_windows[0],
                "last": initial_windows[-1],
            },
        },
        "retained_result_matches": matches,
        "baseline_brackets_match": baseline_matches,
        "summary_builder_sha256": EVALUATION["GUIDANCE"]["digest"](Path(__file__).resolve()),
        "evaluation_builder_sha256": EVALUATION["GUIDANCE"]["digest"](
            Path(__file__).with_name("m3_evaluation.py")
        ),
        "source_sha256": {
            str(path.relative_to(root)): EVALUATION["GUIDANCE"]["digest"](path)
            for path in [
                *sorted(operator.glob("*.py")),
                operator / "before-pickup.json",
                operator / "pickup-result.json",
                operator / "pickup-prompt.txt",
                Path(phase["transcript"]),
            ]
        },
    }
    result["acceptance"] = (
        summary["acceptance"]
        and matches
        and baseline_matches
        and missing_check_first
        and not repeated_points
        and not repeated_checks
        and not unlinked
        and before["client_exit_code"] == -9
        and before["terms_equal_evaluation"]
        and len(earlier) == 12
        and len(later) == 18
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    EVALUATION["dump"](args.output, report(args.workspace.expanduser().resolve()))


if __name__ == "__main__":
    main()
