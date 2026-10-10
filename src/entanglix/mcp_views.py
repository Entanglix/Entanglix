"""Compact research views; backing records and CLI output remain complete."""

import json
from pathlib import Path

from entanglix import runner

METRIC_LIMIT = 512


def metrics_view(metrics):
    return {
        name: value
        if len(json.dumps(value, separators=(",", ":"), default=str).encode()) <= METRIC_LIMIT
        else {"omitted": True, "reason": f"metric exceeds {METRIC_LIMIT} JSON bytes"}
        for name, value in metrics.items()
    }


def check_view(check):
    keys = {"name", "outcome", "scope", "coverage", "reason"}
    return {
        **{key: value for key, value in check.items() if key in keys},
        **({"detail_omitted": sorted(set(check) - keys)} if set(check) - keys else {}),
    }


def history_view(history, *, compact=False):
    metadata = {"run_id", "source_matches", "diagnostic"}
    check_keys = {"name", "outcome"}
    return [
        {
            key: value
            for key, value in item.items()
            if key != "checks"
            and (not compact or key in metadata or (key == "path" and "run_id" not in item))
        }
        | {
            "checks": [
                {key: value for key, value in check.items() if key in check_keys}
                if compact
                else check_view(check)
                for check in item.get("checks", [])
            ]
        }
        for item in history
    ]


def run_view(record):
    summary = dict(record)
    if "manifest" in record:
        omitted = {"spec_sha256", "code", "environment"} & record["manifest"].keys()
        summary["manifest"] = {
            key: value for key, value in record["manifest"].items() if key not in omitted
        } | ({"detail_omitted": sorted(omitted)} if omitted else {})
    if "result" in record:
        result = dict(record["result"])
        result["metrics"] = metrics_view(result.get("metrics", {}))
        if record.get("manifest", {}).get("protocol") == "verification" and "report" in result:
            result["report"] = {
                key: value
                for key, value in result["report"].items()
                if key in {"kind", "requested_checks", "refinement_origin", "work_seconds", "limit"}
            } | {"checks": [check_view(check) for check in result["report"].get("checks", [])]}
        summary["result"] = result
    if "verification" in record:
        summary["verification"] = history_view(record["verification"])
    if "sweep_progress" in record:
        summary["sweep_progress"] = dict(record["sweep_progress"])
        summary["sweep_progress"]["completed_points"] = [
            {key: value for key, value in point.items() if key in {"index", "values", "error"}}
            | {
                "metrics": metrics_view(point.get("metrics", {})),
            }
            for point in record["sweep_progress"]["completed_points"]
        ]
    return summary


def investigation_view(record, *, detail=False):
    summary = dict(record)
    summary["attempts"] = investigation_attempts(record["attempts"], detail=detail)
    if detail:
        return summary
    approved = {
        event["revision"]
        for event in record["events"]
        if event["kind"] == "decision" and event["action"] in {"start", "approve-revision"}
    }
    summary["revisions"] = [
        revision
        if revision["revision"] == record["active_revision"]
        else {
            "revision": revision["revision"],
            "status": "approved" if revision["revision"] in approved else "proposed",
        }
        for revision in record["revisions"]
    ]
    if record["active_revision"] is None:
        summary["revisions"][-1] = record["revisions"][-1]
    events, latest = [], None
    for event in record["events"]:
        if event["kind"] != "finding":
            events.append(event)
            continue
        finding = event["finding"]
        events.append(
            {key: event[key] for key in ("event_id", "recorded_at", "kind", "revision")}
            | {
                "finding": {
                    key: value
                    for key, value in finding.items()
                    if key in {"kind", "decision", "rationale", "next_experiment", "review_name"}
                }
                | {"run_count": len(finding["runs"])}
            }
        )
        latest = {key: value for key, value in event.items() if key != "evidence"}
    summary["events"] = events
    summary["latest_finding"] = latest
    return summary


def investigation_attempts(entries, *, detail=False):
    from entanglix.corrector.verification import verification_history

    attempts = []
    for attempt in entries:
        try:
            run = runner._read_attempt(Path(attempt["path"]))
            manifest = run["manifest"]
            item = attempt | {
                "protocol": manifest["protocol"],
                "metrics": run.get("result", {}).get("metrics", {})
                if detail
                else metrics_view(run.get("result", {}).get("metrics", {})),
            }
            if manifest["protocol"] == "verification":
                item["source_run_ids"] = manifest["parent_run_ids"]
                item["policy_version"] = manifest["launch"]["policy_version"]
                item["checks"] = [
                    check_view(check)
                    for check in run.get("result", {}).get("report", {}).get("checks", [])
                ]
            elif manifest["protocol"] in {"simulate", "sweep", "plan"}:
                item["verification"] = history_view(
                    verification_history(Path(attempt["path"])), compact=True
                )
            attempts.append(item)
        except (OSError, ValueError, KeyError) as exc:
            attempts.append(attempt | {"result_error": runner.failure_message(exc)})
    return attempts


def finding_ack(event, record):
    return {
        **{key: event[key] for key in ("event_id", "recorded_at", "revision")},
        "kind": event["finding"]["kind"],
        "decision": event["finding"]["decision"],
        **{key: record[key] for key in ("investigation_id", "checkpoint", "open_reviews", "usage")},
    }
