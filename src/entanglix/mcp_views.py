"""Compact research views; backing records and CLI output remain complete."""

import json
from collections import Counter
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
    keys = {"name", "outcome", "solution", "scope", "coverage", "reason"}
    return {
        **{key: value for key, value in check.items() if key in keys},
        **({"detail_omitted": sorted(set(check) - keys)} if set(check) - keys else {}),
    }


def history_view(history):
    return [
        {key: value for key, value in item.items() if key != "checks"}
        | {"checks": [check_view(check) for check in item.get("checks", [])]}
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


PAGE_SIZE = 50


def table(columns, rows, after=0):
    if type(after) is not int or after < 0 or after > len(rows):
        raise ValueError("after must be an integer offset between 0 and the row count")
    end = min(after + PAGE_SIZE, len(rows))
    return {
        "columns": columns,
        "rows": rows[after:end],
        "total": len(rows),
        "next_after": end if end < len(rows) else None,
    }


def relative_path(path, root):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def investigation_status(record):
    latest = record["revisions"][-1]["revision"]
    root = Path(record["path"]).parent.parent
    result = {
        key: record[key]
        for key in (
            "path",
            "investigation_id",
            "state",
            "active_revision",
            "usage",
            "checkpoint",
            "open_reviews",
            "unresolved_launches",
        )
    }
    result["revisions"] = [
        r if r["revision"] in {record["active_revision"], latest} else {"revision": r["revision"]}
        for r in record["revisions"]
    ]
    findings = [e for e in record["events"] if e["kind"] == "finding"]
    result["attempt_counts"] = dict(Counter(a["status"] for a in record["attempts"]))
    result["finding_counts"] = dict(Counter(e["finding"]["kind"] for e in findings))
    result["latest_finding"] = (
        {k: v for k, v in findings[-1].items() if k != "evidence"} if findings else None
    )
    result["last_stop"] = next((e for e in reversed(record["events"]) if e["kind"] == "stop"), None)
    result["attention"] = [
        {"path": relative_path(a["path"], root), "status": a["status"]}
        for a in record["attempts"]
        if a["status"] in {"active", "unknown"}
        or (
            a["status"] == "interrupted"
            and not (Path(a["path"]) / "runner/reconciled.toml").exists()
        )
    ]
    return result


def control_fields(run, result):
    """The seed used, derived frame transition and solver termination of a control attempt."""
    frozen = runner._load_record(run / "spec/control.toml")
    trace = result.get("artifacts", {}).get("solver_trace")
    if trace:
        # GRAPE records the seed at top level, Piccolo under its initialization.
        recorded = runner._load_record(run / trace["path"])
        seed = recorded.get(
            "effective_seed", recorded.get("initialization", {}).get("effective_seed")
        )
    else:
        # Without a result, the declared seed is the one the attempt was launched with.
        seed = frozen["pulse"].get("seed")
    frame = frozen["frame"]
    transition = {
        name: {"levels": levels, "frequency_ghz": frame["frequencies_ghz"][name]}
        for name, levels in frame.get("transitions", {}).items()
    }
    return {
        "seed": seed,
        "transition": transition or None,
        "termination": result.get("termination"),
    }


def _pair(check):
    """A check's name and outcome, and the calibration solution it checked, if any."""
    return [
        check["name"],
        check.get("outcome"),
        *([check["solution"]] if "solution" in check else []),
    ]


def attempt_data(record):
    from entanglix.corrector.verification import verification_history

    root = Path(record["path"]).parent.parent
    fixed = ["path", "protocol", "base", "status", "parents", "checks", "points"]
    items = []
    for a in sorted(record["attempts"], key=lambda a: Path(a["path"]).name):
        item = {"path": relative_path(a["path"], root), "base": a["base"], "status": a["status"]}
        try:
            run = runner._read_attempt(Path(a["path"]))
            manifest, result = run["manifest"], run.get("result", {})
            protocol = manifest["protocol"]
            item.update(protocol=protocol, parents=manifest["parent_run_ids"], checks=[])
            if protocol == "verification":
                item["checks"] = [_pair(c) for c in result.get("report", {}).get("checks", [])]
            elif protocol in {"simulate", "sweep", "plan", "control", "calibrate"}:
                item["checks"] = [
                    [
                        h.get("run_id", h.get("path")),
                        h.get("source_matches"),
                        [_pair(c) for c in h.get("checks", [])],
                        h.get("diagnostic"),
                    ]
                    for h in verification_history(Path(a["path"]))
                ]
                if protocol == "control":
                    item.update(control_fields(Path(a["path"]), result))
                if protocol == "calibrate" and "outcome" in result:
                    item["outcome"] = result["outcome"]
            if protocol == "sweep":
                from entanglix.protocols.sweep import partial_results

                item["points"] = partial_results(Path(a["path"]))["completed_count"]
            else:
                values = a["varied_values"]
                values = values[0] if len(values) == 1 else {}
                item.update({f"values.{key}": value for key, value in values.items()})
            item.update(
                {
                    f"metrics.{key}": value
                    for key, value in metrics_view(result.get("metrics", {})).items()
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            item["result_error"] = runner.failure_message(exc)
        items.append(item)
    columns = fixed + sorted({key for item in items for key in item} - set(fixed))
    return columns, [[item.get(key) for key in columns] for item in items]


def event_rows(record):
    return [
        [
            e["event_id"],
            e["recorded_at"],
            e["kind"],
            e.get("revision"),
            e.get("action", e.get("reason", e.get("finding", {}).get("kind"))),
            {k: v for k, v in e.get("finding", {}).items() if k in {"decision", "review_name"}},
        ]
        for e in record["events"]
    ]


EVENT_COLUMNS = ["event_id", "recorded_at", "kind", "revision", "action", "finding"]


def investigation_view(record, *, view="status", after=0, event_id=None):
    if event_id is not None:
        if view != "status" or after != 0:
            raise ValueError("event_id cannot be combined with view or after")
        event = next((e for e in record["events"] if e["event_id"] == event_id), None)
        if event is None:
            raise ValueError(f"unknown investigation event: {event_id}")
        return {"investigation_id": record["investigation_id"], "event": event}
    if view == "attempts":
        return table(*attempt_data(record), after)
    if view == "events":
        return table(EVENT_COLUMNS, event_rows(record), after)
    if view != "status" or after != 0:
        raise ValueError("view must be status, attempts or events; after is only for tables")
    return investigation_status(record)


def runs_view(records, root, after=0):
    rows = [
        [
            relative_path(r["path"], root),
            r.get("run_id", r.get("launch_id")),
            r.get("system_id"),
            r["status"],
            r.get("reason", r.get("diagnostic")),
        ]
        for r in sorted(records, key=lambda r: Path(r["path"]).name)
    ]
    return table(["path", "id", "system_id", "status", "reason"], rows, after)


def finding_ack(event, record):
    return {
        **{key: event[key] for key in ("event_id", "recorded_at", "revision")},
        "kind": event["finding"]["kind"],
        "decision": event["finding"]["decision"],
        **{key: record[key] for key in ("investigation_id", "checkpoint", "open_reviews", "usage")},
    }
