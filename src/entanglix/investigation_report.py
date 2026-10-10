"""Render retained investigation records as a deterministic, on-request report."""

import json
from pathlib import Path

from entanglix.investigations import read_investigation
from entanglix.mcp_views import attempt_data, investigation_status


def text(value, depth=0):
    indent = "  " * depth
    if isinstance(value, dict):
        return (
            "\n".join(
                f"{indent}- **{key}:**"
                + ("\n" + text(item, depth + 1) if isinstance(item, (dict, list)) else f" {item}")
                for key, item in value.items()
            )
            or f"{indent}None."
        )
    if isinstance(value, list):
        return (
            "\n".join(
                f"{indent}- {text(item, depth + 1).lstrip().removeprefix('- ')}" for item in value
            )
            or f"{indent}None."
        )
    return str(value)


def finding_text(finding):
    parts = [
        f"This is the agent's interpretation; {len(finding.get('runs', []))} referenced attempts."
    ]
    parts.append(finding.get("learned", ""))
    parts.extend(
        f"**{key.replace('_', ' ').capitalize()}:**\n\n{text(value)}"
        for key, value in finding.items()
        if key not in {"kind", "runs", "learned"}
    )
    return "\n\n".join(parts)


def markdown_table(columns, rows):
    def cell(value):
        if value is None:
            return ""
        rendered = (
            value
            if isinstance(value, str)
            else json.dumps(value, default=str, separators=(",", ":"))
        )
        return rendered.replace("|", "&#124;").replace("\n", "<br>")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def report_investigation(path: Path) -> str:
    record = read_investigation(path)
    status, events = investigation_status(record), record["events"]
    revisions = {r["revision"]: r["terms"] for r in record["revisions"]}
    terms = revisions.get(record["active_revision"], record["revisions"][-1]["terms"])
    findings = [e for e in events if e["kind"] == "finding"]
    latest = findings[-1] if findings else None
    latest_kind = latest["finding"]["kind"] if latest else None
    counts = status["checkpoint"]
    parts = [
        f"# Investigation {record['investigation_id']}",
        f"State: {record['state']}. Effective revision: {record['active_revision']}.",
        f"Covers through event {events[-1]['event_id'] if events else 'none'} ({len(events)} events).",
        "## Question",
        terms["question"],
        f"## Latest finding ({latest_kind})" if latest else "## Latest finding",
        finding_text(latest["finding"]) if latest else "No finding recorded.",
        "## Open items",
        text({key: status[key] for key in ("open_reviews", "attention", "unresolved_launches")}),
        "## Usage",
        text(status["usage"]),
        f"Checkpoint counts: {counts.get('finished_attempts', 'unknown')} finished attempts; {counts.get('since_assessment', 'unknown')} since assessment.",
        "## Attempts and checks",
        "Paths are relative to the runs root. Source checks show the check run ID, source identity match, outcomes and any diagnostic. Null outcomes are diagnostics, not pass/fail evidence. Omitted metrics and sweep point values remain in the run files.",
        markdown_table(*attempt_data(record)),
        "## History",
    ]
    for event in events:
        kind = event.get("finding", {}).get("kind", event["kind"])
        parts.append(f"### Event {event['event_id']} — {kind} ({event['recorded_at']})")
        if event["kind"] == "finding":
            parts.append(
                f"See [Latest finding ({latest_kind})](#latest-finding-{latest_kind})."
                if event is latest
                else finding_text(event["finding"])
            )
        else:
            fields = {
                key: value
                for key, value in event.items()
                if key not in {"event_id", "record_version", "recorded_at", "kind", "runs"}
            }
            parts.append(text(fields))
            if event.get("runs"):
                parts.append(f"{len(event['runs'])} affected attempts.")
    parts.append("## Terms")
    for revision in record["revisions"]:
        parts.extend([f"### Revision {revision['revision']}", text(revision["terms"])])
    return "\n\n".join(parts) + "\n"
