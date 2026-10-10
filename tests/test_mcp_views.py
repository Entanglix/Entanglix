"""Status stays focused; table cursors and reports preserve retained history."""

import copy
import json
from pathlib import Path

import pytest

from entanglix import runner
from entanglix.cli import main
from entanglix.mcp_views import PAGE_SIZE, investigation_view, runs_view, table


def records(tmp_path, count):
    root = tmp_path / "runs"
    return {
        "path": str(root / "_investigations/i1"),
        "investigation_id": "i1",
        "state": "active",
        "active_revision": 1,
        "revisions": [
            {
                "revision": 1,
                "terms": {"question": "Measure a crossing.", "limits": {"job_count": 1000}},
            }
        ],
        "usage": {"jobs": count, "job_seconds": count / 2},
        "checkpoint": {"due": False, "elapsed_seconds": 1},
        "open_reviews": [],
        "unresolved_launches": [],
        "events": [
            {
                "event_id": "00000001",
                "kind": "decision",
                "action": "start",
                "revision": 1,
                "recorded_at": "2026-10-10T00:00:00Z",
            }
        ],
        "attempts": [
            {
                "path": str(root / f"system/r{i:06d}"),
                "run_id": f"r{i:06d}",
                "base": "base",
                "status": "completed",
                "varied_values": {"amplitude": i},
            }
            for i in range(count)
        ],
    }


def fake_attempt(path):
    index = int(path.name[1:])
    return {
        "manifest": {
            "protocol": "verification" if index == 1 else "script",
            "parent_run_ids": ["r000000"] if index == 1 else [],
        },
        "result": {
            "metrics": {"late_metric": 1} if index == 499 else {"loss": index / 100},
            "report": {"checks": [{"name": "integrity", "outcome": "pass"}]},
        },
    }


def test_status_size_does_not_follow_finished_history(tmp_path):
    small = investigation_view(records(tmp_path, 5))
    large = investigation_view(records(tmp_path, 500))
    # Only digit widths of the usage and counts change.
    for result in (small, large):
        result.pop("usage")
        result.pop("attempt_counts")
    assert json.dumps(small, sort_keys=True) == json.dumps(large, sort_keys=True)
    assert "attempts" not in large and "events" not in large


def test_oldest_interruption_is_visible_and_reconciliation_clears_attention(tmp_path):
    record = records(tmp_path, 500)
    oldest = record["attempts"][0]
    oldest["status"] = "interrupted"
    assert investigation_view(record)["attention"] == [
        {"path": "system/r000000", "status": "interrupted"}
    ]
    marker = Path(oldest["path"]) / "runner/reconciled.toml"
    marker.parent.mkdir(parents=True)
    marker.write_text("record_version = 1\n")
    assert investigation_view(record)["attention"] == []
    oldest["status"] = "unknown"
    assert investigation_view(record)["attention"][0]["status"] == "unknown"


def test_table_pages_cover_every_attempt_once_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_read_attempt", fake_attempt)
    record = records(tmp_path, 500)
    record["attempts"].reverse()
    after, rows = 0, []
    while True:
        page = investigation_view(record, view="attempts", after=after)
        assert len(page["rows"]) <= PAGE_SIZE
        rows.extend(page["rows"])
        if page["next_after"] is None:
            break
        after = page["next_after"]
    assert [row[0] for row in rows] == [f"system/r{i:06d}" for i in range(500)]
    check = dict(zip(page["columns"], rows[1]))
    assert check["metrics.loss"] == 0.01
    assert check["parents"] == ["r000000"]
    assert check["checks"] == [["integrity", "pass"]]
    assert dict(zip(page["columns"], rows[0]))["metrics.late_metric"] is None
    assert dict(zip(page["columns"], rows[-1]))["metrics.late_metric"] == 1
    assert all(len(row) == len(page["columns"]) for row in rows)
    listings = [
        {"path": a["path"], "run_id": a["run_id"], "system_id": "system", "status": a["status"]}
        for a in record["attempts"]
    ]
    page = runs_view(listings, tmp_path / "runs", PAGE_SIZE)
    assert page["rows"][0][0] == f"system/r{PAGE_SIZE:06d}"
    assert page["next_after"] == PAGE_SIZE * 2


@pytest.mark.parametrize("after", [-1, 3, 1.5, True])
def test_invalid_table_cursor_is_rejected(after):
    with pytest.raises(ValueError, match="after"):
        table(["value"], [[1], [2]], after)


def test_sweep_is_one_row_with_point_count_and_metric_omission(tmp_path, monkeypatch):
    from entanglix.corrector import verification
    from entanglix.protocols import sweep

    record = records(tmp_path, 1)
    record["attempts"][0]["varied_values"] = [{"amplitude": 0.1}, {"amplitude": 0.2}]
    run = fake_attempt(Path("r000000"))
    run["manifest"]["protocol"] = "sweep"
    run["result"]["metrics"] = {"small": [0.1, 0.2], "large": list(range(500))}
    monkeypatch.setattr(runner, "_read_attempt", lambda path: run)
    monkeypatch.setattr(verification, "verification_history", lambda path: [])
    count = 2
    monkeypatch.setattr(sweep, "partial_results", lambda path: {"completed_count": count})
    page = investigation_view(record, view="attempts")
    assert page["total"] == 1
    row = dict(zip(page["columns"], page["rows"][0]))
    assert row["points"] == 2
    assert not any(key.startswith("values.") for key in page["columns"])
    assert row["metrics.small"] == [0.1, 0.2]
    assert row["metrics.large"]["omitted"] is True
    record["attempts"][0]["varied_values"] *= 250
    count = 500
    large = investigation_view(record, view="attempts")
    row["points"] = 500
    assert dict(zip(large["columns"], large["rows"][0])) == row


def test_event_index_and_lookup_preserve_order_and_full_text(tmp_path):
    record = records(tmp_path, 0)
    record["events"] += [
        {
            "event_id": f"{i + 2:08d}",
            "kind": "finding",
            "revision": 1,
            "recorded_at": "now",
            "finding": {"kind": "assessment", "decision": "continue", "learned": f"Learning {i}"},
        }
        for i in range(120)
    ]
    page = investigation_view(record, view="events")
    assert len(page["rows"]) == PAGE_SIZE
    next_page = investigation_view(record, view="events", after=page["next_after"])
    assert next_page["rows"][0][0] == f"{PAGE_SIZE + 1:08d}"
    assert investigation_view(record, event_id="00000002")["event"] == record["events"][1]
    with pytest.raises(ValueError, match="combined"):
        investigation_view(record, view="events", event_id="00000002")


def test_report_command_covers_all_attempts_and_findings_deterministically(
    tmp_path, monkeypatch, capsys
):
    from entanglix import investigation_report

    record = records(tmp_path, 65)
    record["attempts"][2]["status"] = "interrupted"
    record["events"].append(
        {
            "event_id": "00000002",
            "kind": "finding",
            "revision": 1,
            "recorded_at": "now",
            "finding": {"kind": "interpretation", "learned": "Earlier interpretation."},
        }
    )
    record["events"].append(
        {
            "event_id": "00000003",
            "kind": "finding",
            "revision": 1,
            "recorded_at": "now",
            "finding": {
                "kind": "report",
                "decision": "stop",
                "learned": "Measured the crossing.\n\nA second paragraph of interpretation.",
                "runs": [f"r{i:06d}" for i in range(65)],
                "unresolved": ["Remaining uncertainty."],
            },
        }
    )
    monkeypatch.setattr(runner, "_read_attempt", fake_attempt)
    monkeypatch.setattr(
        investigation_report, "read_investigation", lambda path: copy.deepcopy(record)
    )
    assert main(["report-investigation", record["path"]]) == 0
    first = capsys.readouterr().out
    record["checkpoint"]["elapsed_seconds"] = 99
    output = tmp_path / "report.md"
    assert main(["report-investigation", record["path"], "--output", str(output)]) == 0
    assert capsys.readouterr().out == ""
    assert output.read_text() == first
    assert all(a["run_id"] in first for a in record["attempts"])
    assert "Measured the crossing." in first and "Remaining uncertainty." in first
    assert "Earlier interpretation." in first
    assert "Covers through event 00000003" in first
    assert "agent's interpretation" in first
    assert "65 referenced attempts" in first
    assert "```json" not in first and "\\n" not in first
    assert first.index("Measured the crossing.") < first.index("## Open items")
    assert first.index("## Open items") < first.index("## Attempts and checks")
    assert (
        first.index("## Attempts and checks") < first.index("## History") < first.index("## Terms")
    )
    assert first.count("r000064") == 1
