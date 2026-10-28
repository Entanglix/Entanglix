"""Run the scripted investigation baseline in operator-controlled acceptance.

This driver creates the explicit start decision for the declared fixture; the
research client itself cannot start. Full records remain under research_runs/.
An existing workspace continues its investigation without another start.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

from entanglix.investigations import read_investigation, record_user_decision

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "examples/readout/investigate.py"


def accept(workspace, runs_root, case, output):
    module = runpy.run_path(str(CLIENT))
    if not workspace.exists():
        record = module["draft"](
            workspace,
            runs_root=runs_root,
            case=case,
            checkpoint=10,
            job_count=60 if case == "normal" else 100,
        )
        record_user_decision(
            Path(record["path"]),
            "start",
            revision=1,
            note="Operator-controlled scripted baseline acceptance.",
        )
    client = module["Client"](workspace)
    if client.case != case or client.runs_root != runs_root.resolve():
        raise ValueError("existing workspace belongs to another case or runs root")
    result = client.run()
    investigation = read_investigation(client.path)
    checks, computations = [], []
    provenance = None
    for run_id, (path, record) in client.records.items():
        manifest = record["manifest"]
        run_provenance = {key: manifest[key] for key in ("env_hash", "code")}
        if provenance is None:
            provenance = run_provenance
        else:
            assert run_provenance == provenance, "baseline code or environment changed between jobs"
        common = {
            "run_id": run_id,
            "status": record["status"],
            "job_seconds": record.get("finished", {}).get("usage", {}).get("wall_seconds", 0),
            "base": manifest["investigation_base"],
            "revision": manifest["investigation_revision"],
        }
        if manifest["protocol"] == "verification":
            report = record.get("result", {}).get("report", {})
            checks.append(
                {
                    **common,
                    "sources": manifest["parent_run_ids"],
                    "policy_version": manifest["launch"]["policy_version"],
                    "checks": [
                        {key: entry[key] for key in ("name", "outcome", "scope") if key in entry}
                        for entry in report.get("checks", [])
                    ],
                }
            )
        else:
            computations.append(
                {
                    **common,
                    "protocol": manifest["protocol"],
                    "values": manifest["launch"].get("varied_values", {}),
                    "metrics": record.get("result", {}).get("metrics", {}),
                }
            )
    assert investigation["state"] == "stopped"
    assert all(item["status"] == "completed" for item in checks + computations)
    assert all(
        entry.get("outcome") not in {"fail", "inapplicable"}
        for item in checks
        for entry in item["checks"]
    )
    normal = [item for item in result["brackets"] if item["stage"] == "normal"]
    assert len(normal) == 2 and all(item["confirmed"] for item in normal)
    assert result["no_crossing"][0]["ej_ghz"] == 12
    assert result["user_decisions_after_start"] == 0
    if case == "early-stop":
        assert result["outcome"] == "review_handoff"
        assert investigation["open_reviews"] == ["precision"]
        assert not result["brackets"][-1]["confirmed"]
        failed_ej = result["brackets"][-1]["ej_ghz"]
        assert result["unattempted"] == [
            {"ej_ghz": item["ej_ghz"], "stage": "tighter"}
            for item in normal
            if item["ej_ghz"] > failed_ej
        ]
    else:
        assert result["outcome"] == "evidence_completion"
        assert not investigation["open_reviews"]
    computation_seconds = sum(item["job_seconds"] for item in computations)
    check_seconds = sum(item["job_seconds"] for item in checks)
    summary = {
        **result,
        "question": investigation["revisions"][0]["terms"]["question"],
        "evidence_requirements": investigation["revisions"][0]["terms"]["evidence_requirements"],
        "provenance": provenance,
        "computation_seconds": computation_seconds,
        "check_seconds": check_seconds,
        "time_outside_jobs_seconds": result["elapsed_seconds"]
        - computation_seconds
        - check_seconds,
        "assessments": sum(
            event["kind"] == "finding" and event["finding"]["kind"] == "assessment"
            for event in investigation["events"]
        ),
        "checks": checks,
        "computations": computations,
        "client_sha256": hashlib.sha256(CLIENT.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src/entanglix").rglob("*.py"))
        },
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "case",
                    "investigation_id",
                    "outcome",
                    "usage",
                    "elapsed_seconds",
                    "computation_seconds",
                    "check_seconds",
                    "assessments",
                    "user_decisions_after_start",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("normal", "early-stop"), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accept(args.workspace, args.runs_root, args.case, args.output)
