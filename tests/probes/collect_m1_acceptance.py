"""Append JUnit results to the retained acceptance report without relabeling earlier evidence.

Run from the repository root, after tests finish and before changing their sources:
    python tests/probes/collect_m1_acceptance.py --junit LABEL REPORT.xml 'TEST COMMAND'
Repeat --junit to collect several runs. The existing JSON supplies earlier measurements,
observed environment, and pinned limits; XML files may live outside the repository.
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def source_snapshot(root):
    paths = list((root / "src/entanglix").rglob("*.py"))
    paths += list((root / "src/entanglix").rglob("*.toml"))
    paths += list((root / "tests").glob("test_engine*.py"))
    paths += [root / "tests/engine_cases.py", root / "tests/conftest.py", Path(__file__).resolve()]
    paths += list((root / "examples").rglob("*.toml"))
    paths += list((root / "examples").rglob("*.npz"))
    paths += list((root / "examples").glob("*.py"))
    paths += [root / ".gitignore", root / ".github/workflows/ci.yml", root / "uv.lock"]
    return {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        ),
        "sources_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))
        },
    }


def snapshot_id(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def collect(payload, runs, root):
    snapshots = payload["source_snapshots"]
    current = source_snapshot(root)
    current_id = snapshot_id(current)
    snapshots[current_id] = current
    existing_labels = {run["label"] for run in payload["test_runs"]}
    for label, path, command in runs:
        if label in existing_labels:
            raise ValueError(f"run label {label!r} already exists; use a new label")
        xml = ET.parse(path).getroot()
        suites = list(xml.iter("testsuite"))
        counts = {
            name: sum(int(suite.get(name, 0)) for suite in suites)
            for name in ("tests", "failures", "errors", "skipped")
        }
        if not suites or counts["failures"] or counts["errors"]:
            raise ValueError(f"{path}: expected completed passing test suites")
        payload["test_runs"].append(
            {
                "label": label,
                "command": command,
                "source_snapshot": current_id,
                "source_snapshot_scope": "sources collected after this run",
                "timestamps": [suite.get("timestamp") for suite in suites],
                "seconds": sum(float(suite.get("time", 0)) for suite in suites),
                **counts,
            }
        )
        existing_labels.add(label)
        for case in xml.iter("testcase"):
            if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
                continue
            properties = {}
            for prop in case.iter("property"):
                value = prop.get("value", "")
                try:
                    value = float(value)
                except ValueError:
                    pass
                properties[prop.get("name")] = value
            if properties:
                key = case.get("classname") + "::" + case.get("name")
                payload["measurements"][key] = {"source_run": label, **properties}
    payload["code"] = current
    payload["generated_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    payload["interpretation"] = (
        "Numerical acceptance for the included models and regimes. Cross-engine agreement "
        "is bounded via independent exponential references. Each run identifies its retained "
        "source snapshot; earlier measurements are not relabeled as new runs. Non-gradient "
        "selection defaults to CPU."
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("tests/probes/m1_acceptance_results.json")
    )
    parser.add_argument(
        "--junit", nargs=3, action="append", required=True, metavar=("LABEL", "PATH", "COMMAND")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    payload = collect(json.loads(args.results.read_text()), args.junit, root)
    args.results.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Collected {len(args.junit)} runs in {args.results}")


if __name__ == "__main__":
    main()
