"""Retain readout grids, requested checks, replay and refinement under research_runs/."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import tomli_w

from entanglix.corrector.verification import check_run
from entanglix.protocols.sweep import sweep
from entanglix.runner import _load_record, read_run, rerun

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/readout"


def launch(device, *, parent=None, narrower=False):
    with tempfile.TemporaryDirectory(prefix="entanglix-readout-grid-") as directory:
        directory = Path(directory)
        for name in ("system.toml", "readout.toml", "readout-state.npz", "dephasing.toml"):
            shutil.copyfile(EXAMPLE / name, directory / name)
        base = _load_record(directory / "readout.toml")
        base["needs"]["device"] = device
        (directory / "readout.toml").write_text(tomli_w.dumps(base))
        if narrower:
            grid = _load_record(directory / "dephasing.toml")
            axis = grid["axes"][0]
            axis["values"] = [13.5]
            axis["linked_values"] = {
                name: [values[1]] for name, values in axis["linked_values"].items()
            }
            grid["axes"][1]["values"] = [0.001, 0.0015, 0.002]
            (directory / "dephasing.toml").write_text(tomli_w.dumps(grid))
        result = sweep(
            directory / "dephasing.toml",
            runs_root=ROOT / "research_runs",
            parent_run_ids=(parent.name,) if parent else (),
        )
    if result["exit_code"]:
        raise RuntimeError(f"sweep failed: {read_run(Path(result['path']))}")
    return Path(result["path"])


def report_summary(report):
    checks = []
    for item in report["checks"]:
        summary = {
            key: item[key] for key in ("name", "scope", "outcome", "coverage") if key in item
        }
        residuals = {}

        def collect(value):
            if isinstance(value, dict):
                if "residual" in value:
                    key = (
                        value["name"],
                        value.get("observable"),
                        value.get("metric"),
                        value["tolerance"],
                    )
                    residuals.setdefault(key, []).append(value["residual"])
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(item)
        summary["residuals"] = [
            {
                "name": name,
                **({"observable": observable} if observable else {}),
                **({"metric": metric} if metric else {}),
                "tolerance": tolerance,
                "minimum_residual" if name == "positivity" else "maximum_residual": min(values)
                if name == "positivity"
                else max(values),
            }
            for (name, observable, metric, tolerance), values in residuals.items()
        ]
        if item["name"] == "segments":
            summary["local_coverage"] = [
                {
                    "point": point["index"],
                    **point["coverage"],
                    "windows": [
                        {key: segment[key] for key in ("start_ns", "end_ns")}
                        for segment in point["segments"]
                    ],
                }
                for point in item["points"]
            ]
        if item["name"] == "diagnostics":
            populations = {}
            for point in item["points"]:
                for value in point["values"]["top_level_population"]:
                    populations.setdefault(value["subsystem_index"], []).append(value)
            summary["top_level_population"] = [
                {
                    "subsystem_index": index,
                    "maximum_population": max(v["population"] for v in values),
                    "warning_points": sum(v["triggered"] for v in values),
                    "threshold": values[0]["threshold"],
                }
                for index, values in populations.items()
            ]
        checks.append(summary)
    return {
        "kind": report["kind"],
        "requested_checks": report["requested_checks"],
        "work_seconds": report["work_seconds"],
        "checks": checks,
    }


def requested(run, names):
    record = read_run(run)
    if record["status"] != "completed" or record["verification"]:
        raise RuntimeError("sweep must finish without automatic checks")
    checked = check_run(run, names)
    checking = read_run(Path(checked["path"]))
    if checked["exit_code"]:
        raise RuntimeError(f"check execution failed: {checking}")
    report = checking["result"]["report"]
    if any(item.get("outcome", "pass") != "pass" for item in report["checks"]):
        raise RuntimeError(f"requested check did not pass: {report}")
    return {
        "run_id": run.name,
        "path": str(run),
        "parent_run_ids": record["manifest"]["parent_run_ids"],
        "engine": record["result"]["engine"],
        "device": record["result"]["device"],
        "shape": record["result"]["shape"],
        "metrics": record["result"]["metrics"],
        "code": record["manifest"]["code"],
        "env_hash": record["manifest"]["env_hash"],
        "plan_sha256": record["manifest"]["launch"]["plan_sha256"],
        "usage": record["finished"]["usage"],
        "automatic_checks": False,
        "verification_run_id": checking["manifest"]["run_id"],
        "policy_version": checking["result"]["policy_version"],
        "report": report_summary(report),
    }


def study(device, output):
    names = ("integrity", "segments", "diagnostics", "declared_expectation")
    source = launch(device)
    first = requested(source, names)
    print(f"{device} source and requested checks passed: {source}", flush=True)
    replay = rerun(source)
    if replay["exit_code"] or not all(replay["replay"].values()):
        raise RuntimeError(f"sweep replay failed: {replay}")
    repeated = requested(Path(replay["path"]), names)
    residual = 0.0
    for index in range(9):
        name = f"artifacts/points/{index:06d}/output.npz"
        with np.load(source / name) as old, np.load(Path(replay["path"]) / name) as new:
            residual = max(
                residual, *(float(np.max(abs(old[key] - new[key]))) for key in old.files)
            )
    if residual > 1e-7:
        raise RuntimeError(f"sweep replay residual exceeds 1e-7: {residual}")
    summary = {
        "complete": True,
        "pass": True,
        "device": device,
        "records": [first, repeated],
        "replay_provenance": replay["replay"],
        "replay_maximum_residual": residual,
        "replay_atol": 1e-7,
    }
    if device == "cpu":
        spectral = sweep(EXAMPLE / "dispersive.toml", runs_root=ROOT / "research_runs")
        if spectral["exit_code"]:
            raise RuntimeError(f"static companion failed: {spectral}")
        summary["static_companion"] = requested(Path(spectral["path"]), ("spectral_refinement",))
        narrower = launch("cpu", parent=source, narrower=True)
        summary["refined_grid"] = requested(narrower, names)
    output.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(
        f"{device} replay and checks passed; maximum numerical difference {residual:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.device, args.output)
