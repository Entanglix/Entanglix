"""Retain Purcell computations, requested checks and replay; runs stay in research_runs/."""

import argparse
import json
import shutil
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import tomli_w

from entanglix.corrector.verification import check_run
from entanglix.protocols.simulate import rerun_simulation, simulate
from entanglix.runner import read_run

ROOT = Path(__file__).resolve().parents[2]


def requested(run, names):
    before = read_run(run)
    if before["status"] != "completed" or before["verification"]:
        raise RuntimeError(f"computation acceptance failed: {before}")
    job = check_run(run, names)
    record = read_run(Path(job["path"]))
    if job["exit_code"]:
        raise RuntimeError(f"requested checks failed execution: {record}")
    report = record["result"]["report"]
    for item in report["checks"]:
        if item["name"] == "diagnostics":
            if "outcome" in item or "values" not in item:
                raise RuntimeError(f"invalid diagnostic report: {item}")
        elif item["outcome"] != "pass":
            raise RuntimeError(f"requested check failed: {item}")
    return {
        "run_id": before["manifest"]["run_id"],
        "path": before["path"],
        "parent_run_ids": before["manifest"]["parent_run_ids"],
        "engine": before["result"]["engine"],
        "device": before["result"]["device"],
        "dimensions": before["result"]["dimensions"],
        "env_hash": before["manifest"]["env_hash"],
        "environment": before["manifest"]["environment"],
        "code": before["manifest"]["code"],
        "plan_sha256": before["manifest"]["launch"]["plan_sha256"],
        "metrics": before["result"]["metrics"],
        "usage": before["finished"]["usage"],
        "automatic_checks": False,
        "verification_run_id": record["manifest"]["run_id"],
        "policy_version": record["result"]["policy_version"],
        "report": report,
    }


def launch(device, *, larger=False):
    with tempfile.TemporaryDirectory(prefix="entanglix-purcell-") as directory:
        directory = Path(directory)
        document = tomllib.loads((ROOT / "examples/readout/system.toml").read_text())
        if larger:
            document["model"]["subsystems"][0].update(retained_levels=6, cutoff=32)
            document["model"]["subsystems"][1]["retained_levels"] = 8
        (directory / "system.toml").write_text(tomli_w.dumps(document))
        name = "purcell-cuda.toml" if device == "cuda" else "purcell.toml"
        shutil.copyfile(ROOT / "examples/readout" / name, directory / "purcell.toml")
        job = simulate(directory / "purcell.toml", runs_root=ROOT / "research_runs")
    if job["exit_code"]:
        raise RuntimeError(f"simulation failed acceptance: {job}")
    return Path(job["path"])


def study(device, output):
    names = ("integrity", "segments", "diagnostics")
    source = launch(device)
    first = requested(source, names)
    print(f"{device} source completed with requested checks: {source}", flush=True)
    replay = rerun_simulation(source)
    if replay["exit_code"] or not all(replay["replay"].values()):
        raise RuntimeError(f"replay failed acceptance: {replay}")
    repeated = requested(Path(replay["path"]), names)
    with (
        np.load(source / "artifacts/output.npz") as old,
        np.load(Path(replay["path"]) / "artifacts/output.npz") as new,
    ):
        residual = max(float(np.max(abs(old[name] - new[name]))) for name in old.files)
    if residual > 1e-7:
        raise RuntimeError(f"replay output residual exceeds 1e-7: {residual}")
    summary = {
        "complete": True,
        "pass": True,
        "device": device,
        "records": [first, repeated],
        "replay_maximum_residual": residual,
        "replay_atol": 1e-7,
        "replay_provenance": replay["replay"],
    }
    if device == "cpu":
        larger = launch(device, larger=True)
        summary["more_levels"] = requested(larger, ("integrity",))
        print(f"dimension-48 computation and integrity passed: {larger}", flush=True)
    output.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"{device} replay/checks passed; maximum output difference {residual:.3g}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.device, args.output)
