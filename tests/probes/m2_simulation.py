"""Retain full-window Purcell simulation, verification, and replay acceptance."""

import argparse
import json
import shutil
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import tomli_w

from entanglix.protocols.simulate import rerun_simulation, simulate
from entanglix.runner import read_run

ROOT = Path(__file__).resolve().parents[2]


def study(device, output):
    with tempfile.TemporaryDirectory(prefix="entanglix-purcell-") as directory:
        directory = Path(directory)
        shutil.copyfile(ROOT / "examples/readout/system.toml", directory / "system.toml")
        name = "purcell-cuda.toml" if device == "cuda" else "purcell.toml"
        spec = tomllib.loads((ROOT / "examples/readout" / name).read_text())
        protocol = directory / "purcell.toml"
        protocol.write_text(tomli_w.dumps(spec))
        first = simulate(protocol, runs_root=ROOT / "research_runs")
    run = Path(first["path"])
    record = read_run(run)
    if first["exit_code"] or record["verification"]["current"].get("verdict") != "VERIFIED":
        raise RuntimeError(f"simulation failed acceptance: {first}; {record['log_tail']}")
    print(f"{device} completed and VERIFIED: {run}", flush=True)
    replay = rerun_simulation(run)
    repeated = read_run(Path(replay["path"]))
    if not all(replay["replay"].values()):
        raise RuntimeError(f"acceptance requires matching code/environment: {replay['replay']}")
    if replay["exit_code"] or repeated["verification"]["current"].get("verdict") != "VERIFIED":
        raise RuntimeError(f"replay failed acceptance: {replay}; {repeated['log_tail']}")
    with (
        np.load(run / "artifacts/output.npz") as old,
        np.load(Path(replay["path"]) / "artifacts/output.npz") as new,
    ):
        residual = max(float(np.max(abs(old[name] - new[name]))) for name in old.files)
    if residual > 1e-7:
        raise RuntimeError(f"replay output residual exceeds 1e-7: {residual}")
    records = []
    for item in (record, repeated):
        verification = read_run(Path(item["verification"]["current"]["path"]))
        records.append(
            {
                "run_id": item["manifest"]["run_id"],
                "path": item["path"],
                "parent_run_ids": item["manifest"]["parent_run_ids"],
                "engine": item["result"]["engine"],
                "device": item["result"]["device"],
                "env_hash": item["manifest"]["env_hash"],
                "environment": item["manifest"]["environment"],
                "code": item["manifest"]["code"],
                "plan_sha256": item["manifest"]["launch"]["plan_sha256"],
                "metrics": item["result"]["metrics"],
                "usage": item["finished"]["usage"],
                "verdict": verification["result"]["verdict"],
                "verification_run_id": verification["manifest"]["run_id"],
                "report": verification["result"]["report"],
            }
        )
    output.write_text(
        json.dumps(
            {
                "complete": True,
                "pass": True,
                "device": device,
                "records": records,
                "replay_maximum_residual": residual,
                "replay_atol": 1e-7,
                "replay_provenance": replay["replay"],
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"{device} replay VERIFIED; maximum output difference {residual:.3g}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.device, args.output)
