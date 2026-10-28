"""Summarize a retained Piccolo control candidate; no solve or check is run."""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from entanglix.runner import read_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = read_run(args.run)
    if record["status"] != "completed" or record["result"].get("engine") != "piccolo":
        parser.error("a completed Piccolo candidate is required")
    run = Path(record["path"])
    result, manifest = record["result"], record["manifest"]
    trace = tomllib.loads((run / "artifacts/solver-trace.toml").read_text())
    native = tomllib.loads((run / result["artifacts"]["native_result"]["path"]).read_text())
    spec = tomllib.loads((run / "spec/control.toml").read_text())
    with np.load(run / "artifacts/pulse.npz") as pulse:
        times, amplitudes = pulse["times_ns"], pulse["amplitudes_ghz"]
        dt = np.diff(times)[0]
        neutral = {
            "kind": pulse["kind"].item(),
            "duration_ns": float(times[-1]),
            "channel_amplitude_maxima_ghz": np.max(abs(amplitudes), axis=1).tolist(),
            "du_maximum_ghz_per_ns": float(np.max(abs(np.diff(amplitudes, axis=1) / dt))),
            "ddu_maximum_ghz_per_ns2": float(np.max(abs(np.diff(amplitudes, n=2, axis=1) / dt**2))),
        }
    baseline = json.loads(Path(__file__).with_name("m4_baseline_results.json").read_text())
    summary = {
        "run": str(run),
        "run_id": manifest["run_id"],
        "status": record["status"],
        "code": manifest["code"],
        "env_hash": manifest["env_hash"],
        "runtime": manifest["environment"]["solver_runtime"],
        "package_versions": trace["package_versions"],
        "problem_sha256": result["problem_sha256"],
        "energy_origin_ghz": manifest["launch"]["control_model"]["energy_origin_ghz"],
        "baseline_run_id": baseline["run_id"],
        "same_problem_as_baseline": result["problem_sha256"] == baseline["problem_sha256"],
        "objective": result["objective"],
        "solver_reported_fidelity": result["metrics"]["solver_reported_fidelity"],
        "requested_fidelity": spec["problem"].get("final_fidelity"),
        "termination": result["termination"],
        "declared_options": trace["declared_options"],
        "effective_options": trace["effective_options"],
        "initialization": trace["initialization"],
        "usage": record["finished"]["usage"],
        "native_solve_seconds": native["wall_seconds"],
        "neutral_pulse": neutral,
        "artifacts": result["artifacts"],
        "evidence_note": "Solver-reported candidate and pulse differences only; no independent control check. CPU workstation solve with OPENBLAS_NUM_THREADS=1.",
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
