"""Summarize a retained CPU control candidate; no solve or built-in check is run."""

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
    if record["status"] != "completed" or record["manifest"]["protocol"] != "control":
        parser.error("a completed control candidate is required")
    run = Path(record["path"])
    manifest, result = record["manifest"], record["result"]
    spec = tomllib.loads((run / "spec/control.toml").read_text())
    trace = tomllib.loads((run / "artifacts/solver-trace.toml").read_text())
    with np.load(run / "artifacts/pulse.npz", allow_pickle=False) as pulse:
        amplitudes = pulse["amplitudes_ghz"]
        dt = np.diff(pulse["times_ns"])[0]
        pulse_summary = {
            "kind": pulse["kind"].item(),
            "duration_ns": float(pulse["times_ns"][-1]),
            "channel_amplitude_maxima_ghz": np.max(abs(amplitudes), axis=1).tolist(),
            "du_maximum_ghz_per_ns": float(np.max(abs(np.diff(amplitudes, axis=1) / dt))),
            "ddu_maximum_ghz_per_ns2": float(np.max(abs(np.diff(amplitudes, n=2, axis=1) / dt**2))),
        }
    calls = [item["seconds"] for item in trace["objective_calls"]]
    iterations = [item["seconds"] for item in trace["iterations"]]
    summary = {
        "run": str(run),
        "run_id": manifest["run_id"],
        "status": record["status"],
        "code": manifest["code"],
        "env_hash": manifest["env_hash"],
        "problem_sha256": result["problem_sha256"],
        "energy_origin_ghz": manifest["launch"]["control_model"]["energy_origin_ghz"],
        "solver": result["solver"],
        "solver_reported_fidelity": result["metrics"]["solver_reported_fidelity"],
        "requested_fidelity": spec["problem"].get("final_fidelity"),
        "termination": result["termination"],
        "usage": record["finished"]["usage"],
        "ignored_guidance_fields": result["ignored_guidance_fields"],
        "effective_seed": trace.get("effective_seed"),
        "neutral_pulse": pulse_summary,
        "declared_bounds": {
            "channel_amplitude_ghz": [
                channel["amplitude_bound_ghz"] for channel in spec["channels"]
            ],
            "du_ghz_per_ns": spec["problem"].get("du_bound"),
            "ddu_ghz_per_ns2": spec["problem"].get("ddu_bound"),
        },
        "timings_seconds": {
            "first_two_objective_gradient_calls": calls[:2],
            "subsequent_objective_gradient_median": float(np.median(calls[2:])),
            "subsequent_iteration_median": float(np.median(iterations[2:])),
        },
        "timing_note": "The first call includes tracing and JAX compilation; the second call is reported separately. Subsequent costs are medians after the first two calls/iterations. CPU workstation measurements.",
        "evidence_note": "Solver-reported fidelity and pulse differences only; independent control checks are not yet available.",
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
