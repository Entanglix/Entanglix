"""Measure adaptive readout brackets before deciding the investigation lifecycle.

Uses existing CPU sweep/simulate/check operations. Full inputs, arrays and check
reports stay in git-ignored research_runs/m3-readout/. This probe has no agent
runtime or investigation API and measures no agent acceleration.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np
import tomli_w

from entanglix.corrector.reference import embed_state
from entanglix.corrector.verification import check_run
from entanglix.models import load_system
from entanglix.protocols.simulate import simulate
from entanglix.protocols.sweep import sweep
from entanglix.runner import read_run

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/readout"
TARGET = 0.01
FREQUENCIES = {12.0: 7.0017536727417085, 13.5: 7.002206406997239, 15.0: 7.002760675881172}


def retain(job, elapsed):
    path = Path(job["path"])
    record = read_run(path)
    if job["exit_code"] or record["status"] != "completed":
        raise RuntimeError(f"probe job failed: {path}; {record['status']}")
    return {
        "run_id": path.name,
        "path": str(path.relative_to(ROOT)),
        "parent_run_ids": record["manifest"]["parent_run_ids"],
        "plan_sha256": record["manifest"]["launch"].get("plan_sha256"),
        "env_hash": record["manifest"]["env_hash"],
        "code": record["manifest"]["code"],
        "job_wall_seconds": record["finished"]["usage"]["wall_seconds"],
        "launch_elapsed_seconds": elapsed,
        "metrics": record["result"].get("metrics", {}),
    }


class Probe:
    def __init__(self, output):
        self.output = output
        self.runs_root = ROOT / "research_runs/m3-readout"
        self.samples = {}
        self.result = {
            "complete": False,
            "target_additional_coherence_loss": TARGET,
            "duration_ns": 100,
            "frequencies_ghz": FREQUENCIES,
            "normal_width_ghz": 0.000125,
            "tighter_width_ghz": 0.000015625,
            "comparison": "QuTiP vern9 lab 5x6 versus lab 6x8, cutoff 30 versus 32",
            "limits": [
                "Single crossing assumes decreasing coherence; checks cover sampled amplitudes only.",
                "Refinement differences are observed convergence evidence, not rigorous error bounds.",
                "Bisection width describes the RWA crossing; lab confirmation is reported separately.",
                "Costs include fresh runner processes; this is not an agent-versus-script benchmark.",
            ],
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((ROOT / "src/entanglix").rglob("*.py"))
            },
            "fixture_sha256": {
                name: hashlib.sha256((EXAMPLE / name).read_bytes()).hexdigest()
                for name in ("system.toml", "readout.toml", "readout-state.npz", "dephasing.toml")
            },
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runs": [],
            "checks": [],
            "brackets": [],
        }

    def save(self):
        self.output.write_text(json.dumps(self.result, indent=2) + "\n")

    def check(self, source, names):
        started = time.perf_counter()
        job = check_run(ROOT / source["path"], names, timeout_seconds=180)
        item = retain(job, time.perf_counter() - started)
        result = read_run(ROOT / item["path"])["result"]
        report = result["report"]
        item.update(
            policy_version=result["policy_version"],
            outcomes=[
                {
                    key: entry[key]
                    for key in ("name", "outcome", "scope", "coverage")
                    if key in entry
                }
                for entry in report["checks"]
            ],
        )
        self.result["checks"].append(item)
        self.save()
        if any(entry.get("outcome") in {"fail", "inapplicable"} for entry in report["checks"]):
            raise RuntimeError(f"requested check did not pass: {item['path']}")

    def seed(self):
        with tempfile.TemporaryDirectory(prefix="entanglix-m3-seed-") as directory:
            directory = Path(directory)
            for name in ("system.toml", "readout.toml", "readout-state.npz", "dephasing.toml"):
                shutil.copyfile(EXAMPLE / name, directory / name)
            spec = tomllib.loads((directory / "readout.toml").read_text())
            spec["needs"]["device"] = "cpu"
            (directory / "readout.toml").write_text(tomli_w.dumps(spec))
            started = time.perf_counter()
            job = sweep(directory / "dephasing.toml", runs_root=self.runs_root)
        item = retain(job, time.perf_counter() - started)
        item["profile"] = "seed_rwa_sweep"
        self.result["runs"].append(item)
        for row, ej in enumerate(FREQUENCIES):
            for column, amplitude in enumerate((0.0, 0.001, 0.002)):
                self.samples[ej, amplitude, "rwa"] = {
                    "amplitude_ghz": amplitude,
                    "coherence": item["metrics"]["coherence"][row][column],
                    "source_run_id": item["run_id"],
                    "point_index": row * 3 + column,
                }
        self.seed_run = item
        self.save()
        self.check(item, ("integrity", "diagnostics", "declared_expectation"))

    def solve(self, ej, amplitude, profile, parent):
        key = (ej, amplitude, profile)
        if key in self.samples:
            return self.samples[key]
        with tempfile.TemporaryDirectory(prefix="entanglix-m3-point-") as directory:
            directory = Path(directory)
            model = tomllib.loads((EXAMPLE / "system.toml").read_text())
            model["parameters"]["EJ"]["value"] = ej
            path = directory / "system.toml"
            path.write_text(tomli_w.dumps(model))
            with np.load(EXAMPLE / "readout-state.npz", allow_pickle=False) as archive:
                ket = archive["psi"].copy()
            if profile == "lab_refined":
                source = load_system(path)
                model["model"]["subsystems"][0].update(retained_levels=6, cutoff=32)
                model["model"]["subsystems"][1]["retained_levels"] = 8
                path.write_text(tomli_w.dumps(model))
                ket = embed_state(source, load_system(path), ket)
            np.savez(directory / "readout-state.npz", psi=ket)
            spec = tomllib.loads((EXAMPLE / "readout.toml").read_text())
            spec["parameters"]["EJ"] = ej
            spec["needs"]["device"] = "cpu"
            spec["timeout_seconds"] = 180
            evolution = spec["evolution"]
            evolution["drive_amplitudes_ghz"]["readout"] = amplitude
            evolution["carriers"]["readout"]["frequency_ghz"] = FREQUENCIES[ej]
            if profile == "rwa":
                evolution["frame"]["frequencies_ghz"] = dict.fromkeys(("q", "r"), FREQUENCIES[ej])
            else:
                evolution.pop("frame")
            if profile == "lab_refined":
                evolution.update(rtol=1e-11, atol=1e-13)
            protocol = directory / "simulate.toml"
            protocol.write_text(tomli_w.dumps(spec))
            started = time.perf_counter()
            job = simulate(protocol, runs_root=self.runs_root, parent_run_ids=(parent,))
        item = retain(job, time.perf_counter() - started)
        item.update(ej_ghz=ej, amplitude_ghz=amplitude, profile=profile)
        self.result["runs"].append(item)
        self.samples[key] = {
            "amplitude_ghz": amplitude,
            "coherence": item["metrics"]["coherence"],
            "source_run_id": item["run_id"],
        }
        self.save()
        print(
            f"EJ={ej:g} A={amplitude:.9g} {profile}: C={item['metrics']['coherence']:.12g}, "
            f"job={item['job_wall_seconds']:.2f}s, launch={item['launch_elapsed_seconds']:.2f}s",
            flush=True,
        )
        return self.samples[key]

    def bisect(self, ej, bracket, steps):
        baseline = self.samples[ej, 0.0, "rwa"]["coherence"]
        lower, upper = bracket
        for _ in range(steps):
            midpoint = (lower["amplitude_ghz"] + upper["amplitude_ghz"]) / 2
            sample = self.solve(ej, midpoint, "rwa", upper["source_run_id"])
            if baseline - sample["coherence"] < TARGET:
                lower = sample
            else:
                upper = sample
        return lower, upper

    def compare(self, ej, bracket, stage):
        rows = []
        for amplitude in (0.0, *(sample["amplitude_ghz"] for sample in bracket)):
            parent = self.solve(ej, amplitude, "rwa", self.seed_run["run_id"])["source_run_id"]
            coarse = self.solve(ej, amplitude, "lab", parent)
            refined = self.solve(ej, amplitude, "lab_refined", coarse["source_run_id"])
            rows.append(
                {"rwa": self.samples[ej, amplitude, "rwa"], "lab": coarse, "lab_refined": refined}
            )
        endpoints = []
        for row in rows[1:]:
            losses = {
                profile: rows[0][profile]["coherence"] - row[profile]["coherence"]
                for profile in ("rwa", "lab", "lab_refined")
            }
            endpoints.append(
                {
                    "amplitude_ghz": row["rwa"]["amplitude_ghz"],
                    "additional_loss": losses,
                    "lab_minus_rwa_loss": losses["lab_refined"] - losses["rwa"],
                    "observed_lab_refinement_change": abs(losses["lab_refined"] - losses["lab"]),
                    "source_runs": {profile: row[profile]["source_run_id"] for profile in row},
                }
            )
        differences = [row["additional_loss"]["lab_refined"] - TARGET for row in endpoints]
        changes = [row["observed_lab_refinement_change"] for row in endpoints]
        confirmed = differences[0] < -changes[0] and differences[1] > changes[1]
        item = {
            "ej_ghz": ej,
            "stage": stage,
            "width_ghz": bracket[1]["amplitude_ghz"] - bracket[0]["amplitude_ghz"],
            "baseline_runs": {profile: rows[0][profile]["source_run_id"] for profile in rows[0]},
            "endpoints": endpoints,
            "lab_confirmation": "observed_bracket_with_refinement_margin"
            if confirmed
            else "not_confirmed",
            "target_margin_lab_refined": differences,
            "suggested_action": "report conditional bracket"
            if confirmed
            else "handoff: additional lab-frame search needed; more RWA bisection is insufficient",
        }
        self.result["brackets"].append(item)
        self.save()
        print(
            f"EJ={ej:g} {stage}: width={item['width_ghz']:.9g}, {item['lab_confirmation']}, "
            f"lab target margins={differences}",
            flush=True,
        )

    def study(self):
        started = time.perf_counter()
        self.seed()
        self.result["ej12_finding"] = {
            "source_run_id": self.seed_run["run_id"],
            "sampled_additional_losses": [
                self.samples[12.0, 0.0, "rwa"]["coherence"]
                - self.samples[12.0, a, "rwa"]["coherence"]
                for a in (0.0, 0.001, 0.002)
            ],
            "claim": "No crossing observed at sampled amplitudes in the allowed range; "
            "under the monotonicity assumption, no crossing in this range.",
            "proposal": "Ask for scope expansion if a crossing at EJ=12 is needed.",
        }
        for ej in (13.5, 15.0):
            bracket = self.bisect(
                ej, (self.samples[ej, 0.001, "rwa"], self.samples[ej, 0.002, "rwa"]), 3
            )
            self.compare(ej, bracket, "normal")
            tighter = self.bisect(ej, bracket, 3)
            self.compare(ej, tighter, "tighter")
            for sample in (*bracket, *tighter):
                source = next(
                    item
                    for item in self.result["runs"]
                    if item["run_id"] == sample["source_run_id"]
                )
                if any(
                    item["parent_run_ids"] == [source["run_id"]] for item in self.result["checks"]
                ):
                    continue
                self.check(source, ("integrity", "segments", "diagnostics"))
        for item in self.result["runs"]:
            if item.get("profile") == "lab_refined":
                self.check(item, ("integrity",))
        self.result.update(
            complete=True,
            elapsed_seconds=time.perf_counter() - started,
            simulation_point_count=9 + len(self.result["runs"]) - 1,
            computation_job_count=len(self.result["runs"]),
            check_job_count=len(self.result["checks"]),
            job_wall_seconds=sum(
                item["job_wall_seconds"] for item in self.result["runs"] + self.result["checks"]
            ),
        )
        self.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    Probe(parser.parse_args().output).study()
