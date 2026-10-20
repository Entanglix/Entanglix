"""Bounded pilot for the co-design example's duration, bound and truncation; CPU, no runner jobs.

Each regime rescales the example base: the duration, the Rabi-rate amplitude bound and the
derivative bounds scaled with the duration. Every candidate and seed is solved in-process with
the example's GRAPE settings and evaluated with the uncertainty diagnostic and its refinement.
The comparison quantity, the worst fidelity over the declared samples, was fixed beforehand.

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv-runner/bin/python \
  tests/probes/m5_pilot.py --output tests/probes/m5_pilot_results.json
"""

import argparse
import itertools
import json
import tempfile
import time
import tomllib
from pathlib import Path

import tomli_w

from entanglix.corrector import load_policy
from entanglix.corrector.control import sampled_values
from entanglix.protocols.control import load_control
from entanglix.solvers import select_solver

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/transmon-codesign"
REGIMES = ((20.0, 0.06), (10.0, 0.1), (6.0, 0.15))
LEVELS = (3, 5)
CAPACITANCES = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0)
SEEDS = (7, 19, 43)


def base(directory, levels, duration, rabi):
    system = tomllib.loads((EXAMPLE / "system.toml").read_text())
    system["truncation"]["retained_levels"] = levels
    (directory / "system.toml").write_text(tomli_w.dumps(system))
    control = tomllib.loads((EXAMPLE / "control.toml").read_text())
    scale = control["pulse"]["T"] / duration
    bounds = control["rabi_bounds"]
    bounds.update(
        amplitude_ghz=rabi,
        du_ghz_per_ns=bounds["du_ghz_per_ns"] * scale,
        ddu_ghz_per_ns2=bounds["ddu_ghz_per_ns2"] * scale**2,
    )
    control["pulse"]["T"] = duration
    (directory / "control.toml").write_text(tomli_w.dumps(control))
    return directory / "control.toml", bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend, _ = select_solver("grape")
    policy = load_policy()
    runs, summary = [], []
    with tempfile.TemporaryDirectory(prefix="entanglix-m5-pilot-") as scratch:
        for levels, (duration, rabi) in itertools.product(LEVELS, REGIMES):
            directory = Path(scratch) / f"L{levels}-T{duration:g}"
            directory.mkdir()
            path, bounds = base(directory, levels, duration, rabi)
            for csh in CAPACITANCES:
                rows = []
                for seed in SEEDS:
                    started = time.monotonic()
                    loaded = load_control(path, values={"parameters.Csh": csh, "pulse.seed": seed})
                    problem = directory / "problem.toml"
                    problem.write_bytes(loaded.problem_bytes)
                    solution = backend.solve(problem, loaded.definition.solver.options)
                    samples = sampled_values(
                        loaded, solution.pulse, policy, loaded.definition.evaluation.refinement
                    )
                    nominal, declared = samples[0], samples[1:]
                    rows.append(
                        {
                            "levels": levels,
                            "T_ns": duration,
                            "rabi_bounds": bounds,
                            "Csh_fF": csh,
                            "seed": seed,
                            "solver_reported_fidelity": solution.solver_reported_fidelity,
                            "termination": solution.termination,
                            "nominal_fidelity": nominal["fidelity"],
                            "worst_fidelity": min(row["fidelity"] for row in declared),
                            "refined_worst_fidelity": min(
                                row["refined"]["fidelity"] for row in declared
                            ),
                            "largest_refinement_change": max(
                                abs(row["refined"]["fidelity_change"]) for row in samples
                            ),
                            "seconds": time.monotonic() - started,
                        }
                    )
                runs.extend(rows)
                worst = [row["worst_fidelity"] for row in rows]
                best = max(rows, key=lambda row: row["worst_fidelity"])
                summary.append(
                    {
                        "levels": levels,
                        "T_ns": duration,
                        "rabi_ghz": rabi,
                        "Csh_fF": csh,
                        "best_worst_fidelity": best["worst_fidelity"],
                        "best_seed": best["seed"],
                        "best_refined_worst_fidelity": best["refined_worst_fidelity"],
                        "seed_spread": max(worst) - min(worst),
                        "largest_refinement_change": max(
                            row["largest_refinement_change"] for row in rows
                        ),
                        "lowest_nominal_fidelity": min(row["nominal_fidelity"] for row in rows),
                    }
                )
                print(json.dumps(summary[-1]), flush=True)
    result = {
        "comparison_quantity": "worst fidelity over the example's declared evaluation samples",
        "evaluation": tomllib.loads((EXAMPLE / "control.toml").read_text())["evaluation"],
        "seconds": sum(row["seconds"] for row in runs),
        "summary": summary,
        "runs": runs,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
