"""CPU-only device/control measurements; standalone jobs, no agent session.

Run with the checkout's existing runner extras, for example:
ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12 JAX_PLATFORMS=cpu \
  CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv-runner/bin/python \
  tests/probes/m5_codesign.py

The workspace retains inputs, runner records and a solve index. Repeating this
command reuses indexed solves/checks and rebuilds evaluations and the summary.
Use a new --workspace for a new solve matrix. No GPU is initialized or used.
"""

import argparse
import copy
import hashlib
import itertools
import json
import os
import subprocess
import time
import tomllib
from contextlib import redirect_stderr
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scqubits as scq
import tomli_w

from entanglix.corrector import load_policy
from entanglix.corrector.control import constraints, replay_values
from entanglix.corrector.metrics import spectral_arrays
from entanglix.corrector.verification import check_run
from entanglix.engines import QuTiPIntegrator
from entanglix.evolution import EvolutionInput
from entanglix.models import ConstructedSystem, load_system
from entanglix.protocols.control import control, load_control
from entanglix.runner import read_run
from entanglix.solvers import Pulse

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "examples/transmon-control"
CAPACITANCES = (10.0, 30.0, 60.0)
SEEDS = (7, 19, 43)
CHARGES = (0.0, 0.125, 0.25, 0.375, 0.5)
EJ_FRACTION = 0.001


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_toml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(value))
    return path


def document(capacitance, *, charge=0.23, ej_fraction=0.0, levels=3, cutoff=30):
    data = tomllib.loads((EXAMPLE / "system.toml").read_text())
    data["parameters"]["Csh"]["value"] = capacitance
    data["parameters"]["EJ"]["value"] *= 1 + ej_fraction
    data["model"]["offset_charges"]["1"] = charge
    data["truncation"] = {"retained_levels": levels, "cutoffs": {"theta1": cutoff}}
    return data


def spectrum(workspace):
    rows = []
    for cap in (2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0):
        root = workspace / "screen" / f"c{cap:g}"
        source = write_toml(root / "system.toml", document(cap))
        system = load_system(source)
        energies = spectral_arrays(system, 3, {})["energies_ghz"]
        branches = system.provenance["converted_branches"]
        ec = 1 / sum(1 / branch["charging_energy_ghz"] for branch in branches.values())
        frequencies = []
        for charge in CHARGES:
            path = write_toml(root / f"ng{charge:g}.toml", document(cap, charge=charge))
            e = spectral_arrays(load_system(path), 2, {})["energies_ghz"]
            frequencies.append(float(e[1] - e[0]))
        shifted = []
        for fraction in (-EJ_FRACTION, EJ_FRACTION):
            path = write_toml(root / f"ej{fraction:g}.toml", document(cap, ej_fraction=fraction))
            e = spectral_arrays(load_system(path), 2, {})["energies_ghz"]
            shifted.append(float(e[1] - e[0]))
        rows.append(
            {
                "Csh_fF": cap,
                "EJ_over_EC": 15.0 / ec,
                "f01_ghz": float(energies[1] - energies[0]),
                "anharmonicity_ghz": float(energies[2] - 2 * energies[1] + energies[0]),
                "n01": float(abs(system.operators["charge"][0, 1])),
                "n12": float(abs(system.operators["charge"][1, 2])),
                "charge_samples": list(CHARGES),
                "charge_f01_ghz": frequencies,
                "sampled_charge_dispersion_ghz": float(np.ptp(frequencies)),
                "EJ_fraction": EJ_FRACTION,
                "EJ_shifted_f01_ghz": shifted,
                "selected": cap in CAPACITANCES,
            }
        )
    return rows


def control_input(directory, row, policy, seed, backend="grape"):
    data = tomllib.loads((EXAMPLE / "control.toml").read_text())
    write_toml(directory / "system.toml", document(row["Csh_fF"]))
    frequency = row["f01_ghz"]
    data["frame"]["frequencies_ghz"] = {"junction_with_shunt": frequency}
    for channel in data["channels"]:
        channel["frequency_ghz"] = frequency
        if policy == "rabi_bound":
            channel["amplitude_bound_ghz"] = 0.06 / row["n01"]
    if policy == "rabi_bound":
        data["problem"].update(du_bound=0.1 / row["n01"], ddu_bound=0.1 / row["n01"])
    data["pulse"]["seed"] = seed
    data["solver"]["backend"] = backend
    return write_toml(directory / "control.toml", data)


def detuning_cases(workspace, row):
    source = control_input(workspace / "detuning", row, "rabi_bound", SEEDS[0])
    base = tomllib.loads(source.read_text())
    results = []
    for label in ("frame_and_carriers", "carriers_only"):
        data = copy.deepcopy(base)
        for channel in data["channels"]:
            channel["frequency_ghz"] += 0.001
        if label == "frame_and_carriers":
            data["frame"]["frequencies_ghz"]["junction_with_shunt"] += 0.001
        path = write_toml(source.parent / f"{label}.toml", data)
        try:
            loaded = load_control(path)
        except ValueError as exc:
            results.append({"case": label, "accepted": False, "error": str(exc)})
        else:
            drift = loaded.system.hamiltonian_ghz
            results.append(
                {
                    "case": label,
                    "accepted": True,
                    "rotating_f01_ghz": float((drift[1, 1] - drift[0, 0]).real),
                    "listing": loaded.listing,
                }
            )
    return results


def retained_job(workspace, index, label, source):
    entry = index.get(label)
    if entry is None:
        print(f"solve {label}", flush=True)
        started = time.monotonic()
        with (workspace / "launches.log").open("a") as stream, redirect_stderr(stream):
            result = control(source, runs_root=workspace / "runs", repository=REPOSITORY)
        entry = {"path": result["path"], "launch_elapsed_seconds": time.monotonic() - started}
        index[label] = entry
        save(workspace / "solve-index.json", index)
    record = read_run(Path(entry["path"]))
    if record["status"] == "completed" and "check_path" not in entry:
        with (workspace / "launches.log").open("a") as stream, redirect_stderr(stream):
            checked = check_run(Path(entry["path"]), ("replay", "constraints"))
        entry["check_path"] = checked["path"]
        save(workspace / "solve-index.json", index)
    return entry, record


def summarize_job(label, entry, record):
    result = record.get("result", {})
    row = {
        "label": label,
        "path": entry["path"],
        "run_id": record["manifest"]["run_id"],
        "status": record["status"],
        "usage": record.get("finished", {}).get("usage", {}),
        "launch_elapsed_seconds": entry["launch_elapsed_seconds"],
        "env_hash": record["manifest"]["env_hash"],
        "problem_sha256": result.get("problem_sha256"),
        "solver_fidelity": result.get("metrics", {}).get("solver_reported_fidelity"),
        "termination": result.get("termination", {}),
    }
    if record["status"] != "completed":
        return {**row, "error": record["log_tail"]}
    trace = tomllib.loads((Path(entry["path"]) / "artifacts/solver-trace.toml").read_text())
    calls = trace.get("objective_calls", [])
    row["first_objective_gradient_seconds"] = calls[0]["seconds"] if calls else None
    row["later_objective_gradient_median_seconds"] = (
        float(np.median([item["seconds"] for item in calls[1:]])) if len(calls) > 1 else None
    )
    if (native := Path(entry["path"]) / "artifacts/piccolo/result.toml").exists():
        row["native_solve_seconds"] = tomllib.loads(native.read_text())["wall_seconds"]
    checked = read_run(Path(entry["check_path"]))
    row["check_run_id"], row["check_path"] = checked["manifest"]["run_id"], entry["check_path"]
    row["check_usage"] = checked.get("finished", {}).get("usage", {})
    row["checks"] = checked.get("result", {}).get("report", {}).get("checks", [])
    replay = next((item for item in row["checks"] if item["name"] == "replay"), {})
    constraints = next((item for item in row["checks"] if item["name"] == "constraints"), {})
    row.update(
        replay_fidelity=replay.get("fidelity"),
        replay_leakage=replay.get("leakage"),
        constraints_outcome=constraints.get("outcome", "not run"),
    )
    row["iteration_cap_reached"] = result["termination"]["iterations"] >= 100
    print(
        f"  {label}: F={row['replay_fidelity']} constraints={row['constraints_outcome']} "
        f"job={row['usage']['wall_seconds']:.1f}s",
        flush=True,
    )
    return row


def circuit_vectors(data, system):
    """Repeat this fixture's circuit construction solely to observe its basis signs."""
    branches = system.provenance["converted_branches"]
    ej = branches["junction"]["josephson_energy_ghz"]
    ecj = branches["junction"]["charging_energy_ghz"]
    ecs = branches["shunt"]["charging_energy_ghz"]
    count = data["truncation"]["retained_levels"]
    circuit = scq.Circuit(
        f"branches:\n- [JJ, 0, 1, {ej:.17g}, {ecj:.17g}]\n- [C, 0, 1, {ecs:.17g}]\n",
        from_file=False,
        truncated_dim=count,
        ext_basis="harmonic",
    )
    circuit.cutoff_n_1 = data["truncation"]["cutoffs"]["theta1"]
    circuit.ng1 = -data["model"]["offset_charges"]["1"]
    _, vectors = circuit.eigensys(evals_count=count)
    charge = vectors.conj().T @ circuit.n1_operator() @ vectors - np.eye(count) * -circuit.ng1
    residual = float(np.max(abs(charge - system.operators["charge"])))
    if residual > 1e-10:
        raise ValueError(f"basis-sign reconstruction disagrees with loader: {residual}")
    return vectors


def aligned_control(path, nominal_vectors):
    loaded = load_control(path)
    data = tomllib.loads((path.parent / "system.toml").read_text())
    original = load_system(path.parent / "system.toml")
    vectors = circuit_vectors(data, original)
    rows = max(len(vectors), len(nominal_vectors))
    old = np.zeros((rows, nominal_vectors.shape[1]), complex)
    new = np.zeros((rows, vectors.shape[1]), complex)
    old_start, new_start = (rows - len(nominal_vectors)) // 2, (rows - len(vectors)) // 2
    old[old_start : old_start + len(nominal_vectors)] = nominal_vectors
    new[new_start : new_start + len(vectors)] = vectors
    common = min(old.shape[1], new.shape[1])
    overlaps = np.einsum("ij,ij->j", old[:, :common].conj(), new[:, :common])
    phases = np.ones(vectors.shape[1], complex)
    phases[:common] = np.where(abs(overlaps) > 0, np.exp(-1j * np.angle(overlaps)), 1)
    system = replace(
        loaded.system,
        drives={
            name: phases.conj()[:, None] * operator * phases[None, :]
            for name, operator in loaded.system.drives.items()
        },
    )
    return replace(loaded, system=system), {
        "matching_overlaps": abs(overlaps).tolist(),
        "signs": phases.real.tolist(),
        "unmatched_extra_levels": vectors.shape[1] - common,
    }


def evaluate(loaded, pulse, engine, rate):
    """Same entangled-input fidelity as replay_values, with optional relaxation."""
    n, d = len(loaded.system.hamiltonian_ghz), 2
    identity = np.eye(d)
    lower = np.diag(np.sqrt(np.arange(1, n)), 1)
    model = ConstructedSystem(
        id="m5-authored-evaluation",
        hamiltonian_ghz=np.kron(
            loaded.system.hamiltonian_ghz - loaded.listing["energy_origin_ghz"] * np.eye(n),
            identity,
        ),
        operators={},
        dissipators={"assumed_relaxation": (np.kron(lower, identity), rate)},
        dimensions=(n, d),
        basis="declared-control-frame",
        provenance={},
        drives={
            name: np.kron(operator, identity) for name, operator in loaded.system.drives.items()
        },
    )
    initial = np.zeros((n, d), complex)
    initial[:d] = np.eye(d) / np.sqrt(d)
    target = np.zeros_like(initial)
    target[:d] = loaded.definition.goal.operator() / np.sqrt(d)
    trace = engine.evolve(
        model,
        EvolutionInput(
            initial.ravel(),
            pulse.times_ns,
            drive_amplitudes_ghz={
                channel.id: pulse.amplitudes_ghz[i, :-1]
                for i, channel in enumerate(loaded.definition.channels)
            },
            rtol=1e-11,
            atol=1e-13,
        ),
    )
    rho = trace.states[-1]
    survival = float(np.diag(rho).real[: d * d].sum())
    overlap = float((target.ravel().conj() @ rho @ target.ravel()).real)
    return {
        "fidelity": (survival + d * overlap) / (d + 1),
        "leakage": 1 - survival,
        "engine": trace.engine,
        "method": trace.method,
        "device": trace.device,
        "rate_per_ns": rate,
    }


def sampled_evaluation(workspace, best, pulse, sample, vectors, *, levels=3, cutoff=30):
    started = time.monotonic()
    cap = best["Csh_fF"]
    label = f"{sample['label']}-l{levels}-c{cutoff}"
    directory = workspace / "evaluations" / f"c{cap:g}" / label
    source = tomllib.loads((Path(best["path"]) / "spec/control.toml").read_text())
    data = document(
        cap, charge=sample["ng"], ej_fraction=sample["ej_fraction"], levels=levels, cutoff=cutoff
    )
    write_toml(directory / "system.toml", data)
    path = write_toml(directory / "control.toml", source)
    loaded, matching = aligned_control(path, vectors)
    values = replay_values(loaded, pulse, load_policy())
    return {
        **sample,
        **values,
        "basis_matching": matching,
        "system_path": str(directory / "system.toml"),
        "control_path": str(path),
        "retained_levels": levels,
        "theta1_cutoff": cutoff,
        "elapsed_seconds": time.monotonic() - started,
    }


def uncertainty_and_refinement(workspace, best):
    run = Path(best["path"])
    data = tomllib.loads((run / "spec/system.toml").read_text())
    vectors = circuit_vectors(data, load_system(run / "spec/system.toml"))
    with np.load(run / "artifacts/pulse.npz", allow_pickle=False) as archive:
        pulse = Pulse(archive["times_ns"].copy(), archive["amplitudes_ghz"].copy())
    nominal = {"label": "nominal", "ng": 0.23, "ej_fraction": 0.0}
    samples = [nominal]
    samples.extend({"label": f"ng{ng:g}", "ng": ng, "ej_fraction": 0.0} for ng in CHARGES)
    samples.extend(
        {"label": f"ej{fraction:g}", "ng": 0.23, "ej_fraction": fraction}
        for fraction in (-EJ_FRACTION, EJ_FRACTION)
    )
    measured = [sampled_evaluation(workspace, best, pulse, sample, vectors) for sample in samples]
    worst_charge = min(measured[1:6], key=lambda item: item["fidelity"])
    worst_ej = min(measured[6:], key=lambda item: item["fidelity"])
    combined = {
        "label": "combined",
        "ng": worst_charge["ng"],
        "ej_fraction": worst_ej["ej_fraction"],
    }
    measured.append(sampled_evaluation(workspace, best, pulse, combined, vectors))
    worst = min(measured, key=lambda item: item["fidelity"])
    refined = []
    for point in (measured[0], worst):
        for levels, cutoff in ((5, 30), (3, 40)):
            value = sampled_evaluation(
                workspace, best, pulse, point, vectors, levels=levels, cutoff=cutoff
            )
            value["fidelity_change"] = abs(value["fidelity"] - point["fidelity"])
            value["leakage_change"] = abs(value["leakage"] - point["leakage"])
            refined.append(value)
    return {
        "Csh_fF": best["Csh_fF"],
        "source_run_id": best["run_id"],
        "samples": measured,
        "worst_sample": worst["label"],
        "worst_sampled_fidelity": worst["fidelity"],
        "nominal_replay_agreement": abs(measured[0]["fidelity"] - best["replay_fidelity"]),
        "offset_charge_fidelity_span": float(np.ptp([item["fidelity"] for item in measured[1:6]])),
        "refinements": refined,
        "nominal_refinement_change": max(item["fidelity_change"] for item in refined[:2]),
        "original_worst_point_refinement_change": max(
            item["fidelity_change"] for item in refined[2:]
        ),
        "uncertainty_ranking": "unresolved: only the original worst sample was refined; the minimum over the sample set was not refined",
    }, pulse


def reading_results():
    return {
        "transition": "entanglix.corrector.metrics.spectral_arrays(system, 2, {}) returns energies; f01 is their difference. There is no dedicated named-transition resolver.",
        "per_candidate_inputs": "Copy system.toml and change Csh; compute f01; set the same nominal frame and both carriers; set amplitude/du/ddu bounds under the declared policy. Evaluation retains those values.",
        "perturbed_inputs": "EJ is a named parameter, but control has no parameter override. Offset charge is literal model data. Both currently need a separate system file and adjacent control spec; evaluations here are probe code.",
        "sweep_point": "A copied adjacent sweep-point system.toml can be control.system. CLI/Python parent_run_ids can name the sweep run, but no field binds its point index. MCP control exposes no parent_run_ids. A referenced file inside spec/points cannot be named directly because control.system must be beside the control input.",
        "decisions": "Three known exact control bases can share one revision and one start decision, provided all job/check operations are allowed. A fourth adaptive candidate requires one approve-revision decision. Checkpoint assessments are findings, not user decisions; optional reviews or limits can add decisions only if declared.",
        "authored_evaluation": "An exact run-script base can hold the evaluation script, system.toml, pulse.npz and numerical data beside the system. Each candidate/perturbation-dependent bundle needs its own approved base. Its authored fidelity metrics and evidence stay authored data, not built-in check outcomes.",
        "variation_binding": "Control variation would need to bind system parameter overrides, nominal transition selection, frame/carrier derivation and the chosen amplitude/derivative-bound policy. Current admission accepts exact control bundles only.",
        "sources": [
            "src/entanglix/protocols/control.py:load_control, control",
            "src/entanglix/corrector/metrics.py:spectral_arrays",
            "src/entanglix/mcp.py:control",
            "src/entanglix/investigations.py:_in_scope",
            "src/entanglix/runner.py:_script_gate, _system_inputs",
            "docs/operations/investigations.md:Launch work within scope",
        ],
    }


def interpret_results(results):
    raw = next(job for job in results["jobs"] if job["label"] == "c30-raw_bound-s7")
    primary = next(job for job in results["jobs"] if job["label"] == "c30-rabi_bound-s7")
    raw_model = load_control(Path(raw["path"]) / "spec/control.toml")
    primary_model = load_control(Path(primary["path"]) / "spec/control.toml")
    same_model = np.array_equal(
        raw_model.system.hamiltonian_ghz, primary_model.system.hamiltonian_ghz
    ) and all(
        np.array_equal(operator, primary_model.system.drives[name])
        for name, operator in raw_model.system.drives.items()
    )
    with np.load(Path(raw["path"]) / "artifacts/pulse.npz", allow_pickle=False) as archive:
        pulse = Pulse(archive["times_ns"].copy(), archive["amplitudes_ghz"].copy())
    admissibility = constraints(primary_model, pulse, load_policy())
    results["optimizer_interpretation"] = {
        "nominal_candidate_ranking": "unresolved: solver variability prevents interpreting the nominal candidate differences",
        "c30_raw_pulse_under_rabi_bounds": {
            "source_run_id": raw["run_id"],
            "rabi_input_run_id": primary["run_id"],
            "same_model_and_drives": same_model,
            "replay_fidelity": raw["replay_fidelity"],
            "constraints": admissibility,
            "evidence": "In-process constraints evaluation of the retained raw-bound pulse against the retained Rabi-rate input; no solve or verification job added.",
        },
        "extra_solves_not_run": [
            {
                "Csh_fF": cap,
                "backend": "grape",
                "comparison_policy": "rabi_bound",
                "seeds": [73, 101, 211],
                "max_iter": 100,
                "tol": 1e-8,
            }
            for cap in CAPACITANCES
        ],
        "followup_scope": "The named matched-seed solves would test whether the attained fidelities stabilize under the primary policy; a finite seed set cannot certify a global optimum. The retained admissible raw-bound pulse already disproves a device-limited explanation for the 30 fF primary deficit.",
    }
    local_solution = same_model and admissibility["outcome"] == "pass"
    for candidate in results["uncertainty"]:
        if candidate["Csh_fF"] == 30 and local_solution:
            candidate["pulse_interpretation"] = (
                "These uncertainty figures use the selected primary-policy local-optimum pulse, "
                "not the higher-fidelity raw-bound pulse that also satisfies the primary bounds; "
                "they do not establish the device's best attainable robustness."
            )
    selected = [row for row in results["spectrum_screen"] if row["selected"]]
    best = results["best_pulses"]
    uncertain = results["uncertainty"]
    timing = results["timings"]
    dispersion = "/".join(f"{row['sampled_charge_dispersion_ghz'] * 1000:.6g}" for row in selected)
    anharmonicity = "/".join(f"{row['anharmonicity_ghz']:.6g}" for row in selected)
    spread = "/".join(f"{row['seed_spread']:.6g}" for row in best)
    separation = next(
        item["separation"]
        for item in results["nominal_comparisons"]
        if item["Csh_pair_fF"] == [10.0, 60.0]
    )
    worst = "/".join(f"{row['worst_sampled_fidelity']:.9f}" for row in uncertain)
    nominal_change = "/".join(f"{row['nominal_refinement_change']:.3g}" for row in uncertain)
    worst_change = "/".join(
        f"{row['original_worst_point_refinement_change']:.3g}" for row in uncertain
    )
    cutoff_change = max(
        item["fidelity_change"]
        for row in uncertain
        for item in row["refinements"]
        if item["theta1_cutoff"] == 40
    )
    initial_elapsed = timing["initial_full_pass_elapsed_seconds"]
    original = f"{initial_elapsed:.3f} s" if initial_elapsed is not None else "not retained"
    primary_best = next(row["nominal_fidelity"] for row in best if row["Csh_fF"] == 30)
    results["answers"] = {
        "1": f"At Csh=10/30/60 fF, sampled charge dispersions are {dispersion} MHz and anharmonicities are {anharmonicity} GHz. The selected set spans charge sensitivity and leakage-related level spacing while retaining the first-gate device.",
        "2": "Frame and carriers shifted together by 1 MHz are accepted; carriers shifted alone are refused by channel name. See the retained detuning entries for exact outcomes.",
        "3": f"The 12 retained solves include a 30 fF raw-bound pulse with replay fidelity {raw['replay_fidelity']:.10f} that also passes the Rabi-rate constraints, versus the primary best {primary_best:.10f}; the primary results are suboptimal local solutions. Solver variability prevents interpreting nominal candidate differences: seed spreads at 10/30/60 fF are {spread}, versus the 10-to-60 fF separation {separation:.3g}; the additional matched-seed solves are named but were not run.",
        "4": f"Worst sampled fidelities at 10/30/60 fF are {worst}; the 30 fF values come from the selected local-optimum pulse, not the higher-fidelity admissible raw-bound pulse. All uncertainty rankings remain unresolved because only each original worst point was refined, and the samples do not establish a continuous-region worst case.",
        "5": f"Levels 3->5 change nominal fidelity by {nominal_change} and original-worst-point fidelity by {worst_change} at 10/30/60 fF; cutoff 30->40 changes fidelity by at most {cutoff_change:.3g}. Four replays per candidate cover only those selected points and do not establish convergence of the sampled minimum.",
        "6": f"With the assumed 1e-5/ns relaxation rate, the passive-ancilla evaluations agree between CPU engines within {results['open_system']['finite_rate_engine_fidelity_difference']:.3g} in fidelity. This is feasibility evidence for an assumed channel, not design-dependent relaxation.",
        "7": "Existing operations compose with exact bases. Particular sweep-point binding, MCP parent lineage and parameter-dependent control data are the recorded gaps; authored evaluation is another exact script bundle.",
        "8": f"The original item 3-5 full-pass elapsed time is {original}; retained solver job time totals {timing['solve_job_seconds']:.3f} s and check job time {timing['check_job_seconds']:.3f} s. This invocation took {timing['item_3_to_5_elapsed_seconds']:.3f} s for items 3-5, including {timing['evaluation_seconds']:.3f} s of direct evaluations; indexed-job reuse is {timing['reused_indexed_jobs']}.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path.home() / "Projects/entanglix-research/m5-probe"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("m5_codesign_results.json")
    )
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false"
    )
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        parser.error("set OPENBLAS_NUM_THREADS=1 before starting the probe")
    profile = tomllib.loads((REPOSITORY / ".venv-runner-extras.toml").read_text())["extras"]
    os.environ["ENTANGLIX_RUNNER_EXTRAS"] = ",".join(profile)
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    previous = json.loads(args.output.read_text()) if args.output.exists() else {}
    previous_timing = (
        previous.get("timings", {}) if previous.get("workspace") == str(workspace) else {}
    )
    screen = spectrum(workspace)
    selected = [row for row in screen if row["selected"]]
    results = {
        "measured_at": datetime.now(UTC).isoformat(),
        "workspace": str(workspace),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "script_sha256": digest(Path(__file__)),
        "provenance_note": "script_sha256 identifies this rebuild/evaluation script. Indexed solves and checks retain their original executed probe source in the recorded code snapshots.",
        "versions": {
            name: version(name)
            for name in ("numpy", "scipy", "qutip", "scqubits", "jax", "dynamiqs", "tomli-w")
        },
        "runner_extras": profile,
        "conventions": {
            "comparison": "Fixed nominal 0-1 Rabi-rate bound: amplitude 0.06/|n01|, du and ddu 0.1/|n01|, in authored units. Alternate: unchanged first-gate raw bounds.",
            "best": "Highest replay fidelity among the three primary GRAPE seeds passing constraints; failures retained.",
            "evaluation": "Returned amplitudes and knots; nominal carriers, phases, frame and bounds fixed. Perturbed charge operator at authored scale. Closed RWA model only.",
            "logical_states": "Each perturbed model's eigenstates, with overlapping charge-basis eigenvectors matched in sign. Quasistatic variation, preparation in actual levels.",
            "resolution": "Sampled refinement sensitivity only: nominal separation exceeds 1e-7 and the sum of nominal refinement changes. This says nothing about optimizer resolution; solver variability prevents a nominal device ranking. 1e-7 is an agreement floor, not an error bound; uncertainty rankings also remain unresolved.",
            "sampling": "Five charge samples, two one-at-a-time EJ samples and the combined individually worst pair; worst sampled, not worst case over a region.",
            "EJ_fraction": EJ_FRACTION,
            "EJ_sampling_reason": "A common +/-0.1% EJ perturbation gives approximately 2.9-5.5 MHz detunings across the selected devices, a visible fixed-pulse stress over 20 ns while remaining a small fractional parameter change. This is a declared probe assumption, not a fabrication distribution.",
            "refinement": "Four replays per candidate: nominal/original worst point, levels 3->5 and cutoff 30->40 separately. No inference about which refined sample is worst.",
            "timings": "CPU workstation, OPENBLAS_NUM_THREADS=1. Runner job wall time excludes launch checks; launch elapsed includes them. First objective call includes compilation; evaluations include model/sign construction. Dynamiqs open-system time includes cold compilation.",
        },
        "spectrum_screen": screen,
        "candidate_choice": "10, 30, 60 fF span sampled dispersion of about 58 MHz, 0.75 MHz and 5.6 kHz; smaller Csh increases anharmonicity while increasing charge sensitivity. 60 fF retains the first gate. Actual gate sensitivity is measured below, not inferred from dispersion alone.",
        "detuning": detuning_cases(workspace, selected[-1]),
        "jobs": [],
        "installed_or_downloaded": "No packages requested or installed by this probe; runner uses its existing locked extras and Piccolo project. Runner logs retain any uv synchronization output.",
        "runtime_session_launched": False,
        "gpu_job_run": False,
    }
    index_path = workspace / "solve-index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    repeated = bool(index)
    started = time.monotonic()
    provenance = {}
    for row in selected:
        for policy, seeds in (("rabi_bound", SEEDS), ("raw_bound", SEEDS[:1])):
            for seed in seeds:
                label = f"c{row['Csh_fF']:g}-{policy}-s{seed}"
                if policy == "raw_bound" and row["Csh_fF"] == 60:
                    results.setdefault("reused_cases", []).append(
                        {"case": label, "reuse": "c60-rabi_bound-s7"}
                    )
                    continue
                source = control_input(workspace / "inputs" / label, row, policy, seed)
                entry, record = retained_job(workspace, index, label, source)
                job = summarize_job(label, entry, record)
                job.update(
                    Csh_fF=row["Csh_fF"], comparison_policy=policy, seed=seed, backend="grape"
                )
                results["jobs"].append(job)
                provenance.setdefault(
                    record["manifest"]["env_hash"],
                    {
                        "environment": record["manifest"]["environment"],
                        "code": record["manifest"]["code"],
                    },
                )
    row = selected[-1]
    label = "c60-piccolo-s7"
    source = control_input(workspace / "inputs" / label, row, "rabi_bound", SEEDS[0], "piccolo")
    entry, record = retained_job(workspace, index, label, source)
    job = summarize_job(label, entry, record)
    job.update(Csh_fF=60.0, comparison_policy="rabi_bound", seed=SEEDS[0], backend="piccolo")
    results["jobs"].append(job)
    provenance.setdefault(
        record["manifest"]["env_hash"],
        {"environment": record["manifest"]["environment"], "code": record["manifest"]["code"]},
    )
    results["provenance_by_env_hash"] = provenance
    save(workspace / "solves-summary.json", results)
    results["best_pulses"], results["uncertainty"] = [], []
    pulses = {}
    for cap in CAPACITANCES:
        eligible = [
            item
            for item in results["jobs"]
            if item["Csh_fF"] == cap
            and item["backend"] == "grape"
            and item["comparison_policy"] == "rabi_bound"
            and item.get("constraints_outcome") == "pass"
        ]
        if not eligible:
            results["best_pulses"].append({"Csh_fF": cap, "reason": "no admissible pulse"})
            continue
        best = max(eligible, key=lambda item: item["replay_fidelity"])
        results["best_pulses"].append(
            {
                "Csh_fF": cap,
                "source_run_id": best["run_id"],
                "label": best["label"],
                "nominal_fidelity": best["replay_fidelity"],
                "seed_spread": float(np.ptp([item["replay_fidelity"] for item in eligible])),
            }
        )
        print(f"uncertainty/refinement c{cap:g}", flush=True)
        values, pulse = uncertainty_and_refinement(workspace, best)
        results["uncertainty"].append(values)
        pulses[cap] = (best, pulse)
    results["timings"] = {
        "item_3_to_5_elapsed_seconds": time.monotonic() - started,
        "reused_indexed_jobs": repeated,
        "solve_job_seconds": sum(
            item.get("usage", {}).get("wall_seconds", 0) for item in results["jobs"]
        ),
        "check_job_seconds": sum(
            item.get("check_usage", {}).get("wall_seconds", 0) for item in results["jobs"]
        ),
        "evaluation_seconds": sum(
            item["elapsed_seconds"]
            for candidate in results["uncertainty"]
            for item in candidate["samples"] + candidate["refinements"]
        ),
    }
    initial_elapsed = previous_timing.get("initial_full_pass_elapsed_seconds")
    if initial_elapsed is None and previous_timing.get("reused_indexed_jobs") is False:
        initial_elapsed = previous_timing["item_3_to_5_elapsed_seconds"]
    results["timings"]["initial_full_pass_elapsed_seconds"] = (
        initial_elapsed if repeated else results["timings"]["item_3_to_5_elapsed_seconds"]
    )
    results["nominal_comparisons"] = []
    for left, right in itertools.combinations(results["uncertainty"], 2):
        a, b = left["samples"][0]["fidelity"], right["samples"][0]["fidelity"]
        separation = abs(a - b)
        movement = left["nominal_refinement_change"] + right["nominal_refinement_change"]
        results["nominal_comparisons"].append(
            {
                "Csh_pair_fF": [left["Csh_fF"], right["Csh_fF"]],
                "separation": separation,
                "combined_refinement_change": movement,
                "exceeds_refinement_resolution": separation > max(1e-7, movement),
                "scope": "Fixed returned pulses and sampled refinements only; not optimizer resolution or a device ranking.",
                "optimizer_ranking": "unresolved",
            }
        )
    results["open_system"] = {"reason": "no admissible 60 fF pulse"}
    if 60.0 in pulses:
        from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

        best, pulse = pulses[60.0]
        loaded = load_control(Path(best["path"]) / "spec/control.toml")
        measurements = []
        for engine in (QuTiPIntegrator(), DynamiqsIntegrator(device_kind="cpu")):
            for rate in (0.0, 1e-5):
                tick = time.monotonic()
                value = evaluate(loaded, pulse, engine, rate)
                measurements.append({**value, "elapsed_seconds": time.monotonic() - tick})
                print(
                    f"open evaluation {value['engine']} rate={rate}: F={value['fidelity']:.9f}",
                    flush=True,
                )
        results["open_system"] = {
            "source_run_id": best["run_id"],
            "operator": "A=sum(sqrt(k)|k-1><k|), collapse=sqrt(gamma)*A tensor I_ancilla; gamma=1e-5/ns (assumed 100 us T1)",
            "measurements": measurements,
            "zero_rate_closed_replay_residuals": {
                item["engine"]: abs(item["fidelity"] - best["replay_fidelity"])
                for item in measurements
                if item["rate_per_ns"] == 0
            },
            "finite_rate_engine_fidelity_difference": abs(
                measurements[1]["fidelity"] - measurements[3]["fidelity"]
            ),
            "scope": "Feasibility/cost within the declared RWA model; assumed common relaxation rate, no design-dependent relaxation claim. Authored data, no new check.",
        }
    results["reading"] = reading_results()
    interpret_results(results)
    save(args.output, results)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
