"""CPU feasibility measurements; carrier/frame code is local to this probe."""

import argparse
import hashlib
import json
import platform
import tempfile
import time
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import qutip as qt
import tomli_w
from qutip.solver.integrator.qutip_integrator import IntegratorVern9

from entanglix.corrector.checks import physical_residuals
from entanglix.corrector.reference import embed_state
from entanglix.models import load_system

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "examples/readout/system.toml"


class CountedVern9(IntegratorVern9):
    """Count accepted internal advances, excluding dense-output interpolation.

    Only the probe uses QuTiP's internal stepping API; production stays on vern9.
    Rejected trial steps are not exposed by this API and are not counted.
    """

    def _prepare(self):
        super()._prepare()
        self.accepted_steps = 0

    def integrate(self, t, copy=True):
        interval_steps = 0
        while self._ode_solver.t < t:
            previous = self._ode_solver.t_front
            super().mcstep(t, copy=False)
            if self._ode_solver.t_front > previous:
                self.accepted_steps += 1
                interval_steps += 1
            if interval_steps > self.options["nsteps"]:
                raise ValueError("probe exceeded max_steps per output interval")
        return self.get_state(copy)


qt.MESolver.add_integrator(CountedVern9, "counted-vern9")


def construct(ej, dimensions):
    document = tomllib.loads(FIXTURE.read_text())
    document["parameters"]["EJ"]["value"] = ej
    q, r = document["model"]["subsystems"]
    q["retained_levels"], r["retained_levels"] = dimensions
    q["cutoff"] = 32 if dimensions[0] > 5 else 30
    with tempfile.TemporaryDirectory(prefix="entanglix-feasibility-") as directory:
        path = Path(directory) / "system.toml"
        path.write_text(tomli_w.dumps(document))
        return load_system(path), document


def spectrum(system, document):
    nq, nr = system.dimensions
    energies, vectors = np.linalg.eigh(system.hamiltonian_ghz)
    labels = {}
    for q, r in ((0, 0), (0, 1), (1, 0), (1, 1)):
        weights = abs(vectors[q * nr + r]) ** 2
        index = int(np.argmax(weights))
        labels[f"{q},{r}"] = {"index": index, "weight": float(weights[index])}
    if len({entry["index"] for entry in labels.values()}) != 4:
        raise ValueError("dressed-state assignments are not unique")
    e = {name: energies[value["index"]] for name, value in labels.items()}
    ground_resonance = e["0,1"] - e["0,0"]
    excited_resonance = e["1,1"] - e["1,0"]
    charge = np.array(system.operator_factors["q.charge"][0])
    coupling = document["model"]["interactions"][0]["coefficient"]["value"]
    resonator_frequency = document["model"]["subsystems"][1]["frequency"]["value"]
    loss_rate = document["dissipators"][0]["rate"]["value"]
    g = abs(coupling * charge[0, 1])
    # The first TensorTerm is the uncoupled qubit Hamiltonian.
    qubit_frequency = float(system.hamiltonian_terms[0].factors[0][1, 1].real)
    qubit_frequency -= float(system.hamiltonian_terms[0].factors[0][0, 0].real)
    detuning = qubit_frequency - resonator_frequency
    return {
        "labels": labels,
        "chi_ghz": float((excited_resonance - ground_resonance) / 2),
        "drive_frequency_ghz": float((excited_resonance + ground_resonance) / 2),
        "qubit_frequency_ghz": qubit_frequency,
        "g01_ghz": float(g),
        "detuning_ghz": detuning,
        "critical_photons": float((detuning / (2 * g)) ** 2),
        "purcell_rate_per_ns": float(loss_rate * (g / detuning) ** 2),
    }


def solve(
    *,
    ej,
    amplitude,
    duration,
    frame,
    dimensions=(5, 6),
    rtol=1e-10,
    kind="readout",
    frequency=None,
    spacing=1.0,
    method="counted-vern9",
):
    system, document = construct(ej, dimensions)
    spectral = spectrum(system, document)
    nq, nr = dimensions
    qdims = [list(dimensions)] * 2
    numbers = (np.arange(nq)[:, None] + np.arange(nr)[None, :]).ravel()
    h = system.hamiltonian_ghz
    position = system.drives["readout"]
    frequency = spectral["drive_frequency_ghz"] if frequency is None else frequency
    if frame == "rotating":
        # R = exp(i 2 pi f t N); retain only excitation-conserving interaction
        # terms and the resonant half of the real cosine carrier.
        h = np.where(numbers[:, None] == numbers[None, :], h, 0)
        h = h - frequency * np.diag(numbers) + amplitude / 2 * position
        hamiltonian = qt.Qobj(2 * np.pi * h, dims=qdims, dtype="CSR")
    else:
        hamiltonian = [qt.Qobj(2 * np.pi * h, dims=qdims, dtype="CSR")]
        if amplitude:
            hamiltonian.append(
                [
                    qt.Qobj(2 * np.pi * amplitude * position, dims=qdims, dtype="CSR"),
                    lambda t: np.cos(2 * np.pi * frequency * t),
                ]
            )
    ket = np.zeros(nq * nr, dtype=complex)
    if kind == "purcell":
        ket[nr] = 1
    else:
        ket[0], ket[nr] = 1 / np.sqrt(2), 1 / np.sqrt(2)
    if dimensions in {(4, 12), (6, 8)}:
        source_dimensions = (3, 10) if dimensions == (4, 12) else (5, 6)
        source, _ = construct(ej, source_dimensions)
        source_ket = np.zeros(np.prod(source_dimensions), dtype=complex)
        if kind == "purcell":
            source_ket[source_dimensions[1]] = 1
        else:
            source_ket[0] = source_ket[source_dimensions[1]] = 1 / np.sqrt(2)
        ket = embed_state(source, system, source_ket)
    initial = qt.Qobj(np.outer(ket, ket.conj()), dims=qdims)
    collapse = [
        qt.Qobj(np.sqrt(rate) * operator, dims=qdims, dtype="CSR")
        for operator, rate in system.dissipators.values()
    ]
    times = np.linspace(0, duration, int(round(duration / spacing)) + 1)
    solver = qt.MESolver(
        qt.QobjEvo(hamiltonian),
        collapse,
        options={
            "method": method,
            "rtol": rtol,
            "atol": rtol / 100,
            "nsteps": 100_000,
            "store_states": True,
            "normalize_output": False,
            "progress_bar": "",
        },
    )
    started = time.perf_counter()
    result = solver.run(initial, times)
    elapsed = time.perf_counter() - started
    states = np.array([state.full() for state in result.states])
    reduced = np.trace(states.reshape(-1, nq, nr, nq, nr), axis1=2, axis2=4)
    coherence = 2 * abs(reduced[:, 0, 1])
    photons = np.einsum("tij,ji->t", states, system.operators["r.number"]).real
    populations = np.diagonal(states, axis1=1, axis2=2).real.reshape(-1, nq, nr)
    record = {
        "kind": kind,
        "frame": frame,
        "ej_ghz": ej,
        "amplitude_ghz": amplitude,
        "dimensions": list(dimensions),
        "cutoff": document["model"]["subsystems"][0]["cutoff"],
        "duration_ns": duration,
        "output_spacing_ns": float(times[1] - times[0]),
        "rtol": rtol,
        "atol": rtol / 100,
        "max_steps_per_interval": 100_000,
        "wall_seconds": elapsed,
        "accepted_steps": getattr(solver._integrator, "accepted_steps", None),
        "drive_frequency_ghz": frequency,
        "coherence_at_end": float(coherence[-1]),
        "excited_population_at_end": float(reduced[-1, 1, 1].real),
        "peak_photons": float(photons.max()),
        "top_qubit_population": float(populations[:, -1, :].sum(axis=1).max()),
        "top_resonator_population": float(populations[:, :, -1].sum(axis=1).max()),
        "physicality": physical_residuals(states),
        "spectrum": spectral,
        "system_sha256": hashlib.sha256(tomli_w.dumps(document).encode()).hexdigest(),
    }
    return record, {"coherence": coherence, "excited": reduced[:, 1, 1].real, "photons": photons}


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("qutip", "numpy", "scqubits")},
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
    }


def assessment(records, comparisons):
    numerical_limit = 1e-6
    frame_limits = {"coherence": 0.003, "excited": 0.005, "photons": 0.008}
    endpoint_limit = 5e-4
    physicality = all(
        record["physicality"]["trace_error"] <= 1e-8
        and record["physicality"]["hermiticity_error"] <= 1e-8
        and record["physicality"]["minimum_eigenvalue"] >= -1e-8
        for record in records
    )
    numerical = all(
        residual <= numerical_limit
        for comparison in comparisons
        for refinement in ("rotating_refinement", "lab_refinement")
        for residual in comparison[refinement].values()
    )
    frames = all(
        residual <= frame_limits[name]
        for comparison in comparisons
        for name, residual in comparison["combined_frame_and_refinement"].items()
    )
    rows = [
        record
        for record in records
        if record["label"].endswith("-rotating") and record["kind"] == "readout"
    ]
    monotonic = all(
        np.all(np.diff([row["coherence_at_end"] for row in rows if row["ej_ghz"] == ej]) <= 0)
        for ej in (12.0, 13.5, 15.0)
    )
    by_label = {record["label"]: record for record in records}
    endpoint_errors = [
        abs(
            by_label[f"readout-{c['ej_ghz']}-{c['amplitude_ghz']}-rotating"]["coherence_at_end"]
            - by_label[f"readout-{c['ej_ghz']}-{c['amplitude_ghz']}-lab-refined"][
                "coherence_at_end"
            ]
        )
        for c in comparisons
        if c["kind"] == "readout"
    ]
    drive_response = []
    for ej in (12.0, 13.5, 15.0):
        coherence = [
            by_label[f"readout-{ej}-{amplitude}-rotating"]["coherence_at_end"]
            for amplitude in (0.0, 0.001, 0.002)
        ]
        increments = [coherence[0] - coherence[1], coherence[1] - coherence[2]]
        drive_response.append(
            {
                "ej_ghz": ej,
                "adjacent_endpoint_decreases": increments,
                "decrease_over_candidate_difference_bound": [
                    step / (2 * endpoint_limit) for step in increments
                ],
                "doubling_amplitude_loss_ratio": (coherence[0] - coherence[2]) / increments[0],
            }
        )
    return {
        "numerical_observable_atol": numerical_limit,
        "candidate_rwa_trajectory_atol": frame_limits,
        "candidate_rwa_scope": "trajectory diagnostics; endpoint coherence is assessed separately; independent-engine acceptance remains required",
        "candidate_endpoint_coherence_atol": endpoint_limit,
        "maximum_sampled_endpoint_coherence_error": max(endpoint_errors),
        "candidate_endpoint_bounds_pass": max(endpoint_errors) <= endpoint_limit,
        "endpoint_scope": "100 ns readout at five sampled reference points; amplitudes of 0.001 GHz have no lab reference; bounds do not certify resolution of every grid difference",
        "candidate_endpoint_difference_bound": 2 * endpoint_limit,
        "drive_response": drive_response,
        "physicality_pass": physicality,
        "numerical_refinement_pass": numerical,
        "candidate_frame_bounds_pass": frames,
        "coherence_decreases_with_amplitude_at_fixed_ej": monotonic,
    }


def study(output):
    """Measure the small rotating grid and full-window references at its corners."""
    records, comparisons = [], []
    traces = {}

    def measure(label, **settings):
        record, arrays = solve(**settings)
        record["label"] = label
        records.append(record)
        traces[label] = arrays
        output.write_text(json.dumps({"complete": False, "runs": records}, indent=2) + "\n")
        print(
            f"{label}: {record['wall_seconds']:.2f}s, {record['accepted_steps']} accepted steps",
            flush=True,
        )
        return record

    def difference(left, right):
        return {
            name: float(np.max(abs(traces[left][name] - traces[right][name])))
            for name in traces[left]
        }

    # Verify that step counting preserves ordinary vern9's observable trajectory.
    counter_checks = []
    for frame in ("lab", "rotating"):
        _, counted = solve(ej=13.5, amplitude=0.002, duration=2, frame=frame)
        _, normal = solve(ej=13.5, amplitude=0.002, duration=2, frame=frame, method="vern9")
        residual = max(float(np.max(abs(counted[name] - normal[name]))) for name in normal)
        if residual > 1e-10:
            raise ValueError("counted vern9 differs from ordinary vern9")
        counter_checks.append({"frame": frame, "maximum_observable_difference": residual})

    # Retain why the guide's initial allocation was not selected.
    frequency = spectrum(*construct(13.5, (3, 10)))["drive_frequency_ghz"]
    for name, dims, tolerance in (
        ("initial-production", (3, 10), 1e-10),
        ("initial-refined", (4, 12), 1e-11),
    ):
        measure(
            name,
            ej=13.5,
            amplitude=0.003,
            duration=100,
            frame="rotating",
            dimensions=dims,
            rtol=tolerance,
            frequency=frequency,
        )
    rejected_difference = difference("initial-production", "initial-refined")

    for ej in (12.0, 13.5, 15.0):
        frequency = spectrum(*construct(ej, (5, 6)))["drive_frequency_ghz"]
        for amplitude in (0.0, 0.001, 0.002):
            key = f"readout-{ej}-{amplitude}"
            settings = dict(ej=ej, amplitude=amplitude, duration=100, frequency=frequency)
            production = measure(key + "-rotating", frame="rotating", **settings)
            if ej != 13.5 and amplitude == 0.001:
                continue
            if ej == 13.5 and amplitude != 0.002:
                continue
            refined = measure(
                key + "-rotating-refined",
                frame="rotating",
                dimensions=(6, 8),
                rtol=1e-11,
                **settings,
            )
            lab = measure(key + "-lab", frame="lab", **settings)
            reference = measure(
                key + "-lab-refined", frame="lab", dimensions=(6, 8), rtol=1e-11, **settings
            )
            comparisons.append(
                {
                    "kind": "readout",
                    "ej_ghz": ej,
                    "amplitude_ghz": amplitude,
                    "rotating_refinement": difference(production["label"], refined["label"]),
                    "lab_refinement": difference(lab["label"], reference["label"]),
                    "combined_frame_and_refinement": difference(
                        production["label"], reference["label"]
                    ),
                    "chi_refinement_ghz": abs(
                        production["spectrum"]["chi_ghz"] - refined["spectrum"]["chi_ghz"]
                    ),
                    "standalone_rotating_plus_lab_reference_seconds": production["wall_seconds"]
                    + reference["wall_seconds"],
                }
            )

    # A measurable Purcell decay needs a much longer window than readout.
    settings = dict(ej=15.0, amplitude=0, duration=1600, kind="purcell")
    for frame in ("rotating", "lab"):
        for suffix, dims, tolerance in (("", (5, 6), 1e-10), ("-refined", (6, 8), 1e-11)):
            measure(
                "purcell-" + frame + suffix,
                frame=frame,
                dimensions=dims,
                rtol=tolerance,
                **settings,
            )
    comparisons.append(
        {
            "kind": "purcell",
            "ej_ghz": 15.0,
            "duration_ns": 1600,
            "rotating_refinement": difference("purcell-rotating", "purcell-rotating-refined"),
            "lab_refinement": difference("purcell-lab", "purcell-lab-refined"),
            "combined_frame_and_refinement": difference("purcell-rotating", "purcell-lab-refined"),
        }
    )
    # Isolate integration and output-grid effects at the strongest selected drive.
    settings = dict(ej=13.5, amplitude=0.002, duration=100, frame="rotating")
    tight = measure("readout-tight", rtol=1e-11, **settings)
    grid = measure("readout-half-spacing", spacing=0.5, **settings)
    base = "readout-13.5-0.002-rotating"
    integration = difference(base, tight["label"])
    output_grid = {
        name: float(np.max(abs(traces[base][name] - traces[grid["label"]][name][::2])))
        for name in traces[base]
    }
    result = {
        "complete": True,
        "environment": environment(),
        "step_count_definition": "accepted internal vern9 advances; rejected trials and dense-output interpolation excluded",
        "counter_validation": counter_checks,
        "initial_allocation_difference": rejected_difference,
        "runs": records,
        "comparisons": comparisons,
        "integration_refinement": integration,
        "output_grid_refinement": output_grid,
        "assessment": assessment(records, comparisons),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.output)


if __name__ == "__main__":
    main()
