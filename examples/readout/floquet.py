"""Record closed-system readout quasienergies and local numerical evidence."""

import hashlib
import os
from pathlib import Path

import numpy as np
import qutip as qt

from entanglix.models import load_system
from entanglix.runner import publish_toml

FREQUENCY_GHZ = 6.9
AMPLITUDES_GHZ = np.array([0.0, 0.005, 0.01, 0.02])
LABELS = ((0, 0), (1, 0))
MIN_OVERLAP = 0.9
ENERGY_LIMIT_GHZ = 1e-8
UNITARITY_LIMIT = 1e-8


def folded(value):
    """Choose the representative in [-f/2, f/2), in GHz."""
    return (value + FREQUENCY_GHZ / 2) % FREQUENCY_GHZ - FREQUENCY_GHZ / 2


def assign(weights, context):
    """Assign distinct modes with dominant squared overlaps."""
    indices = np.argmax(weights, axis=1)
    ordered = np.sort(weights, axis=1)
    overlaps = ordered[:, -1]
    margins = overlaps - ordered[:, -2]
    if len(set(indices)) != len(indices) or np.min(overlaps) < MIN_OVERLAP:
        raise ValueError(
            f"{context}: labels {LABELS} matched mode indices {indices.tolist()} "
            f"with squared overlaps {overlaps.tolist()}; require distinct modes "
            f"and overlaps >= {MIN_OVERLAP}"
        )
    return indices, overlaps, margins


def main():
    run = Path(os.environ["ENTANGLIX_RUN_DIR"])
    system = load_system(os.environ["ENTANGLIX_SYSTEM_SPEC"])
    nq, nr = system.dimensions
    energies, vectors = np.linalg.eigh(system.hamiltonian_ghz)
    static_indices, static_overlaps, _ = assign(
        abs(vectors[[q * nr + r for q, r in LABELS], :]) ** 2, "static dressed-state assignment"
    )
    dressed = vectors[:, static_indices]
    period = 1 / FREQUENCY_GHZ
    phase_times = np.linspace(0, period, 32, endpoint=False)
    h0 = qt.Qobj(2 * np.pi * system.hamiltonian_ghz, dims=[list(system.dimensions)] * 2)
    drive = qt.Qobj(2 * np.pi * system.drives["readout"], dims=h0.dims)
    spectra, selected, overlaps, margins = [], [], [], []
    qubit_populations, resonator_populations, unitarity, energy_changes = [], [], [], []
    zero_error = None
    for amplitude in AMPLITUDES_GHZ:

        def carrier(t, amplitude=float(amplitude)):
            return amplitude * np.cos(2 * np.pi * FREQUENCY_GHZ * t)

        hamiltonian = qt.QobjEvo([h0, [drive, carrier]])
        bases = [
            qt.FloquetBasis(
                hamiltonian,
                period,
                options={"method": "vern9", "rtol": rtol, "atol": atol, "nsteps": 100_000},
                precompute=phase_times,
            )
            for rtol, atol in ((1e-10, 1e-12), (1e-11, 1e-13))
        ]
        basis, refined = bases
        modes = basis.mode(0, data=True).to_array()
        context = f"Floquet assignment at A={amplitude:g} GHz, f={FREQUENCY_GHZ:g} GHz"
        indices, weights, gaps = assign(abs(dressed.conj().T @ modes) ** 2, context)
        refined_indices, _, _ = assign(
            abs(dressed.conj().T @ refined.mode(0, data=True).to_array()) ** 2,
            f"{context} (tightened tolerances)",
        )
        quasienergies = basis.e_quasi / (2 * np.pi)
        tightened = refined.e_quasi / (2 * np.pi)
        change = float(np.max(abs(folded(tightened[refined_indices] - quasienergies[indices]))))
        propagator = basis.U(period).full()
        unitary_error = float(
            np.linalg.norm(propagator.conj().T @ propagator - np.eye(nq * nr), ord="fro")
        )
        if change > ENERGY_LIMIT_GHZ:
            raise ValueError(
                f"A={amplitude:g} GHz: labelled quasienergy refinement residual "
                f"{change:.6g} GHz exceeds {ENERGY_LIMIT_GHZ:.6g} GHz"
            )
        if unitary_error > UNITARITY_LIMIT:
            raise ValueError(
                f"A={amplitude:g} GHz: one-period unitarity Frobenius residual "
                f"{unitary_error:.6g} exceeds {UNITARITY_LIMIT:.6g}"
            )
        if amplitude == 0:
            # Compare every static state modulo f; sorted zone positions are not identities.
            zero_indices = np.argmax(abs(vectors.conj().T @ modes) ** 2, axis=1)
            if len(set(zero_indices)) != len(energies):
                raise ValueError(
                    f"zero-drive static assignment repeats mode indices: {zero_indices.tolist()}"
                )
            zero_error = float(np.max(abs(folded(quasienergies[zero_indices] - energies))))
            if zero_error > ENERGY_LIMIT_GHZ:
                raise ValueError(
                    f"zero-drive static spectrum residual {zero_error:.6g} GHz "
                    f"exceeds {ENERGY_LIMIT_GHZ:.6g} GHz"
                )
        top_qubit, top_resonator = [], []
        for t in phase_times:
            labelled = (
                basis.mode(float(t), data=True).to_array()[:, indices].reshape(nq, nr, len(LABELS))
            )
            top_qubit.append(np.sum(abs(labelled[-1, :, :]) ** 2, axis=0))
            top_resonator.append(np.sum(abs(labelled[:, -1, :]) ** 2, axis=0))
        spectra.append(quasienergies)
        selected.append(indices)
        overlaps.append(weights)
        margins.append(gaps)
        qubit_populations.append(np.max(top_qubit, axis=0))
        resonator_populations.append(np.max(top_resonator, axis=0))
        unitarity.append(unitary_error)
        energy_changes.append(change)

    labelled_energies = np.take_along_axis(np.array(spectra), np.array(selected), axis=1)
    transitions = labelled_energies[:, 1] - labelled_energies[:, 0]
    shifts = folded(transitions - transitions[0])
    artifact = run / "artifacts/quasienergies.npz"
    np.savez(
        artifact,
        amplitudes_ghz=AMPLITUDES_GHZ,
        frequency_ghz=FREQUENCY_GHZ,
        quasienergies_ghz=spectra,
        labels=LABELS,
        label_indices=selected,
        label_overlaps=overlaps,
        label_margins=margins,
        phase_times_ns=phase_times,
        max_top_qubit_populations=qubit_populations,
        max_top_resonator_populations=resonator_populations,
        qubit_transition_shifts_ghz=shifts,
    )
    publish_toml(
        run / "result.toml",
        {
            "engine": "qutip-cpu",
            "solver": "FloquetBasis/vern9",
            "dimensions": list(system.dimensions),
            "settings": {
                "frequency_ghz": FREQUENCY_GHZ,
                "amplitudes_ghz": AMPLITUDES_GHZ.tolist(),
                "period_ns": period,
                "phase_rad": 0.0,
                "rtol": 1e-10,
                "atol": 1e-12,
                "refined_rtol": 1e-11,
                "refined_atol": 1e-13,
                "max_steps": 100_000,
                "omitted_dissipator_rates_per_ns": {
                    name: rate for name, (_, rate) in system.dissipators.items()
                },
            },
            "metrics": {
                "qubit_shift_r0_ghz": shifts.tolist(),
            },
            "numerical_evidence": {
                "zero_drive_error_ghz": zero_error,
                "labelled_refinement_errors_ghz": energy_changes,
                "unitarity_errors_frobenius": unitarity,
                "min_static_overlap": float(min(static_overlaps)),
                "min_label_overlaps": np.min(overlaps, axis=1).tolist(),
                "max_top_qubit_populations": np.max(qubit_populations, axis=1).tolist(),
                "max_top_resonator_populations": np.max(resonator_populations, axis=1).tolist(),
                "phase_sample_count": len(phase_times),
                "min_overlap_required": MIN_OVERLAP,
                "energy_limit_ghz": ENERGY_LIMIT_GHZ,
                "unitarity_limit_frobenius": UNITARITY_LIMIT,
            },
            "artifacts": {
                "quasienergies": {
                    "path": "artifacts/quasienergies.npz",
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            },
        },
        replace=True,
    )
    print("Qubit-transition shifts (GHz), r=0:")
    print(shifts)


if __name__ == "__main__":
    main()
