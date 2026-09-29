"""Run two-level decay and inspect CPU circuit/composite spectra."""

from pathlib import Path

import numpy as np

from entanglix.engines import evolve_cpu
from entanglix.models import load_system

EXAMPLES = Path(__file__).resolve().parent

two_level = load_system(EXAMPLES / "two-level/system.toml")
excited = np.array([0, 1], dtype=complex)
projector = np.outer(excited, excited.conj())
trace = evolve_cpu(two_level, excited, np.linspace(0, 100, 11), observables={"excited": projector})
print(f"Excited population at 100 ns: {trace.expectations['excited'][-1]:.6f}")
print(f"Analytic exp(-1): {np.exp(-1):.6f}")

junction = load_system(EXAMPLES / "junction-shunt/system.toml")
energies = np.diag(junction.hamiltonian_ghz).real
print("Junction excitation energies (GHz):", energies[1:4] - energies[0])

coupled = load_system(EXAMPLES / "qubit-resonator/system.toml")
energies = np.linalg.eigvalsh(coupled.hamiltonian_ghz)
print("Qubit–resonator dimensions:", coupled.dimensions)
print("Coupled excitation energies (GHz):", energies[1:4] - energies[0])
