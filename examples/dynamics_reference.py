"""Compare populated oscillator and resonator loss with loss-disabled evolution."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from entanglix.engine_cuquantum import CuQuantumIntegrator
from entanglix.engine_dynamiqs import DynamiqsIntegrator
from entanglix.engines import EvolutionInput, QuTiPIntegrator
from entanglix.models import load_system

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--engine",
    choices=["qutip-cpu", "cudensitymat", "dynamiqs", "dynamiqs-cpu"],
    default="qutip-cpu",
)
args = parser.parse_args()
if args.engine == "qutip-cpu":
    engine = QuTiPIntegrator()
elif args.engine == "cudensitymat":
    engine = CuQuantumIntegrator()
else:
    engine = DynamiqsIntegrator(device_kind="cpu" if args.engine.endswith("cpu") else "gpu")
root = Path(__file__).resolve().parent
for name, duration, number in (
    ("lossy-oscillator", 50.0, "number"),
    ("qubit-resonator", 10.0, "r.number"),
):
    system = load_system(root / name / "system.toml")
    initial = np.zeros(np.prod(system.dimensions))
    initial[1] = 1  # One photon, with the qubit in its ground state when present.
    request = EvolutionInput(
        initial, np.linspace(0, duration, 11), {"photons": system.operators[number]}
    )
    lossy = engine.evolve(system, request)
    closed = engine.evolve(replace(system, dissipators={}, dissipator_factors={}), request)
    print(f"{name}: engine={lossy.engine}, method={lossy.method}, device={lossy.device}")
    print(
        f"  photons at {duration:g} ns: loss={lossy.expectations['photons'][-1].real:.8f}, loss-disabled={closed.expectations['photons'][-1].real:.8f}"
    )
    if name == "lossy-oscillator":
        print(f"  analytic exp(-0.01 * {duration:g}): {np.exp(-0.01 * duration):.8f}")
