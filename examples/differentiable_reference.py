"""Differentiate the final excited population with respect to a constant GHz drive."""

import argparse
from pathlib import Path

import jax
import numpy as np

from entanglix.engine_dynamiqs import DynamiqsIntegrator
from entanglix.engines import EvolutionInput, QuTiPIntegrator
from entanglix.models import load_system

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
args = parser.parse_args()
jax.config.update("jax_enable_x64", True)
system = load_system(Path(__file__).resolve().parent / "two-level/system.toml")
request = EvolutionInput(np.array([1, 0]), np.linspace(0, 2, 5), {"excited": np.diag([0.0, 1.0])})
engine = DynamiqsIntegrator(device_kind=args.device)


def objective(amplitude):
    return (
        engine.evolve_jax(system, request, drive_amplitudes_ghz={"x_drive": amplitude})
        .expectations["excited"][-1]
        .real
    )


amplitude = 0.1
value, derivative = jax.value_and_grad(objective)(amplitude)
epsilon = 1e-4
populations = []
for control in (amplitude + epsilon, amplitude - epsilon):
    reference = EvolutionInput(
        request.initial_state, request.times_ns, request.observables, {"x_drive": control}
    )
    populations.append(QuTiPIntegrator().evolve(system, reference).expectations["excited"][-1].real)
finite_difference = (populations[0] - populations[1]) / (2 * epsilon)
print(f"Excited population at 2 ns: {float(value):.8f}")
print(
    f"Derivative (expectation/GHz): autodiff={float(derivative):.8f}, QuTiP finite difference={finite_difference:.8f}"
)
