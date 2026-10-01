"""Publish a two-level decay trajectory and its final population as a CPU job."""

import hashlib
import os
from pathlib import Path

import numpy as np

from entanglix.engines import evolve_cpu
from entanglix.models import load_system
from entanglix.runner import publish_toml

run = Path(os.environ["ENTANGLIX_RUN_DIR"])
system = load_system(os.environ["ENTANGLIX_SYSTEM_SPEC"])
excited = np.array([0, 1], dtype=complex)
projector = np.outer(excited, excited.conj())
trace = evolve_cpu(system, excited, np.linspace(0, 100, 11), observables={"excited": projector})
trajectory = run / "artifacts/trajectory.npz"
np.savez(trajectory, times_ns=trace.times_ns, excited=trace.expectations["excited"])
population = float(trace.expectations["excited"][-1])
print(f"Excited population at 100 ns: {population:.9f}")
publish_toml(
    run / "result.toml",
    {
        "engine": trace.engine,
        "solver": trace.method,
        "metrics": {"excited_population": population},
        "artifacts": {
            "trajectory": {
                "path": "artifacts/trajectory.npz",
                "sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            }
        },
    },
    replace=True,
)
