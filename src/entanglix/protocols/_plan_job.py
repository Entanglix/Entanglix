"""Plan computation shared by simulation and sweep workers."""

import os
from pathlib import Path

import numpy as np

from entanglix.corrector.metrics import metric_values, spectral_arrays
from entanglix.protocols.plans import load_plan
from entanglix.runner import _digest, _load_record, publish_toml


def compute(loaded):
    if loaded.definition.kind == "spectrum":
        arrays = spectral_arrays(loaded.system, loaded.definition.levels, loaded.definition.metrics)
        engine, method, diagnostics = (
            "qutip-cpu",
            ("numpy-eigh" if loaded.definition.metrics else "eigenenergies"),
            {},
        )
    else:
        if loaded.definition.engine == "cudensitymat":
            from entanglix.engines.cudensitymat import CuQuantumIntegrator

            integrator = CuQuantumIntegrator()
        elif loaded.definition.engine == "dynamiqs":
            from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

            integrator = DynamiqsIntegrator(
                device_kind="gpu" if loaded.definition.device == "cuda" else "cpu"
            )
        elif loaded.definition.engine == "qutip-cpu":
            from entanglix.engines import QuTiPIntegrator

            integrator = QuTiPIntegrator()
        else:
            from entanglix.engines.selection import load_engine

            integrator, identity = load_engine(loaded.definition.engine)
            launch = _load_record(Path(os.environ["ENTANGLIX_RUN_DIR"]) / "run.toml")["launch"]
            if identity["record_sha256"] != launch["selection"].get("record_sha256"):
                raise ValueError("engine installation changed after launch validation")
        result = integrator.evolve(loaded.system, loaded.request)
        arrays = {"times_ns": result.times_ns, "states": np.asarray(result.states)}
        arrays.update({f"observable.{name}": value for name, value in result.expectations.items()})
        engine, method, diagnostics = result.engine, result.method, result.diagnostics
    return arrays, {"engine": engine, "method": method, "diagnostics": diagnostics}


def execute(plan: Path) -> None:
    run = Path(os.environ["ENTANGLIX_RUN_DIR"])
    launch = _load_record(run / "run.toml")["launch"]
    loaded = load_plan(run / "spec/system.toml", plan)
    if loaded.digest != launch["plan_sha256"]:
        raise ValueError("frozen execution plan digest changed")
    arrays, details = compute(loaded)
    engine, method, diagnostics = details["engine"], details["method"], details["diagnostics"]
    output = run / "artifacts/output.npz"
    np.savez(output, **arrays)
    artifacts = {"output": {"path": "artifacts/output.npz", "sha256": _digest(output.read_bytes())}}
    summary = {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    if loaded.request is None:
        axes.plot(np.arange(len(arrays["energies_ghz"])), arrays["energies_ghz"], "o-")
        axes.set(xlabel="Level", ylabel="Energy (GHz)")
    else:
        for name in loaded.request.observables:
            values = arrays[f"observable.{name}"]
            summary[name] = {
                "initial_real": float(values[0].real),
                "final_real": float(values[-1].real),
                "maximum_magnitude": float(abs(values).max()),
            }
            axes.plot(arrays["times_ns"], values.real, label=f"{name} (real)")
            if np.max(abs(values.imag)) > 1e-12:
                axes.plot(arrays["times_ns"], values.imag, "--", label=f"{name} (imaginary)")
        axes.set(xlabel="Time (ns)", ylabel="Declared observable expectation")
        if loaded.request.observables:
            axes.legend()
    frame = diagnostics.get("frame", {}).get("kind", "lab")
    axes.set_title(f"{engine} / {loaded.definition.device}; {frame} frame")
    plot = run / "artifacts/observables.png"
    figure.savefig(plot)
    plt.close(figure)
    artifacts["plot"] = {
        "path": "artifacts/observables.png",
        "sha256": _digest(plot.read_bytes()),
    }
    publish_toml(
        run / "result.toml",
        {
            "kind": loaded.definition.kind,
            "engine": engine,
            "method": method,
            "device": loaded.definition.device,
            "dimensions": list(loaded.system.dimensions),
            "plan_sha256": loaded.digest,
            "metrics": metric_values(loaded.definition.metrics, arrays),
            "summary": summary,
            "diagnostics": diagnostics,
            "artifacts": artifacts,
        },
        replace=True,
    )
