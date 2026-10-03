"""Preflight and execution of a built-in CPU plan behind the runner start barrier."""

import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from entanglix.corrector.metrics import metric_values
from entanglix.protocols.plans import load_plan
from entanglix.runner import _digest, _load_record, capture_environment, publish_toml


def gate(system: str, plan: str, repository: str, extras: list[str]) -> dict:
    loaded = load_plan(Path(system), Path(plan))
    # Availability is checked in the exact environment that will execute the job.
    if loaded.definition.engine == "dynamiqs":
        try:
            version("dynamiqs")
        except PackageNotFoundError as exc:
            raise ValueError(
                'configure runner extras with "differentiable" to use CPU dynamiqs'
            ) from exc
    return {
        "system_id": loaded.system.id,
        "inputs": {name: str(path) for name, path in loaded.inputs.items()},
        "spec_sha256": loaded.input_digests,
        "engine": loaded.definition.engine,
        "kind": loaded.definition.kind,
        "plan_sha256": loaded.digest,
        "environment": capture_environment(Path(repository) / "uv.lock", extras=tuple(extras)),
    }


def execute(plan: Path) -> None:
    run = Path(os.environ["ENTANGLIX_RUN_DIR"])
    launch = _load_record(run / "run.toml")["launch"]
    loaded = load_plan(run / "spec/system.toml", plan)
    if loaded.digest != launch["plan_sha256"]:
        raise ValueError("frozen execution plan digest changed")
    if loaded.definition.kind == "spectrum":
        arrays = {
            "energies_ghz": loaded.system.qutip_hamiltonian().eigenenergies()[
                : loaded.definition.levels
            ]
        }
        engine, method, diagnostics = "qutip-cpu", "eigenenergies", {}
    else:
        if loaded.definition.engine == "cudensitymat":
            from entanglix.engines.cudensitymat import CuQuantumIntegrator

            integrator = CuQuantumIntegrator()
        elif loaded.definition.engine == "dynamiqs":
            from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

            integrator = DynamiqsIntegrator(
                device_kind="gpu" if loaded.definition.device == "cuda" else "cpu"
            )
        else:
            from entanglix.engines import QuTiPIntegrator

            integrator = QuTiPIntegrator()
        result = integrator.evolve(loaded.system, loaded.request)
        arrays = {"times_ns": result.times_ns, "states": np.asarray(result.states)}
        arrays.update({f"observable.{name}": value for name, value in result.expectations.items()})
        engine, method, diagnostics = result.engine, result.method, result.diagnostics
    output = run / "artifacts/output.npz"
    np.savez(output, **arrays)
    artifacts = {"output": {"path": "artifacts/output.npz", "sha256": _digest(output.read_bytes())}}
    summary = {}
    if _load_record(run / "run.toml")["protocol"] == "simulate":
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


def main() -> int:
    try:
        if sys.argv[1] == "check":
            print(json.dumps(gate(**json.load(sys.stdin))))
        else:
            plan = Path(sys.argv[2])
            os.fstat(int(sys.argv[3]))
            if sys.stdin.buffer.read(1) != b"1":
                return 64
            execute(plan)
    except Exception as exc:
        from entanglix.runner import failure_message

        print(f"plan rejected: {failure_message(exc)}", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
