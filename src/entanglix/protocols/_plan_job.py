"""Preflight and execution of a built-in CPU plan behind the runner start barrier."""

import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

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
        "purpose": loaded.definition.purpose,
        "plan_sha256": loaded.digest,
        "policy_version": loaded.version,
        "environment": capture_environment(Path(repository) / "uv.lock", extras=tuple(extras)),
    }


def execute(plan: Path) -> None:
    run = Path(os.environ["ENTANGLIX_RUN_DIR"])
    launch = _load_record(run / "run.toml")["launch"]
    loaded = load_plan(run / "spec/system.toml", plan, expected_policy=launch["policy_version"])
    if loaded.digest != launch["plan_sha256"]:
        raise ValueError("frozen execution plan digest changed")
    if loaded.definition.kind == "spectrum":
        arrays = {"energies_ghz": loaded.system.qutip_hamiltonian().eigenenergies()}
        engine, method, diagnostics = "qutip-cpu", "eigenenergies", {}
    else:
        if loaded.definition.engine == "dynamiqs":
            from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

            integrator = DynamiqsIntegrator(device_kind="cpu")
        else:
            from entanglix.engines import QuTiPIntegrator

            integrator = QuTiPIntegrator()
        result = integrator.evolve(loaded.system, loaded.request)
        arrays = {"times_ns": result.times_ns, "states": np.asarray(result.states)}
        arrays.update({f"observable.{name}": value for name, value in result.expectations.items()})
        engine, method, diagnostics = result.engine, result.method, result.diagnostics
    output = run / "artifacts/output.npz"
    np.savez(output, **arrays)
    publish_toml(
        run / "result.toml",
        {
            "kind": loaded.definition.kind,
            "engine": engine,
            "method": method,
            "device": "cpu",
            "dimensions": list(loaded.system.dimensions),
            "plan_sha256": loaded.digest,
            "policy_version": loaded.version,
            "metrics": {},
            "diagnostics": diagnostics,
            "artifacts": {
                "output": {"path": "artifacts/output.npz", "sha256": _digest(output.read_bytes())}
            },
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
        print(f"plan rejected: {exc}", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
