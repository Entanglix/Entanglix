"""One public simulation, compiled to a frozen plan with checks available on request."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import tomli_w
from pydantic import Field

from entanglix.corrector.dynamics import check_listing
from entanglix.corrector.metrics import EndpointMetric
from entanglix.engines.selection import select_engine
from entanglix.models import StrictModel, load_system
from entanglix.protocols.plans import EvolutionSpec, Refinement, load_plan
from entanglix.runner import (
    _cpu_job_context,
    _digest,
    _execute_job,
    _load_record,
    _read_attempt,
    _system_inputs,
    acquire_gpu,
    capture_code,
    capture_environment,
    check_gpu_environment,
    create_run,
    process_failure,
    record_finish,
)


class Needs(StrictModel):
    gradients: bool = Field(strict=True, default=False)
    device: Literal["cpu", "cuda"] = "cpu"


class OutputGrid(StrictModel):
    start: float = Field(allow_inf_nan=False)
    stop: float = Field(allow_inf_nan=False)
    count: int = Field(strict=True, ge=2)


class SimulationEvolution(EvolutionSpec):
    array_file: str | None = None
    initial_state: str | list[Annotated[int, Field(strict=True, ge=0)]]
    times_ns: list[float] | OutputGrid


class SimulationSpec(StrictModel):
    format_version: Literal[1]
    system: str
    parameters: dict[str, float] = Field(default_factory=dict)
    kind: Literal["lindblad", "spectrum"] = "lindblad"
    needs: Needs = Field(default_factory=Needs)
    levels: int | None = None
    evolution: SimulationEvolution | None = None
    refinement: dict[str, Refinement] = Field(default_factory=dict)
    metrics: dict[str, EndpointMetric] = Field(default_factory=dict)
    segment_tolerances: dict[str, float] = Field(default_factory=dict)
    timeout_seconds: float | None = None


def _adjacent(root, name):
    path = root / name
    if Path(name).name != name or path.resolve().parent != root.resolve():
        raise ValueError("protocol input files must stay beside the protocol spec")
    return path


def compile_simulation(protocol: Path, directory: Path, extras: list[str], repository: Path):
    """Compile and check in the exactly synced job environment, before admission."""
    content = protocol.read_bytes()
    spec = SimulationSpec.model_validate(tomllib.loads(content.decode()))
    if spec.timeout_seconds is not None and (
        not math.isfinite(spec.timeout_seconds) or spec.timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    source = _adjacent(protocol.parent, spec.system)
    system = load_system(source)
    inputs, hashes = _system_inputs(source, system, reserved={"plan.toml", "simulate.toml"})
    for name, path in inputs.items():
        shutil.copyfile(path, directory / name)
        if _digest((directory / name).read_bytes()) != hashes[name]:
            raise ValueError(f"system input changed during compilation: {name}")
    document = tomllib.loads((directory / "system.toml").read_text())
    for name, value in spec.parameters.items():
        if name not in document.get("parameters", {}) or not math.isfinite(value):
            raise ValueError(f"parameter override must name a finite declared parameter: {name}")
        document["parameters"][name]["value"] = value
    if spec.parameters:
        (directory / "system.toml").write_text(tomli_w.dumps(document))
        system = load_system(directory / "system.toml")
    definition = {
        "format_version": 1,
        "kind": spec.kind,
        "engine": "qutip-cpu",
        "refinement": {name: step.model_dump() for name, step in spec.refinement.items()},
        "metrics": {
            name: metric.model_dump(exclude_none=True) for name, metric in spec.metrics.items()
        },
        "segment_tolerances": spec.segment_tolerances,
    }
    if spec.levels is not None:
        definition["levels"] = spec.levels
    if spec.evolution is not None:
        evolution = spec.evolution.model_dump(exclude_none=True)
        if isinstance(spec.evolution.times_ns, OutputGrid):
            grid = spec.evolution.times_ns
            if grid.stop <= grid.start:
                raise ValueError("output grid stop must exceed start")
            evolution["times_ns"] = np.linspace(grid.start, grid.stop, grid.count).tolist()
        initial = spec.evolution.initial_state
        if isinstance(initial, list):
            if spec.evolution.array_file is not None:
                raise ValueError("a basis initial_state must not declare array_file")
            if len(initial) != len(system.dimensions) or any(
                type(index) is not int or not 0 <= index < size
                for index, size in zip(initial, system.dimensions)
            ):
                raise ValueError("initial_state must contain one valid level per subsystem")
            name = "initial-state.npz"
            if name in inputs:
                raise ValueError("initial-state.npz is reserved for a generated basis state")
            state = np.zeros(math.prod(system.dimensions), dtype=complex)
            state[np.ravel_multi_index(tuple(initial), system.dimensions)] = 1
            np.savez(directory / name, initial_state=state)
            evolution.update(array_file=name, initial_state="initial_state")
        else:
            name = spec.evolution.array_file
            if name is None or name in {*inputs, "simulate.toml", "plan.toml"}:
                raise ValueError("an array initial_state requires a distinct adjacent array_file")
            shutil.copyfile(_adjacent(protocol.parent, name), directory / name)
        definition["evolution"] = evolution
    plan = directory / "plan.toml"
    plan.write_text(tomli_w.dumps(definition))
    loaded = load_plan(directory / "system.toml", plan)
    if spec.kind == "spectrum":
        if spec.needs.device != "cpu" or spec.needs.gradients:
            raise ValueError("spectrum requires device cpu without gradients")
        selection = None
    else:
        available = {"qutip-cpu"}
        for engine, package in (("dynamiqs", "dynamiqs"), ("cudensitymat", "qutip-cuquantum")):
            try:
                version(package)
                available.add(engine)
            except PackageNotFoundError:
                pass
        selection = select_engine(
            loaded.system,
            loaded.request,
            available_engines=available,
            needs_gradients=spec.needs.gradients,
            device=spec.needs.device,
        )
        definition.update(engine=selection.engine, device=spec.needs.device)
        plan.write_text(tomli_w.dumps(definition))
        loaded = load_plan(directory / "system.toml", plan)
    canonical = spec.model_dump(exclude_none=True)
    canonical["system"] = "system.toml"
    (directory / "simulate.toml").write_text(tomli_w.dumps(canonical))
    gpu_check = None
    if spec.needs.device == "cuda":
        try:
            import cupy as cp
        except ImportError as exc:
            raise ValueError(
                "GPU environment rejected: install the cuquantum extra for CuPy"
            ) from exc

        if cp.cuda.runtime.getDeviceCount() != 1:
            raise ValueError("CUDA simulation requires exactly one visible GPU")
        properties = cp.cuda.runtime.getDeviceProperties(0)
        device = f"GPU-{uuid.UUID(bytes=properties['uuid'])}"
        gpu_check = check_gpu_environment(device)
    paths = {**loaded.inputs, "simulate.toml": directory / "simulate.toml"}
    selection_details = (
        {key: value for key, value in selection.__dict__.items() if value is not None}
        if selection
        else {}
    )
    return {
        "preview": check_listing(
            loaded,
            timeout_seconds=spec.timeout_seconds,
            selection=selection_details,
        ),
        "system_id": loaded.system.id,
        "inputs": {name: str(path) for name, path in paths.items()},
        "spec_sha256": {name: _digest(path.read_bytes()) for name, path in paths.items()},
        "engine": loaded.definition.engine,
        "device": spec.needs.device,
        "needs": canonical["needs"],
        "kind": loaded.definition.kind,
        "plan_sha256": loaded.digest,
        "selection": selection_details,
        "gpu_check": gpu_check,
        "timeout_seconds": spec.timeout_seconds,
        "protocol_sha256": _digest(content),
        "environment": capture_environment(
            repository / "uv.lock",
            extras=tuple(extras),
            gpu=gpu_check["gpu"] if gpu_check else None,
        ),
    }


def simulate(
    protocol: Path,
    *,
    runs_root: Path = Path("research_runs"),
    repository: Path | None = None,
    parent_run_ids: tuple[str, ...] = (),
    dry_run: bool = False,
) -> dict:
    """Launch one simulation without unrequested scientific checks."""
    repository = (repository or Path(__file__).resolve().parents[3]).resolve()
    command, environment, extras = _cpu_job_context(repository, "entanglix.protocols._simulate_job")
    gate_environment = environment.copy()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gate_environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        gate_environment.pop("CUDA_VISIBLE_DEVICES", None)
    with tempfile.TemporaryDirectory(prefix="entanglix-simulate-") as scratch:
        check = subprocess.run(
            [*command, "check"],
            input=json.dumps(
                {
                    "protocol": str(protocol.resolve()),
                    "directory": scratch,
                    "repository": str(repository),
                    "extras": extras,
                }
            ),
            env=gate_environment,
            text=True,
            capture_output=True,
        )
        if check.returncode:
            raise ValueError(process_failure(check, "simulation launch rejected"))
        gate = json.loads(check.stdout)
        if dry_run:
            return {"exit_code": 0, "preview": gate["preview"]}
        print(json.dumps({"preview": gate["preview"]}, indent=2), file=sys.stderr, flush=True)
        launch = {key: gate[key] for key in ("kind", "plan_sha256", "selection", "protocol_sha256")}
        if gate["timeout_seconds"] is not None:
            launch["timeout_seconds"] = gate["timeout_seconds"]
        with create_run(
            system_id=gate["system_id"],
            protocol="simulate",
            inputs={name: Path(path) for name, path in gate["inputs"].items()},
            environment=gate["environment"],
            code=capture_code(repository),
            engine=gate["engine"],
            device=gate["device"],
            needs=gate["needs"],
            runs_root=runs_root.resolve(),
            parent_run_ids=parent_run_ids,
            launch=launch,
        ) as owner:
            run = owner.path
            gpu = None
            try:
                if _load_record(run / "run.toml")["spec_sha256"] != gate["spec_sha256"]:
                    raise ValueError("inputs changed after simulation validation")
                if gate["device"] == "cuda":
                    gpu = acquire_gpu(gate["gpu_check"]["device_uuid"], owner)
                    environment.update(
                        CUDA_VISIBLE_DEVICES=gpu.device,
                        JAX_PLATFORMS="cuda,cpu",
                        XLA_PYTHON_CLIENT_PREALLOCATE="false",
                        ENTANGLIX_GPU_LOCK_FD=str(gpu.fileno()),
                    )
                environment.update(ENTANGLIX_RUN_DIR=str(run))
                result = _execute_job(
                    owner,
                    command,
                    environment,
                    timeout_seconds=gate["timeout_seconds"],
                    gpu=gpu,
                    job_input="plan.toml",
                )
            except (Exception, KeyboardInterrupt) as exc:
                status, exit_code = (
                    ("aborted", 130) if isinstance(exc, KeyboardInterrupt) else ("failed", 64)
                )
                if not (run / "FINISHED").exists():
                    (run / "run.log").write_text(f"simulation launch failed: {exc}\n")
                    record_finish(
                        owner, status=status, exit_code=exit_code, wall_seconds=0, gpu_seconds=0
                    )
                result = {"path": str(run), "exit_code": exit_code, "status": status}
            finally:
                if gpu is not None:
                    gpu.close()
    return result


def rerun_simulation(run: Path, *, repository: Path | None = None, runs_root: Path | None = None):
    record = _read_attempt(run)
    if record["manifest"]["protocol"] != "simulate" or record["status"] in {"active", "unknown"}:
        raise ValueError("rerun requires an inactive simulation attempt")
    run = Path(record["path"])
    repository = (repository or Path(__file__).resolve().parents[3]).resolve()
    for name, digest in record["manifest"]["spec_sha256"].items():
        if _digest((run / "spec" / name).read_bytes()) != digest:
            raise ValueError(f"frozen input digest mismatch: {name}")
    result = simulate(
        run / "spec/simulate.toml",
        repository=repository,
        runs_root=runs_root or run.parent.parent,
        parent_run_ids=(run.name,),
    )
    replay = _read_attempt(Path(result["path"]))["manifest"]
    result["replay"] = {
        "code_matches": replay["code"] == record["manifest"]["code"],
        "environment_matches": replay["env_hash"] == record["manifest"]["env_hash"],
    }
    return result
