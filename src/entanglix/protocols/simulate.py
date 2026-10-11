"""One public simulation, compiled to a frozen plan with checks available on request."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import tomli_w
from pydantic import Field

from entanglix.corrector.dynamics import check_listing
from entanglix.corrector.metrics import Metric
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
    select_gpu,
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
    metrics: dict[str, Metric] = Field(default_factory=dict)
    segment_tolerances: dict[str, float] = Field(default_factory=dict)
    timeout_seconds: float | None = None


def _adjacent(root, name):
    path = root / name
    if Path(name).name != name or path.resolve().parent != root.resolve():
        raise ValueError("protocol input files must stay beside the protocol spec")
    return path


def compile_simulation(
    protocol: Path,
    directory: Path,
    extras: list[str],
    repository: Path,
    *,
    probe_device=True,
    values=None,
    retain_base=False,
):
    """Compile and check in the exactly synced job environment, before admission."""
    content = protocol.read_bytes()
    raw = tomllib.loads(content.decode())
    original = SimulationSpec.model_validate(raw)
    if values:
        from entanglix.protocols.sweep import _set_target

        for target, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"varied value must be finite: {target}")
            _set_target(raw, target, value)
    spec = SimulationSpec.model_validate(raw)
    if spec.timeout_seconds is not None and (
        not math.isfinite(spec.timeout_seconds) or spec.timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    source = _adjacent(protocol.parent, spec.system)
    system = load_system(source)
    inputs, hashes = _system_inputs(source, system, reserved={"plan.toml", "simulate.toml"})
    base_inputs = {}
    base_paths = {}
    if retain_base:
        originals = {**inputs, "simulate.toml": protocol}
        if original.evolution is not None and original.evolution.array_file is not None:
            name = original.evolution.array_file
            originals[name] = _adjacent(protocol.parent, name)
        base_directory = directory / "base"
        base_directory.mkdir()
        for name, path in originals.items():
            frozen = base_directory / name
            data = content if name == "simulate.toml" else path.read_bytes()
            if name in hashes and _digest(data) != hashes[name]:
                raise ValueError(f"system input changed during compilation: {name}")
            frozen.write_bytes(data)
            base_inputs[name] = _digest(data)
            base_paths[f"base/{name}"] = frozen
        if spec.system != "system.toml":
            alias = base_directory / spec.system
            shutil.copyfile(base_directory / "system.toml", alias)
            base_paths[f"base/{spec.system}"] = alias
        inputs = {name: base_directory / name for name in inputs}
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
            shutil.copyfile(
                directory / "base" / name if retain_base else _adjacent(protocol.parent, name),
                directory / name,
            )
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
    if spec.needs.device == "cuda" and probe_device:
        descriptor = os.environ.get("ENTANGLIX_GPU_LOCK_FD")
        if descriptor is None:
            raise ValueError("GPU launch check requires the inherited device lock")
        os.fstat(int(descriptor))
        gpu_check = check_gpu_environment(os.environ["CUDA_VISIBLE_DEVICES"])
    paths = {**loaded.inputs, "simulate.toml": directory / "simulate.toml", **base_paths}
    selection_details = (
        {key: value for key, value in selection.__dict__.items() if value is not None}
        if selection
        else {}
    )
    return {
        "base_inputs": base_inputs,
        "scope_values": [values or {}],
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
    investigation_id: str | None = None,
    base: str | None = None,
    values: dict[str, float] | None = None,
) -> dict:
    """Launch one simulation without unrequested scientific checks."""
    return _launch_protocol(
        protocol,
        runs_root=runs_root,
        repository=repository,
        parent_run_ids=parent_run_ids,
        dry_run=dry_run,
        operation="simulate",
        investigation_id=investigation_id,
        base=base,
        values=values,
    )


def _launch_protocol(
    protocol,
    *,
    runs_root,
    repository,
    parent_run_ids,
    dry_run,
    operation,
    investigation_id=None,
    base=None,
    values=None,
):
    if investigation_id is None and (base is not None or values is not None):
        raise ValueError("base and varied values require an investigation")
    if investigation_id is not None and base is None:
        raise ValueError("an investigation launch requires its base name")
    repository = (repository or Path(__file__).resolve().parents[3]).resolve()
    command, environment, extras = _cpu_job_context(
        repository, f"entanglix.protocols._{operation}_job"
    )
    try:
        source = protocol
        if operation == "sweep":
            from entanglix.protocols.sweep import SweepSpec

            spec = SweepSpec.model_validate(_load_record(protocol))
            source = _adjacent(protocol.parent, spec.base)
        document = _load_record(source)
        device = (
            "cpu"
            if operation == "control"
            else Needs.model_validate(document.get("needs", {})).device
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{operation} rejected: {exc}") from exc
    with acquire_gpu(select_gpu()) if device == "cuda" else nullcontext() as gpu:
        gate_environment = environment.copy()
        if gpu is not None:
            gate_environment.update(
                CUDA_VISIBLE_DEVICES=gpu.device, ENTANGLIX_GPU_LOCK_FD=str(gpu.fileno())
            )
        with tempfile.TemporaryDirectory(prefix=f"entanglix-{operation}-") as scratch:
            check = subprocess.run(
                [*command, "check"],
                input=json.dumps(
                    {
                        "protocol": str(protocol.resolve()),
                        "directory": scratch,
                        "repository": str(repository),
                        "extras": extras,
                        "retain_base": investigation_id is not None,
                        **({"values": values} if operation == "simulate" else {}),
                    }
                ),
                env=gate_environment,
                text=True,
                capture_output=True,
                pass_fds=(gpu.fileno(),) if gpu is not None else (),
            )
            if check.returncode:
                raise ValueError(process_failure(check, f"{operation} launch rejected"))
            gate = json.loads(check.stdout)
            if gate["device"] != device:
                raise ValueError("device request changed during launch validation")
            if dry_run and investigation_id is not None:
                from entanglix.investigations import admit, investigation_path
                from entanglix.runner import _scope_defaults

                path = investigation_path(runs_root, investigation_id)
                with admit(
                    path,
                    operation=operation,
                    base_name=base,
                    hashes=gate["base_inputs"],
                    points=gate["scope_values"],
                    defaults=_scope_defaults(
                        {name: Path(path) for name, path in gate["inputs"].items()}
                    ),
                    timeout_seconds=gate["timeout_seconds"],
                ):
                    pass
            if dry_run:
                return {"exit_code": 0, "preview": gate["preview"]}
            print(json.dumps({"preview": gate["preview"]}, indent=2), file=sys.stderr, flush=True)
            launch = {
                key: gate[key]
                for key in (
                    "kind",
                    "selection",
                    "protocol_sha256",
                    "problem_sha256" if operation == "control" else "plan_sha256",
                )
            }
            if operation == "control":
                launch["control_model"] = {
                    key: gate["preview"][key]
                    for key in (
                        "omitted_dissipators",
                        "channel_order",
                        "frame",
                        "energy_origin_ghz",
                    )
                }
            if investigation_id is not None:
                launch["varied_values"] = values or {}
            if gate["timeout_seconds"] is not None:
                launch["timeout_seconds"] = gate["timeout_seconds"]
            with create_run(
                system_id=gate["system_id"],
                protocol=operation,
                investigation_id=investigation_id,
                base=base,
                base_inputs=gate["base_inputs"] if investigation_id is not None else None,
                values=gate["scope_values"] if investigation_id is not None else None,
                inputs={name: Path(path) for name, path in gate["inputs"].items()},
                environment=gate["environment"],
                code=capture_code(repository),
                engine=gate["engine"],
                device=gate["device"],
                needs=gate["needs"],
                runs_root=runs_root.resolve(),
                parent_run_ids=parent_run_ids,
                launch=launch,
                gpu=gpu,
            ) as owner:
                run = owner.path
                try:
                    if _load_record(run / "run.toml")["spec_sha256"] != gate["spec_sha256"]:
                        raise ValueError(f"inputs changed after {operation} validation")
                    if gpu is not None:
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
                        job_input={
                            "sweep": "sweep-manifest.toml",
                            "simulate": "plan.toml",
                            "control": "problem.toml",
                        }[operation],
                    )
                except (Exception, KeyboardInterrupt) as exc:
                    status, exit_code = (
                        ("aborted", 130) if isinstance(exc, KeyboardInterrupt) else ("failed", 64)
                    )
                    if not (run / "FINISHED").exists():
                        (run / "run.log").write_text(f"{operation} launch failed: {exc}\n")
                        record_finish(
                            owner,
                            status=status,
                            exit_code=exit_code,
                            wall_seconds=0,
                            gpu_seconds=time.monotonic() - gpu.allocated_at if gpu else 0,
                        )
                    result = {"path": str(run), "exit_code": exit_code, "status": status}
    return result


def rerun_simulation(run: Path, *, repository: Path | None = None, runs_root: Path | None = None):
    return _rerun_protocol(run, operation="simulate", repository=repository, runs_root=runs_root)


def _rerun_protocol(run, *, operation, repository=None, runs_root=None):
    record = _read_attempt(run)
    if record["manifest"]["protocol"] != operation or record["status"] in {"active", "unknown"}:
        raise ValueError(f"rerun requires an inactive {operation} attempt")
    run = Path(record["path"])
    repository = (repository or Path(__file__).resolve().parents[3]).resolve()
    for name, digest in record["manifest"]["spec_sha256"].items():
        if _digest((run / "spec" / name).read_bytes()) != digest:
            raise ValueError(f"frozen input digest mismatch: {name}")
    manifest = record["manifest"]
    investigation_id = manifest.get("investigation_id")
    if investigation_id and runs_root is not None and runs_root.resolve() != run.parent.parent:
        raise ValueError("investigation reruns must retain their runs root")
    protocol = run / "spec" / f"{operation}.toml"
    if investigation_id:
        protocol = run / "spec/base" / f"{operation}.toml"
    result = _launch_protocol(
        protocol,
        operation=operation,
        dry_run=False,
        repository=repository,
        runs_root=runs_root or run.parent.parent,
        parent_run_ids=(run.name,),
        investigation_id=investigation_id,
        base=manifest.get("investigation_base"),
        values=manifest.get("launch", {}).get("varied_values")
        if operation == "simulate" and investigation_id
        else None,
    )
    replay = _read_attempt(Path(result["path"]))["manifest"]
    result["replay"] = {
        "code_matches": replay["code"] == record["manifest"]["code"],
        "environment_matches": replay["env_hash"] == record["manifest"]["env_hash"],
    }
    return result
