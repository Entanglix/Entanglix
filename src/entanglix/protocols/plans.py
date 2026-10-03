"""Validated CPU computation plans with numeric inputs kept in adjacent NPZ files."""

import hashlib
import io
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from entanglix.corrector.metrics import EndpointMetric, Population, observable_operator
from entanglix.engines.frames import prepare_dynamics
from entanglix.evolution import Carrier, EvolutionInput, RotatingFrame, normalize_input
from entanglix.models import ConstructedSystem, StrictModel, load_system


class Refinement(StrictModel):
    retained_levels_step: int = Field(strict=True, gt=0)
    cutoff_steps: dict[str, Annotated[int, Field(strict=True, gt=0)]] = Field(default_factory=dict)


class CarrierSpec(StrictModel):
    frequency_ghz: float
    phase_rad: float = 0.0


class FrameSpec(StrictModel):
    frequencies_ghz: dict[str, float]
    rwa: Literal[True]
    excitation_numbers: dict[str, list[int]] = Field(default_factory=dict)


class EvolutionSpec(StrictModel):
    array_file: str
    initial_state: str
    times_ns: list[float]
    observables: dict[str, str | Population] = Field(default_factory=dict)
    drive_amplitudes_ghz: dict[str, float | list[float]] = Field(default_factory=dict)
    rtol: float = 1e-10
    atol: float = 1e-12
    max_steps: int = Field(strict=True, gt=0, default=100_000)
    carriers: dict[str, CarrierSpec] = Field(default_factory=dict)
    frame: FrameSpec | None = None


class ExecutionPlan(StrictModel):
    format_version: Literal[1]
    kind: Literal["lindblad", "spectrum"]
    engine: Literal["qutip-cpu", "dynamiqs", "cudensitymat"] = "qutip-cpu"
    device: Literal["cpu", "cuda"] = "cpu"
    levels: int | None = Field(default=None, strict=True, gt=0)
    evolution: EvolutionSpec | None = None
    refinement: dict[str, Refinement] = Field(default_factory=dict)
    metrics: dict[str, EndpointMetric] = Field(default_factory=dict)
    segment_tolerances: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_kind(self):
        if (self.kind == "lindblad") != (self.evolution is not None):
            raise ValueError("lindblad requires evolution; spectrum must not contain evolution")
        if (self.kind == "spectrum") != (self.levels is not None):
            raise ValueError("spectrum requires levels; lindblad must not contain levels")
        if self.kind == "spectrum" and self.engine != "qutip-cpu":
            raise ValueError("spectrum uses the qutip-cpu eigensolver")
        if self.engine == "cudensitymat" and self.device != "cuda":
            raise ValueError("cudensitymat requires device cuda")
        if self.engine == "qutip-cpu" and self.device != "cpu":
            raise ValueError("qutip-cpu requires device cpu")
        if self.metrics and self.evolution is None:
            raise ValueError("endpoint metrics require evolution")
        if self.evolution is not None:
            names = self.evolution.observables.keys()
            if any(metric.observable not in names for metric in self.metrics.values()):
                raise ValueError("metrics must name declared observables")
        if any(
            not math.isfinite(value) or value <= 0 for value in self.segment_tolerances.values()
        ):
            raise ValueError("segment_tolerances must be finite and positive")
        if self.segment_tolerances.keys() - (
            self.evolution.observables.keys() if self.evolution else set()
        ):
            raise ValueError("segment_tolerances must name declared observables")
        return self


@dataclass(frozen=True)
class LoadedPlan:
    definition: ExecutionPlan
    system: ConstructedSystem
    request: EvolutionInput | None
    digest: str
    inputs: dict[str, Path]
    input_digests: dict[str, str]


def load_plan(system_path: Path, plan_path: Path) -> LoadedPlan:
    from entanglix.runner import _system_inputs

    content = plan_path.read_bytes()
    plan = ExecutionPlan.model_validate(tomllib.loads(content.decode("utf-8")))
    system = load_system(system_path)
    paths, hashes = _system_inputs(system_path.resolve(), system, reserved={"plan.toml"})
    paths["plan.toml"] = plan_path.resolve()
    hashes["plan.toml"] = hashlib.sha256(content).hexdigest()
    if plan.levels is not None and plan.levels > math.prod(system.dimensions):
        raise ValueError("spectrum levels exceed the model dimension")
    request = None
    if plan.evolution is not None:
        spec = plan.evolution
        relative = Path(spec.array_file)
        if relative.name != spec.array_file or relative.suffix != ".npz":
            raise ValueError("evolution.array_file must name an NPZ file beside the plan")
        if spec.array_file in paths:
            raise ValueError("evolution.array_file conflicts with a system or reserved input")
        path = plan_path.parent / relative
        if path.resolve().parent != plan_path.resolve().parent:
            raise ValueError("evolution.array_file must stay beside the plan")
        data = path.read_bytes()
        # Hash the same bytes that are decoded, so gate/freezing detects input changes.
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            state = np.array(archive[spec.initial_state])
        if state.dtype.kind not in "iufc":
            raise ValueError("initial_state must be a numeric array")
        operators = {
            name: observable_operator(system, value) for name, value in spec.observables.items()
        }
        for metric in plan.metrics.values():
            operator = operators[metric.observable]
            if metric.kind == "endpoint_real" and not np.allclose(
                operator, operator.conj().T, rtol=0, atol=1e-12
            ):
                raise ValueError("endpoint_real requires a Hermitian observable")
        request = EvolutionInput(
            state,
            np.asarray(spec.times_ns),
            operators,
            {name: np.asarray(value) for name, value in spec.drive_amplitudes_ghz.items()},
            spec.rtol,
            spec.atol,
            spec.max_steps,
            {name: Carrier(**value.model_dump()) for name, value in spec.carriers.items()},
            RotatingFrame(**spec.frame.model_dump()) if spec.frame is not None else None,
        )
        prepare_dynamics(system, normalize_input(system, request))
        paths[spec.array_file] = path.resolve()
        hashes[spec.array_file] = hashlib.sha256(data).hexdigest()
    identity = {
        "definition": plan.model_dump(exclude_none=True),
        "inputs": {name: value for name, value in hashes.items() if name != "plan.toml"},
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return LoadedPlan(plan, system, request, digest, paths, hashes)
