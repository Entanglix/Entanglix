"""Validated CPU computation plans with numeric inputs kept in adjacent NPZ files."""

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from entanglix.corrector import load_policy, validate_refinement
from entanglix.engines.selection import selection_table
from entanglix.evolution import EvolutionInput, normalize_input
from entanglix.models import ConstructedSystem, StrictModel, load_system


class Refinement(StrictModel):
    retained_levels_step: int = Field(strict=True, gt=0)
    cutoff_steps: dict[str, int] = Field(default_factory=dict)


class EvolutionSpec(StrictModel):
    array_file: str
    initial_state: str
    times_ns: list[float]
    observables: dict[str, str] = Field(default_factory=dict)
    drive_amplitudes_ghz: dict[str, float | list[float]] = Field(default_factory=dict)
    rtol: float = 1e-10
    atol: float = 1e-12
    max_steps: int = Field(strict=True, gt=0, default=100_000)


class ExecutionPlan(StrictModel):
    format_version: Literal[1]
    kind: Literal["lindblad", "spectrum"]
    engine: Literal["qutip-cpu", "dynamiqs"] = "qutip-cpu"
    purpose: Literal["computation", "reference"] = "computation"
    evolution: EvolutionSpec | None = None
    refinement: dict[str, Refinement] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_kind(self):
        if (self.kind == "lindblad") != (self.evolution is not None):
            raise ValueError("lindblad requires evolution; spectrum must not contain evolution")
        if self.kind == "spectrum" and self.engine != "qutip-cpu":
            raise ValueError("spectrum uses the qutip-cpu eigensolver")
        return self


@dataclass(frozen=True)
class LoadedPlan:
    definition: ExecutionPlan
    system: ConstructedSystem
    request: EvolutionInput | None
    version: str
    digest: str
    inputs: dict[str, Path]
    input_digests: dict[str, str]


def load_plan(
    system_path: Path, plan_path: Path, *, expected_policy: str | None = None
) -> LoadedPlan:
    from entanglix.runner import _system_inputs

    content = plan_path.read_bytes()
    plan = ExecutionPlan.model_validate(tomllib.loads(content.decode("utf-8")))
    policy = load_policy(expected_policy)
    system = load_system(system_path)
    paths, hashes = _system_inputs(system_path.resolve(), system, reserved={"plan.toml"})
    paths["plan.toml"] = plan_path.resolve()
    hashes["plan.toml"] = hashlib.sha256(content).hexdigest()
    document = tomllib.loads(system_path.read_text())
    validate_refinement(plan, document, system_path.parent, policy)
    limit = (
        policy.thresholds["reference"]["max_dimension"]
        if plan.purpose == "reference"
        else selection_table()["engines"][plan.engine]["max_dimension"]
    )
    if math.prod(system.dimensions) > limit:
        raise ValueError(f"plan dimension exceeds validated {plan.purpose} limit {limit}")
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
        import io

        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            state = np.array(archive[spec.initial_state])
        if state.dtype.kind not in "iufc":
            raise ValueError("initial_state must be a numeric array")
        unknown = set(spec.observables.values()) - system.operators.keys()
        if unknown:
            raise ValueError(f"unknown observable operators: {sorted(unknown)}")
        request = EvolutionInput(
            state,
            np.asarray(spec.times_ns),
            {name: system.operators[operator] for name, operator in spec.observables.items()},
            {name: np.asarray(value) for name, value in spec.drive_amplitudes_ghz.items()},
            spec.rtol,
            spec.atol,
            spec.max_steps,
        )
        normalize_input(system, request)
        paths[spec.array_file] = path.resolve()
        hashes[spec.array_file] = hashlib.sha256(data).hexdigest()
    identity = {
        "definition": plan.model_dump(exclude_none=True),
        "inputs": {name: value for name, value in hashes.items() if name != "plan.toml"},
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return LoadedPlan(plan, system, request, policy.version, digest, paths, hashes)
