"""Explicit serial grids, frozen point plans, partial results and linked replay."""

import copy
import hashlib
import itertools
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import Field, model_validator

from entanglix.models import StrictModel
from entanglix.protocols.plans import load_plan
from entanglix.protocols.simulate import (
    _adjacent,
    _launch_protocol,
    _rerun_protocol,
    compile_simulation,
)
from entanglix.runner import _digest


class Axis(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    target: str
    values: list[float] = Field(min_length=1)
    linked_values: dict[str, list[float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_values(self):
        if any(not math.isfinite(v) for v in self.values) or len(set(self.values)) != len(
            self.values
        ):
            raise ValueError("axis values must be finite and unique")
        if any(
            len(v) != len(self.values) or not all(math.isfinite(x) for x in v)
            for v in self.linked_values.values()
        ):
            raise ValueError("linked_values must be finite with one value per axis entry")
        return self


class Expectation(StrictModel):
    metric: str
    axis: str
    direction: Literal["increasing", "decreasing"]


class SweepSpec(StrictModel):
    format_version: Literal[1]
    base: str
    axes: list[Axis] = Field(min_length=1)
    evidence_metric: str | None = None
    expectation: Expectation | None = None
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique_axes(self):
        if len({axis.id for axis in self.axes}) != len(self.axes):
            raise ValueError("axis IDs must be unique")
        targets = [target for axis in self.axes for target in [axis.target, *axis.linked_values]]
        if len(set(targets)) != len(targets):
            raise ValueError("each sweep target belongs to exactly one axis")
        if self.expectation and self.expectation.axis not in {axis.id for axis in self.axes}:
            raise ValueError("expectation must name a sweep axis")
        return self


def _set_target(document, target, value):
    fields = target.split(".")
    allowed = (
        len(fields) == 2
        and fields[0] == "parameters"
        or len(fields) == 3
        and fields[:2] == ["evolution", "drive_amplitudes_ghz"]
        or len(fields) == 4
        and fields[:2] == ["evolution", "carriers"]
        and fields[-1] == "frequency_ghz"
        or len(fields) == 4
        and fields[:3] == ["evolution", "frame", "frequencies_ghz"]
    )
    if not allowed:
        raise ValueError(f"unsupported sweep target: {target}")
    cursor = document
    for field in fields[:-1]:
        if field not in cursor or not isinstance(cursor[field], dict):
            raise ValueError(f"sweep target must name a declared scalar: {target}")
        cursor = cursor[field]
    if fields[-1] not in cursor or not isinstance(cursor[fields[-1]], (int, float)):
        raise ValueError(f"sweep target must name a declared scalar: {target}")
    cursor[fields[-1]] = value


@dataclass(frozen=True)
class LoadedSweep:
    definition: SweepSpec
    manifest: dict
    points: tuple
    digest: str
    inputs: dict
    input_digests: dict

    @property
    def system(self):
        return self.points[0].system


def load_sweep(root):
    root = Path(root).resolve()
    definition = SweepSpec.model_validate(tomllib.loads((root / "sweep.toml").read_text()))
    manifest = tomllib.loads((root / "sweep-manifest.toml").read_text())
    points = []
    inputs = {name: root / name for name in manifest["base_inputs"]}
    inputs.update({name: root / name for name in ("sweep.toml", "sweep-manifest.toml")})
    for index, entry in enumerate(manifest["points"]):
        directory = root / "points" / f"{index:06d}"
        point = load_plan(directory / "system.toml", directory / "plan.toml")
        if point.digest != entry["plan_sha256"]:
            raise ValueError(f"frozen point {index} plan digest changed")
        points.append(point)
        inputs.update({str(path.relative_to(root)): path for path in point.inputs.values()})
    identity = {"spec": definition.model_dump(exclude_none=True), "manifest": manifest}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return LoadedSweep(
        definition,
        manifest,
        tuple(points),
        digest,
        inputs,
        {name: _digest(path.read_bytes()) for name, path in inputs.items()},
    )


def compile_sweep(
    protocol,
    directory,
    extras,
    repository,
    *,
    retain_base=False,
    runs_root=Path("research_runs"),
    replay_source=None,
):
    content = protocol.read_bytes()
    spec = SweepSpec.model_validate(tomllib.loads(content.decode()))
    base_path = _adjacent(protocol.parent, spec.base)
    gate = compile_simulation(
        base_path,
        directory,
        extras,
        repository,
        retain_base=retain_base,
        runs_root=runs_root,
        replay_source=replay_source,
    )
    base = tomllib.loads((directory / "simulate.toml").read_text())
    if (spec.evidence_metric is not None and spec.evidence_metric not in base["metrics"]) or (
        spec.expectation and spec.expectation.metric not in base["metrics"]
    ):
        raise ValueError("evidence_metric and expectation must name declared metrics")
    # A parameter axis must be declared in both the system and the base overrides.
    for axis in spec.axes:
        for target in [axis.target, *axis.linked_values]:
            _set_target(copy.deepcopy(base), target, axis.values[0])
    shape = [len(axis.values) for axis in spec.axes]
    points = []
    scope_values = []
    for index, coordinate in enumerate(itertools.product(*(range(n) for n in shape))):
        point_spec = copy.deepcopy(base)
        varied = {}
        for axis, offset in zip(spec.axes, coordinate):
            varied[axis.target] = axis.values[offset]
            _set_target(point_spec, axis.target, axis.values[offset])
            for target, values in axis.linked_values.items():
                varied[target] = values[offset]
                _set_target(point_spec, target, values[offset])
        scope_values.append(varied)
        point_path = directory / "point-source.toml"
        point_path.write_text(tomli_w.dumps(point_spec))
        target = directory / "points" / f"{index:06d}"
        target.mkdir(parents=True)
        point_gate = compile_simulation(point_path, target, extras, repository, probe_device=False)
        (target / "simulate.toml").unlink()
        points.append(
            {
                "index": index,
                "coordinate": list(coordinate),
                "plan_sha256": point_gate["plan_sha256"],
            }
        )
    (directory / "point-source.toml").unlink()
    canonical = spec.model_dump(exclude_none=True)
    canonical["base"] = "simulate.toml"
    (directory / "sweep.toml").write_text(tomli_w.dumps(canonical))
    (directory / "sweep-manifest.toml").write_text(
        tomli_w.dumps(
            {
                "format_version": 1,
                "kind": gate["kind"],
                "shape": shape,
                "base_inputs": list(gate["inputs"]),
                "points": points,
            }
        )
    )
    base_paths = {}
    if retain_base:
        replay = spec.model_dump(exclude_none=True)
        replay["base"] = "simulate.toml"
        (directory / "base/sweep.toml").write_text(tomli_w.dumps(replay))
        base_paths["base/sweep.toml"] = directory / "base/sweep.toml"
    loaded = load_sweep(directory)
    from entanglix.corrector.sweeps import sweep_listing

    gate.update(
        inputs={name: str(path) for name, path in {**loaded.inputs, **base_paths}.items()},
        spec_sha256={
            **loaded.input_digests,
            **{name: _digest(path.read_bytes()) for name, path in base_paths.items()},
        },
        scope_values=scope_values,
        plan_sha256=loaded.digest,
        protocol_sha256=_digest(content),
        timeout_seconds=spec.timeout_seconds,
        preview=sweep_listing(loaded),
    )
    return gate


def sweep(
    protocol,
    *,
    runs_root=Path("research_runs"),
    repository=None,
    parent_run_ids=(),
    dry_run=False,
    investigation_id=None,
    base=None,
):
    return _launch_protocol(
        protocol,
        runs_root=runs_root,
        repository=repository,
        parent_run_ids=parent_run_ids,
        dry_run=dry_run,
        operation="sweep",
        investigation_id=investigation_id,
        base=base,
    )


def rerun_sweep(run, *, repository=None, runs_root=None):
    return _rerun_protocol(run, operation="sweep", repository=repository, runs_root=runs_root)


def partial_results(run):
    points = []
    for path in sorted((run / "artifacts/points").glob("*/result.toml")):
        try:
            item = tomllib.loads(path.read_text())
            output = path.parent / "output.npz"
            if _digest(output.read_bytes()) != item["output_sha256"]:
                raise ValueError("point output digest mismatch")
            points.append(item)
        except (OSError, ValueError, KeyError) as exc:
            points.append({"index": int(path.parent.name), "error": str(exc)})
    return {
        "completed_points": points,
        "completed_count": sum("error" not in item for item in points),
    }
