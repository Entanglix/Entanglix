"""The calibrate protocol: a validated spec, frozen inputs, and the fitting job's launch."""

import copy
import hashlib
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import tomli_w
from pydantic import Field, model_validator

from entanglix.calibration import (
    EQUIVALENCE_RTOL,
    JACOBIAN_STEP,
    RANK_RTOL,
    UNDETERMINED_TOLERANCE,
    Matching,
    Member,
    Peaks,
    Problem,
    Spectrum,
)
from entanglix.extraction import ExtractionSettings, extract, frequency_step
from entanglix.measurements import Dataset, load_dataset
from entanglix.models import ConstructedSystem, StrictModel, load_system
from entanglix.protocols import SystemSource, base_files, read_inputs, write_frozen
from entanglix.protocols.plans import Refinement

ENGINE = "scipy-least-squares"
Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Level = Annotated[int, Field(strict=True, ge=0)]
Count = Annotated[int, Field(strict=True, gt=0)]
State = Level | Annotated[list[Level], Field(min_length=1)]
GENERATED = re.compile(r"^\d+-\d+(/\d+)?$")
NUISANCE = ("offset", "period")


class Fitted(StrictModel):
    scale: Positive


class Nuisance(StrictModel):
    lower: Finite
    upper: Finite
    scale: Positive

    @model_validator(mode="after")
    def check_bounds(self):
        if not self.lower < self.upper:
            raise ValueError("nuisance bounds require lower < upper")
        return self


class AxisMap(StrictModel):
    flux: str = Field(min_length=1)
    offset: Nuisance
    period: Nuisance

    @model_validator(mode="after")
    def check_period(self):
        if self.period.lower <= 0:
            raise ValueError("axis_map.period must be positive: its lower bound must exceed 0")
        return self


class Line(StrictModel):
    initial: State
    final: State
    photons: Count = 1

    @model_validator(mode="after")
    def check_states(self):
        if isinstance(self.initial, int) != isinstance(self.final, int):
            raise ValueError("a line's states are both levels or both bare product states")
        if isinstance(self.initial, int) and self.final <= self.initial:
            raise ValueError("a line's final level must exceed its initial level")
        if isinstance(self.initial, list) and (
            len(self.initial) != len(self.final) or self.initial == self.final
        ):
            raise ValueError("a line's bare states must be distinct and of equal length")
        return self


class Family(StrictModel):
    initial: list[Level] = Field(default_factory=list)
    final: list[Level] = Field(default_factory=list)
    max_level: Level | None = None
    photons: list[Count] = Field(default_factory=lambda: [1])
    lines: dict[str, Line] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_ranges(self):
        if self.final and self.max_level is not None:
            raise ValueError("family declares final or max_level, not both")
        if bool(self.initial) != bool(self.final or self.max_level is not None):
            raise ValueError("family.initial requires final or max_level, and they require it")
        if any(GENERATED.match(name) for name in self.lines):
            raise ValueError("named lines must not use the generated label form i-f or i-f/k")
        if {"unexplained", "unresolved"} & set(self.lines):
            raise ValueError("named lines must not be called unexplained or unresolved")
        return self

    def members(self) -> tuple[Member, ...]:
        finals = self.final or range(self.max_level + 1 if self.max_level is not None else 0)
        members = [
            Member(f"{i}-{f}" if k == 1 else f"{i}-{f}/{k}", i, f, k)
            for i in self.initial
            for f in finals
            if f > i
            for k in self.photons
        ]
        for name, line in self.lines.items():
            states = [tuple(s) if isinstance(s, list) else s for s in (line.initial, line.final)]
            members.append(Member(name, *states, line.photons))
        transitions = [(m.initial, m.final, m.photons) for m in members]
        if not members:
            raise ValueError("the candidate family has no member")
        if len(set(transitions)) != len(transitions):
            raise ValueError("candidate family members must be distinct transitions")
        return tuple(members)


class MatchingSpec(StrictModel):
    anchor: str | None = Field(default=None, min_length=1)
    range_ghz: Positive
    cutoff: Positive
    resolution: Positive
    range_iterations: Count
    cutoff_iterations: Count


class Conventions(StrictModel):
    offset_window: Finite | None = None
    exchange: list[Annotated[list[str], Field(min_length=2, max_length=2)]] = Field(
        default_factory=list
    )


class CalibrationSpec(SystemSource):
    format_version: Literal[1]
    datasets: list[str] = Field(min_length=1)
    parameters: dict[str, Fitted] = Field(min_length=1)
    axis_map: AxisMap | None = None
    family: Family
    matching: MatchingSpec | None = None
    extraction: ExtractionSettings | None = None
    starts: list[dict[str, Finite]] = Field(min_length=1)
    merge_tolerance: Positive
    conventions: Conventions = Field(default_factory=Conventions)
    refinement: dict[str, Refinement] = Field(default_factory=dict)
    timeout_seconds: Positive | None = None


@dataclass(frozen=True)
class LoadedCalibration:
    spec: CalibrationSpec
    system_path: Path
    document: dict
    system: ConstructedSystem
    datasets: dict[str, Dataset]
    members: tuple[Member, ...]
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    scales: np.ndarray
    labelled: bool
    inputs: dict[str, Path]
    input_digests: dict[str, str]
    listing: dict
    catalog: dict


def _references(value) -> set[str]:
    """Parameter names referenced anywhere below a parsed model table."""
    if isinstance(value, dict):
        if value.keys() == {"parameter"}:
            return {value["parameter"]}
        return set().union(*(_references(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_references(item) for item in value))
    return set()


def _beside(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).name != name or path.resolve().parent != root:
        raise ValueError("calibration inputs must name files beside the calibration spec")
    return path


def load_calibration(
    protocol: Path,
    *,
    runs_root=Path("research_runs"),
    replay_source=None,
) -> LoadedCalibration:
    """Validate a calibration spec, its system and datasets; nothing is extracted or fitted."""
    from entanglix.runner import _system_inputs

    protocol = protocol.resolve()
    content = protocol.read_bytes()
    spec = CalibrationSpec.model_validate(tomllib.loads(content.decode("utf-8")))
    source, catalog = spec.resolve_system(protocol, runs_root, replay_source)
    authored = source.read_bytes()
    system = load_system(source, content=authored)
    document = tomllib.loads(authored.decode("utf-8"))
    paths, hashes = _system_inputs(source, system, reserved={"calibrate.toml"})
    if protocol in {path.resolve() for path in paths.values()}:
        raise ValueError("calibration spec must be distinct from its system inputs")
    paths["calibrate.toml"], hashes["calibrate.toml"] = (
        protocol,
        hashlib.sha256(content).hexdigest(),
    )

    declared = document.get("parameters", {})
    referenced = _references(document["model"])
    axis = spec.axis_map
    for name in [*spec.parameters, *([axis.flux] if axis else [])]:
        if name not in declared:
            raise ValueError(f"{name!r} is not in the system file's parameter table")
        if name not in referenced:
            raise ValueError(f"parameter {name!r} enters no construction field of the system")
    if axis is not None and axis.flux in spec.parameters:
        raise ValueError("the axis map's flux parameter is not also fitted directly")
    for name in spec.parameters:
        if "prior" not in declared[name]:
            raise ValueError(f"fitted parameter {name!r} needs a uniform prior in the system file")
    if axis is not None:
        if declared[axis.flux]["unit"] != "Phi0":
            raise ValueError(f"axis_map.flux {axis.flux!r} must be a parameter in Phi0")
        if set(NUISANCE) & set(spec.parameters):
            raise ValueError("fitted parameters must not be named offset or period")
    names = (*spec.parameters, *(NUISANCE if axis else ()))
    bounds = [
        (declared[n]["prior"]["lower"], declared[n]["prior"]["upper"]) for n in spec.parameters
    ]
    scales = [fitted.scale for fitted in spec.parameters.values()]
    if axis is not None:
        bounds += [(axis.offset.lower, axis.offset.upper), (axis.period.lower, axis.period.upper)]
        scales += [axis.offset.scale, axis.period.scale]
    lower, upper = (np.array(side, dtype=float) for side in zip(*bounds, strict=True))

    if len(set(spec.datasets)) != len(spec.datasets):
        raise ValueError("training datasets must be distinct")
    datasets = {}
    for name in spec.datasets:
        path = _beside(protocol.parent, name)
        dataset = load_dataset(path)
        # Files are frozen under the names the spec and the dataset use, even through a link.
        files = {
            name: dataset.files[path.resolve().name],
            dataset.spec.array_file: dataset.files[dataset.spec.array_file],
        }
        for file_name, data in files.items():
            if file_name in paths:
                raise ValueError(
                    f"dataset file conflicts with another calibration input: {file_name}"
                )
            paths[file_name] = protocol.parent / file_name
            hashes[file_name] = hashlib.sha256(data).hexdigest()
        level = "map" if spec.extraction is not None else "points"
        if getattr(dataset.spec, level) is None:
            raise ValueError(
                f"dataset {name!r} has no {level}; "
                + (
                    "extraction reads maps"
                    if level == "map"
                    else "without extraction, points are fitted"
                )
            )
        if level == "map":
            try:
                frequency_step(dataset)
            except ValueError as exc:
                raise ValueError(f"dataset {name!r}: {exc}") from exc
        datasets[name] = dataset
    units = {d.spec.bias_unit for d in datasets.values()} - {None}
    if len(units) > 1:
        raise ValueError("training datasets with a bias axis must share one bias unit")
    if units and axis is None:
        raise ValueError("a dataset with a bias axis needs an axis_map to its flux parameter")
    if axis is not None and not units:
        raise ValueError("an axis_map needs a training dataset with a bias axis")

    members = spec.family.members()
    labels = {member.label: index for index, member in enumerate(members)}
    supplied = [
        dataset.arrays.get("points.assignment", np.full(dataset.arrays["points.centre"].size, ""))
        for dataset in datasets.values()
        if spec.extraction is None
    ]
    given = sum(int(np.count_nonzero(array != "")) for array in supplied)
    total = sum(array.size for array in supplied)
    if 0 < given < total:
        raise ValueError(f"partly labelled input: {given} of {total} points carry an assignment")
    labelled = total > 0 and given == total
    if labelled:
        if spec.matching is not None:
            raise ValueError("supplied assignments use no anchor or matching settings")
        unknown = sorted(set(np.concatenate(supplied).tolist()) - labels.keys())
        if unknown:
            raise ValueError(f"label {unknown[0]!r} is not a member of the candidate family")
    else:
        if spec.matching is None:
            raise ValueError("automatic assignment needs [matching] settings")
        anchor = spec.matching.anchor
        if anchor is not None and anchor not in labels:
            raise ValueError(f"anchor {anchor!r} is not a member of the candidate family")
        # Extracted peaks carry every field; points must supply what matching uses.
        needed = ["linewidth"] + (["amplitude"] if anchor is not None else [])
        for name, dataset in datasets.items() if spec.extraction is None else ():
            for field in needed:
                if f"points.{field}" not in dataset.arrays:
                    raise ValueError(
                        f"dataset {name!r}: automatic assignment needs each point's {field}"
                    )

    dimensions = system.dimensions
    for member in members:
        for state in (member.initial, member.final):
            if isinstance(state, int) and state >= math.prod(dimensions):
                raise ValueError(
                    f"line {member.label!r}: level {state} exceeds the system dimension"
                )
            if isinstance(state, tuple) and (
                len(state) != len(dimensions) or any(s >= d for s, d in zip(state, dimensions))
            ):
                raise ValueError(
                    f"line {member.label!r}: bare state {list(state)} does not fit dimensions "
                    f"{list(dimensions)}"
                )
    for index, start in enumerate(spec.starts):
        if start.keys() != set(names):
            raise ValueError(f"starts[{index}] must give exactly {sorted(names)}")
        for k, name in enumerate(names):
            if not lower[k] <= start[name] <= upper[k]:
                raise ValueError(f"starts[{index}].{name} lies outside its bounds")
    paired = [name for pair in spec.conventions.exchange for name in pair]
    if len(set(paired)) != len(paired):
        raise ValueError("each parameter appears in at most one exchange pair")
    for a, b in spec.conventions.exchange:
        if a not in spec.parameters or b not in spec.parameters:
            raise ValueError("exchange pairs name fitted parameters")
        if (
            declared[a]["unit"] != declared[b]["unit"]
            or declared[a]["prior"] != declared[b]["prior"]
            or spec.parameters[a] != spec.parameters[b]
        ):
            raise ValueError(f"exchange {a}/{b} needs equal units, priors and scales")
    if spec.conventions.offset_window is not None and axis is None:
        raise ValueError("conventions.offset_window needs an axis_map")

    listing = {
        "protocol": "calibrate",
        "system_id": system.id,
        "dimensions": list(dimensions),
        "datasets": {
            name: {
                "sha256": dataset.sha256,
                **({"bias_unit": unit} if (unit := dataset.spec.bias_unit) else {}),
                **(
                    {"map_shape": list(np.shape(dataset.arrays["map.signal"]))}
                    if dataset.spec.map is not None
                    else {}
                ),
                **(
                    {"points": int(dataset.arrays["points.centre"].size)}
                    if dataset.spec.points is not None
                    else {}
                ),
            }
            for name, dataset in datasets.items()
        },
        "procedure": {
            "points": "extracted from maps" if spec.extraction is not None else "supplied",
            "assignment": "supplied labels" if labelled else "automatic",
            **({"anchor": spec.matching.anchor} if spec.matching and spec.matching.anchor else {}),
        },
        "parameters": {
            name: {"lower": float(lower[k]), "upper": float(upper[k]), "scale": scales[k]}
            for k, name in enumerate(names)
        },
        **({"axis_map": {"flux": axis.flux, "nuisance": list(NUISANCE)}} if axis else {}),
        "family": [member.label for member in members],
        "starts": len(spec.starts),
        "tolerances": tolerances(spec),
    }
    return LoadedCalibration(
        spec,
        source,
        document,
        system,
        datasets,
        members,
        names,
        lower,
        upper,
        np.array(scales, dtype=float),
        labelled,
        paths,
        hashes,
        listing,
        catalog,
    )


def tolerances(spec: CalibrationSpec) -> dict:
    return {
        "rank_rtol": RANK_RTOL,
        "undetermined_tolerance": UNDETERMINED_TOLERANCE,
        "jacobian_step": JACOBIAN_STEP,
        "equivalence_rtol": EQUIVALENCE_RTOL,
        "merge_tolerance": spec.merge_tolerance,
    }


def chosen_solution(result: dict, solution: int | None, action: str) -> int | None:
    """The named solution's index, or the only one's; None when the run returned none."""
    count = len(result["solutions"])
    if solution is None and count > 1:
        raise ValueError(f"the source run has {count} distinct solutions; name the one to {action}")
    if solution is not None and (type(solution) is not int or not 0 <= solution < count):
        raise ValueError(f"the source run has no solution {solution!r}")
    return (solution or 0) if count else None


def dataset_peaks(name, dataset, extraction, labels, projection=None):
    """One dataset's bias columns, frequency windows, peaks, extraction failures and record.

    With extraction settings the map is extracted, under ``projection`` when one is given;
    otherwise the points are read, with supplied labels as member indices from ``labels``.
    Peak columns count from 0 within the dataset.
    """
    columns, windows = [], []
    if extraction is not None:
        found = extract(dataset, extraction, projection)
        frequency, bias = dataset.arrays["map.frequency"], dataset.arrays.get("map.bias")
        for c in range(1 if bias is None else bias.size):
            columns.append(
                {"dataset": name, "map_column": c}
                | ({} if bias is None else {"bias": float(bias[c])})
            )
            windows.append((float(frequency[0]), float(frequency[-1])))
        points = found.points
        peaks = {
            "column": points["column"],
            "centre": points["centre"],
            "uncertainty": points["centre_uncertainty"],
            "linewidth": points["linewidth"],
            "amplitude": points["amplitude"],
            "line": np.full(points["centre"].size, -1),
        }
        failures = [{"dataset": name, **failure} for failure in found.failures]
        record = {
            "sha256": dataset.sha256,
            "projection": list(found.projection),
            "peaks": int(points["centre"].size),
            "extraction_failures": len(found.failures),
        }
        return columns, windows, peaks, failures, record
    arrays = dataset.arrays
    size = arrays["points.centre"].size
    bias = arrays.get("points.bias")
    if bias is None:
        columns.append({"dataset": name})
        index = np.zeros(size, dtype=int)
    else:
        values, index = np.unique(bias, return_inverse=True)
        columns += [{"dataset": name, "bias": float(value)} for value in values]
    windows += [None] * len(columns)
    missing = np.full(size, np.nan)
    peaks = {
        "column": index,
        "centre": arrays["points.centre"],
        "uncertainty": arrays["points.centre_uncertainty"],
        "linewidth": arrays.get("points.linewidth", missing),
        "amplitude": arrays.get("points.amplitude", missing),
        "line": np.array(
            [labels.get(label, -1) for label in arrays.get("points.assignment", [""] * size)],
            dtype=int,
        ),
    }
    return columns, windows, peaks, [], {"sha256": dataset.sha256, "peaks": int(size)}


def merge_peaks(rows) -> Peaks:
    return Peaks(
        *(
            np.concatenate([peaks[field] for peaks in rows])
            for field in ("column", "centre", "uncertainty", "linewidth", "amplitude", "line")
        )
    )


def spectrum_for(loaded: LoadedCalibration, columns, document=None, path=None) -> Spectrum:
    """The forward model at the given bias columns, on the frozen system or a refined copy."""
    axis = loaded.spec.axis_map
    return Spectrum(
        path or loaded.system_path,
        copy.deepcopy(document or loaded.document),
        loaded.members,
        loaded.spec.parameters,
        [column.get("bias", np.nan) for column in columns],
        axis.flux if axis else None,
    )


def prepare(loaded: LoadedCalibration) -> tuple[Problem, dict]:
    """Extract or read the peaks and build the search problem; returns it and the peak record."""
    spec, members = loaded.spec, loaded.members
    labels = {member.label: index for index, member in enumerate(members)}
    columns, windows, rows, failures, datasets = [], [], [], [], {}
    for name, dataset in loaded.datasets.items():
        added, ranges, peaks, failed, datasets[name] = dataset_peaks(
            name, dataset, spec.extraction, labels
        )
        peaks["column"] = len(columns) + peaks["column"]
        columns += added
        windows += ranges
        rows.append(peaks)
        failures += failed
    merged = merge_peaks(rows)
    spectrum = spectrum_for(loaded, columns)
    matching = None
    if not loaded.labelled:
        settings = spec.matching.model_dump()
        anchor = settings.pop("anchor")
        matching = Matching(**settings, anchor=None if anchor is None else labels[anchor])
    index = {name: k for k, name in enumerate(loaded.names)}
    problem = Problem(
        spectrum,
        merged,
        loaded.names,
        loaded.lower,
        loaded.upper,
        loaded.scales,
        spec.merge_tolerance,
        matching,
        tuple((index[a], index[b]) for a, b in spec.conventions.exchange),
        spec.conventions.offset_window,
        tuple(windows),
    )
    record = {"columns": columns, "failures": failures, "datasets": datasets}
    return problem, record


def compile_calibration(
    protocol: Path,
    directory: Path,
    extras,
    repository: Path,
    *,
    retain_base=False,
    runs_root=Path("research_runs"),
    replay_source=None,
):
    """Freeze the spec, system inputs and datasets for the runner; this creates no attempt."""
    from entanglix.corrector.calibration import calibration_listing
    from entanglix.runner import capture_environment

    loaded = load_calibration(protocol, runs_root=runs_root, replay_source=replay_source)
    frozen = read_inputs(loaded.inputs, loaded.input_digests, "calibration")
    base = base_files(frozen, loaded.spec.system, "calibration") if retain_base else {}
    frozen["calibrate.toml"] = tomli_w.dumps(loaded.spec.frozen_document()).encode("utf-8")
    frozen.update(base)
    write_frozen(directory, frozen)
    # The job loads exactly these files, so they must validate as frozen.
    load_calibration(directory / "calibrate.toml")
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in frozen.items()}
    return {
        **loaded.catalog,
        "system_id": loaded.system.id,
        "inputs": {name: str(directory / name) for name in frozen},
        "spec_sha256": digests,
        "calibration_sha256": digests["calibrate.toml"],
        # A launch and its preview list the checks the run can take, as reads of it do.
        "preview": {**loaded.listing, **calibration_listing(loaded)},
        "base_inputs": loaded.input_digests if retain_base else {},
        "scope_values": [{}],
        "engine": ENGINE,
        "device": "cpu",
        "needs": {"gradients": False, "device": "cpu"},
        "environment": capture_environment(repository / "uv.lock", extras=tuple(extras)),
        "timeout_seconds": loaded.spec.timeout_seconds,
        "kind": "calibrate",
        "selection": {},
        "protocol_sha256": loaded.input_digests["calibrate.toml"],
    }


def calibrate(
    protocol: Path,
    *,
    runs_root: Path = Path("research_runs"),
    repository: Path | None = None,
    parent_run_ids=(),
    dry_run=False,
    investigation_id: str | None = None,
    base: str | None = None,
):
    """Fit device parameters to training spectroscopy as one owned CPU job."""
    from entanglix.protocols.simulate import _launch_protocol

    return _launch_protocol(
        protocol,
        operation="calibrate",
        runs_root=runs_root,
        repository=repository,
        parent_run_ids=parent_run_ids,
        dry_run=dry_run,
        investigation_id=investigation_id,
        base=base,
    )


def rerun_calibration(run: Path, *, repository: Path | None = None, runs_root: Path | None = None):
    """Re-fit retained calibration inputs as a new linked attempt under current terms."""
    from entanglix.protocols.simulate import _rerun_protocol

    return _rerun_protocol(run, operation="calibrate", repository=repository, runs_root=runs_root)
