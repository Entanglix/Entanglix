"""Uniform measurement datasets: validation, digests, and holdout splits."""

from __future__ import annotations

import hashlib
import io
import math
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from zipfile import BadZipFile

import numpy as np
import tomli_w
from pydantic import Field, ValidationError, model_validator

from entanglix.models import StrictModel

SHA256 = r"^[0-9a-f]{64}$"


class DatasetError(ValueError):
    """A measurement dataset cannot be validated or split as authored."""


class Converter(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class Provenance(StrictModel):
    device: str = Field(min_length=1)
    cooldown: str | None = None
    measured_at: datetime
    raw_file_sha256: str | None = Field(default=None, pattern=SHA256)
    converter: Converter

    @model_validator(mode="after")
    def check_time(self) -> Provenance:
        if self.measured_at.utcoffset() is None:
            raise ValueError("measured_at requires a UTC offset")
        return self


class Condition(StrictModel):
    value: float
    unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_value(self) -> Condition:
        if not math.isfinite(self.value):
            raise ValueError("condition value must be finite")
        return self


class Array(StrictModel):
    key: str = Field(min_length=1)
    unit: str = Field(min_length=1)


class Frequencies(StrictModel):
    key: str = Field(min_length=1)
    unit: Literal["GHz"]


class Labels(StrictModel):
    key: str = Field(min_length=1)


class MapLevel(StrictModel):
    signal: Array
    frequency: Frequencies
    bias: Array | None = None


class PointLevel(StrictModel):
    centre: Frequencies
    centre_uncertainty: Frequencies
    bias: Array | None = None
    linewidth: Frequencies | None = None
    amplitude: Array | None = None
    assignment: Labels | None = None


class Split(StrictModel):
    parent_sha256: str = Field(pattern=SHA256)
    part: Literal["holdout", "remainder"]
    lower: float
    upper: float
    columns: list[list[int]] = Field(default_factory=list)
    points: list[list[int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_record(self) -> Split:
        if not (
            math.isfinite(self.lower) and math.isfinite(self.upper) and self.lower < self.upper
        ):
            raise ValueError("split requires finite lower < upper")
        for name in ("columns", "points"):
            end = -1
            for item in getattr(self, name):
                if len(item) != 2 or not end < item[0] < item[1]:
                    raise ValueError(
                        f"{name}: expected increasing, separated [start, stop) index ranges"
                    )
                end = item[1]
        return self


class DatasetSpec(StrictModel):
    format_version: Literal[1]
    id: str = Field(min_length=1)
    kind: Literal["spectroscopy"]
    array_file: str = Field(min_length=1)
    conditions: dict[str, Condition | int | str]
    provenance: Provenance
    map: MapLevel | None = None
    points: PointLevel | None = None
    split: Split | None = None

    @model_validator(mode="after")
    def check_levels(self) -> DatasetSpec:
        levels = [level for level in (self.map, self.points) if level is not None]
        if not levels:
            raise ValueError("a dataset declares a map, points, or both")
        units = {level.bias.unit if level.bias is not None else None for level in levels}
        if len(units) > 1:
            raise ValueError("map and points must share one bias axis and unit, or have none")
        return self

    @property
    def bias_unit(self) -> str | None:
        level = self.map if self.map is not None else self.points
        return level.bias.unit if level.bias is not None else None


@dataclass(frozen=True)
class Dataset:
    spec: DatasetSpec
    arrays: dict[str, np.ndarray]
    files: dict[str, bytes]
    sha256: str
    raw_file_checked: bool = False


def dataset_sha256(sidecar: bytes, archive: bytes) -> str:
    """Identify a dataset by its TOML and NPZ bytes, independent of file names."""
    digest = hashlib.sha256()
    for content in (sidecar, archive):
        digest.update(str(len(content)).encode() + b"\0" + content)
    return digest.hexdigest()


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"cannot read {label} {path}: {exc}") from exc


def _parse(content: bytes) -> DatasetSpec:
    try:
        return DatasetSpec.model_validate(tomllib.loads(content.decode("utf-8")))
    except tomllib.TOMLDecodeError as exc:
        raise DatasetError(f"invalid TOML: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetError(f"dataset file must be UTF-8: {exc}") from exc
    except ValidationError as exc:
        issues = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
        )
        raise DatasetError(issues) from exc


def _references(spec: DatasetSpec) -> dict[str, str]:
    """Map each array role to its NPZ key."""
    references = {}
    for prefix, level in (("map", spec.map), ("points", spec.points)):
        if level is None:
            continue
        for name in type(level).model_fields:
            item = getattr(level, name)
            if item is not None:
                references[f"{prefix}.{name}"] = item.key
    return references


def _vector(value: np.ndarray, label: str, size: int | None = None) -> np.ndarray:
    if value.ndim != 1 or value.size == 0:
        raise DatasetError(f"{label}: expected a non-empty one-dimensional array")
    if size is not None and value.size != size:
        raise DatasetError(f"{label}: expected {size} values, matching points.centre")
    if value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)):
        raise DatasetError(f"{label}: expected finite real values")
    return np.asarray(value, dtype=np.float64)


def _increasing(value: np.ndarray, label: str) -> None:
    if np.any(np.diff(value) <= 0):
        raise DatasetError(f"{label}: values must be strictly increasing")


def _positive(value: np.ndarray, label: str) -> None:
    if np.any(value <= 0):
        raise DatasetError(f"{label}: values must be positive")


def _validated(spec: DatasetSpec, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays = {}
    if spec.map is not None:
        frequency = _vector(raw["map.frequency"], "map.frequency")
        _increasing(frequency, "map.frequency")
        arrays["map.frequency"] = frequency
        expected = (frequency.size,)
        if spec.map.bias is not None:
            bias = _vector(raw["map.bias"], "map.bias")
            _increasing(bias, "map.bias")
            arrays["map.bias"] = bias
            expected = (bias.size, frequency.size)
        signal = raw["map.signal"]
        if signal.shape != expected:
            raise DatasetError(
                f"map.signal: shape {signal.shape} does not match its axes, expected {expected}"
            )
        if signal.dtype.kind not in "iufc" or not np.all(np.isfinite(signal)):
            raise DatasetError("map.signal: expected finite numeric values")
        arrays["map.signal"] = signal
    if spec.points is not None:
        centre = _vector(raw["points.centre"], "points.centre")
        size = centre.size
        arrays["points.centre"] = centre
        for name in ("centre_uncertainty", "bias", "linewidth", "amplitude"):
            label = f"points.{name}"
            if label in raw:
                arrays[label] = _vector(raw[label], label, size)
        _positive(arrays["points.centre_uncertainty"], "points.centre_uncertainty")
        if "points.linewidth" in arrays:
            _positive(arrays["points.linewidth"], "points.linewidth")
        if "points.assignment" in raw:
            labels = raw["points.assignment"]
            if labels.ndim != 1 or labels.dtype.kind != "U" or labels.size != size:
                raise DatasetError(
                    f"points.assignment: expected {size} text labels, matching points.centre"
                )
            arrays["points.assignment"] = labels
    if spec.split is not None:
        counts = {
            "columns": arrays["map.bias"].size if "map.bias" in arrays else 0,
            "points": arrays["points.centre"].size if "points.centre" in arrays else 0,
        }
        for name, count in counts.items():
            recorded = sum(stop - start for start, stop in getattr(spec.split, name))
            if recorded != count:
                raise DatasetError(f"split.{name}: records {recorded} indices for {count} kept")
    return arrays


def _load_arrays(spec: DatasetSpec, content: bytes) -> dict[str, np.ndarray]:
    references = _references(spec)
    keys = list(references.values())
    if len(set(keys)) != len(keys):
        raise DatasetError("each array role needs its own NPZ key")
    try:
        archive = np.load(io.BytesIO(content), allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise DatasetError("array_file must be an NPZ archive")
        with archive:
            missing = sorted(set(keys) - set(archive.files))
            if missing:
                raise DatasetError(f"array_file: missing NPZ keys {missing}")
            unreferenced = sorted(set(archive.files) - set(keys))
            if unreferenced:
                raise DatasetError(f"array_file: unreferenced NPZ arrays {unreferenced}")
            raw = {role: archive[key] for role, key in references.items()}
            for role, value in raw.items():
                # NumPy returns a member's raw bytes when it is not a valid NPY array.
                if not isinstance(value, np.ndarray):
                    raise DatasetError(f"array_file: {references[role]!r} is not a valid NPY array")
    except DatasetError:
        raise
    except (OSError, ValueError, TypeError, BadZipFile) as exc:
        raise DatasetError(f"array_file: cannot read numeric NPZ: {exc}") from exc
    return _validated(spec, raw)


def _beside(source: Path, name: str) -> Path:
    path = Path(name)
    if path.name != name or path.suffix != ".npz":
        raise DatasetError("array_file must name an NPZ file beside the dataset file")
    return source.parent / path


def load_dataset(path: str | Path, *, raw_file: str | Path | None = None) -> Dataset:
    """Validate a dataset; a supplied raw file must match the declared digest."""
    source = Path(path).resolve()
    sidecar = _read(source, "dataset file")
    spec = _parse(sidecar)
    archive = _read(_beside(source, spec.array_file), "array_file")
    arrays = _load_arrays(spec, archive)
    if raw_file is not None:
        declared = spec.provenance.raw_file_sha256
        if declared is None:
            raise DatasetError(
                "a raw file was supplied but the dataset declares no raw_file_sha256"
            )
        try:
            with Path(raw_file).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as exc:
            raise DatasetError(f"cannot read raw file {raw_file}: {exc}") from exc
        if actual != declared:
            raise DatasetError(
                f"raw file digest {actual} does not match provenance.raw_file_sha256 {declared}"
            )
    return Dataset(
        spec,
        arrays,
        {source.name: sidecar, spec.array_file: archive},
        dataset_sha256(sidecar, archive),
        raw_file is not None,
    )


def dataset_report(dataset: Dataset) -> dict:
    """Summarize a validated dataset, stating what Entanglix did and did not verify."""
    spec = dataset.spec
    provenance = spec.provenance.model_dump(exclude_none=True)
    report = {
        "id": spec.id,
        "kind": spec.kind,
        "sha256": dataset.sha256,
        "bias_unit": spec.bias_unit,
        "provenance": provenance,
        "conversion": "recorded by its converter; not verified by Entanglix",
        "raw_file": "digest matched" if dataset.raw_file_checked else "not supplied",
    }
    arrays = dataset.arrays
    if spec.map is not None:
        report["map"] = {
            "shape": list(arrays["map.signal"].shape),
            "signal": "complex" if arrays["map.signal"].dtype.kind == "c" else "real",
            "frequency_range_ghz": [
                float(arrays["map.frequency"][0]),
                float(arrays["map.frequency"][-1]),
            ],
        }
    if spec.points is not None:
        labels = arrays.get("points.assignment")
        report["points"] = {
            "count": int(arrays["points.centre"].size),
            "labelled": 0 if labels is None else int(np.count_nonzero(labels != "")),
            "fields": sorted(
                name.removeprefix("points.") for name in arrays if name.startswith("points.")
            ),
        }
    if spec.split is not None:
        report["split"] = spec.split.model_dump()
    return report


def _ranges(indices: np.ndarray) -> list[list[int]]:
    ranges: list[list[int]] = []
    for index in indices.tolist():
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] += 1
        else:
            ranges.append([index, index + 1])
    return ranges


def split_dataset(
    path: str | Path, lower: float, upper: float, *, holdout: str | Path, remainder: str | Path
) -> dict:
    """Write the part inside a closed bias range and the rest as two new datasets."""
    parent = load_dataset(path)
    spec = parent.spec
    if spec.split is not None:
        raise DatasetError("dataset is already a split part; split the original dataset")
    if spec.bias_unit is None:
        raise DatasetError("a split needs a bias axis")
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise DatasetError("a split needs finite lower < upper")
    outputs = {"holdout": Path(holdout).resolve(), "remainder": Path(remainder).resolve()}
    if outputs["holdout"] == outputs["remainder"]:
        raise DatasetError("holdout and remainder need different paths")
    for name, output in outputs.items():
        if output.suffix != ".toml":
            raise DatasetError(f"{name}: expected a new .toml path")
        if not output.parent.is_dir():
            raise DatasetError(f"{name}: directory {output.parent} does not exist")
        for target in (output, output.with_suffix(".npz")):
            if target.exists():
                raise DatasetError(f"{name}: {target} already exists")
    arrays = parent.arrays
    inside = {
        level: (arrays[f"{level}.bias"] >= lower) & (arrays[f"{level}.bias"] <= upper)
        for level, declared in (("map", spec.map), ("points", spec.points))
        if declared is not None
    }
    # A level with no entries in a part is omitted from it; a part needs at least one level.
    selections = {}
    for part, chosen in (("holdout", True), ("remainder", False)):
        selections[part] = {
            level: indices
            for level, mask in inside.items()
            if (indices := np.flatnonzero(mask == chosen)).size
        }
        if not selections[part]:
            raise DatasetError(f"the {part} would be empty for the range [{lower}, {upper}]")
    keys = _references(spec)
    contents = {}
    for part, output in outputs.items():
        selected = selections[part]
        subset = {}
        for role, key in keys.items():
            level = role.split(".")[0]
            if level in selected:
                value = arrays[role]
                subset[key] = value if role == "map.frequency" else value[selected[level]]
        buffer = io.BytesIO()
        np.savez(buffer, **subset)
        document = spec.model_dump(exclude_none=True)
        for level in ("map", "points"):
            if level not in selected:
                document.pop(level, None)
        document["id"] = f"{spec.id}-{part}"
        document["array_file"] = output.with_suffix(".npz").name
        document["split"] = {
            "parent_sha256": parent.sha256,
            "part": part,
            "lower": float(lower),
            "upper": float(upper),
            "columns": _ranges(selected["map"]) if "map" in selected else [],
            "points": _ranges(selected["points"]) if "points" in selected else [],
        }
        contents[output.with_suffix(".npz")] = buffer.getvalue()
        contents[output] = tomli_w.dumps(document).encode()
    created = []
    try:
        for target, content in contents.items():
            with target.open("xb") as stream:
                created.append(target)
                stream.write(content)
        result = {"parent_sha256": parent.sha256}
        for part, output in outputs.items():
            result[part] = dataset_report(load_dataset(output))
    except BaseException:
        # Remove only the files this call created, so a retry starts clean.
        for target in created:
            target.unlink(missing_ok=True)
        raise
    return result
