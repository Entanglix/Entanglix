"""Staged design candidates in the runs root's catalog, and reads of their lineage."""

import io
import os
import re
import secrets
import shutil
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

CATALOG = "_catalog"
DESIGNS = ("device", "pulse", "device_and_pulse")
STAGED_ID = re.compile(r"s\d{8}T\d{12}Z-[0-9a-f]{8}")
RUN_ID = re.compile(r"r\d{8}T\d{12}Z-[0-9a-f]{8}")
# Each selector picks one design among several held by a run of this protocol.
SELECTORS = {"point": "sweep", "solution": "calibrate", "artifact": "script"}
# Checks on these computations bind the run's result and its primary artifact.
CHECKED = {"plan", "simulate", "sweep", "control", "calibrate"}
PULSE_ARRAYS = ("kind", "times_ns", "amplitudes_ghz", "channels")


def _neutral_pulse(path: Path, channels: list[str] | None = None) -> dict:
    """A pulse's arrays in the neutral form, unchanged; ``channels`` names a control run's."""
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("a neutral pulse is an NPZ archive of named arrays")
    with loaded as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    expected = set(PULSE_ARRAYS) - ({"channels"} if channels is not None else set())
    if set(arrays) != expected:
        raise ValueError(f"a neutral pulse holds exactly {sorted(expected)}")
    if channels is not None:
        arrays["channels"] = np.array(channels)
    kind, times, amplitudes, names = (arrays[name] for name in PULSE_ARRAYS)
    if kind.shape != () or str(kind.item()) != "zero_order":
        raise ValueError("a neutral pulse has kind zero_order")
    if (
        times.ndim != 1
        or times.size < 2
        or times.dtype.kind not in "fiu"
        or not np.isfinite(times).all()
        or times[0] != 0
        # Adjacent values are compared directly: a difference wraps for unsigned integers.
        or not (times[1:] > times[:-1]).all()
    ):
        raise ValueError("a neutral pulse's times_ns increase from zero over at least two knots")
    if (
        amplitudes.ndim != 2
        or amplitudes.shape[0] < 1
        or amplitudes.shape[1] != times.size
        or amplitudes.dtype.kind not in "fiu"
        or not np.isfinite(amplitudes).all()
    ):
        raise ValueError("a neutral pulse's amplitudes_ghz are finite, one row per channel")
    if (
        names.ndim != 1
        or names.dtype.kind != "U"
        or names.size != amplitudes.shape[0]
        or len(set(names.tolist())) != names.size
        or not all(names.tolist())
    ):
        raise ValueError("a neutral pulse names each channel once, one name per amplitude row")
    return {name: arrays[name] for name in PULSE_ARRAYS}


def _device(run: Path, system: Path, prefix: str, content: bytes | None = None):
    """A system file and every file it references, by name, with their source digests."""
    from entanglix.models import load_system
    from entanglix.runner import _digest, _relative_file, _system_inputs

    constructed = load_system(system, content=content)
    paths, _ = _system_inputs(system, constructed, reserved=set())
    files, sources = {}, {}
    for name, path in paths.items():
        _relative_file(name)
        if not path.resolve().is_relative_to(system.parent.resolve()):
            raise ValueError(f"a device file must stay beside its system file: {name}")
        data = path.read_bytes()
        sources[f"{prefix}/{path.relative_to(run)}"] = _digest(data)
        files[name] = content if name == "system.toml" and content is not None else data
    return files, sources


def _sweep_point(run: Path, point: int | None) -> tuple[Path, dict]:
    """A sweep point's frozen system, which holds its parameter values, and the point's values."""
    from entanglix.protocols.sweep import SweepSpec
    from entanglix.runner import _load_record

    if point is None:
        raise ValueError("a sweep holds one device per point; name the point")
    manifest = _load_record(run / "spec/sweep-manifest.toml")
    if type(point) is not int or not 0 <= point < len(manifest["points"]):
        raise ValueError(f"the sweep has no point {point!r}")
    spec = SweepSpec.model_validate(_load_record(run / "spec/sweep.toml"))
    coordinate = manifest["points"][point]["coordinate"]
    values = {}
    for axis, offset in zip(spec.axes, coordinate, strict=True):
        values[axis.target] = axis.values[offset]
        values.update({target: linked[offset] for target, linked in axis.linked_values.items()})
    context = {"point": point, "coordinate": coordinate, "values": values}
    return run / "spec/points" / f"{point:06d}" / "system.toml", context


def _artifact(run: Path, result: dict, name: str | None) -> Path:
    if name is None:
        raise ValueError("name the artifact a script published")
    if name not in result.get("artifacts", {}):
        raise ValueError(f"the run published no artifact {name!r}")
    # The result check has kept the path inside artifacts/ and its digest current.
    return (run / result["artifacts"][name]["path"]).resolve()


def _publish(root: Path, identifier: str, files: dict[str, bytes], record: dict) -> Path:
    """Write one immutable folder under a temporary name and rename it into staging."""
    from entanglix.runner import publish_toml

    catalog = root / CATALOG
    (catalog / "staging").mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=catalog))
    target = catalog / "staging" / identifier
    try:
        for name, data in files.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o444)
        publish_toml(temporary / "record.toml", record)
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return target


def register_result(
    run: Path,
    design: str,
    *,
    claim: str,
    point: int | None = None,
    solution: int | None = None,
    artifact: str | None = None,
) -> dict:
    """Stage a design from a completed run in its runs root's catalog; the run is unchanged."""
    from entanglix.corrector import _source_artifact
    from entanglix.proposals import write_parameters
    from entanglix.protocols.calibrate import chosen_solution
    from entanglix.runner import (
        _digest,
        _load_record,
        _read_attempt,
        _relative_file,
        _script_result,
    )

    if design not in DESIGNS:
        raise ValueError(f"design must be one of {', '.join(DESIGNS)}")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("a candidate needs a claim note")
    run = run.resolve()
    record = _read_attempt(run)
    manifest = record["manifest"]
    protocol = manifest["protocol"]
    if record["status"] != "completed":
        raise ValueError(f"a candidate needs a completed run; this attempt is {record['status']}")
    if protocol == "verification":
        raise ValueError("a verification run holds no design")
    if (run.parent.name, run.name) != (manifest["system_id"], manifest["run_id"]):
        raise ValueError("the run is not at <runs_root>/<system_id>/<run_id>")
    for name, value in (("point", point), ("solution", solution), ("artifact", artifact)):
        if value is not None and protocol != SELECTORS[name]:
            raise ValueError(f"{name} selects a design of a {SELECTORS[name]} run")
    for name, digest in manifest["spec_sha256"].items():
        if _digest((run / "spec" / _relative_file(name)).read_bytes()) != digest:
            raise ValueError(f"source input digest mismatch: {name}")
    result = _script_result(run)
    result_sha256 = _digest((run / "result.toml").read_bytes())
    prefix = f"{manifest['system_id']}/{manifest['run_id']}"
    created = datetime.now(UTC)
    identifier = f"s{created.strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(4)}"
    folder = f"{CATALOG}/staging/{identifier}"
    selected, context, files, sources = {}, {}, {}, {}
    entry = {}

    if design == "device_and_pulse" and protocol != "control":
        raise ValueError("a device and a pulse together come from a control run")
    if design in {"device", "device_and_pulse"}:
        content = None
        device = {"path": f"{folder}/device"}
        if protocol == "sweep":
            system, context["sweep"] = _sweep_point(run, point)
            selected["point"] = point
        elif protocol == "calibrate":
            number = chosen_solution(result, solution, "stage")
            if number is None:
                raise ValueError(f"the source run returned no solution ({result['outcome']})")
            fit = result["solutions"][number]
            fitted = result["parameters"]["fitted"]
            system = run / "spec/system.toml"
            # Every fitted device value is written, so the device is the one its checks evaluated.
            written = {name: fit["values"][name] for name in fitted}
            content = write_parameters(
                tomllib.loads(system.read_text()),
                written,
                f"calibration run {prefix}, solution {number}, result sha256 {result_sha256}",
            )
            selected["solution"] = number
            device["written"] = written
            device["undetermined"] = [name for name in fitted if name not in fit["determined"]]
            context["calibration"] = {
                "solution": number,
                "outcome": fit["outcome"],
                "fit_report": f"{prefix}/result.toml",
            }
            if "flux" in result["parameters"]:
                context["calibration"]["axis_map"] = {
                    "flux": result["parameters"]["flux"],
                    **{name: fit["values"][name] for name in result["parameters"]["nuisance"]},
                }
        elif protocol == "script" and artifact is not None:
            system = _artifact(run, result, artifact)
            selected["artifact"] = artifact
        else:
            system = run / "spec/system.toml"
        device_files, device_sources = _device(run, system, prefix, content)
        files.update({f"device/{name}": data for name, data in device_files.items()})
        sources.update(device_sources)
        device["files"] = {name: _digest(data) for name, data in device_files.items()}
        entry["device"] = device
    if design in {"pulse", "device_and_pulse"}:
        if protocol == "control":
            path = (run / result["artifacts"]["pulse"]["path"]).resolve()
            channels = [
                channel["id"] for channel in _load_record(run / "spec/control.toml")["channels"]
            ]
            arrays = _neutral_pulse(path, channels)
        elif protocol == "script":
            path = _artifact(run, result, artifact)
            selected["artifact"] = artifact
            arrays = _neutral_pulse(path)
        else:
            raise ValueError(f"a {protocol} run holds no pulse")
        sources[f"{prefix}/{path.relative_to(run)}"] = _digest(path.read_bytes())
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        files["pulse.npz"] = buffer.getvalue()
        entry["pulse"] = {
            "path": f"{folder}/pulse.npz",
            "sha256": _digest(files["pulse.npz"]),
            "channels": arrays["channels"].tolist(),
            "knots": int(arrays["times_ns"].size),
            "duration_ns": float(arrays["times_ns"][-1]),
        }

    evidence = {"result_sha256": result_sha256}
    if protocol in CHECKED:
        evidence["artifact_sha256"] = _source_artifact(result)["sha256"]
    evidence["source_files"] = sources
    staged = {
        "record_version": 1,
        "kind": "staged_candidate",
        "staged_id": identifier,
        "created_at": created,
        "design": design,
        "claim": claim,
        "source": {
            "run": prefix,
            "protocol": protocol,
            **(
                {"investigation_id": manifest["investigation_id"]}
                if "investigation_id" in manifest
                else {}
            ),
            **selected,
        },
        "parent_entry_ids": manifest.get("source_entry_ids", []),
        "evidence": evidence,
        **entry,
        **({"context": context} if context else {}),
    }
    path = _publish(run.parent.parent, identifier, files, staged)
    return {"path": str(path), **staged}


def list_catalog(runs_root: Path = Path("research_runs")) -> list[dict]:
    """Staged candidates in creation order, with their designs, sources and claims."""
    from entanglix.runner import _load_record

    items = []
    for path in sorted((runs_root / CATALOG / "staging").glob("*/record.toml")):
        try:
            record = _load_record(path)
            items.append(
                {
                    "id": record["staged_id"],
                    "status": "staged",
                    "path": f"{CATALOG}/staging/{path.parent.name}",
                    "design": record["design"],
                    "source_run": record["source"]["run"],
                    "claim": record["claim"],
                    "created_at": record["created_at"],
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            items.append({"id": path.parent.name, "status": "invalid", "diagnostic": str(exc)})
    return items


def _lineage(root: Path, source: str) -> dict:
    """The source run and its ancestors through parent_run_ids; missing runs are unavailable."""
    from entanglix.runner import _read_attempt

    system_id, run_id = source.split("/")
    order, seen, ancestors = [(system_id, run_id)], {run_id}, []
    for system_id, run_id in order:
        candidates = [root / system_id / run_id, *sorted(root.glob(f"*/{run_id}"))]
        path = next((path for path in candidates if (path / "run.toml").is_file()), None)
        if path is None:
            ancestors.append({"run_id": run_id, "status": "unavailable"})
            continue
        try:
            attempt = _read_attempt(path)
        except (OSError, ValueError, KeyError) as exc:
            ancestors.append({"run_id": run_id, "status": "invalid", "diagnostic": str(exc)})
            continue
        manifest = attempt["manifest"]
        parents = [item for item in manifest["parent_run_ids"] if RUN_ID.fullmatch(item)]
        ancestors.append(
            {
                "run": f"{path.parent.name}/{path.name}",
                "run_id": run_id,
                "protocol": manifest["protocol"],
                "status": attempt["status"],
                "parent_run_ids": manifest["parent_run_ids"],
            }
        )
        for parent in parents:
            if parent not in seen:
                seen.add(parent)
                order.append((path.parent.name, parent))
    return {"source_run": ancestors[0], "ancestors": ancestors[1:]}


def read_entry(identifier: str, runs_root: Path = Path("research_runs")) -> dict:
    """A staged candidate's record, with its source run and that run's ancestors."""
    from entanglix.runner import _load_record

    if not isinstance(identifier, str) or not STAGED_ID.fullmatch(identifier):
        raise ValueError(f"not a staged candidate ID: {identifier!r}")
    path = runs_root / CATALOG / "staging" / identifier / "record.toml"
    if not path.is_file():
        raise ValueError(f"no staged candidate {identifier} under {runs_root}")
    record = _load_record(path)
    return {
        "path": f"{CATALOG}/staging/{identifier}",
        "status": "staged",
        **record,
        "lineage": _lineage(runs_root, record["source"]["run"]),
    }
