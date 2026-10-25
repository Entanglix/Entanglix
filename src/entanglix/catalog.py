"""Design staging, operator promotion, and catalog reads with source lineage."""

import errno
import io
import os
import pwd
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
ENTRY_ID = re.compile(r"e\d{8}T\d{12}Z-[0-9a-f]{8}")
RUN_ID = re.compile(r"r\d{8}T\d{12}Z-[0-9a-f]{8}")
# Each selector picks one design among several held by a run of this protocol.
SELECTORS = {"point": "sweep", "solution": "calibrate", "artifact": "script"}
# Checks on these computations bind the run's result and its primary artifact.
CHECKED = {"plan", "simulate", "sweep", "control", "calibrate"}
PULSE_ARRAYS = ("kind", "times_ns", "amplitudes_ghz", "channels")
BLOCKING_CHECKS = {"integrity", "segments", "spectral_refinement", "replay", "constraints"}


def device_source(identifier: str, runs_root: Path):
    """Resolve a promoted device and its catalog lineage."""
    from entanglix.runner import _load_record, _relative_file

    if not ENTRY_ID.fullmatch(identifier):
        raise ValueError("system_entry must name a promoted catalog entry")
    path = runs_root.resolve() / CATALOG / "entries" / identifier / "record.toml"
    if not path.is_file():
        raise ValueError(f"unknown catalog entry: {identifier}")
    entry = _load_record(path)
    if "device" not in entry:
        raise ValueError(f"catalog entry {identifier} holds no device")
    source = runs_root.resolve() / _relative_file(entry["device"]["path"]) / "system.toml"
    return source, {
        "source_entry_ids": [identifier],
        "parent_run_ids": [Path(entry["source"]["run"]).name],
    }


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


def _publish(
    root: Path, identifier: str, files: dict[str, bytes], record: dict, *, collection="staging"
) -> Path:
    """Write one immutable folder under a temporary name and rename it into place."""
    from entanglix.runner import publish_toml

    catalog = root / CATALOG
    (catalog / collection).mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=catalog))
    target = catalog / collection / identifier
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


def _covering_check(item: dict, source: dict) -> dict | None:
    """Select a point's own outcome, rather than a sweep's aggregate outcome."""
    if "point" in source:
        point = source["point"]
        if "points" in item:
            return next((row for row in item["points"] if row["index"] == point), None)
        if item.get("outcome") != "inapplicable" and point not in item.get("coverage", {}).get(
            "checked_points", []
        ):
            return None
    return item


def _promotion_checks(run: Path, source: dict) -> dict:
    """Record covering outcomes and unusable attempts, without executing a check."""
    from entanglix.corrector.verification import check_run, verification_history
    from entanglix.runner import _load_record, _relative_file

    names = [item["name"] for item in check_run(run)["checks"]]
    if not names:
        return {"checks": [], "checks_not_run": [], "unusable_evidence": []}
    checks, unusable, attempted = [], [], set()
    for attempt in verification_history(run):
        reference = str(Path(attempt["path"]).relative_to(run.parent.parent))
        request = None
        try:
            manifest = _load_record(Path(attempt["path"]) / "run.toml")
            request = _load_record(
                Path(attempt["path"]) / "spec" / _relative_file(manifest["launch"]["request_file"])
            )
        except (OSError, ValueError, KeyError):
            pass  # An unreadable request remains an unusable attempt below.
        if (
            request is not None
            and "solution" in source
            and request.get("solution") != source["solution"]
        ):
            continue
        if attempt["status"] != "completed" or not attempt.get("source_matches"):
            item = {"run": reference, **{k: v for k, v in attempt.items() if k != "path"}}
            # An unusable attempt is history, not evidence that a check was never requested.
            if request is not None and "checks" in request:
                item["requested_checks"] = request["checks"]
                attempted.update(request["checks"])
            unusable.append(item)
            continue
        for original in attempt["checks"]:
            item = _covering_check(original, source)
            if item is None:
                continue
            name = original["name"]
            attempted.add(name)
            outcome = item.get("outcome", "descriptive")
            if name in BLOCKING_CHECKS and outcome == "fail":
                raise ValueError(f"promotion refused: {name} failed in {reference}")
            checks.append(
                {
                    "name": name,
                    "outcome": outcome,
                    "run": reference,
                    "policy_version": attempt["policy_version"],
                }
            )
    return {
        "checks": checks,
        "checks_not_run": [name for name in names if name not in attempted],
        "unusable_evidence": unusable,
    }


def promote(identifier: str, runs_root: Path = Path("research_runs")) -> dict:
    """Operator-only action: publish one promoted entry from an unchanged staged design."""
    from entanglix.corrector import _source_artifact
    from entanglix.runner import (
        _digest,
        _load_record,
        _read_attempt,
        _relative_file,
        _script_result,
    )

    if not isinstance(identifier, str) or not STAGED_ID.fullmatch(identifier):
        raise ValueError(f"not a staged candidate ID: {identifier!r}")
    root = runs_root.resolve()
    staged_path = root / CATALOG / "staging" / identifier
    staged = _load_record(staged_path / "record.toml")
    entry_id = "e" + identifier[1:]
    if (root / CATALOG / "entries" / entry_id).exists():
        raise ValueError(f"candidate {identifier} is already promoted as {entry_id}")
    run = root / _relative_file(staged["source"]["run"])
    if not (run / "run.toml").is_file():
        raise ValueError(f"promotion source is unavailable: {staged['source']['run']}")
    source = _read_attempt(run)
    if source["status"] != "completed":
        raise ValueError("promotion needs a completed source run")
    result = _script_result(run)
    evidence = staged["evidence"]
    if _digest((run / "result.toml").read_bytes()) != evidence["result_sha256"]:
        raise ValueError("source result changed since staging")
    if (
        "artifact_sha256" in evidence
        and _source_artifact(result)["sha256"] != evidence["artifact_sha256"]
    ):
        raise ValueError("source artifact changed since staging")
    for name, digest in evidence["source_files"].items():
        if _digest((root / _relative_file(name)).read_bytes()) != digest:
            raise ValueError(f"source file changed since staging: {name}")
    files = {}
    folder = f"{CATALOG}/entries/{entry_id}"
    entry = {**staged, "kind": "catalog_entry", "entry_id": entry_id}
    if "device" in staged:
        for name in staged["device"]["files"]:
            files[f"device/{name}"] = (staged_path / "device" / _relative_file(name)).read_bytes()
        entry["device"] = {**staged["device"], "path": f"{folder}/device"}
    if "pulse" in staged:
        files["pulse.npz"] = (staged_path / "pulse.npz").read_bytes()
        entry["pulse"] = {**staged["pulse"], "path": f"{folder}/pulse.npz"}
    entry.update(_promotion_checks(run, staged["source"]))
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        user = str(os.getuid())
    entry.update(promoted_at=datetime.now(UTC), promoter={"kind": "operator-cli", "os_user": user})
    try:
        path = _publish(root, entry_id, files, entry, collection="entries")
    except OSError as exc:
        if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise ValueError(f"candidate {identifier} is already promoted as {entry_id}") from exc
        raise
    return {"path": str(path), **entry}


def list_catalog(runs_root: Path = Path("research_runs")) -> list[dict]:
    """Staged candidates and promoted entries, in creation order."""
    from entanglix.runner import _load_record

    items = []
    for path in sorted((runs_root / CATALOG).glob("*/*/record.toml")):
        collection = path.parents[1].name
        if collection not in {"staging", "entries"}:
            continue
        try:
            record = _load_record(path)
            identifier = record["entry_id"] if collection == "entries" else record["staged_id"]
            items.append(
                {
                    "id": identifier,
                    "status": "promoted" if collection == "entries" else "staged",
                    "path": f"{CATALOG}/{collection}/{path.parent.name}",
                    "design": record["design"],
                    "source_run": record["source"]["run"],
                    "claim": record["claim"],
                    "created_at": record["created_at"],
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            items.append({"id": path.parent.name, "status": "invalid", "diagnostic": str(exc)})
    return sorted(items, key=lambda item: item["id"][1:])


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
    """A staged candidate or promoted entry, with its source run and ancestors."""
    from entanglix.runner import _load_record

    if not isinstance(identifier, str) or not (
        STAGED_ID.fullmatch(identifier) or ENTRY_ID.fullmatch(identifier)
    ):
        raise ValueError(f"not a staged candidate ID or entry ID: {identifier!r}")
    collection = "staging" if identifier.startswith("s") else "entries"
    path = runs_root / CATALOG / collection / identifier / "record.toml"
    if not path.is_file():
        raise ValueError(
            f"no {'staged candidate' if collection == 'staging' else 'entry'} {identifier} under {runs_root}"
        )
    record = _load_record(path)
    return {
        "path": f"{CATALOG}/{collection}/{identifier}",
        "status": "staged" if collection == "staging" else "promoted",
        **record,
        "lineage": _lineage(runs_root, record["source"]["run"]),
        **({"reuses": _reuses(identifier, runs_root)} if collection == "entries" else {}),
    }


def _reuses(identifier: str, root: Path) -> dict:
    """Find recorded reuse and base references without loading scientific outputs."""
    from entanglix.runner import _load_record

    runs, bases = [], []
    for path in sorted(root.glob("*/*/run.toml")):
        try:
            manifest = _load_record(path)
            if identifier in manifest.get("source_entry_ids", []):
                runs.append(
                    {
                        "run": str(path.parent.relative_to(root)),
                        "protocol": manifest["protocol"],
                        **(
                            {"investigation_id": manifest["investigation_id"]}
                            if "investigation_id" in manifest
                            else {}
                        ),
                    }
                )
        except (OSError, ValueError, KeyError):
            continue
    for path in sorted((root / "_investigations").glob("*/revisions/*.toml")):
        try:
            revision = _load_record(path)
            for base in revision["terms"]["scope"]["bases"]:
                if base.get("system_entry") == identifier:
                    bases.append(
                        {
                            "investigation_id": path.parents[1].name,
                            "revision": revision["revision"],
                            "base": base["name"],
                        }
                    )
        except (OSError, ValueError, KeyError):
            continue
    return {"runs": runs, "investigation_bases": bases}
