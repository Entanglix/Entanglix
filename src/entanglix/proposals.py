"""Device-update proposals from calibration runs, and their application as a human action."""

import contextlib
import fcntl
import os
import pwd
import re
import stat
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import tomli_w

NOT_DETERMINED = "not numerically determined at the fit's tolerances"


def write_parameters(document: dict, values: dict[str, float], source: str) -> bytes:
    """Set values and their source in a parsed system file's parameter table.

    Units, priors and every other field are kept; the file is canonical TOML.
    """
    for name, value in values.items():
        if name not in document.get("parameters", {}):
            raise ValueError(f"{name!r} is not in the system file's parameter table")
        document["parameters"][name]["value"] = value
        document["parameters"][name]["source"] = source
    return tomli_w.dumps(document).encode("utf-8")


def _source(run: Path) -> tuple[dict, dict, dict]:
    """A completed calibrate run's manifest, result and model-input digests, checked."""
    from entanglix.models import load_system
    from entanglix.runner import (
        _digest,
        _read_attempt,
        _relative_file,
        _script_result,
        _system_inputs,
    )

    record = _read_attempt(run)
    manifest = record["manifest"]
    if manifest["protocol"] != "calibrate" or record["status"] != "completed":
        raise ValueError("a device-update proposal needs a completed calibrate run")
    frozen = manifest["spec_sha256"]
    for name, digest in frozen.items():
        if _digest((run / "spec" / _relative_file(name)).read_bytes()) != digest:
            raise ValueError(f"source input digest mismatch: {name}")
    result = _script_result(run)
    if not (
        result.get("kind") == manifest["launch"]["kind"] == "calibrate"
        and result.get("calibration_sha256")
        == manifest["launch"]["calibration_sha256"]
        == frozen["calibrate.toml"]
    ):
        raise ValueError("source result is inconsistent with its frozen calibration spec")
    system = run / "spec" / "system.toml"
    names = _system_inputs(system, load_system(system), reserved={"calibrate.toml"})[1]
    return manifest, result, {name: frozen[name] for name in names}


def _checks(run: Path) -> list[dict]:
    """Each verification attempt on the run, with its checks' solutions and outcomes."""
    from entanglix.corrector.verification import verification_history

    attempts = []
    for item in verification_history(run):
        attempt = {"run_id": Path(item["path"]).name, "status": item["status"]}
        for key in ("policy_version", "source_matches", "diagnostic"):
            if key in item:
                attempt[key] = item[key]
        if "checks" in item:
            attempt["checks"] = [
                {
                    "name": check["name"],
                    **({"solution": check["solution"]} if "solution" in check else {}),
                    "outcome": check.get("outcome", "descriptive"),
                }
                for check in item["checks"]
            ]
        attempts.append(attempt)
    return attempts


def propose_device_update(run: Path, *, solution: int | None = None) -> dict:
    """Write a proposal from one solution of a calibrate run, inside the run's directory."""
    from entanglix.protocols.calibrate import chosen_solution
    from entanglix.runner import _digest, publish_toml

    run = run.resolve()
    manifest, result, inputs = _source(run)
    number = chosen_solution(result, solution, "propose")
    if number is None:
        raise ValueError(f"the source run returned no solution ({result['outcome']})")
    fit = result["solutions"][number]
    declared = tomllib.loads((run / "spec" / "system.toml").read_text())["parameters"]
    proposed, not_proposed = {}, {}
    for name in result["parameters"]["fitted"]:
        if name not in fit["determined"]:
            not_proposed[name] = {"reason": NOT_DETERMINED}
            continue
        proposed[name] = {
            "unit": declared[name]["unit"],
            "old": declared[name]["value"],
            "new": fit["values"][name],
            "uncertainty": fit["uncertainty"][name],
            "uncertainty_over_prior_width": fit["uncertainty_over_prior_width"][name],
        }
    if not proposed:
        raise ValueError(f"no fitted parameter is numerically determined in solution {number}")
    record = {
        "record_version": 1,
        "kind": "device_update_proposal",
        "created_at": datetime.now(UTC),
        "source_run_id": manifest["run_id"],
        "system_id": manifest["system_id"],
        "solution": number,
        "solution_outcome": fit["outcome"],
        "bounds_hit": fit["bounds_hit"],
        "source_result_sha256": _digest((run / "result.toml").read_bytes()),
        "model_inputs": inputs,
        "parameters": proposed,
        "not_proposed": not_proposed,
        "verification": _checks(run),
    }
    directory = run / "proposals"
    directory.mkdir(exist_ok=True)
    taken = [int(p.stem) for p in directory.iterdir() if re.fullmatch(r"\d+\.toml", p.name)]
    index = max(taken, default=0) + 1
    while True:
        path = directory / f"{index}.toml"
        try:
            publish_toml(path, record)
            break
        except FileExistsError:
            index += 1
    return {"path": str(path), **record}


def _locked(directory: Path) -> int:
    """An exclusive lock on a directory, held until the returned descriptor is closed."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise ValueError(f"another apply is in progress in {directory}") from exc
    return descriptor


def _changed(system: Path, inputs: dict, applied: str | None = None) -> str | None:
    """The first model input beside the system file whose digest is not the frozen one.

    The system file may instead hold ``applied``, the content this proposal writes.
    """
    from entanglix.runner import _digest

    for name, digest in inputs.items():
        path = system if name == "system.toml" else system.parent / name
        current = _digest(path.read_bytes()) if path.is_file() else None
        if current != digest and not (
            name == "system.toml" and applied is not None and current == applied
        ):
            return name
    return None


def apply_device_update(proposal: Path, system: Path) -> dict:
    """Operator action: write a proposal's values and provenance into the unchanged system file.

    The system file, each subsystem source and each array file beside it must have the digests
    the proposal's source run froze. Priors, units and every other field are left unchanged.
    The new content is built from the run's frozen copy, so a file that already holds it, after
    an apply whose record was not published, only gains the record.
    """
    from entanglix.models import load_system
    from entanglix.runner import _digest, publish_toml

    proposal, system = proposal.resolve(), system.resolve()
    content = proposal.read_bytes()
    record = tomllib.loads(content.decode("utf-8"))
    if record.get("kind") != "device_update_proposal" or record.get("record_version") != 1:
        raise ValueError("not a device-update proposal")
    if proposal.parent.name != "proposals" or proposal.parents[1].name != record["source_run_id"]:
        raise ValueError("a proposal is applied from its calibration run's proposals directory")
    if system.is_relative_to(proposal.parents[3]):
        raise ValueError("apply writes an authored system file, not one in the runs root")
    inputs = record["model_inputs"]
    frozen = (proposal.parents[1] / "spec" / "system.toml").read_bytes()
    if _digest(frozen) != inputs["system.toml"]:
        raise ValueError("the source run's frozen system file changed")
    document = tomllib.loads(frozen.decode("utf-8"))
    reference = Path(*proposal.parts[-4:])
    source = (
        f"calibration proposal {reference}, solution {record['solution']}, "
        f"sha256 {_digest(content)}"
    )
    for name, item in record["parameters"].items():
        parameter = document["parameters"][name]
        if parameter["value"] != item["old"] or parameter["unit"] != item["unit"]:
            raise ValueError(f"the proposal does not match the system file's parameter {name!r}")
    updated = write_parameters(
        document, {name: item["new"] for name, item in record["parameters"].items()}, source
    )
    after = _digest(updated)
    applied = proposal.with_name(f"{proposal.stem}.applied.toml")
    with contextlib.ExitStack() as locks:
        # Competing applies to the same folder, or of the same proposal, are refused.
        for directory in (system.parent, proposal.parent):
            locks.callback(os.close, _locked(directory))
        if applied.exists():
            raise ValueError(f"the proposal was already applied: {applied}")
        if name := _changed(system, inputs, after):
            raise ValueError(f"{name} changed since the source run froze it")
        recovered = _digest(system.read_bytes()) == after
        if not recovered:
            load_system(system, content=updated)
            # An edit made while the update was validated is refused, not overwritten.
            if name := _changed(system, inputs):
                raise ValueError(f"{name} changed since the source run froze it")
            mode = stat.S_IMODE(system.stat().st_mode)
            publish_toml(system, document, replace=True)
            os.chmod(system, mode)
        try:
            user = pwd.getpwuid(os.getuid()).pw_name
        except KeyError:
            user = str(os.getuid())
        application = {
            "record_version": 1,
            "kind": "device_update_application",
            "applied_at": datetime.now(UTC),
            "source": {"kind": "operator-cli", "os_user": user},
            "proposal_sha256": _digest(content),
            "system": str(system),
            "system_sha256_before": inputs["system.toml"],
            "system_sha256_after": after,
            "parameters": list(record["parameters"]),
            **({"recovered": True} if recovered else {}),
        }
        try:
            publish_toml(applied, application)
        except OSError as exc:
            raise ValueError(
                f"{system} was updated, but its application record could not be published "
                f"({exc}); apply the same proposal to it again to record it"
            ) from exc
    return {"path": str(applied), **application}
