"""Immutable investigation terms, operator decisions, and research findings."""

import fcntl
import math
import os
import pwd
import re
import secrets
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from entanglix.corrector import CHECK_DEFINITIONS
from entanglix.runner import (
    _load_record,
    _read_attempt,
    _relative_file,
    failure_message,
    publish_toml,
)

Text = Annotated[str, Field(min_length=1)]
Name = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")]
PositiveCount = Annotated[int, Field(strict=True, gt=0)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveSeconds = Annotated[Number, Field(gt=0)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Operation = Literal["simulate", "sweep", "check", "run_script"]
TARGET = re.compile(
    r"^(parameters\.[^.]+|evolution\.drive_amplitudes_ghz\.[^.]+|"
    r"evolution\.carriers\.[^.]+\.frequency_ghz|evolution\.frame\.frequencies_ghz\.[^.]+)$"
)


class Fields(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Variation(Fields):
    target: Text
    values: list[Number] | None = Field(default=None, min_length=1)
    range: tuple[Number, Number] | None = None
    linked_values: dict[Text, list[Number]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_region(self):
        if not all(TARGET.fullmatch(name) for name in [self.target, *self.linked_values]):
            raise ValueError("variation targets must use the supported sweep target syntax")
        if (self.values is None) == (self.range is None):
            raise ValueError("declare exactly one of values or range")
        if self.range is not None and self.range[0] > self.range[1]:
            raise ValueError("range must be ordered lower to upper")
        if self.values is not None and len(set(self.values)) != len(self.values):
            raise ValueError("values must be unique")
        if self.linked_values and (
            self.values is None
            or any(len(v) != len(self.values) for v in self.linked_values.values())
        ):
            raise ValueError("linked_values require one value per entry in values")
        return self


class BaseSpec(Fields):
    name: Name
    inputs: dict[Text, Digest] = Field(min_length=1)
    variations: list[Variation] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_inputs(self):
        if "system.toml" not in self.inputs:
            raise ValueError("every base requires a system.toml digest")
        paths = [_relative_file(name) for name in self.inputs]
        if len(set(paths)) != len(paths) or any(
            parent in paths for path in paths for parent in path.parents
        ):
            raise ValueError("base inputs have duplicate paths or file/directory conflicts")
        targets = [name for v in self.variations for name in [v.target, *v.linked_values]]
        if len(set(targets)) != len(targets):
            raise ValueError("each varied target belongs to exactly one variation")
        return self


class Scope(Fields):
    operations: list[Operation] = Field(min_length=1)
    bases: list[BaseSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_entries(self):
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("permitted operations must be unique")
        if len({base.name for base in self.bases}) != len(self.bases):
            raise ValueError("base names must be unique")
        return self


class Checkpoint(Fields):
    finished_attempts: PositiveCount | None = None
    elapsed_seconds: PositiveSeconds | None = None

    @model_validator(mode="after")
    def one_interval(self):
        if (self.finished_attempts is None) == (self.elapsed_seconds is None):
            raise ValueError("checkpoint requires exactly one count or elapsed interval")
        return self


class Limits(Fields):
    job_count: PositiveCount | None = None
    job_seconds: PositiveSeconds | None = None
    deadline: datetime | None = None

    @field_validator("deadline", mode="before")
    @classmethod
    def timestamp_input(cls, value):
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    "deadline must be an absolute timestamp with a UTC offset"
                ) from exc
        if value is not None and not isinstance(value, datetime):
            raise ValueError("deadline must be an absolute timestamp with a UTC offset")
        return value

    @model_validator(mode="after")
    def absolute_deadline(self):
        if self.deadline is not None and self.deadline.utcoffset() is None:
            raise ValueError("deadline must include a UTC offset")
        return self


class Evidence(Fields):
    requirements: list[Text] = Field(min_length=1)
    checks: dict[Name, list[Text]] = Field(default_factory=dict)


class InvestigationSpec(Fields):
    format_version: Literal[1]
    question: Text
    assumptions: list[Text]
    scope: Scope
    evidence_requirements: Evidence
    checkpoint: Checkpoint
    review_points: dict[Name, PositiveCount] = Field(default_factory=dict)
    review_criteria: list[Text]
    stop_criteria: list[Text]
    limits: Limits

    @model_validator(mode="after")
    def evidence_bases(self):
        unknown = self.evidence_requirements.checks.keys() - {b.name for b in self.scope.bases}
        if unknown:
            raise ValueError(f"requested checks name unknown bases: {sorted(unknown)}")
        for names in self.evidence_requirements.checks.values():
            unknown_checks = set(names) - CHECK_DEFINITIONS.keys()
            if unknown_checks:
                raise ValueError(f"unknown requested checks: {sorted(unknown_checks)}")
            if len(set(names)) != len(names):
                raise ValueError("requested checks must be unique within each base")
        return self


class Finding(Fields):
    kind: Literal["assessment", "interpretation", "report"]
    runs: list[Text]
    learned: Text
    unresolved: list[Text]
    decision: Literal["continue", "refine", "stop", "review"]
    rationale: Text
    next_experiment: str
    review_name: Name | None = None

    @model_validator(mode="after")
    def valid_references(self):
        if len(set(self.runs)) != len(self.runs):
            raise ValueError("finding run references must be unique")
        if (self.decision == "review") != (self.review_name is not None):
            raise ValueError("review_name is required only for a review decision")
        return self


def _terms(spec: Path | dict) -> dict:
    data = _load_record(spec) if isinstance(spec, Path) else spec
    return InvestigationSpec.model_validate(data).model_dump(exclude_none=True)


@contextmanager
def investigation_lock(path: Path, *, shared: bool = False):
    """Serialize publications and read a consistent snapshot using the directory itself."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _read_locked(path: Path) -> dict:
    revisions = [_load_record(file) for file in sorted((path / "revisions").glob("*.toml"))]
    if not revisions:
        raise ValueError("investigation has no revisions")
    events = [_load_record(file) for file in sorted((path / "events").glob("*.toml"))]
    for record in revisions + events:
        if record["record_version"] != 1:
            raise ValueError("unsupported investigation record_version")
    active = None
    for event in events:
        if event["kind"] == "decision" and event["action"] in {"start", "approve-revision"}:
            active = event["revision"]
    return {
        "path": str(path),
        "investigation_id": path.name,
        "state": "draft" if active is None else "active",
        "active_revision": active,
        "revisions": revisions,
        "events": events,
    }


def _event(path: Path, kind: str, **fields) -> dict:
    sequence = 1 + max((int(file.stem) for file in (path / "events").glob("*.toml")), default=0)
    record = {
        "record_version": 1,
        "event_id": f"{sequence:08d}",
        "recorded_at": datetime.now(UTC),
        "kind": kind,
        **fields,
    }
    publish_toml(path / "events" / f"{sequence:08d}.toml", record)
    return record


def create_investigation(spec: Path | dict, *, runs_root: Path = Path("research_runs")) -> dict:
    """Write a draft, without creating a user decision or an active revision."""
    terms = _terms(spec)
    created = datetime.now(UTC)
    name = f"i{created.strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(4)}"
    path = (runs_root / "_investigations" / name).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    with investigation_lock(path):
        (path / "revisions").mkdir()
        (path / "events").mkdir()
        publish_toml(
            path / "revisions/0001.toml",
            {"record_version": 1, "revision": 1, "created_at": created, "terms": terms},
        )
    return read_investigation(path)


def revise_investigation(path: Path, spec: Path | dict) -> dict:
    """Append proposed terms; the current effective revision is unchanged."""
    path = path.resolve()
    terms = _terms(spec)
    with investigation_lock(path):
        record = _read_locked(path)
        revision = max(item["revision"] for item in record["revisions"]) + 1
        publish_toml(
            path / "revisions" / f"{revision:04d}.toml",
            {
                "record_version": 1,
                "revision": revision,
                "created_at": datetime.now(UTC),
                "terms": terms,
            },
        )
    return {"path": str(path), "investigation_id": path.name, "proposed_revision": revision}


def record_user_decision(path: Path, action: str, *, revision: int, note: str = "") -> dict:
    """Operator-only backing operation; research tools must never expose this function."""
    if action not in {"start", "approve-revision"}:
        raise ValueError("available decision actions are start and approve-revision")
    if type(revision) is not int or revision < 1:
        raise ValueError("revision must be a positive integer")
    path = path.resolve()
    with investigation_lock(path):
        record = _read_locked(path)
        if not any(item["revision"] == revision for item in record["revisions"]):
            raise ValueError(f"unknown revision: {revision}")
        active = record["active_revision"]
        if action == "start" and active is not None:
            raise ValueError("investigation has already started")
        if action == "approve-revision" and active is None:
            raise ValueError("a draft must be started before approving another revision")
        if any(
            event["kind"] == "decision" and event["revision"] == revision
            for event in record["events"]
        ):
            raise ValueError("revision has already taken effect")
        uid = os.getuid()
        try:
            os_user = pwd.getpwuid(uid).pw_name
        except KeyError:
            os_user = str(uid)
        _event(
            path,
            "decision",
            action=action,
            revision=revision,
            source={"kind": "operator-cli", "os_user": os_user},
            note=note,
        )
    return read_investigation(path)


def investigation_path(runs_root, investigation_id):
    if not re.fullmatch(r"i[0-9TZ]+-[0-9a-f]{8}", investigation_id):
        raise ValueError("invalid investigation ID")
    return runs_root.resolve() / "_investigations" / investigation_id


def _runs(path):
    """Read linked manifests and executor records without loading scientific models."""
    records = []
    for manifest_path in sorted(path.parent.parent.glob("*/*/run.toml")):
        manifest = _load_record(manifest_path)
        if manifest.get("investigation_id") != path.name:
            continue
        run = manifest_path.parent
        terminal = run / "FINISHED"
        finished = _load_record(terminal) if terminal.exists() else None
        records.append((run, manifest, finished))
    return records


def _usage(records):
    usage = {
        "jobs": len(records),
        "evaluated_points": 0,
        "job_seconds": 0.0,
        "gpu_seconds": 0.0,
        "committed_job_seconds": 0.0,
        "unsettled_jobs": 0,
        "unbounded_jobs": 0,
    }
    for run, manifest, finished in records:
        if finished is not None and not finished["usage"].get("uncertain", False):
            usage["job_seconds"] += finished["usage"]["wall_seconds"]
            usage["gpu_seconds"] += finished["usage"]["gpu_seconds"]
        else:
            usage["unsettled_jobs"] += 1
            usage["unbounded_jobs"] += int(
                manifest["investigation_commitment"].get("unbounded", False)
            )
            usage["committed_job_seconds"] += manifest["investigation_commitment"]["job_seconds"]
        if manifest["protocol"] == "sweep":
            usage["evaluated_points"] += sum(
                1 for _ in (run / "artifacts/points").glob("*/result.toml")
            )
        elif manifest["protocol"] == "simulate" and finished and finished["status"] == "completed":
            usage["evaluated_points"] += 1
    return usage


def _in_scope(base, hashes, points, defaults):
    if hashes != base["inputs"]:
        expected = base["inputs"]
        differences = []
        for label, names in (
            ("missing names", expected.keys() - hashes.keys()),
            ("unexpected names", hashes.keys() - expected.keys()),
            (
                "changed digests",
                {
                    name
                    for name in hashes.keys() & expected.keys()
                    if hashes[name] != expected[name]
                },
            ),
        ):
            if names:
                differences.append(f"{label}: {', '.join(sorted(names))}")
        raise ValueError(
            f"base input digests do not match: {base['name']}; {'; '.join(differences)}"
        )
    variations = base["variations"]
    allowed = {target for item in variations for target in [item["target"], *item["linked_values"]]}
    for point in points:
        if point.keys() - allowed:
            raise ValueError("varied values name targets outside the investigation scope")
        effective = dict(point)
        for target in allowed:
            if target not in effective and defaults is not None:
                cursor = defaults
                try:
                    for field in target.split("."):
                        cursor = cursor[field]
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"scope target is not declared in base: {target}") from exc
                effective[target] = cursor
        point = effective
        for variation in variations:
            target = variation["target"]
            if target not in point:
                continue
            value = point[target]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"varied value must be finite: {target}")
            if "values" in variation:
                if value not in variation["values"]:
                    raise ValueError(f"varied value outside permitted set: {target}")
                index = variation["values"].index(value)
                for linked, values in variation["linked_values"].items():
                    if point.get(linked) != values[index]:
                        raise ValueError(f"linked value does not match {target}: {linked}")
            elif not variation["range"][0] <= value <= variation["range"][1]:
                raise ValueError(f"varied value outside permitted range: {target}")
        for variation in variations:
            if variation["target"] not in point and any(
                key in point for key in variation["linked_values"]
            ):
                raise ValueError(f"linked values require {variation['target']}")


@contextmanager
def admit(path, *, operation, base_name, hashes, points, timeout_seconds, defaults=None):
    """Hold the investigation lock until the run's manifest commits admission."""
    with investigation_lock(path):
        record = _read_locked(path)
        revision = record["active_revision"]
        if revision is None:
            raise ValueError("investigation must be started before admission")
        terms = next(item["terms"] for item in record["revisions"] if item["revision"] == revision)
        if operation not in terms["scope"]["operations"]:
            raise ValueError(f"operation outside investigation scope: {operation}")
        base = next((item for item in terms["scope"]["bases"] if item["name"] == base_name), None)
        if operation != "check":
            if base is None:
                raise ValueError(f"unknown investigation base: {base_name}")
            _in_scope(base, hashes, points, defaults)
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        limits = terms["limits"]
        deadline = limits.get("deadline")
        if deadline is not None and datetime.now(UTC) >= deadline:
            raise ValueError("investigation deadline has passed")
        records = _runs(path)
        usage = _usage(records)
        commitment = timeout_seconds if timeout_seconds is not None else 0.0
        if "job_count" in limits and usage["jobs"] + 1 > limits["job_count"]:
            raise ValueError("investigation job_count limit exhausted")
        if "job_seconds" in limits:
            for run, manifest, finished in records:
                if (finished is None or finished["usage"].get("uncertain", False)) and manifest[
                    "investigation_commitment"
                ].get("unbounded", False):
                    raise ValueError(
                        f"cannot bound unsettled job usage from earlier run: {run.name}"
                    )
            remaining = (
                limits["job_seconds"] - usage["job_seconds"] - usage["committed_job_seconds"]
            )
            if timeout_seconds is None:
                commitment = remaining
            if remaining <= 0 or commitment > remaining:
                raise ValueError("investigation job_seconds limit exhausted")
        yield {
            "investigation_id": path.name,
            "investigation_revision": revision,
            "investigation_base": base_name,
            "investigation_limits": limits,
            "investigation_commitment": {
                "job_seconds": commitment,
                "unbounded": timeout_seconds is None and "job_seconds" not in limits,
            },
        }


class CancelledStart(RuntimeError):
    """An admitted attempt must end without computing."""


@contextmanager
def start_permission(run):
    """Serialize the final deadline check and start publication with decisions."""
    manifest = _load_record(run / "run.toml")
    investigation_id = manifest.get("investigation_id")
    if investigation_id is None:
        yield None
        return
    path = run.parent.parent / "_investigations" / investigation_id
    with investigation_lock(path):
        deadline = manifest["investigation_limits"].get("deadline")
        if deadline is not None and datetime.now(UTC) >= deadline:
            raise CancelledStart("admitted attempt's deadline has passed")
        yield deadline


def record_finding(path: Path, finding: Path | dict) -> dict:
    """Append authored reasoning and member run references against the effective revision."""
    path = path.resolve()
    data = _load_record(finding) if isinstance(finding, Path) else finding
    author = Finding.model_validate(data).model_dump(exclude_none=True)
    with investigation_lock(path):
        record = _read_locked(path)
        if record["active_revision"] is None:
            raise ValueError("a draft investigation cannot receive findings before its start")
        wanted = set(author["runs"])
        found = set()
        for manifest_path in sorted(path.parent.parent.glob("*/*/run.toml")):
            if manifest_path.parent.name not in wanted:
                continue
            manifest = _load_record(manifest_path)
            if manifest.get("investigation_id") != path.name:
                raise ValueError(
                    f"run does not belong to this investigation: {manifest_path.parent.name}"
                )
            run_id = manifest["run_id"]
            if run_id in found:
                raise ValueError(f"ambiguous run ID: {run_id}")
            found.add(run_id)
        if wanted - found:
            raise ValueError(f"unknown run references: {sorted(wanted - found)}")
        runs = _runs(path)
        usage = _usage(runs)
        previous = next(
            (
                event["usage"]["total"]
                for event in reversed(record["events"])
                if event["kind"] == "finding"
                and event["finding"]["kind"] == "assessment"
                and "usage" in event
            ),
            {},
        )
        evidence = []
        for run, manifest, _ in runs:
            if run.name not in wanted:
                continue
            read = _read_attempt(run)
            relevant = set()
            unavailable = None
            if manifest["protocol"] in {"simulate", "sweep", "plan", "verification"}:
                from entanglix.corrector.verification import check_run, verification_history

                source = (
                    (run.parent / manifest["launch"]["source_path"]).resolve()
                    if manifest["protocol"] == "verification"
                    else run
                )
                read["verification"] = verification_history(source)
                if manifest["protocol"] != "verification":
                    # As in read_run, an unloadable frozen plan leaves the record readable.
                    try:
                        relevant = {item["name"] for item in check_run(run)["checks"]}
                    except (OSError, ValueError, RuntimeError, KeyError, MemoryError) as exc:
                        unavailable = failure_message(exc)
                if manifest["protocol"] == "verification":
                    read["verification"] = [
                        item for item in read["verification"] if item.get("run_id") == run.name
                    ]
            revision = next(
                item["terms"]
                for item in record["revisions"]
                if item["revision"] == record["active_revision"]
            )
            requested = revision["evidence_requirements"]["checks"].get(
                manifest["investigation_base"], []
            )
            if manifest["protocol"] == "verification":
                request = _load_record(run / "spec" / manifest["launch"]["request_file"])
                requested = request["checks"]
            checks = [
                {
                    "run_id": item["run_id"],
                    "policy_version": item["policy_version"],
                    "name": check["name"],
                    **({"outcome": check["outcome"]} if "outcome" in check else {}),
                }
                for item in read.get("verification", [])
                if item.get("source_matches") and item["status"] == "completed"
                for check in item.get("checks", [])
            ]
            names = {item["name"] for item in checks}
            evidence.append(
                {
                    "run_id": run.name,
                    "status": read["status"],
                    "checks": checks,
                    **(
                        {"relevant_checks_unavailable": unavailable}
                        if unavailable is not None
                        else {"relevant_checks_not_run": sorted(relevant - names)}
                    ),
                    "requested_checks_not_run": sorted(set(requested) - names),
                }
            )
        return _event(
            path,
            "finding",
            revision=record["active_revision"],
            finding=author,
            usage={
                "total": usage,
                "since_assessment": {
                    key: value - previous.get(key, 0)
                    for key, value in usage.items()
                    if key not in {"unsettled_jobs", "committed_job_seconds", "unbounded_jobs"}
                },
            },
            evidence=evidence,
        )


def read_investigation(path: Path) -> dict:
    path = path.resolve()
    with investigation_lock(path, shared=True):
        record = _read_locked(path)
        record["usage"] = _usage(_runs(path))
        return record


def list_investigations(runs_root: Path = Path("research_runs")) -> list[dict]:
    items = []
    for path in sorted((runs_root / "_investigations").glob("*")):
        if not path.is_dir():
            continue
        try:
            with investigation_lock(path, shared=True):
                record = _read_locked(path)
            items.append(
                {
                    name: record[name]
                    for name in ("path", "investigation_id", "state", "active_revision")
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            items.append({"path": str(path.resolve()), "state": "invalid", "diagnostic": str(exc)})
    return items
