"""Immutable investigation terms, operator decisions, and research findings."""

import fcntl
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
from entanglix.runner import _load_record, _relative_file, publish_toml

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
    gpu_seconds: PositiveSeconds | None = None
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
        return _event(
            path,
            "finding",
            revision=record["active_revision"],
            finding=author,
        )


def read_investigation(path: Path) -> dict:
    path = path.resolve()
    with investigation_lock(path, shared=True):
        return _read_locked(path)


def list_investigations(runs_root: Path = Path("research_runs")) -> list[dict]:
    items = []
    for path in sorted((runs_root / "_investigations").glob("*")):
        if not path.is_dir():
            continue
        try:
            record = read_investigation(path)
            items.append(
                {
                    name: record[name]
                    for name in ("path", "investigation_id", "state", "active_revision")
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            items.append({"path": str(path.resolve()), "state": "invalid", "diagnostic": str(exc)})
    return items
