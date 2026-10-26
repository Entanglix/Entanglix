"""Block 12 format probe over the retained runs; read-only, no solve, GPU or agent session.

The retained runs are copied to a disposable directory first. Every reader, CLI command and MCP
tool runs against the copy, because MCP calls write session and call audits and the readers take
locks. The probe records:
- the file kinds in runs and runs-root ledgers, each record and array writer and reader in
  ``src/entanglix``, and whether the writing function names a version field;
- per record and input kind, version values, optional keys, mixed value types and datetimes;
- the current readers over every retained run, root and ledger, and the status variants present;
- the seven input-model schemas, their acceptance of retained inputs compared with the models,
  and integer/float mutations of one instance of each;
- one record (``run.toml``) drafted as a hand-written schema and as a record model, and one reply
  (``read_run``: CLI, MCP detail and MCP compact) drafted as a schema, checked on every instance;
- each NumPy archive kind's arrays, including the example explicit-operator archives;
- the read-only CLI commands' and MCP tools' reply shapes, and the SDK payload with an output
  schema on the largest reply.

TMPDIR=/var/tmp/claude-84 OPENBLAS_NUM_THREADS=1 .venv-mcp/bin/python tests/probes/s4_formats.py \
  --runs research_runs --output tests/probes/s4_formats_results.json
"""

import argparse
import ast
import asyncio
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter, defaultdict
from datetime import date, datetime
from datetime import time as clock_time
from fnmatch import fnmatch
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal

import jsonschema
import numpy as np
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, Implementation, TextContent
from pydantic import BaseModel, ConfigDict, Field, model_validator

from entanglix import catalog, investigations, runner
from entanglix.investigations import InvestigationSpec
from entanglix.launches import read_launch
from entanglix.mcp import Configuration, _json, build_server
from entanglix.measurements import DatasetSpec
from entanglix.models import SystemSpec
from entanglix.protocols.calibrate import CalibrationSpec
from entanglix.protocols.control import ControlSpec
from entanglix.protocols.simulate import SimulationSpec
from entanglix.protocols.sweep import SweepSpec

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src/entanglix"
LEDGERS = ("_investigations", "_catalog", "_launches", "_audit")
ID = re.compile(r"[a-z]?\d{8}T\d{6,}Z-[0-9a-f]{8}|[a-z]?[0-9a-f]{16,}|\d+")
MAP_LIMIT = 12
STRICT_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
SPACE_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?(\+\d{2}:\d{2})?")

# Tentative classes for the ADR: record, input, array, opaque (contents not published), internal.
RUN_KINDS = {
    "run.toml": ("record", "run"),
    "runner/started.toml": ("record", "started"),
    "FINISHED": ("record", "terminal"),
    "runner/reconciled.toml": ("record", "reconciled"),
    "runner/abort.toml": ("record", "abort request"),
    "result.toml": ("record", "result"),
    "artifacts/points/<id>/result.toml": ("record", "sweep point result"),
    "artifacts/assignment.toml": ("record", "calibration assignment"),
    "artifacts/solver-trace.toml": ("opaque", "solver trace"),
    "proposals/<id>.toml": ("record", "proposal"),
    "proposals/<id>.applied.toml": ("record", "proposal application"),
    "artifacts/output.npz": ("array", "output"),
    "artifacts/points/<id>/output.npz": ("array", "sweep point output"),
    "artifacts/pulse.npz": ("array", "neutral pulse"),
    "artifacts/piccolo/*": ("opaque", "solver-native"),
    "artifacts/*.png": ("internal", "plot"),
    "run.log": ("internal", "log"),
    "spec/code/*": ("internal", "code provenance"),
    "spec/system.toml": ("input", "system"),
    "spec/base/system.toml": ("input", "system"),
    "spec/points/<id>/system.toml": ("input", "system"),
    "spec/simulate.toml": ("input", "simulation"),
    "spec/base/simulate.toml": ("input", "simulation"),
    "spec/sweep.toml": ("input", "sweep"),
    "spec/base/sweep.toml": ("input", "sweep"),
    "spec/control.toml": ("input", "control"),
    "spec/base/control.toml": ("input", "control"),
    "spec/calibrate.toml": ("input", "calibration"),
    "spec/training.toml": ("input", "dataset"),
    "spec/holdout/holdout.toml": ("input", "dataset"),
    "spec/problem.toml": ("input", "ProblemSpec (amicode schema)"),
    "spec/check-request.toml": ("input", "check request (harness-written)"),
    "spec/plan.toml": ("internal", "execution plan"),
    "spec/points/<id>/plan.toml": ("internal", "execution plan"),
    "spec/sweep-manifest.toml": ("internal", "sweep manifest"),
    "spec/readout-state.npz": ("array", "initial state"),
    "spec/base/readout-state.npz": ("array", "initial state"),
    "spec/points/<id>/readout-state.npz": ("array", "initial state"),
    "spec/initial-state.npz": ("array", "initial state"),
    "spec/starting-pulse.npz": ("array", "neutral pulse"),
    "spec/training.npz": ("array", "measurement archive"),
    "spec/holdout/holdout.npz": ("array", "measurement archive"),
}
LEDGER_KINDS = {
    "_investigations/<id>/revisions/<id>.toml": ("record", "investigation revision"),
    "_investigations/<id>/events/<id>.toml": ("record", "investigation event"),
    "_catalog/staging/<id>/record.toml": ("record", "staged candidate"),
    "_catalog/entries/<id>/record.toml": ("record", "catalog entry"),
    "_catalog/*/<id>/device/system.toml": ("input", "system"),
    "_catalog/*/<id>/pulse.npz": ("array", "neutral pulse"),
    "_audit/sessions/<id>.toml": ("record", "MCP session audit"),
    "_audit/calls/<id>/request.toml": ("record", "MCP call request audit"),
    "_audit/calls/<id>/outcome.toml": ("record", "MCP call outcome audit"),
    "_launches/<id>/request.toml": ("record", "launch request"),
    "_launches/<id>/call.toml": ("record", "launch call"),
    "_launches/<id>/attempt.toml": ("record", "launch attempt"),
    "_launches/<id>/supervisor.log": ("internal", "log"),
}
MODELS = {
    "system": SystemSpec,
    "simulation": SimulationSpec,
    "sweep": SweepSpec,
    "control": ControlSpec,
    "calibration": CalibrationSpec,
    "dataset": DatasetSpec,
    "investigation": InvestigationSpec,
}
# Every core array reader and writer, by (module, function), mapped to an array format.
ARRAY_SITES = {
    ("catalog.py", "_neutral_pulse"): "neutral pulse",
    ("catalog.py", "register_result"): "neutral pulse",
    ("protocols/_control_job.py", "execute"): "neutral pulse",
    ("protocols/control.py", "_match_start"): "neutral pulse",
    ("corrector/sweeps.py", "_point_output"): "sweep point output",
    ("corrector/sweeps.py", "sweep_checks"): "sweep output",
    ("protocols/_sweep_job.py", "execute"): "sweep output",
    ("protocols/_plan_job.py", "execute"): "simulation output",
    ("corrector/verification.py", "_output"): "simulation output",
    ("corrector/verification.py", "execute_verification"): "simulation output",
    ("protocols/simulate.py", "compile_simulation"): "initial state",
    ("protocols/plans.py", "load_plan"): "initial state",
    ("corrector/reference.py", "prepare_refinement"): "reference state",
    ("measurements.py", "_load_arrays"): "measurement archive",
    ("measurements.py", "split_dataset"): "measurement archive",
    ("models.py", "_construct_operators"): "explicit-operator archive",
}

# --- run.toml, drafted by hand ------------------------------------------------------------------

DIGEST = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
STAMP = r"\d{8}T\d{12}Z-[0-9a-f]{8}"
RUN_ID = {"type": "string", "pattern": f"^r{STAMP}$"}
DATETIME = {"type": "string", "format": "date-time"}
RUN_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "run record, version 1",
    "type": "object",
    "required": [
        "record_version",
        "run_id",
        "system_id",
        "protocol",
        "created_at",
        "parent_run_ids",
        "engine",
        "device",
        "executor",
        "needs",
        "spec_sha256",
        "code",
        "env_hash",
        "environment",
    ],
    "properties": {
        "record_version": {"const": 1},
        "run_id": RUN_ID,
        "system_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]*$"},
        "protocol": {
            "enum": ["simulate", "sweep", "control", "calibrate", "verification", "script"]
        },
        "created_at": DATETIME,
        "parent_run_ids": {"type": "array", "items": RUN_ID},
        "engine": {"type": "string"},
        "device": {"enum": ["cpu", "cuda"]},
        "executor": {"const": "local"},
        "needs": {
            "type": "object",
            "required": ["gradients", "device"],
            "properties": {"gradients": {"type": "boolean"}, "device": {"type": "string"}},
            "additionalProperties": False,
        },
        "spec_sha256": {"type": "object", "additionalProperties": DIGEST},
        "code": {
            "type": "object",
            "required": ["revision", "dirty", "patch_sha256", "untracked_sha256"],
            "properties": {
                "revision": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                "dirty": {"type": "boolean"},
                "patch_sha256": DIGEST,
                "untracked_sha256": {"type": "object", "additionalProperties": DIGEST},
            },
            "additionalProperties": False,
        },
        "env_hash": DIGEST,
        "environment": {
            "type": "object",
            "required": ["lock_sha256", "python", "extras", "env_hash", "platform", "gpu"],
            "properties": {
                "lock_sha256": DIGEST,
                "python": {"type": "string"},
                "extras": {"type": "array", "items": {"type": "string"}},
                "env_hash": DIGEST,
                "platform": {"type": "object"},
                "gpu": {"type": "object"},
                "solver_runtime": {"type": "object"},
            },
            "additionalProperties": False,
        },
        # Protocol-specific; the probe records its variants instead of drafting them.
        "launch": {"type": "object"},
        "launch_id": {"type": "string", "pattern": f"^l{STAMP}$"},
        "source_entry_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": f"^e{STAMP}$"},
        },
        "investigation_id": {"type": "string", "pattern": f"^i{STAMP}$"},
        "investigation_revision": {"type": "integer", "minimum": 1},
        "investigation_base": {"type": "string"},
        "investigation_commitment": {"type": "object"},
        "investigation_limits": {"type": "object"},
        "investigation_values": {"type": "array", "items": {"type": "object"}},
    },
    "dependentRequired": {
        "investigation_id": [
            "investigation_revision",
            "investigation_base",
            "investigation_commitment",
            "investigation_limits",
        ]
    },
    "additionalProperties": False,
}

# --- run.toml, drafted as a record model ---------------------------------------------------------

Digest = Annotated[str, Field(pattern="^[0-9a-f]{64}$")]
RunId = Annotated[str, Field(pattern=f"^r{STAMP}$")]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Needs(Record):
    gradients: bool
    device: str


class Code(Record):
    revision: Annotated[str, Field(pattern="^[0-9a-f]{40}$")]
    dirty: bool
    patch_sha256: Digest
    untracked_sha256: dict[str, Digest]


class Environment(Record):
    lock_sha256: Digest
    python: str
    extras: list[str]
    env_hash: Digest
    platform: dict[str, Any]
    gpu: dict[str, Any]
    solver_runtime: dict[str, Any] | None = None


COMPANIONS = (
    "investigation_revision",
    "investigation_base",
    "investigation_commitment",
    "investigation_limits",
)


class RunRecord(Record):
    # A cross-field rule reaches the generated schema only through an explicit extra.
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={"dependentRequired": {"investigation_id": list(COMPANIONS)}},
    )

    record_version: Literal[1]
    run_id: RunId
    system_id: Annotated[str, Field(pattern="^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    protocol: Literal["simulate", "sweep", "control", "calibrate", "verification", "script"]
    created_at: datetime
    parent_run_ids: list[RunId]
    engine: str
    device: Literal["cpu", "cuda"]
    executor: Literal["local"]
    needs: Needs
    spec_sha256: dict[str, Digest]
    code: Code
    env_hash: Digest
    environment: Environment
    launch: dict[str, Any] | None = None
    launch_id: Annotated[str, Field(pattern=f"^l{STAMP}$")] | None = None
    source_entry_ids: list[Annotated[str, Field(pattern=f"^e{STAMP}$")]] | None = None
    investigation_id: Annotated[str, Field(pattern=f"^i{STAMP}$")] | None = None
    investigation_revision: Annotated[int, Field(ge=1)] | None = None
    investigation_base: str | None = None
    investigation_commitment: dict[str, Any] | None = None
    investigation_limits: dict[str, Any] | None = None
    investigation_values: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def investigation_fields(self):
        if self.investigation_id is not None and any(getattr(self, f) is None for f in COMPANIONS):
            raise ValueError("investigation_id requires its revision, base, commitment and limits")
        return self


# --- read_run reply, drafted by hand -------------------------------------------------------------

STARTED = {
    "type": "object",
    "required": ["record_version", "started_at", "process_group", "gpu_devices"],
    "properties": {
        "record_version": {"const": 1},
        "started_at": DATETIME,
        "process_group": {"type": "integer", "minimum": 1},
        "gpu_devices": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}
USAGE = {
    "type": "object",
    "required": ["wall_seconds", "gpu_seconds", "uncertain"],
    "properties": {
        "wall_seconds": {"type": "number", "minimum": 0},
        "gpu_seconds": {"type": "number", "minimum": 0},
        "uncertain": {"const": False},
        "deadline_overrun_seconds": {"type": "number", "minimum": 0},
        "job_seconds_overrun": {"type": "number", "minimum": 0},
    },
    "additionalProperties": False,
}
FINISHED = {
    "type": "object",
    "required": ["record_version", "status", "exit_code", "ended_at", "usage"],
    "properties": {
        "record_version": {"const": 1},
        "status": {"enum": ["completed", "failed", "aborted"]},
        "exit_code": {"type": "integer", "minimum": 0, "maximum": 255},
        "ended_at": DATETIME,
        "usage": USAGE,
    },
    "additionalProperties": False,
}
RECONCILED = {
    "type": "object",
    "required": ["record_version", "observed_at", "last_observed_state", "usage"],
    "properties": {
        "record_version": {"const": 1},
        "observed_at": DATETIME,
        "last_observed_state": {"type": "string"},
        "started_at": DATETIME,
        "process_group": {"type": "integer"},
        "gpu_devices": {"type": "array", "items": {"type": "string"}},
        "usage": {
            "type": "object",
            "required": ["uncertain"],
            "properties": {"uncertain": {"const": True}},
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}
ABORT = {
    "type": "object",
    "required": ["requested_at"],
    "properties": {"requested_at": DATETIME},
    "additionalProperties": False,
}


def reply_schema(manifest):
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["path", "status", "manifest", "log_tail"],
        "properties": {
            "path": {"type": "string"},
            "status": {
                "enum": ["active", "interrupted", "unknown", "completed", "failed", "aborted"]
            },
            "manifest": manifest,
            "start": STARTED,
            "finished": FINISHED,
            "reconciliation": RECONCILED,
            "abort_request": ABORT,
            # Protocol results are their own published formats; referenced, not drafted here.
            "result": {"type": "object"},
            "result_error": {"type": "string"},
            "log_tail": {"type": "string"},
            "verification": {"type": "array", "items": {"type": "object"}},
            "check_listing": {"type": "object"},
            "sweep_progress": {"type": "object"},
        },
        "additionalProperties": False,
    }


def compact_manifest():
    schema = copy.deepcopy(RUN_SCHEMA)
    del schema["$schema"]
    for key in ("spec_sha256", "code", "environment"):
        schema["required"].remove(key)
        del schema["properties"][key]
    schema["properties"]["detail_omitted"] = {
        "type": "array",
        "items": {"enum": ["spec_sha256", "code", "environment"]},
    }
    return schema


READ_RUN = reply_schema({k: v for k, v in RUN_SCHEMA.items() if k != "$schema"})
READ_RUN_COMPACT = reply_schema(compact_manifest())

# --- helpers -------------------------------------------------------------------------------------

FORMATS = jsonschema.FormatChecker(formats=())


@FORMATS.checks("date-time")
def _strict_datetime(value):
    return not isinstance(value, str) or bool(STRICT_DATETIME.match(value))


def validator(schema):
    return jsonschema.Draft202012Validator(schema, format_checker=FORMATS)


def to_json(value):
    """The stated TOML-to-JSON mapping: datetimes become RFC 3339 strings with ``T``."""
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json(item) for item in value]
    if isinstance(value, (datetime, date, clock_time)):
        return value.isoformat()
    return value


def type_name(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, datetime):
        return "datetime" if value.tzinfo else "local-datetime"
    for kind, name in ((date, "date"), (clock_time, "time"), (int, "int"), (float, "float")):
        if isinstance(value, kind):
            return name
    return {dict: "table", list: "array", str: "string"}.get(type(value), type(value).__name__)


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, (*path, index))
    else:
        yield path, value


def replaced(value, path, new):
    value = copy.deepcopy(value)
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = new
    return value


def nonfinite(value):
    return sum(isinstance(v, float) and not math.isfinite(v) for _, v in leaves(value))


def errors(schema_validator, instance, limit=3):
    found = sorted(schema_validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:200]}" for e in found][
        :limit
    ]


def load_toml(path):
    return tomllib.loads(path.read_text())


def normalize(parts):
    return "/".join(ID.sub("<id>", part) for part in parts)


def classify(kind, table):
    for pattern, value in table.items():
        if fnmatch(kind, pattern):
            return value
    return ("unclassified", None)


# --- sections ------------------------------------------------------------------------------------


def layout(runs):
    attempts = sorted(p.parent for p in runs.rglob("run.toml"))
    roots = sorted({run.parent.parent for run in attempts})
    roots += sorted(
        {d.parent for name in LEDGERS for d in runs.rglob(name) if d.is_dir()} - set(roots)
    )
    return attempts, roots


def inventory(runs, attempts, roots):
    members = set(attempts)
    kinds = defaultdict(lambda: {"files": 0, "roots": set()})
    other = Counter()
    files = defaultdict(list)
    for path in runs.rglob("*"):
        if not path.is_file():
            continue
        run = next((parent for parent in path.parents if parent in members), None)
        if run is not None:
            relative = path.relative_to(run).parts
            if relative[:2] in {("spec", "code"), ("artifacts", "piccolo")}:
                relative = (*relative[:2], "*")
            kind, table, root = normalize(relative), RUN_KINDS, run.parent.parent
            if fnmatch(kind, "artifacts/metric-*.png"):
                kind = "artifacts/metric-<name>.png"
            key = f"run: {kind}"
        else:
            root = max(
                (r for r in roots if path.is_relative_to(r)),
                key=lambda r: len(r.parts),
                default=None,
            )
            relative = path.relative_to(root).parts if root else ()
            if not relative or relative[0] not in LEDGERS:
                other[path.relative_to(runs).parts[0]] += 1
                continue
            kind, table = normalize(relative), LEDGER_KINDS
            key = f"root: {kind}"
        entry = kinds[key]
        entry["files"] += 1
        entry["roots"].add(str(root.relative_to(runs)))
        entry["class"], entry["format"] = classify(kind, table)
        files[key].append(path)
    table = {key: {**value, "roots": len(value["roots"])} for key, value in sorted(kinds.items())}
    return table, files, dict(other)


def site_rows():
    """Every TOML/NPZ writer and NPZ reader call in src/entanglix, with its enclosing function."""
    rows = []
    for path in sorted(SOURCE.rglob("*.py")):
        module = str(path.relative_to(SOURCE))
        tree = ast.parse(path.read_text())

        def visit(node, function, versioned):
            for child in ast.iter_child_nodes(node):
                name_, flag = function, versioned
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name_ = f"{function}.{child.name}" if function else child.name
                    flag = any(
                        isinstance(k, ast.Constant)
                        and k.value in {"record_version", "format_version"}
                        for d in ast.walk(child)
                        if isinstance(d, ast.Dict)
                        for k in d.keys
                    )
                if isinstance(child, ast.Call):
                    func = child.func
                    attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                    base = ast.unparse(func.value) if isinstance(func, ast.Attribute) else ""
                    call = None
                    if attr == "publish_toml":
                        call = "publish_toml"
                    elif attr in {"dump", "dumps"} and base == "tomli_w":
                        call = "tomli_w"
                    elif attr in {"savez", "savez_compressed"}:
                        call = "np.savez"
                    elif attr == "load" and base in {"np", "numpy"}:
                        call = "np.load"
                    if call:
                        names = sorted(
                            {
                                c.value
                                for c in ast.walk(child.args[0])
                                if isinstance(c, ast.Constant) and isinstance(c.value, str)
                            }
                            if child.args
                            else set()
                        )
                        rows.append(
                            {
                                "module": module,
                                "function": name_,
                                "line": child.lineno,
                                "call": call,
                                "names": names,
                                "version_key_in_function": flag,
                            }
                        )
                visit(child, name_, flag)

        visit(tree, "", False)
    return rows


def writers():
    rows = site_rows()
    arrays = [row for row in rows if row["call"] in {"np.load", "np.savez"}]
    mapped = {
        f"{row['module']}:{row['line']} {row['function']} {row['call']}": ARRAY_SITES.get(
            (row["module"], row["function"].split(".")[-1])
        )
        for row in arrays
    }
    used = {(r["module"], r["function"].split(".")[-1]) for r in arrays}
    return {
        "record_writers": [row for row in rows if row["call"] == "publish_toml"],
        "toml_dumps": [row for row in rows if row["call"] == "tomli_w"],
        "array_sites": mapped,
        "array_sites_unmapped": sorted(site for site, value in mapped.items() if value is None),
        "array_table_unused": sorted(f"{m}:{f}" for m, f in ARRAY_SITES if (m, f) not in used),
    }


def variation(files, runs, kinds):
    result = {}
    for key, paths in files.items():
        if kinds[key]["class"] not in {"record", "input"} or key.endswith(".npz"):
            continue
        documents = []
        failures = []
        for path in paths:
            try:
                documents.append((path, load_toml(path)))
            except (OSError, ValueError) as exc:
                failures.append(f"{path.relative_to(runs)}: {exc}")
        maps = set()
        for _ in range(4):
            children = defaultdict(set)
            for _, document in documents:
                for path, _value in walk(document, maps):
                    if len(path) > 1:
                        children[path[:-1]].add(path[-1])
            wide = {p for p, keys in children.items() if len(keys) > MAP_LIMIT} - maps
            if not wide:
                break
            maps |= wide
        paths_seen = defaultdict(lambda: {"types": set(), "present": 0})
        for _, document in documents:
            for path, value in set(walk(document, maps, typed=True)):
                entry = paths_seen[".".join(map(str, path))]
                entry["types"].add(value)
                entry["present"] += 1
        versions = Counter(
            str(d.get("record_version", d.get("format_version", "absent"))) for _, d in documents
        )
        top = Counter(tuple(sorted(d)) for _, d in documents)
        mixed = {
            p: sorted(e["types"])
            for p, e in paths_seen.items()
            if len(e["types"] - {"table", "array"}) > 1
            and e["types"] - {"table", "array"} != {"int", "float"}
        }
        result[key] = {
            "instances": len(documents),
            "unreadable": failures,
            "versions": dict(versions),
            "top_level_key_sets": len(top),
            "optional_top_level": sorted(
                set().union(*map(set, top)) - set.intersection(*map(set, top))
            )
            if top
            else [],
            "maps_collapsed": sorted(".".join(map(str, m)) for m in maps),
            "int_and_float": sorted(
                p for p, e in paths_seen.items() if {"int", "float"} <= e["types"]
            ),
            "mixed_types": mixed,
            "datetime_paths": sorted(
                p for p, e in paths_seen.items() if e["types"] & {"datetime", "local-datetime"}
            ),
            "local_datetime_paths": sorted(
                p for p, e in paths_seen.items() if "local-datetime" in e["types"]
            ),
            "nonfinite_values": sum(nonfinite(d) for _, d in documents),
            "path_count": len(paths_seen),
        }
    return result


def walk(value, maps, path=(), typed=False):
    if isinstance(value, dict):
        if path:
            yield path, "table" if typed else None
        for key, item in value.items():
            yield from walk(item, maps, (*path, "*" if path in maps else key), typed)
    elif isinstance(value, list):
        yield path, "array" if typed else None
        for item in value:
            yield from walk(item, maps, (*path, "[]"), typed)
    else:
        yield path, type_name(value) if typed else None


def readability(runs, attempts, roots):
    statuses = Counter()
    variants = Counter()
    failures = []
    timings = []
    records = {}
    refusals = Counter()
    for run in attempts:
        start = time.perf_counter()
        try:
            record = runner.read_run(run)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{run.relative_to(runs)}: {type(exc).__name__}: {exc}"[:300])
            continue
        timings.append(time.perf_counter() - start)
        records[run] = record
        protocol = record["manifest"]["protocol"]
        statuses[f"{protocol}/{record['status']}"] += 1
        for label in ("reconciliation", "abort_request", "result_error"):
            if label in record:
                variants[label] += 1
        if reason := record.get("check_listing", {}).get("reason"):
            variants["check listing refused"] += 1
            refusals[f"{protocol}: {reason[:120]}"] += 1
        progress = record.get("sweep_progress")
        if progress and record["status"] != "completed":
            variants["partial sweep"] += 1
        if record.get("verification"):
            variants["runs with check history"] += 1
    present = {key.split("/")[1] for key in statuses} | set(variants)
    expected = [
        "completed",
        "failed",
        "aborted",
        "interrupted",
        "unknown",
        "reconciliation",
        "abort_request",
        "result_error",
        "partial sweep",
        "check listing refused",
    ]
    listing = {}
    for root in roots:
        rows = runner.list_runs(root)
        listing[str(root.relative_to(runs)) or "."] = dict(Counter(row["status"] for row in rows))
    ledgers = {"investigations": Counter(), "catalog": Counter(), "launches": Counter()}
    for root in roots:
        if (root / "_investigations").is_dir():
            for row in investigations.list_investigations(root):
                try:
                    investigations.read_investigation(Path(row["path"]))
                    ledgers["investigations"]["read"] += 1
                except Exception as exc:  # noqa: BLE001
                    ledgers["investigations"][f"error: {type(exc).__name__}"] += 1
        if (root / "_catalog").is_dir():
            for row in catalog.list_catalog(root):
                identifier = row["id"]
                try:
                    catalog.read_entry(identifier, root)
                    ledgers["catalog"]["read"] += 1
                except Exception as exc:  # noqa: BLE001
                    ledgers["catalog"][f"error: {type(exc).__name__}: {exc}"[:120]] += 1
        if (root / "_launches").is_dir():
            for path in sorted((root / "_launches").iterdir()):
                try:
                    ledgers["launches"][read_launch(path)["status"]] += 1
                except Exception as exc:  # noqa: BLE001
                    ledgers["launches"][f"error: {type(exc).__name__}"] += 1
    return records, {
        "attempts": len(attempts),
        "read_failures": failures,
        "statuses": dict(statuses),
        "variants": dict(variants),
        "variants_absent": [name for name in expected if name not in present],
        "check_listing_refusals": dict(refusals),
        "read_run_seconds": {
            "total": sum(timings),
            "max": max(timings, default=0),
        },
        "list_runs_by_root": listing,
        "ledger_readers": {k: dict(v) for k, v in ledgers.items()},
        "readers_absent": [
            "MCP session and call audits (written only)",
            "proposal records (read only inside apply_device_update)",
        ],
    }


def input_models(files, kinds):
    instances = defaultdict(list)
    for key, paths in files.items():
        if kinds[key]["class"] == "input" and kinds[key]["format"] in MODELS:
            instances[kinds[key]["format"]].extend((str(p), load_toml(p)) for p in paths)
        if kinds[key]["format"] == "investigation revision":
            instances["investigation"].extend((str(p), load_toml(p)["terms"]) for p in paths)
    committed = json.loads((REPOSITORY / "schemas/system.schema.json").read_text())
    result = {}
    mutations = {}
    for name, model in MODELS.items():
        schema = model.model_json_schema()
        entry = {
            "schema_bytes": len(json.dumps(schema)),
            "defs": len(schema.get("$defs", {})),
            "schema_valid_2020_12": not list(
                jsonschema.Draft202012Validator(
                    jsonschema.Draft202012Validator.META_SCHEMA
                ).iter_errors(schema)
            ),
            "instances": len(instances[name]),
        }
        if name == "system":
            entry["matches_committed_schema"] = schema == committed
        check = validator(schema)
        counts = Counter()
        examples = []
        first = None
        for path, document in instances[name]:
            mapped = to_json(document)
            schema_ok = not list(check.iter_errors(mapped))
            try:
                model.model_validate(document)
                model_ok = True
            except ValueError as exc:
                model_ok, message = False, str(exc).splitlines()[:3]
            try:
                model.model_validate_json(json.dumps(mapped))
                json_ok = True
            except ValueError:
                json_ok = False
            counts[f"schema={schema_ok} model={model_ok} model_json={json_ok}"] += 1
            if has_datetime(document):
                counts["contains datetimes"] += 1
            if schema_ok != model_ok and len(examples) < 3:
                examples.append(
                    {
                        "path": path,
                        "schema": errors(check, mapped),
                        "model": None if model_ok else message,
                    }
                )
            if first is None and model_ok:
                first = document
        entry["agreement"] = dict(counts)
        entry["disagreements"] = examples
        result[name] = entry
        if first is not None:
            mutations[name] = mutate(model, check, first)
    return result, mutations


def has_datetime(value):
    return any(isinstance(v, (datetime, date, clock_time)) for _, v in leaves(value))


def mutate(model, check, document):
    """Each integer written as a float, and each integral float written as an integer."""
    out = {}
    for label, select, change in (
        ("int_as_float", lambda v: type(v) is int, float),
        ("integral_float_as_int", lambda v: type(v) is float and v.is_integer(), int),
    ):
        tally = Counter()
        accepted = []
        for path, value in leaves(document):
            if not select(value):
                continue
            candidate = replaced(document, path, change(value))
            tally["leaves"] += 1
            tally["schema_accepts"] += not list(check.iter_errors(to_json(candidate)))
            try:
                validated = model.model_validate(candidate)
            except ValueError:
                continue
            tally["model_accepts"] += 1
            try:
                kept = lookup(validated.model_dump(by_alias=True), path)
            except (KeyError, IndexError, TypeError):
                kept = "<not at the same path>"
            accepted.append(
                {
                    "path": ".".join(map(str, path)),
                    "written": change(value),
                    "model_returns": type_name(kept) if kept != "<not at the same path>" else kept,
                }
            )
        out[label] = dict(tally) | {"accepted_paths": accepted}
    return out


def lookup(value, path):
    for key in path:
        value = value[key]
    return value


def drafted_run(attempts):
    model_schema = RunRecord.model_json_schema()
    checks = {
        "hand": validator(RUN_SCHEMA),
        "model_schema": validator(model_schema),
    }
    tally = Counter()
    examples = defaultdict(list)
    launch = defaultdict(Counter)
    sample = None
    for run in attempts:
        document = load_toml(run / "run.toml")
        if sample is None and "investigation_id" in document:
            sample = document
        launch[document["protocol"]][tuple(sorted(document.get("launch", {})))] += 1
        mapped = to_json(document)
        for label, check in checks.items():
            found = errors(check, mapped)
            tally[f"{label} valid"] += not found
            if found and len(examples[label]) < 3:
                examples[label].append(found)
        try:
            RunRecord.model_validate(document)
            tally["model valid"] += 1
        except ValueError as exc:
            if len(examples["model"]) < 3:
                examples["model"].append(str(exc).splitlines()[:3])
    return {
        "instances": len(attempts),
        "valid": dict(tally),
        "examples": dict(examples),
        "constraint_cases": constraint_cases(sample, checks),
        "source_lines": source_lines(),
        "model_schema_bytes": len(json.dumps(model_schema)),
        "hand_schema_bytes": len(json.dumps(RUN_SCHEMA)),
        "launch_variants": {
            protocol: [{"keys": list(k), "runs": n} for k, n in variants.items()]
            for protocol, variants in launch.items()
        },
    }


def constraint_cases(document, checks):
    """Constructed records that the retained ones never exercise; True means accepted."""
    cases = {
        "investigation_id without its companion fields": {
            k: v for k, v in document.items() if k not in COMPANIONS
        },
        "optional launch_id written as null (JSON only)": document | {"launch_id": None},
        "launch with an arbitrary key (launch left untyped)": document
        | {"launch": {"anything": 1}},
    }
    out = {}
    for label, case in cases.items():
        mapped = to_json(case)
        try:
            RunRecord.model_validate(case)
            model = True
        except ValueError:
            model = False
        out[label] = {name: not list(check.iter_errors(mapped)) for name, check in checks.items()}
        out[label]["model"] = model
    return out


def source_lines():
    """Lines of each run.toml draft in this file, shared field definitions included."""
    text = Path(__file__).read_text().splitlines()
    marks = [i for i, line in enumerate(text) if line.startswith("# --- ")]
    hand, model, reply = (
        next(i for i in marks if marker in text[i])
        for marker in ("drafted by hand --", "record model", "read_run reply")
    )
    return {"hand_schema": model - hand - 1, "record_model": reply - model - 1}


def drafted_reply(cli_replies, mcp_replies):
    checks = {"detail": validator(READ_RUN), "compact": validator(READ_RUN_COMPACT)}
    out = {}
    for label, replies, check in (
        ("cli", cli_replies, checks["detail"]),
        ("mcp_detail", mcp_replies["detail"], checks["detail"]),
        ("mcp_compact", mcp_replies["compact"], checks["compact"]),
    ):
        tally = Counter()
        examples = []
        for reply in replies:
            found = list(check.iter_errors(reply))
            tally["valid"] += not found
            kinds = Counter(
                "date-time format" if e.validator == "format" else "structure" for e in found
            )
            tally.update(f"errors: {k}" for k in kinds)
            structural = [e for e in found if e.validator != "format"]
            if structural and len(examples) < 3:
                examples.append(errors(check, reply))
        out[label] = {"replies": len(replies), **dict(tally), "structural_examples": examples}
    out["schema_bytes"] = {
        "detail": len(json.dumps(READ_RUN)),
        "compact": len(json.dumps(READ_RUN_COMPACT)),
    }
    return out


def arrays(files, kinds, runs):
    out = {}
    groups = defaultdict(list)
    for key, paths in files.items():
        if key.endswith(".npz"):
            groups[(kinds[key]["format"], key)] = paths
    examples = sorted((REPOSITORY / "examples").glob("*/operators.npz"))
    groups[("explicit-operator archive", "examples/*/operators.npz")] = examples
    for (fmt, key), paths in sorted(groups.items()):
        layouts = Counter()
        protocols = defaultdict(Counter)
        empty = []
        first = {}
        shapes = defaultdict(set)
        failures = []
        for path in sorted(paths):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    signature = []
                    for name in sorted(archive.files):
                        array = archive[name]
                        signature.append(f"{name}:{array.dtype.str}:{array.ndim}d")
                        if len(shapes[name]) < 4:
                            shapes[name].add(str(array.shape))
                    layout = " ".join(signature) or "(no arrays)"
                    layouts[layout] += 1
                    run = next((d for d in path.parents if (d / "run.toml").is_file()), None)
                    if run is not None:
                        protocols[layout][load_toml(run / "run.toml")["protocol"]] += 1
                        if not signature:
                            empty.append(empty_cause(run, path, runs))
                    base = runs if path.is_relative_to(runs) else REPOSITORY
                    first.setdefault(layout, str(path.relative_to(base)))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{path.name}: {type(exc).__name__}: {exc}"[:200])
        out[key] = {
            "format": fmt,
            "files": len(paths),
            "layouts": {
                layout: {"files": n, "first": first[layout], "protocols": dict(protocols[layout])}
                for layout, n in layouts.items()
            },
            **({"empty_archives": empty} if empty else {}),
            "example_shapes": {name: sorted(values) for name, values in shapes.items()},
            "failures": failures,
        }
    return out


def empty_cause(run, path, runs):
    """Whether an empty archive is a sweep aggregate whose points declare no metrics."""
    points = sorted((run / "artifacts/points").glob("*"))
    metrics = [load_toml(point / "result.toml").get("metrics", {}) for point in points]
    point_arrays = []
    for point in points:
        with np.load(point / "output.npz", allow_pickle=False) as archive:
            point_arrays.append(sorted(archive.files))
    return {
        "archive": str(path.relative_to(runs)),
        "aggregate": path.parent.name == "artifacts" and bool(points),
        "points": len(points),
        "points_with_metrics": sum(bool(m) for m in metrics),
        "points_with_arrays": sum(bool(a) for a in point_arrays),
        "point_arrays": sorted({name for names in point_arrays for name in names}),
    }


def selection(records):
    chosen = {}
    for run, record in records.items():
        manifest = record["manifest"]
        key = (
            manifest["protocol"],
            record["status"],
            "reconciliation" in record,
            "abort_request" in record,
            "investigation_id" in manifest,
            "launch_id" in manifest,
            "source_entry_ids" in manifest,
            bool(record.get("verification")),
        )
        chosen.setdefault(key, run)
    return sorted(chosen.values())


def cli(arguments, work, state):
    start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "entanglix", *arguments],
        cwd=work,
        env=os.environ | {"XDG_STATE_HOME": str(state), "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    text = completed.stdout
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    return data, {
        "command": arguments[0],
        "exit_code": completed.returncode,
        "json": data is not None,
        "bytes": len(text.encode()),
        "seconds": round(time.perf_counter() - start, 2),
        "top_level": shape(data),
        "space_datetimes": len(SPACE_DATETIME.findall(text)),
        "nonstandard_numbers": len(re.findall(r"\b(NaN|-?Infinity)\b", text)),
        "stderr_tail": completed.stderr[-300:] if completed.returncode else "",
    }


def shape(data):
    if isinstance(data, dict):
        return {"type": "object", "keys": sorted(data)}
    if isinstance(data, list):
        keys = sorted({k for item in data if isinstance(item, dict) for k in item})
        return {"type": "array", "items": len(data), "item_keys": keys}
    return {"type": type(data).__name__}


def cli_replies(runs, roots, chosen, work, state):
    replies = []
    detail = []
    data, row = cli(["capabilities"], work, state)
    replies.append(row | {"versions": {k: v for k, v in (data or {}).items() if "version" in k}})
    for root in roots:
        name = str(root.relative_to(runs)) or "."
        _, row = cli(["list-runs", "--runs-root", str(root)], work, state)
        replies.append(row | {"root": name})
        if (root / "_catalog").is_dir():
            data, row = cli(["list-catalog", "--runs-root", str(root)], work, state)
            replies.append(row | {"root": name})
            for item in data or []:
                identifier = item["id"]
                _, row = cli(["read-entry", identifier, "--runs-root", str(root)], work, state)
                replies.append(row | {"root": name})
        if (root / "_investigations").is_dir():
            data, row = cli(["list-investigations", "--runs-root", str(root)], work, state)
            replies.append(row | {"root": name})
            first = (data or [{}])[0].get("path")
            if first:
                _, row = cli(["read-investigation", first], work, state)
                replies.append(row | {"root": name})
    for run in chosen:
        data, row = cli(["read-run", str(run)], work, state)
        replies.append(row | {"run": str(run.relative_to(runs))})
        if data is not None:
            detail.append(data)
    dataset = next(runs.rglob("spec/training.toml"), None)
    if dataset is not None:
        _, row = cli(["validate-dataset", str(dataset)], work, state)
        replies.append(row)
    return replies, detail


async def mcp_replies(runs, roots, chosen, state):
    replies = []
    collected = {"detail": [], "compact": []}
    largest = (0, None, None)
    tools_bytes = None
    for root in roots:
        name = str(root.relative_to(runs)) or "."
        runs_here = [run for run in chosen if run.parent.parent == root]
        if not runs_here and not any((root / ledger).is_dir() for ledger in LEDGERS[:2]):
            continue
        config = Configuration(
            runs_root=root,
            device="",
            runner_extras=(),
            state_directory=state,
            repository=REPOSITORY,
        )
        server = build_server(config)
        calls = [("list_capabilities", {}), ("list_runs", {})]
        for run in runs_here:
            relative = str(run.relative_to(root))
            calls += [
                ("read_run", {"run": relative}),
                ("read_run", {"run": relative, "detail": True}),
            ]
        if (root / "_catalog").is_dir():
            calls.append(("list_catalog", {}))
        if (root / "_investigations").is_dir():
            calls.append(("list_investigations", {}))
        async with Client(
            server, client_info=Implementation(name="s4-probe", version="1")
        ) as client:
            if tools_bytes is None:
                listed = await client.list_tools()
                tools_bytes = {
                    "tools": len(listed.tools),
                    "bytes": len(listed.model_dump_json(by_alias=True, exclude_none=True)),
                    "with_output_schema": sum(t.output_schema is not None for t in listed.tools),
                }
            index = 0
            while index < len(calls):
                tool, arguments = calls[index]
                index += 1
                result = await client.call_tool(tool, arguments)
                text = result.content[0].text if result.content else ""
                try:
                    data = json.loads(text)
                except ValueError:
                    data = None
                replies.append(
                    {
                        "tool": tool,
                        "root": name,
                        "arguments": arguments,
                        "error": bool(result.is_error),
                        "bytes": len(text.encode()),
                        "top_level": shape(data),
                        "space_datetimes": len(SPACE_DATETIME.findall(text)),
                        "structured": result.structured_content is not None,
                    }
                )
                if data is None or result.is_error:
                    continue
                if len(text) > largest[0]:
                    largest = (len(text), tool, data)
                if tool == "read_run":
                    collected["detail" if arguments.get("detail") else "compact"].append(data)
                if tool == "list_catalog":
                    for item in data.get("result", []):
                        identifier = item["id"]
                        calls.append(("read_entry", {"identifier": identifier}))
                if tool == "list_investigations":
                    first = (data.get("result") or [{}])[0].get("path")
                    if first:
                        for view in ("status", "attempts", "events"):
                            calls.append(
                                ("read_investigation", {"investigation": first, "view": view})
                            )
    return replies, collected, largest, tools_bytes


async def output_schema_payload(data):
    """The SDK's result size for the same reply: text only, SDK structured, own result."""
    server = MCPServer("s4-probe")
    text = _json(data)

    def text_only() -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text=text)])

    def sdk_structured() -> dict[str, Any]:
        return data

    def own_result() -> Annotated[CallToolResult, dict[str, Any]]:
        return CallToolResult(
            content=[TextContent(type="text", text="See structuredContent.")],
            structured_content=data,
        )

    server.add_tool(text_only, structured_output=False)
    server.add_tool(sdk_structured, structured_output=True)
    server.add_tool(own_result, structured_output=True)
    out = {}
    async with Client(server, client_info=Implementation(name="s4-probe", version="1")) as client:
        listed = {t.name: t for t in (await client.list_tools()).tools}
        for name in ("text_only", "sdk_structured", "own_result"):
            try:
                result = await client.call_tool(name, {})
            except Exception as exc:  # noqa: BLE001
                out[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                continue
            content = sum(len(c.text.encode()) for c in result.content if c.type == "text")
            structured = (
                len(json.dumps(result.structured_content, separators=(",", ":")).encode())
                if result.structured_content is not None
                else 0
            )
            out[name] = {
                "result_bytes": len(result.model_dump_json(by_alias=True, exclude_none=True)),
                "text_bytes": content,
                "structured_bytes": structured,
                "output_schema": listed[name].output_schema,
                "error": bool(result.is_error),
            }
    return out


def size(kinds, cli_rows, mcp_rows, tools):
    classes = Counter(value["class"] for value in kinds.values())
    formats = defaultdict(set)
    for value in kinds.values():
        formats[value["class"]].add(value["format"])
    parser = ast.parse((SOURCE / "cli.py").read_text())
    commands = sum(
        isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_parser"
        for n in ast.walk(parser)
    )
    cli_shapes = {
        command: len({json.dumps(unsized(r["top_level"]), sort_keys=True) for r in rows})
        for command, rows in group(cli_rows, "command").items()
    }
    mcp_shapes = {
        tool: len({json.dumps(unsized(r["top_level"]), sort_keys=True) for r in rows})
        for tool, rows in group(mcp_rows, "tool").items()
    }
    return {
        "file_kinds_by_class": dict(classes),
        "formats_by_class": {k: sorted(map(str, v)) for k, v in sorted(formats.items())},
        "cli_commands": commands,
        "mcp_tools": tools,
        "distinct_reply_shapes": {"cli": cli_shapes, "mcp": mcp_shapes},
    }


def unsized(top_level):
    return {key: value for key, value in top_level.items() if key != "items"}


def group(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return out


def source_identity():
    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(REPOSITORY), *arguments], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "src_dirty": bool(git("status", "--porcelain", "--", "src")),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=REPOSITORY / "research_runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clock = time.perf_counter()
    seconds = {}
    with tempfile.TemporaryDirectory(prefix="s4-probe-") as scratch:
        work = Path(scratch)
        runs, state = work / "runs", work / "state"
        state.mkdir()
        start = time.perf_counter()
        shutil.copytree(args.runs.resolve(), runs, symlinks=True)
        seconds["copy"] = time.perf_counter() - start

        def timed(name, function, *arguments):
            start = time.perf_counter()
            value = function(*arguments)
            seconds[name] = time.perf_counter() - start
            print(f"{name}: {seconds[name]:.1f} s", flush=True)
            return value

        attempts, roots = layout(runs)
        kinds, files, other = timed("inventory", inventory, runs, attempts, roots)
        writer_table = timed("writers", writers)
        varied = timed("variation", variation, files, runs, kinds)
        records, readable = timed("readability", readability, runs, attempts, roots)
        models, mutations = timed("input_models", input_models, files, kinds)
        run_draft = timed("drafted_run", drafted_run, attempts)
        archives = timed("arrays", arrays, files, kinds, runs)
        chosen = selection(records)
        cli_rows, cli_detail = timed("cli", cli_replies, runs, roots, chosen, work, state)
        start = time.perf_counter()
        mcp_rows, collected, largest, tools = asyncio.run(mcp_replies(runs, roots, chosen, state))
        seconds["mcp"] = time.perf_counter() - start
        print(f"mcp: {seconds['mcp']:.1f} s", flush=True)
        reply_draft = drafted_reply(cli_detail, collected)
        payload = asyncio.run(output_schema_payload(largest[2]))
        payload = {"tool": largest[1], "reply_bytes": largest[0], **payload}
        result = {
            "probe": "s4_formats",
            "source": source_identity(),
            "versions": {
                name: version(name)
                for name in ("entanglix", "pydantic", "jsonschema", "mcp", "numpy")
            },
            "python": sys.version.split()[0],
            "retained": {
                "roots": len(roots),
                "attempts": len(attempts),
                "files_outside_harness_layout": other,
            },
            "inventory": kinds,
            "writers": writer_table,
            "variation": varied,
            "readability": readable,
            "input_models": models,
            "int_float_mutations": mutations,
            "drafted_run_record": run_draft,
            "drafted_read_run_reply": reply_draft,
            "arrays": archives,
            "read_run_selection": [str(run.relative_to(runs)) for run in chosen],
            "cli_replies": cli_rows,
            "mcp_replies": mcp_rows,
            "mcp_output_schema_payload": payload,
            "size": size(kinds, cli_rows, mcp_rows, tools),
            "seconds": {**seconds, "total": time.perf_counter() - clock},
        }
        # Messages quote the disposable copy; keep the tracked results free of its location.
        text = json.dumps(result, indent=2, allow_nan=False, default=str)
        text = text.replace(str(runs), "<copy>").replace(str(work), "<work>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
