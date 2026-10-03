"""Trusted check definitions and bounded, versioned verification policy."""

import hashlib
import itertools
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POLICY_DIRECTORY = Path(__file__).parent


def policy_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted([*POLICY_DIRECTORY.glob("*.py"), POLICY_DIRECTORY / "thresholds.toml"]):
        content = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + str(len(content)).encode() + b"\0" + content)
    return digest.hexdigest()


@dataclass(frozen=True)
class Policy:
    version: str
    thresholds: dict


def load_policy(expected_version: str | None = None) -> Policy:
    version = policy_digest()
    if expected_version is not None and version != expected_version:
        raise ValueError("verification policy changed between launch and load")
    thresholds = tomllib.loads((POLICY_DIRECTORY / "thresholds.toml").read_text())
    if thresholds["format_version"] != 1:
        raise ValueError("unsupported verification policy format")
    return Policy(version, thresholds)


def validate_refinement(plan, document: dict, root: Path, policy: Policy) -> None:
    model = document["model"]
    if model["kind"] == "composite":
        subsystems = {}
        for item in model["subsystems"]:
            if "source" in item:
                leaf = tomllib.loads((root / item["source"]).read_text())
            else:
                leaf = {"model": item}
            subsystems[item["id"]] = leaf
    else:
        subsystems = {document["id"]: document}
    required = {name for name, leaf in subsystems.items() if leaf["model"]["kind"] != "operators"}
    if plan.refinement.keys() != required:
        raise ValueError(
            f"refinement must declare exactly the truncated subsystems: {sorted(required)}"
        )
    minimum = policy.thresholds["refinement"]
    for name in required:
        leaf = subsystems[name]
        cutoffs = set(leaf.get("truncation", {}).get("cutoffs", {}))
        if "cutoff" in leaf["model"]:
            cutoffs.add("cutoff")
        step = plan.refinement[name]
        if step.retained_levels_step < minimum["min_retained_levels_step"]:
            raise ValueError(f"refinement.{name}: retained-level step below policy minimum")
        if step.cutoff_steps.keys() != cutoffs or any(
            value < minimum["min_cutoff_step"] for value in step.cutoff_steps.values()
        ):
            raise ValueError(
                f"refinement.{name}: cutoff steps must cover {sorted(cutoffs)} and meet policy minimum"
            )


CHECK_DEFINITIONS = {
    "integrity": "Corrupted or unphysical stored states, wrong initial state, or inconsistent reported values",
    "segments": "Local integration or truncation disagreement over sampled windows",
    "diagnostics": "Top-level population and static shifts; descriptive only, with no pass/fail outcome",
    "spectral_refinement": "Truncation sensitivity of the reported excitation energies",
    "declared_expectation": "A result contradicting the expectation declared in the sweep spec",
}


def relevant_checks(kind: str) -> tuple[str, ...]:
    if kind == "lindblad":
        return ("integrity", "segments", "diagnostics")
    if kind == "spectrum":
        return ("spectral_refinement",)
    raise ValueError(f"unknown computation definition: {kind!r}")


@dataclass(frozen=True)
class EvidenceSelection:
    points: tuple[int, ...]
    reasons: dict[int, tuple[str, ...]]
    uncovered: tuple[int, ...]


def select_evidence(
    shape: tuple[int, ...],
    *,
    metric: np.ndarray | None = None,
    warnings: np.ndarray | None = None,
    policy: Policy | None = None,
) -> EvidenceSelection:
    """Corners first, then extrema, steepest edge, and ranked truncation warnings."""
    policy = policy or load_policy()
    if not shape or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("grid shape must contain positive integers")
    bound = policy.thresholds["evidence"]["max_points"]
    if math.prod(1 if size == 1 else 2 for size in shape) > bound:
        raise ValueError("grid corners exceed the verification evidence-point bound")
    candidates: dict[int, list[str]] = {}

    def add(index, reason):
        candidates.setdefault(int(index), []).append(reason)

    for corner in itertools.product(*[(0,) if size == 1 else (0, size - 1) for size in shape]):
        add(np.ravel_multi_index(corner, shape), "corner")
    if metric is not None:
        values = np.asarray(metric, dtype=float)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError("metric must be finite and match the grid shape")
        add(np.argmin(values), "metric_minimum")
        add(np.argmax(values), "metric_maximum")
        edges = []
        for axis, size in enumerate(shape):
            if size == 1:
                continue
            differences = np.abs(np.diff(values, axis=axis))
            coordinate = np.unravel_index(np.argmax(differences), differences.shape)
            first = int(np.ravel_multi_index(coordinate, shape))
            second_coordinate = list(coordinate)
            second_coordinate[axis] += 1
            second = int(np.ravel_multi_index(tuple(second_coordinate), shape))
            edges.append((-float(differences[coordinate]), first, second))
        if edges:
            _, first, _ = min(edges)
            add(first, "steepest_change")
    if warnings is not None:
        values = np.asarray(warnings, dtype=float)
        if values.shape != shape or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("warnings must be finite, nonnegative, and match the grid shape")
        for index in np.argsort(-values.ravel(), kind="stable"):
            if values.ravel()[index] > 0:
                add(index, "truncation_warning")
    points = tuple(list(candidates)[:bound])
    return EvidenceSelection(
        points,
        {index: tuple(reasons) for index, reasons in candidates.items()},
        tuple(list(candidates)[bound:]),
    )


def create_verification_run(
    source: Path,
    *,
    checks: tuple[str, ...],
    environment: dict,
    code,
    runs_root: Path,
    timeout_seconds: float | None = None,
    refinement: Path | None = None,
):
    """Freeze a named request and bind its linked attempt to the unchanged source."""
    import tempfile

    import tomli_w
    from pydantic import TypeAdapter

    from entanglix.protocols.plans import Refinement, load_computation
    from entanglix.runner import _digest, _read_attempt, _relative_file, _script_result, create_run

    if not checks or set(checks) - CHECK_DEFINITIONS.keys():
        raise ValueError("name at least one known check")
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    source = source.resolve()
    record = _read_attempt(source)
    if record["status"] != "completed" or record["manifest"]["protocol"] not in {
        "plan",
        "simulate",
        "sweep",
    }:
        raise ValueError("checks require a completed built-in computation run")
    manifest = record["manifest"]
    for name, digest in manifest["spec_sha256"].items():
        if _digest((source / "spec" / _relative_file(name)).read_bytes()) != digest:
            raise ValueError(f"source input digest mismatch: {name}")
    result = _script_result(source)
    loaded = load_computation(source / "spec")
    point = loaded.points[0] if hasattr(loaded, "points") else loaded
    if (
        loaded.digest != manifest["launch"]["plan_sha256"]
        or result.get("plan_sha256") != loaded.digest
        or result.get("kind") != point.definition.kind
        or manifest["launch"]["kind"] != point.definition.kind
        or result.get("engine") != point.definition.engine
        or manifest["engine"] != point.definition.engine
        or result.get("dimensions") != list(loaded.system.dimensions)
    ):
        raise ValueError("source result is inconsistent with its trusted execution plan")
    declaration = point.definition.refinement
    if refinement is not None:
        document = tomllib.loads(refinement.read_text())
        if set(document) != {"refinement"}:
            raise ValueError("replacement file must contain only [refinement.*] tables")
        declaration = TypeAdapter(dict[str, Refinement]).validate_python(document["refinement"])
    request = {
        "checks": list(dict.fromkeys(checks)),
        "refinement": {name: value.model_dump() for name, value in declaration.items()},
        "refinement_origin": "request" if refinement is not None else "source spec",
    }
    policy = load_policy()
    binding = {
        "source_run_id": manifest["run_id"],
        "source_path": os.path.relpath(source, runs_root.resolve() / loaded.system.id),
        "source_result_sha256": _digest((source / "result.toml").read_bytes()),
        "source_artifact_sha256": result["artifacts"]["output"]["sha256"],
        "plan_sha256": loaded.digest,
        "policy_version": policy.version,
        **({"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}),
    }
    name = "check-request.toml"
    while name in loaded.inputs:
        name = "check-" + name
    binding["request_file"] = name
    with tempfile.TemporaryDirectory(prefix="entanglix-check-request-") as directory:
        path = Path(directory) / name
        path.write_text(tomli_w.dumps(request))
        owner = create_run(
            system_id=loaded.system.id,
            protocol="verification",
            inputs={**loaded.inputs, name: path},
            environment=environment,
            code=code,
            engine="corrector",
            device="cpu",
            needs={"gradients": False, "device": "cpu"},
            runs_root=runs_root,
            parent_run_ids=(manifest["run_id"],),
            launch=binding,
        )
    frozen = _read_attempt(owner.path)["manifest"]["spec_sha256"]
    if any(frozen.get(key) != value for key, value in loaded.input_digests.items()):
        from entanglix.runner import record_finish

        try:
            record_finish(owner, status="failed", exit_code=64, wall_seconds=0, gpu_seconds=0)
        finally:
            owner.close()
        raise ValueError(f"source changed during check preparation: {owner.path}")
    return owner


def load_verification(run: Path):
    """Recheck request/input digests, policy, and the recorded completed parent."""
    from entanglix.protocols.plans import load_computation
    from entanglix.runner import (
        _digest,
        _load_record,
        _read_attempt,
        _relative_file,
        _script_result,
    )

    manifest = _load_record(run / "run.toml")
    if manifest["protocol"] != "verification":
        raise ValueError("attempt is not a verification run")
    binding = manifest["launch"]
    load_policy(binding["policy_version"])
    for name, digest in manifest["spec_sha256"].items():
        if _digest((run / "spec" / _relative_file(name)).read_bytes()) != digest:
            raise ValueError(f"verification input digest mismatch: {name}")
    loaded = load_computation(run / "spec")
    if loaded.digest != binding["plan_sha256"] or manifest["parent_run_ids"] != [
        binding["source_run_id"]
    ]:
        raise ValueError("verification source binding is inconsistent")
    source = (run.parent / binding["source_path"]).resolve()
    record = _read_attempt(source)
    if record["status"] != "completed" or record["manifest"]["run_id"] != binding["source_run_id"]:
        raise ValueError("verification source is not the completed recorded parent")
    result = _script_result(source)
    if (
        _digest((source / "result.toml").read_bytes()) != binding["source_result_sha256"]
        or result["artifacts"]["output"]["sha256"] != binding["source_artifact_sha256"]
    ):
        raise ValueError("verification source digest mismatch")
    request = _load_record(run / "spec" / _relative_file(binding["request_file"]))
    if not request["checks"] or set(request["checks"]) - CHECK_DEFINITIONS.keys():
        raise ValueError("invalid frozen check request")
    return loaded, binding, request
