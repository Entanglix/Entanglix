"""Linked scientific verification, validated reference reuse, and verdict inspection."""

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from entanglix.corrector import (
    create_verification_run,
    load_policy,
    load_verification,
    policy_digest,
    required_checks,
    select_evidence,
)
from entanglix.corrector.checks import (
    check,
    coverage_report,
    inapplicable,
    output_checks,
    spectral_checks,
)
from entanglix.corrector.reference import prepare_reference
from entanglix.protocols.plans import load_plan
from entanglix.runner import (
    _cpu_job_context,
    _digest,
    _execute_job,
    _load_record,
    _read_attempt,
    _script_result,
    abort_run,
    capture_code,
    publish_toml,
    run_plan,
)


def _output(run):
    result = _script_result(run)
    path = run / result["artifacts"]["output"]["path"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("computation output contains nonfinite values")
    return arrays


def _records(root):
    for path in sorted(root.glob("*/run.toml")):
        try:
            manifest = _load_record(path)
        except (OSError, ValueError):
            continue
        try:
            yield path.parent, _read_attempt(path.parent)
        except (OSError, ValueError, KeyError) as exc:
            yield path.parent, {"manifest": manifest, "status": "invalid", "log_tail": str(exc)}


def _reference(expected, verification, repository):
    root = verification.parent.parent
    # Check completion, raw input integrity, semantic identity, policy and output
    # integrity before reusing a solve. A result left by a lost runner is not evidence.
    for candidate, record in _records(verification.parent):
        manifest = record["manifest"]
        if (
            record["status"] != "completed"
            or manifest["protocol"] != "reference"
            or manifest["launch"].get("plan_sha256") != expected.digest
            or manifest["launch"].get("policy_version") != expected.version
        ):
            continue
        try:
            if any(
                _digest((candidate / "spec" / name).read_bytes()) != digest
                for name, digest in manifest["spec_sha256"].items()
            ):
                continue
            retained = load_plan(
                candidate / "spec/system.toml",
                candidate / "spec/plan.toml",
                expected_policy=expected.version,
            )
            result = _script_result(candidate)
            if (
                retained.digest != expected.digest
                or result.get("plan_sha256") != expected.digest
                or result.get("policy_version") != expected.version
                or result.get("engine") != expected.definition.engine
                or result.get("kind") != expected.definition.kind
            ):
                continue
            return candidate, _output(candidate), True
        except (OSError, ValueError, KeyError):
            continue
    job = run_plan(
        expected.inputs["system.toml"],
        expected.inputs["plan.toml"],
        runs_root=root,
        repository=repository,
        parent_run_ids=(verification.name,),
    )
    reference = Path(job["path"])
    if job["exit_code"]:
        raise ValueError(
            f"reference solve failed: {reference}; {_read_attempt(reference)['log_tail']}"
        )
    return reference, _output(reference), False


def execute_verification(run: Path, repository: Path):
    loaded, binding = load_verification(run)
    policy = load_policy(binding["policy_version"])
    source = (run.parent / binding["source_path"]).resolve()
    arrays = _output(source)
    scope = "single evidence point; entire reported time window"
    report = {
        "kind": loaded.definition.kind,
        "required_checks": list(required_checks(loaded.definition.kind)),
        "coverage": coverage_report(select_evidence((1,), policy=policy), (1,)),
        "checks": [],
        "warnings": [],
        "recommendations": [],
    }
    checks = report["checks"]
    if loaded.request is not None:
        if not np.array_equal(arrays["times_ns"], loaded.request.times_ns):
            raise ValueError("stored times do not match the validated execution plan")
        if len(arrays["states"]) != len(loaded.request.times_ns):
            raise ValueError("stored state count does not match the execution plan")
        checks[:], report["warnings"] = output_checks(
            arrays["states"], loaded.system.dimensions, policy=policy
        )
        for name, operator in loaded.request.observables.items():
            reported = arrays[f"observable.{name}"]
            calculated = np.einsum("tij,ji->t", arrays["states"], operator)
            if reported.shape != calculated.shape:
                raise ValueError(f"observable {name}: stored shape does not match the plan")
            checks.append(
                check(
                    "observable_consistency",
                    float(np.max(abs(reported - calculated))),
                    policy.thresholds["reference"]["observable_atol"],
                    scope=scope,
                    observable=name,
                )
            )
    else:
        energies = arrays["energies_ghz"]
        if energies.shape != (loaded.definition.levels,):
            raise ValueError("stored spectrum level count does not match the plan")
        # A second eigensolver checks the declared matrix. It does not establish
        # correctness of an arbitrary circuit's physical model.
        checks.extend(spectral_checks(loaded.system, energies, policy=policy))

    dimension = (
        math.prod(
            count
            + (
                loaded.definition.refinement[name].retained_levels_step
                if name in loaded.definition.refinement
                else 0
            )
            for name, count in zip(loaded.system.subsystem_ids, loaded.system.dimensions)
        )
        if loaded.definition.refinement
        else math.prod(loaded.system.dimensions)
    )
    reference_check = "cpu_reference" if loaded.request is not None else "truncation_convergence"
    if dimension > policy.thresholds["reference"]["max_dimension"]:
        checks.append(
            {
                "name": reference_check,
                "scope": scope,
                "outcome": "fail",
                "reason": "combined refinement exceeds the validated CPU reference limit",
                "reference_dimension": dimension,
                "dimension_limit": policy.thresholds["reference"]["max_dimension"],
            }
        )
    elif loaded.request is not None and not loaded.request.observables:
        checks.append(
            {
                "name": reference_check,
                "scope": scope,
                "outcome": "fail",
                "reason": "no reported observables to compare after refinement",
            }
        )
    elif loaded.request is None and not loaded.definition.refinement:
        checks.append(
            inapplicable(
                reference_check,
                "finite explicit matrix has no truncation",
                scope="declared finite matrix",
            )
        )
    else:
        with tempfile.TemporaryDirectory(prefix="entanglix-reference-") as directory:
            expected = prepare_reference(loaded, Path(directory), policy)
            reference, refined, reused = _reference(expected, run, repository)
        report["reference"] = {
            "run_id": reference.name,
            "reused": reused,
            "engine": expected.definition.engine,
            "dimensions": list(expected.system.dimensions),
            "plan_sha256": expected.digest,
            "result_sha256": _digest((reference / "result.toml").read_bytes()),
            "artifact_sha256": _script_result(reference)["artifacts"]["output"]["sha256"],
            "initial_state_mapping": "ordered low-level subspace, with local operator gauge alignment"
            if loaded.request
            else "not applicable to spectra",
        }
        if loaded.request is not None:
            report["reference"].update(
                rtol=expected.definition.evolution.rtol,
                atol=expected.definition.evolution.atol,
                max_steps=expected.definition.evolution.max_steps,
            )
            reference_checks, _ = output_checks(
                refined["states"], expected.system.dimensions, policy=policy
            )
            for item in reference_checks:
                item["name"] = "reference_" + item["name"]
            checks.extend(reference_checks)
            for name in loaded.request.observables:
                original, target = arrays[f"observable.{name}"], refined[f"observable.{name}"]
                if original.shape != target.shape or not np.array_equal(
                    arrays["times_ns"], refined["times_ns"]
                ):
                    raise ValueError("reference output grid does not match the source")
                checks.append(
                    check(
                        reference_check,
                        float(np.max(abs(original - target))),
                        policy.thresholds["reference"]["observable_atol"],
                        scope=scope,
                        observable=name,
                        reference_engine=expected.definition.engine,
                        reference_dimension=math.prod(expected.system.dimensions),
                    )
                )
        elif loaded.definition.refinement:
            original = arrays["energies_ghz"] - arrays["energies_ghz"][0]
            target = refined["energies_ghz"][: len(original)] - refined["energies_ghz"][0]
            checks.append(
                check(
                    reference_check,
                    float(np.max(abs(original - target))),
                    policy.thresholds["reference"]["excitation_convergence_atol_ghz"],
                    scope="every reported excitation energy",
                    reference_engine="qutip-cpu",
                    reference_dimension=math.prod(expected.system.dimensions),
                )
            )
    checks.extend(
        [
            inapplicable(
                "seed_reproducibility", "deterministic computation has no random seed", scope=scope
            ),
            inapplicable(
                "declared_monotonicity",
                "single computation has no parameter-series metric",
                scope=scope,
            ),
        ]
    )
    verdict = "UNVERIFIED" if any(item["outcome"] == "fail" for item in checks) else "VERIFIED"
    if verdict == "UNVERIFIED":
        report["recommendations"].append(
            "Refine truncations or integration tolerances; inspect failed checks."
        )
    # Recheck the source and pinned policy after the potentially long reference solve.
    load_verification(run)
    publish_toml(
        run / "result.toml",
        {"engine": "corrector", "metrics": {}, "verdict": verdict, **binding, "report": report},
        replace=True,
    )


def verification_result(run):
    _, binding = load_verification(run)
    result = _load_record(run / "result.toml")
    if result.get("verdict") not in {"VERIFIED", "UNVERIFIED"} or any(
        result.get(name) != value for name, value in binding.items()
    ):
        raise ValueError("verification result does not match its source and policy binding")
    return result


def compare_reference(
    source: Path, *, repository: Path | None = None, timeout_seconds: float | None = None
):
    repository = (repository or Path(__file__).resolve().parents[3]).resolve()
    source = source.resolve()
    command, environment, extras = _cpu_job_context(repository, "entanglix.corrector._job")
    check_gate = subprocess.run(
        [*command, "check"],
        input=json.dumps({"repository": str(repository), "extras": extras}),
        text=True,
        capture_output=True,
        env=environment,
    )
    if check_gate.returncode:
        raise ValueError(check_gate.stderr.strip() or "verification environment rejected")
    with create_verification_run(
        source,
        environment=json.loads(check_gate.stdout),
        code=capture_code(repository),
        runs_root=source.parent.parent,
        timeout_seconds=timeout_seconds,
    ) as owner:
        environment.update(ENTANGLIX_RUN_DIR=str(owner.path), ENTANGLIX_REPOSITORY=str(repository))

        def cleanup():
            # Nested reference jobs have their own groups. Stop survivors even if
            # the verification worker was forcibly killed during its own cleanup.
            for path, record in _records(source.parent):
                if (
                    record["manifest"]["parent_run_ids"] == [owner.path.name]
                    and record["status"] == "active"
                    and "start" in record
                ):
                    abort_run(path)

        return _execute_job(
            owner,
            command,
            environment,
            timeout_seconds=timeout_seconds,
            job_input="plan.toml",
            validate_result=verification_result,
            cleanup=cleanup,
        )


def verification_history(source, *, policy_version=None):
    policy_version = policy_version or policy_digest()
    history = []
    source_identity = None
    source_error = None
    references = {}
    for path, record in _records(source.parent):
        manifest = record["manifest"]
        if manifest["protocol"] != "verification" or manifest["parent_run_ids"] != [source.name]:
            continue
        binding = manifest["launch"]
        entry = {
            "path": str(path),
            "run_id": manifest["run_id"],
            "status": record["status"],
            "created_at": manifest["created_at"],
            "policy_version": binding["policy_version"],
            "source_result_sha256": binding["source_result_sha256"],
            "source_artifact_sha256": binding["source_artifact_sha256"],
            "verdict_available": False,
        }
        if record["status"] == "completed":
            try:
                result = record["result"]
                if source_identity is None and source_error is None:
                    try:
                        source_identity = (
                            _digest((source / "result.toml").read_bytes()),
                            _script_result(source)["artifacts"]["output"]["sha256"],
                        )
                    except (OSError, ValueError, KeyError) as exc:
                        source_error = str(exc)
                if source_error is not None:
                    raise ValueError(source_error)
                if (
                    result.get("verdict") not in {"VERIFIED", "UNVERIFIED"}
                    or any(result.get(name) != value for name, value in binding.items())
                    or source_identity
                    != (binding["source_result_sha256"], binding["source_artifact_sha256"])
                ):
                    raise ValueError("verification no longer matches its retained source digests")
                if "reference" in result["report"]:
                    evidence = result["report"]["reference"]
                    reference = source.parent / evidence["run_id"]
                    if reference not in references:
                        try:
                            references[reference] = (
                                _read_attempt(reference)["status"],
                                _digest((reference / "result.toml").read_bytes()),
                                _script_result(reference)["artifacts"]["output"]["sha256"],
                            )
                        except (OSError, ValueError, KeyError) as exc:
                            references[reference] = str(exc)
                    if references[reference] != (
                        "completed",
                        evidence["result_sha256"],
                        evidence["artifact_sha256"],
                    ):
                        raise ValueError("verification reference evidence is missing or changed")
                entry.update(verdict_available=True, verdict=result["verdict"])
            except (OSError, ValueError, KeyError) as exc:
                entry["diagnostic"] = str(exc)
        else:
            entry["diagnostic"] = record.get("result_error") or record["log_tail"]
        history.append(entry)
    history.sort(key=lambda entry: (entry["created_at"], entry["run_id"]))
    matching = [entry for entry in history if entry["policy_version"] == policy_version]
    return {
        "requested_policy": policy_version,
        "latest": history[-1] if history else {},
        "current": matching[-1] if matching else {},
        "history": history,
    }
