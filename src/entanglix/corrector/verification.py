"""Named checks on completed computations, with linked reports and scoped history."""

import json
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from entanglix.corrector import (
    _load_check_source,
    _source_artifact,
    create_verification_run,
    load_policy,
    load_verification,
    relevant_checks,
)
from entanglix.corrector.checks import check, inapplicable, output_checks
from entanglix.corrector.dynamics import (
    WHOLE_WINDOW_LIMIT,
    check_listing,
    diagnostics,
    refined_segments,
    refinement_reason,
)
from entanglix.corrector.metrics import metric_values, spectral_arrays
from entanglix.corrector.reference import prepare_refinement
from entanglix.evolution import normalize_input
from entanglix.protocols.plans import Refinement
from entanglix.runner import (
    _cpu_job_context,
    _digest,
    _execute_job,
    _load_record,
    _read_attempt,
    _script_result,
    capture_code,
    process_failure,
    publish_toml,
)


def _output(run):
    result = _script_result(run)
    with np.load(run / result["artifacts"]["output"]["path"], allow_pickle=False) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def integrity(loaded, arrays, result, policy):
    states = arrays["states"]
    scope = "every stored state and reported value"
    if any(not np.isfinite(value).all() for value in arrays.values()):
        return {
            "name": "integrity",
            "outcome": "fail",
            "scope": scope,
            "checks": [check("finite_outputs", 1, 0, scope=scope)],
        }
    dimension = int(np.prod(loaded.system.dimensions))
    if states.shape != (len(loaded.request.times_ns), dimension, dimension):
        return {
            "name": "integrity",
            "outcome": "fail",
            "scope": scope,
            "checks": [check("state_shape", 1, 0, scope=scope)],
        }
    checks, _ = output_checks(states, loaded.system.dimensions, policy=policy)
    initial = normalize_input(loaded.system, loaded.request).initial_density
    checks.append(
        check(
            "initial_state",
            float(np.linalg.norm(states[0] - initial)),
            policy.thresholds["output"]["initial_state_atol"],
            scope="first stored density matrix in the lab basis; Frobenius norm",
        )
    )
    grid_matches = np.array_equal(arrays["times_ns"], loaded.request.times_ns)
    checks.append(
        check(
            "output_grid",
            0 if grid_matches else 1,
            0,
            scope="declared output grid and stored state count",
        )
    )
    for name, operator in loaded.request.observables.items():
        expected = np.einsum("tij,ji->t", states, operator)
        reported = arrays[f"observable.{name}"]
        residual = (
            float(np.max(abs(expected - reported)))
            if reported.shape == expected.shape
            else float("inf")
        )
        checks.append(
            check(
                "observable_consistency",
                residual,
                policy.thresholds["refinement"]["observable_atol"],
                scope="every stored value",
                observable=name,
            )
        )
    expected_metrics = metric_values(loaded.definition.metrics, arrays)
    for name, value in expected_metrics.items():
        reported = result.get("metrics", {}).get(name)
        residual = abs(value - reported) if isinstance(reported, (int, float)) else float("inf")
        checks.append(
            check(
                "metric_consistency",
                residual,
                policy.thresholds["refinement"]["observable_atol"],
                scope="reported endpoint formula",
                metric=name,
            )
        )
    checks.append(
        check(
            "metric_names",
            0 if set(result.get("metrics", {})) == set(expected_metrics) else 1,
            0,
            scope="trusted metric definitions",
        )
    )
    return {
        "name": "integrity",
        "outcome": "fail" if any(item["outcome"] == "fail" for item in checks) else "pass",
        "scope": scope,
        "checks": checks,
    }


def spectral_refinement(loaded, arrays, policy):
    reason = refinement_reason(loaded, policy)
    if reason or not loaded.definition.refinement:
        return inapplicable(
            "spectral_refinement",
            reason or "no refinable subsystem",
            scope="reported excitation energies",
        )
    with tempfile.TemporaryDirectory(prefix="entanglix-spectral-refinement-") as directory:
        refined = prepare_refinement(loaded, Path(directory), policy)
        refined_arrays = spectral_arrays(
            refined.system, loaded.definition.levels, loaded.definition.metrics
        )
        energies = refined_arrays["energies_ghz"]
    stored = arrays["energies_ghz"]
    if stored.shape != energies.shape or not np.isfinite(stored).all():
        raise ValueError("stored spectrum does not match the declared reported levels")
    residual = float(np.max(abs(stored - stored[0] - energies + energies[0])))
    for name, value in metric_values(loaded.definition.metrics, arrays).items():
        residual = max(
            residual, abs(value - metric_values(refined.definition.metrics, refined_arrays)[name])
        )
    return check(
        "spectral_refinement",
        residual,
        policy.thresholds["refinement"]["excitation_convergence_atol_ghz"],
        scope="reported lowest excitation energies at the refined cutoff",
        engine="qutip-cpu",
        dimensions=list(refined.system.dimensions),
    )


def execute_verification(run: Path):
    started = time.monotonic()
    loaded, binding, request = load_verification(run)
    policy = load_policy(binding["policy_version"])
    # Applying a check's replacement affects only the local definition, never the source identity.
    declaration = {
        name: Refinement.model_validate(value) for name, value in request["refinement"].items()
    }
    source = (run.parent / binding["source_path"]).resolve()
    if "problem_sha256" in binding:
        from entanglix.corrector.control import CONTROL_LIMIT, control_checks
        from entanglix.solvers import Pulse

        result = _script_result(source)
        with np.load(source / _source_artifact(result)["path"], allow_pickle=False) as archive:
            pulse = Pulse(
                archive["times_ns"], archive["amplitudes_ghz"], str(archive["kind"].item())
            )
        kind, has_dynamics = "control", False
        outcomes = control_checks(
            loaded, pulse, result, request["checks"], policy, declaration=declaration
        )
    elif hasattr(loaded, "points"):
        from entanglix.corrector.sweeps import sweep_checks

        kind = loaded.manifest["kind"]
        has_dynamics = loaded.points[0].request is not None
        outcomes = sweep_checks(loaded, source, request, declaration, policy)
    else:
        kind = loaded.definition.kind
        has_dynamics = loaded.request is not None
        effective = replace(
            loaded, definition=loaded.definition.model_copy(update={"refinement": declaration})
        )
        arrays = _output(source)
        outcomes = []
        for name in request["checks"]:
            if name not in relevant_checks(kind):
                item = inapplicable(name, f"not applicable to {kind}", scope="source computation")
            elif name == "integrity":
                item = integrity(effective, arrays, _script_result(source), policy)
            elif name == "segments":
                item = refined_segments(effective, arrays, policy)
            elif name == "diagnostics":
                item = diagnostics(effective, arrays, policy)
            else:
                item = spectral_refinement(effective, arrays, policy)
            outcomes.append(item)
    report = {
        "kind": kind,
        "requested_checks": request["checks"],
        "checks": outcomes,
        "refinement": request["refinement"],
        "refinement_origin": request["refinement_origin"],
        "work_seconds": time.monotonic() - started,
        **({"limit": WHOLE_WINDOW_LIMIT} if has_dynamics else {}),
        **({"limit": CONTROL_LIMIT} if kind == "control" else {}),
    }
    load_verification(run)
    publish_toml(
        run / "result.toml",
        {"engine": "corrector", "metrics": {}, **binding, "report": report},
        replace=True,
    )


def _validate_report_binding(result, binding, request):
    if (
        any(result.get(name) != value for name, value in binding.items())
        or result.get("report", {}).get("requested_checks") != request["checks"]
        or [item["name"] for item in result.get("report", {}).get("checks", [])]
        != request["checks"]
    ):
        raise ValueError("check report does not match its source and request binding")


def verification_result(run):
    _, binding, request = load_verification(run)
    result = _load_record(run / "result.toml")
    _validate_report_binding(result, binding, request)
    return result


def check_run(
    source: Path,
    checks: tuple[str, ...] = (),
    *,
    repository: Path | None = None,
    timeout_seconds: float | None = None,
    refinement: Path | None = None,
):
    source = source.resolve()
    if not checks:
        if refinement is not None:
            raise ValueError("a replacement refinement requires named checks")
        record = _read_attempt(source)
        if record["manifest"]["protocol"] not in {"plan", "simulate", "sweep", "control"}:
            return {"checks": [], "reason": "run has no trusted built-in computation definition"}
        loaded = _load_check_source(source / "spec", record["manifest"]["protocol"])
        if record["manifest"]["protocol"] == "control":
            from entanglix.corrector.control import control_listing

            return control_listing(loaded)
        if hasattr(loaded, "points"):
            from entanglix.corrector.sweeps import sweep_listing

            return sweep_listing(loaded)
        return check_listing(loaded)
    repository = (repository or Path(__file__).resolve().parents[3]).resolve()
    command, environment, extras = _cpu_job_context(repository, "entanglix.corrector._job")
    gate = subprocess.run(
        [*command, "check"],
        input=json.dumps({"repository": str(repository), "extras": extras}),
        text=True,
        capture_output=True,
        env=environment,
    )
    if gate.returncode:
        raise ValueError(process_failure(gate, "check environment rejected"))
    with create_verification_run(
        source,
        checks=checks,
        environment=json.loads(gate.stdout),
        code=capture_code(repository),
        runs_root=source.parent.parent,
        timeout_seconds=timeout_seconds,
        refinement=refinement,
    ) as owner:
        environment.update(ENTANGLIX_RUN_DIR=str(owner.path))
        return _execute_job(
            owner,
            command,
            environment,
            timeout_seconds=timeout_seconds,
            job_input=(
                "control.toml"
                if "problem_sha256" in _load_record(owner.path / "run.toml")["launch"]
                else "plan.toml"
            ),
            validate_result=verification_result,
        )


def verification_history(source):
    """List all linked attempts, including failures, without replacing their outcomes."""
    source = source.resolve()
    result = _script_result(source) if (source / "result.toml").exists() else None
    source_digest = _digest((source / "result.toml").read_bytes()) if result else None
    history = []
    for path in sorted(source.parent.glob("*/run.toml")):
        try:
            manifest = _load_record(path)
            if manifest["protocol"] != "verification" or manifest["parent_run_ids"] != [
                source.name
            ]:
                continue
            record = _read_attempt(path.parent)
            binding = manifest["launch"]
            item = {
                "path": str(path.parent),
                "run_id": manifest["run_id"],
                "status": record["status"],
                "policy_version": binding["policy_version"],
                "source_matches": source_digest == binding["source_result_sha256"]
                and result is not None
                and _source_artifact(result)["sha256"] == binding["source_artifact_sha256"],
            }
            if record["status"] == "completed" and item["source_matches"]:
                request_path = path.parent / "spec" / binding["request_file"]
                if (
                    _digest(request_path.read_bytes())
                    != manifest["spec_sha256"][binding["request_file"]]
                ):
                    raise ValueError("frozen check request digest mismatch")
                request = _load_record(request_path)
                report = record["result"]
                _validate_report_binding(report, binding, request)
                item["checks"] = report["report"]["checks"]
            history.append(item)
        except (OSError, ValueError, KeyError) as exc:
            history.append({"path": str(path.parent), "status": "invalid", "diagnostic": str(exc)})
    return history
