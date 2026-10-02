"""Scoped scientific verdicts, independent refinement, reuse, and visible failures."""

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tomli_w

from entanglix.corrector import load_policy, select_evidence
from entanglix.corrector.checks import coverage_report, output_checks, series_checks, series_verdict
from entanglix.corrector.reference import embed_state, prepare_reference
from entanglix.corrector.verification import compare_reference
from entanglix.models import load_system
from entanglix.protocols.plans import load_plan
from entanglix.runner import (
    _digest,
    _group_alive,
    _read_attempt,
    abort_run,
    publish_toml,
    read_run,
    run_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def files(directory, *, kind="lindblad"):
    directory.mkdir(exist_ok=True)
    for name in ("system.toml", "operators.npz"):
        shutil.copyfile(ROOT / "examples/two-level" / name, directory / name)
    np.savez(directory / "state.npz", initial=np.array([1, 1j]) / np.sqrt(2))
    plan = {"format_version": 1, "kind": kind, "engine": "qutip-cpu"}
    if kind == "lindblad":
        plan["evolution"] = {
            "array_file": "state.npz",
            "initial_state": "initial",
            "times_ns": [0, 10, 20],
            "observables": {"coherence": "lower"},
        }
    else:
        plan["levels"] = 2
    (directory / "plan.toml").write_text(tomli_w.dumps(plan))
    return plan


def source_run(directory, repository):
    job = run_plan(
        directory / "system.toml",
        directory / "plan.toml",
        repository=repository,
        runs_root=directory / "runs",
    )
    assert job["exit_code"] == 0, read_run(Path(job["path"]))["log_tail"]
    return Path(job["path"])


def verified(source, repository):
    job = compare_reference(source, repository=repository)
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 0, record["log_tail"]
    return record


def test_output_checks_cover_all_times_and_warning_is_not_a_gate():
    states = np.array([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])], dtype=complex)
    checks, warnings = output_checks(states, (2,))
    assert all(item["outcome"] == "pass" for item in checks)
    assert warnings[0]["triggered"]
    states[1] = [[-0.1, 0.2j], [0.1, 1.2]]
    checks, _ = output_checks(states, (2,))
    assert {item["name"] for item in checks if item["outcome"] == "fail"} == {
        "trace",
        "hermiticity",
        "positivity",
    }


def test_coverage_reports_uncovered_warnings_without_extra_solves():
    selection = select_evidence((12,), warnings=np.ones(12))
    report = coverage_report(selection, (12,))
    assert len(report["evidence_points"]) == 8
    assert not report["complete"]
    assert report["uncovered_points"]
    assert set(report["output_checked_points"]).isdisjoint(report["evidence_points"])
    passed = [{"outcome": "pass"}]
    assert (
        series_verdict(
            [passed] * 12, {point: passed for point in selection.points}, selection, point_count=12
        )
        == "UNVERIFIED"
    )
    selection = select_evidence((2,))
    assert (
        series_verdict([passed] * 2, {0: passed, 1: passed}, selection, point_count=2) == "VERIFIED"
    )
    assert series_verdict([passed] * 2, {0: passed}, selection, point_count=2) == "UNVERIFIED"
    assert (
        series_verdict([passed], {0: passed, 1: passed}, selection, point_count=2) == "UNVERIFIED"
    )
    assert (
        series_verdict([[], []], {0: passed, 1: passed}, selection, point_count=2) == "UNVERIFIED"
    )


def test_series_checks_add_to_trusted_point_definitions():
    checks = series_checks("lindblad", [0, 1, 2], monotonic_direction="increasing")
    assert checks[0]["outcome"] == "inapplicable"
    assert checks[1]["outcome"] == "pass"
    checks = series_checks(
        "spectrum",
        [0, 2, 1],
        monotonic_direction="increasing",
    )
    assert checks[0]["outcome"] == "inapplicable"
    assert checks[1]["outcome"] == "fail"
    assert series_checks("spectrum", [1, 2])[1]["outcome"] == "inapplicable"
    with pytest.raises(ValueError, match="unknown or inconsistent"):
        series_checks("spectral-label", [0, 1])


def test_spectral_verification_and_cli_inspection(tmp_path, runner_repository):
    files(tmp_path, kind="spectrum")
    source = source_run(tmp_path, repository=runner_repository)
    launched = subprocess.run(
        [sys.executable, "-m", "entanglix", "compare-reference", str(source)],
        capture_output=True,
        text=True,
        env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
    )
    assert launched.returncode == 0, launched.stderr
    current = read_run(source)["verification"]["current"]
    record = read_run(Path(current["path"]))
    assert current["verdict_available"]
    assert record["result"]["verdict"] == "VERIFIED"
    checks = {item["name"]: item for item in record["result"]["report"]["checks"]}
    assert checks["spectral_consistency"]["outcome"] == "pass"
    assert checks["spectral_reference"]["outcome"] == "inapplicable"
    assert checks["truncation_convergence"]["outcome"] == "inapplicable"
    assert "reference" not in record["result"]["report"]
    command = subprocess.run(
        [
            sys.executable,
            "-m",
            "entanglix",
            "read-run",
            str(source),
            "--policy-version",
            load_policy().version,
        ],
        capture_output=True,
        text=True,
        env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
    )
    assert command.returncode == 0, command.stderr
    assert '"verdict_available": true' in command.stdout


def test_dynamiqs_combined_reference_reuse_and_visible_new_failure(
    tmp_path, differentiable_repository
):
    files(tmp_path)
    source = source_run(tmp_path, differentiable_repository)
    before = {
        name: (source / name).read_bytes() for name in ("result.toml", "run.toml", "FINISHED")
    }
    first = verified(source, differentiable_repository)
    assert first["result"]["verdict"] == "VERIFIED"
    report = first["result"]["report"]
    assert report["reference"]["engine"] == "dynamiqs"
    assert not report["reference"]["reused"]
    second = verified(source, differentiable_repository)
    assert second["result"]["report"]["reference"]["reused"]
    assert second["result"]["report"]["reference"]["run_id"] == report["reference"]["run_id"]
    assert {name: (source / name).read_bytes() for name in before} == before
    from entanglix.corrector import create_verification_run
    from entanglix.runner import CodeSnapshot, capture_environment, record_finish

    with create_verification_run(
        source,
        environment=capture_environment(ROOT / "uv.lock"),
        code=CodeSnapshot("test", b"", {}),
        runs_root=source.parent.parent,
    ) as owner:
        record_finish(owner, status="failed", exit_code=64, wall_seconds=0, gpu_seconds=0)
    history = read_run(source)["verification"]
    assert history["latest"]["run_id"] == owner.path.name
    assert history["current"]["status"] == "failed"
    assert not history["current"]["verdict_available"]
    assert history["history"][0]["verdict"] == "VERIFIED"
    assert read_run(source, policy_version="other-policy")["verification"]["current"] == {}
    # Matching semantic plans are not enough when a retained artifact has changed.
    reference = source.parent / report["reference"]["run_id"]
    (reference / "artifacts/output.npz").write_bytes(b"damaged")
    third = verified(source, differentiable_repository)
    assert not third["result"]["report"]["reference"]["reused"]
    # Different sources can reuse identical tightened reference plans.
    plan = files(tmp_path)
    plan["evolution"]["rtol"] = 1e-9
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    changed = verified(source_run(tmp_path, differentiable_repository), differentiable_repository)
    assert changed["result"]["report"]["reference"]["reused"]
    assert (
        changed["result"]["report"]["reference"]["run_id"]
        == third["result"]["report"]["reference"]["run_id"]
    )


def test_missing_reference_engine_leaves_failed_verification_without_verdict(
    tmp_path, runner_repository
):
    files(tmp_path)
    source = source_run(tmp_path, runner_repository)
    job = compare_reference(source, repository=runner_repository)
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 64
    assert "differentiable" in record["log_tail"]
    assert "result" not in record
    assert not read_run(source)["verification"]["current"]["verdict_available"]


@pytest.mark.parametrize("failure", ["truncation", "tolerance"])
def test_dynamiqs_underconverged_run_is_unverified(tmp_path, differentiable_repository, failure):
    plan = files(tmp_path)
    if failure == "truncation":
        system = {
            "format_version": 1,
            "id": "driven_oscillator",
            "units": {"energy": "GHz", "time": "ns"},
            "model": {
                "kind": "composite",
                "subsystems": [
                    {
                        "id": "r",
                        "kind": "oscillator",
                        "frequency": {"value": 0.2, "unit": "GHz"},
                        "retained_levels": 2,
                    },
                    {
                        "id": "s",
                        "kind": "oscillator",
                        "frequency": {"value": 0.3, "unit": "GHz"},
                        "retained_levels": 2,
                    },
                ],
            },
            "drives": [{"id": "drive", "factors": ["r.position"]}],
        }
        (tmp_path / "system.toml").write_text(tomli_w.dumps(system))
        np.savez(tmp_path / "state.npz", initial=np.array([1, 0, 0, 0]))
        plan["refinement"] = {"r": {"retained_levels_step": 1}, "s": {"retained_levels_step": 1}}
        plan["evolution"].update(
            times_ns=[0, 1, 2],
            observables={"occupation": "r.number"},
            drive_amplitudes_ghz={"drive": 0.15},
        )
    else:
        plan["evolution"].update(rtol=0.1, atol=0.1, times_ns=list(range(11)))
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    source = source_run(tmp_path, differentiable_repository)
    record = verified(source, differentiable_repository)
    assert record["result"]["verdict"] == "UNVERIFIED"
    assert any(
        item["name"] == "cpu_reference" and item["outcome"] == "fail"
        for item in record["result"]["report"]["checks"]
    )
    assert record["result"]["report"]["recommendations"]


def test_spectral_refinement_detects_small_cutoff(tmp_path, runner_repository):
    files(tmp_path, kind="spectrum")
    shutil.copyfile(ROOT / "examples/fluxonium/system.toml", tmp_path / "system.toml")
    text = (tmp_path / "system.toml").read_text().replace("theta1 = 60", "theta1 = 20")
    (tmp_path / "system.toml").write_text(text)
    plan = {
        "format_version": 1,
        "kind": "spectrum",
        "levels": 4,
        "refinement": {
            "fluxonium_circuit": {"retained_levels_step": 1, "cutoff_steps": {"theta1": 20}}
        },
    }
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    record = verified(
        source_run(tmp_path, repository=runner_repository), repository=runner_repository
    )
    assert record["result"]["verdict"] == "UNVERIFIED"
    assert record["result"]["report"]["reference"]["dimensions"] == [7]


def test_lc_spectrum_has_analytic_and_refinement_evidence(tmp_path, runner_repository):
    files(tmp_path, kind="spectrum")
    shutil.copyfile(ROOT / "examples/lc-circuit/system.toml", tmp_path / "system.toml")
    plan = {
        "format_version": 1,
        "kind": "spectrum",
        "levels": 4,
        "refinement": {"lc_circuit": {"retained_levels_step": 1, "cutoff_steps": {"theta1": 2}}},
    }
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    record = verified(
        source_run(tmp_path, repository=runner_repository), repository=runner_repository
    )
    assert record["result"]["verdict"] == "VERIFIED"
    checks = record["result"]["report"]["checks"]
    assert any(
        item["name"] == "spectral_reference"
        and item.get("reference_engine") == "analytic-LC"
        and item["outcome"] == "pass"
        for item in checks
    )


def test_policy_mismatch_fails_real_verifier_without_verdict(
    tmp_path, monkeypatch, runner_repository
):
    from entanglix.corrector import verification

    files(tmp_path, kind="spectrum")
    source = source_run(tmp_path, repository=runner_repository)
    execute = verification._execute_job

    def mismatch(owner, *args, **kwargs):
        manifest = read_run(owner.path)["manifest"]
        manifest["launch"]["policy_version"] = "changed-policy"
        publish_toml(owner.path / "run.toml", manifest, replace=True)
        return execute(owner, *args, **kwargs)

    monkeypatch.setattr(verification, "_execute_job", mismatch)
    job = compare_reference(source, repository=runner_repository)
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 64
    assert "policy changed" in record["log_tail"]
    assert "result" not in record
    assert read_run(source)["verification"]["latest"]["status"] == "failed"


def test_verification_timeout_is_retained_and_produces_no_verdict(tmp_path, runner_repository):
    files(tmp_path, kind="spectrum")
    source = source_run(tmp_path, repository=runner_repository)
    job = compare_reference(source, timeout_seconds=0.0001, repository=runner_repository)
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 130
    assert record["manifest"]["launch"]["timeout_seconds"] == 0.0001
    assert "result" not in record
    assert "timeout" in record["log_tail"]


@pytest.mark.parametrize("target", ["verification", "reference"])
def test_dynamiqs_cancelling_verification_stops_nested_reference(
    tmp_path, differentiable_repository, target
):
    files(tmp_path)
    source = source_run(tmp_path, differentiable_repository)
    # Slow only the reference execution in this isolated checkout, after admission.
    worker = differentiable_repository / "src/entanglix/protocols/_plan_job.py"
    worker.write_text(
        worker.read_text().replace(
            "result = integrator.evolve",
            "import time\n        time.sleep(30)\n        result = integrator.evolve",
        )
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(compare_reference, source, repository=differentiable_repository)
        deadline = time.monotonic() + 20
        reference = None
        while time.monotonic() < deadline:
            paths = list((tmp_path / "runs").glob("*/*/runner/started.toml"))
            references = [
                _read_attempt(path.parent.parent)
                for path in paths
                if _read_attempt(path.parent.parent)["manifest"]["protocol"] == "reference"
            ]
            if references:
                reference = references[0]
                break
            time.sleep(0.02)
        try:
            assert reference is not None, "reference did not start"
            active = read_run(source)["verification"]["latest"]
            abort_run(Path(reference["path"] if target == "reference" else active["path"]))
            job = future.result(timeout=10)
            assert job["exit_code"] == 130
            assert not _group_alive(reference["start"]["process_group"])
            record = read_run(Path(job["path"]))
            assert record["status"] == "aborted"
            assert "Traceback" not in record["log_tail"]
            assert not record.get("result", {}).get("verdict")
        finally:
            if not future.done():
                active = read_run(source)["verification"]["latest"]
                abort_run(Path(active["path"]))
                future.result(timeout=10)


def test_initial_state_embedding_preserves_complex_operator_gauge(tmp_path):
    shutil.copyfile(ROOT / "examples/junction-shunt/system.toml", tmp_path / "system.toml")
    system = load_system(tmp_path / "system.toml")
    phase = np.exp(1j * np.arange(6) * 0.3)
    operators = {
        name: phase[:, None].conj() * value * phase[None, :]
        for name, value in system.operators.items()
    }
    refined = replace(system, operators=operators)
    rng = np.random.default_rng(43)
    initial = rng.normal(size=6) + 1j * rng.normal(size=6)
    initial /= np.linalg.norm(initial)
    mapped = embed_state(system, refined, initial)
    for name in operators:
        np.testing.assert_allclose(
            mapped.conj() @ operators[name] @ mapped,
            initial.conj() @ system.operators[name] @ initial,
            atol=1e-10,
        )


def test_reference_refines_every_subsystem_and_keeps_tensor_order(tmp_path):
    for name in ("system.toml", "qubit.toml"):
        shutil.copyfile(ROOT / "examples/qubit-resonator" / name, tmp_path / name)
    np.savez(tmp_path / "state.npz", initial=np.eye(30)[7])
    plan = {
        "format_version": 1,
        "kind": "lindblad",
        "refinement": {
            "q": {"retained_levels_step": 1, "cutoff_steps": {"theta1": 2}},
            "r": {"retained_levels_step": 1},
        },
        "evolution": {
            "array_file": "state.npz",
            "initial_state": "initial",
            "times_ns": [0, 0.01],
            "observables": {"number": "r.number"},
        },
    }
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    loaded = load_plan(tmp_path / "system.toml", tmp_path / "plan.toml")
    target = tmp_path / "refined"
    target.mkdir()
    reference = prepare_reference(loaded, target, load_policy())
    assert reference.system.dimensions == (7, 6)
    assert abs(reference.request.initial_state.reshape(7, 6)[1, 2]) == pytest.approx(1)
    assert reference.definition.engine == "dynamiqs"
    assert reference.definition.evolution.rtol <= 1e-11


def test_reference_does_not_overwrite_another_leaf_with_a_generated_name(tmp_path):
    leaf = (
        (ROOT / "examples/qubit-resonator/qubit.toml")
        .read_text()
        .replace("retained_levels = 6", "retained_levels = 3")
    )
    for name in ("qubit.toml", "refined-subsystem-0.toml"):
        (tmp_path / name).write_text(leaf)
    system = {
        "format_version": 1,
        "id": "two_qubits",
        "units": {"energy": "GHz", "time": "ns"},
        "model": {
            "kind": "composite",
            "subsystems": [
                {"id": "q", "source": "qubit.toml"},
                {"id": "p", "source": "refined-subsystem-0.toml"},
            ],
        },
    }
    (tmp_path / "system.toml").write_text(tomli_w.dumps(system))
    plan = {
        "format_version": 1,
        "kind": "spectrum",
        "levels": 4,
        "refinement": {
            "q": {"retained_levels_step": 1, "cutoff_steps": {"theta1": 2}},
            "p": {"retained_levels_step": 2, "cutoff_steps": {"theta1": 4}},
        },
    }
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    loaded = load_plan(tmp_path / "system.toml", tmp_path / "plan.toml")
    target = tmp_path / "refined"
    target.mkdir()
    reference = prepare_reference(loaded, target, load_policy())
    assert reference.system.dimensions == (4, 5)
    assert reference.system.provenance["subsystems"]["q"]["cutoffs"]["theta1"] == 32
    assert reference.system.provenance["subsystems"]["p"]["cutoffs"]["theta1"] == 34


@pytest.mark.parametrize("source_engine", ["qutip-cpu", "dynamiqs"])
def test_dynamiqs_composite_complex_state_has_combined_reference(
    tmp_path, differentiable_repository, source_engine
):
    for name in ("system.toml", "qubit.toml"):
        shutil.copyfile(ROOT / "examples/qubit-resonator" / name, tmp_path / name)
    ket = np.zeros(30, dtype=complex)
    ket[0], ket[5] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    np.savez(tmp_path / "state.npz", initial=ket)
    plan = {
        "format_version": 1,
        "kind": "lindblad",
        "refinement": {
            "q": {"retained_levels_step": 1, "cutoff_steps": {"theta1": 2}},
            "r": {"retained_levels_step": 1},
        },
        "evolution": {
            "array_file": "state.npz",
            "initial_state": "initial",
            "times_ns": [0, 0.005, 0.01],
            "observables": {"charge": "q.charge", "number": "r.number"},
        },
    }
    plan["engine"] = source_engine
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    record = verified(source_run(tmp_path, differentiable_repository), differentiable_repository)
    assert record["result"]["verdict"] == "VERIFIED", record["result"]["report"]
    assert record["result"]["report"]["reference"]["dimensions"] == [7, 6]


def test_refinement_above_reference_cap_does_not_launch_a_solve(tmp_path, runner_repository):
    for name in ("system.toml", "qubit.toml"):
        shutil.copyfile(ROOT / "examples/qubit-resonator" / name, tmp_path / name)
    plan = {
        "format_version": 1,
        "kind": "spectrum",
        "levels": 4,
        "refinement": {
            "q": {"retained_levels_step": 2, "cutoff_steps": {"theta1": 2}},
            "r": {"retained_levels_step": 2},
        },
    }
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    record = verified(
        source_run(tmp_path, repository=runner_repository), repository=runner_repository
    )
    assert record["result"]["verdict"] == "UNVERIFIED"
    assert "reference" not in record["result"]["report"]
    assert any(item.get("dimension_limit") == 48 for item in record["result"]["report"]["checks"])
    assert not any(
        read_run(path.parent)["manifest"]["protocol"] == "reference"
        for path in (tmp_path / "runs").glob("*/*/run.toml")
    )


@pytest.mark.parametrize("spoof", ["kind", "metrics"])
def test_changed_source_invalidates_inspected_verdict_and_cannot_spoof_kind(
    tmp_path, spoof, runner_repository
):
    files(tmp_path, kind="spectrum")
    source = source_run(tmp_path, repository=runner_repository)
    verified(source, repository=runner_repository)
    result = read_run(source)["result"]
    if spoof == "kind":
        result["kind"] = "lindblad"
    else:
        result["metrics"] = {"undeclared_fidelity": 1.0}
    publish_toml(source / "result.toml", result, replace=True)
    assert not read_run(source)["verification"]["current"]["verdict_available"]
    with pytest.raises(ValueError, match="inconsistent"):
        compare_reference(source, repository=runner_repository)


def test_output_label_cannot_waive_physical_checks(tmp_path, runner_repository):
    files(tmp_path, kind="spectrum")
    source = source_run(tmp_path, repository=runner_repository)
    result = read_run(source)["result"]
    result["checks"] = []
    result["exemptions"] = ["spectral_reference"]
    output = source / "artifacts/output.npz"
    np.savez(output, energies_ghz=[0, 1])
    result["artifacts"]["output"]["sha256"] = _digest(output.read_bytes())
    publish_toml(source / "result.toml", result, replace=True)
    record = verified(source, repository=runner_repository)
    assert record["result"]["verdict"] == "UNVERIFIED"
    assert record["result"]["report"]["checks"][0]["outcome"] == "fail"


def test_composite_low_spectrum_and_evidence_survive_relocation(
    tmp_path, monkeypatch, runner_repository
):
    from entanglix.corrector import load_verification, verification

    for name in ("system.toml", "qubit.toml"):
        shutil.copyfile(ROOT / "examples/qubit-resonator" / name, tmp_path / name)
    plan = {
        "format_version": 1,
        "kind": "spectrum",
        "levels": 4,
        "refinement": {
            "q": {"retained_levels_step": 1, "cutoff_steps": {"theta1": 2}},
            "r": {"retained_levels_step": 1},
        },
    }
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    source = source_run(tmp_path, repository=runner_repository)
    first = verified(source, repository=runner_repository)
    assert first["result"]["verdict"] == "VERIFIED", first["result"]["report"]
    with np.load(source / "artifacts/output.npz") as arrays:
        assert arrays["energies_ghz"].shape == (4,)
    checks = {item["name"]: item for item in first["result"]["report"]["checks"]}
    assert checks["spectral_consistency"]["outcome"] == "pass"
    assert checks["spectral_reference"]["outcome"] == "inapplicable"
    assert checks["truncation_convergence"]["outcome"] == "pass"
    reference = first["result"]["report"]["reference"]
    assert not {"rtol", "atol", "max_steps", "path"} & reference.keys()
    assert reference["dimensions"] == [7, 6]

    relocated = tmp_path / "moved-runs"
    shutil.move(tmp_path / "runs", relocated)
    source = relocated / source.parent.name / source.name
    first_path = source.parent / Path(first["path"]).name
    load_verification(first_path)
    assert read_run(source)["verification"]["current"]["verdict_available"]
    second = verified(source, repository=runner_repository)
    assert second["result"]["report"]["reference"]["reused"]
    assert second["result"]["report"]["reference"]["run_id"] == reference["run_id"]

    # A different system directory must not be inspected. Multiple verification
    # reports sharing evidence must not each rehash the retained output archives.
    other = relocated / "other-system" / "not-an-attempt"
    other.mkdir(parents=True)
    (other / "run.toml").write_text('protocol = "plan"\n')
    inspected, hashed = [], []
    read_attempt, script_result = verification._read_attempt, verification._script_result

    def inspect(path):
        inspected.append(path)
        return read_attempt(path)

    def hash_result(path):
        hashed.append(path)
        return script_result(path)

    monkeypatch.setattr(verification, "_read_attempt", inspect)
    monkeypatch.setattr(verification, "_script_result", hash_result)
    history = read_run(source)["verification"]
    assert len(history["history"]) == 2
    assert all(entry["verdict_available"] for entry in history["history"])
    assert all(path.parent == source.parent for path in inspected)
    assert hashed.count(source) == 1
    assert hashed.count(source.parent / reference["run_id"]) == 1
