"""On-request checks, immutable binding, scoped outcomes, and failure history."""

import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tomli_w

from entanglix.corrector import load_policy
from entanglix.corrector.checks import output_checks, series_checks
from entanglix.corrector.reference import embed_state, prepare_refinement
from entanglix.corrector.verification import check_run
from entanglix.models import load_system
from entanglix.protocols.plans import load_plan
from entanglix.runner import _digest, _load_record, publish_toml, read_run, run_plan

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
    reference = prepare_refinement(loaded, target, load_policy())
    assert reference.system.dimensions == (7, 6)
    assert abs(reference.request.initial_state.reshape(7, 6)[1, 2]) == pytest.approx(1)
    assert reference.definition.engine == "qutip-cpu"
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
    reference = prepare_refinement(loaded, target, load_policy())
    assert reference.system.dimensions == (4, 5)
    assert reference.system.provenance["subsystems"]["q"]["cutoffs"]["theta1"] == 32
    assert reference.system.provenance["subsystems"]["p"]["cutoffs"]["theta1"] == 34


def checked(source, repository, names=("integrity", "segments", "diagnostics"), **kwargs):
    job = check_run(source, names, repository=repository, **kwargs)
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 0, record["log_tail"]
    assert "verdict" not in record["result"]
    return record


def test_no_checks_without_request_and_named_cli_report(tmp_path, runner_repository):
    files(tmp_path)
    source = source_run(tmp_path, runner_repository)
    assert read_run(source)["verification"] == []
    count = len(list((tmp_path / "runs").glob("*/*/run.toml")))
    listing = check_run(source)
    assert {item["name"] for item in listing["checks"]} == {"integrity", "segments", "diagnostics"}
    assert "whole-window" in listing["limit"]
    command = [sys.executable, "-m", "entanglix", "check", str(source)]
    env = os.environ | {"PYTHONPATH": str(runner_repository / "src")}
    listed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert listed.returncode == 0, listed.stderr
    assert len(list((tmp_path / "runs").glob("*/*/run.toml"))) == count
    ran = subprocess.run(
        command + ["integrity", "spectral_refinement"], env=env, capture_output=True, text=True
    )
    assert ran.returncode == 0, ran.stderr
    history = read_run(source)["verification"]
    assert len(history) == 1
    outcomes = history[0]["checks"]
    assert outcomes[0]["name"] == "integrity" and outcomes[0]["outcome"] == "pass"
    assert outcomes[1]["outcome"] == "inapplicable"
    assert len(list((tmp_path / "runs").glob("*/*/run.toml"))) == count + 1


def test_unknown_checks_rejected_and_authored_runs_have_no_checks(tmp_path, runner_repository):
    files(tmp_path)
    source = source_run(tmp_path, runner_repository)
    with pytest.raises(ValueError, match="known check"):
        check_run(source, ("fake",), repository=runner_repository)
    manifest = read_run(source)["manifest"]
    manifest["protocol"] = "script"
    publish_toml(source / "run.toml", manifest, replace=True)
    assert check_run(source)["checks"] == []
    with pytest.raises(ValueError, match="completed built-in"):
        check_run(source, ("integrity",), repository=runner_repository)


@pytest.mark.parametrize("protocol", ["plan", "simulate", "reference"])
def test_inspection_preserves_records_and_history_for_obsolete_plan(
    tmp_path, runner_repository, protocol
):
    files(tmp_path)
    source = source_run(tmp_path, runner_repository)
    checked(source, runner_repository, ("integrity",))
    original = read_run(source)
    manifest = original["manifest"]
    manifest["protocol"] = protocol
    publish_toml(source / "run.toml", manifest, replace=True)
    plan = source / "spec/plan.toml"
    definition = _load_record(plan)
    definition["purpose"] = "reference"  # An older internal plan field.
    plan.chmod(0o644)
    plan.write_text(tomli_w.dumps(definition))
    manifest["spec_sha256"]["plan.toml"] = _digest(plan.read_bytes())
    publish_toml(source / "run.toml", manifest, replace=True)
    record = read_run(source)
    assert record["status"] == original["status"] == "completed"
    assert record["result"] == original["result"]
    assert record["verification"] == original["verification"]
    assert record["check_listing"]["checks"] == []
    if protocol != "reference":
        assert "purpose" in record["check_listing"]["reason"]
        with pytest.raises(ValueError, match="purpose"):
            check_run(source, ("integrity",), repository=runner_repository)
    else:
        assert "no trusted built-in" in record["check_listing"]["reason"]


def test_check_gate_preserves_signal_diagnostic(tmp_path, monkeypatch, runner_repository):
    from entanglix.corrector import verification

    monkeypatch.setattr(
        verification.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=137, stderr="", stdout=""),
    )
    with pytest.raises(ValueError, match="SIGKILL encoding.*cause unknown") as error:
        check_run(tmp_path, ("integrity",), repository=runner_repository)
    assert "exhausted memory is a possible cause" in str(error.value)


@pytest.mark.parametrize("change", ["initial", "physicality", "observable", "metric"])
def test_integrity_detects_stored_faults(tmp_path, runner_repository, change):
    files(tmp_path)
    source = source_run(tmp_path, runner_repository)
    result = read_run(source)["result"]
    archive = source / "artifacts/output.npz"
    with np.load(archive) as data:
        arrays = {name: np.array(data[name]) for name in data.files}
    if change == "initial":
        arrays["states"][0] = np.diag([1, 0])
        arrays["observable.coherence"][0] = 0
    elif change == "physicality":
        arrays["states"][1] = np.diag([-0.1, 1.1])
    elif change == "observable":
        arrays["observable.coherence"][1] += 0.1
    else:
        result["metrics"] = {"invented": 1.0}
    np.savez(archive, **arrays)
    result["artifacts"]["output"]["sha256"] = _digest(archive.read_bytes())
    result["exemptions"] = ["integrity"]  # Candidate labels never change trusted relevance.
    publish_toml(source / "result.toml", result, replace=True)
    record = checked(source, runner_repository, ("integrity",))
    outcome = record["result"]["report"]["checks"][0]
    assert outcome["outcome"] == "fail"
    if change == "initial":
        assert (
            next(item for item in outcome["checks"] if item["name"] == "initial_state")["outcome"]
            == "fail"
        )
        assert (
            next(item for item in outcome["checks"] if item["name"] == "trace")["outcome"] == "pass"
        )


def test_complex_nonzero_start_frame_initial_state_agrees(tmp_path, runner_repository):
    definition = files(tmp_path)
    definition["evolution"].update(
        times_ns=[0.23, 0.73, 1.23],
        drive_amplitudes_ghz={"x_drive": 0.02},
        carriers={"x_drive": {"frequency_ghz": 0.2}},
        frame={
            "frequencies_ghz": {"two_level_reference": 0.2},
            "rwa": True,
            "excitation_numbers": {"two_level_reference": [0, 1]},
        },
    )
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(definition))
    record = checked(source_run(tmp_path, runner_repository), runner_repository, ("integrity",))
    assert record["result"]["report"]["checks"][0]["outcome"] == "pass"


def test_spectral_refinement_detects_cutoff_and_reports_only_requested_check(
    tmp_path, runner_repository
):
    files(tmp_path, kind="spectrum")
    shutil.copyfile(ROOT / "examples/fluxonium/system.toml", tmp_path / "system.toml")
    (tmp_path / "system.toml").write_text(
        (tmp_path / "system.toml").read_text().replace("theta1 = 60", "theta1 = 20")
    )
    plan = {
        "format_version": 1,
        "kind": "spectrum",
        "levels": 4,
        "refinement": {
            "fluxonium_circuit": {"retained_levels_step": 1, "cutoff_steps": {"theta1": 20}}
        },
    }
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    record = checked(
        source_run(tmp_path, runner_repository), runner_repository, ("spectral_refinement",)
    )
    report = record["result"]["report"]
    assert [item["name"] for item in report["checks"]] == ["spectral_refinement"]
    assert report["checks"][0]["outcome"] == "fail"


def test_replacement_refinement_runs_without_repeating_source_and_binds_request(
    tmp_path, runner_repository
):
    from test_simulate import protocol_files

    from entanglix.corrector import create_verification_run, load_verification
    from entanglix.protocols.simulate import simulate
    from entanglix.runner import CodeSnapshot, capture_environment

    path, spec = protocol_files(tmp_path, kind="spectrum")
    declaration = spec.pop("refinement")
    path.write_text(tomli_w.dumps(spec))
    source = Path(simulate(path, runs_root=tmp_path / "runs", repository=runner_repository)["path"])
    before = {file: file.read_bytes() for file in source.rglob("*") if file.is_file()}
    original = checked(source, runner_repository, ("spectral_refinement",))
    assert original["result"]["report"]["checks"][0]["outcome"] == "inapplicable"
    override = tmp_path / "refinement.toml"
    override.write_text(tomli_w.dumps({"refinement": declaration}))
    updated = checked(source, runner_repository, ("spectral_refinement",), refinement=override)
    assert updated["result"]["report"]["checks"][0]["outcome"] == "pass"
    assert updated["result"]["report"]["refinement_origin"] == "request"
    assert {file: file.read_bytes() for file in before} == before
    with create_verification_run(
        source,
        checks=("spectral_refinement",),
        refinement=override,
        environment=capture_environment(ROOT / "uv.lock"),
        code=CodeSnapshot("test", b"", {}),
        runs_root=tmp_path / "runs",
    ) as owner:
        _, binding, request = load_verification(owner.path)
        frozen = owner.path / "spec" / binding["request_file"]
        frozen.chmod(0o644)
        frozen.write_text(frozen.read_text() + "\n# tampered\n")
        with pytest.raises(ValueError, match="input digest mismatch"):
            load_verification(owner.path)
        assert request["checks"] == ["spectral_refinement"]


def test_failed_newer_attempt_remains_visible_and_relocation_preserves_binding(
    tmp_path, runner_repository
):
    from entanglix.corrector import load_verification

    files(tmp_path)
    source = source_run(tmp_path, runner_repository)
    first = checked(source, runner_repository, ("integrity",))
    job = check_run(source, ("segments",), repository=runner_repository, timeout_seconds=0.0001)
    assert job["exit_code"] == 130
    failed = read_run(Path(job["path"]))
    assert "result" not in failed
    history = read_run(source)["verification"]
    assert len(history) == 2 and history[-1]["status"] == "aborted"
    assert "checks" not in history[-1]
    assert history[0]["checks"][0]["outcome"] == "pass"
    destination = tmp_path / "moved-runs"
    shutil.move(tmp_path / "runs", destination)
    moved = destination / source.parent.name / source.name
    load_verification(moved.parent / Path(first["path"]).name)
    assert all(entry["source_matches"] for entry in read_run(moved)["verification"])


def test_policy_mismatch_fails_verifier_without_outcomes(tmp_path, monkeypatch, runner_repository):
    from entanglix.corrector import verification

    files(tmp_path)
    source = source_run(tmp_path, runner_repository)
    execute = verification._execute_job

    def mismatch(owner, *args, **kwargs):
        manifest = read_run(owner.path)["manifest"]
        manifest["launch"]["policy_version"] = "changed-policy"
        publish_toml(owner.path / "run.toml", manifest, replace=True)
        return execute(owner, *args, **kwargs)

    monkeypatch.setattr(verification, "_execute_job", mismatch)
    job = check_run(source, ("integrity",), repository=runner_repository)
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 64 and "policy changed" in record["log_tail"]
    assert "result" not in record


def test_declared_expectation_is_an_additive_check():
    report = series_checks("lindblad", [0, 2, 1], monotonic_direction="increasing")
    assert len(report) == 1 and report[0]["outcome"] == "fail"
    assert report[0]["failure_meaning"] == "declared expectation violated"


@pytest.mark.parametrize("step", [0, -1, True, 1.5, "2"])
def test_malformed_cutoff_steps_are_schema_errors(tmp_path, step):
    plan = files(tmp_path)
    plan["refinement"] = {"unused": {"retained_levels_step": 1, "cutoff_steps": {"theta1": step}}}
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    with pytest.raises(ValueError, match="cutoff_steps"):
        load_plan(tmp_path / "system.toml", tmp_path / "plan.toml")


def test_below_policy_refinement_is_reported_and_does_not_restrict_plan(tmp_path):
    from entanglix.corrector.dynamics import check_listing

    for name in ("system.toml", "qubit.toml"):
        shutil.copyfile(ROOT / "examples/qubit-resonator" / name, tmp_path / name)
    plan = {
        "format_version": 1,
        "kind": "spectrum",
        "levels": 4,
        "refinement": {
            "q": {"retained_levels_step": 1, "cutoff_steps": {"theta1": 1}},
            "r": {"retained_levels_step": 1},
        },
    }
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    loaded = load_plan(tmp_path / "system.toml", tmp_path / "plan.toml")
    entry = check_listing(loaded)["checks"][0]
    assert not entry["available"] and "policy minimum" in entry["reason"]


def test_drive_change_windows_spread_and_report_uncovered_events(tmp_path):
    from entanglix.corrector.dynamics import check_listing

    plan = files(tmp_path)
    plan["evolution"].update(
        times_ns=list(np.linspace(0, 10, 21)), drive_amplitudes_ghz={"x_drive": [0.02, 0.01] * 10}
    )
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    listing = check_listing(load_plan(tmp_path / "system.toml", tmp_path / "plan.toml"))
    segments = next(item for item in listing["checks"] if item["name"] == "segments")
    windows = segments["windows"]
    assert len(windows) == 8
    assert windows[0]["start_ns"] == 0 and windows[2]["end_ns"] == 10
    assert windows[3]["start_ns"] < 2 and windows[-1]["end_ns"] > 8
    assert segments["coverage"]["uncovered_drive_changes"]["count"] > 0
    assert segments["refined_dimension"] == 2 and segments["engine"] == "qutip-cpu"
