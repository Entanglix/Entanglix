"""Built-in CPU plans share runner records, failures, provenance, and cancellation."""

import hashlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import tomli_w

from entanglix.corrector import create_verification_run, load_policy, load_verification
from entanglix.protocols.plans import load_plan
from entanglix.runner import (
    CodeSnapshot,
    _execute_job,
    capture_environment,
    publish_toml,
    read_run,
    run_plan,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan_files(tmp_path):
    for name in ("system.toml", "operators.npz"):
        shutil.copyfile(ROOT / "examples/two-level" / name, tmp_path / name)
    np.savez(tmp_path / "state.npz", initial_state=np.array([1, 1j]) / np.sqrt(2))
    definition = {
        "format_version": 1,
        "kind": "lindblad",
        "engine": "qutip-cpu",
        "evolution": {
            "array_file": "state.npz",
            "initial_state": "initial_state",
            "times_ns": [0, 10, 20],
            "observables": {"coherence": "lower"},
        },
    }
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(definition))
    return tmp_path, definition


def test_plan_job_records_and_replays_frozen_complex_inputs(plan_files, runner_repository):
    root, _ = plan_files
    first = run_plan(
        root / "system.toml",
        root / "plan.toml",
        runs_root=root / "runs",
        repository=runner_repository,
    )
    assert first["exit_code"] == 0
    run = Path(first["path"])
    record = read_run(run)
    assert record["manifest"]["protocol"] == "plan"
    assert record["manifest"]["engine"] == "qutip-cpu"
    assert record["result"]["method"] == "vern9"
    assert record["start"]["process_group"] > 0
    assert record["finished"]["usage"]["wall_seconds"] > 0
    assert "verdict" not in record["result"]
    assert record["manifest"]["launch"]["policy_version"] == load_policy().version
    (root / "plan.toml").write_text("invalid original")
    replay = run_plan(
        run / "spec/system.toml",
        run / "spec/plan.toml",
        runs_root=root / "runs",
        parent_run_ids=(run.name,),
        repository=runner_repository,
    )
    repeated = read_run(Path(replay["path"]))
    assert repeated["result"] == record["result"]
    assert repeated["manifest"]["env_hash"] == record["manifest"]["env_hash"]
    assert repeated["manifest"]["parent_run_ids"] == [run.name]
    with np.load(run / "artifacts/output.npz") as output:
        assert abs(output["states"][0, 0, 1].imag) > 0.4


def test_carrier_frame_plan_freezes_settings_and_reference_uses_lab_frame(
    plan_files, runner_repository
):
    from entanglix.corrector.reference import prepare_reference
    from entanglix.engines import QuTiPIntegrator
    from entanglix.models import load_system

    root, definition = plan_files
    system_id = load_system(root / "system.toml").id
    definition["evolution"].update(
        times_ns=[0.23, 0.73, 1.23],
        drive_amplitudes_ghz={"x_drive": [0.02, -0.01]},
        carriers={"x_drive": {"frequency_ghz": 0.2, "phase_rad": 0.37}},
        frame={
            "frequencies_ghz": {system_id: 0.2},
            "rwa": True,
            "excitation_numbers": {system_id: [0, 1]},
        },
    )
    (root / "plan.toml").write_text(tomli_w.dumps(definition))
    loaded = load_plan(root / "system.toml", root / "plan.toml")
    expected = QuTiPIntegrator().evolve(loaded.system, loaded.request)
    run = Path(
        run_plan(
            root / "system.toml",
            root / "plan.toml",
            runs_root=root / "runs",
            repository=runner_repository,
        )["path"]
    )
    with np.load(run / "artifacts/output.npz") as output:
        np.testing.assert_allclose(output["states"], expected.states, rtol=0, atol=1e-12)
    assert read_run(run)["result"]["diagnostics"]["frame"]["rwa"]
    target = root / "reference"
    target.mkdir()
    reference = prepare_reference(loaded, target, load_policy())
    assert reference.request.frame is None
    assert reference.request.carriers == loaded.request.carriers
    np.testing.assert_array_equal(reference.request.times_ns, loaded.request.times_ns)


@pytest.mark.parametrize("engine", ["qutip-cpu", "dynamiqs"])
@pytest.mark.parametrize("controls", [{}, {"x_drive": 0.02}, {"x_drive": [0.02, 0.01]}])
def test_reference_step_allowance_matches_solver_scope(plan_files, engine, controls):
    from entanglix.corrector.reference import prepare_reference
    from entanglix.models import load_system

    root, definition = plan_files
    system_id = load_system(root / "system.toml").id
    definition["engine"] = engine
    definition["evolution"]["drive_amplitudes_ghz"] = controls
    definition["evolution"].update(
        carriers={"x_drive": {"frequency_ghz": 0.2}},
        frame={
            "frequencies_ghz": {system_id: 0.2},
            "rwa": True,
            "excitation_numbers": {system_id: [0, 1]},
        },
    )
    (root / "plan.toml").write_text(tomli_w.dumps(definition))
    source = load_plan(root / "system.toml", root / "plan.toml")
    target = root / "reference"
    target.mkdir()
    reference = prepare_reference(source, target, load_policy())
    multiplier = (
        len(source.request.times_ns) - 1
        if engine == "qutip-cpu" and not any(np.ndim(value) for value in controls.values())
        else 1
    )
    assert reference.request.max_steps == source.request.max_steps * multiplier
    assert reference.request.frame is None


def test_verification_admission_binds_source_and_leaves_source_unchanged(
    plan_files, runner_repository
):
    root, _ = plan_files
    source = Path(
        run_plan(
            root / "system.toml",
            root / "plan.toml",
            runs_root=root / "runs",
            repository=runner_repository,
        )["path"]
    )
    before = {
        path: path.read_bytes()
        for path in (source / "run.toml", source / "result.toml", source / "FINISHED")
    }
    with create_verification_run(
        source,
        environment=capture_environment(ROOT / "uv.lock"),
        code=CodeSnapshot("test", b"", {}),
        runs_root=root / "verification-runs",
    ) as owner:
        verification = read_run(owner.path)
        binding = verification["manifest"]["launch"]
        assert verification["manifest"]["parent_run_ids"] == [source.name]
        assert (
            binding["source_result_sha256"]
            == hashlib.sha256(before[source / "result.toml"]).hexdigest()
        )
        assert not Path(binding["source_path"]).is_absolute()
        assert (owner.path.parent / binding["source_path"]).resolve() == source
        assert (
            binding["source_artifact_sha256"]
            == hashlib.sha256((source / "artifacts/output.npz").read_bytes()).hexdigest()
        )
        assert "source-output.npz" not in verification["manifest"]["spec_sha256"]
        assert "source-result.toml" not in verification["manifest"]["spec_sha256"]
        assert verification["status"] == "active"
        assert "result" not in verification
        loaded, retained_binding = load_verification(owner.path)
        assert loaded.digest == retained_binding["plan_sha256"]
        manifest = verification["manifest"]
        manifest["launch"]["policy_version"] = "changed-policy"
        publish_toml(owner.path / "run.toml", manifest, replace=True)
        with pytest.raises(ValueError, match="policy changed"):
            load_verification(owner.path)
    assert {path: path.read_bytes() for path in before} == before
    # Admission alone is not completed verification and cannot fabricate a verdict.
    assert read_run(owner.path)["status"] == "interrupted"
    result = read_run(source)["result"]
    result["kind"] = "spectrum"
    publish_toml(source / "result.toml", result, replace=True)
    with pytest.raises(ValueError, match="inconsistent"):
        create_verification_run(
            source,
            environment=capture_environment(ROOT / "uv.lock"),
            code=CodeSnapshot("test", b"", {}),
            runs_root=root / "runs",
        )


@pytest.mark.parametrize("change", ["kind", "array_path", "unknown_operator", "inconsistent"])
def test_invalid_plan_is_rejected_before_admission(plan_files, change, runner_repository):
    root, definition = plan_files
    if change == "kind":
        definition["kind"] = "authored-script"
    elif change == "array_path":
        definition["evolution"]["array_file"] = "../state.npz"
    elif change == "unknown_operator":
        definition["evolution"]["observables"]["coherence"] = "unavailable"
    else:
        definition["kind"] = "spectrum"
    (root / "plan.toml").write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError):
        run_plan(
            root / "system.toml",
            root / "plan.toml",
            runs_root=root / "runs",
            repository=runner_repository,
        )
    assert not (root / "runs").exists()


def test_policy_mismatch_fails_job_without_a_verdict(plan_files, monkeypatch, runner_repository):
    import entanglix.runner as runner

    root, _ = plan_files
    execute = runner._execute_job

    def changed_policy(owner, command, environment, **kwargs):
        # Supply a launch/load mismatch in an admitted attempt, without editing policy files.
        manifest = read_run(owner.path)["manifest"]
        manifest["launch"]["policy_version"] = "different-policy"
        publish_toml(owner.path / "run.toml", manifest, replace=True)
        return execute(owner, command, environment, **kwargs)

    monkeypatch.setattr(runner, "_execute_job", changed_policy)
    job = run_plan(
        root / "system.toml",
        root / "plan.toml",
        runs_root=root / "runs",
        repository=runner_repository,
    )
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 64
    assert record["status"] == "failed"
    assert "policy changed" in record["log_tail"]
    assert "result" not in record


def test_policy_mismatch_fails_verification_execution_without_a_verdict(
    plan_files, runner_repository
):
    root, _ = plan_files
    source = Path(
        run_plan(
            root / "system.toml",
            root / "plan.toml",
            runs_root=root / "runs",
            repository=runner_repository,
        )["path"]
    )
    with create_verification_run(
        source,
        environment=capture_environment(ROOT / "uv.lock"),
        code=CodeSnapshot("test", b"", {}),
        runs_root=root / "runs",
    ) as owner:
        manifest = read_run(owner.path)["manifest"]
        manifest["launch"]["policy_version"] = "changed-policy"
        publish_toml(owner.path / "run.toml", manifest, replace=True)
        worker = """import os, sys
from pathlib import Path
from entanglix.corrector import load_verification
os.fstat(int(sys.argv[-1]))
if sys.stdin.buffer.read(1) != b'1':
    raise SystemExit(64)
load_verification(Path(os.environ['ENTANGLIX_RUN_DIR']))
"""
        job = _execute_job(
            owner,
            [sys.executable, "-c", worker],
            os.environ | {"ENTANGLIX_RUN_DIR": str(owner.path)},
            timeout_seconds=None,
        )
    record = read_run(owner.path)
    assert job["status"] == "failed"
    assert "policy changed" in record["log_tail"]
    assert "result" not in record


@pytest.mark.parametrize("change", ["script", "incomplete", "artifact"])
def test_verification_rejects_untrusted_or_incomplete_source(plan_files, change, runner_repository):
    root, _ = plan_files
    source = Path(
        run_plan(
            root / "system.toml",
            root / "plan.toml",
            runs_root=root / "runs",
            repository=runner_repository,
        )["path"]
    )
    if change == "script":
        manifest = read_run(source)["manifest"]
        manifest["protocol"] = "script"
        publish_toml(source / "run.toml", manifest, replace=True)
    elif change == "incomplete":
        (source / "FINISHED").unlink()
    else:
        (source / "artifacts/output.npz").write_bytes(b"changed output")
    with pytest.raises(ValueError, match="completed built-in plan|digest mismatch"):
        create_verification_run(
            source,
            environment=capture_environment(ROOT / "uv.lock"),
            code=CodeSnapshot("test", b"", {}),
            runs_root=root / "runs",
        )
    assert len(list((root / "runs").glob("*/*/run.toml"))) == 1


@pytest.mark.parametrize("change", ["result", "artifact", "missing"])
def test_verification_rechecks_retained_source_after_admission(
    plan_files, change, runner_repository
):
    root, _ = plan_files
    source = Path(
        run_plan(
            root / "system.toml",
            root / "plan.toml",
            runs_root=root / "runs",
            repository=runner_repository,
        )["path"]
    )
    with create_verification_run(
        source,
        environment=capture_environment(ROOT / "uv.lock"),
        code=CodeSnapshot("test", b"", {}),
        runs_root=root / "verification-runs",
    ) as owner:
        load_verification(owner.path)
        if change == "result":
            with (source / "result.toml").open("a") as stream:
                stream.write("\n# changed source result\n")
        elif change == "artifact":
            (source / "artifacts/output.npz").write_bytes(b"changed output")
        else:
            (source / "artifacts/output.npz").unlink()
        with pytest.raises((ValueError, OSError), match="digest mismatch|No such file"):
            load_verification(owner.path)


def test_plan_identity_ignores_toml_formatting_but_tracks_inputs_and_settings(plan_files):
    root, definition = plan_files
    path = root / "plan.toml"
    first = load_plan(root / "system.toml", path)
    path.write_text("# reformatting\n\n" + path.read_text())
    reformatted = load_plan(root / "system.toml", path)
    assert first.digest == reformatted.digest
    assert first.input_digests["plan.toml"] != reformatted.input_digests["plan.toml"]
    definition["evolution"]["rtol"] = 1e-11
    path.write_text(tomli_w.dumps(definition))
    assert load_plan(root / "system.toml", path).digest != first.digest
    path.write_text(
        tomli_w.dumps(definition | {"evolution": definition["evolution"] | {"rtol": 1e-10}})
    )
    np.savez(root / "state.npz", initial_state=np.array([0, 1]))
    assert load_plan(root / "system.toml", path).digest != first.digest


def test_reference_plan_limit_is_separate_and_refinement_cannot_be_waived(
    tmp_path, monkeypatch, runner_repository
):
    system = {
        "format_version": 1,
        "id": "oscillator",
        "units": {"energy": "GHz", "time": "ns"},
        "model": {
            "kind": "composite",
            "subsystems": [
                {
                    "id": "r",
                    "kind": "oscillator",
                    "frequency": {"value": 0.1, "unit": "GHz"},
                    "retained_levels": 6,
                },
                {
                    "id": "q",
                    "kind": "oscillator",
                    "frequency": {"value": 0.15, "unit": "GHz"},
                    "retained_levels": 8,
                },
            ],
        },
    }
    (tmp_path / "system.toml").write_text(tomli_w.dumps(system))
    definition = {
        "format_version": 1,
        "kind": "spectrum",
        "levels": 4,
        "purpose": "reference",
        "refinement": {"r": {"retained_levels_step": 1}, "q": {"retained_levels_step": 1}},
    }
    plan = tmp_path / "plan.toml"
    plan.write_text(tomli_w.dumps(definition))
    job = run_plan(
        tmp_path / "system.toml", plan, runs_root=tmp_path / "runs", repository=runner_repository
    )
    assert job["exit_code"] == 0
    assert read_run(Path(job["path"]))["manifest"]["protocol"] == "reference"
    definition["purpose"] = "computation"
    plan.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match="limit 30"):
        load_plan(tmp_path / "system.toml", plan)
    from entanglix.protocols import plans

    table = plans.selection_table()
    table["engines"]["qutip-cpu"]["max_dimension"] = 48
    monkeypatch.setattr(plans, "selection_table", lambda: table)
    assert load_plan(tmp_path / "system.toml", plan).system.dimensions == (6, 8)
    definition["purpose"] = "reference"
    definition["refinement"] = {}
    plan.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match="truncated subsystems"):
        load_plan(tmp_path / "system.toml", plan)


def test_cutoff_refinement_must_meet_the_corrector_minimum(tmp_path):
    for name in ("system.toml", "qubit.toml"):
        shutil.copyfile(ROOT / "examples/qubit-resonator" / name, tmp_path / name)
    definition = {
        "format_version": 1,
        "kind": "spectrum",
        "levels": 4,
        "refinement": {
            "q": {"retained_levels_step": 1, "cutoff_steps": {"theta1": 1}},
            "r": {"retained_levels_step": 1},
        },
    }
    path = tmp_path / "plan.toml"
    path.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match="cutoff steps.*policy minimum"):
        load_plan(tmp_path / "system.toml", path)
    definition["refinement"]["q"]["cutoff_steps"]["theta1"] = load_policy().thresholds["reference"][
        "min_cutoff_step"
    ]
    path.write_text(tomli_w.dumps(definition))
    assert load_plan(tmp_path / "system.toml", path).system.dimensions == (6, 5)


def test_plan_timeout_uses_runner_lifecycle(plan_files, runner_repository):
    root, _ = plan_files
    job = run_plan(
        root / "system.toml",
        root / "plan.toml",
        runs_root=root / "runs",
        timeout_seconds=0.0001,
        repository=runner_repository,
    )
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 130
    assert record["status"] == "aborted"
    assert "timeout" in record["log_tail"]


def test_dynamiqs_plan_runs_in_configured_exact_cpu_environment(
    plan_files, differentiable_repository
):
    root, definition = plan_files
    repository = differentiable_repository
    definition["engine"] = "dynamiqs"
    (root / "plan.toml").write_text(tomli_w.dumps(definition))
    job = run_plan(
        root / "system.toml", root / "plan.toml", repository=repository, runs_root=root / "runs"
    )
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 0, record["log_tail"]
    assert record["result"]["engine"] == "dynamiqs"
    assert record["result"]["device"] == "cpu"
    assert record["manifest"]["environment"]["extras"] == ["differentiable"]
    expected = load_plan(root / "system.toml", root / "plan.toml")
    from entanglix.engines import QuTiPIntegrator

    comparison = QuTiPIntegrator().evolve(expected.system, expected.request)
    with np.load(Path(job["path"]) / "artifacts/output.npz") as output:
        np.testing.assert_allclose(
            output["states"], np.asarray(comparison.states), atol=1e-7, rtol=0
        )


@pytest.mark.parametrize("levels", [None, 0, True, 3])
def test_spectrum_requires_valid_reported_levels(plan_files, levels):
    root, _ = plan_files
    plan = {"format_version": 1, "kind": "spectrum"}
    if levels is not None:
        plan["levels"] = levels
    (root / "plan.toml").write_text(tomli_w.dumps(plan))
    with pytest.raises(ValueError, match="levels"):
        load_plan(root / "system.toml", root / "plan.toml")


def test_dynamics_rejects_spectral_levels(plan_files):
    root, plan = plan_files
    plan["levels"] = 2
    (root / "plan.toml").write_text(tomli_w.dumps(plan))
    with pytest.raises(ValueError, match="lindblad must not contain levels"):
        load_plan(root / "system.toml", root / "plan.toml")
