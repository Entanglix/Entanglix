"""Requested independent control replay, target, constraints, and source binding."""

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from test_control import control_files as control_files

from entanglix.corrector import load_policy
from entanglix.corrector.control import constraints, control_checks, control_listing, replay_values
from entanglix.corrector.verification import check_run
from entanglix.protocols.control import compile_control, load_control
from entanglix.runner import (
    CodeSnapshot,
    _cpu_job_context,
    _digest,
    capture_environment,
    create_run,
    publish_toml,
    read_run,
    record_finish,
    record_start,
)
from entanglix.solvers import Pulse

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def candidate(control_files):
    path, definition = control_files
    definition["pulse"]["T"] = 10.0
    definition["problem"].update(final_fidelity=0.999, du_bound=0.04, ddu_bound=0.016)
    path.write_text(tomli_w.dumps(definition))
    loaded = load_control(path)
    pulse = Pulse(np.linspace(0, 10, 5), np.array([[0, 0.1, 0.1, 0, 0], [0, 0, 0, 0, 0]]))
    return loaded, pulse


def test_control_replay_target_and_active_constraints_pass(candidate):
    loaded, pulse = candidate
    items = control_checks(
        loaded,
        pulse,
        {"metrics": {"solver_reported_fidelity": 1.0}},
        ("replay", "target", "constraints"),
        load_policy(),
    )
    assert [item["outcome"] for item in items] == ["pass", "pass", "pass"]
    assert items[0]["fidelity"] == pytest.approx(1, abs=1e-9)
    assert items[0]["leakage"] == pytest.approx(0, abs=1e-9)
    assert items[0]["engine"] == "qutip-cpu"
    assert items[-1]["uncovered_constraints"] == []


def test_accurately_reported_bad_gate_fails_only_target(candidate):
    loaded, pulse = candidate
    pulse = replace(pulse, amplitudes_ghz=np.zeros_like(pulse.amplitudes_ghz))
    items = control_checks(
        loaded,
        pulse,
        {"metrics": {"solver_reported_fidelity": 1 / 3}},
        ("replay", "target", "constraints"),
        load_policy(),
    )
    assert [item["outcome"] for item in items] == ["pass", "fail", "pass"]
    assert items[1]["failure_meaning"] == "unmet declared target"


def test_inaccurate_solver_report_fails_replay(candidate):
    loaded, pulse = candidate
    (item,) = control_checks(
        loaded, pulse, {"metrics": {"solver_reported_fidelity": 0.5}}, ("replay",), load_policy()
    )
    assert item["outcome"] == "fail"
    assert item["residual"] == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize(
    "change,failed",
    [
        ("amplitude", "amplitude"),
        ("du", "du"),
        ("ddu", "ddu"),
        ("duration", "duration"),
        ("equal_steps", "equal_steps"),
        ("shape", "pulse_structure"),
        ("finite", "pulse_structure"),
    ],
)
def test_control_constraints_detect_violations(candidate, change, failed):
    loaded, pulse = candidate
    times, amplitudes = pulse.times_ns.copy(), pulse.amplitudes_ghz.copy()
    if change == "amplitude":
        amplitudes[0, 2] = 0.2
    elif change in {"du", "ddu"}:
        problem = loaded.definition.problem.model_copy(update={change + "_bound": 0.001})
        loaded = replace(
            loaded, definition=loaded.definition.model_copy(update={"problem": problem})
        )
    elif change == "duration":
        times *= 1.1
    elif change == "equal_steps":
        times[2] += 0.1
    elif change == "shape":
        amplitudes = amplitudes[:, :-1]
    else:
        amplitudes[0, 2] = np.nan
    item = constraints(loaded, Pulse(times, amplitudes), load_policy())
    assert item["outcome"] == "fail"
    assert failed in [check["name"] for check in item["checks"] if check["outcome"] == "fail"]


@pytest.mark.parametrize("relative_change", [-1.1e-7, -1e-7, -0.9e-7, 0.9e-7, 1e-7, 1.1e-7])
def test_control_timing_uses_frozen_step_bracket(candidate, relative_change):
    loaded, pulse = candidate
    # At 20 ns, valid bracket endpoints exceed the former 1e-6 ns duration slack.
    path = loaded.inputs["control.toml"]
    definition = loaded.definition.model_dump(exclude_none=True)
    definition["pulse"]["T"] = 20.0
    path.write_text(tomli_w.dumps(definition))
    loaded = load_control(path)
    pulse = replace(pulse, times_ns=pulse.times_ns * 2 * (1 + relative_change))
    report = constraints(loaded, pulse, load_policy())
    expected = "pass" if abs(relative_change) <= 1e-7 else "fail"
    assert report["outcome"] == expected
    timing = [item for item in report["checks"] if item["name"] in {"duration", "equal_steps"}]
    assert [item["outcome"] for item in timing] == [expected, expected]
    lower, upper = loaded.problem_spec["problem"]["free_dt"]
    assert timing[0]["bounds_ns"] == [4 * lower, 4 * upper]
    assert timing[1]["bounds_ns"] == [lower, upper]


def test_control_step_outside_bracket_fails_at_correct_total_duration(candidate):
    loaded, pulse = candidate
    times = pulse.times_ns.copy()
    times[2] += (times[1] - times[0]) * 1.1e-7
    report = constraints(loaded, replace(pulse, times_ns=times), load_policy())
    timing = {item["name"]: item["outcome"] for item in report["checks"]}
    assert timing["duration"] == "pass"
    assert timing["equal_steps"] == "fail"


def test_control_optional_target_report_and_unrelated_checks(candidate):
    loaded, pulse = candidate
    loaded = replace(
        loaded,
        definition=loaded.definition.model_copy(
            update={
                "problem": loaded.definition.problem.model_copy(update={"final_fidelity": None})
            }
        ),
    )
    items = control_checks(
        loaded, pulse, {"metrics": {}}, ("target", "replay", "integrity"), load_policy()
    )
    assert [item["outcome"] for item in items] == ["inapplicable"] * 3
    assert "no fidelity target" in items[0]["reason"]
    assert "solver-reported" in items[1]["reason"]
    assert not control_listing(loaded)["checks"][1]["available"]


def test_control_checks_outside_shared_subset_are_inapplicable(candidate):
    loaded, pulse = candidate
    problem = {
        **loaded.problem_spec,
        "problem": {**loaded.problem_spec["problem"], "free_phase": True},
    }
    loaded = replace(loaded, problem_spec=problem)
    assert all(not item["available"] for item in control_listing(loaded)["checks"])
    items = control_checks(loaded, pulse, {}, ("replay", "target", "constraints"), load_policy())
    assert all(item["outcome"] == "inapplicable" for item in items)
    assert all("free_phase" in item["reason"] for item in items)


def test_control_replay_complex_goal_and_leakage(candidate):
    loaded, pulse = candidate
    from scipy.linalg import expm

    # A non-contiguous subspace and complex goal catch transpose and ordering mistakes.
    h = np.array([[0, 0.1j, 0.07], [-0.1j, 0.2, 0.06j], [0.07, -0.06j, 0.3]])
    model = replace(
        loaded.system,
        hamiltonian_ghz=h,
        dimensions=(3,),
        drives={name: np.zeros((3, 3)) for name in loaded.system.drives},
    )
    scale = float(1 / np.sqrt(2))
    goal = loaded.definition.goal.model_copy(
        update={
            "gate": None,
            "matrix": [[scale, [0, scale]], [scale, [0, -scale]]],
            "subspace": [[3, 1]],
        }
    )
    loaded = replace(
        loaded, system=model, definition=loaded.definition.model_copy(update={"goal": goal})
    )
    values = replay_values(loaded, pulse, load_policy())
    u = expm(-2j * np.pi * h * pulse.times_ns[-1])
    m = u[np.ix_([2, 0], [2, 0])]
    fidelity = (
        np.trace(m.conj().T @ m).real + abs(np.trace(goal.operator().conj().T @ m)) ** 2
    ) / 6
    assert values["fidelity"] == pytest.approx(fidelity, abs=1e-9)
    assert values["leakage"] == pytest.approx(1 - np.trace(m.conj().T @ m).real / 2, abs=1e-9)
    assert values["leakage"] > 0.001


def retained_candidate(candidate, directory):
    loaded, pulse = candidate
    frozen = directory / "frozen"
    frozen.mkdir()
    compiled = compile_control(loaded.inputs["control.toml"], frozen)
    selection = {
        "requested": "fixture",
        "actual": "fixture",
        "version": "test",
        "fallback_used": False,
    }
    with create_run(
        system_id=loaded.system.id,
        protocol="control",
        inputs={name: Path(value) for name, value in compiled["inputs"].items()},
        environment=capture_environment(ROOT / "uv.lock"),
        code=CodeSnapshot("fixture", b"", {}),
        engine="fixture",
        device="cpu",
        needs={},
        runs_root=directory / "runs",
        launch={"kind": "control", "selection": selection, "problem_sha256": loaded.digest},
    ) as owner:
        run = owner.path
        np.savez(
            run / "artifacts/pulse.npz",
            kind=pulse.kind,
            times_ns=pulse.times_ns,
            amplitudes_ghz=pulse.amplitudes_ghz,
        )
        publish_toml(
            run / "result.toml",
            {
                "kind": "control",
                "engine": "fixture",
                "solver": selection,
                "problem_sha256": loaded.digest,
                "metrics": {"solver_reported_fidelity": 1.0},
                "artifacts": {
                    "pulse": {
                        "path": "artifacts/pulse.npz",
                        "sha256": _digest((run / "artifacts/pulse.npz").read_bytes()),
                    }
                },
            },
        )
        record_start(owner, process_group=os.getpgrp(), gpu_devices=[])
        record_finish(owner, status="completed", exit_code=0, wall_seconds=0, gpu_seconds=0)
    return run


def test_control_cli_check_freezes_binds_and_lists_history(candidate, runner_repository, tmp_path):
    run = retained_candidate(candidate, tmp_path)
    assert read_run(run)["verification"] == []
    assert [item["name"] for item in read_run(run)["check_listing"]["checks"]] == [
        "replay",
        "target",
        "constraints",
    ]
    command, environment, _ = _cpu_job_context(runner_repository, "entanglix")
    completed = subprocess.run(
        [*command, "check", str(run), "replay", "target", "constraints"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    history = read_run(run)["verification"]
    assert len(history) == 1 and history[0]["source_matches"]
    assert [item["outcome"] for item in history[0]["checks"]] == ["pass"] * 3
    verification = read_run(Path(history[0]["path"]))
    assert verification["manifest"]["parent_run_ids"] == [run.name]
    assert (Path(history[0]["path"]) / "spec/problem.toml").read_bytes() == (
        run / "spec/problem.toml"
    ).read_bytes()
    # Source artifacts stay in their own attempt and changes invalidate the binding.
    result = read_run(run)["result"]
    result["metrics"]["solver_reported_fidelity"] = 0.5
    publish_toml(run / "result.toml", result, replace=True)
    assert not read_run(run)["verification"][0]["source_matches"]
    (run / "artifacts/pulse.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        check_run(run, ("constraints",), repository=runner_repository)


def test_control_check_rejects_inconsistent_source_and_refinement(
    candidate, runner_repository, tmp_path
):
    run = retained_candidate(candidate, tmp_path)
    with pytest.raises(ValueError, match="do not use a refinement"):
        check_run(
            run, ("replay",), repository=runner_repository, refinement=tmp_path / "absent.toml"
        )
    result = read_run(run)["result"]
    result["problem_sha256"] = "wrong"
    publish_toml(run / "result.toml", result, replace=True)
    with pytest.raises(ValueError, match="trusted execution plan"):
        check_run(run, ("replay",), repository=runner_repository)
