"""Control solver constraints, gradients, discovery, and owned CPU launches."""

import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tomli_w
from scipy.linalg import expm
from test_control import control_files as control_files

from entanglix.protocols.control import control, load_control
from entanglix.runner import read_run
from entanglix.solvers import Pulse, Solution, select_solver, validate_problem
from entanglix.solvers.grape import GrapeBackend, decode, objective, validate
from entanglix.solvers.piccolo import PiccoloBackend


def test_control_entry_point_and_declared_fallback():
    backend, selection = select_solver("absent", "grape")
    assert isinstance(backend, GrapeBackend)
    assert selection["requested"] == "absent"
    assert selection["actual"] == "grape"
    assert selection["fallback_used"]
    assert selection["version"]
    with pytest.raises(ValueError, match="solver 'absent' is unavailable"):
        select_solver("absent")


def test_piccolo_missing_julia_is_refused_before_attempt(
    control_files, runner_repository, tmp_path, monkeypatch
):
    path, definition = control_files
    definition["solver"]["backend"] = "piccolo"
    path.write_text(tomli_w.dumps(definition))
    # Exercise the gate in-process with a backend-independent preflight.
    monkeypatch.setattr("entanglix.installation.shutil.which", lambda _: None)
    from entanglix.protocols.control import compile_control

    directory = tmp_path / "compiled"
    directory.mkdir()
    with pytest.raises(ValueError, match="piccolo requires Julia"):
        compile_control(path, directory, repository=runner_repository)
    assert not list(directory.iterdir())


@pytest.mark.parametrize("options", [{"max_iter": 0}, {"tol": 0}, {"alien": 3}])
def test_piccolo_bad_options_need_no_julia(control_files, options):
    with pytest.raises(ValueError, match="solver.options"):
        PiccoloBackend().validate(load_control(control_files[0]).problem_spec, options)


def test_solver_preflight_is_not_selected_by_backend_name(control_files, tmp_path, monkeypatch):
    from entanglix.protocols.control import compile_control

    calls = []
    backend = SimpleNamespace(
        validate=lambda problem, options: calls.append((problem, options)),
        capabilities=lambda: GrapeBackend().capabilities(),
    )
    monkeypatch.setattr(
        "entanglix.solvers.select_solver", lambda *_: (backend, {"actual": "plugin"})
    )
    directory = tmp_path / "compiled"
    directory.mkdir()
    compile_control(control_files[0], directory, repository=Path(__file__).resolve().parents[1])
    assert len(calls) == 1
    validate_problem(calls[0][0], calls[0][1], "plugin")


def test_solver_runtime_participates_in_environment_hash(tmp_path):
    from entanglix.runner import capture_environment

    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    first = capture_environment(lock, solver_runtime={"manifest_sha256": "first"})
    second = capture_environment(lock, solver_runtime={"manifest_sha256": "second"})
    assert first["env_hash"] != second["env_hash"]


def test_control_plugin_discovery_and_duplicate_names(monkeypatch):
    dist = SimpleNamespace(
        version="2.0", metadata={"Name": "plugin-dist"}, read_text=lambda _: "RECORD lines"
    )
    entry = SimpleNamespace(name="plugin", dist=dist, load=lambda: GrapeBackend)
    monkeypatch.setattr("entanglix.solvers.entry_points", lambda **kw: [entry])
    selection = select_solver("plugin")[1]
    assert (selection["version"], selection["distribution"]) == ("2.0", "plugin-dist")
    assert len(selection["record_sha256"]) == 64
    monkeypatch.setattr("entanglix.solvers.entry_points", lambda **kw: [entry, entry])
    with pytest.raises(ValueError, match="duplicate solver entry point"):
        select_solver("plugin")


@pytest.mark.parametrize("init", [None, "random"])
def test_control_generated_seed_reproduces_initialization(control_files, monkeypatch, init):
    problem = load_control(control_files[0]).problem_spec
    problem["pulse"].pop("seed", None)
    if init is None:
        problem["pulse"].pop("init", None)
    else:
        problem["pulse"]["init"] = init

    def quadratic(problem):
        def evaluate(amplitudes):
            return np.sum((amplitudes - 0.03) ** 2), 2 * (amplitudes - 0.03)

        return evaluate, np.linspace(0, problem["pulse"]["T"], problem["problem"]["N"])

    monkeypatch.setattr("entanglix.solvers.grape.objective", quadratic)
    monkeypatch.setattr("entanglix.solvers.grape.version", lambda _: "test")
    problem_path = control_files[0].parent / "problem.toml"
    problem_path.write_text(tomli_w.dumps(problem))
    first = GrapeBackend().solve(problem_path, {"max_iter": 1})
    seed = first.trace["effective_seed"]
    assert type(seed) is int and 0 <= seed < 2**63
    problem["pulse"]["seed"] = seed
    problem_path.write_text(tomli_w.dumps(problem))
    second = GrapeBackend().solve(problem_path, {"max_iter": 1})
    np.testing.assert_array_equal(first.pulse.amplitudes_ghz, second.pulse.amplitudes_ghz)
    assert first.objective == second.objective
    assert (
        first.trace["objective_calls"][0]["objective"]
        == second.trace["objective_calls"][0]["objective"]
    )


@pytest.mark.parametrize("fidelity", [0.8, None])
def test_control_job_keeps_native_objective_separate_from_fidelity(tmp_path, monkeypatch, fidelity):
    from entanglix.protocols._control_job import execute
    from entanglix.runner import _digest, _load_record, publish_toml

    (tmp_path / "spec").mkdir()
    (tmp_path / "artifacts").mkdir()
    problem_path = tmp_path / "spec/problem.toml"
    problem_path.write_text("schema_version = 1\n")
    selection = {"actual": "plugin"}
    publish_toml(
        tmp_path / "run.toml",
        {"launch": {"selection": selection, "problem_sha256": _digest(problem_path.read_bytes())}},
    )
    publish_toml(tmp_path / "spec/control.toml", {"solver": {"backend": "plugin"}})
    solution = Solution(
        Pulse(np.array([0, 1]), np.zeros((1, 2))),
        2.5,
        {"converged": True},
        {"iterations": []},
        solver_reported_fidelity=fidelity,
    )
    backend = SimpleNamespace(solve=lambda *_: solution)
    monkeypatch.setattr(
        "entanglix.protocols._control_job.select_solver", lambda *_: (backend, selection)
    )
    monkeypatch.setenv("ENTANGLIX_RUN_DIR", str(tmp_path))
    execute(problem_path)
    result = _load_record(tmp_path / "result.toml")
    assert result["objective"] == 2.5
    assert result["metrics"] == (
        {"solver_reported_fidelity": fidelity} if fidelity is not None else {}
    )


def exact_fidelity(problem, controls):
    drift = decode(problem["system"]["H_drift"])
    drives = [decode(value) for value in problem["system"]["H_drives"]]
    dt = problem["pulse"]["T"] / (problem["problem"]["N"] - 1)
    unitary = np.eye(len(drift), dtype=complex)
    for values in controls[:, :-1].T:
        h = drift + sum(value * drive for value, drive in zip(values, drives, strict=True))
        unitary = expm(-1j * dt * h) @ unitary
    indices = np.asarray(problem["goal"]["subspace"][0]) - 1
    projected = unitary[np.ix_(indices, indices)]
    from entanglix.protocols.control import Goal

    target = Goal.model_validate(problem["goal"]).operator()
    dimension = len(indices)
    return float(
        (
            np.trace(projected.conj().T @ projected).real
            + abs(np.trace(target.conj().T @ projected)) ** 2
        )
        / (dimension * (dimension + 1))
    )


@pytest.mark.filterwarnings("error::numpy.exceptions.ComplexWarning")
@pytest.mark.parametrize("noncontiguous_goal", [False, True])
def test_control_gradient_and_leakage_aware_objective(
    control_files, noncontiguous_goal, monkeypatch
):
    pytest.importorskip("jax")
    from jax.scipy import linalg

    original_expm = linalg.expm

    def complex128_expm(matrix):
        assert str(matrix.dtype) == "complex128"
        return original_expm(matrix)

    monkeypatch.setattr(linalg, "expm", complex128_expm)
    path, _ = control_files
    problem = copy.deepcopy(load_control(path).problem_spec)
    # A complex drive leaks from the declared subspace into a third state.
    h = np.diag([0, 0.1, 0.4])
    dx = np.array([[0, 1j, 0.2], [-1j, 0, 0.3j], [0.2, -0.3j, 0]])
    dy = np.array([[0, 1, 0], [1, 0, 0.2], [0, 0.2, 0]])

    def encoded(a):
        return np.stack([a.real, a.imag], axis=-1).tolist()

    problem["system"]["H_drift"] = encoded(h)
    problem["system"]["H_drives"] = [encoded(dx), encoded(dy)]
    if noncontiguous_goal:
        problem["goal"] = {
            "kind": "unitary",
            "subspace": [[3, 1]],
            "matrix": encoded(np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)),
        }
    controls = np.array([[0, 0.07, -0.03, 0.08, 0], [0, -0.02, 0.04, 0.06, 0]])
    interior_zero = controls.copy()
    interior_zero[:, 2] = 0
    value_and_grad, _ = objective(problem)
    for amplitudes in (controls, interior_zero, np.zeros_like(controls)):
        value, gradient = value_and_grad(amplitudes)
        assert str(value.dtype) == str(gradient.dtype) == "float64"
        assert all(
            device.platform == "cpu" for item in (value, gradient) for device in item.devices()
        )
        assert float(value) == pytest.approx(1 - exact_fidelity(problem, amplitudes), abs=2e-9)
        delta = 1e-5
        for index in np.ndindex(amplitudes.shape):
            plus, minus = amplitudes.copy(), amplitudes.copy()
            plus[index] += delta
            minus[index] -= delta
            expected = (exact_fidelity(problem, minus) - exact_fidelity(problem, plus)) / (
                2 * delta
            )
            assert float(gradient[index]) == pytest.approx(expected, abs=2e-7)
        assert np.all(np.asarray(gradient)[:, -1] == 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("options", {"foreign": True}),
        ("template", "BangBangPulseProblem"),
        ("free_phase", True),
        ("unhandled", 1),
    ],
)
def test_control_unsupported_problem_fields_are_named(control_files, field, value):
    pytest.importorskip("jax")
    problem = load_control(control_files[0]).problem_spec
    problem["problem"][field] = value
    with pytest.raises(ValueError, match=f"problem.{field}"):
        validate(problem, {})


@pytest.mark.parametrize(
    "options",
    [{"max_iter": 0}, {"max_iter": True}, {"tol": float("nan")}, {"tol": 0}, {"alien": 3}],
)
def test_control_invalid_solver_options(control_files, options):
    pytest.importorskip("jax")
    with pytest.raises(ValueError, match="solver.options"):
        validate(load_control(control_files[0]).problem_spec, options)


def test_control_active_derivative_constraints_and_termination(control_files):
    pytest.importorskip("jax")
    path, definition = control_files
    definition["pulse"].update(T=10, init="random", seed=7)
    definition["problem"].update(du_bound=0.01, ddu_bound=0.003, Q=1000)
    path.write_text(tomli_w.dumps(definition))
    problem = load_control(path).problem_spec
    problem_path = control_files[0].parent / "problem.toml"
    problem_path.write_text(tomli_w.dumps(problem))
    solution = GrapeBackend().solve(problem_path, {"max_iter": 15, "tol": 1e-8})
    from entanglix.corrector import load_policy
    from entanglix.corrector.control import control_checks

    checks = control_checks(
        load_control(path),
        solution.pulse,
        {"metrics": {"solver_reported_fidelity": solution.solver_reported_fidelity}},
        ["replay", "constraints"],
        load_policy(),
    )
    assert all(item["outcome"] == "pass" for item in checks)
    amplitudes = solution.pulse.amplitudes_ghz
    dt = np.diff(solution.pulse.times_ns)[0]
    assert np.max(abs(amplitudes)) <= 0.1 + 1e-10
    assert np.max(abs(np.diff(amplitudes, axis=1) / dt)) <= 0.01 + 1e-10
    assert np.max(abs(np.diff(amplitudes, n=2, axis=1) / dt**2)) == pytest.approx(0.003, abs=1e-8)
    np.testing.assert_array_equal(amplitudes[:, [0, -1]], 0)
    assert 1 - solution.objective == pytest.approx(exact_fidelity(problem, amplitudes), abs=2e-9)
    assert "problem.Q" in solution.trace["ignored_guidance_fields"]
    assert solution.trace["effective_seed"] == 7
    assert solution.solver_reported_fidelity == pytest.approx(
        exact_fidelity(problem, amplitudes), abs=2e-9
    )
    assert solution.termination["objective_evaluations"] == len(solution.trace["objective_calls"])
    assert len(solution.trace["objective_calls"]) >= 2
    assert solution.trace["objective_calls"][0]["seconds"] > 0
    assert solution.trace["iterations"]


def test_control_missing_solver_rejected_before_attempt(control_files, runner_repository, tmp_path):
    path, definition = control_files
    definition["solver"]["backend"] = "absent"
    path.write_text(tomli_w.dumps(definition))
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="solver 'absent' is unavailable"):
        control(path, repository=runner_repository, runs_root=root)
    assert not root.exists()


def test_control_missing_extra_rejected_before_attempt(control_files, runner_repository, tmp_path):
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="differentiable extra"):
        control(control_files[0], repository=runner_repository, runs_root=root)
    assert not root.exists()


def test_control_cli_retains_fallback_pulse_and_nonconvergence(
    control_files, differentiable_repository, tmp_path
):
    path, definition = control_files
    definition["solver"] = {
        "backend": "absent",
        "fallback": "grape",
        "options": {"max_iter": 1, "tol": 1e-12},
    }
    definition["timeout_seconds"] = 90
    path.write_text(tomli_w.dumps(definition))
    from entanglix.runner import _cpu_job_context

    command, environment, _ = _cpu_job_context(differentiable_repository, "entanglix")
    # The installed CLI uses the same isolated checkout for its runner.
    result = subprocess.run(
        [*command, "control", str(path), "--runs-root", str(tmp_path / "runs")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    import json

    record = read_run(Path(json.loads(result.stdout)["path"]))
    assert record["status"] == "completed"
    assert record["manifest"]["launch"]["control_model"]["omitted_dissipators"] == ["relaxation"]
    assert not record["result"]["termination"]["converged"]
    assert record["result"]["solver"]["actual"] == "grape"
    assert record["result"]["solver"]["fallback_used"]
    assert "check_history" not in record
    run = Path(record["path"])
    with np.load(run / "artifacts/pulse.npz", allow_pickle=False) as pulse:
        assert pulse["kind"].item() == "zero_order"
        assert pulse["times_ns"][-1] == 2
        assert pulse["amplitudes_ghz"].shape == (2, 5)
    assert (run / "artifacts/solver-trace.toml").exists()
    assert record["manifest"]["launch"]["problem_sha256"] == load_control(path).digest


def test_control_fixed_zero_pulse_handles_upstream_no_variables(control_files):
    pytest.importorskip("jax")
    problem = load_control(control_files[0]).problem_spec
    problem["system"]["params"]["drive_bounds"] = [0, 0]
    problem["pulse"].update(init="zeros")
    problem_path = control_files[0].parent / "problem.toml"
    problem_path.write_text(tomli_w.dumps(problem))
    solution = GrapeBackend().solve(problem_path, {})
    np.testing.assert_array_equal(solution.pulse.amplitudes_ghz, 0)
    assert solution.termination["iterations"] == 0
    assert "fixed by bounds" in solution.termination["message"]
    assert "pulse.seed" in solution.trace["ignored_guidance_fields"]
    assert "effective_seed" not in solution.trace


def test_control_dry_run_and_bad_settings_create_no_attempt(
    control_files, differentiable_repository, tmp_path
):
    path, definition = control_files
    root = tmp_path / "runs"
    preview = control(path, repository=differentiable_repository, runs_root=root, dry_run=True)
    assert preview["preview"]["solver_selection"]["actual"] == "grape"
    assert not root.exists()
    definition["solver"]["options"] = {"unsupported": 1}
    path.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match="solver.options.unsupported"):
        control(path, repository=differentiable_repository, runs_root=root)
    assert not root.exists()


def test_control_invalid_cuda_request_never_acquires_gpu(
    control_files, runner_repository, tmp_path, monkeypatch
):
    path, definition = control_files
    definition["needs"] = {"device": "cuda"}
    path.write_text(tomli_w.dumps(definition))

    def unexpected():
        pytest.fail("CPU control validation must not acquire a GPU")

    monkeypatch.setattr("entanglix.protocols.simulate.select_gpu", unexpected)
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="needs"):
        control(path, repository=runner_repository, runs_root=root)
    assert not root.exists()
