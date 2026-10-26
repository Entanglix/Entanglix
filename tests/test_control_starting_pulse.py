"""Supplied starting pulses: matching, freezing, solver support, delivery, and lineage."""

import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from test_catalog import ROOT, control_run
from test_control import control_files as control_files

from entanglix.catalog import _neutral_pulse, promote, register_result
from entanglix.corrector.control import load_candidate
from entanglix.corrector.verification import check_run
from entanglix.investigations import create_investigation, record_user_decision
from entanglix.protocols.control import (
    STARTING_PULSE,
    ControlSpec,
    channel_pulse,
    compile_control,
    control,
    load_control,
    rerun_control,
)
from entanglix.runner import _digest, read_run
from entanglix.solvers import Pulse
from entanglix.solvers.grape import GrapeBackend

# Problem order is x, y; the file names them in the other order.
AMPLITUDES = np.array([[0, 0.05, 0.08, 0.03, 0], [0, -0.02, 0, 0.01, 0]])
# Knot times of a retained Piccolo control run (N = 21, T = 20 ns), with their roundoff.
PICCOLO_TIMES = [
    0.0,
    1.0000000000119054,
    2.0000000000238107,
    3.0000000000357163,
    4.0000000000476215,
    5.000000000059527,
    6.000000000071432,
    7.000000000083337,
    8.000000000095243,
    9.000000000107148,
    10.000000000119053,
    11.000000000130958,
    12.000000000142863,
    13.000000000154769,
    14.000000000166674,
    15.000000000178579,
    16.000000000190486,
    17.00000000020239,
    18.000000000214296,
    19.0000000002262,
    20.000000000238106,
]


def write_pulse(path, amplitudes, channels, times=None):
    times = np.linspace(0, 2, amplitudes.shape[1]) if times is None else np.asarray(times)
    np.savez(
        path,
        kind="zero_order",
        times_ns=times,
        amplitudes_ghz=amplitudes,
        channels=np.array(channels),
    )


@pytest.fixture
def start_files(control_files):
    """The two-level control with its seed removed and a file start naming y before x."""
    path, definition = control_files
    del definition["pulse"]["seed"]
    write_pulse(path.with_name("start.npz"), AMPLITUDES[::-1], ["y", "x"])
    definition["starting_pulse"] = {"file": "start.npz"}
    path.write_text(tomli_w.dumps(definition))
    return path, definition


def compiled(path, name="compiled", **kwargs):
    directory = path.parent / name
    directory.mkdir()
    return directory, compile_control(path, directory, **kwargs)


def test_file_start_is_frozen_beside_an_unchanged_problem_spec(start_files):
    path, definition = start_files
    plain = path.with_name("plain.toml")
    plain.write_text(
        tomli_w.dumps({key: value for key, value in definition.items() if key != "starting_pulse"})
    )
    _, without = compiled(plain, "without")
    directory, gate = compiled(path)
    assert gate["problem_sha256"] == without["problem_sha256"]
    authored = path.with_name("start.npz").read_bytes()
    assert (directory / STARTING_PULSE).read_bytes() == authored
    digest = _digest(authored)
    assert gate["starting_pulse"] == {
        "sha256": digest,
        "source": {"file": "start.npz", "sha256": digest},
    }
    assert gate["source_entry_ids"] == gate["parent_run_ids"] == []
    frozen = tomllib.loads((directory / "control.toml").read_text())
    assert frozen["starting_pulse"] == {"file": STARTING_PULSE}
    # Checks, and the models they rebuild for samples, never read the start.
    assert STARTING_PULSE not in load_candidate(directory).inputs
    sample = path.with_name("sample")
    sample.mkdir()
    for name in ("control.toml", "system.toml", "operators.npz"):
        shutil.copyfile(directory / name, sample / name)
    load_control(sample / "control.toml")


@pytest.mark.parametrize(
    "change",
    [{"pulse": {"init": "random"}}, {"pulse": {"seed": 3}}, {"start": {"entry": "e", "run": "r"}}],
)
def test_init_seed_and_a_second_source_are_refused(start_files, change):
    _, definition = start_files
    definition["pulse"].update(change.get("pulse", {}))
    definition["starting_pulse"] = change.get("start", definition["starting_pulse"])
    with pytest.raises(ValueError, match="starting pulse replaces|exactly one of entry"):
        ControlSpec.model_validate(definition)


@pytest.mark.parametrize(
    ("amplitudes", "channels", "times", "ddu_bound", "message"),
    [
        (AMPLITUDES[:1], ["x"], None, None, r"missing \['y'\]"),
        (np.vstack([AMPLITUDES, AMPLITUDES[:1]]), ["x", "y", "z"], None, None, r"extra \['z'\]"),
        (np.zeros((2, 6)), ["x", "y"], None, None, "shape does not match"),
        (AMPLITUDES, ["x", "y"], np.linspace(0, 2 * (1 + 2e-7), 5), None, "equal_steps exceeds"),
        (np.array([[0, 0.05, 0.1 + 1e-6, 0.05, 0], [0] * 5]), ["x", "y"], None, None, "amplitude"),
        (np.array([[0, 0.09, -0.02, 0, 0], [0] * 5]), ["x", "y"], None, None, "du exceeds"),
        (np.array([[0, 0.08, 0, 0.08, 0], [0] * 5]), ["x", "y"], None, 0.5, "ddu exceeds"),
    ],
    ids=["missing", "extra", "knots", "times", "amplitude", "du", "ddu"],
)
def test_incompatible_starts_are_refused_unchanged(
    start_files, amplitudes, channels, times, ddu_bound, message
):
    path, definition = start_files
    write_pulse(path.with_name("start.npz"), np.asarray(amplitudes, dtype=float), channels, times)
    if ddu_bound is not None:
        definition["problem"]["ddu_bound"] = ddu_bound
        path.write_text(tomli_w.dumps(definition))
    directory = path.with_name("compiled")
    directory.mkdir()
    with pytest.raises(ValueError, match=message):
        compile_control(path, directory)
    assert not list(directory.iterdir())


def test_retained_piccolo_knot_times_are_accepted(start_files):
    path, definition = start_files
    definition["pulse"]["T"] = 20.0
    definition["problem"]["N"] = 21
    path.write_text(tomli_w.dumps(definition))
    amplitudes = 0.02 * np.sin(np.pi * np.arange(21) / 20) * np.array([[1], [0.5]])
    write_pulse(path.with_name("start.npz"), amplitudes, ["x", "y"], PICCOLO_TIMES)
    directory, _ = compiled(path)
    with np.load(directory / STARTING_PULSE) as frozen:
        assert frozen["times_ns"].tolist() == PICCOLO_TIMES


@pytest.mark.parametrize(
    "solver", [{"backend": "piccolo"}, {"backend": "absent", "fallback": "piccolo"}]
)
def test_piccolo_is_refused_also_as_fallback(start_files, solver):
    path, definition = start_files
    definition["solver"] = solver
    path.write_text(tomli_w.dumps(definition))
    directory = path.with_name("compiled")
    directory.mkdir()
    with pytest.raises(ValueError, match="'piccolo' does not support a starting pulse"):
        compile_control(path, directory, repository=ROOT)
    assert not list(directory.iterdir())


@pytest.mark.parametrize("relative_shift", [0, 5e-8])
def test_baseline_first_evaluation_is_the_start_on_its_own_grid(
    start_files, monkeypatch, relative_shift
):
    pytest.importorskip("jax")
    from entanglix.solvers import grape

    path, _ = start_files
    # A permitted grid: every step lies within the bracket's relative half-width of 1e-7.
    times = np.arange(5) * 0.5 * (1 + relative_shift)
    write_pulse(path.with_name("start.npz"), AMPLITUDES[::-1], ["y", "x"], times)
    directory, _ = compiled(path)
    start = channel_pulse(_neutral_pulse(directory / STARTING_PULSE), ["x", "y"])
    evaluated = []
    objective = grape.objective

    def recorded(problem):
        evaluate, grid = objective(problem)

        def call(amplitudes):
            evaluated.append(np.array(amplitudes))
            return evaluate(amplitudes)

        return call, grid

    monkeypatch.setattr("entanglix.solvers.grape.objective", recorded)
    solution = GrapeBackend().solve(directory / "problem.toml", {"max_iter": 2}, start)
    np.testing.assert_array_equal(evaluated[0], AMPLITUDES)
    assert solution.pulse.times_ns.tolist() == np.linspace(0, 2, 5).tolist()
    assert (solution.pulse.times_ns != times).any() == bool(relative_shift)
    assert "effective_seed" not in solution.trace


@pytest.mark.parametrize(("knot", "value"), [(2, 0.1 + 5e-8), (0, 0.01)], ids=["slack", "end"])
def test_baseline_refuses_a_start_it_would_clip(start_files, knot, value):
    pytest.importorskip("jax")
    path, _ = start_files
    amplitudes = AMPLITUDES.copy()
    amplitudes[0, knot] = value
    write_pulse(path.with_name("start.npz"), amplitudes, ["x", "y"])
    directory = path.with_name("compiled")
    directory.mkdir()
    # The constraints rule admits both; only the baseline's own bounds refuse them.
    with pytest.raises(ValueError, match="grape refuses a starting pulse outside its bounds"):
        compile_control(path, directory, repository=ROOT)
    assert not list(directory.iterdir())
    problem = load_control(path).problem_spec
    with pytest.raises(ValueError, match="outside its bounds"):
        GrapeBackend().validate(problem, {}, Pulse(np.linspace(0, 2, 5), amplitudes))


@pytest.fixture
def starting_repository(runner_repository):
    # Real distribution entry points: one solver declares support, one does not.
    data = Path(__file__).parent / "data"
    for source, module in (
        ("starting_pulse_plugin.py", "_fixture_starting"),
        ("control_plugin.py", "_fixture_control"),
    ):
        shutil.copyfile(data / source, runner_repository / f"src/entanglix/{module}.py")
    project = runner_repository / "pyproject.toml"
    config = tomllib.loads(project.read_text())
    config["project"]["entry-points"]["entanglix.solvers"].update(
        starting="entanglix._fixture_starting:Backend",
        fixture="entanglix._fixture_control:Backend",
    )
    project.write_text(tomli_w.dumps(config))
    return runner_repository


def test_sources_are_delivered_recorded_and_replayed_without_catalog(
    control_files, starting_repository
):
    path, definition = control_files
    source = control_run(control_files, path.parent, amplitudes=AMPLITUDES)
    root = source.parent.parent
    entry = promote(register_result(source, "pulse", claim="two-level pulse")["staged_id"], root)
    write_pulse(path.with_name("start.npz"), AMPLITUDES[::-1], ["y", "x"])
    del definition["pulse"]["seed"]
    definition["solver"] = {"backend": "starting"}
    path.write_text(tomli_w.dumps(definition))
    problem = load_control(path).digest
    runs = {}
    for kind, value, lineage in (
        ("entry", entry["entry_id"], {"source_entry_ids": [entry["entry_id"]]}),
        ("run", f"{source.parent.name}/{source.name}", {"parent_run_ids": [source.name]}),
        ("file", "start.npz", {}),
    ):
        definition["starting_pulse"] = {kind: value}
        path.write_text(tomli_w.dumps(definition))
        job = control(path, runs_root=root, repository=starting_repository)
        record = read_run(Path(job["path"]))
        assert record["status"] == "completed", record["log_tail"]
        manifest = record["manifest"]
        launch = manifest["launch"]
        assert launch["selection"]["initial_pulse"] is True
        assert record["result"]["problem_sha256"] == problem
        frozen = Path(job["path"]) / "spec" / STARTING_PULSE
        assert launch["starting_pulse"]["sha256"] == _digest(frozen.read_bytes())
        assert launch["starting_pulse"]["source"].items() >= {kind: value}.items()
        assert manifest.get("source_entry_ids", []) == lineage.get("source_entry_ids", [])
        assert manifest["parent_run_ids"] == lineage.get("parent_run_ids", [])
        trace = tomllib.loads((Path(job["path"]) / "artifacts/solver-trace.toml").read_text())
        assert trace["starting_pulse"] == launch["starting_pulse"]
        # The fixture returns its start: the job delivered the amplitudes in problem order.
        with np.load(Path(job["path"]) / "artifacts/pulse.npz") as pulse:
            np.testing.assert_array_equal(pulse["amplitudes_ghz"], AMPLITUDES)
        runs[kind] = record
    # Checks are unchanged: they test the returned pulse.
    checked = check_run(
        Path(runs["file"]["path"]), ("constraints",), repository=starting_repository
    )
    assert checked["exit_code"] == 0
    report = read_run(Path(checked["path"]))["result"]["report"]
    assert [item["outcome"] for item in report["checks"]] == ["pass"]
    entry_run = Path(runs["entry"]["path"])
    assert (entry_run / "spec" / STARTING_PULSE).read_bytes() == (
        root / entry["pulse"]["path"]
    ).read_bytes()
    shutil.rmtree(root / "_catalog")
    replay = rerun_control(entry_run, repository=starting_repository)
    record = read_run(Path(replay["path"]))
    assert record["status"] == "completed", record["log_tail"]
    assert record["manifest"]["source_entry_ids"] == [entry["entry_id"]]
    assert record["manifest"]["parent_run_ids"] == [entry_run.name]
    original = runs["entry"]["manifest"]["launch"]["starting_pulse"]
    assert record["manifest"]["launch"]["starting_pulse"] == original


def test_a_run_source_reached_through_a_link_outside_the_runs_root_is_refused(control_files):
    path, definition = control_files
    source = control_run(control_files, path.parent, amplitudes=AMPLITUDES)
    # Another runs root whose system folder links to the first one's.
    root = path.with_name("linked")
    root.mkdir()
    (root / source.parent.name).symlink_to(source.parent)
    del definition["pulse"]["seed"]
    definition["starting_pulse"] = {"run": f"{source.parent.name}/{source.name}"}
    path.write_text(tomli_w.dumps(definition))
    directory = path.with_name("compiled")
    directory.mkdir()
    with pytest.raises(ValueError, match="in the runs root"):
        compile_control(path, directory, runs_root=root)
    assert not list(directory.iterdir())


def test_an_early_run_pulse_without_kind_is_frozen_as_a_zero_order_start(control_files):
    path, definition = control_files
    source = control_run(control_files, path.parent, amplitudes=AMPLITUDES, kind=None)
    del definition["pulse"]["seed"]
    definition["starting_pulse"] = {"run": f"{source.parent.name}/{source.name}"}
    path.write_text(tomli_w.dumps(definition))
    directory, gate = compiled(path, runs_root=source.parent.parent)
    assert gate["parent_run_ids"] == [source.name]
    with np.load(directory / STARTING_PULSE) as frozen:
        assert frozen["kind"].item() == "zero_order"
        assert frozen["channels"].tolist() == ["x", "y"]
        np.testing.assert_array_equal(frozen["amplitudes_ghz"], AMPLITUDES)


def test_base_binds_its_starting_file_and_replays_from_the_frozen_copy(
    start_files, starting_repository
):
    path, definition = start_files
    definition["solver"] = {"backend": "starting"}
    path.write_text(tomli_w.dumps(definition))
    # The authored start is a base input under its fixed name.
    authored = path.with_name("start.npz").read_bytes()
    inputs = {**load_control(path).input_digests, STARTING_PULSE: _digest(authored)}
    terms = {
        "format_version": 1,
        "question": "Does the gate solve from the supplied start?",
        "assumptions": ["Closed rotating-frame model."],
        "scope": {
            "operations": ["control", "check"],
            "bases": [{"name": "gate", "inputs": inputs}],
        },
        "evidence_requirements": {
            "requirements": ["Report the solve."],
            "checks": {"gate": ["constraints"]},
        },
        "checkpoint": {"finished_attempts": 10},
        "review_criteria": [],
        "stop_criteria": [],
        "limits": {},
    }
    root = path.with_name("runs")
    investigation = Path(create_investigation(terms, runs_root=root)["path"])
    record_user_decision(investigation, "start", revision=1)
    arguments = {
        "runs_root": root,
        "repository": starting_repository,
        "investigation_id": investigation.name,
        "base": "gate",
    }
    run = Path(control(path, **arguments)["path"])
    record = read_run(run)
    assert record["status"] == "completed", record["log_tail"]
    assert (run / "spec/base" / STARTING_PULSE).read_bytes() == authored
    write_pulse(path.with_name("start.npz"), AMPLITUDES, ["x", "y"])
    with pytest.raises(ValueError, match=f"changed digests: {STARTING_PULSE}"):
        control(path, **arguments)
    replay = read_run(Path(rerun_control(run, repository=starting_repository)["path"]))
    assert replay["status"] == "completed", replay["log_tail"]
    original = record["manifest"]["launch"]["starting_pulse"]
    assert replay["manifest"]["launch"]["starting_pulse"] == original


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        ("starting", "the starting fixture refuses this pulse"),
        # A refusal at the capability, not a TypeError: the argument never reaches it.
        ("fixture", "solver 'fixture' does not support a starting pulse"),
    ],
)
def test_refused_starts_leave_no_attempt(start_files, starting_repository, backend, message):
    path, definition = start_files
    amplitudes = AMPLITUDES.copy()
    amplitudes[0, 1] = 0.0625  # The value the starting fixture refuses in its preflight.
    write_pulse(path.with_name("start.npz"), amplitudes, ["x", "y"])
    definition["solver"] = {"backend": backend}
    path.write_text(tomli_w.dumps(definition))
    root = path.with_name("runs")
    with pytest.raises(ValueError, match=message):
        control(path, runs_root=root, repository=starting_repository)
    assert not list(root.glob("*/*/run.toml"))
