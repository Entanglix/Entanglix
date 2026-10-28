"""Serial grids, requested coverage, partial results, replay and GPU dispatch."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from test_simulate import ROOT, protocol_files, wait_for
from test_simulate import gpu_repository as gpu_repository

from entanglix.corrector.verification import check_run
from entanglix.protocols.sweep import compile_sweep, load_sweep, sweep
from entanglix.runner import _digest, _load_record, publish_toml, read_run, rerun


def sweep_files(root, *, kind="lindblad", device="cpu", axes=2):
    base, spec = protocol_files(root, kind=kind, device=device)
    document = _load_record(root / "system.toml")
    document["parameters"] = {f"f{i}": {"value": 0.2, "unit": "GHz"} for i in range(axes)}
    document["model"]["subsystems"][0]["frequency"] = {"parameter": "f0"}
    (root / "system.toml").write_text(tomli_w.dumps(document))
    spec["parameters"] = {f"f{i}": 0.2 for i in range(axes)}
    if kind == "spectrum":
        # The two uncoupled oscillators provide an exact chi=0 spectral companion.
        spec["levels"] = 5
        spec["metrics"] = {"chi": {"kind": "dispersive_shift", "qubit": "r", "resonator": "s"}}
    else:
        spec["evolution"]["drive_amplitudes_ghz"] = {}
    base.write_text(tomli_w.dumps(spec))
    grid = {
        "format_version": 1,
        "base": base.name,
        "axes": [
            {
                "id": f"f{i}",
                "target": f"parameters.f{i}",
                "values": [0.3, 0.35] if kind == "spectrum" else [0.2, 0.3],
            }
            for i in range(axes)
        ],
        "evidence_metric": "chi" if kind == "spectrum" else "excited_population",
    }
    path = root / "sweep.toml"
    path.write_text(tomli_w.dumps(grid))
    return path, grid


def check_report(run, names, repository):
    job = check_run(run, names, repository=repository)
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 0, record["log_tail"]
    return record["result"]["report"]["checks"]


@pytest.mark.parametrize("kind", ["lindblad", "spectrum"])
def test_static_and_dynamic_grids_replay_frozen_inputs(tmp_path, runner_repository, kind):
    path, _ = sweep_files(tmp_path, kind=kind)
    first = sweep(path, runs_root=tmp_path / "runs", repository=runner_repository)
    run = Path(first["path"])
    record = read_run(run)
    assert first["exit_code"] == 0, record["log_tail"]
    assert record["manifest"]["protocol"] == "sweep" and record["verification"] == []
    assert record["result"]["shape"] == [2, 2]
    assert record["sweep_progress"]["completed_count"] == 4
    rows = record["sweep_progress"]["completed_points"]
    assert [row["coordinate"] for row in rows] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    values = record["result"]["metrics"]["chi" if kind == "spectrum" else "excited_population"]
    np.testing.assert_allclose(values, 0 if kind == "spectrum" else np.exp(-0.1), atol=1e-9)
    names = (
        ("spectral_refinement",) if kind == "spectrum" else ("integrity", "segments", "diagnostics")
    )
    checks = check_report(run, names, runner_repository)
    assert all(item.get("outcome", "pass") == "pass" for item in checks)
    assert checks[0]["coverage"]["checked_points"] == [0, 1, 2, 3]
    path.unlink()
    (tmp_path / "system.toml").unlink()
    replay = rerun(run, repository=runner_repository)
    assert replay["exit_code"] == 0
    repeated = read_run(Path(replay["path"]))
    assert repeated["manifest"]["parent_run_ids"] == [run.name]
    assert repeated["result"]["metrics"] == record["result"]["metrics"]
    frozen = run / "spec/points/000000/plan.toml"
    frozen.chmod(0o644)
    frozen.write_text(frozen.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        rerun(run, repository=runner_repository)


def test_cli_preview_and_declared_expectation(tmp_path, runner_repository):
    path, grid = sweep_files(tmp_path, axes=1)
    grid["axes"][0]["values"] = [0.3, 0.2, 0.25]
    grid["expectation"] = {"metric": "excited_population", "axis": "f0", "direction": "decreasing"}
    path.write_text(tomli_w.dumps(grid))
    environment = os.environ | {"PYTHONPATH": str(runner_repository / "src")}
    command = [
        sys.executable,
        "-m",
        "entanglix",
        "sweep",
        str(path),
        "--runs-root",
        str(tmp_path / "runs"),
    ]
    dry = subprocess.run([*command, "--dry-run"], capture_output=True, text=True, env=environment)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["preview"]["grid"]["shape"] == [3]
    assert not (tmp_path / "runs").exists()
    launched = subprocess.run(command, capture_output=True, text=True, env=environment)
    assert launched.returncode == 0, launched.stderr
    run = Path(json.loads(launched.stdout)["path"])
    check = check_report(run, ("declared_expectation",), runner_repository)[0]
    assert check["outcome"] == "pass"
    # Spoof a contradictory aggregate grid, updating its digest as an authored job could.
    output = run / "artifacts/output.npz"
    np.savez(output, excited_population=np.array([0.8, 0.5, 0.6]))
    result = _load_record(run / "result.toml")
    result["artifacts"]["output"]["sha256"] = _digest(output.read_bytes())
    publish_toml(run / "result.toml", result, replace=True)
    reports = check_report(run, ("declared_expectation", "integrity"), runner_repository)
    assert reports[0]["outcome"] == reports[1]["outcome"] == "fail"
    assert reports[0]["checks"][0]["failure_meaning"] == "declared expectation violated"


def test_corner_bound_never_restricts_grid(tmp_path, runner_repository):
    path, _ = sweep_files(tmp_path, axes=4)
    launched = sweep(path, runs_root=tmp_path / "runs", repository=runner_repository)
    run = Path(launched["path"])
    assert launched["exit_code"] == 0
    assert read_run(run)["sweep_progress"]["completed_count"] == 16
    checks = check_report(run, ("integrity", "segments"), runner_repository)
    assert checks[0]["outcome"] == "pass"
    assert checks[1]["outcome"] == "inapplicable" and "corners" in checks[1]["reason"]


def test_warning_selection_stays_bounded(tmp_path, runner_repository):
    path, grid = sweep_files(tmp_path, axes=1)
    grid["axes"][0]["values"] = np.linspace(0.2, 0.4, 12).tolist()
    path.write_text(tomli_w.dumps(grid))
    base = _load_record(tmp_path / "simulate.toml")
    base["evolution"]["initial_state"] = [1, 1]
    (tmp_path / "simulate.toml").write_text(tomli_w.dumps(base))
    launched = sweep(path, runs_root=tmp_path / "runs", repository=runner_repository)
    run = Path(launched["path"])
    checks = check_report(run, ("segments", "diagnostics"), runner_repository)
    assert checks[0]["outcome"] == "pass"
    assert len(checks[0]["coverage"]["checked_points"]) <= 8
    assert checks[0]["coverage"]["uncovered_warning_points"]
    assert len(checks[1]["points"]) == 12 and "outcome" not in checks[1]


def test_linked_axis_values_and_invalid_targets(tmp_path, monkeypatch):
    path, grid = sweep_files(tmp_path, axes=2)
    grid["axes"] = [grid["axes"][0]]
    grid["axes"][0]["linked_values"] = {"parameters.f1": [0.4, 0.5]}
    path.write_text(tomli_w.dumps(grid))
    target = tmp_path / "compiled"
    target.mkdir()
    compile_sweep(path, target, [], ROOT)
    monkeypatch.chdir(tmp_path)
    loaded = load_sweep(Path("compiled"))
    assert [point.definition.evolution.times_ns for point in loaded.points] == [[0, 0.5, 1]] * 2
    for index in range(2):
        assert (
            _load_record(target / "points" / f"{index:06d}" / "system.toml")["parameters"]["f1"][
                "value"
            ]
            == [0.4, 0.5][index]
        )
        assert not (target / "points" / f"{index:06d}" / "simulate.toml").exists()
    grid["axes"][0]["target"] = "evolution.unknown"
    path.write_text(tomli_w.dumps(grid))
    other = tmp_path / "invalid"
    other.mkdir()
    with pytest.raises(ValueError, match="unsupported sweep target"):
        compile_sweep(path, other, [], ROOT)


@pytest.mark.parametrize("stop", ["abort", "loss"])
def test_partial_points_survive_and_rerun_recomputes_grid(tmp_path, runner_repository, stop):
    path, _ = sweep_files(tmp_path, axes=1)
    worker = runner_repository / "src/entanglix/protocols/_sweep_job.py"
    worker.write_text(
        worker.read_text().replace(
            "rows.append(metrics)",
            'rows.append(metrics)\n        if index == 0 and not _load_record(run / "run.toml")["parent_run_ids"]:\n            import time\n            (run / "artifacts/ready").touch()\n            time.sleep(30)',
        )
    )
    runs = tmp_path / "runs"
    supervisor = subprocess.Popen(
        [sys.executable, "-m", "entanglix", "sweep", str(path), "--runs-root", str(runs)],
        env=os.environ | {"PYTHONPATH": str(runner_repository / "src")},
        cwd=runner_repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    run = None
    try:
        run = wait_for(
            lambda: next((p.parent.parent for p in runs.glob("*/*/artifacts/ready")), None)
        )
        from entanglix.runner import abort_run, record_reconciliation

        if stop == "loss":
            supervisor.kill()
            supervisor.communicate(timeout=10)
        abort_run(run)
        if stop == "loss":
            wait_for(lambda: read_run(run)["status"] == "interrupted")
            record_reconciliation(run, last_observed_state="runner lost; job stopped")
        else:
            supervisor.communicate(timeout=10)
        assert read_run(run)["sweep_progress"]["completed_count"] == 1
        assert not (run / "result.toml").exists()
        replay = rerun(run, repository=runner_repository)
        assert replay["exit_code"] == 0
        assert read_run(Path(replay["path"]))["sweep_progress"]["completed_count"] == 2
    finally:
        if run is not None and read_run(run)["status"] == "active":
            os.killpg(read_run(run)["start"]["process_group"], signal.SIGKILL)
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.communicate(timeout=10)


@pytest.mark.gpu
def test_gpu_sweep_checks_and_replay(tmp_path, gpu_repository):
    path, _ = sweep_files(tmp_path, axes=1, device="cuda")
    job = sweep(path, runs_root=tmp_path / "runs", repository=gpu_repository)
    run = Path(job["path"])
    record = read_run(run)
    assert job["exit_code"] == 0, record["log_tail"]
    assert record["result"]["engine"] == "cudensitymat"
    assert record["finished"]["usage"]["gpu_seconds"] > 0 and record["verification"] == []
    assert all(
        item["outcome"] == "pass"
        for item in check_report(run, ("integrity", "segments"), gpu_repository)
    )
    assert rerun(run, repository=gpu_repository)["exit_code"] == 0


def test_readout_coherence_and_dressed_frequency_conventions():
    from entanglix.corrector.metrics import (
        Coherence,
        DispersiveShift,
        EndpointMetric,
        metric_values,
        observable_operator,
        spectral_arrays,
    )
    from entanglix.models import load_system

    system = load_system(ROOT / "examples/readout/system.toml")
    psi = np.zeros(30, dtype=complex)
    psi[0], psi[6] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    density = np.outer(psi, psi.conj())
    operator = observable_operator(system, Coherence(kind="coherence", subsystem="q"))
    expectation = np.trace(density @ operator)
    assert expectation == pytest.approx(-0.5j)
    definition = EndpointMetric(kind="endpoint_coherence", observable="coherence")
    assert metric_values(
        {"coherence": definition}, {"observable.coherence": np.array([expectation])}
    )["coherence"] == pytest.approx(1)
    spectral = DispersiveShift(kind="dispersive_shift", qubit="q", resonator="r")
    arrays = spectral_arrays(system, 5, {"chi": spectral})
    np.testing.assert_array_equal(arrays["dressed_indices.chi"], [0, 2, 1, 4])
    assert metric_values({"chi": spectral}, arrays)["chi"] == pytest.approx(
        -0.0005591160933464678, abs=1e-10
    )
    e00, e01, e10, e11 = arrays["dressed.chi"]
    frequency = (e01 - e00 + e11 - e10) / 2
    grid = _load_record(ROOT / "examples/readout/dephasing.toml")
    assert grid["axes"][0]["linked_values"]["evolution.carriers.readout.frequency_ghz"][
        1
    ] == pytest.approx(frequency, abs=1e-12)


@pytest.mark.parametrize("frame", ["lab", "rotating"])
def test_driven_sweep_in_both_frames_with_complex_coherence(tmp_path, runner_repository, frame):
    path, grid = sweep_files(tmp_path, axes=1)
    document = _load_record(tmp_path / "system.toml")
    document["drives"] = [{"id": "drive", "factors": ["r.position"]}]
    (tmp_path / "system.toml").write_text(tomli_w.dumps(document))
    base = _load_record(tmp_path / "simulate.toml")
    psi = np.zeros(6, dtype=complex)
    psi[0], psi[2] = 1 / np.sqrt(2), 1j / np.sqrt(2)
    np.savez(tmp_path / "state.npz", psi=psi)
    base["evolution"].update(
        initial_state="psi",
        array_file="state.npz",
        drive_amplitudes_ghz={"drive": 0.0},
        carriers={"drive": {"frequency_ghz": 0.2, "phase_rad": 0.3}},
    )
    base["evolution"]["observables"]["coherence"] = {"kind": "coherence", "subsystem": "r"}
    base["metrics"]["coherence"] = {"kind": "endpoint_coherence", "observable": "coherence"}
    if frame == "rotating":
        base["evolution"]["frame"] = {"rwa": True, "frequencies_ghz": {"r": 0.2}}
    (tmp_path / "simulate.toml").write_text(tomli_w.dumps(base))
    grid["axes"][0].update(
        id="amplitude", target="evolution.drive_amplitudes_ghz.drive", values=[0.0, 0.001]
    )
    grid["evidence_metric"] = "coherence"
    path.write_text(tomli_w.dumps(grid))
    result = sweep(path, repository=runner_repository, runs_root=tmp_path / "runs")
    run = Path(result["path"])
    assert result["exit_code"] == 0, read_run(run)["log_tail"]
    assert read_run(run)["result"]["metrics"]["coherence"][0] == pytest.approx(
        np.exp(-0.05), abs=1e-8
    )
    report = check_report(run, ("integrity", "segments"), runner_repository)
    assert all(item["outcome"] == "pass" for item in report)
    assert report[1]["coverage"]["checked_points"] == [0, 1]
    assert "evidence_points" not in check_run(run)["checks"][1]


@pytest.mark.parametrize(
    "bad",
    [
        "duplicate_axis",
        "duplicate_target",
        "nonfinite",
        "linked_length",
        "metric",
        "expectation_axis",
        "path",
    ],
)
def test_invalid_sweeps_reject_before_admission(tmp_path, runner_repository, bad):
    path, grid = sweep_files(tmp_path)
    if bad == "duplicate_axis":
        grid["axes"][1]["id"] = grid["axes"][0]["id"]
    elif bad == "duplicate_target":
        grid["axes"][1]["target"] = grid["axes"][0]["target"]
    elif bad == "nonfinite":
        grid["axes"][0]["values"] = [float("nan")]
    elif bad == "linked_length":
        grid["axes"][0]["linked_values"] = {"parameters.f2": [0.2]}
    elif bad == "metric":
        grid["evidence_metric"] = "unknown"
    elif bad == "expectation_axis":
        grid["expectation"] = {
            "metric": "excited_population",
            "axis": "unknown",
            "direction": "increasing",
        }
    else:
        grid["base"] = "../simulate.toml"
    path.write_text(tomli_w.dumps(grid))
    with pytest.raises(ValueError):
        sweep(path, repository=runner_repository, runs_root=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_readout_refinement_preserves_declared_coherence_gauge(tmp_path):
    import shutil

    from entanglix.corrector import load_policy
    from entanglix.corrector.reference import prepare_refinement
    from entanglix.evolution import normalize_input
    from entanglix.protocols.plans import load_plan
    from entanglix.protocols.simulate import compile_simulation

    for name in ("system.toml", "readout.toml", "readout-state.npz"):
        shutil.copyfile(ROOT / "examples/readout" / name, tmp_path / name)
    base = _load_record(tmp_path / "readout.toml")
    base["parameters"]["EJ"] = 15.0
    base["needs"]["device"] = "cpu"
    (tmp_path / "readout.toml").write_text(tomli_w.dumps(base))
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    compile_simulation(tmp_path / "readout.toml", compiled, [], ROOT)
    source = load_plan(compiled / "system.toml", compiled / "plan.toml")
    target = tmp_path / "refined"
    target.mkdir()
    refined = prepare_refinement(source, target, load_policy())
    original_density = normalize_input(source.system, source.request).initial_density
    target_density = normalize_input(refined.system, refined.request).initial_density
    assert np.trace(original_density @ source.request.observables["coherence"]) == pytest.approx(
        0.5
    )
    assert np.trace(target_density @ refined.request.observables["coherence"]) == pytest.approx(0.5)


def test_dynamiqs_sweep_uses_declared_engine_for_cpu_segments(tmp_path, differentiable_repository):
    path, _ = sweep_files(tmp_path, axes=1)
    base = _load_record(tmp_path / "simulate.toml")
    base["needs"]["gradients"] = True
    (tmp_path / "simulate.toml").write_text(tomli_w.dumps(base))
    result = sweep(path, repository=differentiable_repository, runs_root=tmp_path / "runs")
    run = Path(result["path"])
    assert result["exit_code"] == 0, read_run(run)["log_tail"]
    assert read_run(run)["result"]["engine"] == "dynamiqs"
    checks = check_report(run, ("integrity", "segments"), differentiable_repository)
    assert all(item["outcome"] == "pass" for item in checks)
    assert {
        segment["engine"] for point in checks[1]["points"] for segment in point["segments"]
    } == {"dynamiqs"}


@pytest.mark.parametrize("kind", ["lindblad", "spectrum"])
def test_sweep_without_evidence_metric_keeps_corner_checks(tmp_path, runner_repository, kind):
    path, grid = sweep_files(tmp_path, kind=kind)
    del grid["evidence_metric"]
    path.write_text(tomli_w.dumps(grid))
    launched = sweep(path, repository=runner_repository, runs_root=tmp_path / "runs")
    assert launched["exit_code"] == 0
    run = Path(launched["path"])
    listing = check_run(run)
    assert "points" not in listing
    names = ("segments", "diagnostics") if kind == "lindblad" else ("spectral_refinement",)
    reports = check_report(run, names, runner_repository)
    assert reports[0]["outcome"] == "pass"
    assert reports[0]["coverage"]["checked_points"] == [0, 1, 2, 3]
    assert all(
        "corner" in reasons for reasons in reports[0]["coverage"]["selection_reasons"].values()
    )


@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (1, 1), (1, 2, 3)])
def test_metric_plot_uses_varying_axes(shape):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from entanglix.protocols._sweep_job import draw_metric
    from entanglix.protocols.sweep import Axis

    grid_axes = [
        Axis(id=f"a{i}", target=f"parameters.a{i}", values=list(range(size)))
        for i, size in enumerate(shape)
    ]
    values = np.arange(np.prod(shape)).reshape(shape)
    figure, axes = plt.subplots()
    try:
        draw_metric(axes, grid_axes, values, "metric")
        varying = [axis for axis in grid_axes if len(axis.values) > 1]
        if len(varying) == 2:
            assert axes.collections[0].get_array().size == values.size
            assert axes.get_xlabel() == varying[0].id
            assert axes.get_ylabel() == varying[1].id
        else:
            np.testing.assert_array_equal(axes.lines[0].get_ydata(), values.ravel())
            if varying:
                np.testing.assert_array_equal(axes.lines[0].get_xdata(), varying[0].values)
                assert axes.get_xlabel() == varying[0].id
    finally:
        plt.close(figure)
