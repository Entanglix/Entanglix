"""Staged design candidates from completed runs, and catalog reads with lineage."""

import io
import json
import os
import shutil
import threading
import tomllib
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from test_calibrate import TRUTH, displaced, sweep_points, write_spec
from test_calibration_checks import source_run as calibration_run
from test_control import control_files as control_files
from test_proposals import BOUND, oscillator_run
from test_sweep import sweep_files

from entanglix.catalog import list_catalog, read_entry, register_result
from entanglix.cli import main
from entanglix.models import load_system
from entanglix.protocols.control import compile_control, override_system
from entanglix.protocols.simulate import compile_simulation
from entanglix.protocols.sweep import compile_sweep
from entanglix.runner import (
    CodeSnapshot,
    _digest,
    capture_environment,
    create_run,
    publish_toml,
    read_run,
    record_finish,
    record_start,
)

ROOT = Path(__file__).resolve().parents[1]
CODE = CodeSnapshot("fixture", b"", {})


def npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def completed(runs, gate, protocol, artifacts, *, kind, status="completed", parents=()):
    """A finished attempt from frozen inputs, whose result cites the given artifacts."""
    with create_run(
        system_id=gate["system_id"],
        protocol=protocol,
        inputs={name: Path(path) for name, path in gate["inputs"].items()},
        environment=capture_environment(ROOT / "uv.lock"),
        code=CODE,
        engine="fixture",
        device="cpu",
        needs={},
        runs_root=runs,
        parent_run_ids=parents,
    ) as owner:
        run = owner.path
        table = {}
        for name, (file_name, data) in artifacts.items():
            (run / "artifacts" / file_name).write_bytes(data)
            table[name] = {"path": f"artifacts/{file_name}", "sha256": _digest(data)}
        result = {"kind": kind, "engine": "fixture", "metrics": {}, "artifacts": table}
        publish_toml(run / "result.toml", result)
        record_start(owner, process_group=os.getpgrp(), gpu_devices=[])
        record_finish(
            owner,
            status=status,
            exit_code=0 if status == "completed" else 1,
            wall_seconds=0,
            gpu_seconds=0,
        )
    return run


def simulation(root, *, f0=0.25, status="completed", parents=()):
    """A spectrum simulation whose spec overrides the system's f0 = 0.2."""
    root.mkdir(exist_ok=True)
    sweep_files(root, kind="spectrum", axes=1)
    base = tomllib.loads((root / "simulate.toml").read_text())
    base["parameters"]["f0"] = f0
    (root / "simulate.toml").write_text(tomli_w.dumps(base))
    frozen = root / "frozen"
    frozen.mkdir()
    gate = compile_simulation(root / "simulate.toml", frozen, [], ROOT)
    output = {"output": ("output.npz", b"output")}
    return completed(
        root.parent / "runs",
        gate,
        "simulate",
        output,
        kind="spectrum",
        status=status,
        parents=parents,
    )


def sweep_run(root, *, expectation=False):
    """Two points over f0, each with a linked drive amplitude, an evolution setting."""
    root.mkdir()
    path, grid = sweep_files(root, axes=1)
    document = tomllib.loads((root / "system.toml").read_text())
    document["drives"] = [{"id": "drive", "factors": ["r.position"]}]
    (root / "system.toml").write_text(tomli_w.dumps(document))
    base = tomllib.loads((root / "simulate.toml").read_text())
    base["evolution"].update(
        drive_amplitudes_ghz={"drive": 0.0},
        carriers={"drive": {"frequency_ghz": 0.2, "phase_rad": 0.0}},
    )
    (root / "simulate.toml").write_text(tomli_w.dumps(base))
    grid["axes"][0]["linked_values"] = {"evolution.drive_amplitudes_ghz.drive": [0.0, 0.001]}
    if expectation:
        grid["expectation"] = {
            "metric": "excited_population",
            "axis": "f0",
            "direction": "decreasing",
        }
    path.write_text(tomli_w.dumps(grid))
    frozen = root / "frozen"
    frozen.mkdir()
    gate = compile_sweep(path, frozen, [], ROOT)
    output = {"output": ("output.npz", b"grid")}
    return completed(root.parent / "runs", gate, "sweep", output, kind="lindblad")


def control_run(control_files, root, amplitudes=((0, 0.05, 0.05, 0, 0), (0, 0, 0.02, 0, 0))):
    path, _ = control_files
    frozen = root / "frozen-control"
    frozen.mkdir(parents=True)
    compiled = compile_control(path, frozen)
    pulse = npz(
        kind="zero_order", times_ns=np.linspace(0, 2, 5), amplitudes_ghz=np.array(amplitudes)
    )
    gate = {"system_id": compiled["system_id"], "inputs": compiled["inputs"]}
    return completed(
        root / "runs", gate, "control", {"pulse": ("pulse.npz", pulse)}, kind="control"
    )


def tree(run):
    return {
        str(path.relative_to(run)): (path.stat().st_mode, path.read_bytes())
        for path in sorted(run.rglob("*"))
        if path.is_file()
    }


def test_simulation_device_holds_its_override_and_the_run_is_unchanged(tmp_path):
    run = simulation(tmp_path / "simulation")
    before = tree(run)
    staged = register_result(run, "device", claim="f0 raised to 0.25")
    identifier = staged["staged_id"]
    assert staged.pop("path") == f"_catalog/staging/{identifier}"
    folder = tmp_path / "runs/_catalog/staging" / identifier
    assert tomllib.loads((folder / "record.toml").read_text()) == staged
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in folder.rglob("*") if path.is_file())
    frozen = run / "spec/system.toml"
    system = folder / "device/system.toml"
    # The frozen system already holds the spec's override; it is copied as frozen.
    assert system.read_bytes() == frozen.read_bytes()
    assert tomllib.loads(system.read_text())["parameters"]["f0"]["value"] == 0.25
    np.testing.assert_array_equal(
        load_system(system).hamiltonian_ghz, load_system(frozen).hamiltonian_ghz
    )
    prefix = f"oscillator/{run.name}"
    result = read_run(run)["result"]
    assert staged["source"] == {"run": prefix, "protocol": "simulate"}
    assert staged["evidence"] == {
        "result_sha256": _digest((run / "result.toml").read_bytes()),
        "artifact_sha256": result["artifacts"]["output"]["sha256"],
        "source_files": {f"{prefix}/spec/system.toml": _digest(frozen.read_bytes())},
    }
    assert staged["device"] == {
        "path": f"_catalog/staging/{identifier}/device",
        "files": {"system.toml": _digest(frozen.read_bytes())},
    }
    assert staged["parent_entry_ids"] == []
    assert (staged["design"], staged["claim"]) == ("device", "f0 raised to 0.25")
    assert tree(run) == before


def test_sweep_point_device_and_evolution_values_as_context(tmp_path):
    run = sweep_run(tmp_path / "sweep")
    staged = register_result(run, "device", claim="the f0 = 0.3 point", point=1)
    system = run.parent.parent / staged["path"] / "device/system.toml"
    frozen = run / "spec/points/000001/system.toml"
    assert system.read_bytes() == frozen.read_bytes()
    assert tomllib.loads(system.read_text())["parameters"]["f0"]["value"] == 0.3
    # The linked drive amplitude is an evolution setting: context, not device.
    assert staged["context"] == {
        "sweep": {
            "point": 1,
            "coordinate": [1],
            "values": {"parameters.f0": 0.3, "evolution.drive_amplitudes_ghz.drive": 0.001},
        }
    }
    assert staged["source"]["point"] == 1
    assert list(staged["evidence"]["source_files"]) == [
        f"oscillator/{run.name}/spec/points/000001/system.toml"
    ]
    with pytest.raises(ValueError, match="one device per point; name the point"):
        register_result(run, "device", claim="grid")
    with pytest.raises(ValueError, match="no point 2"):
        register_result(run, "device", claim="grid", point=2)
    with pytest.raises(ValueError, match="a sweep run holds no pulse"):
        register_result(run, "pulse", claim="grid", point=1)


def test_control_run_stages_its_device_and_its_pulse_with_channel_ids(control_files, tmp_path):
    run = control_run(control_files, tmp_path)
    prefix = f"{read_run(run)['manifest']['system_id']}/{run.name}"
    staged = register_result(run, "device_and_pulse", claim="X gate in 2 ns")
    folder = run.parent.parent / staged["path"]
    # The operator array the system references travels with it.
    assert sorted(path.name for path in (folder / "device").iterdir()) == [
        "operators.npz",
        "system.toml",
    ]
    for name in ("system.toml", "operators.npz"):
        assert (folder / "device" / name).read_bytes() == (run / "spec" / name).read_bytes()
    with np.load(folder / "pulse.npz") as copy, np.load(run / "artifacts/pulse.npz") as source:
        assert sorted(copy.files) == ["amplitudes_ghz", "channels", "kind", "times_ns"]
        assert copy["channels"].tolist() == ["x", "y"]
        for name in ("kind", "times_ns", "amplitudes_ghz"):
            np.testing.assert_array_equal(copy[name], source[name])
    assert staged["pulse"] == {
        "path": f"_catalog/staging/{staged['staged_id']}/pulse.npz",
        "sha256": _digest((folder / "pulse.npz").read_bytes()),
        "channels": ["x", "y"],
        "knots": 5,
        "duration_ns": 2.0,
    }
    assert staged["evidence"]["artifact_sha256"] == _digest(
        (run / "artifacts/pulse.npz").read_bytes()
    )
    assert sorted(staged["evidence"]["source_files"]) == [
        f"{prefix}/artifacts/pulse.npz",
        f"{prefix}/spec/operators.npz",
        f"{prefix}/spec/system.toml",
    ]
    pulse = register_result(run, "pulse", claim="the pulse alone")
    assert "device" not in pulse and pulse["pulse"]["channels"] == ["x", "y"]
    device = register_result(run, "device", claim="the device alone")
    assert "pulse" not in device and device["device"] == staged["device"] | {
        "path": f"_catalog/staging/{device['staged_id']}/device"
    }


def test_calibration_device_carries_every_fitted_value_and_lists_the_undetermined(tmp_path):
    # One line at a alone determines a; b enters the model but no assigned line.
    run = oscillator_run(tmp_path, ["qa"], [5.0005], system=BOUND)
    fit = read_run(run)["result"]["solutions"][0]
    staged = register_result(run, "device", claim="a from one line")
    folder = run.parent.parent / staged["path"]
    values = {name: fit["values"][name] for name in ("a", "b")}
    assert staged["device"]["written"] == values
    assert staged["device"]["undetermined"] == ["b"]
    # The subsystem source and its array file are copied with their digests.
    assert staged["device"]["files"] == {
        name: _digest((folder / "device" / name).read_bytes())
        for name in ("system.toml", "fixed.toml", "fixed.npz")
    }
    for name in ("fixed.toml", "fixed.npz"):
        assert (folder / "device" / name).read_bytes() == (run / "spec" / name).read_bytes()
    parameters = tomllib.loads((folder / "device/system.toml").read_text())["parameters"]
    source = (
        f"calibration run bound/{run.name}, solution 0, "
        f"result sha256 {staged['evidence']['result_sha256']}"
    )
    for name, value in values.items():
        assert parameters[name]["value"] == value and parameters[name]["source"] == source
    frozen = run / "spec/system.toml"
    expected = load_system(
        frozen,
        content=override_system(
            frozen.read_bytes(), {f"parameters.{name}": value for name, value in values.items()}
        ),
    )
    np.testing.assert_array_equal(
        load_system(folder / "device/system.toml").hamiltonian_ghz, expected.hamiltonian_ghz
    )
    assert staged["context"] == {
        "calibration": {
            "solution": 0,
            "outcome": "insufficient evidence",
            "fit_report": f"bound/{run.name}/result.toml",
        }
    }
    assert staged["source"]["solution"] == 0
    assert sorted(staged["evidence"]["source_files"]) == [
        f"bound/{run.name}/spec/{name}" for name in ("fixed.npz", "fixed.toml", "system.toml")
    ]
    with pytest.raises(ValueError, match="a calibrate run holds no pulse"):
        register_result(run, "pulse", claim="none")


def test_calibration_solution_selector_and_axis_map_context(tmp_path):
    axis = {
        "flux": "phi",
        "offset": {"lower": -0.5, "upper": 1.5, "scale": 0.01},
        "period": {"lower": 0.8, "upper": 1.6, "scale": 0.01},
    }
    alias = {**displaced(-1), "offset": displaced(-1)["offset"] + TRUTH["period"]}
    dataset = sweep_points(tmp_path, np.random.default_rng(16))
    run = calibration_run(
        tmp_path, write_spec(tmp_path, [dataset], axis_map=axis, starts=[displaced(1), alias])
    )
    with pytest.raises(ValueError, match="2 distinct solutions; name the one to stage"):
        register_result(run, "device", claim="fluxonium")
    with pytest.raises(ValueError, match="no solution 2"):
        register_result(run, "device", claim="fluxonium", solution=2)
    fit = read_run(run)["result"]["solutions"][1]
    staged = register_result(run, "device", claim="fluxonium", solution=1)
    assert staged["device"]["written"] == {name: fit["values"][name] for name in ("EJ", "EL")}
    assert staged["device"]["undetermined"] == []
    # The axis map describes the measurement setup; the flux parameter is not written.
    assert staged["context"]["calibration"]["axis_map"] == {
        "flux": "phi",
        "offset": fit["values"]["offset"],
        "period": fit["values"]["period"],
    }
    parameters = tomllib.loads(
        (run.parent.parent / staged["path"] / "device/system.toml").read_text()
    )
    assert parameters["parameters"]["phi"] == {"value": 0.25, "unit": "Phi0"}


def test_files_a_script_published_stage_with_the_files_they_reference(tmp_path, capsys):
    for name in ("system.toml", "operators.npz"):
        shutil.copyfile(ROOT / "examples/two-level" / name, tmp_path / name)
    (tmp_path / "script.py").write_text("print('fixture')\n")
    gate = {
        "system_id": "two_level_reference",
        "inputs": {name: tmp_path / name for name in ("system.toml", "operators.npz", "script.py")},
    }
    pulse = npz(
        kind="zero_order",
        times_ns=np.array([0.0, 1.0, 2.0]),
        amplitudes_ghz=np.array([[0.0, 0.1, 0.0]]),
        channels=np.array(["drive"]),
    )
    artifacts = {
        "chip": ("chip.toml", (tmp_path / "system.toml").read_bytes()),
        "operators": ("operators.npz", (tmp_path / "operators.npz").read_bytes()),
        "pulse": ("pulse.npz", pulse),
        "unnamed": (
            "unnamed.npz",
            npz(kind="zero_order", times_ns=[0.0, 1.0], amplitudes_ghz=[[0.0, 0.0]]),
        ),
        "unsigned": (
            "unsigned.npz",
            npz(
                kind="zero_order",
                times_ns=np.array([0, 2, 1], dtype=np.uint64),
                amplitudes_ghz=np.zeros((1, 3)),
                channels=np.array(["drive"]),
            ),
        ),
        "single": ("single.npy", npy(np.zeros(3))),
    }
    run = completed(tmp_path / "runs", gate, "script", artifacts, kind="script")
    prefix = f"two_level_reference/{run.name}"
    device = register_result(run, "device", claim="published chip", artifact="chip")
    folder = run.parent.parent / device["path"] / "device"
    assert (folder / "system.toml").read_bytes() == (run / "artifacts/chip.toml").read_bytes()
    assert (folder / "operators.npz").read_bytes() == (run / "artifacts/operators.npz").read_bytes()
    assert sorted(device["evidence"]["source_files"]) == [
        f"{prefix}/artifacts/chip.toml",
        f"{prefix}/artifacts/operators.npz",
    ]
    # A script has no built-in checks, so no primary artifact binds them.
    assert "artifact_sha256" not in device["evidence"]
    assert device["source"]["artifact"] == "chip"
    staged = register_result(run, "pulse", claim="published pulse", artifact="pulse")
    assert staged["pulse"]["channels"] == ["drive"]
    frozen = register_result(run, "device", claim="the run's own system")
    assert sorted(frozen["evidence"]["source_files"]) == [
        f"{prefix}/spec/operators.npz",
        f"{prefix}/spec/system.toml",
    ]
    with pytest.raises(ValueError, match="published no artifact 'missing'"):
        register_result(run, "pulse", claim="x", artifact="missing")
    with pytest.raises(ValueError, match="name the artifact a script published"):
        register_result(run, "pulse", claim="x")
    with pytest.raises(ValueError, match="a neutral pulse holds exactly"):
        register_result(run, "pulse", claim="x", artifact="unnamed")
    with pytest.raises(ValueError, match="times_ns increase from zero"):
        register_result(run, "pulse", claim="x", artifact="unsigned")
    with pytest.raises(ValueError, match="a neutral pulse is an NPZ archive"):
        register_result(run, "pulse", claim="x", artifact="single")
    command = ["register-result", str(run), "--design", "pulse", "--claim", "x"]
    assert main([*command, "--artifact", "single"]) == 64
    assert "a neutral pulse is an NPZ archive" in capsys.readouterr().err


def test_refusals_write_nothing(control_files, tmp_path):
    run = simulation(tmp_path / "simulation")
    for design, options, message in [
        ("pulse", {}, "a simulate run holds no pulse"),
        ("device_and_pulse", {}, "a device and a pulse together come from a control run"),
        ("both", {}, "design must be one of device, pulse, device_and_pulse"),
        ("device", {"point": 0}, "point selects a design of a sweep run"),
        ("device", {"solution": 0}, "solution selects a design of a calibrate run"),
        ("device", {"artifact": "output"}, "artifact selects a design of a script run"),
        ("device", {"claim": " "}, "a candidate needs a claim note"),
    ]:
        with pytest.raises(ValueError, match=message):
            register_result(run, design, **{"claim": "c", **options})
    failed = simulation(tmp_path / "failed", status="failed")
    with pytest.raises(ValueError, match="needs a completed run; this attempt is failed"):
        register_result(failed, "device", claim="c")
    gate = {"system_id": "oscillator", "inputs": {"system.toml": run / "spec/system.toml"}}
    check = completed(tmp_path / "runs", gate, "verification", {}, kind="verification")
    with pytest.raises(ValueError, match="a verification run holds no design"):
        register_result(check, "device", claim="c")
    malformed = control_run(control_files, tmp_path / "control", amplitudes=((0, 0.1, 0, 0, 0),))
    with pytest.raises(ValueError, match="one name per amplitude row"):
        register_result(malformed, "pulse", claim="c")
    frozen = run / "spec/system.toml"
    frozen.chmod(0o644)
    frozen.write_bytes(frozen.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source input digest mismatch: system.toml"):
        register_result(run, "device", claim="c")
    assert list_catalog(tmp_path / "runs") == []
    assert not (tmp_path / "runs/_catalog/staging").exists() or not any(
        (tmp_path / "runs/_catalog/staging").iterdir()
    )


def test_concurrent_staging_publishes_distinct_records(tmp_path):
    run = simulation(tmp_path / "simulation")
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(register_result(run, "device", claim="c")))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    identifiers = sorted(item["staged_id"] for item in results)
    assert len(set(identifiers)) == 4
    assert [item["id"] for item in list_catalog(tmp_path / "runs")] == identifiers
    # Folders are renamed into place, so no temporary folder remains.
    assert os.listdir(tmp_path / "runs/_catalog") == ["staging"]
    for item in results:
        record = tmp_path / "runs" / item["path"] / "record.toml"
        assert tomllib.loads(record.read_text()) == {k: v for k, v in item.items() if k != "path"}


def test_read_entry_follows_lineage_and_reports_a_removed_source(tmp_path):
    parent = simulation(tmp_path / "parent")
    child = simulation(tmp_path / "child", f0=0.3, parents=(parent.name,))
    staged = register_result(child, "device", claim="child device")
    runs = tmp_path / "runs"
    entry = read_entry(staged["staged_id"], runs)
    assert entry["path"] == f"_catalog/staging/{staged['staged_id']}"
    assert entry["status"] == "staged"
    assert entry["evidence"] == staged["evidence"]
    assert entry["lineage"]["source_run"] == {
        "run": f"oscillator/{child.name}",
        "run_id": child.name,
        "protocol": "simulate",
        "status": "completed",
        "parent_run_ids": [parent.name],
    }
    assert [item["run_id"] for item in entry["lineage"]["ancestors"]] == [parent.name]
    shutil.rmtree(child)
    entry = read_entry(staged["staged_id"], runs)
    assert entry["lineage"]["source_run"] == {"run_id": child.name, "status": "unavailable"}
    assert entry["lineage"]["ancestors"] == []
    # The copied design stays readable without its source run.
    load_system(runs / entry["device"]["path"] / "system.toml")
    with pytest.raises(ValueError, match="not a staged candidate ID"):
        read_entry("../escape", runs)
    with pytest.raises(ValueError, match="no staged candidate"):
        read_entry("s20260101T000000000000Z-00000000", runs)


def test_cli_stages_lists_and_reads(tmp_path, capsys):
    run = simulation(tmp_path / "simulation")
    runs = str(tmp_path / "runs")
    assert main(["register-result", str(run), "--design", "device", "--claim", "c"]) == 0
    staged = json.loads(capsys.readouterr().out)
    assert main(["list-catalog", "--runs-root", runs]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert [(item["id"], item["status"], item["design"]) for item in listing] == [
        (staged["staged_id"], "staged", "device")
    ]
    assert main(["read-entry", staged["staged_id"], "--runs-root", runs]) == 0
    entry = json.loads(capsys.readouterr().out)
    assert entry["evidence"] == staged["evidence"]
    assert main(["register-result", str(run), "--design", "pulse", "--claim", "c"]) == 64
    assert "a simulate run holds no pulse" in capsys.readouterr().err
