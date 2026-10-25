"""Promoted devices are frozen into protocols and retain lineage across replay."""

import copy
import shutil
import tomllib
from pathlib import Path

import pytest
import tomli_w
from test_catalog import ROOT, control_run
from test_control import control_files as control_files
from test_investigations import terms as terms
from test_proposals import BOUND, oscillator_run

from entanglix.catalog import promote, read_entry, register_result
from entanglix.corrector.verification import check_run
from entanglix.investigations import (
    create_investigation,
    record_user_decision,
    revise_investigation,
)
from entanglix.protocols.calibrate import CalibrationSpec, compile_calibration
from entanglix.protocols.control import ControlSpec, compile_control
from entanglix.protocols.simulate import (
    SimulationSpec,
    rerun_simulation,
    simulate,
)
from entanglix.protocols.sweep import rerun_sweep, sweep
from entanglix.runner import _digest, read_run, rerun_script, run_script


@pytest.fixture
def device(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    source = oscillator_run(directory, ["qa", "qb"], [5.0, 7.0], system=BOUND)
    staged = register_result(source, "device", claim="two calibrated oscillator frequencies")
    root = source.parent.parent
    entry = promote(staged["staged_id"], root)
    return source, root, entry


def spectrum(directory, root, entry):
    directory.mkdir()
    path = directory / "simulate.toml"
    nominal = tomllib.loads((root / entry["path"] / "device/system.toml").read_text())[
        "parameters"
    ]["a"]["value"]
    path.write_text(
        tomli_w.dumps(
            {
                "format_version": 1,
                "system_entry": entry["entry_id"],
                "kind": "spectrum",
                "levels": 3,
                "parameters": {"a": nominal},
            }
        )
    )
    return path


def test_entry_sweep_freezes_all_device_files_and_replays_without_catalog(
    device, tmp_path, runner_repository
):
    source, root, entry = device
    protocol = spectrum(tmp_path / "work", root, entry)
    grid = protocol.with_name("sweep.toml")
    grid.write_text(
        tomli_w.dumps(
            {
                "format_version": 1,
                "base": protocol.name,
                "axes": [{"id": "a", "target": "parameters.a", "values": [5.0, 5.2]}],
            }
        )
    )
    job = sweep(grid, runs_root=root, repository=runner_repository)
    assert job["exit_code"] == 0, read_run(Path(job["path"]))["log_tail"]
    run = Path(job["path"])
    manifest = read_run(run)["manifest"]
    assert manifest["source_entry_ids"] == [entry["entry_id"]]
    assert source.name in manifest["parent_run_ids"]
    for name in entry["device"]["files"]:
        assert (run / "spec" / name).read_bytes() == (
            root / entry["device"]["path"] / name
        ).read_bytes()
    frozen = tomllib.loads((run / "spec/simulate.toml").read_text())
    assert frozen["system"] == "system.toml" and "system_entry" not in frozen
    assert [item["name"] for item in check_run(run)["checks"]] == ["spectral_refinement"]
    reuse = read_entry(entry["entry_id"], root)["reuses"]["runs"]
    assert reuse == [{"run": f"bound/{run.name}", "protocol": "sweep"}]
    child = register_result(run, "device", claim="second sweep point", point=1)
    assert child["parent_entry_ids"] == [entry["entry_id"]]
    shutil.rmtree(root / "_catalog")
    replay = rerun_sweep(run, repository=runner_repository)
    assert replay["exit_code"] == 0
    repeated = read_run(Path(replay["path"]))
    assert repeated["manifest"]["source_entry_ids"] == [entry["entry_id"]]
    assert repeated["manifest"]["parent_run_ids"] == [run.name]
    assert repeated["result"]["metrics"] == read_run(run)["result"]["metrics"]


@pytest.mark.parametrize("schema", [SimulationSpec, ControlSpec, CalibrationSpec])
def test_source_choice_is_exclusive(schema, control_files, device):
    if schema is SimulationSpec:
        document = {"format_version": 1, "system": "system.toml", "kind": "spectrum", "levels": 2}
    elif schema is ControlSpec:
        document = copy.deepcopy(control_files[1])
    else:
        document = tomllib.loads((device[0] / "spec/calibrate.toml").read_text())
    document["system_entry"] = device[2]["entry_id"]
    with pytest.raises(ValueError, match="exactly one of system or system_entry"):
        schema.model_validate(document)
    document.pop("system")
    document.pop("system_entry")
    with pytest.raises(ValueError, match="exactly one of system or system_entry"):
        schema.model_validate(document)


@pytest.mark.parametrize("choice", ["unknown", "staged", "both"])
def test_invalid_entry_requests_create_no_attempt(device, tmp_path, runner_repository, choice):
    source, root, entry = device
    protocol = spectrum(tmp_path / "work", root, entry)
    document = tomllib.loads(protocol.read_text())
    message = {
        "unknown": "unknown catalog entry",
        "staged": "promoted catalog entry",
        "both": "exactly one",
    }[choice]
    if choice == "unknown":
        document["system_entry"] = "e20260101T000000000000Z-00000000"
    elif choice == "staged":
        document["system_entry"] = entry["staged_id"]
    else:
        document["system"] = "system.toml"
    protocol.write_text(tomli_w.dumps(document))
    before = sorted(root.glob("*/*/run.toml"))
    with pytest.raises(ValueError, match=message):
        simulate(protocol, runs_root=root, repository=runner_repository)
    assert sorted(root.glob("*/*/run.toml")) == before


def test_control_compilation_uses_entry_arrays_and_refuses_pulse_only_entries(
    control_files, tmp_path
):
    source = control_run(control_files, tmp_path)
    root = source.parent.parent
    staged = register_result(source, "device_and_pulse", claim="two-level device and pulse")
    entry = promote(staged["staged_id"], root)
    protocol, document = control_files
    document.pop("system")
    document["system_entry"] = entry["entry_id"]
    protocol.write_text(tomli_w.dumps(document))
    directory = tmp_path / "compiled"
    directory.mkdir()
    gate = compile_control(protocol, directory, runs_root=root)
    assert gate["source_entry_ids"] == [entry["entry_id"]]
    assert gate["parent_run_ids"] == [source.name]
    assert (directory / "operators.npz").read_bytes() == (
        source / "spec/operators.npz"
    ).read_bytes()
    assert "system_entry" not in tomllib.loads((directory / "control.toml").read_text())
    pulse = promote(register_result(source, "pulse", claim="pulse alone")["staged_id"], root)
    document["system_entry"] = pulse["entry_id"]
    protocol.write_text(tomli_w.dumps(document))
    with pytest.raises(ValueError, match="holds no device"):
        compile_control(protocol, directory, runs_root=root)


def test_calibration_compilation_uses_entry_model_and_adjacent_measurements(device, tmp_path):
    source, root, entry = device
    work = tmp_path / "calibration"
    work.mkdir()
    for name, digest in read_run(source)["manifest"]["spec_sha256"].items():
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "spec" / name, target)
    protocol = work / "calibrate.toml"
    document = tomllib.loads(protocol.read_text())
    document.pop("system")
    document["system_entry"] = entry["entry_id"]
    protocol.write_text(tomli_w.dumps(document))
    directory = tmp_path / "compiled"
    directory.mkdir()
    gate = compile_calibration(protocol, directory, [], ROOT, runs_root=root)
    assert gate["source_entry_ids"] == [entry["entry_id"]]
    assert gate["parent_run_ids"] == [source.name]
    for name in entry["device"]["files"]:
        assert (directory / name).read_bytes() == (
            root / entry["device"]["path"] / name
        ).read_bytes()
    assert "system_entry" not in tomllib.loads((directory / "calibrate.toml").read_text())


def entry_terms(terms, protocol, entry):
    spec = copy.deepcopy(terms)
    base = spec["scope"]["bases"][0]
    base["inputs"] = {**entry["device"]["files"], "simulate.toml": _digest(protocol.read_bytes())}
    base["system_entry"] = entry["entry_id"]
    nominal = tomllib.loads(protocol.read_text())["parameters"]["a"]
    base["variations"] = [{"target": "parameters.a", "values": [nominal, 5.2]}]
    spec["scope"]["operations"] = ["simulate"]
    spec["checkpoint"] = {"finished_attempts": 10}
    return spec


@pytest.mark.parametrize("changed", ["system.toml", "fixed.toml", "fixed.npz"])
def test_investigation_entry_binds_every_device_file_before_drafting(
    device, tmp_path, terms, changed
):
    _, root, entry = device
    protocol = spectrum(tmp_path / "work", root, entry)
    spec = entry_terms(terms, protocol, entry)
    spec["scope"]["bases"][0]["inputs"][changed] = "0" * 64
    with pytest.raises(ValueError, match=changed):
        create_investigation(spec, runs_root=root)
    assert not (root / "_investigations").exists()


def test_investigation_run_and_replay_keep_entry_and_current_revision(
    device, tmp_path, terms, runner_repository
):
    source, root, entry = device
    protocol = spectrum(tmp_path / "work", root, entry)
    spec = entry_terms(terms, protocol, entry)
    draft = create_investigation(spec, runs_root=root)
    investigation = Path(draft["path"])
    record_user_decision(investigation, "start", revision=1)
    job = simulate(
        protocol,
        runs_root=root,
        repository=runner_repository,
        investigation_id=investigation.name,
        base="rwa",
        values={"parameters.a": 5.2},
    )
    assert job["exit_code"] == 0
    run = Path(job["path"])
    manifest = read_run(run)["manifest"]
    assert manifest["source_entry_ids"] == [entry["entry_id"]]
    assert source.name in manifest["parent_run_ids"]
    assert (run / "spec/base/simulate.toml").read_bytes() == protocol.read_bytes()
    references = read_entry(entry["entry_id"], root)["reuses"]
    assert references["investigation_bases"] == [
        {"investigation_id": investigation.name, "revision": 1, "base": "rwa"}
    ]
    assert references["runs"][0]["investigation_id"] == investigation.name
    revised = copy.deepcopy(spec)
    revised["limits"] = {"job_count": 4}
    revise_investigation(investigation, revised)
    record_user_decision(investigation, "approve-revision", revision=2)
    shutil.rmtree(root / "_catalog")
    replay = rerun_simulation(run, repository=runner_repository)
    assert replay["exit_code"] == 0
    repeated = read_run(Path(replay["path"]))["manifest"]
    assert repeated["source_entry_ids"] == [entry["entry_id"]]
    assert repeated["investigation_revision"] == 2
    assert repeated["parent_run_ids"] == [run.name]


def test_base_entry_provenance_is_recorded_when_protocol_uses_a_local_copy(
    device, tmp_path, terms, runner_repository
):
    _, root, entry = device
    protocol = spectrum(tmp_path / "work", root, entry)
    document = tomllib.loads(protocol.read_text())
    document.pop("system_entry")
    document["system"] = "system.toml"
    protocol.write_text(tomli_w.dumps(document))
    for name in entry["device"]["files"]:
        shutil.copyfile(root / entry["device"]["path"] / name, protocol.parent / name)
    spec = entry_terms(terms, protocol, entry)
    investigation = Path(create_investigation(spec, runs_root=root)["path"])
    record_user_decision(investigation, "start", revision=1)
    shutil.rmtree(root / "_catalog")
    job = simulate(
        protocol,
        runs_root=root,
        repository=runner_repository,
        investigation_id=investigation.name,
        base="rwa",
    )
    assert job["exit_code"] == 0
    manifest = read_run(Path(job["path"]))["manifest"]
    assert manifest["source_entry_ids"] == [entry["entry_id"]]


@pytest.mark.parametrize("remove_entry", [False, True])
def test_script_base_keeps_entry_provenance_on_replay_without_catalog(
    device, tmp_path, terms, runner_repository, remove_entry
):
    _, root, entry = device
    work = tmp_path / "script"
    work.mkdir()
    for name in entry["device"]["files"]:
        shutil.copyfile(root / entry["device"]["path"] / name, work / name)
    script = work / "script.py"
    script.write_text(
        "import os\nfrom pathlib import Path\nfrom entanglix.runner import publish_toml\n"
        "publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', "
        "{'engine': 'authored', 'metrics': {}})\n"
    )
    spec = copy.deepcopy(terms)
    spec["scope"]["operations"] = ["run_script"]
    base = spec["scope"]["bases"][0]
    base.update(
        system_entry=entry["entry_id"],
        inputs={**entry["device"]["files"], "script.py": _digest(script.read_bytes())},
        variations=[],
    )
    investigation = Path(create_investigation(spec, runs_root=root)["path"])
    record_user_decision(investigation, "start", revision=1)
    job = run_script(
        work / "system.toml",
        script,
        runs_root=root,
        repository=runner_repository,
        investigation_id=investigation.name,
        base="rwa",
    )
    assert job["exit_code"] == 0
    run = Path(job["path"])
    assert read_run(run)["manifest"]["source_entry_ids"] == [entry["entry_id"]]
    if remove_entry:
        revised = copy.deepcopy(spec)
        revised["scope"]["bases"][0].pop("system_entry")
        revise_investigation(investigation, revised)
        record_user_decision(investigation, "approve-revision", revision=2)
    shutil.rmtree(root / "_catalog")
    repeated = rerun_script(run, repository=runner_repository)
    assert repeated["exit_code"] == 0
    manifest = read_run(Path(repeated["path"]))["manifest"]
    assert manifest["source_entry_ids"] == [entry["entry_id"]]
    assert manifest["parent_run_ids"] == [run.name]
    assert manifest["investigation_revision"] == (2 if remove_entry else 1)
