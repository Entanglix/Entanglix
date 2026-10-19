"""Control export agrees with the shared frame and CPU evolution conventions."""

import hashlib
import json
import shutil
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from jsonschema import Draft7Validator
from scipy.linalg import expm

from entanglix.engines import QuTiPIntegrator
from entanglix.evolution import Carrier, EvolutionInput, RotatingFrame
from entanglix.models import load_system
from entanglix.protocols.control import AMICODE_REVISION, compile_control, load_control
from entanglix.runner import CodeSnapshot, capture_environment, create_run, record_finish

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "tests/data/amicode/problemspec.oss.schema.json"
EXAMPLE = ROOT / "examples/transmon-control/control.toml"


def decode(encoded):
    values = np.asarray(encoded)
    return values[..., 0] + 1j * values[..., 1]


@pytest.fixture
def control_files(tmp_path):
    for name in ("system.toml", "operators.npz"):
        shutil.copyfile(ROOT / "examples/two-level" / name, tmp_path / name)
    system = tomllib.loads((tmp_path / "system.toml").read_text())
    # This dissipator mixes excitation shifts; control must omit it before frame preparation.
    system["dissipators"][0]["factors"] = ["x"]
    (tmp_path / "system.toml").write_text(tomli_w.dumps(system))
    definition = {
        "format_version": 1,
        "system": "system.toml",
        "frame": {
            "frequencies_ghz": {system["id"]: 0.2},
            "excitation_numbers": {system["id"]: [0, 1]},
            "rwa": True,
        },
        "channels": [
            {
                "id": name,
                "drive": "x_drive",
                "frequency_ghz": 0.2,
                "phase_rad": phase,
                "amplitude_bound_ghz": 0.1,
            }
            for name, phase in (("x", 0.0), ("y", np.pi / 2))
        ],
        "goal": {"kind": "unitary", "gate": "X", "subspace": [[1, 2]]},
        "pulse": {"kind": "zero_order", "T": 2.0, "seed": 7},
        "problem": {"template": "SmoothPulseProblem", "N": 5, "du_bound": 0.2},
        "solver": {"backend": "grape"},
    }
    path = tmp_path / "control.toml"
    path.write_text(tomli_w.dumps(definition))
    return path, definition


def test_export_conforms_to_recorded_oss_schema_and_is_solver_independent(control_files):
    path, definition = control_files
    first = load_control(path)
    schema = json.loads(SCHEMA.read_text())
    Draft7Validator.check_schema(schema)
    Draft7Validator(schema).validate(tomllib.loads(first.problem_bytes.decode()))
    assert first.listing["amicode_revision"] == AMICODE_REVISION
    assert "solver" not in first.problem_spec
    assert first.digest == hashlib.sha256(first.problem_bytes).hexdigest()
    assert first.listing["omitted_dissipators"] == ["relaxation"]
    assert not first.system.dissipators
    assert first.listing["channel_order"] == ["x", "y"]
    assert first.problem_spec["problem"]["options"]["timesteps_all_equal"] is True
    step_bounds = first.problem_spec["problem"]["free_dt"]
    assert step_bounds[0] < 0.5 < step_bounds[1]
    assert first.definition.solver.options == {}
    assert "final_fidelity" not in first.problem_spec["problem"]
    # Solver convergence settings and runner limits do not alter the shared problem.
    definition["solver"] = {"backend": "piccolo", "options": {"max_iter": 100, "tol": 1e-8}}
    definition["timeout_seconds"] = 10.0
    path.write_text("# another solver\n" + tomli_w.dumps(definition))
    second = load_control(path)
    assert second.digest == first.digest
    assert second.problem_bytes == first.problem_bytes
    definition["channels"][0]["phase_rad"] = 0.3
    path.write_text(tomli_w.dumps(definition))
    assert load_control(path).digest != first.digest


def test_quadrature_sign_and_single_angular_frequency_conversion(control_files):
    path, _ = control_files
    loaded = load_control(path)
    raw = loaded.problem_spec["system"]
    np.testing.assert_allclose(decode(raw["H_drift"]), np.zeros((2, 2)), atol=1e-14)
    np.testing.assert_allclose(decode(raw["H_drives"][0]), np.pi * np.array([[0, 1], [1, 0]]))
    np.testing.assert_allclose(
        decode(raw["H_drives"][1]), np.pi * np.array([[0, 1j], [-1j, 0]]), atol=1e-14
    )


@pytest.mark.parametrize("first_gate", [False, True])
def test_exported_fixed_pulse_matches_reference_in_the_control_frame(control_files, first_gate):
    path, definition = control_files
    if first_gate:
        path = EXAMPLE
        definition = tomllib.loads(path.read_text())
    loaded = load_control(path)
    original = load_system(path.parent / "system.toml")
    channels = definition["channels"]
    reference_model = replace(
        original,
        drives={channel["id"]: original.drives[channel["drive"]] for channel in channels},
        drive_factors={},
        dissipators={},
        dissipator_factors={},
    )
    times = np.linspace(0, definition["pulse"]["T"], definition["problem"]["N"])
    controls = {
        "x": np.resize(np.array([0.05, -0.04, 0.02, 0.03]), len(times) - 1),
        "y": np.resize(np.array([0.02, 0.04, -0.01, 0.01]), len(times) - 1),
    }
    initial = np.array([1, 1j, 0.25 + 0.17j] if first_gate else [1, 1j])
    initial /= np.linalg.norm(initial)
    reference = QuTiPIntegrator().evolve(
        reference_model,
        EvolutionInput(
            initial,
            times,
            drive_amplitudes_ghz=controls,
            carriers={
                channel["id"]: Carrier(channel["frequency_ghz"], channel["phase_rad"])
                for channel in channels
            },
            frame=RotatingFrame(**definition["frame"]),
        ),
    )
    raw = loaded.problem_spec["system"]
    drift = decode(raw["H_drift"])
    drives = [decode(operator) for operator in raw["H_drives"]]
    state = initial
    exact_states = [np.outer(state, state.conj())]
    for index, interval in enumerate(np.diff(times)):
        generator = drift + sum(
            controls[channel["id"]][index] * drive for channel, drive in zip(channels, drives)
        )
        state = expm(-1j * generator * interval) @ state
        exact_states.append(np.outer(state, state.conj()))
    # Engine output is in the lab basis; the declared goal is in the control frame.
    frequency = definition["frame"]["frequencies_ghz"][original.id]
    phase = np.exp(2j * np.pi * times[:, None] * frequency * np.arange(len(initial)))
    rotated = phase[:, :, None] * np.asarray(reference.states) * phase.conj()[:, None, :]
    np.testing.assert_allclose(exact_states, rotated, rtol=0, atol=1e-9)


def test_first_gate_keeps_charge_scale_and_probe_quadrature_orientation():
    loaded = load_control(EXAMPLE)
    source = load_system(EXAMPLE.parent / "system.toml")
    origin = loaded.listing["energy_origin_ghz"]
    assert origin == pytest.approx(source.hamiltonian_ghz[0, 0].real, abs=1e-12)
    assert abs(origin) > 1
    np.testing.assert_allclose(
        decode(loaded.problem_spec["system"]["H_drift"]),
        2 * np.pi * (loaded.system.hamiltonian_ghz - origin * np.eye(3)),
        rtol=0,
        atol=1e-12,
    )
    charge = source.operators["charge"]
    scale = abs(charge[0, 1])
    lowering = np.diag(np.diag(charge, 1), 1)
    np.testing.assert_allclose(
        loaded.system.drives["x"], (lowering + lowering.conj().T) / 2, atol=1e-13
    )
    np.testing.assert_allclose(
        loaded.system.drives["y"], 1j * (lowering - lowering.conj().T) / 2, atol=1e-13
    )
    frequency = float((source.hamiltonian_ghz[1, 1] - source.hamiltonian_ghz[0, 0]).real)
    assert loaded.definition.frame.frequencies_ghz[source.id] == pytest.approx(
        frequency, rel=0, abs=1e-9
    )
    np.testing.assert_allclose(
        [channel.amplitude_bound_ghz * scale for channel in loaded.definition.channels],
        [0.06, 0.06],
    )
    assert loaded.definition.problem.du_bound * scale == pytest.approx(0.1)
    assert loaded.definition.problem.ddu_bound * scale == pytest.approx(0.1)
    assert loaded.definition.problem.final_fidelity == 0.999
    Draft7Validator(json.loads(SCHEMA.read_text())).validate(loaded.problem_spec)


def test_compilation_freezes_inputs_in_existing_run_records(control_files, tmp_path):
    path, definition = control_files
    # A noncanonical authored system filename is canonicalized in frozen inputs.
    source = path.parent / "system.toml"
    source.rename(path.parent / "chip.toml")
    definition["system"] = "chip.toml"
    path.write_text(tomli_w.dumps(definition))
    directory = tmp_path / "compiled"
    directory.mkdir()
    compiled = compile_control(path, directory)
    generated = (directory / "problem.toml").read_bytes()
    assert compiled["problem_sha256"] == hashlib.sha256(generated).hexdigest()
    assert load_control(directory / "control.toml").problem_bytes == generated
    with create_run(
        system_id=compiled["system_id"],
        protocol="control",
        inputs={name: Path(value) for name, value in compiled["inputs"].items()},
        environment=capture_environment(ROOT / "uv.lock"),
        code=CodeSnapshot("export-test", b"", {}),
        engine="not-selected",
        device="cpu",
        needs={},
        runs_root=tmp_path / "runs",
        launch={"problem_sha256": compiled["problem_sha256"], "preview": compiled["preview"]},
    ) as owner:
        (path.parent / "chip.toml").write_text("source changed")
        assert (owner.path / "spec/problem.toml").read_bytes() == generated
        manifest = tomllib.loads((owner.path / "run.toml").read_text())
        assert manifest["spec_sha256"]["problem.toml"] == compiled["problem_sha256"]
        assert manifest["launch"]["preview"]["omitted_dissipators"] == ["relaxation"]
        assert not (owner.path / "spec/problem.toml").stat().st_mode & 0o222
        record_finish(owner, status="failed", exit_code=64, wall_seconds=0, gpu_seconds=0)


def test_parameters_and_varied_values_override_only_the_frozen_inputs(tmp_path):
    for name in ("system.toml", "control.toml"):
        shutil.copyfile(EXAMPLE.parent / name, tmp_path / name)
    path = tmp_path / "control.toml"
    definition = tomllib.loads(path.read_text())
    definition["parameters"] = {"Csh": 60.0}
    path.write_text(tomli_w.dumps(definition))
    nominal = load_control(path)
    assert nominal.problem_bytes == load_control(EXAMPLE).problem_bytes
    values = {"parameters.Csh": 30.0, "pulse.seed": 3}
    varied = load_control(path, values=values)
    # Varied values leave the authored base and its digests unchanged.
    assert varied.input_digests == nominal.input_digests
    assert varied.system.provenance["parameters"]["Csh"]["value"] == 30.0
    assert varied.problem_spec["pulse"]["seed"] == 3
    assert varied.problem_spec["system"]["H_drift"] != nominal.problem_spec["system"]["H_drift"]
    directory = tmp_path / "compiled"
    directory.mkdir()
    compiled = compile_control(path, directory, retain_base=True, values=values)
    assert compiled["base_inputs"] == nominal.input_digests
    assert compiled["scope_values"] == [values]
    frozen = tomllib.loads((directory / "control.toml").read_text())
    assert frozen["parameters"] == {"Csh": 30.0} and frozen["pulse"]["seed"] == 3
    system = tomllib.loads((directory / "system.toml").read_text())
    assert system["parameters"]["Csh"]["value"] == 30.0
    assert (directory / "base/control.toml").read_bytes() == path.read_bytes()
    assert (directory / "base/system.toml").read_bytes() == (tmp_path / "system.toml").read_bytes()
    generated = (directory / "problem.toml").read_bytes()
    assert generated == varied.problem_bytes
    assert load_control(directory / "control.toml").problem_bytes == generated
    seeded = load_control(path, values={"pulse.seed": 2**53 + 1})
    assert seeded.problem_spec["pulse"]["seed"] == 2**53 + 1
    for values, message in (
        ({"solver.options.max_iter": 10}, "control varies only parameters"),
        ({"parameters.EJ": 14.0}, "must name a declared scalar: parameters.EJ"),
        ({"parameters.Csh": float("nan")}, "must be finite: parameters.Csh"),
        ({"pulse.seed": 1.5}, "non-negative integers"),
        ({"pulse.seed": -1}, "non-negative integers"),
        ({"pulse.seed": 2**63}, "pulse.seed"),
    ):
        with pytest.raises(ValueError, match=message):
            load_control(path, values=values)
    del definition["pulse"]["seed"]
    definition["parameters"] = {"Cx": 1.0}
    path.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match="declared value: pulse.seed"):
        load_control(path, values={"pulse.seed": 1})
    with pytest.raises(ValueError, match="declared parameter: Cx"):
        load_control(path)


def test_composite_local_factors_and_leaf_inputs_are_retained(control_files, tmp_path):
    path, definition = control_files
    for name in ("system.toml", "qubit.toml"):
        shutil.copyfile(ROOT / "examples/qubit-resonator" / name, tmp_path / name)
    qubit = load_system(tmp_path / "qubit.toml")
    frequency = float((qubit.hamiltonian_ghz[1, 1] - qubit.hamiltonian_ghz[0, 0]).real)
    definition["frame"] = {"frequencies_ghz": {"q": frequency, "r": frequency}, "rwa": True}
    for channel in definition["channels"]:
        channel.update(drive="charge_drive", frequency_ghz=frequency)
    path.write_text(tomli_w.dumps(definition))
    loaded = load_control(path)
    assert loaded.system.dimensions == (6, 5)
    assert loaded.listing["omitted_dissipators"] == ["resonator_loss"]
    assert "qubit.toml" in loaded.inputs
    lowering = np.diag(np.diag(qubit.operators["charge"], 1), 1)
    np.testing.assert_allclose(
        loaded.system.drives["x"],
        np.kron((lowering + lowering.conj().T) / 2, np.eye(5)),
        atol=1e-13,
    )
    assert all(len(term.factors) == 2 for term in loaded.system.hamiltonian_terms)


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        (None, "system", "../system.toml", "beside"),
        ("goal", "subspace", [[1, 3]], "exceeds"),
        ("goal", "subspace", [[1, 1]], "distinct"),
        ("goal", "gate", "unknown", "goal.gate"),
        ("goal", "kind", "ket", "goal.kind"),
        ("pulse", "T", -1.0, "pulse.T"),
        ("pulse", "kind", "cubic_spline", "pulse.kind"),
        ("problem", "N", 1, "problem.N"),
        ("problem", "N", True, "problem.N"),
        ("problem", "du_bound", -0.1, "problem.du_bound"),
        ("problem", "ddu_bound", float("inf"), "problem.ddu_bound"),
        ("problem", "final_fidelity", 1.1, "problem.final_fidelity"),
        ("problem", "free_phase", True, "problem.free_phase"),
        ("problem", "free_dt", True, "problem.free_dt"),
        ("problem", "options", {"timesteps_all_equal": True}, "problem.options"),
        ("problem", "options", {"tol": 1e-8}, "problem.options"),
        ("problem", "template", "BangBangPulseProblem", "problem.template"),
        ("problem", "R_du", [0.1, 0.1], "problem.R_du"),
        ("frame", "rwa", False, "frame.rwa"),
        ("frame", "frequencies_ghz", {"unknown": 0.2}, "must name subsystems"),
    ],
)
def test_invalid_control_specs_fail_before_compilation(control_files, section, key, value, message):
    path, definition = control_files
    (definition if section is None else definition[section])[key] = value
    path.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match=message):
        load_control(path)


def test_invalid_channels_and_matrix_goals_have_diagnostics(control_files):
    path, definition = control_files
    definition["channels"][1]["id"] = "x"
    path.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match="unique"):
        load_control(path)
    definition["channels"][1]["id"] = "y"
    definition["channels"][0]["drive"] = "missing"
    path.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match="unknown drive.*available"):
        load_control(path)
    definition["channels"][0]["drive"] = "x_drive"
    definition["goal"] = {"kind": "unitary", "matrix": [[1, 1], [0, 1]], "subspace": [[1, 2]]}
    path.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match="unitary"):
        load_control(path)
    definition["goal"]["matrix"] = [[0, [0, -1]], [[0, 1], 0]]
    path.write_text(tomli_w.dumps(definition))
    loaded = load_control(path)
    np.testing.assert_array_equal(
        decode(loaded.problem_spec["goal"]["matrix"]), [[0, -1j], [1j, 0]]
    )
    Draft7Validator(json.loads(SCHEMA.read_text())).validate(loaded.problem_spec)


def test_carrier_mismatch_rejects_a_zero_projected_channel(control_files):
    path, definition = control_files
    definition["channels"][0]["frequency_ghz"] += 1e-9
    path.write_text(tomli_w.dumps(definition))
    with pytest.raises(ValueError, match="channel 'x': exported drive is zero.*carrier frequency"):
        load_control(path)


def test_changed_system_input_is_refused_before_freezing(control_files, tmp_path, monkeypatch):
    from entanglix.protocols import control

    path, _ = control_files
    real_load = control.load_control

    def load_then_mutate(protocol, **kwargs):
        loaded = real_load(protocol, **kwargs)
        (path.parent / "operators.npz").write_bytes(b"changed after loading")
        return loaded

    monkeypatch.setattr(control, "load_control", load_then_mutate)
    target = tmp_path / "compiled"
    target.mkdir()
    with pytest.raises(ValueError, match="changed during compilation: operators.npz"):
        compile_control(path, target)
    assert list(target.iterdir()) == []
