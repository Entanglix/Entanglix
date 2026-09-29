"""Independent CPU references for the first two authored system paths."""

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scqubits as scq
from scipy.constants import e, h

from entanglix.engines import EvolutionInputError, evolve_cpu
from entanglix.models import SpecError, SystemSpec, load_system

ROOT = Path(__file__).resolve().parents[1]
TWO_LEVEL = ROOT / "examples/two-level/system.toml"
JUNCTION = ROOT / "examples/junction-shunt/system.toml"


def test_explicit_model_decay_and_closed_coherence():
    model = load_system(TWO_LEVEL)
    np.testing.assert_allclose(model.hamiltonian_ghz, np.diag([0, 0.2]))
    np.testing.assert_allclose(model.qutip_hamiltonian().full(), model.hamiltonian_ghz)
    np.testing.assert_allclose(model.operators["lower"], [[0, 1], [0, 0]])
    assert model.basis == "ground_excited"
    assert len(model.provenance["asset_sha256"]) == 64

    excited = np.array([0, 1], dtype=complex)
    projector = np.outer(excited, excited.conj())
    times = np.linspace(0, 100, 11)
    result = evolve_cpu(model, excited, times, observables={"excited": projector})
    np.testing.assert_allclose(result.expectations["excited"], np.exp(-0.01 * times), atol=2e-8)
    assert np.isrealobj(result.expectations["excited"])
    assert all(abs(np.trace(state) - 1) < 1e-9 for state in result.states)

    closed = replace(model, dissipators={})
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    closed_result = evolve_cpu(closed, plus, np.linspace(0, 2, 9))
    expected = 0.5 * np.exp(1j * 2 * np.pi * 0.2 * closed_result.times_ns)
    np.testing.assert_allclose([state[0, 1] for state in closed_result.states], expected, atol=2e-8)


def test_junction_shunt_matches_transmon_at_nonzero_offset():
    model = load_system(JUNCTION)
    c_junction = e**2 / (2 * h * 2.0e9)
    e_total = e**2 / (2 * h * (c_junction + 60e-15) * 1e9)
    reference = scq.Transmon(EJ=15.0, EC=e_total, ng=0.23, ncut=30, truncated_dim=6)
    evals, evecs = reference.eigensys(evals_count=6)
    np.testing.assert_allclose(np.diag(model.hamiltonian_ghz), evals, atol=1e-8)
    assert model.dimensions == (6,) and model.basis == "isolated_eigenstates"
    # Eigenvector signs may differ; diagonal charge expectations do not.
    expected_charge = evecs.conj().T @ reference.n_operator() @ evecs - 0.23 * np.eye(6)
    np.testing.assert_allclose(
        np.diag(model.operators["charge"]), np.diag(expected_charge), atol=1e-8
    )
    expected_cos = evecs.conj().T @ reference.cos_phi_operator() @ evecs
    np.testing.assert_allclose(
        np.diag(model.operators["cos_phase"]), np.diag(expected_cos), atol=1e-8
    )
    np.testing.assert_allclose(
        np.abs(model.operators["cos_phase"]), np.abs(expected_cos), atol=1e-8
    )
    expected_sin = evecs.conj().T @ reference.sin_phi_operator() @ evecs
    assert np.linalg.norm(expected_sin) > 0.1
    np.testing.assert_allclose(
        np.abs(model.operators["sin_phase"]), np.abs(expected_sin), atol=1e-8
    )
    np.testing.assert_allclose(
        model.operators["sin_phase"], model.operators["sin_phase"].conj().T, atol=1e-8
    )
    assert model.provenance["converted_branches"]["shunt"]["charging_energy_ghz"] > 0


def _copy_two_level(tmp_path: Path) -> Path:
    shutil.copyfile(TWO_LEVEL, tmp_path / "system.toml")
    shutil.copyfile(TWO_LEVEL.parent / "operators.npz", tmp_path / "operators.npz")
    return tmp_path / "system.toml"


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("dimensions = [2]", "dimensions = [3]", "shape"),
        ("dimensions = [2]", 'dimensions = ["2"]', "dimensions"),
        ('unit = "1/ns"', 'unit = "GHz"', "expected 1/ns"),
        ("value = 0.01", "value = -0.01", "nonnegative"),
        ('kind = "operators"', 'kind = "composite"', "model"),
    ],
)
def test_invalid_spec_has_actionable_diagnostic(tmp_path, original, replacement, message):
    source = _copy_two_level(tmp_path)
    source.write_text(source.read_text().replace(original, replacement))
    with pytest.raises(SpecError, match=message):
        load_system(source)


def test_numeric_archive_rejects_object_arrays(tmp_path):
    source = _copy_two_level(tmp_path)
    np.savez(
        tmp_path / "operators.npz",
        H0=np.array([[1, 2], [3]], dtype=object),
        sigma_x=np.eye(2),
        lowering=np.zeros((2, 2)),
    )
    with pytest.raises(SpecError, match="numeric NPZ"):
        load_system(source)


@pytest.mark.parametrize("asset_name", ["../operators.npz", "/tmp/operators.npz"])
def test_numeric_archive_must_be_beside_spec(tmp_path, asset_name):
    source = _copy_two_level(tmp_path)
    source.write_text(
        source.read_text().replace('array_file = "operators.npz"', f'array_file = "{asset_name}"')
    )
    with pytest.raises(SpecError, match="beside the system spec"):
        load_system(source)


def test_numeric_archive_symlink_outside_spec_is_reported(tmp_path):
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    source = _copy_two_level(spec_dir)
    outside = tmp_path / "outside.npz"
    (spec_dir / "operators.npz").rename(outside)
    (spec_dir / "operators.npz").symlink_to(outside)
    with pytest.raises(SpecError, match="outside the spec's directory"):
        load_system(source)


def test_evolution_rejects_nonuniform_grid():
    model = load_system(TWO_LEVEL)
    with pytest.raises(EvolutionInputError, match="uniform grid"):
        evolve_cpu(model, np.array([0, 1]), np.array([0.0, 1.0, 3.0]))


def test_invalid_text_and_missing_archive_key_are_reported(tmp_path):
    source = tmp_path / "system.toml"
    source.write_bytes(b"\xff")
    with pytest.raises(SpecError, match="UTF-8"):
        load_system(source)
    source = _copy_two_level(tmp_path)
    np.savez(tmp_path / "operators.npz", H0=np.diag([0.0, 0.2]))
    with pytest.raises(SpecError, match="missing NPZ key 'sigma_x'"):
        load_system(source)


def test_insufficient_circuit_cutoff_is_reported(tmp_path):
    source = tmp_path / "system.toml"
    source.write_text(JUNCTION.read_text().replace("theta1 = 30", "theta1 = 1"))
    with pytest.raises(SpecError, match="retainable levels"):
        load_system(source)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            'capacitance = { parameter = "Csh" }\ncharging_energy = { value = 0.3, unit = "GHz" }',
            "exactly one",
        ),
        ('capacitance = { value = 60.0, unit = "GHz" }', "expected fF"),
    ],
)
def test_component_quantity_mistakes_are_reported(tmp_path, replacement, message):
    source = tmp_path / "system.toml"
    source.write_text(
        JUNCTION.read_text().replace('capacitance = { parameter = "Csh" }', replacement)
    )
    with pytest.raises(SpecError, match=message):
        load_system(source)


def test_cli_and_generated_schema(tmp_path):
    valid = subprocess.run(
        [sys.executable, "-m", "entanglix", "validate", str(TWO_LEVEL)],
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0 and "lower" in valid.stdout
    invalid_source = _copy_two_level(tmp_path)
    invalid_source.write_text(invalid_source.read_text().replace('unit = "1/ns"', 'unit = "GHz"'))
    invalid = subprocess.run(
        [sys.executable, "-m", "entanglix", "validate", str(invalid_source)],
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 64 and "expected 1/ns" in invalid.stderr
    assert invalid.stdout == ""
    committed_schema = json.loads((ROOT / "schemas/system.schema.json").read_text())
    assert committed_schema == SystemSpec.model_json_schema()
