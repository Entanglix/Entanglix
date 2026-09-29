"""Physical-component references and separate basis/refinement evidence."""

from pathlib import Path

import numpy as np
import pytest
import scqubits as scq
from scipy.constants import e, h, pi

from entanglix.cli import main
from entanglix.models import SpecError, load_system

ROOT = Path(__file__).resolve().parents[1] / "examples"


def _variant(tmp_path, fixture: str, old: str, new: str) -> Path:
    source = tmp_path / "system.toml"
    source.write_text((ROOT / fixture / "system.toml").read_text().replace(old, new))
    return source


def test_lc_component_conversion_matches_resonance_formula():
    model = load_system(ROOT / "lc-circuit/system.toml")
    expected_ghz = 1 / (2 * pi * np.sqrt(80e-15 * 12e-9)) / 1e9
    energies = np.diag(model.hamiltonian_ghz).real
    np.testing.assert_allclose(np.diff(energies), expected_ghz, rtol=1e-10)
    assert model.provenance["converted_branches"]["inductor"]["inductive_energy_ghz"] > 0
    assert {"charge", "phase"} <= model.operators.keys()


def test_lc_operators_follow_node_orientation():
    model = load_system(ROOT / "lc-circuit/system.toml")
    ec = e**2 / (2 * h * 80e-15 * 1e9)
    el = (h / (2 * e * 2 * pi)) ** 2 / (12e-9 * h * 1e9)
    circuit = scq.Circuit(
        f"branches:\n- [C, 0, 1, {ec:.17g}]\n- [L, 0, 1, {el:.17g}]\n",
        from_file=False,
        truncated_dim=6,
        ext_basis="harmonic",
    )
    assert circuit.transformation_matrix[0, 0] == -1
    circuit.cutoff_ext_1 = 30
    _, vectors = circuit.eigensys(evals_count=6)
    # At a fixed basis, the independent circuit's projected operators should
    # agree to roundoff. The absolute floor also covers a zero operator.
    for name, raw in (("charge", -circuit.Q1_operator()), ("phase", -circuit.θ1_operator())):
        expected = vectors.conj().T @ raw @ vectors
        residual = np.linalg.norm(model.operators[name] - expected)
        assert residual <= max(1e-12 * np.linalg.norm(expected), 1e-12)


def test_scqubits_unit_setting_must_match_spec(monkeypatch):
    monkeypatch.setattr(scq, "get_units", lambda: "MHz")
    with pytest.raises(SpecError, match="scqubits units must be GHz"):
        load_system(ROOT / "lc-circuit/system.toml")


def test_fluxonium_circuit_spectrum_and_oriented_flux(tmp_path):
    model = load_system(ROOT / "fluxonium/system.toml")
    reference = scq.Fluxonium(EJ=5.7, EC=1.2, EL=0.4, flux=0.25, cutoff=60, truncated_dim=6)
    expected = reference.eigenvals(evals_count=6)
    # The separate circuit/constructor paths differ by 1.55e-8 GHz here.
    np.testing.assert_allclose(
        np.diag(model.hamiltonian_ghz).real - model.hamiltonian_ghz[0, 0].real,
        expected - expected[0],
        atol=2e-8,
        rtol=0,
    )
    reversed_flux = load_system(_variant(tmp_path, "fluxonium", "value = 0.25", "value = -0.25"))
    np.testing.assert_allclose(
        np.diag(model.hamiltonian_ghz), np.diag(reversed_flux.hamiltonian_ghz), rtol=0, atol=2e-8
    )
    assert model.operators["sin_phase"][0, 0].real > 0.1
    np.testing.assert_allclose(
        model.operators["sin_phase"][0, 0],
        -reversed_flux.operators["sin_phase"][0, 0],
        atol=2e-8,
    )
    assert [loop["branch"] for loop in model.provenance["oriented_loops"]] == ["inductor"]
    reversed_branch = load_system(
        _variant(
            tmp_path,
            "fluxonium",
            'id = "inductor"\nkind = "L"\nnodes = [0, 1]',
            'id = "inductor"\nkind = "L"\nnodes = [1, 0]',
        )
    )
    np.testing.assert_allclose(
        model.operators["sin_phase"][0, 0],
        -reversed_branch.operators["sin_phase"][0, 0],
        atol=2e-8,
    )


def test_extended_offset_is_dropped_and_reported_with_oriented_loop(tmp_path, capsys):
    source = _variant(
        tmp_path, "fluxonium", "offset_charges = {}", 'offset_charges = { "1" = 0.3 }'
    )
    baseline = load_system(ROOT / "fluxonium/system.toml")
    shifted = load_system(source)
    np.testing.assert_array_equal(shifted.hamiltonian_ghz, baseline.hamiltonian_ghz)
    np.testing.assert_array_equal(shifted.operators["charge"], baseline.operators["charge"])
    assert shifted.provenance["coordinate_offsets"] == [0.0]
    assert shifted.provenance["dropped_offsets"] == {"theta1": 0.3}
    assert shifted.provenance["oriented_loops"] == [
        {"branch": "inductor", "nodes": [0, 1], "value": 0.25, "unit": "Phi0"}
    ]
    assert main(["validate", str(source)]) == 0
    output = capsys.readouterr().out
    assert "derived coordinate offset charges: [0.0]" in output
    assert "dropped extended-coordinate offsets: {'theta1': 0.3}" in output
    assert "closure branch inductor (0→1): 0.25 Phi0" in output
    assert "charge: n − ng in Cooper-pair units" in output
    assert "phase: extended-coordinate phase in radians" in output


def test_fluxonium_harmonic_cutoff_refines_independently_of_retained_levels(tmp_path):
    values = []
    sin_expectations = []
    for cutoff in (40, 60, 80):
        model = load_system(_variant(tmp_path, "fluxonium", "theta1 = 60", f"theta1 = {cutoff}"))
        energies = np.diag(model.hamiltonian_ghz).real
        values.append(energies[:4] - energies[0])
        sin_expectations.append(model.operators["sin_phase"][0, 0].real)
    # The 40-state cutoff visibly fails; 60→80 changes energies and a signed
    # ground-state observable by much less than their declared fixture limits.
    assert np.max(np.abs(values[0] - values[1])) > 1e-5
    assert np.max(np.abs(values[1] - values[2])) <= 1e-5
    assert abs(sin_expectations[0] - sin_expectations[1]) > 1e-6
    assert abs(sin_expectations[1] - sin_expectations[2]) <= 1e-6
    larger = load_system(
        _variant(tmp_path, "fluxonium", "retained_levels = 6", "retained_levels = 8")
    )
    original = load_system(ROOT / "fluxonium/system.toml")
    np.testing.assert_allclose(
        np.diag(original.hamiltonian_ghz), np.diag(larger.hamiltonian_ghz)[:6], rtol=0, atol=2e-8
    )
    assert larger.dimensions == (8,) and original.dimensions == (6,)


def test_two_mode_component_circuit_and_node_charge_reference(tmp_path):
    source = _variant(
        tmp_path,
        "two-mode-circuit",
        'offset_charges = { "1" = 0.0, "2" = 0.0 }',
        'offset_charges = { "1" = 0.2, "2" = -0.1 }',
    )
    model = load_system(source)
    ec1 = e**2 / (2 * h * 60e-15 * 1e9)
    ec2 = e**2 / (2 * h * 70e-15 * 1e9)
    ecc = e**2 / (2 * h * 2e-15 * 1e9)
    circuit = scq.Circuit(
        "branches:\n"
        "- [JJ, 0, 1, 15, 2]\n"
        f"- [C, 0, 1, {ec1:.17g}]\n"
        "- [JJ, 0, 2, 18, 2]\n"
        f"- [C, 0, 2, {ec2:.17g}]\n"
        f"- [C, 1, 2, {ecc:.17g}]\n",
        from_file=False,
        truncated_dim=6,
    )
    circuit.configure(transformation_matrix=np.eye(2, dtype=int))
    circuit.cutoff_n_1 = 10
    circuit.cutoff_n_2 = 10
    circuit.ng1 = -0.2
    circuit.ng2 = 0.1
    energies, vectors = circuit.eigensys(evals_count=6)
    np.testing.assert_allclose(np.diag(model.hamiltonian_ghz), energies, rtol=0, atol=1e-8)
    for node, offset in ((1, 0.2), (2, -0.1)):
        expected = vectors.conj().T @ getattr(circuit, f"n{node}_operator")() @ vectors
        expected -= offset * np.eye(6)
        np.testing.assert_allclose(
            np.diag(model.operators[f"node{node}.charge"]), np.diag(expected), rtol=0, atol=1e-8
        )
    assert model.provenance["transformation"] == [[1.0, 0.0], [0.0, 1.0]]
    assert model.provenance["transformation_source"] == "authored"


def test_two_mode_cutoff_refinement(tmp_path):
    values = []
    for cutoff in (5, 8, 10):
        model = load_system(
            _variant(
                tmp_path,
                "two-mode-circuit",
                "theta1 = 10, theta2 = 10",
                f"theta1 = {cutoff}, theta2 = {cutoff}",
            )
        )
        energies = np.diag(model.hamiltonian_ghz).real
        values.append(energies[:4] - energies[0])
    assert np.max(np.abs(values[0] - values[1])) > 1e-5
    assert np.max(np.abs(values[1] - values[2])) <= 1e-5


def test_nontrivial_unimodular_transform_maps_node_charges(tmp_path):
    source = _variant(tmp_path, "two-mode-circuit", "[[1, 0], [0, 1]]", "[[1, 1], [1, 0]]")
    source.write_text(
        source.read_text().replace(
            'offset_charges = { "1" = 0.0, "2" = 0.0 }',
            'offset_charges = { "1" = 0.2, "2" = -0.1 }',
        )
    )
    model = load_system(source)
    np.testing.assert_allclose(model.provenance["coordinate_offsets"], [0.1, 0.2])
    ec1 = e**2 / (2 * h * 60e-15 * 1e9)
    ec2 = e**2 / (2 * h * 70e-15 * 1e9)
    ecc = e**2 / (2 * h * 2e-15 * 1e9)
    circuit = scq.Circuit(
        "branches:\n- [JJ, 0, 1, 15, 2]\n"
        f"- [C, 0, 1, {ec1:.17g}]\n"
        "- [JJ, 0, 2, 18, 2]\n"
        f"- [C, 0, 2, {ec2:.17g}]\n"
        f"- [C, 1, 2, {ecc:.17g}]\n",
        from_file=False,
        truncated_dim=6,
    )
    circuit.configure(transformation_matrix=np.array([[1, 1], [1, 0]]))
    circuit.cutoff_n_1 = 10
    circuit.cutoff_n_2 = 10
    circuit.ng1 = -0.1
    circuit.ng2 = -0.2
    energies, vectors = circuit.eigensys(evals_count=6)
    np.testing.assert_allclose(np.diag(model.hamiltonian_ghz), energies, rtol=0, atol=1e-8)
    q1 = circuit.n2_operator() - 0.2 * np.eye(441)
    q2 = circuit.n1_operator() - circuit.n2_operator() + 0.1 * np.eye(441)
    for name, raw in (("node1.charge", q1), ("node2.charge", q2)):
        expected = vectors.conj().T @ raw @ vectors
        np.testing.assert_allclose(
            np.diag(model.operators[name]), np.diag(expected), rtol=0, atol=1e-8
        )


@pytest.mark.parametrize(
    ("fixture", "old", "new", "message"),
    [
        ("fluxonium", 'branch = "inductor"', 'branch = "missing"', "unknown closure"),
        (
            "fluxonium",
            '[[model.external_fluxes]]\nbranch = "inductor"\nvalue = 0.25\nunit = "Phi0"\n\n',
            "",
            "external_fluxes",
        ),
        ("two-mode-circuit", "[[1, 0], [0, 1]]", "[[2, 0], [0, 1]]", "unimodular"),
        ("two-mode-circuit", '"2" = 0.0', '"2" = "zero"', "offset_charges"),
    ],
)
def test_invalid_circuit_conventions_have_diagnostics(tmp_path, fixture, old, new, message):
    with pytest.raises(SpecError, match=message):
        load_system(_variant(tmp_path, fixture, old, new))


def test_missing_multinode_transformation_shows_default_to_author(tmp_path, capsys):
    source = _variant(tmp_path, "two-mode-circuit", "transformation = [[1, 0], [0, 1]]\n", "")
    assert main(["validate", str(source)]) == 64
    captured = capsys.readouterr()
    assert "scqubits default to adopt: [[" in captured.err
    assert captured.out == ""
