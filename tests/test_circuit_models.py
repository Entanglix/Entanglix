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


def test_harmonic_cutoff_keeps_retained_levels_below_spurious_state(tmp_path):
    expected_ghz = 1 / (2 * pi * np.sqrt(80e-15 * 12e-9)) / 1e9
    with pytest.raises(SpecError, match="theta1 = 30 gives at most 15 usable levels"):
        load_system(_variant(tmp_path, "lc-circuit", "retained_levels = 6", "retained_levels = 16"))
    # At N = 2 × retained_levels, the spurious index N/2 is just outside the spectrum.
    boundary = load_system(
        _variant(
            tmp_path,
            "lc-circuit",
            "theta1 = 30 }\nretained_levels = 6",
            "theta1 = 32 }\nretained_levels = 16",
        )
    )
    np.testing.assert_allclose(
        np.diff(np.diag(boundary.hamiltonian_ghz).real), expected_ghz, rtol=1e-10
    )


def test_lc_operators_follow_node_orientation():
    model = load_system(ROOT / "lc-circuit/system.toml")
    ec = e**2 / (2 * h * 80e-15 * 1e9)
    el = (h / (2 * e * 2 * pi)) ** 2 / (12e-9 * h * 1e9)
    circuit = scq.Circuit.from_yaml_string(
        f"branches:\n- [C, 0, 1, {ec:.17g}]\n- [L, 0, 1, {el:.17g}]\n",
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
    circuit = scq.Circuit.from_yaml_string(
        "branches:\n"
        "- [JJ, 0, 1, 15, 2]\n"
        f"- [C, 0, 1, {ec1:.17g}]\n"
        "- [JJ, 0, 2, 18, 2]\n"
        f"- [C, 0, 2, {ec2:.17g}]\n"
        f"- [C, 1, 2, {ecc:.17g}]\n",
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
    # Cutoff 5 gives 11 charge states, too few for the example's 10 group levels.
    for cutoff in (6, 8, 10):
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


def _two_mode_reference(transformation, cutoff, offsets):
    ec1 = e**2 / (2 * h * 60e-15 * 1e9)
    ec2 = e**2 / (2 * h * 70e-15 * 1e9)
    ecc = e**2 / (2 * h * 2e-15 * 1e9)
    circuit = scq.Circuit.from_yaml_string(
        "branches:\n- [JJ, 0, 1, 15, 2]\n"
        f"- [C, 0, 1, {ec1:.17g}]\n"
        "- [JJ, 0, 2, 18, 2]\n"
        f"- [C, 0, 2, {ec2:.17g}]\n"
        f"- [C, 1, 2, {ecc:.17g}]\n",
        truncated_dim=6,
    )
    circuit.configure(transformation_matrix=np.array(transformation))
    circuit.cutoff_n_1 = cutoff
    circuit.cutoff_n_2 = cutoff
    circuit.ng1, circuit.ng2 = offsets
    return circuit


def _shear(tmp_path, cutoff, levels):
    source = _variant(tmp_path, "two-mode-circuit", "[[1, 0], [0, 1]]", "[[1, 1], [1, 0]]")
    source.write_text(
        source.read_text()
        .replace(
            'offset_charges = { "1" = 0.0, "2" = 0.0 }',
            'offset_charges = { "1" = 0.2, "2" = -0.1 }',
        )
        .replace("theta1 = 10, theta2 = 10", f"theta1 = {cutoff}, theta2 = {cutoff}")
        .replace("hierarchy_levels = [10, 10]", f"hierarchy_levels = [{levels}, {levels}]")
    )
    return load_system(source)


def test_nontrivial_unimodular_transform_maps_node_charges(tmp_path):
    # The shear couples the two groups strongly, so they keep nearly their whole bases.
    model = _shear(tmp_path, 15, 29)
    np.testing.assert_allclose(model.provenance["coordinate_offsets"], [0.1, 0.2])
    circuit = _two_mode_reference([[1, 1], [1, 0]], 15, (-0.1, -0.2))
    energies, vectors = circuit.eigensys(evals_count=6)
    np.testing.assert_allclose(np.diag(model.hamiltonian_ghz), energies, rtol=0, atol=1e-8)
    q1 = circuit.n2_operator() - 0.2 * np.eye(961)
    q2 = circuit.n1_operator() - circuit.n2_operator() + 0.1 * np.eye(961)
    for name, raw in (("node1.charge", q1), ("node2.charge", q2)):
        expected = vectors.conj().T @ raw @ vectors
        np.testing.assert_allclose(
            np.diag(model.operators[name]), np.diag(expected), rtol=0, atol=1e-8
        )


def test_group_levels_converge_more_slowly_under_a_shear(tmp_path):
    def error(model, transformation, offsets):
        energies = _two_mode_reference(transformation, 10, offsets).eigenvals(evals_count=6)
        return np.max(np.abs(np.diag(model.hamiltonian_ghz).real - energies))

    identity = load_system(ROOT / "two-mode-circuit/system.toml")
    assert error(identity, [[1, 0], [0, 1]], (0.0, 0.0)) < 1e-9
    # The same partition and group levels are far from converged under the shear.
    errors = [
        error(_shear(tmp_path, 10, levels), [[1, 1], [1, 0]], (-0.1, -0.2)) for levels in (10, 19)
    ]
    assert errors[0] > 1e-2 and errors[1] < 1e-5


def test_flux_bias_survives_the_hierarchy(tmp_path):
    source = tmp_path / "system.toml"
    source.write_text(
        """format_version = 2
id = "junction_loop"

[units]
energy = "GHz"
time = "ns"

[model]
kind = "circuit"
nodes = [0, 1, 2]
ground = 0
transformation = [[1, 0], [0, 1]]
offset_charges = { "1" = 0.1, "2" = 0.0 }

[[model.branches]]
id = "left"
kind = "JJ"
nodes = [0, 1]
josephson_energy = { value = 12.0, unit = "GHz" }
charging_energy = { value = 1.0, unit = "GHz" }

[[model.branches]]
id = "right"
kind = "JJ"
nodes = [0, 2]
josephson_energy = { value = 14.0, unit = "GHz" }
charging_energy = { value = 1.2, unit = "GHz" }

[[model.branches]]
id = "bridge"
kind = "JJ"
nodes = [1, 2]
josephson_energy = { value = 4.0, unit = "GHz" }
charging_energy = { value = 3.0, unit = "GHz" }

[[model.external_fluxes]]
branch = "bridge"
value = 0.3
unit = "Phi0"

[truncation]
cutoffs = { theta1 = 8, theta2 = 8 }
hierarchy = [[1], [2]]
hierarchy_levels = [12, 12]
retained_levels = 6
"""
    )
    model = load_system(source)
    circuit = scq.Circuit.from_yaml_string(
        "branches:\n- [JJ, 0, 1, 12, 1]\n- [JJ, 0, 2, 14, 1.2]\n- [JJ, 1, 2, 4, 3]\n",
        truncated_dim=6,
    )
    circuit.configure(
        transformation_matrix=np.eye(2, dtype=int), closure_branches=[circuit.branches[2]]
    )
    circuit.cutoff_n_1 = 8
    circuit.cutoff_n_2 = 8
    circuit.ng1 = -0.1
    unbiased = circuit.eigenvals(evals_count=6)
    circuit.Φ1 = 0.3
    energies, vectors = circuit.eigensys(evals_count=6)
    # The bias moves these levels by about 3 GHz; the hierarchy keeps it.
    assert np.max(np.abs(energies - unbiased)) > 1
    np.testing.assert_allclose(np.diag(model.hamiltonian_ghz), energies, rtol=0, atol=1e-8)
    for node, offset in ((1, 0.1), (2, 0.0)):
        expected = vectors.conj().T @ getattr(circuit, f"n{node}_operator")() @ vectors
        np.testing.assert_allclose(
            np.diag(model.operators[f"node{node}.charge"]),
            np.diag(expected) - offset,
            rtol=0,
            atol=1e-8,
        )


def test_group_of_two_coordinates_matches_native_diagonalization(tmp_path):
    import tomli_w

    def branch(kind, nodes, ec, ej=None):
        energies = {"charging_energy": {"value": ec, "unit": "GHz"}}
        if ej is not None:
            energies["josephson_energy"] = {"value": ej, "unit": "GHz"}
        return {"id": f"{kind}{nodes[0]}{nodes[1]}", "kind": kind, "nodes": nodes, **energies}

    branches = [
        branch("JJ", [0, 1], 1.0, 10.0),
        branch("JJ", [0, 2], 1.1, 12.0),
        branch("JJ", [0, 3], 0.9, 15.0),
        branch("C", [1, 2], 2.0),
        branch("C", [2, 3], 8.0),
    ]
    offsets = {"1": 0.1, "2": -0.2, "3": 0.15}
    document = {
        "format_version": 2,
        "id": "three_node",
        "units": {"energy": "GHz", "time": "ns"},
        "model": {
            "kind": "circuit",
            "nodes": [0, 1, 2, 3],
            "ground": 0,
            "transformation": np.eye(3, dtype=int).tolist(),
            "offset_charges": offsets,
            "branches": branches,
        },
        # Groups of unequal size, one of them holding two coordinates out of order.
        "truncation": {
            "cutoffs": {"theta1": 4, "theta2": 4, "theta3": 5},
            "hierarchy": [[2, 1], [3]],
            "hierarchy_levels": [60, 9],
            "retained_levels": 6,
        },
    }
    (tmp_path / "system.toml").write_text(tomli_w.dumps(document))
    model = load_system(tmp_path / "system.toml")
    circuit = scq.Circuit.from_yaml_string(
        "branches:\n- [JJ, 0, 1, 10, 1]\n- [JJ, 0, 2, 12, 1.1]\n- [JJ, 0, 3, 15, 0.9]\n"
        "- [C, 1, 2, 2]\n- [C, 2, 3, 8]\n",
        truncated_dim=6,
    )
    circuit.configure(transformation_matrix=np.eye(3, dtype=int))
    circuit.cutoff_n_1, circuit.cutoff_n_2, circuit.cutoff_n_3 = 4, 4, 5
    circuit.ng1, circuit.ng2, circuit.ng3 = -0.1, 0.2, -0.15
    energies, vectors = circuit.eigensys(evals_count=6)
    np.testing.assert_allclose(np.diag(model.hamiltonian_ghz), energies, rtol=0, atol=1e-8)
    # The levels are nondegenerate, so magnitudes do not depend on eigenvector phases.
    for node, offset in offsets.items():
        raw = getattr(circuit, f"n{node}_operator")()
        expected = vectors.conj().T @ raw @ vectors - offset * np.eye(6)
        np.testing.assert_allclose(
            np.abs(model.operators[f"node{node}.charge"]), np.abs(expected), rtol=0, atol=1e-8
        )


def test_multinode_circuits_need_version_2_and_a_hierarchy(tmp_path, capsys):
    version_1 = _variant(tmp_path, "two-mode-circuit", "format_version = 2", "format_version = 1")
    version_1.write_text(
        version_1.read_text().replace("hierarchy = [[1], [2]]\nhierarchy_levels = [10, 10]\n", "")
    )
    assert main(["validate", str(version_1)]) == 64
    captured = capsys.readouterr()
    assert "need format_version = 2 with truncation.hierarchy" in captured.err
    assert "partition to adopt: hierarchy = [[1], [2]]" in captured.err
    version_2 = _variant(tmp_path, "two-mode-circuit", "hierarchy = [[1], [2]]\n", "")
    version_2.write_text(version_2.read_text().replace("hierarchy_levels = [10, 10]\n", ""))
    with pytest.raises(SpecError, match="multi-node circuits require truncation.hierarchy"):
        load_system(version_2)
    with pytest.raises(SpecError, match="requires format_version = 2"):
        load_system(
            _variant(tmp_path, "two-mode-circuit", "format_version = 2", "format_version = 1")
        )
    assert main(["validate", str(ROOT / "two-mode-circuit/system.toml")]) == 0
    assert "circuit hierarchy: [[1], [2]], keeping [10, 10] eigenstates per group" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("hierarchy = [[1], [2]]", "hierarchy = [[1, 2]]", "at least two nonempty groups"),
        ("hierarchy = [[1], [2]]", "hierarchy = [[1], []]", "at least two nonempty groups"),
        ("hierarchy = [[1], [2]]", "hierarchy = [[1], [1, 2]]", "each coordinate in one"),
        ("hierarchy = [[1], [2]]", "hierarchy = [[1], [3]]", "each coordinate in one"),
        ("hierarchy_levels = [10, 10]", "hierarchy_levels = [10]", "one positive count per group"),
        (
            "hierarchy_levels = [10, 10]",
            "hierarchy_levels = [10, 0]",
            "one positive count per group",
        ),
        ("hierarchy_levels = [10, 10]\n", "", "declared together"),
        ("hierarchy_levels = [10, 10]", "hierarchy_levels = [20, 10]", "exceeds 19, the most kept"),
        ("retained_levels = 6", "retained_levels = 99", "exceeds 98, the most kept"),
    ],
)
def test_hierarchy_refusals(tmp_path, old, new, message):
    with pytest.raises(SpecError, match=message):
        load_system(_variant(tmp_path, "two-mode-circuit", old, new))


def test_eigensolver_keeps_two_fewer_states_than_the_groups_product(tmp_path):
    source = _variant(tmp_path, "two-mode-circuit", "[10, 10]", "[3, 3]")
    assert load_system(source).dimensions == (6,)
    source.write_text(source.read_text().replace("retained_levels = 6", "retained_levels = 7"))
    assert load_system(source).dimensions == (7,)
    source.write_text(source.read_text().replace("retained_levels = 7", "retained_levels = 8"))
    with pytest.raises(SpecError, match="retained_levels = 8 exceeds 7"):
        load_system(source)


def test_single_coordinate_circuit_takes_no_hierarchy(tmp_path):
    source = _variant(tmp_path, "lc-circuit", "format_version = 1", "format_version = 2")
    source.write_text(
        source.read_text().replace(
            "retained_levels = 6", "retained_levels = 6\nhierarchy = [[1]]\nhierarchy_levels = [6]"
        )
    )
    with pytest.raises(SpecError, match="single-coordinate circuit takes no truncation.hierarchy"):
        load_system(source)


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


FLUX_LITERAL = 'value = 0.25\nunit = "Phi0"'


def _flux_parameter(tmp_path, value=0.25, unit="Phi0", reference='value = { parameter = "phi" }'):
    source = _variant(tmp_path, "fluxonium", FLUX_LITERAL, reference)
    source.write_text(
        source.read_text() + f'\n[parameters]\nphi = {{ value = {value!r}, unit = "{unit}" }}\n'
    )
    return source


def test_flux_parameter_constructs_the_literal_model(tmp_path, capsys):
    bound = load_system(_flux_parameter(tmp_path))
    literal = load_system(ROOT / "fluxonium/system.toml")
    np.testing.assert_array_equal(bound.hamiltonian_ghz, literal.hamiltonian_ghz)
    for name, operator in literal.operators.items():
        np.testing.assert_array_equal(bound.operators[name], operator)
    loop = bound.provenance["oriented_loops"][0]
    assert loop == {
        "branch": "inductor",
        "nodes": [0, 1],
        "value": 0.25,
        "unit": "Phi0",
        "parameter": "phi",
    }
    assert "parameter" not in literal.provenance["oriented_loops"][0]
    assert main(["validate", str(tmp_path / "system.toml")]) == 0
    assert "closure branch inductor (0→1): 0.25 Phi0 from parameter phi" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("unit", "reference", "message"),
    [
        ("GHz", 'value = { parameter = "phi" }', "external_fluxes.inductor: expected Phi0"),
        ("Phi0", 'value = { parameter = "missing" }', "unknown parameter 'missing'"),
        ("Phi0", 'value = { parameter = "phi" }\nunit = "Phi0"', "omit unit"),
        ("Phi0", "value = 0.25", 'requires unit = "Phi0"'),
    ],
)
def test_flux_parameter_refusals(tmp_path, unit, reference, message):
    with pytest.raises(SpecError, match=message):
        load_system(_flux_parameter(tmp_path, unit=unit, reference=reference))


def test_sweep_over_a_flux_parameter_matches_separately_built_models(tmp_path):
    import tomli_w

    from entanglix.protocols.sweep import compile_sweep, load_sweep

    _flux_parameter(tmp_path)
    spec = {
        "format_version": 1,
        "system": "system.toml",
        "kind": "spectrum",
        "levels": 3,
        "needs": {"device": "cpu"},
        "parameters": {"phi": 0.25},
        "refinement": {
            "fluxonium_circuit": {"retained_levels_step": 1, "cutoff_steps": {"theta1": 2}}
        },
    }
    (tmp_path / "simulate.toml").write_text(tomli_w.dumps(spec))
    values = [0.0, 0.25, 0.4]
    grid = {
        "format_version": 1,
        "base": "simulate.toml",
        "axes": [{"id": "phi", "target": "parameters.phi", "values": values}],
    }
    (tmp_path / "sweep.toml").write_text(tomli_w.dumps(grid))
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    compile_sweep(tmp_path / "sweep.toml", compiled, [], ROOT.parent)
    points = load_sweep(compiled).points
    assert len(points) == len(values)
    for value, point in zip(values, points, strict=True):
        separate = tmp_path / f"literal-{value}"
        separate.mkdir()
        literal = _variant(separate, "fluxonium", "value = 0.25", f"value = {value!r}")
        np.testing.assert_array_equal(
            point.system.hamiltonian_ghz, load_system(literal).hamiltonian_ghz
        )
        assert point.system.provenance["oriented_loops"][0]["value"] == value
