"""Independent CPU references for coupled systems and constant drives."""

from dataclasses import replace
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
import scqubits as scq
from scipy.constants import e, h

from entanglix.engines import EvolutionInputError, evolve_cpu
from entanglix.models import SpecError, load_system

ROOT = Path(__file__).resolve().parents[1]
QUBIT_RESONATOR = ROOT / "examples/qubit-resonator/system.toml"
THREE = ROOT / "examples/three-subsystem/system.toml"
TWO_LEVEL = ROOT / "examples/two-level/system.toml"


def test_qubit_resonator_agrees_with_independent_transmon_reference():
    model = load_system(QUBIT_RESONATOR)
    c_junction = e**2 / (2 * h * 2.0e9)
    ec = e**2 / (2 * h * (c_junction + 60e-15) * 1e9)
    qubit = scq.Transmon(EJ=15, EC=ec, ng=0.23, ncut=30, truncated_dim=6)
    energies, vectors = qubit.eigensys(evals_count=6)
    charge = vectors.conj().T @ qubit.n_operator() @ vectors - 0.23 * np.eye(6)
    resonator = scq.Oscillator(E_osc=6.2, truncated_dim=5)
    lower = resonator.annihilation_operator()
    position = lower + lower.conj().T
    reference = (
        np.kron(np.diag(energies), np.eye(5))
        + np.kron(np.eye(6), np.diag(resonator.eigenvals(evals_count=5)))
        + 0.05 * np.kron(charge, position)
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(model.hamiltonian_ghz), np.linalg.eigvalsh(reference), rtol=0, atol=2e-8
    )
    np.testing.assert_allclose(model.operators["r.position"], np.kron(np.eye(6), position))
    assert model.dimensions == (6, 5) and model.subsystem_ids == ("q", "r")
    assert model.qutip_hamiltonian().dims == [[6, 5], [6, 5]]
    assert len(model.hamiltonian_terms) == 3
    assert all(len(term.factors) == 2 for term in model.hamiltonian_terms)
    np.testing.assert_allclose(
        sum(
            (term.coefficient_ghz * np.kron(*term.factors) for term in model.hamiltonian_terms),
            np.zeros((30, 30)),
        ),
        model.hamiltonian_ghz,
    )
    assert set(model.drive_factors) == set(model.drives) == {"charge_drive"}
    assert set(model.dissipator_factors) == set(model.dissipators) == {"resonator_loss"}
    for name, factors in model.drive_factors.items():
        assert tuple(factor.shape for factor in factors) == ((6, 6), (5, 5))
        np.testing.assert_allclose(np.kron(*factors), model.drives[name], rtol=0, atol=1e-12)
    for name, factors in model.dissipator_factors.items():
        assert tuple(factor.shape for factor in factors) == ((6, 6), (5, 5))
        np.testing.assert_allclose(
            np.kron(*factors), model.dissipators[name][0], rtol=0, atol=1e-12
        )
    np.testing.assert_allclose(model.drive_factors["charge_drive"][1], np.eye(5))
    np.testing.assert_allclose(model.dissipator_factors["resonator_loss"][0], np.eye(6))
    np.testing.assert_allclose(
        np.diag(model.drive_factors["charge_drive"][0]), np.diag(charge), atol=1e-10
    )
    np.testing.assert_allclose(model.dissipator_factors["resonator_loss"][1], lower)
    assert model.provenance["subsystems"]["q"]["parameters"]["Csh"]["prior"]["kind"] == "uniform"


def test_qubit_resonator_refines_cutoff_and_retained_levels_separately(tmp_path):
    qubit_text = (QUBIT_RESONATOR.parent / "qubit.toml").read_text()
    composite_text = QUBIT_RESONATOR.read_text()

    def result(cutoff: int, qubit_levels: int, resonator_levels: int):
        (tmp_path / "qubit.toml").write_text(
            qubit_text.replace("theta1 = 30", f"theta1 = {cutoff}").replace(
                "retained_levels = 6", f"retained_levels = {qubit_levels}"
            )
        )
        (tmp_path / "system.toml").write_text(
            composite_text.replace("retained_levels = 5", f"retained_levels = {resonator_levels}")
        )
        model = load_system(tmp_path / "system.toml")
        energies, vectors = np.linalg.eigh(model.hamiltonian_ghz)
        occupation = np.vdot(vectors[:, 0], model.operators["r.number"] @ vectors[:, 0]).real
        return energies[:4] - energies[0], occupation

    baseline = result(30, 6, 5)
    for refined in (result(40, 6, 5), result(30, 8, 5), result(30, 6, 7)):
        np.testing.assert_allclose(refined[0], baseline[0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(refined[1], baseline[1], rtol=0, atol=1e-7)


def test_resonator_loss_acts_on_populated_state():
    model = load_system(QUBIT_RESONATOR)
    uncoupled = sum(
        (np.kron(*term.factors) for term in model.hamiltonian_terms[:2]), np.zeros((30, 30))
    )
    model = replace(model, hamiltonian_ghz=uncoupled)
    initial = np.zeros(30)
    initial[1] = 1  # q ground, one resonator excitation; rightmost index varies fastest.
    one_photon = np.kron(np.eye(6), np.diag([0, 1, 0, 0, 0]))
    times = np.linspace(0, 10, 6)
    result = evolve_cpu(model, initial, times, observables={"one": one_photon})
    np.testing.assert_allclose(result.expectations["one"].real, np.exp(-0.01 * times), atol=2e-7)
    lossless = evolve_cpu(
        replace(model, dissipators={}), initial, times, observables={"one": one_photon}
    )
    np.testing.assert_allclose(lossless.expectations["one"].real, np.ones(len(times)), atol=2e-8)


def test_nonuniform_tensor_dimensions_and_order():
    model = load_system(THREE)
    assert model.dimensions == (3, 4, 5)
    assert model.subsystem_ids == ("q1", "q2", "r")
    lower = scq.Oscillator(E_osc=6.2, truncated_dim=5).annihilation_operator()
    np.testing.assert_allclose(
        model.operators["r.position"], np.kron(np.eye(12), lower + lower.conj().T)
    )
    assert len(model.hamiltonian_terms) == 5
    assert all(
        tuple(f.shape[0] for f in term.factors) == (3, 4, 5) for term in model.hamiltonian_terms
    )


def test_fluxonium_constructor_matches_direct_class(tmp_path):
    source = tmp_path / "system.toml"
    source.write_text("""format_version = 1
id = "flux_and_mode"
[units]
energy = "GHz"
time = "ns"
[model]
kind = "composite"
[[model.subsystems]]
id = "f"
kind = "fluxonium"
josephson_energy = { value = 5.7, unit = "GHz" }
charging_energy = { value = 1.2, unit = "GHz" }
inductive_energy = { value = 0.4, unit = "GHz" }
flux = { value = 0.25, unit = "Phi0" }
cutoff = 60
retained_levels = 6
[[model.subsystems]]
id = "r"
kind = "oscillator"
frequency = { value = 6.2, unit = "GHz" }
retained_levels = 2
""")
    model = load_system(source)
    reference = scq.Fluxonium(EJ=5.7, EC=1.2, EL=0.4, flux=0.25, cutoff=60, truncated_dim=6)
    energies, vectors = reference.eigensys(evals_count=6)
    np.testing.assert_allclose(
        np.diag(model.hamiltonian_ghz)[::2] - np.diag(model.hamiltonian_ghz)[0],
        energies - energies[0],
        rtol=0,
        atol=2e-8,
    )
    expected_phase = vectors.conj().T @ reference.phi_operator() @ vectors
    np.testing.assert_allclose(
        np.diag(model.operators["f.phase"])[::2], np.diag(expected_phase), rtol=0, atol=2e-8
    )
    circuit_text = (ROOT / "examples/fluxonium/system.toml").read_text()
    forward_path = tmp_path / "forward.toml"
    forward_path.write_text(circuit_text)
    reversed_path = tmp_path / "reversed.toml"
    reversed_path.write_text(
        circuit_text.replace(
            'id = "inductor"\nkind = "L"\nnodes = [0, 1]',
            'id = "inductor"\nkind = "L"\nnodes = [1, 0]',
        )
    )
    forward = load_system(forward_path)
    reversed_circuit = load_system(reversed_path)
    for name in ("phase", "sin_phase"):
        constructor_diagonal = np.diag(model.operators[f"f.{name}"])[::2]
        np.testing.assert_allclose(
            constructor_diagonal, np.diag(reversed_circuit.operators[name]), rtol=0, atol=1e-7
        )
        np.testing.assert_allclose(
            constructor_diagonal, -np.diag(forward.operators[name]), rtol=0, atol=1e-7
        )


def test_transmon_constructor_keeps_conventional_ng(tmp_path):
    source = tmp_path / "system.toml"
    source.write_text(THREE.read_text().replace("ng = 0.0", "ng = 0.23"))
    model = load_system(source)
    reference = scq.Transmon(EJ=15, EC=0.25, ng=0.23, ncut=20, truncated_dim=3)
    energies, vectors = reference.eigensys(evals_count=3)
    np.testing.assert_allclose(
        np.diag(model.hamiltonian_ghz)[::20] - model.hamiltonian_ghz[0, 0],
        energies - energies[0],
        rtol=0,
        atol=1e-8,
    )
    charge = vectors.conj().T @ reference.n_operator() @ vectors - 0.23 * np.eye(3)
    np.testing.assert_allclose(
        np.diag(model.operators["q1.charge"])[::20], np.diag(charge), rtol=0, atol=1e-8
    )


def test_constant_and_piecewise_drives_use_ghz_once():
    model = load_system(TWO_LEVEL)
    model = replace(model, hamiltonian_ghz=np.zeros((2, 2)), dissipators={})
    initial = np.array([1, 0])
    excited = np.diag([0, 1])
    times = np.array([0.0, 1.0, 2.0])
    constant = evolve_cpu(
        model,
        initial,
        times,
        observables={"excited": excited},
        drive_amplitudes_ghz={"x_drive": 0.125},
    )
    np.testing.assert_allclose(constant.expectations["excited"].real, [0, 0.5, 1], atol=2e-8)
    piecewise = evolve_cpu(
        model,
        initial,
        times,
        observables={"excited": excited},
        drive_amplitudes_ghz={"x_drive": np.array([0.125, 0.0])},
    )
    np.testing.assert_allclose(piecewise.expectations["excited"].real, [0, 0.5, 0.5], atol=2e-8)
    assert piecewise.diagnostics["segments"] == 2
    with pytest.raises(EvolutionInputError, match="one value per output interval"):
        evolve_cpu(model, initial, times, drive_amplitudes_ghz={"x_drive": [0.125]})


def test_nonhermitian_interaction_requires_conjugate(tmp_path):
    (tmp_path / "qubit.toml").write_text(TWO_LEVEL.read_text().split("[[drives]]")[0])
    copyfile(TWO_LEVEL.parent / "operators.npz", tmp_path / "operators.npz")
    source = tmp_path / "system.toml"
    source.write_text("""format_version = 1
id = "exchange"
[units]
energy = "GHz"
time = "ns"
[model]
kind = "composite"
[[model.subsystems]]
id = "q"
source = "qubit.toml"
[[model.subsystems]]
id = "r"
kind = "oscillator"
frequency = { value = 6.2, unit = "GHz" }
retained_levels = 3
[[model.interactions]]
coefficient = { value = 0.05, unit = "GHz" }
factors = ["q.lower", "r.annihilation"]
add_hermitian_conjugate = true
""")
    model = load_system(source)
    lower_q = np.array([[0, 1], [0, 0]])
    lower_r = scq.Oscillator(E_osc=6.2, truncated_dim=3).annihilation_operator()
    expected = np.kron(np.diag([0, 0.2]), np.eye(3)) + np.kron(np.eye(2), np.diag([0, 6.2, 12.4]))
    exchange = np.kron(lower_q, lower_r)
    expected += 0.05 * (exchange + exchange.conj().T)
    np.testing.assert_allclose(model.hamiltonian_ghz, expected)
    assert len(model.hamiltonian_terms) == 4
    source.write_text(
        source.read_text().replace(
            "add_hermitian_conjugate = true", "add_hermitian_conjugate = false"
        )
    )
    with pytest.raises(SpecError, match="Hermitian"):
        load_system(source)


def test_bad_composite_references_and_operator_names(tmp_path):
    text = QUBIT_RESONATOR.read_text()
    (tmp_path / "qubit.toml").write_text((QUBIT_RESONATOR.parent / "qubit.toml").read_text())
    source = tmp_path / "system.toml"
    source.write_text(text.replace("q.charge", "q.missing"))
    with pytest.raises(SpecError, match="available"):
        load_system(source)
    source.write_text(text.replace('source = "qubit.toml"', 'source = "../qubit.toml"'))
    with pytest.raises(SpecError, match="beside the composite"):
        load_system(source)
    source.write_text(text.replace('id = "r"', 'id = "q"'))
    with pytest.raises(SpecError, match="unique"):
        load_system(source)
    source.write_text(text)
    (tmp_path / "qubit.toml").write_text(text)
    with pytest.raises(SpecError, match="cannot itself be a composite"):
        load_system(source)


def test_invalid_uniform_prior_is_reported(tmp_path):
    source = tmp_path / "system.toml"
    original = (ROOT / "examples/junction-shunt/system.toml").read_text()
    source.write_text(
        original.replace(
            '{ kind = "uniform", lower = 55.0, upper = 65.0 }',
            '{ kind = "uniform", lower = 60.0, upper = 55.0 }',
        )
    )
    with pytest.raises(SpecError, match="lower < upper"):
        load_system(source)
