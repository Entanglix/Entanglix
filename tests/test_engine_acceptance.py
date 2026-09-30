"""Three-engine dynamics, integration refinement, and physical-state acceptance."""

from dataclasses import replace

import numpy as np
import pytest
from engine_cases import physical_residuals

from entanglix.engine_cuquantum import CuQuantumIntegrator
from entanglix.engine_dynamiqs import DynamiqsIntegrator
from entanglix.engines import QuTiPIntegrator
from entanglix.models import load_system


@pytest.mark.parametrize(
    "case_name",
    [
        "two-level-loss",
        "two-level-coherence",
        "two-level-drive",
        "oscillator-loss",
        "oscillator-drive",
        "composite-loss",
        "composite-drive",
    ],
)
@pytest.mark.parametrize(
    "engine_name",
    [
        "qutip-cpu",
        "dynamiqs-cpu",
        pytest.param("cudensitymat", marks=pytest.mark.gpu),
        pytest.param("dynamiqs", marks=pytest.mark.gpu),
    ],
)
def test_reference_dynamics_and_refinement(
    engine_name, case_name, acceptance_cases, record_property
):
    if engine_name == "qutip-cpu":
        engine = QuTiPIntegrator()
    elif engine_name.startswith("dynamiqs"):
        pytest.importorskip("dynamiqs")
        engine = DynamiqsIntegrator(device_kind="cpu" if engine_name.endswith("cpu") else "gpu")
    else:
        pytest.importorskip("qutip_cuquantum")
        engine = CuQuantumIntegrator()
    case = acceptance_cases[case_name]
    result = engine.evolve(case.system, case.request)
    tighter = engine.evolve(
        case.system, replace(case.request, rtol=1e-10, atol=1e-12, max_steps=400_000)
    )
    reference_error = max(np.linalg.norm(a - b) for a, b in zip(result.states, case.expected))
    refinement_error = max(np.linalg.norm(a - b) for a, b in zip(result.states, tighter.states))
    assert reference_error < 1e-7, reference_error
    assert refinement_error < 1e-7, refinement_error
    assert len(result.states) == len(case.request.times_ns)
    if engine_name in ("cudensitymat", "dynamiqs"):
        assert result.device.startswith("cuda:")
        assert "RTX 5090" in result.diagnostics.get(
            "gpu_name", result.diagnostics.get("device_kind", "")
        )
    assert result.engine == ("dynamiqs" if engine_name.startswith("dynamiqs") else engine_name)
    validity = []
    for states in (result.states, tighter.states):
        residuals = physical_residuals(states)
        validity.append(residuals)
        assert residuals["trace_error"] < 1e-8, residuals
        assert residuals["hermiticity_error"] < 1e-8, residuals
        assert residuals["minimum_eigenvalue"] >= -1e-8, residuals
    expected_observables = {
        name: np.asarray([np.trace(operator @ state) for state in case.expected])
        for name, operator in case.request.observables.items()
    }
    for name, expected in expected_observables.items():
        np.testing.assert_allclose(result.expectations[name], expected, rtol=0, atol=1e-6)
    if case.analytic_population is not None:
        np.testing.assert_allclose(
            result.expectations["population"], case.analytic_population, rtol=0, atol=1e-6
        )
    if case.closed_expected is not None:
        closed_system = replace(case.system, dissipators={}, dissipator_factors={})
        closed = engine.evolve(closed_system, case.request)
        closed_error = max(
            np.linalg.norm(a - b) for a, b in zip(closed.states, case.closed_expected)
        )
        assert closed_error < 1e-7, closed_error
        residuals = physical_residuals(closed.states)
        validity.append(residuals)
        assert residuals["trace_error"] < 1e-8, residuals
        assert residuals["hermiticity_error"] < 1e-8, residuals
        assert residuals["minimum_eigenvalue"] >= -1e-8, residuals
        loss_difference = np.linalg.norm(result.states[-1] - closed.states[-1])
        assert loss_difference > 0.02, loss_difference
        record_property("loss_disabled_difference", float(loss_difference))
        record_property("loss_disabled_reference_error", float(closed_error))
    record_property("engine", result.engine)
    record_property("device", result.device)
    record_property("dimension", int(np.prod(result.dimensions)))
    record_property("duration_ns", float(case.request.times_ns[-1] - case.request.times_ns[0]))
    record_property("reference_error", float(reference_error))
    record_property("refinement_error", float(refinement_error))
    for name in validity[0]:
        values = [stats[name] for stats in validity]
        record_property(name, min(values) if name == "minimum_eigenvalue" else max(values))


def test_oscillator_dynamics_truncation(tmp_path, acceptance_cases, record_property):
    from engine_cases import ROOT

    case = acceptance_cases["oscillator-drive"]
    baseline = QuTiPIntegrator().evolve(case.system, case.request)
    text = (ROOT / "lossy-oscillator/system.toml").read_text()
    frequency = case.system.hamiltonian_ghz[1, 1].real
    errors = {}
    for levels in (8, 2):
        lower = np.diag(np.sqrt(np.arange(1, levels)), k=1)
        number = np.diag(np.arange(levels))
        np.savez(
            tmp_path / "operators.npz",
            H0=frequency * number,
            number=number,
            position=lower + lower.T,
            lowering=lower,
        )
        source = tmp_path / "system.toml"
        source.write_text(text.replace("dimensions = [6]", f"dimensions = [{levels}]"))
        system = load_system(source)
        request = replace(
            case.request,
            initial_state=np.eye(levels)[1],
            observables={"population": system.operators["number"]},
        )
        result = QuTiPIntegrator().evolve(system, request)
        errors[levels] = float(
            np.max(np.abs(result.expectations["population"] - baseline.expectations["population"]))
        )
    assert errors[8] < 1e-6, errors
    assert errors[2] > 1e-6, errors
    record_property("refined_population_change", errors[8])
    record_property("insufficient_population_change", errors[2])


@pytest.mark.parametrize(
    "cutoff, qubit_levels, resonator_levels", [(40, 6, 5), (30, 8, 5), (30, 6, 7)]
)
def test_composite_dynamics_truncation(
    tmp_path, acceptance_cases, cutoff, qubit_levels, resonator_levels, record_property
):
    from engine_cases import ROOT

    case = acceptance_cases["composite-drive"]
    baseline = QuTiPIntegrator().evolve(case.system, case.request)
    (tmp_path / "qubit.toml").write_text(
        (ROOT / "qubit-resonator/qubit.toml")
        .read_text()
        .replace("theta1 = 30", f"theta1 = {cutoff}")
        .replace("retained_levels = 6", f"retained_levels = {qubit_levels}")
    )
    source = tmp_path / "system.toml"
    source.write_text(
        (ROOT / "qubit-resonator/system.toml")
        .read_text()
        .replace("retained_levels = 5", f"retained_levels = {resonator_levels}")
    )
    system = load_system(source)
    request = replace(
        case.request,
        initial_state=np.eye(qubit_levels * resonator_levels)[1],
        observables={"population": system.operators["r.number"]},
    )
    result = QuTiPIntegrator().evolve(system, request)
    error = float(
        np.max(np.abs(result.expectations["population"] - baseline.expectations["population"]))
    )
    assert error < 1e-6, error
    record_property("refined_population_change", error)
