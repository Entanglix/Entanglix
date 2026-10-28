"""Policy integrity, applicability, and bounded deterministic evidence coverage."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from entanglix.corrector import load_policy, policy_digest, relevant_checks, select_evidence


@pytest.mark.parametrize("name", ["__init__.py", "thresholds.toml"])
def test_policy_changed_after_launch_is_rejected(tmp_path, monkeypatch, name):
    from entanglix import corrector

    for path in corrector.POLICY_DIRECTORY.iterdir():
        if path.suffix in {".py", ".toml"}:
            shutil.copyfile(path, tmp_path / path.name)
    monkeypatch.setattr(corrector, "POLICY_DIRECTORY", tmp_path)
    expected = load_policy().version
    path = tmp_path / name
    path.write_text(path.read_text() + "\n# changed policy\n")
    assert policy_digest() != expected
    with pytest.raises(ValueError, match="between launch and load"):
        load_policy(expected)


def test_trusted_check_definitions_reject_unknown_labels():
    assert relevant_checks("lindblad") == ("integrity", "segments", "diagnostics")
    assert relevant_checks("spectrum") == ("spectral_refinement",)
    for kind in ("authored-script", "spectral", "unknown"):
        with pytest.raises(ValueError, match="unknown computation"):
            relevant_checks(kind)


def test_evidence_is_bounded_reports_uncovered_warnings_and_breaks_ties_by_grid_order():
    policy = load_policy()
    values = np.array([0, 3, 3, -2, -2, 1, 1, 1, 1, 1, 1, 0], dtype=float)
    warnings = np.ones(values.shape)
    selected = select_evidence(values.shape, metric=values, warnings=warnings, policy=policy)
    assert selected.points[:2] == (0, 11)
    assert selected.points[2:4] == (3, 1)  # First tied minimum and maximum.
    assert len(selected.points) == policy.thresholds["evidence"]["max_points"]
    assert selected.uncovered
    assert set(selected.points) | set(selected.uncovered) == set(range(12))
    assert selected == select_evidence(
        values.shape, metric=values, warnings=warnings, policy=policy
    )
    assert "steepest_change" in selected.reasons[2]


def test_evidence_uses_grid_edges_instead_of_flattening_across_rows():
    values = np.array([[0, 1, 2], [10, 11, 12]])
    selected = select_evidence(values.shape, metric=values)
    assert selected.points[:4] == (0, 2, 3, 5)
    assert "steepest_change" in selected.reasons[0]
    assert "steepest_change" not in selected.reasons[3]


def test_corner_bound_is_a_check_prerequisite():
    with pytest.raises(ValueError, match="corners exceed"):
        select_evidence((2, 2, 2, 2))
    assert select_evidence((1,)).points == (0,)


@pytest.mark.parametrize("engine_name", ["qutip-cpu", "dynamiqs"])
def test_dimension_48_exact_liouvillian(tmp_path, engine_name, record_property):
    """Numerical evidence at dimension 48, without a policy admission ceiling."""
    from engine_cases import exact_states

    from entanglix.engines import QuTiPIntegrator
    from entanglix.evolution import EvolutionInput
    from entanglix.models import load_system

    if engine_name == "dynamiqs":
        pytest.importorskip("dynamiqs")
        from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

        engine = DynamiqsIntegrator(device_kind="cpu")
    else:
        engine = QuTiPIntegrator()
    root = Path(__file__).resolve().parents[1]
    text = (
        (root / "examples/qubit-resonator/system.toml")
        .read_text()
        .replace("retained_levels = 5", "retained_levels = 6")
    )
    leaf = (
        (root / "examples/qubit-resonator/qubit.toml")
        .read_text()
        .replace("retained_levels = 6", "retained_levels = 8")
    )
    (tmp_path / "system.toml").write_text(text)
    (tmp_path / "qubit.toml").write_text(leaf)
    system = load_system(tmp_path / "system.toml")
    thresholds = load_policy().thresholds["refinement"]
    dimension = 48
    assert np.prod(system.dimensions) == dimension
    rng = np.random.default_rng(42)
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    state /= np.linalg.norm(state)
    request = EvolutionInput(
        state,
        np.linspace(0, 2, 5),
        rtol=thresholds["rtol"],
        atol=thresholds["atol"],
        max_steps=400_000,
    )
    expected = exact_states(system, request)
    result = engine.evolve(system, request)
    error = max(np.linalg.norm(a - b) for a, b in zip(result.states, expected))
    assert error < 1e-7, error
    record_property("engine", result.engine)
    record_property("dimension", dimension)
    record_property("duration_ns", 2)
    record_property("reference_error", float(error))
