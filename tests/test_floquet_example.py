"""The authored Floquet example retains its spectrum and numerical evidence."""

import hashlib
from pathlib import Path

import numpy as np

from entanglix.runner import read_run, run_script

ROOT = Path(__file__).resolve().parents[1]


def test_readout_floquet_job(tmp_path, runner_repository, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    attempt = run_script(
        ROOT / "examples/readout/system.toml",
        ROOT / "examples/readout/floquet.py",
        runs_root=tmp_path / "runs",
        repository=runner_repository,
        timeout_seconds=60,
    )
    assert attempt["exit_code"] == 0, read_run(Path(attempt["path"]))["log_tail"]
    path = Path(attempt["path"])
    record = read_run(path)
    assert record["status"] == "completed"
    assert record["manifest"]["protocol"] == "script"
    result = record["result"]
    assert result["engine"] == "qutip-cpu"
    assert result["solver"] == "FloquetBasis/vern9"
    assert result["dimensions"] == [5, 6]
    assert result["settings"]["omitted_dissipator_rates_per_ns"] == {"resonator_loss": 0.03}
    evidence = result["numerical_evidence"]
    assert evidence["zero_drive_error_ghz"] < 1e-8
    assert max(evidence["labelled_refinement_errors_ghz"]) < 1e-8
    assert max(evidence["unitarity_errors_frobenius"]) < 1e-8
    assert evidence["min_static_overlap"] > 0.9
    assert min(evidence["min_label_overlaps"]) > 0.9
    assert max(evidence["max_top_qubit_populations"]) < 1e-6
    assert max(evidence["max_top_resonator_populations"]) < 1e-6
    np.testing.assert_allclose(
        np.array(result["metrics"]["qubit_shift_r0_ghz"]) * 1e6,
        [0, -0.736031, -2.944122, -11.776446],
        rtol=0,
        atol=0.01,
    )
    artifact = path / result["artifacts"]["quasienergies"]["path"]
    assert (
        hashlib.sha256(artifact.read_bytes()).hexdigest()
        == result["artifacts"]["quasienergies"]["sha256"]
    )
    with np.load(artifact, allow_pickle=False) as data:
        assert data["quasienergies_ghz"].shape == (4, 30)
        np.testing.assert_array_equal(data["labels"], [[0, 0], [1, 0]])
        assert all(len(set(row)) == 2 for row in data["label_indices"])
        np.testing.assert_array_equal(data["amplitudes_ghz"], [0, 0.005, 0.01, 0.02])
        energies = data["quasienergies_ghz"]
        assert np.all(energies >= -6.9 / 2)
        assert np.all(energies < 6.9 / 2)
