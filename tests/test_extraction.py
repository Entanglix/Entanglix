"""Extraction under declared settings: coverage regression, failures, and projection reuse."""

import io
import sys
from datetime import UTC, datetime

import numpy as np
import pytest
import tomli_w
from pydantic import ValidationError

import entanglix.extraction as extraction
from entanglix.extraction import ExtractionError, ExtractionSettings, extract, learn_projection
from entanglix.measurements import load_dataset, split_dataset

FREQUENCY = np.round(np.linspace(4.0, 6.0, 2001), 6)
BIAS = np.linspace(0.0, 1.0, 16)
HWHM = 0.002
NOISE = 0.02
PHASE = 0.7
SETTINGS = ExtractionSettings(
    background_window_ghz=0.1,
    threshold=6.0,
    min_separation_ghz=0.01,
    fit_half_window_ghz=0.02,
    linewidth_bounds_ghz=[2e-4, 0.02],
    line_shape="lorentzian",
)


def two_lines(bias):
    strong = 5.0 + 0.4 * np.cos(np.pi * bias)
    return [(strong, 1.0), (strong - 0.25, 0.3)]


def profile(detuning, shape):
    if shape == "lorentzian":
        return HWHM**2 / (detuning**2 + HWHM**2)
    return 2.0 ** (-((detuning / HWHM) ** 2))


def synthetic(rng, bias=BIAS, lines=two_lines, shape="lorentzian"):
    clean = np.zeros((bias.size, FREQUENCY.size))
    for centres, amplitude in lines(bias):
        clean += amplitude * profile(FREQUENCY[None, :] - np.asarray(centres)[:, None], shape)
    background = (0.5 + 0.3j) + 0.02 * (FREQUENCY - FREQUENCY[0])
    noise = NOISE * (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
    return np.exp(1j * PHASE) * clean + background + noise


def write_map(directory, signal, bias=BIAS, frequency=FREQUENCY):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"signal": signal, "frequency": frequency}
    levels = {
        "signal": {"key": "signal", "unit": "V"},
        "frequency": {"key": "frequency", "unit": "GHz"},
    }
    if bias is not None:
        arrays["bias"] = bias
        levels["bias"] = {"key": "bias", "unit": "mA"}
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    (directory / "map.npz").write_bytes(buffer.getvalue())
    document = {
        "format_version": 1,
        "id": "synthetic",
        "kind": "spectroscopy",
        "array_file": "map.npz",
        "conditions": {},
        "provenance": {
            "device": "synthetic",
            "measured_at": datetime(2026, 10, 1, tzinfo=UTC),
            "converter": {"name": "test", "version": "1"},
        },
        "map": levels,
    }
    path = directory / "map.toml"
    path.write_text(tomli_w.dumps(document))
    return path


def coverage(result, bias=BIAS, lines=two_lines):
    """Count generated lines, those extracted, and those within one and two uncertainties."""
    counts = np.zeros(4, dtype=int)
    for centres, _ in lines(bias):
        for column, truth in enumerate(np.atleast_1d(centres)):
            counts[0] += 1
            mine = result.points["column"] == column
            offsets = result.points["centre"][mine] - truth
            if not offsets.size or np.abs(offsets).min() > 5 * HWHM:
                continue
            nearest = np.abs(offsets).argmin()
            z = abs(offsets[nearest]) / result.points["centre_uncertainty"][mine][nearest]
            counts[1:] += [1, z <= 1, z <= 2]
    return counts.tolist()


@pytest.mark.parametrize(
    ("shape", "recorded"),
    [("lorentzian", [96, 96, 70, 93]), ("gaussian", [96, 96, 67, 91])],
)
def test_fixed_seed_coverage_matches_recorded_values(tmp_path, shape, recorded):
    settings = SETTINGS.model_copy(update={"line_shape": shape})
    totals = np.zeros(4, dtype=int)
    for seed in (1, 2, 3):
        signal = synthetic(np.random.default_rng(seed), shape=shape)
        result = extract(load_dataset(write_map(tmp_path / f"{shape}-{seed}", signal)), settings)
        totals += coverage(result)
        assert result.failures == ()
        np.testing.assert_allclose(result.points["linewidth"], HWHM, rtol=0.5)
        np.testing.assert_allclose(result.points["bias"], BIAS[result.points["column"]])
    # Failures stay in the denominator: [generated, extracted, within 1σ, within 2σ].
    assert totals.tolist() == recorded


def test_projection_is_learned_from_the_map_and_reused_when_given(tmp_path):
    rng = np.random.default_rng(4)
    source = write_map(tmp_path / "parent", synthetic(rng))
    outputs = {"holdout": tmp_path / "holdout.toml", "remainder": tmp_path / "training.toml"}
    split_dataset(source, 0.7, 1.0, **outputs)
    training, held = load_dataset(outputs["remainder"]), load_dataset(outputs["holdout"])
    learned = learn_projection(training, SETTINGS)
    np.testing.assert_allclose(learned, (np.cos(PHASE), np.sin(PHASE)), atol=1e-3)
    assert extract(training, SETTINGS).projection == learned
    reused = extract(held, SETTINGS, projection=learned)
    assert reused.projection == learned
    assert reused.projection != learn_projection(held, SETTINGS)
    assert coverage(reused, BIAS[BIAS >= 0.7])[1] == 2 * np.count_nonzero(BIAS >= 0.7)


def test_failures_are_reported_per_column(tmp_path, monkeypatch):
    def lines(bias):
        # A second line 4.5 MHz away in the last four columns, inside the 10 MHz separation.
        return [(np.full(bias.size, 4.5), 1.0), (np.where(bias >= 0.75, 4.5045, 100.0), 1.0)]

    signal = synthetic(np.random.default_rng(5), lines=lines)
    signal[3] = 0.5 + 0.3j
    result = extract(load_dataset(write_map(tmp_path / "map", signal)), SETTINGS)
    assert [item for item in result.failures if item["column"] == 3] == [
        {"column": 3, "bias": BIAS[3], "reason": "no noise estimate"}
    ]
    overlaps = [item for item in result.failures if item["reason"] == "unresolved overlap"]
    assert [item["column"] for item in overlaps] == [12, 13, 14, 15]
    for item in overlaps:
        assert item["bias"] == BIAS[item["column"]]
        np.testing.assert_allclose(item["frequencies_ghz"], [4.5, 4.5045], atol=1e-3)
    assert len(result.failures) == 5
    assert sorted(result.points["column"]) == [c for c in range(12) if c != 3]

    empty = synthetic(np.random.default_rng(6), lines=lambda b: [])
    result = extract(load_dataset(write_map(tmp_path / "empty", empty)), SETTINGS)
    assert {item["reason"] for item in result.failures} == {"no peak"}
    assert len(result.failures) == BIAS.size and result.points["centre"].size == 0

    original = extraction.least_squares

    def failing(*args, **kwargs):
        found = original(*args, **kwargs)
        found.success = False
        return found

    monkeypatch.setattr(extraction, "least_squares", failing)
    result = extract(
        load_dataset(write_map(tmp_path / "fit", synthetic(np.random.default_rng(7)))), SETTINGS
    )
    assert {item["reason"] for item in result.failures} == {"line fit failed"}
    assert {item["detail"] for item in result.failures} == {"the optimizer did not converge"}
    assert len(result.failures) == 2 * BIAS.size


def test_rank_deficient_fits_fail_beside_a_full_rank_fit(tmp_path, monkeypatch):
    # Two lines 2 MHz apart share a five-sample window, which cannot determine seven parameters;
    # the isolated line's window determines its four.
    monkeypatch.setattr(sys.modules[__name__], "HWHM", 0.0006)

    def lines(bias):
        return [
            (np.full(bias.size, 4.5), 1.0),
            (np.full(bias.size, 5.0), 1.0),
            (np.full(bias.size, 5.002), 1.0),
        ]

    settings = SETTINGS.model_copy(
        update={
            "line_shape": "gaussian",
            "min_separation_ghz": 0.0015,
            "fit_half_window_ghz": 0.002,
            "linewidth_bounds_ghz": [1e-4, 0.02],
        }
    )
    signal = synthetic(np.random.default_rng(10), lines=lines, shape="gaussian")
    result = extract(load_dataset(write_map(tmp_path / "pair", signal)), settings)
    assert len(result.failures) == 2 * BIAS.size
    for item in result.failures:
        assert item["reason"] == "line fit failed"
        assert item["detail"].startswith("rank-deficient")
        assert item["frequency_ghz"] in (5.0, 5.002)
    assert result.points["column"].tolist() == list(range(BIAS.size))
    np.testing.assert_allclose(result.points["centre"], 4.5, atol=1e-3)
    assert np.all(np.isfinite(result.points["centre_uncertainty"]))
    assert np.all(result.points["centre_uncertainty"] > 0)


def test_fixed_bias_map_is_one_column(tmp_path):
    signal = synthetic(np.random.default_rng(8))[0]
    result = extract(load_dataset(write_map(tmp_path / "fixed", signal, bias=None)), SETTINGS)
    assert "bias" not in result.points
    assert result.points["column"].tolist() == [0, 0]
    np.testing.assert_allclose(sorted(result.points["centre"]), [5.15, 5.4], atol=5e-4)


def test_refusals(tmp_path):
    with pytest.raises(ValidationError, match="0 < lower < upper"):
        ExtractionSettings.model_validate(
            {**SETTINGS.model_dump(), "linewidth_bounds_ghz": [0.02, 0.01]}
        )
    with pytest.raises(ValidationError, match="line_shape"):
        ExtractionSettings.model_validate({**SETTINGS.model_dump(), "line_shape": "voigt"})
    uneven = FREQUENCY.copy()
    uneven[10] += 2e-4
    dataset = load_dataset(
        write_map(tmp_path / "uneven", synthetic(np.random.default_rng(9)), frequency=uneven)
    )
    with pytest.raises(ExtractionError, match="uniform frequency grid"):
        extract(dataset, SETTINGS)
    dataset = load_dataset(write_map(tmp_path / "map", synthetic(np.random.default_rng(9))))
    with pytest.raises(ExtractionError, match="unit I/Q direction"):
        extract(dataset, SETTINGS, projection=(1.0, 1.0))
