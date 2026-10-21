"""Calibration from spectroscopy: assignment, fit, outcomes, report, refusals and the job."""

import io
import json
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import tomli_w

from entanglix.calibration import ANCHOR_FAILED, UNRESOLVED, calibrate_problem
from entanglix.models import load_system
from entanglix.protocols.calibrate import load_calibration, prepare

FLUXONIUM = """format_version = 1
id = "fluxonium"
[units]
energy = "GHz"
time = "ns"
[parameters]
EJ = { value = 4.0, unit = "GHz", prior = { kind = "uniform", lower = 2.0, upper = 8.0 } }
EL = { value = 1.0, unit = "GHz", prior = { kind = "uniform", lower = 0.5, upper = 2.0 } }
phi = { value = 0.25, unit = "Phi0" }
[model]
kind = "composite"
[[model.subsystems]]
id = "f"
kind = "fluxonium"
josephson_energy = { parameter = "EJ" }
charging_energy = { value = 1.0, unit = "GHz" }
inductive_energy = { parameter = "EL" }
flux = { parameter = "phi" }
cutoff = 40
retained_levels = 5
[[model.subsystems]]
id = "r"
kind = "oscillator"
frequency = { value = 30.0, unit = "GHz" }
retained_levels = 2
"""
TRANSMONS = """format_version = 1
id = "pair"
[units]
energy = "GHz"
time = "ns"
[parameters]
EJa = { value = 20.0, unit = "GHz", prior = { kind = "uniform", lower = 10.0, upper = 30.0 } }
EJb = { value = 15.0, unit = "GHz", prior = { kind = "uniform", lower = 10.0, upper = 30.0 } }
[model]
kind = "composite"
[[model.subsystems]]
id = "a"
kind = "transmon"
josephson_energy = { parameter = "EJa" }
charging_energy = { value = 0.25, unit = "GHz" }
ng = 0.0
cutoff = 10
retained_levels = 3
[[model.subsystems]]
id = "b"
kind = "transmon"
josephson_energy = { parameter = "EJb" }
charging_energy = { value = 0.25, unit = "GHz" }
ng = 0.0
cutoff = 10
retained_levels = 3
"""
RESONATOR = """format_version = 1
id = "qr"
[units]
energy = "GHz"
time = "ns"
[parameters]
EJ = { value = 20.0, unit = "GHz", prior = { kind = "uniform", lower = 10.0, upper = 30.0 } }
fr = { value = 1.0, unit = "GHz", prior = { kind = "uniform", lower = 1e-6, upper = 10.0 } }
[model]
kind = "composite"
[[model.subsystems]]
id = "q"
kind = "transmon"
josephson_energy = { parameter = "EJ" }
charging_energy = { value = 0.25, unit = "GHz" }
ng = 0.0
cutoff = 10
retained_levels = 3
[[model.subsystems]]
id = "r"
kind = "oscillator"
frequency = { parameter = "fr" }
retained_levels = 2
"""
TRUTH = {"EJ": 4.0, "EL": 1.0, "offset": 0.1, "period": 1.2}
BIAS = np.linspace(0.1, 0.1 + 0.4 * 1.2, 12)
# Visible lines and their amplitudes; the anchor 0-1 is the strongest.
VISIBLE = {(0, 1, 1): 1.0, (1, 2, 1): 0.3, (0, 2, 2): 0.2}
SIGMA = 1e-3
HWHM = 0.002
FAMILY = {"initial": [0, 1], "max_level": 3, "photons": [1, 2]}
MATCHING = {
    "anchor": "0-1",
    "range_ghz": 0.3,
    "cutoff": 5.0,
    "resolution": 2.0,
    "range_iterations": 20,
    "cutoff_iterations": 20,
}
EXTRACTION = {
    "background_window_ghz": 0.1,
    "threshold": 6.0,
    "min_separation_ghz": 0.01,
    "fit_half_window_ghz": 0.02,
    "linewidth_bounds_ghz": [2e-4, 0.02],
    "line_shape": "lorentzian",
}
AXIS_MAP = {
    "flux": "phi",
    "offset": {"lower": -0.5, "upper": 0.5, "scale": 0.01},
    "period": {"lower": 0.8, "upper": 1.6, "scale": 0.01},
}


def lines(values, bias=None, visible=VISIBLE, system=FLUXONIUM):
    """Transition frequencies of the generating model, one row per bias (or one row)."""
    rows = []
    for flux in [None] if bias is None else (bias - values["offset"]) / values["period"]:
        parameters = {n: v for n, v in values.items() if n not in {"offset", "period"}}
        if flux is not None:
            parameters["phi"] = float(flux)
        energies = np.linalg.eigvalsh(build(system, parameters).hamiltonian_ghz)
        rows.append([(energies[f] - energies[i]) / k for i, f, k in visible])
    return np.array(rows)


def build(system, parameters):
    document = tomllib.loads(system)
    for name, value in parameters.items():
        document["parameters"][name]["value"] = value
    return load_system(Path("system.toml"), content=tomli_w.dumps(document).encode())


def provenance():
    return {
        "device": "synthetic",
        "measured_at": datetime(2026, 10, 1, tzinfo=UTC),
        "converter": {"name": "test", "version": "1"},
    }


def write_dataset(directory, name, arrays, level):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    (directory / f"{name}.npz").write_bytes(buffer.getvalue())
    document = {
        "format_version": 1,
        "id": name,
        "kind": "spectroscopy",
        "array_file": f"{name}.npz",
        "conditions": {},
        "provenance": provenance(),
        **level,
    }
    (directory / f"{name}.toml").write_text(tomli_w.dumps(document))
    return f"{name}.toml"


def write_points(
    directory, centre, bias=None, *, labels=None, fields=("linewidth", "amplitude"), amplitude=None
):
    centre = np.ravel(centre)
    arrays = {"centre": centre, "sigma": np.full(centre.size, SIGMA)}
    level = {
        "centre": {"key": "centre", "unit": "GHz"},
        "centre_uncertainty": {"key": "sigma", "unit": "GHz"},
    }
    if bias is not None:
        arrays["bias"] = np.ravel(bias)
        level["bias"] = {"key": "bias", "unit": "V"}
    if "linewidth" in fields:
        arrays["hwhm"] = np.full(centre.size, HWHM)
        level["linewidth"] = {"key": "hwhm", "unit": "GHz"}
    if "amplitude" in fields:
        arrays["amplitude"] = np.ravel(amplitude) if amplitude is not None else np.ones(centre.size)
        level["amplitude"] = {"key": "amplitude", "unit": "V"}
    if labels is not None:
        arrays["label"] = np.asarray(labels)
        level["assignment"] = {"key": "label"}
    return write_dataset(directory, "points", arrays, {"points": level})


def sweep_points(directory, rng, values=TRUTH, bias=BIAS, **options):
    centres = lines(values, bias)
    noisy = centres + rng.normal(0, SIGMA, centres.shape)
    amplitude = np.tile(list(VISIBLE.values()), (bias.size, 1))
    return write_points(
        directory, noisy, np.repeat(bias, len(VISIBLE)), amplitude=amplitude, **options
    )


def sweep_map(directory, rng, values=TRUTH, bias=np.linspace(0.1, 0.58, 16)):
    frequency = np.round(np.arange(2.0, 10.5, 0.001), 6)
    clean = np.zeros((bias.size, frequency.size))
    for column, centres in enumerate(lines(values, bias)):
        for centre, amplitude in zip(centres, VISIBLE.values(), strict=True):
            clean[column] += amplitude * HWHM**2 / ((frequency - centre) ** 2 + HWHM**2)
    noise = 0.02 * (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
    signal = np.exp(0.7j) * clean + (0.5 + 0.3j) + 0.02 * (frequency - 2.0) + noise
    level = {
        "signal": {"key": "signal", "unit": "V"},
        "frequency": {"key": "frequency", "unit": "GHz"},
        "bias": {"key": "bias", "unit": "V"},
    }
    arrays = {"signal": signal, "frequency": frequency, "bias": bias}
    return write_dataset(directory, "map", arrays, {"map": level})


def displaced(sign, factor=0.03, values=TRUTH):
    start = {name: value * (1 + factor * sign) for name, value in values.items()}
    if "period" in values:
        start["offset"] = values["offset"] + factor * sign * values["period"]
    return start


def write_spec(directory, datasets, *, system=FLUXONIUM, **overrides):
    (directory / "system.toml").write_text(system)
    spec = {
        "format_version": 1,
        "system": "system.toml",
        "datasets": datasets,
        "parameters": {"EJ": {"scale": 0.1}, "EL": {"scale": 0.1}},
        "axis_map": AXIS_MAP,
        "family": FAMILY,
        "matching": MATCHING,
        "starts": [displaced(1)],
        "merge_tolerance": 1e-3,
    }
    spec.update(overrides)
    spec = {key: value for key, value in spec.items() if value is not None}
    path = directory / "calibrate.toml"
    path.write_text(tomli_w.dumps(spec))
    return path


def fit(path):
    loaded = load_calibration(path)
    problem, record = prepare(loaded)
    starts = [[start[name] for name in loaded.names] for start in loaded.spec.starts]
    return calibrate_problem(problem, starts), loaded, record


def covered(solution, truth):
    """Parameters whose truth lies within one and two reported uncertainties."""
    sigma = solution["uncertainty"]
    return [
        sum(
            name in sigma and abs(solution["values"][name] - value) <= k * sigma[name]
            for name, value in truth.items()
        )
        for k in (1, 2)
    ]


def test_fixed_seed_regression_set_matches_recorded_outcomes_and_coverage(tmp_path):
    outcomes, counts = [], np.zeros(3, dtype=int)
    cases = [("points", seed) for seed in range(1, 7)] + [("map", seed) for seed in (1, 2)]
    for kind, seed in cases:
        directory = tmp_path / f"{kind}-{seed}"
        directory.mkdir()
        rng = np.random.default_rng(seed)
        dataset = sweep_points(directory, rng) if kind == "points" else sweep_map(directory, rng)
        found, _, _ = fit(
            write_spec(
                directory,
                [dataset],
                starts=[displaced((-1) ** seed)],
                extraction=EXTRACTION if kind == "map" else None,
            )
        )
        outcomes.append(found["outcome"])
        if found["solutions"]:
            counts += [len(TRUTH), *covered(found["solutions"][0], TRUTH)]
    # [parameter estimates, truth within 1σ, within 2σ] over the best solutions.
    assert outcomes == ["solution"] * 8
    assert counts.tolist() == [32, 30, 32]


def test_fit_reports_statistics_diagnostics_nuisance_and_tolerances(tmp_path):
    dataset = sweep_points(tmp_path, np.random.default_rng(11))
    found, loaded, record = fit(write_spec(tmp_path, [dataset]))
    assert found["outcome"] == "solution"
    (solution,) = found["solutions"]
    assert loaded.names == ("EJ", "EL", "offset", "period")
    assert set(solution["values"]) == {"EJ", "EL", "offset", "period"}
    assert solution["rank"] == 4 and solution["undetermined_combinations"] == []
    assert solution["determined"] == ["EJ", "EL", "offset", "period"]
    # At the ninth bias f03 = 2 f01, so 0-1, 0-3/2 and 1-3 coincide and that f01 peak is
    # unresolved: left out of the fit and penalised.
    assert solution["counts"] == {"assigned": 35, "unexplained": 0, "unresolved": 1, "missing": 0}
    assert solution["assignment"][3 * 8] == UNRESOLVED
    statistics = solution["statistics"]
    assert statistics["assigned"] == 35 and statistics["fitted"] == 4
    assert statistics["reduced_chi2"] == pytest.approx(statistics["chi2"] / (35 - 4))
    width = {"EJ": 6.0, "EL": 1.5, "offset": 1.0, "period": 0.8}
    for name, sigma in solution["uncertainty"].items():
        assert solution["uncertainty_over_prior_width"][name] == pytest.approx(sigma / width[name])
    matrix = np.array(solution["correlation"]["matrix"])
    np.testing.assert_allclose(np.diag(matrix), 1)
    assert solution["lines"] == {"0-1": 11, "1-2": 12, "0-2/2": 12}
    assert found["starts"][0]["anchor"] == {
        "columns": 12,
        "assigned": 12,
        "unexplained": 0,
        "unresolved": 0,
    }
    assert [column["bias"] for column in record["columns"]] == BIAS.tolist()


def test_supplied_labels_are_kept_and_fit_without_matching(tmp_path):
    rng = np.random.default_rng(12)
    labels = np.tile(["0-1", "1-2", "0-2/2"], BIAS.size)
    dataset = sweep_points(tmp_path, rng, labels=labels, fields=())
    path = write_spec(tmp_path, [dataset], matching=None)
    found, loaded, _ = fit(path)
    assert loaded.labelled and loaded.listing["procedure"]["assignment"] == "supplied labels"
    assert found["outcome"] == "solution"
    assert found["starts"][0]["cycles"] == {"fit": 1} and "anchor" not in found["starts"][0]
    labels_of = [loaded.members[j].label for j in found["solutions"][0]["assignment"]]
    assert labels_of == labels.tolist()
    write_spec(tmp_path, [dataset], matching=MATCHING)
    with pytest.raises(ValueError, match="supplied assignments use no anchor or matching"):
        load_calibration(path)


def test_fixed_bias_data_fits_without_an_axis_map(tmp_path):
    values = {"EJ": 4.0, "EL": 1.0}
    visible = [(0, 1, 1), (0, 2, 1), (1, 2, 1), (0, 3, 1)]
    centre = lines(values, visible=visible)[0] + np.random.default_rng(13).normal(0, SIGMA, 4)
    dataset = write_points(tmp_path, centre, labels=["0-1", "0-2", "1-2", "0-3"], fields=())
    starts = [{"EJ": 4.1, "EL": 0.97}]
    found, loaded, record = fit(
        write_spec(tmp_path, [dataset], axis_map=None, matching=None, starts=starts)
    )
    assert loaded.names == ("EJ", "EL")
    assert record["columns"] == [{"dataset": "points.toml"}]
    assert found["outcome"] == "solution"
    assert covered(found["solutions"][0], values)[1] == 2
    with pytest.raises(ValueError, match="an axis_map needs a training dataset with a bias axis"):
        load_calibration(write_spec(tmp_path, [dataset], matching=None, starts=starts))


def test_single_bias_two_parameter_fit_is_insufficient_evidence(tmp_path):
    centre = lines({"EJ": 4.0, "EL": 1.0}, visible=[(0, 1, 1)])[0]
    dataset = write_points(tmp_path, centre, labels=["0-1"], fields=())
    found, _, _ = fit(
        write_spec(
            tmp_path, [dataset], axis_map=None, matching=None, starts=[{"EJ": 4.2, "EL": 0.95}]
        )
    )
    assert found["outcome"] == "insufficient evidence"
    (solution,) = found["solutions"]
    assert solution["outcome"] == "insufficient evidence"
    assert solution["rank"] == 1 and len(solution["undetermined_combinations"]) == 1
    assert solution["determined"] == [] and solution["uncertainty"] == {}
    assert solution["statistics"]["assigned"] == 1 and solution["statistics"]["fitted"] == 2
    assert "reduced_chi2" not in solution["statistics"]


def test_anchor_failure_is_reported_with_its_counts(tmp_path):
    # Each column's strongest peak is an unmodelled line far from every 0-1 prediction.
    rng = np.random.default_rng(14)
    centres = lines(TRUTH, BIAS) + rng.normal(0, SIGMA, (BIAS.size, len(VISIBLE)))
    centres = np.column_stack([centres, np.full(BIAS.size, 11.5)])
    amplitude = np.tile([*VISIBLE.values(), 2.0], (BIAS.size, 1))
    dataset = write_points(tmp_path, centres, np.repeat(BIAS, 4), amplitude=amplitude)
    found, _, _ = fit(write_spec(tmp_path, [dataset]))
    assert found["outcome"] == ANCHOR_FAILED and found["solutions"] == []
    (start,) = found["starts"]
    assert start["status"] == ANCHOR_FAILED
    assert start["anchor"] == {"columns": 12, "assigned": 0, "unexplained": 12, "unresolved": 0}


def test_reaching_an_iteration_limit_is_non_convergence(tmp_path):
    dataset = sweep_points(tmp_path, np.random.default_rng(15))
    start = {"EJ": 3.9, "EL": 1.03, "offset": 0.097, "period": 1.0}
    found, _, _ = fit(write_spec(tmp_path, [dataset], starts=[start]))
    assert found["starts"][0]["cycles"]["anchor range"] > 1
    matching = {**MATCHING, "range_iterations": 1}
    found, _, _ = fit(write_spec(tmp_path, [dataset], starts=[start], matching=matching))
    assert found["outcome"] == "no fit" and found["solutions"] == []
    assert found["starts"][0]["status"] == (
        "non-converged: the anchor range phase reached its iteration limit"
    )


def test_conventions_remove_offset_aliases(tmp_path):
    dataset = sweep_points(tmp_path, np.random.default_rng(16))
    axis = {**AXIS_MAP, "offset": {"lower": -0.5, "upper": 1.5, "scale": 0.01}}
    alias = {**displaced(-1), "offset": displaced(-1)["offset"] + TRUTH["period"]}
    starts = [displaced(1), alias]
    found, _, _ = fit(write_spec(tmp_path, [dataset], axis_map=axis, starts=starts))
    assert found["outcome"] == "several distinct solutions"
    found, _, _ = fit(
        write_spec(
            tmp_path,
            [dataset],
            axis_map=axis,
            starts=starts,
            conventions={"offset_window": -0.5},
        )
    )
    assert found["outcome"] == "solution"
    (solution,) = found["solutions"]
    assert solution["starts"] == [0, 1]
    assert solution["equivalents_removed"] == ["offset alias: shifted by -1 period(s)"]
    assert solution["conventions_not_applied"] == []
    assert [start["equivalents_removed"] for start in found["starts"]] == [
        [],
        ["offset alias: shifted by -1 period(s)"],
    ]
    assert abs(solution["values"]["offset"] - TRUTH["offset"]) < 0.01


def test_conventions_remove_exchanged_parameters(tmp_path):
    values = {"EJa": 20.0, "EJb": 15.0}
    centre = lines(values, visible=[(0, 1, 1), (0, 2, 1)], system=TRANSMONS)[0]
    dataset = write_points(tmp_path, centre, labels=["0-1", "0-2"], fields=())
    options = {
        "system": TRANSMONS,
        "parameters": {"EJa": {"scale": 1.0}, "EJb": {"scale": 1.0}},
        "axis_map": None,
        "matching": None,
        "family": {"initial": [0], "final": [1, 2]},
        "starts": [{"EJa": 21.0, "EJb": 14.5}, {"EJa": 14.5, "EJb": 21.0}],
    }
    found, _, _ = fit(write_spec(tmp_path, [dataset], **options))
    assert found["outcome"] == "several distinct solutions"
    found, _, _ = fit(
        write_spec(tmp_path, [dataset], conventions={"exchange": [["EJa", "EJb"]]}, **options)
    )
    assert found["outcome"] == "solution"
    (solution,) = found["solutions"]
    assert solution["starts"] == [0, 1]
    assert solution["equivalents_removed"] == ["exchange EJa/EJb"]
    assert solution["conventions_not_applied"] == []
    assert solution["values"] == pytest.approx(values, abs=1e-4)
    unequal = TRANSMONS.replace(
        "lower = 10.0, upper = 30.0 } }\nEJb", "lower = 9.0, upper = 30.0 } }\nEJb"
    )
    with pytest.raises(ValueError, match="exchange EJa/EJb needs equal units, priors and scales"):
        load_calibration(
            write_spec(
                tmp_path,
                [dataset],
                **{**options, "system": unequal},
                conventions={"exchange": [["EJa", "EJb"]]},
            )
        )


def test_a_convention_that_moves_the_predictions_is_not_applied(tmp_path):
    # Bare labels tie each line to one transmon, so exchanging EJa and EJb moves both lines.
    values = {"EJa": 15.0, "EJb": 20.0}
    hamiltonian = build(TRANSMONS, values).hamiltonian_ghz.real
    centre = [hamiltonian[3, 3] - hamiltonian[0, 0], hamiltonian[1, 1] - hamiltonian[0, 0]]
    dataset = write_points(tmp_path, centre, labels=["qa", "qb"], fields=())
    found, _, _ = fit(
        write_spec(
            tmp_path,
            [dataset],
            system=TRANSMONS,
            parameters={"EJa": {"scale": 1.0}, "EJb": {"scale": 1.0}},
            axis_map=None,
            matching=None,
            family={
                "lines": {
                    "qa": {"initial": [0, 0], "final": [1, 0]},
                    "qb": {"initial": [0, 0], "final": [0, 1]},
                }
            },
            starts=[{"EJa": 14.5, "EJb": 21.0}],
            conventions={"exchange": [["EJa", "EJb"]]},
        )
    )
    assert found["outcome"] == "solution"
    (solution,) = found["solutions"]
    assert solution["values"] == pytest.approx(values, abs=1e-4)
    assert solution["equivalents_removed"] == []
    assert solution["conventions_not_applied"] == ["exchange EJa/EJb"]
    (refused,) = found["starts"][0]["conventions_not_applied"]
    assert refused["convention"] == "exchange EJa/EJb"
    assert refused["largest_line_change_ghz"] == pytest.approx(abs(centre[0] - centre[1]), rel=1e-3)
    # The published objective is the cost at the published values.
    assert solution["statistics"]["chi2"] < 1e-6
    assert solution["objective"] == pytest.approx(solution["statistics"]["chi2"], abs=1e-9)


@pytest.mark.parametrize(
    ("prior", "line", "start", "expected", "bounds_hit", "rtol"),
    [
        # The line at 0 GHz pins fr to its 1e-6 GHz lower bound, within one diagnostic step
        # of the non-positive frequencies that construction refuses.
        ("lower = 1e-6, upper = 10.0", 0.0, 0.5, pytest.approx(1e-6, abs=1e-5), ["fr"], 1e-6),
        # A prior far narrower than the scale, whose lower endpoint construction refuses. The
        # step is 1e-5 of the interval, 1e-11 GHz, so rounding limits σ(fr) to about 1e-4.
        ("lower = 0.0, upper = 1e-6", 5e-7, 2e-7, pytest.approx(5e-7, abs=1e-12), [], 1e-3),
    ],
)
def test_diagnostics_stay_inside_the_bounds(
    tmp_path, prior, line, start, expected, bounds_hit, rtol
):
    system = RESONATOR.replace("lower = 1e-6, upper = 10.0", prior)
    hamiltonian = build(RESONATOR, {"EJ": 20.0, "fr": 1.0}).hamiltonian_ghz.real
    centre = [line, hamiltonian[2, 2] - hamiltonian[0, 0]]
    dataset = write_points(tmp_path, centre, labels=["0-1", "0-2"], fields=())
    found, _, _ = fit(
        write_spec(
            tmp_path,
            [dataset],
            system=system,
            parameters={"EJ": {"scale": 1.0}, "fr": {"scale": 1.0}},
            axis_map=None,
            matching=None,
            family={"initial": [0], "final": [1, 2]},
            starts=[{"EJ": 21.0, "fr": start}],
        )
    )
    assert found["outcome"] == "solution"
    (solution,) = found["solutions"]
    assert solution["values"]["fr"] == expected
    assert solution["bounds_hit"] == bounds_hit
    # The resonator line's residual falls by one per uncertainty of fr: σ(fr) = σ.
    assert solution["uncertainty"]["fr"] == pytest.approx(SIGMA, rel=rtol)


def test_a_linked_dataset_is_frozen_under_its_referenced_name(tmp_path):
    from entanglix.protocols.calibrate import compile_calibration

    dataset = sweep_points(tmp_path, np.random.default_rng(20))
    (tmp_path / "alias.toml").symlink_to(tmp_path / dataset)
    path = write_spec(tmp_path, ["alias.toml"])
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    gate = compile_calibration(path, frozen, [], Path(__file__).resolve().parents[1])
    assert sorted(gate["inputs"]) == ["alias.toml", "calibrate.toml", "points.npz", "system.toml"]
    reloaded = load_calibration(frozen / "calibrate.toml")
    assert (
        reloaded.datasets["alias.toml"].sha256
        == load_calibration(path).datasets["alias.toml"].sha256
    )
    assert prepare(reloaded)[1]["datasets"]["alias.toml"]["peaks"] == 36


def test_a_map_without_a_uniform_grid_is_refused_at_launch(tmp_path):
    frequency = np.array([4.0, 4.001, 4.003, 4.004])
    arrays = {"signal": np.zeros((2, 4)), "frequency": frequency, "bias": np.array([0.1, 0.2])}
    level = {
        "signal": {"key": "signal", "unit": "V"},
        "frequency": {"key": "frequency", "unit": "GHz"},
        "bias": {"key": "bias", "unit": "V"},
    }
    dataset = write_dataset(tmp_path, "map", arrays, {"map": level})
    path = write_spec(tmp_path, [dataset], extraction=EXTRACTION)
    with pytest.raises(ValueError, match="'map.toml': extraction needs a uniform frequency grid"):
        load_calibration(path)


def test_bare_labels_follow_overlap_through_a_crossing(tmp_path):
    # With a 7 GHz resonator, fluxonium levels cross resonator states in energy order.
    system = FLUXONIUM.replace("frequency = { value = 30.0", "frequency = { value = 7.0")
    values = {"EJ": 4.0, "EL": 1.0, "phi": 0.0}
    direct = build(system, values)
    energies, vectors = np.linalg.eigh(direct.hamiltonian_ghz)
    order = np.argmax(abs(vectors[[0, 2, 4]]) ** 2, axis=1)  # bare (0,0), (1,0), (2,0)
    assert order.tolist() == [0, 1, 3]
    centre = [energies[1] - energies[0], energies[3] - energies[0]]
    dataset = write_points(tmp_path, centre, labels=["qubit", "second"], fields=())
    found, _, _ = fit(
        write_spec(
            tmp_path,
            [dataset],
            system=system.replace('value = 0.25, unit = "Phi0"', 'value = 0.0, unit = "Phi0"'),
            axis_map=None,
            matching=None,
            family={
                "lines": {
                    "qubit": {"initial": [0, 0], "final": [1, 0]},
                    "second": {"initial": [0, 0], "final": [2, 0]},
                }
            },
            starts=[{"EJ": 4.1, "EL": 0.98}],
        )
    )
    assert found["outcome"] == "solution"
    assert found["solutions"][0]["values"] == pytest.approx({"EJ": 4.0, "EL": 1.0}, abs=1e-6)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"parameters": {"EJ": {"scale": 0.1}, "EC": {"scale": 0.1}}}, "'EC' is not in the system"),
        (
            {"parameters": {"EJ": {"scale": 0.1}, "unused": {"scale": 0.1}}},
            "'unused' enters no construction",
        ),
        ({"parameters": {"EJ": {"scale": 0.1}, "phi": {"scale": 0.1}}}, "not also fitted directly"),
        ({"axis_map": None}, "needs an axis_map"),
        ({"starts": [{"EJ": 4.0, "EL": 1.0}]}, r"starts\[0\] must give exactly"),
        ({"starts": [{**TRUTH, "EJ": 9.0}]}, r"starts\[0\].EJ lies outside its bounds"),
        ({"matching": {**MATCHING, "anchor": "0-7"}}, "anchor '0-7' is not a member"),
        ({"matching": None}, r"automatic assignment needs \[matching\]"),
        ({"family": {"initial": [0], "max_level": 10}}, "level 10 exceeds the system dimension"),
        (
            {"family": {**FAMILY, "lines": {"0-9": {"initial": 0, "final": 1}}}},
            "generated label form",
        ),
        ({"conventions": {"exchange": [["EJ", "phi"]]}}, "exchange pairs name fitted parameters"),
    ],
)
def test_launch_refusals_for_parameters_settings_and_family(tmp_path, change, message):
    system = FLUXONIUM.replace(
        "phi = {",
        'unused = { value = 1.0, unit = "GHz", prior = { kind = "uniform", lower = 0.0, upper = 2.0 } }\nphi = {',
    )
    dataset = sweep_points(tmp_path, np.random.default_rng(17))
    path = write_spec(tmp_path, [dataset], system=system, **change)
    with pytest.raises(ValueError, match=message):
        load_calibration(path)


def test_launch_refusals_for_labels_and_point_fields(tmp_path):
    rng = np.random.default_rng(18)
    labels = np.tile(["0-1", "1-2", "0-2/2"], BIAS.size)
    cases = [
        (
            {"labels": np.where(np.arange(labels.size) < 6, labels, "")},
            None,
            "partly labelled input: 6 of 36",
        ),
        ({"labels": np.where(labels == "1-2", "0-5", labels)}, None, "label '0-5' is not a member"),
        ({"fields": ("amplitude",)}, MATCHING, "needs each point's linewidth"),
        ({"fields": ("linewidth",)}, MATCHING, "needs each point's amplitude"),
    ]
    for options, matching, message in cases:
        dataset = sweep_points(tmp_path, rng, **options)
        with pytest.raises(ValueError, match=message):
            load_calibration(write_spec(tmp_path, [dataset], matching=matching))
    # Without an anchor, automatic assignment needs no amplitude.
    dataset = sweep_points(tmp_path, rng, fields=("linewidth",))
    unanchored = {key: value for key, value in MATCHING.items() if key != "anchor"}
    load_calibration(write_spec(tmp_path, [dataset], matching=unanchored))


def test_calibrate_cli_freezes_inputs_and_records_the_result(tmp_path, runner_repository):
    from entanglix.runner import _cpu_job_context, read_run

    dataset = sweep_points(tmp_path, np.random.default_rng(19))
    path = write_spec(tmp_path, [dataset], timeout_seconds=300)
    command, environment, _ = _cpu_job_context(runner_repository, "entanglix")
    root = tmp_path / "runs"

    def launch(*extra):
        return subprocess.run(
            [*command, "calibrate", str(path), "--runs-root", str(root), *extra],
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )

    preview = launch("--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert json.loads(preview.stdout)["preview"]["procedure"] == {
        "points": "supplied",
        "assignment": "automatic",
        "anchor": "0-1",
    }
    assert not root.exists()
    launched = launch()
    assert launched.returncode == 0, launched.stderr
    record = read_run(Path(json.loads(launched.stdout)["path"]))
    run = Path(record["path"])
    assert record["status"] == "completed" and record["manifest"]["protocol"] == "calibrate"
    assert sorted(p.name for p in (run / "spec").iterdir() if p.is_file()) == [
        "calibrate.toml",
        "points.npz",
        "points.toml",
        "system.toml",
    ]
    result = record["result"]
    assert result["outcome"] == "solution"
    assert result["tolerances"] == {
        "rank_rtol": 1e-6,
        "undetermined_tolerance": 1e-3,
        "jacobian_step": 1e-5,
        "equivalence_rtol": 1e-8,
        "merge_tolerance": 1e-3,
    }
    assert result["parameters"] == {
        "fitted": ["EJ", "EL"],
        "nuisance": ["offset", "period"],
        "flux": "phi",
    }
    assignment = (run / result["artifacts"]["assignment"]["path"]).read_text()
    assert assignment.count("[[peaks]]") == 36
    detached = launch("--detach")
    assert detached.returncode == 0, detached.stderr
    admission = json.loads(detached.stdout)
    assert admission["status"] == "admitted"
    finished = subprocess.run(
        [*command, "read-run", admission["path"], "--wait", "120"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert json.loads(finished.stdout)["result"]["outcome"] == "solution"
    # A refused launch exits 64 with a diagnostic and leaves no attempt.
    attempts = sorted(root.rglob("run.toml"))
    write_spec(tmp_path, [dataset], parameters={"EJ": {"scale": 0.1}, "EC": {"scale": 1.0}})
    refused = launch()
    assert refused.returncode == 64
    assert "'EC' is not in the system file's parameter table" in refused.stderr
    assert sorted(root.rglob("run.toml")) == attempts
