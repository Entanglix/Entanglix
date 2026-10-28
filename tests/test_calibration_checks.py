"""Requested calibration checks: holdout, identifiability and refinement, and their requests."""

import json
import math
import os
import subprocess
import tomllib
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from test_calibrate import (
    BIAS,
    EXTRACTION,
    SIGMA,
    TRUTH,
    VISIBLE,
    displaced,
    lines,
    sweep_map,
    sweep_points,
    write_points,
    write_spec,
)

from entanglix.corrector import create_verification_run, load_policy
from entanglix.corrector.calibration import (
    _agreement,
    _training,
    calibration_request,
    holdout_form,
    holdout_values,
)
from entanglix.corrector.verification import execute_verification, verification_result
from entanglix.measurements import load_dataset, split_dataset
from entanglix.protocols import _calibrate_job
from entanglix.protocols.calibrate import compile_calibration, load_calibration
from entanglix.runner import (
    CodeSnapshot,
    _cpu_job_context,
    capture_environment,
    create_run,
    read_run,
    record_finish,
    record_start,
)

ROOT = Path(__file__).resolve().parents[1]
CODE = CodeSnapshot("fixture", b"", {})
# Flux 0 to 0.48 in steps of 0.03 under the generating axis map.
PARENT = TRUTH["offset"] + TRUTH["period"] * np.linspace(0, 0.48, 17)
HOLDOUT = (0.53, 0.68)


def source_run(directory, spec):
    """A completed calibrate attempt, fitted in this process from the frozen inputs."""
    frozen = directory / "frozen"
    frozen.mkdir()
    gate = compile_calibration(spec, frozen, [], ROOT)
    with create_run(
        system_id=gate["system_id"],
        protocol="calibrate",
        inputs={name: Path(path) for name, path in gate["inputs"].items()},
        environment=gate["environment"],
        code=CODE,
        engine=gate["engine"],
        device="cpu",
        needs=gate["needs"],
        runs_root=directory / "runs",
        launch={
            key: gate[key] for key in ("kind", "selection", "protocol_sha256", "calibration_sha256")
        },
    ) as owner:
        record_start(owner, process_group=os.getpgrp(), gpu_devices=[])
        with mock.patch.dict(os.environ, {"ENTANGLIX_RUN_DIR": str(owner.path)}):
            _calibrate_job.execute(owner.path / "spec" / "calibrate.toml")
        record_finish(owner, status="completed", exit_code=0, wall_seconds=0, gpu_seconds=0)
    return owner.path


def run_checks(source, *checks, **request):
    """Run named checks in this process; returns the verification attempt and its reports."""
    with create_verification_run(
        source,
        checks=checks,
        environment=capture_environment(ROOT / "uv.lock"),
        code=CODE,
        runs_root=source.parent.parent,
        **request,
    ) as owner:
        record_start(owner, process_group=os.getpgrp(), gpu_devices=[])
        execute_verification(owner.path)
        record_finish(owner, status="completed", exit_code=0, wall_seconds=0, gpu_seconds=0)
    report = verification_result(owner.path)["report"]
    return owner.path, {item["name"]: item for item in report["checks"]}


def split_points(directory, seed, **options):
    """A swept points dataset split into a holdout part and the training remainder."""
    directory.mkdir(exist_ok=True)
    parent = sweep_points(directory, np.random.default_rng(seed), bias=PARENT, **options)
    split_dataset(
        directory / parent,
        *HOLDOUT,
        holdout=directory / "holdout.toml",
        remainder=directory / "training.toml",
    )
    return directory / parent


@pytest.fixture
def mapped(tmp_path):
    """A fit to a training map, with a separate holdout map over a further bias range."""
    training = sweep_map(tmp_path, np.random.default_rng(3))
    holdout = tmp_path / "separate"
    holdout.mkdir()
    sweep_map(holdout, np.random.default_rng(4), bias=PARENT[12:])
    spec = write_spec(tmp_path, [training], extraction=EXTRACTION)
    return source_run(tmp_path, spec), holdout / "map.toml"


def test_holdout_map_is_supplied_frozen_and_scored(mapped):
    run, holdout = mapped
    # The fitting job's frozen inputs hold the training map only.
    assert sorted(p.name for p in (run / "spec").iterdir() if p.is_file()) == [
        "calibrate.toml",
        "map.npz",
        "map.toml",
        "system.toml",
    ]
    listing = read_run(run)["check_listing"]
    assert [item["name"] for item in listing["checks"]] == [
        "holdout",
        "identifiability",
        "refinement",
    ]
    assert listing["holdout_forms"] == ["map", "labelled points", "unlabelled points"]
    verification, items = run_checks(run, "holdout", holdout=holdout)
    item = items["holdout"]
    frozen = load_dataset(verification / "spec" / "holdout" / "map.toml")
    assert item["dataset"] == {
        "file": "holdout/map.toml",
        "id": "map",
        "sha256": load_dataset(holdout).sha256,
    }
    assert frozen.sha256 == item["dataset"]["sha256"]
    request = tomllib.loads((verification / "spec" / "check-request.toml").read_text())
    assert request["solution"] == 0 and request["holdout"]["dataset"] == "holdout/map.toml"
    # Descriptive without a declared expectation.
    assert "outcome" not in item
    values = item["values"]
    assert values["form"] == "map"
    counts = values["counts"]
    # Near half flux some lines fall below the map's 2 GHz window: 12 peaks in 5 columns.
    assert counts == {
        "peaks": 12,
        "assigned": 12,
        "not_determined": 0,
        "unresolved": 0,
        "unexplained": 0,
        "scored": 12,
        "extraction_failures": 0,
    }
    assert len(values["peaks"]) == 12 and len(values["columns"]) == 5
    assert values["coverage_over_peaks"] == {"within_1": 7 / 12, "within_2": 1.0}
    assert values["coverage_over_scored"] == values["coverage_over_peaks"]
    assert values["mean_squared_standardised_residual"] == pytest.approx(1.095, abs=1e-3)
    row = values["peaks"][1]
    assert row["line"] == "0-1" and row["determined"]
    # The prediction uncertainty adds the propagated covariance to the centre uncertainty.
    assert row["prediction_uncertainty_ghz"] > row["uncertainty_ghz"]
    assert read_run(run)["verification"][0]["checks"][0]["name"] == "holdout"


def test_a_displaced_model_lowers_holdout_coverage(mapped):
    run, holdout = mapped
    loaded = load_calibration(run / "spec" / "calibrate.toml")
    training = _training(loaded, run, read_run(run)["result"], 0)
    problem, record, assignment, theta, residual = training
    projection = tuple(record["datasets"]["map.toml"]["projection"])
    dataset = load_dataset(holdout)

    def values(training):
        return holdout_values(
            loaded, "map.toml", dataset, "map", training, load_policy(), projection
        )

    fitted = values(training)
    shifted = theta.copy()
    shifted[0] *= 1.01
    moved = values((problem, record, assignment, shifted, residual))
    assert fitted["counts"]["peaks"] == moved["counts"]["peaks"]
    assert moved["coverage_over_peaks"]["within_2"] < fitted["coverage_over_peaks"]["within_2"] / 2


def test_a_declared_expectation_is_checked_as_declared(mapped):
    run, holdout = mapped
    _, items = run_checks(run, "holdout", holdout=holdout, expect_coverage=[2, 0.5])
    item = items["holdout"]
    assert item["outcome"] == "pass" and item["tolerance"] == 0.5
    assert item["residual"] == item["values"]["coverage_over_peaks"]["within_2"]
    assert item["expectation"] == {"within": 2, "minimum": 0.5}
    _, items = run_checks(run, "holdout", holdout=holdout, expect_coverage=[1, 1.0])
    assert items["holdout"]["outcome"] == "fail"
    assert items["holdout"]["failure_meaning"] == "declared expectation violated"
    for expectation, message in [
        ([3, 0.5], "within 1 or 2"),
        ([2, 0.0], "minimum in"),
        ([2, 1.5], "minimum in"),
    ]:
        with pytest.raises(ValueError, match=message):
            run_checks(run, "holdout", holdout=holdout, expect_coverage=expectation)
    with pytest.raises(ValueError, match="needs a holdout dataset"):
        run_checks(run, "holdout", expect_coverage=[2, 0.5])
    with pytest.raises(ValueError, match="used only by the holdout check"):
        run_checks(run, "identifiability", holdout=holdout)


def test_identifiability_agrees_with_the_fit_and_records_both_tolerances(mapped):
    run, _ = mapped
    _, items = run_checks(run, "identifiability")
    item = items["identifiability"]
    assert "outcome" not in item
    fit = read_run(run)["result"]
    assert item["tolerances"]["fit"] == fit["tolerances"]
    assert item["tolerances"]["check"] == load_policy().thresholds["calibration"]
    agreement = item["values"]["agreement"]
    assert agreement["agrees"] and agreement["rank_matches"] and agreement["determined_matches"]
    assert agreement["largest_uncertainty_difference"] <= 1e-6
    solution = fit["solutions"][0]
    assert item["values"]["uncertainty"] == pytest.approx(solution["uncertainty"], rel=1e-6)
    # A report whose uncertainties differ by 0.1% does not agree.
    values = {key: value for key, value in item["values"].items() if key != "agreement"}
    widened = {
        **solution,
        "uncertainty": {n: 1.001 * s for n, s in solution["uncertainty"].items()},
    }
    differing = _agreement(values, solution["outcome"], widened, 1e-6)
    assert not differing["agrees"]
    assert differing["largest_uncertainty_difference"] == pytest.approx(1 - 1 / 1.001)


def test_identifiability_agrees_on_an_undetermined_fit(tmp_path):
    centre = lines({"EJ": 4.0, "EL": 1.0}, visible=[(0, 1, 1)])[0]
    dataset = write_points(tmp_path, centre, labels=["0-1"], fields=())
    spec = write_spec(
        tmp_path, [dataset], axis_map=None, matching=None, starts=[{"EJ": 4.2, "EL": 0.95}]
    )
    run = source_run(tmp_path, spec)
    _, items = run_checks(run, "identifiability", "holdout", "refinement")
    values = items["identifiability"]["values"]
    assert values["rank"] == 1 and values["determined"] == []
    agreement = values["agreement"]
    assert agreement["agrees"] and agreement["undetermined_difference"] <= 1e-6
    assert items["holdout"]["outcome"] == "inapplicable"
    assert items["holdout"]["reason"] == "no holdout dataset supplied with the check request"
    assert items["refinement"]["outcome"] == "inapplicable"
    assert items["refinement"]["reason"].startswith("no refinement declared")


def test_split_holdout_points_are_matched_after_a_points_fit(tmp_path):
    split_points(tmp_path, 5)
    run = source_run(tmp_path, write_spec(tmp_path, ["training.toml"]))
    holdout = tmp_path / "holdout.toml"
    assert not any("holdout" in p.name for p in (run / "spec").iterdir())
    _, items = run_checks(run, "holdout", holdout=holdout)
    values = items["holdout"]["values"]
    assert values["form"] == "unlabelled points"
    assert values["counts"] == {
        "peaks": 15,
        "assigned": 15,
        "not_determined": 0,
        "unresolved": 0,
        "unexplained": 0,
        "scored": 15,
    }
    # Each column's peaks match the lines that generated them.
    assert [row["line"] for row in values["peaks"]] == ["0-1", "1-2", "0-2/2"] * 5
    assert values["coverage_over_peaks"] == {"within_1": 11 / 15, "within_2": 1.0}


def test_labelled_holdout_points_are_evaluated_after_a_labelled_fit(tmp_path):
    labels = np.tile(["0-1", "1-2", "0-2/2"], PARENT.size)
    split_points(tmp_path, 6, labels=labels, fields=())
    run = source_run(tmp_path, write_spec(tmp_path, ["training.toml"], matching=None))
    _, items = run_checks(run, "holdout", holdout=tmp_path / "holdout.toml")
    values = items["holdout"]["values"]
    assert values["form"] == "labelled points"
    assert values["counts"]["assigned"] == values["counts"]["peaks"] == 15
    assert [row["line"] for row in values["peaks"]] == labels[:15].tolist()


def test_holdout_uses_only_the_recorded_procedure(tmp_path):
    labelled = tmp_path / "labelled"
    labels = np.tile(["0-1", "1-2", "0-2/2"], PARENT.size)
    split_points(labelled, 7, labels=labels, fields=())
    loaded = load_calibration(write_spec(labelled, ["training.toml"], matching=None))
    unlabelled = tmp_path / "unlabelled"
    unlabelled.mkdir()
    sweep_points(unlabelled, np.random.default_rng(8), bias=PARENT[12:])
    form, reason = holdout_form(loaded, load_dataset(unlabelled / "points.toml"))
    assert form is None and "recorded no matching settings" in reason
    maps = tmp_path / "maps"
    maps.mkdir()
    sweep_map(maps, np.random.default_rng(9), bias=PARENT[12:])
    form, reason = holdout_form(loaded, load_dataset(maps / "map.toml"))
    assert form is None and "no extraction settings" in reason
    form, reason = holdout_form(loaded, load_dataset(labelled / "holdout.toml"))
    assert (form, reason) == ("labelled points", None)


def test_holdout_datasets_sharing_bias_points_are_refused(tmp_path):
    parent = split_points(tmp_path, 10)
    # A second split of the same parent: its holdout part overlaps the training remainder.
    split_dataset(
        parent, 0.2, 0.3, holdout=tmp_path / "other.toml", remainder=tmp_path / "rest.toml"
    )
    result = {"solutions": [{}]}

    def request(training, holdout):
        loaded = load_calibration(write_spec(tmp_path, [training]))
        return calibration_request(loaded, result, ["holdout"], holdout=tmp_path / holdout)

    for training, holdout, message in [
        ("training.toml", "training.toml", "is the fitting input"),
        ("training.toml", "points.toml", "is the parent of the fitting input"),
        ("points.toml", "holdout.toml", "is a split part of the fitting input"),
        ("training.toml", "other.toml", "overlaps the fitting input"),
    ]:
        with pytest.raises(ValueError, match=message):
            request(training, holdout)
    fields, files = request("training.toml", "holdout.toml")
    assert fields["holdout"]["dataset"] == "holdout/holdout.toml"
    assert sorted(files) == ["holdout/holdout.npz", "holdout/holdout.toml"]


def test_holdout_label_refusals(tmp_path):
    loaded = load_calibration(write_spec(tmp_path, [split_points(tmp_path, 11).name]))
    result = {"solutions": [{}]}
    labels = np.tile(["0-1", "1-2", "0-2/2"], 5)
    for given, message in [
        (np.where(np.arange(15) < 3, labels, ""), "partly labelled holdout dataset: 3 of 15"),
        (np.where(labels == "1-2", "0-7", labels), "holdout label '0-7' is not a member"),
    ]:
        directory = tmp_path / f"holdout-{message[:7]}"
        directory.mkdir()
        centres = lines(TRUTH, PARENT[12:]).ravel()
        write_points(directory, centres, np.repeat(PARENT[12:], 3), labels=given)
        with pytest.raises(ValueError, match=message):
            calibration_request(loaded, result, ["holdout"], holdout=directory / "points.toml")


def test_several_solutions_need_a_named_solution(tmp_path):
    dataset = sweep_points(tmp_path, np.random.default_rng(16))
    axis = {
        "flux": "phi",
        "offset": {"lower": -0.5, "upper": 1.5, "scale": 0.01},
        "period": {"lower": 0.8, "upper": 1.6, "scale": 0.01},
    }
    alias = {**displaced(-1), "offset": displaced(-1)["offset"] + TRUTH["period"]}
    spec = write_spec(tmp_path, [dataset], axis_map=axis, starts=[displaced(1), alias])
    run = source_run(tmp_path, spec)
    assert read_run(run)["result"]["outcome"] == "several distinct solutions"
    with pytest.raises(ValueError, match="2 distinct solutions; name the one to check"):
        run_checks(run, "identifiability")
    with pytest.raises(ValueError, match="no solution 2"):
        run_checks(run, "identifiability", solution=2)
    _, items = run_checks(run, "identifiability", solution=1)
    assert items["identifiability"]["solution"] == 1
    assert items["identifiability"]["values"]["agreement"]["agrees"]


def test_checks_are_inapplicable_without_fitted_parameters(tmp_path):
    rng = np.random.default_rng(14)
    centres = lines(TRUTH, BIAS) + rng.normal(0, 1e-3, (BIAS.size, len(VISIBLE)))
    centres = np.column_stack([centres, np.full(BIAS.size, 11.5)])
    amplitude = np.tile([*VISIBLE.values(), 2.0], (BIAS.size, 1))
    dataset = write_points(tmp_path, centres, np.repeat(BIAS, 4), amplitude=amplitude)
    run = source_run(tmp_path, write_spec(tmp_path, [dataset]))
    assert read_run(run)["result"]["solutions"] == []
    listing = read_run(run)["check_listing"]
    assert {item["available"] for item in listing["checks"]} == {False}
    _, items = run_checks(run, "holdout", "identifiability", "refinement")
    assert {item["reason"] for item in items.values()} == {
        "the source run returned no fitted parameters"
    }
    with pytest.raises(ValueError, match="no solution 0"):
        run_checks(run, "identifiability", solution=0)


def test_refinement_reports_standardised_shifts(tmp_path):
    dataset = sweep_points(tmp_path, np.random.default_rng(21))
    declared = {
        "f": {"retained_levels_step": 1, "cutoff_steps": {"cutoff": 10}},
        "r": {"retained_levels_step": 1},
    }
    run = source_run(tmp_path, write_spec(tmp_path, [dataset], refinement=declared))
    _, items = run_checks(run, "refinement")
    item = items["refinement"]
    assert "outcome" not in item
    values = item["values"]
    assert values["dimensions"] == [5, 2] and values["refined_dimensions"] == [6, 3]
    assert values["assigned"] == 35 and set(values["lines"]) == {"0-1", "1-2", "0-2/2"}
    # A converged fluxonium cutoff moves no assigned line by a measurable fraction of σ.
    assert values["largest_standardised_shift"] < 1e-3
    assert values["refined_chi2"] == pytest.approx(values["chi2"], rel=1e-3)
    replacement = tmp_path / "replacement.toml"
    replacement.write_text(
        "[refinement.f]\nretained_levels_step = 1\ncutoff_steps = { cutoff = 1 }\n[refinement.r]\nretained_levels_step = 1\n"
    )
    _, items = run_checks(run, "refinement", refinement=replacement)
    assert items["refinement"]["outcome"] == "inapplicable"
    assert "policy minimum" in items["refinement"]["reason"]


def test_listings_mark_an_invalid_refinement_unavailable(tmp_path):
    # A cutoff step below the policy minimum annotates the listings; the calibration still runs.
    dataset = sweep_points(tmp_path, np.random.default_rng(23))
    declared = {
        "f": {"retained_levels_step": 1, "cutoff_steps": {"cutoff": 1}},
        "r": {"retained_levels_step": 1},
    }
    spec = write_spec(tmp_path, [dataset], refinement=declared)
    preview = tmp_path / "preview"
    preview.mkdir()
    listings = [compile_calibration(spec, preview, [], ROOT)["preview"]]
    run = source_run(tmp_path, spec)
    assert read_run(run)["result"]["outcome"] == "solution"
    listings.append(read_run(run)["check_listing"])
    for listing in listings:
        holdout, identifiability, refinement = listing["checks"]
        assert holdout["available"] and identifiability["available"]
        assert not refinement["available"] and "policy minimum" in refinement["reason"]
    assert "solutions" not in listings[0] and listings[1]["solutions"] == 1
    _, items = run_checks(run, "refinement")
    assert items["refinement"]["reason"] == listings[1]["checks"][2]["reason"]


OSCILLATORS = """format_version = 1
id = "oscillators"
[units]
energy = "GHz"
time = "ns"
[parameters]
a = { value = 5.0, unit = "GHz", prior = { kind = "uniform", lower = 1.0, upper = 10.0 } }
b = { value = 7.0, unit = "GHz", prior = { kind = "uniform", lower = 1.0, upper = 10.0 } }
[model]
kind = "composite"
[[model.subsystems]]
id = "x"
kind = "oscillator"
frequency = { parameter = "a" }
retained_levels = 2
[[model.subsystems]]
id = "y"
kind = "oscillator"
frequency = { parameter = "b" }
retained_levels = 2
"""


def test_holdout_scores_a_determined_combination_of_undetermined_parameters(tmp_path):
    # One line at a + b determines only the sum. A holdout line at a + b has a finite prediction
    # uncertainty; a line at a alone is assigned but not scored.
    family = {
        "lines": {
            "sum": {"initial": [0, 0], "final": [1, 1]},
            "qa": {"initial": [0, 0], "final": [1, 0]},
        }
    }
    dataset = write_points(tmp_path, [12.0], labels=["sum"], fields=())
    spec = write_spec(
        tmp_path,
        [dataset],
        system=OSCILLATORS,
        parameters={"a": {"scale": 1.0}, "b": {"scale": 1.0}},
        axis_map=None,
        matching=None,
        family=family,
        starts=[{"a": 5.2, "b": 6.9}],
    )
    run = source_run(tmp_path, spec)
    fit = read_run(run)["result"]["solutions"][0]
    assert fit["outcome"] == "insufficient evidence" and fit["determined"] == []
    separate = tmp_path / "separate"
    separate.mkdir()
    write_points(separate, [12.0005, 5.0], labels=["sum", "qa"], fields=())
    _, items = run_checks(run, "holdout", holdout=separate / "points.toml")
    values = items["holdout"]["values"]
    assert values["counts"] == {
        "peaks": 2,
        "assigned": 2,
        "not_determined": 1,
        "unresolved": 0,
        "unexplained": 0,
        "scored": 1,
    }
    total, alone = values["peaks"]
    # The fit determines a + b to σ, which adds in quadrature to the holdout centre's σ.
    spread = math.sqrt(2) * SIGMA
    assert total["determined"]
    assert total["prediction_uncertainty_ghz"] == pytest.approx(spread, rel=1e-6)
    assert total["standardised_residual"] == pytest.approx(0.5e-3 / spread, rel=1e-6)
    assert alone["line"] == "qa" and not alone["determined"]
    assert "prediction_uncertainty_ghz" not in alone
    assert values["coverage_over_peaks"] == {"within_1": 0.5, "within_2": 0.5}
    assert values["coverage_over_scored"] == {"within_1": 1.0, "within_2": 1.0}


def test_check_cli_runs_calibration_checks(tmp_path, runner_repository):
    split_points(tmp_path, 22)
    run = source_run(tmp_path, write_spec(tmp_path, ["training.toml"]))
    command, environment, _ = _cpu_job_context(runner_repository, "entanglix")

    def check(*arguments):
        return subprocess.run(
            [*command, "check", str(run), *arguments],
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )

    listing = check()
    assert listing.returncode == 0, listing.stderr
    assert json.loads(listing.stdout)["solutions"] == 1
    holdout = str(tmp_path / "holdout.toml")
    completed = check(
        "holdout", "identifiability", "--holdout", holdout, "--expect-coverage", "2", "0.5"
    )
    assert completed.returncode == 0, completed.stderr
    (history,) = read_run(run)["verification"]
    assert [item["name"] for item in history["checks"]] == ["holdout", "identifiability"]
    assert history["checks"][0]["outcome"] == "pass"
    attempts = sorted((tmp_path / "runs").rglob("run.toml"))
    refused = check("holdout", "--holdout", str(tmp_path / "training.toml"))
    assert refused.returncode == 64
    assert "the holdout dataset is the fitting input 'training.toml'" in refused.stderr
    assert sorted((tmp_path / "runs").rglob("run.toml")) == attempts
