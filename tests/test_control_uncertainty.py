"""The requested uncertainty diagnostic over declared evaluation samples of a control run."""

import math
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from test_control import EXAMPLE
from test_control import control_files as control_files
from test_control import transmon_control as transmon_control
from test_control_checks import retained_candidate

from entanglix.corrector import Policy, load_policy
from entanglix.corrector.control import control_checks, control_listing, replay_values
from entanglix.corrector.reference import _phases
from entanglix.corrector.verification import check_run
from entanglix.models import load_system
from entanglix.protocols.control import load_control
from entanglix.runner import read_run
from entanglix.solvers import Pulse

CHARGE = "model.offset_charges.1"
EJ = "parameters.EJ"
REFINEMENT = {"junction_with_shunt": {"retained_levels_step": 2, "cutoff_steps": {"theta1": 10}}}


def pulse():
    amplitudes = np.random.default_rng(1).uniform(-0.05, 0.05, (2, 21))
    amplitudes[:, [0, -1]] = 0
    return Pulse(np.linspace(0, 20, 21), amplitudes)


def with_table(path, definition, evaluation):
    path.write_text(tomli_w.dumps(definition | {"evaluation": evaluation}))
    return load_control(path)


@pytest.fixture
def sampled(transmon_control):
    path, definition = transmon_control
    evaluation = {
        "quantities": {CHARGE: "drifting", EJ: "calibrated"},
        "samples": [{CHARGE: 0.0}, {CHARGE: 0.5}, {EJ: 14.985}, {EJ: 14.985, CHARGE: 0.5}],
        "refinement": REFINEMENT,
    }
    return with_table(path, definition, evaluation), path, definition


def diagnose(loaded, names, policy=None):
    return control_checks(
        loaded,
        pulse(),
        {"metrics": {"solver_reported_fidelity": 0.5}},
        names,
        policy or load_policy(),
    )


def model_energies(path, changes):
    """Independently construct a sampled system and return its eigenenergies."""
    document = tomllib.loads(path.read_text())
    for target, value in changes.items():
        if target == EJ:
            document["parameters"]["EJ"]["value"] = value
        else:
            document["model"]["offset_charges"]["1"] = value
    reference = path.parent / "reference" / "system.toml"
    reference.parent.mkdir(exist_ok=True)
    reference.write_text(tomli_w.dumps(document))
    return np.linalg.eigvalsh(load_system(reference).hamiltonian_ghz)


def test_table_is_validated_and_excluded_from_the_problem(transmon_control):
    path, definition = transmon_control
    plain = load_control(path)
    loaded = with_table(
        path, definition, {"quantities": {CHARGE: "drifting"}, "samples": [{CHARGE: 0.0}]}
    )
    assert loaded.problem_bytes == plain.problem_bytes
    for quantity, mode, samples, message in (
        ("frame.rwa", "drifting", [{"frame.rwa": 0.0}], "parameters.<name> or"),
        (CHARGE, "fixed", [{CHARGE: 0.0}], "drifting"),
        (CHARGE, "drifting", [{}], r"samples\[0\]"),
        (CHARGE, "drifting", [{EJ: 1.0}], r"samples\[0\]"),
        (CHARGE, "drifting", [{CHARGE: 0.0}] * 2, "unique"),
        ("parameters.EL", "drifting", [{"parameters.EL": 1.0}], "declared parameter: EL"),
        ("model.offset_charges.2", "drifting", [{"model.offset_charges.2": 0.0}], "node: 2"),
    ):
        evaluation = {"quantities": {quantity: mode}, "samples": samples}
        with pytest.raises(ValueError, match=message):
            with_table(path, definition, evaluation)


def test_nominal_sample_reproduces_replay_without_an_outcome(sampled):
    loaded, _, _ = sampled
    assert {item["name"]: item["available"] for item in control_listing(loaded)["checks"]} == {
        "replay": True,
        "target": True,
        "constraints": True,
        "uncertainty": True,
        "uncertainty_refinement": True,
    }
    replay, item = diagnose(loaded, ("replay", "uncertainty"))
    assert "outcome" not in item and "closed-system" in item["scope"]
    values = item["values"]
    assert values["coverage"].startswith("all 4 declared samples;")
    nominal = values["nominal"]
    assert nominal["values"] == {} and nominal["state_overlaps"] == pytest.approx([1, 1, 1])
    assert nominal["fidelity"] == pytest.approx(replay["fidelity"], abs=1e-12)
    assert nominal["leakage"] == pytest.approx(replay["leakage"], abs=1e-12)
    assert nominal["dimensions"] == [3]
    assert nominal["integration"] == {
        key: replay[key] for key in ("engine", "method", "rtol", "atol")
    }
    fidelities = [sample["fidelity"] for sample in values["samples"]]
    assert [sample["values"] for sample in values["samples"]] == (
        loaded.definition.evaluation.samples
    )
    assert values["minimum_fidelity"] == min(fidelities)
    assert values["minimum_sample"] == fidelities.index(min(fidelities))
    assert values["ambiguous_samples"] == []


def test_minimum_excludes_a_lower_nominal_model(sampled):
    loaded, path, definition = sampled
    single = with_table(
        path, definition, {"quantities": {CHARGE: "drifting"}, "samples": [{CHARGE: 0.0}]}
    )
    (item,) = diagnose(single, ("uncertainty",))
    values = item["values"]
    assert values["nominal"]["fidelity"] < values["samples"][0]["fidelity"]
    assert values["minimum_fidelity"] == values["samples"][0]["fidelity"]
    assert values["minimum_sample"] == 0


def test_drifting_only_table_uses_the_declared_frequencies(tmp_path):
    path = tmp_path / "control.toml"
    for name in ("system.toml", "control.toml"):
        path.with_name(name).write_bytes((EXAMPLE.parent / name).read_bytes())
    definition = tomllib.loads(path.read_text())
    loaded = with_table(
        path,
        definition,
        {"quantities": {CHARGE: "drifting"}, "samples": [{CHARGE: 0.0}, {CHARGE: 0.5}]},
    )
    (item,) = diagnose(loaded, ("uncertainty",))
    frame = definition["frame"]["frequencies_ghz"]
    carriers = {channel["id"]: channel["frequency_ghz"] for channel in definition["channels"]}
    for sample in item["values"]["samples"]:
        assert sample["frame_frequencies_ghz"] == frame
        assert sample["carrier_frequencies_ghz"] == carriers
        assert sample["calibrated"] == []


def test_calibrated_sample_matches_an_independent_replay(sampled, tmp_path):
    loaded, path, definition = sampled
    (item,) = diagnose(loaded, ("uncertainty",))
    sample = item["values"]["samples"][2]
    assert sample["values"] == {EJ: 14.985} and sample["calibrated"] == [EJ]
    energies = model_energies(path.with_name("system.toml"), {EJ: 14.985})
    frequency = energies[1] - energies[0]
    assert math.isclose(
        sample["frame_frequencies_ghz"]["junction_with_shunt"], frequency, rel_tol=1e-12
    )
    # An explicit input at the hand-computed frequency, aligned by operator elements.
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "system.toml").write_bytes(path.with_name("system.toml").read_bytes())
    document = tomllib.loads(EXAMPLE.read_text())
    document["parameters"] = {"EJ": 14.985}
    document["frame"]["frequencies_ghz"] = {"junction_with_shunt": float(frequency)}
    for channel in document["channels"]:
        channel["frequency_ghz"] = float(frequency)
    (explicit / "control.toml").write_text(tomli_w.dumps(document))
    independent = load_control(explicit / "control.toml")
    names = sorted(loaded.system.operators)
    phases = _phases(
        [loaded.system.operators[name] for name in names],
        [independent.system.operators[name] for name in names],
        3,
    )
    aligned = replace(
        independent,
        system=replace(
            independent.system,
            drives={
                name: phases.conj()[:, None] * operator * phases[None, :]
                for name, operator in independent.system.drives.items()
            },
        ),
    )
    reference = replay_values(aligned, pulse(), load_policy())
    assert sample["fidelity"] == pytest.approx(reference["fidelity"], abs=1e-10)
    assert sample["leakage"] == pytest.approx(reference["leakage"], abs=1e-10)


def test_combined_sample_is_recalibrated_for_its_calibrated_change_only(sampled):
    loaded, path, _ = sampled
    (item,) = diagnose(loaded, ("uncertainty",))
    calibrated, combined = item["values"]["samples"][2:]
    assert combined["calibrated"] == [EJ]
    assert combined["frame_frequencies_ghz"] == calibrated["frame_frequencies_ghz"]
    energies = model_energies(path.with_name("system.toml"), {EJ: 14.985, CHARGE: 0.5})
    frequency = combined["frame_frequencies_ghz"]["junction_with_shunt"]
    assert abs(frequency - (energies[1] - energies[0])) > 1e-6
    assert combined["fidelity"] != calibrated["fidelity"]


def test_calibrated_samples_need_a_followed_transition(transmon_control, tmp_path):
    path, definition = transmon_control
    evaluation = {"quantities": {EJ: "calibrated"}, "samples": [{EJ: 14.985}]}
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "system.toml").write_bytes(path.with_name("system.toml").read_bytes())
    unfollowed = definition | {"channels": [dict(c) for c in definition["channels"]]}
    del unfollowed["channels"][0]["follows"]
    unfollowed["channels"][0]["frequency_ghz"] = 5.482065688844835
    for loaded in (
        with_table(explicit / "control.toml", tomllib.loads(EXAMPLE.read_text()), evaluation),
        with_table(path, unfollowed, evaluation),
    ):
        listing = {item["name"]: item for item in control_listing(loaded)["checks"]}
        assert not listing["uncertainty"]["available"]
        (item,) = diagnose(loaded, ("uncertainty",))
        assert item["outcome"] == "inapplicable"
        assert "named frame transition with every carrier following it" in item["reason"]
        assert listing["uncertainty"]["reason"] == item["reason"]


def test_unsupported_state_correspondence_is_inapplicable(control_files):
    path, definition = control_files
    system = path.with_name("system.toml")
    document = tomllib.loads(system.read_text())
    document["parameters"] = {"gamma": {"value": 0.01, "unit": "1/ns"}}
    document["dissipators"][0]["rate"] = {"parameter": "gamma"}
    system.write_text(tomli_w.dumps(document))
    loaded = with_table(
        path,
        definition,
        {"quantities": {"parameters.gamma": "drifting"}, "samples": [{"parameters.gamma": 0.02}]},
    )
    pulse_ = Pulse(np.linspace(0, 2, 5), np.zeros((2, 5)))
    (item,) = control_checks(loaded, pulse_, {}, ("uncertainty",), load_policy())
    assert item["outcome"] == "inapplicable"
    assert "single periodic circuit coordinate" in item["reason"]


def test_refinement_covers_every_sample_and_reports_the_minimum_change(sampled):
    loaded, path, definition = sampled
    base, refined = diagnose(loaded, ("uncertainty", "uncertainty_refinement"))
    assert "outcome" not in refined and refined["refinement"] == REFINEMENT
    samples, values = base["values"]["samples"], refined["values"]
    assert len(values["samples"]) == len(samples) == 4
    for before, after in zip(
        [base["values"]["nominal"], *samples], [values["nominal"], *values["samples"]]
    ):
        # Each refined row keeps its sample's identity and resolved drive.
        for key in ("values", "calibrated", "frame_frequencies_ghz", "carrier_frequencies_ghz"):
            assert after[key] == before[key]
        assert len(after["state_overlaps"]) == 3 and after["dimensions"] == [5]
        assert after["integration"] == before["integration"]
        assert after["fidelity_change"] == pytest.approx(after["fidelity"] - before["fidelity"])
    minimum = min(sample["fidelity"] for sample in values["samples"])
    assert values["minimum_fidelity"] == minimum
    assert values["unrefined_minimum_fidelity"] == base["values"]["minimum_fidelity"]
    assert values["minimum_change"] == pytest.approx(minimum - base["values"]["minimum_fidelity"])
    assert values["largest_sample_change"] == max(
        abs(sample["fidelity_change"]) for sample in values["samples"]
    )
    undeclared = with_table(
        path, definition, {**loaded.definition.evaluation.model_dump(), "refinement": {}}
    )
    (item,) = diagnose(undeclared, ("uncertainty_refinement",))
    assert item["outcome"] == "inapplicable" and "no refinement declared" in item["reason"]


def test_refinement_extends_canonical_excitation_numbers_only(sampled):
    loaded, path, definition = sampled
    evaluation = loaded.definition.evaluation.model_dump()
    canonical = definition | {
        "frame": definition["frame"] | {"excitation_numbers": {"junction_with_shunt": [0, 1, 2]}}
    }
    (item,) = diagnose(with_table(path, canonical, evaluation), ("uncertainty_refinement",))
    assert "outcome" not in item
    assert all(row["dimensions"] == [5] for row in item["values"]["samples"])
    reordered = definition | {
        "frame": definition["frame"] | {"excitation_numbers": {"junction_with_shunt": [0, 1, 3]}}
    }
    loaded = with_table(path, reordered, evaluation)
    listing = {item["name"]: item for item in control_listing(loaded)["checks"]}
    assert listing["uncertainty"]["available"]
    assert not listing["uncertainty_refinement"]["available"]
    (item,) = diagnose(loaded, ("uncertainty_refinement",))
    assert item["outcome"] == "inapplicable"
    assert "canonical frame excitation numbers" in item["reason"]


def test_ambiguous_correspondence_is_reported(sampled):
    loaded, _, _ = sampled
    policy = load_policy()
    strict = Policy(
        policy.version, {**policy.thresholds, "uncertainty": {"min_state_overlap": 0.999}}
    )
    (item,) = diagnose(loaded, ("uncertainty",), strict)
    flagged = item["values"]["ambiguous_samples"]
    assert flagged and not item["values"]["nominal"]["ambiguous_correspondence"]
    assert all(min(item["values"]["samples"][i]["state_overlaps"][:2]) < 0.999 for i in flagged)


def test_requested_refinement_replacement_is_frozen(sampled, runner_repository, tmp_path):
    loaded, _, _ = sampled
    run = retained_candidate((loaded, pulse()), tmp_path)
    listing = read_run(run)["check_listing"]["checks"]
    assert [item["name"] for item in listing][-2:] == ["uncertainty", "uncertainty_refinement"]
    replacement = tmp_path / "refinement.toml"
    step = {"retained_levels_step": 1, "cutoff_steps": {"theta1": 4}}
    replacement.write_text(tomli_w.dumps({"refinement": {"junction_with_shunt": step}}))
    checked = check_run(
        run,
        ("uncertainty", "uncertainty_refinement"),
        repository=runner_repository,
        refinement=replacement,
    )
    assert checked["exit_code"] == 0
    report = read_run(Path(checked["path"]))["result"]["report"]
    assert report["refinement_origin"] == "request"
    assert report["checks"][1]["refinement"] == {"junction_with_shunt": step}
    assert [item.get("outcome") for item in report["checks"]] == [None, None]
