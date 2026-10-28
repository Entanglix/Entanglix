"""Device-update proposals from calibrate runs, and the human apply command."""

import fcntl
import json
import os
import tomllib
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from test_calibrate import (
    BIAS,
    TRUTH,
    VISIBLE,
    displaced,
    lines,
    sweep_points,
    write_points,
    write_spec,
)
from test_calibration_checks import OSCILLATORS, run_checks, source_run

from entanglix import models, runner
from entanglix.cli import main
from entanglix.corrector import load_policy
from entanglix.models import load_system
from entanglix.proposals import NOT_DETERMINED, apply_device_update, propose_device_update
from entanglix.runner import _digest, list_capabilities, read_run

# Two fitted oscillators beside a fixed subsystem whose source and array file are model inputs.
BOUND = (
    OSCILLATORS.replace('id = "oscillators"', 'id = "bound"')
    + """[[model.subsystems]]
id = "z"
source = "fixed.toml"
"""
)
FIXED = """format_version = 1
id = "fixed"
[units]
energy = "GHz"
time = "ns"
[model]
kind = "operators"
dimensions = [2]
basis = "ground_excited"
array_file = "fixed.npz"
hamiltonian = "H0"
operators = { lower = "lowering" }
"""


def oscillator_run(directory, labels, centres, *, system=OSCILLATORS):
    """A labelled-point fit of the oscillator frequencies a and b."""
    if system == BOUND:
        (directory / "fixed.toml").write_text(FIXED)
        np.savez(directory / "fixed.npz", H0=np.diag([0.0, 9.0]), lowering=np.eye(2, k=1))
    size = 3 if system == BOUND else 2
    family = {
        "lines": {
            name: {"initial": [0] * size, "final": [int(k == j) for k in range(size)]}
            for j, name in enumerate(("qa", "qb"))
        }
        | {"sum": {"initial": [0] * size, "final": [1, 1] + [0] * (size - 2)}}
    }
    dataset = write_points(directory, centres, labels=labels, fields=())
    spec = write_spec(
        directory,
        [dataset],
        system=system,
        parameters={"a": {"scale": 1.0}, "b": {"scale": 1.0}},
        axis_map=None,
        matching=None,
        family=family,
        starts=[{"a": 5.2, "b": 6.9}],
    )
    return source_run(directory, spec)


def test_proposal_lists_device_parameters_and_the_checks_that_ran(tmp_path):
    run = source_run(
        tmp_path, write_spec(tmp_path, [sweep_points(tmp_path, np.random.default_rng(5))])
    )
    fit = read_run(run)["result"]["solutions"][0]
    assert fit["outcome"] == "solution"
    verification, _ = run_checks(run, "identifiability")
    proposal = propose_device_update(run)
    path = Path(proposal.pop("path"))
    assert path == run / "proposals/1.toml"
    assert path.stat().st_mode & 0o777 == 0o444
    assert tomllib.loads(path.read_text()) == proposal
    # Device parameters only: the axis map's offset, period and flux are not proposed.
    assert proposal["parameters"] == {
        name: {
            "unit": "GHz",
            "old": TRUTH[name],
            "new": fit["values"][name],
            "uncertainty": fit["uncertainty"][name],
            "uncertainty_over_prior_width": fit["uncertainty_over_prior_width"][name],
        }
        for name in ("EJ", "EL")
    }
    assert proposal["not_proposed"] == {}
    assert proposal["source_run_id"] == run.name
    assert (proposal["solution"], proposal["solution_outcome"]) == (0, "solution")
    assert proposal["model_inputs"] == {
        "system.toml": _digest((tmp_path / "system.toml").read_bytes())
    }
    assert proposal["verification"] == [
        {
            "run_id": verification.name,
            "status": "completed",
            "policy_version": load_policy().version,
            "source_matches": True,
            "checks": [{"name": "identifiability", "solution": 0, "outcome": "descriptive"}],
        }
    ]
    assert Path(propose_device_update(run)["path"]) == run / "proposals/2.toml"

    system = tmp_path / "system.toml"
    authored = tomllib.loads(system.read_text())
    applied = apply_device_update(path, system)
    updated = tomllib.loads(system.read_text())
    for name in ("EJ", "EL"):
        parameter = updated["parameters"].pop(name)
        assert parameter.pop("value") == fit["values"][name]
        source = parameter.pop("source")
        assert source == (
            f"calibration proposal fluxonium/{run.name}/proposals/1.toml, solution 0, "
            f"sha256 {_digest(path.read_bytes())}"
        )
        assert parameter == {
            k: v for k, v in authored["parameters"].pop(name).items() if k != "value"
        }
    # Priors, units, the flux parameter and every other field are unchanged.
    assert updated == authored
    load_system(system)
    assert Path(applied.pop("path")) == run / "proposals/1.applied.toml"
    assert applied["source"]["kind"] == "operator-cli"
    assert applied["system_sha256_after"] == _digest(system.read_bytes())
    assert applied["parameters"] == ["EJ", "EL"]
    assert "recovered" not in applied
    assert tomllib.loads((run / "proposals/1.applied.toml").read_text()) == applied
    with pytest.raises(ValueError, match="already applied"):
        apply_device_update(path, system)
    with pytest.raises(ValueError, match="system.toml changed since the source run froze it"):
        apply_device_update(run / "proposals/2.toml", system)


def test_undetermined_parameters_are_not_proposed_and_apply_binds_every_model_input(tmp_path):
    # One line at a alone determines a; b enters the model but no assigned line.
    run = oscillator_run(tmp_path, ["qa"], [5.0005], system=BOUND)
    proposal = propose_device_update(run)
    assert proposal["solution_outcome"] == "insufficient evidence"
    assert list(proposal["parameters"]) == ["a"]
    assert proposal["not_proposed"] == {"b": {"reason": NOT_DETERMINED}}
    assert proposal["model_inputs"] == {
        name: _digest((tmp_path / name).read_bytes())
        for name in ("system.toml", "fixed.toml", "fixed.npz")
    }
    system = tmp_path / "system.toml"
    for name in proposal["model_inputs"]:
        original = (tmp_path / name).read_bytes()
        (tmp_path / name).write_bytes(original + b"\n")
        with pytest.raises(ValueError, match=f"{name} changed since the source run froze it"):
            apply_device_update(Path(proposal["path"]), system)
        (tmp_path / name).unlink()
        with pytest.raises(ValueError, match=f"{name} changed since the source run froze it"):
            apply_device_update(Path(proposal["path"]), system)
        (tmp_path / name).write_bytes(original)
    with pytest.raises(ValueError, match="not one in the runs root"):
        apply_device_update(Path(proposal["path"]), run / "spec/system.toml")
    apply_device_update(Path(proposal["path"]), system)
    parameters = tomllib.loads(system.read_text())["parameters"]
    assert parameters["a"]["value"] == read_run(run)["result"]["solutions"][0]["values"]["a"]
    assert parameters["b"] == {
        "value": 7.0,
        "unit": "GHz",
        "prior": {"kind": "uniform", "lower": 1.0, "upper": 10.0},
    }


def test_proposal_refusals(tmp_path):
    first, second, third = (tmp_path / name for name in ("several", "none", "sum"))
    for directory in (first, second, third):
        directory.mkdir()
    axis = {
        "flux": "phi",
        "offset": {"lower": -0.5, "upper": 1.5, "scale": 0.01},
        "period": {"lower": 0.8, "upper": 1.6, "scale": 0.01},
    }
    alias = {**displaced(-1), "offset": displaced(-1)["offset"] + TRUTH["period"]}
    dataset = sweep_points(first, np.random.default_rng(16))
    run = source_run(
        first, write_spec(first, [dataset], axis_map=axis, starts=[displaced(1), alias])
    )
    with pytest.raises(ValueError, match="2 distinct solutions; name the one to propose"):
        propose_device_update(run)
    with pytest.raises(ValueError, match="no solution 2"):
        propose_device_update(run, solution=2)
    assert propose_device_update(run, solution=1)["solution"] == 1

    rng = np.random.default_rng(14)
    centres = lines(TRUTH, BIAS) + rng.normal(0, 1e-3, (BIAS.size, len(VISIBLE)))
    centres = np.column_stack([centres, np.full(BIAS.size, 11.5)])
    amplitude = np.tile([*VISIBLE.values(), 2.0], (BIAS.size, 1))
    dataset = write_points(second, centres, np.repeat(BIAS, 4), amplitude=amplitude)
    with pytest.raises(ValueError, match="returned no solution"):
        propose_device_update(source_run(second, write_spec(second, [dataset])))

    # One line at a + b determines neither parameter alone.
    with pytest.raises(ValueError, match="no fitted parameter is numerically determined"):
        propose_device_update(oscillator_run(third, ["sum"], [12.0]))
    verification, _ = run_checks(run, "identifiability", solution=1)
    with pytest.raises(ValueError, match="needs a completed calibrate run"):
        propose_device_update(verification)


def test_apply_refuses_concurrent_edits_and_recovers_an_unpublished_record(tmp_path):
    run = oscillator_run(tmp_path, ["qa", "qb"], [5.0005, 6.9995])
    path = Path(propose_device_update(run)["path"])
    system = tmp_path / "system.toml"
    original = system.read_bytes()

    # An edit made while apply validates the update is kept, and apply refuses.
    validate = models.load_system

    def edit_during_validation(*arguments, **options):
        system.write_bytes(original + b"# edited\n")
        return validate(*arguments, **options)

    with mock.patch.object(models, "load_system", edit_during_validation):
        with pytest.raises(ValueError, match="system.toml changed since the source run froze it"):
            apply_device_update(path, system)
    assert system.read_bytes() == original + b"# edited\n"
    assert not (run / "proposals/1.applied.toml").exists()
    system.write_bytes(original)

    # A competing apply holds the folder.
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="another apply is in progress"):
            apply_device_update(path, system)
    finally:
        os.close(descriptor)
    assert system.read_bytes() == original

    # The file is replaced but its record fails to publish; applying again records it.
    publish = runner.publish_toml

    def refuse_record(target, *arguments, **options):
        if target.name.endswith(".applied.toml"):
            raise PermissionError("injected")
        return publish(target, *arguments, **options)

    with mock.patch.object(runner, "publish_toml", refuse_record):
        with pytest.raises(ValueError, match="could not be published .*injected.*again"):
            apply_device_update(path, system)
    updated = system.read_bytes()
    assert updated != original
    assert not (run / "proposals/1.applied.toml").exists()
    applied = apply_device_update(path, system)
    assert applied["recovered"] is True
    assert applied["system_sha256_before"] == _digest(original)
    assert applied["system_sha256_after"] == _digest(updated) == _digest(system.read_bytes())
    with pytest.raises(ValueError, match="already applied"):
        apply_device_update(path, system)


def test_cli_proposes_and_applies_and_research_tools_cannot_apply(tmp_path, capsys):
    run = oscillator_run(tmp_path, ["qa", "qb"], [5.0005, 6.9995])
    assert main(["propose-device-update", str(run)]) == 0
    proposal = json.loads(capsys.readouterr().out)
    assert list(proposal["parameters"]) == ["a", "b"]
    system = tmp_path / "system.toml"
    assert main(["apply-device-update", proposal["path"], str(system)]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["system_sha256_after"] == _digest(system.read_bytes())
    assert main(["apply-device-update", proposal["path"], str(system)]) == 64
    assert "already applied" in capsys.readouterr().err
    # The research surface, whose tools the MCP server lists, has no write-back.
    assert "apply_device_update" not in list_capabilities()["operations"]
