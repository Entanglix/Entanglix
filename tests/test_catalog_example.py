"""The catalog client pauses for promotion and reuses the exact device and pulse."""

import json
import os
import shutil
import subprocess
import sys
import tomllib

import numpy as np
import pytest
import tomli_w
from test_catalog import ROOT, completed, npz
from test_control_starting_pulse import starting_repository as starting_repository

from entanglix.protocols.control import compile_control

CLIENT = ROOT / "examples/catalog/reuse.py"


@pytest.mark.parametrize("via", ["cli", "mcp"])
def test_example_stages_waits_for_operator_and_reuses(via, starting_repository, tmp_path):
    if via == "mcp":
        pytest.importorskip("mcp")
    source_inputs = tmp_path / "source"
    source_inputs.mkdir()
    shutil.copyfile(ROOT / "examples/transmon-codesign/system.toml", source_inputs / "system.toml")
    control = tomllib.loads((ROOT / "examples/transmon-codesign/control.toml").read_text())
    control["solver"] = {"backend": "starting"}  # Test-only solver echoes the supplied pulse.
    control["pulse"]["seed"] = 19
    (source_inputs / "control.toml").write_text(tomli_w.dumps(control))
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    gate = compile_control(source_inputs / "control.toml", frozen)
    source = completed(
        tmp_path / "runs",
        gate,
        "control",
        {
            "pulse": (
                "pulse.npz",
                npz(
                    kind="zero_order",
                    times_ns=np.linspace(0, 6, 21),
                    amplitudes_ghz=np.zeros((2, 21)),
                ),
            )
        },  # fmt: skip
        kind="control",
    )
    root = source.parent.parent
    environment = os.environ | {
        "PYTHONPATH": str(starting_repository / "src"),
        "ENTANGLIX_RUNNER_EXTRAS": "",
    }

    def invoke(*arguments):
        result = subprocess.run(
            [sys.executable, *map(str, arguments)], capture_output=True, text=True, env=environment
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    staged = invoke(CLIENT, "stage", "--via", via, "--source", source)
    assert staged["promotion_command"].startswith("entanglix promote " + staged["staged_id"])
    claim = tomllib.loads(
        (root / "_catalog/staging" / staged["staged_id"] / "record.toml").read_text()
    )["claim"]
    assert "60 and 90 fF" in claim and "ranking unresolved" in claim
    assert not (root / "_catalog/entries").exists()
    # The fixture acts as the operator; the example client has no promotion operation.
    entry = invoke("-m", "entanglix", "promote", staged["staged_id"], "--runs-root", root)
    output = tmp_path / "summary.json"
    summary = invoke(
        CLIENT, "reuse", "--via", via, "--entry", entry["entry_id"], "--runs-root", root,
        "--workspace", tmp_path / "inputs", "--output", output,
    )  # fmt: skip
    assert json.loads(output.read_text()) == summary
    assert summary["device"] == {"Csh_fF": 60.0, "EJ_ghz": 15.0}
    assert summary["entry"]["lineage"]["source_run"]["run_id"] == source.name
    assert [row["EJ_ghz"] for row in summary["spectrum"]] == [14.5, 15.0, 15.5]
    assert summary["spectrum"][1]["f01_ghz"] == pytest.approx(5.48, abs=0.005)
    assert set(summary["runs"]) == {"sweep", "warm_start", "control_check"}
    assert summary["warm_start"]["starting_pulse"]["sha256"] == entry["pulse"]["sha256"]
    assert summary["warm_start"]["starting_pulse"]["source"] == {"entry": entry["entry_id"]}
    assert (
        next(c for c in summary["warm_start"]["checks"] if c["name"] == "constraints")["outcome"]
        == "pass"
    )
    for name in ("sweep", "warm_start"):
        run = summary["runs"][name]
        assert run["source_entry_ids"] == [entry["entry_id"]]
        assert run["parent_run_ids"] == [source.name]
    assert summary["runs"]["control_check"]["parent_run_ids"] == [
        summary["runs"]["warm_start"]["run_id"]
    ]
    assert {row["protocol"] for row in summary["entry"]["reuses"]["runs"]} == {"sweep", "control"}
    spectrum = tomllib.loads((tmp_path / "inputs/spectrum.toml").read_text())
    assert spectrum["parameters"] == {"EJ": 15.0}  # Csh comes only from the entry.
    warm = root / summary["runs"]["warm_start"]["run"]
    repeated = tomllib.loads((warm / "spec/problem.toml").read_text())
    original = tomllib.loads((source / "spec/problem.toml").read_text())
    for name in ("init", "seed"):
        original["pulse"].pop(name, None)
    assert repeated == original  # The gate and model stayed fixed; the start replaces guidance.
