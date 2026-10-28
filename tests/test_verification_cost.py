"""Same-engine failure and false-alarm fixtures from the feasibility probe."""

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from test_simulate import protocol_files
from test_verification import ROOT, files

from entanglix.corrector import load_policy
from entanglix.corrector.dynamics import refined_segments
from entanglix.engines import QuTiPIntegrator
from entanglix.protocols.plans import load_plan
from entanglix.protocols.simulate import compile_simulation


@pytest.mark.parametrize("case", ["truncation", "strong_drive", "two_photon", "top_fock"])
@pytest.mark.parametrize("engine_name", ["qutip-cpu", "dynamiqs"])
def test_same_engine_segment_scientific_regressions(tmp_path, case, engine_name):
    if engine_name == "dynamiqs":
        pytest.importorskip("dynamiqs")
    from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator
    from entanglix.protocols.simulate import compile_simulation

    path, spec = protocol_files(tmp_path)
    system = {
        "format_version": 1,
        "id": "segment_regression",
        "units": {"energy": "GHz", "time": "ns"},
        "model": {
            "kind": "composite",
            "subsystems": [
                {
                    "id": "r",
                    "kind": "oscillator",
                    "frequency": {"value": 0.2, "unit": "GHz"},
                    "retained_levels": 2 if case == "truncation" else 4,
                }
            ],
        },
        "drives": [{"id": "drive", "factors": ["r.position"]}],
    }
    spec["refinement"] = {"r": {"retained_levels_step": 1 if case == "truncation" else 4}}
    evolution = spec["evolution"]
    evolution.update(
        initial_state=[0],
        times_ns=list(np.linspace(0, 10, 21)),
        observables={"number": "r.number"},
        drive_amplitudes_ghz={"drive": 0.15},
    )
    spec["metrics"] = {"number": {"kind": "endpoint_real", "observable": "number"}}
    if case == "strong_drive":
        evolution["carriers"] = {"drive": {"frequency_ghz": 0.2}}
    if case == "top_fock":
        evolution.update(
            initial_state=[3], times_ns=list(np.linspace(0, 100, 21)), drive_amplitudes_ghz={}
        )
        system["dissipators"] = [
            {"id": "loss", "factors": ["r.annihilation"], "rate": {"value": 0.01, "unit": "1/ns"}}
        ]
    if case == "two_photon":
        import scqubits as scq

        levels = scq.Transmon(EJ=15, EC=0.25, ng=0, ncut=30).eigenvals(evals_count=3)
        frequency = float((levels[2] - levels[0]) / 2)
        system["model"]["subsystems"] = [
            {
                "id": "r",
                "kind": "transmon",
                "josephson_energy": {"value": 15, "unit": "GHz"},
                "charging_energy": {"value": 0.25, "unit": "GHz"},
                "ng": 0.0,
                "cutoff": 30,
                "retained_levels": 3,
            }
        ]
        system["drives"][0]["factors"] = ["r.charge"]
        spec["refinement"] = {"r": {"retained_levels_step": 3, "cutoff_steps": {"cutoff": 2}}}
        evolution.update(
            times_ns=list(np.linspace(0, 100, 101)),
            observables={"number": {"kind": "population", "subsystem": "r", "level": 2}},
            drive_amplitudes_ghz={"drive": 0.03},
            carriers={"drive": {"frequency_ghz": frequency}},
            frame={"frequencies_ghz": {"r": frequency}, "rwa": True},
        )
    system["model"]["subsystems"].append(
        {
            "id": "s",
            "kind": "oscillator",
            "frequency": {"value": 0.3, "unit": "GHz"},
            "retained_levels": 2,
        }
    )
    spec["refinement"]["s"] = {"retained_levels_step": 1}
    evolution["initial_state"].append(0)
    (tmp_path / "system.toml").write_text(tomli_w.dumps(system))
    path.write_text(tomli_w.dumps(spec))
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    compile_simulation(path, compiled, [], Path(__file__).resolve().parents[1])
    plan = tomllib.loads((compiled / "plan.toml").read_text())
    plan["engine"] = engine_name
    (compiled / "plan.toml").write_text(tomli_w.dumps(plan))
    loaded = load_plan(compiled / "system.toml", compiled / "plan.toml")
    from entanglix.engines import QuTiPIntegrator

    engine = (
        DynamiqsIntegrator(device_kind="cpu") if engine_name == "dynamiqs" else QuTiPIntegrator()
    )
    trace = engine.evolve(loaded.system, loaded.request)
    arrays = {"states": np.asarray(trace.states), "observable.number": trace.expectations["number"]}
    report = refined_segments(loaded, arrays, load_policy())
    failures = report["outcome"] == "fail"
    if case == "top_fock":
        assert not failures
    else:
        assert failures, report


@pytest.mark.parametrize("rwa", [False, True])
def test_qubit_segments_at_delayed_pulse_and_rwa_false_alarm(tmp_path, rwa):
    from dataclasses import replace

    from test_verification import files

    from entanglix.engines import QuTiPIntegrator

    plan = files(tmp_path)
    plan["evolution"].update(
        times_ns=list(np.linspace(0, 2, 21)),
        drive_amplitudes_ghz={"x_drive": [0.0] * 4 + [0.1] * 2 + [0.0] * 14},
    )
    if rwa:
        plan["evolution"].update(
            carriers={"x_drive": {"frequency_ghz": 0.2}},
            frame={
                "frequencies_ghz": {"two_level_reference": 0.2},
                "rwa": True,
                "excitation_numbers": {"two_level_reference": [0, 1]},
            },
        )
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    loaded = load_plan(tmp_path / "system.toml", tmp_path / "plan.toml")
    engine = QuTiPIntegrator()
    for fault in (False, True):
        trace = engine.evolve(
            loaded.system,
            replace(loaded.request, drive_amplitudes_ghz={}) if fault else loaded.request,
        )
        arrays = {
            "states": np.asarray(trace.states),
            "observable.coherence": trace.expectations["coherence"],
        }
        report = refined_segments(loaded, arrays, load_policy())
        assert report["outcome"] == ("fail" if fault else "pass")
        assert not report["coverage"]["uncovered_drive_changes"]["count"]


@pytest.mark.parametrize("engine_name", ["qutip-cpu", "dynamiqs"])
@pytest.mark.parametrize("slow", [False, True])
def test_same_engine_loose_tolerance_and_slow_drift(tmp_path, engine_name, slow):
    plan = files(tmp_path)
    plan["engine"] = engine_name
    plan["evolution"].update(
        times_ns=list(np.linspace(0, 10000, 1001)) if slow else list(np.arange(11.0)),
        rtol=1e-3 if slow else 0.1,
        atol=1e-5 if slow else 0.1,
    )
    if slow:
        document = tomllib.loads((tmp_path / "system.toml").read_text())
        document.pop("dissipators", None)
        (tmp_path / "system.toml").write_text(tomli_w.dumps(document))
        with np.load(tmp_path / "operators.npz") as original:
            arrays = {name: np.array(original[name]) for name in original.files}
        arrays["H0"] = np.diag([0.0, 0.02])
        np.savez(tmp_path / "operators.npz", **arrays)
        state = np.array([1, 1j]) / np.sqrt(2)
        np.savez(
            tmp_path / "state.npz", initial=0.25 * np.eye(2) + 0.5 * np.outer(state, state.conj())
        )
    (tmp_path / "plan.toml").write_text(tomli_w.dumps(plan))
    loaded = load_plan(tmp_path / "system.toml", tmp_path / "plan.toml")
    if engine_name == "dynamiqs":
        pytest.importorskip("dynamiqs")
        from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator

        engine = DynamiqsIntegrator(device_kind="cpu")
    else:
        engine = QuTiPIntegrator()
    trace = engine.evolve(loaded.system, loaded.request)
    report = refined_segments(
        loaded,
        {
            "states": np.asarray(trace.states),
            "observable.coherence": trace.expectations["coherence"],
        },
        load_policy(),
    )
    assert report["outcome"] == "fail", report


@pytest.mark.parametrize(
    "ej,amplitude",
    [(12.0, 0.0), (12.0, 0.001), (12.0, 0.002), (13.5, 0.002), (15.0, 0.0), (15.0, 0.002)],
)
def test_same_frame_readout_false_alarm_cases(tmp_path, ej, amplitude):
    path, spec = protocol_files(tmp_path)
    document = tomllib.loads((ROOT / "examples/readout/system.toml").read_text())
    (tmp_path / "system.toml").write_text(tomli_w.dumps(document))
    frame_cases = json.loads((ROOT / "tests/probes/m2_frames_results.json").read_text())["cases"]
    frequency = next(
        item["frequency_ghz"]
        for item in frame_cases
        if item["kind"] == "readout" and item["ej_ghz"] == ej
    )
    state = np.zeros(30, complex)
    state[0] = state[6] = 1 / np.sqrt(2)
    np.savez(tmp_path / "state.npz", initial=state)
    spec.update(
        parameters={"EJ": ej},
        refinement={
            "q": {"retained_levels_step": 1, "cutoff_steps": {"cutoff": 2}},
            "r": {"retained_levels_step": 2},
        },
        metrics={},
    )
    spec["evolution"] = {
        "array_file": "state.npz",
        "initial_state": "initial",
        "times_ns": list(np.arange(101.0)),
        "rtol": 1e-10,
        "atol": 1e-12,
        "observables": {"charge": "q.charge", "photons": "r.number"},
        "drive_amplitudes_ghz": {"readout": amplitude},
        "carriers": {"readout": {"frequency_ghz": frequency}},
        "frame": {"frequencies_ghz": {"q": frequency, "r": frequency}, "rwa": True},
    }
    path.write_text(tomli_w.dumps(spec))
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    compile_simulation(path, compiled, [], ROOT)
    loaded = load_plan(compiled / "system.toml", compiled / "plan.toml")
    trace = QuTiPIntegrator().evolve(loaded.system, loaded.request)
    arrays = {
        "states": np.asarray(trace.states),
        **{f"observable.{name}": value for name, value in trace.expectations.items()},
    }
    report = refined_segments(loaded, arrays, load_policy())
    assert report["outcome"] == "pass", report
