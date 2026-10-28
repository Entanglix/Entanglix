"""Selection by source run, spec, user default, or rule; refusals have no fallback."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from entanglix.engines import EngineCapabilities, EvolutionInput, QuTiPIntegrator
from entanglix.engines.cudensitymat import CuQuantumIntegrator
from entanglix.engines.dynamiqs_jax import DynamiqsIntegrator
from entanglix.engines.selection import (
    EngineSelectionError,
    select_engine,
    selection_table,
)
from entanglix.models import load_system
from entanglix.runner import engine_defaults

ROOT = Path(__file__).resolve().parents[1] / "examples"


def inputs(name):
    system = load_system(ROOT / name / "system.toml")
    initial = np.zeros(np.prod(system.dimensions))
    initial[0] = 1
    return system, EvolutionInput(initial, np.array([0.0, 1.0]))


def test_policy_agrees_with_implemented_capabilities():
    table = selection_table()
    assert table["format_version"] == 1
    for engine in (QuTiPIntegrator(), CuQuantumIntegrator(), DynamiqsIntegrator()):
        caps = engine.capabilities()
        row = table["engines"][caps.name]
        for field in ("device_kind", "open_system", "differentiable", "batched"):
            assert row[field] == getattr(caps, field)


@pytest.mark.parametrize(
    "system_name, gradients, selected",
    [
        ("two-level", False, "qutip-cpu"),
        ("qubit-resonator", False, "qutip-cpu"),
        ("two-level", True, "dynamiqs"),
        ("qubit-resonator", True, "dynamiqs"),
    ],
)
def test_selection_follows_declared_needs(system_name, gradients, selected):
    system, request = inputs(system_name)
    result = select_engine(
        system,
        request,
        available_engines={"qutip-cpu", "cudensitymat", "dynamiqs"},
        needs_gradients=gradients,
    )
    assert result.engine == selected and result.table_id == "declared-needs-engines-v3"
    assert result.source == "rule"


def test_missing_or_incapable_engines_refuse_without_fallback():
    system, request = inputs("qubit-resonator")
    with pytest.raises(EngineSelectionError, match="qutip-cpu \\(rule\\) is not installed"):
        select_engine(system, request, available_engines={"cudensitymat"})
    with pytest.raises(EngineSelectionError, match="cudensitymat \\(spec\\) is not installed"):
        select_engine(system, request, available_engines={"qutip-cpu"}, engine="cudensitymat")
    with pytest.raises(EngineSelectionError, match="does not support gradients"):
        select_engine(
            system,
            request,
            available_engines={"qutip-cpu", "dynamiqs"},
            needs_gradients=True,
            defaults={"cpu": "qutip-cpu"},
        )


def test_precedence_is_source_run_spec_default_then_rule():
    system, request = inputs("two-level")
    available = {"qutip-cpu", "cudensitymat", "dynamiqs"}
    choose = lambda **kwargs: select_engine(  # noqa: E731
        system, request, available_engines=available, device="cpu", **kwargs
    )
    defaults = {"cpu": "dynamiqs", "cuda": "cudensitymat"}
    assert (choose().engine, choose().source) == ("qutip-cpu", "rule")
    assert (choose(defaults=defaults).engine, choose(defaults=defaults).source) == (
        "dynamiqs",
        "default",
    )
    named = choose(defaults=defaults, engine="qutip-cpu")
    assert (named.engine, named.source) == ("qutip-cpu", "spec")
    pinned = choose(defaults=defaults, engine="qutip-cpu", pinned="dynamiqs")
    assert (pinned.engine, pinned.source) == ("dynamiqs", "source_run")
    assert choose().distribution == "qutip" and len(choose().record_sha256) == 64


def test_user_defaults_parse_per_device():
    assert engine_defaults("") == {}
    assert engine_defaults(" cpu = mine , cuda=cudensitymat ") == {
        "cpu": "mine",
        "cuda": "cudensitymat",
    }
    for value in ("mine", "gpu=mine", "cpu=", "cpu=a,cpu=b"):
        with pytest.raises(ValueError, match="ENTANGLIX_ENGINES"):
            engine_defaults(value)


@pytest.fixture
def plugin(monkeypatch):
    capabilities = {"value": EngineCapabilities("mine", "cpu", True, True)}

    class Engine:
        def capabilities(self):
            return capabilities["value"]

    dist = SimpleNamespace(
        metadata={"Name": "mine-engine"}, version="2.0", read_text=lambda _: "RECORD lines"
    )
    entries = [SimpleNamespace(name="mine", load=lambda: Engine, dist=dist)]
    monkeypatch.setattr("entanglix.engines.selection.entry_points", lambda group: entries)
    return capabilities, entries


def test_plugin_engines_are_named_and_refused_when_incapable(plugin):
    capabilities, entries = plugin
    system, request = inputs("qubit-resonator")
    selection = select_engine(
        system, request, available_engines={"qutip-cpu"}, defaults={"cpu": "mine"}
    )
    assert (selection.engine, selection.source, selection.distribution) == (
        "mine",
        "default",
        "mine-engine",
    )
    assert selection.version == "2.0" and len(selection.record_sha256) == 64
    # A plugin's declared differentiability does not admit gradient runs.
    with pytest.raises(EngineSelectionError, match="mine \\(spec\\) does not support gradients"):
        select_engine(
            system, request, available_engines={"qutip-cpu"}, engine="mine", needs_gradients=True
        )
    with pytest.raises(EngineSelectionError, match="does not support device cuda"):
        select_engine(
            system, request, available_engines={"qutip-cpu"}, engine="mine", device="cuda"
        )
    capabilities["value"] = EngineCapabilities("mine", "cpu", False, False)
    with pytest.raises(EngineSelectionError, match="open-system evolution"):
        select_engine(system, request, available_engines={"qutip-cpu"}, engine="mine")
    with pytest.raises(EngineSelectionError, match="'other' is not installed"):
        select_engine(system, request, available_engines={"qutip-cpu"}, engine="other")
    entries.append(SimpleNamespace(name="qutip-cpu", load=None, dist=None))
    with pytest.raises(EngineSelectionError, match="duplicate engine entry point: qutip-cpu"):
        select_engine(system, request, available_engines={"qutip-cpu"}, engine="mine")


@pytest.mark.parametrize(
    "device, gradients, engine",
    [
        ("cpu", False, "qutip-cpu"),
        ("cuda", False, "cudensitymat"),
        ("cpu", True, "dynamiqs"),
        ("cuda", True, "dynamiqs"),
    ],
)
def test_explicit_device_dispatch_and_storage(device, gradients, engine):
    system, request = inputs("two-level")
    selection = select_engine(
        system,
        request,
        available_engines={"qutip-cpu", "dynamiqs", "cudensitymat"},
        device=device,
        needs_gradients=gradients,
    )
    assert selection.engine == engine


def test_large_model_dispatch_and_unsupported_batch():
    system, request = inputs("three-subsystem")
    selection = select_engine(system, request, available_engines={"qutip-cpu"})
    assert selection.engine == "qutip-cpu"
    with pytest.raises(EngineSelectionError, match="does not support batching"):
        select_engine(system, request, available_engines={"qutip-cpu"}, batched=True)
