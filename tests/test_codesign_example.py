"""The co-design example's terms admit its launches, and its next-candidate rule."""

import importlib.util
from pathlib import Path

import pytest

from entanglix.investigations import _in_scope, _terms
from entanglix.protocols.control import compile_control
from entanglix.protocols.sweep import compile_sweep

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/transmon-codesign"
spec = importlib.util.spec_from_file_location("codesign", EXAMPLE / "codesign.py")
codesign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codesign)


def test_terms_admit_the_screen_and_candidates_inside_the_range(tmp_path):
    bases = {base["name"]: base for base in _terms(codesign.terms())["scope"]["bases"]}
    screen = tmp_path / "screen"
    screen.mkdir()
    gate = compile_sweep(EXAMPLE / "screen.toml", screen, [], EXAMPLE.parents[1], retain_base=True)
    _in_scope(bases["screen"], gate["base_inputs"], gate["scope_values"], None)
    for index, (capacitance, seed) in enumerate([(10.0, 7), (45.0, 19), (90.0, 43), (95.0, 7)]):
        directory = tmp_path / f"control-{index}"
        directory.mkdir()
        values = {"parameters.Csh": capacitance, "pulse.seed": seed}
        compiled = compile_control(
            EXAMPLE / "control.toml", directory, retain_base=True, values=values
        )
        assert "evaluation" not in (directory / "problem.toml").read_text()
        check = lambda: _in_scope(  # noqa: E731
            bases["gate"], compiled["base_inputs"], compiled["scope_values"], None
        )
        if capacitance > codesign.RANGE[1]:
            with pytest.raises(ValueError, match="outside permitted range: parameters.Csh"):
                check()
        else:
            check()


def scores(**values):
    return {
        float(name.removeprefix("c")): {"score": score, "spread": spread}
        for name, (score, spread) in values.items()
    }


def test_next_candidate_follows_the_declared_rule():
    rule = codesign.next_candidate
    pilot = scores(c10=(0.922, 5e-3), c30=(0.99993, 7e-5), c60=(0.99999985, 3e-7))
    assert rule(pilot, set(pilot))[0] == 90.0
    pilot[90.0] = {"score": 0.99999989, "spread": 2e-7}
    following, reason, pair = rule(pilot, set(pilot))
    assert following is None and "unresolved" in reason and pair == [60.0, 90.0]
    resolved = scores(c10=(0.9, 1e-3), c30=(0.999, 1e-4), c60=(0.99, 1e-4), c90=(0.98, 1e-4))
    assert rule(resolved, set(resolved))[0] == 45.0
    assert rule(resolved, {*resolved, 45.0})[0] is None
    assert rule(resolved, {*resolved, 45.0, 75.0})[1:] == ("5 candidates solved", None)
    assert rule({}, {10.0, 30.0, 60.0})[0] is None


def test_conclusion_retains_evaluated_candidates_or_reports_insufficient_evidence():
    rows = {
        60.0: {"score": 0.9999998545, "selected_refined_worst_fidelity": 0.9999998545},
        90.0: {"score": 0.9999998872, "selected_refined_worst_fidelity": 0.9999998348},
    }
    retained, text, following, stop = codesign.conclusion(rows, [60.0, 90.0])
    assert retained == [60.0, 90.0] and "reverses their order" in text and following
    assert stop == "evidence_completion"
    assert codesign.conclusion(rows, None)[0] == [90.0]
    retained, text, _, stop = codesign.conclusion({}, None)
    assert retained == [] and "No candidate is retained" in text
    assert stop == "insufficient_progress"
