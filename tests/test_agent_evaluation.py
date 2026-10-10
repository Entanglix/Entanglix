"""The evaluation distinguishes an RWA crossing from lab-confirmed evidence."""

import json
import math
import runpy
from pathlib import Path

import pytest

EVALUATION = runpy.run_path(str(Path(__file__).resolve().parent / "probes/m3_evaluation.py"))


def test_concurrent_job_time_is_not_double_counted():
    assert EVALUATION["occupied_seconds"]([(0, 10), (2, 6), (10, 12), (14, 15)]) == 13


@pytest.mark.parametrize("lab_loss,confirmed", [(0.0098, True), (0.01001, False)])
@pytest.mark.parametrize("rounded_report", [False, True])
def test_rwa_bracket_does_not_imply_lab_confirmation(lab_loss, confirmed, rounded_report):
    samples = {}
    for profile in ("rwa", "lab", "lab_refined"):
        for amplitude, loss in (
            (0.0, 0.0),
            (0.001, 0.009 if profile == "rwa" else lab_loss),
            (0.002, 0.011),
        ):
            if profile == "lab" and amplitude:
                loss += 1e-6
            point = (
                math.nextafter(amplitude, math.inf) if amplitude and rounded_report else amplitude
            )
            samples[13.5, point, profile] = {
                "coherence": 1 - loss,
                "source_run_id": f"{profile}-{amplitude}",
            }
    evidence = EVALUATION["confirmation"](samples, 13.5, 0.001, 0.002)
    assert evidence["confirmed"] is confirmed
    assert (
        evidence["endpoints"][0]["profiles"]["lab_refined"]["endpoint_run"] == "lab_refined-0.001"
    )


def test_unsampled_endpoint_cannot_supply_confirmation():
    with pytest.raises(KeyError):
        EVALUATION["sample_at"]({(13.5, 0.001, "lab"): {}}, 13.5, 0.001001, "lab")


def test_runtime_failure_before_compute_retains_usage_and_reason(tmp_path):
    workspace = tmp_path / "workspace"
    EVALUATION["CLIENT"]["draft"](
        workspace, runs_root=workspace / "runs", case="normal", checkpoint=10, job_count=60
    )
    phase = {
        "exit_code": 1,
        "errors": [{"type": "error", "message": "usage limit reached"}],
        "conduct_in_initial_context": True,
        "development_instructions_present": False,
        "suspicious_commands": [],
    }
    summary = EVALUATION["summarize"](workspace, "normal", [phase])
    assert not summary["acceptance"]
    assert summary["state"] == "draft"
    assert summary["usage"]["jobs"] == 0
    assert "runtime phase failed" in summary["problems"]
    assert summary["phases"][0]["errors"] == phase["errors"]


@pytest.mark.parametrize(
    "jobs,error,accepted", [(0, True, True), (1, True, False), (0, False, False)]
)
def test_acceptance_allows_only_reported_runtime_failures_before_admission(jobs, error, accepted):
    formulation = {
        "exit_code": 0,
        "errors": [],
        "conduct_in_initial_context": True,
        "development_instructions_present": False,
        "suspicious_commands": [],
    }
    agents = [
        {"case": case, "acceptance": True} for case in ("normal", "early-stop") for _ in range(2)
    ]
    agents.append(
        {
            "case": "early-stop",
            "acceptance": False,
            "usage": {"jobs": jobs},
            "phases": [{"exit_code": 1, "errors": ["usage limit"] if error else []}],
        }
    )
    verdict, counts = EVALUATION["evaluation_acceptance"](formulation, agents)
    assert verdict is accepted
    assert counts["early-stop"]["accepted"] == 2
    assert counts["early-stop"]["failed"] == 1
    assert counts["early-stop"]["runtime_failed_before_admission"] == int(accepted)
    assert not EVALUATION["evaluation_acceptance"](formulation | {"exit_code": 1}, agents)[0]
    assert not EVALUATION["evaluation_acceptance"](formulation, agents[1:])[0]


def test_resumed_runtime_usage_subtracts_prior_snapshot(tmp_path):
    def event(index, tokens, window):
        return {
            "timestamp": str(index),
            "ordinal": index,
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {"total_token_usage": tokens},
                "rate_limits": {"primary": {"used_percent": window}, "secondary": None},
            },
        }

    request = {"timestamp": "1", "ordinal": 1, "type": "token_usage_record"}
    first = [
        request,
        event(2, {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 5}, 32),
    ]
    next_request = request | {"timestamp": "3", "ordinal": 3}
    second = first + [
        next_request,
        event(4, {"input_tokens": 250, "cached_input_tokens": 200, "output_tokens": 9}, 34),
    ]
    prior, resumed = tmp_path / "draft.jsonl", tmp_path / "execute.jsonl"
    for path, events in ((prior, first), (resumed, second)):
        path.write_text("\n".join(json.dumps(row) for row in events))
    usage = EVALUATION["runtime_usage"](resumed, prior)
    assert (usage["input_tokens"], usage["cached_input_tokens"], usage["output_tokens"]) == (
        150,
        120,
        4,
    )
    assert usage["model_requests"] == 1
    assert usage["windows_used_percent"] == {"five_hour": {"first": 34, "last": 34}, "weekly": None}
    assert EVALUATION["runtime_usage"](prior, prior)["input_tokens"] == 0


def test_read_call_does_not_establish_launch_provenance(tmp_path):
    for index, tool in enumerate(("read_run", "simulate")):
        call = tmp_path / "runs/_audit/calls" / str(index)
        call.mkdir(parents=True)
        (call / "request.toml").write_text(f'tool = "{tool}"\n')
        (call / "outcome.toml").write_text(f'[links]\nrun_id = "run-{index}"\n')
    assert EVALUATION["launch_links"](tmp_path) == {"run-1": "simulate"}
