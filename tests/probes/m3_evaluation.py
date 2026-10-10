"""Workstation M3 evaluation: formulation, six agent cases, and current baselines.

Only input copies and owner-approved terms reach each fresh research workspace.
The operator-controlled driver starts the draft; all research jobs and findings
must go through the real MCP tools. Raw records/transcripts stay outside the repo.
"""

import argparse
import json
import os
import runpy
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

from entanglix.investigations import read_investigation

ROOT = Path(__file__).resolve().parents[2]
GUIDANCE = runpy.run_path(str(ROOT / "tests/probes/m3_guidance.py"))
CLIENT = runpy.run_path(str(ROOT / "examples/readout/investigate.py"))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def workspace(path, python, extras, bundle=None):
    GUIDANCE["prepare"](path, python, "", extras)
    shutil.rmtree(path / "inputs")
    (path / "cuda_visibility.py").unlink()
    (path / "evidence/driver.py").unlink()
    if bundle:
        for base in ("rwa", "lab", "refined-12", "refined-13_5", "refined-15"):
            shutil.copytree(bundle / base, path / "inputs" / base)
    return path


def session(path, name, prompt, python, model, effort, *, thread=None):
    evidence = path / "evidence"
    command = ["codex", "exec"]
    if thread:
        command += ["resume", thread]
    else:
        command += ["-C", str(path), "-s", "workspace-write"]
    command += [
        "--json", "--skip-git-repo-check", "-m", model,
        "-c", f'model_reasoning_effort="{effort}"',
        "-c", f'projects={{"{path}"={{trust_level="trusted"}}}}',
        "-c", 'shell_environment_policy.inherit="all"',
        "-o", str(evidence / f"{name}-answer.txt"), prompt,
    ]  # fmt: skip
    start = time.monotonic()
    with (
        (evidence / f"{name}.jsonl").open("w") as stdout,
        (evidence / f"{name}.stderr").open("w") as stderr,
    ):
        done = subprocess.run(
            command,
            cwd=path,
            env=os.environ
            | {"CUDA_VISIBLE_DEVICES": "", "PATH": f"{python.parent}:{os.environ['PATH']}"},
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
        )
    events = [json.loads(line) for line in (evidence / f"{name}.jsonl").read_text().splitlines()]
    thread = next(
        (item["thread_id"] for item in events if item["type"] == "thread.started"), thread
    )
    paths = list((Path.home() / ".codex/sessions").rglob(f"*{thread}*")) if thread else []
    transcript = paths[0].read_text() if paths else ""
    if paths:
        shutil.copyfile(paths[0], evidence / f"{name}-transcript.jsonl")
    initial = []
    for line in transcript.splitlines():
        item = json.loads(line)
        payload = item.get("payload", {})
        if item.get("type") == "response_item" and payload.get("role") == "assistant":
            break
        initial.extend(v.get("text", "") for v in payload.get("content", []) if isinstance(v, dict))
    suspicious = []
    # Retain the exact read commands for human inspection, rather than infer
    # absence of contamination from a separate runs root or a model assertion.
    commands = []
    for item in events:
        entry = item.get("item", {})
        if entry.get("type") == "command_execution":
            command_text = entry.get("command", "")
            commands.append(command_text)
            if any(
                marker in command_text
                for marker in (
                    str(ROOT),
                    "m3_investigation_",
                    "m3_readout_results",
                    "investigate.py",
                )
            ):
                suspicious.append(command_text)
    dump(evidence / f"{name}-commands.json", commands)
    return {
        "exit_code": done.returncode,
        "thread_id": thread,
        "elapsed_seconds": time.monotonic() - start,
        "errors": [item for item in events if item["type"] in {"error", "turn.failed"}],
        "transcript": str(evidence / f"{name}-transcript.jsonl") if paths else None,
        "conduct_in_initial_context": (path / "AGENTS.md").read_text().strip()
        in "\n".join(initial),
        "development_instructions_present": any(
            mark in transcript for mark in GUIDANCE["DEVELOPMENT_MARKERS"]
        ),
        "suspicious_commands": suspicious,
    }


def sample_at(samples, ej, amplitude, profile):
    key = (ej, amplitude, profile)
    if key in samples:
        return samples[key]
    # Decimal report formatting can differ by one floating-point ULP from a
    # midpoint. Match only a unique existing sample; never interpolate evidence.
    matches = [
        sample
        for (value, point, kind), sample in samples.items()
        if value == ej and kind == profile and abs(point - amplitude) <= 1e-15
    ]
    if len(matches) != 1:
        raise KeyError(key)
    return matches[0]


def confirmation(samples, ej, lower, upper):
    """Derive the claim from retained metrics, never from the agent's boolean."""
    endpoints = []
    for amplitude in (lower, upper):
        profiles = {}
        for profile in ("rwa", "lab", "lab_refined"):
            zero = sample_at(samples, ej, 0.0, profile)
            sample = sample_at(samples, ej, amplitude, profile)
            profiles[profile] = {
                "additional_loss": zero["coherence"] - sample["coherence"],
                "zero_run": zero["source_run_id"],
                "endpoint_run": sample["source_run_id"],
            }
        change = abs(
            profiles["lab"]["additional_loss"] - profiles["lab_refined"]["additional_loss"]
        )
        margin = profiles["lab_refined"]["additional_loss"] - CLIENT["TARGET"]
        endpoints.append(
            {
                "amplitude_ghz": amplitude,
                "profiles": profiles,
                "margin": margin,
                "refinement_change": change,
            }
        )
    return {
        "ej_ghz": ej,
        "width_ghz": upper - lower,
        "endpoints": endpoints,
        "confirmed": endpoints[0]["margin"] < -endpoints[0]["refinement_change"]
        and endpoints[1]["margin"] > endpoints[1]["refinement_change"],
    }


def occupied_seconds(intervals):
    """Execution coverage, counting simultaneous jobs only once."""
    end = total = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def runtime_usage(transcript, previous=None):
    """Read phase usage, subtracting an earlier snapshot of the same thread."""
    rows = [json.loads(line) for line in Path(transcript).read_text().splitlines()]
    prior = (
        [json.loads(line) for line in Path(previous).read_text().splitlines()] if previous else []
    )

    def totals(events):
        return next(
            (
                event["payload"]["info"]["total_token_usage"]
                for event in reversed(events)
                if event.get("payload", {}).get("type") == "token_count"
                and event["payload"].get("info")
            ),
            {},
        )

    end, start = totals(rows), totals(prior)
    seen = {(row.get("timestamp"), row.get("ordinal"), row["type"]) for row in prior}
    new = [
        row for row in rows if (row.get("timestamp"), row.get("ordinal"), row["type"]) not in seen
    ]
    limits = [
        row["payload"]["rate_limits"]
        for row in new
        if row.get("payload", {}).get("type") == "token_count" and row["payload"].get("rate_limits")
    ]
    windows = {}
    for name, key in (("five_hour", "primary"), ("weekly", "secondary")):
        observed = [item[key]["used_percent"] for item in limits if item.get(key)]
        windows[name] = {"first": observed[0], "last": observed[-1]} if observed else None
    return {
        **{
            key: end[key] - start.get(key, 0) if key in end else None
            for key in ("input_tokens", "cached_input_tokens", "output_tokens")
        },
        "model_requests": sum(row["type"] == "token_usage_record" for row in new),
        "windows_used_percent": windows,
    }


LAUNCH_TOOLS = {
    "simulate",
    "sweep",
    "check",
    "rerun_simulation",
    "rerun_sweep",
    "rerun_script",
    "run_script",
}


def launch_links(path):
    links = {}
    for outcome in (path / "runs/_audit/calls").glob("*/outcome.toml"):
        request = tomllib.loads((outcome.parent / "request.toml").read_text())
        record = tomllib.loads(outcome.read_text())
        if request["tool"] in LAUNCH_TOOLS and (run := record.get("links", {}).get("run_id")):
            links[run] = request["tool"]
    return links


def summarize(path, case, phases, *, scripted_runs=()):
    investigations = list((path / "runs/_investigations").glob("*/revisions/0001.toml"))
    if len(investigations) != 1:
        return {
            "case": case,
            "acceptance": False,
            "reason": "expected one investigation",
            "phases": phases,
        }
    investigation = investigations[0].parent.parent
    # The scripted client's recovery reader only reconstructs samples; it never
    # runs its policy or any job during evaluation.
    dump(path / "client.json", {"investigation": str(investigation), "case": case})
    client = CLIENT["Client"](path)
    record = read_investigation(investigation)
    report_path = path / "evidence/claims.json"
    claims = json.loads(report_path.read_text()) if report_path.exists() else {}
    evidence, problems, needed = [], [], {}
    for phase in phases:
        if phase["exit_code"] or phase["errors"]:
            problems.append("runtime phase failed")
        if not phase["conduct_in_initial_context"] or phase["development_instructions_present"]:
            problems.append("initial guidance boundary not demonstrated")
        if phase["suspicious_commands"]:
            problems.append("transcript requires contamination review")
    if client.seed_run is None:
        problems.append("seed sweep missing")
    else:
        needed[client.seed_run.name] = {"integrity", "diagnostics", "declared_expectation"}
    for claim in claims.get("brackets", []):
        try:
            item = confirmation(
                client.samples,
                claim["ej_ghz"],
                claim["lower_amplitude_ghz"],
                claim["upper_amplitude_ghz"],
            )
            item["stage"] = claim["stage"]
            if item["confirmed"] != claim["confirmed"]:
                problems.append("agent confirmation disagrees with retained evidence")
            for endpoint in item["endpoints"]:
                needed[endpoint["profiles"]["rwa"]["endpoint_run"]] = {
                    "integrity",
                    "segments",
                    "diagnostics",
                }
                for name in ("zero_run", "endpoint_run"):
                    needed[endpoint["profiles"]["lab_refined"][name]] = {"integrity"}
            evidence.append(item)
        except KeyError as exc:
            problems.append(f"missing source for claim: {exc}")
    normal = [item for item in evidence if item["stage"] == "normal"]
    if {item["ej_ghz"] for item in normal} != {13.5, 15.0} or any(
        not item["confirmed"] or item["width_ghz"] > CLIENT["WIDTH"] * (1 + 1e-10)
        for item in normal
    ):
        problems.append("normal brackets incomplete or unsupported")
    if client.seed_run:
        zero = client.samples[12.0, 0.0, "rwa"]["coherence"]
        seed_loss = [
            zero - client.samples[12.0, a, "rwa"]["coherence"] for a in (0.0, 0.001, 0.002)
        ]
        if max(seed_loss) >= CLIENT["TARGET"] or claims.get("no_crossing_ej") != [12.0]:
            problems.append("no-crossing finding does not match samples")
    if case == "early-stop":
        tighter = [item for item in evidence if item["stage"] == "tighter"]
        failed_ej = tighter[0]["ej_ghz"] if tighter else None
        crossing_ejs = sorted(item["ej_ghz"] for item in normal)
        if (
            len(tighter) != 1
            or failed_ej not in crossing_ejs
            or tighter[0]["confirmed"]
            or tighter[0]["width_ghz"] > CLIENT["TIGHT_WIDTH"] * (1 + 1e-10)
        ):
            problems.append("precision handoff missing or unsupported")
        if (
            claims.get("unattempted_tighter_ej")
            != [ej for ej in crossing_ejs if failed_ej is not None and ej > failed_ej]
            or "precision" not in record["open_reviews"]
        ):
            problems.append("remaining tighter work or precision review not reported")
    reports = [
        e for e in record["events"] if e["kind"] == "finding" and e["finding"]["kind"] == "report"
    ]
    if (
        not reports
        or not reports[-1]["finding"]["unresolved"]
        or not reports[-1]["finding"]["next_experiment"]
    ):
        problems.append("report lacks limits or next experiments")
    if record["state"] != "stopped":
        problems.append("investigation not stopped after report")
    if not any(event["kind"] == "stop" for event in record["events"]):
        return {
            "case": case,
            "investigation_id": investigation.name,
            "acceptance": False,
            "problems": problems,
            "phases": phases,
            "state": record["state"],
            "usage": record["usage"],
            "report": None,
        }
    seconds = {"computation": 0.0, "checks": 0.0}
    intervals = []
    for run_id, (run_path, item) in client.records.items():
        if item["status"] != "completed":
            problems.append(f"non-completed attempt: {run_id}")
        role = "checks" if item["manifest"]["protocol"] == "verification" else "computation"
        seconds[role] += item.get("finished", {}).get("usage", {}).get("wall_seconds", 0)
        if "start" in item and "finished" in item:
            intervals.append(
                (
                    item["start"]["started_at"].timestamp(),
                    item["finished"]["ended_at"].timestamp(),
                )
            )
        if role == "computation":
            if item["manifest"]["investigation_base"].startswith("refined-"):
                needed.setdefault(run_id, set()).add("integrity")
            passed = set()
            for check in item.get("verification", []):
                if check.get("source_matches") and check["status"] == "completed":
                    for entry in check.get("checks", []):
                        if entry.get("outcome") not in {"fail", "inapplicable"}:
                            passed.add(entry["name"])
            if missing := needed.get(run_id, set()) - passed:
                problems.append(f"requested checks missing: {run_id}: {sorted(missing)}")
    calls = [
        tomllib.loads(p.read_text()) for p in (path / "runs/_audit/calls").glob("*/request.toml")
    ]
    linked = launch_links(path)
    if set(client.records) - set(scripted_runs) - set(linked):
        problems.append("attempt without launching MCP call link")
    findings = [e for e in record["events"] if e["kind"] == "finding"]
    if reports and not set(client.records).issubset(set(reports[-1]["finding"]["runs"])):
        problems.append("final report omits source runs")
    start = next(
        e["recorded_at"]
        for e in record["events"]
        if e["kind"] == "decision" and e["action"] == "start"
    )
    stop = next(e["recorded_at"] for e in reversed(record["events"]) if e["kind"] == "stop")
    elapsed = (stop - start).total_seconds()
    return {
        "case": case,
        "investigation_id": investigation.name,
        "acceptance": not problems,
        "state": record["state"],
        "problems": problems,
        "phases": phases,
        "usage": record["usage"],
        "elapsed_seconds": elapsed,
        "computation_seconds": seconds["computation"],
        "check_seconds": seconds["checks"],
        "execution_coverage_seconds": occupied_seconds(intervals),
        "time_outside_jobs_seconds": elapsed - occupied_seconds(intervals),
        "user_decisions_after_start": sum(
            e["kind"] == "decision" and e["action"] != "start" for e in record["events"]
        ),
        "assessments": [
            {key: e["finding"][key] for key in ("rationale", "decision", "next_experiment")}
            for e in findings
            if e["finding"]["kind"] == "assessment"
        ],
        "report": {
            key: reports[-1]["finding"][key]
            for key in ("decision", "rationale", "unresolved", "next_experiment")
        }
        if reports
        else None,
        "claims": evidence,
        "agent_claims": claims,
        "tools_used": sorted({c["tool"] for c in calls}),
    }


EXECUTION = """The operator has started your investigation at {investigation}. Read its effective
terms, then execute the agreed method through Entanglix MCP tools only. Use
simulate/sweep/check for every solve/check, record_finding for assessments/reports,
and stop_investigation after your report. Do not call a solver, the Python harness
API, an operator command, or a separate MCP client from the shell. Shell arithmetic
and reading your own retained outputs are allowed. Do not read outside this workspace.

Run your own 3x3 seed sweep. At each EJ use C(0)-C(A) against 0.01; bisect the RWA
crossing at midpoints to width 0.000125 GHz. Do not refine a no-crossing line outside
the range. Check seed integrity/diagnostics/declared_expectation; each final RWA
endpoint integrity/segments/diagnostics; each refined lab run integrity. Confirmation
requires lab and refined-lab runs at zero and both endpoints, comparing their loss
margins with the observed refinement change. Intermediate RWA points and coarse lab
runs do not need checks. Reuse your own retained zero/endpoint runs when unchanged.
For the early-stop case establish all normal results first, then tighten in ascending
EJ to width 0.000015625 GHz; stop at the first failed lab confirmation, open review
'precision', and leave later tighter requests unattempted. Before a due checkpoint
record an assessment with all run IDs, evidence, learned/unresolved items, rationale,
and next experiment. Unused allowance is not a reason to continue after a handoff.

Remember each job returns admission: wait on the returned path until completed.
Use lineage to cite the source of each new choice. Launch independent CPU work
concurrently only when its scientific inputs are known and checkpoints permit it.
Report source values and IDs, relevant checks not run, all four limitations in the
terms, and the next useful experiment. Delta-C and brackets are your interpretations,
not a built-in metric. Record the final report via MCP, then stop. Also write
'evidence/claims.json' with this exact compact shape for the external reviewer:
{{"outcome":"evidence_completion or review_handoff", "no_crossing_ej":[...],
"brackets":[{{"ej_ghz":number,"stage":"normal or tighter",
"lower_amplitude_ghz":number,"upper_amplitude_ghz":number,"confirmed":boolean}}],
"unattempted_tighter_ej":[...], "limitations":[...], "next_experiments":[...]}}.
Do not compute beyond the agreed method or change any base input.
"""


def evaluation_acceptance(formulation, agents):
    counts = {}
    for case in ("normal", "early-stop"):
        sessions = [item for item in agents if item["case"] == case]
        accepted = sum(item["acceptance"] for item in sessions)
        failures = [item for item in sessions if not item["acceptance"]]
        runtime_failures = sum(
            item.get("usage", {}).get("jobs") == 0
            and any(phase["errors"] and phase["exit_code"] for phase in item["phases"])
            for item in failures
        )
        counts[case] = {
            "accepted": accepted,
            "failed": len(failures),
            "runtime_failed_before_admission": runtime_failures,
        }
    formulation_ok = (
        not formulation["exit_code"]
        and not formulation["errors"]
        and formulation["conduct_in_initial_context"]
        and not formulation["development_instructions_present"]
        and not formulation["suspicious_commands"]
    )
    return formulation_ok and all(
        value["accepted"] >= 2 and value["failed"] == value["runtime_failed_before_admission"]
        for value in counts.values()
    ), counts


def finish_report(root, result, output):
    """Rebuild measured summaries without running or correcting agent work."""
    formulation = result["formulation"]
    formulation["proposal"] = (root / "formulation/evidence/formulation-answer.txt").read_text()
    formulation["runtime_usage"] = runtime_usage(formulation["transcript"])
    for index, prior in enumerate(result["agents"]):
        case, repetition = prior["case"], prior["repetition"]
        previous = None
        for phase in prior["phases"]:
            phase["runtime_usage"] = runtime_usage(phase["transcript"], previous)
            previous = phase["transcript"]
        try:
            item = summarize(root / f"{case}-{repetition}", case, prior["phases"])
        except (ValueError, KeyError, OSError, RuntimeError, StopIteration) as exc:
            item = {
                "case": case,
                "acceptance": False,
                "reason": str(exc) or type(exc).__name__,
                "phases": prior["phases"],
            }
        item["repetition"] = repetition
        baseline = result["baselines"][case]
        if "elapsed_seconds" in item:
            item["comparison"] = {
                "elapsed_ratio_to_baseline": item["elapsed_seconds"] / baseline["elapsed_seconds"],
                "job_seconds_ratio_to_baseline": item["usage"]["job_seconds"]
                / baseline["usage"]["job_seconds"],
                "additional_jobs": item["usage"]["jobs"] - baseline["usage"]["jobs"],
            }
        result["agents"][index] = item
    execution = result["source_sha256"].pop("tests/probes/m3_evaluation.py", None)
    result.setdefault("execution_driver_sha256", execution)
    result["summary_driver_sha256"] = GUIDANCE["digest"](Path(__file__).resolve())
    phases = [formulation] + [phase for item in result["agents"] for phase in item["phases"]]
    result.setdefault(
        "completed_at",
        max(
            json.loads(Path(phase["transcript"]).read_text().splitlines()[-1])["timestamp"]
            for phase in phases
        ),
    )
    result["runtime_usage"] = {
        key: sum(phase["runtime_usage"][key] for phase in phases)
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "model_requests")
    }
    result["acceptance"], result["case_counts"] = evaluation_acceptance(
        formulation, result["agents"]
    )
    dump(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--server-python", type=Path)
    parser.add_argument("--runner-extras", nargs="+")
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild summaries from retained records without running sessions",
    )
    args = parser.parse_args()
    root = args.workspace.expanduser().resolve()
    if args.report_only:
        finish_report(
            root, json.loads((root / "evaluation-progress.json").read_text()), args.output
        )
        return
    if not all((args.server_python, args.runner_extras, args.model, args.effort)):
        parser.error("execution requires server-python, runner-extras, model, and effort")
    if root.exists() or root.is_relative_to(ROOT):
        parser.error("choose a fresh persistent directory outside the checkout")
    root.mkdir(parents=True)
    python = args.server_python.expanduser().absolute()
    operator = root / "operator"
    operator.mkdir()
    shutil.copyfile(Path(__file__).resolve(), operator / "execution-driver.py")
    terms, bundles = {}, {}
    for case in ("normal", "early-stop"):
        bundle = operator / case
        draft = CLIENT["draft"](
            bundle,
            runs_root=operator / f"{case}-draft",
            case=case,
            checkpoint=10,
            job_count=60 if case == "normal" else 100,
        )
        terms[case] = draft["revisions"][0]["terms"]
        bundles[case] = bundle
    result = {
        "runtime": subprocess.check_output(["codex", "--version"], text=True).strip(),
        "model": args.model,
        "effort": args.effort,
        "workspace": str(root),
        "source_sha256": {
            str(p.relative_to(ROOT)): GUIDANCE["digest"](p)
            for p in [
                Path(__file__).resolve(),
                *sorted((ROOT / "src/entanglix").rglob("*.py")),
                ROOT / "examples/readout/investigate.py",
                ROOT / "docs/research-agent.md",
                *sorted((ROOT / "skills").glob("*/SKILL.md")),
            ]
        },
        "operator_decisions": "Driver starts only a draft identical to the approved baseline terms; no agent approves itself.",
        "baselines": {},
        "agents": [],
    }
    for case in ("normal", "early-stop"):
        baseline_output = operator / f"{case}-baseline.json"
        subprocess.run(
            [
                str(python),
                str(ROOT / "tests/probes/m3_investigation.py"),
                "--case",
                case,
                "--workspace",
                str(operator / f"{case}-baseline-inputs"),
                "--runs-root",
                str(operator / f"{case}-baseline-runs"),
                "--output",
                str(baseline_output),
            ],
            check=True,
            env=os.environ | {"ENTANGLIX_RUNNER_EXTRAS": ",".join(args.runner_extras)},
        )
        baseline = json.loads(baseline_output.read_text())
        result["baselines"][case] = {
            key: baseline[key]
            for key in (
                "outcome",
                "usage",
                "elapsed_seconds",
                "computation_seconds",
                "check_seconds",
                "time_outside_jobs_seconds",
                "user_decisions_after_start",
                "assessments",
            )
        }
    formulation = workspace(root / "formulation", python, args.runner_extras, bundles["normal"])
    prompt = "I want to understand how a readout drive affects qubit coherence. Inspect the shipped input models and research references, propose a bounded next investigation and explain the decisions you need from me. Do not launch jobs, draft an effective investigation, read outside this workspace, or infer measurement/readout quality from coherence alone."
    result["formulation"] = session(
        formulation, "formulation", prompt, python, args.model, args.effort
    )
    for case in ("normal", "early-stop"):
        for repetition in range(1, 4):
            path = workspace(
                root / f"{case}-{repetition}", python, args.runner_extras, bundles[case]
            )
            dump(path / "agreed-terms.json", terms[case])
            prompt = "The operator has settled the next investigation of how readout drive reduces endpoint coherence. Read the conduct and relevant procedures/reference. The approved declaration is agreed-terms.json and its input copies are under inputs/<base>/. Draft it unchanged using create_investigation through MCP; validate the base system through MCP. Do not start it, launch compute, read outside this workspace, or change its terms. Explain the bounded question, evidence, limits and handoff decisions; then wait for the operator start."
            phases = [session(path, "draft", prompt, python, args.model, args.effort)]
            investigations = list((path / "runs/_investigations").glob("*/revisions/0001.toml"))
            if len(investigations) == 1:
                investigation = investigations[0].parent.parent
                record = read_investigation(investigation)
                if (
                    not phases[0]["exit_code"]
                    and not phases[0]["errors"]
                    and record["state"] == "draft"
                    and record["revisions"][0]["terms"] == terms[case]
                ):
                    subprocess.run(
                        [
                            str(python.parent / "entanglix"),
                            "investigation-decision",
                            str(investigation),
                            "start",
                            "--revision",
                            "1",
                            "--note",
                            "Operator-controlled M3 agent evaluation; approved fixture terms.",
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                    phases.append(
                        session(
                            path,
                            "execute",
                            EXECUTION.format(investigation=investigation),
                            python,
                            args.model,
                            args.effort,
                            thread=phases[0]["thread_id"],
                        )
                    )
            try:
                item = summarize(path, case, phases)
            except (ValueError, KeyError, OSError, RuntimeError, StopIteration) as exc:
                item = {
                    "case": case,
                    "acceptance": False,
                    "reason": str(exc) or type(exc).__name__,
                    "phases": phases,
                }
            item["repetition"] = repetition
            result["agents"].append(item)
            dump(root / "evaluation-progress.json", result)
            print(
                json.dumps(
                    {
                        "case": case,
                        "repetition": repetition,
                        "acceptance": item["acceptance"],
                        "problems": item.get("problems", item.get("reason")),
                        "usage": item.get("usage"),
                    }
                ),
                flush=True,
            )
    finish_report(root, result, args.output)


if __name__ == "__main__":
    main()
