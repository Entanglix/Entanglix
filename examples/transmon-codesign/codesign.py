"""Scripted device–control co-design over shunt capacitance, through the CLI or an MCP client.

`draft` proposes the investigation terms; the user starts the investigation. `run` screens
the range, solves each candidate with every required seed inside the approved base, issues
the standing-requested checks, chooses each next candidate from the evidence by the declared
rule, and records assessments and a final report. Both paths make the same calls.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tomli_w

EXAMPLE = Path(__file__).resolve().parent
RANGE = (10.0, 90.0)
SEEDS = (7, 19, 43)
INITIAL = (10.0, 30.0, 60.0)
MAX_CANDIDATES = 5
CHECKS = ["replay", "constraints", "uncertainty", "uncertainty_refinement"]
LIMITS = [
    "Conclusions hold for this model, its declared samples, the Rabi-rate bound policy and the "
    "declared search policy only; they are not general device-design conclusions.",
    "Closed-system evaluations under the rotating-wave approximation, on the declared samples only.",
    "Each score is a lower bound from the pulses found; finite search establishes no upper bound.",
    "Seed spread and refinement change are reported as evidence, not as thresholds on a ranking.",
]


def runner_extras():
    """The runner profile the CLI uses: ENTANGLIX_RUNNER_EXTRAS, else the project's runner table."""
    if "ENTANGLIX_RUNNER_EXTRAS" in os.environ:
        return [v.strip() for v in os.environ["ENTANGLIX_RUNNER_EXTRAS"].split(",") if v.strip()]
    config = tomllib.loads((EXAMPLE.parents[1] / "pyproject.toml").read_text())
    return config.get("tool", {}).get("entanglix", {}).get("runner", {}).get("extras", [])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terms():
    system = digest(EXAMPLE / "system.toml")
    capacitance = {"target": "parameters.Csh", "range": list(RANGE)}
    return {
        "format_version": 1,
        "question": (
            "Which shunt capacitance in 10-90 fF gives the highest worst-sample fidelity of a "
            "6 ns X gate under a 0.15 GHz Rabi-rate bound, over the declared offset-charge and "
            "EJ samples?"
        ),
        "assumptions": [
            "Closed five-level model of the junction-shunt transmon under the rotating-wave "
            "approximation, with the 0-1 transition followed by both quadratures.",
            "The worst fidelity over the declared samples is the compared quantity.",
            "Samples are quasistatic offsets, not time-dependent noise; calibrated samples "
            "assume perfect recalibration of the named transition.",
            "Equal Rabi-rate bounds do not imply equal hardware drive voltage or power.",
            "The 6 ns, 0.15 GHz regime was chosen in an exploratory pilot to exercise the "
            "workflow with distinguishable outcomes; it is not a preferred-regime finding, and "
            "the leading candidates' order differs in other regimes.",
        ],
        "scope": {
            "operations": ["sweep", "control", "check"],
            "bases": [
                {
                    "name": "screen",
                    "inputs": {
                        "system.toml": system,
                        "simulate.toml": digest(EXAMPLE / "spectrum.toml"),
                    },
                    "variations": [capacitance],
                },
                {
                    "name": "gate",
                    "inputs": {
                        "system.toml": system,
                        "control.toml": digest(EXAMPLE / "control.toml"),
                    },
                    "variations": [capacitance, {"target": "pulse.seed", "values": list(SEEDS)}],
                },
            ],
        },
        "evidence_requirements": {
            "requirements": [
                "Solve every candidate with seeds 7, 19 and 43; allocate no additional attempts.",
                "Score a candidate by its best worst-sample fidelity over pulses passing replay "
                "and constraints; report seed spread, refinement change and termination.",
                "Initial candidates are 10, 30 and 60 fF. If the best candidate is the smallest "
                "or largest solved and the range extends beyond it, solve that range end; "
                "otherwise solve the midpoint between it and its better neighbour.",
            ],
            "checks": {"gate": CHECKS},
        },
        "checkpoint": {"finished_attempts": 6},
        "review_criteria": [],
        "stop_criteria": [
            "Stop when the best candidate and its better neighbour differ by no more than the "
            "larger of their seed spreads, or after five candidates.",
        ],
        "limits": {},
    }


def next_candidate(scores, attempted):
    """Apply the declared rule to {capacitance: {"score", "spread"}}.

    Returns the next capacitance or None, the reason, and any pair left unresolved.
    """
    if not scores:
        return None, "no candidate has a pulse with checked evidence", None
    if len(attempted) >= MAX_CANDIDATES:
        return None, f"{MAX_CANDIDATES} candidates solved", None
    following, reason, pair = _rule(scores)
    if following in attempted:
        return None, f"{following:g} fF was already solved without a checked pulse", None
    return following, reason, pair


def _rule(scores):
    solved = sorted(scores)
    best = max(solved, key=lambda c: scores[c]["score"])
    if best == solved[-1] and best < RANGE[1]:
        return RANGE[1], f"{best:g} fF is best and the largest solved; solve the range end", None
    if best == solved[0] and best > RANGE[0]:
        return RANGE[0], f"{best:g} fF is best and the smallest solved; solve the range end", None
    index = solved.index(best)
    neighbours = [solved[i] for i in (index - 1, index + 1) if 0 <= i < len(solved)]
    other = max(neighbours, key=lambda c: scores[c]["score"])
    difference = scores[best]["score"] - scores[other]["score"]
    spread = max(scores[best]["spread"], scores[other]["spread"])
    if difference <= spread:
        reason = (
            f"{best:g} and {other:g} fF differ by {difference:.2e}, within the larger seed "
            f"spread {spread:.2e}; the ranking between them is unresolved"
        )
        return None, reason, sorted((best, other))
    return (
        (best + other) / 2,
        f"bisect between {best:g} fF and its better neighbour {other:g} fF",
        None,
    )


def conclusion(scores, pair):
    """The retained candidates, the report's recommendation and next experiment, and stop reason."""
    if not scores:
        return (
            [],
            "No candidate is retained: none has a pulse that passes replay and constraints with "
            "uncertainty evidence. The failed attempts are retained.",
            "Review the failed attempts; another regime, bound or solver setting needs a revised base.",
            "insufficient_progress",
        )
    lower_bound = (
        " Each score is a lower bound: the best checked pulse's minimum fidelity over the "
        "declared samples."
    )
    if not pair:
        best = max(scores, key=lambda c: scores[c]["score"])
        return (
            [best],
            f"Retain the evaluated {best:g} fF candidate, the highest known worst-sample "
            "fidelity." + lower_bound,
            "",
            "evidence_completion",
        )
    first, second = (scores[c] for c in pair)
    refined = "selected_refined_worst_fidelity"
    order = (first["score"] - second["score"]) * (first[refined] - second[refined])
    return (
        pair,
        f"Retain the evaluated {pair[0]:g} and {pair[1]:g} fF candidates; their ranking is "
        f"unresolved. Their selected pulses' worst-sample fidelities are {first['score']:.10f} "
        f"and {second['score']:.10f}, and {first[refined]:.10f} and {second[refined]:.10f} "
        f"after refinement, which {'reverses' if order < 0 else 'keeps'} their order."
        + lower_bound,
        "Resolving the remaining pair needs attempts the evidence requirements do not "
        "allocate, such as more seeds or another solver.",
        "evidence_completion",
    )


class Cli:
    """The same calls as the MCP tools, issued as `entanglix` commands."""

    def __init__(self, runs_root):
        self.runs_root = runs_root

    async def call(self, name, arguments):
        args = dict(arguments)
        command = [name.replace("_", "-")]
        if name in {"sweep", "control"}:
            command += [args.pop("protocol"), "--runs-root", str(self.runs_root)]
            command += ["--investigation", args.pop("investigation_id"), "--base", args.pop("base")]
            for target, value in args.pop("values", {}).items():
                command += ["--vary", f"{target}={value}"]
            for parent in args.pop("parent_run_ids", []):
                command += ["--parent-run", parent]
        elif name == "create_investigation":
            command += ["--runs-root", str(self.runs_root)]
        elif name == "read_run":
            command += [args.pop("run"), "--wait", str(args.pop("wait_seconds", 0))]
            args.pop("detail", None)
        elif name == "check":
            command += [args.pop("run"), *args.pop("checks")]
        else:
            command.append(args.pop("investigation"))
            if name == "stop_investigation":
                command.append(args.pop("reason"))
        with tempfile.TemporaryDirectory() as scratch:
            for key in ("spec", "finding"):
                if key in args:
                    path = Path(scratch) / f"{key}.toml"
                    path.write_text(tomli_w.dumps(args.pop(key)))
                    command.append(str(path))
            if args:
                raise ValueError(f"unmapped arguments for {name}: {sorted(args)}")
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "entanglix", *command,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )  # fmt: skip
            stdout, stderr = await process.communicate()
        if not stdout.strip():
            raise RuntimeError(f"{name} failed: {stderr.decode()[-2000:]}")
        return json.loads(stdout)


class Mcp:
    def __init__(self, session):
        self.session = session

    async def call(self, name, arguments):
        result = await self.session.call_tool(name, arguments)
        text = result.content[0].text
        if result.is_error:
            raise RuntimeError(f"{name} failed: {text}")
        return json.loads(text)


@asynccontextmanager
async def transport(via, runs_root):
    if via == "cli":
        yield Cli(runs_root)
        return
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "entanglix.mcp", "--runs-root", str(runs_root), "--device", "",
              "--runner-extras", *runner_extras()],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
    )  # fmt: skip
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            yield Mcp(session)


class Study:
    def __init__(self, tool, investigation):
        self.tool, self.investigation = tool, investigation
        self.runs, self.candidates, self.screen = [], {}, None

    async def job(self, name, **arguments):
        if name != "check":
            arguments.update(investigation_id=Path(self.investigation).name)
        admitted = await self.tool.call(name, arguments)
        record = {"status": "active"}
        while record["status"] in {"active", "pending"}:
            record = await self.tool.call(
                "read_run", {"run": admitted["path"], "wait_seconds": 30, "detail": True}
            )
        record["run_id"] = record["manifest"]["run_id"]
        self.runs.append(record["run_id"])
        return record

    def learned(self):
        """A compact candidate table; individual pulses stay in the attempts and the summary."""
        points = self.screen["points"]
        frequencies = [p["f01_ghz"] for p in points]
        anharmonicities = [p["anharmonicity_ghz"] for p in points]
        lines = [
            f"Screen {self.screen['run_id']}: f01 {min(frequencies):.2f} to "
            f"{max(frequencies):.2f} GHz and anharmonicity {min(anharmonicities):.2f} to "
            f"{max(anharmonicities):.2f} GHz over the screened Csh values."
        ]
        for row in self.table():
            if "score" not in row:
                states = ", ".join(
                    f"seed {p['seed']} {p.get('check_status', p['status'])}" for p in row["pulses"]
                )
                lines.append(f"- {row['Csh_fF']:g} fF: no pulse with checked evidence ({states}).")
                continue
            lines.append(
                f"- {row['Csh_fF']:g} fF: worst-sample fidelity {row['score']:.10f} "
                f"(seed {row['selected_seed']}, run {row['selected_run_id']}), "
                f"{row['selected_refined_worst_fidelity']:.10f} refined; seed spread "
                f"{row['spread']:.1e}; largest refinement change "
                f"{row['largest_refinement_change']:.1e}; unconverged seeds "
                f"{row['unconverged_seeds'] or 'none'}."
            )
        return "\n".join(lines)

    async def finding(self, kind, decision, rationale, next_experiment):
        return await self.tool.call(
            "record_finding",
            {
                "investigation": self.investigation,
                "finding": {
                    "kind": kind,
                    "runs": list(self.runs),
                    "learned": self.learned(),
                    "unresolved": LIMITS,
                    "decision": decision,
                    "rationale": rationale,
                    "next_experiment": next_experiment,
                },
            },
        )

    async def screen_range(self):
        protocol = EXAMPLE / "screen.toml"
        record = await self.job("sweep", protocol=str(protocol), base="screen")
        if record["status"] != "completed":
            raise RuntimeError(f"screen did not complete: {record['path']}")
        self.screen = {"run_id": record["run_id"], "points": screen_points(Path(record["path"]))}

    async def solve(self, capacitance):
        pulses = []
        for seed in SEEDS:
            solve = await self.job(
                "control",
                protocol=str(EXAMPLE / "control.toml"),
                base="gate",
                values={"parameters.Csh": capacitance, "pulse.seed": seed},
                parent_run_ids=[self.screen["run_id"]],
            )
            item = {"seed": seed, "run_id": solve["run_id"], "status": solve["status"]}
            if solve["status"] == "completed":
                result = solve["result"]
                item.update(
                    termination=result["termination"],
                    solver_reported_fidelity=result["metrics"].get("solver_reported_fidelity"),
                )
                report = await self.job("check", run=solve["path"], checks=CHECKS)
                # A check job that did not complete leaves this pulse without checked evidence.
                listed = report.get("result", {}).get("report", {}).get("checks", [])
                checks = {name: {} for name in CHECKS} | {c["name"]: c for c in listed}
                item.update(
                    check_run_id=report["run_id"],
                    check_status=report["status"],
                    outcomes={name: checks[name].get("outcome") for name in CHECKS},
                    replayed_fidelity=checks["replay"].get("fidelity"),
                    leakage=checks["replay"].get("leakage"),
                )
                uncertainty = checks["uncertainty"].get("values")
                refined = checks["uncertainty_refinement"].get("values")
                if uncertainty and refined:
                    item.update(
                        worst_fidelity=uncertainty["minimum_fidelity"],
                        worst_sample=uncertainty["samples"][uncertainty["minimum_sample"]][
                            "values"
                        ],
                        nominal_fidelity=uncertainty["nominal"]["fidelity"],
                        refined_worst_fidelity=refined["minimum_fidelity"],
                        refinement_minimum_change=refined["minimum_change"],
                        largest_sample_change=refined["largest_sample_change"],
                        frequency_ghz=uncertainty["nominal"]["frame_frequencies_ghz"],
                    )
            pulses.append(item)
        self.candidates[capacitance] = pulses

    def table(self):
        rows = []
        for capacitance, pulses in sorted(self.candidates.items()):
            admissible = [
                p for p in pulses
                if "worst_fidelity" in p
                and p["outcomes"]["replay"] == p["outcomes"]["constraints"] == "pass"
            ]  # fmt: skip
            row = {"Csh_fF": capacitance, "pulses": pulses}
            if admissible:
                best = max(admissible, key=lambda p: p["worst_fidelity"])
                worst = [p["worst_fidelity"] for p in admissible]
                row.update(
                    score=best["worst_fidelity"],
                    selected_refined_worst_fidelity=best["refined_worst_fidelity"],
                    selected_seed=best["seed"],
                    selected_run_id=best["run_id"],
                    spread=max(worst) - min(worst),
                    largest_refinement_change=max(
                        abs(p["largest_sample_change"]) for p in admissible
                    ),
                    unconverged_seeds=[
                        p["seed"] for p in pulses if not p.get("termination", {}).get("converged")
                    ],
                )
            rows.append(row)
        return rows

    def scores(self):
        return {row["Csh_fF"]: row for row in self.table() if "score" in row}


def screen_points(run):
    """Transition frequency and anharmonicity at each retained screening point."""
    axis = tomllib.loads((run / "spec/sweep.toml").read_text())["axes"][0]
    rows = []
    for point in sorted(run.glob("artifacts/points/*")):
        coordinate = tomllib.loads((point / "result.toml").read_text())["coordinate"]
        with np.load(point / "output.npz") as arrays:
            energies = arrays["energies_ghz"]
        rows.append(
            {
                axis["target"]: axis["values"][coordinate[0]],
                "f01_ghz": float(energies[1] - energies[0]),
                "anharmonicity_ghz": float(energies[2] - 2 * energies[1] + energies[0]),
            }
        )
    return rows


async def draft(via, runs_root):
    async with transport(via, runs_root) as tool:
        created = await tool.call("create_investigation", {"spec": terms()})
    print(json.dumps(created, indent=2))
    print(
        "Review the terms, then start the investigation as the operator:\n"
        f"  entanglix investigation-decision {created['path']} start --revision 1"
    )


async def run(via, runs_root, investigation, output):
    async with transport(via, runs_root) as tool:
        study = Study(tool, investigation)
        await study.screen_range()
        await study.finding(
            "assessment", "continue",
            "The screen covers the approved range; solve the declared initial candidates.",
            "Solve 10, 30 and 60 fF with every required seed.",
        )  # fmt: skip
        queue, reason = list(INITIAL), "declared initial candidate"
        while True:
            capacitance = queue.pop(0)
            await study.solve(capacitance)
            if queue:
                await study.finding(
                    "assessment", "continue", f"Continue the {reason}s.",
                    f"Solve {queue[0]:g} fF with every required seed.",
                )  # fmt: skip
                continue
            following, reason, pair = next_candidate(study.scores(), set(study.candidates))
            if following is None:
                break
            queue.append(following)
            await study.finding(
                "assessment", "continue", f"Next candidate by the declared rule: {reason}.",
                f"Solve {following:g} fF with every required seed.",
            )  # fmt: skip
        retained, recommendation, following, stop = conclusion(study.scores(), pair)
        report = await study.finding(
            "report", "stop", f"Stop rule: {reason}. {recommendation}", following
        )
        stopped = await tool.call(
            "stop_investigation", {"investigation": investigation, "reason": stop}
        )
        status = await tool.call("read_investigation", {"investigation": investigation})
    summary = {
        "via": via,
        "investigation": investigation,
        "screen": study.screen,
        "candidates": study.table(),
        "stop_reason": reason,
        "retained_candidates_fF": retained,
        "recommendation": recommendation,
        "report_event": report["event_id"],
        "stopped": stopped.get("state", stopped),
        "usage": status["usage"],
        "runs": study.runs,
    }
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("stop_reason", "recommendation")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("draft", "run"):
        command = commands.add_parser(name)
        command.add_argument("--via", choices=("cli", "mcp"), required=True)
        command.add_argument("--runs-root", type=Path, required=True)
        if name == "run":
            command.add_argument("--investigation", required=True)
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "draft":
        asyncio.run(draft(args.via, args.runs_root))
    else:
        asyncio.run(run(args.via, args.runs_root, args.investigation, args.output))


if __name__ == "__main__":
    main()
