"""Scripted calibration of the example's transmon, through the CLI or an MCP client.

`run` fits the training map, requests the holdout, identifiability and refinement checks with
the separate holdout measurement, writes a device-update proposal and a summary, and prints the
operator's apply command. Both paths make the same calls. Applying the proposal is the
operator's action; afterwards `applied` adds its record and the updated system copy to the
summary.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path

from generate import OFFSET, PERIOD, TRUTH

EXAMPLE = Path(__file__).resolve().parent
CHECKS = ["holdout", "identifiability", "refinement"]
KNOWN = {**TRUTH, "offset": OFFSET, "period": PERIOD}


def runner_extras():
    """The runner profile the CLI uses: ENTANGLIX_RUNNER_EXTRAS, else the project's runner table."""
    if "ENTANGLIX_RUNNER_EXTRAS" in os.environ:
        return [v.strip() for v in os.environ["ENTANGLIX_RUNNER_EXTRAS"].split(",") if v.strip()]
    config = tomllib.loads((EXAMPLE.parents[1] / "pyproject.toml").read_text())
    return config.get("tool", {}).get("entanglix", {}).get("runner", {}).get("extras", [])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Cli:
    """The same calls as the MCP tools, issued as `entanglix` commands."""

    def __init__(self, runs_root):
        self.runs_root = runs_root

    async def call(self, name, arguments):
        args = dict(arguments)
        command = [name.replace("_", "-")]
        if name == "calibrate":
            command += [args.pop("protocol"), "--runs-root", str(self.runs_root)]
        elif name == "read_run":
            command += [args.pop("run"), "--wait", str(args.pop("wait_seconds", 0))]
            args.pop("detail", None)
        elif name == "check":
            command += [args.pop("run"), *args.pop("checks"), "--holdout", args.pop("holdout")]
        elif name == "propose_device_update":
            command.append(args.pop("run"))
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


async def job(tool, name, arguments):
    admitted = await tool.call(name, arguments)
    record = {"status": "active"}
    while record["status"] in {"active", "pending"}:
        record = await tool.call(
            "read_run", {"run": admitted["path"], "wait_seconds": 30, "detail": True}
        )
    if record["status"] != "completed":
        raise RuntimeError(f"{name} {record['status']}: {record['log_tail'][-2000:]}")
    return record


def fitted(record):
    """The fit's outcome, the solution's values against the known ones, and its diagnostics."""
    result = record["result"]
    summary = {
        "outcome": result["outcome"],
        "datasets": {name: item["sha256"] for name, item in result["datasets"].items()},
        "frozen_inputs": sorted(record["manifest"]["spec_sha256"]),
        "starts": [
            {key: start[key] for key in ("status", "cycles", "anchor", "solution") if key in start}
            for start in result["starts"]
        ],
        "job_seconds": record["finished"]["usage"]["wall_seconds"],
    }
    if result["outcome"] != "solution":
        return summary
    solution = result["solutions"][0]
    sigma = solution["uncertainty"]
    summary["parameters"] = {
        name: {
            "known": known,
            "fitted": solution["values"][name],
            "uncertainty": sigma.get(name),
            "standardised_error": (
                (solution["values"][name] - known) / sigma[name] if name in sigma else None
            ),
            "uncertainty_over_prior_width": solution["uncertainty_over_prior_width"].get(name),
        }
        for name, known in KNOWN.items()
    }
    for key in ("statistics", "counts", "rank", "bounds_hit", "starts"):
        summary[f"solution_{key}"] = solution[key]
    summary["equivalents_removed"] = solution["equivalents_removed"]
    summary["conventions_not_applied"] = solution["conventions_not_applied"]
    summary["correlation"] = solution["correlation"]
    return summary


def checked(record):
    items = {item["name"]: item for item in record["result"]["report"]["checks"]}
    holdout = items["holdout"]
    values = holdout.get("values", {})
    return {
        "run_id": record["manifest"]["run_id"],
        "job_seconds": record["finished"]["usage"]["wall_seconds"],
        "holdout": {
            "outcome": holdout.get("outcome", "descriptive"),
            "dataset": holdout.get("dataset"),
            "frozen": sorted(
                name for name in record["manifest"]["spec_sha256"] if "holdout" in name
            ),
            **{
                key: values[key]
                for key in (
                    "form",
                    "counts",
                    "coverage_over_peaks",
                    "coverage_over_scored",
                    "mean_squared_standardised_residual",
                )
                if key in values
            },
        },
        "identifiability": {
            "outcome": items["identifiability"].get("outcome", "descriptive"),
            "tolerances": items["identifiability"].get("tolerances"),
            "agreement": items["identifiability"].get("values", {}).get("agreement"),
        },
        "refinement": {
            "outcome": items["refinement"].get("outcome", "descriptive"),
            "declaration": items["refinement"].get("refinement"),
            **items["refinement"].get("values", {}),
        },
    }


def relative(path):
    return os.path.relpath(path, EXAMPLE.parents[1])


async def run(via, runs_root, output):
    holdout = EXAMPLE / "holdout.toml"
    summary = {"via": via, "holdout_sha256": None}
    async with transport(via, runs_root) as tool:
        fit = await job(tool, "calibrate", {"protocol": str(EXAMPLE / "calibrate.toml")})
        summary.update(source_run_id=fit["manifest"]["run_id"], fit=fitted(fit))
        if summary["fit"]["outcome"] == "solution":
            report = await job(
                tool, "check", {"run": fit["path"], "checks": CHECKS, "holdout": str(holdout)}
            )
            summary["checks"] = checked(report)
            summary["holdout_sha256"] = summary["checks"]["holdout"]["dataset"]["sha256"]
            proposal = await tool.call("propose_device_update", {"run": fit["path"]})
            summary["proposal"] = {
                "path": relative(proposal["path"]),
                **{
                    key: proposal.get(key, {})
                    for key in (
                        "solution",
                        "solution_outcome",
                        "bounds_hit",
                        "model_inputs",
                        "parameters",
                        "not_proposed",
                    )
                },
                "verification": [item["checks"] for item in proposal["verification"]],
            }
    summary["holdout_in_fitting_inputs"] = summary["holdout_sha256"] in summary["fit"][
        "datasets"
    ].values() or any("holdout" in name for name in summary["fit"]["frozen_inputs"])
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if "proposal" in summary:
        print(
            "operator: copy examples/transmon-calibration/system.toml outside the runs root and "
            f"run\n  entanglix apply-device-update {summary['proposal']['path']} <copy>"
        )
    print(json.dumps({"outcome": summary["fit"]["outcome"], "summary": str(output)}))


def applied(summary_path, device):
    """Add the operator's application record and the updated system copy to the summary."""
    from entanglix.models import load_system

    summary = json.loads(summary_path.read_text())
    proposal = EXAMPLE.parents[1] / summary["proposal"]["path"]
    record = tomllib.loads(proposal.with_suffix(".applied.toml").read_text())
    document = tomllib.loads(device.read_text())
    system = load_system(device)
    summary["application"] = {
        **{key: record[key] for key in record if key not in {"source", "system"}},
        "source": {"kind": record["source"]["kind"]},
        "system_before_is_the_example": record["system_sha256_before"]
        == digest(EXAMPLE / "system.toml"),
        "updated": {name: document["parameters"][name] for name in record["parameters"]},
        "validated": {"id": system.id, "dimensions": list(system.dimensions)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False, default=str) + "\n")
    print(json.dumps(summary["application"]["updated"], default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run")
    running.add_argument("--via", choices=("cli", "mcp"), required=True)
    running.add_argument("--runs-root", type=Path, required=True)
    running.add_argument("--output", type=Path, required=True)
    recording = commands.add_parser("applied")
    recording.add_argument("--summary", type=Path, required=True)
    recording.add_argument("--device", type=Path, required=True, help="the applied system copy")
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args.via, args.runs_root, args.output))
    else:
        applied(args.summary, args.device)


if __name__ == "__main__":
    main()
