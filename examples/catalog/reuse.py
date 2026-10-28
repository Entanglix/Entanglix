"""Stage a retained co-design candidate, then reuse it after the operator promotes it.

Both transports use ordinary harness operations. This client never promotes.
The source run supplies the frozen gate specification; the entry supplies its device and pulse.
"""

import argparse
import asyncio
import json
import os
import shlex
import sys
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tomli_w

CHECKS = ["replay", "constraints", "uncertainty", "uncertainty_refinement"]


class Client:
    def __init__(self, root, session=None):
        self.root, self.session = root, session

    async def call(self, name, **args):
        if self.session is not None:
            result = await self.session.call_tool(name, args)
            if result.is_error:
                raise RuntimeError(result.content[0].text)
            return json.loads(result.content[0].text)
        command = [sys.executable, "-m", "entanglix", name.replace("_", "-")]
        if name == "register_result":
            command += [args["run"], "--design", args["design"], "--claim", args["claim"]]
        elif name in {"list_catalog", "read_entry"}:
            if name == "read_entry":
                command.append(args["identifier"])
            command += ["--runs-root", str(self.root)]
        elif name in {"control", "sweep"}:
            command += [args["protocol"], "--runs-root", str(self.root)]
        elif name == "check":
            command += [args["run"], *args["checks"]]
        elif name == "read_run":
            command += [args["run"], "--wait", "30"]
        else:
            raise ValueError(f"unmapped example operation: {name}")
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            raise RuntimeError(stderr.decode().strip() or stdout.decode())
        result = json.loads(stdout)
        return {"result": result} if isinstance(result, list) else result

    async def job(self, name, **args):
        launched = await self.call(name, **args)
        while True:
            record = await self.call("read_run", run=launched["path"], wait_seconds=30, detail=True)
            if record["status"] not in {"active", "pending"}:
                break
        if record["status"] != "completed":
            raise RuntimeError(
                f"{record['path']}: {record['status']}\n{record.get('log_tail', '')}"
            )
        return record


@asynccontextmanager
async def connection(via, root):
    if via == "cli":
        yield Client(root)
        return
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from entanglix import runner

    if "ENTANGLIX_RUNNER_EXTRAS" in os.environ:
        extras = [value.strip() for value in os.environ["ENTANGLIX_RUNNER_EXTRAS"].split(",")]
        extras = [value for value in extras if value]
    else:
        project = Path(runner.__file__).resolve().parents[2] / "pyproject.toml"
        extras = (
            tomllib.loads(project.read_text())
            .get("tool", {})
            .get("entanglix", {})
            .get("runner", {})
            .get("extras", [])
        )
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "entanglix.mcp", "--runs-root", str(root), "--device", "",
              "--runner-extras", *extras],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
    )  # fmt: skip
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            yield Client(root, session)


def inputs(root, entry, workspace):
    """Use the copied device without overriding Csh, and repeat the source's gate."""
    device = root / entry["device"]["path"] / "system.toml"
    model = tomllib.loads(device.read_text())
    gate = tomllib.loads((root / entry["source"]["run"] / "spec/control.toml").read_text())
    gate.pop("system", None)
    gate.pop("parameters", None)  # The entry already holds the values the source solve used.
    gate["system_entry"] = entry["entry_id"]
    gate["starting_pulse"] = {"entry": entry["entry_id"]}
    for name in ("init", "seed"):
        gate["pulse"].pop(name, None)
    ej = model["parameters"]["EJ"]["value"]
    spectrum = {
        "format_version": 1,
        "system_entry": entry["entry_id"],
        "kind": "spectrum",
        "levels": 3,
        "parameters": {"EJ": ej},
    }
    sweep = {
        "format_version": 1,
        "base": "spectrum.toml",
        "axes": [{"id": "EJ", "target": "parameters.EJ", "values": [ej - 0.5, ej, ej + 0.5]}],
    }
    workspace.mkdir(parents=True, exist_ok=True)
    for name, document in (("control", gate), ("spectrum", spectrum), ("sweep", sweep)):
        (workspace / f"{name}.toml").write_text(tomli_w.dumps(document))
    return {"Csh_fF": model["parameters"]["Csh"]["value"], "EJ_ghz": ej}


def check_summary(record):
    checks = []
    for item in record["result"]["report"]["checks"]:
        row = {key: item[key] for key in (
            "name", "outcome", "reason", "scope", "coverage", "residual", "tolerance", "fidelity", "leakage"
        ) if key in item}  # fmt: skip
        values = item.get("values", {})
        row.update({key: values[key] for key in (
            "coverage", "minimum_fidelity", "minimum_sample", "largest_sample_change"
        ) if key in values})  # fmt: skip
        checks.append(row)
    return checks


async def reuse(client, root, identifier, workspace):
    entry = await client.call("read_entry", identifier=identifier)
    parameters = inputs(root, entry, workspace)
    sweep = await client.job("sweep", protocol=str(workspace / "sweep.toml"))
    warm = await client.job("control", protocol=str(workspace / "control.toml"))
    checked = await client.job("check", run=warm["path"], checks=CHECKS)
    trace = tomllib.loads((Path(warm["path"]) / "artifacts/solver-trace.toml").read_text())
    rows = []
    axis = tomllib.loads((workspace / "sweep.toml").read_text())["axes"][0]["values"]
    for point in sorted(Path(sweep["path"]).glob("artifacts/points/*")):
        with np.load(point / "output.npz", allow_pickle=False) as arrays:
            energies = arrays["energies_ghz"]
        index = tomllib.loads((point / "result.toml").read_text())["coordinate"][0]
        rows.append({"EJ_ghz": axis[index], "f01_ghz": float(energies[1] - energies[0])})
    entry = await client.call("read_entry", identifier=identifier)
    listed = await client.call("list_catalog")
    if identifier not in {row["id"] for row in listed["result"]}:
        raise RuntimeError("the entry is absent from the catalog listing")
    return {
        "runs_root": str(root),
        "entry": {
            key: entry[key]
            for key in (
                "entry_id",
                "staged_id",
                "source",
                "promoter",
                "promoted_at",
                "claim",
                "parent_entry_ids",
                "checks",
                "checks_not_run",
                "unusable_evidence",
                "lineage",
                "reuses",
            )
        },  # fmt: skip
        "device": parameters,
        "spectrum": rows,
        "warm_start": {
            "starting_pulse": warm["manifest"]["launch"]["starting_pulse"],
            "termination": warm["result"]["termination"],
            "first_objective": trace.get("objective_calls", [{}])[0].get("objective"),
            "solver_reported_fidelity": warm["result"]["metrics"].get("solver_reported_fidelity"),
            "checks": check_summary(checked),
        },
        "runs": {
            name: {
                "run": str(Path(record["path"]).relative_to(root)),
                "run_id": record["manifest"]["run_id"],
                "parent_run_ids": record["manifest"]["parent_run_ids"],
                "source_entry_ids": record["manifest"].get("source_entry_ids", []),
                "usage": record["finished"]["usage"],
            }
            for name, record in (
                ("sweep", sweep),
                ("warm_start", warm),
                ("control_check", checked),
            )
        },  # fmt: skip
        "provenance": {
            "code": warm["manifest"]["code"],
            "env_hash": warm["manifest"]["env_hash"],
            "policy_version": checked["manifest"]["launch"]["policy_version"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("stage", "reuse"):
        command = commands.add_parser(name)
        command.add_argument("--via", choices=("cli", "mcp"), required=True)
        if name == "stage":
            command.add_argument("--source", type=Path, required=True)
        else:
            command.add_argument("--entry", required=True)
            command.add_argument("--runs-root", type=Path, required=True)
            command.add_argument("--workspace", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = (
        args.source.resolve().parent.parent if args.command == "stage" else args.runs_root.resolve()
    )

    async def execute():
        async with connection(args.via, root) as client:
            if args.command == "stage":
                staged = await client.call(
                    "register_result", run=str(args.source.resolve()), design="device_and_pulse",
                    claim="60 fF device and seed-19 pulse from the transmon co-design comparison: "
                    "60 and 90 fF were retained with their ranking unresolved. This candidate is "
                    "offered for reuse; its standing rests on its requested checks.",
                )  # fmt: skip
                return {
                    "staged_id": staged["staged_id"], "source": staged["source"],
                    "promotion_command": f"entanglix promote {staged['staged_id']} --runs-root {shlex.quote(str(root))}",
                }  # fmt: skip
            return await reuse(client, root, args.entry, args.workspace.resolve())

    result = {"via": args.via, **asyncio.run(execute())}
    if args.command == "reuse":
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
