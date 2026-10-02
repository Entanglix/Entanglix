"""System validation, authored CPU jobs, and retained attempt inspection."""

import argparse
import json
import signal
import sys
from pathlib import Path

from entanglix import __version__


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


def _operator_meaning(name: str, provenance: dict[str, object]) -> str:
    if "asset_sha256" in provenance:
        return "declared by the spec"
    if "subsystems" in provenance:
        subsystem, _, local = name.partition(".")
        return _operator_meaning(local, provenance["subsystems"][subsystem])
    if name == "charge" or (name.startswith("node") and name.endswith(".charge")):
        return "n − ng in Cooper-pair units"
    return {
        "phase": "extended-coordinate phase in radians",
        "cos_phase": "cosine of phase",
        "sin_phase": "sine of phase",
        "annihilation": "oscillator lowering operator a",
        "number": "oscillator occupation a†a",
        "position": "oscillator quadrature a + a†",
    }.get(name, "declared by the spec")


def _print_circuit_diagnostics(provenance: dict[str, object], label: str) -> None:
    if "transformation" in provenance:
        print(
            f"{label} transformation ({provenance['transformation_source']}): {provenance['transformation']}"
        )
        print(f"{label} node offset charges: {provenance['node_offsets']}")
        print(f"{label} derived coordinate offset charges: {provenance['coordinate_offsets']}")
        if provenance["dropped_offsets"]:
            print(f"{label} dropped extended-coordinate offsets: {provenance['dropped_offsets']}")
        for loop in provenance["oriented_loops"]:
            first, second = loop["nodes"]
            print(
                f"{label} closure branch {loop['branch']} ({first}→{second}): "
                f"{loop['value']} {loop['unit']}"
            )
    for name, child in provenance.get("subsystems", {}).items():
        _print_circuit_diagnostics(child, name)


def main(argv: list[str] | None = None) -> int:
    parser = Parser(prog="entanglix", description="Quantum system research harness.")
    parser.add_argument("--version", action="version", version=f"entanglix {__version__}")
    subcommands = parser.add_subparsers(dest="command")
    validate = subcommands.add_parser("validate", help="validate a system TOML")
    validate.add_argument("system", help="path to system.toml")
    script = subcommands.add_parser(
        "run-script", help="run an authored CPU script with retained records"
    )
    script.add_argument("script", type=Path, help="standalone Python script")
    script.add_argument(
        "--system", required=True, type=Path, help="system.toml required for every research script"
    )
    script.add_argument(
        "--runs-root",
        type=Path,
        default=Path("research_runs"),
        help="runs root (default: ./research_runs)",
    )
    script.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="freeze an additional input under spec/; repeat as needed",
    )
    script.add_argument(
        "--parent-run", action="append", default=[], help="parent run ID for lineage"
    )
    script.add_argument(
        "--timeout", type=float, help="job time limit in seconds; default: unlimited"
    )
    abort = subcommands.add_parser("abort-run", help="stop an attempt's job process group")
    abort.add_argument("run", type=Path, help="attempt directory")
    reconcile = subcommands.add_parser(
        "reconcile-run", help="record an interrupted attempt's uncertainty"
    )
    reconcile.add_argument("run", type=Path, help="attempt directory")
    reconcile.add_argument(
        "--state", required=True, help="last observed state or recovery diagnostic"
    )
    rerun = subcommands.add_parser(
        "rerun-script", help="replay frozen inputs under a new linked run ID"
    )
    rerun.add_argument("run", type=Path, help="earlier attempt directory")
    rerun.add_argument(
        "--runs-root", type=Path, help="output root; default: the earlier attempt's root"
    )
    rerun.add_argument("--timeout", type=float, help="override the recorded time limit in seconds")
    subcommands.add_parser(
        "capabilities", help="list runner operations and installed engine/device support"
    )
    listing = subcommands.add_parser("list-runs", help="list all retained attempts")
    listing.add_argument(
        "--runs-root",
        type=Path,
        default=Path("research_runs"),
        help="runs root (default: ./research_runs)",
    )
    reading = subcommands.add_parser("read-run", help="read an attempt's records and log tail")
    reading.add_argument("run", type=Path, help="attempt directory")
    reading.add_argument("--policy-version", help="verification policy digest; default: current")
    comparison = subcommands.add_parser(
        "compare-reference", help="verify a completed computation independently"
    )
    comparison.add_argument("run", type=Path, help="source attempt directory")
    comparison.add_argument("--timeout", type=float, help="verification time limit in seconds")
    simulation = subcommands.add_parser("simulate", help="run and verify one simulation protocol")
    simulation.add_argument("protocol", type=Path, help="path to simulate.toml")
    simulation.add_argument("--runs-root", type=Path, default=Path("research_runs"))
    simulation.add_argument("--parent-run", action="append", default=[])
    replay = subcommands.add_parser(
        "rerun-simulation", help="replay a frozen simulation with linked lineage"
    )
    replay.add_argument("run", type=Path, help="earlier simulation attempt")
    replay.add_argument("--runs-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        from entanglix.models import SpecError, load_system

        try:
            system = load_system(args.system)
        except SpecError as exc:
            print(f"entanglix: invalid system: {exc}", file=sys.stderr)
            return 64
        print(f"valid system: {system.id} ({system.basis}, dimensions {system.dimensions})")
        print("operators:")
        for name in sorted(system.operators):
            print(f"  {name}: {_operator_meaning(name, system.provenance)}")
        _print_circuit_diagnostics(system.provenance, "circuit")
        return 0
    if args.command in {
        "run-script",
        "capabilities",
        "list-runs",
        "read-run",
        "abort-run",
        "reconcile-run",
        "rerun-script",
        "compare-reference",
        "simulate",
        "rerun-simulation",
    }:
        from entanglix.runner import (
            abort_run,
            list_capabilities,
            list_runs,
            read_run,
            record_reconciliation,
            rerun_script,
            run_script,
        )

        if args.command in {
            "run-script",
            "rerun-script",
            "compare-reference",
            "simulate",
            "rerun-simulation",
        }:
            # A supervising command stops its job on termination or hangup, as on Ctrl-C.
            signal.signal(signal.SIGTERM, signal.default_int_handler)
            signal.signal(signal.SIGHUP, signal.default_int_handler)
        try:
            if args.command == "run-script":
                inputs = {}
                for item in args.input:
                    name, separator, path = item.partition("=")
                    if not separator or not name or not path or name in inputs:
                        raise ValueError("--input requires a unique NAME=PATH")
                    inputs[name] = Path(path)
                result = run_script(
                    args.system,
                    args.script,
                    inputs=inputs,
                    runs_root=args.runs_root,
                    parent_run_ids=tuple(args.parent_run),
                    timeout_seconds=args.timeout,
                )
            elif args.command == "abort-run":
                result = abort_run(args.run)
            elif args.command == "reconcile-run":
                record_reconciliation(args.run, last_observed_state=args.state)
                result = read_run(args.run)
            elif args.command == "rerun-script":
                result = rerun_script(
                    args.run, runs_root=args.runs_root, timeout_seconds=args.timeout
                )
            elif args.command == "capabilities":
                result = list_capabilities()
            elif args.command == "compare-reference":
                from entanglix.corrector.verification import compare_reference

                result = compare_reference(args.run, timeout_seconds=args.timeout)
            elif args.command in {"simulate", "rerun-simulation"}:
                from entanglix.protocols.simulate import rerun_simulation, simulate

                result = (
                    simulate(
                        args.protocol,
                        runs_root=args.runs_root,
                        parent_run_ids=tuple(args.parent_run),
                    )
                    if args.command == "simulate"
                    else rerun_simulation(args.run, runs_root=args.runs_root)
                )
            elif args.command == "list-runs":
                result = list_runs(args.runs_root)
            else:
                result = read_run(args.run, policy_version=args.policy_version)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            print(f"entanglix: {exc}", file=sys.stderr)
            return 64
        print(json.dumps(result, default=str, indent=2))
        return (
            result["exit_code"]
            if args.command
            in {"run-script", "rerun-script", "compare-reference", "simulate", "rerun-simulation"}
            else 0
        )
    parser.print_help()
    return 0
