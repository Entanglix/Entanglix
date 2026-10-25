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
            bound = f" from parameter {loop['parameter']}" if "parameter" in loop else ""
            print(
                f"{label} closure branch {loop['branch']} ({first}→{second}): "
                f"{loop['value']} {loop['unit']}{bound}"
            )
    for name, child in provenance.get("subsystems", {}).items():
        _print_circuit_diagnostics(child, name)


def _varied_values(items: list[str]) -> dict[str, int | float] | None:
    values = {}
    for item in items:
        target, separator, value = item.partition("=")
        if not separator or not target or target in values:
            raise ValueError("--vary requires a unique TARGET=VALUE")
        # Integer literals stay exact, so seeds above 2**53 keep their value.
        try:
            values[target] = int(value)
        except ValueError:
            values[target] = float(value)
    return values or None


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("entanglix: interrupted", file=sys.stderr)
        return 130


def _main(argv: list[str] | None = None) -> int:
    parser = Parser(prog="entanglix", description="Quantum system research harness.")
    parser.add_argument("--version", action="version", version=f"entanglix {__version__}")
    subcommands = parser.add_subparsers(dest="command")
    validate = subcommands.add_parser("validate", help="validate a system TOML")
    validate.add_argument("system", help="path to system.toml")
    dataset = subcommands.add_parser("validate-dataset", help="validate a measurement dataset")
    dataset.add_argument("dataset", help="path to the dataset TOML")
    dataset.add_argument("--raw-file", help="raw laboratory file whose declared digest is checked")
    splitting = subcommands.add_parser(
        "split-dataset", help="split a dataset by a closed bias range into two new datasets"
    )
    splitting.add_argument("dataset", help="path to the dataset TOML")
    splitting.add_argument("--lower", type=float, required=True, help="range start, in bias units")
    splitting.add_argument("--upper", type=float, required=True, help="range end, in bias units")
    splitting.add_argument("--holdout", required=True, help="new TOML path for the part inside")
    splitting.add_argument("--remainder", required=True, help="new TOML path for the rest")
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
    reading.add_argument("--wait", type=float, default=0, help="wait seconds; no command-line cap")
    checking = subcommands.add_parser("check", help="list relevant checks, or run named checks")
    checking.add_argument("run", type=Path, help="source attempt directory")
    from entanglix.corrector import CHECK_DEFINITIONS

    checking.add_argument("names", nargs="*", help=", ".join(CHECK_DEFINITIONS))
    checking.add_argument("--timeout", type=float, help="check time limit in seconds")
    checking.add_argument("--refinement", type=Path, help="replacement [refinement.*] declarations")
    checking.add_argument("--holdout", type=Path, help="holdout dataset for a calibration run")
    checking.add_argument(
        "--solution", type=int, help="calibration solution index, required among several"
    )
    checking.add_argument(
        "--expect-coverage",
        nargs=2,
        type=float,
        metavar=("WITHIN", "MINIMUM"),
        help="expected holdout coverage within 1 or 2 prediction uncertainties",
    )
    simulation = subcommands.add_parser("simulate", help="run one simulation protocol")
    simulation.add_argument("protocol", type=Path, help="path to simulate.toml")
    simulation.add_argument("--runs-root", type=Path, default=Path("research_runs"))
    simulation.add_argument("--parent-run", action="append", default=[])
    simulation.add_argument(
        "--dry-run", action="store_true", help="preview without launching or creating a run"
    )
    controlling = subcommands.add_parser("control", help="solve a closed-system gate on CPU")
    controlling.add_argument("protocol", type=Path, help="path to control.toml")
    controlling.add_argument("--runs-root", type=Path, default=Path("research_runs"))
    controlling.add_argument(
        "--dry-run", action="store_true", help="validate the model and solver without a run"
    )
    controlling.add_argument("--parent-run", action="append", default=[])
    control_replay = subcommands.add_parser(
        "rerun-control", help="re-solve frozen control inputs with linked lineage"
    )
    control_replay.add_argument("run", type=Path)
    control_replay.add_argument("--runs-root", type=Path)
    calibrating = subcommands.add_parser(
        "calibrate", help="fit device parameters to training spectroscopy on CPU"
    )
    calibrating.add_argument("protocol", type=Path, help="path to calibrate.toml")
    calibrating.add_argument("--runs-root", type=Path, default=Path("research_runs"))
    calibrating.add_argument(
        "--dry-run", action="store_true", help="validate the inputs and preview without a run"
    )
    calibrating.add_argument("--parent-run", action="append", default=[])
    calibration_replay = subcommands.add_parser(
        "rerun-calibration", help="re-fit frozen calibration inputs with linked lineage"
    )
    calibration_replay.add_argument("run", type=Path)
    calibration_replay.add_argument("--runs-root", type=Path)
    proposing = subcommands.add_parser(
        "propose-device-update", help="write a device-update proposal from a calibrate run"
    )
    proposing.add_argument("run", type=Path, help="completed calibrate attempt directory")
    proposing.add_argument(
        "--solution", type=int, help="calibration solution index, required among several"
    )
    applying = subcommands.add_parser(
        "apply-device-update",
        help="operator action: write a proposal into its unchanged system file",
    )
    applying.add_argument("proposal", type=Path, help="proposal file in the run's proposals/")
    applying.add_argument("system", type=Path, help="authored system.toml to update")
    registering = subcommands.add_parser(
        "register-result", help="stage a device, a pulse, or both from a completed run"
    )
    registering.add_argument("run", type=Path, help="completed attempt directory")
    registering.add_argument(
        "--design", required=True, choices=["device", "pulse", "device_and_pulse"]
    )
    registering.add_argument("--claim", required=True, help="what the candidate is offered as")
    registering.add_argument("--point", type=int, help="sweep point index")
    registering.add_argument(
        "--solution", type=int, help="calibration solution index, required among several"
    )
    registering.add_argument("--artifact", help="a system or pulse file the script published")
    catalog_listing = subcommands.add_parser("list-catalog", help="list staged candidates")
    catalog_listing.add_argument("--runs-root", type=Path, default=Path("research_runs"))
    entry = subcommands.add_parser(
        "read-entry", help="read a staged candidate with its source lineage"
    )
    entry.add_argument("id", help="staged candidate ID")
    entry.add_argument("--runs-root", type=Path, default=Path("research_runs"))
    replay = subcommands.add_parser(
        "rerun-simulation",
        help="replay a frozen simulation with linked lineage",
    )
    replay.add_argument("run", type=Path, help="earlier simulation attempt")
    replay.add_argument("--runs-root", type=Path)
    grid = subcommands.add_parser("sweep", help="run a serial static or dynamic grid")
    grid.add_argument("protocol", type=Path, help="path to sweep.toml")
    grid.add_argument("--runs-root", type=Path, default=Path("research_runs"))
    grid.add_argument("--parent-run", action="append", default=[])
    grid.add_argument("--dry-run", action="store_true", help="validate and preview the frozen grid")
    grid_replay = subcommands.add_parser(
        "rerun-sweep", help="recompute a frozen grid with linked lineage"
    )
    grid_replay.add_argument("run", type=Path)
    grid_replay.add_argument("--runs-root", type=Path)
    for job_parser in (
        script,
        rerun,
        checking,
        simulation,
        replay,
        grid,
        grid_replay,
        controlling,
        control_replay,
        calibrating,
        calibration_replay,
    ):
        job_parser.add_argument(
            "--detach",
            action="store_true",
            help="return admission; supervise in a separate session",
        )
    draft = subcommands.add_parser("create-investigation", help="record a draft investigation")
    draft.add_argument("spec", type=Path, help="investigation terms TOML")
    draft.add_argument("--runs-root", type=Path, default=Path("research_runs"))
    revision = subcommands.add_parser("revise-investigation", help="propose complete revised terms")
    revision.add_argument("investigation", type=Path, help="investigation directory")
    revision.add_argument("spec", type=Path, help="complete replacement terms TOML")
    investigations = subcommands.add_parser(
        "list-investigations", help="list retained investigations"
    )
    investigations.add_argument("--runs-root", type=Path, default=Path("research_runs"))
    investigation = subcommands.add_parser(
        "read-investigation", help="read terms and event history"
    )
    investigation.add_argument("investigation", type=Path)
    report = subcommands.add_parser(
        "report-investigation", help="render retained investigation history as Markdown"
    )
    report.add_argument("investigation", type=Path)
    report.add_argument("--output", type=Path, help="write a file instead of standard output")
    finding = subcommands.add_parser(
        "record-finding", help="append research reasoning and references"
    )
    finding.add_argument("investigation", type=Path)
    finding.add_argument("finding", type=Path, help="authored finding TOML")
    decision = subcommands.add_parser(
        "investigation-decision", help="operator action: start, approve, resume, or release review"
    )
    decision.add_argument("investigation", type=Path)
    decision.add_argument(
        "action", choices=["start", "approve-revision", "resume", "release-review"]
    )
    decision.add_argument("--revision", type=int, help="required for start and approve-revision")
    decision.add_argument("--review", help="open review name for release-review")
    decision.add_argument("--note", default="")
    stopping = subcommands.add_parser(
        "stop-investigation", help="close admission and cancel outstanding jobs"
    )
    stopping.add_argument("investigation", type=Path)
    stopping.add_argument("reason", help="stop reason; see the investigation reference")
    stopping.add_argument("--note", default="")
    for launch_parser in (script, simulation, grid, controlling, calibrating):
        launch_parser.add_argument(
            "--investigation", help="started investigation ID under the runs root"
        )
        launch_parser.add_argument("--base", help="permitted investigation base name")
    for launch_parser in (simulation, controlling):
        launch_parser.add_argument(
            "--vary",
            action="append",
            default=[],
            metavar="TARGET=VALUE",
            help="varied value within the investigation scope; repeat as needed",
        )
    args = parser.parse_args(argv)
    if args.command in {
        "create-investigation",
        "revise-investigation",
        "list-investigations",
        "read-investigation",
        "report-investigation",
        "record-finding",
        "investigation-decision",
        "stop-investigation",
    }:
        from entanglix.investigations import (
            create_investigation,
            list_investigations,
            read_investigation,
            record_finding,
            record_user_decision,
            revise_investigation,
            stop_investigation,
        )
        from entanglix.runner import failure_message

        try:
            if args.command == "create-investigation":
                result = create_investigation(args.spec, runs_root=args.runs_root)
            elif args.command == "revise-investigation":
                result = revise_investigation(args.investigation, args.spec)
            elif args.command == "list-investigations":
                result = list_investigations(args.runs_root)
            elif args.command == "read-investigation":
                result = read_investigation(args.investigation)
            elif args.command == "report-investigation":
                from entanglix.investigation_report import report_investigation

                text = report_investigation(args.investigation)
                if args.output:
                    args.output.write_text(text)
                else:
                    print(text, end="")
                return 0
            elif args.command == "record-finding":
                result = record_finding(args.investigation, args.finding)
            elif args.command == "stop-investigation":
                result = stop_investigation(args.investigation, args.reason, note=args.note)
            else:
                result = record_user_decision(
                    args.investigation,
                    args.action,
                    revision=args.revision,
                    review_name=args.review,
                    note=args.note,
                )
        except (OSError, ValueError, RuntimeError, KeyError, MemoryError) as exc:
            print(f"entanglix: {failure_message(exc)}", file=sys.stderr)
            return 64
        print(json.dumps(result, default=str, indent=2))
        return 0
    if args.command == "validate":
        from entanglix.models import SpecError, load_system
        from entanglix.runner import failure_message

        try:
            system = load_system(args.system)
        except SpecError as exc:
            print(f"entanglix: invalid system: {failure_message(exc)}", file=sys.stderr)
            return 64
        except MemoryError as exc:
            print(f"entanglix: {failure_message(exc)}", file=sys.stderr)
            return 64
        print(f"valid system: {system.id} ({system.basis}, dimensions {system.dimensions})")
        print("operators:")
        for name in sorted(system.operators):
            print(f"  {name}: {_operator_meaning(name, system.provenance)}")
        _print_circuit_diagnostics(system.provenance, "circuit")
        return 0
    if args.command in {"validate-dataset", "split-dataset"}:
        from entanglix.measurements import DatasetError, dataset_report, load_dataset, split_dataset
        from entanglix.runner import failure_message

        try:
            if args.command == "validate-dataset":
                result = dataset_report(load_dataset(args.dataset, raw_file=args.raw_file))
            else:
                result = split_dataset(
                    args.dataset,
                    args.lower,
                    args.upper,
                    holdout=args.holdout,
                    remainder=args.remainder,
                )
        except (OSError, DatasetError, MemoryError) as exc:
            prefix = "invalid dataset: " if args.command == "validate-dataset" else ""
            print(f"entanglix: {prefix}{failure_message(exc)}", file=sys.stderr)
            return 64
        print(json.dumps(result, default=str, indent=2))
        return 0
    if args.command in {"propose-device-update", "apply-device-update"}:
        from entanglix.proposals import apply_device_update, propose_device_update
        from entanglix.runner import failure_message

        try:
            if args.command == "propose-device-update":
                result = propose_device_update(args.run, solution=args.solution)
            else:
                result = apply_device_update(args.proposal, args.system)
        except (OSError, ValueError, KeyError, MemoryError) as exc:
            print(f"entanglix: {failure_message(exc)}", file=sys.stderr)
            return 64
        print(json.dumps(result, default=str, indent=2))
        return 0
    if args.command in {"register-result", "list-catalog", "read-entry"}:
        from entanglix.catalog import list_catalog, read_entry, register_result
        from entanglix.runner import failure_message

        try:
            if args.command == "register-result":
                result = register_result(
                    args.run,
                    args.design,
                    claim=args.claim,
                    point=args.point,
                    solution=args.solution,
                    artifact=args.artifact,
                )
            elif args.command == "list-catalog":
                result = list_catalog(args.runs_root)
            else:
                result = read_entry(args.id, args.runs_root)
        except (OSError, ValueError, KeyError, MemoryError) as exc:
            print(f"entanglix: {failure_message(exc)}", file=sys.stderr)
            return 64
        print(json.dumps(result, default=str, indent=2))
        return 0
    if args.command in {
        "run-script",
        "capabilities",
        "list-runs",
        "read-run",
        "abort-run",
        "reconcile-run",
        "rerun-script",
        "check",
        "calibrate",
        "rerun-calibration",
        "control",
        "rerun-control",
        "simulate",
        "rerun-simulation",
        "sweep",
        "rerun-sweep",
    }:
        from entanglix.runner import (
            abort_run,
            failure_message,
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
            "check",
            "calibrate",
            "rerun-calibration",
            "control",
            "rerun-control",
            "simulate",
            "rerun-simulation",
            "sweep",
            "rerun-sweep",
        }:
            # A supervising command stops its job on termination or hangup, as on Ctrl-C.
            signal.signal(signal.SIGTERM, signal.default_int_handler)
            signal.signal(signal.SIGHUP, signal.default_int_handler)
        try:

            def execute(function, *positional, **arguments):
                if not args.detach:
                    return function(*positional, **arguments)
                import inspect

                from entanglix.launches import launch_job

                arguments = dict(
                    inspect.signature(function).bind(*positional, **arguments).arguments
                )
                return launch_job(
                    "check" if function.__name__ == "check_run" else function.__name__,
                    arguments,
                )

            if args.command == "run-script":
                inputs = {}
                for item in args.input:
                    name, separator, path = item.partition("=")
                    if not separator or not name or not path or name in inputs:
                        raise ValueError("--input requires a unique NAME=PATH")
                    inputs[name] = Path(path)
                result = execute(
                    run_script,
                    args.system,
                    args.script,
                    inputs=inputs,
                    runs_root=args.runs_root,
                    parent_run_ids=tuple(args.parent_run),
                    timeout_seconds=args.timeout,
                    investigation_id=args.investigation,
                    base=args.base,
                )
            elif args.command == "abort-run":
                result = abort_run(args.run)
            elif args.command == "reconcile-run":
                record_reconciliation(args.run, last_observed_state=args.state)
                result = read_run(args.run)
            elif args.command == "rerun-script":
                result = execute(
                    rerun_script, args.run, runs_root=args.runs_root, timeout_seconds=args.timeout
                )
            elif args.command == "capabilities":
                result = list_capabilities()
            elif args.command == "check":
                from entanglix.corrector.verification import check_run

                result = execute(
                    check_run,
                    args.run,
                    tuple(args.names),
                    timeout_seconds=args.timeout,
                    refinement=args.refinement,
                    holdout=args.holdout,
                    solution=args.solution,
                    expect_coverage=args.expect_coverage,
                )
            elif args.command in {"calibrate", "rerun-calibration"}:
                from entanglix.protocols.calibrate import calibrate, rerun_calibration

                result = (
                    execute(
                        calibrate,
                        args.protocol,
                        runs_root=args.runs_root,
                        parent_run_ids=tuple(args.parent_run),
                        dry_run=args.dry_run,
                        investigation_id=args.investigation,
                        base=args.base,
                    )
                    if args.command == "calibrate"
                    else execute(rerun_calibration, args.run, runs_root=args.runs_root)
                )
            elif args.command in {"control", "rerun-control"}:
                from entanglix.protocols.control import control, rerun_control

                result = (
                    execute(
                        control,
                        args.protocol,
                        values=_varied_values(args.vary),
                        runs_root=args.runs_root,
                        parent_run_ids=tuple(args.parent_run),
                        dry_run=args.dry_run,
                        investigation_id=args.investigation,
                        base=args.base,
                    )
                    if args.command == "control"
                    else execute(rerun_control, args.run, runs_root=args.runs_root)
                )
            elif args.command in {"simulate", "rerun-simulation"}:
                from entanglix.protocols.simulate import rerun_simulation, simulate

                result = (
                    execute(
                        simulate,
                        args.protocol,
                        values=_varied_values(args.vary),
                        runs_root=args.runs_root,
                        parent_run_ids=tuple(args.parent_run),
                        dry_run=args.dry_run,
                        investigation_id=args.investigation,
                        base=args.base,
                    )
                    if args.command == "simulate"
                    else execute(rerun_simulation, args.run, runs_root=args.runs_root)
                )
            elif args.command in {"sweep", "rerun-sweep"}:
                from entanglix.protocols.sweep import rerun_sweep, sweep

                result = (
                    execute(
                        sweep,
                        args.protocol,
                        runs_root=args.runs_root,
                        parent_run_ids=tuple(args.parent_run),
                        dry_run=args.dry_run,
                        investigation_id=args.investigation,
                        base=args.base,
                    )
                    if args.command == "sweep"
                    else execute(rerun_sweep, args.run, runs_root=args.runs_root)
                )
            elif args.command == "list-runs":
                result = list_runs(args.runs_root)
            else:
                result = read_run(args.run, wait_seconds=args.wait)
        except (OSError, ValueError, RuntimeError, KeyError, MemoryError) as exc:
            print(f"entanglix: {failure_message(exc)}", file=sys.stderr)
            return 64
        print(json.dumps(result, default=str, indent=2))
        return (
            result.get("exit_code", 0)
            if args.command
            in {
                "run-script",
                "rerun-script",
                "check",
                "calibrate",
                "rerun-calibration",
                "control",
                "rerun-control",
                "simulate",
                "rerun-simulation",
                "sweep",
                "rerun-sweep",
            }
            else 0
        )
    parser.print_help()
    return 0
