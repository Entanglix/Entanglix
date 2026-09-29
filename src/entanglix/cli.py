"""The foundation CLI; research operations arrive with their capabilities."""

import argparse
import sys

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
    parser.print_help()
    return 0
