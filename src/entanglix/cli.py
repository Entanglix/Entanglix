"""The foundation CLI; research operations arrive with their capabilities."""

import argparse
import sys

from entanglix import __version__


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


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
        print("operators: " + ", ".join(sorted(system.operators)))
        return 0
    parser.print_help()
    return 0
