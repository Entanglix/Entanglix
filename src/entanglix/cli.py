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
    parser.parse_args(argv)
    parser.print_help()
    return 0
