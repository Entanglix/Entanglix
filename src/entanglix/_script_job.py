"""Locked-environment preflight and the start barrier for an authored CPU job."""

import json
import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    if sys.argv[1] == "check":
        from entanglix.runner import _script_gate

        try:
            print(json.dumps(_script_gate(**json.load(sys.stdin))))
        except Exception as exc:
            from entanglix.runner import failure_message

            print(f"launch rejected: {failure_message(exc)}", file=sys.stderr)
            return 64
        return 0
    script = Path(sys.argv[2])
    os.fstat(int(sys.argv[3]))  # The job must inherit attempt ownership through uv.
    if sys.stdin.buffer.read(1) != b"1":
        return 64
    sys.argv = [str(script)]
    sys.path[0] = str(script.parent)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except MemoryError as exc:
        from entanglix.runner import failure_message

        print(f"job failed: {failure_message(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
