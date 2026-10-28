"""Verification worker with the same start barrier and ownership as computation jobs."""

import json
import os
import signal
import sys
from pathlib import Path

from entanglix.runner import capture_environment


def main():
    # Interruptions return through the worker's normal shutdown path.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    signal.signal(signal.SIGHUP, signal.default_int_handler)
    try:
        if sys.argv[1] == "check":
            request = json.load(sys.stdin)
            print(
                json.dumps(
                    capture_environment(
                        Path(request["repository"]) / "uv.lock", extras=tuple(request["extras"])
                    )
                )
            )
        else:
            os.fstat(int(sys.argv[3]))
            if sys.stdin.buffer.read(1) != b"1":
                return 64
            from entanglix.corrector.verification import execute_verification

            execute_verification(Path(os.environ["ENTANGLIX_RUN_DIR"]))
    except KeyboardInterrupt:
        print("verification aborted", file=sys.stderr)
        return 130
    except Exception as exc:
        from entanglix.runner import failure_message

        print(f"verification failed: {failure_message(exc)}", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
