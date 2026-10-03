"""Simulation gate and plan worker using the runner's inherited start barrier."""

import json
import os
import sys
from pathlib import Path


def main():
    try:
        if sys.argv[1] == "check":
            from entanglix.protocols.simulate import compile_simulation

            request = json.load(sys.stdin)
            for name in ("protocol", "directory", "repository"):
                request[name] = Path(request[name])
            print(json.dumps(compile_simulation(**request)))
        else:
            from entanglix.protocols._plan_job import execute

            os.fstat(int(sys.argv[3]))
            if sys.stdin.buffer.read(1) != b"1":
                return 64
            if "ENTANGLIX_GPU_LOCK_FD" in os.environ:
                from entanglix.runner import _load_record, check_gpu_environment

                os.fstat(int(os.environ["ENTANGLIX_GPU_LOCK_FD"]))
                record = _load_record(Path(os.environ["ENTANGLIX_RUN_DIR"]) / "run.toml")
                observed = check_gpu_environment(os.environ["CUDA_VISIBLE_DEVICES"])
                if observed["gpu"] != record["environment"]["gpu"]:
                    raise ValueError("GPU runtime changed after launch validation")
            execute(Path(sys.argv[2]))
    except Exception as exc:
        from entanglix.runner import failure_message

        print(f"simulation rejected: {failure_message(exc)}", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
