"""Simulation gate and plan worker using the runner's inherited start barrier."""

import json
import os
import sys
from pathlib import Path


def main(operation="simulate"):
    try:
        if sys.argv[1] == "check":
            if operation == "control":
                from entanglix.protocols.control import compile_control as compile_protocol
            elif operation == "sweep":
                from entanglix.protocols.sweep import compile_sweep as compile_protocol
            else:
                from entanglix.protocols.simulate import compile_simulation as compile_protocol

            request = json.load(sys.stdin)
            for name in ("protocol", "directory", "repository"):
                request[name] = Path(request[name])
            if operation == "control":
                request.pop("retain_base")
            print(json.dumps(compile_protocol(**request)))
        else:
            if operation == "control":
                from entanglix.protocols._control_job import execute
            elif operation == "sweep":
                from entanglix.protocols._sweep_job import execute
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

        print(f"{operation} rejected: {failure_message(exc)}", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
