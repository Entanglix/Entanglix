"""Live setup-project CUDA/MCP checks; requires logged-in Claude Code and Codex.

Run with the checkout's Python and --installation pointing at a GPU installation.
Workspace copies and raw runtime output stay outside the repository. No inherited
CUDA-hiding prefix or replacement runtime settings supply the settings under test.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from entanglix.project_setup import setup_project

ROOT = Path(__file__).resolve().parents[2]
VISIBILITY = """import ctypes, json, os
from pathlib import Path
driver = ctypes.CDLL("libcuda.so.1")
status = driver.cuInit(0)
count = ctypes.c_int()
if status == 0:
    driver.cuDeviceGetCount(ctypes.byref(count))
result = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "cuInit": status, "device_count": count.value}
Path("session-cuda.json").write_text(json.dumps(result) + "\\n")
print(json.dumps(result))
"""
PROTOCOL = """format_version = 1
system = "system.toml"
kind = "lindblad"
timeout_seconds = 60
[needs]
device = "cuda"
[evolution]
initial_state = [1]
times_ns = [0.0, 0.1]
[evolution.observables.excited]
kind = "population"
subsystem = "two_level_reference"
level = 1
"""


def load(path):
    return tomllib.loads(path.read_text()) if path.exists() else {}


def run_session(runtime, workspace, installation, device):
    setup_project(workspace, runtimes=(runtime,), device=device, root=installation)
    for name in ("system.toml", "operators.npz"):
        shutil.copyfile(ROOT / "examples/two-level" / name, workspace / name)
    (workspace / "simulate.toml").write_text(PROTOCOL)
    (workspace / "visibility.py").write_text(VISIBILITY)
    prompt = (
        "This is an authorized setup conformance test. Run exactly one session-shell "
        f"command: `{installation / '.venv/bin/python'} visibility.py`. Do not override "
        "environment variables or edit configuration. Then use the Entanglix MCP simulate "
        "tool on simulate.toml once; wait with read_run on its returned path until terminal. "
        "Report the exact visibility JSON, run ID, status, and engine. No checks, other "
        "computations, investigation decisions, or subagents are requested."
    )
    if runtime == "codex":
        command = [
            "codex", "exec", "--json", "--skip-git-repo-check",
            "-s", "danger-full-access", "-c",
            f'projects={{"{workspace}"={{trust_level="trusted"}}}}', prompt,
        ]  # fmt: skip
    else:
        command = [
            "claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
            "--allowedTools", "Bash", "mcp__entanglix__simulate", "mcp__entanglix__read_run",
        ]  # fmt: skip
    (workspace / "invocation.json").write_text(json.dumps(command, indent=2) + "\n")
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": device}
    started = time.monotonic()
    with (
        (workspace / "runtime.jsonl").open("w") as out,
        (workspace / "runtime.stderr").open("w") as err,
    ):
        done = subprocess.run(
            command, cwd=workspace, env=environment, stdout=out, stderr=err, timeout=300
        )
    visibility = workspace / "session-cuda.json"
    runs = []
    for manifest in (workspace / "research_runs").glob("*/*/run.toml"):
        run = manifest.parent
        runs.append(
            {
                "run_id": load(manifest)["run_id"],
                "path": str(run),
                "status": load(run / "FINISHED").get("status", "nonterminal"),
                "engine": load(run / "result.toml").get("engine"),
                "gpu_devices": load(run / "runner/started.toml").get("gpu_devices", []),
            }
        )
    observed = json.loads(visibility.read_text()) if visibility.exists() else {}
    configs = (
        [".claude/settings.json", ".mcp.json"] if runtime == "claude" else [".codex/config.toml"]
    )
    return {
        "version": subprocess.check_output([runtime, "--version"], text=True).strip(),
        "workspace": str(workspace),
        "command": command,
        "parent_cuda_visible_devices": device,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "exit_code": done.returncode,
        "configurations": {
            name: (
                json.loads((workspace / name).read_text())
                if name.endswith(".json")
                else load(workspace / name)
            )
            for name in configs
        },
        "session_cuda": observed,
        "runner_jobs": runs,
        "passed": done.returncode == 0
        and observed.get("cuda_visible_devices") == ""
        and observed.get("device_count") == 0
        and len(runs) == 1
        and runs[0]["status"] == "completed"
        and bool(runs[0]["gpu_devices"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    if workspace.exists() or workspace.is_relative_to(ROOT):
        parser.error("choose a fresh workspace outside the repository")
    workspace.mkdir(parents=True)
    sources = [Path(__file__).resolve(), ROOT / "src/entanglix/project_setup.py"]
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "installation": str(args.installation),
        "sessions": {},
        "limits": [
            "One non-interactive session per runtime; no interactive first-start prompt claim.",
            "Codex uses an explicit trust override and no device sandbox; Claude tool approvals are supplied for this test.",
            "Models and effort use the user's runtime settings; no command-restriction test is included.",
            "Setup templates come from the worktree; the MCP server and job environment use the recorded installation.",
        ],
    }
    for runtime in ("claude", "codex"):
        result["sessions"][runtime] = run_session(
            runtime, workspace / runtime, args.installation, args.device
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(runtime, "passed" if result["sessions"][runtime]["passed"] else "failed", flush=True)
        if not result["sessions"][runtime]["passed"]:
            raise SystemExit("inspect retained evidence before another runtime session")


if __name__ == "__main__":
    main()
