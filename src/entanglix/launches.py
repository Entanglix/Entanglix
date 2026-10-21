"""Detached launches and recovery from records, without resubmission."""

import fcntl
import inspect
import math
import os
import secrets
import subprocess
import sys
import time
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

from entanglix.runner import RunHandle, _load_record, publish_toml

# Only the supervisor sets this. The manifest binds the call before any inputs copy.
current_launch: ContextVar[Path | None] = ContextVar("current_launch", default=None)


def _operation(name):
    from entanglix.corrector.verification import check_run
    from entanglix.protocols.calibrate import calibrate
    from entanglix.protocols.control import control, rerun_control
    from entanglix.protocols.simulate import rerun_simulation, simulate
    from entanglix.protocols.sweep import rerun_sweep, sweep
    from entanglix.runner import rerun_script, run_script

    operations = {
        "calibrate": calibrate,
        "control": control,
        "rerun_control": rerun_control,
        "simulate": simulate,
        "sweep": sweep,
        "run_script": run_script,
        "rerun_script": rerun_script,
        "rerun_simulation": rerun_simulation,
        "rerun_sweep": rerun_sweep,
        "check": check_run,
    }
    if name not in operations:
        raise ValueError(f"not a launch operation: {name}")
    return operations[name]


def _arguments(arguments):
    result = dict(arguments)
    for name in (
        "system",
        "script",
        "protocol",
        "run",
        "source",
        "repository",
        "runs_root",
        "refinement",
    ):
        if result.get(name) is not None:
            result[name] = Path(result[name]).resolve()
    if result.get("inputs") is not None:
        result["inputs"] = {name: Path(path).resolve() for name, path in result["inputs"].items()}
    return result


def reserve_launch(runs_root: Path, operation: str, *, investigation_id=None) -> RunHandle:
    """Own a call before auditing it; a lost caller leaves a recoverable refusal."""
    launch_id = f"l{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(4)}"
    parent = runs_root.resolve() / "_launches"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / launch_id
    path.mkdir()
    owner = RunHandle(path, os.open(path, os.O_RDONLY | os.O_DIRECTORY))
    fcntl.flock(owner.fileno(), fcntl.LOCK_EX)
    try:
        publish_toml(
            path / "call.toml",
            {
                "record_version": 1,
                "launch_id": launch_id,
                "operation": operation,
                **({"investigation_id": investigation_id} if investigation_id else {}),
            },
        )
    except BaseException:
        owner.close()
        path.rmdir()
        raise
    return owner


def prepare_launch(
    operation: str,
    arguments: dict,
    *,
    runs_root: Path | None = None,
    owner: RunHandle | None = None,
) -> RunHandle:
    """Record a request while owning it, before a supervisor can start."""
    arguments = _arguments(arguments)
    function = _operation(operation)
    inspect.signature(function).bind(**arguments)
    if arguments.get("dry_run") or (operation == "check" and not arguments.get("checks")):
        raise ValueError(
            "detached launches require a job; use synchronous previews and check listings"
        )
    source = arguments.get("run", arguments.get("source"))
    expected = arguments.get("runs_root") or (
        source.parent.parent if source is not None else Path("research_runs").resolve()
    )
    runs_root = (runs_root or expected).resolve()
    if expected != runs_root:
        raise ValueError("launch records and attempts must use the same runs root")
    investigation_id = arguments.get("investigation_id")
    if source is not None:
        investigation_id = _load_record(source / "run.toml").get("investigation_id")
    # Record absolute inputs: a supervisor's working directory is not the client's.
    serialised = {
        key: (
            {name: str(path) for name, path in value.items()}
            if key == "inputs"
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in arguments.items()
        if value is not None
    }
    borrowed = owner is not None
    owner = owner or reserve_launch(runs_root, operation, investigation_id=investigation_id)
    path = owner.path
    if path.parent.parent != runs_root:
        raise ValueError("launch owner and attempts must use the same runs root")
    try:
        publish_toml(
            path / "request.toml",
            {
                "record_version": 1,
                "launch_id": path.name,
                "created_at": datetime.now(UTC),
                "operation": operation,
                "arguments": serialised,
                **({"investigation_id": investigation_id} if investigation_id else {}),
            },
        )
    except BaseException:
        if not borrowed:
            owner.close()
        raise
    return owner


def start_supervisor(owner: RunHandle, *, environment: dict[str, str] | None = None):
    """Hand ownership to a new session with streams independent of the caller."""
    with (owner.path / "supervisor.log").open("ab") as log:
        return subprocess.Popen(
            [sys.executable, "-m", "entanglix._supervisor", str(owner.path), str(owner.fileno())],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(owner.fileno(),),
            env=environment,
        )


def read_launch(path: Path) -> dict:
    """Check ownership and the prepublished link without scanning other attempts."""
    path = path.resolve()
    request = _load_record(
        path / ("request.toml" if (path / "request.toml").exists() else "call.toml")
    )
    root = path.parent.parent
    with RunHandle(path, os.open(path, os.O_RDONLY | os.O_DIRECTORY)) as owner:
        owned = False
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            owned = True
        # Shared readers cannot be mistaken for the exclusive launch owner.
        link = path / "attempt.toml"
        result = {
            "launch_id": request["launch_id"],
            "launch_path": str(path),
            "operation": request["operation"],
            "investigation_id": request.get("investigation_id"),
        }
        if link.exists():
            binding = _load_record(link)
            run = root / binding["run_path"]
            if (run / "run.toml").is_file():
                return {
                    **result,
                    "status": "admitted",
                    "path": str(run),
                    "run_id": run.name,
                    **(
                        {"investigation_revision": binding["investigation_revision"]}
                        if "investigation_revision" in binding
                        else {}
                    ),
                }
        if owned:
            return {**result, "status": "pending"}
        refusal = path / "refused.toml"
        diagnostic = (
            _load_record(refusal)["reason"]
            if refusal.exists()
            else "launch ownership ended with no retained admitted manifest"
        )
        return {**result, "status": "refused", "reason": diagnostic}


def unresolved_launches(runs_root: Path, *, investigation_id: str | None = None) -> list[dict]:
    result = []
    for path in sorted((runs_root / "_launches").glob("*")):
        request = path / "request.toml"
        if not request.exists():
            request = path / "call.toml"
        if not request.exists():
            continue
        if investigation_id is not None:
            if _load_record(request).get("investigation_id") != investigation_id:
                continue
        launch = read_launch(request.parent)
        if launch["status"] == "pending":
            result.append(launch)
    return result


def launch_job(
    operation: str,
    arguments: dict,
    *,
    runs_root: Path | None = None,
    environment: dict[str, str] | None = None,
) -> dict:
    """Return admission (or refusal), leaving execution under a detached supervisor."""
    with prepare_launch(operation, arguments, runs_root=runs_root) as owner:
        try:
            process = start_supervisor(owner, environment=environment)
        except OSError as exc:
            publish_toml(owner.path / "refused.toml", {"reason": str(exc)})
            raise
        path = owner.path
    while True:
        result = read_launch(path)
        if result["status"] != "pending":
            if process.poll() is not None:
                process.wait()
            if result["status"] == "refused":
                raise ValueError(f"launch {result['launch_id']} refused: {result['reason']}")
            return result
        time.sleep(0.01)


def validate_wait(wait_seconds: float) -> None:
    if not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise ValueError("wait_seconds must be finite and non-negative")
