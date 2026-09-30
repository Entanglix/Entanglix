"""Run records and frozen provenance; process supervision is supplied separately."""

import fcntl
import hashlib
import json
import math
import os
import platform
import re
import secrets
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import tomli_w


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def capture_environment(
    lockfile: Path, *, extras: tuple[str, ...] = (), gpu: dict[str, str] | None = None
) -> dict:
    """Fingerprint an exactly synced environment; GPU observations come from the gate."""
    gpu = gpu or {}
    unknown = gpu.keys() - {"driver", "cuda", "cuquantum"}
    if unknown:
        raise ValueError(f"unknown GPU observation keys: {sorted(unknown)}")
    record = {
        "lock_sha256": _digest(lockfile.read_bytes()),
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "extras": sorted(set(extras)),
        "gpu": {key: gpu.get(key, "unobserved") for key in ("driver", "cuda", "cuquantum")},
    }
    record["env_hash"] = _digest(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
    return record


@dataclass(frozen=True)
class CodeSnapshot:
    revision: str
    patch: bytes
    untracked: dict[str, bytes]

    def provenance(self) -> dict:
        return {
            "revision": self.revision,
            "dirty": bool(self.patch or self.untracked),
            "patch_sha256": _digest(self.patch),
            "untracked_sha256": {name: _digest(data) for name, data in self.untracked.items()},
        }


def capture_code(repository: Path, *, max_snapshot_bytes: int = 16 * 1024 * 1024) -> CodeSnapshot:
    """Capture staged and unstaged changes against HEAD, including binary changes."""

    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(repository), *args], check=True, capture_output=True
        ).stdout

    root = Path(os.fsdecode(git("rev-parse", "--show-toplevel")).strip())
    repository = root
    revision = git("rev-parse", "HEAD").decode().strip()
    if max_snapshot_bytes < 0:
        raise ValueError("max_snapshot_bytes must be nonnegative")
    patch = git(
        "diff",
        "--binary",
        "--no-color",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
        ".",
    )
    names = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    files = {}
    for raw in filter(None, names):
        name = os.fsdecode(raw)
        path = root / name
        if path.is_symlink():
            raise ValueError(f"cannot freeze an untracked symlink: {name}")
        files[name] = path
    size = len(patch) + sum(path.stat().st_size for path in files.values())
    if size > max_snapshot_bytes:
        raise ValueError(
            f"code snapshot needs {size} bytes, exceeding max_snapshot_bytes={max_snapshot_bytes}; "
            "ignore unrelated files, supply data as explicit inputs, or raise the snapshot limit"
        )
    untracked = {name: path.read_bytes() for name, path in files.items()}
    return CodeSnapshot(revision, patch, untracked)


def publish_toml(path: Path, record: dict, *, replace: bool = False) -> None:
    """Publish a complete record atomically; runner records refuse replacement."""
    content = tomli_w.dumps(record).encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            if not replace:
                os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_file(name: str) -> Path:
    name.encode("utf-8")
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"frozen file must have a relative path without '..': {name!r}")
    return path


class RunCreationError(RuntimeError):
    """An attempt was created but could not finish preparing; retain its path."""

    def __init__(self, run: Path, cause: Exception):
        self.run = run
        super().__init__(f"failed to prepare run {run}: {cause}")


@dataclass
class RunHandle:
    """Own an attempt until this descriptor and all inherited copies are closed."""

    path: Path
    _descriptor: int | None

    def fileno(self) -> int:
        if self._descriptor is None:
            raise ValueError("attempt owner handle is closed")
        return self._descriptor

    def close(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def __enter__(self):
        self.fileno()
        return self

    def __exit__(self, *_):
        self.close()


def create_run(
    *,
    system_id: str,
    protocol: str,
    inputs: dict[str, Path],
    environment: dict,
    code: CodeSnapshot,
    engine: str,
    device: Literal["cpu", "cuda"],
    needs: dict,
    runs_root: Path = Path("research_runs"),
    parent_run_ids: tuple[str, ...] = (),
) -> RunHandle:
    """Create an admitted attempt with its manifest first and its inputs frozen."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", system_id):
        raise ValueError(
            "system_id must be a single path component starting with a letter or digit"
        )
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if "system.toml" not in inputs:
        raise ValueError("every research run requires a system.toml input, including scripts")
    frozen = {}
    for name, source in inputs.items():
        relative = _relative_file(name)
        if relative.parts[0] == "code":
            raise ValueError("spec/code is reserved for code provenance")
        frozen[name] = source.read_bytes()
    files = {**frozen, "code/changes.patch": code.patch}
    files.update({f"code/untracked/{name}": data for name, data in code.untracked.items()})
    for name in code.untracked:
        _relative_file(name)
    paths = [_relative_file(name) for name in files]
    unique = set(paths)
    if len(paths) != len(unique) or any(
        parent in unique for path in paths for parent in path.parents
    ):
        raise ValueError("frozen file names have duplicate paths or file/directory conflicts")
    created = datetime.now(UTC)
    run_id = f"r{created.strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(4)}"
    record = {
        "record_version": 1,
        "run_id": run_id,
        "system_id": system_id,
        "protocol": protocol,
        "created_at": created,
        "parent_run_ids": list(parent_run_ids),
        "engine": engine,
        "device": device,
        "executor": "local",
        "needs": needs,
        "spec_sha256": {name: _digest(data) for name, data in frozen.items()},
        "code": code.provenance(),
        "env_hash": environment["env_hash"],
        "environment": environment,
    }
    # Serialization and input errors must happen before the attempt directory exists.
    tomli_w.dumps(record).encode("utf-8")
    parent = runs_root / system_id
    parent.mkdir(parents=True, exist_ok=True)
    run = parent / run_id
    run.mkdir()
    owner = None
    try:
        owner = RunHandle(run, os.open(run, os.O_RDONLY | os.O_DIRECTORY))
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX)
        publish_toml(run / "run.toml", record)
        for name in ("runner", "spec", "artifacts"):
            (run / name).mkdir()
        for name, data in files.items():
            target = run / "spec" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o444)
    except Exception as exc:
        error = RunCreationError(run, exc)
        if (run / "run.toml").exists():
            try:
                record_finish(owner, status="failed", exit_code=64, wall_seconds=0, gpu_seconds=0)
            except Exception as finish_error:
                error.add_note(f"could not record creation failure: {finish_error}")
        else:
            try:
                run.rmdir()
            except OSError as cleanup_error:
                error.add_note(f"could not remove unadmitted attempt directory: {cleanup_error}")
        if owner is not None:
            owner.close()
        raise error from exc
    return owner


def record_start(
    owner: RunHandle, *, process_group: int, gpu_devices: tuple[str, ...] = ()
) -> None:
    if process_group <= 0:
        raise ValueError("process_group must be positive")
    owner.fileno()
    run = owner.path
    if (run / "FINISHED").exists():
        raise ValueError("a finished attempt cannot start")
    publish_toml(
        run / "runner" / "started.toml",
        {
            "record_version": 1,
            "started_at": datetime.now(UTC),
            "process_group": process_group,
            "gpu_devices": list(gpu_devices),
        },
    )


def record_finish(
    owner: RunHandle,
    *,
    status: Literal["completed", "failed", "aborted"],
    exit_code: int,
    wall_seconds: float,
    gpu_seconds: float,
) -> None:
    if status not in {"completed", "failed", "aborted"}:
        raise ValueError("unknown terminal status")
    if type(exit_code) is not int or not 0 <= exit_code <= 255:
        raise ValueError("exit_code must be an integer from 0 to 255; signals use 128 + signal")
    if (status == "completed" and exit_code != 0) or (status == "aborted" and exit_code != 130):
        raise ValueError("completed requires exit 0; aborted requires exit 130")
    if status == "failed" and exit_code == 0:
        raise ValueError("failed requires a nonzero exit code")
    if any(not math.isfinite(value) or value < 0 for value in (wall_seconds, gpu_seconds)):
        raise ValueError("usage must be finite and nonnegative")
    owner.fileno()
    run = owner.path
    if not (run / "runner" / "started.toml").exists():
        if status != "failed" or exit_code != 64 or wall_seconds != 0 or gpu_seconds != 0:
            raise ValueError("without a start record only failed/64 with zero usage is valid")
    publish_toml(
        run / "FINISHED",
        {
            "record_version": 1,
            "status": status,
            "exit_code": exit_code,
            "ended_at": datetime.now(UTC),
            "usage": {
                "wall_seconds": wall_seconds,
                "gpu_seconds": gpu_seconds,
                "uncertain": False,
            },
        },
    )


def record_reconciliation(run: Path, *, last_observed_state: str) -> None:
    descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("attempt is still owned by its runner or inheriting job") from exc
        if not (run / "run.toml").exists():
            raise ValueError("attempt has no published run.toml")
        if (run / "FINISHED").exists():
            raise ValueError("a finished attempt does not need reconciliation")
        start_path = run / "runner" / "started.toml"
        if start_path.exists():
            with start_path.open("rb") as stream:
                start = tomllib.load(stream)
            try:
                os.killpg(start["process_group"], 0)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                raise ValueError("cannot confirm recorded process group is gone") from exc
            else:
                raise ValueError("recorded process group is still alive")
            start = {key: start[key] for key in ("started_at", "process_group", "gpu_devices")}
        else:
            start = {}
        (run / "runner").mkdir(exist_ok=True)
        publish_toml(
            run / "runner" / "reconciled.toml",
            {
                "record_version": 1,
                "observed_at": datetime.now(UTC),
                "last_observed_state": last_observed_state,
                **start,
                "usage": {"uncertain": True},
            },
        )
    finally:
        os.close(descriptor)
