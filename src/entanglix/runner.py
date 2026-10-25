"""Owned run records, CPU authored jobs, and inspection of retained attempts."""

import fcntl
import hashlib
import json
import math
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import time
import tomllib
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions, version
from pathlib import Path
from typing import Literal

import tomli_w

TERMINATION_GRACE_SECONDS = 2.0


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def capture_environment(
    lockfile: Path,
    *,
    extras: tuple[str, ...] = (),
    gpu: dict[str, str] | None = None,
    solver_runtime: dict | None = None,
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
    if solver_runtime is not None:
        record["solver_runtime"] = solver_runtime
    record["env_hash"] = _digest(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
    return record


def failure_message(exc: Exception) -> str:
    """Identify observed allocation failures without guessing at signal causes."""
    if "out of memory:" in str(exc).lower():
        return str(exc)
    detail = f"{type(exc).__name__}: {exc}"
    if (
        isinstance(exc, MemoryError)
        or "out of memory" in detail.lower()
        or "ALLOC_FAILED" in detail
    ):
        return f"out of memory: {detail}"
    return str(exc)


def _signal_failure(code: int) -> str:
    """Describe raw signals or uncertain shell-encoded exits consistently."""
    encoded_signal = next((sig for sig in signal.Signals if code == 128 + sig), None)
    if code < 0 or encoded_signal is not None:
        observed = (
            f"terminated by {signal.Signals(-code).name}"
            if code < 0
            else f"exited {code} ({encoded_signal.name} encoding)"
        )
        detail = f"{observed}; cause unknown"
        if code in {-signal.SIGKILL, 128 + signal.SIGKILL}:
            detail += "; exhausted memory is a possible cause"
        return detail
    return ""


def process_failure(process, context: str) -> str:
    detail = process.stderr.strip()
    observed = _signal_failure(process.returncode)
    if observed:
        detail += f"\n{context}: {observed}"
    return detail.strip() or context


def check_gpu_environment(device_uuid: str) -> dict:
    """Check a selected device in the caller's exact job environment before dispatch."""
    try:
        import ctypes

        import cupy as cp
        from cuquantum.densitymat import WorkStream

        query = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_uuid}",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        observed_uuid, driver = (value.strip() for value in query.stdout.strip().split(","))
        if observed_uuid != device_uuid:
            raise ValueError("NVIDIA-SMI returned a different device UUID")
        for device_id in range(cp.cuda.runtime.getDeviceCount()):
            properties = cp.cuda.runtime.getDeviceProperties(device_id)
            if f"GPU-{uuid.UUID(bytes=properties['uuid'])}" == device_uuid:
                break
        else:
            raise ValueError(f"selected device {device_uuid} is not visible to CUDA")
        with cp.cuda.Device(device_id):
            cuda = cp.cuda.runtime.runtimeGetVersion()
            # Creating a handle checks that the native library initializes on this device.
            work = WorkStream(device_id=device_id)
            # The 26.9 binding misreads GetVersion's size_t result as a status; use the
            # already loaded library's C API instead of substituting a package version.
            native = ctypes.CDLL("libcudensitymat.so.0")
            native.cudensitymatGetVersion.restype = ctypes.c_size_t
            native.cudensitymatGetVersion.argtypes = []
            library_version = native.cudensitymatGetVersion()
            del work
        return {
            "device_uuid": device_uuid,
            "device_id": device_id,
            "gpu": {
                "driver": driver,
                "cuda": f"{cuda // 1000}.{cuda % 1000 // 10}.{cuda % 10}",
                "cuquantum": (
                    f"{library_version // 10000}.{library_version % 10000 // 100}."
                    f"{library_version % 100}"
                ),
            },
        }
    except Exception as exc:
        raise ValueError(
            f"GPU environment rejected for {device_uuid}: {exc}; "
            "check device visibility, driver access, the cuquantum extra, and free memory"
        ) from exc


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


def _scope_defaults(inputs, protocol):
    # Choose by operation: an authored system may itself be named simulate.toml.
    name = {"simulate": "simulate.toml", "sweep": "simulate.toml", "control": "control.toml"}
    path = inputs.get(f"base/{name[protocol]}") if protocol in name else None
    return _load_record(path) if path is not None else None


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
    source_entry_ids: tuple[str, ...] = (),
    launch: dict | None = None,
    investigation_id: str | None = None,
    base: str | None = None,
    base_inputs: dict[str, str] | None = None,
    values: list[dict[str, float]] | None = None,
    gpu: "GpuLease | None" = None,
) -> RunHandle:
    """Admit standalone or investigation work and retain ownership of the attempt."""
    arguments = dict(
        system_id=system_id,
        protocol=protocol,
        inputs=inputs,
        environment=environment,
        code=code,
        engine=engine,
        device=device,
        needs=needs,
        runs_root=runs_root,
        parent_run_ids=parent_run_ids,
        source_entry_ids=source_entry_ids,
        launch=launch,
        gpu=gpu,
    )
    if protocol == "verification" and parent_run_ids:
        source_path = (
            runs_root.resolve() / system_id / (launch or {}).get("source_path", parent_run_ids[0])
        ).resolve()
        source = _load_record(source_path / "run.toml")
        investigation_id = source.get("investigation_id")
        if investigation_id and source_path.parent.parent != runs_root.resolve():
            raise ValueError("investigation checks must retain their source runs root")
        base = source.get("investigation_base")
    if investigation_id is None:
        if base is not None or base_inputs is not None or values is not None:
            raise ValueError("base and varied values require an investigation")
        return _create_run(**arguments)
    from entanglix.investigations import admit, investigation_path

    path = investigation_path(runs_root, investigation_id)
    if protocol not in {"simulate", "sweep", "script", "verification", "control", "calibrate"}:
        raise ValueError("unsupported investigation operation")
    hashes = (
        base_inputs
        if base_inputs is not None
        else {name: _digest(source.read_bytes()) for name, source in inputs.items()}
    )
    with admit(
        path,
        operation={"script": "run_script", "verification": "check"}.get(protocol, protocol),
        base_name=base,
        hashes=hashes,
        points=values or [{}],
        defaults=_scope_defaults(inputs, protocol),
        timeout_seconds=(launch or {}).get("timeout_seconds"),
    ) as binding:
        arguments["source_entry_ids"] = (*source_entry_ids, *binding.pop("source_entry_ids", []))
        return _create_run(
            **arguments, investigation={**binding, "investigation_values": values or [{}]}
        )


def _create_run(
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
    source_entry_ids: tuple[str, ...] = (),
    launch: dict | None = None,
    investigation: dict | None = None,
    gpu: "GpuLease | None" = None,
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
    if investigation is not None:
        record.update(investigation)
    if source_entry_ids:
        record["source_entry_ids"] = list(dict.fromkeys(source_entry_ids))
    if launch is not None:
        record["launch"] = launch
    from entanglix.launches import current_launch

    launch_path = current_launch.get()
    if launch_path is not None:
        record["launch_id"] = launch_path.name
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
        if launch_path is not None:
            publish_toml(
                launch_path / "attempt.toml",
                {
                    "run_path": str(run.relative_to(runs_root)),
                    **(
                        {"investigation_revision": record["investigation_revision"]}
                        if "investigation_revision" in record
                        else {}
                    ),
                },
            )
        publish_toml(run / "run.toml", record)
        if gpu is not None:
            gpu.bind_run(owner)
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
                record_finish(
                    owner,
                    status="failed",
                    exit_code=64,
                    wall_seconds=0,
                    gpu_seconds=time.monotonic() - gpu.allocated_at if gpu else 0,
                )
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
        if (status, exit_code) not in {("failed", 64), ("aborted", 130)} or wall_seconds != 0:
            raise ValueError(
                "without a start record only failed/64 or aborted/130 with zero wall time is valid"
            )
    ended = datetime.now(UTC)
    usage = {"wall_seconds": wall_seconds, "gpu_seconds": gpu_seconds, "uncertain": False}
    manifest = _load_record(run / "run.toml")
    if "investigation_id" in manifest:
        deadline = manifest["investigation_limits"].get("deadline")
        commitment = manifest["investigation_commitment"]
        usage["deadline_overrun_seconds"] = (
            max(0.0, (ended - deadline).total_seconds()) if deadline else 0.0
        )
        usage["job_seconds_overrun"] = (
            max(0.0, wall_seconds - commitment["job_seconds"])
            if not commitment.get("unbounded")
            else 0.0
        )
    publish_toml(
        run / "FINISHED",
        {
            "record_version": 1,
            "status": status,
            "exit_code": exit_code,
            "ended_at": ended,
            "usage": usage,
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


def _system_inputs(source: Path, constructed, *, reserved: set[str]) -> tuple[dict, dict]:
    """Collect the files and loader digests of an already validated CPU model."""
    paths = {"system.toml": source}
    # Reuse the loader's digests so freezing checks the bytes actually validated.
    hashes = {"system.toml": constructed.provenance["system_sha256"]}
    document = tomllib.loads(source.read_text())
    documents = [(document, constructed.provenance)]
    for subsystem in document["model"].get("subsystems", []):
        if "source" in subsystem:
            name = subsystem["source"]
            if name in {"system.toml", *reserved}:
                raise ValueError(f"subsystem source conflicts with reserved input: {name}")
            if name in paths:
                continue
            paths[name] = source.parent / name
            provenance = constructed.provenance["subsystems"][subsystem["id"]]
            hashes[name] = provenance["system_sha256"]
            documents.append((tomllib.loads(paths[name].read_text()), provenance))
    for document, provenance in documents:
        if "array_file" in document["model"]:
            name = document["model"]["array_file"]
            if name in {"system.toml", *reserved}:
                raise ValueError(f"array file conflicts with reserved input: {name}")
            paths[name] = source.parent / name
            hashes[name] = provenance["asset_sha256"]
    return paths, hashes


def _script_gate(
    system: str, script: str, inputs: dict[str, str], repository: str, extras: list[str]
) -> dict:
    """Called inside uv's exact environment before an attempt is admitted."""
    from entanglix.models import load_system

    source = Path(system).resolve()
    constructed = load_system(source)
    paths, hashes = _system_inputs(source, constructed, reserved={"script.py"})
    paths["script.py"] = Path(script).resolve()
    for name, path in inputs.items():
        _relative_file(name)
        if name in paths or Path(name).parts[0] == "code":
            raise ValueError(f"additional input conflicts with a reserved or system input: {name}")
        paths[name] = Path(path).resolve()
    for name, path in paths.items():
        if name in hashes:
            continue
        content = path.read_bytes()
        hashes[name] = _digest(content)
        if name == "script.py" or path.suffix == ".py":
            compile(content.decode("utf-8"), str(path), "exec")
    return {
        "system_id": constructed.id,
        "inputs": {name: str(path) for name, path in paths.items()},
        "spec_sha256": hashes,
        "environment": capture_environment(Path(repository) / "uv.lock", extras=tuple(extras)),
    }


def _load_record(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True  # Permission errors leave ownership uncertain and propagate.


@dataclass
class GpuLease:
    """A device lock inherited by the job; a lost owner leaves its allocation record."""

    device: str
    run: Path | None
    descriptor: int | None
    record_path: Path
    allocated_at: float

    def fileno(self) -> int:
        if self.descriptor is None:
            raise ValueError("GPU lease is closed")
        return self.descriptor

    def bind_run(self, owner: RunHandle) -> None:
        self.fileno()
        owner.fileno()
        if self.run is not None:
            raise ValueError("GPU lease already names an attempt")
        run = owner.path.resolve()
        publish_toml(self.record_path, {"device": self.device, "run": str(run)}, replace=True)
        self.run = run

    def bind_process(self, group: int) -> None:
        self.fileno()
        if self.run is None:
            raise ValueError("GPU lease has no admitted attempt")
        if group <= 0:
            raise ValueError("process_group must be positive")
        publish_toml(
            self.record_path,
            {"device": self.device, "run": str(self.run), "process_group": group},
            replace=True,
        )

    def close(self) -> None:
        # Never unlink the lock file or explicitly unlock an inherited descriptor.
        if self.descriptor is not None:
            try:
                if self.run is not None and (self.run / "FINISHED").exists():
                    self.record_path.unlink(missing_ok=True)
            finally:
                os.close(self.descriptor)
                self.descriptor = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def select_gpu() -> str:
    """Resolve the configured single device without creating a CUDA context."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible == "":
        raise ValueError("CUDA simulation requires exactly one visible GPU")
    command = ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
    if visible is not None:
        command.append(f"--id={visible}")
    try:
        query = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"GPU selection failed: {exc}; check device visibility and driver access"
        ) from exc
    devices = query.stdout.split()
    if len(devices) != 1:
        raise ValueError("CUDA simulation requires exactly one visible GPU")
    return devices[0]


def _state_directory() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))


def acquire_gpu(device: str) -> GpuLease:
    """Acquire a canonical device UUID outside all runs roots, without waiting."""
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", device):
        raise ValueError("GPU lock requires a canonical GPU-... device identifier")
    root = _state_directory() / "entanglix/gpu"
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / f"{device}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    record_path = root / f"{device}.toml"
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"GPU {device} is still owned by a runner or inheriting job") from exc
        if record_path.exists():
            previous = _load_record(record_path)
            previous_run = Path(previous["run"])
            if previous.get("process_group") and _group_alive(previous["process_group"]):
                raise ValueError(f"GPU {device} has a live recorded process group")
            if not any(
                (previous_run / name).exists() for name in ("FINISHED", "runner/reconciled.toml")
            ):
                raise ValueError(f"GPU {device} requires reconciliation of {previous_run}")
        return GpuLease(device, None, descriptor, record_path, time.monotonic())
    except BaseException:
        os.close(descriptor)
        raise


def _signal_group(group: int, sig: int) -> None:
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def _terminate_group(group: int, alive: Callable[[], bool]) -> None:
    """Stop the group, allowing a fixed grace before forced termination."""
    _signal_group(group, signal.SIGTERM)
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if alive():
        _signal_group(group, signal.SIGKILL)


def abort_run(run: Path) -> dict:
    """Request abort and signal this attempt's group, including orphaned jobs."""
    record = _read_attempt(run)
    if record["status"] != "active" or "start" not in record:
        raise ValueError("only an active, started attempt can be aborted")
    run = Path(record["path"])
    group = record["start"]["process_group"]
    if not _group_alive(group):
        raise ValueError("recorded job group has already exited")
    try:
        publish_toml(run / "runner/abort.toml", {"requested_at": datetime.now(UTC)})
    except FileExistsError:
        pass  # Repeated requests do not replace the first event.
    _terminate_group(group, lambda: _group_alive(group))
    return {"path": str(run), "abort_requested": True}


def _execute_job(
    owner: RunHandle,
    command: list[str],
    environment: dict[str, str],
    *,
    timeout_seconds: float | None,
    gpu: GpuLease | None = None,
    job_input: str = "script.py",
    validate_result: Callable[[Path], dict] | None = None,
) -> dict:
    """Supervise the local job and its process group while retaining allocation ownership."""
    from entanglix.investigations import CancelledStart, start_permission

    run = owner.path
    manifest = _load_record(run / "run.toml")
    job_seconds = (
        manifest["investigation_commitment"]["job_seconds"]
        if timeout_seconds is None and "job_seconds" in manifest.get("investigation_limits", {})
        else None
    )
    try:
        log_stream = (run / "run.log").open("w")
    except OSError:
        record_finish(
            owner,
            status="failed",
            exit_code=64,
            wall_seconds=0,
            gpu_seconds=time.monotonic() - gpu.allocated_at if gpu else 0,
        )
        raise
    with log_stream as log:
        child = None
        clock = None
        stopped = None
        try:
            descriptors = (owner.fileno(),) + ((gpu.fileno(),) if gpu is not None else ())
            with start_permission(run) as deadline:
                child = subprocess.Popen(
                    [*command, "run", str(run / "spec" / job_input), str(owner.fileno())],
                    cwd=run / "artifacts",
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=descriptors,
                )
                record_start(
                    owner, process_group=child.pid, gpu_devices=(gpu.device,) if gpu else ()
                )
                clock = time.monotonic()
                if gpu is not None:
                    gpu.bind_process(child.pid)
                child.stdin.write(b"1")
                child.stdin.close()
            while child.poll() is None or _group_alive(child.pid):
                if (run / "runner/abort.toml").exists():
                    stopped = "abort"
                elif timeout_seconds is not None and time.monotonic() - clock >= timeout_seconds:
                    stopped = "timeout"
                elif job_seconds is not None and time.monotonic() - clock >= job_seconds:
                    stopped = "job_seconds limit"
                if deadline is not None and datetime.now(UTC) >= deadline:
                    stopped = "deadline"
                if stopped:
                    log.write(f"entanglix: {stopped}; terminating job process group\n")
                    log.flush()
                    _terminate_group(
                        child.pid, lambda: child.poll() is None or _group_alive(child.pid)
                    )
                    break
                time.sleep(0.01)
            code = child.wait()
            observed = _signal_failure(code)
            if observed and not stopped and not (run / "runner/abort.toml").exists():
                log.write(f"entanglix: job {observed}\n")
                log.flush()
            code = 128 - code if code < 0 else code
            if (run / "runner/abort.toml").exists():
                stopped = "abort"
        except CancelledStart as exc:
            stopped = "abort"
            log.write(f"entanglix: {exc}; aborting job\n")
            log.flush()
            code = 130
        except KeyboardInterrupt:
            stopped = "abort" if (run / "runner/started.toml").exists() else None
            if child is not None:
                log.write("entanglix: interrupt; terminating job process group\n")
                log.flush()
                _terminate_group(child.pid, lambda: child.poll() is None or _group_alive(child.pid))
                child.wait()
            code = 130 if stopped else 64
        except Exception as exc:
            log.write(f"entanglix: execution failed: {exc}\n")
            log.flush()
            if child is not None:
                if not child.stdin.closed:
                    child.stdin.close()
                _signal_group(child.pid, signal.SIGKILL)
                child.wait()
            code = 64
        if child is not None:
            while _group_alive(child.pid):
                time.sleep(0.01)
        elapsed = time.monotonic() - clock if clock is not None else 0
        if stopped:
            code = 130
        if code == 0:
            try:
                (validate_result or _script_result)(run)
            except Exception as exc:
                log.write(f"entanglix: execution failed: {exc}\n")
                code = 64
        status = "completed" if code == 0 else "aborted" if code == 130 else "failed"
        record_finish(
            owner,
            status=status,
            exit_code=code,
            wall_seconds=elapsed,
            gpu_seconds=time.monotonic() - gpu.allocated_at if gpu else 0,
        )
    return {"path": str(run), "exit_code": code, "status": status}


def _script_result(run: Path) -> dict:
    result = _load_record(run / "result.toml")
    if not isinstance(result.get("engine"), str) or not result["engine"]:
        raise ValueError("script result must name its actual engine")
    if not isinstance(result.get("metrics"), dict):
        raise ValueError("script result must contain a metrics table")
    if not isinstance(result.get("artifacts", {}), dict):
        raise ValueError("result artifacts must be a table")
    for name, artifact in result.get("artifacts", {}).items():
        path = (run / artifact["path"]).resolve()
        if not path.is_relative_to((run / "artifacts").resolve()):
            raise ValueError(f"artifact {name}: path must stay inside artifacts/")
        if _digest(path.read_bytes()) != artifact["sha256"]:
            raise ValueError(f"artifact {name}: digest mismatch")
    return result


def _claim_runner_extras(repository: Path, extras: list[str]) -> None:
    """Pin the shared environment's profile before any uv synchronization."""
    profile = repository / ".venv-runner-extras.toml"
    if (repository / ".venv-runner/pyvenv.cfg").exists() and not profile.exists():
        raise ValueError(
            "existing .venv-runner has no extras profile; stop its jobs and confirm "
            "its extras before recording .venv-runner-extras.toml or recreating it"
        )
    try:
        publish_toml(profile, {"extras": extras})
    except FileExistsError:
        pass  # Another launcher claimed it first; compare its complete record.
    pinned = _load_record(profile)["extras"]
    if pinned != extras:
        raise ValueError(
            f"runner extras mismatch: .venv-runner is pinned to {pinned}, requested {extras}; "
            "use the pinned extras or a separate checkout; recreate the environment/profile "
            "only after all its jobs have stopped"
        )


def _cpu_job_context(repository: Path, module: str) -> tuple[list[str], dict, list[str]]:
    """Use one configured extras set in the shared exact runner environment."""
    config = _load_record(repository / "pyproject.toml")
    extras = config.get("tool", {}).get("entanglix", {}).get("runner", {}).get("extras", [])
    if "ENTANGLIX_RUNNER_EXTRAS" in os.environ:
        extras = [
            value.strip()
            for value in os.environ["ENTANGLIX_RUNNER_EXTRAS"].split(",")
            if value.strip()
        ]
    if not isinstance(extras, list) or any(not isinstance(extra, str) for extra in extras):
        raise ValueError("tool.entanglix.runner.extras must be a list of extra names")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv is required for locked, exactly synced job execution")
    extras = sorted(set(extras))
    _claim_runner_extras(repository, extras)
    command = [uv, "run", "--locked", "--exact", "--no-env-file", "--project", str(repository)]
    for extra in extras:
        command.extend(("--extra", extra))
    command.extend(("python", "-m", module))
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        JAX_PLATFORMS="cpu",
        PYTHONUNBUFFERED="1",
        UV_PROJECT_ENVIRONMENT=str(repository / ".venv-runner"),
    )
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("PYTHONPATH", None)
    environment.pop("UV_NO_SYNC", None)
    environment.pop("UV_FROZEN", None)
    return command, environment, extras


def run_script(
    system: Path,
    script: Path,
    *,
    inputs: dict[str, Path] | None = None,
    runs_root: Path = Path("research_runs"),
    repository: Path | None = None,
    parent_run_ids: tuple[str, ...] = (),
    timeout_seconds: float | None = None,
    investigation_id: str | None = None,
    base: str | None = None,
    _source_entry_ids: tuple[str, ...] = (),
) -> dict:
    """Validate and synchronously execute one CPU script."""
    repository = (repository or Path(__file__).resolve().parents[2]).resolve()
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    command, environment, extras = _cpu_job_context(repository, "entanglix._script_job")
    request = {
        "system": str(system.resolve()),
        "script": str(script.resolve()),
        "inputs": {name: str(path.resolve()) for name, path in (inputs or {}).items()},
        "repository": str(repository),
        "extras": extras,
    }
    check = subprocess.run(
        [*command, "check"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        env=environment,
    )
    if check.returncode:
        raise ValueError(process_failure(check, "locked-environment launch validation failed"))
    gate = json.loads(check.stdout)
    with create_run(
        system_id=gate["system_id"],
        protocol="script",
        investigation_id=investigation_id,
        base=base,
        inputs={name: Path(path) for name, path in gate["inputs"].items()},
        environment=gate["environment"],
        code=capture_code(repository),
        engine="authored-script",
        device="cpu",
        needs={"gradients": False, "device": "cpu"},
        runs_root=runs_root.resolve(),
        parent_run_ids=parent_run_ids,
        source_entry_ids=_source_entry_ids,
        launch={
            "input_names": sorted(inputs or {}),
            **({"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}),
        },
    ) as owner:
        run = owner.path
        if _load_record(run / "run.toml")["spec_sha256"] != gate["spec_sha256"]:
            (run / "run.log").write_text("entanglix: inputs changed after launch validation\n")
            record_finish(owner, status="failed", exit_code=64, wall_seconds=0, gpu_seconds=0)
            return {"path": str(run), "exit_code": 64, "status": "failed"}
        environment.update(
            ENTANGLIX_RUN_DIR=str(run), ENTANGLIX_SYSTEM_SPEC=str(run / "spec/system.toml")
        )
        return _execute_job(
            owner,
            command,
            environment,
            timeout_seconds=timeout_seconds,
        )


def rerun_script(
    run: Path,
    *,
    runs_root: Path | None = None,
    repository: Path | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    """Replay a retained script and its recorded inputs under a new linked ID."""
    record = read_run(run)
    if record["status"] in {"active", "unknown"}:
        raise ValueError("cannot rerun an active or uncertain attempt")
    manifest = record["manifest"]
    if manifest["protocol"] != "script" or "launch" not in manifest:
        raise ValueError("attempt has no recorded script launch; replay its frozen inputs manually")
    run = Path(record["path"])
    for name, digest in manifest["spec_sha256"].items():
        if _digest((run / "spec" / _relative_file(name)).read_bytes()) != digest:
            raise ValueError(f"frozen input digest mismatch: {name}")
    launch = manifest["launch"]
    if (
        manifest.get("investigation_id")
        and runs_root is not None
        and runs_root.resolve() != run.parent.parent
    ):
        raise ValueError("investigation reruns must retain their runs root")
    return run_script(
        run / "spec/system.toml",
        run / "spec/script.py",
        inputs={name: run / "spec" / name for name in launch["input_names"]},
        runs_root=runs_root if runs_root is not None else run.parent.parent,
        repository=repository,
        parent_run_ids=(manifest["run_id"],),
        _source_entry_ids=tuple(manifest.get("source_entry_ids", [])),
        investigation_id=manifest.get("investigation_id"),
        base=manifest.get("investigation_base"),
        timeout_seconds=timeout_seconds
        if timeout_seconds is not None
        else launch.get("timeout_seconds"),
    )


def list_capabilities(repository: Path | None = None) -> dict:
    """Inspect the runner environment without creating or synchronizing it."""
    repository = (repository or Path(__file__).resolve().parents[2]).resolve()
    paths = [str(path) for path in (repository / ".venv-runner/lib").glob("python*/site-packages")]
    found = {
        package.metadata["Name"].lower(): package.version for package in distributions(path=paths)
    }
    return {
        "version": version("entanglix"),
        "run_record_version": 1,
        "investigation_record_version": 1,
        "operations": [
            "validate_system",
            "run_script",
            "list_capabilities",
            "list_runs",
            "read_run",
            "abort_run",
            "reconcile_run",
            "rerun_script",
            "check",
            "simulate",
            "rerun_simulation",
            "sweep",
            "rerun_sweep",
            "control",
            "rerun_control",
            "calibrate",
            "rerun_calibration",
            "propose_device_update",
            "create_investigation",
            "revise_investigation",
            "list_investigations",
            "read_investigation",
            "record_finding",
            "stop_investigation",
        ],
        "protocols": ["simulate", "sweep", "control", "calibrate"],
        "runner_devices": ["cpu", "cuda"],
        "engines": {
            engine: {"version": found[package], "devices": devices}
            for engine, package, devices in (
                ("qutip-cpu", "qutip", ["cpu"]),
                ("cudensitymat", "qutip-cuquantum", ["cuda"]),
                (
                    "dynamiqs",
                    "dynamiqs",
                    ["cpu", "cuda"] if "jax-cuda12-plugin" in found else ["cpu"],
                ),
            )
            if package in found
        },
    }


def _read_attempt(run: Path) -> dict:
    """Read one attempt without modifying its records or reconciling it."""
    run = run.resolve()
    descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        active = False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            active = True
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        records = {"manifest": _load_record(run / "run.toml")}
        for label, name in (
            ("start", "runner/started.toml"),
            ("finished", "FINISHED"),
            ("reconciliation", "runner/reconciled.toml"),
            ("abort_request", "runner/abort.toml"),
            ("result", "result.toml"),
        ):
            if (run / name).exists():
                try:
                    records[label] = _load_record(run / name)
                except (OSError, ValueError) as exc:
                    if label != "result":
                        raise
                    records["result_error"] = str(exc)
        if "finished" in records:
            status = records["finished"]["status"]
        else:
            status = "active" if active else "interrupted"
            if not active and "start" in records:
                try:
                    os.killpg(records["start"]["process_group"], 0)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    status = "unknown"
                else:
                    status = "active"
    finally:
        os.close(descriptor)
    log_path = run / "run.log"
    log = ""
    if log_path.exists():
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 8192))
            log = stream.read(8192).decode("utf-8", errors="replace")
    record = {
        "path": str(run),
        "status": status,
        **records,
        "log_tail": log,
    }
    return record


def read_run(run: Path, *, wait_seconds: float = 0) -> dict:
    """Read retained records and check history, even if the frozen plan is obsolete."""
    run = run.resolve()
    from entanglix.launches import read_launch, validate_wait

    validate_wait(wait_seconds)
    expires = time.monotonic() + wait_seconds
    if run.parent.name == "_launches":
        while True:
            launch = read_launch(run)
            if launch["status"] != "pending" or time.monotonic() >= expires:
                return launch
            time.sleep(min(0.25, max(0, expires - time.monotonic())))
    while True:
        state = _read_attempt(run)["status"]
        if state != "active" or time.monotonic() >= expires:
            break
        time.sleep(min(0.25, max(0, expires - time.monotonic())))
    record = _read_attempt(run)
    if record["manifest"]["protocol"] in {
        "plan",
        "reference",
        "simulate",
        "sweep",
        "control",
        "calibrate",
    }:
        from entanglix.corrector.verification import check_run, verification_history

        record["verification"] = verification_history(run)
        try:
            record["check_listing"] = check_run(run)
        except (OSError, ValueError, RuntimeError, KeyError, MemoryError) as exc:
            record["check_listing"] = {"checks": [], "reason": failure_message(exc)}
    if record["manifest"]["protocol"] == "sweep":
        from entanglix.protocols.sweep import partial_results

        record["sweep_progress"] = partial_results(run)
    return record


def list_runs(runs_root: Path = Path("research_runs")) -> list[dict]:
    attempts = []
    for manifest in sorted(runs_root.glob("*/*/run.toml")):
        try:
            run = _read_attempt(manifest.parent)
            attempts.append(
                {
                    "path": run["path"],
                    "run_id": run["manifest"]["run_id"],
                    "system_id": run["manifest"]["system_id"],
                    "status": run["status"],
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            attempts.append(
                {
                    "path": str(manifest.parent.resolve()),
                    "status": "invalid",
                    "diagnostic": str(exc),
                }
            )
    from entanglix.launches import unresolved_launches

    return attempts + unresolved_launches(runs_root)
