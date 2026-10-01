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
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions, version
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


def _script_gate(
    system: str, script: str, inputs: dict[str, str], repository: str, extras: list[str]
) -> dict:
    """Called inside uv's exact environment before an attempt is admitted."""
    from entanglix.imports import forbidden_imports
    from entanglix.models import load_system

    source = Path(system).resolve()
    constructed = load_system(source)
    paths = {"system.toml": source, "script.py": Path(script).resolve()}
    # Reuse the loader's digests so freezing checks the bytes actually validated.
    hashes = {"system.toml": constructed.provenance["system_sha256"]}
    document = tomllib.loads(source.read_text())
    documents = [(document, constructed.provenance)]
    for subsystem in document["model"].get("subsystems", []):
        if "source" in subsystem:
            name = subsystem["source"]
            if name in {"system.toml", "script.py"}:
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
            paths[name] = source.parent / name
            hashes[name] = provenance["asset_sha256"]
    for name, path in inputs.items():
        _relative_file(name)
        if name in paths or Path(name).parts[0] == "code":
            raise ValueError(f"additional input conflicts with a reserved or system input: {name}")
        paths[name] = Path(path).resolve()
    allowed = set(
        tomllib.loads((Path(repository) / "pyproject.toml").read_text())["tool"]["entanglix"][
            "import-boundary"
        ]["allowed-third-party"]
    )
    for name, path in paths.items():
        if name in hashes:
            continue
        content = path.read_bytes()
        hashes[name] = _digest(content)
        if name == "script.py" or path.suffix == ".py":
            source_text = content.decode("utf-8")
            compile(source_text, str(path), "exec")
            forbidden = forbidden_imports(source_text, allowed, standalone=True)
            if forbidden:
                raise ValueError(f"{name}: imports outside the allowlist: {forbidden}")
    return {
        "system_id": constructed.id,
        "inputs": {name: str(path) for name, path in paths.items()},
        "spec_sha256": hashes,
        "environment": capture_environment(Path(repository) / "uv.lock", extras=tuple(extras)),
    }


def _load_record(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _script_result(run: Path) -> dict:
    result = _load_record(run / "result.toml")
    if "verdict" in result:
        raise ValueError("authored scripts cannot publish verification verdicts")
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


def run_script(
    system: Path,
    script: Path,
    *,
    inputs: dict[str, Path] | None = None,
    runs_root: Path = Path("research_runs"),
    repository: Path | None = None,
    parent_run_ids: tuple[str, ...] = (),
) -> dict:
    """Validate and synchronously execute one standalone CPU script."""
    repository = (repository or Path(__file__).resolve().parents[2]).resolve()
    config = _load_record(repository / "pyproject.toml")
    extras = config["tool"]["entanglix"].get("runner", {}).get("extras", [])
    if not isinstance(extras, list) or any(not isinstance(extra, str) for extra in extras):
        raise ValueError("tool.entanglix.runner.extras must be a list of extra names")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv is required for locked, exactly synced job execution")
    command = [uv, "run", "--locked", "--exact", "--no-env-file", "--project", str(repository)]
    for extra in sorted(set(extras)):
        command.extend(("--extra", extra))
    command.extend(("python", "-m", "entanglix._script_job"))
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
        raise ValueError(check.stderr.strip() or "locked-environment launch validation failed")
    gate = json.loads(check.stdout)
    with create_run(
        system_id=gate["system_id"],
        protocol="script",
        inputs={name: Path(path) for name, path in gate["inputs"].items()},
        environment=gate["environment"],
        code=capture_code(repository),
        engine="authored-script",
        device="cpu",
        needs={"gradients": False, "device": "cpu"},
        runs_root=runs_root.resolve(),
        parent_run_ids=parent_run_ids,
    ) as owner:
        run = owner.path
        try:
            log_stream = (run / "run.log").open("w")
        except OSError:
            record_finish(owner, status="failed", exit_code=64, wall_seconds=0, gpu_seconds=0)
            raise
        with log_stream as log:
            child = None
            clock = None
            elapsed = 0
            try:
                if _load_record(run / "run.toml")["spec_sha256"] != gate["spec_sha256"]:
                    raise ValueError("inputs changed between validation and freezing")
                environment.update(
                    ENTANGLIX_RUN_DIR=str(run), ENTANGLIX_SYSTEM_SPEC=str(run / "spec/system.toml")
                )
                child = subprocess.Popen(
                    [*command, "run", str(run / "spec/script.py"), str(owner.fileno())],
                    cwd=run / "artifacts",
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(owner.fileno(),),
                )
                record_start(owner, process_group=child.pid)
                clock = time.monotonic()
                child.stdin.write(b"1")
                child.stdin.close()
                code = child.wait()
                code = 128 - code if code < 0 else code
            except Exception as exc:
                log.write(f"entanglix: execution failed: {exc}\n")
                log.flush()
                if child is not None:
                    if not child.stdin.closed:
                        child.stdin.close()
                    child.wait()
                code = 64
            if child is not None:
                while True:
                    try:
                        os.killpg(child.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.01)
            elapsed = time.monotonic() - clock if clock is not None else 0
            if code == 0:
                try:
                    _script_result(run)
                except Exception as exc:
                    log.write(f"entanglix: execution failed: {exc}\n")
                    code = 64
            status = "completed" if code == 0 else "aborted" if code == 130 else "failed"
            record_finish(
                owner,
                status=status,
                exit_code=code,
                wall_seconds=elapsed,
                gpu_seconds=0,
            )
    return {"path": str(run), "exit_code": code, "status": status}


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
        "operations": [
            "validate_system",
            "run_script",
            "list_capabilities",
            "list_runs",
            "read_run",
        ],
        "protocols": [],
        "runner_devices": ["cpu"],
        "engines": {
            engine: {"version": found[package], "device": "cpu"}
            for engine, package in (("qutip-cpu", "qutip"), ("dynamiqs", "dynamiqs"))
            if package in found
        },
    }


def read_run(run: Path) -> dict:
    """Read one attempt without modifying its records or reconciling it."""
    run = run.resolve()
    descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        active = False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            active = True
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        records = {"manifest": _load_record(run / "run.toml")}
        for label, name in (
            ("start", "runner/started.toml"),
            ("finished", "FINISHED"),
            ("reconciliation", "runner/reconciled.toml"),
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
    return {
        "path": str(run),
        "status": status,
        **records,
        "log_tail": log,
    }


def list_runs(runs_root: Path = Path("research_runs")) -> list[dict]:
    attempts = []
    for manifest in sorted(runs_root.glob("*/*/run.toml")):
        try:
            run = read_run(manifest.parent)
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
    return attempts
