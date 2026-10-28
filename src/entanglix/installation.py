"""Installed versions: recorded tools and components, environment locks, and removal."""

import argparse
import fcntl
import hashlib
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import tomli_w

ROOT = Path(__file__).resolve().parents[2]
STATE = ".installation"
COMPONENT_EXTRAS = {
    "base": ("differentiable",),
    "gpu": ("cuquantum", "differentiable-cuda12"),
    "piccolo": (),
}
# uv settings that shape only network access pass through to an installation's calls.
NETWORK_VARIABLES = frozenset(
    {
        "UV_NATIVE_TLS",
        "UV_HTTP_TIMEOUT",
        "UV_HTTP_RETRIES",
        "UV_CONCURRENT_DOWNLOADS",
        "UV_CONCURRENT_BUILDS",
        "UV_CONCURRENT_INSTALLS",
    }
)
JULIA_VERSION = "1.13.0"
JULIA = {
    "x86_64": (
        "https://julialang-s3.julialang.org/bin/linux/x64/1.13/julia-1.13.0-linux-x86_64.tar.gz",
        "8975da61c128a5e5ded3e719e868da8c8781deb7ad7913d37fb99be02a81904b",
    ),
    "aarch64": (
        "https://julialang-s3.julialang.org/bin/linux/aarch64/1.13/julia-1.13.0-linux-aarch64.tar.gz",
        "6cd4a3e4baa2dc5f55638c28e9835fc294f41c78ada4a740dc408436778ab8b4",
    ),
}
# What a refused request lacks: its installation component and a checkout's own remedy.
NEEDS = {
    "cudensitymat": ("gpu", "install the cuquantum runner extra"),
    "dynamiqs-cuda": ("gpu", "install the differentiable-cuda12 runner extra"),
    "dynamiqs": ("base", "install the differentiable runner extra"),
    "piccolo": ("piccolo", "install Julia 1.13 on PATH and instantiate the pinned project"),
}
PLUGIN_GROUPS = {"engines": "entanglix.integrators", "solvers": "entanglix.solvers"}
CHECKOUT_PLUGINS = "installs plugins with uv pip install --python .venv-runner/bin/python"
LAUNCHER = """#!/usr/bin/env bash
# Entanglix launcher: runs the installation a project records, else the newest one.
set -eu
root={root}
version=
project=
if [ "${{1-}}" != setup-project ]; then
  directory=$PWD
  while :; do
    if [ -f "$directory/.entanglix/project.toml" ]; then
      project=$directory
      version=$(sed -n 's/^installation_version = "\\(.*\\)"$/\\1/p' "$directory/.entanglix/project.toml")
      break
    fi
    [ "$directory" = / ] && break
    directory=$(dirname "$directory")
  done
fi
if [ -n "$project" ]; then
  if [ -z "$version" ] || [ ! -f "$root/$version/.installation/installation.toml" ]; then
    echo "entanglix: $project uses Entanglix $version, which is not installed;" \\
      "run: entanglix setup-project $project --refresh" >&2
    exit 64
  fi
else
  version=$(for record in "$root"/*/.installation/installation.toml; do
    [ -f "$record" ] && basename "$(dirname "$(dirname "$record")")"
  done | sort -V | tail -n 1)
  if [ -z "$version" ]; then
    echo "entanglix: no complete installation in $root" >&2
    exit 64
  fi
fi
exec "$root/$version/.venv/bin/entanglix" "$@"
"""

_use_lock: int | None = None


def _load(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _publish(path: Path, record: dict) -> None:
    """Replace a state record atomically."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(tomli_w.dumps(record).encode())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def record(root: Path = ROOT) -> dict | None:
    """The completion record of an installed version; None in a development checkout."""
    path = root / STATE / "installation.toml"
    return _load(path) if path.is_file() else None


def components(root: Path = ROOT) -> dict:
    path = root / STATE / "components.toml"
    if path.is_file():
        return _load(path)
    return {"components": ["base"], "runner_extras": list(COMPONENT_EXTRAS["base"])}


def _require(
    root: Path, checkout: str = "sets its runner extras with ENTANGLIX_RUNNER_EXTRAS"
) -> dict:
    installed = record(root)
    if installed is None:
        raise ValueError(
            f"this command manages an installed Entanglix; a development checkout {checkout}"
        )
    return installed


def uv_environment(base, root: Path = ROOT, installed: dict | None = None) -> dict[str, str]:
    """Inherited variables cannot redirect an installation's uv; network settings pass."""
    installed = installed or _require(root)
    environment = {
        name: value
        for name, value in base.items()
        if not (
            name in {"UV", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"}
            or (name.startswith("UV_") and name not in NETWORK_VARIABLES)
        )
    }
    environment.update(
        UV_CACHE_DIR=installed["cache_dir"],
        UV_PYTHON_INSTALL_DIR=str(root / STATE / "python"),
    )
    return environment


def uv_options(root: Path = ROOT) -> tuple[str, ...]:
    """Options of every installation uv call.

    uv looks for .python-version from the working directory, so a user's project could
    otherwise request another interpreter and rebuild the shared environment.
    """
    python = (root / ".python-version").read_text().strip()
    return ("--no-config", "--no-dev", "--managed-python", "--python", python)


class Lease:
    """A held environment lock; a job that inherits the descriptor keeps it held."""

    def __init__(self, descriptor: int):
        self.descriptor = descriptor

    def fileno(self) -> int:
        if self.descriptor is None:
            raise ValueError("environment lease is closed")
        return self.descriptor

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def environment_lease(root: Path = ROOT, *, exclusive: bool = False) -> Lease:
    """A launch's shared hold on the job environment, or a component change's exclusive one."""
    state = root / STATE
    state.mkdir(exist_ok=True)
    descriptor = os.open(state / "environment.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "a launch or job is using the job environment; change it after they end"
                if exclusive
                else "components or plugins are being changed; launch after the change ends"
            ) from exc
        if not exclusive:
            _refuse_incomplete(root)
        return Lease(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def _refuse_incomplete(root: Path) -> None:
    mark = root / STATE / "incomplete.toml"
    if mark.exists():
        raise ValueError(
            "the job environment is incomplete after a failed component change; rerun "
            f"`entanglix components add {_load(mark)['component']}`"
        )


def _exclusive(path: Path, busy: str) -> int:
    """An exclusive lock taken without waiting; the caller closes the descriptor."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise ValueError(busy) from exc
    return descriptor


def hold_use_lock(root: Path = ROOT) -> None:
    """Keep an installed version from removal while this process lives."""
    global _use_lock
    installed = record(root)
    if _use_lock is not None or installed is None:
        return
    descriptor = os.open(root / STATE / "use.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise ValueError(f"Entanglix {installed['version']} is being removed") from exc
    _use_lock = descriptor


def check_project_paths(paths, root: Path = ROOT) -> None:
    """Refuse records of a project that another installed version serves."""
    installed = record(root)
    if installed is None:
        return
    for path in paths:
        resolved = Path(path).resolve()
        for directory in (resolved, *resolved.parents):
            project = directory / ".entanglix/project.toml"
            if project.is_file():
                version = _load(project).get("installation_version")
                if version != installed["version"]:
                    raise ValueError(
                        f"{directory} uses Entanglix {version}, not {installed['version']}; run "
                        f"this command from {directory}, or move the project with "
                        f"`entanglix setup-project {directory} --refresh`"
                    )
                break


def gpus() -> list[dict[str, str]]:
    """NVIDIA devices the driver reports; none without a driver."""
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in filter(str.strip, query.stdout.splitlines()):
        uuid, _, name = (part.strip() for part in line.partition(","))
        devices.append({"uuid": uuid, "name": name})
    return devices


def missing_hint(need: str, root: Path = ROOT) -> str:
    """How to add what a refused request needs, for an installation or a checkout."""
    if need not in NEEDS:
        return ""
    component, remedy = NEEDS[need]
    if record(root) is None:
        return f"; {remedy}"
    if component in components(root)["components"]:
        return ""
    if component == "gpu" and not gpus():
        return "; it comes with the GPU component, which needs an NVIDIA driver (none found)"
    size = " (several GB)" if component == "gpu" else ""
    return f"; add it with `entanglix components add {component}`{size}"


def julia(root: Path = ROOT) -> tuple[str, dict[str, str]]:
    """The Julia executable and environment Piccolo uses, or a refusal naming the fix."""
    if record(root) is None:
        executable = shutil.which("julia")
        if executable is None:
            raise ValueError("piccolo requires Julia on PATH and its pinned project instantiated")
        return executable, dict(os.environ)
    executable = components(root).get("julia")
    if executable is None:
        raise ValueError("Piccolo is not added" + missing_hint("piccolo", root))
    return executable, _julia_environment(root, Path(executable))


def _julia_environment(root: Path, executable: Path) -> dict[str, str]:
    # Julia's default depots follow the user's; the installation's own depot replaces it.
    home = executable.parent.parent
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("JULIA_") or name == "JULIA_PKG_SERVER"
    }
    environment["JULIA_DEPOT_PATH"] = os.pathsep.join(
        (
            str(root / STATE / "julia-depot"),
            str(home / "local/share/julia"),
            str(home / "share/julia"),
        )
    )
    return environment


def _run(
    command: list[str], environment: dict[str, str], failure: str, lease: Lease | None = None
) -> None:
    """Run a step; one that changes the environment inherits the lease, so it stays held if the
    command that started it dies."""
    pass_fds = (lease.fileno(),) if lease is not None else ()
    if subprocess.run(command, env=environment, pass_fds=pass_fds).returncode:
        raise ValueError(failure)


def _sync_runner(root: Path, extras: list[str], installed: dict, lease: Lease) -> None:
    command = [
        installed["uv"],
        "sync",
        "--project",
        str(root),
        "--locked",
        "--inexact",
        *uv_options(root),
        *(f"--extra={extra}" for extra in extras),
    ]
    environment = uv_environment(os.environ, root, installed)
    environment["UV_PROJECT_ENVIRONMENT"] = str(root / ".venv-runner")
    _run(command, environment, "synchronizing the job environment failed; see the uv output", lease)


def _install_julia(root: Path, lease: Lease) -> str:
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(platform.machine(), platform.machine())
    if machine not in JULIA:
        raise ValueError(f"Piccolo's Julia is available for x86_64 and aarch64, not {machine}")
    url, digest = JULIA[machine]
    target = root / STATE / f"julia-{JULIA_VERSION}"
    executable = target / "bin/julia"
    if not executable.exists():
        with tempfile.TemporaryDirectory(dir=root / STATE) as scratch:
            archive = Path(scratch) / "julia.tar.gz"
            _run(
                ["curl", "-fsSL", "-o", str(archive), url],
                dict(os.environ),
                "Julia download failed",
            )
            if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
                raise ValueError("the Julia download does not match its checksum")
            with tarfile.open(archive) as bundle:
                bundle.extractall(scratch, filter="data")
            (Path(scratch) / f"julia-{JULIA_VERSION}").rename(target)
    project = root / "src/entanglix/solvers/julia"
    _run(
        [
            str(executable),
            "--startup-file=no",
            f"--project={project}",
            "-e",
            "using Pkg; Pkg.instantiate(); Pkg.precompile()",
        ],
        _julia_environment(root, executable),
        "instantiating the Piccolo project failed; see the Julia output",
        lease,
    )
    return str(executable)


def _gpu_check(root: Path, lease: Lease) -> list[dict]:
    """Run the existing GPU environment check on each free device; a failure is reported."""
    from entanglix.runner import acquire_gpu

    results = []
    for device in gpus():
        try:
            gpu = acquire_gpu(device["uuid"])
        except ValueError as exc:
            # CUDA is not initialized on a device another run owns.
            results.append(
                {
                    "device": device["uuid"],
                    "status": "not checked",
                    "detail": f"{exc}; its first CUDA job checks it at launch",
                }
            )
            continue
        environment = uv_environment(os.environ, root)
        environment["CUDA_VISIBLE_DEVICES"] = device["uuid"]
        with gpu:
            check = subprocess.run(
                [
                    str(root / ".venv-runner/bin/python"),
                    "-c",
                    "import sys; from entanglix.runner import check_gpu_environment; "
                    "check_gpu_environment(sys.argv[1], engine='cudensitymat')",
                    device["uuid"],
                ],
                env=environment,
                capture_output=True,
                text=True,
                pass_fds=(lease.fileno(), gpu.fileno()),
            )
        detail = (check.stderr.strip().splitlines() or [""])[-1]
        results.append(
            {
                "device": device["uuid"],
                "status": "passed" if check.returncode == 0 else "failed",
                **({"detail": detail} if check.returncode else {}),
            }
        )
    return results


def add_component(name: str, root: Path = ROOT) -> dict:
    """Add GPU engines or Piccolo; refused while any launch or job uses the environment."""
    installed = _require(root)
    if name not in {"gpu", "piccolo"}:
        raise ValueError("components are gpu and piccolo")
    mark = root / STATE / "incomplete.toml"
    with environment_lease(root, exclusive=True) as lease:
        # Read under the lock, so a change that finished meanwhile is kept. A failed change
        # is finished only by rerunning it; no other change clears its mark.
        current = components(root)
        if mark.exists():
            pending = _load(mark)["component"]
            if pending != name:
                raise ValueError(
                    f"the job environment is incomplete after a failed change to {pending}; "
                    f"rerun `entanglix components add {pending}` first"
                )
        elif name in current["components"]:
            return {"component": name, "status": "already added"}
        if name == "gpu" and not gpus():
            raise ValueError("the GPU component needs an NVIDIA driver, and nvidia-smi found none")
        _publish(mark, {"component": name, "started_at": datetime.now(UTC)})
        updated = {**current, "components": sorted({*current["components"], name})}
        if name == "gpu":
            extras = sorted({*current["runner_extras"], *COMPONENT_EXTRAS["gpu"]})
            _publish(root / ".venv-runner-extras.toml", {"extras": extras})
            _sync_runner(root, extras, installed, lease)
            updated["runner_extras"] = extras
        else:
            updated["julia"] = _install_julia(root, lease)
        _publish(root / STATE / "components.toml", updated)
        mark.unlink()
        result = {"component": name, "status": "added"}
        if name == "gpu":
            # Under the change's lease, so no other change alters the checked environment.
            result["gpu_check"] = _gpu_check(root, lease)
    return result


def status(root: Path = ROOT) -> dict:
    installed = _require(root)
    current = components(root)
    devices = gpus()
    addable = {}
    if "gpu" not in current["components"]:
        addable["gpu"] = (
            "entanglix components add gpu (several GB)"
            if devices
            else "needs an NVIDIA driver (none found)"
        )
    if "piccolo" not in current["components"]:
        addable["piccolo"] = f"entanglix components add piccolo (Julia {JULIA_VERSION})"
    return {
        "version": installed["version"],
        "path": str(root),
        "components": current["components"],
        "runner_extras": current["runner_extras"],
        **({"julia": current["julia"]} if "julia" in current else {}),
        "gpus": devices,
        "can_add": addable,
    }


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def plugins(root: Path = ROOT) -> dict[str, dict]:
    """Distributions in the job environment, besides Entanglix, that register engines or solvers."""
    from importlib.metadata import distributions

    _require(root, CHECKOUT_PLUGINS)
    paths = [str(path) for path in (root / ".venv-runner/lib").glob("python*/site-packages")]
    found = {}
    for dist in distributions(path=paths):
        name = _normalized(dist.metadata["Name"])
        registered = {
            kind: sorted(entry.name for entry in dist.entry_points if entry.group == group)
            for kind, group in PLUGIN_GROUPS.items()
        }
        if name != "entanglix" and any(registered.values()):
            found[name] = {"version": dist.version, **registered}
    return dict(sorted(found.items()))


def _plugin_name(source: Path) -> str:
    """A plugin source's name, refused unless it registers an engine or a solver."""
    from importlib.metadata import PathDistribution

    if source.is_dir():
        path = source / "pyproject.toml"
        project = _load(path).get("project", {}) if path.is_file() else {}
        name = project.get("name")
        groups = {group for group, entries in project.get("entry-points", {}).items() if entries}
    elif source.is_file() and source.suffix == ".whl":
        try:
            with zipfile.ZipFile(source) as wheel:
                [info] = {
                    member.split("/")[0]
                    for member in wheel.namelist()
                    if member.split("/")[0].endswith(".dist-info")
                }
                dist = PathDistribution(zipfile.Path(wheel, at=f"{info}/"))
                name = dist.metadata["Name"]
                groups = {entry.group for entry in dist.entry_points}
        except (zipfile.BadZipFile, ValueError) as exc:
            raise ValueError(f"{source} is not a wheel") from exc
    else:
        raise ValueError(f"a plugin is a project directory or a wheel file; {source} is neither")
    if not isinstance(name, str):
        raise ValueError(f"{source} needs a pyproject.toml whose [project] table names it")
    if _normalized(name) == "entanglix":
        raise ValueError("Entanglix itself is not a plugin")
    if not groups & set(PLUGIN_GROUPS.values()):
        raise ValueError(
            f"{source} registers no entanglix.integrators or entanglix.solvers entry point; "
            "a project declares them in its pyproject.toml"
        )
    return _normalized(name)


def add_plugin(source: Path, root: Path = ROOT) -> dict:
    """Install a plugin from a project directory or a wheel, or refresh it, while no job runs."""
    installed = _require(root, CHECKOUT_PLUGINS)
    source = source.resolve()
    name = _plugin_name(source)
    with environment_lease(root, exclusive=True) as lease:
        _refuse_incomplete(root)
        # A reinstall by name, so a code-only edit at the same version reaches jobs whatever
        # uv's default for local sources.
        _run(
            [
                installed["uv"],
                "pip",
                "install",
                "--no-config",
                "--python",
                str(root / ".venv-runner/bin/python"),
                "--reinstall-package",
                name,
                str(source),
            ],
            uv_environment(os.environ, root, installed),
            "installing the plugin failed; see the uv output",
            lease,
        )
        return {"plugin": name, "status": "installed", **plugins(root).get(name, {})}


def remove_plugin(name: str, root: Path = ROOT) -> dict:
    """Uninstall a plugin while no launch or job uses the environment."""
    installed = _require(root, CHECKOUT_PLUGINS)
    name = _normalized(name)
    with environment_lease(root, exclusive=True) as lease:
        _refuse_incomplete(root)
        found = plugins(root)
        if name not in found:
            raise ValueError(
                f"{name} is not an installed plugin; installed: {', '.join(found) or 'none'}"
            )
        _run(
            [
                installed["uv"],
                "pip",
                "uninstall",
                "--no-config",
                "--python",
                str(root / ".venv-runner/bin/python"),
                name,
            ],
            uv_environment(os.environ, root, installed),
            "removing the plugin failed; see the uv output",
            lease,
        )
    return {"plugin": name, "status": "removed"}


def remove_version(version: str, root: Path = ROOT, bin_directory: Path | None = None) -> dict:
    """Delete an installed version only while none of its processes, launches or jobs run.

    Removing the last version also deletes Entanglix's own uv cache and launcher.
    """
    _require(root)
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+-]*", version):
        raise ValueError(f"not an installed version: {version!r}")
    data = root.parent
    target = data / version
    held = []
    try:
        # Installs take this lock too, so neither sees the other's directories half made.
        held.append(
            _exclusive(
                data / ".lock", "an Entanglix install or removal is running; remove after it ends"
            )
        )
        removed = record(target)
        if removed is None:
            raise ValueError(f"Entanglix {version} is not installed in {data}")
        for name in ("use.lock", "environment.lock"):
            held.append(
                _exclusive(
                    target / STATE / name,
                    f"Entanglix {version} is in use by an MCP server, a CLI process, a launch or "
                    "a job; remove it after they end",
                )
            )
        # The rename hides the version from the launcher before its files go.
        removing = data / f".removing-{version}-{secrets.token_hex(4)}"
        target.rename(removing)
        shutil.rmtree(removing)
        note = _clean_up(data, removed, bin_directory or Path.home() / ".local/bin")
    finally:
        for descriptor in held:
            os.close(descriptor)
    return {"removed": version, "note": note}


def _clean_up(data: Path, removed: dict, bin_directory: Path) -> str:
    """Delete Entanglix's shared cache and launcher once no version directory remains.

    The data lock and its directory stay: an installer that opened the lock file before a
    deletion would lock the old file while the next one locked a new file.
    """
    cache = data / "cache"
    remaining = sorted(path.name for path in data.iterdir() if path.name not in {"cache", ".lock"})
    if remaining:
        return f"Entanglix's uv cache {cache} stays; {data} still holds {', '.join(remaining)}"
    deleted = []
    # An earlier record may name the user's own uv cache, which stays.
    own_cache = Path(removed["cache_dir"]).resolve() == cache.resolve()
    if own_cache and cache.is_dir() and not cache.is_symlink():
        shutil.rmtree(cache)
        deleted.append(str(cache))
    launcher = bin_directory / "entanglix"
    if _own_launcher(launcher, data):
        launcher.unlink()
        deleted.append(str(launcher))
    note = "the last version is removed"
    if deleted:
        note += f"; deleted {' and '.join(deleted)}"
    if not own_cache:
        note += f"; its packages stay in {removed['cache_dir']}, which Entanglix does not own"
    return note


def _quoted(data: Path) -> str:
    return "'" + str(data).replace("'", "'\\''") + "'"


def _own_launcher(launcher: Path, data: Path) -> bool:
    """Whether a launcher is one Entanglix wrote for this data directory."""
    if launcher.is_symlink() or not launcher.is_file():
        return False
    try:
        lines = launcher.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return False
    return lines[:2] == LAUNCHER.splitlines()[:2] and f"root={_quoted(data)}" in lines


def write_launcher(bin_directory: Path, data: Path) -> Path:
    bin_directory.mkdir(parents=True, exist_ok=True)
    launcher = bin_directory / "entanglix"
    descriptor, temporary = tempfile.mkstemp(prefix=".entanglix-", dir=bin_directory)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(LAUNCHER.format(root=_quoted(data)))
    os.chmod(temporary, 0o755)
    os.replace(temporary, launcher)
    return launcher


def install(*, uv: str, cache: str, bin_directory: Path, root: Path = ROOT) -> list[str]:
    """Finish a cloned version: its base job environment, records and launcher."""
    from entanglix import __version__

    if record(root) is None:
        installed = {"uv": uv, "cache_dir": cache}
        extras = list(COMPONENT_EXTRAS["base"])
        with environment_lease(root, exclusive=True) as lease:
            _publish(root / ".venv-runner-extras.toml", {"extras": extras})
            _sync_runner(root, extras, installed, lease)
            _publish(
                root / STATE / "components.toml", {"components": ["base"], "runner_extras": extras}
            )
            # The completion record comes last: the launcher ignores a version without it.
            _publish(
                root / STATE / "installation.toml",
                {
                    "record_version": 1,
                    "version": __version__,
                    "uv": uv,
                    "cache_dir": cache,
                    "installed_at": datetime.now(UTC),
                },
            )
    launcher = write_launcher(bin_directory, root.parent)
    on_path = str(bin_directory) in os.environ.get("PATH", "").split(os.pathsep)
    command = "entanglix" if on_path else str(launcher)
    lines = [f"Entanglix {record(root)['version']} is installed in {root}", f"Command: {launcher}"]
    if not on_path:
        lines.append(
            f"{bin_directory} is not on PATH; add this line to your shell profile: "
            f'export PATH="{bin_directory}:$PATH"'
        )
    current = components(root)["components"]
    devices = gpus()
    if "gpu" not in current:
        lines.append(
            f"NVIDIA GPU found ({', '.join(device['name'] for device in devices)}); add GPU "
            f"engines with: {command} components add gpu (several GB)"
            if devices
            else "No NVIDIA driver found; CPU engines and GRAPE control are installed"
        )
    if "piccolo" not in current:
        lines.append(
            f"Add the Piccolo solver (its own Julia) with: {command} components add piccolo"
        )
    lines.append(f"Set up a project: {command} setup-project DIR --runtime claude|codex|other")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="Finish an Entanglix installation")
    subcommands = parser.add_subparsers(dest="command", required=True)
    finishing = subcommands.add_parser("install")
    finishing.add_argument("--uv", required=True)
    finishing.add_argument("--cache", required=True)
    finishing.add_argument("--bin", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        lines = install(uv=args.uv, cache=args.cache, bin_directory=args.bin)
    except (OSError, ValueError) as exc:
        print(f"entanglix install: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
