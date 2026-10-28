"""Installed versions: recorded tools and extras, environment locks, components and removal."""

import fcntl
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import tomli_w

from entanglix import installation
from entanglix.cli import main
from entanglix.engines.selection import EngineSelectionError, select_engine
from entanglix.runner import _cpu_job_context, acquire_gpu, list_capabilities
from entanglix.solvers import piccolo

REPOSITORY = Path(__file__).resolve().parents[1]
HOSTILE = {
    "UV": "/elsewhere/uv",
    "UV_NO_SYNC": "1",
    "UV_FROZEN": "1",
    "UV_PROJECT_ENVIRONMENT": "/elsewhere/env",
    "UV_CACHE_DIR": "/elsewhere/cache",
    "UV_PYTHON": "/elsewhere/python",
    "VIRTUAL_ENV": "/elsewhere/venv",
    "PYTHONPATH": "/elsewhere/lib",
    "PYTHONHOME": "/elsewhere",
}


def fake_installation(root: Path, version="0.1.0", components=("base",)) -> Path:
    """An installed version's records, without its environments."""
    (root / ".installation").mkdir(parents=True)
    for name in ("pyproject.toml", ".python-version"):
        (root / name).write_bytes((REPOSITORY / name).read_bytes())
    extras = sorted({extra for name in components for extra in installation.COMPONENT_EXTRAS[name]})
    record = {
        "record_version": 1,
        "version": version,
        "uv": str(root / ".installation/bin/uv"),
        "cache_dir": str(root.parent / "uv-cache"),
    }
    (root / ".installation/installation.toml").write_text(tomli_w.dumps(record))
    state = {"components": sorted(components), "runner_extras": extras}
    (root / ".installation/components.toml").write_text(tomli_w.dumps(state))
    return root


def holder(path: Path, mode="shared"):
    """A child process holding a lock, as an idle server or a surviving job does."""
    flag = "fcntl.LOCK_SH" if mode == "shared" else "fcntl.LOCK_EX"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import fcntl, os, sys, time; d = os.open({str(path)!r}, os.O_CREAT | os.O_RDWR); "
            f"fcntl.flock(d, {flag}); print('held', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout.readline().strip() == "held"
    return child


def test_uv_environment_removes_redirecting_variables_and_keeps_network(tmp_path):
    root = fake_installation(tmp_path / "0.1.0")
    environment = installation.uv_environment(
        {**HOSTILE, "UV_NATIVE_TLS": "1", "HTTPS_PROXY": "http://proxy", "HOME": "/home/u"},
        root,
    )
    assert not HOSTILE.keys() - {"UV_CACHE_DIR"} & environment.keys()
    assert environment["UV_CACHE_DIR"] == str(tmp_path / "uv-cache")
    assert environment["UV_PYTHON_INSTALL_DIR"] == str(root / ".installation/python")
    assert environment["UV_NATIVE_TLS"] == "1" and environment["HTTPS_PROXY"] == "http://proxy"


def test_installation_jobs_use_recorded_uv_extras_and_options(tmp_path, monkeypatch):
    root = fake_installation(tmp_path / "0.1.0")
    monkeypatch.delenv("ENTANGLIX_RUNNER_EXTRAS", raising=False)
    for name, value in HOSTILE.items():
        monkeypatch.setenv(name, value)
    command, environment, extras = _cpu_job_context(root, "entanglix._script_job")
    assert command[0] == str(root / ".installation/bin/uv")
    assert {"--no-config", "--no-dev", "--managed-python"} <= set(command)
    pin = (REPOSITORY / ".python-version").read_text().strip()
    assert command[command.index("--python") + 1] == pin
    assert extras == ["differentiable"] and command[command.index("--extra") + 1] == extras[0]
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(root / ".venv-runner")
    assert environment["UV_CACHE_DIR"] == str(tmp_path / "uv-cache")
    for name in ("UV", "UV_NO_SYNC", "UV_FROZEN", "UV_PYTHON", "VIRTUAL_ENV", "PYTHONPATH"):
        assert name not in environment
    profile = tomllib.loads((root / ".venv-runner-extras.toml").read_text())
    assert profile["extras"] == ["differentiable"]


def test_an_explicit_extras_setting_overrides_the_recorded_set(tmp_path, monkeypatch):
    root = fake_installation(tmp_path / "0.1.0")
    monkeypatch.setenv("ENTANGLIX_RUNNER_EXTRAS", "")
    assert _cpu_job_context(root, "entanglix._script_job")[2] == []


def test_a_checkout_keeps_its_own_uv_and_options(runner_repository):
    command, environment, _ = _cpu_job_context(runner_repository, "entanglix._script_job")
    assert "--no-config" not in command and "--managed-python" not in command
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(runner_repository / ".venv-runner")


def test_launches_and_component_changes_exclude_each_other(tmp_path):
    root = fake_installation(tmp_path / "0.1.0")
    with installation.environment_lease(root):
        with installation.environment_lease(root):
            pass  # Launches share the environment.
        with pytest.raises(ValueError, match="a launch or job is using the job environment"):
            installation.environment_lease(root, exclusive=True)
    with installation.environment_lease(root, exclusive=True):
        with pytest.raises(ValueError, match="components are being changed"):
            installation.environment_lease(root)


def test_an_inherited_lease_outlives_its_runner(tmp_path):
    root = fake_installation(tmp_path / "0.1.0")
    lease = installation.environment_lease(root)
    job = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], pass_fds=(lease.fileno(),)
    )
    lease.close()
    try:
        with pytest.raises(ValueError, match="a launch or job is using"):
            installation.environment_lease(root, exclusive=True)
    finally:
        job.kill()
        job.wait()
    installation.environment_lease(root, exclusive=True).close()


def test_a_component_change_is_refused_during_a_launch(tmp_path, monkeypatch):
    root = fake_installation(tmp_path / "0.1.0")
    monkeypatch.setattr(installation, "gpus", lambda: [{"uuid": "GPU-1", "name": "Test"}])
    with installation.environment_lease(root):
        for name in ("gpu", "piccolo"):
            with pytest.raises(ValueError, match="a launch or job is using"):
                installation.add_component(name, root)
    assert not (root / ".installation/incomplete.toml").exists()
    assert installation.components(root)["components"] == ["base"]


def test_a_failed_change_leaves_a_mark_that_refuses_launches(tmp_path, monkeypatch):
    root = fake_installation(tmp_path / "0.1.0")
    monkeypatch.setattr(installation, "gpus", lambda: [{"uuid": "GPU-1", "name": "Test"}])

    def fail(*_):
        raise ValueError("synchronizing the job environment failed")

    monkeypatch.setattr(installation, "_sync_runner", fail)
    with pytest.raises(ValueError, match="synchronizing"):
        installation.add_component("gpu", root)
    assert installation.components(root)["components"] == ["base"]
    with pytest.raises(ValueError, match="rerun `entanglix components add gpu`"):
        installation.environment_lease(root)
    synced = []
    monkeypatch.setattr(installation, "_sync_runner", lambda _, extras, __: synced.append(extras))
    monkeypatch.setattr(installation, "_gpu_check", lambda *_: [])
    assert installation.add_component("gpu", root)["status"] == "added"
    expected = ["cuquantum", "differentiable", "differentiable-cuda12"]
    assert synced == [expected]
    assert installation.components(root) == {
        "components": ["base", "gpu"],
        "runner_extras": expected,
    }
    assert tomllib.loads((root / ".venv-runner-extras.toml").read_text())["extras"] == expected
    installation.environment_lease(root).close()
    assert installation.add_component("gpu", root)["status"] == "already added"


def test_only_rerunning_a_failed_change_clears_its_mark(tmp_path, monkeypatch):
    root = fake_installation(tmp_path / "0.1.0", components=("base", "gpu"))
    monkeypatch.setattr(installation, "gpus", lambda: [{"uuid": "GPU-1", "name": "Test"}])
    synced = []
    monkeypatch.setattr(installation, "_sync_runner", lambda _, extras, __: synced.append(extras))
    monkeypatch.setattr(installation, "_gpu_check", lambda *_: [])
    monkeypatch.setattr(installation, "_install_julia", lambda _: "/julia")
    # The change failed after recording the component, before clearing its mark.
    mark = root / ".installation/incomplete.toml"
    mark.write_text(tomli_w.dumps({"component": "gpu"}))
    with pytest.raises(
        ValueError, match="failed change to gpu; rerun `entanglix components add gpu`"
    ):
        installation.add_component("piccolo", root)
    assert mark.exists() and "julia" not in installation.components(root)
    assert installation.add_component("gpu", root)["status"] == "added"
    assert synced and not mark.exists()
    installation.environment_lease(root).close()


def test_a_change_reads_the_components_under_its_lock(tmp_path, monkeypatch):
    root = fake_installation(tmp_path / "0.1.0")
    monkeypatch.setattr(installation, "gpus", lambda: [{"uuid": "GPU-1", "name": "Test"}])
    monkeypatch.setattr(installation, "_sync_runner", lambda *_: None)
    monkeypatch.setattr(installation, "_gpu_check", lambda *_: [])
    monkeypatch.setattr(installation, "_install_julia", lambda _: "/julia")
    lease = installation.environment_lease
    finished = []

    def another_change_finishes_first(root, *, exclusive=False):
        if exclusive and not finished:
            finished.append("gpu")
            assert installation.add_component("gpu", root)["status"] == "added"
        return lease(root, exclusive=exclusive)

    monkeypatch.setattr(installation, "environment_lease", another_change_finishes_first)
    installation.add_component("piccolo", root)
    assert finished == ["gpu"]
    state = installation.components(root)
    assert state["components"] == ["base", "gpu", "piccolo"] and state["julia"] == "/julia"
    assert state["runner_extras"] == ["cuquantum", "differentiable", "differentiable-cuda12"]


def test_the_gpu_check_holds_the_environment_and_each_free_device(tmp_path, monkeypatch):
    root = fake_installation(tmp_path / "0.1.0")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    devices = [{"uuid": "GPU-free", "name": "A"}, {"uuid": "GPU-busy", "name": "B"}]
    monkeypatch.setattr(installation, "gpus", lambda: devices)
    monkeypatch.setattr(installation, "_sync_runner", lambda *_: None)
    checked = []

    def check(command, **options):
        device = command[-1]
        assert options["env"]["CUDA_VISIBLE_DEVICES"] == device and len(options["pass_fds"]) == 2
        with pytest.raises(ValueError, match="components are being changed"):
            installation.environment_lease(root)
        with pytest.raises(ValueError, match="still owned"):
            acquire_gpu(device)
        checked.append(device)
        return subprocess.CompletedProcess(command, 0, "", "")

    owner = tmp_path / "state/entanglix/gpu/GPU-busy.lock"
    owner.parent.mkdir(parents=True)
    busy = holder(owner, mode="exclusive")
    try:
        monkeypatch.setattr(installation.subprocess, "run", check)
        result = installation.add_component("gpu", root)
    finally:
        busy.kill()
        busy.wait()
    assert checked == ["GPU-free"]
    free, unchecked = result["gpu_check"]
    assert free == {"device": "GPU-free", "status": "passed"}
    assert unchecked["status"] == "not checked" and "still owned" in unchecked["detail"]


def test_the_gpu_component_needs_a_driver(tmp_path, monkeypatch):
    root = fake_installation(tmp_path / "0.1.0")
    monkeypatch.setattr(installation, "gpus", lambda: [])
    with pytest.raises(ValueError, match="needs an NVIDIA driver"):
        installation.add_component("gpu", root)


def test_refusal_hints_name_the_fix(tmp_path, monkeypatch):
    root = fake_installation(tmp_path / "0.1.0")
    assert installation.missing_hint("cudensitymat", tmp_path) == (
        "; install the cuquantum runner extra"
    )
    monkeypatch.setattr(installation, "gpus", lambda: [])
    assert "needs an NVIDIA driver (none found)" in installation.missing_hint("cudensitymat", root)
    monkeypatch.setattr(installation, "gpus", lambda: [{"uuid": "GPU-1", "name": "Test"}])
    assert "`entanglix components add gpu`" in installation.missing_hint("dynamiqs-cuda", root)
    assert "`entanglix components add piccolo`" in installation.missing_hint("piccolo", root)
    assert installation.missing_hint("dynamiqs", root) == ""
    assert installation.missing_hint("qutip-cpu", root) == ""


def test_a_missing_built_in_engine_is_refused_with_its_remedy():
    import numpy as np

    from entanglix.engines import EvolutionInput
    from entanglix.models import load_system

    system = load_system(REPOSITORY / "examples/two-level/system.toml")
    initial = np.zeros(np.prod(system.dimensions))
    initial[0] = 1
    request = EvolutionInput(initial, np.array([0.0, 1.0]))
    with pytest.raises(EngineSelectionError, match="not installed; install the cuquantum"):
        select_engine(system, request, available_engines={"qutip-cpu"}, device="cuda")


def test_piccolo_resolves_julia_once_for_validation_and_the_solve(tmp_path, monkeypatch):
    calls = tmp_path / "calls"
    first, second = tmp_path / "julia-first", tmp_path / "julia-second"
    for script in (first, second):
        script.write_text(
            "#!/bin/sh\n"
            f'echo "$0 $*" >> {calls}\n'
            '[ "$2" = --version ] && echo "julia version 1.13.0" && exit 0\n'
            "exit 3\n"
        )
        script.chmod(0o755)
    monkeypatch.setattr(piccolo, "validate_problem", lambda *_: None)
    monkeypatch.setattr(installation, "julia", lambda: (str(first), dict(os.environ)))
    backend = piccolo.PiccoloBackend()
    runtime = backend.validate({}, {})
    assert runtime["julia_executable"] == str(first)
    assert runtime["julia_version"] == "julia version 1.13.0"
    # A later lookup would find another Julia; the solve keeps the validated one.
    monkeypatch.setattr(installation, "julia", lambda: (str(second), dict(os.environ)))
    run = tmp_path / "run"
    (run / "spec").mkdir(parents=True)
    (run / "artifacts").mkdir()
    with pytest.raises(ValueError, match="Piccolo exited with code 3"):
        backend.solve(run / "spec/problem.toml", {})
    assert [line.split()[0] for line in calls.read_text().splitlines()] == [str(first)] * 2


def test_julia_belongs_to_the_installation(tmp_path, monkeypatch):
    root = fake_installation(tmp_path / "0.1.0")
    with pytest.raises(ValueError, match="Piccolo is not added; add it with `entanglix"):
        installation.julia(root)
    executable = root / ".installation/julia-1.13.0/bin/julia"
    state = installation.components(root) | {"julia": str(executable)}
    (root / ".installation/components.toml").write_text(tomli_w.dumps(state))
    monkeypatch.setenv("JULIA_PROJECT", "/elsewhere")
    monkeypatch.setenv("JULIA_DEPOT_PATH", "/elsewhere/depot")
    monkeypatch.setenv("JULIA_PKG_SERVER", "https://pkg.example")
    found, environment = installation.julia(root)
    assert found == str(executable)
    assert "JULIA_PROJECT" not in environment
    assert environment["JULIA_PKG_SERVER"] == "https://pkg.example"
    depots = environment["JULIA_DEPOT_PATH"].split(os.pathsep)
    assert depots[0] == str(root / ".installation/julia-depot")
    assert all(depot.startswith(str(root)) for depot in depots)


def test_capabilities_report_solvers_and_the_installation(tmp_path):
    root = fake_installation(tmp_path / "0.1.0")
    capabilities = list_capabilities(root)
    assert capabilities["installation"] == {"version": "0.1.0", "components": ["base"]}
    assert capabilities["solvers"]["grape"]["available"] is False
    piccolo_state = capabilities["solvers"]["piccolo"]
    assert piccolo_state["available"] is False
    assert "components add piccolo" in piccolo_state["reason"]


def test_removal_waits_for_servers_processes_launches_and_jobs(tmp_path):
    older = fake_installation(tmp_path / "0.1.0")
    current = fake_installation(tmp_path / "0.2.0", version="0.2.0")
    server = holder(older / ".installation/use.lock")
    try:
        with pytest.raises(ValueError, match="in use by an MCP server"):
            installation.remove_version("0.1.0", current)
        # An idle server does not block a component change.
        installation.environment_lease(older, exclusive=True).close()
    finally:
        server.kill()
        server.wait()
    job = holder(older / ".installation/environment.lock")
    try:
        with pytest.raises(ValueError, match="in use"):
            installation.remove_version("0.1.0", current)
    finally:
        job.kill()
        job.wait()
    assert installation.remove_version("0.1.0", current)["removed"] == "0.1.0"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["0.2.0"]
    for version in ("0.1.0", "../0.2.0", ".hidden"):
        with pytest.raises(ValueError, match="not an installed version|is not installed"):
            installation.remove_version(version, current)


def test_the_use_lock_refuses_a_version_being_removed(tmp_path):
    root = fake_installation(tmp_path / "0.1.0")
    remover = holder(root / ".installation/use.lock", mode="exclusive")
    try:
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; from entanglix import installation; "
                "installation.hold_use_lock(Path(sys.argv[1]))",
                str(root),
            ],
            capture_output=True,
            text=True,
        )
    finally:
        remover.kill()
        remover.wait()
    assert "Entanglix 0.1.0 is being removed" in check.stderr


def launcher_world(tmp_path):
    data = tmp_path / "data"
    for version, complete in (("0.1.0", True), ("0.2.0", True), ("0.10.0", False)):
        root = data / version
        (root / ".venv/bin").mkdir(parents=True)
        program = root / ".venv/bin/entanglix"
        program.write_text(f'#!/bin/sh\necho "{version} $*"\n')
        program.chmod(0o755)
        if complete:
            fake_installation(root, version=version)
    launcher = installation.write_launcher(tmp_path / "bin", data)
    return data, launcher


def project(directory: Path, version: str) -> Path:
    (directory / ".entanglix").mkdir(parents=True)
    (directory / ".entanglix/project.toml").write_text(
        tomli_w.dumps({"installation_version": version})
    )
    return directory


def launch(launcher, *arguments, cwd):
    return subprocess.run(
        [str(launcher), *arguments], cwd=cwd, capture_output=True, text=True, check=False
    )


def test_the_launcher_runs_the_projects_version_else_the_newest(tmp_path):
    _, launcher = launcher_world(tmp_path)
    older = project(tmp_path / "older", "0.1.0")
    (older / "inputs").mkdir()
    assert launch(launcher, "read-run", "x", cwd=tmp_path).stdout == "0.2.0 read-run x\n"
    assert launch(launcher, "promote", "x", cwd=older / "inputs").stdout == "0.1.0 promote x\n"
    # Project setup always runs on the newest version, so a refresh can move the project.
    refreshed = launch(launcher, "setup-project", ".", "--refresh", cwd=older)
    assert refreshed.stdout == "0.2.0 setup-project . --refresh\n"


def test_the_launcher_refuses_a_project_whose_version_was_removed(tmp_path):
    _, launcher = launcher_world(tmp_path)
    removed = project(tmp_path / "removed", "0.0.9")
    refused = launch(launcher, "list-runs", cwd=removed)
    assert refused.returncode == 64
    assert f"run: entanglix setup-project {removed} --refresh" in refused.stderr
    refreshed = launch(launcher, "setup-project", str(removed), "--refresh", cwd=removed)
    assert refreshed.stdout.startswith("0.2.0 setup-project")


def test_commands_refuse_records_of_a_project_on_another_version(tmp_path):
    root = fake_installation(tmp_path / "0.2.0", version="0.2.0")
    older = project(tmp_path / "older", "0.1.0")
    current = project(tmp_path / "current", "0.2.0")
    with pytest.raises(ValueError, match=re.escape(f"{older} uses Entanglix 0.1.0, not 0.2.0")):
        installation.check_project_paths([older / "runs/system/r1"], root)
    installation.check_project_paths([current / "runs/system/r1", tmp_path / "loose"], root)
    installation.check_project_paths([older / "runs"], tmp_path)  # A checkout checks nothing.


@pytest.mark.parametrize(
    ("arguments", "records"),
    [
        (["simulate", "IN/simulate.toml", "--runs-root", "R", "--dry-run"], ["R"]),
        (["run-script", "IN/job.py", "--system", "IN/system.toml", "--runs-root", "R"], ["R"]),
        (["check", "R/r1", "--refinement", "IN/r.toml", "--holdout", "IN/h.toml"], ["R/r1"]),
        (["rerun", "R/r1"], ["R/r1"]),
        (
            ["apply-device-update", "R/r1/proposals/p.toml", "IN/system.toml"],
            ["R/r1/proposals/p.toml"],
        ),
        (["record-finding", "R/i", "IN/finding.toml"], ["R/i"]),
        (["report-investigation", "R/i", "--output", "IN/report.md"], ["R/i"]),
    ],
)
def test_commands_check_record_paths_but_not_authored_inputs(arguments, records, monkeypatch):
    monkeypatch.setattr(installation, "hold_use_lock", lambda: None)
    checked = []

    def check(paths):
        checked.extend(paths)
        raise ValueError("checked")

    monkeypatch.setattr(installation, "check_project_paths", check)
    assert main(arguments) == 64
    assert checked == [Path(record) for record in records]


def test_a_checkout_has_no_components_to_manage(capsys):
    assert main(["components"]) == 64
    assert "manages an installed Entanglix" in capsys.readouterr().err


def test_installing_finishes_once_and_a_reinstall_changes_nothing(tmp_path, monkeypatch):
    root = tmp_path / "data/0.1.0"
    root.mkdir(parents=True)
    synced = []
    monkeypatch.setattr(installation, "_sync_runner", lambda _, extras, __: synced.append(extras))
    monkeypatch.setattr(installation, "gpus", lambda: [])
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    arguments = dict(uv="/data/uv", cache=str(tmp_path / "cache"), bin_directory=tmp_path / "bin")
    lines = installation.install(**arguments, root=root)
    assert synced == [["differentiable"]]
    record = (root / ".installation/installation.toml").read_bytes()
    assert tomllib.loads(record.decode())["uv"] == "/data/uv"
    assert installation.components(root)["components"] == ["base"]
    assert any("is not on PATH" in line for line in lines)
    assert any("No NVIDIA driver found" in line for line in lines)
    assert (tmp_path / "bin/entanglix").stat().st_mode & 0o111
    installation.install(**arguments, root=root)
    assert synced == [["differentiable"]]
    assert (root / ".installation/installation.toml").read_bytes() == record


def test_the_install_script_names_this_release():
    script = (REPOSITORY / "install.sh").read_text()
    project_version = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())["project"]
    assert re.search(rf"^  version={re.escape(project_version['version'])}$", script, re.M)
    uv_version = re.search(r"^  uv_version=(\S+)$", script, re.M).group(1)
    major, minor, _ = (int(part) for part in uv_version.split("."))
    assert (major, minor) == (0, 12)  # pyproject.toml requires uv >=0.12,<0.13
    assert script.rstrip().endswith('main "$@"')
    manifest = tomllib.loads((REPOSITORY / "src/entanglix/solvers/julia/Manifest.toml").read_text())
    assert manifest["julia_version"] == installation.JULIA_VERSION
    subprocess.run(["bash", "-n", str(REPOSITORY / "install.sh")], check=True)


def test_a_lease_file_is_ignored_by_git(tmp_path):
    ignored = subprocess.run(
        ["git", "-C", str(REPOSITORY), "check-ignore", "-q", ".installation/environment.lock"]
    )
    assert ignored.returncode == 0


@pytest.fixture(autouse=True)
def _no_process_use_lock(monkeypatch):
    # The test process itself never holds a use lock between tests.
    monkeypatch.setattr(installation, "_use_lock", None)
    yield
    if installation._use_lock is not None:
        fcntl.flock(installation._use_lock, fcntl.LOCK_UN)
        os.close(installation._use_lock)
