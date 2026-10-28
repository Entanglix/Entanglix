"""A user-installed plugin distribution reaches jobs; checks and reads never need it."""

import shutil
import subprocess
from pathlib import Path

import pytest
import tomli_w
from test_control import control_files as control_files
from test_simulate import protocol_files

from entanglix import installation
from entanglix.corrector.verification import check_run
from entanglix.protocols.control import control
from entanglix.protocols.simulate import simulate
from entanglix.runner import _load_record, read_run, rerun

PLUGIN = "entanglix-test-plugin"
ENGINE = '''"""Test plugin engine: QuTiP under another name, as a separate distribution."""

from dataclasses import replace

from entanglix.engines import EngineCapabilities, QuTiPIntegrator


class Engine:
    def capabilities(self):
        return EngineCapabilities("fixture-engine", "cpu", True, False)

    def evolve(self, system, request):
        result = QuTiPIntegrator().evolve(system, request)
        return replace(result, engine="fixture-engine", method={method!r})
'''
CUDA_ENGINE = '''"""Test CUDA plugin whose factory, like a JAX plugin's, depends on the job's settings."""

import os

from entanglix.engines import EngineCapabilities


class Engine:
    def __init__(self):
        settings = {
            name: os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE")
        }
        if settings != {
            "CUDA_VISIBLE_DEVICES": os.environ.get("ENTANGLIX_TEST_DEVICE"),
            "JAX_PLATFORMS": "cuda,cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }:
            raise RuntimeError(f"plugin loaded without the job's CUDA settings: {settings}")

    def capabilities(self):
        return EngineCapabilities("fixture-cuda-engine", "cuda", True, False)
'''


def write_plugin(root, *, method="fixture-1", dependencies=()):
    package = root / "src/entanglix_test_plugin"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(ENGINE.format(method=method))
    (package / "cuda.py").write_text(CUDA_ENGINE)
    shutil.copyfile(Path(__file__).parent / "data/control_plugin.py", package / "solver.py")
    project = {
        "project": {
            "name": PLUGIN,
            "version": "0.1.0",
            "requires-python": ">=3.13",
            "dependencies": list(dependencies),
            "entry-points": {
                "entanglix.integrators": {
                    "fixture-engine": "entanglix_test_plugin:Engine",
                    "fixture-cuda-engine": "entanglix_test_plugin.cuda:Engine",
                },
                "entanglix.solvers": {"fixture-solver": "entanglix_test_plugin.solver:Backend"},
            },
        },
        "build-system": {"requires": ["uv_build==0.12.0"], "build-backend": "uv_build"},
    }
    (root / "pyproject.toml").write_text(tomli_w.dumps(project))
    return root


def uv_pip(repository, *arguments):
    """Install or remove as the user does, while no job uses the runner environment."""
    python = repository / ".venv-runner/bin/python"
    subprocess.run(
        ["uv", "pip", *arguments, "--python", str(python)], check=True, capture_output=True
    )


def install(repository, plugin, *options):
    uv_pip(
        repository,
        "install",
        "--offline",
        "--no-cache",
        "--reinstall-package",
        PLUGIN,
        *options,
        str(plugin),
    )


def test_plugin_engine_by_default_is_read_checked_and_rerun_without_it(
    tmp_path, runner_repository, monkeypatch
):
    path, _ = protocol_files(tmp_path / "work")
    root = tmp_path / "runs"
    # The first launch creates the runner environment the user installs into.
    simulate(path, runs_root=root, repository=runner_repository, dry_run=True)
    plugin = write_plugin(tmp_path / "plugin")
    install(runner_repository, plugin)
    monkeypatch.setenv("ENTANGLIX_ENGINES", "cpu=fixture-engine")
    preview = simulate(path, runs_root=root, repository=runner_repository, dry_run=True)["preview"]
    selection = preview["execution"]["selection"]
    assert preview["execution"]["engine"] == "fixture-engine"
    assert selection["source"] == "default"
    assert (selection["distribution"], selection["version"]) == (PLUGIN, "0.1.0")
    segments = next(item for item in preview["checks"] if item["name"] == "segments")
    assert segments["engine"] == "qutip-cpu"

    first = simulate(path, runs_root=root, repository=runner_repository)
    assert first["exit_code"] == 0, (Path(first["path"]) / "run.log").read_text()
    run = Path(first["path"])
    manifest = read_run(run)["manifest"]
    recorded = manifest["environment"]["distributions"][PLUGIN]
    assert manifest["engine"] == "fixture-engine"
    assert manifest["launch"]["selection"]["record_sha256"] == recorded["record_sha256"]
    assert _load_record(run / "result.toml")["method"] == "fixture-1"

    uv_pip(runner_repository, "uninstall", PLUGIN)
    assert read_run(run)["status"] == "completed"
    checked = check_run(run, ("integrity", "segments"), repository=runner_repository)
    assert checked["exit_code"] == 0
    report = read_run(Path(checked["path"]))["result"]["report"]["checks"]
    assert [item["outcome"] for item in report] == ["pass", "pass"]
    assert "compares the engines" in report[1]["scope"]
    assert {entry["engine"] for entry in report[1]["segments"]} == {"qutip-cpu"}
    with pytest.raises(ValueError, match="fixture-engine' is not installed"):
        rerun(run, repository=runner_repository)

    # A rebuild at the same version changes the record; a changed default leaves reruns alone.
    install(runner_repository, write_plugin(tmp_path / "plugin", method="fixture-2"))
    monkeypatch.setenv("ENTANGLIX_ENGINES", "cpu=qutip-cpu")
    replay = rerun(run, repository=runner_repository)
    assert replay["exit_code"] == 0
    assert replay["replay"]["environment_matches"] is False
    replayed = read_run(Path(replay["path"]))["manifest"]
    assert replayed["engine"] == "fixture-engine"
    assert replayed["launch"]["selection"]["source"] == "source_run"
    assert replayed["launch"]["selection"]["record_sha256"] != recorded["record_sha256"]
    assert _load_record(Path(replay["path"]) / "result.toml")["method"] == "fixture-2"
    preview = simulate(path, runs_root=root, repository=runner_repository, dry_run=True)["preview"]
    assert preview["execution"]["selection"]["source"] == "default"


def plugins_command(repository, cache, command, *arguments):
    """An installation's plugins command, run on the fixture checkout's runner environment."""
    installed = {"version": "0.1.0", "uv": shutil.which("uv"), "cache_dir": str(cache)}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(installation, "record", lambda root: installed)
        return command(*arguments, repository)


def test_plugins_add_refreshes_a_code_only_edit_and_remove_uninstalls(
    tmp_path, runner_repository, monkeypatch
):
    path, _ = protocol_files(tmp_path / "work")
    root = tmp_path / "runs"
    cache = tmp_path / "uv-cache"
    simulate(path, runs_root=root, repository=runner_repository, dry_run=True)
    plugin = write_plugin(tmp_path / "plugin")
    added = plugins_command(runner_repository, cache, installation.add_plugin, plugin)
    assert added == {
        "plugin": PLUGIN,
        "status": "installed",
        "version": "0.1.0",
        "engines": ["fixture-cuda-engine", "fixture-engine"],
        "solvers": ["fixture-solver"],
    }
    monkeypatch.setenv("ENTANGLIX_ENGINES", "cpu=fixture-engine")
    recorded = []
    for method in ("fixture-1", "fixture-2"):
        if method != "fixture-1":
            # Only the code changes; pyproject.toml is untouched, so uv sees the same project.
            (plugin / "src/entanglix_test_plugin/__init__.py").write_text(
                ENGINE.format(method=method)
            )
            plugins_command(runner_repository, cache, installation.add_plugin, plugin)
        result = simulate(path, runs_root=root, repository=runner_repository)
        assert result["exit_code"] == 0, (Path(result["path"]) / "run.log").read_text()
        run = Path(result["path"])
        assert _load_record(run / "result.toml")["method"] == method
        recorded.append(read_run(run)["manifest"]["environment"]["distributions"][PLUGIN])
    assert recorded[0]["version"] == recorded[1]["version"]
    assert recorded[0]["record_sha256"] != recorded[1]["record_sha256"]

    removed = plugins_command(runner_repository, cache, installation.remove_plugin, PLUGIN)
    assert removed == {"plugin": PLUGIN, "status": "removed"}
    assert plugins_command(runner_repository, cache, installation.plugins) == {}
    with pytest.raises(ValueError, match="fixture-engine' is not installed"):
        simulate(path, runs_root=root, repository=runner_repository, dry_run=True)


def test_cuda_plugin_loads_in_the_gate_with_the_job_cuda_settings(
    tmp_path, runner_repository, monkeypatch
):
    device = "GPU-plugin-test"
    root = tmp_path / "runs"
    cpu, _ = protocol_files(tmp_path / "cpu")
    simulate(cpu, runs_root=root, repository=runner_repository, dry_run=True)
    install(runner_repository, write_plugin(tmp_path / "plugin"))
    path, _ = protocol_files(tmp_path / "cuda", device="cuda")
    monkeypatch.setattr("entanglix.protocols.simulate.select_gpu", lambda: device)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("ENTANGLIX_TEST_DEVICE", device)
    monkeypatch.setenv("ENTANGLIX_ENGINES", "cuda=fixture-cuda-engine")
    # The factory loads and passes; the shared device check then rejects the absent device.
    with pytest.raises(ValueError, match=f"GPU environment rejected for {device}") as error:
        simulate(path, runs_root=root, repository=runner_repository, dry_run=True)
    assert "cuquantum extra" not in str(error.value)


def test_solver_plugin_and_environment_refusals(tmp_path, runner_repository, control_files):
    protocol, definition = control_files
    definition.update(solver={"backend": "fixture-solver"})
    definition["pulse"]["T"] = 10.0
    protocol.write_text(tomli_w.dumps(definition))
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="fixture-solver' is unavailable"):
        control(protocol, runs_root=root, repository=runner_repository, dry_run=True)
    plugin = write_plugin(tmp_path / "plugin")
    install(runner_repository, plugin)
    result = control(protocol, runs_root=root, repository=runner_repository)
    assert result["exit_code"] == 0, (Path(result["path"]) / "run.log").read_text()
    run = Path(result["path"])
    solver = _load_record(run / "result.toml")["solver"]
    recorded = read_run(run)["manifest"]["environment"]["distributions"][PLUGIN]
    assert solver["actual"] == "fixture-solver" and solver["distribution"] == PLUGIN
    assert (solver["version"], solver["record_sha256"]) == (
        recorded["version"],
        recorded["record_sha256"],
    )

    install(runner_repository, plugin, "--editable")
    with pytest.raises(ValueError, match=f"editable distribution {PLUGIN}"):
        control(protocol, runs_root=root, repository=runner_repository, dry_run=True)
    broken = write_plugin(tmp_path / "broken", dependencies=["entanglix-absent-requirement"])
    install(runner_repository, broken, "--no-deps")
    with pytest.raises(ValueError, match="inconsistent"):
        control(protocol, runs_root=root, repository=runner_repository, dry_run=True)
