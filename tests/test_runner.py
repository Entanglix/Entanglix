"""Real locked CPU launches, frozen replay, and retained inspection evidence."""

import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from entanglix.runner import _script_gate, list_capabilities, list_runs, read_run, run_script

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path):
    system = tmp_path / "system.toml"
    shutil.copyfile(ROOT / "examples/two-level/system.toml", system)
    shutil.copyfile(ROOT / "examples/two-level/operators.npz", tmp_path / "operators.npz")
    script = tmp_path / "script.py"
    runs = tmp_path / "runs"

    def execute(source, **kwargs):
        script.write_text(source)
        return run_script(system, script, runs_root=runs, repository=ROOT, **kwargs)

    return execute, system, script, runs


def test_completed_decay_replays_frozen_inputs(launch, tmp_path):
    execute, system, script, runs = launch
    data = tmp_path / "gain.txt"
    data.write_text("2")
    source = (ROOT / "examples/recorded_decay.py").read_text()
    source += "\nprint('gain:', (run / 'spec/gain.txt').read_text())\n"
    source += "print('start barrier:', (run / 'runner/started.toml').exists())\n"
    first = execute(source, inputs={"gain.txt": data})
    assert first["exit_code"] == 0
    record = read_run(Path(first["path"]))
    assert record["status"] == "completed"
    assert record["manifest"]["engine"] == "authored-script"
    assert record["result"]["engine"] == "qutip-cpu"
    assert record["result"]["metrics"]["excited_population"] == pytest.approx(
        math.exp(-1), abs=1e-8
    )
    assert "gain: 2" in record["log_tail"]
    assert "start barrier: True" in record["log_tail"]
    script.write_text("raise RuntimeError('changed original')")
    data.write_text("99")
    system.write_text("invalid original")
    frozen = Path(first["path"]) / "spec"
    replay = run_script(
        frozen / "system.toml",
        frozen / "script.py",
        inputs={"gain.txt": frozen / "gain.txt"},
        runs_root=runs,
        repository=ROOT,
        parent_run_ids=(record["manifest"]["run_id"],),
    )
    repeated = read_run(Path(replay["path"]))
    assert repeated["result"] == record["result"]
    assert repeated["manifest"]["env_hash"] == record["manifest"]["env_hash"]
    assert repeated["manifest"]["parent_run_ids"] == [record["manifest"]["run_id"]]
    assert "gain: 2" in repeated["log_tail"]
    assert len(list_runs(runs)) == 2


@pytest.mark.parametrize("repeated", [False, True])
def test_subsystem_reference_is_frozen_automatically(launch, repeated):
    execute, system, _, _ = launch
    shutil.copyfile(ROOT / "examples/qubit-resonator/system.toml", system)
    shutil.copyfile(ROOT / "examples/qubit-resonator/qubit.toml", system.parent / "qubit.toml")
    if repeated:
        system.write_text("""format_version = 1
id = "two_qubits"
[units]
energy = "GHz"
time = "ns"
[model]
kind = "composite"
[[model.subsystems]]
id = "q"
source = "qubit.toml"
[[model.subsystems]]
id = "r"
source = "qubit.toml"
""")
    job = execute(
        "import os\n"
        "from pathlib import Path\n"
        "from entanglix.models import load_system\n"
        "from entanglix.runner import publish_toml\n"
        "model = load_system(os.environ['ENTANGLIX_SYSTEM_SPEC'])\n"
        "publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', {'engine': 'scqubits', "
        "'metrics': {'dimensions': list(model.dimensions)}}, replace=True)\n"
    )
    assert job["exit_code"] == 0
    record = read_run(Path(job["path"]))
    assert record["result"]["metrics"]["dimensions"] == [6, 6 if repeated else 5]
    assert "qubit.toml" in record["manifest"]["spec_sha256"]
    assert (Path(job["path"]) / "spec/qubit.toml").read_bytes() == (
        system.parent / "qubit.toml"
    ).read_bytes()


@pytest.mark.parametrize(
    "source",
    [
        "import unapproved_example_package",
        "import importlib; importlib.import_module(name)",
        "this is not valid Python !!!",
    ],
)
def test_gate_rejection_creates_no_attempt(launch, source):
    execute, _, _, runs = launch
    with pytest.raises(ValueError, match="launch rejected"):
        execute(source)
    assert not runs.exists()


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda _: "not valid TOML", "invalid TOML"),
        (lambda text: text.split("[model]")[0], "model: Field required"),
    ],
)
def test_invalid_system_rejects_before_creation(launch, change, message):
    execute, system, _, runs = launch
    system.write_text(change(system.read_text()))
    with pytest.raises(ValueError, match=message):
        execute("print('must not run')")
    assert not runs.exists()


def test_invalid_subsystem_reference_uses_model_diagnostic(launch):
    execute, system, _, runs = launch
    content = (ROOT / "examples/qubit-resonator/system.toml").read_text()
    system.write_text(content.replace('source = "qubit.toml"', 'source = "../qubit.toml"'))
    with pytest.raises(ValueError, match="source must name a TOML file beside the composite"):
        execute("print('must not run')")
    assert not runs.exists()


def test_gate_retains_digests_of_validated_model_inputs(launch, monkeypatch):
    from entanglix.models import load_system

    _, system, script, _ = launch
    script.write_text("print('unused')")
    asset = system.parent / "operators.npz"
    originals = {"system.toml": system.read_bytes(), "operators.npz": asset.read_bytes()}

    def edit_after_validation(path):
        model = load_system(path)
        system.write_text(system.read_text().replace("two_level_reference", "changed_reference"))
        asset.write_bytes(b"changed after model validation")
        return model

    monkeypatch.setattr("entanglix.models.load_system", edit_after_validation)
    gate = _script_gate(str(system), str(script), {}, str(ROOT), [])
    for name, content in originals.items():
        assert gate["spec_sha256"][name] == hashlib.sha256(content).hexdigest()
        assert (
            gate["spec_sha256"][name]
            != hashlib.sha256((system.parent / name).read_bytes()).hexdigest()
        )


def test_job_environment_and_capabilities_are_separate(launch, monkeypatch):
    from importlib.metadata import version

    execute, _, _, _ = launch
    development = Path(sys.prefix)
    before = (development / "pyvenv.cfg").read_bytes()
    job = execute(
        "import os, sys\n"
        "from pathlib import Path\n"
        "from importlib.metadata import version\n"
        "from entanglix.runner import publish_toml\n"
        "publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', {'engine': 'qutip-cpu', "
        "'metrics': {'python_prefix': sys.prefix, 'qutip_version': version('qutip')}}, replace=True)\n"
    )
    result = read_run(Path(job["path"]))["result"]
    assert Path(result["metrics"]["python_prefix"]) == ROOT / ".venv-runner"
    assert Path(result["metrics"]["python_prefix"]) != development
    assert (development / "pyvenv.cfg").read_bytes() == before

    def no_host_engine_metadata(name):
        if name != "entanglix":
            pytest.fail("capabilities inspected the development environment's engines")
        return version(name)

    monkeypatch.setattr("entanglix.runner.version", no_host_engine_metadata)
    capabilities = list_capabilities()
    assert capabilities["engines"]["qutip-cpu"]["version"] == result["metrics"]["qutip_version"]


def test_capabilities_do_not_create_an_environment(tmp_path):
    assert list_capabilities(repository=tmp_path)["engines"] == {}
    assert not list(tmp_path.iterdir())


def test_relative_outputs_land_in_artifacts(launch):
    execute, _, _, _ = launch
    job = execute(
        "import hashlib, os\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from entanglix.runner import publish_toml\n"
        "np.save('samples.npy', np.arange(3))\n"
        "publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', "
        "{'engine': 'numpy', 'metrics': {}, 'artifacts': {'samples': "
        "{'path': 'artifacts/samples.npy', 'sha256': "
        "hashlib.sha256(Path('samples.npy').read_bytes()).hexdigest()}}}, replace=True)\n"
    )
    assert job["exit_code"] == 0
    run = Path(job["path"])
    assert (run / "artifacts/samples.npy").is_file()
    assert not (run / "samples.npy").exists()
    assert not (run / "artifacts/result.toml").exists()
    assert read_run(run)["status"] == "completed"


@pytest.mark.parametrize("valid,code", [(True, 0), (False, 64)])
def test_lingering_child_finishes_before_result_validation(launch, valid, code):
    execute, _, _, _ = launch
    background = (
        "import os, time\n"
        "from pathlib import Path\n"
        "from entanglix.runner import publish_toml\n"
        "time.sleep(0.3)\n"
        f"publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', {{'engine': 'numpy', 'metrics': "
        f"{repr({'generation': 1} if valid else 1)}}}, replace=True)\n"
        "print('background finished', flush=True)\n"
    )
    job = execute(
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        "from entanglix.runner import publish_toml\n"
        "publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', {'engine': 'numpy', "
        "'metrics': {'generation': 0}}, replace=True)\n"
        f"subprocess.Popen([sys.executable, '-c', {background!r}])\n"
    )
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == code
    assert record["status"] == ("completed" if valid else "failed")
    assert "background finished" in record["log_tail"]
    assert "reconciliation" not in record
    assert record["finished"]["usage"]["wall_seconds"] >= 0.3
    if valid:
        assert record["result"]["metrics"]["generation"] == 1
    else:
        assert "script result must contain a metrics table" in record["log_tail"]


@pytest.mark.parametrize(
    "ending,code,status",
    [
        ("raise SystemExit(7)", 7, "failed"),
        ("raise SystemExit(130)", 130, "aborted"),
        ("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", 137, "failed"),
    ],
)
def test_job_failure_is_recorded_with_log(launch, ending, code, status):
    execute, _, _, runs = launch
    job = execute(f"print('diagnostic from job', flush=True)\n{ending}")
    assert job["exit_code"] == code
    record = read_run(Path(job["path"]))
    assert record["status"] == status
    assert record["finished"]["exit_code"] == code
    assert record["start"]["process_group"] > 0
    assert "diagnostic from job" in record["log_tail"]
    assert list_runs(runs)[0]["status"] == status


def test_spawn_failure_finishes_admitted_attempt(launch, monkeypatch):
    execute, _, _, runs = launch
    popen = subprocess.Popen

    def failing_spawn(*args, **kwargs):
        if kwargs.get("start_new_session"):
            raise OSError("job spawn failed")
        return popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", failing_spawn)
    job = execute("print('must not run')")
    assert job["exit_code"] == 64
    record = read_run(Path(job["path"]))
    assert record["status"] == "failed"
    assert "start" not in record
    assert record["finished"]["usage"]["wall_seconds"] == 0
    assert "job spawn failed" in record["log_tail"]
    assert len(list_runs(runs)) == 1


@pytest.mark.parametrize(
    "source,reason",
    [
        ("print('no result')", "result.toml"),
        (
            "import os\nfrom pathlib import Path\n"
            "(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml').write_text('invalid TOML')",
            "execution failed:",
        ),
        (
            "import os\nfrom pathlib import Path\n"
            "from entanglix.runner import publish_toml\n"
            "publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', "
            "{'engine': 'numpy', 'metrics': {}, 'verdict': 'VERIFIED'}, replace=True)",
            "cannot publish verification",
        ),
        (
            "import os\nfrom pathlib import Path\n"
            "from entanglix.runner import publish_toml\n"
            "Path('data.txt').write_text('changed')\n"
            "publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', {'engine': 'numpy', 'metrics': {}, "
            "'artifacts': {'data': {'path': 'artifacts/data.txt', 'sha256': 'wrong'}}}, replace=True)",
            "digest mismatch",
        ),
    ],
)
def test_invalid_result_preserves_failed_execution(launch, source, reason):
    execute, _, _, _ = launch
    job = execute(source)
    assert job["exit_code"] == 64
    record = read_run(Path(job["path"]))
    assert record["status"] == "failed"
    assert reason in record["log_tail"]


def test_active_inspection_is_read_only(launch):
    execute, _, _, runs = launch
    source = """import os, time
from pathlib import Path
from entanglix.runner import publish_toml
print('live trace', flush=True)
Path('ready').write_text('ready')
deadline = time.monotonic() + 30
while not Path('release').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('release timed out')
    time.sleep(0.01)
publish_toml(Path(os.environ['ENTANGLIX_RUN_DIR']) / 'result.toml', {'engine': 'numpy', 'metrics': {}}, replace=True)
"""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute, source)
        run = None
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                attempts = list_runs(runs)
                if attempts:
                    run = Path(attempts[0]["path"])
                    if (run / "artifacts/ready").exists():
                        break
                if future.done():
                    pytest.fail(f"job ended before inspection: {future.result()}")
                time.sleep(0.01)
            else:
                pytest.fail("job never reached its ready point")
            before = (run / "run.toml").read_bytes()
            inspected = read_run(run)
            assert inspected["status"] == "active"
            assert "live trace" in inspected["log_tail"]
            assert (run / "run.toml").read_bytes() == before
            assert not (run / "runner/reconciled.toml").exists()
        finally:
            if run is not None:
                (run / "artifacts/release").write_text("release")
        assert future.result(timeout=10)["exit_code"] == 0


def test_capabilities_and_usage_error_cli(launch, tmp_path):
    _, system, _, runs = launch
    capabilities = list_capabilities()
    assert "run_script" in capabilities["operations"]
    source = tmp_path / "script.txt"
    source.write_text("import unapproved_example_package")
    job = subprocess.run(
        ["entanglix", "run-script", str(source), "--system", str(system), "--runs-root", str(runs)],
        capture_output=True,
        text=True,
    )
    assert job.returncode == 64
    assert "allowlist" in job.stderr
    assert not runs.exists()
    listed = subprocess.run(
        ["entanglix", "list-runs", "--runs-root", str(runs)], capture_output=True, text=True
    )
    assert json.loads(listed.stdout) == []
