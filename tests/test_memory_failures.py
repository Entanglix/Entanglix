"""Allocation errors and uncertain signal deaths retain distinct diagnostics."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_simulate import protocol_files

from entanglix.protocols.simulate import simulate
from entanglix.runner import failure_message, process_failure, read_run, run_script

ROOT = Path(__file__).resolve().parents[1]
ALLOCATION_FAILURE = """import resource
    from pathlib import Path as AllocPath
    import os as alloc_os
    current = int(AllocPath('/proc/self/statm').read_text().split()[0]) * alloc_os.sysconf('SC_PAGE_SIZE')
    ceiling = current + 16 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
    bytearray(64 * 1024 * 1024)
    """


@pytest.mark.parametrize("phase", ["gate", "job"])
def test_allocation_failure_identified_at_gate_or_in_retained_job(
    tmp_path, runner_repository, phase
):
    path, _ = protocol_files(tmp_path, kind="spectrum")
    if phase == "gate":
        worker = runner_repository / "src/entanglix/protocols/simulate.py"
        worker.write_text(
            worker.read_text().replace(
                "    system = load_system(source)",
                "    " + ALLOCATION_FAILURE + "system = load_system(source)",
                1,
            )
        )
        with pytest.raises(ValueError, match="out of memory: MemoryError") as error:
            simulate(path, repository=runner_repository, runs_root=tmp_path / "runs")
        diagnostic = failure_message(error.value)
        assert diagnostic.count("out of memory:") == 1
        assert "ValueError:" not in diagnostic
        assert not (tmp_path / "runs").exists()
    else:
        worker = runner_repository / "src/entanglix/protocols/_plan_job.py"
        worker.write_text(
            worker.read_text().replace(
                '    if loaded.definition.kind == "spectrum":',
                "    " + ALLOCATION_FAILURE + 'if loaded.definition.kind == "spectrum":',
                1,
            )
        )
        job = simulate(path, repository=runner_repository, runs_root=tmp_path / "runs")
        record = read_run(Path(job["path"]))
        assert job["exit_code"] == 64 and record["status"] == "failed"
        assert "out of memory: MemoryError" in record["log_tail"]
        assert "result" not in record


def test_unexplained_signal_death_is_not_identified_as_oom(tmp_path, runner_repository):
    script = tmp_path / "kill.py"
    script.write_text("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
    job = run_script(
        ROOT / "examples/two-level/system.toml",
        script,
        repository=runner_repository,
        runs_root=tmp_path / "runs",
    )
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 137 and record["status"] == "failed"
    assert "SIGKILL encoding); cause unknown" in record["log_tail"]
    assert "exhausted memory is a possible cause" in record["log_tail"]
    assert "out of memory:" not in record["log_tail"]


def test_authored_allocation_failure_is_identified(tmp_path, runner_repository):
    script = tmp_path / "allocate.py"
    # Use the same small address-space limit without global host memory pressure.
    script.write_text(ALLOCATION_FAILURE.replace("\n    ", "\n"))
    job = run_script(
        ROOT / "examples/two-level/system.toml",
        script,
        repository=runner_repository,
        runs_root=tmp_path / "runs",
    )
    record = read_run(Path(job["path"]))
    assert job["exit_code"] == 1 and record["status"] == "failed"
    assert "out of memory: MemoryError" in record["log_tail"]


def test_cupy_allocation_exception_names_oom():
    cp = pytest.importorskip("cupy")
    exception = cp.cuda.memory.OutOfMemoryError(1024, 0, 0)
    assert failure_message(exception).startswith("out of memory:")


@pytest.mark.parametrize("code", [-9, 137, -15, 143])
def test_launch_gate_signal_diagnostic_preserves_uncertainty(code):
    message = process_failure(
        SimpleNamespace(stderr="gate diagnostic", returncode=code), "launch rejected"
    )
    assert "gate diagnostic" in message
    assert "launch rejected" in message
    assert "cause unknown" in message
    assert ("SIGKILL" if code in {-9, 137} else "SIGTERM") in message
    assert ("encoding" in message) == (code > 0)
    assert ("exhausted memory is a possible cause" in message) == (code in {-9, 137})
    assert "out of memory:" not in message
