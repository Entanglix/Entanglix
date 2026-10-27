"""GPU pre-dispatch observations and rejection diagnostics; CPU imports stay optional."""

import json
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from entanglix.runner import capture_environment, check_gpu_environment

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "GPU-f2a6e8aa-b32b-ae8e-6dc6-03612dd6b829"


@pytest.fixture
def gpu_runtime(monkeypatch):
    runtime = SimpleNamespace(
        getDeviceCount=lambda: 1,
        getDeviceProperties=lambda _: {"uuid": uuid.UUID(DEVICE[4:]).bytes},
        runtimeGetVersion=lambda: 12090,
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(cuda=SimpleNamespace(runtime=runtime, Device=nullcontext)),
    )
    monkeypatch.setitem(sys.modules, "cuquantum", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "cuquantum.densitymat", SimpleNamespace(WorkStream=lambda **_: object())
    )
    monkeypatch.setattr(
        "ctypes.CDLL", lambda _: SimpleNamespace(cudensitymatGetVersion=lambda: 700)
    )

    def query(command, **_):
        if command[1:3] == ["pip", "check"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert f"--id={DEVICE}" in command
        return SimpleNamespace(stdout=f"{DEVICE}, 595.91.07\n")

    monkeypatch.setattr("entanglix.runner.subprocess.run", query)
    return runtime


def test_gpu_observations_feed_identity(gpu_runtime):
    observed = check_gpu_environment(DEVICE, engine="cudensitymat")
    assert observed["device_uuid"] == DEVICE
    assert observed["gpu"] == {"driver": "595.91.07", "cuda": "12.9.0", "cuquantum": "0.7.0"}
    identity = capture_environment(ROOT / "uv.lock", extras=("cuquantum",), gpu=observed["gpu"])
    assert "free_memory_bytes" not in observed
    assert identity["gpu"] == observed["gpu"]


@pytest.mark.parametrize(
    "change,message",
    [
        ({"getDeviceCount": lambda: 0}, "not visible to CUDA"),
    ],
)
def test_gpu_gate_rejections_are_actionable(gpu_runtime, change, message):
    for key, value in change.items():
        setattr(gpu_runtime, key, value)
    with pytest.raises(ValueError, match=message) as error:
        check_gpu_environment(DEVICE, engine="cudensitymat")
    assert "GPU environment rejected" in str(error.value)


def test_driver_and_native_library_failures_reject(gpu_runtime, monkeypatch):
    def fail(*_, **__):
        raise OSError("driver or library unavailable")

    for boundary in ("entanglix.runner.subprocess.run", "ctypes.CDLL"):
        with monkeypatch.context() as patch:
            patch.setattr(boundary, fail)
            with pytest.raises(ValueError, match="driver or library unavailable"):
                check_gpu_environment(DEVICE, engine="cudensitymat")


def test_missing_gpu_extra_rejects_without_loading_gpu_stack(gpu_runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ValueError, match="cuquantum extra"):
        check_gpu_environment(DEVICE, engine="cudensitymat")


def test_other_cuda_engines_pass_shared_device_checks_without_cuquantum(gpu_runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(sys.modules, "cuquantum.densitymat", None)
    observed = check_gpu_environment(DEVICE, engine="mine")
    assert observed == {
        "device_uuid": DEVICE,
        "gpu": {"driver": "595.91.07", "cuda": "unobserved", "cuquantum": "unobserved"},
    }

    def fail(*_, **__):
        raise OSError("driver unavailable")

    monkeypatch.setattr("entanglix.runner.subprocess.run", fail)
    with pytest.raises(ValueError, match="driver unavailable") as error:
        check_gpu_environment(DEVICE, engine="mine")
    assert "cuquantum extra" not in str(error.value)


@pytest.mark.gpu
def test_workstation_gpu_environment(record_property):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("cuquantum.densitymat")
    import numpy as np

    from entanglix.engines.selection import select_engine
    from entanglix.evolution import EvolutionInput
    from entanglix.models import load_system

    properties = cp.cuda.runtime.getDeviceProperties(0)
    device = f"GPU-{uuid.UUID(bytes=properties['uuid'])}"
    model = load_system(ROOT / "examples/two-level/system.toml")
    selection = select_engine(
        model,
        EvolutionInput(np.array([0, 1]), np.array([0, 1])),
        available_engines={"cudensitymat"},
        device="cuda",
    )
    observed = check_gpu_environment(device, engine=selection.engine)
    environment = capture_environment(
        ROOT / "uv.lock", extras=("cuquantum", "differentiable-cuda12"), gpu=observed["gpu"]
    )
    assert selection.engine == "cudensitymat"
    record_property("gpu_environment", json.dumps(observed, sort_keys=True))
    record_property("env_hash", environment["env_hash"])
    record_property("device_name", properties["name"].decode())
