"""M1 compatibility check: a two-level system with analytic amplitude damping."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")


def solve(engine):
    import numpy as np

    times = np.linspace(0.0, 50.0, 11)
    gamma = 0.01
    expected = np.exp(-gamma * times)
    population = np.diag([0.0, 1.0]).astype(np.complex128)
    hamiltonian = 2 * np.pi * 0.2 * population
    collapse = np.sqrt(gamma) * np.array([[0, 1], [0, 0]], dtype=np.complex128)
    if engine == "cuquantum":
        import cupy as cp
        import qutip as qt
        import qutip_cuquantum as qcu
        from cuquantum.densitymat import WorkStream

        device = cp.cuda.Device(0)
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        with qcu.CuQuantumBackend(WorkStream(device_id=device.id)):
            h = qt.Qobj(hamiltonian, dtype="Dense").to("CuOperator")
            collapse_op = qt.Qobj(collapse, dtype="Dense").to("CuOperator")
            rho0 = qt.Qobj(population, dtype="Dense").to("CuState")
            observable = qt.Qobj(population, dtype="Dense").to("CuOperator")
            result = qt.mesolve(
                h,
                rho0,
                times,
                c_ops=[collapse_op],
                e_ops=[observable],
                options={
                    "method": "CuVern7",
                    "atol": 1e-11,
                    "rtol": 1e-9,
                    "store_states": True,
                    "progress_bar": "",
                },
            )
            assert isinstance(result.states[-1].data, qcu.CuState)
            storage = result.states[-1].data.to_cupy()
            assert isinstance(storage, cp.ndarray) and storage.device.id == device.id
            assert str(storage.dtype) == "complex128"
            cp.cuda.get_current_stream().synchronize()
            measured = np.real(cp.asnumpy(cp.asarray(result.expect[0])))
            final_state = cp.asnumpy(storage).reshape((2, 2), order="F")
            identity = {
                "gpu": properties["name"].decode(),
                "device_id": device.id,
                "compute_capability": f"{properties['major']}.{properties['minor']}",
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver_api": cp.cuda.runtime.driverGetVersion(),
                "state_storage": type(result.states[-1].data).__name__,
                "state_dtype": str(storage.dtype),
                "array_device": str(storage.device),
                "method": "CuVern7",
            }
    else:
        import jax

        jax.config.update("jax_enable_x64", True)
        import dynamiqs as dq

        dq.set_precision("double")
        gpu = jax.devices("gpu")[0]
        with jax.default_device(gpu):
            result = dq.mesolve(
                dq.asqarray(hamiltonian),
                [dq.asqarray(collapse)],
                dq.asqarray(population),
                times,
                exp_ops=[dq.asqarray(population)],
                method=dq.method.Tsit5(atol=1e-11, rtol=1e-9),
                progress_meter=False,
            )
            states = result.states.to_jax().block_until_ready()
            assert all(device.platform == "gpu" for device in states.devices())
            assert str(states.dtype) == "complex128"
            measured = np.real(np.asarray(result.expects[0]))
            final_state = np.asarray(states[-1])
            identity = {
                "gpu": gpu.device_kind,
                "device": str(gpu),
                "platform": gpu.platform,
                "state_dtype": str(states.dtype),
                "array_devices": [str(d) for d in states.devices()],
                "method": "Tsit5",
            }
    residual = float(np.max(np.abs(measured - expected)))
    trace_residual = float(abs(np.trace(final_state) - 1))
    assert residual <= 1e-7, residual
    assert trace_residual <= 1e-9, trace_residual
    return {
        "engine": engine,
        "device": identity,
        "times_ns": times.tolist(),
        "rate_per_ns": gamma,
        "energy_gap_GHz": 0.2,
        "solver_atol": 1e-11,
        "solver_rtol": 1e-9,
        "population_error_limit": 1e-7,
        "trace_error_limit": 1e-9,
        "populations": measured.tolist(),
        "analytic_populations": expected.tolist(),
        "max_population_error": residual,
        "final_trace_error": trace_residual,
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["cuquantum", "dynamiqs"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.engine:
        result = solve(args.engine)
    else:
        results = []
        for engine in ("cuquantum", "dynamiqs"):
            child = subprocess.run(
                [sys.executable, __file__, "--engine", engine],
                text=True,
                capture_output=True,
                timeout=180,
            )
            if child.returncode:
                raise RuntimeError(f"{engine} exited {child.returncode}: {child.stderr}")
            if child.stderr:
                print(child.stderr, file=sys.stderr, end="")
            results.append(json.loads(child.stdout))
        result = {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "gpu_query": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
            "python": platform.python_version(),
            "system": platform.system(),
            "architecture": platform.machine(),
            "kernel": platform.release(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "environment": {
                key: os.environ.get(key)
                for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_PLATFORMS", "MPLCONFIGDIR")
            },
            "results": results,
        }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
