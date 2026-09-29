# GPU Environment

Initial compatibility was demonstrated on 2026-09-29 with Python 3.13.12 and both GPU dynamics stacks. The supported Entanglix release still provides the CPU foundation; these observations establish a starting environment for engine integration, not full engine conformance.

## Observed workstation and packages

| Component | Observed value |
|---|---|
| OS / architecture / kernel | Linux / x86_64 / `7.0.0-34-generic` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0 |
| GPU memory reported by NVIDIA-SMI | 32607 MiB |
| NVIDIA driver | 595.91.07 |
| Loaded CUDA runtime / driver API | 12.9 / 13.2 |
| Python | 3.13.12 |
| QuTiP / qutip-cuquantum | 5.3.1 / 0.3.1 |
| cuquantum-python-cu12 / cudensitymat-cu12 | 26.9.0 / 0.7.0 |
| cupy-cuda12x | 14.2.0 |
| dynamiqs | 0.3.6 |
| JAX / jaxlib / CUDA plugin / CUDA PJRT | 0.6.2 |
| NumPy / SciPy | 2.5.3 / 1.18.1 |

The driver API value and loaded runtime identify different components. All Python dependencies were installed in a disposable environment, using `qutip-cuquantum[cuda12]` and `jax[cuda12]`; the full observed package snapshot is [m1_gpu.requirements.txt](../tests/probes/m1_gpu.requirements.txt). Project dependency integration follows when the corresponding capabilities are implemented.

Dynamiqs 0.3.6 with JAX 0.11.2 and jaxtyping 0.3.11 failed at import with `TypeError: unsupported operand type(s) for |: 'types.UnionType' and 'str'` in its nested `QArrayLike` definition. JAX 0.6.2 resolved that failure and passed the GPU solve on the same Python version. This tested combination supports retaining Python 3.13.

## Compatibility result

The [probe](../tests/probes/m1_gpu.py) evolves an initially excited two-level system with `H = 2π × diag(0, 0.2) rad/ns` and `L = sqrt(0.01/ns) × |0><1|`. Eleven output times cover 0–50 ns. The excited-state population must agree with `exp(-0.01 × t)` within `1e-7`; the final trace error must be at most `1e-9`. Both engines use complex128 states, `rtol = 1e-9`, and `atol = 1e-11`.

| Engine / method | Maximum population error | Final trace error | Device evidence |
|---|---|---|---|
| cuDensityMat through qutip-cuquantum / CuVern7 | `9.7244e-13` | `0` | `CuState` backed by a CuPy array on CUDA device 0; RTX 5090 identity queried |
| dynamiqs / Tsit5 | `1.0738e-10` | `0` | Returned complex128 state arrays on `cuda:0`; RTX 5090 identity queried |

[The retained JSON result](../tests/probes/m1_gpu_results.json) records all installed package versions, device identities, times, populations, solver settings, residuals, and execution timestamp. The engines ran sequentially in separate processes. This case establishes basic installation and GPU execution; driven cases, gradients, model refinement, storage thresholds, and engine-selection limits require their own validation.

## Reproduce from a source checkout

Run where the NVIDIA driver and GPU device are accessible.

```sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv venv --python 3.13.12 /tmp/entanglix-m1-probe/.venv
uv pip sync --python /tmp/entanglix-m1-probe/.venv/bin/python \
  tests/probes/m1_gpu.requirements.txt
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MPLCONFIGDIR=/tmp/entanglix-m1-probe/matplotlib \
  JAX_PLATFORMS=cuda,cpu \
  /tmp/entanglix-m1-probe/.venv/bin/python tests/probes/m1_gpu.py \
  --output /tmp/entanglix-m1-probe/reproduced.json
```

Allow CPU devices for JAX host callbacks while explicitly checking GPU placement of the simulated states. `JAX_PLATFORMS=cuda` alone failed a dynamiqs callback in this probe. Disable JAX preallocation before initialization; each engine process releases its allocations when it exits. The Matplotlib cache uses a writable temporary directory.
