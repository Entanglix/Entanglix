# GPU Environment

Initial compatibility was demonstrated on 2026-09-29 with Python 3.13.12 and both GPU dynamics stacks. The project now locks optional cuDensityMat and dynamiqs dependencies and tests the adapters on this workstation. These observations establish reference-size integration, not full engine conformance or a performance claim.

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

The driver API value and loaded runtime identify different components. The initial compatibility probe used a disposable environment with `qutip-cuquantum[cuda12]` and `jax[cuda12]`; its full package snapshot is [m1_gpu.requirements.txt](../tests/probes/m1_gpu.requirements.txt). The project extras `cuquantum`, `differentiable-cuda12`, and CPU-only `differentiable` are now resolved in `uv.lock`. An exact sync prevents stray installed packages from influencing a run.

Dynamiqs 0.3.6 with JAX 0.11.2 and jaxtyping 0.3.11 failed at import with `TypeError: unsupported operand type(s) for |: 'types.UnionType' and 'str'` in its nested `QArrayLike` definition. JAX 0.6.2 resolved that failure and passed the GPU solve on the same Python version. This tested combination supports retaining Python 3.13.

## Compatibility result

The [probe](../tests/probes/m1_gpu.py) evolves an initially excited two-level system with `H = 2π × diag(0, 0.2) rad/ns` and `L = sqrt(0.01/ns) × |0><1|`. Eleven output times cover 0–50 ns. The excited-state population must agree with `exp(-0.01 × t)` within `1e-7`; the final trace error must be at most `1e-9`. Both engines use complex128 states, `rtol = 1e-9`, and `atol = 1e-11`.

| Engine / method | Maximum population error | Final trace error | Device evidence |
|---|---|---|---|
| cuDensityMat through qutip-cuquantum / CuVern7 | `9.7244e-13` | `0` | `CuState` backed by a CuPy array on CUDA device 0; RTX 5090 identity queried |
| dynamiqs / Tsit5 | `1.0738e-10` | `0` | Returned complex128 state arrays on `cuda:0`; RTX 5090 identity queried |

[The retained JSON result](../tests/probes/m1_gpu_results.json) records all installed package versions, device identities, times, populations, solver settings, residuals, and execution timestamp for the initial probe. The engines ran sequentially in separate processes.

The [cuDensityMat](../tests/test_engine_cuquantum.py) and [dynamiqs](../tests/test_engine_dynamiqs.py) adapter suites passed on the RTX 5090 on 2026-09-30. Small closed, dissipative, and piecewise-driven density matrices agree with QuTiP at a `1e-6` Frobenius limit; complex initial coherences exercise state layout. All three adapters also pass a `1e-7` exact-Liouvillian comparison on the qubit–resonator over 0–10 ns with 1 ns output spacing and a seeded random complex initial state ([reference construction](../tests/conftest.py), [CPU test](../tests/test_engine_interface.py), [numerical rationale](operations/system-spec.md#numerical-checks-for-the-bundled-models)). This uses the default `rtol = 1e-10`, `atol = 1e-12`, and `max_steps = 100_000`; the resonator-loss rate gives `γT = 0.1`. The full cross-engine oscillator and populated-resonator/loss-disabled fixture matrix, gradient verification, integration refinement, and dispatch limits still need validation.

cuDensityMat storage tests compare evolved states with QuTiP below `1e-7` for diagonal, narrow-band, and dense operators, and compare factorized composites with the CPU and exact references. Dense is selected at 16 levels, Dia becomes `MultidiagonalOperator` at 17 levels (diagonal and banded), and a wide 20-level operator becomes `DenseOperator`. These checks establish functional selection on this stack; they do not measure a performance crossover.

Run the GPU adapter tests in separate processes after exact synchronization of both extras:

```sh
uv sync --locked --exact --extra cuquantum --extra differentiable-cuda12
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_engine_cuquantum.py --require-gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_engine_dynamiqs.py -m gpu --require-gpu
```

For the CPU-only dynamiqs check, use `JAX_PLATFORMS=cpu uv run --locked --exact --extra differentiable pytest -q tests/test_engine_dynamiqs.py::test_dynamiqs_cpu_matches_reference`. `--require-gpu` turns missing devices and skipped GPU tests into failures. The GPU suites assert the engine and actual device identity; this workstation reports an RTX 5090.

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
