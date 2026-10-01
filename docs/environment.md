# GPU Environment

Initial compatibility was demonstrated on 2026-09-29 with Python 3.13.12 and both GPU dynamics stacks. Reference-size dynamics, drive gradients, storage selection, and numerical refinement passed on this workstation on 2026-09-30. The project locks the optional cuDensityMat and dynamiqs dependencies. Validation covers the bundled fixtures and conservative selection bounds; it makes no performance or arbitrary-system accuracy claim.

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

The [cuDensityMat](../tests/test_engine_cuquantum.py) and [dynamiqs](../tests/test_engine_dynamiqs.py) adapter suites passed on the RTX 5090. The [acceptance matrix](../tests/test_engine_acceptance.py) adds two-level, oscillator, and populated qubit–resonator evolution with constant and interval-valued controls, loss-disabled comparisons, independent Lindblad exponentials, and integration refinement. CPU QuTiP, CPU dynamiqs, and both CUDA adapters pass the same numerical limits. Separate oscillator/composite truncation refinements and scalar/piecewise nonzero drive gradients also pass. Complex coherences and seeded random complex initial states exercise state layout.

[Retained acceptance results](../tests/probes/m1_acceptance_results.json) contain exact commands, test counts, per-case residuals, package/device observations, lockfile digest, and source snapshots identified by each run. The largest state-reference residual is approximately `3.7e-8`, and the largest integration-refinement change is `3.4e-8`, below the `1e-7` limits. The lowest tested density-matrix eigenvalue is approximately `-7.1e-10`, above the `-1e-8` floor. [The numerical reference](operations/system-spec.md#numerical-checks-for-the-bundled-models) owns thresholds and rationale; [the selection policy](operations/system-spec.md#engine-selection-policy) owns size/known-array storage bounds and explicit fallback behavior. Storage accounting excludes solver workspace and autodiff storage.

After a test run, the [results collector](../tests/probes/collect_m1_acceptance.py) appends its JUnit counts and properties while retaining earlier runs and source snapshots. Run it before changing the tested sources: `python tests/probes/collect_m1_acceptance.py --junit new-run /tmp/report.xml 'exact test command'`. Generate the XML with `-o junit_family=legacy --junitxml=/tmp/report.xml` to retain numerical properties. Repeated `--junit` arguments collect multiple runs.

cuDensityMat storage tests compare evolved states with QuTiP below `1e-7` for diagonal, narrow-band, and dense operators, and compare factorized composites with the CPU and exact references. Dense is selected at 16 levels, Dia becomes `MultidiagonalOperator` at 17 levels (diagonal and banded), and a wide 20-level operator becomes `DenseOperator`. These checks establish functional selection on this stack; they do not measure a performance crossover.

Run the GPU adapter tests in separate processes after exact synchronization of both extras:

```sh
uv sync --locked --exact --extra cuquantum --extra differentiable-cuda12
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_engine_cuquantum.py tests/test_engine_acceptance.py -k 'not dynamiqs' -m gpu --require-gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_engine_dynamiqs.py tests/test_engine_acceptance.py -k dynamiqs -m gpu --require-gpu
```

For CPU acceptance, use `uv run --locked --exact pytest -q`. The CPU-only optional lane uses `JAX_PLATFORMS=cpu uv run --locked --exact --extra differentiable pytest -q tests/test_engine_dynamiqs.py tests/test_engine_acceptance.py -m 'not gpu' -k 'dynamiqs or gradient'`. These exact runs keep GPU extras out of the CPU environment. `--require-gpu` turns missing devices and skipped GPU tests into failures. The GPU suites assert the engine and actual device identity; this workstation reports an RTX 5090. [Loss comparisons](../examples/dynamics_reference.py) and [differentiable evolution](../examples/differentiable_reference.py) provide runnable examples.

## Carrier and frame acceptance

The [carrier/frame suite](../tests/test_carriers_frames.py) exercises continuous
cosines at nonzero initial time and phase, scalar and interval envelopes,
phase-sensitive rotating-wave evolution, lab-basis state reconstruction, and
factorized composites. All three engines pass the analytic and independent ODE
references and physicality limits described in the
[numerical interface](operations/system-spec.md#carriers-and-rotating-frames).
CPU and GPU dynamiqs also pass carrier-envelope gradient comparisons.
Retained GPU reports cover [cuDensityMat](../tests/probes/m2_carriers_cu.xml) and
[GPU dynamiqs](../tests/probes/m2_carriers_dq.xml) on the RTX 5090.

Run the stacks separately, using an exact environment with the appropriate extras:

```sh
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra differentiable pytest -q tests/test_carriers_frames.py -m 'not gpu'
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra cuquantum pytest -q tests/test_carriers_frames.py -k cudensitymat -m gpu --require-gpu -o junit_family=legacy --junitxml=tests/probes/m2_carriers_cu.xml
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra differentiable-cuda12 pytest -q tests/test_carriers_frames.py -k dynamiqs -m gpu --require-gpu -o junit_family=legacy --junitxml=tests/probes/m2_carriers_dq.xml
```

The [readout example](../examples/readout/README.md#independent-engine-frame-acceptance)
records the full-window CPU dynamiqs refinement, approximation bounds, resolution
limits, truncation warning, and measured reference costs. GPU scientific runner
dispatch and metric-specific protocol verification remain unavailable.

## Runner GPU environment check

On 2026-10-01 the [runner environment test](../tests/test_gpu_environment.py) passed on the RTX 5090 using a separate exact test environment with both GPU extras. It matched device UUID `GPU-f2a6e8aa-b32b-ae8e-6dc6-03612dd6b829`, observed driver `595.91.07`, loaded CUDA `12.9.0` and native cuDensityMat `0.7.0`, compared available memory against the existing selection estimate for the two-level fixture, and rejected an oversized request. [The retained JUnit report](../tests/probes/block3_gpu_environment.xml) contains the device/memory observations and environment hash. This validates pre-dispatch environment facilities; scientific GPU dispatch is still unavailable.

Reproduce this check in a separate exact environment without reshaping the development or CPU runner environment:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/entanglix-block3-gpu uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_gpu_environment.py -m gpu --require-gpu -o junit_family=legacy --junitxml=tests/probes/block3_gpu_environment.xml
```

The CPU cases in the same test file exercise missing dependencies, driver/library failures, invisible devices, memory rejection, and memory-independent environment identity. The [run reference](operations/runs.md#environment-and-code-provenance) defines the API and recorded fields.

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
