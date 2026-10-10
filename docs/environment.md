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

[Retained acceptance results](../tests/probes/m1_acceptance_results.json) contain exact commands, test counts, per-case residuals, package/device observations, lockfile digest, and source snapshots identified by each run. The largest state-reference residual is approximately `3.7e-8`, and the largest integration-refinement change is `3.4e-8`, below the `1e-7` limits. The lowest tested density-matrix eigenvalue is approximately `-7.1e-10`, above the `-1e-8` floor. [The numerical reference](operations/system-spec.md#numerical-checks-for-the-bundled-models) owns thresholds and rationale; [the selection policy](operations/system-spec.md#engine-selection-policy) owns declared-needs dispatch and explicit fallback behavior. No harness dimension cap or pre-launch memory estimate is imposed.

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
limits, truncation warning, and measured reference costs. These fixture measurements do not replace checks requested on a new run.

## Runner GPU environment check

The [runner environment test](../tests/test_gpu_environment.py) passes on the RTX 5090 in a separate exact test environment with both GPU extras. It matches device UUID `GPU-f2a6e8aa-b32b-ae8e-6dc6-03612dd6b829` and observes driver `595.91.07`, loaded CUDA `12.9.0` and native cuDensityMat `0.7.0`, with no memory estimate. The [GPU report](../tests/probes/m2_simulation_gpu.xml) from the simulation acceptance below retains the device observations and environment hash. This records pre-dispatch environment compatibility.

Run this check alone in a separate exact environment without reshaping the development or CPU runner environment:

```sh
UV_PROJECT_ENVIRONMENT=/tmp/entanglix-block3-gpu uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_gpu_environment.py -m gpu --require-gpu
```

The CPU cases in the same test file exercise missing dependencies, driver/library failures, invisible devices, and environment identity independent of memory availability. The [run reference](operations/runs.md#environment-and-code-provenance) defines the API and recorded fields.

## Simulation acceptance

The [simulation integration tests](../tests/test_simulate.py) cover CLI spectra,
CPU dynamics, trusted endpoint metrics, segment tolerances, frozen-input replay,
and rejected inputs. The [GPU report](../tests/probes/m2_simulation_gpu.xml) covers
cuDensityMat dispatch, requested post-run checks, timeout/cancellation, runner loss
with a surviving job, exclusion until reconciliation, and linked replay. Core CPU
tests need no CUDA dependencies. The [same-engine regression cases](../tests/test_verification_cost.py)
exercise insufficient truncation, strong drive, two-photon dynamics, a delayed pulse,
and false alarms. The [allocation/signal cases](../tests/test_memory_failures.py)
use a small process address-space limit and deliberate signal death.

The [Purcell acceptance probe](../tests/probes/m2_simulation.py) runs the
[CPU and CUDA protocols](../examples/readout/README.md), requests integrity, local
segments and diagnostics, and replays frozen inputs with a `1e-7` output residual
limit. No check follows computation automatically. The CPU study also runs a 6 × 8,
dimension-48 lab-frame simulation and requests integrity. Its
[CPU summary](../tests/probes/m2_simulation_cpu.json) and
[CUDA summary](../tests/probes/m2_simulation_cuda.json) retain IDs, provenance,
metrics, usage, reports, local coverage and work time. Full attempts remain under
`research_runs/` on the workstation; arrays and plots are not shipped. Passing
segments establishes local convergence, not whole-window or RWA accuracy.

The [larger-engine test](../tests/test_engine_acceptance.py) compares an 8 × 8,
dimension-64 complex-state dissipative evolution across QuTiP, dynamiqs and
cuDensityMat. The [CPU dynamiqs report](../tests/probes/m2_checks_cpu.xml) and
[GPU dynamiqs report](../tests/probes/m2_checks_dynamiqs.xml), together with the
cuDensityMat GPU report above, retain agreement properties. Tested sizes are
acceptance evidence, not a ceiling.

```sh
export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact python tests/probes/m2_simulation.py --device cpu --output /tmp/m2_simulation_cpu.json
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact python tests/probes/m2_simulation.py --device cuda --output /tmp/m2_simulation_cuda.json
UV_PROJECT_ENVIRONMENT=/tmp/entanglix-simulation-tests XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_simulate.py tests/test_gpu_environment.py tests/test_engine_acceptance.py tests/test_memory_failures.py -k 'not dynamiqs' -m gpu --require-gpu -o junit_family=legacy --junitxml=/tmp/m2_simulation_gpu.xml
UV_PROJECT_ENVIRONMENT=/tmp/entanglix-simulation-tests JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_engine_acceptance.py -k 'dimension_64 and dynamiqs' -m gpu --require-gpu -o junit_family=legacy --junitxml=/tmp/m2_checks_dynamiqs.xml
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra differentiable pytest -q tests/test_engine_dynamiqs.py tests/test_engine_acceptance.py tests/test_carriers_frames.py tests/test_corrector.py tests/test_plans.py tests/test_verification.py tests/test_simulate.py tests/test_sweep.py tests/test_verification_cost.py -m 'not gpu' -k 'dynamiqs or gradient' -o junit_family=legacy --junitxml=/tmp/m2_checks_cpu.xml
```

With CUDA extras installed and GPU visibility disabled for CPU checks, JAX plugin
discovery may log `CUDA_ERROR_NO_DEVICE` before selecting CPU. Recorded outcomes
and devices identify whether execution succeeded.

Keep the code, docs and runner profile unchanged through the acceptance studies.
Write reports outside the checkout and copy them into `tests/probes/` after all
studies finish. Recorded provenance describes the tree before replacing the reports.

## Sweep acceptance

The [sweep tests](../tests/test_sweep.py) cover static/dynamic grids, driven lab and
rotating frames, complex coherence, paired axis settings, frozen replay, invalid
inputs, bounded warning selection, and partial results after cancellation or runner
loss. Grids whose corners exceed the check-selection bound still run. The
[GPU integration report](../tests/probes/m2_sweep_gpu.xml) covers cuDensityMat,
post-run CPU checks, GPU usage, and full-grid replay.

The [readout acceptance probe](../tests/probes/m2_sweep.py) runs the 3 × 3 grid on
CPU and CUDA, requests integrity, local segments, diagnostics and the declared
amplitude expectation, and reproduces every point's arrays on replay within `1e-7`.
The [CPU summary](../tests/probes/m2_sweep_cpu.json) also retains the static `χ`
companion and a narrower linked CPU grid. The [CUDA summary](../tests/probes/m2_sweep_cuda.json)
retains GPU execution and usage without a speedup claim. Reports identify the
outcomes, maximum residuals, selected and uncovered points, local windows and warnings;
full per-point reports remain in the linked verification attempts. Passing sampled
checks says nothing about convergence at unsampled points or RWA accuracy.

```sh
export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12
MPLCONFIGDIR=/tmp/entanglix-m2-matplotlib uv run --locked --exact python tests/probes/m2_sweep.py --device cpu --output /tmp/m2_sweep_cpu.json
MPLCONFIGDIR=/tmp/entanglix-m2-matplotlib uv run --locked --exact python tests/probes/m2_sweep.py --device cuda --output /tmp/m2_sweep_cuda.json
UV_PROJECT_ENVIRONMENT=/tmp/entanglix-simulation-tests XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_sweep.py -m gpu --require-gpu -o tmp_path_retention_policy=failed --junitxml=/tmp/m2_sweep_gpu.xml
```

Keep the tree and runner profile fixed until all studies finish, then copy summaries
and reports into `tests/probes/`. Full attempts and arrays stay in `research_runs/`.

## Threshold characterization probe

The [threshold probe](../tests/probes/m3_readout.py) runs the readout example's
[threshold study](../examples/readout/README.md#threshold-characterization) on CPU
with the existing `sweep`, `simulate`, and `check` operations: the seed grid,
rotating-frame bisection, lab-frame and refined lab-frame confirmation, and the
requested checks. Its [results](../tests/probes/m3_readout_results.json) retain every
run's identity, metrics, job time, and check outcomes, with each bracket and its
lab-frame confirmation. The retained run used the shared runner profile and took
about 329 s.

```sh
export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12
MPLCONFIGDIR=/tmp/entanglix-m3-matplotlib uv run --locked --exact python tests/probes/m3_readout.py --output /tmp/m3_readout_results.json
```

Full attempts and arrays stay in `research_runs/m3-readout/`.

## Reproduce GPU compatibility from a source checkout

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

## Investigation admission acceptance

The [admission tests](../tests/test_investigation_admission.py) cover scope and
linked checks/reruns, concurrent job-count/execution-time commitments, retained
uncertain usage, jobs without timeouts ending at their allowance, and deadlines
before start and during execution. The workstation
[GPU report](../tests/probes/m3_admission_gpu.xml) records one linked CUDA job's
actual GPU-seconds, execution time counted toward `job_seconds`, and refusal of a
second job whose commitment would exceed the
remaining allowance. Use an isolated test environment with the workstation extras:

```sh
uv run --locked --exact pytest -q tests/test_investigations.py tests/test_investigation_admission.py tests/test_investigation_lifecycle.py -m 'not gpu'
UV_PROJECT_ENVIRONMENT=/var/tmp/entanglix-6a2-gpu-test XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_investigation_admission.py -m gpu --require-gpu -o junit_family=legacy --junitxml=tests/probes/m3_admission_gpu.xml
```

Tests that launch jobs build runner environments under the temporary directory. On
this workstation `/tmp` is a quota-limited tmpfs: one session of the runner tests
took about 15 GB there and failed with `Disk quota exceeded` when less was free, and
took under 0.5 GB with `TMPDIR=/var/tmp`. Set `TMPDIR` to a disk-backed directory
when `/tmp` is a tmpfs.

## Scripted investigation acceptance

The [operator acceptance driver](../tests/probes/m3_investigation.py) drafts and
starts each declared CPU fixture, then runs the
[readout client](../examples/readout/README.md#scripted-investigation-baseline).
An existing workspace continues its investigation; choose fresh workspace and
runs-root paths for a new measurement. Full inputs, arrays, check reports and
findings stay in `research_runs/`; summaries retain source identities and costs.
These commands use this workstation's existing runner extras, without changing
its shared environment profile:

```sh
export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact python tests/probes/m3_investigation.py --case normal --workspace research_runs/m3-threshold-normal-client --runs-root research_runs/m3-threshold-normal --output tests/probes/m3_investigation_normal.json
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact python tests/probes/m3_investigation.py --case early-stop --workspace research_runs/m3-threshold-early-stop-client --runs-root research_runs/m3-threshold-early-stop --output tests/probes/m3_investigation_early_stop.json
TMPDIR=/var/tmp uv run --locked --exact pytest -q tests/test_investigations.py tests/test_investigation_admission.py tests/test_investigation_lifecycle.py tests/test_readout_investigation.py -m 'not gpu'
```

## Busy-device admission acceptance

The [device admission tests](../tests/test_gpu_admission.py) and
[workstation GPU report](../tests/probes/m3_busy_gpu.xml) retain ownership and
launch evidence. Regenerate them with an isolated test environment:

```sh
uv run --locked --exact pytest -q tests/test_gpu_admission.py tests/test_lifecycle.py -m 'not gpu'
TMPDIR=/var/tmp UV_PROJECT_ENVIRONMENT=/var/tmp/entanglix-6b0-gpu-test XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_simulate.py tests/test_sweep.py tests/test_investigation_admission.py -m gpu --require-gpu -o junit_family=legacy --junitxml=tests/probes/m3_busy_gpu.xml
```

## Detached supervision

Shared-operation tests cover admission recovery, ownership handoff, requesting-process loss, concurrent reads and cancellation, linked checks/replays, and finite waits. The workstation report retains CUDA deadline and continuous-ownership cases before computation and while the adapter is executing.

Use disk-backed temporary storage for these commands: the full suite creates isolated job environments, which can exceed this workstation's `/tmp` quota.

```sh
TMPDIR=/var/tmp uv run --locked --exact pytest -m "not gpu" -q
TMPDIR=/var/tmp uv run --locked --exact pytest tests/test_detached_launches.py -m "not gpu" -q
TMPDIR=/var/tmp UV_PROJECT_ENVIRONMENT=/var/tmp/entanglix-6b-supervision-tests XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra cuquantum --extra differentiable-cuda12 pytest tests/test_detached_launches.py -m gpu --require-gpu -q -o junit_family=legacy --junitxml=tests/probes/m3_supervision_gpu.xml
```

The GPU command uses its own test environment. Each fixture also has a separate, exact runner environment; neither changes the checkout's runner extras.

## Stdio MCP acceptance

The server uses its own environment; jobs keep the runner environment. The
[transport reference](operations/mcp.md) describes the command and explicit settings.
Run the scripted-client tests with the SDK extra, then the workstation CUDA cases:

```sh
TMPDIR=/var/tmp UV_PROJECT_ENVIRONMENT=/var/tmp/entanglix-mcp-test uv run --locked --exact --extra mcp pytest -q tests/test_mcp.py -m 'not gpu'
TMPDIR=/var/tmp UV_PROJECT_ENVIRONMENT=/var/tmp/entanglix-6b-mcp-gpu XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra mcp --extra cuquantum --extra differentiable-cuda12 pytest -q tests/test_mcp.py -m gpu --require-gpu -o junit_family=legacy --junitxml=tests/probes/m3_mcp_gpu.xml
```

The [GPU report](../tests/probes/m3_mcp_gpu.xml) retains disconnection/deadline
ownership evidence and CUDA launch-to-manifest timings. The optional SDK changes
the lockfile digest in newly captured environments without changing the numerical
package pins.

## Research guidance conformance

The [workspace reference](operations/mcp.md#research-workspace) describes conduct,
plain-text skills, command settings, and the operator decision path. Scripted
initialization tests check that the real server sends the shipped conduct and
records supplied versions and operator-declared restriction mode:

```sh
TMPDIR=/var/tmp UV_PROJECT_ENVIRONMENT=/var/tmp/entanglix-mcp-test uv run --locked --exact --extra mcp pytest -q tests/test_research_guidance.py tests/test_mcp.py -m 'not gpu'
```

The [runtime driver](../tests/probes/m3_guidance.py) uses fresh sessions outside the
checkout and retains raw transcripts under its workspace. It exercises initial
conduct, native skill metadata and plain-procedure reading, configured bare-command
refusals, and session/server CUDA separation using one tiny simulation. Use an
available logged-in Codex account and choose a fresh workspace path:

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.venv-mcp" uv sync --locked --exact --extra mcp
TMPDIR=/var/tmp uv run --locked --exact python tests/probes/m3_guidance.py --workspace "$HOME/Projects/entanglix-research/6b2-guidance-new" --server-python "$PWD/.venv-mcp/bin/python" --device GPU-f2a6e8aa-b32b-ae8e-6dc6-03612dd6b829 --runner-extras cuquantum differentiable-cuda12 --model gpt-6.1-sol --effort low --output tests/probes/m3_guidance_results.json
```

The retained [conformance evidence](../tests/probes/m3_guidance_results.json) shows
conduct in initial context, all four native skill descriptions and plain-text
procedures, with no repository development instructions. The real MCP server
validated the copied model. Runtime tool replies show both bare human-action
commands refused by the configured rule, while `entanglix --version` ran. The
session's CUDA probe found zero devices; a server-launched cuDensityMat job
completed on the configured RTX 5090. All acceptance criteria passed.

These measurements cover
one non-interactive Codex 0.160.0 configuration, not all runtimes or interactive use.

## Agent investigation evaluation

The [evaluation driver](../tests/probes/m3_evaluation.py) runs fresh scripted
baselines, a broad-request formulation session, and three normal plus three
early-stop agent investigations. It retains full inputs, runs, audits and runtime
transcripts outside the checkout; the compact
[comparison](../tests/probes/m3_evaluation_results.json) includes unsuccessful
sessions. The [readout example](../examples/readout/README.md#agent-investigation-comparison)
states the measured outcomes and limits.

Use an available logged-in Codex account, choose the model and effort explicitly,
and supply a fresh persistent workspace. The driver consumes real runtime usage;
acceptance requires successful formulation and at least two accepted sessions per
case. Any other session must be a reported runtime failure before admitting a job;
a failed session that admitted jobs fails the evaluation. Match `--runner-extras`
to the checkout's existing runner profile:

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.venv-mcp" uv sync --locked --exact --extra mcp
TMPDIR=/var/tmp uv run --locked --exact python tests/probes/m3_evaluation.py --workspace "$HOME/Projects/entanglix-research/6b3-evaluation-new" --server-python "$PWD/.venv-mcp/bin/python" --runner-extras cuquantum differentiable-cuda12 --model YOUR_MODEL --effort YOUR_EFFORT --output tests/probes/m3_evaluation_results.json
TMPDIR=/var/tmp UV_PROJECT_ENVIRONMENT="$PWD/.venv-mcp" uv run --locked --exact --extra mcp pytest -q tests/test_agent_evaluation.py tests/test_research_guidance.py
```

To rebuild only the summary from an existing workspace, without launching jobs or
runtime sessions:

```sh
uv run --locked --exact python tests/probes/m3_evaluation.py --report-only --workspace "$HOME/Projects/entanglix-research/6b3-evaluation-20261006" --output tests/probes/m3_evaluation_results.json
```

The report records separate fingerprints for the driver used to execute the
sessions and the current summary builder. The executed driver is retained under
the workspace's `operator/` directory. Reporting fixes do not replace scientific
outputs or hide failed sessions.

The [pick-up summary probe](../tests/probes/m3_pickup.py) rebuilds the separately
retained 2026-10-09 session's [result](../tests/probes/m3_pickup_results.json). It
reads the interrupted-client snapshot, completed runs, MCP launch audit and runtime
transcript, and compares its rebuilt measurements against the operator's original
report. This command launches no jobs or runtime sessions:

```sh
uv run --locked --exact python tests/probes/m3_pickup.py --workspace "$HOME/Projects/entanglix-research/6b3-pickup-20261009" --output tests/probes/m3_pickup_results.json
```

The [Claude Code pick-up probe](../tests/probes/m3_pickup_claude.py) rebuilds the
three retained 2026-10-10 sessions' [results](../tests/probes/m3_pickup_claude_results.json)
from each workspace's runs, MCP launch audit, and session output. Each workspace's
`operator/` directory holds the scripts and prompt that produced it. This command
launches no jobs or runtime sessions:

```sh
uv run --locked --exact python tests/probes/m3_pickup_claude.py --workspace "$HOME/Projects/entanglix-research/6b4-pickup-claude-20261010" "$HOME/Projects/entanglix-research/6b4-pickup-claude-20261010-b" "$HOME/Projects/entanglix-research/6b4-pickup-claude-20261010-c" --output tests/probes/m3_pickup_claude_results.json
```

## MCP response summaries

The [scripted stdio tests](../tests/test_mcp.py) and [view tests](../tests/test_mcp_views.py)
exercise investigation status, paged attempt and event tables, single events,
run detail, finding acknowledgments, deterministic reports and retained audit
links. No agent-runtime session is needed:

```sh
TMPDIR=/var/tmp UV_PROJECT_ENVIRONMENT="$PWD/.venv-mcp" uv run --locked --exact --extra mcp pytest -q tests/test_mcp.py tests/test_mcp_views.py tests/test_research_guidance.py tests/test_agent_evaluation.py -m "not gpu"
```

Read-only measurements on the second retained [Claude Code pick-up investigation](../tests/probes/m3_pickup_claude_results.json),
`i20261010T165839906785Z-85f1d3ed`, cover 42 attempts and 8 events in
`6b4-pickup-claude-20261010-b/normal/runs`. JSON views use the compact representation
sent by MCP; the report is Markdown:

| View | Characters |
|---|---:|
| Status | 11,035 |
| Attempt table (42 rows) | 9,420 |
| Event index (8 rows) | 867 |
| `list_runs` (42 rows) | 4,589 |
| Report | 33,225 |

These are character counts, not measured token savings. See the
[MCP reference](operations/mcp.md#investigation-views-and-tables).

## Investigation usability walk-through

On 2026-10-10, at commit `18ff984`, the example investigation was started, read
and stopped by following only the
[scripted investigation baseline](../examples/readout/README.md#scripted-investigation-baseline)
and the [investigation reference](operations/investigations.md), with a fresh
workspace and runs root and no agent runtime:

| Step | Command | Observed |
|---|---|---|
| Draft | `investigate.py draft` | State `draft` |
| List and read | `list-investigations`, `read-investigation` | The draft and its terms |
| Start | `investigation-decision ... start --revision 1` | State `active` |
| Run and read | `investigate.py run`, `read-investigation` | 8 jobs admitted, 8 finished |
| Stop | `stop-investigation ... user_cancellation` | State `stopped`; the client's next job was refused and no process remained |
| Read and report | `read-investigation`, `report-investigation` | 8 completed attempts retained; the report rendered |
| Resume | `investigation-decision ... resume` | State `active` |

This was a manual check; its scratch records were not retained. The stop fell
between jobs, so it did not cancel a running solve; the
[session tests](../tests/test_readout_investigation.py) cover that. On this
workstation the run command needs the pinned runner extras,
`export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12`.
