# Simulation protocols

`entanglix simulate simulate.toml` runs one declared computation and retains its
inputs, arrays, observable plot, summaries, and execution outcome. No check runs
automatically. `--runs-root PATH` chooses the history root; `--parent-run ID` links
an earlier attempt. Inspect the returned path with `read-run`.

The [Purcell example](../../examples/readout/purcell.toml) evolves an excited qubit
with an empty lossy resonator for 1600 ns. It uses the
[system conventions](system-spec.md) and [run lifecycle](runs.md).

```sh
uv run --locked --exact entanglix simulate examples/readout/purcell.toml
uv run --locked --exact entanglix check research_runs/<system_id>/<run_id>
uv run --locked --exact entanglix check research_runs/<system_id>/<run_id> integrity segments diagnostics
uv run --locked --exact entanglix read-run research_runs/<system_id>/<run_id>
uv run --locked --exact entanglix rerun-simulation research_runs/<system_id>/<run_id>
```

## Specification

Version 1 uses `format_version = 1`, an adjacent `system` filename, and `kind =
"lindblad"` or `"spectrum"`. Files must stay beside the protocol; referenced subsystem
files follow the system specification's rules. Optional `[parameters]` overrides
named system parameters in their declared units. Frozen `system.toml` contains the
values actually used.

`[needs]` declares `gradients` (default false) and `device` (`"cpu"`, the default, or
`"cuda"`). Non-gradient CPU requests use QuTiP; CUDA requests use cuDensityMat;
gradient requests use dynamiqs on the declared device. This selects capability,
not performance. There is no harness-imposed dimension cap or pre-launch memory
estimate. CUDA requests require exactly one visible GPU, selected automatically;
`CUDA_VISIBLE_DEVICES` can restrict visibility. Its UUID is recorded in the attempt,
not the portable protocol. Spectra require CPU without gradients.

For `"lindblad"`, `[evolution]` supplies:

- `initial_state = [1, 0]`: one isolated retained-basis level per subsystem, in
  tensor order. For arbitrary numeric states, use `array_file = "state.npz"` and
  `initial_state = "psi"` to select an NPZ array. Pickle arrays are never loaded.
- `times_ns`: explicit output times, or `{start = 0, stop = 100, count = 101}`,
  including both endpoints. Times must increase.
- `observables`: system operator names, or retained-level projectors such as
  `excited = {kind = "population", subsystem = "q", level = 1}`.
- Scalar or interval-valued `drive_amplitudes_ghz`, `carriers`, and an optional
  `frame`, using the [shared evolution conventions](system-spec.md#carriers-and-rotating-frames).
- Optional `rtol`, `atol`, and `max_steps`, with the shared solver defaults.

For `"spectrum"`, provide a positive `levels` count instead of evolution; only
those lowest energies are reported.

Optional `[refinement.<subsystem>]` declares `retained_levels_step` and applicable
`cutoff_steps`. Steps must be positive integers. Checks currently require every
truncated subsystem to be covered, at least one added retained level and two cutoff
units. A missing or below-policy declaration does not block computation: the launch
lists the affected checks as unavailable, with the reason. Explicit finite matrices
have no truncation refinement. A check request can replace the entire declaration
with `--refinement FILE`, using the same `[refinement.*]` tables. The request is frozen
with its linked check run, and the source remains unchanged.

`[metrics.<name>]` may declare `kind = "endpoint_real"` for a Hermitian observable,
with `observable = "excited"`. The endpoint is evaluated from stored expectations.
Unknown formulas and undeclared observables are rejected. `result.toml` also
summarizes observable endpoints and maximum magnitudes. `artifacts/output.npz`
stores all states and expectations; `artifacts/observables.png` plots real and
nonzero imaginary parts.

Optional `timeout_seconds` limits the computation, measured from its start record,
including job imports. Validation and environment synchronization precede it.
Requested checks have their own `--timeout SECONDS`. Cancellation and recovery use
`abort-run` and `reconcile-run` as described in the run reference.

## GPU configuration and checks

Set one `ENTANGLIX_RUNNER_EXTRAS` list for every invocation sharing `.venv-runner`.
It overrides `[tool.entanglix.runner].extras`; an empty value selects no extras.
CPU dynamiqs needs `differentiable`. On the CUDA workstation, use:

```sh
export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12
uv run --locked --exact entanglix simulate examples/readout/purcell-cuda.toml
```

CPU jobs and requested checks use the same pinned extras with CUDA hidden. A different
extras list is rejected before synchronization; the [run reference](runs.md)
explains the environment profile. CUDA jobs see only the allocated UUID, and JAX
preallocation is disabled before initialization. The gate probes the exact job
environment, records native runtime versions, and initializes the selected device.
GPU ownership persists through runner loss while an inheriting job survives;
reconcile a lost attempt before reusing that GPU. No scientific check precedes a
GPU launch. Requested checks run afterwards on CPU.

## Requested checks and segment tolerances

`check RUN` lists relevant checks without computing them. Supply names to run them
as one linked verification attempt. Each check reports pass, fail, or inapplicable
with a reason; diagnostics report quantities without pass/fail. There is no run-level
verdict. Execution code `0` means the requested checks completed, even if a check failed.

- `integrity`: trace, Hermiticity and minimum eigenvalue at every stored time,
  initial-state agreement in the lab basis, and consistency of stored observables
  and metrics. The initial-state Frobenius limit is `1e-8`; it cannot detect a fault
  shared by the declared and stored state.
- `segments`: local convergence comparisons in the source frame, using the source's
  CPU engine or QuTiP for GPU runs. Tightened tolerances and refined truncation can
  share systematic error with the source and do not validate the engine.
- `diagnostics`: top-level populations, ordered static cutoff shifts and outside
  weights, and static full/RWA shifts when applicable. Missing refinement omits
  cutoff quantities with the reason. These are not dynamics error estimates.
- `spectral_refinement`: changes in reported excitation energies at refined truncation,
  using an absolute `1e-5 GHz` limit. It applies only to spectra with refinable subsystems.

No built-in check measures whole-window error or the accuracy of a rotating-wave
approximation. To compare frames or more levels, run another `simulate` with the
changed declaration and compare the reported quantities. Copy the protocol and model
files into a separate directory, increase the model's retained levels and applicable
cutoffs, and retain the same initial-state meaning, frame, drives and output grid. The
[readout example](../../examples/readout/README.md) demonstrates lab-frame CPU and
rotating-frame CUDA computations and records limits of its sampled frame comparisons.

`[segment_tolerances]` overrides the absolute local trajectory tolerance for named
observables. Each `[metrics.<name>]` has its own optional positive
`segment_tolerance`, in the metric's units. Both default to `1e-6`; endpoint metrics
never inherit observable tolerances. A local endpoint comparison does not bound the
final metric's error. Physicality and consistency thresholds are fixed separately.

At most eight segments span two stored output intervals each (one when only one
exists). Start, middle, end and drive changes are candidates; extra change candidates
are spread evenly across the remaining slots. Reports give sampled windows, covered
interval fraction, and uncovered changes as a count with the first and last times.
A coarse output grid can make a segment long. The listing shows windows, source and
refined dimensions, CPU engine, and effective tolerances before a check is requested.
Segments run in one process; `--timeout` and cancellation control execution cost.

`simulate --dry-run simulate.toml` validates and previews model dimensions, output
count, chosen engine/device, and the same check listing shown before a real launch
and by `read_run`. It creates no attempt and runs no scientific solve. Environment
synchronization and a requested-device probe still run.

## Replay

`rerun-simulation RUN` uses the frozen protocol and data under a new ID with `RUN`
as its parent. It rejects changed input digests and records current code and its
actual environment. The command reports `replay.code_matches` and
`replay.environment_matches` against the parent; differing provenance does not block
execution. The earlier record remains unchanged. For an interrupted attempt,
reconcile its process ownership first. Matching provenance and numerical output
agreement provide reproduction evidence. A replay runs no checks automatically.
Plots and elapsed times need not match byte for byte.

Static and dynamic grid sweeps are not yet available.
