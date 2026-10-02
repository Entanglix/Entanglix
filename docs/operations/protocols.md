# Simulation protocols

`entanglix simulate simulate.toml` runs one declared computation, retains its inputs,
arrays, observable plot, summaries, and execution outcome, then starts a separate
verification attempt. `--runs-root PATH` chooses the history root; `--parent-run ID`
links an earlier attempt. Inspect the returned paths with `read-run`. The command's
exit code describes computation execution; the linked verification has its own
execution outcome and scientific verdict.

The [Purcell example](../../examples/readout/purcell.toml) evolves an excited qubit
with an empty lossy resonator for 1600 ns. It uses the
[system conventions](system-spec.md) and [run lifecycle](runs.md).

```sh
export ENTANGLIX_RUNNER_EXTRAS=differentiable
uv run --locked --exact entanglix simulate examples/readout/purcell.toml
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
not performance. The validated production dimension cap is 30. CUDA requests require
exactly one visible GPU, selected automatically; `CUDA_VISIBLE_DEVICES` can restrict
visibility. Its UUID is recorded in the attempt, not the portable protocol. Spectra
require CPU without gradients.

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
those lowest energies are reported and checked. `[refinement.<subsystem>]` declares
`retained_levels_step` and applicable `cutoff_steps`. Every truncated subsystem must
be covered. Minimum steps are one retained level and two cutoff units; explicit
finite matrices have no refinement. References are bounded at dimension 48.

`[metrics.<name>]` may declare `kind = "endpoint_real"` for a Hermitian observable,
with `observable = "excited"`. The endpoint is evaluated from stored expectations
and rechecked by the corrector. Unknown formulas and undeclared metrics cannot
verify. `result.toml` also summarizes the observable
endpoints and maximum magnitudes. `artifacts/output.npz` stores all states and
expectations; `artifacts/observables.png` plots real and nonzero imaginary parts.

Optional `timeout_seconds` limits each computation, pre-launch reference, and
verification job. It counts from the respective start record, including job imports;
validation and environment synchronization precede it. Cancellation and recovery use
`abort-run` and `reconcile-run` as described in the run reference.

## GPU configuration and verification

Set one `ENTANGLIX_RUNNER_EXTRAS` list for every invocation sharing `.venv-runner`.
It overrides `[tool.entanglix.runner].extras`; an empty value selects no extras.
CPU dynamiqs references need `differentiable`. On the CUDA workstation, use:

```sh
export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12
uv run --locked --exact entanglix simulate my-gpu-simulation.toml
```

Declare `needs.device = "cuda"` in that protocol. CPU jobs and references use the
same pinned extras with CUDA hidden. A different extras list is rejected before
synchronization; the [run reference](runs.md) explains the environment profile. CUDA jobs see only the allocated UUID, and
JAX preallocation is disabled before initialization. The gate probes the actual
exact job environment, records native runtime versions, and checks known-array
storage against free device memory. It checks again while holding the inherited GPU
lease before solving. Solver workspace is not estimated.

A CUDA simulation first runs its refined independent CPU reference. Failed reference
execution or physicality checks prevent GPU allocation. The reference is linked to
the simulation, completes before its start record, and is reused by verification.
Before GPU execution starts, `abort-run` targets the linked reference attempt;
interrupting the simulation command also stops that reference. Both attempts record
`aborted/130`, with zero job wall time and GPU usage on the unstarted simulation.
GPU ownership persists through runner loss while an inheriting job survives; a lost
attempt requires reconciliation before that GPU can be reused.

Lab-frame observable comparisons use the corrector's numerical `1e-6` limit.
For an RWA computation, `[rwa_bounds]` can declare a positive approximation bound
per observable. The combined lab-reference comparison uses that declared bound plus
the numerical limit, and reports it explicitly; undeclared observables retain the
numerical limit. Each `[metrics.<name>]` declares its own optional nonnegative
`rwa_bound`, added to the numerical limit for that endpoint. Omission means zero;
trajectory bounds are never inherited by metrics. Approximation bounds require a
rotating frame. These are user-declared model approximation requirements, not
universal accepted tolerances.
Trace, Hermiticity, positivity, and output consistency limits remain unchanged.
New systems and windows need their own frame study. The
[readout evidence](../../examples/readout/README.md#independent-engine-frame-acceptance)
qualifies bounds for its sampled cases and records the unresolved low-drive steps.

## Replay

`rerun-simulation RUN` uses the frozen protocol and data under a new ID with `RUN`
as its parent. It rejects changed input digests and records current code and
its actual environment. The command reports `replay.code_matches` and
`replay.environment_matches` against the parent; differing provenance does not block
execution. The earlier record remains unchanged. For an interrupted attempt,
reconcile its process ownership first. Matching provenance and numerical output
agreement provide reproduction evidence. Every completed replay automatically starts
fresh verification, including when code or environment differs. Plots and elapsed times need not match byte for byte.

Static and dynamic grid sweeps are not yet available.
