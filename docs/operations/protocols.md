# Simulation and sweep protocols

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

Version 1 uses `format_version = 1`, an adjacent `system` filename or a promoted
device's `system_entry` ID (exactly one), and `kind =
"lindblad"` or `"spectrum"`. Files must stay beside the protocol; referenced subsystem
files follow the system specification's rules. Optional `[parameters]` overrides
named system parameters in their declared units. Frozen `system.toml` contains the
values actually used.

Catalog entries resolve in the configured runs root; see [device reuse](catalog.md#reusing-a-device).
The run freezes their system and referenced files and records the entry and source-run lineage.

`[needs]` declares `gradients` (default false) and `device` (`"cpu"`, the default, or
`"cuda"`). Optional top-level `engine` names the engine, built in or from an installed
plugin; leave it unset to use the user's default for the device or the built-in rule:
non-gradient CPU requests use QuTiP, CUDA requests use cuDensityMat, and gradient
requests use dynamiqs on the declared device. A named or default engine that is
missing or cannot meet the needs refuses the run; no other engine replaces it
([engine selection](system-spec.md#engine-selection-policy)). This selects capability,
not performance. There is no harness-imposed dimension cap or pre-launch memory
estimate. CUDA requests require exactly one visible GPU, selected automatically;
`CUDA_VISIBLE_DEVICES` can restrict visibility. Its UUID is recorded in the attempt,
not the portable protocol. Spectra require CPU without gradients and use the QuTiP CPU eigensolver; defaults do
not apply to them, and any other named engine is refused.

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
have no truncation refinement. When a rotating frame declares `excitation_numbers` for
a refined subsystem, they must be the canonical `0, 1, …`; refinement extends them to
the added levels. Other numbers make the refinement checks unavailable, with the
reason. A check request can replace the entire declaration with `--refinement FILE`,
using the same `[refinement.*]` tables. The request is frozen with its linked check
run, and the source remains unchanged.

`[metrics.<name>]` may declare `kind = "endpoint_real"` for a Hermitian observable,
with `observable = "excited"`. An observable `{kind = "coherence", subsystem = "q"}`
is the retained-basis operator `|1⟩⟨0|` on `q` tensored with identity on other
subsystems; its complex expectation is `ρq,01`. A metric `kind = "endpoint_coherence"`
requires that observable and reports `2|ρq,01(T)|`. It is a coherence proxy, not a
fitted dephasing rate. Refinement carries its phase with the source eigenstate gauge
while retaining identity on all spectator levels. Endpoints are evaluated from stored expectations.

A spectral metric `kind = "dispersive_shift"`, with `qubit = "q"` and
`resonator = "r"`, requires exactly those two subsystems. Within the reported
`levels`, it assigns dressed states `(0,0), (0,1), (1,0), (1,1)` by largest squared
bare-product overlap. Assignments must be distinct and overlaps at least `0.9`;
otherwise computation fails with the indices and overlaps. It reports
`χ = (E11 − E10 − E01 + E00)/2` in GHz. The archive retains assigned energies,
indices and overlaps as well as the lowest spectrum. Spectral refinement compares
both reported excitation energies and these metric values.
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
explains the environment profile. Numeric indices in `CUDA_VISIBLE_DEVICES` are
resolved in `nvidia-smi` order, which may differ from CUDA's order on a multi-GPU
machine; use a GPU UUID to select a device unambiguously. CUDA jobs see only the
allocated UUID, and JAX preallocation is disabled before initialization. The gate
probes the job environment and confirms the device and driver; for cuDensityMat it
also records native runtime versions and initializes the library on the device. A
CUDA plugin engine needs no cuQuantum stack.
Device ownership is taken before the probe and admission. A busy device refuses
the launch without creating an attempt or using investigation allowance; the
[ownership reference](runs.md#local-gpu-ownership) covers reconciliation and dry runs.
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
  built-in CPU engine, or QuTiP for GPU and plugin-engine runs. Tightened tolerances
  and refined truncation can share systematic error with the source and do not
  validate the engine. For a plugin-engine run the comparison also includes engine
  differences, and its scope says so; no check needs the plugin, so a run is read
  and checked where its plugin is not installed.
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
count, chosen engine/device with the source of the choice and the engine's distribution
version and record digest, and the same check listing shown before a real launch
and by `read_run`. It creates no attempt and runs no scientific solve. Environment
synchronization and a requested-device probe still run.

## Replay

`rerun-simulation RUN` uses the frozen protocol and data under a new ID with `RUN`
as its parent. It rejects changed input digests and records current code and its
actual environment. It runs on the parent's engine, whatever the current default, and
is refused if that engine is not installed. The command reports `replay.code_matches` and
`replay.environment_matches` against the parent; differing provenance does not block
execution. The earlier record remains unchanged. For an interrupted attempt,
reconcile its process ownership first. Matching provenance and numerical output
agreement provide reproduction evidence. A replay runs no checks automatically.
Plots and elapsed times need not match byte for byte.

## Serial grid sweeps

`entanglix sweep sweep.toml` runs one static or dynamic grid as one attempt.
Version 1 requires an adjacent `base` simulation spec and at least one axis.
Optional `evidence_metric` names a metric in the base spec to select extrema
and steepest-change evidence points when a refinement check is requested.
Without it, selection uses corners and any requested diagnostics warnings.
The base supports either computation kind; dynamic sweeps use its declared lab
or rotating frame, drives, initial state and solver settings.

```toml
format_version = 1
base = "simulate.toml"
evidence_metric = "excited_population"

[[axes]]
id = "frequency"
target = "parameters.frequency"
values = [0.2, 0.3, 0.4]
```

Axis IDs and targets are unique. Values are finite and distinct, and the last
axis varies fastest. Targets name declared scalar fields: `parameters.NAME`,
`evolution.drive_amplitudes_ghz.NAME`,
`evolution.carriers.NAME.frequency_ghz`, or
`evolution.frame.frequencies_ghz.SUBSYSTEM`. A parameter must already appear in
the base's `[parameters]` overrides and be declared by its system.

Optional `[axes.linked_values]` maps other targets to a list of the same length,
paired with that axis's values. The [readout sweep](../../examples/readout/dephasing.toml)
uses this for the carrier and frame frequencies at each `EJ`. It is explicit data,
with no expressions or automatic frequency updates. A target belongs to one axis.
Every point's effective model and plan are frozen before admission;
refinement retains that point's frequencies and initial-state meaning.

An optional `[expectation]` declares `metric`, `axis`, and `direction = "increasing"`
or `"decreasing"`. Request `declared_expectation` to check every grid line along
that axis in numeric-value order, regardless of the declared value order.
The check uses the corrector's numerical tolerance (`1e-6` for dynamic metrics,
`1e-5 GHz` for spectral metrics). Its failure means "declared expectation violated";
the expectation may itself be wrong. No expectation is inferred from the model.

```sh
uv run --locked --exact entanglix sweep sweep.toml --dry-run
uv run --locked --exact entanglix sweep sweep.toml
uv run --locked --exact entanglix check research_runs/<system_id>/<run_id> integrity segments diagnostics declared_expectation
uv run --locked --exact entanglix rerun-sweep research_runs/<system_id>/<run_id>
```

Points execute serially using the base's selected engine/device; a sweep rerun keeps
the parent's engine. All states and
observables, or reported spectral arrays, remain in per-point artifacts. A point
publishes its array archive first and then its atomic result. `read-run` shows
completed points while the run is active or interrupted. No aggregate result
establishes completion until the runner writes `FINISHED`. The sweep result retains
metric grids, extrema and plot links: line plots for one varying axis, heatmaps for two,
and point-order plots for larger grids or a single point. No scientific diagnostic runs unless requested.

On request, integrity and diagnostics cover every retained dynamic point. Segments
and spectral refinement select at most eight points: corners first, metric extrema,
the first endpoint of the steepest grid edge, and ranked truncation warnings when
`diagnostics` is requested in the same check run. Selection and ties are deterministic
in grid order. Listings describe the checks once using the first point; reports
record the actual selection. Every warning is reported, and uncovered
warning points are identified without expanding the bound. If corners alone exceed
eight, the sweep still runs; the bounded checks are inapplicable. Other checks remain
available. Reports list each checked and uncovered point, local windows, reasons,
residuals and tolerances. Passing sampled checks makes no convergence claim at
unsampled points or about whole-window/RWA accuracy.

A sweep-level `timeout_seconds` covers the whole job, including imports and all points;
the base's single-simulation timeout does not apply separately at each point.
Cancellation and runner-loss recovery use the existing run lifecycle. `rerun-sweep`
checks frozen input digests, recomputes the entire grid with current code/environment,
and reports provenance matches against its parent. To explore a narrower or finer
grid, copy and edit the specs and launch with `--parent-run ID`. Partial points are
retained for inspection; replay does not resume them. Checks remain separate commands.
