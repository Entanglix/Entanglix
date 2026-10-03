# M2 Guide — Runs, Verification, and First Sweeps

**Status:** Implemented through local simulations, requested checks and serial sweeps. Completion and acceptance evidence belong to the checklist. The decisions in §8 are recorded in [ADR-0003](adr/0003-runs-verification-and-dispatch.md), with matching contract, roadmap, and checklist edits. Implemented syntax and operating conventions live in the operational references.
**Scope:** M2, covering Block 3 (Runner and Run Contract), Block 4 (Corrector v0), and Block 5 (Protocols `simulate` and `sweep`).
**Authority:** [roadmap](roadmap.md#block-3--runner-and-run-contract), [accepted mandate](adr/0001-mandate.md#4-later-decisions-and-implementation-evidence), and [harness contract](../HARNESS_CONTRACT.md#2-execution-and-artifacts). [The checklist](checklist.md#m2--local-sweep-with-checks-blocks-35) owns completion status and acceptance evidence.
**Audience:** the owner and implementation agents. This is a planning document; operational docs and production agents must work without it.

## 1. Outcome and boundary

M2 turns the M1 library into retained research runs on the workstation that a user can check on request. At completion, a developer or expert user can:

1. Launch a simulation, a sweep, or an authored script through `entanglix`. The runner validates inputs, checks the environment, selects the engine from the versioned table, and creates a run directory.
2. List and read every attempt, including failed, aborted, and incomplete ones, with lineage, resource use, and the live trace of a running job; abort a stalled job; rerun interrupted work under a new linked ID.
3. Reproduce a completed run from its frozen inputs, code version, and environment hash.
4. Request built-in checks on a completed `simulate` or `sweep` run and receive a report listing each check's outcome, scope, tolerance, residual, and coverage, and inapplicable checks with reasons.
5. Run static (spectral) and dynamic sweeps, undriven or driven, in the lab frame or a rotating frame, and use plots and summaries to choose a refined sweep.

M2 covers standalone runs. Investigations, progress checkpoints, and limits (contract §4) arrive in Block 6a; MCP transport and agent permissions in Block 6b; scheduler execution in Block 9; control in Block 7. M2 supplies the Python backing operations and CLI commands that the MCP surface will later wrap.

GPU runs in M2 demonstrate the execution path, not a performance choice. Crossover and peak-memory measurements, batching, and automatic GPU selection belong to a later cuDensityMat usage study, tracked in the checklist as a prerequisite for automatic performance-based dispatch. Costs recorded during M2 therefore describe an unstudied GPU path; investigation sizing after M2 should treat them accordingly.

## 2. Build by block

```text
M1: models, engines, selection table
    |
Block 3: run records --> CPU runner --> lifecycle and GPU lock
    |
Block 4: corrector and plans --> checks
    |
Block 5: feasibility --> drives and frames --> simulate --> checks on request --> sweep + M2 acceptance
```

Each step belongs to one block, and blocks complete in order. Block 3 accepts runner facilities through authored CPU jobs, CPU process/lock tests, and a workstation GPU environment check. Plan-job integration belongs to 4A, linked verification and inspection to 4A–4B, and built-in protocol records, reproduction, and scientific GPU dispatch integration to 5C–5D. Block 4 owns trusted computation definitions, shared series-check functions, and spoofing tests; 5C–5D supply protocol-specific metric definitions and apply bounded evidence selection and declared-expectation checks to actual sweep grids. Those integration checks remain required for the later blocks. Step 5A needs only M1 and QuTiP, so it can run before Blocks 3 and 4 finish.

| Step | Work | Evidence before advancing |
|---|---|---|
| 3A — Run records | Fields of `run.toml`, the runner's start and reconciliation records, `result.toml`, and `FINISHED`; run IDs; `env_hash` inputs; code provenance including uncommitted changes; exit codes | Records and their meanings written in an operational run reference; contract wording agrees |
| 3B — CPU runner | Gate (spec validation, import scan, environment check); launch of an authored script in its own process group; frozen `spec/`; `FINISHED` last; `list_capabilities`, `list_runs`, `read_run` | Completed and failed runs leave valid records; launches rejected before `run.toml` exists exit `64` with a diagnostic; a completed run reproduces |
| 3C — Lifecycle and GPU | Timeouts, `abort_run`, termination of the job or runner, an attempt lock, a GPU lock across runner instances, reconciliation of unknown process state, live trace, rerun lineage, documented recovery | Abrupt termination never yields a completed `FINISHED`; an attempt whose runner or job is alive is never reconciled; a runner killed while its job survives leaves the GPU blocked until reconciliation; interrupted attempts, including those that never started, remain queryable with their records and uncertain usage |
| 4A — Corrector and plans | Versioned thresholds, policy digest, deterministic bounded evidence selection, trusted computation definitions, internal execution plans and CPU jobs, linked check records | Plan jobs preserve runner records and provenance; source bindings and policy digests are checked; dimension-48 exact regression qualifies the CPU engines without imposing a cap |
| 4B — Shared checks | Integrity and spectral-refinement functions, declared sweep-expectation checks, coverage and linked inspection | Deliberate faults are detected; labels cannot change check applicability; reports identify scope, tolerances, residuals, coverage and inapplicable reasons |
| 5A — Feasibility and metric | Disposable QuTiP probe of the first example; metric, region, windows, grids, drive-frequency policy, refined dimensions, and candidate thresholds | Choices recorded with measured durations and step counts before drive, frame, or protocol implementation |
| 5B — Drives and frames | Lab-frame carrier drives on all three engines; rotating frames with an explicit rotating-wave approximation; frame comparison in the reference solve | Carrier fixtures pass M1-style agreement, refinement, and physicality checks; rotating and lab frames agree within declared tolerances |
| 5C — `simulate` | Protocol spec, declared needs/device selection, GPU environment and ownership/lifecycle integration, frozen inputs, summaries, plots and replay; Purcell example | Documented CPU/CUDA computations complete and reproduce; cancellation, runner loss, reconciliation and linked rerun tests pass; checks are requested separately through 5V |
| 5V — Checks on request | A run computes and stores its results only. Built-in checks (integrity, same-frame refined segments, diagnostics, spectral refinement) run on request with per-check outcomes and no verdict. A check's prerequisites never restrict a run, and the dimension caps and memory estimates are removed (§5) | A run completes with no verification run and with no refinement declared; requested segments catch the probe's segment-detectable failures and not its false alarms; a run above dimension 30 completes, and an allocation failure is reported as out of memory; the 5C simulation acceptance re-runs with its requested checks passing |
| 5D — `sweep` and M2 acceptance | Static and dynamic grid sweeps, frozen grids and per-point settings, plots and summaries, sweep checks including declared expectations; readout sweep | The agreed sweep runs end to end on the GPU and reproduces; its requested checks pass and report their coverage; interrupted partial results and rerun lineage remain queryable; M2 exit criteria pass |

## 3. Block 3 — runs as the history

### Run directory

The runner follows amicode's run-directory layout and terminal semantics without its Julia-specific fields:

```text
research_runs/<system_id>/<run_id>/
  run.toml      # runner, first: run ID, parent run IDs, protocol and declared needs,
                # spec digests, code provenance, env_hash, engine, device, executor
  runner/       # runner, read-only: started.toml before computation (start time,
                # GPU allocation, process group); reconciled.toml after runner loss
  spec/         # frozen system and protocol specs, authored script and data
  run.log       # job output and runner diagnostics; read_run tails it while running
  result.toml   # job, published by temporary-file rename
  artifacts/    # job: arrays with digests, plots
  FINISHED      # runner, last: status, exit_code, end time, usage
```

Run IDs use amicode's `r<UTC-timestamp>Z-<hex>` form, and the runner creates each directory exclusively. Exit codes follow contract §2: `0` completed, `130` aborted, `64` usage or orchestration fault, otherwise the job's code. Run Inspector conveniences and control-specific records (`index`, `latest`, `AMICODE_ITER` lines, `iter_<N>.png` names, amicode's `[julia]` table and `fidelity`/`iterations` fields) wait for Block 12 or the protocol that needs them.

The runner's records are the attempt history; there is no separate ledger store. Jobs write only `result.toml` and `artifacts/`; runner records use separate, read-only paths. `list_runs` scans the runs root. An attempt without a valid `FINISHED` is incomplete: active while its attempt lock is held or its process group is alive, otherwise interrupted. Reconciliation records an interrupted attempt in `runner/reconciled.toml` with its last observed state, its start record if one was written, and usage marked uncertain; it never writes `FINISHED`, and a result left by such an attempt cannot substantiate a completed result. Reruns get new IDs with the earlier attempt as a parent. Check outcomes come from linked verification runs (§4).

### Environment and provenance

`env_hash` covers the `uv.lock` digest, Python and platform, the installed extras, and observed driver, CUDA, and cuQuantum versions. Jobs sharing an environment run through `uv run --locked --exact` with one declared extras set: an exact sync with a different set would reshape the environment between jobs. Scientific GPU dispatch probes the actual job environment. A workstation-local extras list declares both GPU extras for every job sharing the environment, so CPU-only installations stay lean (§8). Separate processes still keep JAX and cuQuantum initialization apart.

Code provenance records the Git revision and dirty state. For uncommitted changes, freeze a patch and untracked source files into `spec/` so the run can be reproduced; digests alone only detect differences. Authored scripts and data are copied into `spec/` with digests. Reproduction reruns the frozen inputs in an environment with the same `env_hash`, using determinism checks where the engine supports them and declared tolerances otherwise.

### Launch and lifecycle

The gate validates specs through `load_system` and the protocol loader, scans authored-script imports against the import-boundary allowlist, and, when the selected engine uses CUDA, checks the driver and cuQuantum. There is no daemon or queue: the runner process supervises its job in a new process group, and separate CLI invocations read and abort through the run records. The job begins computing only after the runner has written `runner/started.toml`.

The runner is outside its job's process group, so an empty group does not show that the runner is gone. Each attempt therefore has an attempt lock, a `flock` on its directory, taken before `run.toml` is published. The job inherits it, and the runner releases it and any GPU lock only after writing `FINISHED`.

- Timeouts: the runner timer signals the job's process group and escalates after a documented grace period.
- `abort_run` signals the recorded process group; the supervising runner writes `FINISHED` with `aborted` and exit code `130`.
- If the runner dies, no process writes `FINISHED` and the job may still be running. Reconciliation acquires the attempt lock without waiting and writes its record while holding it, so it never acts on an attempt whose runner or inheriting job is alive and cannot race `FINISHED`. With the lock held, a live recorded process group keeps the GPU blocked; otherwise the attempt receives `runner/reconciled.toml`. An attempt without `started.toml` has no recorded group, and the lock alone decides. Test a runner paused between its job's exit and `FINISHED`: reconciliation must not run.
- The GPU lock is per device and lives outside any runs root, so it excludes across runner instances. The job inherits the lock, which stays held while any inheriting process keeps it open ([`flock` semantics](https://man7.org/linux/man-pages/man2/flock.2.html)); the lock record also names the run and process group, and reconciliation checks that group before reuse. Test a runner killed with `SIGKILL` while its job survives: GPU dispatch is refused until the job exits and reconciliation is recorded.

The lock is a file lock, so lifecycle tests use short CPU jobs and real signals and run in CPU CI.

## 4. Block 4 — corrector

### Package, thresholds, and policy digest

`src/entanglix/corrector/` ships in the same distribution, with a versioned thresholds file as package data, like the engine-selection table. Neither lives in run directories or authored-script workspaces, and research tools cannot write them. The corrector policy version is the digest of the corrector code and thresholds: the verification run records it at launch, the job recomputes it when the corrector loads, and a mismatch fails the verification run without a report. Version-1 thresholds start from the M1 pinned limits and also set minimum refinement steps and the evidence-point bound.

### Plans and applicability

Checks are keyed to computation kinds M1 already provides: Lindblad evolution, spectrum, and parameter series. A plan is the runner-validated execution record for one computation: frozen system, evolution or spectral request, and engine decision. Step 4A pins the plan format and adds a built-in plan job that evaluates a plan on a named engine. Block 4 tests check plan runs directly; Block 5 protocols compile to plans and add metrics, summaries, and plots.

The corrector derives the checks relevant to a run from the plan's content and the trusted definition of its kind, not from labels in specs or results. For example, dynamics checks do not apply to a spectral run, and rotating-wave shifts are reported only for a rotating-frame run. Runs without a trusted definition, including authored scripts, have no built-in checks.

A check's prerequisites never restrict a run. A refinement declaration is optional: a plan without one loads and runs.

- A malformed declaration, such as a wrong type or a nonpositive step, is rejected by the spec schema like any other field.
- A declaration that is missing, leaves out a truncated subsystem, or is below the thresholds' minimum does not stop the run. The listing names the checks that cannot run and why, and those checks are reported as inapplicable with that reason.
- A check request may carry a replacement declaration, so correcting one never requires repeating the run.

### Verification runs

Verification is an ordinary run whose parent is the source run. It executes the checks the user requested, and its `result.toml` holds a report bound to the source artifact digests and the corrector policy version. A replacement refinement declaration supplied with the request is frozen in the verification run's `spec/`; the source run is unchanged. The report lists each check's outcome (pass, fail, or inapplicable with a reason), scope, tolerance, residual, and the engine, dimension, and refinement declaration of any refined solve, and marks each point's coverage. There is no verdict (5V).

A failed or interrupted verification run reports no outcome. No verification run starts unless the user requests a check.

`read_run` lists the checks relevant to the run and its verification runs with their status, policy version, source artifact digests, and each check's outcome. A failed or incomplete attempt is shown with diagnostics and no outcomes. Reports under an earlier policy remain history labelled with that policy.

### Dynamics checks and cost

The dynamics checks are integrity of the stored states at every point, refined segments at evidence points, and diagnostics; the [protocol reference](operations/protocols.md#requested-checks-and-segment-tolerances) defines each. Segments and spectral refinement build a refined plan in the same format and solve it inside the verification job; no separate plan job is launched.

For sweeps, step 4A pins a deterministic selection rule and a small maximum evidence-point count in the corrector policy. A starting rule is grid corners plus the metric's global minimum and maximum, one point for its steepest change, and one for the worst truncation warning, deduplicated with ties resolved by grid order. The bound limits checks, not grids: a sweep whose required corners exceed it still runs, and a requested segments or refinement check is reported as inapplicable with that reason. Report every warning, but select only the bounded worst cases for refined solves. If the evidence points cannot cover the warnings within the bound, report that instead of expanding the set. The report names selected points, reasons, and uncovered warnings. Warning candidates are included when diagnostics is requested with the refinement check; no diagnostic is computed without a request. A declared expectation, such as a metric's direction along an axis, comes from the sweep spec (5D).

Refined solves run on CPU, so a check never waits for the GPU lock. Segments use the run's own engine on CPU and QuTiP for GPU runs, so checking a QuTiP run needs no optional extra.

The spectral check refines cutoffs and retained levels for the reported quantities, following the M1 CPU fixtures. Comparisons with a second eigensolver, an analytic case, or an independently constructed model stay in the M1 acceptance tests and are not run-time checks.

The checklist links the check and sweep evidence, including grids whose corners exceed the evidence-point bound and warnings that do not expand the evidence set.

## 5. Block 5 — feasibility, drives, frames, and protocols

### Feasibility and metric

Step 5A's CPU measurements are retained in the [feasibility probe](../tests/probes/m2_feasibility.py) and [results](../tests/probes/m2_feasibility_results.json). The [readout example](../examples/readout/README.md) records the selected model, metric, windows, grids, drive-frequency policy, refinement, costs, and candidate frame bounds, separating endpoint metric resolution from full-trajectory diagnostics. Its low-drive resolution limits and thin lab-frame refinement margin qualify the selected fixture. The probe keeps its own carrier and frame calculations; independent-engine acceptance of the sampled RWA bounds uses the shared implementation (see the readout example). The current corrector numerical thresholds are unchanged.

### Drives and frames

Step 5B implements continuous carrier drives and numeric rotating-wave frames through the shared evolution input. Settled field names, transformation conventions, supported dissipators, and numerical limits are in the [operational interface](operations/system-spec.md#carriers-and-rotating-frames). The [independent-engine study](../examples/readout/README.md#independent-engine-frame-acceptance) supplies the full-window acceptance for the first fixture. The required behavior is:

- **Carrier drives in the lab frame:** a declared envelope times `cos(2π f t + φ)`, with `f` in GHz, `t` in ns, and `φ` in radians. Each engine evaluates it continuously through its time-dependent coefficient path: QuTiP coefficients, dynamiqs modulated operators, and cuDensityMat callbacks through qutip-cuquantum, which evaluates them on the CPU at every step. Constant and piecewise-constant envelopes come first; other envelopes arrive with the example that needs them.
- **Rotating frames:** declared in the evolution input as frame frequencies applied to excitation-number operators in each subsystem's retained eigenbasis; explicit-operator models must declare theirs. The transformation is an explicit numeric operation on the constructed operators through the common engine interface, not a general symbolic framework. A declared rotating-wave approximation removes counter-rotating terms, and provenance records the frame and the removed terms. Comparisons across frames use frame-invariant observables or states transformed back to the lab frame.
- **Frame comparison:** 5B accepted the first fixture's frames with a full-window lab-frame reference. From 5V, no built-in check measures the approximation; a user compares a rotating-frame run with a lab-frame run of the same spec.

The pinned adaptive solvers stay; there are no stiff or specialized integrators. The rotating frame is the cost lever; checking its approximation against a lab-frame run is the user's choice. Carrier fixtures need M1-style acceptance against independent references on all three engines. Before 5C relies on the lab-frame 5 × 6 readout/Purcell fixture, 5B must compare its full-window observables against CPU dynamiqs at tightened tolerances and 6 × 8 levels. Differentiable controls extend to carrier envelopes only where gradient tests cover them; otherwise `evolve_jax` rejects them.

### `simulate`

A `simulate.toml` protocol spec names the system, frame, initial state, output grid, drives, observables, solver tolerances, optional refinement steps, declared needs, and an optional timeout, and compiles to one plan. Declared needs are `gradients` and `device`. A new selection-table version maps a declared `device = "cuda"` without gradients to cuDensityMat; without a declared device, non-gradient requests stay on QuTiP CPU. The table records needs, not performance. The harness imposes no dimension cap, and a run that exhausts memory fails as the [run reference](operations/runs.md#execution-outcomes) describes.

GPU launches follow the contract's cheap-to-expensive funnel. No scientific check precedes or follows a launch unless the user requests one. Results include observable and metric summaries in `result.toml`, stored states and observables in `artifacts/`, and plots. Plots use matplotlib, today a transitive scqubits dependency; declare it directly and add it to the import-boundary allowlist.

### Checks on request (5V)

Step 5V implements [ADR-0003 §2.3–§2.5](adr/0003-runs-verification-and-dispatch.md#23-verification-runs-and-checks-on-request). The [protocol reference](operations/protocols.md#requested-checks-and-segment-tolerances) and the [run reference](operations/runs.md#corrector-policy-and-verification-records) describe the implemented interface, and the checklist links the acceptance evidence. The [probe](../tests/probes/m2_verification.py) and its [measurements](../tests/probes/m2_verification_results.json) supply the measured basis and the regression cases.

- **Requests:** a run computes and stores its results. `check` lists the relevant checks, or runs named ones as one linked verification run with per-check outcomes and no verdict.
- **Checks:** integrity, same-frame refined segments on the run's own CPU engine, diagnostics, and spectral refinement. No built-in check measures whole-window error or the rotating-wave approximation; a user compares two runs.
- **Prerequisites:** they never restrict a run. The refinement declaration is optional, and a check request may replace it.
- **Size:** the harness imposes no dimension cap and makes no memory estimate before launch. An allocation failure is reported as out of memory, and an unexplained signal death as its signal.
- **Removed:** automatic verification, verdicts, waivers and the allowance, the full-window reference and `compare-reference`, the spectral cross-checks, the dimension caps, and the memory estimates.

Step 5D applies the same rules to sweeps: segments and spectral refinement at the bounded evidence points (§4), and declared expectations from the sweep spec.

### `sweep`

A `sweep.toml` names a base `simulate` or spectral spec and grid axes over system parameters, such as a constructor's `josephson_energy`, and protocol quantities, such as a drive amplitude. Static sweeps compute spectral quantities; dynamic sweeps evolve each point, undriven or driven, in either frame. One run covers one sweep: points run serially inside the job, completed points remain in `artifacts/` if the run is interrupted, and a rerun is a new linked run that recomputes the grid. On request, integrity and diagnostics cover every point whose states the sweep retained, and segments or spectral refinement run at the bounded evidence points (§4) selected after the sweep. Reports mark which points were sampled. The bound limits these checks, never the grid. A sweep spec may declare an expectation, such as a metric's direction along an axis, which becomes a check the user can request.

Summaries and plots show the metric over the grid. `read_run` adds the outcomes and coverage of any checks run, including requested diagnostics such as per-point top-level populations; a sweep computes no diagnostic unrequested. From these a user can choose a narrower or finer grid and rerun with the earlier sweep as parent. Batch-versus-serial checks are needed only before batching is advertised.

## 6. First example

One qubit–resonator model serves the Purcell simulation, measurement-induced-dephasing sweep, and static dispersive-shift companion. The settled CPU feasibility choices and their limits are in the [readout example](../examples/readout/README.md), with a [validated model](../examples/readout/system.toml) and [retained measurements](../tests/probes/m2_feasibility_results.json). Production uses 5 × 6 = 30 retained levels; refinement uses 6 × 8 = 48. The public protocols and serial GPU sweep use the shared implementation. A Floquet quasienergy example ships as an authored script, claiming quasienergies under periodic driving only.

## 7. Dependencies, layout, and validation

The runner uses the standard library for processes, signals, and file locks.

Suggested locations, created as code arrives:

```text
src/entanglix/runner.py          # records, gate, launch, lifecycle, GPU lock, list/read/abort
src/entanglix/corrector/         # definitions, checks, thresholds.toml, verification job
src/entanglix/protocols/         # plan job, simulate, sweep
src/entanglix/evolution.py       # carrier drives and frames in the evolution input
tests/probes/m2_feasibility.py   # disposable 5A probe and its retained results
tests/probes/m2_verification.py  # 5V verification probe and its retained results
docs/operations/runs.md          # run directory, records, exit codes, recovery
examples/readout/                # first example specs
```

Frames and carrier drives extend `docs/operations/system-spec.md`. Keep optional GPU imports inside their adapters, and keep `import entanglix` free of engine packages. The M1 required-GPU pattern continues: GPU tests assert the actual engine and device, run each engine stack in its own process, and fail rather than skip when required.

Proposed validation commands after the corresponding capabilities exist:

```sh
uv run --locked --exact pytest -m "not gpu"
uv run --locked --exact --extra differentiable pytest -m "not gpu" -k "corrector"
uv run --locked --exact ruff check .
uv run --locked --exact ruff format --check .
uv run --locked --exact --extra differentiable-cuda12 --extra cuquantum entanglix simulate examples/readout/purcell.toml
uv run --locked --exact --extra differentiable-cuda12 --extra cuquantum entanglix sweep examples/readout/dephasing.toml
```

## 8. Settled decisions and open choices

Recorded in [ADR-0003](adr/0003-runs-verification-and-dispatch.md), with matching contract, roadmap, and checklist edits:

1. Run directories are the attempt history: runner-owned `run.toml`, start and reconciliation records, and `FINISHED`, with no separate ledger store, `index`, or `latest`. amicode's layout and terminal semantics are adopted without Julia fields; Inspector compatibility waits for Block 12.
2. Verification is a separate linked run whose `result.toml` holds a report of per-check outcomes bound to source digests and the policy version; protocol results carry no check outcomes.
3. Nothing is checked unless the user requests it. The built-in checks are:
   - integrity of stored states;
   - same-frame segments as local evidence;
   - diagnostics;
   - spectral refinement;
   - declared expectations in sweeps.

   Each check reports its own outcome, with no verdict, waiver, or allowance. A check's prerequisites, such as a refinement declaration or the evidence-point bound, never restrict a run. No built-in check measures whole-window error or the rotating-wave approximation. In a sweep, segments and refinement run at a bounded, deterministically selected set of points, and reports state coverage.
4. The harness imposes no dimension cap, on dispatch or on the corrector's refined solves, and makes no memory estimate before launch; tested sizes are acceptance evidence, not a ceiling. An observed allocation failure is reported as out of memory, and an unexplained signal death as its signal.
5. A declared `device` need routes non-gradient requests to cuDensityMat. Engine efficiency and the corresponding Block 5 checklist items move to a cuDensityMat usage study, tracked as a prerequisite for automatic performance-based dispatch.
6. Lab-frame carrier drives and rotating frames are M2 capabilities, using the pinned adaptive solvers and explicit numeric transformations.
7. The first example is the qubit–resonator readout model in §6; 5A settled the metric and feasibility choices recorded in the readout example.

Run records and implemented plans are documented in [the operational run reference](operations/runs.md); evidence-point selection is implemented in the corrector policy and reaches the operational references with sweeps (5D). The readout example records the settled feasibility choices and independently accepted frame bounds with their resolution limits. Frame declarations and rotating-wave rules are implemented as described in the operational system reference. Sweep syntax, CLI commands, full per-point retention and replay are implemented in the protocol reference. Checks remain separate requests; a sweep spec does not trigger them. Simulation syntax, trusted endpoint metrics, segment tolerances, replay, and workstation-local runner extras are documented in the [protocol reference](operations/protocols.md). The extras choice must preserve one declared set for jobs sharing an environment and run the GPU probe in the actual job environment.

## 9. Exit evidence and handoff

[Checklist M2](checklist.md#m2--local-sweep-with-checks-blocks-35) is the sole acceptance list; this guide adds no completion gates. Acceptance runs are themselves Entanglix run directories: retain the cited runs and their verification runs, and link them with test commands and results, as M1 retained its acceptance results. Publish run records, exit codes, and recovery in `docs/operations/runs.md`, frames and carrier drives in `system-spec.md`, protocol usage in an operational protocol reference, and available commands in the root README. Operational documents stand alone without linking to planning docs.

The investigation decisions after M2 build on the runner, run history, verification, and sweep outputs. Block 6a adds investigation revisions to run records and admission checks before launch; Block 6b wraps the backing operations in MCP. The cuDensityMat usage study can proceed once M2 runs exist and updates the selection table only from its measurements.

Before retiring this guide, retain consequential design rationale in ADRs and necessary conventions and instructions in operational docs.
