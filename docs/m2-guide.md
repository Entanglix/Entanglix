# M2 Guide — Runs, Verification, and First Sweeps

**Status:** Draft implementation plan; no M2 capability is claimed by this document. The decisions in §8 are recorded in [ADR-0003](adr/0003-runs-verification-and-dispatch.md), with matching contract, roadmap, and checklist edits. Formats, syntax, fixture values, and file names below are proposals.
**Scope:** M2, covering Block 3 (Runner and Run Contract), Block 4 (Corrector v0), and Block 5 (Protocols `simulate` and `sweep`).
**Authority:** [roadmap](roadmap.md#block-3--runner-and-run-contract), [accepted mandate](adr/0001-mandate.md#4-later-decisions-and-implementation-evidence), and [harness contract](../HARNESS_CONTRACT.md#2-execution-and-artifacts). [The checklist](checklist.md#m2--verified-local-sweep-blocks-35) owns completion status and acceptance evidence.
**Audience:** the owner and implementation agents. This is a planning document; operational docs and production agents must work without it.

## 1. Outcome and boundary

M2 turns the M1 library into retained, verified research runs on the workstation. At completion, a developer or expert user can:

1. Launch a simulation, a sweep, or an authored script through `entanglix`. The runner validates inputs, checks the environment, selects the engine from the versioned table, and creates a run directory.
2. List and read every attempt, including failed, aborted, and incomplete ones, with lineage, resource use, and the live trace of a running job; abort a stalled job; rerun interrupted work under a new linked ID.
3. Reproduce a completed run from its frozen inputs, code version, and environment hash.
4. Receive an independent verdict for completed `simulate` and `sweep` runs, listing checks, tolerances, residuals, evidence coverage, and inapplicable checks with reasons.
5. Run static (spectral) and dynamic sweeps, undriven or driven, in the lab frame or a rotating frame, and use plots and summaries to choose a refined sweep.

M2 covers standalone runs. Investigations, progress checkpoints, and limits (contract §4) arrive in Block 6a; MCP transport and agent permissions in Block 6b; scheduler execution in Block 9; control in Block 7. M2 supplies the Python backing operations and CLI commands that the MCP surface will later wrap.

GPU runs in M2 demonstrate the execution path, not a performance choice. Crossover and peak-memory measurements, batching, automatic GPU selection, and dispatch caps above 30 belong to a later cuDensityMat usage study, tracked in the checklist as a prerequisite for automatic performance-based dispatch. Costs recorded during M2 therefore describe an unstudied GPU path; investigation sizing after M2 should treat them accordingly.

## 2. Build by block

```text
M1: models, engines, selection table
    |
Block 3: run records --> CPU runner --> lifecycle and GPU lock
    |
Block 4: corrector and plans --> checks and compare_reference
    |
Block 5: feasibility --> drives and frames --> simulate --> sweep + M2 acceptance
```

Each step belongs to one block, and blocks complete in order. Step 5A needs only M1 and QuTiP, so it can run before Blocks 3 and 4 finish.

| Step | Work | Evidence before advancing |
|---|---|---|
| 3A — Run records | Fields of `run.toml`, the runner's start and reconciliation records, `result.toml`, and `FINISHED`; run IDs; `env_hash` inputs; code provenance including uncommitted changes; exit codes | Records and their meanings written in an operational run reference; contract wording agrees |
| 3B — CPU runner | Gate (spec validation, import scan, environment check); launch of an authored script in its own process group; frozen `spec/`; `FINISHED` last; `list_capabilities`, `list_runs`, `read_run` | Completed, failed, and usage-error runs leave valid records; a completed run reproduces |
| 3C — Lifecycle and GPU | Timeouts, `abort_run`, termination of the job or runner, a GPU lock across runner instances, reconciliation of unknown process state, live trace, rerun lineage, documented recovery | Abrupt termination never yields a completed `FINISHED`; a runner killed while its job survives leaves the GPU blocked until reconciliation; interrupted attempts remain queryable with start, allocation, and uncertain usage |
| 4A — Corrector and plans | `entanglix.corrector`, versioned thresholds including reference dimension and evidence-point limits, deterministic point selection, policy digest, check definitions per computation kind, execution-plan format and plan job, verification run records | Unknown or inconsistent definitions cannot verify; a hash mismatch fails verification; both CPU reference engines pass an exact-Liouvillian regression at the reference limit |
| 4B — Checks and `compare_reference` | Output checks, combined reference solves at evidence points, spectral checks, sweep-level checks, coverage reports, evidence reuse | An under-converged run is `UNVERIFIED`; label-spoof cases cannot skip checks; reports show scope, tolerances, residuals, reference engine and dimension, coverage, and inapplicable checks with reasons |
| 5A — Feasibility and metric | Disposable QuTiP probe of the first example; metric, region, windows, grids, drive-frequency policy, refined dimensions, and candidate thresholds | Choices recorded with measured durations and step counts before drive, frame, or protocol implementation |
| 5B — Drives and frames | Lab-frame carrier drives on all three engines; rotating frames with an explicit rotating-wave approximation; frame comparison in the reference solve | Carrier fixtures pass M1-style agreement, refinement, and physicality checks; rotating and lab frames agree within declared tolerances |
| 5C — `simulate` | Protocol spec, declared needs including device, engine selection, pre-launch reference for GPU runs, summaries and artifacts, automatic verification; Purcell example | The documented simulation runs on CPU and, when declared, on the GPU, and verifies |
| 5D — `sweep` and M2 acceptance | Static and dynamic grid sweeps, plots and summaries, sweep checks; readout sweep | The agreed sweep runs end to end on the GPU, verifies with stated coverage, and reproduces; M2 exit criteria pass |

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

The runner's records are the attempt history; there is no separate ledger store. Jobs write only `result.toml` and `artifacts/`; runner records use separate, read-only paths. `list_runs` scans the runs root. An attempt without a valid `FINISHED` is incomplete: running while its process group is alive, otherwise interrupted. Reconciliation records an interrupted attempt in `runner/reconciled.toml` with its last observed state and usage from `started.toml` marked uncertain; it never writes `FINISHED`, and a result left by such an attempt cannot substantiate a completed result. Reruns get new IDs with the earlier attempt as a parent. Verification status comes from linked verification runs (§4).

### Environment and provenance

`env_hash` covers the `uv.lock` digest, Python and platform, the installed extras, and observed driver, CUDA, and cuQuantum versions. Jobs run through `uv run --locked --exact` with one declared extras set for every job (both GPU extras on the workstation): an exact sync with a different set would reshape the environment between jobs. Separate processes still keep JAX and cuQuantum initialization apart.

Code provenance records the Git revision and dirty state. For uncommitted changes, freeze a patch and untracked source files into `spec/` so the run can be reproduced; digests alone only detect differences. Authored scripts and data are copied into `spec/` with digests. Reproduction reruns the frozen inputs in an environment with the same `env_hash`, using determinism checks where the engine supports them and declared tolerances otherwise.

### Launch and lifecycle

The gate validates specs through `load_system` and the protocol loader, scans authored-script imports against the import-boundary allowlist, and, when the selected engine uses CUDA, checks the driver, cuQuantum, and free GPU memory against the selection estimate. There is no daemon or queue: the runner process supervises its job in a new process group, and separate CLI invocations read and abort through the run records. The job begins computing only after the runner has written `runner/started.toml`.

- Timeouts: the runner timer signals the job's process group and escalates after a documented grace period.
- `abort_run` signals the recorded process group; the supervising runner writes `FINISHED` with `aborted` and exit code `130`.
- If the runner dies, no process writes `FINISHED` and the job may still be running. The next runner reconciles from `runner/started.toml`: a live process group keeps the GPU blocked; a finished one yields `runner/reconciled.toml`.
- The GPU lock is per device and lives outside any runs root, so it excludes across runner instances. The job inherits the lock, which stays held while any inheriting process keeps it open ([`flock` semantics](https://man7.org/linux/man-pages/man2/flock.2.html)); the lock record also names the run and process group, and reconciliation checks that group before reuse. Test a runner killed with `SIGKILL` while its job survives: GPU dispatch is refused until the job exits and reconciliation is recorded.

The lock is a file lock, so lifecycle tests use short CPU jobs and real signals and run in CPU CI.

## 4. Block 4 — corrector

### Package, thresholds, and policy digest

`src/entanglix/corrector/` ships in the same distribution, with a versioned thresholds file as package data, like the engine-selection table. Neither lives in run directories or authored-script workspaces, and research tools cannot write them. The verification policy version is the digest of the corrector code and thresholds: the verification run records it at launch, the job recomputes it when the corrector loads, and a mismatch fails the verification run without a verdict. Version-1 thresholds start from the M1 pinned limits and also set minimum refinement steps, the reference dimension limit, and the evidence-point bound.

### Plans and applicability

Checks are keyed to computation kinds M1 already provides: Lindblad evolution, spectrum, and parameter series. A plan is the runner-validated execution record for one computation: frozen system, evolution or spectral request, and engine decision. Step 4A pins the plan format and adds a built-in plan job that evaluates a plan on a named engine. Reference solves use it, and Block 4 tests verify plan runs directly; Block 5 protocols compile to plans and add metrics, summaries, and plots.

The corrector derives applicable checks from the plan's content and the trusted definition of its kind. For example, dissipators require trace and positivity checks, and a rotating frame requires a lab-frame reference. Labels in specs or results cannot remove checks, and declared refinement steps below the thresholds' minimum are rejected. Runs without a trusted definition, including authored scripts, cannot verify.

### Verdict and verification runs

Verification is an ordinary run whose parent is the source run. Its `result.toml` holds the verdict, `VERIFIED` or `UNVERIFIED`, and a report bound to the source artifact digests and the verification policy version. The report lists each check's scope, tolerance, residual, reference engine and dimension, and outcome (pass, fail, or inapplicable with a reason), and marks each point as evidence-checked or output-checked. `VERIFIED` means every output check passed and every evidence point passed its reference comparison; it certifies nothing about convergence at output-checked points.

A failed verification execution yields no verdict, and the source result remains available only for diagnosis. The runner starts verification after each completed `simulate` or `sweep` run; `compare_reference` starts it on demand. Evidence is reused when plan digests and the policy version match.

`read_run` shows the latest verification attempt's status and whether a verdict is available for the requested policy, defaulting to the current corrector policy, and the source artifact digests. A failed or incomplete latest attempt is shown with diagnostics and no verdict for that attempt. Earlier verdicts remain available as historical evidence labeled with their policy and source digests; they cannot substitute for a missing current verdict. For example, success under policy A followed by a failed attempt under policy B shows B's failure and no B verdict, with A's success retained in the history.

### Dynamics checks and cost

Verification never repeats a run on all three engines:

| Check | Where | Extra solves |
|---|---|---|
| Trace, Hermiticity, and minimum eigenvalue of stored states; top retained-level population as a truncation warning | Every point | None |
| Combined reference solve: an independent CPU engine (QuTiP, or dynamiqs when the run used QuTiP) at tightened tolerances, with each truncated subsystem refined by its declared step, compared on the reported observables and metrics | Evidence points | One per evidence point |

One comparison bounds engine, integration, and truncation differences together; a failure does not say which, and a diagnostic rerun can separate them. A top-level population is a warning, not convergence evidence.

A single run is its own evidence point, so `VERIFIED` requires its combined solve. A standalone simulation therefore requires an additional full-window CPU solve at refined truncation and tightened tolerances, unless matching evidence already exists. For GPU runs that solve runs before launch (§5) and is reused during verification. Step 5A measures this reference cost separately from production execution and includes both in the end-to-end cost when choosing feasible windows.

For sweeps, step 4A pins a deterministic selection rule and a small maximum evidence-point count in the verification policy. The count includes pre-launch corners and later selections. A starting rule is grid corners plus the metric's global minimum and maximum, one point for its steepest change, and one for the worst truncation warning, deduplicated with ties resolved by grid order. Reject grids whose required corners exceed the bound before launch. Report every warning, but select only the bounded worst cases for additional solves. If required evidence cannot be covered within the bound, return `UNVERIFIED` with a refinement recommendation instead of expanding the set automatically. The report names selected points, reasons, and uncovered warnings. Monotonicity applies where the protocol's metric definition predicts it. Seed reproducibility is reported as inapplicable to deterministic solves.

Reference solves run on CPU engines, so verification never waits for the GPU lock. They are plan jobs on a named engine, bounded by the reference dimension limit rather than the dispatch cap, following the M1 separation of CPU convergence references from policy eligibility. Validate that limit for both CPU reference engines with the M1 exact-Liouvillian regression; at dimension 48 the dense Liouvillian is 2304 × 2304. Verifying QuTiP runs needs dynamiqs: CPU-only installs use the `differentiable` extra, and CPU CI covers that path.

Spectral checks compare with analytic or independently constructed references where available and refine cutoffs and retained levels for the reported quantities, following the M1 CPU fixtures. The M1 acceptance helpers move into the corrector, and the M1 tests call them there.

Required Block 4 cases: a truncation that is too small and a tolerance that is too loose each yield `UNVERIFIED`; a corrector file changed between launch and load fails verification; a spec or result claiming a different kind, omitting checks, or declaring a refinement step below the minimum cannot avoid them; an authored-script run cannot verify; repeated verification with matching digests and policy reuses evidence; a newer failed verification is visible without substituting an older successful verdict; warnings across many grid points do not exceed the evidence-point bound.

## 5. Block 5 — feasibility, drives, frames, and protocols

### Feasibility and metric

Step 5A probes the first example (§6) with disposable QuTiP scripts, as step P did for M1, and records the choices the later steps implement:

- The exact dynamics metric, such as the reduced qubit `|ρ01|` at the end of the readout window or a fitted dephasing rate.
- Drive-amplitude range, window durations, output-grid and sweep-grid sizes, and whether the readout drive follows the dressed resonator frequency at each point or stays fixed with its detuning reported.
- Purcell parameters with feasible windows: the §6 starting ranges imply `γ_P = κ(g/Δ)²` of about `1.5e-5` to `1.2e-4/ns`, so `γT = 0.1–1` spans roughly 0.85–68 μs.
- Refined dimensions within the reference limit, and candidate evidence thresholds from a convergence study, including measured durations and step counts in both frames. Record production and full-window refined CPU-reference costs separately, including their combined cost for standalone simulation.

### Drives and frames

Today a drive amplitude is constant or constant within each output interval, engines restart at each boundary, and evolution uses the lab frame. A resonant GHz drive would need output intervals of about 0.01 ns. Step 5B adds:

- **Carrier drives in the lab frame:** a declared envelope times `cos(2π f t + φ)`, with `f` in GHz, `t` in ns, and `φ` in radians. Each engine evaluates it continuously through its time-dependent coefficient path: QuTiP coefficients, dynamiqs modulated operators, and cuDensityMat callbacks through qutip-cuquantum, which evaluates them on the CPU at every step. Constant and piecewise-constant envelopes come first; other envelopes arrive with the example that needs them.
- **Rotating frames:** declared in the evolution input as frame frequencies applied to excitation-number operators in each subsystem's retained eigenbasis; explicit-operator models must declare theirs. The transformation is an explicit numeric operation on the constructed operators through the common engine interface, not a general symbolic framework. A declared rotating-wave approximation removes counter-rotating terms, and provenance records the frame and the removed terms. Comparisons across frames use frame-invariant observables or states transformed back to the lab frame.
- **Frame comparison:** for rotating-frame runs, the combined reference solve uses the lab frame, so the same solve also bounds the rotating-wave approximation.

The pinned adaptive solvers stay; there are no stiff or specialized integrators. The rotating frame is the cost lever, and the lab-frame reference bounds its approximation. Carrier fixtures need M1-style acceptance against independent references on all three engines. Differentiable controls extend to carrier envelopes only where gradient tests cover them; otherwise `evolve_jax` rejects them.

### `simulate`

A `simulate.toml` protocol spec names the system, frame, initial state, output grid, drives, observables, solver tolerances, refinement steps, declared needs, and an optional timeout, and compiles to one plan. Declared needs are `gradients` and `device`. A new selection-table version maps a declared `device = "cuda"` without gradients to cuDensityMat; without a declared device, non-gradient requests stay on QuTiP CPU. The table records needs, not performance, and the dispatch cap stays at 30.

GPU launches follow the contract's cheap-to-expensive funnel: the combined CPU reference solve runs first as a linked run, the GPU run proceeds only when that solve passes its output checks, and verification compares against it. Results include observable and metric summaries in `result.toml`, stored states and observables in `artifacts/`, and plots. Plots use matplotlib, today a transitive scqubits dependency; declare it directly and add it to the import-boundary allowlist.

### `sweep`

A `sweep.toml` names a base `simulate` or spectral spec and grid axes over system parameters, such as a constructor's `josephson_energy`, and protocol quantities, such as a drive amplitude. Static sweeps compute spectral quantities; dynamic sweeps evolve each point, undriven or driven, in either frame. One run covers one sweep: points run serially inside the job, completed points remain in `artifacts/` if the run is interrupted, and a rerun is a new linked run that recomputes the grid. For a GPU sweep, the pre-launch reference covers the grid corners; verification adds selected metric-feature and truncation-warning points within the total evidence-point bound (§4) after the sweep and reuses the corner solves.

Summaries and plots show the metric over the grid, per-point top-level populations, and each point's verification coverage, so a user can choose a narrower or finer grid and rerun with the earlier sweep as parent. Batch-versus-serial checks are needed only before batching is advertised.

## 6. First example

One qubit–resonator model serves the documented simulation, the agreed dynamics sweep, and a static companion sweep. Values are starting inputs to the 5A probe. Production stays within the dispatch cap at 3 retained transmon levels × 10 resonator levels = 30; refinement to about 4 × 12 = 48 runs under the reference limit.

| Item | Proposal |
|---|---|
| Model | Transmon with `EC = 0.25 GHz` and `EJ = 11–16 GHz` (qubit frequency about 4.4–5.4 GHz); 7 GHz resonator; charge–position coupling chosen for `g ≈ 0.1 GHz`; resonator loss `κ ≈ 0.01–0.03/ns` |
| Documented simulation | Purcell decay: qubit excited, resonator empty, no drive. Excited population over a feasible 5A window, compared with the `κ(g/Δ)²` trend, in both frames |
| Agreed dynamics sweep | Measurement-induced dephasing: qubit in `(|0⟩ + |1⟩)/√2`, resonator empty, constant resonator drive at the readout frequency; the 5A coherence metric over the readout window. Axes `EJ` × drive amplitude; production in a common rotating frame at the drive frequency on the GPU, with lab-frame reference solves at evidence points |
| Region | Dispersive regime with mean photon number below about 1, far below the critical photon number `(Δ/2g)²` |
| Static companion | Dispersive shift `χ` over the same `EJ` values, from the spectrum |
| Expected trends | Dephasing grows with drive amplitude squared in the dispersive limit, and `χ` from the static sweep predicts the dynamic trend. These are scientific cross-checks in the example; the corrector checks only the monotonicity the metric definition declares |

The example states its assumptions: a weak-measurement regime chosen to fit the dispatch cap, not typical readout photon numbers. Reference agreement does not establish device accuracy.

## 7. Dependencies, layout, and validation

No new runtime dependency is expected beyond declaring matplotlib: the runner uses the standard library for processes, signals, and file locks.

Suggested locations, created as code arrives:

```text
src/entanglix/runner.py          # records, gate, launch, lifecycle, GPU lock, list/read/abort
src/entanglix/corrector/         # definitions, checks, thresholds.toml, verification job
src/entanglix/protocols/         # plan job, simulate, sweep
src/entanglix/evolution.py       # carrier drives and frames in the evolution input
tests/probes/m2_feasibility.py   # disposable 5A probe and its retained results
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

Recorded in [ADR-0003](adr/0003-runs-verification-and-dispatch.md), with matching edits to contract §2, §3, and §6 and to the roadmap and checklist:

1. Run directories are the attempt history: runner-owned `run.toml`, start and reconciliation records, and `FINISHED`, with no separate ledger store, `index`, or `latest`. amicode's layout and terminal semantics are adopted without Julia fields; Inspector compatibility waits for Block 12.
2. Verification is a separate linked run whose `result.toml` holds the verdict and a report bound to source digests and the policy version; protocol results carry no verdict.
3. Dynamics verification has scoped coverage: output checks at every point, and one combined reference solve per evidence point on an independent CPU engine at tightened tolerances and refined truncation. A single run is its own evidence point; sweeps use a bounded, deterministically selected set. `VERIFIED` certifies nothing about convergence at output-checked points, and reports state coverage.
4. Corrector reference solves have their own dimension limit, validated like the M1 exact-Liouvillian regression; the dispatch cap stays at 30.
5. A declared `device` need routes non-gradient requests to cuDensityMat. Engine efficiency and the corresponding Block 5 checklist items move to a cuDensityMat usage study, tracked as a prerequisite for automatic performance-based dispatch.
6. Lab-frame carrier drives and rotating frames are M2 capabilities, using the pinned adaptive solvers and explicit numeric transformations.
7. The first example is the qubit–resonator readout model in §6, with feasibility and metric choices made in 5A before implementation.

Open, decided in the named step: run-record fields (3A); plan format, reference limit value, and evidence-point selection rule and bound (4A); metric, region, windows, grids, drive-frequency policy, refined dimensions, and evidence thresholds (5A); frame declaration and rotating-wave rules (5B); protocol spec syntax (5C); CLI command names.

## 9. Exit evidence and handoff

[Checklist M2](checklist.md#m2--verified-local-sweep-blocks-35) is the sole acceptance list; this guide adds no completion gates. Acceptance runs are themselves Entanglix run directories: retain the cited runs and their verification runs, and link them with test commands and results, as M1 retained its acceptance results. Publish run records, exit codes, and recovery in `docs/operations/runs.md`, frames and carrier drives in `system-spec.md`, protocol usage in an operational protocol reference, and available commands in the root README. Operational documents stand alone without linking to planning docs.

The investigation decisions after M2 build on the runner, run history, verification, and sweep outputs. Block 6a adds investigation revisions to run records and admission checks before launch; Block 6b wraps the backing operations in MCP. The cuDensityMat usage study can proceed once M2 runs exist and updates the selection table only from its measurements.

Before retiring this guide, retain consequential design rationale in ADRs and necessary conventions and instructions in operational docs.
