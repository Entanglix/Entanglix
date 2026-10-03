# Entanglix — Harness Contract

**Date:** 2026-09-28 · **Status:** v0. Normative for each capability when available. See [README.md](README.md) for current usage and availability.

**Audience:** research users, agents, and integrators. This contract is a retained operational part of the harness: it defines shared interfaces and execution guarantees. [Research-agent guidance](docs/research-agent.md) is a draft pending runtime integration; [ADRs](docs/adr/) retain decision history for optional consultation.

This contract defines required interfaces and guarantees enforced through supported execution paths under the cooperative trust model (§7). Task-specific research procedures belong in operational skills when available. Required configuration must be explicit; draft recommendations do not supply defaults.

## 1. Objects and interfaces

| Object | Definition |
|---|---|
| System | `system.toml`: component-defined circuit, coupled subsystems, or explicit operators; truncation, drive operators, dissipators, parameter values and priors, units |
| Protocol spec | `simulate`, `sweep`, `control`, or `calibrate`; declares engine needs, observables/metrics, and any configured execution limits |
| Investigation | Human-started research question with versioned scope, assumptions, evidence/progress requirements, review/stop criteria, optional limits, and recorded user decisions (§4) |
| Run | One protocol execution with frozen inputs, artifacts, and parent/investigation provenance |
| Result | Metrics, provenance, and actual engine/solver; distinct from execution status. Check outcomes come from linked verification runs (§3) |
| Finding | Interpretation, progress assessment, or next-step/stop proposal referencing its investigation revision, runs, the checks run and their outcomes, assumptions, uncertainty, and limitations |
| Run ledger | Append-only history of every attempt, its lineage, outcome, and resource use |
| Catalog entry | Human-promoted result retaining source-run and parent-entry lineage |
| Device model | Current device parameters; updated by humans from calibration proposals |

```text
SolverBackend
  capabilities() -> {gradients: bool, constraints: [...], gpu: bool, problem_kinds: [...]}
  solve(problem_spec, system, budget) -> Solution{pulse, objective, certificate, trace}
Integrator
  capabilities() -> {name: str, device_kind: "cpu" | "cuda",
                     differentiable: bool, open_system: bool, batched: bool}
  evolve(system, request: EvolutionInput) -> EvolutionResult
```

`differentiable` means evolution exposes results preserving derivatives through supported continuous controls with a fixed constructed system and basis.

Evolution requests declare carrier frequencies in GHz and phases in radians, and any rotating frame with an explicit rotating-wave approximation. Returned states and observables use the lab basis; diagnostics record the frame and removed frequency sectors. Unsupported frame or control requirements are rejected.

Control uses amicode's ProblemSpec OSS schema; `solver.backend` records a string identifier. Entry-point groups are `entanglix.solvers` and `entanglix.integrators`. The repo ships open implementations: a GRAPE-style baseline and a Piccolo adapter through `amico-run`. Private implementations attach through discovery; repo code never imports them directly, and tests run with open implementations only.

## 2. Execution and artifacts

The runner selects an engine from a versioned table using declared protocol needs (`gradients` and `device`). Research agents cannot select an engine directly. The table records needs, not performance; automatic performance-based dispatch requires measured CPU/GPU crossover and peak memory.

| Gradients | Declared device | Engine |
|---|---|---|
| No | None | QuTiP CPU |
| No | `cuda` | cuDensityMat through `qutip-cuquantum` |
| Yes | Any | dynamiqs |

The harness imposes no dimension cap, and no memory estimate precedes a launch; tested sizes are acceptance evidence, not a ceiling. An observed allocation failure is reported as out of memory, including one during model construction at the launch gate, before a run exists. A job killed by a signal the runner did not send is reported as that signal, with exhausted memory as a possible cause only. If the runner dies as well, the interruption and reconciliation rules apply. Batched requests before batching is validated are refused. A gradient request may use cuDensityMat only through a declared gradient-free fallback. Engine and solver fallbacks must satisfy the declared constraints and be recorded. Missing plugins or unsupported requirements produce actionable configuration errors unless a compatible fallback is declared. cuDensityMat and JAX exchange data through DLPack / CuPy; their gradients do not compose.

```text
research_runs/<system_id>/<run_id>/
  run.toml           # runner writes first: run_id, parent run IDs, spec digests,
                     # code provenance, env_hash, engine, device, executor
  runner/            # runner-owned, read-only: started.toml before the job computes
                     # (start, GPU allocation, process group); reconciled.toml after runner loss
  spec/              # frozen system and protocol specs, authored code and data
  run.log            # job output and diagnostics
  result.toml        # job writes atomically: metrics, actual engine/solver; no check outcomes
  artifacts/         # small arrays and plots; large arrays referenced by path + digest
  FINISHED           # runner writes last: status, exit_code, end time, usage
```

- The runner writes each applicable record once; jobs write only `result.toml` and `artifacts/`. `FINISHED.status` is `completed`, `failed`, or `aborted`. Without a valid terminal marker, a run is incomplete and cannot substantiate a completed scientific result. After runner loss, reconciliation records the last observed state and uncertain usage in `runner/reconciled.toml` and never writes `FINISHED`. Completed execution does not show that a result is accurate; requested checks provide evidence within their scope (§3).
- `result.toml` is published by temporary-file rename. Frozen inputs include authored code and data references with versions or content digests, and a patch of uncommitted code. Record seeds, solver settings, and `env_hash` covering pinned CUDA, cuQuantum, driver, and Python packages. Jobs sharing an environment run exactly synced with the same declared extras.
- A sweep is one run. Its points run inside the job, completed points remain in `artifacts/` after interruption, and a rerun is a new linked run.
- Units and time-grid conventions are explicit; no implicit conversion. Every completed scientific result records its truncation and solver settings; convergence is checked only on request (§3). Reproduction uses determinism checks where supported and declared tolerances otherwise.
- Exit codes follow amicode: `0` completed; `130` aborted; `64` usage/configuration/orchestrator fault; otherwise the job's return code. Run directories follow amicode's layout, terminal marker, atomic result, run-ID form, and exit codes; amicode's Julia- and control-specific fields and Run Inspector files (`index`, `latest`) are not yet provided.
- Abrupt termination retains diagnostics and incomplete-attempt history. Recovery cannot fabricate success; a rerun receives a new ID linked to the prior attempt. Each executor provides a documented recovery/rerun path.
- Local and cluster executors share these semantics. Large arrays remain on execution storage; mirrored artifacts are integrity-checked. Lost contact never implies completion. The ledger preserves uncertain outcomes and resource accounting until reconciled.

## 3. Verification

The corrector is an independent, deterministic package, hashed at load, with versioned thresholds configured outside the candidate workspace. Research tools cannot replace the corrector, its thresholds, or its reports. A hash mismatch fails execution. Verification never depends on a private solver.

The harness runs what the user requests. A protocol run computes and stores its results, and no check follows unless the user asks. Verification is a set of built-in checks that the user requests on a completed run. Each check targets one error mode and reports its own outcome: pass, fail, or inapplicable with a reason. There is no run-level verdict. A launch, its dry-run preview, and `read_run` list the checks relevant to a run, with the error mode each one catches and any that cannot run, so the user can choose.

Verification is a separate run linked to its source run. Its result holds a report bound to the source artifact digests and the corrector policy version, the digest of the corrector code and thresholds; protocol results carry no check outcomes. Reads list a run's verification runs with their status, policy version, and each check's outcome. A failed or interrupted verification run reports no outcome; reports under an earlier policy remain history labelled with that policy.

| Protocol | Built-in checks |
|---|---|
| Dynamics `simulate` / `sweep` | Integrity: trace, hermiticity, and positivity of stored states, the first stored state matching the declared initial state, and reported values consistent with the stored states. Segments: short windows restarted from stored states in the run's frame, on CPU at tightened tolerances and refined truncation, which detect local disagreement. Diagnostics: top retained-level population and static shifts, reported without a pass or fail outcome |
| Spectral `simulate` / `sweep` | Truncation refinement of the reported quantities |
| `sweep` (additional check) | A declared expectation, such as a metric's direction along a sweep axis |
| `control` | Pulse rollout on a different engine; comparison on the same ProblemSpec with an open solver; independently calculated metric and constraint satisfaction for each candidate |
| `calibrate` | Fitting receives training data only; corrector evaluates holdout data without refitting; identifiability report using covariance or profile likelihood |

The corrector derives the checks relevant to a run from trusted protocol/metric definitions and the validated execution plan, not from labels in specs or results. A check requested where it does not apply is reported as inapplicable. A check's prerequisites, such as a declared refinement or the evidence-point bound, never restrict a run; a check that lacks one is reported as inapplicable with that reason. A check request may supply a replacement refinement declaration, which is frozen in the verification run and leaves the source run unchanged. A declared expectation is checked as the user wrote it, and its failure is reported as a violated declared expectation. Runs without a trusted definition have no built-in checks. The `control` and `calibrate` rows follow the same on-request rule when those protocols are built. Reports identify:

- each check's scope, tolerance, residual, and outcome;
- the coverage of each segment and, in a sweep, of each point;
- the engine, dimension, and refinement declaration of any refined solve;
- inapplicable checks, with reasons.

A passing check establishes only what its scope states:

- Passing segments show local agreement over their coverage. A restart inherits any error already accumulated. Segments on the run's own engine are convergence comparisons, not independent validation of that engine.
- No built-in check measures error accumulated over the whole window or the accuracy of a rotating-wave approximation, and check listings and dynamics reports say so. A user who needs that evidence requests another solve and compares the runs.
- In a sweep, segment and refinement checks run at evidence points chosen by a deterministic rule with a maximum count set by the corrector policy. They say nothing about unsampled points, and reports mark each point's coverage.

Refined solves run on CPU. Numerical agreement does not establish device-model accuracy. Cross-solver comparisons report discrepancies; distinct pulses or local optima alone do not invalidate a candidate passing independent checks. Promotion follows §6.

## 4. Bounded investigations

The runner enforces the investigation lifecycle through supported CLI/MCP paths. Standalone runs retain their own provenance and any configured limits without requiring an investigation or grant.

- **Scope and start:** record the question, assumptions, parameter region, permitted changes/protocols, evidence requirements, progress checkpoints, review/stop criteria, and optional limits. A scalar objective is optional. An agent may draft an investigation; a user action starts it.
- **User decisions:** the CLI/runtime user-input path records the initial scope, broader changes, review releases, and changes to configured ceilings. Research-tool arguments and agent-authored notes cannot create these events. Routine revisions within recorded scope need no renewed approval; no separate grant issuer is required.
- **Revisions:** investigation revisions are immutable and runs identify their dispatch revision. Resume, reruns, and session changes preserve unsuccessful attempts, resource use, findings, and outstanding reviews.
- **Admission:** check active revision, supported protocol, mechanically checkable scope constraints, review/stop state, and any run/investigation limits before dispatch. Free-text research aims are recorded context, not machine-certified scientific judgments.
- **Progress:** progress checkpoints, not preset budgets, are the primary control on continuing work. A checkpoint is declared as a count of finished attempts, including failed and aborted runs, or an elapsed interval; when one is due, the runner refuses further dispatch until a progress assessment is recorded. Assessments reference available results, check outcomes, constraint/convergence evidence, observed cost, learning or uncertainty, and a continue/refine/stop rationale. The runner does not certify that continuation is worthwhile. Findings may recommend stopping with resources still available.
- **Stopping and resume:** a running job's live trace is readable through `read_run`, and `abort_run` stops a stalled run without waiting for it to finish. Stopping an investigation closes admission and requests cancellation of active jobs. Record a reason distinguishing evidence completion, insufficient progress, unresolved model/strategy limitations, user cancellation, configured limits, interruption, and review handoff. Resume retains accounting and requires outstanding user reviews to be released. Restart files are versioned artifacts where supported; restarted computation gets a new linked run ID.
- **Findings:** link scientific claims to the checks that ran and their outcomes, and state which relevant checks did not run; retain diagnostics separately when execution or verification fails. Early-stop reports preserve the best available results, limitations, and suggested next experiments. Negative/exploratory findings can satisfy declared reporting criteria without a finished device.
- **Composition:** sweep/search requires no control, calibration, or catalog. Control accepts an existing system model; calibration accepts its declared data. Co-design links device/model, control, and evaluation runs with shared review state, usage accounting, and any configured limits; declare constraints, uncertainty, and comparable evaluation conditions.

Usage is always recorded from executor reports (§7). Limits are optional, and the absence of a limit is recorded explicitly. Unknown limit names are rejected.

| Limit | Scope | Meaning and enforcement |
|---|---|---|
| `timeout_seconds` | Run | Executor-native timeout: a process timer on the workstation, the scheduler time limit on a cluster |
| `deadline` | Investigation | Absolute calendar timestamp enforced by the executor, including queued jobs: no new dispatch or job start after it; cancel pending jobs and terminate active jobs at expiry. Cap each run's timeout at the time remaining when it starts |
| `gpu_seconds` | Investigation | Allocated GPUs × allocation time, including verification, retries, and failed work |
| `job_count` | Investigation | Admitted submissions, including queued jobs, retries, and verification jobs |

Before dispatch, the runner compares recorded usage with any limits. When `gpu_seconds` is limited, every job needs a timeout. Admission includes the proposed job and all admitted but unsettled jobs, including queued and running jobs, at their maximum commitment (timeout × requested GPUs). The runner atomically checks allowances and records the commitment before submitting the job, using its existing records rather than a separate reservation service. Confirmed final usage replaces the commitment; usage the executor cannot confirm remains uncertain and never returns allowance. Lost contact never permits conflicting GPU dispatch.

Executors document timeout resolution and termination grace, and record actual usage and any overrun. A submission-time duration alone does not enforce an investigation deadline across queueing; unsupported deadline enforcement is rejected.

## 5. Tools and permissions

CLI and MCP share backing operations and expose available capabilities. Research-tool requests are distinct from user-input events; approval claims in request fields are rejected. Unavailable protocols cannot be dispatched. These application-level checks operate under §7, without claiming an authenticated OS boundary around an unrestricted shell.

| Operation | Permission class | Effect |
|---|---|---|
| `validate_system` | Read-only | Check the system schema and model adapter inputs |
| `list_capabilities` | Read-only | Report installed protocols, supported features, and interface versions |
| `list_runs` / `read_run` | Read-only | Inspect artifacts, all attempt states, relevant checks and their outcomes, and the live trace of running jobs |
| `list_investigations` / `read_investigation` | Read-only | Inspect revisions, findings, history, measured usage, remaining configured allowances, and review/stop state |
| `list_catalog` / `read_entry` | Read-only | Inspect promoted entries and provenance |
| `simulate` / `sweep` / `control` / `calibrate` | Controlled job | Execute through the runner, enforcing any configured limits |
| `run_script` | Controlled job | Execute an authored script through the runner, enforcing any configured limits |
| `rerun_script` | Controlled job | Replay frozen script inputs as a new attempt linked to the prior run |
| `rerun_simulation` | Controlled job | Replay frozen simulation inputs as a new linked attempt, recording current code/environment provenance |
| `check` | Controlled job | Run the named built-in checks on a completed run (§3) |
| `abort_run` / `stop_investigation` | Controlled job | Cancel work; investigation stop also closes admission and records its reason |
| `create_investigation` / `revise_investigation` | Controlled write | Draft a new investigation or publish in-scope revisions; broader changes remain proposals until a user decision |
| `record_finding` | Controlled write | Append interpretation and evidence references |
| `reconcile_run` | Controlled write | Record an interrupted attempt's uncertain state only after ownership and process-group checks permit it |
| `register_result` | Controlled write | Stage a candidate for human promotion |
| `propose_device_update` | Controlled write | Write a calibration proposal |
| Investigation start/resume, scope/limit changes, review release; `promote`; device-model apply | Human action | Record user decisions, release named reviews, promote eligible results (§6), or apply proposals |

Human actions use the operator CLI or runtime user-input path and record the action source and affected revision. The research-tool surface cannot synthesize these events, promote results, or apply device updates. The agent runtime's permission settings block the human-action commands, so an agent cannot run them from its shell by accident. This separates supported workflow roles; it is not authentication, and stronger identity and credential isolation is deferred deployment work.

Read-only operations do not mutate research state. Controlled jobs use the runner/executor. Controlled writes are limited to staged outputs and runner-managed investigation records; they cannot change policy, verification thresholds, configured ceilings, promoted entries, or device models through the research API. Typed tools validate structured scope constraints; data and notes cannot change execution policy.

## 6. History, audit, and promotion

The runner is the single writer for history and publishes immutable events atomically, one file per event. For runs, the ledger events are the runner-owned records in each run directory (§2), and check outcomes come from linked verification runs (§3). Concurrent requests cannot overwrite or lose events. Every attempt retains lineage, including incomplete, failed, aborted, and unpromoted work. Trial counts include unsuccessful attempts within the declared investigation or hypothesis family; session changes do not reset them.

| Record | Required content |
|---|---|
| Run ledger event | Run ID; spec/code/environment provenance; actual engine/solver when known; parent run IDs; outcome; resource use; check outcomes when available; investigation revision and source catalog IDs when applicable |
| Tool-call audit | Timestamp, session ID, tool, permission class, argument/result hashes, outcome, and linked investigation revision/run IDs |
| Session audit | One entry per agent session: runtime name/version, model ID, skill names/versions, and the path to the runtime's own transcript when it keeps one |
| User-decision event | Timestamp, operator or runtime user-input source, affected investigation revision, start/resume, scope/limit changes, or named review release; separate from agent proposals |

Audit records are append-only through supported APIs and stored outside candidate run outputs; these are provenance guarantees, not protection against a hostile OS user. Supported tools expose histories without catalog promotion; unknown outcomes remain explicit until reconciled.

`register_result` writes to `catalog/staging/`, referencing existing evidence. Human `entanglix promote <staged-id>` records the promoter, source runs, and `parent_entry_ids`. The entry records the checks run on its source runs, their outcomes, and the relevant checks that did not run, and claims citing it state that scope. A result with a failed check remains available for diagnosis but cannot be promoted or cited as design evidence. A violated declared expectation is the exception: the expectation may be wrong, so it does not block promotion and the entry records it. Reuse records both the source entry and the new run. Continuing an authorized investigation requires no promotion.

`propose_device_update` writes next to the calibration run. Applying it to the device model is a separate human command. Research tools can stage proposals and findings but expose neither promotion nor device-model write-back.

## 7. Execution environments and trust boundaries

Development and testing run on a single-owner workstation; research runs may also use a cluster through its scheduler. Both follow a cooperative trust model: one owner, one coordinating agent, and scripts that use the supported runner path. This contract does not claim containment of arbitrary code, protection from an agent deliberately bypassing tools through its shell, or tamper-proof history against the owner. Coding-agent development permissions remain separate from research workflow roles.

Executors provide the same guarantees from facilities that already exist:

| Need | Workstation | Cluster |
|---|---|---|
| GPU exclusivity | Runner lock: one active GPU job by default, including across runner instances; a job may batch calculations; unknown process state blocks reuse until reconciled | Scheduler GPU allocation |
| Timeout and cancellation | Runner timer; signal to the job's process group and descendants | Scheduler time limit and cancellation |
| Usage | Measured start/end times and assigned GPUs | Scheduler accounting |
| Outputs | Local run directory | Execution storage; small artifacts mirrored back with integrity checks |

| Boundary | Requirement |
|---|---|
| Launch checks | Runner validates specs, scans authored imports, and checks the environment before launch; no scientific check precedes a launch |
| Run ownership | Unique directories, frozen inputs, and assigned outputs; supported job paths cannot overwrite another run |
| Scientific records | Runner owns ledger/audit publication; research tools cannot modify the corrector policy or write check reports |
| Human actions | Research APIs stage proposals; human actions follow §5 |
| Agent GPU access | The agent session sees no GPU by default (`CUDA_VISIBLE_DEVICES=""`); only runner-launched jobs do, so ad hoc analysis cannot take memory from a running job. On a cluster, the agent does not run on GPU nodes |

Configured limits and cancellation apply to runner-launched scripts as well as built-in protocols. Static scanning is preflight, not a security sandbox. Restart files are used only where a protocol supports them. Entanglix builds no job queue, reservation service, grant issuer, or container layer; it reuses the process, scheduler, and agent-runtime facilities above.

A deployment requiring resistance to bypass or execution of untrusted code must first define and test stronger identity, process, filesystem, credential, network, and device boundaries. Those guarantees require a separate deployment decision and conformance evidence; cooperative tests cannot establish them.

## 8. Versioning and conformance

Public schemas and CLI/MCP interfaces declare versions and compatibility expectations. Incompatible inputs fail with actionable diagnostics. Skill metadata declares `name`, `description`, and `version`; runtime and skill versions are pinned in CI and recorded in audit. Instructions and skills cannot substitute for tool-enforced invariants.

Changes to normative invariants require a retained ADR and corresponding schema/interface versions and conformance updates where affected.

Conformance evidence must cover the observable guarantees below. Implementations add concrete test/schema references as they are built; these requirements do not claim existing tests.

| Contract surface | Responsible component | Required evidence |
|---|---|---|
| Objects and interfaces (§1, §5) | Schemas, adapters, CLI/MCP | Validation, transport consistency, capability checks, open-only/plugin-boundary tests |
| Execution (§2) | Runner and executors | Reproduction, atomic artifacts, no refusal of a request for its size, allocation failures reported as out of memory and unexplained signal deaths as their signal, cancellation/interruption recovery including runner loss while the job survives, amicode run-directory layout and ProblemSpec compatibility |
| Verification (§3) | Corrector | No check without a request, convergence failures caught by the requested check, relevance from trusted definitions and inapplicable requests, per-check outcomes with scope, a terminated verification run reporting no outcome, declared-expectation checks, a run unrestricted by check prerequisites, a replacement refinement on a check request, bounded evidence-point coverage, report presentation after a failed newer attempt, independent constraint checks, training/holdout data separation, hash-mismatch rejection |
| Investigations (§4) | Runner and tools | User-start/review event handling, rejection of tool-authored approvals, checkpoint gating including failed/aborted attempts, early-stop reports, no-limit and limit cases, atomic admission accounting for proposed/queued/running jobs, deadlines across queueing and execution, timeout resolution/grace and recorded overruns, rejection of unknown limits, retry/verification usage, cancellation and resume accounting |
| History and promotion (§6) | Ledger, audit, catalog | Atomic concurrent events, failed-attempt lineage, recorded check scope, refusal of a failed check other than a violated declared expectation, and human-promotion boundaries |
| Execution environments (§7) | Runner, executors, runtime adapter | Run/GPU ownership, conflicting-launch rejection, process cancellation, controlled history writes, blocked human-action commands, GPU-hidden agent session, and rejection of policy/verification bypass through supported tools; one executor test suite passes against the local executor and a stubbed scheduler; no OS-level containment claim |
