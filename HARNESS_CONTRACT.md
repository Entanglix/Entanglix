# Entanglix — Harness Contract

**Date:** 2026-09-28 · **Status:** v0. Normative for each capability when available. Version 0.1.0 provides the CPU foundation only; research execution and MCP operations described below are not yet available. See [README.md](README.md) for current usage and availability.

**Audience:** research users, agents, and integrators. This contract is a retained operational part of the harness: it defines shared interfaces and execution guarantees. [AGENTS.md](AGENTS.md) provides agent entry instructions; [ADRs](docs/adr/) retain decision history for optional consultation.

This contract defines required interfaces and guarantees enforced through supported execution paths under the cooperative trust model (§7). Task-specific research procedures belong in operational skills when available. Required configuration must be explicit; draft recommendations do not supply defaults.

## 1. Objects and interfaces

| Object | Definition |
|---|---|
| System | `system.toml`: component-defined circuit, coupled subsystems, or explicit operators; truncation, drive operators, dissipators, parameter values and priors, units |
| Protocol spec | `simulate`, `sweep`, `control`, or `calibrate`; declares engine needs, observables/metrics, and any configured execution limits |
| Investigation | Human-started research question with versioned scope, assumptions, evidence/progress requirements, review/stop criteria, optional limits, and recorded user decisions (§4) |
| Run | One protocol execution with frozen inputs, artifacts, and parent/investigation provenance |
| Result | Metrics, provenance, actual engine/solver, and verification verdict; distinct from execution status |
| Finding | Interpretation, progress assessment, or next-step/stop proposal referencing its investigation revision, runs, verification status, assumptions, uncertainty, and limitations |
| Run ledger | Append-only history of every attempt, its lineage, outcome, and resource use |
| Catalog entry | Human-promoted result retaining source-run and parent-entry lineage |
| Device model | Current device parameters; updated by humans from calibration proposals |

```text
SolverBackend
  capabilities() -> {gradients: bool, constraints: [...], gpu: bool, problem_kinds: [...]}
  solve(problem_spec, system, budget) -> Solution{pulse, objective, certificate, trace}
Integrator
  capabilities() -> {differentiable: bool, open_system: bool, batched: bool}
  evolve(system, controls, grid) -> trajectory
```

Control uses amicode's ProblemSpec OSS schema; `solver.backend` records a string identifier. Entry-point groups are `entanglix.solvers` and `entanglix.integrators`. The repo ships open implementations: a GRAPE-style baseline and a Piccolo adapter through `amico-run`. Private implementations attach through discovery; repo code never imports them directly, and tests run with open implementations only.

## 2. Execution and artifacts

The runner selects an engine from a versioned table using declared protocol needs and configured size/memory limits. Research agents cannot select an engine directly.

| Gradients | Size | Engine |
|---|---|---|
| No | Reference-sized | QuTiP CPU |
| No | Large / batched | cuDensityMat through `qutip-cuquantum` |
| Yes | Fits JAX memory budget | dynamiqs |
| Yes | Does not fit | Refuse; use batched cuDensityMat only through a declared gradient-free fallback |

Engine and solver fallbacks must satisfy the declared constraints and be recorded. Missing plugins or unsupported requirements produce actionable configuration errors unless a compatible fallback is declared. cuDensityMat and JAX exchange data through DLPack / CuPy; their gradients do not compose.

```text
research_runs/<system_id>/<run_id>/
  run.toml           # runner writes first: run_id, spec hashes, code_version,
                     # env_hash, engine, executor
  spec/              # frozen system and protocol specs
  run.log            # AMICODE_ITER iter=<n> f=<obj> …; diagnostics
  iter_<N>.png       # plots where applicable
  result.toml        # protocol writes atomically: metrics, solver.backend, verdict
  artifacts/         # small arrays; large arrays referenced by path + digest
  FINISHED           # runner writes last: status, exit_code
```

- `FINISHED.status` is `completed`, `failed`, or `aborted`. Without a valid terminal marker, a run is incomplete and cannot substantiate a completed scientific result. Completed execution may still yield an `UNVERIFIED` result.
- `result.toml` is published by temporary-file rename. Frozen inputs include authored code and data references with versions or content digests. Record seeds, solver settings, and `env_hash` covering pinned CUDA, cuQuantum, driver, and Python packages.
- Units and time-grid conventions are explicit; no implicit conversion. Every completed scientific result reports truncation and convergence. Reproduction uses determinism checks where supported and declared tolerances otherwise.
- Exit codes follow amicode: `0` completed; `130` aborted; `64` usage/configuration/orchestrator fault; otherwise the job's return code. Run artifacts must satisfy the adopted amicode run-directory interface.
- Abrupt termination retains diagnostics and incomplete-attempt history. Recovery cannot fabricate success; a rerun receives a new ID linked to the prior attempt. Each executor provides a documented recovery/rerun path.
- Local and cluster executors share these semantics. Large arrays remain on execution storage; mirrored artifacts are integrity-checked. Lost contact never implies completion. The ledger preserves uncertain outcomes and resource accounting until reconciled.

## 3. Verification

The corrector is an independent, deterministic package, hashed at load, with versioned thresholds configured outside the candidate workspace. Research tools cannot replace the corrector, thresholds, or verdicts. A hash mismatch fails execution. It verifies every completed protocol; failed or interrupted attempts retain diagnostics without a successful verdict. Verification never depends on a private solver.

| Protocol | Required checks |
|---|---|
| Dynamics `simulate` / `sweep` | Reference-size agreement across QuTiP CPU, cuDensityMat, and dynamiqs; step-size and truncation convergence; trace, hermiticity, positivity |
| Spectral `simulate` / `sweep` | Reference spectra or analytic cases where available; truncation convergence; checks applicable to the declared observable |
| `control` | Pulse rollout on a different engine; comparison on the same ProblemSpec with an open solver; independently calculated metric and constraint satisfaction for each candidate |
| `calibrate` | Fitting receives training data only; corrector evaluates holdout data without refitting; identifiability report using covariance or profile likelihood |
| Design `sweep` (additional checks) | Reproducibility across seeds; metric monotonicity where physics predicts it |

The corrector derives applicability from trusted protocol/metric definitions and the validated execution plan, not agent-written labels or exemptions. Unknown or inconsistent definitions cannot receive a verified verdict. Design-sweep checks stack with applicable dynamics or spectral checks. Reports identify checks, tolerances, residuals, reference size, and reasons for inapplicability. Reference-size agreement does not certify the full production problem; numerical agreement does not establish device-model accuracy. Cross-solver comparisons report discrepancies; distinct pulses or local optima alone do not invalidate a candidate passing independent checks. Promotion eligibility follows §6.

## 4. Bounded investigations

The runner enforces the investigation lifecycle through supported CLI/MCP paths. Standalone runs retain their own provenance and any configured limits without requiring an investigation or grant.

- **Scope and start:** record the question, assumptions, parameter region, permitted changes/protocols, evidence requirements, progress checkpoints, review/stop criteria, and optional limits. A scalar objective is optional. An agent may draft an investigation; a user action starts it.
- **User decisions:** the CLI/runtime user-input path records the initial scope, broader changes, review releases, and changes to configured ceilings. Research-tool arguments and agent-authored notes cannot create these events. Routine revisions within recorded scope need no renewed approval; no separate grant issuer is required.
- **Revisions:** investigation revisions are immutable and runs identify their dispatch revision. Resume, reruns, and session changes preserve unsuccessful attempts, resource use, findings, and outstanding reviews.
- **Admission:** check active revision, supported protocol, mechanically checkable scope constraints, review/stop state, and any run/investigation limits before dispatch. Free-text research aims are recorded context, not machine-certified scientific judgments.
- **Progress:** progress checkpoints, not preset budgets, are the primary control on continuing work. A checkpoint is declared as a count of finished attempts, including failed and aborted runs, or an elapsed interval; when one is due, the runner refuses further dispatch until a progress assessment is recorded. Assessments reference available results, verification status, constraint/convergence evidence, observed cost, learning or uncertainty, and a continue/refine/stop rationale. The runner does not certify that continuation is worthwhile. Findings may recommend stopping with resources still available.
- **Stopping and resume:** a running job's live trace is readable through `read_run`, and `abort_run` stops a stalled run without waiting for it to finish. Stopping an investigation closes admission and requests cancellation of active jobs. Record a reason distinguishing evidence completion, insufficient progress, unresolved model/strategy limitations, user cancellation, configured limits, interruption, and review handoff. Resume retains accounting and requires outstanding user reviews to be released. Restart files are versioned artifacts where supported; restarted computation gets a new linked run ID.
- **Findings:** link scientific claims to applicable verified evidence; retain diagnostics separately when execution or verification fails. Early-stop reports preserve the best available results, limitations, and suggested next experiments. Negative/exploratory findings can satisfy declared reporting criteria without a finished device.
- **Composition:** sweep/search requires no control, calibration, or catalog. Control accepts an existing system model; calibration accepts its declared data. Co-design links device/model, control, and evaluation runs with shared review state, usage accounting, and any configured limits; declare constraints, uncertainty, and comparable evaluation conditions.

Usage is always recorded from executor reports (§7). Limits are optional, and the absence of a limit is recorded explicitly. Unknown limit names are rejected.

| Limit | Scope | Meaning and enforcement |
|---|---|---|
| `timeout_seconds` | Run | Executor-native timeout: a process timer on the workstation, the scheduler time limit on a cluster |
| `deadline` | Investigation | Absolute calendar timestamp enforced by the executor, including queued jobs: no new dispatch or job start after it; cancel pending jobs and terminate active jobs at expiry. Cap each run's timeout at the time remaining when it starts |
| `gpu_seconds` | Investigation | Allocated GPUs × allocation time, including verification, retries, and failed work |
| `job_count` | Investigation | Executor launches, including retries and verification jobs |

Before dispatch, the runner compares recorded usage with any limits. When `gpu_seconds` is limited, every job needs a timeout. Admission includes the proposed job and all admitted but unsettled jobs, including queued and running jobs, at their maximum commitment (timeout × requested GPUs). The runner atomically checks allowances and records the commitment before submitting the job, using its existing records rather than a separate reservation service. Confirmed final usage replaces the commitment; usage the executor cannot confirm remains uncertain and never returns allowance. Lost contact never permits conflicting GPU dispatch.

Executors document timeout resolution and termination grace, and record actual usage and any overrun. A submission-time duration alone does not enforce an investigation deadline across queueing; unsupported deadline enforcement is rejected.

## 5. Tools and permissions

CLI and MCP share backing operations and expose available capabilities. Research-tool requests are distinct from user-input events; approval claims in request fields are rejected. Unavailable protocols cannot be dispatched. These application-level checks operate under §7, without claiming an authenticated OS boundary around an unrestricted shell.

| Operation | Permission class | Effect |
|---|---|---|
| `validate_system` | Read-only | Check the system schema and model adapter inputs |
| `list_capabilities` | Read-only | Report installed protocols, supported features, and interface versions |
| `list_runs` / `read_run` | Read-only | Inspect artifacts, all attempt states, and the live trace of running jobs |
| `list_investigations` / `read_investigation` | Read-only | Inspect revisions, findings, history, measured usage, remaining configured allowances, and review/stop state |
| `list_catalog` / `read_entry` | Read-only | Inspect promoted entries and provenance |
| `simulate` / `sweep` / `control` / `calibrate` | Controlled job | Execute through the runner, enforcing any configured limits |
| `compare_reference` | Controlled job | Execute corrector checks |
| `abort_run` / `stop_investigation` | Controlled job | Cancel work; investigation stop also closes admission and records its reason |
| `create_investigation` / `revise_investigation` | Controlled write | Draft a new investigation or publish in-scope revisions; broader changes remain proposals until a user decision |
| `record_finding` | Controlled write | Append interpretation and evidence references |
| `register_result` | Controlled write | Stage a candidate for human promotion |
| `propose_device_update` | Controlled write | Write a calibration proposal |
| Investigation start/resume, scope/limit changes, review release; `promote`; device-model apply | Human action | Record user decisions, release named reviews, promote verified results, or apply proposals |

Human actions use the operator CLI or runtime user-input path and record the action source and affected revision. The research-tool surface cannot synthesize these events, promote results, or apply device updates. The agent runtime's permission settings block the human-action commands, so an agent cannot run them from its shell by accident. This separates supported workflow roles; it is not authentication, and stronger identity and credential isolation is deferred deployment work.

Read-only operations do not mutate research state. Controlled jobs use the runner/executor. Controlled writes are limited to staged outputs and runner-managed investigation records; they cannot change policy, verification thresholds, configured ceilings, promoted entries, or device models through the research API. Typed tools validate structured scope constraints; data and notes cannot change execution policy.

## 6. History, audit, and promotion

The runner is the single writer for history and publishes immutable events atomically, one file per event. Concurrent requests cannot overwrite or lose events. Every attempt retains lineage, including incomplete, failed, aborted, and unpromoted work. Trial counts include unsuccessful attempts within the declared investigation or hypothesis family; session changes do not reset them.

| Record | Required content |
|---|---|
| Run ledger event | Run ID; spec/code/environment provenance; actual engine/solver when known; parent run IDs; outcome; resource use; verdict when available; investigation revision and source catalog IDs when applicable |
| Tool-call audit | Timestamp, session ID, tool, permission class, argument/result hashes, outcome, and linked investigation revision/run IDs |
| Session audit | One entry per agent session: runtime name/version, model ID, skill names/versions, and the path to the runtime's own transcript when it keeps one |
| User-decision event | Timestamp, operator or runtime user-input source, affected investigation revision, start/resume, scope/limit changes, or named review release; separate from agent proposals |

Audit records are append-only through supported APIs and stored outside candidate run outputs; these are provenance guarantees, not protection against a hostile OS user. Supported tools expose histories without catalog promotion; unknown outcomes remain explicit until reconciled.

`register_result` writes to `catalog/staging/`, referencing existing evidence. Human `entanglix promote <staged-id>` records the promoter, source runs, and `parent_entry_ids`. Promotion requires verified evidence; `UNVERIFIED` results remain available for diagnosis but cannot be promoted or represented as verified design evidence. Reuse records both the source entry and the new run. Continuing an authorized investigation requires no promotion.

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
| Launch checks | Runner validates specs, scans authored imports, checks the environment, and requires a passing reference-size check before GPU launch |
| Run ownership | Unique directories, frozen inputs, and assigned outputs; supported job paths cannot overwrite another run |
| Scientific records | Runner owns ledger/audit publication; research tools cannot modify verification policy or substitute successful verdicts |
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
| Execution (§2) | Runner and executors | Reproduction, atomic artifacts, cancellation/interruption recovery, amicode run-directory and ProblemSpec compatibility |
| Verification (§3) | Corrector | Reference cases, convergence failures, trusted applicability and label-spoof rejection, additive sweep checks, independent constraint checks, training/holdout data separation, hash-mismatch rejection |
| Investigations (§4) | Runner and tools | User-start/review event handling, rejection of tool-authored approvals, checkpoint gating including failed/aborted attempts, early-stop reports, no-limit and limit cases, atomic admission accounting for proposed/queued/running jobs, deadlines across queueing and execution, timeout resolution/grace and recorded overruns, rejection of unknown limits, retry/verification usage, cancellation and resume accounting |
| History and promotion (§6) | Ledger, audit, catalog | Atomic concurrent events, failed-attempt lineage, verification eligibility and human-promotion boundaries |
| Execution environments (§7) | Runner, executors, runtime adapter | Run/GPU ownership, conflicting-launch rejection, process cancellation, controlled history writes, blocked human-action commands, GPU-hidden agent session, and rejection of policy/verification bypass through supported tools; one executor test suite passes against the local executor and a stubbed scheduler; no OS-level containment claim |
