# Entanglix — Harness Contract

**Date:** 2026-09-28 · **Status:** v0. Derived from the pinned [roadmap](docs/roadmap.md); normative once Decision 0 is Accepted, as each capability is introduced. The [mandate ADR](docs/adr/0001-mandate.md) is Draft.

This contract defines required interfaces and guarantees enforced through supported execution paths under the cooperative local trust model (§7). The roadmap owns scope, milestones, quality standards, and open decisions; the [checklist](docs/checklist.md) tracks implementation and acceptance evidence. Working instructions and research procedures belong in `AGENTS.md` and skills when introduced. Required configuration must be explicit; draft recommendations do not supply defaults.

## 1. Objects and interfaces

| Object | Definition |
|---|---|
| System | `system.toml`: scqubits circuit or explicit operators, truncation, drive operators, dissipators, parameter values and priors, units |
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
- **User decisions:** the CLI/runtime user-input path records the initial scope, broader changes, review releases, and changes to configured ceilings. Research-tool arguments and agent-authored notes cannot create these events. Routine revisions within recorded scope need no renewed approval; the local workflow requires no separate grant issuer.
- **Revisions:** investigation revisions are immutable and runs identify their dispatch revision. Resume, reruns, and session changes preserve unsuccessful attempts, resource use, findings, and outstanding reviews.
- **Admission:** check active revision, supported protocol, mechanically checkable scope constraints, review/stop state, and any run/investigation limits before dispatch. Free-text research aims are recorded context, not machine-certified scientific judgments.
- **Progress:** checkpoint assessments reference available results, verification status, constraint/convergence evidence, observed cost, learning or uncertainty, and a continue/refine/stop rationale. The runner requires these records at declared checkpoints; it does not certify that continuation is worthwhile. Findings may recommend stopping with resources still available.
- **Stopping and resume:** stop closes admission and requests cancellation of active jobs. Record a reason distinguishing evidence completion, insufficient progress, unresolved model/strategy limitations, user cancellation, configured limits, interruption, and review handoff. Resume retains accounting and requires outstanding user reviews to be released. Checkpoints are versioned artifacts where supported; restarted computation gets a new linked run ID.
- **Findings:** link scientific claims to applicable verified evidence; retain diagnostics separately when execution or verification fails. Early-stop reports preserve the best available results, limitations, and suggested next experiments. Negative/exploratory findings can satisfy declared reporting criteria without a finished device.
- **Composition:** sweep/search requires no control, calibration, or catalog. Control accepts an existing system model; calibration accepts its declared data. Co-design links device/model, control, and evaluation runs with shared review state, usage accounting, and any configured limits; declare constraints, uncertainty, and comparable evaluation conditions.

Usage is recorded even without a ceiling. Each optional limit names its scope (run or investigation), quantity, and unit; absence is explicit. Unknown units or unsupported enforcement are rejected. Estimates are recorded separately from ceilings.

| Unit | Accounting meaning |
|---|---|
| `gpu_device_seconds` | Allocated GPU device count × allocation duration in seconds, including verification and failed work; charge each allocation once |
| `job_count` | Executor job launches, including separately launched retries and child/verification jobs |
| `wall_seconds` | Elapsed time from recorded run/investigation start, including queueing and review pauses; resume does not restart the clock |

Before launching, check remaining configured allowances; time caps set executor deadlines with documented metering resolution and termination grace. Record actual usage and any overrun or uncertainty; lost contact never restores an allowance or permits conflicting GPU dispatch. The serialized local path needs no concurrent credit-allocation service. Any later concurrent execution must enforce aggregate configured limits before it is enabled.

## 5. Tools and permissions

CLI and MCP share backing operations and expose available capabilities. Research-tool requests are distinct from user-input events; approval claims in request fields are rejected. Unavailable protocols cannot be dispatched. These application-level checks operate under §7, without claiming an authenticated OS boundary around an unrestricted shell.

| Operation | Permission class | Effect |
|---|---|---|
| `validate_system` | Read-only | Check the system schema and model adapter inputs |
| `list_capabilities` | Read-only | Report installed protocols, supported features, and interface versions |
| `list_runs` / `read_run` | Read-only | Inspect artifacts and all attempt states |
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

Human actions use the operator CLI or runtime user-input path and record the action source and affected revision. The research-tool surface cannot synthesize these events, promote results, or apply device updates. This separates supported workflow roles; stronger identity and credential isolation is deferred deployment work.

Read-only operations do not mutate research state. Controlled jobs use the runner/executor. Controlled writes are limited to staged outputs and runner-managed investigation records; they cannot change policy, verification thresholds, configured ceilings, promoted entries, or device models through the research API. Typed tools validate structured scope constraints; data and notes cannot change execution policy.

## 6. History, audit, and promotion

The runner is the single writer for history and publishes immutable events atomically, one file per event. Concurrent requests cannot overwrite or lose events. Every attempt retains lineage, including incomplete, failed, aborted, and unpromoted work. Trial counts include unsuccessful attempts within the declared investigation or hypothesis family; session changes do not reset them.

| Record | Required content |
|---|---|
| Run ledger event | Run ID; spec/code/environment provenance; actual engine/solver when known; parent run IDs; outcome; resource use; verdict when available; investigation revision and source catalog IDs when applicable |
| Tool-call audit | Timestamp, session ID, tool, permission class, argument/result hashes, outcome, and linked investigation revision/run IDs |
| Prompt audit | One entry per turn: system prompt hash, skill names/versions, model ID |
| User-decision event | Timestamp, operator or runtime user-input source, affected investigation revision, start/resume, scope/limit changes, or named review release; separate from agent proposals |

Audit records are append-only through supported APIs and stored outside candidate run outputs; these are provenance guarantees, not protection against a hostile OS user. Supported tools expose histories without catalog promotion; unknown outcomes remain explicit until reconciled.

`register_result` writes to `catalog/staging/`, referencing existing evidence. Human `entanglix promote <staged-id>` records the promoter, source runs, and `parent_entry_ids`. Promotion requires verified evidence; `UNVERIFIED` results remain available for diagnosis but cannot be promoted or represented as verified design evidence. Reuse records both the source entry and the new run. Continuing an authorized investigation requires no promotion.

`propose_device_update` writes next to the calibration run. Applying it to the device model is a separate human command. Research tools can stage proposals and findings but expose neither promotion nor device-model write-back.

## 7. Local execution and trust boundaries

The initial deployment is a cooperative, single-owner workstation workflow. One coordinating agent and its scripts use the supported runner path. This contract does not claim containment of arbitrary code, protection from an agent deliberately bypassing tools through its shell, or tamper-proof history against the host owner. Coding-agent development permissions remain separate from research workflow roles.

| Boundary | Enforcing component and requirement |
|---|---|
| Launch checks | Runner validates specs, scans authored imports, checks the environment, and requires a passing reference-size check before GPU launch |
| Run ownership | Unique directories, frozen inputs, and assigned outputs; supported job paths cannot overwrite another run |
| GPU ownership | Shared local ownership check permits one active GPU job by default, including across runner instances; a job may batch calculations; unknown process state blocks reuse pending reconciliation |
| Process lifecycle | Executor owns the job process group; cancellation targets that job and descendants; checkpoint recovery is used only where supported |
| Scientific records | Runner owns ledger/audit publication; research tools cannot modify verification policy or substitute successful verdicts |
| Human actions | Research APIs stage proposals; operator/runtime user-input actions record scope changes, review releases, promotion, and device-model application |

Configured limits and cancellation apply to runner-launched scripts as well as built-in protocols. Static scanning is preflight, not a security sandbox. Reuse available runtime/OS protections; a separate grant service, inaccessible administrative credentials, mandatory containers, and device/network restrictions are not prerequisites for this local workflow.

A deployment requiring resistance to bypass or execution of untrusted code must first define and test stronger identity, process, filesystem, credential, network, and device boundaries. Those guarantees require a separate deployment decision and conformance evidence; cooperative local tests cannot establish them.

## 8. Versioning and conformance

Public schemas and CLI/MCP interfaces declare versions and compatibility expectations. Incompatible inputs fail with actionable diagnostics. Skill metadata declares `name`, `description`, and `version`; runtime and skill versions are pinned in CI and recorded in audit. Instructions and skills cannot substitute for tool-enforced invariants.

Once this contract is normative, invariant changes require an ADR and corresponding schema/interface versions and conformance updates where affected. Broad quality standards and milestone acceptance remain in roadmap §4 and the checklist.

Conformance evidence must cover the observable guarantees below. Implementations add concrete test/schema references as they are built; these requirements do not claim existing tests.

| Contract surface | Responsible component | Required evidence |
|---|---|---|
| Objects and interfaces (§1, §5) | Schemas, adapters, CLI/MCP | Validation, transport consistency, capability checks, open-only/plugin-boundary tests |
| Execution (§2) | Runner and executors | Reproduction, atomic artifacts, cancellation/interruption recovery, amicode run-directory and ProblemSpec compatibility |
| Verification (§3) | Corrector | Reference cases, convergence failures, trusted applicability and label-spoof rejection, additive sweep checks, independent constraint checks, training/holdout data separation, hash-mismatch rejection |
| Investigations (§4) | Runner and tools | User-start/review event handling, rejection of tool-authored approvals, progress records and early-stop reports, no-limit/configured-limit cases, unit validation, nested/retry usage, cancellation and resume accounting |
| History and promotion (§6) | Ledger, audit, catalog | Atomic concurrent events, failed-attempt lineage, verification eligibility and human-promotion boundaries |
| Local execution (§7) | Runner, executor, runtime adapter | Run/GPU ownership, conflicting launch rejection, correct process cancellation, controlled history writes, and rejection of policy/verification bypass through supported tools; no OS-level containment claim |
