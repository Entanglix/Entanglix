# Entanglix — Harness Contract

**Date:** 2026-09-28 · **Status:** v0. Derived from the pinned [roadmap](docs/roadmap.md); normative once Decision 0 is Accepted, as each capability is introduced. The [mandate ADR](docs/adr/0001-mandate.md) is Draft.

This contract defines required interfaces, execution behavior, and enforceable guarantees. The roadmap owns scope, milestones, quality standards, and open decisions; the [checklist](docs/checklist.md) tracks implementation and acceptance evidence. Working instructions and research procedures belong in `AGENTS.md` and skills when introduced. Required configuration must be explicit; draft recommendations do not supply defaults.

## 1. Objects and interfaces

| Object | Definition |
|---|---|
| System | `system.toml`: scqubits circuit or explicit operators, truncation, drive operators, dissipators, parameter values and priors, units |
| Protocol spec | `simulate`, `sweep`, `control`, or `calibrate`; declares engine needs, budgets, and observables or metrics |
| Grant | Protected, versioned authorization: authenticated issuer, authorized principal, scope bounds, resource ceilings, review conditions, validity, and revocation state (§4) |
| Investigation | Bounded research question with versioned scope, assumptions, evidence requirements, limits, stopping/review criteria, and a grant reference (§4) |
| Run | One protocol execution with frozen inputs, artifacts, and parent/investigation provenance |
| Result | Metrics, provenance, actual engine/solver, and verification verdict; distinct from execution status |
| Finding | Interpretation or next-step proposal referencing its investigation revision, runs, verification status, assumptions, uncertainty, and limitations |
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

The corrector is an independent, deterministic package, hashed at load, with versioned thresholds approved outside the research-agent-editable tree. A hash mismatch fails execution. It verifies every completed protocol; failed or interrupted attempts retain diagnostics without a successful verdict. Verification never depends on a private solver.

| Protocol | Required checks |
|---|---|
| Dynamics `simulate` / `sweep` | Reference-size agreement across QuTiP CPU, cuDensityMat, and dynamiqs; step-size and truncation convergence; trace, hermiticity, positivity |
| Spectral `simulate` / `sweep` | Reference spectra or analytic cases where available; truncation convergence; checks applicable to the declared observable |
| `control` | Pulse rollout on a different engine; comparison on the same ProblemSpec with an open solver; independently calculated metric and constraint satisfaction for each candidate |
| `calibrate` | Holdout evaluation without refitting or fitting-path access; identifiability report using covariance or profile likelihood |
| Design `sweep` (additional checks) | Reproducibility across seeds; metric monotonicity where physics predicts it |

The corrector derives applicability from trusted protocol/metric definitions and the validated execution plan, not agent-written labels or exemptions. Unknown or inconsistent definitions cannot receive a verified verdict. Design-sweep checks stack with applicable dynamics or spectral checks. Reports identify checks, tolerances, residuals, reference size, and reasons for inapplicability. Reference-size agreement does not certify the full production problem; numerical agreement does not establish device-model accuracy. Cross-solver comparisons report discrepancies; distinct pulses or local optima alone do not invalidate a candidate passing independent checks. Promotion eligibility follows §6.

## 4. Bounded investigations

The runner enforces these requirements for delegated work. All research-runtime jobs require a valid grant, including standalone runs without an investigation; those runs retain their own budgets and provenance.

- **Scope:** record the question, assumptions, parameter region, permitted changes/protocols, evidence requirements, limits, review/stop criteria, and grant ID/revision. A scalar objective is optional.
- **Authorization:** the authenticated caller must match the grant; proposals and investigation revisions stay within its bounds. Only the human administrative channel (§5) can issue or amend grants, raise limits, or release reviews. In-scope iterations need no renewed approval. Expiry or revocation closes admission and requests cancellation of active work.
- **Revisions:** grant and investigation revisions are immutable; each run identifies both at dispatch. Restarts and revisions retain prior findings, unsuccessful attempts, and resource use.
- **Admission and accounting:** atomically check current grant validity, scope, review state, protocol availability, and run/investigation/grant limits before dispatch. Reserve enforceable job bounds across concurrent work, including nested solves, verification, retries, and failures. Charge each executor job once; new investigations, revisions, or sessions cannot reset the shared grant balance. Uncertain completion retains its reservation until reconciled.
- **Stopping:** limits and review points close affected admission; executor deadlines enforce running-job bounds. User cancellation also requests cancellation of active work. Record resource use and a reason distinguishing evidence completion, limits, interruption, review handoff, and grant expiry/revocation.
- **Findings:** link interpretations and next-step/stopping recommendations to run evidence and verification status; state assumptions, uncertainty, and limitations. Failed-run diagnostics can support troubleshooting; scientific design claims require applicable verified evidence. Exploratory or negative findings may satisfy declared evidence requirements without a finished device.
- **Composition:** sweep/search has no control, calibration, or catalog prerequisite. Control accepts an existing system model; calibration accepts its declared data. Co-design links device/model, control, and evaluation runs under the same limits, review points, and verification requirements. It declares constraints, parameter uncertainty, and a consistent comparison/optimization-budget policy, reporting failed candidates and differing evaluation conditions.

Limits are typed quantities with versioned accounting semantics; unknown units are rejected. Delegated grants require finite job-count and elapsed-time ceilings, plus a GPU allowance for GPU work. Numeric allowances are explicit configuration.

| Unit | Accounting meaning |
|---|---|
| `gpu_device_seconds` | Sum of allocated GPU device count × allocation duration in seconds, including verification and failed work; measures allocation, not utilization |
| `job_count` | Executor job launches; retries and separately launched child/verification jobs each count |
| `wall_seconds` | Elapsed time from the recorded grant activation time, including queueing and review pauses; revisions do not restart the clock |

Reservations include termination overhead; unused reservations are released only after measured usage is reconciled. Record metering uncertainty; enforcement must use conservative bounds.

## 5. Tools and permissions

CLI and MCP share backing operations and expose available capabilities. The runner authenticates caller authority independently of request fields; invoking the CLI from an agent shell grants no additional privileges. Unavailable protocols cannot be dispatched.

| Operation | Permission class | Effect |
|---|---|---|
| `validate_system` | Read-only | Check the system schema and model adapter inputs |
| `list_capabilities` | Read-only | Report installed protocols, supported features, and interface versions |
| `list_runs` / `read_run` | Read-only | Inspect artifacts and all attempt states |
| `list_investigations` / `read_investigation` | Read-only | Inspect revisions, findings, history, remaining resources, and review/stop state |
| `list_catalog` / `read_entry` | Read-only | Inspect promoted entries and provenance |
| `simulate` / `sweep` / `control` / `calibrate` | Controlled job | Execute a protocol under enforced limits |
| `compare_reference` | Controlled job | Execute corrector checks |
| `abort_run` / `stop_investigation` | Controlled job | Cancel work; investigation stop also closes admission and records its reason |
| `create_investigation` / `revise_investigation` | Controlled write | Publish versioned scope within existing authorization |
| `record_finding` | Controlled write | Append interpretation and evidence references |
| `register_result` | Controlled write | Stage a candidate for human promotion |
| `propose_device_update` | Controlled write | Write a calibration proposal |
| `authorize` / `raise-limit` / `release-review` / `revoke`; `promote`; device-model apply | Human administration | Issue/amend grants, change ceilings, release a named review, revoke authority, promote verified results, or apply proposals |

Human administration requires an authenticated issuer channel whose credentials and execution context are inaccessible to research agents and submitted jobs. The backing implementation rejects unauthorized calls over every transport; omitting commands from MCP is insufficient.

Read-only operations do not mutate research state. Controlled jobs use the runner/executor. Controlled writes are limited to staged outputs and runner-managed investigation records; they cannot alter executable policy, verification thresholds, authorized ceilings, the promoted catalog, or device models. Typed tools validate requests against scope; data and notes cannot change permissions or execution policy.

## 6. History, audit, and promotion

The trusted runner publishes immutable ledger events atomically, one file per event. Concurrent jobs cannot overwrite or lose events. Every attempt retains lineage, including incomplete, failed, aborted, and unpromoted work. Trial counts include unsuccessful attempts within the declared investigation or hypothesis family; session changes do not reset them.

| Record | Required content |
|---|---|
| Run ledger event | Run ID; spec/code/environment provenance; actual engine/solver when known; parent run IDs; outcome; resource use; verdict when available; grant ID/revision, investigation revision, and source catalog IDs when applicable |
| Tool-call audit | Timestamp, session ID, tool, permission class, argument/result hashes, outcome, and linked investigation revision/run IDs |
| Prompt audit | One entry per turn: system prompt hash, skill names/versions, model ID |
| Authorization event | Authenticated issuer, timestamp, grant ID/revision, scope/limit changes, review releases, and revocation; administrative denials; distinct from agent proposals |

Audit records are append-only and outside the research-agent-editable tree. Supported tools expose histories without catalog promotion; unknown outcomes remain explicit until reconciled.

`register_result` writes to `catalog/staging/`, referencing existing evidence. Human `entanglix promote <staged-id>` records the promoter, source runs, and `parent_entry_ids`. Promotion requires verified evidence; `UNVERIFIED` results remain available for diagnosis but cannot be promoted or represented as verified design evidence. Reuse records both the source entry and the new run. Continuing an authorized investigation requires no promotion.

`propose_device_update` writes next to the calibration run. Applying it to the device model is a separate human command. Research agents can stage proposals and findings but cannot perform either promotion or device-model write-back.

## 7. Isolation and enforcement

These boundaries govern the research runtime and submitted jobs. Coding agents developing the repository follow its working instructions and development permissions; research-runtime restrictions do not prohibit authorized implementation work.

| Boundary | Enforcing component and requirement |
|---|---|
| Launch gate | Runner validates specs, scans authored imports, checks the environment, and requires a passing reference-size check before GPU launch |
| Compute access | Research-agent environment has no direct local GPU access, scheduler credentials, or GPU-node shell; controlled compute is reachable only through executors |
| Trusted state | Research agents may edit candidate specs, skill drafts, and analysis scripts; runner, schemas, corrector, thresholds, grants, accounting, ledger, audit, promoted catalog, and device model remain protected |
| Job writes | Submitted code writes only assigned outputs; it cannot modify trusted state or access calibration holdout. The trusted runner publishes controlled records |
| Network | Research-agent access is limited to MCP and the package index needed for the pinned environment |
| Human actions | Grant administration, review release, promotion, and device-model application require an isolated authenticated human channel |

Isolation must hold before research-agent execution, locally and on clusters. The chosen process, filesystem, credential, and device controls must demonstrate these boundaries, including with an agent shell. The concrete local mechanism is an implementation decision; containers alone do not establish the boundary. Static import scanning is preflight, not isolation. Authorization and budget checks apply to authored code as well as tool calls.

## 8. Versioning and conformance

Public schemas and CLI/MCP interfaces declare versions and compatibility expectations. Incompatible inputs fail with actionable diagnostics. Skill metadata declares `name`, `description`, and `version`; runtime and skill versions are pinned in CI and recorded in audit. Instructions and skills cannot substitute for tool-enforced invariants.

Once this contract is normative, invariant changes require an ADR and corresponding schema/interface versions and conformance updates where affected. Broad quality standards and milestone acceptance remain in roadmap §4 and the checklist.

Conformance evidence must cover the observable guarantees below. Implementations add concrete test/schema references as they are built; these requirements do not claim existing tests.

| Contract surface | Responsible component | Required evidence |
|---|---|---|
| Objects and interfaces (§1, §5) | Schemas, adapters, CLI/MCP | Validation, transport consistency, capability checks, open-only/plugin-boundary tests |
| Execution (§2) | Runner and executors | Reproduction, atomic artifacts, cancellation/interruption recovery, amicode run-directory and ProblemSpec compatibility |
| Verification (§3) | Corrector | Reference cases, convergence failures, trusted applicability and label-spoof rejection, additive sweep checks, independent constraint checks, holdout isolation, hash-mismatch rejection |
| Investigations (§4) | Runner and tools | Grant issuance/revocation, unit validation, concurrent/nested reservations and accounting, deadlines, review gates, cancellation, shared balances across investigations/restarts |
| History and promotion (§6) | Ledger, audit, catalog | Atomic concurrent events, failed-attempt lineage, verification eligibility and human-promotion boundaries |
| Isolation (§7) | Execution environment | Rejection of unauthorized writes, direct compute access, prompt-injection-driven policy changes, and scope/budget escalation through tools, shell/CLI, and authored code |
