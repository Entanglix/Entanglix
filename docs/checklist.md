# Entanglix Build Checklist

**Status:** Implementation tracker for the pinned [`roadmap.md`](roadmap.md) and derived [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md). The roadmap defines scope, milestones, and dependencies; the contract defines enforceable behavior; this file tracks completion and acceptance evidence.

Status legend: `[ ]` open · `[-]` in progress / partial · `[x]` done. Dependencies below follow the roadmap. Supporting milestones can proceed when their own prerequisites are ready. A block is complete when its required functional and applicable quality criteria are checked; numbering imposes no additional dependencies. No calendar: exit criteria gate progress, not dates.

The mandate ADR is Accepted and Decision 0 is closed. Concrete modeling schemas and fixtures follow during Block 1; later choices are tracked with the blocks that need them. Development and testing run on the owner's workstation; research runs may also use a cluster through its scheduler, under the same cooperative trust model. Stronger authorization and isolation are conditional deployment work, not M3 prerequisites.

---

## Milestone Map

Core capability progression; users may enter, iterate within, and stop at any supported stage:

| Milestone | Blocks | Status |
|---|---|---|
| M0 — Mandate + foundation | Decision 0, Block 0 | **complete** |
| M1 — Model + engines | Blocks 1–2 | open |
| M2 — Verified local sweep | Blocks 3–5 | open |
| M3 — Bounded agent-assisted exploration | Blocks 6a–6b | open |
| M4 — Control | Block 7 | open |
| M5 — Device–control co-design | Block 8 | open |

Supporting milestones, scheduled by dependency and research need:

| Milestone | Blocks | Status |
|---|---|---|
| S1 — Cluster execution | Block 9 | open |
| S2 — Calibration | Block 10 | open |
| S3 — Results catalog | Block 11 | open |
| S4 — Interoperability | Block 12 | open |

The Harmoniqs conversation requires the verified shared ProblemSpec and open-solver artifacts specified in roadmap §5, normally delivered at M4. Agent integration is not a prerequisite. The audience/license decision applies, and `roadmap.md` Appendix A must be removed before sharing.

## Acceptance Evidence for Every Block

Each block's quality checkbox covers the applicable standards in [roadmap §4](roadmap.md#4-standing-disciplines-every-block) and conformance requirements in [contract §8](../HARNESS_CONTRACT.md#8-versioning-and-conformance). Record links to the relevant tests, reports, examples, or run artifacts when checking it.

M3 demonstrates quality for a complete sweep/search workflow. The preferred sequence puts M3 before control; standalone control has its own build prerequisites and quality gates. Conditional features require evidence when their prerequisite becomes available; record the condition explicitly rather than marking unimplemented work complete. Optional extensions do not gate completion.

---

## M0 — Mandate + Foundation (Decision 0, Block 0)

### [x] Decision 0 — Mandate ADR

[`docs/adr/0001-mandate.md`](adr/0001-mandate.md) is **Accepted**. Foundation choices are closed; detailed modeling and validation choices are implemented during Block 1.

- [x] Project name: Entanglix. Command: `entanglix`. Repository established.
- [x] Runner language: Python (`uv`), same substrate as the engines.
- [x] Substrate fixed (roadmap §0): scqubits, cuDensityMat via `qutip-cuquantum`, dynamiqs, QuTiP CPU reference; ProblemSpec OSS control spec; private solvers as entry-point plugins.
- [x] Research direction pinned: expert-guided formulation, bounded autonomy, independently useful workflows, and enterprise-grade quality across milestones.
- [x] Core progression pinned: verified local sweep → bounded agent-assisted exploration → control → co-design; supporting capabilities follow their own dependencies.
- [x] System-modeling mandate: component-defined circuits, coupled subsystems, and explicit operators; representative validation required, with concrete coverage and fixtures selected during Block 1 (ADR §3.1)
- [x] Initial verification target: this Linux workstation with an owner-confirmed NVIDIA RTX 5090; one local GPU environment plus CPU-only foundation CI; compatible version pins and GPU validation follow in Block 2 (ADR §3.2)
- [x] GHz/ns conventions with explicit angular-frequency conversion; uniform output grid with internal stepping allowed; separate basis/retained-level cutoffs and model-specific convergence policy (ADR §3.3)
- [x] Private development by the owner and explicitly permitted collaborators; distribution license selected before external distribution; existing sharing gates retained (ADR §3.4)
- [x] Foundation decisions recorded in the Accepted mandate ADR
- **Unblocks:** Block 0.

### [x] Block 0 — Repo Foundation

**Depends on:** Decision 0 (Accepted). Reduced single-project foundation implemented and validated. GPU environment validation is deferred to Block 2, environment capture/hashing to Block 3, schemas to Block 1, and MCP/skills to Block 6b.

- [x] `HARNESS_CONTRACT.md` at repo root, aligned with the pinned roadmap
- [x] `docs/README.md` index with the documentation boundary and sharing rule
- [x] `docs/roadmap.md` and `docs/checklist.md`, with aligned milestones and dependencies
- [x] Single-package layout: `src/entanglix/`, `docs/`, and `tests/`
- [x] Root [project configuration](../pyproject.toml), [`uv.lock`](../uv.lock), and [Python pin](../.python-version)
- [x] [`.gitignore`](../.gitignore) covers runs, caches, `.env`, and generated numerical arrays, with exceptions for small reference fixtures
- [x] [CPU CI workflow](../.github/workflows/ci.yml): locked installation, lint/format checks, and pytest; [GPU selection policy](../tests/conftest.py) skips unavailable devices and rejects empty/unavailable required-GPU runs
- [x] [Import-boundary test](../tests/test_import_boundary.py): reviewed public imports only outside `tests/`, including literal/aliased dynamic imports; private-implementation and scheduler-client cases rejected
- [x] Root [AGENTS.md](../AGENTS.md) and [CLAUDE.md](../CLAUDE.md) are one-line development pointers to [development.md](development.md); [research-agent.md](research-agent.md) preserves draft research conduct separately until Block 6b
- [x] Operational [setup/CLI usage](../README.md#setup-and-cli) and [release notes](../README.md#changes-and-compatibility), independent of planning docs
- [x] [Installed entry-point and CPU import checks](../tests/test_cli.py), import-boundary checks, lint/format, and documentation checks pass
- [x] **Done:** a fresh source copy syncs from the lockfile and runs `pytest` successfully without GPU dependencies or device access

---

## M1 — Model + Engines (Blocks 1–2)

The [M1 guide](m1-guide.md) develops both blocks as one implementation plan. Its proposed schemas, fixtures, and numerical targets remain draft choices; this checklist records implemented and validated work.

### [ ] M1 entry — GPU compatibility probe

- [ ] Before step A/implementation: a disposable environment runs one tiny cuDensityMat solve through qutip-cuquantum and one dynamiqs solve on the actual RTX 5090; both match a known reference
- [ ] Record commands, Python/package versions, driver/CUDA observations, and device evidence; retain Python 3.13 unless an observed blocker requires a pin change. This probe is not full engine acceptance

### [ ] Block 1 — System Spec and Model Adapter

**Depends on:** Block 0.

- [ ] `system.toml` JSON schema + pydantic model (circuit, truncation, drives, dissipators, parameters + priors, units)
- [ ] scqubits adapter: circuit → truncated operators in a chosen basis
- [ ] Explicit-operator fallback
- [ ] CPU-testable QuTiP `Qobj`, numerical arrays suitable for JAX, and per-subsystem operator terms with dimensions/order for the cuDensityMat adapter; no GPU packages required
- [ ] `entanglix validate <system.toml>` / `validate_system` backing capability
- [ ] Concrete representation coverage and fixture systems/parameters selected during this block under ADR §3.1, exercising component, composition, and explicit-operator paths; fixed units, array/output unit semantics, and one level of leaf system references
- [ ] Pin numerical acceptance thresholds/rationale after independent CPU reference refinement; operator/spectral comparisons and separate basis/retained-level convergence evidence pass
- [ ] Malformed units, rates, dimensions, references, and unsupported forms produce actionable diagnostics; intentionally insufficient truncation demonstrates the refinement check
- [ ] Implemented syntax, tensor/basis order, 2π conversion, collapse normalization, units, and operator names documented in `docs/operations/system-spec.md`; worked examples and README usage/compatibility describe available behavior
- [ ] Applicable quality evidence recorded
- [ ] **Done:** fixtures validate and export CPU objects/arrays/factorized terms with preserved units/bases; reference spectra/operators, model refinement, and input diagnostics pass without GPU dependencies

### [ ] Block 2 — Engines v0

**Depends on:** Block 1.

- [ ] `Integrator` protocol with `capabilities()`
- [ ] QuTiP CPU reference engine
- [ ] cuDensityMat engine via `qutip-cuquantum`
- [ ] dynamiqs engine
- [ ] Wrap CPU factorized terms and select Dense/Dia storage in the cuDensityMat adapter; reimplement the unreleased scqubits selection rule and check the threshold on the RTX 5090
- [ ] Cross-engine agreement tests at reference size (closed, open, time-dependent drive); dissipative fixtures populate affected states and show measurable loss, with `γT` around `0.1–1`
- [ ] Nonzero evolution gradient agrees with an analytic/finite-difference reference; integration refinement and trace/Hermiticity/positivity checks meet the pinned tolerances
- [ ] Required-GPU acceptance identifies the actual engines and devices used
- [ ] Verified environment pins in `uv.lock` and observations in `docs/environment.md`: GPU, driver, CUDA, cuQuantum, Python, and engine packages; hashing follows with the runner
- [ ] Versioned engine-selection table (contract §2); configured memory/size limits; declared fallbacks recorded without silently changing required capabilities; advertise only implemented/tested capabilities (`batched = false` until sweep batching is validated)
- [ ] Applicable quality evidence recorded
- [ ] **Done:** required agreement, gradient, storage, and refinement checks pass with retained commands/results and actual GPU evidence from CI or a documented manual run; all M1 acceptance items above are satisfied

---

## M2 — Verified Local Sweep (Blocks 3–5)

### [ ] Block 3 — Runner and Run Contract

**Depends on:** Blocks 0–2.

- [ ] Gate: spec validation, static import scan, environment check (driver, cuQuantum, memory)
- [ ] Run dir per contract §2: `run.toml` first, `spec/` frozen, `FINISHED` last, atomic `result.toml`
- [ ] Minimal retained environment record and `env_hash`: lockfile digest, Python/platform, and observed GPU driver/CUDA/cuQuantum versions; exactly synced execution through `uv run --locked --exact`
- [ ] Reproduction inputs: authored code/data references, versions or digests, seeds, solver settings, and code/environment provenance; Git revision and dirty state plus frozen inputs/digests for uncommitted code
- [ ] Job spawn in its own process group; exit-code contract; per-run timeouts when configured
- [ ] Separate run directories and assigned outputs prevent accidental interference; a GPU lock permits one active job by default across runner instances; a job may batch calculations
- [ ] Cancellation targets the intended job and descendants; unknown process state blocks GPU reuse pending reconciliation
- [ ] `index` and `latest` at the runs root
- [ ] Local executor, diagnostics, and documented interruption recovery/rerun path
- [ ] Append-only ledger from first execution: parent run IDs, provenance, outcomes, resource use, and verification status when available; include failed, aborted, and incomplete attempts
- [ ] Atomic ledger events retain concurrent attempts; reruns get new IDs and preserve lineage
- [ ] `list_capabilities`, `list_runs`, `read_run`, and `abort_run` backing capabilities; `read_run` shows the live trace of running jobs
- [ ] Execution outcome remains distinct from scientific verification
- [ ] Applicable quality evidence recorded
- [ ] **Done:** abrupt termination leaves no fabricated completion; completed results validate and reproduce; interrupted attempts remain queryable with diagnostics and lineage

### [ ] Block 4 — Corrector v0

**Depends on:** Blocks 1–3.

- [ ] Separate corrector package; code hashed at load; versioned thresholds outside the candidate workspace; research tools cannot replace policy or verdicts
- [ ] Cross-engine, convergence, trace, hermiticity, and positivity checks for dynamics
- [ ] Reference and truncation checks applicable to spectral studies
- [ ] Corrector selects checks from trusted protocol/metric definitions and validated execution plans; unknown/inconsistent definitions cannot verify; agent-written labels cannot waive checks
- [ ] Design-sweep seed/monotonicity checks add to applicable dynamics or spectral checks
- [ ] Reports identify check scope, tolerances, residuals, reference size, and inapplicable checks with reasons (contract §3)
- [ ] `compare_reference` backing capability
- [ ] Applicable quality evidence recorded
- [ ] **Done:** a deliberately under-converged run is `UNVERIFIED`; a hash mismatch fails the run; label-spoof cases cannot evade checks; verification is available before Block 5 completion

### [ ] Block 5 — Protocols `simulate` and `sweep`

**Depends on:** Blocks 1–4.

- [ ] Standalone simulation and grid sweep over scqubits parameters against declared observables or a simulated metric
- [ ] Before implementation: choose the exact dynamics metric/observable, pulse if needed, parameter region, and evidence thresholds
- [ ] First end-to-end sweep exercises open-system dynamics on the local GPU (metric class already fixed)
- [ ] Spectral-only studies remain usable independently
- [ ] Representative batched sweep agrees member-by-member with serial execution under declared tolerances before batching is advertised
- [ ] Plots and `result.toml` summaries expose evidence needed to inspect and refine a sweep
- [ ] Documented CLI examples for simulation and iterative sweep refinement
- [ ] Applicable quality evidence recorded
- [ ] **Done:** a documented simulation and agreed dynamics sweep run through `entanglix`, GPU execution and corrector checks pass, and a user can inspect/refine the sweep without control or calibration

---

## M3 — Bounded Agent-assisted Exploration (Blocks 6a–6b)

### [ ] Investigation decisions — after M2

- [ ] Before Block 6a: first investigation question, scope, assumptions, evidence requirements, progress checkpoints, and stop/review criteria; any requested ceilings informed by measured M2 costs
- [ ] Before Block 6a: what the contract leaves open (user-decision recording, progress-assessment fields, default checkpoint intervals, resume behaviour) in a follow-up ADR
- [ ] Before Block 6a: evaluation protocol defines the scripted selection policy, comparable benchmark allowances, evidence thresholds, and outcome/time/compute/human-intervention measures; include supported early stopping after insufficient progress
- [ ] Before Block 6b: runtime integration and research-agent instruction delivery (MCP instructions, skills, or a generated research-workspace instruction file); user-input event handling, progress reporting, runtime permission settings, and execution checks under the cooperative trust model; no separate grant issuer or OS security boundary required

### [ ] Block 6a — Investigation Execution

**Depends on:** Blocks 3–5 and the decisions identified above as required before Block 6a.

- [ ] Investigation records with immutable revisions: question, scope, assumptions, permitted changes/protocols, evidence requirements, progress checkpoints, review/stop criteria, and optional limits (contract §4)
- [ ] Human-started investigations; user-input events for broader scope changes, review releases, and configured-limit changes are distinct from agent proposals
- [ ] `create_investigation`, `revise_investigation`, `list_investigations`, `read_investigation`, `record_finding`, and `stop_investigation`; drafts cannot self-approve or start through research-tool fields
- [ ] Findings and progress assessments link investigation revisions, evidence, verification status, constraints/convergence, observed cost, learning/uncertainty, and continue/refine/stop rationale
- [ ] Usage recorded from executor reports across protocol runs, nested work, verification, retries, and failures
- [ ] Optional limits per contract §4 (run timeout; investigation deadline, GPU-seconds, job count); absence is explicit; unknown limit names are rejected
- [ ] Admission checks and commitment records are atomic; proposed and admitted-but-unsettled jobs, including queued/running jobs, count at their maximum when GPU-seconds are limited; uncertain usage never returns allowance
- [ ] Executor enforces absolute investigation deadlines across queueing and execution; timeout resolution and termination grace documented; actual usage and overruns recorded
- [ ] Progress checkpoints (finished-attempt count, including failed/aborted runs, or elapsed interval) gate dispatch until an assessment is recorded
- [ ] Stop/review state gates dispatch; cancellation, resume, and supported restart files preserve accounting, outstanding reviews, and stopping reasons
- [ ] Early-stop reports preserve best available results, limitations, and next experiments; remaining allowance does not require continued execution
- [ ] User-decision events record source, affected revision, starts/resumes, scope/limit changes, and named review releases
- [ ] Scripted iterative sweep/search client, with Bayesian search where appropriate; documented selection policy qualifies it as the M3 baseline
- [ ] Tests cover no-limit and configured-limit cases, atomic admission with proposed/queued/running commitments, deadlines across queueing and execution, timeout resolution/grace and overruns, rejected approval claims, review and checkpoint gating including failed/aborted attempts, nested/retry accounting, cancellation/resume, and accidental run interference
- [ ] Applicable quality evidence recorded
- [ ] **Done:** scripted investigation reaches declared evidence or a supported stop/review handoff; progress/history survive resume; optional limits, review gates, cancellation, and local execution protections work through supported interfaces

### [ ] Block 6b — Agent Integration

**Depends on:** Block 6a and the local runtime integration decision.

- [ ] Formulation workflow refines broad ambitions into bounded next investigations; partial workflows and exploratory questions are supported
- [ ] Agent assesses evidence, constraints, convergence, observed cost, and learning before continuing/refining/stopping; roles of numerical search and agent reasoning documented
- [ ] `entanglix.mcp` behind an optional `mcp` extra, introduced with `skills/`; MCP over stdio uses CLI backing operations, advertises available capabilities, and distinguishes research requests from runtime user-input events (contract §5)
- [ ] Tool-call audit log and per-session runtime/model/skill records (contract §6), including scope revisions and linked run IDs
- [ ] Initial skills: `system-modeler`, `simulate`, `design-sweep`, `gpu-budget`; versioned metadata; resource guidance covers measured usage and optional limits; updated research-agent conduct and workflow entry points delivered through the selected mechanism
- [ ] Instruction-delivery check in a research workspace: intended research guidance reaches the runtime without repository development instructions
- [ ] One coordinating agent uses separate run directories, frozen inputs, runner-owned history, one active GPU job by default, and cancellation of owned jobs; model/catalog changes are staged
- [ ] Runtime permission settings block human-action commands for the agent; the agent session sees no GPU; the cooperative trust assumption is documented; no claim of resistance to arbitrary shell/code bypass
- [ ] Tests reject agent-authored approval claims, skipped verification, out-of-scope tool requests, and configured-limit changes through research tools; pinned runtime in CI
- [ ] Demonstrate evidence-driven refinement and a supported early-stop report in an insufficient-progress case; preserve available results and suggested next experiments
- [ ] Execute the declared baseline comparison; report scientific outcome, elapsed time, compute use, human interventions, and continuation/stopping rationale, including worse or inconclusive performance
- [ ] Applicable quality evidence recorded for the complete independent sweep/search workflow
- [ ] **Done:** logged agent investigation meets declared evidence/reporting criteria and independent usability; refinement and justified stopping are demonstrated; local execution protections and any configured limits hold; baseline comparison complete
- M3 does not require outperforming the baseline. Claims of acceleration require measured comparative benefit.

---

## M4 — Control (Block 7)

### [ ] Block 7 — Protocol `control`

**Builds on:** Blocks 1–4 for standalone control; Block 6b for agent integration. Prefer to sequence after M3; that order is not a technical prerequisite. Cluster execution and calibration are optional.

- [ ] Resolve the roadmap's open ProblemSpec choice: verbatim adoption or documented additive `solver.options` extension
- [ ] `SolverBackend` protocol and entry-point discovery
- [ ] Open GRAPE-style baseline
- [ ] Piccolo adapter via `amico-run`
- [ ] Open-only operation and declared plugin fallback behavior; actual solver recorded
- [ ] Independent pulse rollout, metric and constraint checks; cross-solver comparison on the same ProblemSpec (contract §3)
- [ ] Standalone control from an existing system model through the CLI
- [ ] When Block 6b is available: MCP capability and `control` skill with integration evidence
- [ ] Applicable quality evidence recorded
- [ ] **Done:** open baseline and Piccolo solve the same gate problem, both results verified and in run dirs; a private plugin, if present, is discovered and its result is verified through the open path
- **Unblocks:** the Harmoniqs conversation once the shared ProblemSpec, verified open-baseline/Piccolo runs, and corrector reports are inspectable; audience/license and Appendix A rules still apply.

---

## M5 — Device–Control Co-design (Block 8)

### [ ] Block 8 — Device–Control Co-design

**Depends on:** Blocks 5, 6a, and 7; Block 6b for agent-driven co-design.

- [ ] Compose circuit-parameter proposal, model construction, admissible-control optimization, verified evaluation, and device-search refinement
- [ ] Declared device/control constraints, relevant parameter uncertainty, and a consistent candidate comparison/optimization-budget policy; report any limits and differing conditions
- [ ] Candidate lineage links device/model, control, and evaluation runs, including unsuccessful attempts
- [ ] Nested work retains investigation usage accounting, verification, user-chosen review points, and any configured limits
- [ ] Comparison report preserves failed candidates, limitations, and any differences in evaluation conditions
- [ ] Documented co-design example; standalone sweep/search and control entry points remain supported
- [ ] Applicable quality evidence recorded
- [ ] **Done:** one joint device/control investigation produces a verified candidate comparison and supported recommendation or explicit account of limitations under the declared constraints and uncertainty

---

## S1 — Cluster Execution (Block 9)

### [ ] Block 9 — Cluster Executor

**Depends on:** Blocks 3–5. Schedule when remote compute is needed; local exploration and control do not wait for this block.

- [ ] Scheduler target and environment confirmed when this capability is scheduled
- [ ] Slurm or Kubernetes executor behind the local `Executor` interface
- [ ] Artifact mirroring: small arrays back; large arrays by path + digest with integrity checks
- [ ] Warming and lost-contact timeouts; abort; documented recovery/rerun path
- [ ] Scheduler time limits, GPU allocation, cancellation, and accounting used instead of Entanglix equivalents; run ownership, audit, and lineage preserved; site-specific access controls and any stronger deployment guarantees defined before claiming them
- [ ] One executor test suite runs on the workstation against the local executor and a stubbed scheduler (fake submit/status/accounting commands)
- [ ] One documented smoke run on the target cluster
- [ ] Applicable quality evidence recorded
- [ ] **Done:** `sweep` runs unchanged locally and on the cluster; the executor test suite passes against the stubbed scheduler; a lost job is aborted, never completed; usage comes from scheduler accounting and recovery diagnostics are recorded

---

## S2 — Calibration (Block 10)

### [ ] Block 10 — Protocol `calibrate`

**Depends on:** Blocks 1–4; Block 6b for agent access. Control and co-design are optional.

- [ ] Calibration input contract with point-in-time provenance
- [ ] Gradient lane (dynamiqs) and gradient-free / Bayesian lane
- [ ] Holdout evaluation and identifiability report; fitting receives training data only and the corrector evaluates holdout data
- [ ] `propose_device_update` proposal file; separate human apply command
- [ ] Documented standalone calibration and inspectable report, independent of control
- [ ] When Block 6b is available: MCP capabilities and `calibrate` skill; training/holdout data separation and staged device-model updates verified through research tools
- [ ] Applicable quality evidence recorded
- [ ] **Done:** synthetic-data round trip recovers known parameters within reported uncertainty; holdout is excluded from fitting inputs; a user can inspect the report and update proposal

---

## S3 — Results Catalog (Block 11)

### [ ] Block 11 — Results Catalog

**Depends on:** Blocks 3–5; Block 7 for pulse warm-starts.

- [ ] Catalog builds on the existing run ledger; all failed and unpromoted history remains independently queryable
- [ ] `catalog/staging/`, `register_result`, `list_catalog`, and `read_entry`
- [ ] Human `promote` command verifies evidence eligibility and records source lineage and promoter; agents cannot promote
- [ ] Lineage queries and trial counts retain unsuccessful attempts within the declared investigation or hypothesis family
- [ ] Subsequent sweep reuses a catalogued candidate with both entry and run provenance
- [ ] When Block 7 is available: control warm-start from a catalogued pulse with recorded lineage
- [ ] Applicable quality evidence recorded
- [ ] **Done:** a catalogued candidate is reused in a sweep and promotion is auditable; pulse warm-start is demonstrated when control is available

---

## S4 — Interoperability (Block 12)

### [ ] Block 12 — Interoperability

**Depends on:** Blocks 3 and 7; shared calibration schemas extend conformance when Block 10 is available.

- [ ] A `control` run dir opens in amicode's Run Inspector
- [ ] A ProblemSpec authored here solves under `amico-run` with no edits
- [ ] Shared device / experiment-card schemas where they overlap; calibration conformance when Block 10 is available
- [ ] Applicable quality evidence recorded
- [ ] **Done:** both run-directory and ProblemSpec conformance checks hold
- Optional: amicode Domain Pack, if wanted by both sides.

---

## Upgrade Path (no boxes until a milestone claims them)

Multi-GPU cuDensityMat · cuTensorNet engine for lattice-scale systems · multistart and continuation for control · longer scheduled unattended campaigns after bounded workflows demonstrate reliability · hardware-in-the-loop behind the same run contract.

Stronger authorization and isolation are scoped when a deployment needs bypass resistance or containment of untrusted code. Record its threat model and required identity/process/filesystem/network/device controls before adding acceptance boxes; cooperative checks do not prove those guarantees.
