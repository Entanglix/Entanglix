# Entanglix Build Checklist

**Status:** Implementation tracker for the pinned [`roadmap.md`](roadmap.md) and derived [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md). The roadmap defines scope, milestones, and dependencies; the contract defines enforceable behavior; this file tracks completion and acceptance evidence.

Status legend: `[ ]` open · `[-]` in progress / partial · `[x]` done. Dependencies below follow the roadmap. Supporting milestones can proceed when their own prerequisites are ready. A block is complete when its required functional and applicable quality criteria are checked; numbering imposes no additional dependencies. No calendar: exit criteria gate progress, not dates.

The mandate ADR is Draft. Complete and accept its foundation decisions during Decision 0; later choices are tracked with the blocks that need them.

---

## Milestone Map

Core capability progression; users may enter, iterate within, and stop at any supported stage:

| Milestone | Blocks | Status |
|---|---|---|
| M0 — Mandate + foundation | Decision 0, Block 0 | **in progress** |
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

### [-] Decision 0 — Mandate ADR

Block 0 implementation begins once `docs/adr/0001-mandate.md` reads **Accepted**. The preparatory documentation already exists. Items marked *walkthrough* need the owner's decision; the draft's recommendations remain proposals.

- [x] Project name: Entanglix. Command: `entanglix`. Repository established.
- [x] Runner language: Python (`uv`), same substrate as the engines.
- [x] Substrate fixed (roadmap §0): scqubits, cuDensityMat via `qutip-cuquantum`, dynamiqs, QuTiP CPU reference; ProblemSpec OSS control spec; private solvers as entry-point plugins.
- [x] Research direction pinned: expert-guided formulation, bounded autonomy, independently useful workflows, and enterprise-grade quality across milestones.
- [x] Core progression pinned: verified local sweep → bounded agent-assisted exploration → control → co-design; supporting capabilities follow their own dependencies.
- [ ] Platform family and fixture systems for v1 — *walkthrough*
- [ ] Target local GPU machine and environment policy — *walkthrough*; actual pins recorded during foundation/engine validation
- [ ] Unit conventions (energy, time, angular vs. cyclic frequency), time-grid convention, and default truncations — *walkthrough*
- [ ] License for the shareable harness and who may see it — *walkthrough*
- [ ] Record foundation decisions in the Draft mandate ADR during the walkthrough; status → Accepted
- **Unblocks:** Block 0.

### [-] Block 0 — Repo Foundation

**Depends on:** Decision 0. Current progress is preparatory documentation only.

- [x] `HARNESS_CONTRACT.md` at repo root, aligned with the pinned roadmap
- [x] `docs/README.md` index with the sharing rule
- [x] `docs/roadmap.md` and `docs/checklist.md`, with aligned milestones and dependencies
- [ ] Layout: `packages/entanglix`, `packages/entanglix-mcp`, `schemas/`, `skills/`, `tests/`
- [ ] `uv` project and `uv.lock`; environment manifest and hashing convention; GPU pins completed in Block 2
- [ ] `.gitignore` (runs, caches, `.env`, large arrays)
- [ ] CI: CPU tests for foundation behavior; `gpu` pytest marker skipped when no device
- [ ] Import-boundary test: no direct private-implementation imports or scheduler-client imports outside `tests/`, per roadmap Block 0
- [ ] Short `AGENTS.md` for coding-agent working instructions, navigation, and commands; `CLAUDE.md` one-line pointer; research-runtime permissions referenced separately
- [ ] Contributor setup, architecture overview, test instructions, initial versioning and compatibility policy
- [ ] Applicable quality evidence recorded
- [ ] **Done:** clone → `uv sync` → `pytest` green on a machine without a GPU

---

## M1 — Model + Engines (Blocks 1–2)

### [ ] Block 1 — System Spec and Model Adapter

**Depends on:** Block 0.

- [ ] `system.toml` JSON schema + pydantic model (circuit, truncation, drives, dissipators, parameters + priors, units)
- [ ] scqubits adapter: circuit → truncated operators in a chosen basis
- [ ] Explicit-operator fallback
- [ ] Operator export to QuTiP `Qobj` and to arrays for JAX
- [ ] Storage-format selection for cuDensityMat operators (CSR-probe rule from the scqubits GPU work)
- [ ] `entanglix validate <system.toml>` / `validate_system` backing capability
- [ ] Fixture systems confirmed in Decision 0; roadmap currently proposes transmon–resonator and fluxonium
- [ ] Applicable quality evidence recorded
- [ ] **Done:** the Decision-0 fixture systems validate and export to the three engine input formats; spectra match scqubits to tolerance

### [ ] Block 2 — Engines v0

**Depends on:** Block 1.

- [ ] `Integrator` protocol with `capabilities()`
- [ ] QuTiP CPU reference engine
- [ ] cuDensityMat engine via `qutip-cuquantum`
- [ ] dynamiqs engine
- [ ] DLPack / CuPy zero-copy handoff where needed
- [ ] Cross-engine agreement tests at reference size (closed, open, time-dependent drive)
- [ ] Verified environment pins in the hashed manifest and `docs/environment.md`: GPU, driver, CUDA, cuQuantum, Python, and engine packages
- [ ] Versioned engine-selection table (contract §2); configured memory/size limits; declared fallbacks recorded without silently changing required capabilities
- [ ] Applicable quality evidence recorded
- [ ] **Done:** agreement tests green on GPU CI or a documented manual GPU run

---

## M2 — Verified Local Sweep (Blocks 3–5)

### [ ] Block 3 — Runner and Run Contract

**Depends on:** Blocks 0–2.

- [ ] Gate: spec validation, static import scan, environment check (driver, cuQuantum, memory)
- [ ] Run dir per contract §2: `run.toml` first, `spec/` frozen, `FINISHED` last, atomic `result.toml`
- [ ] Reproduction inputs: authored code/data references, versions or digests, seeds, solver settings, and environment provenance
- [ ] Job spawn in its own process group; exit-code contract; per-run resource limits
- [ ] `index` and `latest` at the runs root
- [ ] Local executor, diagnostics, and documented interruption recovery/rerun path
- [ ] Append-only ledger from first execution: parent run IDs, provenance, outcomes, resource use, and verification status when available; include failed, aborted, and incomplete attempts
- [ ] Atomic ledger events retain concurrent attempts; reruns get new IDs and preserve lineage
- [ ] `list_capabilities`, `list_runs`, `read_run`, and `abort_run` backing capabilities
- [ ] Execution outcome remains distinct from scientific verification
- [ ] Applicable quality evidence recorded
- [ ] **Done:** abrupt termination leaves no fabricated completion; completed results validate and reproduce; interrupted attempts remain queryable with diagnostics and lineage

### [ ] Block 4 — Corrector v0

**Depends on:** Blocks 1–3.

- [ ] Separate package; code hashed at load; versioned thresholds outside the agent-editable tree
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
- [ ] Plots and `result.toml` summaries expose evidence needed to inspect and refine a sweep
- [ ] Documented CLI examples for simulation and iterative sweep refinement
- [ ] Applicable quality evidence recorded
- [ ] **Done:** a documented simulation and agreed dynamics sweep run through `entanglix`, GPU execution and corrector checks pass, and a user can inspect/refine the sweep without control or calibration

---

## M3 — Bounded Agent-assisted Exploration (Blocks 6a–6b)

### [ ] Investigation decisions — after M2

- [ ] Before Block 6a: first investigation question, scope, assumptions, evidence requirements, numeric resource allowances, and stop/review criteria informed by measured M2 costs
- [ ] Before Block 6a: v1 grant issuer/authentication, metering, reservation, and cancellation policy recorded in a follow-up ADR (planned ADR-0002); routine investigation specs/grants remain research records
- [ ] Before Block 6a: evaluation protocol defines the scripted experiment-selection policy, comparable allowances, evidence thresholds, and outcome/time/compute/human-intervention measures
- [ ] Before Block 6b: local process, filesystem, credential, and GPU isolation mechanism selected and documented, including an authenticated human channel inaccessible from the agent shell and jobs

### [ ] Block 6a — Investigation Execution

**Depends on:** Blocks 3–5 and the decisions identified above as required before Block 6a.

- [ ] Investigation records and immutable revisions: question, scope, assumptions, permitted changes/protocols, evidence requirements, limits, stop/review criteria, and grant reference (contract §4)
- [ ] Protected grants record authenticated issuer, authorized principal, scope bounds, typed ceilings, review conditions, validity, and revocation
- [ ] Human administrative operations: `authorize`, `raise-limit`, `release-review`, `revoke`; backing implementation authenticates authority independently of request fields on every transport
- [ ] `create_investigation`, `revise_investigation`, `list_investigations`, `read_investigation`, `record_finding`, and `stop_investigation`
- [ ] Findings and run provenance retain grant/investigation revisions, evidence, verification status, and limitations
- [ ] Typed resource units and accounting semantics follow contract §4; unknown units rejected; finite configured allowances and conservative bounds enforced
- [ ] Atomic reservations cover concurrent jobs, nested solves, verification, retries, failures, and termination overhead; uncertain completion retains reservations
- [ ] New investigations, revisions, and sessions cannot reset the shared grant balance; standalone research-runtime jobs also require grants
- [ ] Review/stop state, expiry/revocation, cancellation, and executor deadlines gate and bound work; resume preserves accounting and stop reasons
- [ ] Authorization events record issuance/amendments, limit changes, review releases, revocation, and administrative denials
- [ ] Scripted iterative sweep/search client, with Bayesian search where appropriate; documented selection policy qualifies it as the M3 baseline
- [ ] Tests cover scope/issuer spoofing, unauthorized administration/review release, concurrent budget exhaustion, revocation, and restart accounting
- [ ] Applicable quality evidence recorded
- [ ] **Done:** scripted investigation reaches declared evidence or a supported stop/review handoff; restart retains evidence/accounting; limits and authorization hold under concurrency and fault cases without an agent runtime

### [ ] Block 6b — Agent Integration

**Depends on:** Block 6a and the local isolation decision.

- [ ] Formulation workflow refines broad ambitions into bounded next investigations; partial workflows and exploratory questions are supported
- [ ] Agent makes evidence-driven sweep/search refinements; roles of numerical search and agent reasoning documented
- [ ] MCP over stdio uses CLI backing operations, advertises available capabilities, and enforces caller authority (contract §5)
- [ ] Tool-call and prompt audit logs (contract §6), including scope revisions and linked run IDs
- [ ] Initial skills: `system-modeler`, `simulate`, `design-sweep`, `gpu-budget`; versioned metadata; `AGENTS.md` links to available workflows
- [ ] Local isolation protects grants, accounting, runner, corrector, ledger, catalog, device model, and human credentials; research-agent shell has no direct GPU or administrative access; jobs remain confined (contract §7)
- [ ] Prompt-injection, direct shell/CLI escalation, and process/filesystem/device isolation cases; pinned runtime version in CI
- [ ] Execute the declared comparison against the scripted baseline; report scientific outcome, elapsed time, compute use, human interventions, and limitations, including worse or inconclusive performance
- [ ] Applicable quality evidence recorded for the complete independent sweep/search workflow
- [ ] **Done:** agent completes the declared investigation through logged tools with evidence-driven refinement and justified stop/review handoff; required evidence, correct execution, enforced resources/isolation, and independent usability pass; baseline comparison complete
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

**Depends on:** Blocks 5, 6a, 6b, and 7.

- [ ] Compose circuit-parameter proposal, model construction, admissible-control optimization, verified evaluation, and device-search refinement
- [ ] Declared device/control constraints, relevant parameter uncertainty, and consistent candidate comparison and optimization budgets
- [ ] Candidate lineage links device/model, control, and evaluation runs, including unsuccessful attempts
- [ ] Nested work inherits investigation limits, verification, and user-chosen review points
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
- [ ] Cluster preserves local isolation, budget, audit, and lineage guarantees
- [ ] Applicable quality evidence recorded
- [ ] **Done:** `sweep` runs unchanged locally and on the cluster; a lost job is aborted, never completed; resource use and recovery diagnostics are recorded

---

## S2 — Calibration (Block 10)

### [ ] Block 10 — Protocol `calibrate`

**Depends on:** Blocks 1–4; Block 6b for agent access. Control and co-design are optional.

- [ ] Calibration input contract with point-in-time provenance
- [ ] Gradient lane (dynamiqs) and gradient-free / Bayesian lane
- [ ] Holdout evaluation and identifiability report; fitting code cannot access holdout
- [ ] `propose_device_update` proposal file; separate human apply command
- [ ] Documented standalone calibration and inspectable report, independent of control
- [ ] When Block 6b is available: MCP capabilities and `calibrate` skill; holdout and device-model boundaries verified for agent access
- [ ] Applicable quality evidence recorded
- [ ] **Done:** synthetic-data round trip recovers known parameters within reported uncertainty; holdout remains unreachable from the fitting path; a user can inspect the report and update proposal

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
