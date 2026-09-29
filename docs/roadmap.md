# Entanglix — Implementation Plan

**Date:** 2026-09-28
**Project:** Entanglix — quantum device physics research harness: composable modeling, simulation, search, control, and calibration
**Status:** Direction pinned on 2026-09-28: expert-guided research, bounded autonomy, independently useful workflows, and professional quality across milestones. Authoritative for scope, milestones, and build dependencies. Project name: Entanglix; command: `entanglix` (no `-run` suffix). `amico-run` remains the external Piccolo launcher. Concrete choices still open are listed in §7.
**Authority:** [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md) derives enforceable guarantees from this roadmap; [`checklist.md`](checklist.md) tracks implementation and acceptance evidence. [`adr/0001-mandate.md`](adr/0001-mandate.md) is the Draft agenda for Decision 0; recommendations do not settle open choices.
**Source documents:** amicode `packages/extension/CONTRACT.md` (run-dir shape), amicode `packages/schema/schemas/problemspec.oss.schema.json` (control spec), amicode ADR-0008 (domain-pack boundary), ADR-0011 (harness contract).

---

## 0. Working Assumptions

### Mandate

Entanglix enables expert-guided quantum system research through composable modeling, simulation, search, control, and calibration capabilities. Agents assist with problem formulation and carry out bounded investigations under explicit assumptions and resource limits. Users can enter at any supported stage and progressively compose these capabilities into increasingly autonomous design workflows.

- Domain: superconducting-circuit device physics and quantum control. System modeling (circuit parameters → Hamiltonians), open-system simulation on GPU, and three optimization tasks: optimal control (pulse/gate synthesis), model calibration (fit Hamiltonian and noise parameters to data), and device design (optimize circuit parameters against simulated metrics).
- Not in scope: gate-model algorithms (VQE, QAOA, annealing schedules), circuit compilation, hardware drivers.
- Users: the author and close collaborators; expert users. No product, no UI, no VS Code extension. The agent runtime is a commodity (Codex CLI or equivalent) driven through `AGENTS.md`, skills, and an MCP server.

### Research workflow and autonomy

- **Problem formulation is part of the workflow.** A broad ambition such as "design a five-qubit processor with two-qubit gate fidelity > 0.999" starts an expert–agent discussion about architecture, assumptions, constraints, models, and the evidence needed. Expensive work begins once the next investigation is sufficiently defined.
- **Autonomy is bounded by the investigation.** A task may explore a parameter region, investigate an unexpected result, refine a model, or compare candidates. Its question, permitted changes, assumptions, evidence requirements, resource limits, and stopping or review criteria are explicit. Users choose where to retain checkpoints and where to delegate execution.
- **Useful findings are valid outcomes.** An investigation may yield a promising candidate, a characterized region, an infeasibility finding within the stated assumptions, or unresolved uncertainty with suggested next experiments. A finished device and a single ultimate optimization objective are not prerequisites for useful work.
- **Capabilities have independent entry points.** Modeling → sweep → search → refined sweep is a complete research workflow. Control can start from an existing device model; calibration can start from measured data. Device–control co-design composes these capabilities when the research question calls for it.
- **Milestones describe capability development.** Users may enter, iterate within, and stop at any supported stage. The build progression does not prescribe a single user journey.

### Enterprise-grade quality

"Enterprise grade" means professional, high-quality, mature scientific software. Reproducibility is one part of that standard:

- **Scientific quality:** validated methods, explicit assumptions, uncertainty and convergence evidence, independent verification, and results reproducible from specs, code versions, and recorded environments.
- **Engineering quality:** maintainable architecture, appropriate automated tests, dependable execution, manageable dependencies, and explicit compatibility expectations.
- **Operational quality:** controlled launches, clear diagnostics, interruption handling and recovery, resource accounting, and observable execution. GPU work goes through the runner and its executor; cluster execution uses the site's scheduler.
- **Professional usability:** coherent CLI/MCP interfaces, useful documentation and examples, actionable errors, and predictable, versioned releases.

Every milestone must demonstrate these qualities for the capabilities it introduces. The acceptance evidence in §4 develops alongside functionality.

### Substrate decisions (fixed)

- **Python-first, single substrate.** Modeling with `scqubits`; dynamics on cuQuantum's cuDensityMat through `qutip-cuquantum`; differentiable dynamics with `dynamiqs` (JAX); QuTiP CPU as the exact small-size reference.
- **Two engines that do not compose through autodiff.** cuDensityMat is large-scale, batched, non-differentiable. JAX (dynamiqs) is differentiable and moderate-size. Data crosses via DLPack / CuPy zero-copy; gradients do not. Engine selection is a declared property of a protocol run ("needs gradients?" × Hilbert-space size), never an agent's judgment call.
- **Solver-agnostic control spec.** amicode's ProblemSpec OSS schema (`system`, `goal`, `pulse`, `problem`, `integrator`, `solver.backend`) is adopted as the control problem file. `solver.backend` is a string; which solver ran is recorded, never assumed.
- **Private solver plugins.** High-performance solvers may be proprietary. They attach through a `SolverBackend` / `Integrator` protocol via Python entry points and are never imported by this repo. The repo's tests run with open solvers only; an import-boundary test enforces it. Verification never depends on a private solver (contract §1, §3).
- **amicode's role is contracts and a corrector, not substrate.** Run-dir shape and schemas are adopted for interoperability; Piccolo (via `amico-run`) is an open second-opinion control solver. Dynamics never route through Julia.

### Key principles

- Prose steers, tools enforce. Invariants live in the runner, the schemas, the corrector, and the environment.
- Stage, never promote. Agents write candidates and results; humans promote into the catalog and the device model.
- Every attempted run contributes to research history, including failed and unpromoted attempts. Human promotion records endorsement; run lineage exists from execution onward.
- Cheap-to-expensive funnel. Spec validation → small-size reference → GPU run → corrector. Nothing skips a stage.
- Truncation and convergence are reported on every run, never assumed.

---

## 1. Architecture

```text
Model plane      system.toml ──► scqubits circuit ──► truncated operators, dissipators, parameter priors
Engine plane     QuTiP CPU (reference) │ cuDensityMat via qutip-cuquantum (large, batched) │ dynamiqs (JAX, differentiable)
Protocol plane   simulate │ sweep │ calibrate │ control        (each: spec in, run dir out)
Workflow plane   problem formulation │ bounded sweep/search │ device–control co-design
Solver plane     open GRAPE baseline │ Piccolo via amico-run │ private plugins (entry points)
Run plane        research_runs/<system_id>/<run_id>/ … FINISHED   (amicode run-dir shape)
Verify plane     corrector: cross-engine, cross-solver, convergence, positivity, holdout   (hashed, outside agent tree)
Evidence plane   run ledger (all attempts, lineage, trial counts) + human-promoted results catalog
Harness plane    entanglix CLI  ⇄  MCP server  ⇄  skills + AGENTS.md  ⇄  Codex (or any runtime)
Executors        local GPU │ Slurm │ Kubernetes   (artifacts mirrored back; large arrays stay on the cluster)
```

The workflow plane composes available protocols around a research question. Sweep/search is usable independently; control and calibration extend the available workflows. The detailed invariants behind this diagram — object model, engine-selection table, run directory and exit codes, solver and integrator protocols, corrector checks — belong in [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md) §1–§3.

---

## 2. Build Blocks and Dependencies

No calendar. A block is done when its functional criteria and the applicable quality criteria in §4 hold. Dependencies below govern what must be available; numbering does not impose additional dependencies.

The core capability progression is **foundation → verified local sweep → bounded agent-assisted exploration → control → device–control co-design**. Cluster execution, calibration, catalog, and interoperability are supporting capabilities with their own dependencies. This is the preferred sequence; technical prerequisites are listed separately. Sweep/search must meet its own quality gates independently of control.

### Decision 0 — Mandate ADR

- Name, command, and repo are settled: Entanglix, `entanglix`, this repository. License for the shareable harness, and who may see it, remain open.
- First platform family and fixture systems (recommendation: transmon and fluxonium, matching existing scqubits work).
- Target local GPU machine and environment policy. Actual compatible version pins are recorded during foundation and engine validation; hardware measurements do not gate scaffolding.
- Unit conventions, including angular vs. cyclic frequency, time-grid conventions, and default truncations.
- **Done when:** `docs/adr/0001-mandate.md` records these foundation choices and is marked Accepted; unblocks Block 0. The first sweep metric is chosen before Block 5; investigation budgets and evaluation design follow M2 measurements before Blocks 6a–6b.

### Block 0 — Repo Foundation

- **Depends on:** Decision 0.
- Layout: `packages/entanglix` (library and the `entanglix` command), `packages/entanglix-mcp`, `schemas/`, `skills/`, `docs/`, `tests/`; `uv` project and dependency lockfile; environment manifest and hashing convention, with compatible GPU version pins completed during Block 2.
- CI: unit tests on CPU (no GPU required), a `gpu` marker for GPU tests, an **import-boundary test** asserting no module outside `tests/` imports a private plugin or a scheduler client.
- `HARNESS_CONTRACT.md` and `docs/README.md` exist (2026-09-28); `AGENTS.md` and the `CLAUDE.md` one-line pointer are written once the layout is real. `AGENTS.md` provides coding-agent working instructions and navigation; research-runtime permissions remain in the contract.
- Contributor setup, architecture overview, test instructions, and an initial versioning and compatibility policy.
- **Done when:** clone → `uv sync` → `pytest` green on a machine without a GPU.

### Block 1 — System Spec and Model Adapter

- **Depends on:** Block 0.
- `system.toml` JSON schema and pydantic model; scqubits adapter (circuit → truncated operators in a chosen basis); explicit-operator fallback; dissipators; drive operators; parameter values + priors; unit handling.
- Operator export to QuTiP `Qobj` and to arrays for JAX; storage-format selection for cuDensityMat operators (reuse the CSR-probe decision rule already prototyped in the scqubits GPU work).
- `entanglix validate <system.toml>`.
- **Done when:** the Decision-0 fixture systems validate, export to the three engine input formats, and spectra match scqubits to tolerance in tests.

### Block 2 — Engines v0

- **Depends on:** Block 1.
- QuTiP CPU reference engine; cuDensityMat engine via `qutip-cuquantum`; dynamiqs engine; each behind the `Integrator` protocol with `capabilities()`.
- DLPack / CuPy zero-copy handoff for arrays between cuDensityMat and JAX where needed.
- Cross-engine agreement tests at reference size (closed and open system, time-dependent drive).
- Verify the target GPU environment; record GPU, driver, CUDA, cuQuantum, and package versions in the hashed environment manifest and `docs/environment.md`.
- **Done when:** the engine-selection table (contract §2) is a versioned file, and the agreement tests are green on GPU CI or a documented manual GPU run.

### Block 3 — Runner and Run Contract

- **Depends on:** Blocks 0–2.
- `entanglix` gate + launch: spec validation, static import scan of any authored script, environment check (driver, cuQuantum, memory), run-dir creation (`run.toml` first), job spawn in its own process group, `FINISHED` last, exit-code contract, `index` and `latest` at the runs root.
- Local executor only in this block.
- Basic append-only run ledger from the first execution: provenance, parent runs, resource use, and outcome for completed, failed, and aborted attempts. This history is available before catalog promotion or agent use.
- Diagnostics and a documented recovery or rerun path for interrupted work.
- **Done when:** a killed job leaves no `FINISHED`; a completed job's `result.toml` validates; a run is reproducible from `spec/` + `code_version` + `env_hash`; attempts and their lineage remain queryable after interruption.

### Block 4 — Corrector v0

- **Depends on:** Blocks 1–3.
- Cross-engine, convergence, positivity checks as a separate package with hashed code and a versioned thresholds file; `compare_reference` verb.
- The corrector derives applicable checks from trusted protocol/metric definitions and the validated execution plan. Agent-written labels cannot waive checks; design-sweep checks add to the applicable dynamics or spectral checks.
- **Done when:** a deliberately under-converged run is flagged `UNVERIFIED`; a hash mismatch fails the run; mislabeled studies cannot evade required checks. Verification is available before completion of Block 5.

### Block 5 — Protocols `simulate` and `sweep` (first end-to-end)

- **Depends on:** Blocks 1–4.
- Standalone simulation and a grid sweep over scqubits parameters against declared observables or a simulated metric (for example dispersive shift, T1 limit from a noise model, gate error under a fixed pulse).
- The first end-to-end example exercises open-system dynamics on the local GPU; this class of metric is settled. Choose the exact observable, fixed pulse if needed, parameter region, and evidence thresholds before implementation of this block. Spectral-only studies remain useful supported workflows.
- Plots and `result.toml` summaries with enough evidence for a user to inspect results and choose a refined sweep.
- **Done when:** a documented simulation and one agreed dynamics sweep run end to end through `entanglix`, the sweep exercises the local GPU, corrector checks pass, and a user can inspect and refine the sweep without control or calibration.

### Investigation decisions — after M2, before Blocks 6a–6b

Use measured sweep costs to define the first investigation's question, scope, assumptions, evidence requirements, numeric allowances, and stop/review criteria. A complete device-design target is optional. Before Block 6a, record the v1 grant/issuer model, resource metering, reservation and cancellation policy, and evaluation protocol in a follow-up ADR (planned ADR-0002). Ordinary investigation specifications and grants remain research records, not new ADRs. Before Block 6b, select and document the local process, filesystem, credential, and GPU isolation mechanism.

The evaluation protocol names a scripted experiment-selection policy, comparable resource allowances, evidence thresholds, and measures of scientific outcome, elapsed time, compute use, and human interventions. A scripted client alone is not a scientific baseline. For optimization tasks, compare best verified objective or time to a feasible candidate; for exploratory tasks, use the declared evidence requirements.

### Block 6a — Investigation Execution

- **Depends on:** Blocks 3–5 and the decisions identified above as required before Block 6a.
- Runner-managed investigation records, immutable revisions, findings, and resumable histories.
- Protected grants issued through an authenticated human administrative channel: scope, typed resource ceilings, review conditions, issuer, validity, and revocation. The runner enforces the same authority checks for CLI, MCP, and submitted code; access to the CLI binary confers no administrative authority.
- Resource reservation and accounting across concurrent jobs, nested solves, verification, retries, and failures; review/stop gating and cancellation. Contract §4 defines the resource units; choose numeric allowances using M2 measurements.
- Scripted iterative sweep/search client, with Bayesian search where appropriate. Document its experiment-selection policy so it can serve as the M3 baseline.
- **Done when:** the scripted client completes the declared investigation or a supported review/stop handoff; evidence, revisions, and accounting survive restart; concurrent dispatch cannot exceed grants; unauthorized grant changes and review releases are rejected. Fault cases demonstrate enforcement independently of an agent runtime.

### Block 6b — Agent Integration

- **Depends on:** Block 6a and the local isolation decision.
- Support expert–agent problem formulation, explicit review points, and bounded execution after the next investigation is defined. A broad device ambition is refined before it becomes a job request.
- MCP server over stdio exposing available CLI capabilities; tool-call and prompt audit logs; initial skills (`system-modeler`, `simulate`, `design-sweep`, `gpu-budget`); `AGENTS.md` finalized for these workflows. Control and calibration tools and skills arrive with their capabilities.
- Local process/filesystem isolation and separate administrative credentials protect grants, runner policy, corrector, ledger, catalog, and device model. The research agent retains a shell but cannot directly use the GPU or exercise human administrative authority; submitted jobs have bounded access through the executor.
- Prompt-injection, shell/CLI escalation, and isolation tests; a pinned runtime version in CI. Numerical search and agent reasoning have distinct, documented roles.
- **Done when:** an agent completes the declared sweep/search investigation through logged tools, with an evidence-driven refinement and a justified stop or review handoff; applicable evidence requirements and quality gates pass; resource and isolation guarantees hold. The workflow is documented and usable independently of control and calibration.
- **M3 evaluation:** execute the declared comparison against the Block 6a baseline and report outcomes, including worse performance or inconclusive benefit. M3 passes on correct execution, required evidence, enforced limits/isolation, independent usability, and completion of the comparison; outperforming the baseline is not required. Claims of acceleration require measured comparative benefit.

### Block 7 — Protocol `control`

- **Builds on:** Blocks 1–4 for standalone control; Block 6b for agent integration. Prefer to sequence after M3; that order is not a technical prerequisite. Cluster execution and calibration are optional.
- Adopt ProblemSpec OSS, resolving the additive `solver.options` choice before implementation; `SolverBackend` protocol and entry-point discovery; open GRAPE-style baseline; Piccolo adapter via `amico-run`.
- Cross-solver corrector check (contract §3) and re-simulation on a different engine.
- Standalone control on an existing system model through the CLI; extend MCP and add the `control` skill when Block 6b is available. Existing sweep/search workflows remain independently supported.
- **Done when:** one gate (Decision-0 platform) is solved by the open baseline and by Piccolo from the same ProblemSpec, both results verified, both in run dirs; a private plugin, if present, is discovered and its result is verified by the open solver.

### Block 8 — Device–Control Co-design

- **Depends on:** Blocks 5, 6a, 6b, and 7.
- Compose device search and control: propose circuit parameters → construct the model → optimize admissible controls → evaluate verified performance → refine the device search.
- Declare device and control constraints, relevant parameter uncertainty, and a consistent evaluation and optimization budget for comparing candidates.
- Support expert checkpoints and bounded agent execution, with evidence linking device candidates to their control runs. Calibration can supply model updates once available.
- **Done when:** one documented investigation iterates jointly on circuit parameters and controls, reports a verified candidate comparison under the declared constraints and uncertainty, and ends with a supported recommendation or an explicit account of limitations. Users can still choose the standalone sweep/search or control workflows.

### Block 9 — Cluster Executor

- **Depends on:** Blocks 3–5. Build when remote compute is needed; it does not gate local exploration or control.
- Slurm or Kubernetes executor behind the same `Executor` interface as local; artifact mirroring (small arrays back, large arrays referenced by path + digest); warming and lost-contact timeouts; abort.
- **Done when:** `sweep` runs unchanged on the cluster and locally; a lost job terminates as `aborted`, never as `completed`; resource use and recovery diagnostics are recorded.

### Block 10 — Protocol `calibrate`

- **Depends on:** Blocks 1–4; Block 6b for agent access. Control and co-design are optional.
- Data contract for calibration inputs (measured spectra, Ramsey/T1/T2 traces, gate-error tables) with point-in-time provenance.
- Gradient lane (dynamiqs) and gradient-free / Bayesian lane; holdout split; identifiability report; **write-back to the device model is a proposal file a human applies**.
- Documented standalone calibration workflow; extend the MCP surface and add the `calibrate` skill when the agent harness is available.
- **Done when:** synthetic-data round trip recovers known parameters within reported uncertainty; the holdout is unreachable from the fitting code path; a user can inspect a calibration report and device-update proposal independently of control.

### Block 11 — Results Catalog

- **Depends on:** Blocks 3–5; Block 7 for pulse warm-starts.
- Build on the run ledger: catalog of human-promoted results, provenance and lineage queries, trial counts per hypothesis family, and reusable candidates for subsequent investigations.
- Promotion is a human command. Failed and unpromoted run history remains queryable independently of the catalog.
- **Done when:** a subsequent sweep reuses a catalogued candidate with recorded lineage; promotion is auditable; once control is available, a control run warm-starts from a catalogued pulse.

### Block 12 — Interoperability

- **Depends on:** Blocks 3 and 7; shared calibration schemas extend conformance when Block 10 is available.
- Conformance: run dirs readable by amicode's Run Inspector; ProblemSpec files accepted by `amico-run` unchanged; shared device / experiment-card schemas.
- Optional: package as an amicode Domain Pack if wanted by both sides.
- **Done when:** a `control` run dir opens in the Run Inspector; a ProblemSpec authored here solves under `amico-run` with no edits.

### Upgrade path (not blocks)

Multi-GPU cuDensityMat; cuTensorNet engine for lattice-scale systems; multistart and continuation for control; longer scheduled unattended campaigns after bounded workflows demonstrate reliability; hardware-in-the-loop via an external device bridge behind the same run contract.

---

## 3. Harness Contract

[`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md) defines bounded investigations (§4), tools and permissions (§5), history/audit/promotion (§6), isolation (§7), and versioning/conformance (§8). Quality standards and open decisions remain in this roadmap (§4 and §7); the checklist tracks acceptance evidence. Working instructions and research procedures arrive through `AGENTS.md` and skills at their planned stages.

Block 6a adds investigation execution to the runner and ledger from Block 3 and corrector from Block 4; Block 6b adds isolated agent access through MCP. Blocks 7 and 10 extend agent access to control and calibration when both capabilities are present; Block 11 adds the promoted catalog.

---

## 4. Standing Disciplines (every block)

Each milestone's exit includes evidence for the capabilities it introduces:

| Quality dimension | Required acceptance evidence |
|---|---|
| Scientific | Representative reference cases; documented assumptions and applicable uncertainty, convergence, and verification reports; reproducible examples |
| Engineering | Appropriate automated tests and regression coverage; documented interfaces and dependencies; supported environment and compatibility expectations |
| Operational | Observable execution and resource use; actionable failures; interruption and recovery behavior tested where jobs are introduced; budget enforcement for delegated work |
| Professional usability | A documented entry point and worked example; clear diagnostics; versioned changes and release notes for user-visible behavior |

Apply these criteria to each block's scope: foundation proves installation and development workflows; later blocks add scientific and operational evidence for their capabilities. M3 must meet them for a complete sweep/search workflow. The preferred sequence puts M3 before control; independent control development retains its own quality gates.

- Pin CUDA, cuQuantum, driver, and Python package versions; record an environment hash in every `run.toml`.
- Seeds recorded; determinism tests for engines where the vendor guarantees it, tolerance tests where it does not.
- Units and time-grid conventions from Decision 0 everywhere; no implicit unit conversion.
- Every run reports truncation levels and a convergence check; every `control` run reports constraint satisfaction outside the solver.
- Tests on anything touching money-equivalents: GPU hours, cluster submission, catalog promotion.

---

## 5. Milestones

Core progression (capability development; users may enter or stop at any supported stage):

| Milestone | Blocks | Exit |
|---|---|---|
| M0 — Mandate + foundation | Decision 0, Block 0 | mandate ADR accepted; CPU tests green; contributor and compatibility guidance available |
| M1 — Model + engines | Blocks 1–2 | spectra match scqubits; engines agree at reference size |
| M2 — Verified local sweep | Blocks 3–5 | documented local GPU sweep, verified and reproducible; run history and interruption diagnostics available |
| M3 — Bounded agent-assisted exploration | Blocks 6a–6b | declared evidence, correct execution, resource/isolation guarantees, independent workflow usability, and baseline comparison pass Block 6b criteria; no performance advantage required |
| M4 — Control | Block 7 | standalone control workflow; open baseline and Piccolo solve the same gate problem, both verified; agent integration demonstrated when Block 6b is available |
| M5 — Device–control co-design | Block 8 | joint device/control investigation produces a verified comparison and supported findings under declared constraints and uncertainty |

Supporting milestones are scheduled by their dependencies and research need:

| Milestone | Blocks | Exit |
|---|---|---|
| S1 — Cluster execution | Block 9 | same sweep on the scheduler; resource accounting and failure handling demonstrated |
| S2 — Calibration | Block 10 | standalone synthetic round trip with uncertainty and holdout evaluation; inspectable model-update proposal |
| S3 — Results catalog | Block 11 | promoted candidate reused with lineage; pulse warm-start demonstrated when control is available |
| S4 — Interoperability | Block 12 | Run Inspector opens a run; `amico-run` accepts a ProblemSpec |

The Harmoniqs conversation is gated by concrete artifacts: the shared ProblemSpec solved by the open baseline and Piccolo, both verified, with inspectable run directories and corrector reports. These are normally delivered at M4, but standalone control may supply them before M3. Agent integration is not a sharing prerequisite; the audience/license decision and removal of Appendix A still apply.

---

## 6. Things Not to Build Yet

- Any UI or editor extension.
- Gate-model algorithm layers.
- A scheduler or job queue of our own.
- Hardware drivers.
- Unattended multi-day campaigns before bounded workflows and their operational quality are demonstrated.
- A Julia substrate.

---

## 7. Open Decisions

Settled: project name Entanglix; command `entanglix`; runner in Python (`uv`), same substrate as the engines. Expert-guided problem formulation and bounded autonomous investigations; independently useful entry points; the core milestone progression in §5; supporting capabilities scheduled by dependency and need; enterprise-grade quality across every milestone. The initial agent harness supports sweep/search; control and calibration extend it as those capabilities become available. The first end-to-end sweep exercises open-system GPU dynamics; its exact metric remains open.

- **Decision 0 (Blocks 0–2):** license/audience and sharing policy; platform family and fixture systems; target local machine and environment policy; units, time grid, and truncation defaults.
- **Before Block 5:** exact dynamics observable/metric, pulse if needed, parameter region, and evidence thresholds for the first GPU sweep.
- **After M2, before Block 6a:** first investigation and evaluation protocol, including a defined scripted selection policy and comparable budgets; v1 grant issuer/authentication, resource metering, reservation, and cancellation policy. Contract §4 fixes resource units; numeric allowances use measured costs.
- **Before Block 6b:** concrete local isolation mechanism covering processes, filesystem, credentials, GPU access, and the authenticated human channel. A separate OS identity and device controls or a container/privilege boundary must enforce the contract; naming commands "human-only" is insufficient.
- **Before Block 7:** ProblemSpec OSS verbatim vs. an additive `solver.options` table.
- **Before Block 9:** scheduler target (Slurm or Kubernetes) and remote environment when needed.

---

## Appendix A — Private solver plugins

- A private solver stays in its own package outside this repository. Entanglix code and docs never name or import it, and need no change to use it.
- The package registers a backend factory under the `entanglix.solvers` entry-point group, implements the solver interface in the [control reference](operations/control.md#launch-and-solver-selection), and pins the version of the private solver it wraps.
- Each run records the requested and actual solver and its distribution version. A spec that names a private solver can declare an open `fallback`, used only where the plugin is not installed.
- A private solver's results receive the same open checks as any other solver's, and no check depends on a private solver.
- Evolution engines have no plugin entry point yet. One would follow the same pattern when a plugin needs it, with the open engines remaining the default.

## Appendix B — amicode touchpoints (read-only references)

- `packages/extension/CONTRACT.md` — run-dir shape and exit codes adopted in contract §2.
- `packages/schema/schemas/problemspec.oss.schema.json` — the control spec adopted in Block 7.
- `packages/amico-run/src/local_executor.ts`, `remote_executor.ts`, `types.ts` — reference for the executor interface and mirroring lifecycle.
- `packages/extension/julia/verify_rollout.jl` — their re-rollout corrector; ours differs by using a different engine and solver.
- `docs/adr/0008-autoresearch-identity-domain-pack-boundary.md`, `docs/adr/0011-harness-contract.md` — domain-pack boundary and harness-as-contract, the basis for Block 12.
