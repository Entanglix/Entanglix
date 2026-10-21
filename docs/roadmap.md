# Entanglix — Implementation Plan

**Date:** 2026-09-28
**Project:** Entanglix — quantum device physics research harness: composable modeling, simulation, search, control, and calibration
**Status:** Direction pinned on 2026-09-28: expert-guided research, bounded autonomy, independently useful workflows, and professional quality across milestones. Authoritative for scope, milestones, and build dependencies. Project name: Entanglix; command: `entanglix` (no `-run` suffix). `amico-run` remains the external Piccolo launcher. Concrete choices still open are listed in §7.
**Authority:** [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md) derives enforceable guarantees from this roadmap; [`checklist.md`](checklist.md) tracks implementation and acceptance evidence. [`adr/0001-mandate.md`](adr/0001-mandate.md) records the Accepted foundation mandate from Decision 0; later choices retain their own gates.
**Source documents:** amicode `packages/extension/CONTRACT.md` (run-dir shape), amicode `packages/schema/schemas/problemspec.oss.schema.json` (control spec), amicode ADR-0008 (domain-pack boundary), ADR-0011 (harness contract).

---

## 0. Working Assumptions

### Mandate

Entanglix enables expert-guided quantum system research through composable modeling, simulation, search, control, and calibration capabilities. Agents assist with problem formulation and carry out bounded investigations under explicit assumptions, review/stop criteria, and any user-specified resource limits. Users can enter at any supported stage and progressively compose these capabilities into increasingly autonomous design workflows.

- Domain: superconducting-circuit device physics and quantum control. System modeling (circuit parameters → Hamiltonians), open-system simulation on GPU, and three optimization tasks: optimal control (pulse/gate synthesis), model calibration (fit Hamiltonian and noise parameters to data), and device design (optimize circuit parameters against simulated metrics).
- Not in scope: gate-model algorithms (VQE, QAOA, annealing schedules), circuit compilation, hardware drivers.
- Users: the author and close collaborators; expert users. No product, no UI, no VS Code extension. The research-agent runtime is a commodity (Codex CLI or equivalent) integrated through MCP and skills; research instructions reach it through a conduct text supplied in the session's initial context, the MCP server's instructions, and skills. Repository `AGENTS.md` and `CLAUDE.md` guide development agents.

### Research workflow and autonomy

- **Problem formulation is part of the workflow.** A broad ambition such as "design a five-qubit processor with two-qubit gate fidelity > 0.999" starts an expert–agent discussion about architecture, assumptions, constraints, models, and the evidence needed. Expensive work begins once the next investigation is sufficiently defined.
- **Autonomy is bounded by the investigation.** A task may explore a parameter region, investigate an unexpected result, refine a model, or compare candidates. Its question, permitted changes, assumptions, evidence requirements, and stopping or review criteria are explicit. Resource usage is recorded; user-specified ceilings and deadlines are optional and enforced when present. Users choose where to retain checkpoints and where to delegate execution.
- **Progress determines continuation.** Start with an informative experiment or optimization segment, inspect verified performance, constraints, convergence, cost, and what was learned, then continue, refine, or stop. Remaining budget is not a reason to continue; insufficient progress can justify an early report with the best available evidence and proposed next experiments. Initial cost measurements do not predict total time to convergence.
- **Useful findings are valid outcomes.** An investigation may yield a promising candidate, a characterized region, an infeasibility finding within the stated assumptions, or unresolved uncertainty with suggested next experiments. A finished device and a single ultimate optimization objective are not prerequisites for useful work.
- **Capabilities have independent entry points.** Modeling → sweep → search → refined sweep is a complete research workflow. Control can start from an existing device model; calibration can start from measured data. Device–control co-design composes these capabilities when the research question calls for it.
- **Milestones describe capability development.** Users may enter, iterate within, and stop at any supported stage. The build progression does not prescribe a single user journey.

### Enterprise-grade quality

"Enterprise grade" means professional, high-quality, mature scientific software. Reproducibility is one part of that standard:

- **Scientific quality:** validated methods, explicit assumptions, uncertainty and convergence evidence, independent verification, and results reproducible from specs, code versions, and recorded environments.
- **Engineering quality:** maintainable architecture, appropriate automated tests, dependable execution, manageable dependencies, and explicit compatibility expectations.
- **Operational quality:** controlled launches, clear diagnostics, interruption handling and recovery, resource accounting, and observable execution. GPU work goes through the runner and its executor; cluster execution uses the site's scheduler.
- **Professional usability:** coherent CLI/MCP interfaces, useful documentation and examples, actionable errors, and predictable, versioned releases.

Every milestone must demonstrate these qualities for the capabilities it introduces. The acceptance evidence in §4 develops alongside functionality. Guarantees match the deployment: professional local research execution comes first; stronger security mechanisms follow an explicit operational need.

### Substrate decisions (fixed)

- **Python-first, single substrate.** Modeling with `scqubits`; dynamics on cuQuantum's cuDensityMat through `qutip-cuquantum`; differentiable dynamics with `dynamiqs` (JAX); QuTiP CPU as the exact small-size reference.
- **Two engines that do not compose through autodiff.** cuDensityMat is large-scale, batched, non-differentiable. JAX (dynamiqs) is differentiable and moderate-size. When a workflow needs direct cross-engine array exchange, use DLPack / CuPy with measured copy/zero-copy behavior; gradients do not cross. Build that boundary with its first consumer, potentially control or co-design. Engine selection is a declared property of a protocol run ("needs gradients?" × declared device × Hilbert-space size), never an agent's judgment call.
- **Solver-agnostic control spec.** A control problem is one generated file in amicode's ProblemSpec OSS schema (`system`, `goal`, `pulse`, `problem`), shared unchanged by every solver. The solver is named in Entanglix's `control` spec; which solver ran is recorded, never assumed.
- **Private solver plugins.** High-performance solvers may be proprietary. They attach through a `SolverBackend` / `Integrator` protocol via Python entry points and are never imported by this repo. The repo's tests run with open solvers only; an import-boundary test enforces it. Verification never depends on a private solver (contract §1, §3).
- **amicode's role is contracts and a corrector, not substrate.** Run-dir shape and schemas are adopted for interoperability; Piccolo, run through its ProblemSpec runner, is an open second-opinion control solver. Dynamics never route through Julia.

### Key principles

- Prose steers, tools enforce. Invariants live in the runner, the schemas, and the corrector, on supported execution paths.
- Lean by default. Reuse process, scheduler, and agent-runtime facilities for timeouts, GPU allocation, accounting, and permissions; build only what they cannot provide. Professional quality with infrastructure proportional to demonstrated needs.
- Stage, never promote. Agents write candidates and results; humans promote into the catalog and the device model.
- Every attempted run contributes to research history, including failed and unpromoted attempts. Human promotion records endorsement; run lineage exists from execution onward.
- Run what the user requests. A protocol run computes and stores its results; built-in checks run on request, and the harness lists the ones relevant to a run. The harness imposes no cap on the systems a user can model.
- Cheap-to-expensive funnel. Spec validation → environment check → GPU run. Nothing skips a stage.
- Truncation and solver settings are recorded on every run; convergence is checked on request, never assumed.

---

## 1. Architecture

```text
Model plane      system.toml ──► scqubits circuit ──► truncated operators, dissipators, parameter priors
Engine plane     QuTiP CPU (reference) │ cuDensityMat via qutip-cuquantum (large, batched) │ dynamiqs (JAX, differentiable)
Protocol plane   simulate │ sweep │ calibrate │ control        (each: spec in, run dir out)
Workflow plane   problem formulation │ bounded sweep/search │ device–control co-design
Solver plane     open GRAPE baseline │ Piccolo (Julia) │ private plugins (entry points)
Run plane        research_runs/<system_id>/<run_id>/ … FINISHED   (amicode run-dir shape)
Verify plane     corrector, on request: integrity, convergence, pulse replay, holdout    (versioned, outside candidate workspace)
Evidence plane   run ledger (all attempts, lineage, trial counts) + human-promoted results catalog
Harness plane    entanglix CLI  ⇄  MCP server  ⇄  skills + research instructions  ⇄  Codex (or any runtime)
Executors        local GPU │ Slurm │ Kubernetes   (artifacts mirrored back; large arrays stay on the cluster)
```

Development and testing run on the owner's workstation with one coordinating agent and one active GPU job by default; a job may still batch calculations. Research runs may also use a cluster through its scheduler (Block 9). Both follow the cooperative trust model in contract §7: unique run directories, frozen inputs, runner-owned history, cancellation, and executor-native timeouts and GPU allocation prevent accidental interference. Supported CLI/MCP paths enforce declared checks and any configured limits. This deployment does not claim containment of arbitrary shell/code bypass. Multi-agent concurrency, a grant issuer, a reservation service, and OS-enforced GPU/credential separation are not prerequisites; stronger deployment guarantees are scoped when needed.

The workflow plane composes available protocols around a research question. Sweep/search is usable independently; control and calibration extend the available workflows. The detailed invariants behind this diagram — object model, engine-selection table, run directory and exit codes, solver and integrator protocols, corrector checks — belong in [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md) §1–§3.

---

## 2. Build Blocks and Dependencies

No calendar. A block is done when its functional criteria and the applicable quality criteria in §4 hold. Dependencies below govern what must be available; numbering does not impose additional dependencies.

The core capability progression is **foundation → local sweep with checks → bounded agent-assisted exploration → control → device–control co-design**. Cluster execution, calibration, catalog, and interoperability are supporting capabilities with their own dependencies. This is the preferred sequence; technical prerequisites are listed separately. Sweep/search must meet its own quality gates independently of control.

### Decision 0 — Mandate ADR

- Name, command, and repo are settled: Entanglix, `entanglix`, this repository. Audience and license timing are agreed: private development by the owner and explicitly permitted collaborators; choose a distribution license before external distribution and retain the sharing gates.
- Actual system modeling includes component-defined circuits, coupled quantum subsystems, and explicit-operator models. This mandate is accepted in ADR §3.1; concrete schemas, supported representation details, and fixtures follow during Block 1. Named qubit examples do not define the capability's limits.
- Initial verification target agreed: the owner's current Linux workstation with an NVIDIA RTX 5090 (owner-confirmed), plus CPU-only foundation CI. Compatible version pins and GPU execution are verified during foundation and Block 2; hardware measurements do not gate scaffolding.
- Numerical policy agreed: user-facing energy/frequency parameters in GHz, time in ns, explicit angular-frequency conversion for dynamics, a uniform output grid with internal stepping allowed, and separate basis/retained-level cutoffs with model-specific convergence checks. Concrete fixture defaults follow the selected models.
- **Complete:** `docs/adr/0001-mandate.md` records these foundation choices and is Accepted; Block 0 is unblocked. The first sweep metric is chosen before Block 5; investigation policy, any requested limits, and evaluation design follow M2 measurements before Blocks 6a–6b.

### Block 0 — Repo Foundation

- **Depends on:** Decision 0.
- Layout: one installable project with `src/entanglix/` (library and the `entanglix` command), `docs/`, and `tests/`; root `pyproject.toml`, `uv.lock`, `.python-version`, and `.gitignore`. CLI help/version, CPU importability, and development checks establish the foundation. Schemas arrive in Block 1, GPU environment validation in Block 2, run environment identity in Block 3, and MCP/skills in Block 6b.
- CI: unit tests on CPU (no GPU required), a `gpu` marker for GPU tests, an **import-boundary test** asserting no module outside `tests/` imports a private plugin or a scheduler client.
- Foundation documentation: operational `README.md` and `HARNESS_CONTRACT.md`; repository `AGENTS.md` and `CLAUDE.md` are one-line pointers to internal `docs/development.md`; research conduct is a draft in `docs/research-agent.md` until Block 6b; planning index `docs/README.md`; decision records in `docs/adr/`. Operational docs state capability availability and work without the plans ([documentation boundary](README.md#documentation-boundary)).
- The root README covers setup, available commands, and an initial versioning and compatibility policy; CI defines the development checks.
- **Done when:** clone → `uv sync` → `pytest` green on a machine without a GPU.

### Block 1 — System Spec and Model Adapter

The [M1 guide](m1-guide.md) covers the joint implementation of Blocks 1–2. Its first execution step is a disposable GPU compatibility probe before model implementation: one tiny cuDensityMat solve and one dynamiqs solve on the RTX 5090, with actual device and known-result checks. Record the working Python/dependency stack; retain Python 3.13 unless an observed incompatibility requires changing it.

- **Depends on:** Block 0.
- Introduce `schemas/` with the `system.toml` JSON schema and pydantic model; scqubits adapter (circuit → truncated operators in a chosen basis); explicit-operator fallback; dissipators; drive operators; parameter values + priors; unit handling.
- CPU-testable exports to QuTiP `Qobj`, numerical arrays suitable for JAX, and per-subsystem operator terms with dimensions/order for the cuDensityMat adapter. GPU wrapping and storage selection belong to Block 2.
- `entanglix validate <system.toml>`.
- Pin concrete representation details and representative fixtures/parameters against the accepted modeling mandate during this block: fixed units, explicit array/output unit semantics, and one level of leaf system references. Publish implemented conventions and syntax in `docs/operations/system-spec.md`.
- **Done when:** the Block-1 fixture systems validate and export CPU objects/arrays/factorized terms with preserved units and bases; reference spectra/operators and model-refinement checks pass; invalid inputs produce documented diagnostics. Block 1 requires no GPU packages or device.

### Block 2 — Engines v0

- **Depends on:** Block 1.
- QuTiP CPU reference engine; cuDensityMat engine via `qutip-cuquantum`; dynamiqs engine; each behind the `Integrator` protocol with `capabilities()`.
- GPU wrapping and Dense/Dia storage selection inside the cuDensityMat adapter; reimplement the small rule from the unreleased scqubits GPU work and check its threshold of 16 on the RTX 5090. Reuse the rule, not unreleased code or a branch dependency.
- Cross-engine agreement tests at reference size (closed and open system, time-dependent drive), integration refinement, and a nonzero gradient checked against an analytic/finite-difference reference. Defer batching to Block 5 and advertise only implemented, tested capabilities.
- Verify the target GPU environment; pin compatible dependencies in `uv.lock` and record GPU, driver, CUDA, cuQuantum, Python, and engine-package observations in `docs/environment.md`. Run environment identity is introduced with the runner in Block 3.
- **Done when:** the engine-selection table (contract §2) is a versioned file, and required cross-engine, gradient, storage, and refinement checks pass, with GPU evidence from CI or a documented manual run. The checklist owns the detailed M1 acceptance requirements.

### Block 3 — Runner and Run Contract

The [M2 guide](m2-guide.md) covers the joint implementation of Blocks 3–5. [ADR-0003](adr/0003-runs-verification-and-dispatch.md) records the run-record, verification, dispatch, and frame decisions.

- **Depends on:** Blocks 0–2.
- `entanglix` gate + launch: spec validation, static import scan of any authored script, environment check (driver, cuQuantum), run-dir creation (`run.toml` first), job spawn in its own process group, `FINISHED` last, exit-code contract.
- Capture and retain a minimal run environment record and `env_hash` from the lockfile digest, Python/platform, and observed GPU driver/CUDA/cuQuantum versions. Execute jobs in an exactly synced environment through `uv run --locked --exact`, with the same declared extras for all jobs sharing it. Record code provenance separately, including the Git revision and dirty state plus frozen authored inputs and a patch of uncommitted code; a dirty flag or digest alone cannot reproduce changes.
- Local executor only in this block; distinct run directories and frozen inputs, a GPU lock with one active job by default, held by the job so runner loss cannot release it, runner-owned history writes, per-run timeouts, and cancellation of the intended process group. Reconcile unknown process state before reusing the GPU; no custom scheduler is required.
- `read_run` shows the live trace of running jobs so a stalled run can be aborted mid-way.
- Run history from the first execution through runner-owned records in each run directory, without a separate ledger store: provenance, parent runs, resource use, and outcome for completed, failed, aborted, and interrupted attempts. This history is available before catalog promotion or agent use.
- Diagnostics and a documented recovery or rerun path for interrupted work.
- This block accepts runner facilities through authored CPU jobs, real process/lock tests, and a GPU environment check on the workstation. Internal computation plans use them in Block 4; linked verification inspection belongs to Block 4, and built-in protocol records, reproduction, and scientific GPU dispatch integration belong to Block 5, with their acceptance evidence required there.
- **Done when:** runner loss leaves the attempt without `FINISHED`; observed job failures receive failed terminal records; a completed CPU job's `result.toml` validates; it reproduces from frozen inputs, recorded code, and matching `env_hash`; the GPU environment gate passes on the workstation; attempts and lineage remain queryable after interruption.

### Block 4 — Corrector v0

- **Depends on:** Blocks 1–3.
- Built-in checks as a separate package with hashed code and a versioned thresholds file: physicality of stored states, spectral truncation refinement, and shared sweep-check functions, with a deterministic, bounded evidence-point selection. Verification is a linked run whose report states each check's outcome and each point's coverage. Block 4 first shipped verdicts, a full-window reference comparison, and spectral cross-checks; Block 5 removes them and makes every check run on request ([ADR-0003 §2.3–§2.4](adr/0003-runs-verification-and-dispatch.md#23-verification-runs-and-checks-on-request)).
- Block 4 owns trusted computation definitions and shared check functions; Block 5 protocols supply metric definitions and sweep application.
- The corrector derives the checks relevant to a run from trusted computation definitions and the validated execution plan, not from labels in specs or results.
- Internal computation plans use the runner's existing gate, records, and lifecycle; linked verification runs integrate with history and `read_run` before this block closes.
- **Done when:** a requested check fails a deliberately under-converged run; a hash mismatch fails the verification run; labels cannot change which checks are relevant. Verification is available before completion of Block 5.

### Block 5 — Protocols `simulate` and `sweep` (first end-to-end)

- **Depends on:** Blocks 1–4.
- Standalone simulation and static and dynamic grid sweeps over scqubits parameters against declared observables or a simulated metric (for example dispersive shift, T1 limit from a noise model, gate error under a fixed pulse).
- The first end-to-end example exercises open-system dynamics on the local GPU; this class of metric is settled. The CPU feasibility probe settles the exact observable, drive, parameter region, and candidate evidence thresholds; these are recorded in the [readout example](../examples/readout/README.md). Independent-engine acceptance of the candidate frame bounds belongs to 5B. Spectral-only studies remain useful supported workflows.
- Lab-frame carrier drives and rotating frames with an explicit rotating-wave approximation; no built-in check measures the approximation, and a user compares against a lab-frame run. A declared `device = "cuda"` runs non-gradient work on cuDensityMat without a performance claim. The harness imposes no dimension cap: Block 5 removes the caps and the pre-launch memory estimates. An allocation failure is reported as out of memory, and an unexplained signal death as its signal.
- Verification on request, before the first sweep. A run computes and stores its results, and checks run only when the user asks: integrity of stored states, refined segments in the source run's frame as local evidence, diagnostics, and spectral refinement. Each check reports its own outcome, and there is no verdict. A check's prerequisites, such as a refinement declaration, never restrict a run. A launch, its dry-run preview, and `read_run` list the relevant checks. Segment tolerances replace declared rotating-wave bounds.
- Built-in protocols integrate the runner's gate, environment identity, GPU ownership, usage, and recovery facilities. Freeze protocol/plan inputs, record seeds and solver settings, and validate reproduction and interrupted-attempt history on CPU and GPU.
- Protocols supply trusted metric definitions and segment tolerances to the corrector; sweep checks apply bounded evidence selection and declared expectations to actual grids.
- Plots and `result.toml` summaries with enough evidence for a user to inspect results and choose a refined sweep.
- **Done when:** a documented simulation and one agreed dynamics sweep run end to end through `entanglix`, the sweep exercises the local GPU, the requested checks pass and report their scope, and a user can inspect and refine the sweep without control or calibration.

### Investigation decisions — after M2, before Blocks 6a–6b

Define the first investigation's question, scope, assumptions, evidence requirements, progress checkpoints, and stop/review criteria. Record any user-specified ceilings or deadlines; ordinary investigations need no mandatory job-count or time allowance. M2 measurements inform cost reporting and experiment sizing without requiring an estimate of total research cost.

[ADR-0004](adr/0004-investigations-and-agent-sessions.md) records what the contract leaves open (how user decisions are recorded, progress-assessment fields, checkpoint declaration, and resume behaviour) and the evaluation protocol; usage recording and limits are fixed by contract §4. Ordinary investigation specifications remain research records. ADR-0004 §2.8 records the agent-session decisions taken before Block 6b: instruction delivery, the decision path, job tools, the permission settings and what holds in a runtime without them, and the session audit.

The evaluation protocol names a scripted experiment-selection policy, comparable resource allowances, evidence thresholds, and measures of scientific outcome, elapsed time, compute use, and human interventions. Fixed allowances support this comparison without becoming mandatory for ordinary investigations. Include a case assessing a supported early stop after insufficient progress. A scripted client alone is not a scientific baseline. For optimization tasks, compare best verified objective or time to a feasible candidate; for exploratory tasks, use the declared evidence requirements.

### Block 6a — Investigation Execution

- **Depends on:** Blocks 3–5 and the decisions identified above as required before Block 6a.
- Runner-managed investigation records, immutable revisions, findings, progress assessments, and resumable histories.
- Human-started investigations with recorded scope and review points. Existing CLI/runtime user-input paths record scope changes, review releases, and optional-limit changes separately from agent proposals. Research tools cannot approve their own proposals; no separate grant issuer or credential service is required.
- Usage recorded from executor reports across protocol runs, nested work, verification, retries, and failures. Optional limits (run timeout; investigation deadline, job seconds, job count) use atomic admission accounting and executor enforcement, including queued jobs and absolute deadlines (contract §4). Absence of a limit is recorded, not treated as a zero budget or an estimate.
- Progress checkpoints, declared as a count of finished attempts (including failed and aborted runs) or an elapsed interval, gate further dispatch until an assessment links observed results, constraints, convergence, learning, cost, and a continue/refine/stop rationale. The runner validates records and review state; research judgment determines whether more computation is worthwhile.
- Reliable cancellation, pause/resume, and restart files where supported; otherwise rerun under a new linked ID. Preserve consumed resources and incomplete attempts across sessions.
- Scripted iterative sweep/search client, with Bayesian search where appropriate. Document its experiment-selection policy so it can serve as the M3 baseline.
- **Done when:** the scripted client reaches the declared evidence or a supported stop/review handoff; progress and accounting survive resume; no-limit and explicit-limit cases work; cancellation, review gating, and prevention of accidental run interference pass through supported interfaces.

### Block 6b — Agent Integration

- **Depends on:** Block 6a and the local runtime integration decision.
- Support expert–agent formulation and bounded execution once the next investigation is defined. Broad ambitions are refined before expensive work.
- Introduce `entanglix.mcp` behind an optional `mcp` dependency extra and add `skills/`; MCP over stdio exposing available CLI capabilities; tool-call audit and per-connection session records; initial skills (`system-modeler`, `simulate`, `design-sweep`, `gpu-budget`); refresh the research-agent draft and deliver its conduct and workflow entry points through the selected mechanism in the user's research workspace, independently of repository development instructions. The resource skill covers measured usage and optional limits. Control and calibration tools and skills arrive with their capabilities.
- One coordinating agent uses the local runner: separate run directories, frozen inputs, controlled history writes, one GPU job at a time by default, and cancellation of owned jobs. Human actions remain separate in the supported workflow. Where the runtime can restrict commands, its settings deny human-action commands for the agent, and an instruction-only session is recorded as such; the agent session sees no CUDA device. A launch on a busy GPU is refused before an attempt exists. Document the cooperative trust assumption.
- Verify research instructions reach the user's runtime in a research workspace without repository development guidance. Tests reject agent-supplied approval claims, out-of-scope tool requests, and changes to configured ceilings through research tools. The runtime is the user's own; each session's audit records the client name and version the runtime reports. OS-level escape resistance and device/credential isolation are deferred deployment work.
- **Done when:** the agent completes the declared sweep/search investigation through logged tools with evidence-driven refinement, recorded continuation decisions, and a supported final report or review handoff; the declared evidence/reporting criteria and independent usability pass. A stagnation case produces a supported early report with available results and next experiments. Execution protections and any configured limits hold through supported interfaces.
- **M3 evaluation:** compare against the Block 6a baseline, including scientific outcome, cost, human intervention, and the evidence supporting continuation/stopping decisions. Report worse performance or inconclusive benefit. M3 requires correct execution, declared evidence, local execution reliability, independent usability, and completion of the comparison; outperforming the baseline is not required. Claims of acceleration require measured comparative benefit.

### Block 7 — Protocol `control`

- **Builds on:** Blocks 1–4 for standalone control; Block 6b for agent integration. Prefer to sequence after M3; that order is not a technical prerequisite. Cluster execution and calibration are optional.
- A `control` spec from which the runner generates one ProblemSpec OSS file shared by every solver ([ADR-0005](adr/0005-control-problems-and-solvers.md)); `SolverBackend` protocol and entry-point discovery; open GRAPE-style baseline; Piccolo adapter that runs Piccolo's ProblemSpec runner in Julia.
- Control checks on request (contract §3): pulse replay on a different engine, the declared fidelity target, and the declared constraints. A second solve of the same ProblemSpec is a separate requested run.
- Standalone control on an existing system model through the CLI; extend MCP and add the `control` skill when Block 6b is available. Existing sweep/search workflows remain independently supported.
- **Done when:** one gate (Decision-0 platform) is solved by the open baseline and by Piccolo from the same ProblemSpec, each candidate passing the checks requested with their scope reported, both in run dirs; a private plugin, if present, is discovered and its candidate is checked by the same open replay.

### Block 8 — Device–Control Co-design

- **Depends on:** Blocks 5, 6a, and 7; Block 6b for agent-driven co-design.
- Compose device search and control: propose circuit parameters → construct the model → optimize admissible controls → evaluate performance with the checks requested → refine the device search.
- Declare device and control constraints, relevant parameter uncertainty, and a consistent evaluation and optimization-budget policy for comparing candidates; report any limits and differing conditions explicitly.
- Support expert checkpoints and bounded agent execution, with evidence linking device candidates to their control runs. Calibration can supply model updates once available.
- **Done when:** one documented investigation iterates jointly on circuit parameters and controls, reports a candidate comparison under the declared constraints and uncertainty, with the checks requested and their scope reported, and ends with a supported recommendation or an explicit account of limitations. Users can still choose the standalone sweep/search or control workflows.

### Block 9 — Cluster Executor

- **Depends on:** Blocks 3–5. Build when remote compute is needed; it does not gate local exploration or control.
- Slurm or Kubernetes executor behind the same `Executor` interface as local; artifact mirroring (small arrays back, large arrays referenced by path + digest); warming and lost-contact timeouts; abort. Use the scheduler's time limits, GPU allocation, cancellation, and accounting rather than Entanglix equivalents; retain lineage. Define any stronger access/isolation guarantees required by the deployment before claiming them.
- Development and CI stay on the workstation: one executor test suite runs against the local executor and a stubbed scheduler (fake submit/status/accounting commands). Acceptance adds one documented smoke run on the target cluster.
- **Done when:** `sweep` runs unchanged on the cluster and locally; the executor test suite passes locally against the stubbed scheduler; a lost job terminates as `aborted`, never as `completed`; usage comes from scheduler accounting and recovery diagnostics are recorded.

### Block 10 — Protocol `calibrate`

- **Depends on:** Blocks 1–4; Block 6b for agent access. Control and co-design are optional.
- One uniform measurement format with point-in-time provenance; laboratory data is converted to it mechanically, outside the harness.
- Spectroscopy first: extraction with declared settings; supplied labels, or one-to-one matching from a declared candidate family with an optional anchor; bounded weighted least squares with a declared search; holdout data separated before the fitting launch; identifiability report; **write-back to the device model is a proposal file a human applies**.
- Documented standalone calibration workflow; extend the MCP surface and add the `calibrate` skill when the agent harness is available.
- **Done when:** synthetic-data round trip recovers known parameters within reported uncertainty; fitting receives training data only and the corrector evaluates holdout data; a user can inspect a calibration report and device-update proposal independently of control.

### Block 11 — Results Catalog

- **Depends on:** Blocks 3–5; Block 7 for pulse warm-starts.
- Build on the run ledger: catalog of human-promoted results, provenance and lineage queries, trial counts per hypothesis family, and reusable candidates for subsequent investigations.
- Promotion is a human command. Failed and unpromoted run history remains queryable independently of the catalog.
- **Done when:** a subsequent sweep reuses a catalogued candidate with recorded lineage; promotion is auditable; once control is available, a control run warm-starts from a catalogued pulse.

### Block 12 — Interoperability

- **Depends on:** Blocks 3 and 7; shared calibration schemas extend conformance when Block 10 is available.
- Conformance: run dirs readable by amicode's Run Inspector, including `index`, `latest`, and the Inspector's required fields; ProblemSpec files accepted by `amico-run` unchanged; shared device / experiment-card schemas.
- Optional: package as an amicode Domain Pack if wanted by both sides.
- **Done when:** a `control` run dir opens in the Run Inspector; a ProblemSpec authored here solves under `amico-run` with no edits.

### Upgrade path (not blocks)

Multi-GPU cuDensityMat; cuTensorNet engine for lattice-scale systems; multistart and continuation for control; longer scheduled unattended campaigns after bounded workflows demonstrate reliability; hardware-in-the-loop via an external device bridge behind the same run contract.

A cuDensityMat usage study reviews open-system operator construction and solver usage, measures CPU/GPU crossover and peak memory with setup separated from repeated execution, and checks batch-versus-serial agreement member by member. It is a prerequisite for automatic performance-based dispatch and for advertising batching. A first observation to start from: at dimension 30 cuDensityMat was about 100 times slower than CPU QuTiP, and a dimension scan that was not retained put the crossover between 320 and 600. The engines use different integrators (QuTiP `vern9`, cuDensityMat `CuVern7`, dynamiqs `Tsit5`), so the cause is not isolated; fixed per-step overhead is a hypothesis.

Stronger authorization and isolation are conditional deployment work. Before a deployment requires containment of untrusted code, resistance to shell bypass, or stronger separation between users, define its threat model and reuse suitable runtime/OS controls. Grant issuers, isolated administrative credentials, restricted filesystem/network/device access, and adversarial escape tests belong to that scope; they do not gate M3. Parallel development or batching alone does not require that security model.

---

## 3. Harness Contract

[`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md) defines bounded investigations (§4), tools and permissions (§5), history/audit/promotion (§6), execution environments and trust boundaries (§7), and versioning/conformance (§8). Quality standards and open decisions remain in this roadmap (§4 and §7); the checklist tracks acceptance evidence. Document classes and lifecycles are in the [documentation boundary](README.md#documentation-boundary).

Block 6a adds investigation execution to the runner and ledger from Block 3 and corrector from Block 4; Block 6b adds cooperative agent access through MCP and evidence-based continuation/stopping. Blocks 7 and 10 extend agent access to control and calibration when both capabilities are present; Block 11 adds the promoted catalog.

---

## 4. Standing Disciplines (every block)

Each milestone's exit includes evidence for the capabilities it introduces:

| Quality dimension | Required acceptance evidence |
|---|---|
| Scientific | Representative reference cases; documented assumptions and applicable uncertainty, convergence, and verification reports; reproducible examples |
| Engineering | Appropriate automated tests and regression coverage; documented interfaces and dependencies; supported environment and compatibility expectations |
| Operational | Observable execution and resource use; actionable failures; interruption and recovery behavior tested where jobs are introduced; enforcement of explicitly configured limits for delegated work |
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
| M0 — Mandate + foundation | Decision 0, Block 0 | mandate ADR accepted; CPU tests green; setup and compatibility guidance available |
| M1 — Model + engines | Blocks 1–2 | spectra match scqubits; engines agree at reference size |
| M2 — Local sweep with checks | Blocks 3–5 | documented local GPU sweep, reproducible, with its requested checks passing; run history and interruption diagnostics available |
| M3 — Bounded agent-assisted exploration | Blocks 6a–6b | declared evidence, progress-based continuation/stopping, reliable local execution, configured-limit enforcement, independent usability, and baseline comparison pass Block 6b criteria; no performance advantage required |
| M4 — Control | Block 7 | standalone control workflow; open baseline and Piccolo solve the same gate problem, each candidate passing the checks requested; agent integration demonstrated when Block 6b is available |
| M5 — Device–control co-design | Block 8 | joint device/control investigation produces a candidate comparison, with the checks requested and their scope reported, and supported findings under declared constraints and uncertainty |

Supporting milestones are scheduled by their dependencies and research need:

| Milestone | Blocks | Exit |
|---|---|---|
| S1 — Cluster execution | Block 9 | same sweep on the scheduler; executor tests pass locally against a stubbed scheduler; scheduler accounting and failure handling demonstrated |
| S2 — Calibration | Block 10 | standalone synthetic round trip with uncertainty and holdout evaluation; inspectable model-update proposal |
| S3 — Results catalog | Block 11 | promoted candidate reused with lineage; pulse warm-start demonstrated when control is available |
| S4 — Interoperability | Block 12 | Run Inspector opens a run; `amico-run` accepts a ProblemSpec |

The Harmoniqs conversation is gated by concrete artifacts: the shared ProblemSpec solved by the open baseline and Piccolo, each candidate with its requested check reports, in inspectable run directories. These are normally delivered at M4, but standalone control may supply them before M3. Agent integration is not a sharing prerequisite; the audience/license decision and removal of Appendix A still apply.

---

## 6. Things Not to Build Yet

- Any UI or editor extension.
- Gate-model algorithm layers.
- A scheduler or job queue of our own.
- Hardware drivers.
- Unattended multi-day campaigns before bounded workflows and their operational quality are demonstrated.
- A Julia substrate.
- Co-design extensions, until a concrete workflow needs them: uncertainty evaluation beyond a single periodic circuit coordinate, flux correspondence, dressed-transition calibration, device-dependent relaxation, and lab-frame validation of the control model.
- Calibration extensions, until a concrete workflow needs them: dynamics-based calibration (time-domain traces, gate-error tables), gate-charge axes, several loops driven from one coil, profile likelihood and posterior sampling, and laboratory-format converters.

---

## 7. Open Decisions

Settled: project name Entanglix; command `entanglix`; runner in Python (`uv`), same substrate as the engines. Expert-guided problem formulation and bounded autonomous investigations; independently useful entry points; the core milestone progression in §5; supporting capabilities scheduled by dependency and need; enterprise-grade quality across every milestone. The initial agent harness supports sweep/search; control and calibration extend it as those capabilities become available. The first end-to-end sweep exercises open-system GPU dynamics; 5A has settled its coherence metric and feasibility choices in the readout example.

- **During Block 1:** concrete modeling schemas/APIs, supported representation details, and validation fixtures/parameters implementing the accepted mandate in ADR §3.1. Decision 0 is closed; the RTX 5090 target/environment policy, numerical conventions/convergence policy, and audience/license timing are accepted in ADR §3.2–§3.4.
- **Settled in 5A:** dynamics metric, drive, parameter region, windows, grids, refinement, and candidate frame bounds for the first GPU sweep ([readout example](../examples/readout/README.md)); independent-engine acceptance of the frame bounds remains in 5B.
- **Settled in [ADR-0004](adr/0004-investigations-and-agent-sessions.md):** user-decision recording, progress-assessment fields, a checkpoint the user declares with no default, resume behaviour, the evaluation protocol with its scripted selection policy and user-declared allowances, and the user's own agent runtime. The first investigation is recorded in the [readout example](../examples/readout/README.md#threshold-characterization). Usage recording and optional limits are fixed by contract §4; ordinary investigations do not require preallocated budgets.
- **Settled in [ADR-0004 §2.8](adr/0004-investigations-and-agent-sessions.md#28-agent-sessions):** instruction delivery, the operator CLI as the decision path, job tools with a wait and recoverable launches, the permission settings and what holds in a runtime without them, session audit per connection, and busy-device admission ([probe results](../tests/probes/m3_agent_session_results.json)).
- **Settled in [ADR-0005](adr/0005-control-problems-and-solvers.md):** the control spec and the generated ProblemSpec, where the solver is named, the shared problem subset, solver options and termination reports, the direct Piccolo adapter, and the control checks ([probe results](../tests/probes/m4_control_results.json)).
- **Settled in [ADR-0007](adr/0007-calibration-from-spectroscopy.md):** spectroscopy first, the uniform measurement format, extraction, assignment, the fit and its uncertainty report, holdout and the calibration checks, flux binding, device-update proposals, and calibration in investigations and tools ([probe results](../tests/probes/s2_calibration_results.json)).
- **Before Block 9:** scheduler target (Slurm or Kubernetes), remote environment, and site execution requirements when needed.
- **When stronger deployment guarantees are needed:** threat model, authorization/identity mechanism, process/filesystem/network/device controls, and evidence of bypass resistance. This is not a Decision-0 or M3 prerequisite.

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
