# Entanglix Build Checklist

**Status:** Implementation tracker for the pinned [`roadmap.md`](roadmap.md) and derived [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md). The roadmap defines scope, milestones, and dependencies; the contract defines enforceable behavior; this file tracks completion status and links to acceptance evidence. Progress narratives belong in conversation, not this checklist.

Status legend: `[ ]` open · `[-]` in progress / partial · `[x]` done. Dependencies below follow the roadmap. Supporting milestones can proceed when their own prerequisites are ready. A block is complete when its required functional and applicable quality criteria are checked; numbering imposes no additional dependencies. No calendar: exit criteria gate progress, not dates.

The mandate ADR is Accepted and Decision 0 is closed. Concrete modeling schemas and fixtures follow during Block 1; later choices are tracked with the blocks that need them. Development and testing run on the owner's workstation; research runs may also use a cluster through its scheduler, under the same cooperative trust model. Stronger authorization and isolation are conditional deployment work, not M3 prerequisites.

---

## Milestone Map

Core capability progression; users may enter, iterate within, and stop at any supported stage:

| Milestone | Blocks | Status |
|---|---|---|
| M0 — Mandate + foundation | Decision 0, Block 0 | **complete** |
| M1 — Model + engines | Blocks 1–2 | **complete** |
| M2 — Local sweep with checks | Blocks 3–5 | **complete** |
| M3 — Bounded agent-assisted exploration | Blocks 6a–6b | **complete** |
| M4 — Control | Block 7 | **complete** |
| M5 — Device–control co-design | Block 8 | open |

Supporting milestones, scheduled by dependency and research need:

| Milestone | Blocks | Status |
|---|---|---|
| S1 — Cluster execution | Block 9 | open |
| S2 — Calibration | Block 10 | open |
| S3 — Results catalog | Block 11 | open |
| S4 — Interoperability | Block 12 | open |

The Harmoniqs conversation requires the shared ProblemSpec and the checked open-solver artifacts specified in roadmap §5, normally delivered at M4. Agent integration is not a prerequisite. The audience/license decision applies, and `roadmap.md` Appendix A must be removed before sharing.

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
- [x] Core progression pinned: local sweep with checks → bounded agent-assisted exploration → control → co-design; supporting capabilities follow their own dependencies.
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

The [M1 guide](m1-guide.md) records the implementation approach. Supported conventions and pinned numerical limits live in the [operational reference](operations/system-spec.md); this checklist records completion and links to acceptance evidence.

### [x] M1 entry — GPU compatibility probe

- [x] Before step A/implementation: a disposable environment runs one tiny cuDensityMat solve through qutip-cuquantum and one dynamiqs solve on the actual RTX 5090; both match a known reference
- [x] Record commands, Python/package versions, driver/CUDA observations, and device evidence; retain Python 3.13 unless an observed blocker requires a pin change. This probe is not full engine acceptance

**Acceptance evidence:** [GPU environment](environment.md).

### [x] Block 1 — System Spec and Model Adapter

**Depends on:** Block 0.

- [x] [`system.toml` JSON schema](../schemas/system.schema.json) + [pydantic model](../src/entanglix/models.py) (circuit, truncation, drives, dissipators, parameters + priors, units)
- [x] scqubits adapter: circuit → truncated operators in a chosen basis; `validate` reports applied and dropped offsets, oriented flux loops, available operators and meanings, the effective transformation, and scqubits' default when a multi-node author must supply one ([circuit reference tests](../tests/test_circuit_models.py), [CLI tests](../tests/test_models.py), [syntax reference](operations/system-spec.md))
- [x] Explicit-operator fallback ([CPU model tests](../tests/test_models.py))
- [x] CPU-testable QuTiP `Qobj`, numerical arrays suitable for JAX, and per-subsystem operator terms with dimensions/order for the cuDensityMat adapter, including drive and dissipator factors; no GPU packages required ([composition tests](../tests/test_composition.py))
- [x] `entanglix validate <system.toml>` / `load_system` backing capability ([CPU model tests](../tests/test_models.py))
- [x] Concrete representation coverage and fixture systems/parameters selected during this block under ADR §3.1, exercising component, composition, and explicit-operator paths; fixed units, array/output unit semantics, and one level of leaf system references ([examples](../examples/), [operational syntax](operations/system-spec.md))
- [x] Pin numerical acceptance thresholds/rationale after independent CPU reference refinement; operator/spectral comparisons and separate basis/retained-level convergence evidence pass ([numerical policy](operations/system-spec.md#numerical-checks-for-the-bundled-models), [circuit tests](../tests/test_circuit_models.py), [composition tests](../tests/test_composition.py))
- [x] Malformed units, rates, dimensions, references, and unsupported forms produce actionable diagnostics; intentionally insufficient truncation demonstrates the refinement check ([model](../tests/test_models.py), [circuit](../tests/test_circuit_models.py), [composition](../tests/test_composition.py) tests)
- [x] Implemented syntax, tensor/basis order, 2π conversion, collapse normalization, units, and operator names documented in [system-spec.md](operations/system-spec.md); [worked examples](../examples/cpu_reference.py) and [README](../README.md) usage/compatibility describe available behavior
- [x] Applicable quality evidence recorded ([model tests](../tests/test_models.py), [circuit tests](../tests/test_circuit_models.py), [composition tests](../tests/test_composition.py), [system interface and numerical checks](operations/system-spec.md), [worked CPU example](../examples/cpu_reference.py), [usage and changes](../README.md))
- [x] **Done:** fixtures validate and export CPU objects/arrays/factorized terms with preserved units/bases; reference spectra/operators, model refinement, and input diagnostics pass without GPU dependencies

### [x] Block 2 — Engines v0

**Depends on:** Block 1.

- [x] `Integrator` protocol with `capabilities()` and shared evolution input/result ([interface](../src/entanglix/evolution.py), [CPU tests](../tests/test_engine_interface.py))
- [x] QuTiP CPU reference engine ([implementation](../src/entanglix/engines/qutip_cpu.py), [analytic tests](../tests/test_composition.py), [exact Liouvillian regression](../tests/test_engine_interface.py))
- [x] cuDensityMat engine via `qutip-cuquantum` ([adapter](../src/entanglix/engines/cudensitymat.py), [GPU tests](../tests/test_engine_cuquantum.py))
- [x] dynamiqs engine on CPU and GPU, with JAX results preserving drive-amplitude derivatives ([adapter](../src/entanglix/engines/dynamiqs_jax.py), [CPU/GPU tests](../tests/test_engine_dynamiqs.py))
- [x] Wrap CPU factorized terms and select Dense/Dia storage in the cuDensityMat adapter; diagonal, narrow-band, dense, and factorized composite states agree with a CPU reference on the RTX 5090 ([GPU tests](../tests/test_engine_cuquantum.py), [storage evidence](environment.md#compatibility-result)); performance thresholds remain separate
- [x] Cross-engine agreement for the required two-level, driven/lossy oscillator, and populated qubit–resonator fixtures, including loss-disabled comparisons and closed/open/time-dependent evolution; dissipative cases have `γT` around `0.1–1` ([fixture references](../tests/engine_cases.py), [acceptance matrix](../tests/test_engine_acceptance.py), [retained results](../tests/probes/m1_acceptance_results.json))
- [x] Nonzero evolution gradient agrees with an analytic/finite-difference reference; integration refinement, separate dynamics truncation refinement, and trace/Hermiticity/positivity checks meet the [pinned tolerances](operations/system-spec.md#numerical-checks-for-the-bundled-models) ([gradient tests](../tests/test_engine_dynamiqs.py), [refinement tests](../tests/test_engine_acceptance.py), [retained results](../tests/probes/m1_acceptance_results.json))
- [x] Required-GPU adapter tests identify the actual engines and RTX 5090 devices; skipped required-GPU tests fail ([test hook](../tests/conftest.py), [GPU tests](../tests/test_engine_cuquantum.py), [GPU/CPU dynamiqs tests](../tests/test_engine_dynamiqs.py))
- [x] Verified optional engine pins in [`uv.lock`](../uv.lock) and [environment observations](environment.md): GPU, driver, CUDA, cuQuantum, Python, and engine packages; hashing follows with the runner
- [x] Versioned engine-selection table (contract §2); no harness-imposed dimension cap or pre-launch memory estimate; non-gradient requests default to CPU, with explicit compatible fallbacks recorded; advertise only implemented/tested capabilities (`batched = false` until sweep batching is validated) ([table](../src/entanglix/engines/selection.toml), [selection tests](../tests/test_engine_selection.py), [selection policy](operations/system-spec.md#engine-selection-policy))
- [x] Applicable quality evidence recorded ([acceptance results and commands](../tests/probes/m1_acceptance_results.json), [environment](environment.md), [interface and numerical reference](operations/system-spec.md), [loss example](../examples/dynamics_reference.py), [gradient example](../examples/differentiable_reference.py), [usage and changes](../README.md), [CPU CI](../.github/workflows/ci.yml))
- [x] **Done:** required agreement, gradient, storage, and refinement checks pass with retained commands/results and actual GPU evidence from CI or a documented manual run; all M1 acceptance items above are satisfied

---

## M2 — Local Sweep with Checks (Blocks 3–5)

The [M2 guide](m2-guide.md) records the implementation approach; [ADR-0003](adr/0003-runs-verification-and-dispatch.md) records the run-record, verification, dispatch, and frame decisions. This checklist records completion and links to acceptance evidence.

### [x] Block 3 — Runner and Run Contract

**Depends on:** Blocks 0–2. Acceptance covers the runner facilities and authored CPU jobs. Computation-plan integration belongs to 4A, linked verification to 4A–4B, and built-in protocol/GPU dispatch integration to 5C–5D; their acceptance checks remain below.

- [x] Gate facilities: system validation, static import scan, and selected-device environment check (driver access and loaded CUDA/cuDensityMat) ([launch tests](../tests/test_runner.py), [import policy](../src/entanglix/imports.py), [GPU gate tests](../tests/test_gpu_environment.py), [workstation evidence](environment.md#runner-gpu-environment-check))
- [x] Run dir per contract §2: `run.toml` first, `runner/started.toml` before computation, frozen `spec/`, `FINISHED` last, atomic `result.toml`, and assigned job outputs ([record API](../src/entanglix/runner.py), [record tests](../tests/test_run_records.py), [launch tests](../tests/test_runner.py), [run reference](operations/runs.md))
- [x] Minimal retained environment record and `env_hash`: lockfile digest, Python/platform, observed GPU driver/CUDA/cuDensityMat versions, and declared extras; exactly synced execution with the same extras for jobs sharing an environment ([environment identity](operations/runs.md#environment-and-code-provenance), [GPU gate tests](../tests/test_gpu_environment.py), [workstation evidence](environment.md#runner-gpu-environment-check))
- [x] Frozen authored code/data with digests; Git revision and dirty state, patch of uncommitted code and retained untracked sources; reconstruction and replay ([frozen provenance](operations/runs.md#environment-and-code-provenance), [reconstruction tests](../tests/test_run_records.py), [replay tests](../tests/test_runner.py), [lifecycle tests](../tests/test_lifecycle.py))
- [x] Job spawn in its own process group; exit-code contract; configured timeouts ([launch tests](../tests/test_runner.py), [lifecycle tests](../tests/test_lifecycle.py), [limit reference](operations/runs.md#limits-interruption-and-recovery))
- [x] Separate run directories and assigned outputs; an inherited GPU lock excludes across runner instances and runs roots ([record tests](../tests/test_run_records.py), [launch tests](../tests/test_runner.py), [lifecycle tests](../tests/test_lifecycle.py), [GPU ownership](operations/runs.md#local-gpu-ownership))
- [x] Cancellation targets the intended job and descendants; unknown process state blocks GPU reuse; inherited ownership survives runner loss; reconciliation records uncertain usage and never writes `FINISHED` ([CPU process and lock tests](../tests/test_lifecycle.py), [ownership tests](../tests/test_run_records.py), [recovery reference](operations/runs.md#limits-interruption-and-recovery))
- [x] Local executor, diagnostics, and documented interruption recovery/rerun path ([run reference](operations/runs.md), [launch tests](../tests/test_runner.py), [lifecycle tests](../tests/test_lifecycle.py))
- [x] Runner-owned attempt history retains parent runs, provenance, outcomes, and observed or uncertain resource use ([record tests](../tests/test_run_records.py), [launch tests](../tests/test_runner.py), [lifecycle tests](../tests/test_lifecycle.py))
- [x] Per-run records retain concurrent attempts; frozen-input reruns receive new IDs and preserve parent lineage ([record tests](../tests/test_run_records.py), [replay tests](../tests/test_runner.py), [lifecycle tests](../tests/test_lifecycle.py))
- [x] `list_capabilities`, `list_runs`, `read_run`, and `abort_run` backing capabilities ([launch tests](../tests/test_runner.py), [lifecycle tests](../tests/test_lifecycle.py), [run reference](operations/runs.md#authored-cpu-jobs))
- [x] Execution outcome remains distinct from scientific verification; authored result fields, including a `checks` table, are data and create no trusted check outcomes ([result tests](../tests/test_runner.py), [authored results](operations/runs.md#authored-cpu-jobs), [execution outcomes](operations/runs.md#execution-outcomes))
- [x] Applicable quality evidence ([reference example](../examples/recorded_decay.py), [run interface and recovery](operations/runs.md), [automated runner tests](../tests/test_runner.py), [lifecycle tests](../tests/test_lifecycle.py), [GPU environment evidence](environment.md#runner-gpu-environment-check), [release notes](../README.md#changes-and-compatibility))
- [x] **Done:** runner loss leaves no fabricated completion; observed job failure has a terminal record; completed CPU-script results validate and reproduce; interrupted attempts retain diagnostics and lineage ([record tests](../tests/test_run_records.py), [launch tests](../tests/test_runner.py), [lifecycle tests](../tests/test_lifecycle.py))

### [x] Block 4 — Corrector v0

**Depends on:** Blocks 1–3. This block owns trusted computation definitions and shared check functions; protocol metrics and actual sweep integration belong to 5C–5D.

- [x] Internal computation plans use the runner gate, frozen records, lifecycle, and provenance through simulations ([plan and simulation tests](../tests/test_plans.py), [retained computation records](operations/runs.md#computation-records))
- [x] Separate corrector package; code hashed at load; versioned thresholds outside the candidate workspace; research tools cannot replace policy or check reports ([policy implementation](../src/entanglix/corrector/__init__.py), [policy tests](../tests/test_corrector.py), [verification execution tests](../tests/test_verification.py))
- [x] Requested dynamics integrity and same-frame CPU refined segments with local coverage; deterministic bounded sweep evidence selection; dimension-48 exact regression retained as engine evidence ([policy tests](../tests/test_corrector.py), [checks and coverage](../tests/test_verification.py), [scientific regressions](../tests/test_verification_cost.py), [semantics](operations/protocols.md#requested-checks-and-segment-tolerances))
- [x] Linked requested checks bound to source digests, frozen requests and policy version; inspection lists history and per-check outcomes without an overall verdict ([binding tests](../tests/test_plans.py), [inspection and relocation](../tests/test_verification.py), [records](operations/runs.md#corrector-policy-and-verification-records))
- [x] Spectral refinement of reported excitation energies runs on request ([implementation](../src/entanglix/corrector/verification.py), [spectral tests](../tests/test_verification.py))
- [x] Corrector lists checks from trusted computation definitions and validated plans; a requested check with missing prerequisites is inapplicable with a reason ([definitions](../tests/test_corrector.py), [plan binding](../tests/test_plans.py), [listing tests](../tests/test_verification.py))
- [x] Shared series function checks a declared monotonicity expectation without removing other checks ([implementation](../src/entanglix/corrector/checks.py), [expectation tests](../tests/test_verification.py)); application to actual sweeps belongs to 5D
- [x] Reports identify check scope, tolerances, residuals, reference engine and dimension, each point's coverage, and inapplicable checks with reasons (contract §3) ([verification tests](../tests/test_verification.py), [report semantics](operations/runs.md#corrector-policy-and-verification-records))
- [x] `check` lists or executes named checks through one linked verification run ([implementation](../src/entanglix/corrector/verification.py), [CLI and execution tests](../tests/test_verification.py))
- [x] Applicable quality evidence recorded ([requested checks](operations/runs.md#corrector-policy-and-verification-records), [scientific and lifecycle tests](../tests/test_verification.py), [CPU reference regression](../tests/test_corrector.py), [CPU CI](../.github/workflows/ci.yml))
- [x] **Done:** deliberately insufficient refinement reports a failed check; a hash mismatch fails execution; check prerequisites do not block computation ([acceptance tests](../tests/test_verification.py), [scientific cases](../tests/test_verification_cost.py))

### [x] Block 5 — Protocols `simulate` and `sweep`

**Depends on:** Blocks 1–4.

- [x] Standalone simulation and static/dynamic serial grid sweeps over declared system parameters and protocol quantities ([protocol reference](operations/protocols.md), [simulation tests](../tests/test_simulate.py), [sweep tests](../tests/test_sweep.py), [static companion](../examples/readout/dispersive.toml))
- [x] CPU feasibility probe settles the dynamics metric, parameter region, windows, grids, frequency policy, refined dimensions, and candidate frame bounds (5A) ([probe](../tests/probes/m2_feasibility.py), [measurements](../tests/probes/m2_feasibility_results.json), [choices and limits](../examples/readout/README.md))
- [x] Lab-frame carrier drives and rotating frames with an explicit rotating-wave approximation: carrier drives pass cross-engine and refinement acceptance, rotating-frame results agree with the lab frame within declared metric tolerances, including endpoint resolution limits, and the lab-frame 5 × 6 readout/Purcell fixture passes a full-window independent CPU dynamiqs comparison at tightened tolerances and 6 × 8 levels (5B) ([interface and numerical limits](operations/system-spec.md#carriers-and-rotating-frames), [CPU/CUDA tests](../tests/test_carriers_frames.py), [acceptance evidence](environment.md#carrier-and-frame-acceptance), [independent-engine evidence and limits](../examples/readout/README.md#independent-engine-frame-acceptance), [measurements](../tests/probes/m2_frames_results.json))
- [x] `simulate` routes a declared `device = "cuda"` to cuDensityMat through a versioned selection table, without a performance claim, and connects the environment check, GPU and attempt locks, usage accounting, cancellation, and reconciliation; completed and interrupted GPU runs pass runner integration (5C) ([table](../src/entanglix/engines/selection.toml), [selection tests](../tests/test_engine_selection.py), [simulation tests](../tests/test_simulate.py), [GPU acceptance](../tests/probes/m2_simulation_gpu.xml), [operating rules](operations/protocols.md#gpu-configuration-and-checks))
- [x] `simulate` and `sweep` freeze protocol/plan specs and data, record solver settings, retain code/environment provenance, and reproduce from retained inputs under new linked IDs; all attempt states remain queryable (5C–5D) ([simulation replay tests](../tests/test_simulate.py), [CPU acceptance](../tests/probes/m2_simulation_cpu.json), [CUDA acceptance](../tests/probes/m2_simulation_cuda.json), [replay rules](operations/protocols.md#replay), [partial results, runner loss and full-grid replay tests](../tests/test_sweep.py), [CPU sweep acceptance](../tests/probes/m2_sweep_cpu.json), [CUDA sweep acceptance](../tests/probes/m2_sweep_cuda.json))
- [x] `simulate`/`sweep` supply trusted metric definitions with segment tolerances and recorded effective values; sweep checks apply bounded selection and declared expectations to actual grids (5C–5D) ([simulation formulas](../src/entanglix/corrector/metrics.py), [metric checks](../tests/test_simulate.py), [tolerance rules](operations/protocols.md#requested-checks-and-segment-tolerances), [grid coverage, declared expectations, corner-bound and warning tests](../tests/test_sweep.py), [sweep check semantics](operations/protocols.md#serial-grid-sweeps))
- [x] Verification feasibility probe measures the candidate checks against retained full-window references, including deliberate failures and false alarms (5V) ([probe](../tests/probes/m2_verification.py), [measurements](../tests/probes/m2_verification_results.json))
- [x] A run computes and stores its results and no check follows; `check` lists the relevant checks or runs named ones as one linked verification run with per-check outcomes and no verdict; prerequisites never restrict a run, and a check request may replace the refinement declaration (5V) ([request, listing and replacement tests](../tests/test_verification.py), [launch and inspection test](../tests/test_simulate.py), [admission binding tests](../tests/test_plans.py), [check records](operations/runs.md#corrector-policy-and-verification-records))
- [x] Integrity, same-frame refined segments, diagnostics, and spectral refinement behave as the check reference defines; the probe's segment-detectable failures are caught and its false alarms are not (5V) ([check reference](operations/protocols.md#requested-checks-and-segment-tolerances), [check tests](../tests/test_verification.py), [same-engine regressions](../tests/test_verification_cost.py), [CPU dynamiqs record](../tests/probes/m2_checks_cpu.xml))
- [x] No harness-imposed dimension cap or pre-launch memory estimate: a dimension-48 `simulate` run completes with its requested checks, and the engines agree at dimension 64; neither size sets a ceiling (5V) ([selection tests](../tests/test_engine_selection.py), [dimension-48 run in the CPU summary](../tests/probes/m2_simulation_cpu.json), [agreement test](../tests/test_engine_acceptance.py), [dimension-48 regression](../tests/test_corrector.py), [CPU dynamiqs record](../tests/probes/m2_checks_cpu.xml), [GPU dynamiqs record](../tests/probes/m2_checks_dynamiqs.xml), [cuDensityMat record](../tests/probes/m2_simulation_gpu.xml))
- [x] An observed allocation failure is reported as out of memory, and a signal death the runner did not request as that signal (5V) ([allocation and signal tests](../tests/test_memory_failures.py), [CLI test](../tests/test_cli.py), [outcome rules](operations/runs.md#execution-outcomes))
- [x] The simulation acceptance re-runs on CPU and CUDA: each run completes with no verification run, and the requested integrity and segments pass (5V) ([CPU summary](../tests/probes/m2_simulation_cpu.json), [CUDA summary](../tests/probes/m2_simulation_cuda.json), [commands](environment.md#simulation-acceptance))
- [x] Floquet quasienergy example through an authored script, claiming quasienergies under periodic driving only ([script](../examples/readout/floquet.py), [CPU runner test](../tests/test_floquet_example.py), [results and limits](../examples/readout/README.md#closed-system-floquet-quasienergies))
- [x] First end-to-end sweep exercises open-system dynamics on the local GPU ([readout grid](../examples/readout/dephasing.toml), [CUDA source/replay and requested checks](../tests/probes/m2_sweep_cuda.json), [GPU integration](../tests/probes/m2_sweep_gpu.xml))
- [x] Spectral-only studies remain usable independently ([CPU spectral simulation and replay test](../tests/test_simulate.py))
- [x] Plots and `result.toml` summaries expose metric grids and extrema for inspection and refinement; completed partial points remain queryable ([worker](../src/entanglix/protocols/_sweep_job.py), [run records](operations/runs.md#computation-records), [partial-result tests](../tests/test_sweep.py))
- [x] Documented CLI examples for simulation and iterative sweep refinement ([readout commands and narrower-grid example](../examples/readout/README.md#serial-readout-and-static-sweeps), [linked narrower grid in CPU acceptance](../tests/probes/m2_sweep_cpu.json))
- [x] Applicable quality evidence recorded ([protocol reference](operations/protocols.md), [CPU CI with dynamiqs sweep coverage](../.github/workflows/ci.yml), [CPU/CUDA simulation acceptance](environment.md#simulation-acceptance), [sweep acceptance and commands](environment.md#sweep-acceptance), [CPU lifecycle and scientific tests](../tests/test_sweep.py), [GPU sweep report](../tests/probes/m2_sweep_gpu.xml))
- [x] **Done:** a documented simulation and agreed dynamics sweep run through `entanglix`, GPU execution passes, the requested checks pass and report their scope, and a user can inspect/refine the sweep without control or calibration ([simulation acceptance](environment.md#simulation-acceptance), [sweep acceptance](environment.md#sweep-acceptance), [example](../examples/readout/README.md#serial-readout-and-static-sweeps))

---

## M3 — Bounded Agent-assisted Exploration (Blocks 6a–6b)

### [x] Investigation decisions — after M2

- [x] Before Block 6a: first investigation question, scope, assumptions, evidence requirements, progress checkpoints, and stop/review criteria; any requested ceilings informed by measured M2 costs ([readout example](../examples/readout/README.md#threshold-characterization), [probe](../tests/probes/m3_readout.py), [results](../tests/probes/m3_readout_results.json))
- [x] Before Block 6a: what the contract leaves open (user-decision recording, progress-assessment fields, checkpoint declaration, resume behaviour) in a follow-up ADR ([ADR-0004](adr/0004-investigations-and-agent-sessions.md))
- [x] Before Block 6a: evaluation protocol defines the scripted selection policy, comparable benchmark allowances, evidence thresholds, and outcome/time/compute/human-intervention measures; include supported early stopping after insufficient progress ([ADR-0004 §2.6](adr/0004-investigations-and-agent-sessions.md#26-evaluation-protocol), [readout example](../examples/readout/README.md#threshold-characterization))
- [x] Before Block 6b: agent-session decisions on instruction delivery, the decision path, job tools, runtime permission settings and what holds in a runtime without them, session audit, and busy-device admission; no separate grant issuer or OS security boundary required ([ADR-0004 §2.8](adr/0004-investigations-and-agent-sessions.md#28-agent-sessions), [probe results](../tests/probes/m3_agent_session_results.json))

### [x] Block 6a — Investigation Execution

**Depends on:** Blocks 3–5 and the decisions identified above as required before Block 6a.

- [x] Investigation records with immutable revisions: question, scope, assumptions, permitted changes/protocols, evidence requirements, progress checkpoints, review/stop criteria, and optional limits (contract §4; [reference](operations/investigations.md), [implementation](../src/entanglix/investigations.py), [tests](../tests/test_investigations.py); `uv run --locked --exact pytest -q tests/test_investigations.py tests/test_cli.py`)
- [x] Human-started investigations; user-input events for broader scope changes, review releases, and configured-limit changes are distinct from agent proposals ([start and approval evidence](../tests/test_investigations.py), [review releases and resumes](../tests/test_investigation_lifecycle.py))
- [x] `create_investigation`, `revise_investigation`, `list_investigations`, `read_investigation`, `record_finding`, and `stop_investigation`; drafts cannot self-approve or start through research-tool fields ([record-operation evidence](../tests/test_investigations.py), [stop and operator lifecycle evidence](../tests/test_investigation_lifecycle.py))
- [x] Findings and progress assessments link investigation revisions, evidence, verification status, constraints/convergence, observed cost, learning/uncertainty, and continue/refine/stop rationale ([finding authority and run membership](../tests/test_investigations.py), [linked check and usage summaries](../tests/test_investigation_admission.py), [assessments clearing checkpoints](../tests/test_investigation_lifecycle.py))
- [x] Usage recorded from executor reports across protocol runs, nested work, verification, retries, and failures ([linked work and findings](../tests/test_investigation_admission.py), [usage rules](operations/investigations.md#usage-and-optional-limits))
- [x] Optional limits per contract §4 (run timeout; investigation deadline, job seconds, job count); absence is explicit; unknown limit names are rejected ([limit validation](../tests/test_investigations.py), [admission tests](../tests/test_investigation_admission.py), [workstation GPU evidence](../tests/probes/m3_admission_gpu.xml))
- [x] Admission checks and commitment records are atomic; proposed and admitted-but-unsettled jobs, including queued/running jobs, count at their maximum when job seconds are limited; uncertain usage never returns allowance ([concurrent admission and reconciliation](../tests/test_investigation_admission.py), [admission reference](operations/investigations.md#usage-and-optional-limits))
- [x] Executor enforces absolute investigation deadlines across queueing and execution; timeout resolution and termination grace documented; actual usage and overruns recorded ([delayed-start and execution tests](../tests/test_investigation_admission.py), [executor rules](operations/investigations.md#usage-and-optional-limits))
- [x] Progress checkpoints (finished-attempt count, including failed/aborted runs, or elapsed interval) gate dispatch until an assessment is recorded ([checkpoint and linked-verification tests](../tests/test_investigation_lifecycle.py), [reference](operations/investigations.md#checkpoints-and-reviews))
- [x] Stop/review state gates dispatch; cancellation and resume preserve accounting, outstanding reviews, and stopping reasons; simulate/sweep have no restart files ([delayed-start, process-group and session tests](../tests/test_investigation_lifecycle.py), [reference](operations/investigations.md#stop-and-resume))
- [x] Early-stop reports preserve best available results, limitations, and next experiments; remaining allowance does not require continued execution ([early-stop acceptance](../tests/probes/m3_investigation_early_stop.json), [claim and handoff policy](../examples/readout/README.md#scripted-investigation-baseline))
- [x] User-decision events record source, affected revision, starts/resumes, scope/limit changes, and named review releases ([authority tests](../tests/test_investigations.py), [operator lifecycle tests](../tests/test_investigation_lifecycle.py))
- [x] Scripted iterative sweep/search client, with Bayesian search where appropriate; documented selection policy qualifies it as the M3 baseline ([client](../examples/readout/investigate.py), [midpoint policy](../examples/readout/README.md#scripted-investigation-baseline), [normal acceptance](../tests/probes/m3_investigation_normal.json), [early-stop acceptance](../tests/probes/m3_investigation_early_stop.json))
- [x] Tests cover no-limit and configured-limit cases, atomic admission with proposed/queued/running commitments, deadlines across queueing and execution, timeout resolution/grace and overruns, rejected approval claims, review and checkpoint gating including failed/aborted attempts, nested/retry accounting, cancellation/resume, and accidental run interference ([record tests](../tests/test_investigations.py), [admission tests](../tests/test_investigation_admission.py), [lifecycle tests](../tests/test_investigation_lifecycle.py), [client interruption and retry tests](../tests/test_readout_investigation.py), [GPU accounting](../tests/probes/m3_admission_gpu.xml))
- [x] Applicable quality evidence recorded ([worked scientific example and limits](../examples/readout/README.md#scripted-investigation-baseline), [operational reference](operations/investigations.md), [normal baseline](../tests/probes/m3_investigation_normal.json), [early-stop baseline](../tests/probes/m3_investigation_early_stop.json), [client tests](../tests/test_readout_investigation.py))
- [x] **Done:** scripted investigation reaches declared evidence or a supported stop/review handoff; progress/history survive resume; optional limits, review gates, cancellation, and local execution protections work through supported interfaces ([baseline evidence and policy](../examples/readout/README.md#scripted-investigation-baseline), [session recovery](../tests/test_readout_investigation.py))

### [x] Block 6b — Agent Integration

**Depends on:** Block 6a and the local runtime integration decision.

- [x] Formulation workflow refines broad ambitions into bounded next investigations; partial workflows and exploratory questions are supported ([formulation transcript identity and initial-context evidence](../tests/probes/m3_evaluation_results.json), [bounded formulation and comparison](../examples/readout/README.md#agent-investigation-comparison))
- [x] Agent assesses evidence, constraints, convergence, observed cost, and learning before continuing/refining/stopping; roles of numerical search and agent reasoning documented ([assessment and stopping records](../tests/probes/m3_evaluation_results.json), [method and limits](../examples/readout/README.md#agent-investigation-comparison))
- [x] A launch on a busy GPU is refused before an attempt exists and uses no investigation allowance; ownership spans the CUDA launch check and execution, and launch failures release it ([admission and failure tests](../tests/test_gpu_admission.py), [workstation lifecycle report](../tests/probes/m3_busy_gpu.xml), [ownership reference](operations/runs.md#local-gpu-ownership))
- [x] `entanglix.mcp` behind an optional `mcp` extra, introduced with `skills/`; MCP over stdio uses CLI backing operations, advertises available capabilities, and distinguishes research requests from runtime user-input events (contract §5; [transport tests](../tests/test_mcp.py), [guidance initialization tests](../tests/test_research_guidance.py), [operations reference](operations/mcp.md))
- [x] Tool-call audit log, with client-reported request metadata, and per-connection session records with unknown fields labelled (contract §6), including scope revisions and linked run IDs ([transport and audit tests](../tests/test_mcp.py), [audit reference](operations/mcp.md#audit-records))
- [x] A launch whose connection or response is lost is recovered as pending, admitted, or refused, without a second attempt ([shared-operation tests](../tests/test_detached_launches.py), [stdio recovery tests](../tests/test_mcp.py), [workstation MCP supervision report](../tests/probes/m3_mcp_gpu.xml))
- [x] Initial skills: `system-modeler`, `simulate`, `design-sweep`, `gpu-budget`; versioned metadata; resource guidance covers measured usage and optional limits; updated research-agent conduct and workflow entry points delivered through the selected mechanism ([skills](../skills/), [conduct](research-agent.md), [initialization and audit tests](../tests/test_research_guidance.py), [initial-context and plain-procedure evidence](../tests/probes/m3_guidance_results.json))
- [x] Instruction-delivery check in a research workspace: intended research guidance reaches the runtime without repository development instructions ([initial-context evidence](../tests/probes/m3_guidance_results.json), [workspace setup](operations/mcp.md#research-workspace))
- [x] One coordinating agent uses separate run directories, frozen inputs, runner-owned history, one active GPU job by default, and cancellation of owned jobs ([retained agent attempts and pickup](../tests/probes/m3_pickup_results.json), [frozen-input replay and cancellation tests](../tests/test_lifecycle.py), [workstation ownership and cancellation evidence](../tests/probes/m3_mcp_gpu.xml))
- [x] The runtime's configured restriction refuses human-action commands for each runtime and command form where it is claimed, and an instruction-only session is recorded as such; the agent session sees no CUDA device; the cooperative trust assumption is documented; no claim of resistance to arbitrary shell/code bypass ([configured forms](operations/mcp.md#codex-example), [restriction-mode audit tests](../tests/test_research_guidance.py), [live runtime conformance](../tests/probes/m3_guidance_results.json))
- [x] Tests reject agent-authored approval claims, out-of-scope tool requests, and configured-limit changes through research tools; each session's audit records the client name and version the runtime reports ([scripted MCP client tests](../tests/test_mcp.py))
- [x] Demonstrate evidence-driven refinement and a supported early-stop report in an insufficient-progress case; preserve available results and suggested next experiments ([source-bound claims, retained reports and precision reviews](../tests/probes/m3_evaluation_results.json), [comparison and limitations](../examples/readout/README.md#agent-investigation-comparison))
- [x] Execute the declared baseline comparison; report scientific outcome, elapsed time, compute use, human interventions, and continuation/stopping rationale, including worse or inconclusive performance ([comparison, including the pre-compute runtime failure](../tests/probes/m3_evaluation_results.json), [evaluation driver](../tests/probes/m3_evaluation.py), [summary regression tests](../tests/test_agent_evaluation.py))
- [x] Research-tool responses are bounded: an investigation is read as a status whose size does not depend on history, a paged attempt table, and an event index; a report is generated from the records; each result is sent once ([stdio views, acknowledgment and audit tests](../tests/test_mcp.py), [paging and report tests](../tests/test_mcp_views.py), [response sizes](environment.md#mcp-response-summaries), [interface and limits](operations/mcp.md#investigation-views-and-tables))
- [x] Applicable quality evidence recorded for the complete independent sweep/search workflow ([comparison](../tests/probes/m3_evaluation_results.json), [pick-up evidence](../tests/probes/m3_pickup_results.json), [Claude Code and Codex pick-up after a cut-off job](../tests/probes/m3_pickup_cutoff_results.json), [summary tests](../tests/test_agent_evaluation.py), [guidance tests](../tests/test_research_guidance.py), [worked example and limitations](../examples/readout/README.md#agent-investigation-comparison))
- [x] **Done:** logged agent investigation meets declared evidence/reporting criteria and independent usability; refinement and justified stopping are demonstrated; local execution protections and any configured limits hold; baseline comparison complete ([comparison and acceptance](../tests/probes/m3_evaluation_results.json), [pick-up evidence](../tests/probes/m3_pickup_results.json), [usability walk-through](environment.md#investigation-usability-walk-through), [refusal and limit tests](../tests/test_mcp.py), [baseline comparison](../examples/readout/README.md#agent-investigation-comparison))
- M3 does not require outperforming the baseline. Claims of acceleration require measured comparative benefit.

---

## M4 — Control (Block 7)

The [M4 guide](m4-guide.md) records the implementation approach; [ADR-0005](adr/0005-control-problems-and-solvers.md) records the control spec, solver, adapter, and check decisions. This checklist records completion and links to acceptance evidence.

### [x] Block 7 — Protocol `control`

**Builds on:** Blocks 1–4 for standalone control; Block 6b for agent integration. Prefer to sequence after M3; that order is not a technical prerequisite. Cluster execution and calibration are optional.

- [x] Control decisions recorded in [ADR-0005](adr/0005-control-problems-and-solvers.md): the control spec and generated ProblemSpec, the shared problem subset, solvers, the Piccolo adapter, and control checks ([probe](../tests/probes/m4_control.py), [results](../tests/probes/m4_control_results.json))
- [x] `control` spec and one generated ProblemSpec per run, with the system exported in the control frame ([validation and export](../src/entanglix/protocols/control.py), [schema, quadrature, pulse, freezing, and invalid-input tests](../tests/test_control.py), [control reference](operations/control.md), [first-gate input and bounds](../examples/transmon-control/README.md))
- [x] `SolverBackend` protocol and entry-point discovery ([interface and discovery](../src/entanglix/solvers/__init__.py), [discovery tests](../tests/test_control_solver.py))
- [x] Open GRAPE-style baseline ([solver](../src/entanglix/solvers/grape.py), [gradient, constraint, and short-solve tests](../tests/test_control_solver.py), [first-gate candidate and measured cost](../tests/probes/m4_baseline_results.json), [pulse convention](operations/control.md#neutral-pulse-and-candidate-record))
- [x] Piccolo adapter running Piccolo's ProblemSpec runner in Julia, with a pinned Julia project ([adapter](../src/entanglix/solvers/piccolo.py), [project](../src/entanglix/solvers/julia/Project.toml), [Julia entry](../src/entanglix/solvers/julia/EntanglixPiccolo/src/EntanglixPiccolo.jl), [integer-tolerance, complex-goal, active-bound, and stop tests](../tests/test_control_piccolo.py), [workstation report](../tests/probes/m4_piccolo.xml), [same-problem first-gate candidate](../tests/probes/m4_piccolo_results.json), [setup and commands](environment.md#piccolo-adapter-acceptance))
- [x] Open-only operation and declared plugin fallback behavior; actual solver recorded ([fallback and refusal tests](../tests/test_control_solver.py), [launch and selection reference](operations/control.md#launch-and-solver-selection))
- [x] Control checks on request: pulse replay on a different engine, the declared fidelity target, and the declared constraints (contract §3); a second solver on the same ProblemSpec as a separate run ([checks](../src/entanglix/corrector/control.py), [constructed-pulse and linked CLI tests](../tests/test_control_checks.py), [GRAPE active-bound test](../tests/test_control_solver.py), [Piccolo active-bound report](../tests/probes/m4_piccolo.xml), [policy and tolerances](operations/control.md#requested-checks), [refreshed acceptance commands and evidence](environment.md#control-check-acceptance))
- [x] Standalone control from an existing system model through the CLI ([runner worker](../src/entanglix/protocols/_control_job.py), [CLI test](../tests/test_control_solver.py), [command and evidence](environment.md#control-baseline-acceptance))
- [x] `control` in investigations: exact-base admission, usage, limits, stop/deadline, and linked rerun; a test-only entry-point plugin candidate passes open replay ([control integration](../src/entanglix/protocols/control.py), [admission, finding, lifecycle, and CLI tests](../tests/test_control_investigations.py), [CPU report](../tests/probes/m4_control_investigations.xml), [reference](operations/control.md#investigations-and-reruns), [both solvers’ first-gate checks](../examples/transmon-control/README.md#requested-check-evidence))
- [x] MCP `control` and `rerun_control` tools, advertised capabilities, and versioned `control` skill; integration evidence is a scripted stdio client and four live sessions, two each on Claude Code and Codex ([live sessions](environment.md#live-control-sessions), [session summary](../tests/probes/m4_sessions_results.json), [tools](../src/entanglix/mcp.py), [procedure](../skills/control/SKILL.md), [CLI parity, check, rerun and refusal tests](../tests/test_mcp_control.py), [guidance tests](../tests/test_research_guidance.py), [CPU report](../tests/probes/m4_mcp.xml), [setup and command](environment.md#control-mcp-and-guidance-acceptance))
- [x] Applicable quality evidence recorded ([export and pulse-convention tests](../tests/test_control.py), [gradient and constraint evidence](../tests/test_control_solver.py), [independent first-gate checks](../tests/probes/m4_control_checks_results.json), [execution and plugin tests](../tests/test_control_investigations.py), [CLI/MCP conformance and guidance report](../tests/probes/m4_mcp.xml), [operational reference](operations/control.md))
- [x] **Done:** open baseline and Piccolo solve the same gate problem from one generated ProblemSpec, each candidate passing the checks requested, both in run dirs; a private plugin, if present, is discovered and its candidate is checked by the same open replay ([first-gate checks on both candidates](../tests/probes/m4_control_checks_results.json), [baseline candidate](../tests/probes/m4_baseline_results.json), [Piccolo candidate](../tests/probes/m4_piccolo_results.json), [measurements and limits](../examples/transmon-control/README.md#requested-check-evidence), [plugin discovery and replay](../tests/test_control_investigations.py))
- **Unblocks:** the Harmoniqs conversation once the shared ProblemSpec, the open-baseline and Piccolo runs, and their check reports are inspectable; audience/license and Appendix A rules still apply.

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
- [ ] Human `promote` command records the checks run and their outcomes, refuses a result with a failed check other than a violated declared expectation, and records source lineage and promoter; agents cannot promote
- [ ] Lineage queries and trial counts retain unsuccessful attempts within the declared investigation or hypothesis family
- [ ] Subsequent sweep reuses a catalogued candidate with both entry and run provenance
- [ ] When Block 7 is available: control warm-start from a catalogued pulse with recorded lineage
- [ ] Applicable quality evidence recorded
- [ ] **Done:** a catalogued candidate is reused in a sweep and promotion is auditable; pulse warm-start is demonstrated when control is available

---

## S4 — Interoperability (Block 12)

### [ ] Block 12 — Interoperability

**Depends on:** Blocks 3 and 7; shared calibration schemas extend conformance when Block 10 is available.

- [ ] A `control` run dir opens in amicode's Run Inspector, including `index`, `latest`, and the Inspector's required fields
- [ ] A ProblemSpec authored here solves under `amico-run` with no edits
- [ ] Shared device / experiment-card schemas where they overlap; calibration conformance when Block 10 is available
- [ ] Applicable quality evidence recorded
- [ ] **Done:** both run-directory and ProblemSpec conformance checks hold
- Optional: amicode Domain Pack, if wanted by both sides.

---

## Upgrade Path (no boxes until a milestone claims them)

cuDensityMat usage study (open-system usage review, CPU/GPU crossover and peak memory, member-by-member batch-versus-serial agreement), a prerequisite for automatic performance-based dispatch and advertised batching · Multi-GPU cuDensityMat · cuTensorNet engine for lattice-scale systems · multistart and continuation for control · longer scheduled unattended campaigns after bounded workflows demonstrate reliability · hardware-in-the-loop behind the same run contract.

Stronger authorization and isolation are scoped when a deployment needs bypass resistance or containment of untrusted code. Record its threat model and required identity/process/filesystem/network/device controls before adding acceptance boxes; cooperative checks do not prove those guarantees.
