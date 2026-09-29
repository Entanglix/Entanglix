# Entanglix — Implementation Plan

**Date:** 2026-09-27
**Project:** Entanglix — quantum device physics research harness: modeling → simulation → optimization (control, calibration, device design)
**Status:** Authoritative for scope and block order. Status lives in [`checklist.md`](checklist.md); invariants in [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md); decisions in [`adr/`](adr/). Moved here from the amicode draft on 2026-09-28. Settled: project name Entanglix; command `entanglix` (no `-run` suffix). `amico-run` remains the external Piccolo launcher.
**Source documents:** amicode `packages/extension/CONTRACT.md` (run-dir shape), amicode `packages/schema/schemas/problemspec.oss.schema.json` (control spec), amicode ADR-0008 (domain-pack boundary), ADR-0011 (harness contract).

---

## 0. Working Assumptions

### Mandate

- Domain: superconducting-circuit device physics and quantum control. System modeling (circuit parameters → Hamiltonians), open-system simulation on GPU, and three optimization tasks: optimal control (pulse/gate synthesis), model calibration (fit Hamiltonian and noise parameters to data), and device design (optimize circuit parameters against simulated metrics).
- Not in scope: gate-model algorithms (VQE, QAOA, annealing schedules), circuit compilation, hardware drivers.
- Users: the author and close collaborators; expert users. No product, no UI, no VS Code extension. The agent runtime is a commodity (Codex CLI or equivalent) driven through `AGENTS.md`, skills, and an MCP server.
- "Enterprise production grade" means: every run reproducible from spec + code version + environment hash; every launch gated; every result independently verified by a different engine or solver; artifacts and lineage on disk; GPU jobs run through a scheduler, not by hand.

### Substrate decisions (fixed)

- **Python-first, single substrate.** Modeling with `scqubits`; dynamics on cuQuantum's cuDensityMat through `qutip-cuquantum`; differentiable dynamics with `dynamiqs` (JAX); QuTiP CPU as the exact small-size reference.
- **Two engines that do not compose through autodiff.** cuDensityMat is large-scale, batched, non-differentiable. JAX (dynamiqs) is differentiable and moderate-size. Data crosses via DLPack / CuPy zero-copy; gradients do not. Engine selection is a declared property of a protocol run ("needs gradients?" × Hilbert-space size), never an agent's judgment call.
- **Solver-agnostic control spec.** amicode's ProblemSpec OSS schema (`system`, `goal`, `pulse`, `problem`, `integrator`, `solver.backend`) is adopted as the control problem file. `solver.backend` is a string; which solver ran is recorded, never assumed.
- **Private solver plugins.** High-performance solvers may be proprietary. They attach through a `SolverBackend` / `Integrator` protocol via Python entry points and are never imported by this repo. The repo's tests run with open solvers only; an import-boundary test enforces it. Verification never depends on a private solver (contract §4–§5).
- **amicode's role is contracts and a corrector, not substrate.** Run-dir shape and schemas are adopted for interoperability; Piccolo (via `amico-run`) is an open second-opinion control solver. Dynamics never route through Julia.

### Key principles

- Prose steers, tools enforce. Invariants live in the runner, the schemas, the corrector, and the environment.
- Stage, never promote. Agents write candidates and results; humans promote into the catalog and the device model.
- Cheap-to-expensive funnel. Spec validation → small-size reference → GPU run → corrector. Nothing skips a stage.
- Truncation and convergence are reported on every run, never assumed.

---

## 1. Architecture

```text
Model plane      system.toml ──► scqubits circuit ──► truncated operators, dissipators, parameter priors
Engine plane     QuTiP CPU (reference) │ cuDensityMat via qutip-cuquantum (large, batched) │ dynamiqs (JAX, differentiable)
Protocol plane   simulate │ sweep │ calibrate │ control        (each: spec in, run dir out)
Solver plane     open GRAPE baseline │ Piccolo via amico-run │ private plugins (entry points)
Run plane        research_runs/<system_id>/<run_id>/ … FINISHED   (amicode run-dir shape)
Verify plane     corrector: cross-engine, cross-solver, convergence, positivity, holdout   (hashed, outside agent tree)
Catalog plane    results catalog + ledger (lineage, trial counts); device-model write-back is a human act
Harness plane    entanglix CLI  ⇄  MCP server  ⇄  skills + AGENTS.md  ⇄  Codex (or any runtime)
Executors        local GPU │ Slurm │ Kubernetes   (artifacts mirrored back; large arrays stay on the cluster)
```

The invariants behind this diagram — object model, engine-selection table, run directory and exit codes, solver and integrator protocols, corrector checks — are specified in [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md) §1–§5. That file is normative; this roadmap says when each part gets built.

---

## 2. Build Blocks (strict dependency order)

No calendar. A block is done when its done criteria hold. Each block is independently useful.

### Decision 0 — Mandate ADR

- Name, command, and repo are settled: Entanglix, `entanglix`, this repository. License for the shareable harness, and who may see it, remain open.
- First platform family (recommendation: transmon and fluxonium, matching existing scqubits work) and first device metric for `sweep`.
- GPU target for v1: local workstation, Slurm, or Kubernetes.
- Unit conventions (GHz / ns, ħ = 1), time grid conventions, default truncations.
- **Done when:** `docs/adr/0001-mandate.md` committed; unblocks Block 0.

### Block 0 — Repo Foundation

- Layout: `packages/entanglix` (library and the `entanglix` command), `packages/entanglix-mcp`, `schemas/`, `skills/`, `docs/`, `tests/`; `uv` project; pinned CUDA / cuQuantum / driver versions recorded in an environment lockfile with a hash.
- CI: unit tests on CPU (no GPU required), a `gpu` marker for GPU tests, an **import-boundary test** asserting no module outside `tests/` imports a private plugin or a scheduler client.
- `HARNESS_CONTRACT.md` and `docs/README.md` exist (2026-09-28); `AGENTS.md` and the `CLAUDE.md` one-line pointer are written once the layout is real.
- **Done when:** clone → `uv sync` → `pytest` green on a machine without a GPU.

### Block 1 — System Spec and Model Adapter

- `system.toml` JSON schema and pydantic model; scqubits adapter (circuit → truncated operators in a chosen basis); explicit-operator fallback; dissipators; drive operators; parameter values + priors; unit handling.
- Operator export to QuTiP `Qobj` and to arrays for JAX; storage-format selection for cuDensityMat operators (reuse the CSR-probe decision rule already prototyped in the scqubits GPU work).
- `entanglix validate <system.toml>`.
- **Done when:** a transmon–resonator and a fluxonium system validate, export to all three engines, and spectra match scqubits to tolerance in tests.

### Block 2 — Engines v0

- QuTiP CPU reference engine; cuDensityMat engine via `qutip-cuquantum`; dynamiqs engine; each behind the `Integrator` protocol with `capabilities()`.
- DLPack / CuPy zero-copy handoff for arrays between cuDensityMat and JAX where needed.
- Cross-engine agreement tests at reference size (closed and open system, time-dependent drive).
- **Done when:** the engine-selection table (contract §2) is a versioned file, and the agreement tests are green on GPU CI or a documented manual GPU run.

### Block 3 — Runner and Run Contract

- `entanglix` gate + launch: spec validation, static import scan of any authored script, environment check (driver, cuQuantum, memory), run-dir creation (`run.toml` first), job spawn in its own process group, `FINISHED` last, exit-code contract, `index` and `latest` at the runs root.
- Local executor only in this block.
- **Done when:** a killed job leaves no `FINISHED`; a completed job's `result.toml` validates; a run is reproducible from `spec/` + `code_version` + `env_hash`.

### Block 4 — Protocol `sweep` (first end-to-end)

- Design sweep over scqubits parameters (grid, then Bayesian) against a simulated metric (for example dispersive shift, T1 limit from a noise model, gate error under a fixed pulse), batched on cuDensityMat.
- Plots per iteration; `result.toml` with the metric surface summary.
- **Done when:** one sweep for the Decision-0 metric runs end to end on GPU through `entanglix`, and the corrector's simulate/sweep checks pass.

### Block 5 — Corrector v0

- Cross-engine, convergence, positivity checks as a separate package with hashed code and a versioned thresholds file; `compare_reference` verb.
- **Done when:** a deliberately under-converged run is flagged `UNVERIFIED`; a hash mismatch fails the run.

### Block 6 — GPU Executor

- Slurm (or Kubernetes, per Decision 0) executor behind the same `Executor` interface as local; artifact mirroring (small arrays back, large arrays referenced by path + digest); warming and lost-contact timeouts; abort.
- **Done when:** `sweep` runs unchanged on the cluster and locally; a lost job terminates as `aborted`, never as `completed`.

### Block 7 — Protocol `control`

- Adopt ProblemSpec OSS verbatim; `SolverBackend` protocol and entry-point discovery; open GRAPE-style baseline; Piccolo adapter via `amico-run`.
- Cross-solver corrector check (contract §5) and re-simulation on a different engine.
- **Done when:** one gate (Decision-0 platform) is solved by the open baseline and by Piccolo from the same ProblemSpec, both results verified, both in run dirs; a private plugin, if present, is discovered and its result is verified by the open solver.

### Block 8 — Protocol `calibrate`

- Data contract for calibration inputs (measured spectra, Ramsey/T1/T2 traces, gate-error tables) with point-in-time provenance.
- Gradient lane (dynamiqs) and gradient-free / Bayesian lane; holdout split; identifiability report; **write-back to the device model is a proposal file a human applies**.
- **Done when:** synthetic-data round trip recovers known parameters within reported uncertainty; the holdout is unreachable from the fitting code path.

### Block 9 — Catalog and Ledger

- Append-only ledger (one file per entry): run, spec hashes, engine, solver backend string, corrector verdict, `parent_entry_ids`, trial counts per hypothesis family.
- Catalog of promoted results for warm-starts; promotion is a human command.
- **Done when:** a control run warm-starts from a catalogued pulse; lineage is queryable.

### Block 10 — Agent Harness

- MCP server projecting the tool table (contract §6) over stdio; tool-call audit log; skills (`system-modeler`, `simulate`, `design-sweep`, `control`, `calibrate`, `gpu-budget`, plus vendored generic research skills); `AGENTS.md` finalized.
- Prompt-injection test cases; pinned runtime version in CI.
- **Done when:** an agent session runs `sweep` and `control` end to end through tools only, every call logged, no path to promotion or device-model write.

### Block 11 — Interoperability

- Conformance: run dirs readable by amicode's Run Inspector; ProblemSpec files accepted by `amico-run` unchanged; shared device / experiment-card schemas.
- Optional: package as an amicode Domain Pack if wanted by both sides.
- **Done when:** a `control` run dir opens in the Run Inspector; a ProblemSpec authored here solves under `amico-run` with no edits.

### Upgrade path (not blocks)

Multi-GPU cuDensityMat; cuTensorNet engine for lattice-scale systems; multistart and continuation for control; scheduled unattended campaigns; hardware-in-the-loop via an external device bridge behind the same run contract.

---

## 3. Harness Contract

Tool table and permission classes, prohibitions with their enforcement, staging and promotion, audit, skills and `AGENTS.md` conventions, and the agent environment are in [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md) §6–§11. Block 10 builds the MCP projection of that contract; Blocks 3, 5, and 9 build the runner, corrector, and ledger it relies on.

---

## 4. Standing Disciplines (every block)

- Pin CUDA, cuQuantum, driver, and Python package versions; record an environment hash in every `run.toml`.
- Seeds recorded; determinism tests for engines where the vendor guarantees it, tolerance tests where it does not.
- Units and time-grid conventions from Decision 0 everywhere; no implicit unit conversion.
- Every run reports truncation levels and a convergence check; every `control` run reports constraint satisfaction outside the solver.
- Tests on anything touching money-equivalents: GPU hours, cluster submission, catalog promotion.

---

## 5. Milestones

| Milestone | Blocks | Exit |
|---|---|---|
| M0 — Mandate + foundation | Decision 0, Block 0 | ADR committed; CPU tests green |
| M1 — Model + engines | Blocks 1–2 | spectra match scqubits; engines agree at reference size |
| M2 — First end-to-end | Blocks 3–5 | one `sweep` on GPU, verified, reproducible |
| M3 — Cluster | Block 6 | same `sweep` on the scheduler |
| M4 — Control | Block 7 | open baseline and Piccolo agree on one gate |
| M5 — Calibration + catalog | Blocks 8–9 | synthetic round trip; warm-start works |
| M6 — Harness | Block 10 | agent runs `sweep` + `control` through tools only |
| M7 — Interop | Block 11 | Run Inspector opens a run; `amico-run` accepts a ProblemSpec |

External conversation with Harmoniqs is scheduled after M4, when the shared ProblemSpec and the cross-solver corrector exist as artifacts.

---

## 6. Things Not to Build Yet

- Any UI or editor extension.
- Gate-model algorithm layers.
- A scheduler or job queue of our own.
- Hardware drivers.
- Unattended multi-day campaigns before Block 10 holds.
- A Julia substrate.

---

## 7. Open Decisions

Settled: project name Entanglix; command `entanglix`; runner in Python (`uv`), same substrate as the engines.

- License (Apache-2.0 matches the schemas adopted from amicode) and who may see the shareable harness.
- GPU target for v1 (workstation vs. Slurm vs. Kubernetes).
- First platform family and first `sweep` metric.
- ProblemSpec OSS verbatim vs. an additive `solver.options` table.
- Whether Block 10 (harness) precedes Block 8 (calibration) if agent-driven sweeps and control are useful before measured data exists.

---

## Appendix A — Private solver plugins

- A private solver stays in its own package outside this repository. Entanglix code and docs never name or import it, and need no change to use it.
- The package registers a backend factory under the `entanglix.solvers` entry-point group, implements the solver interface in the [control reference](operations/control.md#launch-and-solver-selection), and pins the version of the private solver it wraps.
- Each run records the requested and actual solver and its distribution version. A spec that names a private solver can declare an open `fallback`, used only where the plugin is not installed.
- A private solver's results receive the same open checks as any other solver's, and no check depends on a private solver.
- Evolution engines have no plugin entry point yet. One would follow the same pattern when a plugin needs it, with the open engines remaining the default.

## Appendix B — amicode touchpoints (read-only references)

- `packages/extension/CONTRACT.md` — run-dir shape and exit codes adopted in contract §3.
- `packages/schema/schemas/problemspec.oss.schema.json` — the control spec adopted in Block 7.
- `packages/amico-run/src/local_executor.ts`, `remote_executor.ts`, `types.ts` — reference for the executor interface and mirroring lifecycle.
- `packages/extension/julia/verify_rollout.jl` — their re-rollout corrector; ours differs by using a different engine and solver.
- `docs/adr/0008-autoresearch-identity-domain-pack-boundary.md`, `docs/adr/0011-harness-contract.md` — domain-pack boundary and harness-as-contract, the basis for Block 11.
