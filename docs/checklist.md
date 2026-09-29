# Entanglix Build Checklist

Cross-referenced from [`roadmap.md`](roadmap.md) and [`../HARNESS_CONTRACT.md`](../HARNESS_CONTRACT.md). The roadmap is authoritative for scope and rationale, the contract for invariants; this file tracks status only.

Status legend: `[ ]` open · `[-]` in progress / partial · `[x]` done. Blocks are in strict dependency order — a block is done when all its criteria boxes are checked, and the next block starts then. No calendar: exit criteria gate progress, not dates.

---

## Milestone Map

| Milestone | Blocks | Status |
|---|---|---|
| M0 — Mandate + foundation | Decision 0, Block 0 | **in progress** |
| M1 — Model + engines | Blocks 1–2 | open |
| M2 — First end-to-end | Blocks 3–5 | open |
| M3 — Cluster | Block 6 | open |
| M4 — Control | Block 7 | open |
| M5 — Calibration + catalog | Blocks 8–9 | open |
| M6 — Harness | Block 10 | open |
| M7 — Interop | Block 11 | open |

The Harmoniqs conversation waits for M4. Nothing in this repository is shared before `roadmap.md` Appendix A is removed.

---

## M0 — Mandate and Foundation (Decision 0, Block 0)

### [-] Decision 0 — Mandate ADR
Highest-leverage decision in the roadmap. Nothing in Block 0 starts before `docs/adr/0001-mandate.md` reads **Accepted**. Items marked *walkthrough* need the owner's call; see the ADR draft for options.
- [x] Project name: Entanglix. Command: `entanglix`. Repo: this one, private, no remote yet.
- [x] Runner language: Python (`uv`), same substrate as the engines.
- [x] Substrate fixed (roadmap §0): scqubits, cuDensityMat via `qutip-cuquantum`, dynamiqs, QuTiP CPU reference; ProblemSpec OSS as control spec; private solvers as entry-point plugins.
- [ ] Platform family for v1 — *walkthrough*
- [ ] First `sweep` metric — *walkthrough*
- [ ] GPU target for v1 (workstation / Slurm / Kubernetes) and which machine — *walkthrough*
- [ ] Unit conventions (energy, time, angular vs. cyclic frequency), time-grid convention, default truncations — *walkthrough*
- [ ] License for the shareable harness and who may see it — *walkthrough*
- [ ] Environment pins recorded in `docs/environment.md`: GPU model, driver, CUDA, cuQuantum, Python, engine package versions — *needs hardware access*
- [ ] Block order question resolved: Block 10 (harness) before or after Block 8 (calibrate) — *walkthrough*
- [ ] `docs/adr/0001-mandate.md` status → Accepted
- **Unblocks:** Block 0.

### [-] Block 0 — Repo Foundation
- [x] `HARNESS_CONTRACT.md` at repo root
- [x] `docs/README.md` index with the sharing rule
- [x] `docs/roadmap.md`, `docs/checklist.md`
- [ ] Layout: `packages/entanglix`, `packages/entanglix-mcp`, `schemas/`, `skills/`, `tests/`
- [ ] `uv` project; `uv.lock`; environment lockfile with hash (CUDA / cuQuantum / driver)
- [ ] `.gitignore` (runs, caches, `.env`, large arrays)
- [ ] CI: CPU unit tests on a trivial test; `gpu` pytest marker skipped when no device
- [ ] Import-boundary test: no module outside `tests/` imports a private plugin or a scheduler client
- [ ] `AGENTS.md`; `CLAUDE.md` one-line pointer
- [ ] **Done:** clone → `uv sync` → `pytest` green on a machine without a GPU

---

## M1 — Model and Engines (Blocks 1–2)

### [ ] Block 1 — System Spec and Model Adapter
- [ ] `system.toml` JSON schema + pydantic model (circuit, truncation, drives, dissipators, parameters + priors, units)
- [ ] scqubits adapter: circuit → truncated operators in a chosen basis
- [ ] Explicit-operator fallback
- [ ] Operator export to QuTiP `Qobj` and to arrays for JAX
- [ ] Storage-format selection for cuDensityMat operators (CSR-probe rule from the scqubits GPU work)
- [ ] `entanglix validate <system.toml>`
- [ ] **Done:** a transmon–resonator and a fluxonium system validate, export to all three engines, spectra match scqubits to tolerance in tests

### [ ] Block 2 — Engines v0
- [ ] `Integrator` protocol with `capabilities()`
- [ ] QuTiP CPU reference engine
- [ ] cuDensityMat engine via `qutip-cuquantum`
- [ ] dynamiqs engine
- [ ] DLPack / CuPy zero-copy handoff where needed
- [ ] Cross-engine agreement tests at reference size (closed, open, time-dependent drive)
- [ ] Engine-selection table (contract §2) as a versioned file
- [ ] **Done:** agreement tests green on GPU CI or a documented manual GPU run

---

## M2 — First End-to-End (Blocks 3–5)

### [ ] Block 3 — Runner and Run Contract
- [ ] Gate: spec validation, static import scan, environment check (driver, cuQuantum, memory)
- [ ] Run dir per contract §3: `run.toml` first, `spec/` frozen, `FINISHED` last, atomic `result.toml`
- [ ] Job spawn in its own process group; exit-code contract
- [ ] `index` and `latest` at the runs root
- [ ] Local executor
- [ ] **Done:** a killed job leaves no `FINISHED`; a completed job's `result.toml` validates; a run is reproducible from `spec/` + `code_version` + `env_hash`

### [ ] Block 4 — Protocol `sweep`
- [ ] Grid sweep over scqubits parameters against the Decision-0 metric, batched on cuDensityMat
- [ ] Bayesian sweep
- [ ] Per-iteration plots; `result.toml` metric-surface summary
- [ ] **Done:** one sweep runs end to end on GPU through `entanglix`; corrector simulate/sweep checks pass

### [ ] Block 5 — Corrector v0
- [ ] Separate package; code hashed at load; versioned thresholds file outside the agent tree
- [ ] Cross-engine, convergence, positivity checks
- [ ] `compare_reference` verb
- [ ] **Done:** a deliberately under-converged run is flagged `UNVERIFIED`; a hash mismatch fails the run

---

## M3 — Cluster (Block 6)

### [ ] Block 6 — GPU Executor
- [ ] Scheduler executor (per Decision 0) behind the same `Executor` interface as local
- [ ] Artifact mirroring: small arrays back; large arrays by path + digest
- [ ] Warming and lost-contact timeouts; abort
- [ ] **Done:** `sweep` runs unchanged on the cluster and locally; a lost job terminates as `aborted`, never `completed`

---

## M4 — Control (Block 7)

### [ ] Block 7 — Protocol `control`
- [ ] ProblemSpec OSS adopted verbatim (or `solver.options` extension recorded by ADR)
- [ ] `SolverBackend` protocol; entry-point discovery
- [ ] Open GRAPE-style baseline
- [ ] Piccolo adapter via `amico-run`
- [ ] Cross-solver corrector check; re-simulation on a different engine
- [ ] **Done:** one gate solved by the open baseline and by Piccolo from the same ProblemSpec, both verified, both in run dirs; a private plugin, if present, is discovered and verified by the open solver
- **Unblocks:** the Harmoniqs conversation.

---

## M5 — Calibration and Catalog (Blocks 8–9)

### [ ] Block 8 — Protocol `calibrate`
- [ ] Data contract for calibration inputs with point-in-time provenance
- [ ] Gradient lane (dynamiqs)
- [ ] Gradient-free / Bayesian lane
- [ ] Holdout split; identifiability report
- [ ] `propose_device_update` proposal file; human apply command
- [ ] **Done:** synthetic-data round trip recovers known parameters within reported uncertainty; the holdout is unreachable from the fitting code path

### [ ] Block 9 — Catalog and Ledger
- [ ] Append-only ledger, one file per entry, fields per contract §8
- [ ] `catalog/staging/`; `register_result`; human `promote` command
- [ ] Warm-start from a catalogued pulse
- [ ] **Done:** a control run warm-starts from a catalogued pulse; lineage is queryable

---

## M6 — Harness (Block 10)

### [ ] Block 10 — Agent Harness
- [ ] MCP server over stdio projecting contract §6
- [ ] Tool-call audit log and prompt log (contract §9)
- [ ] Skills: `system-modeler`, `simulate`, `design-sweep`, `control`, `calibrate`, `gpu-budget`
- [ ] `AGENTS.md` finalized
- [ ] Prompt-injection test cases; pinned runtime version in CI
- [ ] **Done:** an agent session runs `sweep` and `control` end to end through tools only, every call logged, no path to promotion or device-model write

---

## M7 — Interoperability (Block 11)

### [ ] Block 11 — Interoperability
- [ ] A `control` run dir opens in amicode's Run Inspector
- [ ] A ProblemSpec authored here solves under `amico-run` with no edits
- [ ] Shared device / experiment-card schemas where they overlap
- [ ] Optional: amicode Domain Pack, if wanted by both sides
- [ ] **Done:** both interop checks above hold

---

## Upgrade path (no boxes until a milestone claims them)

Multi-GPU cuDensityMat · cuTensorNet engine for lattice-scale systems · multistart and continuation for control · scheduled unattended campaigns · hardware-in-the-loop behind the same run contract.
