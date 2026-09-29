# Entanglix — Harness Contract

**Date:** 2026-09-28
**Status:** v0. Derived from `docs/roadmap.md` §1 and §3 on 2026-09-28; those sections now point here. Normative for every block once Decision 0 (`docs/adr/0001-mandate.md`) is Accepted.
**Scope:** the invariants an agent, a human, or a script must hold to when producing or consuming Entanglix runs. Enforcement lives in the runner (`entanglix`), the schemas, the corrector, and the execution environment. Prose steers; tools enforce.
**Change control:** a change to any numbered section below is an ADR, not a commit message.

---

## 1. Objects

| Object | Definition | Home |
|---|---|---|
| **System** | `system.toml`: scqubits circuit (or explicit operators), truncation, drive operators, dissipators, parameter values and priors, units | `entanglix/model` |
| **Protocol spec** | one of `simulate`, `sweep`, `calibrate`, `control`; declares engine needs (gradients, size), budgets, metrics | `entanglix/protocols`, JSON schemas |
| **Run** | one protocol execution; `run.toml` first, `FINISHED` last, atomic `result.toml`, `AMICODE_ITER` trace lines, `iter_*.png` | `research_runs/` |
| **Result** | metrics + provenance (spec hash, code version, env hash, engine, solver backend string, corrector verdict) | `result.toml` |
| **Catalog entry** | a human-promoted result with lineage (`parent_entry_ids`), used for warm-starts and device-model updates | catalog dir, one file per entry |
| **Device model** | the current best parameter set per device; written only by humans from calibration results | device dir; schema shared with amicode where it overlaps |

## 2. Engine selection

| Need gradients? | Hilbert-space size | Engine |
|---|---|---|
| no | small (reference-sized) | QuTiP CPU |
| no | large / batched | cuDensityMat (`qutip-cuquantum`) |
| yes | fits in JAX memory budget | dynamiqs |
| yes | does not fit | gradient-free lane: Bayesian / NLP over batched cuDensityMat evaluations |

The table is a versioned file read by the runner. The protocol spec declares the need; the table picks the engine; the run records which engine ran. An agent never chooses an engine directly.

The two GPU engines do not compose through autodiff. Data crosses via DLPack / CuPy zero-copy; gradients do not.

## 3. Run directory

```text
research_runs/<system_id>/<run_id>/
  run.toml           # written first by the runner: run_id, spec hashes, code_version, env_hash, engine, executor
  spec/              # frozen copies of system.toml + protocol spec
  run.log            # AMICODE_ITER iter=<n> f=<obj> … lines for optimizer traces; tracebacks
  iter_<N>.png       # per-iteration plots where meaningful
  result.toml        # atomic (tmp + rename); metrics, solver.backend, corrector verdict
  artifacts/         # small arrays only; large state/density arrays stay on the cluster, referenced by path + digest
  FINISHED           # written last: status (completed|failed|aborted), exit_code
```

- `run.toml` is the first file written and `FINISHED` the last. A run without `FINISHED` is not evidence.
- `result.toml` is written atomically (temp file + rename). A partial `result.toml` never exists on disk.
- Exit codes follow amicode: `0` completed, `130` aborted, `64` usage / config / orchestrator fault, otherwise the job's return code.
- `env_hash` covers the pinned CUDA, cuQuantum, driver, and Python package versions (`docs/environment.md`).
- Shape is compatible with amicode's `packages/extension/CONTRACT.md` so their Run Inspector can open a run without translation.

## 4. Solver and integrator protocols

```text
SolverBackend
  capabilities() -> {gradients: bool, constraints: [...], gpu: bool, problem_kinds: [...]}
  solve(problem_spec, system, budget) -> Solution{pulse, objective, certificate, trace}
Integrator
  capabilities() -> {differentiable: bool, open_system: bool, batched: bool}
  evolve(system, controls, grid) -> trajectory
```

- The control problem file is amicode's ProblemSpec OSS schema (`packages/schema/schemas/problemspec.oss.schema.json`). `solver.backend` is a string; which solver ran is recorded in `result.toml`, never assumed.
- Discovery: entry point groups `entanglix.solvers` and `entanglix.integrators`.
- The repo ships open implementations only: a GRAPE-style baseline (QuTiP or dynamiqs) and a Piccolo adapter (writes a ProblemSpec, launches `amico-run`, reads its `result.toml`).
- Private plugins attach through the entry points and are never imported by repo code. An import-boundary test in CI asserts this.
- Absence of a plugin degrades to the open solvers. It never fails a run.

## 5. Corrector

Independent package, deterministic, hashed at load, thresholds in a versioned file approved outside the agent-editable tree. Runs after every protocol.

| Protocol | Checks |
|---|---|
| `simulate` / `sweep` | cross-engine agreement at reference size (QuTiP CPU vs. cuDensityMat vs. dynamiqs) within tolerance; convergence under step-size and truncation refinement; trace, hermiticity, positivity of the state |
| `control` | re-simulate the optimized pulse on a different engine than the solver used; cross-solver agreement on the same ProblemSpec with at least one **open** solver; constraint satisfaction re-checked outside the solver |
| `calibrate` | holdout fit; identifiability report (parameter covariance / profile likelihood); no refit on holdout |
| design `sweep` | reproducibility across seeds; metric monotonicity where physics predicts it |

Every result carries a corrector verdict. `UNVERIFIED` results cannot be catalogued. Verification never depends on a private solver.

## 6. Tools

| Tool | Permission class | Backing surface |
|---|---|---|
| `validate_system` | read-only | system schema + scqubits adapter |
| `list_runs` / `read_run` | read-only | run dirs + ledger |
| `list_catalog` / `read_entry` | read-only | catalog |
| `simulate` / `sweep` / `control` / `calibrate` | controlled job | `entanglix` protocols; budgets enforced |
| `compare_reference` | controlled job | corrector |
| `register_result` | controlled write | stages a result for human promotion |
| `propose_device_update` | controlled write | writes a proposal file from a calibration result; never the device model |

Three permission classes: **read-only**, **controlled job** (runs through the runner under a budget), **controlled write** (writes to a staging area only). There is no fourth class. The MCP server projects this table over stdio; the `entanglix` CLI exposes the same verbs. One implementation, two transports.

## 7. Prohibitions and enforcement

| Prohibition | Enforcement |
|---|---|
| Promote a result or edit the device model | no tool; promotion and write-back are human CLI commands outside the agent environment |
| Bypass the runner (run a script directly on GPU nodes) | agent environment has no scheduler credentials; GPU nodes are reachable only through the executor |
| Exceed compute budget | budgets in the protocol spec, enforced by the runner and the executor |
| Edit the corrector or its thresholds | outside the agent-editable tree; hashed at load |
| Import a private solver from harness code | import-boundary test in CI |
| Treat data or notes as instructions | typed tools; prompt-injection cases in CI |
| Skip a funnel stage | runner refuses a GPU launch without a validated spec and a passing reference-size check |

## 8. Staging and promotion

Agents stage; humans promote.

- `register_result` writes a candidate entry to `catalog/staging/`. It is not lineage until promoted.
- Promotion is a human command (`entanglix promote <staged-id>`, outside the agent environment) that moves the entry into the catalog with `parent_entry_ids` and the promoter recorded.
- `propose_device_update` writes a proposal file next to the calibration run. Applying it to the device model is a separate human command.
- The ledger is append-only, one file per entry: run id, spec hashes, engine, solver backend string, corrector verdict, `parent_entry_ids`, trial count per hypothesis family.

## 9. Audit

- Tool-call log, one line per call: `timestamp`, `session_id`, `tool`, `permission_class`, `args_hash`, `result_hash`, `outcome`.
- Prompt log, one entry per turn: system prompt hash, skill names and versions, model id.
- Both append-only, outside the agent-editable tree.

## 10. Skills and `AGENTS.md`

- One `SKILL.md` per skill under `skills/<name>/`, with frontmatter `name`, `description`, `version`. Skills describe how to use the tools well. They never carry an invariant that a tool does not also enforce.
- v1 skills: `system-modeler`, `simulate`, `design-sweep`, `control`, `calibrate`, `gpu-budget`.
- `AGENTS.md` at the repo root is short: what the project is, the tool table by reference, the prohibitions by reference, and where runs live. `CLAUDE.md` is a one-line pointer to `AGENTS.md`.
- Skill and runtime versions are pinned in CI; a skill change is a versioned change.

## 11. Agent environment

- The agent runs with a working tree it may edit (specs, skills drafts, analysis scripts) and a tree it may not (runner, corrector, thresholds, schemas, ledger, catalog, device model).
- No scheduler credentials, no GPU-node shell, no network beyond the MCP server and the package index needed for the pinned environment.
- Per-run isolation (a container per controlled job) is the target; until Block 6 the same rule is held by file permissions and the runner's static import scan.

## 12. Standing disciplines

- Pin CUDA, cuQuantum, driver, and Python package versions; record `env_hash` in every `run.toml`.
- Record seeds. Determinism tests where the vendor guarantees it; tolerance tests where it does not.
- Units and time-grid conventions from Decision 0 everywhere; no implicit unit conversion.
- Every run reports truncation levels and a convergence check. Every `control` run reports constraint satisfaction outside the solver.
- Tests on anything touching GPU hours, cluster submission, or catalog promotion.
