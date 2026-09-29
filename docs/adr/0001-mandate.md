# ADR-0001: Mandate and Constraints

**Date:** 2026-09-28
**Status:** Draft — agenda for the Decision 0 walkthrough. Nothing in §3 is decided until this line reads **Accepted**.
**Source documents:** `docs/roadmap.md` §0 and Decision 0, `HARNESS_CONTRACT.md`.
**Purpose:** record the decisions that bind Blocks 0–4 before any code is written. Each open item lists its options, what it binds, and a recommendation. The recommendation is not the decision.

---

## 1. Context

Entanglix is a research harness for superconducting-circuit device physics: system modeling → open-system simulation on GPU → optimization (optimal control, model calibration, device design). It is Python-first on scqubits, cuDensityMat via `qutip-cuquantum`, dynamiqs, and a QuTiP CPU reference. It has no product surface; a commodity agent runtime is driven through `AGENTS.md`, skills, and an MCP server.

## 2. Already decided (recorded here, not reopened)

| Decision | Value | Where |
|---|---|---|
| Name and command | Entanglix; `entanglix` | roadmap header |
| Runner language | Python, `uv` | roadmap §7 |
| Substrate | scqubits · cuDensityMat (`qutip-cuquantum`) · dynamiqs · QuTiP CPU reference | roadmap §0 |
| Two-engine rule | cuDensityMat and JAX do not compose through autodiff; engine chosen by a versioned table | contract §2 |
| Control spec | amicode ProblemSpec OSS, `solver.backend` a recorded string | contract §4 |
| Private solvers | entry-point plugins; never imported by repo code; verification never depends on them | contract §4–§5 |
| amicode's role | contracts and an open second-opinion solver (Piccolo via `amico-run`); never the dynamics substrate | roadmap §0 |
| Posture | stage-never-promote; prose steers, tools enforce | contract §7–§8 |
| Out of scope | gate-model algorithms, circuit compilation, hardware drivers, any UI | roadmap §0, §6 |

## 3. Decisions to make

### 3.1 Platform family for v1

- **Binds:** Block 1 test systems; Block 4 metric; the corrector's reference cases.
- **Options:** (a) transmon–resonator and fluxonium; (b) transmon only; (c) fluxonium only; (d) either plus a bosonic mode (cavity / cat-qubit line).
- **Recommendation:** (a). Both circuits are already in the owner's scqubits work, and the pair covers a weakly and a strongly anharmonic spectrum, which is what the truncation and convergence checks need to be tested against.
- **Decision:** _pending_

### 3.2 First `sweep` metric

- **Binds:** what Block 4 computes; whether Block 4 exercises time evolution or only diagonalization.
- **Options:**
  - (a) Spectra-only: dispersive shift χ or a T1 limit from a scqubits noise model over (E_J/E_C, g). Fast; scqubits alone; **does not touch cuDensityMat**, so the GPU path is untested until Block 7.
  - (b) Dynamics under a fixed pulse: gate error or leakage for an analytically known pulse (Gaussian π pulse on a transmon coupled to a resonator) under T1/Tφ, batched over device parameters on cuDensityMat. Exercises the open-system GPU path that M2 exists to prove.
  - (c) ZZ crosstalk across a two-qubit coupler sweep. Physically rich; larger Hilbert space; heavier first target.
- **Recommendation:** (b). M2's exit is "one sweep on GPU, verified"; a spectra-only sweep would pass that gate without proving the engine. Keep (a) as the Block 1 test, not the Block 4 target.
- **Decision:** _pending_

### 3.3 GPU target for v1

- **Binds:** Block 2 where GPU tests run; Block 6 which executor gets built; Decision on artifact mirroring.
- **Facts observed 2026-09-28:** CUDA 13.2 toolkit at `/usr/local/cuda-13.2`; `nvidia-smi` could not reach a driver from the agent sandbox (verify from a normal shell); `uv` installed; Python 3.13 (miniforge).
- **Options:** (a) local workstation only through M2, scheduler chosen at Block 6; (b) Slurm cluster from Block 2 (name the cluster); (c) Kubernetes from Block 2.
- **Recommendation:** (a), with the cluster named now so Block 6 is a known target, not a survey.
- **Decision:** _pending_ — GPU model(s), machine, cluster name if any.

### 3.4 Units, time grid, truncation defaults

- **Binds:** every `system.toml`, every engine adapter, the corrector tolerances. This is where the 2π bugs live.
- **Options:**
  - Energy: scqubits convention (energies as frequencies in GHz, h = 1) vs. angular (rad/ns, ħ = 1).
  - Time: ns.
  - Where the 2π enters: explicitly at Hamiltonian construction, recorded in a `[units]` table in `system.toml`; never inside an engine adapter.
  - Time grid: uniform grid with `dt` and `n_steps` declared in the protocol spec; engines may sub-step internally but report on the declared grid.
  - Truncation: per-subsystem `n_levels` declared; the corrector re-runs at `n_levels + 2` for the convergence check.
- **Recommendation:** scqubits convention for energies, ns for time, explicit 2π at construction, declared uniform grid, declared truncation with the +2 convergence rule.
- **Decision:** _pending_

### 3.5 License and audience

- **Binds:** whether a `LICENSE` file exists before M4; how `roadmap.md` Appendix A is handled; what Harmoniqs may see.
- **Options:** (a) no license file, repo private until M4, license chosen at share time; (b) Apache-2.0 now (matches the amicode schemas adopted); (c) MIT now.
- **Recommendation:** (a). A license file on a private repo decides nothing and invites an early share. Choose Apache-2.0 at M4 unless the Harmoniqs conversation argues otherwise.
- **Decision:** _pending_

### 3.6 Environment pins

- **Binds:** `env_hash` in every `run.toml`; `docs/environment.md`.
- **Needs:** GPU model, driver version, CUDA version, cuQuantum version, Python version, and versions of `qutip`, `qutip-cuquantum`, `dynamiqs`, `jax`, `scqubits`, `cupy`. These are read off the machine in §3.3, not chosen.
- **Decision:** _pending hardware access_

### 3.7 Block order: harness before calibration?

- **Binds:** whether Block 10 runs after Block 7 (M4) or after Block 9 (M5).
- **Options:** (a) as written — calibrate (8) and catalog (9) before harness (10); (b) harness right after control (7), calibrate later, if agent-driven sweeps and control are useful before measured device data exists.
- **Recommendation:** revisit at M4. If there is no measured data in hand at M4, take (b).
- **Decision:** _pending_

## 4. Consequences

Written when §3 is filled. Expected shape: which Block 1 systems become test fixtures; the Block 4 metric function signature; the Block 6 executor to build; the `[units]` table schema; the sharing rule in `docs/README.md` confirmed or amended.
