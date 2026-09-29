# ADR-0001: Mandate and Constraints

**Date:** 2026-09-28
**Status:** Draft — Decision 0 walkthrough in progress. Agreed decisions are recorded individually in §3; the initial modeling coverage and fixture set remain to be pinned before acceptance.
**Source documents:** [roadmap](../roadmap.md) §0 and Decision 0; [harness contract](../../HARNESS_CONTRACT.md).
**Purpose:** record the foundation choices needed by Blocks 0–2. Recommendations remain proposals. Sweep metrics, investigation design, and deployment mechanisms are decided when their prerequisites are available (§4).

---

## 1. Context

Entanglix supports expert-guided superconducting-circuit research through independently useful modeling, simulation, sweep/search, control, and calibration workflows. Agents help formulate bounded investigations; a complete device-design objective is optional. The substrate is Python-first: scqubits, cuDensityMat via `qutip-cuquantum`, dynamiqs, and a QuTiP CPU reference. A commodity agent runtime uses skills and MCP; the runner enforces authority, execution limits, and verification. There is no product UI.

## 2. Already decided (recorded here, not reopened)

| Decision | Value | Where |
|---|---|---|
| Name and command | Entanglix; `entanglix` | roadmap header |
| Runner language | Python, `uv` | roadmap §7 |
| Substrate | scqubits · cuDensityMat (`qutip-cuquantum`) · dynamiqs · QuTiP CPU reference | roadmap §0 |
| Two-engine rule | cuDensityMat and JAX do not compose through autodiff; engine chosen by a versioned table | contract §2 |
| Control spec | amicode ProblemSpec OSS; `solver.backend` records the solver | contract §1 |
| Private solvers | Entry-point plugins; repo code never imports them directly; verification remains open | contract §1, §3 |
| amicode's role | Contracts and an open second-opinion solver (Piccolo via `amico-run`); never the dynamics substrate | roadmap §0 |
| Research direction | Expert-guided formulation, bounded autonomy, independent entry points, professional quality | roadmap §0, §4 |
| Preferred sequence | Verified local sweep → bounded exploration → control → co-design; technical dependencies govern availability | roadmap §2, §5 |
| Initial exploration | Runner investigations (6a), then isolated agent integration (6b); calibration and cluster execution are independently scheduled | roadmap §2 |
| First end-to-end sweep | Open-system dynamics on the local GPU; exact metric remains open | roadmap Block 5 |
| Posture | Stage-never-promote; tools enforce authority, limits, verification, and isolation | contract §3–§7 |
| Out of scope | Gate-model algorithms, circuit compilation, hardware drivers, UI | roadmap §0, §6 |

## 3. Decision 0 — foundation choices

### 3.1 System-modeling scope and validation fixtures

- **Binds:** Block 1 system representation and model construction; Block 2 engine inputs and reference cases; later sweep/search, control, and calibration inputs.
- **Agreed direction:** system modeling is a research capability used to construct and revise actual device models. Systems can contain circuit components and coupled quantum subsystems; their topology, parameters, interactions, drives, and dissipation must be representable. Named qubit types alone do not define the modeling scope. Validation fixtures exercise this capability without limiting the systems users may model.
- **Proposed initial coverage:** component-level circuit graphs (capacitors, inductors, Josephson junctions, connectivity, and biases) through the scqubits circuit adapter; composition of qubit, resonator, and coupler subsystem models with explicit interactions; the existing explicit-operator path for effective or custom models. Supported inputs produce Hamiltonians and operators with declared bases, units, truncations, and parameter provenance for the engines. Unsupported elements or representations require explicit diagnostics.
- **Proposed validation coverage:** transmon–resonator and fluxonium cases, a custom component-defined circuit, and a coupled multi-subsystem example. Choose specific fixtures and parameters to demonstrate the supported construction and composition paths; agreement with a reference alone does not establish accuracy for a physical device.
- **Decision:** modeling direction agreed; exact initial representation coverage and fixture set _pending_. The distinction between component-defined circuits and composed subsystem models follows scqubits' [Circuit](https://scqubits.readthedocs.io/en/latest/guide/circuit/ipynb/custom_circuit_define.html) and [interaction](https://scqubits.readthedocs.io/en/v4.3/api-doc/_autosummary/scqubits.core.hilbert_space.InteractionTerm.html) interfaces; concrete schema/API design follows the pinned scope.

### 3.2 Target local machine and environment policy

- **Binds:** supported development environment and where Block 2 GPU validation runs. Local execution through M2 is settled; a scheduler is not required for foundation.
- **Decision — agreed 2026-09-28:** use the owner's current Linux workstation with an NVIDIA RTX 5090 (owner-confirmed) as the initial verification environment. Support one documented local GPU environment and CPU-only installation/foundation CI. Select a scheduler when Block 9 is needed. GPU identity is supplied by the owner; sandbox access is not a prerequisite for selecting this target.
- Actual GPU/driver/toolkit compatibility and package versions are verified during foundation and Block 2, then recorded in the hashed environment manifest and `docs/environment.md`. Hardware observations do not block scaffolding; GPU agreement tests must pass before Block 2 completes.

### 3.3 Units, time grid, and truncation defaults

- **Binds:** system schemas, engine adapters, and later corrector thresholds.
- **Decision — agreed 2026-09-28:** user-facing energy/frequency parameters use the scqubits cyclic-frequency convention in GHz; time uses ns. Construct dynamics Hamiltonians with an explicit conversion to angular-frequency units, recorded in `[units]`; adapters must not apply a second implicit conversion. Physical component quantities retain explicit units and documented conversion to model parameters.
- **Output grid:** declare a uniform sampling grid; integrators may take adaptive/internal steps while reporting at the declared output times.
- **Truncation and convergence:** declare model/fixture-specific basis cutoffs and retained subsystem levels separately. Check the relevant observables as these are refined against declared tolerances. A +2-level increment is a possible refinement step, not a universal convergence rule; numerical defaults and refinement policies are justified with the selected fixtures.

### 3.4 License, audience, and sharing policy

- **Binds:** repository/license policy, permitted audience, and conditions for sharing.
- **Decision — agreed 2026-09-28:** private development by the owner and explicitly permitted collaborators. Defer distribution-license selection until before external distribution. Retain the verified-artifact gate in roadmap §5 for the Harmoniqs conversation and the private-content exclusions in `docs/README.md`; collaborator access does not authorize onward distribution.

## 4. Later decisions and implementation evidence

These do not block acceptance of this foundation ADR:

| When | Decision or evidence |
|---|---|
| Foundation / Block 2 | Verify environment compatibility and record actual GPU, driver, CUDA, cuQuantum, Python, and engine-package pins; GPU validation gates Block 2 completion |
| Before Block 5 | Exact dynamics metric/observable, fixed pulse if needed, parameter region, and evidence thresholds. Gate error or leakage under a fixed pulse is a candidate. Spectral studies remain supported but do not satisfy the first GPU dynamics demo |
| After M2, before Block 6a | First investigation's scope, assumptions, evidence requirements, measured-cost budget, review/stop criteria, and baseline evaluation protocol; explicit scripted selection policy and comparable allowances |
| Before Block 6a | V1 grant issuer/authentication, resource metering, reservation, and cancellation policy; contract §4 defines the resource units. Record the design and evaluation protocol in a follow-up ADR (planned ADR-0002), not a new ADR for every investigation |
| Before Block 6b | Concrete local process/filesystem/credential/GPU isolation mechanism and authenticated human channel, including enforcement against an agent with a shell |
| Before Block 7 | ProblemSpec OSS verbatim vs. an additive `solver.options` table |
| Before Block 9 | Scheduler target, remote environment, and executor deployment |

M3 acceptance and baseline-evaluation requirements are defined in roadmap Block 6b.

## 5. Consequences

The local target, unit/grid/truncation policy, and audience/license policy are agreed. Modeling must support actual component-based and coupled-subsystem research; a small fixture list cannot serve as its capability specification. Pin the initial representation coverage and fixture set, then reflect the resulting scope and accepted decisions in the roadmap, contract, and checklist before accepting this ADR. Acceptance unblocks Block 0; later decisions retain their own gates in roadmap §2 and §7.
