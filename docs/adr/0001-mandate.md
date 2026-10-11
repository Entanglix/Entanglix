# ADR-0001: Mandate and Constraints

**Date:** 2026-09-28, revised 2026-10-10
**Status:** Accepted — Decision 0 closed on 2026-09-28. The agreed modeling mandate, local target, numerical conventions, and audience/license policy are recorded in §3; concrete model schemas, supported representation details, and fixtures follow during Block 1.
**Purpose:** record the foundation choices needed by Blocks 0–2. Implementation starting points are distinguished from mandate decisions. Sweep metrics, investigation design, and deployment mechanisms are decided when their prerequisites are available (§4).

---

## 1. Context

Entanglix supports expert-guided superconducting-circuit research through independently useful modeling, simulation, sweep/search, control, and calibration workflows. Agents help formulate bounded investigations; a complete device-design objective is optional. The substrate is Python-first: scqubits, cuDensityMat via `qutip-cuquantum`, dynamiqs, and a QuTiP CPU reference. A commodity agent runtime uses skills and MCP; the runner enforces structured scope/review checks, configured execution limits, and verification through supported interfaces. There is no product UI.

## 2. Already decided (recorded here, not reopened)

| Decision | Value | Where |
|---|---|---|
| Name and command | Entanglix; `entanglix` | Accepted foundation mandate |
| Runner language | Python, `uv` | Accepted foundation mandate |
| Substrate | scqubits · cuDensityMat (`qutip-cuquantum`) · dynamiqs · QuTiP CPU reference | Accepted foundation mandate |
| Two-engine rule | cuDensityMat and JAX do not compose through autodiff; engine chosen by a versioned table | contract §2 |
| Control spec | An Entanglix `control` spec, from which the runner generates one amicode ProblemSpec OSS file shared by every solver; the control spec names the solver and the result records it | contract §1, [ADR-0005](0005-control-problems-and-solvers.md) |
| Private solvers | Entry-point plugins; repo code never imports them directly; verification remains open | contract §1, §3 |
| amicode's role | Contracts and an open second-opinion solver (Piccolo, run in Julia through its ProblemSpec runner); never the dynamics substrate | Accepted foundation mandate; [ADR-0005](0005-control-problems-and-solvers.md) |
| Research direction | Expert-guided formulation, bounded autonomy, independent entry points, professional quality | Accepted foundation mandate |
| Preferred sequence | Local sweep with checks → bounded exploration → control → co-design; technical dependencies govern availability | Accepted foundation mandate |
| Initial exploration | Runner investigations (6a), then cooperative agent integration (6b); progress-based continuation/stopping; calibration and cluster execution are independently scheduled | Accepted foundation mandate |
| Execution environments | Development and testing on the owner's workstation; research runs may also use a cluster through its scheduler. One coordinating agent, one active GPU job by default on the workstation; usage recorded, limits optional, progress checkpoints govern continuation; stronger authorization/isolation follows an explicit deployment need | Accepted foundation mandate; contract §4, §7 |
| First end-to-end sweep | Open-system dynamics on the local GPU; exact metric remains open | Accepted foundation mandate |
| Posture | Stage-never-promote; supported tools enforce scope/review checks, configured limits, and run ownership, and run verification checks on request, under a cooperative trust model | contract §3–§7 |
| Out of scope | Gate-model algorithms, circuit compilation, hardware drivers, UI | Accepted foundation mandate |

## 3. Decision 0 — foundation choices

### 3.1 System-modeling scope and validation fixtures

- **Binds:** Block 1 system representation and model construction; Block 2 engine inputs and reference cases; later sweep/search, control, and calibration inputs.
- **Agreed direction:** system modeling is a research capability used to construct and revise actual device models. Systems can contain circuit components and coupled quantum subsystems; their topology, parameters, interactions, drives, and dissipation must be representable. Named qubit types alone do not define the modeling scope. Validation fixtures exercise this capability without limiting the systems users may model.
- **Decision — agreed 2026-09-28:** modeling supports component-defined circuits, coupled quantum subsystems, and explicit operators for effective/custom models. Outputs declare bases, units, truncations, and parameter provenance. Unsupported elements or representations produce explicit diagnostics. This mandate is sufficient for foundation; concrete schema/API choices and supported representation details are Block 1 implementation work.
- **Implementation starting points:** scqubits' [Circuit](https://scqubits.readthedocs.io/en/latest/guide/circuit/ipynb/custom_circuit_define.html) interface provides component/connectivity/bias descriptions; its [interaction](https://scqubits.readthedocs.io/en/v4.3/api-doc/_autosummary/scqubits.core.hilbert_space.InteractionTerm.html) interface supports subsystem composition.
- **Validation policy:** select concrete fixtures and parameters during Block 1 to exercise each implemented construction and composition path. Transmon–resonator, fluxonium, a custom component-defined circuit, and a coupled multi-subsystem case are useful starting examples. Fixtures demonstrate declared coverage; reference agreement alone does not establish physical-device accuracy.

### 3.2 Target local machine and environment policy

- **Binds:** supported development environment and where Block 2 GPU validation runs. Local execution through M2 is settled; a scheduler is not required for foundation.
- **Decision — agreed 2026-09-28:** use the owner's current Linux workstation with an NVIDIA RTX 5090 (owner-confirmed) as the initial verification environment. Support one documented local GPU environment and CPU-only installation/foundation CI. Select a scheduler when Block 9 is needed. GPU identity is supplied by the owner; sandbox access is not a prerequisite for selecting this target.
- Actual GPU/driver/toolkit compatibility and package versions are verified during Block 2: compatible package versions are pinned in `uv.lock`, and GPU, driver, CUDA, cuQuantum, Python, and engine-package observations are recorded in `docs/environment.md`. The hashed run environment record is introduced with the runner in Block 3. Hardware observations do not block scaffolding; GPU agreement tests must pass before Block 2 completes.

### 3.3 Units, time grid, and truncation defaults

- **Binds:** system schemas, engine adapters, and later corrector thresholds.
- **Decision — agreed 2026-09-28:** user-facing energy/frequency parameters use the scqubits cyclic-frequency convention in GHz; time uses ns. Construct dynamics Hamiltonians with an explicit conversion to angular-frequency units, recorded in `[units]`; adapters must not apply a second implicit conversion. Physical component quantities retain explicit units and documented conversion to model parameters.
- **Output grid:** declare a uniform sampling grid; integrators may take adaptive/internal steps while reporting at the declared output times.
- **Truncation and convergence:** declare model/fixture-specific basis cutoffs and retained subsystem levels separately. Check the relevant observables as these are refined against declared tolerances. A +2-level increment is a possible refinement step, not a universal convergence rule; numerical defaults and refinement policies are justified with the selected fixtures.

### 3.4 License, audience, and sharing policy

- **Binds:** repository/license policy, permitted audience, and conditions for sharing.
- **Decision — agreed 2026-09-28:** private development by the owner and explicitly permitted collaborators. Defer distribution-license selection until before external distribution. The Harmoniqs conversation requires a shared ProblemSpec solved by the open baseline and Piccolo, each candidate independently checked, with inspectable run directories and corrector reports. Agent integration is not a prerequisite. Before sharing, remove private appendices and exclude files marked `*-private.md`; public text uses "private solver plugins" without product names. Collaborator access does not authorize onward distribution.

## 4. Later decisions and implementation evidence

These do not block acceptance of this foundation ADR:

| When | Decision or evidence |
|---|---|
| During Block 1 | Concrete model schemas/APIs, supported representation details, and fixtures/parameters exercising component, composition, and explicit-operator paths; these do not reopen the foundation mandate |
| Foundation / Block 2 | Verify environment compatibility and record actual GPU, driver, CUDA, cuQuantum, Python, and engine-package pins; GPU validation gates Block 2 completion |
| Before Block 5 | Exact dynamics metric/observable, fixed pulse if needed, parameter region, and evidence thresholds. Gate error or leakage under a fixed pulse is a candidate. Spectral studies remain supported but do not satisfy the first GPU dynamics demo |
| After M2, before Block 6a | First investigation's scope, assumptions, evidence requirements, progress checkpoints, review/stop criteria, and any requested limits; baseline evaluation defines a scripted selection policy and comparable benchmark allowances, including an early-stop case |
| Before Block 6a | What the harness contract leaves open: how user decisions are recorded, progress-assessment fields, default checkpoint intervals, and resume behaviour. Record these and the evaluation protocol in a follow-up ADR; ordinary investigations remain research records |
| Before Block 6b | Runtime integration and research-agent instruction delivery (MCP instructions, skills, or a generated research-workspace instruction file); user-input event handling, progress reporting, runtime permission settings (human-action commands blocked, GPU hidden from the agent session), and cooperative execution checks; no separate grant issuer or OS-enforced security boundary required |
| Before Block 7 | ProblemSpec OSS verbatim vs. an additive `solver.options` table; recorded in [ADR-0005](0005-control-problems-and-solvers.md) |
| Before Block 9 | Scheduler target, remote environment, and site execution requirements; the executor reuses the scheduler's time limits, GPU allocation, and accounting and is tested on the workstation against a stubbed scheduler |
| When stronger deployment guarantees are needed | Explicit threat model and identity/process/filesystem/credential/network/device controls, reusing suitable platform facilities; bypass-resistance evidence is separate from cooperative conformance |

Agent-integration acceptance requires a declared sweep/search investigation with logged tools, evidence-driven refinement, supported reporting or review handoff, and a stagnation case that stops with useful results and next experiments. Compare scientific outcome, cost, human intervention, and continuation/stop evidence against a scripted baseline. Correct execution, declared evidence, local reliability, independent usability, and completion of the comparison are required; outperforming the baseline is not. Acceleration claims require measured comparative benefit.

## 5. Consequences

Decision 0 is closed and Block 0 is unblocked. Modeling supports actual component-based and coupled-subsystem research, with an explicit-operator path; a fixture list does not limit that mandate. Concrete representation and validation choices are made during Block 1. The local target, unit/grid/truncation policy, and audience/license policy are accepted; later decisions retain the capability-specific prerequisites recorded in §4.
