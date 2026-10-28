# ADR-0001: Mandate and Constraints

**Date:** 2026-09-28, revised 2026-10-28
**Status:** Accepted — Decision 0 closed on 2026-09-28. The agreed modeling mandate, local target, numerical conventions, license, and installation are recorded in §3; concrete model schemas, supported representation details, and fixtures are recorded in [ADR-0002](0002-system-model-representation.md).
**Purpose:** record the foundation choices needed before model construction and engine integration. Implementation starting points are distinguished from mandate decisions. Sweep metrics, investigation design, and deployment mechanisms are decided when their prerequisites are available (§4).

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
| Initial exploration | Runner investigations, then cooperative agent integration; progress-based continuation/stopping; calibration and cluster execution are independently scheduled | Accepted foundation mandate |
| Execution environments | Development and testing on the owner's workstation; research runs may also use a cluster through its scheduler. One coordinating agent, one active GPU job by default on the workstation; usage recorded, limits optional, progress checkpoints govern continuation; stronger authorization/isolation follows an explicit deployment need | Accepted foundation mandate; contract §4, §7 |
| First end-to-end sweep | Open-system dynamics on the local GPU; its metric is the readout example's [endpoint coherence](../../examples/readout/README.md#readout-and-purcell-cases) | Accepted foundation mandate |
| Posture | Stage-never-promote; supported tools enforce scope/review checks, configured limits, and run ownership, and run verification checks on request, under a cooperative trust model | contract §3–§7 |
| Out of scope | Gate-model algorithms, circuit compilation, hardware drivers, UI | Accepted foundation mandate |

## 3. Decision 0 — foundation choices

### 3.1 System-modeling scope and validation fixtures

- **Binds:** system representation and model construction; engine inputs and reference cases; later sweep/search, control, and calibration inputs.
- **Agreed direction:** system modeling is a research capability used to construct and revise actual device models. Systems can contain circuit components and coupled quantum subsystems; their topology, parameters, interactions, drives, and dissipation must be representable. Named qubit types alone do not define the modeling scope. Validation fixtures exercise this capability without limiting the systems users may model.
- **Decision — agreed 2026-09-28:** modeling supports component-defined circuits, coupled quantum subsystems, and explicit operators for effective/custom models. Outputs declare bases, units, truncations, and parameter provenance. Unsupported elements or representations produce explicit diagnostics. This mandate is sufficient for foundation; concrete schema/API choices and supported representation details are implementation work, recorded in [ADR-0002](0002-system-model-representation.md).
- **Implementation starting points:** scqubits' [Circuit](https://scqubits.readthedocs.io/en/latest/guide/circuit/ipynb/custom_circuit_define.html) interface provides component/connectivity/bias descriptions; its [interaction](https://scqubits.readthedocs.io/en/v4.3/api-doc/_autosummary/scqubits.core.hilbert_space.InteractionTerm.html) interface supports subsystem composition.
- **Validation policy:** select concrete fixtures and parameters with the model implementation to exercise each implemented construction and composition path. Transmon–resonator, fluxonium, a custom component-defined circuit, and a coupled multi-subsystem case are useful starting examples. Fixtures demonstrate declared coverage; reference agreement alone does not establish physical-device accuracy.

### 3.2 Target local machine and environment policy

- **Binds:** supported development environment and where GPU validation runs. Local execution is settled; a scheduler is not required for foundation.
- **Decision — agreed 2026-09-28:** use the owner's current Linux workstation with an NVIDIA RTX 5090 (owner-confirmed) as the initial verification environment. Support one documented local GPU environment and CPU-only installation; checks run on this workstation. Select a scheduler when cluster execution is needed. GPU identity is supplied by the owner; sandbox access is not a prerequisite for selecting this target.
- Actual GPU/driver/toolkit compatibility and package versions are verified with engine integration: compatible package versions are pinned in `uv.lock`, and GPU, driver, CUDA, cuQuantum, Python, and engine-package observations are recorded in `docs/environment.md`. The hashed run environment record is introduced with the runner. Hardware observations do not block scaffolding; GPU agreement tests must pass before engine integration completes.

### 3.3 Units, time grid, and truncation defaults

- **Binds:** system schemas, engine adapters, and later corrector thresholds.
- **Decision — agreed 2026-09-28:** user-facing energy/frequency parameters use the scqubits cyclic-frequency convention in GHz; time uses ns. Construct dynamics Hamiltonians with an explicit conversion to angular-frequency units, recorded in `[units]`; adapters must not apply a second implicit conversion. Physical component quantities retain explicit units and documented conversion to model parameters.
- **Output grid:** declare a uniform sampling grid; integrators may take adaptive/internal steps while reporting at the declared output times.
- **Truncation and convergence:** declare model/fixture-specific basis cutoffs and retained subsystem levels separately. Check the relevant observables as these are refined against declared tolerances. A +2-level increment is a possible refinement step, not a universal convergence rule; numerical defaults and refinement policies are justified with the selected fixtures.

### 3.4 License

- **Binds:** the terms under which Entanglix's code, documentation, and examples are distributed.
- **Decision — agreed 2026-10-27:** Entanglix is licensed under the Apache License 2.0. It is permissive, so private solver plugins can attach through entry points, and it carries an explicit patent grant.

### 3.5 Installation

- **Binds:** how users install Entanglix, add optional components, upgrade, and remove a version, and where its runtimes live.
- **Decision — agreed 2026-10-28:**
  - Entanglix supports Linux. One command installs it per user, without root: a bootstrap script pinned to a release tag clones that tag into a directory for its version, with its own pinned uv and Python and a base job environment, and installs a launcher in the user's local `bin` directory. It edits no shell profile; when that directory is not on `PATH`, it says so and shows the full launcher path.
  - The base installation serves CPU work, including the open control baseline. GPU engines and the Piccolo solver are components the user adds on request; the installation records which it has.
  - Each version keeps its own directory. All versions share one uv cache, Entanglix's own, beside them. The launcher runs the version a project records, otherwise the newest complete installation; project setup always runs on the newest. Installs and removals run one at a time. A version is removed only while none of its processes, launches, or jobs is running. Removing the last version also removes the cache and the launcher; removal deletes only Entanglix-owned paths.
  - An installation's runtimes and state live in one directory that Git ignores, so its runs record clean code.
  - macOS, Windows, and a package-index distribution are not offered.
- **Rationale:**
  - A user-level install needs no administrator. Runs record Git provenance, so the installation is a clone.
  - A clone updated in place would change code under running jobs and under projects set up with it. Separate versions keep both on their code, and a shared cache avoids downloading again and, where uv can hardlink, storing again what versions share. The environments are hardlinks into the cache, so only a cache of Entanglix's own lets removal free the space, and it leaves the user's uv cache alone.
  - The GPU set is large, and a detected GPU may serve only a display, so it is added on request rather than by detection.
  - A version removed under a running process or job would take its code away. Concurrent installs and removals would work on the same directories, and running them in parallel gains little on one workstation.

## 4. Later decisions and implementation evidence

These do not block acceptance of this foundation ADR:

| Decision or evidence | Status |
|---|---|
| Concrete model schemas/APIs, supported representation details, and fixtures/parameters exercising component, composition, and explicit-operator paths; these do not reopen the foundation mandate | Decided in [ADR-0002](0002-system-model-representation.md) |
| Environment compatibility and actual GPU, driver, CUDA, cuQuantum, Python, and engine-package pins | Recorded in the [GPU environment](../environment.md) and `uv.lock` |
| Exact dynamics metric/observable, parameter region, and evidence thresholds for the first GPU dynamics sweep | Recorded with the [readout example](../../examples/readout/README.md) |
| First investigation's scope, assumptions, evidence requirements, progress checkpoints, review/stop criteria, and any requested limits; a baseline evaluation with a scripted selection policy and comparable allowances, including an early-stop case | Decided in [ADR-0004](0004-investigations-and-agent-sessions.md) |
| What the harness contract leaves open: how user decisions are recorded, progress-assessment fields, checkpoints, resume behaviour, and the evaluation protocol; ordinary investigations remain research records | Decided in [ADR-0004](0004-investigations-and-agent-sessions.md) |
| Runtime integration and research-agent instruction delivery; user-input event handling, progress reporting, runtime permission settings, and cooperative execution checks; no separate grant issuer or OS-enforced security boundary | Decided in [ADR-0004 §2.8](0004-investigations-and-agent-sessions.md#28-agent-sessions) |
| ProblemSpec OSS verbatim vs. an additive `solver.options` table | Decided in [ADR-0005](0005-control-problems-and-solvers.md) |
| Scheduler target, remote environment, and site execution requirements; the executor reuses the scheduler's time limits, GPU allocation, and accounting and is tested on the workstation against a stubbed scheduler | Deferred until cluster execution is needed |
| Explicit threat model and identity/process/filesystem/credential/network/device controls, reusing suitable platform facilities; bypass-resistance evidence is separate from cooperative conformance | Deferred until stronger deployment guarantees are needed |

Agent-integration acceptance requires a declared sweep/search investigation with logged tools, evidence-driven refinement, supported reporting or review handoff, and a stagnation case that stops with useful results and next experiments. Compare scientific outcome, cost, human intervention, and continuation/stop evidence against a scripted baseline. Correct execution, declared evidence, local reliability, independent usability, and completion of the comparison are required; outperforming the baseline is not. Acceleration claims require measured comparative benefit.

## 5. Consequences

Decision 0 is closed. Modeling supports actual component-based and coupled-subsystem research, with an explicit-operator path; a fixture list does not limit that mandate. Concrete representation and validation choices are recorded in [ADR-0002](0002-system-model-representation.md). The local target, unit/grid/truncation policy, license, and installation are accepted; later decisions retain the capability-specific prerequisites recorded in §4.
