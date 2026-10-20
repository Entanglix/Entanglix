# ADR-0006: Device–Control Co-design

**Date:** 2026-10-20
**Status:** Accepted
**Purpose:** fix how a co-design investigation composes the existing operations, how device candidates and seeds vary within one approved base, which frequency and bound quantities are derived from the model, how a returned pulse is evaluated under declared uncertainty, how lineage is carried, how candidates are compared, how the solve budget is declared, and what the first version leaves out. Field lists, formats, and command names are documented with the implementation.

---

## 1. Context

The [harness contract](../../HARNESS_CONTRACT.md) has co-design link device/model, control, and evaluation runs with shared review state, usage accounting, and any configured limits, and declare constraints, uncertainty, and comparable evaluation conditions (§4). [ADR-0004](0004-investigations-and-agent-sessions.md) fixes investigations, admission against approved bases, and the user's own agent runtime. [ADR-0005](0005-control-problems-and-solvers.md) fixes the control spec, the generated ProblemSpec, the solvers, and the control checks.

Before this record:

- A control input has no parameter overrides, so each device candidate needs its own system file and its own exact base. Investigation variations exist for simulate and sweep targets only, so a candidate chosen from evidence needs a new revision and a user approval.
- Frame and carrier frequencies are declared numbers. Moving the frame and carriers together is accepted, with any detuning left in the drift; a carrier off the declared frame leaves a zero projected drive and is refused by name.
- Amplitude bounds are stated in the drive operator's own scale. The shared ProblemSpec holds one amplitude bound per channel, and a single scalar for each derivative bound.
- Control checks evaluate the returned pulse on the nominal closed model only. Offset charges are literal model values, not named parameters.
- Control lineage exists in Python and on the CLI; the MCP job tools do not take parent runs.

The [co-design probe](../../tests/probes/m5_codesign.py) exercised these operations on one transmon, with candidates differing in shunt capacitance. The harness does only what the user requests.

## 2. Decisions

### 2.1 Co-design composes existing operations

- Co-design is an investigation that uses the existing operations: simulate or sweep to screen devices, control to solve, and requested checks and diagnostics to evaluate.
- There is no co-design protocol, no device optimiser, no generic expression system, and no sub-investigation. The user or the agent proposes each next device candidate.
- Expert checkpoints, review points, usage accounting, and limits are the investigation's own.
- One agent coordinates, in the user's own runtime. Job tools return on admission, so independent solves run as concurrent jobs. The harness neither starts nor requires subagents, and stays correct when several callers use it at once.

**Rationale:** the existing operations compose under one investigation; what co-design lacked was candidate inputs, evaluation under uncertainty, and lineage through the tools, not a protocol. Device and control choices are coupled, so one coordinator keeps them together, and concurrency comes from the job tools rather than from more agents.

### 2.2 Device parameters and seeds vary within one base

- A control input may declare a `[parameters]` table in the form of simulate's, with the same freezing: the run's frozen system holds the values actually used.
- A control base may vary `parameters.<name>`, by range or values, and `pulse.seed`, by values. Solver settings and every other input stay exact.
- Each control run records the seed it used.
- An allowed seed set alone does not make effort comparable. The investigation's evidence requirements declare which seeds each candidate requires and how any additional attempts are allocated. These are recorded terms; the harness does not enforce them.

**Rationale:** one approved base then covers a device region and a seed set, and the agent chooses candidates and seeds inside it without further approvals; admission keeps its existing scope check. Solver results depend on the initial guess, so a single seed can misstate a candidate. Solver settings stay fixed because varying them within a base breaks comparability.

### 2.3 One derived quantity: a named transition

- A control input may declare its frame frequency as the transition between two named levels of a subsystem's isolated eigenstates. The frequency is computed at launch from the constructed model, with overrides applied, and recorded in the frozen inputs. Channels may follow the frame.
- The amplitude bound and both derivative bounds may be declared as a fixed nominal Rabi-rate bound on that transition, with the computed raw bounds recorded.
  - The normalisation is the named transition's matrix element of the authored drive operator, under the existing cosine-carrier convention, not the projected half-amplitude operator.
  - A zero matrix element is rejected.
  - Derived derivative bounds are allowed only when they resolve to one common raw value across the channels, equal within a relative tolerance fixed with the implementation. Otherwise the launch is refused, naming each channel's value, and the user declares explicit raw derivative bounds.
- Explicit numbers remain valid. Nothing else is derived, and there are no expressions.

**Rationale:** a range variation cannot carry linked values, and a carrier off the frame is refused, so without derivation a value inside an approved range could not be driven on resonance. A Rabi-rate bound keeps drive strength comparable across candidates whose matrix elements differ. The shared ProblemSpec holds a single value per derivative bound, so a derived derivative bound is used only where one value serves every channel; refusing otherwise keeps the shared format and every solver unchanged. A single named quantity keeps admission mechanically checkable.

### 2.4 Uncertainty evaluation is a requested diagnostic

- A control input may declare an evaluation table. It is part of the approved base, so every candidate in the base has the same conditions, and it is not part of the generated ProblemSpec.
- The table lists samples over `parameters.<name>` and over circuit offset charges. There is no general model-path editor.
- Each sampled quantity is marked drifting, meaning the drive frequencies are not recalibrated for it, or calibrated, meaning they follow it.
- A requested diagnostic evaluates the returned pulse on each sample and reports per-sample fidelity and leakage and the worst sampled fidelity, without a pass or fail outcome.
- The drive for a sample is resolved in three steps:
  1. A calibration model applies only the sample's calibrated changes.
  2. The named transition and the carriers that follow it are resolved on that model; with no calibrated changes, this is the nominal frame.
  3. The actual model, with every sampled change applied, is evaluated at those frequencies.
- Calibrated samples need the control run to declare a named transition, with the affected carriers following it. Without one, a table with calibrated samples makes the diagnostic inapplicable, with that reason, and no partial result is reported. Control runs with explicit frequencies stay valid, and tables with only drifting samples are evaluated at the declared frequencies.
- Fixed conventions:
  - The returned amplitudes, phases, and knots never change, and nothing is renormalised.
  - Logical states are each sampled model's own eigenstates, phase-aligned to the nominal ones by overlap. Ambiguous correspondence is reported; unsupported correspondence makes the diagnostic inapplicable, with a reason.
  - The figure of merit is the replay check's leakage-aware fidelity, on the replay engine.
  - Results are described as closed-system evaluations under the rotating-wave approximation.
- Refinements of retained levels and cutoffs are declared in the same table and run only when requested. Each requested refinement re-evaluates the whole sample set, keeping each sample's resolved drive, and recomputes the sampled minimum. The report states coverage and distinguishes per-sample changes from the change in the minimum.

**Rationale:** the pulse exists only after the solve, so an authored evaluation would need a new approved base for every pulse; a diagnostic is linked to its source run and needs none. Declaring the samples in the base keeps the conditions identical across candidates. Drifting and calibrated quantities are physically different: holding the nominal frame under a fabrication offset models a device that was never calibrated. A sample's own eigenstates model quasistatic variation, with preparation in the device's actual levels. Another sample can become the worst after refinement, so refinement covers the whole set. Like the existing diagnostics, the result is a quantity, not a verdict.

### 2.5 Lineage

- A control run's frozen parameter overrides and derived transition identify its device candidate.
- A screening sweep or simulation links to a control run at run level through `parent_run_ids`.
- Every MCP job tool accepts `parent_run_ids`, as the CLI and Python already do.
- There is no sweep-point lineage field.

**Rationale:** with §2.2 a candidate need not come from a sweep point. Binding points would need a new field and an exception to the rule that a control input's system sits beside it.

### 2.6 Comparison

- There is no new report view. The investigation report's attempt table shows each attempt's varied values and derived transition. The user or the agent records a comparison as a finding linked to its runs and checks.
- Comparison guidance is part of the operational reference. It is guidance for findings; the harness does not certify it:
  - Measured performance of the returned pulses under the declared search policy is distinct from a device's attainable performance.
  - An independently checked admissible pulse establishes a lower bound on what its device attains, within the declared model and the reported numerical evidence. It is not a hardware guarantee. Finite search establishes no upper bound.
  - A comparison considers every independently checked pulse known for each candidate, including pulses from other bound policies and other solvers. Each must satisfy the comparison's constraints and be evaluated, on the compared quantity, under the comparison's declared conditions.
  - Seed spread and refinement change are reported as evidence, not as thresholds that resolve a comparison.
  - Spreads stay per solver.
  - A finding states how the selected pulse and the ranked quantity were chosen.
  - A comparison the evidence does not support is reported as unresolved, which is a valid endpoint.

**Rationale:** a search that did not find a better pulse does not show that none exists. A small spread can mean every start found the same poor solution, so a threshold on spread or refinement change can accept a ranking that a known admissible pulse disproves. Solvers stop by different rules on different objectives ([ADR-0005 §2.4](0005-control-problems-and-solvers.md#24-solvers)), so their spreads are not pooled.

### 2.7 Budget policy

- Solver settings and the allowed seed set are declared in the approved base. The required seeds and the allocation of additional attempts are declared in the evidence requirements.
- A comparison reports differing termination, such as a run that reached its iteration cap.
- The harness never adds seeds or solves.

**Rationale:** termination differs by solver and is shown, not normalised away. The harness does only what the user requests.

### 2.8 Scope of the first version

- Evaluation is closed-system. Open-system evaluation is a later addition: it needs a device-dependent relaxation model, not only an operator convention.
- Uncertainty evaluation follows logical states by correspondence for a single periodic circuit coordinate. For other models it is inapplicable, with its reason, and those models can still be modelled, solved and checked. General state correspondence, dressed-transition calibration and lab-frame validation of the control model are later additions, each scheduled with the workflow that needs it.
- The `target` check remains available and unchanged; the first co-design example does not request it.

**Rationale:** an assumed common relaxation rate shows that open-system evaluation is feasible, but a comparison between devices needs relaxation that depends on the device. The example compares performance; a target would turn that comparison into a pass or fail. General correspondence needs a declared logical encoding, physical overlaps across bases, and subspace agreement near degeneracy. That is a design of its own, not an extension of the single-coordinate case.

Measurements: [co-design probe results](../../tests/probes/m5_codesign_results.json), [solver propagator results](../../tests/probes/grape_propagator_results.json), [Piccolo precompile results](../../tests/probes/piccolo_precompile_results.json).

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| A co-design protocol or a built-in device optimiser | The existing operations compose; the user or the agent proposes candidates, as in sweep/search |
| Agents split by protocol | Cuts the coupling between device and control decisions and multiplies runtime token cost |
| One exact base per candidate | Works, but puts a user decision on every adaptive step |
| Varying solver settings within a base | Breaks comparability within the base |
| Linked values only, without a derived transition | Serve listed values, not ranges or adaptive choices |
| A general derivation or expression language | One named quantity is enough, and admission stays mechanically checkable |
| The smallest of the channels' derivative bounds | Silently tightens the other channels |
| An authored evaluation job | Needs a new approved base for every pulse |
| Samples supplied when the diagnostic is requested | Conditions would differ between candidates |
| Probability distributions, or a worst case over a continuous region | Declared samples support only sampled claims |
| A pass or fail threshold on the uncertainty diagnostic | Would be a fidelity target again |
| A sweep-point lineage field, or control reading a sweep point's frozen system | Needs a new field and a path exception; candidates need not come from points |
| Threshold-based resolution of a comparison | Can accept a ranking that a known admissible pulse disproves |
| A dedicated comparison view | The report and a finding cover the first workflow |
| A machine verdict on rankings | Checks report their own scope, with no run-level verdict |
| Automatic multi-start | The harness adds no solves |
| Open-system evaluation in the first version | Needs a device-dependent relaxation model |

## 4. Consequences

- The contract's §3 `control` row adds the uncertainty diagnostic: on request, the returned pulse evaluated on the declared samples, reporting per-sample fidelity and leakage and the worst sampled fidelity without a pass or fail outcome.
- The roadmap and checklist replace M5's "verified" wording with the checks requested and their scope reported.
- The control reference documents parameter overrides, the named transition, Rabi-rate bounds, the evaluation table, the diagnostic, and the comparison guidance. The investigation reference documents control variations; the MCP reference documents `parent_run_ids`.
- The co-design example is recorded with its own directory.
- Tests cover: parameter overrides frozen in a control run; control variations over parameters and seeds admitted, and an out-of-region request refused; a derived transition recorded and matched against the model; a Rabi-rate bound that reproduces explicitly normalised bounds; a zero matrix element rejected; differing derived derivative bounds refused by channel; each diagnostic sample mode, including a combined sample recalibrated for its calibrated change only; calibrated samples without a named transition reported as inapplicable; requested refinement over the whole sample set, with coverage reported; and `parent_run_ids` through every MCP job tool, with CLI parity.
