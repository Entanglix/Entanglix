# ADR-0004: Investigations, User Decisions, and Agent Sessions

**Date:** 2026-10-04
**Status:** Accepted
**Purpose:** fix how an investigation is recorded, who may change its terms, how its jobs are admitted and stopped, how progress is assessed and work resumed, how the first agent-assisted investigation is evaluated, and whose runtime drives the harness. Field lists, formats, and command names are documented with the implementation.

---

## 1. Context

The [harness contract](../../HARNESS_CONTRACT.md) fixes the investigation lifecycle, usage recording, and optional limits (§4), separates research tools from human actions (§5), and requires an append-only history (§6). It leaves open how user decisions are recorded, what a progress assessment contains, how checkpoints are set, and how work resumes. [ADR-0001](0001-mandate.md) assigns those points and the evaluation protocol to this record, and keeps ordinary investigation specifications as research records.

[ADR-0003](0003-runs-verification-and-dispatch.md) made run directories the attempt history and checks a separate request. Two of its properties matter here. An attempt exists from its manifest, but `abort_run` acts only on an attempt with a start record, and the timeout clock starts at that record. A sweep is one job that evaluates many points.

The [readout threshold probe](../../tests/probes/m3_readout.py) ran the first investigation's plan with the existing operations: 52 jobs and 301 s of job time for 41 evaluated points. A checkpoint after every computation and its check run would have required 26 assessments. A lab-frame run cost six times a rotating-frame run. The refined initial state differed between parameter values, because it is embedded in a basis that depends on them.

The harness does only what the user requests. For an investigation it supplies no default checkpoint and no default resource limit.

## 2. Decisions

### 2.1 Investigation records

- An investigation is a directory of runner-written files under the runs root, in a reserved location that cannot be a system ID. It holds immutable revisions and one immutable file per event: user decisions, findings, and stops. Events are published atomically.
- A run launched for an investigation records the investigation and its revision in its manifest.
- There is no separate store or index. Listing and reading an investigation derive its state from these files and the linked runs.

**Rationale:** unique, immutable files give atomic events without a ledger service, as they do for runs. State derived from records is the same in every session.

### 2.2 Revision authority and user decisions

- A revision holds the agreed terms: the question, assumptions, scope fields, evidence requirements, checkpoint, review points, review and stop criteria, and limits. Every field belongs to the user.
- A research tool drafts an investigation or proposes a revision. A revision takes effect only on a user decision.
- A user decision is an event recorded by one operator command. Its actions are start, resume, approve a proposed revision, and release a named review. The event records the action, the affected revision, the time, and the source. No research tool exposes the command, and a research request carrying an approval claim is rejected.
- A limit is a field of a revision, so a limit change is an approved revision.
- Findings need no approval, and choosing experiments within the scope needs no revision.
- The runner enforces who may change the terms. It does not judge what they say.

**Rationale:** a rule that let recorded text change without approval would let an agent remove an evidence requirement while leaving every checked field untouched. Ownership of the whole revision needs no classification of its fields. An agent's working plan and interpretation have their own record, the finding.

### 2.3 Scope and admission

- The runner checks structured scope fields only, in the sweep target syntax:
  - the permitted operations;
  - the permitted base specs, each identified by the digests of its frozen inputs;
  - for each varied target, a value set or a closed range, with linked values where targets move together.
- Each operation is admitted by its own rule:
  - A simulation or sweep is in scope when its frozen inputs match a permitted base apart from the varied targets, and every varied value at every point lies inside its set or range.
  - A check belongs to its source run's investigation. Its check names and any replacement refinement are validated as for a standalone check.
  - A rerun keeps the earlier run's membership and passes admission under the current revision.
  - An authored script is in scope when the script, its data, and its system spec match a permitted base exactly.
- An input derived from a parameter value, such as an initial state embedded in a refined basis, is bound by a base of its own for that value.
- A research procedure that these fields cannot express is part of the agreed terms. The client or agent follows it, and the runner does not enforce it.
- Admission is atomic under one lock per investigation. Decisions, revisions, stops, and findings that open a review take the same lock. Admission requires a started revision, a permitted operation within scope, no stop, no open review, no checkpoint due, and room under every limit, counting recorded usage, admitted attempts without a terminal record, and the proposed job. The admitted run's manifest is the commitment record.
- An admitted attempt keeps the revision and limits under which it was admitted, including its deadline. An approved revision governs later admissions only. To withdraw permission from outstanding attempts, the user stops the investigation, approves the revision, and resumes.
- Immediately before a job computes, the runner takes the lock again. If a stop event names the run, or its deadline has passed, the run ends as aborted without computing. Otherwise the start record is published before the lock is released.
- The supervisor carries the absolute deadline into execution and ends the job at expiry, whatever its timeout.

**Rationale:**
- Bases and ranges describe a permitted region with syntax that sweeps already validate. A base per value binds a derived file with existing digests, so no mechanism for derived inputs is needed.
- The runner cannot infer a script's physical parameters from its system spec, so a script is admitted only as an exact bundle.
- The base-and-range fields permit a lab-frame confirmation and a lab-frame search at the same amplitudes. A rule separating them would be a workflow engine.
- An attempt is admitted before it starts. Without the second check, a stop would miss it, and a timeout fixed at admission would let setup time carry the job past the deadline.
- Applying an approved revision to attempts already admitted would mean reconfiguring jobs in flight. A stop already cancels outstanding attempts, so stop, revise, and resume withdraws permission with what exists.

### 2.4 Checkpoints, assessments, and reviews

- The harness supplies no default checkpoint and no default limit. The user declares the checkpoint when starting an investigation, as a count of finished attempts or an elapsed interval, and declares a limit only when one is wanted.
- Every finished run linked to the investigation counts toward a checkpoint, including verification runs and failed or aborted attempts. An interrupted attempt counts once it is reconciled. When a checkpoint is due, admission is refused until an assessment is recorded.
- One finding record serves assessments, interpretations, and reports.
  - The author supplies the runs considered, what was learned, what remains unresolved, a decision to continue, refine, stop, or ask for review, its rationale, and the next experiment.
  - The runner fills the revision, the usage since the previous assessment and in total, and, for each linked run, the checks run with their outcomes and the relevant checks not run. Usage counts jobs, evaluated points, wall seconds, and GPU-seconds.
  - The runner validates that the fields are present and that the referenced runs belong to the investigation. It does not judge the rationale.
- A review opens from a finding that asks for one, or from a review point the user declared at a finished-attempt count. A declared review point is optional and absent unless the user requests it. An open review refuses admission for the whole investigation until a user decision releases it by name. Runs already admitted finish. A proposed revision does not stop work within the current scope.
- Requested checks are part of the evidence requirements, and a standing request among them is the user's authorization (ADR-0003 §2.3). The client or agent still issues each check as its own job, and the runner reports which runs still lack a requested check.

**Rationale:**
- The right interval depends on the investigation, and many investigations have no predictable job count. A default would be wrong for most of them, so the user states it.
- Usage and check status filled from records cannot be misreported. The author's part is the judgment.
- A declared review point gives the user a pause that does not depend on the agent's judgment, using the counter the checkpoint already keeps.
- A check is a job that counts toward the checkpoint and any limit. Issued by the client, it is scheduled and admitted like every other job; launched by the runner, its timing and admission would be implicit.

### 2.5 Stop and resume

- Stopping an investigation closes admission, aborts running jobs, and records one reason from the contract's list. The stop event names every attempt outstanding at that moment, running or admitted but not yet computing. Their cancellation is permanent.
- An investigation's state comes from its records, not from a session. A new session continues an active investigation without a user action.
- A stopped investigation reopens only through a resume decision, after its open reviews are released. Resume permits new admissions and does not revive a cancelled attempt.
- Completed results are not recomputed because a session changed. An interrupted attempt is reconciled and rerun under a new linked ID, and the rerun counts as a new attempt.
- Usage carries across sessions, and uncertain usage stays counted at its commitment.
- M3 adds no restart files; `simulate` and `sweep` have none.

**Rationale:** a check of the investigation's current state alone would let an attempt admitted before a stop start after a later resume. Naming the cancelled attempts in the stop event makes the outcome independent of what follows.

### 2.6 Evaluation protocol

- The evaluation compares a documented scripted selection policy with the agent. Both receive the same revision: scope, requested checks, checkpoint, and a job-count allowance the user declares. They run in separate investigations, and each runs all of its own jobs.
- The measures are the outcome against the declared claim rule and the requested checks, elapsed time, job time for computation and for checks, jobs and evaluated points, time outside jobs, user decisions after the start, and the rationale recorded at each assessment and at the stop.
- The evaluation includes an early-stop case, in which the permitted method cannot establish the requested claim. Its supported outcome is a handoff that reports the best confirmed result and the next experiment. The first failed confirmation opens a review, and the remaining requests are reported as unattempted.
- The agent runs each case more than once, and every run is reported, including a worse or inconclusive one.
- The baseline is the scripted policy. Bayesian search is built when a task suits it.
- The first investigation is the readout threshold characterization. Its question, scope, numbers, and declared allowance are a research record in the readout example.

**Rationale:**
- A sweep evaluates many points in one job, so points are counted as well as jobs.
- An agent varies between runs, so one run is not a result.
- The early-stop case shows a client recognizing a limit of its permitted method and stopping with the evidence it has. It does not show that further work would be uneconomic.
- On a smooth one-dimensional crossing a midpoint policy is close to optimal, so the expected result is the same outcome at a higher elapsed time. ADR-0001 requires the comparison, not an advantage.

Measurements: [readout threshold probe](../../tests/probes/m3_readout_results.json).

### 2.7 Agent runtime

- The runtime is the user's own. The harness holds no model backend and no API key, and it does not limit the runtime: any runtime that speaks MCP over stdio can drive it.
- Each session's audit records the runtime's name and version.
- CI tests the harness side with a scripted MCP client and needs no runtime. Checks that need a session run on the workstation in the user's runtime and are retained.

**Rationale:** the harness provides tools, records, and procedures; the model and its runtime are the user's choice, as the solver is. CI has no logged-in user, and a runtime the harness does not own is not the harness's to pin.

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| A separate investigation store, database, or index | Duplicates the runner-written records; state derives from files, as for runs |
| Recorded-text revisions effective without approval | An agent could weaken an evidence or stop criterion without changing a checked field |
| Approval through research-tool fields or agent-authored notes | Research tools cannot create user decisions (contract §5) |
| A separate command or event type for limit changes | A limit is a field of a revision |
| Enforcing a research procedure in the runner, such as confirmation but not search | The scope fields permit both at the same values; separating them needs a workflow engine |
| A mechanism for derived inputs at admission | A base per parameter value binds the derived file with existing digests |
| Inferring a script's parameter restrictions from its system spec | The runner does not know a script's physics |
| A timeout fixed at admission as the deadline's enforcement | Setup time before the start record would carry the job past the deadline |
| Checking only the investigation's current state before a job computes | A stop followed by a resume would revive a pending attempt |
| A default checkpoint interval | The right interval depends on the investigation; the user declares it |
| A checkpoint after every computation and its check run | 26 assessments in the 52-job probe |
| Default limits or a mandatory budget | Cost is unknown in advance; a limit exists only when the user declares one |
| The runner launching requested checks itself | Their scheduling and admission would be implicit; each check is a job that counts toward the checkpoint and limits |
| An approved revision applied to attempts already admitted | Needs reconfiguration of jobs in flight; stop, revise, and resume withdraws permission with existing mechanisms |
| Author-reported usage and check status in findings | Can be misreported; the runner derives both from records |
| Reviews opened only by findings | The user sometimes needs a pause that does not depend on the agent's judgment |
| A review that blocks one branch of an investigation | The runner has no notion of a branch; a review refuses admission for the whole investigation |
| Cancelling admitted runs when a review opens | A review is a pause for new work, not a stop |
| Restart files or resumed partial sweeps | No M3 protocol supports them (ADR-0003 §2.1) |
| A seed sweep shared between the baseline and the agent | Each investigation carries its own accounting and exercises `sweep` |
| Bayesian search as the first baseline | A one-dimensional monotone crossing does not need it |
| A model backend, an API key, or a list of supported runtimes | The runtime is the user's own |
| A pinned runtime or a live agent session in CI | CI has no logged-in user, and the harness does not own the runtime |
| A job queue, scheduler, grant service, or workflow engine | ADR-0001 and contract §7 reuse process and runtime facilities |

## 4. Consequences

- The contract changes accordingly:
  - §4 states that every change to a revision takes effect only on a user decision, in place of routine revisions without approval, that admitted attempts retain their dispatch revision and limits, and that a stop cancels its outstanding attempts permanently.
  - §5 describes `revise_investigation` as proposing a revision.
  - §8 records the runtime's name and version in each session's audit, in place of a runtime pinned in CI.
- The roadmap and checklist replace the default checkpoint interval with a checkpoint the user declares, and refer to the user's runtime instead of a selected one.
- A run's manifest gains its investigation and revision. The run reference and a new investigation reference document the records, the operator command, limits, and recovery.
- The readout example records the first investigation: its question, scope, claim rule, measured brackets, costs, and stated limits.
- Tests cover a draft that cannot start or approve itself, a proposed revision taking effect only on a decision, concurrent events, the admission rule of each operation, unknown limit names, concurrent launches against one limit, uncertain usage, the absolute deadline, an approved revision leaving admitted attempts under their dispatch revision, a run admitted but not yet computing after a stop, after a stop followed by a resume, and past the deadline, checkpoint gating that counts failed and aborted attempts, review gating with named release, and a new session continuing from records.
- Instruction delivery, the form of job tools and their supervision, the user-input path inside a session, and permission settings are decided before Block 6b and added to this record. So is ADR-0003's open point on whether a research agent may declare a device.
- Control and calibration add their operations to the permitted operations when those protocols are built. Co-design links device and control runs under one investigation.
