# M3 Guide — Investigations and Agent Integration

**Status:** Draft implementation plan; no M3 capability is claimed by this document. The decisions in §7 are recorded in [ADR-0004](adr/0004-investigations-and-agent-sessions.md), with matching contract, roadmap, and checklist edits. Record layouts, command names, limits, and file names below are proposals.
**Scope:** M3, covering the investigation decisions after M2, Block 6a (Investigation Execution), and Block 6b (Agent Integration).
**Authority:** [roadmap](roadmap.md#investigation-decisions--after-m2-before-blocks-6a6b), [accepted mandate](adr/0001-mandate.md#4-later-decisions-and-implementation-evidence), and [harness contract](../HARNESS_CONTRACT.md#4-bounded-investigations). [The checklist](checklist.md#m3--bounded-agent-assisted-exploration-blocks-6a6b) owns completion status and acceptance evidence.
**Audience:** the owner and implementation agents. This is a planning document; operational docs and production agents must work without it.

## 1. Outcome and boundary

M3 adds investigations to M2's runs. An investigation is a recorded research question with a scope the user started, pursued over many runs by a scripted client or an agent. At completion, an expert user can:

1. Start an investigation from a drafted scope. Its question, assumptions, scope, evidence requirements, limits, and review state change only through recorded user decisions.
2. Launch jobs for it through the existing protocols. Each launch is admitted against the active scope, the stop and review state, the checkpoint, and any limits. Usage is recorded across runs, checks, retries, and failures.
3. Read the investigation at any time: revisions, findings, linked runs and checks, usage, and outstanding work. A new session continues from these records without recomputing completed work.
4. Receive a final report or a review handoff whose claims link to the runs and checks behind them and name the relevant checks that did not run.
5. Hand the same work to an agent through MCP tools and skills, with human actions outside its reach, and compare its outcome with the scripted baseline.

M3 covers one coordinating agent on the workstation and the sweep/search operations: `simulate`, `sweep`, `check`, and authored scripts. The harness holds no model backend and no API key, and it does not limit the runtime: the agent runs in whatever runtime the user brings, which reaches the harness through the MCP server and skills. Claude Code and Codex are the most widely used. Control, calibration, the catalog and promotion, cluster execution, and OS-level isolation belong to later blocks.

The first investigation runs on CPU. Its fixture has dimension 30, where the GPU path was about 100 times slower per point in the M2 acceptance runs. GPU dispatch and the `gpu_seconds` limit inside an investigation are covered by a lifecycle test (§4), not by the science. Outperforming the baseline is not an M3 requirement.

## 2. Build by step

```text
M2: runs, checks on request, simulate, sweep
    |
Decisions: first investigation and evaluation (probe) --> ADR-0004
    |
Block 6a: records and decisions --> admission and limits --> checkpoints, stop, resume --> scripted client + 6a acceptance
    |
Decision: instruction delivery, permissions, job tools
    |
Block 6b: MCP surface and audit --> skills and instructions --> agent investigation + M3 evaluation
```

Steps complete in order. D3 needs nothing from Block 6a and can be taken at any time before 6b-1.

| Step | Work | Evidence before advancing |
|---|---|---|
| D1 — First investigation | Question, scope, evidence plan, resolution, early-stop case, and measured costs (§3) | Probe and results retained; choices recorded in the readout example |
| D2 — ADR-0004 | The decisions in §7, including the evaluation protocol | Accepted, with matching contract, roadmap, and checklist edits |
| 6a-1 — Records and user decisions | Revisions, decision events, findings; `create_investigation`, `revise_investigation`, `list_investigations`, `read_investigation`, `record_finding`; the operator decision command | Records and their meanings written in an operational investigation reference; a draft cannot start or approve itself; a proposed revision takes effect only on a user decision; concurrent events are neither lost nor overwritten |
| 6a-2 — Admission, usage, and limits | Investigation ID on job operations; scope checks per operation; atomic admission and the check before a job computes; usage from run records; `job_count`, `gpu_seconds`, `deadline`, and run timeouts | Out-of-scope launches and unknown limit names are refused; checks, reruns, and scripts are admitted by their own rules; no-limit and limit cases pass; two concurrent launches cannot both pass a limit that one exhausts; uncertain usage never returns allowance; an absolute deadline ends a running job and refuses later starts; an approved revision leaves attempts already admitted under their dispatch revision; one workstation GPU job records its GPU-seconds and a `gpu_seconds` limit refuses the next |
| 6a-3 — Checkpoints, reviews, stop, and resume | Checkpoint gating, assessments, review opening and release, `stop_investigation`, resume | A due checkpoint refuses admission until an assessment is recorded, counting failed and aborted attempts; an open review refuses admission until released by name; a stop cancels active jobs and records its reason; delayed-start tests show that a run admitted but not yet computing does not start after a stop, after a stop followed by a resume, or past the deadline; a new session continues from records with accounting intact |
| 6a-4 — Scripted client and 6a acceptance | The readout threshold client, its documented selection policy, and baseline runs of both cases | The client reproduces the probe's outcomes: the declared evidence in the normal case and a supported handoff in the early-stop case; history survives an interrupted session; Block 6a exit criteria pass |
| D3 — Agent-session decisions | Instruction delivery, user-input path, job-tool form, and permission settings (§5) | Recorded as a section of ADR-0004; instruction delivery demonstrated in a research workspace |
| 6b-1 — MCP surface and audit | `entanglix.mcp` behind an optional `mcp` extra; stdio server over the backing operations; job supervision independent of the connection; tool-call and session audit | A scripted MCP client gets the CLI's results; unavailable capabilities are not advertised; approval claims, out-of-scope requests, and limit changes are refused; a job whose MCP connection is lost, before or during computation, still reaches a valid terminal record with its deadline enforced and the GPU excluded until then; every call leaves an audit record linked to its revision and runs |
| 6b-2 — Skills and instructions | `skills/` with the initial skills; refreshed research-agent conduct through the selected mechanism; runtime permission settings | In a research workspace the runtime receives the research guidance and no repository development instructions, claimed only where the runtime exposes evidence of what it loaded; human-action commands are denied; the session sees no GPU |
| 6b-3 — Agent investigation and M3 evaluation | Formulation from a broad request; both cases through logged tools; comparison with the 6a baseline | Declared evidence and reporting criteria are met; the early-stop case ends in a supported report with next experiments; the comparison is reported, including worse or inconclusive results; M3 exit criteria pass |

## 3. First investigation and evaluation

### Question and scope

At each of three `EJ` values, which readout drive amplitude produces an additional endpoint-coherence loss of `0.01` at 100 ns? The loss is `ΔC(A) = C(0) − C(A)`, with `C` the [readout example](../examples/readout/README.md)'s normalized endpoint coherence. This is a characterization. Coherence alone does not establish readout quality or an optimal setting.

| Item | Choice |
|---|---|
| Fixed | The readout example's system, initial state, 100 ns window, and tolerances; `EJ = 12, 13.5, 15 GHz`, each with its paired carrier and frame frequency |
| Varied | Drive amplitude in `[0, 0.002] GHz` |
| Search | Rotating frame with the declared RWA, on CPU: the 3 × 3 seed sweep, then midpoint bisection between the seed amplitudes that bracket the target |
| Agreed resolution | Bracket width `1.25e-4 GHz`, three halvings of a seed interval |
| Confirmation | Lab-frame runs at zero amplitude and at both final bracket ends: one at 5 × 6 levels, one refined at 6 × 8 levels, cutoff 32, and tolerances `1e-11/1e-13` |
| Claim rule | A bracket is confirmed when the refined lab-frame loss is below the target at the lower end and above it at the upper end, each by more than the observed change between the two lab-frame runs |
| Requested checks | Seed sweep: integrity, diagnostics, declared expectation. RWA bracket ends: integrity, segments, diagnostics. Refined lab-frame runs: integrity |
| Outcomes per `EJ` | A confirmed bracket; no crossing in range; or a handoff when confirmation fails |
| Needs a user decision | A wider amplitude range, other `EJ` values, a changed model or window, a lab-frame search, or weaker evidence |
| Checkpoint | Declared by the user at the start; 10 finished attempts in this example |
| Limits | None unless the user declares one; the evaluation declares a `job_count` |

The claim has four stated limits:

- Coherence is assumed to decrease with amplitude at fixed `EJ`, which gives one crossing per line. The declared expectation checks this at sampled amplitudes only.
- The bracket width describes the RWA crossing. Lab-frame confirmation is reported separately.
- The change between the two lab-frame runs is observed convergence evidence, not an error bound. No check bounds whole-window error.
- When confirmation fails, more RWA bisection cannot settle the bracket. The handoff reports the last confirmed bracket and proposes a lab-frame search.

The runner does not enforce the difference between lab-frame confirmation and a lab-frame search: the checked scope fields permit a lab-frame run at any amplitude in range (§4). Confirming only at bracket ends is a research procedure in the agreed terms, followed by the client or agent.

### Measured basis

The [probe](../tests/probes/m3_readout.py) ran this plan with the existing `sweep`, `simulate`, and `check` operations and no investigation API. Its [results](../tests/probes/m3_readout_results.json) name every run.

| `EJ` (GHz) | Width (GHz) | RWA bracket (GHz) | Refined lab minus RWA loss | Lab-frame confirmation |
|---|---|---|---|---|
| 13.5 | `1.25e-4` | 0.001750–0.001875 | `7.4e-5` to `9.5e-5` | Confirmed |
| 13.5 | `1.5625e-5` | 0.001796875–0.001812500 | `8.5e-5` to `8.7e-5` | Not confirmed: the lower end is `3.6e-6` above the target |
| 15 | `1.25e-4` | 0.001375–0.001500 | `2.8e-5` to `5.1e-5` | Confirmed |
| 15 | `1.5625e-5` | 0.001437500–0.001453125 | `3.7e-5` to `4.0e-5` | Confirmed |

- **`EJ = 12`:** the largest sampled loss is `0.0039`, so there is no crossing in range under the monotonic assumption.
- **Lab-frame convergence:** the two lab-frame runs differ in loss by at most `2.4e-7`.
- **Job time:** seed sweep 4.4 s, RWA point 1.1 s, lab-frame run 6.6 s, refined lab-frame run 20 s, check run 0.8 s.
- **Totals:** the normal case is 30 jobs, 27 evaluated points, and 176 s of job time. The probe's full run, both widths at both `EJ` values, is 52 jobs, 41 points, and 301 s of job time within 329 s elapsed. Two thirds of the job time is refined lab-frame runs.

Two qualifications carry into acceptance:

- The failed confirmation at `EJ = 13.5` rests on a `3.6e-6` margin. That is 19 times the observed refinement change and about four times the `1e-6` cross-engine agreement of the frame study. It is deterministic in the locked environment. Acceptance tests the claim rule, not which `EJ` fails.
- The early-stop case is a supported handoff: the permitted method cannot establish the requested lab-frame claim, and the client stops with the evidence it has. It does not show that further lab-frame work would be uneconomic; a lab-frame run costs only six times an RWA run here.

### Evaluation protocol

- **Cases:** the normal case uses the agreed width. The early-stop case requests `1.5625e-5 GHz`. It first establishes the normal-width results for every `EJ`, then refines in ascending `EJ` order. At the first failed confirmation the client records a handoff, which opens a review, and reports the remaining tighter cases as unattempted. By the probe's numbers this ends at `EJ = 13.5` after 41 jobs, 34 points, and 240 s of job time.
- **Same start:** the baseline and the agent receive the same scope revision, requested checks, checkpoint, and limit, in separate investigations. Each runs the seed sweep itself.
- **Scripted selection policy:** the search, confirmation, and claim rule above, applied without judgment. A line with no crossing in the seed is reported, with a wider range proposed.
- **Allowances:** the user declares one `job_count` per case at the start, the same for the baseline and the agent. This example uses 60 in the normal case and 100 in the early-stop case, against 30 and 41 jobs for the scripted policy. The harness supplies no default, and an ordinary investigation has no limit unless its user declares one.
- **Measures:** outcome per `EJ` against the claim rule and the requested checks; elapsed time; job time for computation and for checks; jobs and evaluated points; time outside jobs; user decisions after the start; and the rationale recorded at each assessment and at the stop.
- **Agent runs:** three per case, each reported.

The expected result is the same outcome at a higher elapsed time: on a smooth one-dimensional curve the midpoint policy is close to optimal. Judgment shows only in the `EJ = 12` report, the stop decision, and the handoff.

## 4. Block 6a — investigation execution

### Records

```text
<runs_root>/_investigations/<investigation_id>/
  revisions/0001.toml   # runner: question, assumptions, scope, evidence plan, checkpoint,
                        # review points, limits with absence explicit, stop and review criteria
  events/               # runner: one immutable file per event, published by rename
                        #   decision: start, resume, approve a revision, release a review
                        #   finding:  assessment, interpretation, or report
                        #   stop:     reason and cancelled runs
```

System IDs begin with a letter or digit, so `_investigations` cannot collide with a system directory. A run launched for an investigation records the investigation ID and revision in its `run.toml`. There is no separate store or index: `list_investigations` and `read_investigation` derive state from these records and the linked runs, as `list_runs` does.

### User decisions

One operator command records a decision event: the action, the affected revision, a timestamp, the source (operator CLI and OS user), and an optional note. The actions are start, resume, approve a proposed revision, and release a named review. No research tool exposes the command, and a research request carrying an approval field is rejected. Limits are fields of a revision, so a limit change is an approved revision.

### Scope and revisions

A revision holds the agreed terms: the question, assumptions, scope fields, evidence requirements, checkpoint, review points, review and stop criteria, and limits. Every one of them is the user's. A research tool can only propose a revision, and it takes effect on a user decision. The runner enforces who may change these fields without judging what they say. Working plans and interpretations go in findings, which need no approval, and choosing experiments inside the scope needs no revision.

The runner checks structured scope fields only. They reuse the sweep target syntax:

- the permitted operations;
- the permitted base specs, each identified by the digests of its frozen inputs;
- for each varied target, a value set or a closed range, with linked values where targets move together.

Each operation is admitted by its own rule:

| Operation | In scope when |
|---|---|
| `simulate`, `sweep` | The frozen inputs match a permitted base apart from the varied targets, and every varied value at every point lies inside its set or range |
| `check` | Its source run belongs to the investigation; the check then belongs to it too. The check names and any replacement refinement are validated as for a standalone check |
| Reruns | The earlier run belongs to the investigation. The rerun keeps that membership and passes admission under the current revision |
| Authored scripts | The script, its data, and its system spec match a permitted base exactly. The runner cannot infer a script's physical parameter restrictions from its system spec |

The first investigation permits `simulate`, `sweep`, and `check`, with five bases: RWA, lab frame, and one refined lab-frame base per `EJ` value. The refined initial state is embedded in the refined basis, which depends on `EJ`, so its array differs between `EJ` values. Each refined base therefore binds its own state file to its `EJ` and paired frequency, and varies only the amplitude. The scope is agreed before the seed sweep shows which lines cross, so every `EJ` gets one. The RWA and lab-frame bases set `EJ` through the spec parameter, which keeps one system file and one state file for all three values.

These fields permit a lab-frame run at any amplitude in range. Confirming only at bracket ends, with a lab-frame search left to a review, is a research procedure in the agreed terms. The client's workflow follows it; the runner does not enforce it.

### Admission, usage, and limits

Job operations take the investigation ID. Under an exclusive lock on the investigation directory, the runner checks:

1. a started, active revision;
2. the permitted operation and the scope fields;
3. no stop and no open review;
4. no checkpoint due;
5. any limits, counting recorded usage, admitted jobs without a terminal record, and the proposed job.

It then creates the run, whose `run.toml` is the commitment record, and releases the lock. Decisions, revisions, stops, and findings that open a review take the same lock, so none can interleave with an admission.

An admitted attempt keeps the revision and limits under which it was admitted, including its deadline. An approved revision governs later admissions only. To withdraw permission from outstanding attempts, the user stops the investigation, approves the revision, and resumes.

An admitted run still freezes its inputs and passes the launch gate before it computes. Immediately before releasing the job to compute, the runner takes the lock again. If a stop event names the run, or its deadline has passed, the run ends as aborted without computing. Otherwise the runner publishes the start record before releasing the lock, so a later stop finds the run and aborts it.

Usage comes from each run's `FINISHED`. An admitted run without one counts at its maximum, its timeout times its assigned GPUs, and a reconciled attempt stays at that maximum. A `gpu_seconds` limit requires a timeout on every job. The supervisor carries the absolute deadline into execution and ends the job at expiry, whatever its timeout; the local executor has no queue.

Limit accounting is tested in CPU CI with short CPU jobs and crafted records. One workstation GPU job inside an investigation confirms GPU-seconds recording and refusal under a `gpu_seconds` limit.

### Checkpoints, assessments, and reviews

A checkpoint is a required scope field with no implicit default: a count of finished attempts or an elapsed interval. Every finished run linked to the investigation counts, including verification runs and failed or aborted attempts. When one is due, admission is refused until an assessment is recorded.

The first investigation declares 10, which gives three assessments in the normal case. A count of two, one computation and its check run, would have meant 26 assessments in the probe.

One finding record serves assessments, interpretations, and final reports:

| Supplied by the author | Filled by the runner |
|---|---|
| The runs considered; what was learned; what remains unresolved; a decision of `continue`, `refine`, `stop`, or `review`; its rationale; the next experiment | The revision; usage since the previous assessment and in total, as jobs, evaluated points, wall seconds, and GPU-seconds; for each linked run, the checks run with their outcomes and the relevant checks not run |

The runner validates that the fields are present and that the referenced runs belong to the investigation. It does not judge the rationale.

A review opens in two ways: a finding whose decision is `review`, which names it, or a review point the user declares in the scope at a finished-attempt count. A declared review point is optional, absent unless the user requests one, and uses the checkpoint's counter. An open review refuses admission for the whole investigation, not one branch of it, until a user decision releases it by name. Runs already admitted finish. A proposed revision does not stop work inside the current scope.

Requested checks are part of the evidence plan, and a standing request among them is the user's authorization. The client or agent still issues each `check` as its own job, so its timing and admission stay explicit; the runner launches no check itself. `read_investigation` and findings show which runs still lack a requested check.

### Stop and resume

`stop_investigation` closes admission, aborts running jobs through `abort_run`, and records one reason from the contract's list. Its event names every attempt outstanding at that moment, running or admitted but not yet computing. Their cancellation is permanent: an admitted attempt finds itself in the stop event at its check before start and never computes, even after a resume. A report can recommend stopping while allowance remains.

- **State comes from records, not from a session.** A new session reads the investigation and continues an active one without a user action.
- **A stopped investigation reopens only through a resume decision,** after its open reviews are released. Resume permits new admissions; it does not revive an attempt a stop cancelled.
- **Completed results are not recomputed** because a session changed. Interrupted attempts are reconciled as in M2 and rerun under new linked IDs; a rerun is a new attempt and counts again.
- **Usage carries across sessions.** Uncertain usage stays counted at its commitment.
- **No restart files in M3:** `simulate` and `sweep` have none.

### Scripted client

The probe already implements the selection policy without the investigation API. Step 6a-4 turns it into a client beside the readout example that works only through the investigation operations. It drafts the scope, waits for the user's start, launches through admission, records an assessment when a checkpoint is due, and ends with a report or a handoff. It establishes the normal-width results before any tighter request and need not repeat every measurement the probe collected.

The client computes the loss and the brackets from several runs' metrics. These are not trusted metrics, so each finding lists its source runs and values. Bayesian search does not suit a one-dimensional monotone crossing and is not built in M3.

### Carried from M2

- The evidence-point bound stays at 8, and a failing point still fails its sweep. Refinement here uses single-point runs and the 3 × 3 seed, so neither binds. Revisit both when an investigation uses three or more axes or batches many points in one job.
- The metric is a coherence magnitude, which the RWA's phase shift does not affect. A phase-sensitive metric would need a lab-frame comparison from the start.

## 5. Block 6b — agent integration

### Decisions before 6b

| Decision | Proposal |
|---|---|
| Runtime | The user's own. Nothing is built or configured as a backend, and the harness requires only MCP over stdio, so any runtime that speaks it can drive the harness. Each session's audit records the runtime's name and version |
| Instruction delivery | Research conduct in the MCP server's instructions and task procedures in skills, both independent of the runtime. Where a runtime does not surface required conduct or procedures, a generated workspace file supplies only the missing content |
| User-input path | The operator CLI in the user's own terminal is the only decision path in M3. A runtime prompt is added only if the agent cannot answer it itself |
| Job tools | A job tool returns the run ID once the manifest exists and the supervisor owns the attempt, and the agent then reads or waits on the run. Reading a live trace or aborting a stalled job then does not depend on how the runtime and transport handle a long call |
| Permissions | Human-action commands are denied to the agent, and its session sees no GPU. The user sets both in their runtime; the operational docs state the requirement and give example settings for Claude Code and Codex. The MCP server's own environment keeps the device for runner-launched jobs |

### MCP surface and audit

`entanglix.mcp` wraps the backing operations the CLI uses and advertises only available capabilities: the read-only operations, the controlled jobs M3 provides, and `create_investigation`, `revise_investigation`, `record_finding`, and `reconcile_run`. Human actions are absent, as are catalog, control, and calibration operations.

Returning at admission needs a supervisor that outlives the call. The job tool starts the existing runner command as a separate process in its own session, with output streams that do not depend on the server. That process holds the attempt and GPU locks, enforces the timeout and deadline, and writes `FINISHED`, exactly as a CLI launch does. The tool returns only when the manifest exists and the supervisor owns the attempt. After a disconnection the run finishes or times out under its supervisor, and the next session reads it. A lost supervisor is reconciled as in M2.

Step 6b-1 owns this work and its tests: one disconnection before computation starts and one during execution. Each shows that the run reaches a valid terminal record, that its deadline is still enforced, and that the GPU stays excluded until then. A separate process alone does not establish these.

The server writes one record per tool call and one per session under `<runs_root>/_audit/`, outside run outputs, with the fields contract §6 requires.

### Skills and instructions

`skills/` holds the roadmap's initial skills, `system-modeler`, `simulate`, `design-sweep`, and `gpu-budget`, each declaring its name, description, and version. The refreshed [research-agent conduct](research-agent.md) adds standing check requests, assessments, and stopping. The delivery check runs in a research workspace and confirms that the research guidance arrives and the repository development instructions do not. Delivery is claimed only where the runtime exposes evidence of what it loaded.

### Agent investigation and evaluation

- **Formulation:** from a broad request about readout drive and qubit coherence, the agent drafts the investigation of §3 and the user starts it.
- **Execution:** the agent runs both cases through logged tools, three runs each, under the evaluation protocol.
- **Refusals:** tests reject approval claims in tool arguments, out-of-scope requests, and limit changes through research tools.
- **CI and workstation:** CI tests the harness side with a scripted MCP client and needs no runtime: advertised tools, refusals, audit records, the server's instructions, and skill metadata. Checks that need a session run on the workstation in the user's logged-in runtime and are retained: instruction delivery, the permission settings, the hidden GPU, and the agent runs.
- **Usability:** a user following only the operational docs can start, read, and stop the example investigation.

## 6. Dependencies, layout, and validation

Investigations use the standard library, as the runner does. The `mcp` extra adds the MCP Python SDK and nothing else.

Suggested locations, created as code arrives:

```text
src/entanglix/investigations.py     # records, decisions, admission, findings, stop and resume
src/entanglix/mcp/                  # optional extra: stdio server over the backing operations, audit
skills/                             # system-modeler, simulate, design-sweep, gpu-budget
examples/readout/threshold.py       # scripted client and M3 baseline
docs/operations/investigations.md   # records, commands, limits, recovery
tests/probes/m3_readout.py          # D1 probe and its retained results
```

Proposed validation commands after the corresponding capabilities exist:

```sh
uv run --locked --exact pytest -m "not gpu"
uv run --locked --exact --extra mcp pytest -m "not gpu" -k "mcp"
uv run --locked --exact ruff check .
uv run --locked --exact ruff format --check .
uv run --locked --exact python examples/readout/threshold.py --investigation <investigation_id>
```

## 7. Settled decisions

Recorded in [ADR-0004](adr/0004-investigations-and-agent-sessions.md), with matching contract, roadmap, and checklist edits:

1. Investigation records are files beside the runs: immutable revisions and one file per event, written by the runner. Runs name their investigation and revision. There is no separate store.
2. A user decision is an event recorded by one operator command. The actions are start, resume, approve a revision, and release a review. Research tools cannot create one.
3. Every field of a revision belongs to the user: a research tool proposes a revision, and a user decision makes it effective. Findings need no approval. The runner checks permitted operations, permitted bases by digest, and value sets or ranges for varied targets, with one admission rule per operation.
4. Admission, decisions, revisions, stops, and review-opening findings share one investigation lock, and the admitted run's manifest is the commitment record. Immediately before a job computes, the runner checks the absolute deadline and whether a stop event names the run. A stop cancels its outstanding attempts permanently; resume does not revive them. An admitted attempt keeps its dispatch revision and limits; an approved revision governs later admissions.
5. A checkpoint is required and has no implicit default. Verification runs and failed or aborted attempts count.
6. One finding record serves assessments, interpretations, and reports. The author supplies interpretation and a decision; the runner fills usage and check status from records.
7. A review opens from a `review` finding or from a review point the user declared at a finished-attempt count, and is released by name. A declared review point is optional and absent unless the user requests one.
8. Resume derives state from records. Completed results are reused, a stopped investigation needs a resume decision, and M3 adds no restart files.
9. The evaluation protocol compares a documented scripted policy and the agent under the same scope, checks, checkpoint, and allowance, with the measures in §3 and an early-stop case. The readout numbers stay in the example.
10. The harness supplies no default checkpoint or limit. The user declares the checkpoint, and any limit, when starting an investigation.
11. The runtime is the user's own and is not limited by the harness. Each session's audit records its name and version, CI covers the harness side with a scripted MCP client, and the roadmap, checklist, and contract §8 say so.
12. Before 6b: the remaining decisions in §5, added to the ADR when taken.

## 8. Exit evidence and handoff

[Checklist M3](checklist.md#m3--bounded-agent-assisted-exploration-blocks-6a6b) is the sole acceptance list; this guide adds no completion gates. Baseline and agent investigations are themselves Entanglix records: retain the cited investigations, runs, verification runs, and audit records, and link them with test commands and results. Publish investigation records, commands, limits, and recovery in `docs/operations/investigations.md`, the investigation's question and settled numbers in the readout example, the MCP and skill entry points in the root README, and the refreshed research-agent conduct through the selected mechanism. Operational documents stand alone without linking to planning docs.

Later blocks extend the same records: control and calibration add their protocols to the permitted operations, co-design links device and control runs under one investigation, and the catalog cites findings.

Before retiring this guide, retain consequential design rationale in ADR-0004 and necessary conventions and instructions in operational docs.
