# Investigations, admission, and user decisions

An investigation records a research question, agreed terms, and findings. Its
revisions and events are retained under `research_runs/_investigations/`, or the
root passed to `--runs-root`. Reads and listings reconstruct state from these files.

The available commands manage drafts, starts, revision approvals, findings,
checkpoint/review gates, stops, resumes, and job admission. `simulate`, `sweep`,
`control`, `calibrate`, and authored scripts can link jobs to an investigation;
checks and reruns inherit that membership. Scope and optional limits are enforced.

## Declare the terms

Write complete terms in an `investigation.toml`. A checkpoint is required, with
exactly one positive finished-attempt count or elapsed interval in seconds. Include
`[limits]`, leaving it empty when no resource limit is requested. The harness
supplies no default checkpoint or resource limit.

This example records an amplitude study. Replace the input digests with the SHA-256
digests of the intended files, for example using `sha256sum system.toml simulate.toml`.
Include all model, protocol, and data files in the permitted bundle.

```toml
format_version = 1
question = "Which amplitudes bracket an additional coherence loss of 0.01?"
assumptions = ["Coherence decreases with amplitude at the sampled points."]
review_criteria = ["A wider amplitude range needs user review."]
stop_criteria = ["Stop when the permitted method cannot establish the claim."]

[scope]
operations = ["simulate", "check"]

[[scope.bases]]
name = "rwa"

[scope.bases.inputs]
"system.toml" = "<64 hexadecimal SHA-256 characters>"
"simulate.toml" = "<64 hexadecimal SHA-256 characters>"

[[scope.bases.variations]]
target = "evolution.drive_amplitudes_ghz.readout"
range = [0.0, 0.002]

[evidence_requirements]
requirements = ["Report the bracket, convergence evidence, and limitations."]

[evidence_requirements.checks]
rwa = ["integrity", "segments", "diagnostics"]

[checkpoint]
finished_attempts = 10

[limits]
```

Each base has a unique name and relative input filenames with lowercase SHA-256
digests. Every base includes `system.toml`. Use the runner's fixed names for the
system (`system.toml`), simulation protocol (`simulate.toml`, even when authored as
`readout.toml`), control input (`control.toml`), a control input's starting-pulse
file (`starting-pulse.npz`), calibration spec (`calibrate.toml`), and script
(`script.py`); subsystem and data filenames, including datasets and their array files,
are retained.
An optional `system_entry` names a promoted catalog device. Drafting and revisions
require matching digests for its system and every referenced file, and launches
under that base record the entry; see [device reuse](catalog.md#reusing-a-device).
A variation declares either a closed
`range = [lower, upper]` or unique `values = [...]`. The supported targets are
`parameters.<name>`, `evolution.drive_amplitudes_ghz.<drive>`,
`evolution.carriers.<drive>.frequency_ghz`, and
`evolution.frame.frequencies_ghz.<subsystem>`; a control base may also vary
`pulse.seed`, by non-negative integer values. Discrete values can have
`linked_values = { "<other target>" = [...] }`, one linked value per entry.
An exact bundle has no variations. Parameter-dependent data requires a separate
base for each value that changes that data.

Requested checks are named per base. They record a standing request; the client
issues each check separately. Free-text assumptions and scientific criteria remain
recorded terms, rather than machine-certified judgments.

For an elapsed checkpoint use `elapsed_seconds = 60.0` instead of a count.
Optional named review points use `[review_points]` with a positive finished-attempt
count per name. Accepted resource limit names are `job_count`, `job_seconds`, and
`deadline`, an absolute TOML timestamp with a UTC offset, such as
`2026-10-05T18:00:00Z`. Positive counts must be integers; seconds must be finite and
positive. Unknown fields, limit names, or check names are rejected.

## Draft, start, and revise

```sh
entanglix create-investigation investigation.toml --runs-root research_runs
entanglix list-investigations --runs-root research_runs
entanglix read-investigation research_runs/_investigations/<investigation_id>
```

Creation returns the directory and an initial draft revision. It creates no user
decision. Every field of a revision belongs to the user, including the evidence
requirements and stop criteria. Research operations cannot make a revision effective.

The user starts a specific revision through the separate operator command:

```sh
entanglix investigation-decision research_runs/_investigations/<investigation_id> start --revision 1
```

To change any term, supply the complete replacement declaration. The previous
effective revision stays effective until the user approves the proposal:

```sh
entanglix revise-investigation research_runs/_investigations/<investigation_id> revised.toml
entanglix investigation-decision research_runs/_investigations/<investigation_id> approve-revision --revision 2 --note "Accept the revised evidence plan."
```

Decision records retain the action, revision, timestamp, operator CLI source,
OS user, and optional note. `investigation-decision` is a human action and is absent
from the research operations advertised by `capabilities`. This separation follows
the cooperative workflow in the [harness contract](../../HARNESS_CONTRACT.md);
it is not OS authentication. Resume and named review release use the same operator
command, as described below.

## Launch work within scope

Supply the started investigation ID and a permitted base name. For a simulation,
the protocol argument is the unchanged base file; pass varied values separately:

```sh
entanglix simulate simulate.toml --runs-root research_runs --investigation <investigation_id> --base rwa --vary evolution.drive_amplitudes_ghz.readout=0.001
entanglix simulate simulate.toml --runs-root research_runs --investigation <investigation_id> --base rwa --vary evolution.drive_amplitudes_ghz.readout=0.001 --dry-run
entanglix sweep sweep.toml --runs-root research_runs --investigation <investigation_id> --base rwa
```

A sweep's `base` field points to the permitted simulation file; its axes supply the
varied values. The declaration binds that simulation file and its model/data inputs,
while the sweep supplies the grid, evidence metric, and declared expectation.
The runner verifies the complete base bundle against its digests, then checks every
varied value at every point. Targets must already exist in the base protocol.
Values omitted from a simulation retain the base's values and still pass scope
checks. Linked targets must match their declared pairings. No other base input can
change. Dry runs perform the same scope and limit checks without admitting a job.

The run freezes the original base files under `spec/base/` as well as the compiled
inputs, and records its investigation ID, revision, base name, limits, and resource
commitment. Reruns use those original bases and varied values, keep membership, and
pass admission under the current approved revision. Investigation reruns retain
the same runs root.

An authored script uses `run-script ... --investigation <investigation_id> --base
<base_name>` and must match an exact bundle, including the script, system, and data.

A [control solve](control.md#investigations-and-reruns) passes varied values in
the same way, with `--vary`; it varies only `parameters.<name>` and `pulse.seed`.
The authored control input, system and data stay exact, and changing the solver or
its options requires another approved base. `rerun-control` inherits membership,
reuses the original values, and uses current approved terms.

A [calibration](calibration.md#investigations-and-reruns) takes no varied values:
`calibrate fit.toml --investigation <investigation_id> --base <base_name>` must match
an exact bundle of the spec, the system and its inputs, and the training datasets.
`rerun-calibration` inherits membership and uses current approved terms.

A named check on a linked source automatically belongs to its investigation:

```sh
entanglix check research_runs/<system_id>/<run_id> integrity segments --timeout 60
entanglix rerun-simulation research_runs/<system_id>/<run_id>
```

Check names and replacement refinements follow the standalone rules. A check is a
separate admitted job; the harness does not issue standing requested checks itself.
It may check an older source even when that source's base is absent from the current
revision, provided `check` remains permitted and the limits allow the job.

## Usage and optional limits

Admission is serialized with user decisions and records its commitment in the new
run manifest. `job_count` counts every admitted job, including checks, failed work,
and reruns. Rejected launches create no attempt and consume no job allowance.
`job_seconds` limits total execution time for CPU and GPU jobs alike, using each
job's `FINISHED.usage.wall_seconds` on the same clock as its timeout. A timeout is
optional on every job. Admission commits its timeout, or, without one, the remaining
allowance, and refuses when nothing remains. The supervisor ends a job without a
timeout when that commitment is used up. Each unsettled attempt counts at its
commitment; confirmed `FINISHED` usage replaces it. An interrupted or reconciled
attempt retains its commitment because reconciliation cannot confirm usage. A new
limit refuses admission while an earlier unsettled job admitted without a limit or
timeout remains, and names that run. `gpu_seconds` is recorded usage, not a limit.

An absolute `deadline` refuses admission after expiry. Immediately before computing,
the runner rechecks the admitted attempt's deadline under the investigation lock
and publishes its start record before releasing the lock. Expiry before start ends
the attempt as aborted with no computation; expiry during execution terminates its
process group. Admitted jobs retain their original revision and deadline when a
new revision is approved. The local executor polls at 10 ms and gives termination
2 seconds before escalating to SIGKILL. `FINISHED` records actual wall/GPU seconds
and deadline and job-commitment overruns; termination and cleanup can extend past
a ceiling. Overruns compare completion time with the deadline, and execution time
with the admitted commitment, which is the timeout when the job has one.

`read-investigation` reports jobs, completed computation points, confirmed execution
time as `job_seconds`, recorded GPU-seconds, outstanding job commitments, and
unsettled jobs. Sweep points count when
their result record is published, including partial sweeps. Usage is retained
across sessions and approved revisions. Standalone runs still need no investigation.

## Findings and retained history

Findings record assessments, interpretations, or final reports without changing
the agreed terms. Write a `finding.toml`, then append it:

```toml
kind = "assessment"
runs = []
learned = "The model and evidence plan have been agreed."
unresolved = ["The seed grid has not run yet."]
decision = "continue"
rationale = "The first experiment fits the agreed scope."
next_experiment = "Run the seed grid."
```

```sh
entanglix record-finding research_runs/_investigations/<investigation_id> finding.toml
```

An empty run list is useful before the first experiment. Nonempty lists must name
run IDs whose manifests link them to this investigation; standalone runs cannot be
claimed as members. The runner checks membership from the manifests and binds the
finding to the effective revision. It fills total usage and the usage since the
previous assessment from run records. For each referenced run it lists execution
status, checks run with their policy version, any outcome and, for a calibration
check, the solution index, and requested check
names not yet run. Requests for a computation come from its base in the effective
revision, so newly requested checks can appear as missing on older runs. A
verification run retains its own frozen check request. Earlier findings remain
unchanged.
For each computation it also lists relevant checks not run, using names from the
corrector's own listing, whether or not those checks were requested. If a run's
frozen plan cannot be loaded, the finding records why that list is unavailable.
Diagnostics have no pass/fail outcome. Full check reports remain with their linked
verification runs; available-check listings are not copied into findings. Authors
cannot supply usage or evidence summaries.

The decision is `continue`, `refine`, `stop`, or `review`. A `review` finding requires
`review_name`; other decisions omit it. A review decision opens a named review;
continue, refine, and stop record recommendations without changing the agreed
terms or executing a stop. A final report can leave `next_experiment` empty.
A final report lists every attempt, including failed and interrupted ones.
A draft must be started before receiving findings.

Use a finding's `learned` field for interpretation, with source run IDs supporting
the claim. Measurements remain in attempt and run records; repeating their full
results in each finding makes later recovery harder. MCP finding writes return
an acknowledgment rather than echoing the submitted text and attached evidence.
MCP reads return status with an attention list by default. Read attempts and event
history through their paged table views, or request one complete event by ID; see
the [MCP reference](mcp.md#investigation-views-and-tables). CLI and Python reads
remain complete. For a user report, including every attempt and finding, run:

```sh
entanglix report-investigation PATH --output report.md
```

Omit `--output` to print Markdown. The report names the last event it covers,
opens with the question, latest finding of any kind and open items, and puts terms
after attempts and history. Findings are agent interpretations, rendered as text
with referenced-run counts. The report is identical over unchanged retained
records. It includes retained checkpoint counts rather than the advancing elapsed
clock. It is generated only when requested and adds no history records.

Records use version 1. Revisions live in `revisions/0001.toml`, `0002.toml`, and so
on; events in `events/00000001.toml`, `00000002.toml`, and so on. Earlier files are
never replaced through supported operations. Writers share a directory lock;
readers obtain a consistent snapshot. A new process reads the same effective
revision and history. CLI output is JSON; invalid declarations or actions return
exit `64` with a diagnostic.

## Checkpoints and reviews

`read-investigation` reports the checkpoint's due state, total finished attempts,
finished attempts since the last assessment, elapsed seconds, and open review names.
Every linked job with `FINISHED` counts, including checks and failed/aborted work.
An interrupted attempt counts once it is reconciled, and its usage stays uncertain.
Unreconciled interrupted attempts do not count, and an attempt counts at most once.
A due checkpoint refuses new admission, including dry runs, until a finding with
`kind = "assessment"` is recorded. Interpretations and reports do not reset it.
Elapsed intervals run from the start or last assessment; approvals and resumes do
not reset them. A stop does not pause an elapsed interval. Already admitted jobs
may finish.

A review opens from a finding's `review` decision or a declared review point at its
total finished-attempt count. It blocks all new admissions until the user releases
it by name. Assessments and approved revisions do not release an open review.

```sh
entanglix investigation-decision research_runs/_investigations/<investigation_id> release-review --review strategy --note "Proceed with the current method."
```

## Stop and resume

```sh
entanglix stop-investigation research_runs/_investigations/<investigation_id> user_cancellation --note "Pause this work."
entanglix investigation-decision research_runs/_investigations/<investigation_id> resume
```

Stop reasons are `evidence_completion`, `insufficient_progress`,
`model_or_strategy_limitations`, `user_cancellation`, `configured_limits`,
`interruption`, and `review_handoff`; the note explains the case. Stopping closes
admission and records the outstanding run IDs before cancelling their jobs.
Running jobs and their descendants are stopped through the runner's process-group
cancellation. A delayed attempt checks its permanent cancellation before computing
and ends aborted, even if the investigation has since resumed. Stop output lists
any cancellation errors; retained records and logs remain inspectable.

Resume is an operator decision on the effective revision, after all open reviews
are released. It allows new admissions without reviving cancelled attempts or
resetting usage or checkpoints. Proposed and approved revisions do not resume a
stopped investigation. A new session reads the same records; it does not recompute
completed results. Interrupted work is reconciled and rerun with new linked IDs.
Simulation and sweep runs have no restart files.

Inspection includes each admitted attempt's run ID, base, varied values, dispatch revision, and execution state. Unresolved detached launches are listed separately and consume no allowance before admission; the [run reference](runs.md#detached-launches-and-waiting) describes admission recovery and waits.
