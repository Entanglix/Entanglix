# Investigation records and user decisions

An investigation records a research question, agreed terms, and findings. Its
revisions and events are retained under `research_runs/_investigations/`, or the
root passed to `--runs-root`. Reads and listings reconstruct state from these files.

The available commands manage drafts, starts, revision approvals, and findings.
`simulate`, `sweep`, and `check` currently run as standalone operations: they do not
yet link jobs to these investigations or enforce their scope, checkpoints, or limits.

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
digests. Every base includes `system.toml`. A variation declares either a closed
`range = [lower, upper]` or unique `values = [...]`. The supported targets are
`parameters.<name>`, `evolution.drive_amplitudes_ghz.<drive>`,
`evolution.carriers.<drive>.frequency_ghz`, and
`evolution.frame.frequencies_ghz.<subsystem>`. Discrete values can have
`linked_values = { "<other target>" = [...] }`, one linked value per entry.
An exact bundle has no variations. Parameter-dependent data requires a separate
base for each value that changes that data.

Requested checks are named per base. They record a standing request; the client
issues each check separately. Free-text assumptions and scientific criteria remain
recorded terms, rather than machine-certified judgments.

For an elapsed checkpoint use `elapsed_seconds = 60.0` instead of a count.
Optional named review points use `[review_points]` with a positive finished-attempt
count per name. Accepted resource limit names are `job_count`, `gpu_seconds`, and
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
it is not OS authentication. Resume and named review release are not available yet.

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
finding to the effective revision. Findings currently retain authored reasoning and
run references; they do not summarize execution, checks, or usage.

The decision is `continue`, `refine`, `stop`, or `review`. A `review` finding requires
`review_name`; other decisions omit it. These currently record recommendations;
checkpoint/review gating and cancellation are not yet implemented. A final report
can leave `next_experiment` empty. A draft must be started before receiving findings.

Records use version 1. Revisions live in `revisions/0001.toml`, `0002.toml`, and so
on; events in `events/00000001.toml`, `00000002.toml`, and so on. Earlier files are
never replaced through supported operations. Writers share a directory lock;
readers obtain a consistent snapshot. A new process reads the same effective
revision and history. CLI output is JSON; invalid declarations or actions return
exit `64` with a diagnostic.
