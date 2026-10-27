---
name: gpu-budget
description: Assess measured Entanglix job costs, optional investigation limits, and GPU ownership before continuing or stopping research.
version: 0.1.1
---

# Resources and GPU use

Read workspace `reference/operations/investigations.md` for limit, checkpoint,
and finding fields, and `reference/operations/runs.md` for GPU ownership/recovery.

Read `read_investigation` for actual usage, commitments, uncertain attempts,
remaining declared limits, checkpoints, and reviews. There is no default budget.
Ask for a resource limit only when needed; an unknown simulation cost calls for a
bounded experiment and measured cost, not an invented prediction.

`job_count` counts all admitted jobs, including checks, failed work, and reruns.
`job_seconds` counts CPU and GPU execution seconds without a GPU
multiplier. GPU-seconds are recorded usage, not an investigation limit. A job's
timeout is optional. With a job-seconds limit, admission commits the timeout or,
without one, remaining allowance; a job without a timeout is ended when that
commitment is spent. Deadlines also apply before an admitted job starts.
Unsettled work counts at its commitment; reconciliation cannot confirm usage or
return allowance. Confirmed terminal usage replaces the commitment.

Keep CUDA hidden in the agent session. Request GPU computation only through
runner tools and an approved CUDA base. The operator configures the device and
runner extras. A busy GPU refuses launch before an attempt exists and consumes
no allowance. There is no job queue: inspect existing work and wait or hand off,
rather than repeatedly launching. CPU may be faster for small systems; make a
performance claim only from measurements for the relevant workload.

If ownership is unknown, do not remove locks or clear the runner's environment
profile. Read the lost attempt and reconcile when the supported operation permits
it. Stop owned work with `abort_run` or `stop_investigation`; a stop cancels both
running and not-yet-started attempts, and resume does not revive them.

Resource availability does not justify continued research without learning.
Assess convergence, meaningful precision, observed cost, and possible next
experiments. Record a supported stop/review handoff when scope or evidence cannot
settle the question. Propose complete revised terms for a limit change and give
the user the operator command. The approval is the user's: run that command only
as the conduct's human-action rule allows.
