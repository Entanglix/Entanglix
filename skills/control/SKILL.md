---
name: control
description: Solve an Entanglix closed-system gate problem, reuse a device or starting pulse, inspect candidates, request checks, or re-solve frozen inputs.
version: 0.1.5
---

# Closed-system control

Read workspace `reference/operations/control.md` for the input, units, subspace,
solver support and neutral pulse. Use `reference/operations/investigations.md`
for approved terms. Read `reference/operations/runs.md` only when an attempt is
interrupted or its state is unknown.
Read `reference/operations/catalog.md` when reusing an entry or citing its checks.

Declare the gate, duration, amplitude and derivative bounds, reported target and
control frame. The supported control model is closed and uses an explicit RWA;
omitted dissipators appear in the launch listing. These choices do not establish
lab-frame or device-model accuracy. Control solves and their checks run on CPU.

Use `system_entry` instead of `system` for a promoted device. A `starting_pulse`
may name an entry, a control run or a neutral pulse file, and replaces `pulse.init`
and `pulse.seed`. Only a supporting solver accepts it, including after fallback;
`grape` supports starts and `piccolo` does not. The runner matches channel IDs,
knots, times and bounds, and refuses a start that does not fit. Reruns use the
frozen start and retain its source lineage. The entry's check scope applies to
the source candidate.

Call `control` with the authored protocol. In an investigation read effective
terms first and name an approved base; pass the device parameters and seeds it
permits in `values`, leaving the authored protocol unchanged.
A different solver or options table needs another approved base, even when the
generated ProblemSpec digest is identical. A fallback applies only to an absent
named solver; it does not hide unsupported input or a failed solve.

The tool returns admission, not a finished gate. Pass the returned path to
`read_run`, optionally with a bounded wait. Inspect execution status, actual
solver, objective, separately reported fidelity and convergence. Request
`detail=True` only for needed provenance or report details. Reaching an iteration
cap can leave a completed candidate with `converged = false`.

Issue `check` for explicitly or standing-requested names: `replay` compares the
solver's reported fidelity with an independent rollout; `target` tests the
declared target; `constraints` tests the returned knots and time grid. Read the
check run with `read_run(detail=True)` for replayed fidelity and leakage, which
the default summary lists under `detail_omitted`. Leakage is diagnostic.
Passing checks applies to the declared control model, not the
RWA or device-model approximation. Different pulses from two solvers can both
pass; compare their shared ProblemSpec digests and requested check outcomes.

For device–control co-design, follow the control reference's co-design
comparison. Use the approved base's declared frame, carriers, bounds and
evaluation conditions. Screen candidates with `simulate` or `sweep`, then solve
each chosen candidate inside the approved base, with its parameters and seed in
`values` and the screening runs in `parent_run_ids`. Launch the seeds the
evidence requirements name for each candidate, and add attempts only as they
allocate them. Issue the checks and diagnostics explicitly or standing-requested
in the evidence requirements. A declared evaluation or refinement table alone
does not request a diagnostic. Compare in a finding: weigh every checked pulse
known for each candidate, report seed spread, refinement change and differing
termination as evidence, keep spreads per solver, state how the pulse and the
compared quantity were chosen, and report an unsupported comparison as
unresolved.

Settle an interrupted attempt once ownership and group-liveness checks permit
reconciliation. Use `rerun`, not a fresh solve of the same inputs, to
retain lineage and investigation membership under current approved terms.
Each rerun needs its own requested checks. Record findings with source run IDs,
attained fidelity, unmet targets or constraints, leakage and the next decision.
