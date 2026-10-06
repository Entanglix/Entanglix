---
name: design-sweep
description: Plan and execute Entanglix static or dynamic parameter sweeps, interpret sampled trends, and choose bounded follow-up experiments.
version: 0.1.0
---

# Sweep and search

Read workspace `reference/operations/protocols.md` for sweep fields and metrics;
use `reference/operations/investigations.md` for terms and finding fields.

Start from a validated simulation base. A spectrum gives static quantities;
Lindblad evolution gives dynamic endpoints. Either supports a partial research
workflow without control optimization. Declare unique values on each axis using
supported parameter, drive-amplitude, carrier-frequency, or frame-frequency
targets. Preserve linked carrier/frame pairings where agreed terms require
them; do not alter a base file to evade admission.

Choose a grid that resolves the reported effect, accounting for known numerical
and frame limitations. `evidence_metric` is optional and guides point selection,
not permission to compute. A declared monotonic expectation adds a check; it
does not remove a check or prove the trend between samples. Use `sweep` for the
declared grid and `simulate` for a small follow-up point when appropriate.

Inspect the retained grid and per-point results, execution outcome, plots, and
relevant checks. Sweeps run serially and stop at a failed point; earlier outputs
remain available and later points may be unattempted. A rerun recomputes the
whole frozen grid, rather than continuing it. Resolve pending launches before
retrying.

When checks are requested, integrity/diagnostics cover applicable computed points;
segments and spectral refinement use bounded evidence points. Read actual
coverage and unavailable reasons in the report. Sampled convergence certifies
nothing at unsampled points; corners can exhaust the evidence-point bound. Do
not treat a plot or clean execution code as sufficient scientific evidence.

Let numerical search locate points within agreed scope, and use reasoning to
interpret constraints, uncertainty, and learning. Bracket resolution is not a
physical error bound. Record derived differences/brackets with source run IDs and
values. At a checkpoint assess whether another sample can resolve the question.
When the effect is below supported precision, or a crossing lies outside scope,
report it and propose a user decision rather than extending the range or
weakening the evidence. Assessments do not release named reviews.
