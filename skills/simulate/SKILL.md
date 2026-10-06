---
name: simulate
description: Run and inspect one Entanglix spectrum or dynamics computation, request relevant checks, or replay frozen inputs.
version: 0.1.0
---

# One computation

Read workspace `reference/operations/protocols.md` when writing a simulation or
check request, and `reference/operations/runs.md` for lifecycle and record fields.

Identify the reported quantity and choose spectrum or Lindblad dynamics. Declare
initial-state meaning, observables, times, drives, and lab or explicit RWA frame.
The frame is an approximation choice, not an automatic acceleration guarantee.
For a spectrum declare the lowest `levels` to report. Request CPU or CUDA through
the protocol's needs, and gradients only when needed; engine choice expresses
capability, not a measured performance crossover.

Validate the system, then call `simulate` with the protocol. In an investigation,
read effective terms first, use an approved base, and send only allowed `values`.
The base protocol stays unchanged. Missing or below-policy refinement does not
block computation, but the check listing explains what cannot run. A job tool
returns admission, not a completed scientific result.

Pass the returned path to `read_run`, optionally requesting a bounded wait;
the bare run ID is not a path. If the response was lost, resolve the recorded
launch path before retrying.
Inspect execution outcome, actual engine, metrics, artifacts, provenance, and the
check listing. Failed or interrupted work stays part of history. Cancel owned
work with `abort_run`; reconcile only after ownership and group-liveness checks
permit it, retaining uncertain usage.

Explain relevant checks and issue `check` only for explicit or standing requests:
integrity tests stored bookkeeping and state properties; segments compare local
same-frame convergence; spectral refinement compares reported low energies and
metrics; diagnostics report quantities without a pass/fail judgment. Passing
segments cover local windows and do not establish RWA or whole-window accuracy.
Compare changed frames or truncations through separately recorded simulations
when that evidence is needed.

`rerun_simulation` replays frozen inputs with current code/environment and records
whether provenance matches its parent. It creates a new attempt and needs new
requested checks. Findings distinguish results, check outcomes, and claims,
and name source runs when deriving a quantity across attempts.
