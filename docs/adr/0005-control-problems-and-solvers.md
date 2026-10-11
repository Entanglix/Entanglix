# ADR-0005: Control Problems, Solvers, and Control Checks

**Date:** 2026-10-10
**Status:** Accepted
**Purpose:** fix how a control problem is stated and shared between solvers, where the solver is named, how solvers attach and what they report, how Piccolo is run, and which checks a control result offers. Field lists, formats, and command names are documented with the implementation.

---

## 1. Context

The [harness contract](../../HARNESS_CONTRACT.md) states control problems in amicode's ProblemSpec OSS, ships an open GRAPE-style baseline and a Piccolo adapter, and attaches private solvers through entry points (§1). [ADR-0001](0001-mandate.md) left the ProblemSpec arrangement to be decided before the `control` protocol is built.

[ADR-0003](0003-runs-verification-and-dispatch.md) made checks a separate request with no run-level verdict, put the frame in the computation's input, and has the runner stop a job by signalling its process group, with a fixed grace before forced termination.

The [control probe](../../tests/probes/m4_control.py) measured the boundaries on the workstation, on one transmon with three retained levels and an X gate on the lowest two:

- ProblemSpec OSS accepts only `"ipopt"` as `solver.backend`. Another name fails the schema and is refused by Piccolo. Piccolo reads `problem.options` as its own options.
- A system exported from an Entanglix model as explicit matrices solves under Piccolo. The returned pulse, replayed on the QuTiP reference engine, agrees with Piccolo's reported fidelity to 2.5e-9.
- Piccolo varies the time step unless a step bracket bounds it. An unbounded solve returned 19.968 ns for a declared 20 ns, and that pulse on the declared grid loses 1.5e-4 in fidelity. A bracket of relative half-width 1e-7 held the duration to 6e-11 ns.
- Calling Piccolo's ProblemSpec runner in Julia directly returns the same result as `amico-run`. Through `amico-run` the iteration cap reaches Piccolo only through the file. The runner's source takes the cap and solver options as arguments; the probe did not exercise them.
- `amico-run` starts Julia in a process group of its own. A termination signal did not end Julia; the wrapper's forced termination after 5 s did. Forced termination of the wrapper left Julia running with no terminal record.
- Piccolo reports convergence as "stopped before the iteration cap". It returns the pulse as a Julia object in a JLD2 file. Its source bounds the discrete derivatives of the pulse, and applies a second-derivative bound of its own when the problem declares none.
- The existing differentiable engine gives the gradient of the gate infidelity with respect to piecewise-constant amplitudes, matching finite differences on the reference engine to 4e-9. The optimisers that come with it are unconstrained; SciPy, already in the locked environment, has optimisers that take bounds and linear constraints.

The harness does only what the user requests.

## 2. Decisions

### 2.1 The control spec and the generated ProblemSpec

- The user writes one `control` spec. It references a system spec, declares the control frame and channels, carries the `goal`, `pulse`, and `problem` tables of ProblemSpec OSS under their own names, and names the solver and its options.
- From it the runner generates one ProblemSpec OSS file and freezes it in the run with its digest. The system is exported as explicit drift and drive matrices in the control frame. The file has no `solver` table.
- Every solver receives that file unchanged. Two runs solve the same problem when their generated files have the same digest.
- The export has one convention. With `H0` the control-frame drift in GHz and `D_j` the dimensionless operator of channel `j`, the file holds `H_drift = 2π H0` and `H_drives[j] = 2π D_j`. The amplitudes `u_j` are in GHz and `t` is in ns, so the propagator obeys `i dU/dt = (H_drift + Σ_j u_j(t) H_drives[j]) U`. Complex entries are written as real and imaginary parts, and channels keep their declared order.
- The generated file is TOML and conforms to ProblemSpec OSS at a recorded amicode revision.

**Rationale:** a ProblemSpec must contain its system. A user-written one would state the system a second time, or name a Piccolo system template that the harness cannot replay on its own model. Its `solver` table names an optimiser inside Piccolo, so it cannot record which solver ran. A file generated once and shared unchanged is what makes two solvers' results comparable, and it is the artifact to share.

### 2.2 The control model

- The control spec declares the frame as an evolution input does ([ADR-0003 §2.6](0003-runs-verification-and-dispatch.md#26-drives-and-frames)). Each control channel is a drive of the system with a declared carrier frequency and phase. Two channels on one drive, a quarter cycle apart, are its two quadratures.
- The existing frame transformation builds the drift and the channel operators under the declared rotating-wave approximation. Provenance records the frame and the removed terms.
- The goal is a unitary in the control frame at the final time, and the fidelity is calculated on the control-frame propagator. Evolution interfaces return lab-basis states, so a replay evaluates the goal in the control frame explicitly.
- The control model is closed. Dissipators are not part of it; the launch listing and the run record name those left out.

**Rationale:** the frame is a choice of computation, and the transformation that simulations already use yields the operators a solver needs. ProblemSpec OSS states state-transfer and unitary problems only.

### 2.3 The shared problem subset

- These fields define a valid candidate and mean the same to every solver:
  - a unitary goal on a declared computational subspace;
  - a fixed duration in equal steps;
  - a piecewise-constant pulse;
  - amplitude bounds per channel, and derivative bounds when declared;
  - the leakage-aware average gate fidelity, with no free phase;
  - a fidelity target, when declared. The harness supplies none.
- With `M` the block of the final propagator on the subspace, `V` the goal, and `d` the subspace dimension, the fidelity is `(Tr(M†M) + |Tr(V†M)|²) / (d(d+1))`.
- The generated file states the duration as equal steps with a step bracket narrow enough to fix it.
- A derivative bound limits the discrete derivative: the change in amplitude between neighbouring steps divided by the step length. The second derivative is the change in that quantity divided by the step length. The end points, including a final knot that is not a propagation interval, follow one convention for every solver and for the check, stated with the implementation.
- A solver enforces every declared bound or refuses the problem, naming the field. The open baseline enforces amplitude and derivative bounds.
- A solver may restrict its pulses further than the problem requires. Such a candidate remains valid.
- Regularisation weights, the initial guess, and the seed guide a solver. A solver may ignore them, and its result lists the fields it ignored.
- A solver refuses a problem that uses a field it does not support, naming the field. No field is dropped silently.
- A problem outside the subset may still be solved by a solver that supports it. The checks of §2.7 are defined for the subset and are reported as inapplicable elsewhere.

**Rationale:** the solvers use different formulations and regularisers, so they do not solve an identical optimisation. What can be the same is the definition of a valid candidate, and that is what the checks test. A bound changes which pulses are admissible, so it cannot be ignored. Derivative bounds are part of regular use, so the baseline and the checks handle them from the start.

### 2.4 Solvers

- A solver implements `capabilities()` and `solve(problem_spec, options)`, and returns the pulse, its objective, its termination report, and its trace. It may also report the fidelity it attained, as a quantity separate from its objective.
- Solvers are discovered through the `entanglix.solvers` entry-point group, the open ones included. Repository code never imports a private solver.
- The control spec names the solver and may declare a fallback. A named solver that is missing, with no fallback, is a configuration error before a run exists. The result records the solver that ran and its version.
- Solver options hold convergence settings such as an iteration cap and a tolerance. The harness adds none. Resource limits are the runner's and are optional.
- The termination report is the solver's own: the status it gives, its iteration count, and whether a cap was reached. It and the solver's reported objective are labelled as solver-reported and certify nothing.
- Every solver writes the pulse in one neutral form, readable without the solver: knot times in ns, amplitudes per channel in GHz, and the interpolation kind. A solver's native file is kept beside it.

**Rationale:** the problem file carries the system, so a solver needs no second system argument. A solver's convergence settings are not a resource limit, and absence of a limit stays valid. Piccolo's convergence flag shows that a solver's report can be a heuristic.

### 2.5 The open baseline

- The baseline is GRAPE-style: the gate infidelity is differentiated with respect to the piecewise-constant amplitudes on the existing differentiable engine, and a constrained optimiser from SciPy takes the steps.
- Amplitude bounds and derivative bounds are explicit constraints of that optimiser, subject to its numerical feasibility tolerances. They are not penalty terms. Whether a returned pulse holds its bounds is established by the requested check.
- The baseline adds no propagation code and no optimiser of its own. A dedicated propagator is adopted only on a measured need.

**Rationale:** the engine already gives a correct gradient of the gate objective. On a piecewise-constant pulse, amplitude and derivative bounds are bounds and linear constraints on the amplitudes, which an existing optimiser takes as explicit constraints. The baseline's cost per iteration is measured once it runs as a loop.

### 2.6 The Piccolo adapter

- The adapter is an ordinary job. It runs Piccolo's ProblemSpec runner in Julia directly, in the job's own process group.
- It passes the iteration cap and the tolerance as arguments, so the shared file is unchanged.
- It writes the neutral pulse inside Julia.
- The repository holds a pinned Julia project for it. A Piccolo run records the Julia version and the digest of that project's manifest.
- Julia is optional. The core and the default tests do not need it, and a Piccolo request without it is a configuration error that names what is missing.

**Rationale:** the runner already writes every record `amico-run` adds, and already stops a job's process group. A wrapper that moves Julia into another group, and needs longer than the runner's grace to stop it, would leave a solve running without a record. Acceptance of the generated file by `amico-run` remains interoperability evidence, not a dependency of the adapter.

### 2.7 Control checks

- The user requests them through `check`, as for any run. Each reports its own outcome.
- For a candidate there are three checks, all on the returned pulse:
  - Replay: the pulse is replayed in the control model on an engine other than the solver's. The check passes when the independently calculated fidelity agrees with the fidelity the solver reported.
  - Target: the independently calculated fidelity meets the problem's declared target. Without a declared target the check is inapplicable. A failure is reported as an unmet declared target.
  - Constraints: the amplitude bounds, the declared derivative bounds, and the duration hold. The report names any declared constraint the check does not cover.
- The replay also reports the leakage, without a pass or fail outcome.
- Numerical tolerances belong to the corrector policy. The fidelity target belongs to the problem.
- The replay does not measure the accuracy of the rotating-wave approximation or of the device model. A user who needs that evidence requests a lab-frame solve.
- A second solve of the same problem by another solver is a separate run that the user requests. A control run's record shows its problem digest. Different pulses or attained objectives do not invalidate a candidate.
- A private solver's candidate is checked by the same replay. No check depends on a private solver.

**Rationale:** these follow [ADR-0003 §2.3](0003-runs-verification-and-dispatch.md#23-verification-runs-and-checks-on-request). Agreement with a solver's report says the report is accurate, not that the gate is good, so the target is a check of its own. A comparison between solvers is evidence about two candidates; it does not certify either.

### 2.8 Investigations and tools

- `control` is a permitted operation of an investigation. A control job is in scope when its frozen inputs match a permitted base exactly. Usage, limits, the deadline, and stops apply as to any job.
- A rerun re-solves a frozen control problem as a new linked attempt.
- The MCP server offers `control` in the form of the existing job tools.

**Rationale:** exact matching uses the digests that admission already checks. A second solver is a second approved base.

Measurements: [control probe results](../../tests/probes/m4_control_results.json).

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| A ProblemSpec written by the user and adopted verbatim | States the system twice or through a template the harness cannot replay; its `solver` table cannot name the solver that ran |
| An Entanglix copy of the schema with `solver.backend` opened and a `solver.options` table | The file fails amicode's schema and Piccolo's parser, and the fork must follow a schema generated from Piccolo |
| Entanglix solver settings under `problem.options` | Piccolo reads that table as its own options |
| Piccolo through `amico-run` | Adds a second runtime, a third input file, and records the runner already writes; Julia leaves the job's process group and outlives a forced stop |
| A `solver` table in the shared file to carry Piccolo's iteration cap | The file would differ between solvers; the direct call takes the cap as an argument |
| A run-level verified verdict, or an automatic second solve | Checks run on request and report their own scope |
| Derivative bounds as guidance a solver may ignore, or refused by the baseline in its first version | They change which pulses are admissible, and they are part of regular use |
| An unconstrained optimiser with penalty terms for the bounds | The bounds would hold only approximately |
| Reads that find other runs with the same problem digest | Needs a search over runs; the digest is enough for a comparison the user requests |
| A Julia project path configured by the user | The Piccolo environment would not be reproducible from the repository |
| Reading Piccolo's native pulse file in the checks | It holds a Julia object, so the check would depend on the solver |
| A dedicated propagator for the baseline from the start | The existing engine gives the gradient; its cost is not yet measured in a loop |

## 4. Consequences

- The contract changes accordingly:
  - §1 describes the control spec and the generated ProblemSpec, names the solver in the control spec and the result, and gives `solve` the problem file and solver options, returning a termination report in place of a certificate.
  - §2 records the Julia version and project digest of a Piccolo run.
  - §3 states the control checks as a replay with independently calculated quantities, and a second solve as a separate requested run.
  - §5 adds `rerun_control`.
- ADR-0001's mandate table describes the control spec and Piccolo's role in these terms.
- The roadmap and checklist replace the open ProblemSpec choice, the `amico-run` adapter, and "both verified" with the control spec, the direct adapter, and candidates passing the checks requested.
- Acceptance of a generated ProblemSpec by `amico-run` stays with the interoperability block.
- The first gate problem, with its duration, bounds, and fidelity target, is recorded with its example.
- Tests cover: one generated file for every solver; the export convention against an independent replay; a solver name the registry lacks, with and without a fallback; a field outside a solver's support refused by name; a pulse from each open solver with an active derivative bound, the bound recomputed from the neutral pulse; ignored guidance fields listed; the iteration cap and tolerance reaching Piccolo as arguments; a stop ending a Piccolo solve with a terminal record; the duration held within tolerance; each control check passing and failing on a constructed pulse, including an accurately reported pulse below its target; a check requested outside the subset reported as inapplicable; and `control` admitted, refused, counted, and rerun inside an investigation.
