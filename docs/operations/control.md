# Closed-system control

Control inputs can be compiled through Python or solved on CPU through `entanglix control`. The open `grape` backend differentiates a matrix-exponential propagator in JAX and uses SciPy SLSQP with explicit constraints. The optional `piccolo` backend runs Piccolo directly in Julia through a pinned project. Named replay, target and constraints checks run on request.

The [transmon example](../../examples/transmon-control/control.toml) declares a closed-system X gate in a rotating frame, with two quadratures of the charge drive. It uses the existing [system format](system-spec.md), including its basis and tensor order.

## Input

A `control.toml` requires `format_version = 1`, `system`, `[frame]`, ordered `[[channels]]`, `[goal]`, `[pulse]`, `[problem]`, and `[solver]`. The system and its referenced files stay beside the control input. The frame has the same syntax as an evolution frame: `frequencies_ghz`, or `transitions` as below, explicit `rwa = true`, and `excitation_numbers` when the system's basis requires them.

An optional `[parameters]` table overrides declared system parameters by name, as in a simulation, for example `Csh = 30.0`. The run's frozen `system.toml` holds the values used; the authored files are unchanged.

Each channel has a unique `id`, a system `drive`, a positive `frequency_ghz` or `follows`, a `phase_rad` (zero when omitted), and a nonnegative `amplitude_bound_ghz` unless `[rabi_bounds]` derives it. Several channels may use the same system drive with different carrier phases. Order is preserved in the exported matrices and bounds. A channel whose projected operator is zero is rejected by name; check its carrier frequency against the declared frame and its source operator. The operator's scale is preserved; changing that scale requires inverse scaling of amplitudes and their derivative bounds.

`[goal]` has `kind = "unitary"`, exactly one of `gate` or `matrix`, and `subspace = [[...]]`. The subspace is one ordered list of distinct **one-based flattened basis indices**, including for a composite system. The rightmost subsystem index varies fastest. Named gates are `I`, `X`, `Y`, `Z`, and `H`; other unitary goals use `matrix`. A matrix entry is a real number or `[real, imaginary]`. The matrix dimension must match the declared subspace, and it must be unitary within `1e-10` absolute tolerance. The goal is expressed in the control frame at the final time.

`[pulse]` has `kind = "zero_order"` and positive duration `T` in ns. `init` (`default`, `random`, or `zeros`) and an integer `seed` from 0 to 2**63 - 1 may be declared. `[problem]` requires `template = "SmoothPulseProblem"` and an integer `N >= 2`. There are `N` knots and `N - 1` equal propagation intervals, of length `T/(N-1)`.

Optional problem fields are `Q`, `R`, `R_u`, `R_du`, `R_ddu`, `du_bound`, `ddu_bound`, `final_fidelity`, `free_dt`, and `free_phase`. An authored `[problem.options]` table is rejected; solver settings belong in `[solver.options]`. Weights and bounds are finite nonnegative scalars, matching the OSS schema. `du_bound` has units GHz/ns, and `ddu_bound` GHz/ns². `final_fidelity`, when declared, is the target in `[0, 1]`; no target is supplied otherwise. A solver must support the fields the problem declares.

Duration is fixed: authored `free_phase` and `free_dt` may only be false or omitted. The generated file writes `options.timesteps_all_equal = true` and a time-step bracket of relative half-width `1e-7` around `T/(N-1)`. This compensates for the upstream runner's handling of fixed steps; it is recorded in the generated file.

`[solver]` names `backend`, may name `fallback`, and carries an optional `options` table. Convergence settings such as `max_iter` and `tol` belong to that table. A positive top-level `timeout_seconds` is an optional runner limit. Neither table is included in the generated ProblemSpec. Plain Python export does not select a solver; the launch gate resolves and validates the solver in the runner environment before creating an attempt.

### Named transitions and Rabi-rate bounds

A frame may name a transition in place of a frequency: `transitions = { <subsystem> = [lower, upper] }` sets that subsystem's frame frequency, in GHz, to the energy difference between two of its levels. The difference is computed at launch from the constructed model, with `[parameters]` and varied values applied. Levels are **zero-based** indices of the subsystem's isolated eigenstates, ordered by energy; they are not the one-based flattened indices of `goal.subspace`. The subsystem must be built in isolated eigenstates (circuits and named subsystems, not explicit-operator models), and `lower` must lie below `upper`. A channel may declare `follows = "<subsystem>"` in place of `frequency_ghz`; its carrier then uses that subsystem's frame frequency.

An optional `[rabi_bounds]` table names a frame transition by `subsystem` and declares any of `amplitude_ghz` (GHz), `du_ghz_per_ns` (GHz/ns) and `ddu_ghz_per_ns2` (GHz/ns²) as nominal Rabi-rate bounds. Each channel's raw bound is the Rabi rate divided by \|⟨lower\|D\|upper⟩\|, where D is the channel's authored drive operator. Under the cosine-carrier convention, an amplitude u drives that transition at Rabi rate u·\|⟨lower\|D\|upper⟩\|. The drive must act on that subsystem alone, its element must be nonzero, and its exported drive must keep the transition under the frame and carrier; otherwise the launch is refused, naming the channel. The ProblemSpec holds one `du_bound` and one `ddu_bound`, so derived values must agree across channels within relative tolerance `1e-9`. Otherwise the launch is refused, listing each channel's value, and an explicit raw bound is needed. Quantities not declared in `[rabi_bounds]` stay explicit.

A declared number next to its derivation must agree within the same tolerance; the declared frame frequency or bound is then used, and following carriers take their frame's number. The frozen `control.toml` keeps both the declaration and the resolved numbers. The launch listing shows them under `resolved`, and the generated ProblemSpec holds only numbers.

### Evaluation samples

An optional `[evaluation]` table declares the conditions for the requested uncertainty diagnostic. It belongs to the control input, so an approved base fixes it for every candidate, and it is not part of the generated ProblemSpec. `quantities` maps each sampled target, `parameters.<name>` or `model.offset_charges.<node>`, to `drifting` (drive frequencies are not recalibrated for it) or `calibrated` (the named transition and its following carriers are re-resolved for it). `samples` is a list of distinct, nonempty tables of absolute values for declared quantities, for example `{ "parameters.EJ" = 14.985, "model.offset_charges.1" = 0.5 }`. Quantities a sample omits keep the run's values. An optional `[evaluation.refinement.<subsystem>]` uses the simulation refinement form (`retained_levels_step`, `cutoff_steps`). Targets are checked against the system at launch.

## Export and preview

```python
from pathlib import Path
from entanglix.protocols.control import load_control, compile_control

loaded = load_control(Path("control.toml"))
print(loaded.listing)

directory = Path("compiled-control")
directory.mkdir()
compiled = compile_control(Path("control.toml"), directory)
print(compiled["problem_sha256"])
```

The preview records the control frame, channel order, duration, dimensions, named solver, ProblemSpec digest, schema revision, and every omitted dissipator. The control model is closed: all dissipators are removed before frame preparation. The source system is preserved, including its original dissipators, in the frozen inputs.

The exporter uses the shared rotating-wave transformation. It retains the source drive scale and applies the carrier phases, including the cosine carrier's half-amplitude convention. The export subtracts `c I`, where `c` is the real first diagonal entry of the control-frame drift, and records `energy_origin_ghz` in the launch listing; this changes only global phase, preserving density matrices and the declared gate fidelity. The exported `H_drift = 2π (H0 − c I)` and `H_drives[j] = 2π D_j` obey

```text
i dU/dt = (H_drift + Σ_j u_j(t) H_drives[j]) U
```

Here `H0` is in GHz, `D_j` is dimensionless, `u_j` is in GHz, and `t` is in ns. Complex matrix entries are `[real, imaginary]` pairs, with rows before columns. The generated file is `problem.toml` in amicode's ProblemSpec OSS format, at revision `cb717f57cb63fa292917188c8ca32f50ddcca539`, with no solver table.

Compilation takes an empty existing directory. It verifies the loaded input digests before writing anything, writes read-only copies of the system, referenced assets, canonicalized control input, and generated ProblemSpec, and returns paths and digests for the runner. The frozen control input refers to `system.toml`. Changing the named solver or its convergence settings leaves the generated ProblemSpec bytes unchanged. Compilation itself creates no run or terminal record.

## MCP and research procedure

The stdio server offers `control(protocol, investigation_id=None, base=None, values=None, parent_run_ids=None)` and `rerun_control(run)`. Its configuration supplies the runs root and runner environment; neither is a tool argument. A successful call returns admission while a separate supervisor executes the CPU job. Pass its returned path to `read_run`, optionally with a bounded wait. Source run paths must resolve inside the configured runs root. The same admission, check, cancellation and rerun rules apply as through the CLI.

The versioned `skills/control/SKILL.md` procedure directs a research agent to this reference, distinguishes solver completion from candidate checks, and explains approved bases, recovery and the co-design procedure. Copy it and this reference into the research workspace following the [MCP setup](mcp.md#research-workspace). Runtime approval may be needed for these job tools; a tool approval does not approve investigation terms.

## Numerical evidence and limits

The [export tests](../../tests/test_control.py) compare a complex-state, piecewise pulse propagated from the exported matrices with QuTiP evolution of the declared control model. The QuTiP outputs are explicitly transformed from the lab basis to the control frame; the absolute state-entry agreement limit is `1e-9`. Separate quadrature checks pin the sign of the imaginary channel and the single `2π` conversion. The example verifies the relation to the probe's normalized charge operators within `1e-13` absolute error.

These tests establish the export convention and bookkeeping. They do not measure device-model or rotating-wave approximation accuracy, optimize a gate, or establish that a declared target can be achieved.

## Launch and solver selection

For `grape`, install `differentiable` in the runner environment by setting `ENTANGLIX_RUNNER_EXTRAS=differentiable` before its first job, or use the runner extras already configured for your workstation. The runner keeps one exact extras profile and refuses a different set; see [runner environment](runs.md). The calling CLI does not itself need JAX.

```sh
export ENTANGLIX_RUNNER_EXTRAS=differentiable
uv run --locked --exact entanglix control examples/transmon-control/control.toml --dry-run
uv run --locked --exact entanglix control examples/transmon-control/control.toml --runs-root research_runs
```

On the documented GPU workstation the established profile is `cuquantum,differentiable-cuda12`; keep that profile. A control job still runs on CPU, with CUDA hidden. `--detach`, `read-run --wait`, timeouts, cancellation and interrupted-attempt reconciliation follow the [run reference](runs.md). Investigation admission and linked control reruns are available through the CLI and [stdio MCP server](mcp.md).

Solvers are discovered through `entanglix.solvers` entry points. Each entry loads a backend factory implementing `capabilities()` and `solve(problem_spec, options)`. The `problem_spec` argument is a `Path` to the frozen, unchanged `RUN/spec/problem.toml`; `options` is the separate solver-options table. Backends may derive `RUN` from that path and list retained native files in `Solution.trace["native_artifacts"]` as names mapped to paths relative to `RUN`, for the job to digest and record. This gives every backend the same file, including one that passes it to another process. The built-in entries are `grape` and `piccolo`. An optional `validate(problem_spec, options)` method checks support before admission and again in the job; any returned runtime provenance is included in the environment hash. A missing named solver fails before any attempt is created, unless a declared `fallback` exists. A fallback is used only when the named entry is absent; it does not hide an installed solver's unsupported input or execution failure. The preview, manifest and result name the requested and actual solver, its distribution version and whether fallback was used.

For `grape`, `solver.options.max_iter` is a positive integer mapped to SLSQP `maxiter`, and `solver.options.tol` is a finite positive value mapped to `ftol`. Unknown options are refused by name. Only declared settings are passed; otherwise SLSQP's defaults apply (currently 100 iterations and `1e-6`). The trace distinguishes declared settings from these upstream defaults. `timeout_seconds`, if present, is enforced independently by the runner.

The baseline supports raw unitary problems, fixed equal steps, zero-order pulses, and amplitude, `du_bound` and `ddu_bound` constraints. It optimizes leakage-aware average gate fidelity on the declared subspace, with projected evolution `M` and goal `V`:

```text
F = (Tr(M†M) + |Tr(V†M)|²) / (d (d + 1))
objective = 1 - F
```

The baseline propagates the computational subspace's columns with `jax.scipy.linalg.expm` inside `jax.lax.scan`, in double precision on CPU. Each interval uses `exp(-i dt (H_drift + Σ_j u_j H_drives[j]))` directly on the exported rad/ns matrices, with no additional `2π` conversion. This is the exact-arithmetic propagator for the declared closed, piecewise-constant model; its numerical evaluation and derivatives carry floating-point error. The trace names this propagator and the JAX version. Scalar `Q`, `R`, `R_u`, `R_du` and `R_ddu` are accepted guidance but ignored by this pure-fidelity baseline, and are listed in its result and trace. `final_fidelity` declares the requested target; solver completion does not certify it. Default or random initialization samples uniformly within 10% of the amplitude bounds and scales to satisfy declared derivative constraints. The trace records `effective_seed` for every random initialization, generating one when the input omits it. Reusing that seed fixes the initialization; `zeros` uses zero amplitudes and lists any seed as ignored. Unsupported semantics are refused before launch.

## Neutral pulse and candidate record

`artifacts/pulse.npz` contains `kind = "zero_order"` and numeric arrays `times_ns` (N knots, starting at zero and ending exactly at T) and `amplitudes_ghz` (channels × N, in the frozen input's channel order). For interval k, the amplitude is knot k's value; the last knot is a terminal value, not another propagation interval. The baseline fixes the first and last amplitudes to zero, matching the upstream smooth-pulse default. It adds no derivative value before or after the knot sequence.

For fixed step `dt = T/(N-1)`, constraints act on `diff(amplitudes)/dt` (N−1 first differences) and `diff(amplitudes, n=2)/dt²` (N−2 second differences). They include the amplitude jump to the terminal knot. No zero endpoint derivative is imposed. Amplitude bounds are SciPy bounds, and derivative bounds are linear constraints, never penalties. The [solver tests](../../tests/test_control_solver.py) recompute an active bound from the pulse and compare its objective and every knot's gradient with SciPy matrix-exponential propagation, including leakage, an undriven interior interval, and an all-zero pulse. They also compare a solved pulse's fidelity with the independent QuTiP ODE replay.

The frozen `spec/` contains `control.toml`, model inputs and `problem.toml`. `result.toml` carries the pulse and trace artifact digests, actual solver/version, the solver-native `objective`, `metrics.solver_reported_fidelity` when supplied by the backend, ignored guidance fields and `termination` (`converged`, solver status/message, iterations, objective evaluations). `artifacts/solver-trace.toml` retains per-objective and per-iteration timings. A backend returns fidelity separately from its native objective, which may include regularizers; the runner never derives fidelity from that objective. These are solver-reported candidate quantities, not built-in check outcomes. A solve reaching its iteration cap can still be an executed, completed job with `converged = false`.

## Investigations and reruns

An investigation may permit `control` and `check` with an approved base. Its input digests cover the authored control, system and referenced assets, using the canonical names `control.toml` and `system.toml`; the generated `problem.toml` is not an authored base input. A control base may vary `parameters.<name>`, by range or values, and `pulse.seed`, by non-negative integer values; both must be declared in the base control input. Pass candidate values at launch with `--vary TARGET=VALUE`, or `values` over MCP; the authored base is unchanged, and values left out keep the base's own. The solver, its options and every other input stay exact: a different solver or solver-options table needs a separately approved base, even when it generates the same ProblemSpec. Each run freezes the values used, the parameters and seed in `spec/control.toml` and the overridden system in `spec/system.toml`. `rerun-control` reuses the original base and values under the current approved revision. See the [investigation reference](investigations.md) for complete terms and user decisions.

```sh
entanglix control control.toml --runs-root research_runs --investigation <investigation_id> --base <base_name>
entanglix control control.toml --runs-root research_runs --investigation <investigation_id> --base <base_name> --vary parameters.Csh=30.0 --vary pulse.seed=2
entanglix check research_runs/<system_id>/<run_id> replay target constraints
entanglix rerun-control research_runs/<system_id>/<run_id> --detach
```

The solve and each requested check are separate admitted jobs. They count toward the investigation's usage and optional limits; its stop and deadline terminate their process groups. Requested checks remain client-issued. Findings report checks run, requested checks not run and relevant checks not run, including for control candidates.

An investigation run retains original base bytes under `spec/base/`, including the original system filename when the authored control uses another name. `rerun-control` verifies the frozen input digests and creates a new attempt linked to the source. It retains investigation membership and the runs root, and passes admission under the current approved revision. Standalone control reruns need no investigation. Reruns use current code and environment, recording whether they match the source; they do not require a provenance match. The [run reference](runs.md) describes detached supervision and recovery.

## Piccolo

Install Julia 1.13 and instantiate the repository's pinned project once before using `piccolo`:

```sh
julia --startup-file=no --project=src/entanglix/solvers/julia -e 'using Pkg; Pkg.instantiate()'
uv run --locked --exact entanglix control examples/transmon-control/piccolo.toml --runs-root research_runs
```

The project pins Piccolo 2.1.0, with transitive dependencies (including JLD2) in its manifest. No Python extra is needed by this backend. Keep any established runner extras profile unchanged. A missing Julia executable is refused before an attempt exists. The manifest is resolved for Julia 1.13.0; its digest, the project digest and the observed Julia version are recorded under `environment.solver_runtime`. A change between the launch gate and execution is refused.

Only declared `solver.options.max_iter` and `solver.options.tol` are passed as arguments to `Piccolo.Specs.solve_spec`; the shared ProblemSpec is unchanged. Without them, upstream defaults apply: 500 iterations and Ipopt tolerance `1e-8`. Integer `tol` values are transported as floating-point values to Julia. The trace records declared and effective settings. Piccolo's `converged` flag means only that it stopped before its iteration cap; its native runner does not expose Ipopt's termination status. The separately reported fidelity is retained. The native objective is the last iteration's `obj` value from Piccolo's progress output, printed to eight decimal places in scientific notation; it includes regularisation and is not an infidelity.

Piccolo honours `init = "random"` with its declared seed, defaulting the seed to zero when omitted. `default` and `zeros` initialise to zero and list a declared seed as ignored. Its native template supplies additional regularisation and a second-derivative bound of 1 GHz/ns² when one is not declared. These upstream restrictions do not replace the input's declared bounds.

The Julia entry exports the stored N knot values without shifting or resampling them. Interval k uses knot k, and the terminal knot participates in the discrete differences. The narrow equal-step bracket holds the declared duration within its relative half-width `1e-7`; returned times are preserved. Native `artifacts/piccolo/result.toml` and the JLD2 pulse are retained with digests beside the neutral pulse. `options.toml` is transport for the settings recorded in the frozen control spec and solver trace. Julia stays in the job's process group, so cancellation and timeout terminate it through the ordinary runner.


## Requested checks

A completed candidate has three relevant checks, five when the run declares an evaluation table, shown at launch and by `read-run`:

```sh
uv run --locked --exact entanglix check RUN replay target constraints
```

With no names, `check RUN` only lists them. Named checks create one linked CPU verification attempt and report their individual outcomes. They never run automatically. Checks use the frozen system, control spec, generated ProblemSpec and neutral pulse; they verify input and artifact digests and preserve the source binding in the history. No solver installation, JAX or Julia is needed to check a retained candidate.

- `replay` evolves the neutral pulse on QuTiP's `vern9` engine and compares its independently calculated leakage-aware average gate fidelity with the solver's separately reported fidelity. Missing solver-reported fidelity makes this check inapplicable. The replay uses an entangled input with a passive ancilla to retain relative column phases and leakage, in the declared control frame. Its leakage is the average probability of leaving the declared computational subspace and has no pass/fail outcome.
- `target` compares that independently calculated fidelity with the declared `final_fidelity`. Without a target it is inapplicable. A below-target pulse can pass replay and fail only target. The replay-agreement tolerance does not lower the target.
- `constraints` checks channel amplitudes, duration, equal intervals and every declared first/second derivative bound. It includes the terminal knot in the discrete differences, adds no endpoint constraint and lists any uncovered constraints. Malformed pulse arrays fail this check; unsupported problem semantics are inapplicable with a reason.
- `uncertainty` evaluates the returned pulse, unchanged and not renormalised, on each declared evaluation sample. The figure of merit is the replay check's fidelity and leakage, on QuTiP. A calibration model applies only the sample's calibrated changes, and the named transition and following carriers are resolved on it; the model with every change applied is evaluated at those frequencies. Without calibrated changes, the run's own frame and carriers are used. Logical states are each sample's own eigenstates, phase-aligned to the nominal ones by charge-basis overlap. Each sample reports its values, calibrated quantities, frame and carrier frequencies, dimensions, integration settings, fidelity, leakage and state overlaps. The nominal model is reported separately and is not part of the minimum, which is the lowest declared-sample fidelity with its index in table order. There is no pass/fail outcome.
- `uncertainty_refinement` re-evaluates the nominal model and every sample at the declared refinement, keeping each one's resolved drive. Each refined row keeps the sample's identity and reports its fidelity and leakage changes; the item reports the refined minimum, the unrefined minimum and the change in the minimum. `check RUN … --refinement FILE`, with `[refinement.<subsystem>]` tables, replaces the declared refinement, and the replacement is frozen in the verification run.
- The two uncertainty diagnostics are inapplicable, with a reason, without an evaluation table; for anything other than a single periodic circuit coordinate, where state correspondence is unsupported; and for calibrated samples without a named frame transition followed by every carrier. Refinement is also inapplicable without a declaration, below the policy minimum steps, or with frame `excitation_numbers` other than the canonical `0, 1, …`, which refinement extends.

Replay and target share one evolution when requested together. A replay of the closed RWA control model measures neither device-model accuracy nor the rotating-wave approximation. Additional evidence requires a separately requested lab-frame calculation.

The versioned corrector policy sets replay agreement to `1e-7` absolute fidelity; QuTiP uses `rtol=1e-11`, `atol=1e-13`, and `max_steps=100000` per interval. The fidelity tolerance exceeds the measured native/independent discrepancies in the bundled gate probe (about `1e-9`) while detecting incorrect solver reports. Constraint slack is `1e-7` GHz, GHz/ns, and GHz/ns² for amplitude, first and second differences respectively. This tolerates floating-point feasibility residuals without certifying stricter bounds. Timing limits come from the frozen ProblemSpec’s `free_dt` bracket: every step must lie within it, and total duration within `(N-1)` times its endpoints. Only roundoff slack, `32 × machine epsilon × max(returned duration, upper duration bound)`, is added to these timing limits. The target belongs to the input, not the policy. Constructed-pulse tests cover pass/fail, complex goals, non-contiguous subspaces, leakage, missing prerequisites and source tampering.

A sample whose logical state overlaps its nominal state by less than `0.7` is listed under `ambiguous_samples` and still evaluated. The uncertainty diagnostics are closed-system evaluations under the rotating-wave approximation, and they say nothing about conditions that were not sampled.

## Co-design comparison

A co-design investigation composes these operations: screen device candidates with `simulate` or `sweep`, solve chosen candidates inside one approved control base with the screening runs as `parent_run_ids`, and evaluate returned pulses with requested checks and the uncertainty diagnostic. There is no comparison view. The investigation report's attempt table shows each control attempt's varied values, `seed`, derived `transition` and solver `termination`; record a comparison as a finding linked to its runs and checks.

**Budget policy.** The approved base fixes the solver settings and the allowed seed set. The evidence requirements declare the seeds each candidate requires and how any additional attempts are allocated; these are recorded terms, which the harness does not enforce. The harness never adds seeds or solves. Solvers stop by different rules, so a comparison reports differing termination, such as a run that reached its iteration cap with `converged = false`.

The following guidance is for findings; the harness does not certify it.

- Measured performance of the returned pulses under the declared search policy is distinct from a device's attainable performance.
- An independently checked admissible pulse establishes a lower bound on what its device attains, within the declared model and the reported numerical evidence. It is not a hardware guarantee. Finite search establishes no upper bound.
- Consider every independently checked pulse known for each candidate, including pulses from other bound policies and other solvers. Each must satisfy the comparison's constraints and be evaluated, on the compared quantity, under the comparison's declared conditions.
- Report seed spread and refinement change as evidence, not as thresholds that resolve a comparison. A small spread can mean every start found the same poor solution.
- Keep spreads per solver; solvers optimise different objectives under different stopping rules.
- State how the selected pulse and the ranked quantity were chosen.
- Report a comparison the evidence does not support as unresolved; that is a valid endpoint.

**Scope of a comparison.** The operations above compose into investigations for every supported model. The uncertainty diagnostic currently supports a single periodic circuit coordinate. For other models it is inapplicable, with its reason, and those models can still be modelled, solved and checked. A comparison's conclusions are conditional on its declared model, samples, constraints and search policy.

- State what was not sampled. Evaluation samples are quasistatic offsets, not time-dependent noise, and a calibrated sample assumes perfect recalibration of the named transition.
- Equal Rabi-rate bounds, or equal raw coefficients, do not imply equal hardware drive voltage or power.
- A ranking found under one duration, bound policy or truncation can change under another; check it there before generalising it.
- Report the evaluated candidates. Performance between them is not evaluated.
