# Closed-system control

Control inputs can be compiled through Python or solved on CPU through `entanglix control`. The open `grape` backend uses JAX gradients through dynamiqs and SciPy SLSQP with explicit constraints. The optional `piccolo` backend runs Piccolo directly in Julia through a pinned project. Named replay, target and constraints checks run on request.

The [transmon example](../../examples/transmon-control/control.toml) declares a closed-system X gate in a rotating frame, with two quadratures of the charge drive. It uses the existing [system format](system-spec.md), including its basis and tensor order.

## Input

A `control.toml` requires `format_version = 1`, `system`, `[frame]`, ordered `[[channels]]`, `[goal]`, `[pulse]`, `[problem]`, and `[solver]`. The system and its referenced files stay beside the control input. The frame has the same syntax as an evolution frame: `frequencies_ghz`, explicit `rwa = true`, and `excitation_numbers` when the system's basis requires them.

Each channel has a unique `id`, a system `drive`, a positive `frequency_ghz`, a `phase_rad` (zero when omitted), and a nonnegative `amplitude_bound_ghz`. Several channels may use the same system drive with different carrier phases. Order is preserved in the exported matrices and bounds. A channel whose projected operator is zero is rejected by name; check its carrier frequency against the declared frame and its source operator. The operator's scale is preserved; changing that scale requires inverse scaling of amplitudes and their derivative bounds.

`[goal]` has `kind = "unitary"`, exactly one of `gate` or `matrix`, and `subspace = [[...]]`. The subspace is one ordered list of distinct **one-based flattened basis indices**, including for a composite system. The rightmost subsystem index varies fastest. Named gates are `I`, `X`, `Y`, `Z`, and `H`; other unitary goals use `matrix`. A matrix entry is a real number or `[real, imaginary]`. The matrix dimension must match the declared subspace, and it must be unitary within `1e-10` absolute tolerance. The goal is expressed in the control frame at the final time.

`[pulse]` has `kind = "zero_order"` and positive duration `T` in ns. `init` (`default`, `random`, or `zeros`) and a nonnegative integer `seed` may be declared. `[problem]` requires `template = "SmoothPulseProblem"` and an integer `N >= 2`. There are `N` knots and `N - 1` equal propagation intervals, of length `T/(N-1)`.

Optional problem fields are `Q`, `R`, `R_u`, `R_du`, `R_ddu`, `du_bound`, `ddu_bound`, `final_fidelity`, `free_dt`, and `free_phase`. An authored `[problem.options]` table is rejected; solver settings belong in `[solver.options]`. Weights and bounds are finite nonnegative scalars, matching the OSS schema. `du_bound` has units GHz/ns, and `ddu_bound` GHz/ns². `final_fidelity`, when declared, is the target in `[0, 1]`; no target is supplied otherwise. A solver must support the fields the problem declares.

Duration is fixed: authored `free_phase` and `free_dt` may only be false or omitted. The generated file writes `options.timesteps_all_equal = true` and a time-step bracket of relative half-width `1e-7` around `T/(N-1)`. This compensates for the upstream runner's handling of fixed steps; it is recorded in the generated file.

`[solver]` names `backend`, may name `fallback`, and carries an optional `options` table. Convergence settings such as `max_iter` and `tol` belong to that table. A positive top-level `timeout_seconds` is an optional runner limit. Neither table is included in the generated ProblemSpec. Plain Python export does not select a solver; the launch gate resolves and validates the solver in the runner environment before creating an attempt.

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

The stdio server offers `control(protocol, investigation_id=None, base=None)` and `rerun_control(run)`. Its configuration supplies the runs root and runner environment; neither is a tool argument. A successful call returns admission while a separate supervisor executes the CPU job. Pass its returned path to `read_run`, optionally with a bounded wait. Source run paths must resolve inside the configured runs root. The same admission, check, cancellation and rerun rules apply as through the CLI.

The versioned `skills/control/SKILL.md` procedure directs a research agent to this reference, distinguishes solver completion from candidate checks, and explains exact bases and recovery. Copy it and this reference into the research workspace following the [MCP setup](mcp.md#research-workspace). Runtime approval may be needed for these job tools; a tool approval does not approve investigation terms.

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

The existing differentiable engine evolves one maximally entangled input with a passive ancilla to evaluate this expression; the ancilla is only objective bookkeeping. It uses double precision, `rtol=1e-10`, `atol=1e-12` and Tsit5. Scalar `Q`, `R`, `R_u`, `R_du` and `R_ddu` are accepted guidance but ignored by this pure-fidelity baseline, and are listed in its result and trace. `final_fidelity` declares the requested target; solver completion does not certify it. Default or random initialization samples uniformly within 10% of the amplitude bounds and scales to satisfy declared derivative constraints. The trace records `effective_seed` for every random initialization, generating one when the input omits it. Reusing that seed fixes the initialization; `zeros` uses zero amplitudes and lists any seed as ignored. Unsupported semantics are refused before launch.

## Neutral pulse and candidate record

`artifacts/pulse.npz` contains `kind = "zero_order"` and numeric arrays `times_ns` (N knots, starting at zero and ending exactly at T) and `amplitudes_ghz` (channels × N, in the frozen input's channel order). For interval k, the amplitude is knot k's value; the last knot is a terminal value, not another propagation interval. The baseline fixes the first and last amplitudes to zero, matching the upstream smooth-pulse default. It adds no derivative value before or after the knot sequence.

For fixed step `dt = T/(N-1)`, constraints act on `diff(amplitudes)/dt` (N−1 first differences) and `diff(amplitudes, n=2)/dt²` (N−2 second differences). They include the amplitude jump to the terminal knot. No zero endpoint derivative is imposed. Amplitude bounds are SciPy bounds, and derivative bounds are linear constraints, never penalties. The [solver tests](../../tests/test_control_solver.py) recompute an active bound from the pulse and compare its objective and gradient with independent matrix-exponential propagation, including leakage.

The frozen `spec/` contains `control.toml`, model inputs and `problem.toml`. `result.toml` carries the pulse and trace artifact digests, actual solver/version, the solver-native `objective`, `metrics.solver_reported_fidelity` when supplied by the backend, ignored guidance fields and `termination` (`converged`, solver status/message, iterations, objective evaluations). `artifacts/solver-trace.toml` retains per-objective and per-iteration timings. A backend returns fidelity separately from its native objective, which may include regularizers; the runner never derives fidelity from that objective. These are solver-reported candidate quantities, not built-in check outcomes. A solve reaching its iteration cap can still be an executed, completed job with `converged = false`.

## Investigations and reruns

An investigation may permit `control` and `check` with an exact-match base. Its input digests cover the authored control, system and referenced assets, using the canonical names `control.toml` and `system.toml`; the generated `problem.toml` is not an authored base input. Control accepts no varied targets. A different solver or solver-options table needs a separately approved base, even when it generates the same ProblemSpec. See the [investigation reference](investigations.md) for complete terms and user decisions.

```sh
entanglix control control.toml --runs-root research_runs --investigation <investigation_id> --base <base_name>
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

A completed candidate has three relevant checks, shown at launch and by `read-run`:

```sh
uv run --locked --exact entanglix check RUN replay target constraints
```

With no names, `check RUN` only lists them. Named checks create one linked CPU verification attempt and report their individual outcomes. They never run automatically. Checks use the frozen system, control spec, generated ProblemSpec and neutral pulse; they verify input and artifact digests and preserve the source binding in the history. No solver installation, JAX or Julia is needed to check a retained candidate.

- `replay` evolves the neutral pulse on QuTiP's `vern9` engine and compares its independently calculated leakage-aware average gate fidelity with the solver's separately reported fidelity. Missing solver-reported fidelity makes this check inapplicable. The replay uses an entangled input with a passive ancilla to retain relative column phases and leakage, in the declared control frame. Its leakage is the average probability of leaving the declared computational subspace and has no pass/fail outcome.
- `target` compares that independently calculated fidelity with the declared `final_fidelity`. Without a target it is inapplicable. A below-target pulse can pass replay and fail only target. The replay-agreement tolerance does not lower the target.
- `constraints` checks channel amplitudes, duration, equal intervals and every declared first/second derivative bound. It includes the terminal knot in the discrete differences, adds no endpoint constraint and lists any uncovered constraints. Malformed pulse arrays fail this check; unsupported problem semantics are inapplicable with a reason.

Replay and target share one evolution when requested together. A replay of the closed RWA control model measures neither device-model accuracy nor the rotating-wave approximation. Additional evidence requires a separately requested lab-frame calculation.

The versioned corrector policy sets replay agreement to `1e-7` absolute fidelity; QuTiP uses `rtol=1e-11`, `atol=1e-13`, and `max_steps=100000` per interval. The fidelity tolerance exceeds the measured native/independent discrepancies in the bundled gate probe (about `1e-9`) while detecting incorrect solver reports. Constraint slack is `1e-7` GHz, GHz/ns, and GHz/ns² for amplitude, first and second differences respectively. This tolerates floating-point feasibility residuals without certifying stricter bounds. Timing limits come from the frozen ProblemSpec’s `free_dt` bracket: every step must lie within it, and total duration within `(N-1)` times its endpoints. Only roundoff slack, `32 × machine epsilon × max(returned duration, upper duration bound)`, is added to these timing limits. The target belongs to the input, not the policy. Constructed-pulse tests cover pass/fail, complex goals, non-contiguous subspaces, leakage, missing prerequisites and source tampering.
