# Control specifications and model export

Control input validation and export are available through Python. Solver execution and control checks are not yet available.

The [transmon example](../../examples/transmon-control/control.toml) declares a closed-system X gate in a rotating frame, with two quadratures of the charge drive. It uses the existing [system format](system-spec.md), including its basis and tensor order.

## Input

A `control.toml` requires `format_version = 1`, `system`, `[frame]`, ordered `[[channels]]`, `[goal]`, `[pulse]`, `[problem]`, and `[solver]`. The system and its referenced files stay beside the control input. The frame has the same syntax as an evolution frame: `frequencies_ghz`, explicit `rwa = true`, and `excitation_numbers` when the system's basis requires them.

Each channel has a unique `id`, a system `drive`, a positive `frequency_ghz`, a `phase_rad` (zero when omitted), and a nonnegative `amplitude_bound_ghz`. Several channels may use the same system drive with different carrier phases. Order is preserved in the exported matrices and bounds. A channel whose projected operator is zero is rejected by name; check its carrier frequency against the declared frame and its source operator. The operator's scale is preserved; changing that scale requires inverse scaling of amplitudes and their derivative bounds.

`[goal]` has `kind = "unitary"`, exactly one of `gate` or `matrix`, and `subspace = [[...]]`. The subspace is one ordered list of distinct **one-based flattened basis indices**, including for a composite system. The rightmost subsystem index varies fastest. Named gates are `I`, `X`, `Y`, `Z`, and `H`; other unitary goals use `matrix`. A matrix entry is a real number or `[real, imaginary]`. The matrix dimension must match the declared subspace, and it must be unitary within `1e-10` absolute tolerance. The goal is expressed in the control frame at the final time.

`[pulse]` has `kind = "zero_order"` and positive duration `T` in ns. `init` (`default`, `random`, or `zeros`) and a nonnegative integer `seed` may be declared. `[problem]` requires `template = "SmoothPulseProblem"` and an integer `N >= 2`. There are `N` knots and `N - 1` equal propagation intervals, of length `T/(N-1)`.

Optional problem fields are `Q`, `R`, `R_u`, `R_du`, `R_ddu`, `du_bound`, `ddu_bound`, `final_fidelity`, `free_dt`, and `free_phase`. An authored `[problem.options]` table is rejected; solver settings belong in `[solver.options]`. Weights and bounds are finite nonnegative scalars, matching the OSS schema. `du_bound` has units GHz/ns, and `ddu_bound` GHz/ns². `final_fidelity`, when declared, is the target in `[0, 1]`; no target is supplied otherwise. A solver must support the fields the problem declares.

Duration is fixed: authored `free_phase` and `free_dt` may only be false or omitted. The generated file writes `options.timesteps_all_equal = true` and a time-step bracket of relative half-width `1e-7` around `T/(N-1)`. This compensates for the upstream runner's handling of fixed steps; it is recorded in the generated file.

`[solver]` names `backend`, may name `fallback`, and carries an optional `options` table. Convergence settings such as `max_iter` and `tol` belong to that table. A positive top-level `timeout_seconds` is an optional runner limit. Neither table is included in the generated ProblemSpec, and the exporter does not discover or run the named solver.

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

The exporter uses the shared rotating-wave transformation. It retains the source drive scale and applies the carrier phases, including the cosine carrier's half-amplitude convention. It does not subtract an additional energy origin from the drift. The exported `H_drift = 2π H0` and `H_drives[j] = 2π D_j` obey

```text
i dU/dt = (H_drift + Σ_j u_j(t) H_drives[j]) U
```

Here `H0` is in GHz, `D_j` is dimensionless, `u_j` is in GHz, and `t` is in ns. Complex matrix entries are `[real, imaginary]` pairs, with rows before columns. The generated file is `problem.toml` in amicode's ProblemSpec OSS format, at revision `cb717f57cb63fa292917188c8ca32f50ddcca539`, with no solver table.

Compilation takes an empty existing directory. It verifies the loaded input digests before writing anything, writes read-only copies of the system, referenced assets, canonicalized control input, and generated ProblemSpec, and returns paths and digests for the runner. The frozen control input refers to `system.toml`. Changing the named solver or its convergence settings leaves the generated ProblemSpec bytes unchanged. Compilation itself creates no run or terminal record.

## Numerical evidence and limits

The [export tests](../../tests/test_control.py) compare a complex-state, piecewise pulse propagated from the exported matrices with QuTiP evolution of the declared control model. The QuTiP outputs are explicitly transformed from the lab basis to the control frame; the absolute state-entry agreement limit is `1e-9`. Separate quadrature checks pin the sign of the imaginary channel and the single `2π` conversion. The example verifies the relation to the probe's normalized charge operators within `1e-13` absolute error.

These tests establish the export convention and bookkeeping. They do not measure device-model or rotating-wave approximation accuracy, optimize a gate, or establish that a declared target can be achieved.
