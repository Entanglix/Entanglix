# System specifications and reference evolution

Entanglix accepts version-1 TOML descriptions of component circuits, coupled subsystems, and finite-dimensional operator models. Construction runs on the CPU; reference-size evolution is available through QuTiP on the CPU and optional cuDensityMat and dynamiqs adapters. Runnable examples cover a [two-level operator model](../../examples/two-level/system.toml), [junction and shunt](../../examples/junction-shunt/system.toml), [LC circuit](../../examples/lc-circuit/system.toml), [lossy oscillator](../../examples/lossy-oscillator/system.toml), [fluxonium circuit](../../examples/fluxonium/system.toml), [two-node circuit](../../examples/two-mode-circuit/system.toml), [qubit–resonator composite](../../examples/qubit-resonator/system.toml), and [three-subsystem composite](../../examples/three-subsystem/system.toml).

Run `entanglix validate path/to/system.toml`. Success lists the model ID, basis, dimensions, each available operator and its meaning. For circuits, including composite leaves, it also shows the effective transformation, the authored node offset charges, applied coordinate offsets, any dropped extended-coordinate offsets, and oriented flux loops. When a multi-node spec omits the required transformation, the diagnostic prints scqubits' default matrix for the author to adopt. Invalid input prints a diagnostic to stderr and exits with code `64`. The Python operation `entanglix.models.load_system(path)` returns a `ConstructedSystem` with its GHz Hamiltonian, named dimensionless operators, drives, dissipators, dimensions, basis, and provenance. `qutip_hamiltonian()` returns a QuTiP `Qobj`; NumPy matrices and ordered `hamiltonian_terms`/`operator_factors` are available for engine adapters.

## Common fields and units

Every file has `format_version = 1`, an `id`, `[units]` with `energy = "GHz"` and `time = "ns"`, and a `[model]`. Values with physical units use `{ value = ..., unit = ... }` or `{ parameter = "name" }`. Parameters carry a value, unit, optional `source`, and optional uniform `{ kind = "uniform", lower = ..., upper = ... }` prior. Prior bounds inherit the parameter unit; construction uses the explicit value and does not sample the prior. Invalid bounds are rejected.

The fixed unit vocabulary is `GHz`, `ns`, `1/ns`, `fF`, `nH`, `Phi0`, and `dimensionless`. Each field accepts only its specified unit. Unlabelled Hamiltonian arrays inherit `[units].energy`; named operator arrays are dimensionless. Circuit and constructor loading requires scqubits' global unit setting to be `GHz`. A dissipator with rate `γ` in `1/ns` and operator `A` gives collapse operator `sqrt(γ) A`, with no `2π` on `γ`. Dynamics convert `H_GHz` to radians/ns once, as `2π H_GHz`.

## Component circuits

`kind = "circuit"` uses `nodes = [0, ..., N]`, `ground = 0`, named `[[model.branches]]`, and `[truncation]`. Branches connect declared nodes and have kind `C`, `L`, or `JJ`. C and JJ branches take exactly one of `charging_energy` (GHz) or `capacitance` (fF); JJ also takes `josephson_energy` (GHz). L branches take exactly one of `inductive_energy` (GHz) or `inductance` (nH). Capacitance converts as `E_C/h = e²/(2hC)`; inductance converts as `E_L/h = (Φ0/2π)²/(Lh)`. Supplied and converted values are retained in provenance.

Each quantized coordinate has a `thetaK` cutoff. Periodic cutoff `N` retains charge basis states `−N…N`; extended cutoff `N` retains `N` harmonic basis states. `retained_levels` then selects the lowest eigenstates. scqubits' truncated harmonic construction adds a spurious state at index `N/2`, so an extended circuit coordinate requires `retained_levels ≤ N/2`; this bound is necessary, not sufficient, because a junction can need a larger cutoff. Refine the construction cutoffs and retained levels separately before treating results as converged.

`offset_charges` names non-ground nodes with dimensionless `n − ng` values. Every node contributing to a periodic coordinate requires an explicit value, including zero. The adapter computes `q_theta = Mᵀ q_node`, drops extended-coordinate components, reverses the sign of retained periodic components for scqubits Circuit's `ngK`, and exports offset-relative charge. Dropped values appear separately in `validate` and provenance; applied coordinate offsets show zero for those coordinates. A multi-node circuit requires `transformation = [[...], ...]`; the current multi-node path supports periodic coordinates with an integer unimodular transformation. Mixed extended/periodic, free, and frozen multi-node coordinates produce diagnostics.

Every independent flux loop needs one `[[model.external_fluxes]]` entry with `branch` and either a literal `value` with `unit = "Phi0"`, including zero bias, or `value = { parameter = "name" }` naming a parameter in `Phi0`, with `unit` omitted. A flux parameter varies like any other parameter, for example through a sweep target `parameters.name`. The named closure branch fixes the positive direction from the branch's first node to its second. `validate` prints each branch's nodes, its applied value, and the parameter it comes from; the oriented loops, with any parameter name, and the transformation are retained in provenance.

For a single periodic coordinate, the available names are `charge`, `cos_phase`, and `sin_phase`. A single extended coordinate exports `charge` and `phase`; JJ circuits also export `cos_phase` and `sin_phase`. The current multi-node path exports `nodeK.charge`. A periodic raw phase operator is unavailable. `validate` lists each name and its physical meaning; names from explicit operator arrays are identified as declared by the spec.

## Coupled subsystems

`kind = "composite"` lists ordered `[[model.subsystems]]` with unique IDs. A subsystem can name a leaf TOML file beside the composite with `source = "qubit.toml"`; referenced files cannot themselves be composites, and their drives or dissipators must be declared on the composite. Constructors are `oscillator` (`frequency`, `retained_levels`), `transmon` (`josephson_energy`, `charging_energy`, `ng`, `cutoff`, `retained_levels`), and `fluxonium` (`josephson_energy`, `charging_energy`, `inductive_energy`, `flux`, `cutoff`, `retained_levels`). Energies are in GHz, `flux` in `Phi0`, and `ng` is dimensionless. The [qubit–resonator example](../../examples/qubit-resonator/system.toml) shows a leaf reference and an oscillator constructor.

Fluxonium's constructor `flux` follows scqubits' conventional sign. For the same single-junction circuit and node orientation, constructor `flux = +0.25 Phi0` corresponds to a circuit with its inductive closure branch directed `[1, 0]` and external-flux value `+0.25 Phi0`. Directing that branch `[0, 1]` reverses the signed `phase` and `sin_phase` expectations at the same positive value, although the spectra agree. Compare signed observables when matching constructor and circuit descriptions.

`[[model.interactions]]` gives a GHz `coefficient`, named `factors`, and optional `add_hermitian_conjugate`. A factor is `<subsystem>.<operator>`; factors are multiplied in listed order within each subsystem, with identities on omitted subsystems. The full Hamiltonian must be Hermitian. Subsystem list order sets the tensor order, and the rightmost index varies fastest in flattened arrays. `hamiltonian_terms` keeps local matrices and interaction factors rather than only the materialized CPU matrix. Composite operators use names such as `q.charge` and `r.position`, where oscillator `position = a + a†`.

Top-level `[[drives]]` entries give unique IDs and Hermitian dimensionless operator factors; a GHz amplitude is supplied at evolution time. Top-level `[[dissipators]]` entries give unique IDs, operator factors, and nonnegative rates in `1/ns`. Both can reference leaf or composite operator names. Unknown names report the available operators. The model retains ordered `drive_factors` and `dissipator_factors` alongside the materialized CPU matrices, with one factor per subsystem and identities for omitted subsystems. Dissipator rates remain in the `dissipators` mapping; the factor arrays do not include `sqrt(γ)`.

## Explicit operators and evolution

`kind = "operators"` declares one finite dimension, its basis, a numeric NPZ `array_file` beside the TOML file, the Hamiltonian key, and a map of operator names to NPZ keys. Matrices must be finite and square with the declared dimension; the Hamiltonian must be Hermitian. Object arrays and pickle loading are rejected. The declared basis is preserved, and the archive digest appears in provenance.

`entanglix.engines.evolve_cpu(system, initial_state, times_ns, observables=..., drive_amplitudes_ghz=...)` remains the QuTiP convenience function. The common Python interface is `EvolutionInput` with a normalized ket or density matrix, an increasing uniform output grid in ns, named finite-matrix observables, named drive amplitudes in GHz, and positive finite `rtol`/`atol`. A drive amplitude can be constant or an array with one value per output interval. Array entry `i` applies on `[times_ns[i], times_ns[i+1])`; every engine restarts at the boundary. Hamiltonian and drive GHz values receive one `2π` conversion to radians/ns, while dissipator rates remain in `1/ns`. The [CPU example](../../examples/cpu_reference.py) demonstrates analytic two-level decay and the coupled-system spectrum.

Pass the same input to `QuTiPIntegrator().evolve(system, request)`, `CuQuantumIntegrator().evolve(system, request)`, or `DynamiqsIntegrator(device_kind="cpu" | "gpu").evolve(system, request)`. The optional classes live in `entanglix.engines.cudensitymat` and `entanglix.engines.dynamiqs_jax`; importing the core package needs neither GPU stack. Each result contains sampled density matrices and expectations, output times and dimensions, plus engine, solver method, dtype, device, and diagnostics. The GPU adapters require a visible CUDA device and use complex128. cuDensityMat consumes factorized operator terms and records selected local Dense/Dia storage and the effective cuDensityMat data layer. Dynamiqs' `evolve` returns host NumPy arrays after checking device placement and dtype; `evolve_jax` preserves derivatives as described below. The [populated-loss example](../../examples/dynamics_reference.py) compares oscillator and coupled-system dynamics with loss-disabled evolution. The [simulation protocol](protocols.md) retains these computations through the runner; grid sweeps are not yet available.

QuTiP uses `vern9`, cuDensityMat uses `CuVern7`, and dynamiqs uses `Tsit5`. `EvolutionInput.max_steps` (also accepted by `evolve_cpu`) is a positive integer, defaulting to `100_000`. It maps to QuTiP/cuDensityMat's `nsteps` limit per output interval and dynamiqs' limit per solver call. For constant controls dynamiqs counts the entire grid as one solve; for interval-valued controls each restarted segment gets a fresh limit. This is an integration-step ceiling; it does not bound elapsed time. Diagnostics record `max_steps` and `step_limit_scope`. Capabilities use `device_kind = "cpu"` or `"cuda"`; dynamiqs translates its constructor's `"gpu"` selector to the public `"cuda"` value. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` in the launch environment before JAX initializes when sharing GPU memory with another stack; the adapter does not change the process environment.

### Carriers and rotating frames

`EvolutionInput.carriers` maps drive IDs to `Carrier(frequency_ghz, phase_rad=0)`.
The amplitude remains the scalar or interval-valued envelope in `drive_amplitudes_ghz`;
the lab-frame coefficient is `A(t) cos(2π f t + φ)`, evaluated continuously at the
solver's internal times. Carrier phase uses absolute time, including across envelope
boundaries and when the output grid starts after zero. Unlisted drives remain constant
within each interval. Carrier frequencies must be finite and positive; phases finite.

`EvolutionInput.frame = RotatingFrame(frequencies_ghz, rwa=True, excitation_numbers=...)`
declares a rotating-wave approximation. Frequencies map subsystem IDs to nonnegative
GHz values; omitted subsystems have zero frame frequency. A single-system model uses
its system ID. Constructed circuits and constructors use excitation numbers
`0, 1, ..., retained_levels-1` in their isolated eigenbasis. Explicit-operator
subsystems with nonzero frame frequency must supply one nonnegative integer excitation
number per basis state in
the `excitation_numbers` mapping. This declaration makes the diagonal generator
explicit; exact rotating-frame evolution without an RWA is not provided.

With `F = Σ f_s N_s`, the implementation uses `ρrot = exp(+i2πFt) ρlab exp(-i2πFt)`.
It retains static Hamiltonian matrix elements with zero `F` difference and subtracts
`F` from the Hamiltonian in GHz. For a cosine carrier, it retains drive transitions
with `F` differences `+f` and `-f`, multiplying them by `exp(-iφ)/2` and
`exp(+iφ)/2`, respectively. Unmodulated drives retain only zero-difference elements.
Frequency matching uses an absolute `1e-10 GHz` tolerance, a numerical equality
threshold rather than a physical near-resonance window. Local tensor factors remain
available to cuDensityMat. Dissipators must have one definite excitation-frequency
shift so their rotating phase cancels in the Lindblad term; operators mixing shifts
are rejected rather than silently secularized.

Initial states are specified in the lab basis at the first output time. All engines
transform rotating outputs back to that basis before returning states and expectations.
Diagnostics record frequencies, excitation numbers, removed Hamiltonian/drive frequency
sectors, and norms of the removed operators (Hamiltonian in GHz; drives dimensionless).
`evolve_cpu` accepts `carriers` and `frame` as keyword arguments. These conventions
are shared across all three adapters; scalar and interval-valued carrier-envelope
gradients are supported by dynamiqs with a fixed frame.

### Differentiable drives

`DynamiqsIntegrator.evolve_jax(system, request, drive_amplitudes_ghz=...)` returns JAX state arrays and named JAX expectation arrays on the requested device. Put model construction and the `EvolutionInput` outside the differentiated objective, then pass scalar or interval-valued JAX drive amplitudes through the override argument. These controls replace matching request amplitudes and may supply drives omitted from the request; the effective controls must be scalar or have one value per output interval. Configure `jax_enable_x64` before constructing JAX parameters, and use real, finite amplitudes. The [gradient example](../../examples/differentiable_reference.py) compares `jax.value_and_grad` with an independent QuTiP finite difference. Taking `.real` from a Hermitian expectation supplies the real scalar objective JAX requires. The derivative is in expectation/GHz. It is the derivative of the adaptive solution, whose step size follows the state's error only. When the state is stationary during an undriven interval and later intervals are driven, the solver takes few steps there, and the derivative with respect to that interval's amplitude can deviate from finite differences. In the [engine gradient probe](../../tests/probes/engine_gradient_results.json) the deviation reached 2e-5 for a transmon population at 1 ns intervals and stayed below 1e-7 at 0.5 ns; tighter tolerances and other gradient modes did not change it. This differentiates evolution with a fixed constructed Hamiltonian/basis; it does not differentiate scqubits construction, priors, or discrete truncation choices.

### Engine-selection policy

The bundled [TOML table](../../src/entanglix/engines/selection.toml) has `format_version = 1` and `table_id = "declared-needs-engines-v3"`. `entanglix.engines.selection.select_engine(system, request, available_engines=..., needs_gradients=..., device=..., batched=..., fallbacks=...)` evaluates declared needs without launching a job. The caller supplies verified engine availability.

| Declared need | Current eligible engine |
|---|---|
| No gradients, CPU or undeclared device | QuTiP CPU |
| No gradients, declared CUDA device | cuDensityMat |
| No gradients, CPU unavailable, explicit compatible fallback | cuDensityMat or dynamiqs on CUDA |
| Gradients | dynamiqs on the declared CPU or CUDA device |
| Batching | Refuse until an engine advertises it |

There is no harness-imposed dimension cap or pre-launch memory estimate. Missing
engines or unsupported capabilities cause `EngineSelectionError`. A fallback must
be explicitly named and satisfy the same needs; the decision records `fallback_from`
and its reason. QuTiP cannot satisfy a gradient request. [Simulation dispatch](protocols.md)
records the decision and checks the actual job environment. Allocation failures and
unexplained signal deaths follow the [run outcome rules](runs.md#execution-outcomes).

## Numerical checks for the bundled models

The [carrier/frame tests](../../tests/test_carriers_frames.py) compare analytic
carrier integrals, an independent lossy-oscillator ODE, phase-sensitive RWA evolution,
and a tensor-projected composite on CPU and CUDA. Their state and integration-refinement
limits are `1e-7` in Frobenius norm, with the existing `1e-8` physicality limits.
Carrier-gradient comparisons use QuTiP finite differences at `1e-3` and `1e-4 GHz`,
with relative limits `1e-3` and `1e-4`, respectively, and absolute `1e-6`. The
[readout example](../../examples/readout/README.md) owns full-window frame bounds
and their metric-resolution limits. These limits are specific to that fixture;
segment tolerances do not bound whole-window or RWA error.

The CPU fixture tests compare constructed circuits with independently configured scqubits or analytic references. For the LC circuit at a fixed basis and shared eigenvector gauge, projected operator matrices use a relative Frobenius residual limit of `1e-12`, with an absolute floor of `1e-12` in the operator's units. Across separately constructed models, phase-invariant operator diagonals or magnitudes use absolute limits of `1e-8` for the junction, two-mode circuit, and transmon offset checks; the direct fluxonium constructor phase check uses `2e-8`. The fluxonium circuit/constructor signed `phase` and `sin_phase` checks use `1e-7`, above observed residuals of about `3.6e-8` and `2.2e-8`. Excitation energies compared with a direct reference use an absolute limit of `2e-8 GHz`; the fluxonium fixture's observed residual is about `1.55e-8 GHz` between separate circuit and constructor calculations. These limits check representation agreement at the same cutoff, not physical convergence.

The bundled fluxonium, two-mode, and qubit–resonator fixtures vary cutoff and retained levels separately where applicable. Accepted refinements shift the first four excitation energies by no more than `1e-5 GHz` and tested ground-state expectations by no more than `1e-6` in their declared dimensionless units. The fluxonium check includes signed `sin_phase`; smaller fluxonium and two-mode cutoffs deliberately fail the energy limit. These are acceptance limits for the included fixtures; a new system needs its own convergence study before treating its numerical outputs as stable.

The qubit–resonator evolution regression uses a seeded random complex initial ket, times from 0 to 10 ns at 1 ns spacing, and a dense `900×900` Lindblad generator built directly from the Hamiltonian and collapse operators. An exact 1 ns matrix exponential supplies the reference at each output. The maximum density-matrix Frobenius difference must be below `1e-7`, one tenth of the `1e-6` cross-engine comparison limit. QuTiP's `vern9` replaced Adams after Adams exceeded the comparison limit on this case despite `rtol = 1e-10` and `atol = 1e-12`. A diagonalization-based solver is not used as the default, so tolerance refinement continues to exercise numerical integration.

The [dynamics acceptance matrix](../../tests/test_engine_acceptance.py) covers two-level decay/coherence/drives, oscillator decay/drives, and populated qubit–resonator decay/drives on all three engines. Dissipative cases use populated initial states and `γT = 0.1–1`; lossy outputs must differ from loss-disabled evolution by a Frobenius norm above `0.02`. Independent Lindblad exponentials use the authored constant or interval-valued drives. Each engine must agree with that reference below `1e-7` in density-matrix Frobenius norm and `1e-6` in dimensionless observables; pairwise engine differences are therefore bounded below the `1e-6` comparison limit. Trace and Hermiticity residuals must be below `1e-8`, and minimum density-matrix eigenvalues at least `-1e-8`, including loss-disabled outputs. Tightening `rtol/atol` from `1e-9/1e-11` to `1e-10/1e-12` and increasing the step ceiling from `200_000` to `400_000` must change states by less than `1e-7`.

Driven oscillator population changes must remain below `1e-6` when increasing from 6 to 8 Fock states; the intentionally insufficient 2-state basis exceeds that limit. Driven coupled-system photon populations must change by less than `1e-6` when refining the circuit cutoff, qubit levels, and resonator levels separately. The [gradient tests](../../tests/test_engine_dynamiqs.py) check nonzero derivatives against QuTiP central differences at `1e-3` and `1e-4 GHz`, plus the analytic closed-system Rabi derivative; accepted derivative differences use `rtol = 1e-4`, `atol = 1e-6 expectation/GHz`. These limits apply to the bundled fixtures and do not establish accuracy for a physical device or an arbitrary new system.

The [dimension-64 agreement case](../../tests/test_engine_acceptance.py) compares a short 8 × 8 composite evolution with a complex initial state and active loss across the engines at `1e-7` in state Frobenius norm and `1e-6` in observable values. The dimension-48 exact regression remains a separate engine test. These tested sizes do not impose a ceiling.
