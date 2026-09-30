# M1 Guide — System Models and Numerical Engines

**Status:** Completed implementation guide. The [checklist](checklist.md#m1--model--engines-blocks-12) owns completion; the [operational reference](operations/system-spec.md) and [environment evidence](environment.md) own supported behavior and results. Fixture recommendations below record the implementation approach rather than a second acceptance authority.
**Scope:** M1, covering Block 1 (System Spec and Model Adapter) and Block 2 (Engines v0).
**Authority:** [roadmap](roadmap.md#block-1--system-spec-and-model-adapter), [accepted mandate](adr/0001-mandate.md#3-decision-0--foundation-choices), and [harness contract](../HARNESS_CONTRACT.md#1-objects-and-interfaces). [The checklist](checklist.md#m1--model--engines-blocks-12) owns completion status and acceptance evidence.
**Audience:** the owner and implementation agents. This is a planning document; operational docs and production agents must work without it.

## 1. Outcome and boundary

M1 establishes a tested path from a description of a superconducting system to spectra and reference-size dynamics. The same model must retain its meaning across the QuTiP CPU, cuDensityMat, and dynamiqs adapters.

At completion, a developer or expert user can:

1. Describe a system using circuit components, coupled subsystems, or explicit operators.
2. Validate its parameters, units, bases, truncations, interactions, drives, and dissipation through `entanglix validate <system.toml>` and its Python backing operation.
3. Construct operators and export equivalent engine inputs without changing subsystem order, normalization, or units.
4. Exercise static, driven, closed, and open-system evolution through the engine API and worked examples.
5. Inspect numerical agreement, truncation/refinement evidence, and the actual engine, precision, and device used.

Modeling actual systems is the capability; named qubits are useful constructors and fixtures within that capability. The initial implementation supports a documented subset of circuit descriptions and operator forms, and gives explicit diagnostics outside that subset.

M1 supplies library operations, validation CLI, examples, and acceptance tests. The runner, run directories, `env_hash`, and cancellation/accounting arrive in Block 3; the independent corrector service arrives in Block 4; public `simulate`/`sweep` protocols arrive in Block 5. Numerical checks written here establish engine evidence and can later be reused by the corrector. They do not issue production verification verdicts.

GPU work here is developer testing on the owner's workstation: a compatibility probe comes first, followed by scientific acceptance against refined CPU references. This does not introduce a research-job launch path before the runner exists.

## 2. Build as one milestone

```text
Block 0: package, CLI, CPU CI, import boundary, GPU test selection
    |
    v
Disposable GPU compatibility probe: Python + both engine stacks
    |
    v
System examples + model/engine conventions
    |
    v
Parse and validate --> construct model --> preserve operators and bases
                                            |
                      +---------------------+---------------------+
                      v                     v                     v
                 QuTiP CPU             cuDensityMat            dynamiqs
                      +---------------------+---------------------+
                                            v
                         analytic checks + cross-engine agreement
                         + refinement + GPU environment evidence
```

Read these as implementation steps, not separate infrastructure projects. Run the compatibility probe before step A or implementation: its outcome may change the Python pin and dependency choices. Once the environment is viable, establish the model boundary before completing engines.

| Step | Work | Evidence needed before advancing |
|---|---|---|
| P — Verify compatibility | In a disposable environment, run one tiny cuDensityMat solve through qutip-cuquantum and one dynamiqs solve on the RTX 5090 | Both complete on the actual GPU and match a known reference; record Python, package, driver/CUDA versions, commands, and device evidence; retain Python 3.13 unless an observed blocker requires changing it |
| A — Pin the representation | Review the examples in §4, operator conventions, and fixture matrix | Supported forms and explicit exclusions are written down; schema fields have a purpose |
| B — CPU vertical slice | Explicit two-level model and one component-defined circuit; validation, exports, analytic/reference spectra, QuTiP evolution | Unit conversion, dimensions, basis order, and basic dynamics work end to end on CPU |
| C — Composition | Coupled subsystems, interactions, drives, dissipators, priors, and basis/level refinement | A nontrivial composite model agrees with an independently constructed reference |
| D — Engine adapters | GPU wrapping and storage selection, cuDensityMat, and dynamiqs | The same normalized inputs work on each adapter; precision and device use are explicit |
| E — M1 acceptance | Complete the fixture matrix, gradient checks, refinement, diagnostics, and worked examples | Block 1 and Block 2 exits both pass with retained evidence |

Block 1 completes with CPU-testable QuTiP objects, numerical arrays suitable for JAX, and per-subsystem operator terms/dimension metadata for the cuDensityMat adapter. Block 2 constructs GPU objects, selects storage, and validates actual GPU execution. The initial probe demonstrates compatibility, not completion of either block.

## 3. One model boundary

### Representation

Use a Pydantic model for the parsed TOML structure and generate its JSON schema into `schemas/`. Keep custom scientific validation in Python: JSON schema alone cannot establish Hermiticity, basis compatibility, or physical parameter validity. Do not maintain a second handwritten copy of the field definitions.

[ADR-0002](adr/0002-system-model-representation.md) fixes the representation: model kinds and constructors, units and `[units]`, circuit coordinates and the pinned transformation, cutoffs, biases, operator names, composite basis, tensor order, and explicit-operator arrays. Proposed top-level fields are `format_version`, `id`, `units`, `parameters`, `model`, `truncation`, `drives`, and `dissipators`; exact syntax is fixed with the implementation.

Parameters have a resolved value and unit, with an optional declared source and prior. The current fixture uses a validated uniform prior. Priors are metadata at M1; construction uses explicit resolved values. Other prior families, sampling, calibration, and prior propagation follow the workflows that need them.

### Constructed model

The shared in-memory model carries:

- Ordered subsystem IDs, dimensions, basis definitions, and projection/retained-level information.
- Resolved parameters and their units/provenance; supplied component quantities remain inspectable.
- Static Hamiltonian terms, named drive operators, and dimensionless dissipator operators with rates.
- Factorized terms for composite models, plus explicit dense operators when supplied by the user.
- The conversion from model energies to dynamics units, and declared construction limitations.

Preserve local factors until an export needs materialized matrices. Requiring every backend to consume one pre-expanded dense Hamiltonian would discard the composite structure needed by cuDensityMat. Materialization is appropriate for the small CPU reference and moderate-size JAX lane, subject to declared memory limits.

Initial state, time grid, drive envelopes, observables, solver options, and gradient requests belong to an evolution input, separate from the physical system spec. An M1 Python example may construct that input directly. It does not require a new protocol-file schema or a public simulation command.

### Units, basis, and dissipation

- Follow ADR-0001: model energies/frequencies use GHz, time uses ns, and physical components retain explicit units. A GHz-valued Hamiltonian becomes `H_dynamics = 2π × H_GHz` in radians/ns. Perform this once at the model-to-dynamics boundary and expose the conversion; no engine adds another factor. scqubits' unit setting must agree with the declared model units. [scqubits units](https://scqubits.readthedocs.io/en/latest/guide/settings/guide-units.html)
- Collapse normalization, tensor ordering, and the separation of cutoffs from retained levels follow ADR-0002.
- Use a uniform output grid in ns. Internal adaptive integration is permitted. Drive sample times and discontinuities are explicit, and all adapters use the same interpolation/end-point convention.
- Start reference validation with float64/complex128. Configure and verify JAX precision before array creation; dynamiqs defaults to single precision and exposes an explicit double-precision setting. Lower-precision support needs separate tolerances and evidence. [dynamiqs precision](https://www.dynamiqs.org/stable/python_api/utils/global_settings/set_precision.html)

## 4. Runnable starting examples

The operational [system-spec reference](operations/system-spec.md) owns implemented syntax. These files own the concrete TOML and fixture values; they are illustrative inputs, not measured device parameters.

- **A — Component-defined circuit:** [junction with shunt](../examples/junction-shunt/system.toml) derives its reference transmon parameters from the supplied junction and shunt capacitances. The authored node offset and charge cutoff make sign and truncation checks observable.
- **B — Coupled circuit and resonator:** [composite](../examples/qubit-resonator/system.toml) references a [leaf circuit](../examples/qubit-resonator/qubit.toml) and an oscillator constructor. The loss test starts from one resonator photon so the dissipator is measurable. Coupling, tensor order, and retained-level refinement are checked independently.
- **C — Explicit operators:** [two-level model](../examples/two-level/system.toml) reads a numeric NPZ archive with a GHz Hamiltonian and dimensionless operators. Analytic decay and coherence checks separate rates from the 2π energy conversion.

The [LC](../examples/lc-circuit/system.toml), [fluxonium](../examples/fluxonium/system.toml), [two-mode component](../examples/two-mode-circuit/system.toml), and [three-subsystem](../examples/three-subsystem/system.toml) fixtures broaden representation coverage. Section 6 defines the evidence each fixture must provide.

## 5. Engine integration and storage

Implement the contract's `Integrator` capability/evolution surface with three adapters. Resolve the concrete initial-state, observable, solver-option, and result types before coding the common interface: these inputs must be explicit even though the contract sketches only `evolve(system, controls, grid)`.

For the first shared comparison, use density matrices and deterministic Lindblad evolution, including zero dissipators for closed systems. State-vector evolution can be supported where the adapter declares it. Results identify output times, requested observables, state representation/dimensions, engine, numerical method, dtype, actual device, and solver diagnostics. Avoid storing every full state for large problems; reference fixtures may retain them for comparison.

| Adapter | M1 responsibility | Required evidence |
|---|---|---|
| QuTiP CPU | Small reference-size evolution with explicit CPU data/method selection | Analytic agreement, drive/rate conventions, refinement, and reference results constructed independently of GPU adapters |
| cuDensityMat through `qutip-cuquantum` | Open/closed dynamics on the RTX 5090, retaining composite terms and supported local operator storage | Actual GPU execution, small-size agreement, and Dense/Dia storage selection; batching is deferred to sweep work |
| dynamiqs/JAX | Differentiable open/closed dynamics with explicit precision and device selection | CPU/GPU agreement on shared fixtures and an autodiff gradient checked against a numerical/analytic reference |

Use solver methods supported by the pinned plugin; selecting a GPU data type alone is insufficient. The current qutip-cuquantum solver guide specifies plugin integration methods. Isolate backend activation from the QuTiP CPU reference so a global plugin setting cannot turn both sides of a comparison into the same engine. Separate test processes are sufficient where necessary. [qutip-cuquantum solver guide](https://qutip-cuquantum.readthedocs.io/en/latest/solver.html)

### Storage rule to reuse and validate

The local scqubits checkout contains `_cuoperator_data` in `scqubits/utils/spectrum_utils.py`, at inspected revision `ca13baa35c349c24ad3dc2ba5036f60112785db9`: it uses Dense for dimension ≤16; otherwise it uses Dia when there are ≤16 occupied diagonals, and Dense otherwise. The helper is on `origin/nvidia-test` (27 commits beyond v4.3.1 in the inspected checkout), absent from `origin/main` and all available release tags. Reimplement the small selection rule inside the Block 2 cuDensityMat adapter; reuse the rule, not unreleased scqubits code or a sibling-checkout dependency. The threshold of 16 is a starting point to check on the RTX 5090, not a validated Entanglix performance threshold.

Select the effective cuDensityMat representation after probing conversion behavior in the pinned adapter stack. Current qutip-cuquantum documentation distinguishes Dense/Dia local terms and notes that CSR conversion routes through Dense. It also explains why flattening a tensor product into a full matrix loses useful structure. [Data-layer and tensor-structure reference](https://qutip-cuquantum.readthedocs.io/en/latest/data.html)

In Block 1, test per-subsystem numerical terms and dimension/order metadata on CPU. In Block 2, wrap those terms and test diagonal, narrow-band, dense, and factorized composite cases against a materialized small CPU reference. Record the actual selected representation and conversions. Preserve tensor factors when available; do not promise sparse GPU execution merely because the authored or CPU operator is CSR. Inspect the selected package versions before carrying over the threshold.

M1 sends shared CPU model inputs independently to each engine; it requires no cuDensityMat–JAX array exchange. Defer DLPack/CuPy handoff until a concrete workflow needs that boundary, potentially control or co-design. That consuming block owns dtype/order/device, lifetime, synchronization, and copy/zero-copy evidence. No gradient is propagated through the cuDensityMat lane.

Batch-versus-serial acceptance belongs with Block 5 sweeps. Until that capability is implemented and tested, the Entanglix adapter reports `batched = false`, and engine selection rejects requirements it cannot meet.

The versioned engine-selection table follows contract §2. M1 tests name adapters to compare them; validated non-gradient selection defaults to CPU. The [operational policy](operations/system-spec.md#engine-selection-policy) defines size caps and known-array storage accounting. Solver workspace and peak memory require measurements alongside the cuDensityMat usage review before automatic GPU sweep selection; this work is tracked in Block 5.

## 6. Scientific fixture and acceptance matrix

These starting values and cutoffs are proposals. Establish reference convergence before fixing tolerances; retain failing comparisons rather than weakening thresholds to fit an adapter. “Reference” below means a separately constructed analytic or library reference, not the same Entanglix export fed back into itself.

| Fixture | Proposed starting point | What it establishes |
|---|---|---|
| Explicit two-level system | Example C; excited state, `γ = 0.01/ns`, output `0–100 ns`; also a coherent initial state with dissipation disabled | Array loading, phase evolution, analytic `P_exc(t) = exp(-γt)`, complex coherences, and the rate/angular-unit distinction on all engines |
| LC circuit / oscillator | `C = 80 fF`, `L = 12 nH`; harmonic basis cutoff 30, retain 6 | Physical-component conversion; level spacings against `1/(2π sqrt(LC))`; lossy dynamics from the one-photon state with `γ = 0.01/ns`, `T = 50 ns`; driven dynamics and cutoff refinement |
| Junction with shunt | Example A; charge cutoffs 20/30/40, retain 4/6/8 | Component construction versus a direct scqubits Transmon using the same total capacitance; charge operator and projection |
| Fluxonium | `EJ = 5.7`, `EC = 1.2`, `EL = 0.4 GHz`, external flux `0.25 Φ0`; harmonic cutoffs 60/80/100, retain 6/8/10 | Inductive/JJ construction, static flux mapping, basis convergence, reference spectra and matrix elements |
| Custom two-mode component circuit | Two JJ/shunt branches to ground: `(EJ, ECJ, Csh) = (15 GHz, 2 GHz, 60 fF)` and `(18 GHz, 2 GHz, 70 fF)`; `2 fF` between their nodes; zero offset charge | Coupled circuit topology beyond named-qubit templates; compare with direct scqubits Circuit construction and the zero-coupling limit |
| Qubit plus resonator | Example B; start at 6×5, refine each subsystem separately | Composite terms, interaction scaling, tensor order, and driven/dissipative comparisons across all three engines; use example B's populated resonator and loss-disabled comparison |
| Three-subsystem composition | Two transmons with `(EJ, EC) = (15, 0.25)` and `(18, 0.28) GHz`, zero offset charge, and a `6.2 GHz` resonator; retained dimensions 3×4×5; charge–position couplings `0.05` and `0.04 GHz` | Nonuniform tensor dimensions, local-operator embedding, omitted-identity mistakes, and decoupled-limit spectra |

Each dissipative fixture must populate a state on which its collapse operator acts, directly or through a demonstrated drive, and show a measurable difference from loss-disabled evolution. Use `γT` around `0.1–1` for the acceptance cases; illustrative weak physical loss parameters alone do not test the dissipator.

The fixture set must exercise component, composition, and explicit-operator paths. CPU construction/spectral tests cover every row. Required three-engine dynamics cover at least the two-level, driven/lossy oscillator, and coupled qubit–resonator cases, with both static and time-dependent Hamiltonians and closed/open evolution. This separates representation coverage from the cross-engine matrix without multiplying every test combination.

For time-dependent tests, first define one constant and one piecewise-constant envelope. Pin left/right boundary semantics and force integration through discontinuities. Add a smooth envelope if it is needed to establish the adapter's supported modulation path. Comparisons use identical physical envelopes and output times, not merely the same number of samples.

For the differentiable lane, compare a nonzero derivative of a final expectation value with respect to a drive amplitude or a dissipative rate, using an analytic result when available and central finite differences at multiple step sizes. This demonstrates differentiation through evolution. Differentiation through scqubits quantization/eigendecomposition is outside the M1 claim. Dynamiqs provides examples of differentiating observables through evolution. [dynamiqs examples](https://www.dynamiqs.org/stable/documentation/getting_started/examples.html)

### Pinned numerical targets

Model and engine limits, refinement requirements, and their rationale are recorded in the [operational reference](operations/system-spec.md#numerical-checks-for-the-bundled-models). The [acceptance matrix](../tests/test_engine_acceptance.py), [gradient checks](../tests/test_engine_dynamiqs.py), and [retained results](../tests/probes/m1_acceptance_results.json) implement and verify them. Diagnose units, bases, dtype, solver settings, or truncation before changing a threshold.

Compare physical observables or aligned subspaces when truncation changes dimensions. Eigenvector signs/phases and rotations within degenerate subspaces are not failures; use projectors or phase-invariant quantities. Include at least one deliberately insufficient cutoff and one malformed-unit/rate/dimension case so failure diagnostics are exercised.

CPU/GPU agreement does not establish physical-device accuracy. State assumptions and unmodeled effects in examples; an M1 pass supports numerical correctness for the tested representations and regimes.

## 7. Dependencies, layout, and validation execution

Keep the single installable project. CPU modeling dependencies live in the default install. The locked optional extras are `differentiable` for CPU JAX, `differentiable-cuda12` for GPU JAX, and `cuquantum` for qutip-cuquantum's CUDA 12 stack. [Upstream dependency metadata](https://github.com/qutip/qutip-cuquantum/blob/main/pyproject.toml)

Preserve `import entanglix` and CPU tests without importing JAX, CuPy, scqubits, or QuTiP at the package root.

Suggested implementation locations, created as code arrives:

```text
src/entanglix/models.py     # spec parsing, validation, construction, operator exports
src/entanglix/engines/      # CPU entry points, selection table, three adapters
src/entanglix/evolution.py # shared engine input/result types and conventions
schemas/system.schema.json # generated from the authored-system model
tests/fixtures/            # small reviewed system specs and numeric references
tests/                     # model, export, engine, and regression checks
examples/                  # supported modeling and reference-evolution examples
docs/environment.md        # validated GPU setup and observed versions
docs/operations/system-spec.md # supported syntax, units, operators, tensor/basis conventions
```

Prefer a few focused modules; this tree does not require separate packages, managers, registries beyond supported names, or a generic serialization framework. Keep optional GPU imports inside their adapters. CPU CI runs schema/model validation, analytic spectra/dynamics, export checks that need no device, and existing foundation checks. Each optional engine test lives where an absent extra cannot break CPU collection.

The RTX 5090 workstation is the accepted GPU target. Package compatibility, actual device execution, driver/toolkit versions, and memory behavior still require observation. Sandbox `nvidia-smi` failure is not hardware evidence: run the documented GPU acceptance outside the restricted sandbox when needed. Record GPU/driver/CUDA/cuQuantum/JAX/plugin versions and exact commands in `docs/environment.md`, and pin Python packages in `uv.lock`; do not invent compatibility pins in this guide.

Proposed validation commands after the corresponding capabilities/extras exist:

```sh
uv sync --locked --exact
uv run --locked --exact pytest -m "not gpu"
uv run --locked --exact ruff check .
uv run --locked --exact ruff format --check .
uv run --locked --exact entanglix validate examples/qubit-resonator/system.toml
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --locked --exact --extra differentiable-cuda12 --extra cuquantum pytest -q tests/test_engine_cuquantum.py --require-gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu uv run --locked --exact --extra differentiable-cuda12 --extra cuquantum pytest -q tests/test_engine_dynamiqs.py -m gpu --require-gpu
```

The GPU suite must assert the actual backend/device used. Missing required engines or initialization failures must fail required-GPU acceptance rather than become `importorskip` successes. Record executed cases and skips; the existing GPU marker alone does not prove all three engines ran. If JAX and cuQuantum initialization compete for memory or alter global settings, run the relevant test groups in separate processes and compare retained small references.

Record compilation/setup and steady-state costs separately if timing is reported. Speedup is useful evidence, not a condition for M1 completion on small fixtures.

## 8. Settled decisions and remaining limits

Representation decisions are retained in [ADR-0002](adr/0002-system-model-representation.md). Evolution, differentiable controls, numerical limits, and selection bounds are documented in the [operational reference](operations/system-spec.md); verified dependency pins and device observations are in [environment.md](environment.md). Dense/Dia thresholds are functionally validated, and selection caps apply to reference-size fixtures. Validated non-gradient requests default to CPU. The [Block 5 checklist](checklist.md) retains the cuDensityMat open-system usage review, comparative timings, and peak-memory measurements before automatic GPU selection. Batching is also future work.

## 9. Exit evidence and handoff

[Checklist M1](checklist.md#m1--model--engines-blocks-12) is the sole implementation acceptance list. This guide supplies the implementation and test approach; supported conventions and pinned targets now live with the implementation. Do not add separate completion gates here or mark unchecked/partial GPU work complete.

Retain fixture inputs, tolerance rationale, test commands/results, engine/device identities, and environment observations with the implementation and link them from the checklist. Publish implemented syntax and conventions in `docs/operations/system-spec.md`, environment setup in `docs/environment.md`, and available usage/compatibility in the root README. These operational documents stand alone without linking to planning docs. Routine convention documentation needs no ADR; significant design rationale may warrant one.

The Block 3 handoff is a validated model loader, stable operator conventions, a common engine interface, tested capability/selection data, and observed environment information. The runner can then add dispatch, retained run provenance, environment hashing, and lifecycle management around these operations. Resolve engine identity and device together when dispatching: dynamiqs supports CPU and CUDA, while its selection-table row identifies CUDA.

Before retiring this guide, retain consequential design rationale in ADRs and necessary conventions/instructions in operational docs. Preserve the scientific acceptance evidence with the implementation.
