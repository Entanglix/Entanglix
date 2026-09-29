# System specifications and CPU reference use

Entanglix accepts version-1 TOML descriptions of component circuits, coupled subsystems, and finite-dimensional operator models. Construction and evolution currently run on the CPU. Runnable examples cover a [two-level operator model](../../examples/two-level/system.toml), [junction and shunt](../../examples/junction-shunt/system.toml), [LC circuit](../../examples/lc-circuit/system.toml), [fluxonium circuit](../../examples/fluxonium/system.toml), [two-node circuit](../../examples/two-mode-circuit/system.toml), [qubit–resonator composite](../../examples/qubit-resonator/system.toml), and [three-subsystem composite](../../examples/three-subsystem/system.toml).

Run `entanglix validate path/to/system.toml`. Success lists the model ID, basis, dimensions, and available operators. Invalid input prints a diagnostic to stderr and exits with code `64`. The Python operation `entanglix.models.load_system(path)` returns a `ConstructedSystem` with its GHz Hamiltonian, named dimensionless operators, drives, dissipators, dimensions, basis, and provenance. `qutip_hamiltonian()` returns a QuTiP `Qobj`; NumPy matrices and ordered `hamiltonian_terms`/`operator_factors` are available for engine adapters.

## Common fields and units

Every file has `format_version = 1`, an `id`, `[units]` with `energy = "GHz"` and `time = "ns"`, and a `[model]`. Values with physical units use `{ value = ..., unit = ... }` or `{ parameter = "name" }`. Parameters carry a value, unit, optional `source`, and optional uniform `{ kind = "uniform", lower = ..., upper = ... }` prior. Prior bounds inherit the parameter unit; construction uses the explicit value and does not sample the prior. Invalid bounds are rejected.

The fixed unit vocabulary is `GHz`, `ns`, `1/ns`, `fF`, `nH`, `Phi0`, and `dimensionless`. Each field accepts only its specified unit. Unlabelled Hamiltonian arrays inherit `[units].energy`; named operator arrays are dimensionless. Circuit and constructor loading requires scqubits' global unit setting to be `GHz`. A dissipator with rate `γ` in `1/ns` and operator `A` gives collapse operator `sqrt(γ) A`, with no `2π` on `γ`. Dynamics convert `H_GHz` to radians/ns once, as `2π H_GHz`.

## Component circuits

`kind = "circuit"` uses `nodes = [0, ..., N]`, `ground = 0`, named `[[model.branches]]`, and `[truncation]`. Branches connect declared nodes and have kind `C`, `L`, or `JJ`. C and JJ branches take exactly one of `charging_energy` (GHz) or `capacitance` (fF); JJ also takes `josephson_energy` (GHz). L branches take exactly one of `inductive_energy` (GHz) or `inductance` (nH). Capacitance converts as `E_C/h = e²/(2hC)`; inductance converts as `E_L/h = (Φ0/2π)²/(Lh)`. Supplied and converted values are retained in provenance.

Each quantized coordinate has a `thetaK` cutoff. Periodic cutoff `N` retains charge basis states `−N…N`; extended cutoff `N` retains `N` harmonic basis states. `retained_levels` then selects the lowest eigenstates. Refine the construction cutoffs and retained levels separately before treating results as converged.

`offset_charges` names non-ground nodes with dimensionless `n − ng` values. Every node contributing to a periodic coordinate requires an explicit value, including zero. The adapter records `q_theta = Mᵀ q_node`, reverses the sign for scqubits Circuit's `ngK`, and exports offset-relative charge. A removable offset on an extended coordinate is reported and does not alter its spectrum. A multi-node circuit requires `transformation = [[...], ...]`; the current multi-node path supports periodic coordinates with an integer unimodular transformation. Mixed extended/periodic, free, and frozen multi-node coordinates produce diagnostics.

Every independent flux loop needs one `[[model.external_fluxes]]` entry with `branch`, `value`, and `unit = "Phi0"`, including zero bias. Its named closure branch fixes the positive direction from the branch's first node to its second. The applied closure set and transformation are recorded in provenance.

For a single periodic coordinate, the available names are `charge`, `cos_phase`, and `sin_phase`. A single extended coordinate exports `charge` and `phase`; JJ circuits also export `cos_phase` and `sin_phase`. The current multi-node path exports `nodeK.charge`. A periodic raw phase operator is unavailable. `validate` lists the names for the constructed model.

## Coupled subsystems

`kind = "composite"` lists ordered `[[model.subsystems]]` with unique IDs. A subsystem can name a leaf TOML file beside the composite with `source = "qubit.toml"`; referenced files cannot themselves be composites, and their drives or dissipators must be declared on the composite. Constructors are `oscillator` (`frequency`, `retained_levels`), `transmon` (`josephson_energy`, `charging_energy`, `ng`, `cutoff`, `retained_levels`), and `fluxonium` (`josephson_energy`, `charging_energy`, `inductive_energy`, `flux`, `cutoff`, `retained_levels`). Energies are in GHz, `flux` in `Phi0`, and `ng` is dimensionless. The [qubit–resonator example](../../examples/qubit-resonator/system.toml) shows a leaf reference and an oscillator constructor.

Fluxonium's constructor `flux` follows scqubits' conventional sign. For the same single-junction circuit and node orientation, constructor `flux = +0.25 Phi0` corresponds to a circuit with its inductive closure branch directed `[1, 0]` and external-flux value `+0.25 Phi0`. Directing that branch `[0, 1]` reverses the signed `phase` and `sin_phase` expectations at the same positive value, although the spectra agree. Compare signed observables when matching constructor and circuit descriptions.

`[[model.interactions]]` gives a GHz `coefficient`, named `factors`, and optional `add_hermitian_conjugate`. A factor is `<subsystem>.<operator>`; factors are multiplied in listed order within each subsystem, with identities on omitted subsystems. The full Hamiltonian must be Hermitian. Subsystem list order sets the tensor order, and the rightmost index varies fastest in flattened arrays. `hamiltonian_terms` keeps local matrices and interaction factors rather than only the materialized CPU matrix. Composite operators use names such as `q.charge` and `r.position`, where oscillator `position = a + a†`.

Top-level `[[drives]]` entries give unique IDs and Hermitian dimensionless operator factors; a GHz amplitude is supplied at evolution time. Top-level `[[dissipators]]` entries give unique IDs, operator factors, and nonnegative rates in `1/ns`. Both can reference leaf or composite operator names. Unknown names report the available operators.

## Explicit operators and CPU evolution

`kind = "operators"` declares one finite dimension, its basis, a numeric NPZ `array_file` beside the TOML file, the Hamiltonian key, and a map of operator names to NPZ keys. Matrices must be finite and square with the declared dimension; the Hamiltonian must be Hermitian. Object arrays and pickle loading are rejected. The declared basis is preserved, and the archive digest appears in provenance.

`entanglix.engines.evolve_cpu(system, initial_state, times_ns, observables=..., drive_amplitudes_ghz=...)` takes a normalized ket or density matrix and a uniform output grid in ns. Named drive amplitudes are constant scalar GHz values. Observables are finite square matrices. The result contains sampled density matrices, expectations, and output times. The [CPU example](../../examples/cpu_reference.py) demonstrates analytic two-level decay and the coupled-system spectrum.

Current evolution is reference-size CPU work. GPU adapters, engine selection, runner records, and public simulation/sweep commands arrive with their respective capabilities.
