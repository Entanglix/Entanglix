# System specifications and CPU reference use

Entanglix currently accepts version-1 TOML files for a finite-dimensional operator model or a single-node junction with a shunt capacitor. Both paths run on the CPU. See the runnable [two-level model](../../examples/two-level/system.toml) and [junction model](../../examples/junction-shunt/system.toml). The coupled-subsystem representation and other circuit topologies are still being implemented.

Validate a file with `entanglix validate path/to/system.toml`. Success prints the model ID, basis, dimensions, and available operator names. Invalid input prints a diagnostic to stderr and exits with code `64`. In Python, `entanglix.models.load_system(path)` returns a `ConstructedSystem` with `hamiltonian_ghz`, named `operators`, `dissipators`, `dimensions`, `basis`, and `provenance`. It can export a QuTiP Hamiltonian with `qutip_hamiltonian()`; named operators are NumPy arrays in `operators`.

## Common fields and units

Every file declares `format_version = 1`, an `id`, `[units]` with `energy = "GHz"` and `time = "ns"`, and a `[model]`. Parameters have explicit `value` and `unit`, with an optional `source`. Construction uses the supplied value.

Accepted unit names are `GHz`, `ns`, `1/ns`, `fF`, `nH`, `Phi0`, and `dimensionless`. A quantity must use the unit required by its field. Energy arrays inherit `[units].energy`; named operator arrays are dimensionless. A dissipator has a nonnegative `rate` in `1/ns` and a dimensionless operator `A`; its collapse operator is `sqrt(rate) × A`. Rates receive no `2π` conversion.

The CPU evolution boundary converts a Hamiltonian in GHz to radians/ns exactly once: `H_dynamics = 2π × H_GHz`. Its output grid is uniform in ns, although the integrator may take internal steps.

## Explicit operators

An `operators` model declares `dimensions`, `basis`, `array_file`, the NPZ key of its Hamiltonian, and a map from Entanglix operator names to NPZ keys. `array_file` must name an NPZ file beside the system TOML; paths to other directories are rejected. The current CPU path supports one dimension of at least two states. Matrices must be finite, numeric, and square with the declared dimension; the Hamiltonian must be Hermitian. Object arrays and pickle loading are rejected. The declared basis is preserved, including its computational-state meaning. The archive's SHA-256 digest is recorded in model provenance.

Each `[[dissipators]]` entry gives a unique `id`, one named operator in `factors`, and a nonnegative rate in `1/ns`. The [two-level example](../../examples/two-level/system.toml) uses `H0 = diag(0, 0.2) GHz`, `x = sigma_x`, and `lower = |0⟩⟨1|`.

## Junction and shunt circuit

The current circuit path requires `nodes = [0, 1]`, `ground = 0`, one JJ branch and one C branch between those nodes, and an explicit dimensionless `offset_charges = { "1" = ... }`. Each branch has a unique ID and exactly one of `charging_energy` in GHz or `capacitance` in fF. A JJ also has `josephson_energy` in GHz. Values may be inline `{ value = ..., unit = ... }` or `{ parameter = "name" }`. Physical capacitance is converted to charging energy as `E_C/h = e²/(2hC)`, expressed in GHz; supplied values and converted branch values remain in provenance.

`[truncation]` supplies `cutoffs = { theta1 = N }` and `retained_levels`. The periodic charge basis includes integer states from `−N` through `N`, and the lowest retained eigenstates form the exported basis. Refine the charge cutoff and retained levels separately for any scientific claim. The authored node offset follows the `n − ng` convention; the adapter reverses its sign when setting scqubits Circuit's `ng1`. `charge` is `n − ng`, projected into the retained eigenbasis. The other available operators are `cos_phase` and `sin_phase`. A periodic coordinate has no single-valued raw phase operator.

## CPU evolution

`entanglix.engines.evolve_cpu(system, initial_state, times_ns, observables=...)` takes a normalized ket or density matrix, a uniform output grid in ns, and optional named finite square observable matrices. It returns sampled density matrices, expectations, and output times. The [CPU example](../../examples/cpu_reference.py) evolves an excited two-level state for 100 ns and compares its final population with `exp(-1)`.

For this release, circuit validation covers the junction-and-shunt topology and evolution uses a static Hamiltonian. Other component circuits, coupled subsystems, and time-dependent envelopes require later model and engine support. Unsupported inputs produce errors.
