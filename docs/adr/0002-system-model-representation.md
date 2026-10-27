# ADR-0002: System Model Representation

**Date:** 2026-09-29, revised 2026-10-27
**Status:** Accepted
**Purpose:** fix how a system specification describes circuits, coupled subsystems, and explicit operators, so that an authored file keeps one physical meaning across scqubits versions and engine exports. The syntax and field reference are documented with the implementation.

---

## 1. Context

[ADR-0001](0001-mandate.md) §3.1 requires modeling of component-defined circuits, coupled quantum subsystems, and explicit operators, with declared bases, units, truncations, and parameter provenance, and it leaves the concrete representation to this record. §3.3 fixes GHz and ns, a single explicit angular-frequency conversion, and declared, separately refined cutoffs and retained levels.

scqubits quantizes circuits through a coordinate transformation `φ = Mθ` from node fluxes `φ` to quantized coordinates `θ`, classifies each coordinate as periodic, extended, or eliminated (frozen or free), and numbers coordinates and loops internally. Those internal choices depend on topology and can change between releases. The decisions below keep authored inputs tied to user-named objects wherever the physics allows it.

## 2. Decisions

### 2.1 Model kinds and constructors

- A system is one of three kinds: `circuit` (grounded circuits built from C, L, and JJ branches), `composite` (ordered subsystems with interactions), or `operators` (finite-dimensional numerical arrays).
- Composites reference leaf systems (circuit or operators specs) one level deep, or use a named constructor. A referenced spec cannot itself be composite, so reference cycles cannot occur.
- Named constructors are a closed list: `oscillator`, `transmon`, and `fluxonium`, built with scqubits' corresponding classes and exposing their conventional parameters. Others are added when a workflow needs them.
- System specifications are data. No user-supplied Python expression is evaluated.

**Rationale:** circuits remain the general path, and the constructors cover the common cases with short files and mature, well-tested scqubits classes. Because every subsystem exposes the same operator names (§2.6), the constructors add no second naming scheme, and the reference tests that compare circuits against scqubits' classes also check the constructors.

### 2.2 Units

- The unit vocabulary is fixed: `GHz`, `ns`, `1/ns`, `fF`, `nH`, `Phi0`, and `dimensionless`. There is no general units library. Unsupported units, or quantities incompatible with a field, are rejected.
- Every parameter carries its unit. A circuit branch takes either a physical quantity or the corresponding energy, never both. The supplied quantity and its conversion are kept as provenance.
- `[units]` declares the units of unlabelled numerical array data and of reported outputs. At present only `energy = "GHz"` and `time = "ns"` are supported.
- Dissipators are a dimensionless operator `A` with a nonnegative rate `γ` in `1/ns`. Engines receive `L = sqrt(γ) A`, and rates receive no angular-frequency conversion.

**Rationale:** a closed vocabulary covers every quantity the supported models need, and it keeps conversion auditable. Physical-to-energy conversions use the same constants as scqubits, so references agree.

### 2.3 Circuit coordinates and the transformation

- Quantized coordinates are identified as `thetaK`, where `K` is the coordinate index under the circuit's transformation.
- Circuits with a single quantized coordinate, and the three constructors, need no transformation input.
- A multi-node circuit specification includes its transformation matrix `M`. It is passed to scqubits as the transformation. `M` must preserve the periodic identification lattice, so that the integer charge basis of §2.4 represents the same quantum system. The periodic columns must form a primitive basis of the circuit's periodic identification lattice; changes between such bases are integer and unimodular. It must also keep the supported coordinate categories separate. A matrix that violates either condition produces a diagnostic. `validate` can print scqubits' default transformation for the author to adopt.

**Rationale:** cutoffs must address quantized coordinates (§2.4), so the meaning of `thetaK` must not depend on scqubits' default choice. Pinning `M` in the file prevents a scqubits upgrade from silently reassigning an existing file's cutoffs.

### 2.4 Construction bases, cutoffs, and retained levels

- Each active quantized coordinate declares an explicit cutoff. There is no universal default.
- Periodic coordinates use a charge basis: cutoff `N` means charge states `−N…N`. Extended coordinates use a harmonic basis: cutoff `N` means `N` oscillator states.
- `retained_levels` is separate from the construction cutoffs and sets the number of eigenstates kept after diagonalization. The `transmon` and `fluxonium` constructors use the same cutoff meanings for their single coordinate. The `oscillator` constructor takes only `retained_levels`, because its Fock basis is already its energy eigenbasis.

**Rationale:** these meanings follow scqubits' conventions, and keeping cutoffs separate from retained levels lets each one be refined independently, as ADR-0001 §3.3 requires.

### 2.5 Biases

- **Offset charges** are authored per circuit node, in dimensionless Cooper-pair units. The harness converts them to coordinate offsets as `q_θ = Mᵀ q_φ`:
  - components on extended coordinates are removable by a gauge transformation, so they are dropped and reported;
  - a nonzero component on an eliminated coordinate produces a diagnostic;
  - every node whose offset reaches a periodic coordinate requires an explicit value, including zero;
  - `validate` reports the derived offset of each coordinate.
- Offsets enter charging terms as `n − ng` on every construction path. Adapters convert to each backend's convention, which differs between scqubits' `Circuit` (`n + ng`) and `Transmon` (`n − ng`) classes.
- The `transmon` constructor keeps its conventional `ng`. The `fluxonium` constructor has no offset, because its coordinate is extended.
- **Static flux biases**, in `Phi0`, are attached to loops identified by closure branches:
  - the specification gives a complete closure set, one closure branch per independent loop, and each branch is passed to scqubits as a closure branch;
  - positive flux follows the closure branch's direction, from its first node to its second;
  - every loop requires an explicit value, including zero;
  - `validate` reports the resulting oriented loops.

**Rationale:** nodes and branches are objects the author names, so biases keep their meaning regardless of scqubits' internal coordinate and loop numbering. Coordinate conversion is the harness's responsibility, and the device model stays tied to physical islands and loops.

### 2.6 Operator names

Operators are referenced as `<subsystem>.<name>`. Available names depend on the model:

| Subsystem | Names |
|---|---|
| Periodic single-coordinate (transmon, single-junction circuit) | `charge`, `cos_phase`, `sin_phase` |
| Extended single-coordinate (fluxonium) | `charge`, `phase`, `cos_phase`, `sin_phase` |
| Oscillator | `annihilation`, `number`, `position` (`a + a†`) |
| Multi-node circuit | `nodeK.charge` and, where available, `nodeK.phase` |
| Explicit operators | the names the specification declares |

- `charge` means charge relative to the offset, `n − ng`, in Cooper-pair units. For a periodic coordinate it therefore need not have integer eigenvalues. For an extended coordinate, removing the offset by a gauge transformation leaves exactly this quantity.
- Node charges are reconstructed from these coordinate charges as `q_φ = M⁻ᵀ q_θ`. A node charge that depends on an eliminated coordinate is unavailable and reported as such.
- A node phase `φ_K = Σⱼ M_Kj θⱼ` is available only when every coordinate with a nonzero weight is a supported extended coordinate. Otherwise a diagnostic gives the reason.
- Branch-phase operators, such as the cosine of a junction phase, are not provided until a coupling or drive needs them.
- `validate` lists each subsystem's available names and their meaning. Unsupported names report the available choices.

**Rationale:** names that describe physical quantities make files independent of scqubits' variable numbering and basis-dependent operator names. A periodic coordinate has no single-valued phase operator, so availability follows the physics rather than a uniform list. Charge relative to the offset is uniform across coordinate types and is the quantity a capacitive coupling derived from the full circuit acts on.

### 2.7 Composite basis and tensor order

- Circuit and constructor subsystems are diagonalized, and their lowest `retained_levels` eigenstates form their basis. Interaction, drive, and dissipator operators are expressed in that basis.
- Explicit-operator subsystems keep their declared basis and dimensions, which may carry computational-state meaning. A later optional basis change must transform operators and state definitions together.
- Interactions are sums of operator products with explicit coefficients.
- Subsystem list order determines the tensor product, and the rightmost subsystem index varies fastest in flattened arrays. Exports keep dimension metadata and per-subsystem factors.

**Rationale:** the eigenbasis gives the smallest dimensions for composites and dynamics. Eigenvector phases are arbitrary, so comparisons use phase-invariant quantities or projectors.

### 2.8 Explicit-operator arrays

- Arrays are read from numeric NPZ files with explicit keys. Object arrays and pickle loading are rejected.
- Shapes are checked against the declared dimensions. The Hamiltonian is in the `[units]` energy unit, and other declared operators are dimensionless. The file's content digest is kept as provenance.

**Rationale:** a numeric-only format supports effective and custom models without executing stored code.

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| scqubits variable names (`n1`, `Q1`, `θ1`) | Depend on the default coordinate transformation and the basis choice; the same qubit has different names on scqubits' class and circuit paths |
| Loop indices for flux biases | scqubits' loop numbering is internal and can change with topology or release |
| Offsets addressed by quantized coordinate | Simpler adapter, but ties authored biases to coordinates instead of physical islands |
| Unpinned transformation for multi-node circuits | A scqubits upgrade could silently reassign cutoffs in existing files |
| Construction basis for composites | Much larger dimensions for composites and dynamics |
| Canonical charge `n` as `charge` | Differs between coordinate types after gauge removal, and interactions built from it omit single-subsystem terms |
| Each backend's own offset sign | scqubits' construction paths disagree, so the same authored offset would mean different physics |
| Oscillator as the only constructor | Once names are shared, the transmon and fluxonium constructors cost little and reuse existing reference tests |

## 4. Consequences

- The schema, `validate`, and the model adapter implement these rules. The derived coordinate offsets, available operator names, and scqubits' default transformation are reported by `validate`.
- The coordinate maps `Mᵀ` and `M⁻ᵀ`, the gauge and eliminated-coordinate rules, the transformation constraint, and the node-phase availability rule need tests. The custom two-mode circuit fixture exercises them.
- Offset and flux signs need observable checks, such as charge expectations or a multi-loop case, because opposite signs can give identical spectra.
- Branch-phase operators, further constructors, and additional units or bases are extensions.
