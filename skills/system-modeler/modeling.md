# Modeling in the session

Session modeling runs on CPU with the Python named in this project's Entanglix
MCP entry, as `PYTHON script.py`. Start from whatever the user has: a schematic,
a netlist, capacitances and junction parameters, design targets, or a model to
explore. `reference/operations/system-spec.md` defines the file fields. Each
entry gives the purpose, when it applies, the call, what the output means, and
what it does not check.

Three kinds of object have different methods:

- Entanglix's `ConstructedSystem`, from `entanglix.models.load_system`;
- scqubits' `HilbertSpace`, from `ConstructedSystem.hilbert_space()`;
- scqubits' `Circuit`, reached through that `HilbertSpace` or built in the session.

## Charging and inductive energies

- Purpose: turn capacitances and inductances into the energies constructors take.
- When: a transmon or fluxonium constructor, or a coupling worked out by hand. A
  circuit netlist takes `capacitance` (fF) and `inductance` (nH) directly and
  converts them itself.
- Call: `E_C/h = e²/(2hC) = 19.370 GHz·fF / C` and
  `E_L/h = (Φ0/2π)²/(hL) = 163.46 GHz·nH / L`. A node joined to others has the
  loaded charging energy `E_C,i = (e²/2)(C⁻¹)ᵢᵢ` from the node capacitance matrix,
  where `Cᵢᵢ` sums every capacitance at node `i`, junction capacitance included,
  and `Cᵢⱼ` is minus the capacitance between nodes `i` and `j`.
- Output: GHz values for `charging_energy` and `inductive_energy`.
- Limits: lumped elements only; stray capacitances are the author's.

## One circuit or a composite

- Purpose: choose how to write a device of several coupled modes.
- When: the modes share a netlist, such as coupled fluxoniums or transmons.
- Call: a circuit with a hierarchy of one group per mode, or a composite of
  per-mode subsystems with derived couplings.
- Output: the circuit gives the device's dressed eigenstates as one block, with
  its modes labelled through `hilbert_space()`; the composite gives the product of
  the modes' eigenbases, with each subsystem's operators.
- Limits: the circuit needs no coupling derivation; its job observables include
  node operators such as `node1.charge` and populations of dressed states, which
  the labels identify per mode, but not a mode's population summed over the other
  modes. The composite needs the couplings from the capacitances, and also offers
  those subsystem populations and two-subsystem metrics.

## Capacitive coupling coefficients

- Purpose: the coefficient of a `charge`·`charge` interaction between two subsystems.
- When: two circuit or constructor subsystems joined by a capacitance and
  composed rather than written as one netlist.
- Call: `g/h = 4e²(C⁻¹)ᵢⱼ/h = 8 × 19.370 GHz·fF × (C⁻¹)ᵢⱼ`, from the same matrix
  as the loaded charging energies, as
  `coefficient = { value = g, unit = "GHz" }` with
  `factors = ["a.charge", "b.charge"]`.
- Output: a positive coefficient on the exported offset-relative charges, counted
  in Cooper pairs.
- Limits: other interactions multiply the exported dimensionless operators as
  named (`position` is a + a†), so convert a physical coupling to that
  normalization. Where the whole circuit fits one netlist, compare the composite
  with it.

## Truncation

- Purpose: choose `cutoffs`, a multi-node circuit's `hierarchy` and
  `hierarchy_levels`, and `retained_levels`.
- When: every circuit or constructor subsystem.
- Call: the fields in `reference/operations/system-spec.md`; `validate_system`
  reports the dimensions and offers a default hierarchy.
- Output: the retained basis that jobs and session code share.
- Limits: validation does not show that a truncation suffices; see "Checking a
  numerical claim".

## The loaded model

- Purpose: the model exactly as a job builds it.
- When: inspecting a model, or computing from it in the session.
- Call: `system = load_system("system.toml")`; at other declared parameter values,
  `load_system("system.toml", parameters={"EJ": 15.2})`, which sets them as a
  job's `[parameters]` does.
- Output: a `ConstructedSystem` with `hamiltonian_ghz` (dense, in the retained
  basis), `operators` (named dimensionless matrices in that basis),
  `dimensions`, `subsystem_ids`, `provenance` (converted branch energies under
  `converted_branches`, and `system_sha256`), `qutip_hamiltonian()`, and, for a
  composite or a hierarchical circuit, `hilbert_space()`.
  `numpy.linalg.eigvalsh(system.hamiltonian_ghz)` is the model's spectrum.
- Limits: `parameters` sets declared parameters only; change truncation or
  topology in a copy of the spec.

## Low states of a scqubits object

- Purpose: a few eigenstates of a scqubits object without building its native
  matrices by hand.
- When: exploring a `HilbertSpace` or a `Circuit` in the session.
- Call: `space.eigensys(evals_count=4)` or `circuit.eigensys(evals_count=4)`;
  `eigenvals` for energies only.
- Output: energies in GHz, and eigenvectors in that object's own basis.
- Limits: for the exported model keep `numpy.linalg.eigvalsh(system.hamiltonian_ghz)`.
  A loaded composite's `HilbertSpace` uses the exported product basis, so its
  eigenvectors apply to the exported matrices, up to arbitrary phases. A
  hierarchical circuit's `HilbertSpace` uses the group product basis, while the
  export uses the circuit's retained eigenbasis.

## Dressed-state labels

- Purpose: each dressed state's bare product label, on the user's request.
- When: a composite or a hierarchical circuit.
- Call: `space = system.hilbert_space()`, `space.generate_lookup()`, then
  `space.bare_index(k)`, `space.dressed_index((1, 0))` or
  `space.energy_by_bare_index((1, 0))`.
- Output: labels in tensor order, one entry per subsystem or group.
- Limits: the ordering and unassigned-state rules in `SKILL.md`.

## A circuit's symbolic Hamiltonian

- Purpose: read a circuit's Hamiltonian as an expression, with its coordinates,
  loaded charging energies and couplings.
- When: a loaded hierarchical circuit, or the bare topology of any circuit.
- Call: `system.hilbert_space().subsystem_list[0].parent.sym_hamiltonian_expr()`
  for a loaded hierarchical circuit, which carries the spec's transformation,
  closure branches, fluxes and offsets. To inspect the bare topology of another
  circuit, build `scqubits.Circuit.from_yaml_string(yaml, ext_basis="harmonic")`
  from the branch energies in `provenance["converted_branches"]` and call
  `sym_hamiltonian_expr()`.
- Output: a sympy expression in GHz with charges, phases, offset symbols `n_gK`
  and flux symbols `ΦK`.
- Limits: scqubits' `n_gK` is the negative of Entanglix's offset. A circuit built
  from the branches alone puts its fluxes on scqubits' default closure branches,
  with zero bias and no offsets, so it is not the loaded model: take numbers from
  the loaded model. Energies alone cannot confirm a flux sign; compare a signed
  expectation such as the phase. Reading an expression needs no truncation study.

## Derivations with sympy

- Purpose: analytic work such as series expansions, perturbative estimates, or
  parameters solved from targets.
- When: any derivation; sympy is an Entanglix dependency.
- Call: `import sympy`.
- Output: exact or symbolic results.
- Limits: nothing ties an expression to the model; evaluate it at one point and
  compare with the loaded model.

## Checking a numerical claim

- Purpose: measure how a reported number depends on truncation.
- When: a numerical claim, such as a small coupling, a transition frequency or a
  shift.
- Call: load a copy of the spec with larger cutoffs, `hierarchy_levels` or
  `retained_levels` and recompute that quantity.
- Output: its sensitivity to the tested truncations, reported with it.
- Limits: a job's built-in spectral refinement compares excitation energies only,
  so a coupling or a derived shift still needs its own recomputation; request a
  built-in check when the user authorizes it and it is relevant to the claim.
  Compare with a reference only when the user has one, such as measured data or a
  published value; fit measured spectra with the `calibrate` protocol.

## Sign conventions

- Offsets: `offset_charges` and a transmon's `ng` give the `ng` in `n − ng`, and
  exported `charge` operators are offset-relative. scqubits' `Circuit` writes
  `n + n_g`, so its `ngK` has the opposite sign.
- Flux: a fluxonium constructor's `flux` follows scqubits' sign; the matching
  circuit depends on its closure-branch direction (`system-spec.md`).
- Eigenvector phases are arbitrary: across separately built models compare
  energies, matrix-element magnitudes or projectors, not signed matrix elements.
- A capacitive `charge`·`charge` coefficient is positive.
- Oscillator `position` is a + a†.

## Tracing a derived parameter

- Purpose: connect a value in a system file to the session script that derived it.
- When: a parameter came from session work.
- Call: `[parameters.EJ]` with `value`, `unit` and
  `source = "derivations/ej_from_target.py"`, the script's project-relative path;
  a finding may name the same path.
- Output: the path stays in the spec, and in every run that freezes it.
- Limits: the script itself is not frozen; `run_script` freezes a script with its
  outputs once a valid system spec exists.

## Example

A session script for a composite whose spec declares `EJ`:

```python
import numpy as np
import entanglix
from entanglix.models import load_system

system = load_system("system.toml", parameters={"EJ": 14.0})
energies = np.linalg.eigvalsh(system.hamiltonian_ghz)
space = system.hilbert_space()
space.generate_lookup()
print(entanglix.__version__, energies[1] - energies[0], space.bare_index(1))
```
