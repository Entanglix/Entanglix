---
name: system-modeler
description: Formulate and validate Entanglix quantum system models from circuit components, constructors, or explicit operators.
version: 0.1.3
---

# System modeling

Before authoring a model, read `reference/operations/system-spec.md` in the
research workspace for system fields, units, operator names, and circuit syntax.

Choose the representation from the question: a circuit for component-level
parameters and topology; an oscillator, transmon, or fluxonium constructor for a
supported isolated subsystem; explicit matrices for an externally derived finite
model. Compose subsystems when coupling and spectator modes matter. The available
constructors are not a limit on the physical systems a circuit can represent.

Use `validate_system` before a compute request. Read its available operator names,
meanings, tensor dimensions, and circuit provenance; fix a diagnostic rather than
inventing an operator or silently replacing the model. For multi-node circuits
the author supplies the transformation and the hierarchy with its group levels,
and validation offers a default transformation and one coordinate per group to
adopt. Refine the group levels like cutoffs; a transformation that mixes the
groups' coordinates can need many more levels. Their coordinates may be
periodic, extended, or both; a node's phase is exported only when every
coordinate it involves is extended. Offset charge addresses physical nodes, not
derived coordinates; dropped extended-coordinate offsets are reported. Flux is tied to an oriented closure
branch. Constructor and circuit flux signs need not coincide.

Keep units explicit: energies/frequencies in GHz and evolution times in ns; the
adapter applies the single 2π conversion. A rate in 1/ns enters a collapse
operator as √γ A without 2π. Oscillator `position` means a + a†. Composite
operators use each circuit or constructor subsystem's retained eigenbasis, or an
explicit subsystem's declared basis, in declared tensor order.
Specs and numeric NPZ data stay beside one another; no pickle data or nested
composite file references.

Dressed-state labels are modeling work on the user's request, not part of every
model or job. `load_system(path).hilbert_space()` returns the scqubits
`HilbertSpace` of a composite or a hierarchical circuit; other models, and a
control job's rotating-frame model, have none. Call its `generate_lookup()` and
read labels with `bare_index` and `dressed_index`. Keep the default ordering
(`DE`, by dressed energy and overlap), and report a state the lookup cannot
assign (`dressed_index` gives `None`, `bare_index` raises) as unassigned rather
than guessing. Use branch analysis (`ordering="LX"`) only when each subsystem's
declared order is its intended excitation ladder and its first state the intended
ground state; an explicit subsystem declared out of energy order breaks that, and
`LX` fails when a subsystem or group other than an oscillator retains one state,
so keep `DE` there. To label at other parameters, change the spec and load it
again: editing the returned `HilbertSpace` changes its lookup but not the loaded
model's matrices. A
label has one entry per subsystem in tensor order, or per group of a hierarchical
circuit; a circuit inside a composite takes one entry, its eigenstate index. Only
dressed indices below a circuit's `retained_levels` belong to its exported basis.

Choose retained levels and cutoffs for the reported quantity. A populated lossy
mode exercises loss; a ground-state mode does not. Describe convergence evidence
and missing refinement rather than equating schema validation with physical
validity. Save a runnable input, its assumptions, and intended observables.

Within an investigation, preserve approved base bytes and pass permitted
variations through job arguments. A different topology, data file, truncation, or
scientific evidence requirement needs a proposed base/revision and user decision.
