---
name: system-modeler
description: Formulate and validate Entanglix quantum system models from circuit components, constructors, or explicit operators.
version: 0.1.0
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
the author supplies the transformation, and validation offers the default to
adopt. Offset charge addresses physical nodes, not derived coordinates; dropped
extended-coordinate offsets are reported. Flux is tied to an oriented closure
branch. Constructor and circuit flux signs need not coincide.

Keep units explicit: energies/frequencies in GHz and evolution times in ns; the
adapter applies the single 2π conversion. A rate in 1/ns enters a collapse
operator as √γ A without 2π. Oscillator `position` means a + a†. Composite
operators use each subsystem's retained eigenbasis and declared tensor order.
Specs and numeric NPZ data stay beside one another; no pickle data or nested
composite file references.

Choose retained levels and cutoffs for the reported quantity. A populated lossy
mode exercises loss; a ground-state mode does not. Describe convergence evidence
and missing refinement rather than equating schema validation with physical
validity. Save a runnable input, its assumptions, and intended observables.

Within an investigation, preserve approved base bytes and pass permitted
variations through job arguments. A different topology, data file, truncation, or
scientific evidence requirement needs a proposed base/revision and user decision.
