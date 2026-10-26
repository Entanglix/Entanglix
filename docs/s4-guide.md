# S4 Guide — Interoperability

**Status:** Planned. The decisions are recorded in [ADR-0009](adr/0009-published-formats.md); completion and acceptance evidence belong to the checklist.
**Scope:** S4, Block 12 (interoperability).
**Authority:** [roadmap](roadmap.md#block-12--interoperability), [ADR-0009](adr/0009-published-formats.md), [ADR-0003](adr/0003-runs-verification-and-dispatch.md), [ADR-0005](adr/0005-control-problems-and-solvers.md), [ADR-0008](adr/0008-results-catalog.md), and [harness contract](../HARNESS_CONTRACT.md#8-versioning-and-conformance). [The checklist](checklist.md#s4--interoperability-block-12) owns completion status and acceptance evidence.
**Audience:** the owner and implementation agents. This is a planning document; operational docs and production agents must work without it.

## 1. Outcome and boundary

S4 publishes Entanglix's formats for programs outside it. At completion, a program that does not import Entanglix can:

1. Find every published input, record, array and reply, with its supported versions, in the formats reference. It can read the schemas and archive descriptors shipped in the package.
2. Validate the records of a run of each protocol, of a check run and of a script run, using TOML, JSON Schema, NumPy and the documented conventions.
3. Construct inputs for each protocol that Entanglix accepts through its normal validation.
4. Read the version of every format from `capabilities`, and validate CLI and MCP replies against their schemas.

Each supported version has committed fixtures, which pass their schemas and Entanglix's own readers. A change that breaks them without a version change fails the tests.

S4 advertises no MCP output schemas and adds no declared formats to artifact references. It promises no forward compatibility and involves no amicode runtime or viewer. Historical files are never rewritten. Reading a historical record promises neither replay nor checks.

No step needs a live agent session. Transport evidence uses scripted CLI and MCP clients.

## 2. Build by step

```text
ADR-0009
    |
12-1 formats package and inputs
    |
12-2 run records and results
    |              \
12-3 ledger records  12-4 arrays
    |              /
12-5 replies
    |
12-6 outside program and S4 evidence
```

Steps complete in order. 12-3 and 12-4 depend only on 12-2 and may run in either order.

| Step | Work | Evidence before advancing |
|---|---|---|
| 12-1 — Formats package and inputs (ADR-0009 §2.1, §2.2, §2.4, §2.5) | The package directory for schemas and descriptors, with the system schema moved there and the root `schemas/` folder retired. Schema generation from the seven input models, with the drift test. The formats reference: published, opaque and internal files; the JSON mapping and the semantic conventions, with the integer rule's limits and each path field's resolution base; the compatibility rule and the supported versions. Format versions in `capabilities` for the formats published so far. Version 1 input fixtures | Each input schema equals the one generated from its model, and a model change without regenerating fails. The installed package resolves the schema files. Each input fixture passes its schema and is accepted by its model and compile path. The fixtures include an explicit-operator system, a dataset with its archive, and a control spec with a starting-pulse file. `capabilities` lists each published input with its version. The reference stands without this guide |
| 12-2 — Run records and results (§2.2, §2.3, §2.8) | Record models, with generated schemas, for the run, start, terminal, reconciliation and abort-request records. Each protocol's result: simulation, sweep aggregate and point, control, calibration with its assignment, and the common structure of an authored script result. The verification result as one envelope with check-specific variants. `record_version = 1` in the writers of results, sweep point results, the assignment and the abort request. Writer-output tests. Version 1 fixtures with their variants | Records written through the real writers validate, for every protocol and for check and script runs. A result fixture without a version is read as version 1 and passes its schema, `read_run` and the check loaders. A fixture exists for each variant: completed, failed, aborted, interrupted with and without reconciliation, unknown, an unreadable result, a partial sweep and an abort request. Each variant's records pass their schemas and `read_run` reads them. The exception is the unreadable result, which fails its schema while `read_run` succeeds and reports `result_error`. An authored result with extra fields passes, and a result missing `engine` or `metrics` fails. Structural dependencies that the models declare reach their generated schemas, and no optional TOML record field accepts null |
| 12-3 — Ledger records (§2.1, §2.8) | Record models and schemas for investigation revisions and events, staged candidates and catalog entries, device-update proposals and their applications, launch requests, calls, attempts and refusals, and MCP session and call audits. `record_version = 1` in the launch attempt and refusal writers | Each ledger record written through its real writer validates. Each fixture passes its schema and its reader: the investigation reads, `read_entry`, the proposal check in the apply command, and `read_launch`. A launch attempt fixture without a version is read as version 1. Audit records validate. They have no reader, so the formats reference marks them as write-only |
| 12-4 — Arrays (§2.2, §2.3, §2.7) | One descriptor for each array kind: the neutral pulse, simulation output, sweep aggregate and point outputs, initial and reference states, measurement archives and explicit-operator archives. A descriptor is generated where an authoritative declaration exists, and maintained directly otherwise. Selection of a descriptor through the spec or record that names the archive. Names from authored specs. Observables real or complex. The empty-aggregate rule. The legacy pulse reading in the catalog's pulse reader | A generated descriptor equals its source. Archives written by the real writers pass their descriptors. Each array fixture passes its descriptor and its reader: the catalog and starting-pulse readers for pulses, the check loaders for outputs, and the system and dataset loaders for operator and measurement archives. A pulse without `kind` is read as `zero_order` and can be staged and used as a starting pulse. An empty sweep aggregate passes only when its points declare no metrics. A complex and a real observable both pass |
| 12-5 — Replies (§2.6) | Reply models and schemas for each CLI command and MCP tool, kept separate where MCP reshapes a reply. ISO datetime encoding in the CLI and MCP. `capabilities` reporting every format's version, replies included. No output schema advertised | Scripted CLI and MCP clients read the records written in 12-2 and 12-3. Every JSON reply validates against its schema, including the compact views, the paged tables and inspection diagnostics such as `result_error`. Existing reply nulls, such as `next_after`, are kept. Every datetime in a reply has the `T` form. CLI failures keep their standard-error message and exit code, and MCP failures stay tool errors; both are tested separately, with no new JSON envelope. `tools/list` advertises no output schema. `capabilities` lists every published format with its versions |
| 12-6 — Outside program and S4 evidence (§2.8) | The outside-program test. The breaking-change test. The local acceptance record over the retained runs. Release notes. Block 12's checklist ticks | The outside program, in an interpreter where Entanglix cannot be imported, validates a test-written run of each protocol, a check run and a script run. It constructs inputs for each protocol, which Entanglix accepts through its normal validation. Removing a field from a supported version fails its fixtures. The local record validates the retained runs of each protocol with the shipped schemas and descriptors. Block 12's checklist items pass |

Each step runs the tests its changes can reach, decided from its actual diff. 12-2 changes the verification result's writer inside `src/entanglix/corrector/`, so the corrector policy version changes. Earlier check reports keep their own policy label, and a policy change alone reruns no acceptance study. The format code stays outside the corrector.

## 3. The evidence

The roadmap's Done criterion is met by two records:

| Item | Choice |
|---|---|
| Outside program | A test runs a short program in a separate interpreter, with an isolated path, where importing Entanglix fails. It uses `tomllib`, `jsonschema` and NumPy, and reads the schemas and descriptors as files. Its inputs: test-written runs of simulate, sweep, control, calibrate, a check and a script; their ledger records; and the replies recorded by the scripted clients. It writes one input per protocol from the schemas and conventions alone. The test then loads each input through Entanglix's normal validation |
| Local acceptance record | A script in the probe style validates the retained runs of each protocol with the shipped schemas and descriptors, on a disposable copy, with no solves. Its summary is committed beside the probes. The run directories stay in the local runs root |

## 4. Conventions to settle with the implementation

- **Schema names and identifiers.** The file name and the `$id` of each schema and descriptor encode the format and its version, and are fixed in 12-1.
- **Generation command.** One command regenerates every generated schema and descriptor; the drift test compares against it. Fixed in 12-1.
- **Null.** How generated record schemas drop null from optional TOML fields, fixed in 12-2. Reply schemas keep the existing nullable fields, fixed in 12-5. Each with a test; no general transformation removes null everywhere.
- **Non-finite numbers.** TOML can hold `nan` and `inf`, and JSON cannot. No retained record holds one. 12-2 fixes the JSON mapping and whether a writer may produce such a value, with a fixture. 12-5 makes replies standard JSON.
- **Fixtures.** They live under `tests/data/formats/<format>/v<N>/`. Each fixture names the reader it must pass.
- **Verification variants.** The envelope's variants are selected by check name, with one variant per check family.
- **`capabilities`.** The field that lists format versions is named in 12-1 and documented in the formats reference.
- **Doc alignment.** The contract, roadmap §7, checklist and docs-index edits of ADR-0009 §4 are made with this guide's commit. The operational references and release notes follow their steps.
- **Live sessions.** Scripted CLI and MCP clients are the required evidence. A live session runs only on the owner's word, after its share of the usage window is estimated.

## 5. Layout and validation

Suggested locations, created as code arrives:

```text
src/entanglix/formats/               # record, reply and array models; generation (12-1 to 12-5)
src/entanglix/schemas/               # shipped schemas and descriptors (package data)
src/entanglix/cli.py, mcp.py         # ISO datetimes; capabilities (12-5)
src/entanglix/catalog.py             # legacy pulse reading (12-4)
tests/data/formats/                  # committed fixtures per format and version
tests/probes/                        # local acceptance record (12-6)
docs/operations/formats.md           # published formats, versions, conventions, compatibility
```

Validation commands, as in S3:

```sh
uv run --locked --exact pytest -m "not gpu"
uv run --locked --exact --extra mcp pytest -m "not gpu" -k "mcp"
uv run --locked --exact ruff check .
uv run --locked --exact ruff format --check .
```

Each step runs the tests its changes can reach and adds its tests to the existing CI jobs.

## 6. Exit evidence and handoff

[Checklist S4](checklist.md#s4--interoperability-block-12) is the sole acceptance list; this guide adds no completion gates. The outside-program test and the fixtures run in CI. The local acceptance record's summary is kept in the repository.

Later work builds on the same formats:
- Block 9's records join the published set when Block 9 is built, with their own fixtures;
- MCP output schemas, declared formats in artifact references and converters to other tools follow the consumers that need them.

Before retiring this guide, retain consequential design rationale in ADR-0009 and necessary conventions and instructions in operational docs.
