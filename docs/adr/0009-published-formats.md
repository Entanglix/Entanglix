# ADR-0009: Published Formats and Compatibility

**Date:** 2026-10-26
**Status:** Accepted
**Purpose:** fix which inputs, records, arrays and replies Entanglix publishes for programs outside it; the form and location of their schemas; how each version is identified; the compatibility rule; the semantic conventions that complete the schemas; and how conformance is shown. Field lists and schema files are documented with the implementation.

---

## 1. Context

The [harness contract](../../HARNESS_CONTRACT.md) §8 requires public schemas and CLI/MCP interfaces to declare versions and compatibility expectations, and incompatible inputs to fail with actionable diagnostics. Block 12 makes this concrete for programs that write Entanglix inputs or read its records without importing it.

Before this record:
- **Inputs:** each authored input kind carries `format_version = 1` and is read by a strict model that rejects unknown fields. The compile functions add the checks a model cannot make: relationships between fields, referenced files and physical quantities. Only the system schema is committed, in a root `schemas/` folder.
- **Records:** the runner, investigation, catalog, proposal, launch and audit records carry `record_version = 1`. Protocol results, sweep point results, the calibration assignment, the launch attempt and the abort request carry no version field.
- **Arrays:** results reference NumPy archives by path and digest, with no declared format. The neutral pulse names its interpolation `kind`; the earliest control pulses do not.
- **Replies:** CLI commands print JSON, except `validate` and the Markdown investigation report. The MCP server returns one JSON text payload per call, with compact views and paged tables for some reads. Both render datetimes with a space separator. `capabilities` reports the run and investigation record versions.
- **Bindings:** check reports and catalog records bind the digests of the results, artifacts and source files they cite.
- **ProblemSpec:** [ADR-0005](0005-control-problems-and-solvers.md) generates a ProblemSpec that follows ProblemSpec OSS at a recorded amicode revision, and runs Piccolo directly.

The [probe](../../tests/probes/s4_formats.py) inventoried the retained runs and the code that writes and reads each file. It compared generated schemas with the models, drafted one record and one reply both ways, and measured MCP payloads with output schemas ([results](../../tests/probes/s4_formats_results.json)).

## 2. Decisions

### 2.1 What is published

- **Inputs:**
  - the system, simulation, sweep, control, calibration and investigation specs, and the measurement dataset;
  - the archives they name: explicit-operator, initial-state and measurement archives, and a starting-pulse file.
- **Records:**
  - **Runs:** the run, start, terminal, reconciliation and abort-request records.
  - **Results:** each protocol's result: simulation, sweep aggregate and sweep point, control, calibration with its assignment, and an authored script's result.
  - **Checks:** the verification result, as one envelope with check-specific variants.
  - **Ledgers:** investigation revisions and events; staged candidates and catalog entries; device-update proposals and their applications; launch requests, calls, attempts and refusals; MCP session and call audits.
- **Arrays:** the neutral pulse wherever it appears (control artifact, starting pulse, catalog), simulation outputs, sweep aggregate and point outputs, initial and reference states, measurement archives and explicit-operator archives.
- **Replies:** the JSON replies of the CLI commands and the MCP tools.
- **By reference:** the generated ProblemSpec is ProblemSpec OSS at its recorded amicode revision. Entanglix does not republish that schema.
- **Opaque extensions:** authored outputs, solver-native files, solver traces, and authored result fields beyond the common structure. A reference publishes their path, digest and any declared format, never their contents.
- **Internal:**
  - in the run directory: execution plans, sweep manifests, check requests, code provenance, logs and plots;
  - outside it: the GPU lock and process records, and temporary and exchange files;
  - in the package: policy and engine tables;
  - on the CLI: the text outputs of `validate` and the investigation report.

  The formats reference names each internal file with its reason.

**Rationale:** the published set is what a pipeline outside Entanglix writes or reads to drive it and use its results. An authored result keeps arbitrary fields, and solvers and scripts write files Entanglix does not define, so referencing a file cannot make its contents an Entanglix format. Plans, manifests and check requests are compiled or harness-written exchange between Entanglix's own steps. Their published counterparts are the authored specs, the check call and the check result. Republishing amicode's schema would create a second copy that could drift from the recorded revision.

### 2.2 Schemas

- **Form:** JSON Schema 2020-12, one file per format and version. Each schema fixes its version field to its version.
- **Location:** shipped as package data with the distribution. The root `schemas/` folder is retired; the system schema moves into the package.
- **TOML:** TOML inputs and records are validated in the JSON data model, under the mapping of §2.5.
- **Inputs:** their schemas are generated from the existing input models.
- **Records and replies:** their schemas are generated from record models, which describe what each writer produces.
  - Writers keep building their tables, and tests validate their output against the models.
  - A structural dependency between fields that JSON Schema can express, such as one field requiring others, is stated in the model and carried explicitly into its generated schema. Other relationships stay in validation and the semantic conventions.
  - In a TOML record, an optional field is absent, never null, and the generated schema says so. Reply models keep the nulls that replies already carry, such as the absent next page of a table or an empty table cell.
- **Arrays:** a NumPy archive has a machine-readable descriptor in place of a JSON Schema, shipped beside the schemas. It lists each required array, or the naming convention its names follow, with its dtype kind and dimensions.
  - A descriptor is generated where an authoritative declaration exists.
  - Otherwise it is maintained directly, and tested against the writers' archives and the readers.
- **Drift:** a committed schema, or a descriptor generated from a declaration, must equal the one generated from the code.
- **Limits:** a schema is a necessary check, not a complete one. Relationships between fields, referenced files and physical quantities stay in Entanglix's validation. The formats reference documents them as semantic conventions (§2.5).

**Rationale:** a schema in the package travels with the code that reads and writes the format, so an installed version carries the schemas it honours. The models already state what the readers accept. Generating schemas from them, and from record models for what the writers produce, gives one source for the schema and the reader check. Hand-written schemas would need their own tests against every writer and reader. A generated schema states a structural dependency only when the model declares it for the schema, so the drift test keeps the two together. TOML has no null, so an optional record field can only be absent. A reply's null already means something to its readers. JSON Schema cannot describe archive contents, and a short descriptor read with NumPy can. The array readers and writers share no declaration today, and a model added only to regenerate a descriptor would be another copy to keep in step.

### 2.3 Version identity

- **Inputs:** `format_version`.
- **Records:** `record_version`.
  - A result, sweep point result, calibration assignment, launch attempt or abort request without the field is version 1.
  - New writes of these records add `record_version = 1`, and the version 1 schema allows the field to be absent.
- **Arrays:** the format declared in the artifact reference, where one is declared. Otherwise, the version of the spec or record that names the archive selects its descriptor.
- **Replies:** no field is added to each reply. `capabilities` reports the version of every published format, replies included; the existing run and investigation record versions stay in its reply. Reply version 1 is the first published form (§2.6).
- **ProblemSpec:** its own `schema_version` and the recorded amicode revision.

**Rationale:** every retained result and point result is version 1 in content, so absence can name it without rewriting a file. A version field in every reply would repeat the same number in every payload, while a client asks `capabilities` once. Selecting an archive's descriptor through the record that names it covers every input archive and every historical artifact without adding headers. A declared format can be added later without rewriting either.

### 2.4 Compatibility rule

- **Supported versions:** named in the formats reference, each with committed fixtures. Support never depends on what a local runs root holds.
- **Direction:**
  - New readers accept every supported version.
  - Older readers are not promised to accept newer files: the models reject unknown fields.
- **What changes a version:** a field removed, renamed, retyped or newly required, or a changed meaning. An added optional field keeps the version and is recorded in the release notes.
- **Leaving support:** a version leaves the supported list only with a release-note entry, and its fixtures go with it. Before 1.0 this may happen in a minor release, as the release notes already allow.
- **Historical files are never rewritten.**
- **Inspection is separate from replay and checks.** Reading a record of a supported version promises neither replay of a historical run nor checks on it.

**Rationale:** a pipeline that upgrades Entanglix must still read what it wrote before. Promising the reverse would forbid every additive change, because strict models reject what they do not know. Fixtures make a supported version a tested fact rather than an observation about someone's files. Check reports and catalog entries bind digests, so rewriting a historical file would break the evidence that cites it. Replay depends on compiling a frozen plan under current code, which a reader of the record does not need.

### 2.5 Semantic conventions

The formats reference documents, with each format, the rules its schema cannot state:

- **JSON mapping:**
  - TOML tables become objects and arrays become arrays.
  - Datetimes become RFC 3339 strings with `T` and an offset.
  - Integers and floats keep their TOML types.
- **Integers:** an integer field is written without a fractional part.
  - JSON Schema cannot tell `3` from `3.0`.
  - Entanglix's models enforce the rule for declared integer fields. `format_version` accepts `1.0` and reads it as `1`. Free-form maps such as solver options and variation values keep the values as given.
- **Paths:** each path field documents its resolution base, or states that it is absolute. Existing conventions are documented as they are, not migrated: attempt-relative artifact paths, runs-root-relative catalog paths, input references beside their spec, and the absolute paths in launch requests and session audits.
- **Units:** units are explicit, as each format states.
- **Cross-field rules:** the relationships between fields and referenced files that each input's validation enforces.

**Rationale:** the Done criterion asks an outside program to use the schemas and these conventions together, so each convention a reader or writer depends on is written down where the format is. Stating the integer rule with its limits keeps it from claiming enforcement that validation does not perform.

### 2.6 Replies

- **One schema per CLI command and per MCP tool reply.** The CLI and MCP definitions are separate where MCP reshapes a reply: compact views, paged tables, and lists wrapped in an object. They are shared where the shapes are equal.
- **What the schemas cover:** JSON payloads, including the diagnostics an inspection reports, such as an unreadable result.
- **Transport errors keep their current form:** a CLI failure is a message on standard error with its exit code, and an MCP failure is a tool error. They are tested separately and gain no JSON envelope.
- **Datetimes:** RFC 3339 with `T`. The CLI and MCP encoders render datetimes in ISO form in place of their string form.
- **MCP output schemas** are not advertised in this version. The reply schemas ship in the package for clients that validate.
- **Text outputs** stay unpublished.

**Rationale:** a reply schema describes what each surface returns, and the compact MCP views are deliberately not the CLI record. An inspection that reads a broken record still succeeds, so its diagnostic belongs to the payload. A failed command already has its transport's error form, and a JSON envelope for it would be a new interface. No reply schema existed before, so the first published reply version can use the standard datetime form that the records' mapping uses. Advertising an output schema through the SDK repeats the payload as text beside the structured copy. Replacing the text with a short note would hide the data from clients that read only text, so advertisement waits for a check of client behaviour.

### 2.7 Arrays

- **Descriptors:** specific to each output kind, with required arrays and dtype rules.
- **Names from authored specs:** names follow the authored spec where it supplies them:
  - observables as `observable.<name>`;
  - sweep aggregate grids by point metric;
  - the measurement and explicit-operator archives by the key mapping in their spec.
- **Observables:** real or complex.
- **An empty archive** is valid only as the aggregate of a sweep whose points declare no metrics.
- **Legacy neutral pulse:** a version 1 neutral pulse without `kind` means `zero_order`. The catalog's pulse reader accepts both forms.

**Rationale:** each output kind has its own writer and its own required content, so a single permissive descriptor would accept archives no writer produces. The earliest control pulses predate the `kind` field. Reading them as the only interpolation that exists keeps "new readers accept every supported version" true for them without rewriting their files.

### 2.8 Conformance

- **Drift:** committed schemas, and descriptors generated from a declaration, equal the generated ones.
- **Writers:** tests write records and archives through the real writers in temporary roots and validate them.
- **Fixtures:**
  - Each supported version has committed fixtures, including historical forms: a result without a version, and a pulse without `kind`.
  - Each fixture passes its schema or descriptor, and is accepted by Entanglix's reader for its format: the model and compile path for inputs; for records and arrays, `read_run`, the check loaders and the catalog, proposal, investigation and launch readers.
- **Variants:** completed, failed, aborted, interrupted with and without reconciliation, unknown, an unreadable result, a partial sweep, and an abort request. An unreadable result fails its schema, while its inspection succeeds and reports the error.
- **An outside program:** a test runs it in a separate interpreter where Entanglix cannot be imported, using TOML, JSON Schema and NumPy. It:
  - validates a test-written run of each protocol, a check run and a script run;
  - constructs inputs for each protocol, which Entanglix then accepts through its normal validation.
- **A breaking change:** a change that breaks a supported version's fixtures without a version change fails these tests. No test claims to detect a changed meaning.

**Rationale:** the retained runs are local and uncommitted, so the evidence that runs everywhere uses test-written records and committed fixtures. Schema validation alone shows the structure, and the readers show the promise of the compatibility rule.

### 2.9 Scope of the first version

- **In:** the published set of §2.1, with schemas, descriptors, fixtures and conventions; version fields for the unversioned records; ISO datetimes in replies; format versions in `capabilities`; the legacy pulse reading.
- **Out:**
  - advertising MCP output schemas;
  - declared formats in artifact references;
  - forward compatibility;
  - a hosted schema registry;
  - the amicode interoperability extensions in roadmap §6;
  - Block 9's records, which join when Block 9 is built.

**Rationale:** each waits for a consumer or workflow that needs it.

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| Publishing every file a result references | Authored and solver files are not Entanglix formats; their contents would bind Entanglix to formats it does not define |
| Hand-written schemas for records and replies | Needs separate tests against every writer and reader; the models already state what the readers accept |
| New models replacing the compile-time input checks | The input models exist; the compile checks cover what a schema cannot |
| Schemas only in the repository, not the package | An installed version would not carry the schemas it honours |
| Every version still present in retained runs counts as supported | Support would depend on a local filesystem and could imply restoring obsolete plans |
| Forward compatibility | Strict readers reject unknown fields; promising it would forbid additive changes |
| Rewriting historical records to add version fields | Breaks the digests that check reports and catalog entries bind |
| A version field in every reply | Repeats one number in every payload; `capabilities` reports it once |
| Keeping space-separated reply datetimes, documented with a pattern | Diverges from the records' mapping and from the standard form clients parse |
| Advertising MCP output schemas through the SDK default | Repeats the payload as text beside the structured copy |
| One permissive archive descriptor, empty archives allowed | Accepts archives no writer produces |
| Refusing pulses without `kind` | Retained control pulses of a supported version would become unreadable |
| Republishing the ProblemSpec OSS schema | A second copy could drift from the recorded revision |
| A JSON envelope for CLI and MCP failures | A new interface; failures keep their transport's error form |
| Making every path relative to one base | A path migration of existing public records; each path field documents its base instead |
| Removing null from every generated schema | Replies already use null with a meaning |
| Array models added only to generate descriptors | Another copy to keep in step; maintained descriptors are tested against writers and readers |

## 4. Consequences

- **Contract:**
  - §8: the opening paragraph states that schemas and descriptors ship with the package, one per format and version, and states the compatibility rule.
  - §8 table: the "Objects and interfaces" evidence adds schema drift, writer and fixture conformance, fixtures read by the readers, and an outside program that validates and constructs without Entanglix.
  - §2: every published record carries its version, and a record written before its version field existed is version 1.
- **Roadmap:** §7 records the settled decisions. Block 12 is unchanged.
- **Checklist:** the conformance item adds that fixtures are read by Entanglix's readers.
- **Operational references:**
  - A new formats reference lists the published, opaque and internal files, the supported versions with their schemas and descriptors, the semantic conventions and the compatibility rule.
  - The runs, protocol, control, calibration, catalog, investigation and MCP references link their field tables to it.
  - The runs reference adds the version of results.
  - The catalog reference adds the legacy pulse reading.
- **README:** release notes for the version fields, ISO reply datetimes, format versions in `capabilities`, the system schema's new location and the legacy pulse reading.
- **Code:**
  - record models and the schema and descriptor package;
  - `record_version = 1` in the writers of the unversioned records, the verification result's writer among them, so the corrector policy version changes;
  - ISO datetime encoding in the CLI and MCP;
  - format versions in `capabilities`;
  - the catalog pulse reader.
- **Tests cover:**
  - **Drift and writers:** drift for every generated schema and descriptor; writer output for every published record and archive, written through the real writers.
  - **Fixtures:** each supported version's fixtures, including a result without a version and a pulse without `kind`, through schema or descriptor and reader.
  - **Variants:** the status variants of §2.8.
  - **Replies:** CLI and MCP JSON replies against their schemas, with ISO datetimes and their existing nulls; inspection diagnostics in the payload; CLI and MCP transport errors unchanged; `capabilities` reporting every format's version.
  - **Arrays:** an empty sweep aggregate accepted only without point metrics.
  - **Outside program:** the outside program over a run of each protocol, a check run and a script run, and over constructed inputs accepted by normal validation.
  - **Breaking change:** a field removed from a supported version failing its fixtures.
