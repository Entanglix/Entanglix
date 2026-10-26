# ADR-0009: Record Compatibility and Deferred Format Publication

**Date:** 2026-10-26
**Status:** Accepted
**Purpose:** fix what Entanglix promises about reading the records and inputs that earlier releases wrote, how a version is identified, and that standalone format publication is deferred.

---

## 1. Context

Entanglix users install it, and their agent drives it through the CLI, MCP and Python. The agent reads records through Entanglix's own readers and validation. People who do not use Entanglix receive artifacts the agent hands over, such as the investigation report and check reports.

Research history outlives a release. Catalog entries cite earlier runs, investigations span upgrades, and collaborators share runs across versions.

Before this record:
- **Inputs:** each authored input kind carries `format_version = 1` and is read by a strict model that rejects unknown fields. The system schema is committed with the Block 1 model.
- **Records:** the runner, investigation, catalog, proposal, launch and audit records carry `record_version = 1`, and the investigation and proposal readers require it. Protocol results, sweep point results, the calibration assignment, the launch attempt and the abort request carry no version field.
- **Pulses:** the neutral pulse names its interpolation `kind`; the earliest control pulses do not, and the catalog's pulse reader refuses them.
- **Bindings:** check reports and catalog records bind the digests of the files they cite.

The [probe](../../tests/probes/s4_formats.py) inventoried the retained runs and the code that writes and reads each file ([results](../../tests/probes/s4_formats_results.json)).

## 2. Decisions

### 2.1 Compatibility

- **Direction:** a new release reads the supported earlier versions of the records, frozen inputs and arrays that earlier releases wrote. Older releases are not promised to read newer files: the models reject unknown fields.
- **Supported versions:** every version written so far is supported. A version leaves support only with a release-note entry; before 1.0 this may happen in a minor release.
- **What reading covers:** inspecting a record's retained contents. Compiling, reusing, replaying and checking an earlier run are separate and not promised.
- **What changes a version:** a field removed, renamed, retyped or newly required, or a changed meaning. An added optional field keeps the version and is recorded in the release notes.
- **Historical files are never rewritten.**
- **Evidence:** supported historical forms retain regression tests, in the existing test modules.

**Rationale:** users keep their research history across upgrades, so a release that cannot inspect it breaks their work. Promising the reverse would forbid every additive change. Compilation, reuse, replay and checks depend on current code and current policy, which inspecting a record does not need. Check reports and catalog records bind digests, so rewriting a file would break the evidence that cites it. Tests tied to the supported forms, rather than to what a reader happens to accept, keep an accidental rejection from redefining support.

### 2.2 Version identity

- **Inputs:** `format_version`.
- **Records:** `record_version`. Formats that historically omitted it, namely protocol results, sweep point results, the calibration assignment, the launch attempt and the abort request, remain implicitly version 1. Their first incompatible revision carries an explicit version. Existing version fields remain required.
- **Arrays:** the version of the spec or record that names the archive.

**Rationale:** every retained record of those formats is version 1 in content, so absence identifies it without a writer change. Adding `record_version = 1` now would change writers, including the verification writer and so the corrector policy version, for no reader that needs it. Records that already carry the field keep requiring it, so the absence rule does not weaken them.

### 2.3 Legacy neutral pulse

A neutral pulse without `kind` is `zero_order`. The catalog's pulse reader accepts both forms, so the pulses of earlier control runs can be staged and used as starting pulses. Malformed arrays and any other `kind` are still refused.

**Rationale:** the earliest control pulses predate the `kind` field, and `zero_order` is the only interpolation that exists. This extends reuse, beyond inspection, to those pulses.

### 2.4 Deferred format publication

- Entanglix does not ship standalone schemas, array descriptors or reply schemas for its files until a consumer needs them.
- CLI/MCP interfaces, and public schemas where provided, keep declaring versions and compatibility expectations. The field references document the records, and the system schema from Block 1 stays as it is.
- The generated control problem remains ProblemSpec OSS at its recorded amicode revision ([ADR-0005](0005-control-problems-and-solvers.md)).

**Rationale:** no identified consumer currently requires standalone schemas. They could help installed users too, but schemas for every record, array and reply would need models kept in step with each writer and definitions for every command and tool, and that maintenance is not justified now. An installed user's agent can use Entanglix's readers and validation, and people without Entanglix receive reports.

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| Package-shipped schemas for every input, record, array and reply, with a test of a program that cannot import Entanglix | No identified consumer requires them; models parallel to the writers would need maintenance |
| Adding `record_version = 1` to the historically unversioned records now | Absence already identifies version 1; the field arrives with the first incompatible revision |
| Implicit version 1 for every record | Would weaken records whose readers already require the field |
| A fixture framework for each supported version | Regression tests in the existing modules protect the same promise |
| Refusing pulses without `kind` | The earliest control runs could not be staged or used as starting pulses |
| Rewriting historical records | Breaks the digests that check reports and catalog records bind |
| Forward compatibility | Strict readers reject unknown fields; promising it would forbid additive changes |

## 4. Consequences

- **Contract:** §2 states implicit version 1 for the historically unversioned formats. §8 states that CLI/MCP interfaces, and public schemas where provided, declare versions; the compatibility rule; and what reading covers. Its "Objects and interfaces" evidence adds regression tests for supported historical record forms.
- **Roadmap:** Block 12 and milestone S4 are removed. §6 lists published schemas, amicode's records and Domain Pack, and converters until a consumer needs them. §7 records this decision.
- **Checklist:** the S4 section is removed.
- **Operational references:** the run reference states the compatibility rule, what reading covers and the implicit version 1; the catalog reference states the legacy pulse reading.
- **README:** a release note for the legacy pulse reading.
- **Code:** the catalog's pulse reader accepts a pulse without `kind`.
- **Tests:** staging a pulse without `kind`, using it as a starting pulse, and refusing another `kind`.
