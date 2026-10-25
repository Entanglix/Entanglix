# ADR-0008: Results Catalog

**Date:** 2026-10-24
**Status:** Accepted
**Purpose:** fix what a catalogued candidate is, where catalog records live, how a candidate is staged and promoted, which recorded checks block promotion, how lineage is read, how a later job reuses a catalogued device, how a control solve starts from a supplied pulse, and what the first version leaves out. Field lists, formats, and command names are documented with the implementation.

---

## 1. Context

The [harness contract](../../HARNESS_CONTRACT.md) fixes these points for the catalog:
- **Catalog entry (§1):** a human-promoted result that keeps its source-run and parent-entry lineage.
- **Tools (§5):** `list_catalog` and `read_entry` are read-only, `register_result` stages a candidate as a controlled write, and `promote` is a human action that research tools cannot perform.
- **Promotion (§6):** the entry records the promoter, its source runs and parent entries, the checks run with their outcomes, and the relevant checks that did not run. A result with a failed check cannot be promoted, except for a violated declared expectation; this record adds an unmet target to that exception (§2.4). Reuse records both the source entry and the new run.

[ADR-0003](0003-runs-verification-and-dispatch.md) makes checks a separate request with no run-level verdict, and has a result's standing rest on the checks its user requested.

Before this record:
- **History:** each run directory's runner records form the run ledger. Every attempt state is readable, check attempts are linked as child runs, and manifests carry `parent_run_ids`. No record names a catalog entry.
- **Designs in runs:** every computation run freezes its system and the files it references. A sweep point, a calibration solution, and a control spec's parameter values are named values applied to that system. A control run publishes its pulse in the neutral form.
- **Check outcomes:** each check reports `pass`, `fail`, `inapplicable`, or a descriptive result. A violated declared expectation and an unmet target carry their own failure meanings.
- **Control starts:** the baseline starts from a default, random, or zero pulse with a seed. The generated ProblemSpec has no field for a starting pulse, and Piccolo's ProblemSpec runner reads none.
- **Human actions:** the documented runtime restriction already denies `entanglix promote`.

Promotion is the user's decision, and the harness does only what the user requests.

## 2. Decisions

### 2.1 What a candidate is

- **A candidate is a design object,** whatever job produced it:
  - **a device:** a system file together with every file it references (subsystem sources and operator arrays), with their digests;
  - **a pulse:** in the neutral form, with the IDs of its channels in order;
  - **a device and a pulse together:** from a control run, whose pulse was solved for its device.
- **Sources.** One completed run supplies the design:
  - **Device:** the run's frozen system. For a sweep point, a calibration solution, or a control spec's parameter values, the values are written into the system file's parameter table by the rewrite that applies device-update proposals ([ADR-0007 §2.9](0007-calibration-from-spectroscopy.md#29-device-update-proposals)), with its refusals.
  - **Calibration solution:** every fitted device value of the solution is written, determined or not, so the device is the one its checks evaluated. The candidate lists the undetermined parameters and references the solution's identifiability report. The axis map's offset and period describe the measurement setup and are recorded as context only. Device-update proposals keep their own rule.
  - **Pulse:** a control run's neutral pulse.
  - **Files a script published:** a system file that validates with the files it references, or a pulse in the neutral form that names its channels.
- **Not device values:** sweep axes on evolution settings, such as drive amplitudes, carriers, and frame frequencies, are protocol settings. The candidate records them as context only.
- **Evidence, not designs:** simulation results such as spectra, populations, and fidelities are cited by the entry, not catalogued.
- **Outside the design scope:** gate-model algorithms and compiled circuits.

**Rationale:** a later job uses a design, not the job that found it, so the candidate is defined by what it is. A device is a whole system: parameter values alone would lose its subsystem files and operator arrays, and reuse would depend on name and unit matching against another system. Writing values through the proposal rewrite keeps one rule for where device values live. A calibration check evaluates the whole fitted solution, so a device that kept some original values would not be the one its checks examined. A proposal stays conservative because applying it changes the device model every later job reads; a catalogued device is a separate, named design whose limitations travel with it.

### 2.2 Catalog records

- **Location:** `_catalog/` in the runs root, beside `_audit/`, `_launches/`, and `_investigations/`. It holds `staging/<staged-id>/` and `entries/<entry-id>/`. Each folder holds one record and the candidate's design files.
- **Immutability:** each folder is published once, atomically, and never changed. The harness offers no edit or withdrawal of an entry.
- **Paths:** every path in a record is relative to the runs root.

**Rationale:** a candidate is a later decision about a run, not part of it, and one location lists the catalog without a scan of every run. Runs-root-relative paths survive a move of the whole root. The underscore prefix keeps the folder apart from system IDs, as for the other records.

### 2.3 Staging

- **`register_result`** stages a candidate from a named source run, its kind, and a selector when the run holds several designs: a sweep point or a calibration solution.
- **The design is copied:** the staging folder receives the device's files, with the system file written out where values were applied, and the pulse. A pulse from a control run takes its channel IDs from the run's frozen control spec.
- **The evidence is referenced:** the record names the source run, its investigation when it has one, and the digests of its result, its artifact, and the source files. Check outcomes are not copied; promotion reads them.
- **Also recorded:** the caller's claim note, and the parent entries named in the source run's manifest.
- **Refusals:** a run that is not completed, holds no design of the requested kind, or lacks a needed selector.
- Staging changes no run.

**Rationale:** a sweep-point device exists as no file until it is written, and the files are small. A copied design stays readable if its source run later moves or disappears; the evidence stays where the runner wrote it, bound by digest.

### 2.4 Promotion

- **`entanglix promote <staged-id>`** is an operator CLI command, recorded with the operator source and the OS user, as investigation decisions and device-update applications are. No research tool performs it. A staged candidate is promoted at most once.
- **No check is required.** An entry with no checks lists the relevant checks that did not run.
- **Evidence binding:** promotion reads the source run's verification history. Only completed check attempts bound to the source's current result and artifact digests count, and only when their recorded scope covers the candidate:
  - for a sweep point, a check at that point or over the whole grid;
  - for a calibration solution, a check on that solution;
  - otherwise, a check on the run's result.
- **Unusable evidence:** an attempt that is invalid, unfinished, or bound to another result is listed as unusable. It neither counts nor reads as a check that did not run.
- **Blocking:** a counted `fail` from `integrity`, `segments`, `spectral_refinement`, `replay`, or `constraints`. These report numerical disagreement or a violated pulse constraint. A recorded failure blocks even when a later attempt of the same check passed.
- **Recorded without blocking:** a violated declared expectation (`declared_expectation`, and `holdout` with a declared coverage expectation), and an unmet target (`target`). Descriptive checks never block.
- **Source mismatch:** promotion refuses when the source run is missing, or when its result, artifact, or source-file digests differ from those recorded at staging.
- **The entry records:** the entry and staged IDs, the promoter, the time, the source run and its investigation, `parent_entry_ids`, each counted check with its outcome, attempt, and policy version, the relevant checks that did not run, the unusable evidence, and the claim note.
- **A refused promotion writes nothing.**

**Rationale:** checks run on request, so the entry states its scope instead of demanding one. A check counts only where it examined the candidate, so a failure elsewhere in a grid does not block a point it did not cover, and a point it did not cover gains no standing from it. The blocking checks say the design's reported numbers or its pulse are not as stated. An expectation may be wrong, and a target is the user's goal: a candidate below it can still be a useful start or comparison. Both attempts of a contradicted check stay on record, and promotion does not choose between them. Binding to digests stops a changed or missing source from passing as one with no checks.

### 2.5 Lineage and reads

- **`list_catalog`** lists entries and staged candidates.
- **`read_entry`** gives an entry or staged candidate with:
  - its source run and that run's ancestors through `parent_run_ids`;
  - its parent entries;
  - the runs and investigation bases that reused it, found from their records.

  A missing source run is reported as unavailable, and the copied design stays readable.
- **No trial count.** The source run and its investigation lead to the full history, failed attempts included, through `list_runs`, `read_run`, and `read_investigation`, whose attempt tables already count attempts by state.
- Failed and unpromoted history stays readable without the catalog.

**Rationale:** promotion records when the user was satisfied, not how strong the evidence is, so a count on the entry would claim more than it shows. The history it would summarise is already readable.

### 2.6 Reusing a device

- **In a job:** the specs of `simulate`, `control`, and `calibrate` may name an entry in place of the system path, but not both. A sweep names it through its base simulation spec. The job tools take no new arguments.
- **Freezing:** the run freezes the entry's device files under `spec/` as usual, so reruns and checks need no catalog. Its manifest records `source_entry_ids`, and the entry's source run joins `parent_run_ids`.
- **In an investigation:** a base may name an entry. Drafting checks that the base's digests for the system and every file it references equal the entry's. Admission still matches exact digests, and each launch under that base records the entry.
- Only a promoted entry can be named. Continuing work from an unpromoted run uses its files and records it as a parent run, as now.

**Rationale:** a spec already carries a job's inputs, so the entry belongs there and every surface that takes a spec gains it at once. An explicit name, rather than a path into the catalog, makes the lineage deliberate; control also requires its system beside its spec. Matching the complete digest map keeps an investigation base exact.

### 2.7 Starting pulse

- **The user chooses the start:** `init` as default, random, or zeros, with a seed, as now; or a starting pulse from an entry, a control run, or a file in the neutral form that names its channels. A starting pulse excludes `init` and `seed`.
- **Matching:** an incompatible pulse is refused, never clipped or resampled.
  - Channels are matched by ID. A missing or extra channel is refused, and the pulse is reordered to the problem's channel order.
  - The knot count equals the problem's. The times start at zero, increase, and fit the problem's step bracket under the rule of the `constraints` check, including its roundoff allowance.
  - Amplitudes and declared derivatives lie within the problem's bounds, under the tolerances of the `constraints` check.
- **Grid:** the amplitudes are used knot for knot on the solver's own grid. The supplied times identify the knots and must fit the bracket; they do not replace the grid. Propagation, derivative constraints, and the returned times are the solver's, as for any start.
- **Freezing:** the pulse is frozen as its own file beside the generated ProblemSpec, which is unchanged. Its digest enters the launch record, and its source is recorded: `source_entry_ids`, a parent run, or the digest of an outside file.
- **Solver support:** a solver supports starting pulses only when its capabilities report `initial_pulse` as true. The launch reads this from the solver that will run, after any fallback, and records it with the selection. Any other solver is refused at launch.
- **Delivery:** a frozen starting pulse is passed to the supporting solver's `validate` preflight, when it has one, at launch and again in the job, and to `solve`. A solver without support never receives it.
- **Binding:** a solver that declares support starts from exactly the supplied amplitudes, or refuses the pulse in its preflight before a run exists. The open baseline supports it and refuses a pulse outside its own bounds, including its zero end knots, instead of letting its optimiser clip it. Piccolo does not support it in this version.
- The solver trace records the starting pulse's digest and source. The control checks are unchanged and test the returned pulse.

**Rationale:** the starting pulse is where the user chose to begin the search. A solver that ignored it would report a result from a start the user did not choose, so it is binding where [ADR-0005 §2.3](0005-control-problems-and-solvers.md#23-the-shared-problem-subset) lets guidance be ignored. Keeping the pulse out of the shared ProblemSpec keeps that file identical between solvers and valid for Piccolo's schema. Retained pulses carry small roundoff in their knot times, which the constraints rule already accepts. Any grid within the step bracket states the same problem, so the start keeps its amplitudes on the solver's grid; adopting the supplied times would make the solver's grid depend on where the start came from. Only the solver can tell whether it can start from a given pulse, so it sees the pulse in its preflight, before admission. Piccolo's runner takes no initial pulse, and building its problem outside that runner would need a probe that no workflow yet asks for.

### 2.8 Tools and conduct

- **MCP:** `register_result` as a controlled write, and `list_catalog` and `read_entry` as read-only tools. No tool promotes. The job tools are unchanged; their specs carry an entry and a starting pulse.
- **Conduct:** stage only when the user asks, never promote, and cite an entry with its check scope.
- **Skills:** the design-sweep and control skills gain the reuse steps.

**Rationale:** the permission classes are the contract's, and live refusals of the bare `promote` command under the documented rule are already retained.

### 2.9 Scope of the first version

- **In:** devices, pulses, and the two together; staging, promotion, and reads; device reuse in jobs and investigation bases; starting pulses for the open baseline.
- **Out:**
  - starting pulses for Piccolo;
  - resampling a pulse onto another grid;
  - editing or withdrawing an entry;
  - a trial count.

**Rationale:** each of these waits for a workflow that needs it.

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| Parameter values alone, substituted by name into a later system | Loses subsystem files and operator arrays, and needs name and unit matching |
| Candidates limited to certain job kinds | A later job uses a design, not the job that found it |
| Records inside the source run's directory | A candidate is a later decision; listing the catalog would scan every run |
| Referencing design files in the source run without a copy | A sweep-point device is no file, and an entry would lose its design when a run moves |
| Checks required before promotion | Checks run on request; the entry states its scope instead |
| Every failed check blocking, a missed target included | A target is the user's goal; a candidate below it can still be a useful start or comparison |
| No failed check blocking | The catalog would hold designs whose own checks found their numbers or pulse not as stated |
| A failure elsewhere in a grid blocking a sweep point | The check did not examine that point |
| Only the latest attempt of a check counting | Both outcomes are on record, and promotion would choose between them |
| A missing source treated as no checks run | The entry would claim a scope it could not read |
| A trial count on each entry | Says how many runs preceded the user's decision, not how strong the evidence is; the history stays readable |
| An entry argument on every job tool | Duplicates the spec, which already carries a job's inputs |
| Reuse by a path into the catalog | Makes lineage depend on where a file sits; control requires its system beside its spec |
| A starting pulse in the shared ProblemSpec | Its pulse table is shared and closed, and Piccolo's schema has no such field |
| A starting pulse as guidance a solver may ignore | The user's start would silently not be used |
| Clipping or resampling an incompatible starting pulse | Changes the user's pulse without saying so |
| A promote tool for agents | Promotion is a human action |
| Piccolo starting pulses through a problem built outside its ProblemSpec runner | Needs a probe and a second Piccolo path, and no workflow needs it yet |

## 4. Consequences

- **Contract:**
  - §1: a catalog entry is a promoted design; a solver's capabilities may declare starting-pulse support, and `solve` takes an optional starting pulse; a control spec may name a starting pulse, frozen beside the unchanged ProblemSpec.
  - §5: `list_catalog` and `read_entry` also show staged candidates.
  - §6: investigation attempt counts include unsuccessful attempts; the promotion paragraph states the staging location, the copied design, the counted checks, the blocking rule with its two exceptions, and the source-mismatch refusal.
  - §8: the history-and-promotion evidence names the counted-check rule and evidence binding.
- **ADR-0003:** the promotion consequence is rewritten in place to the rule of §2.4.
- **ADR-0005:** rewritten in place. A supplied starting pulse is binding, not guidance; `solve` takes it optionally; the baseline declares support; the Piccolo adapter does not.
- **Roadmap:** Block 11 drops trial counts and names designs and reuse in jobs and investigations; the evidence plane counts attempts; §6 adds the catalog extensions of §2.9; §7 records the settled decisions.
- **Checklist:** the Block 11 items change to match.
- **Operational references:** a new catalog reference documents candidates, staging, promotion, the check scope, and reads. The protocol, control, investigation, and MCP references document an entry in a spec, the starting pulse, a base naming an entry, and the new tools.
- **Example:** a catalog example registers, promotes, and reuses a co-design candidate.
- **Tests cover:**
  - **Staging:** candidates from a sweep point, a calibration solution, a control run (device, pulse, and both), a simulation's system, and files a script published; a device with a subsystem source and an operator array keeping both; a calibration-solution device carrying every fitted device value, with the undetermined ones listed; a written system constructing the same model as its source point; evolution axes recorded but not written; a parameter outside the system file's table, an incomplete run, and a missing selector refused; the source run unchanged; concurrent staging giving distinct immutable records.
  - **Promotion:** an entry with no checks listing the relevant checks not run; each blocking check refusing by name; a failure at another sweep point not counted; a violated declared expectation, a missed holdout coverage expectation, and an unmet target recorded and allowed; a failure followed by a pass still refusing; a check bound to an earlier result listed as unusable; an artifact changed after staging and a missing source refused; a refused promotion writing nothing; a second promotion of one candidate refused; the promoter's source and OS user recorded.
  - **Reuse:** a sweep on an entry's device recording the entry and its source run; an entry and a system path together refused; a rerun needing no catalog; an investigation base whose referenced file differs from the entry's refused; launches under a base naming an entry recording it; `read_entry` listing reuses and reporting a missing source as unavailable.
  - **Starting pulse:** the baseline's first evaluation at the supplied amplitudes, including a pulse on a slightly shifted permitted grid, with the solver's own times returned; a supporting test solver refusing a particular pulse in its preflight before any attempt exists; channels reordered by ID; a missing or extra channel, a different knot count, times outside the bracket, and amplitudes or derivatives beyond the bounds refused without clipping; retained Piccolo knot times accepted; Piccolo refused, also as a fallback; a solver without the capability never receiving the pulse; `init` or `seed` beside a starting pulse refused; the source recorded for an entry, a run, and a file.
  - **Tools:** MCP and CLI parity for staging and reads, with no promote tool on the research surface.
