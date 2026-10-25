# S3 Guide — Results Catalog

**Status:** Planned. The decisions are recorded in [ADR-0008](adr/0008-results-catalog.md); completion and acceptance evidence belong to the checklist.
**Scope:** S3, Block 11 (results catalog).
**Authority:** [roadmap](roadmap.md#block-11--results-catalog), [ADR-0008](adr/0008-results-catalog.md), [ADR-0003](adr/0003-runs-verification-and-dispatch.md), [ADR-0005](adr/0005-control-problems-and-solvers.md), and [harness contract](../HARNESS_CONTRACT.md#6-history-audit-and-promotion). [The checklist](checklist.md#s3--results-catalog-block-11) owns completion status and acceptance evidence.
**Audience:** the owner and implementation agents. This is a planning document; operational docs and production agents must work without it.

## 1. Outcome and boundary

S3 adds the results catalog. At completion, an expert user or their agent can:

1. Stage a candidate from any completed run that holds a design: a device with every file its system references, a pulse, or both.
2. List and read staged candidates and entries, with their lineage and the runs that reused them.
3. Promote a staged candidate as a human action. The entry records the checks that covered it and their outcomes, and the relevant checks that did not run.
4. Name an entry in a simulation, sweep, control, or calibration spec, or in an investigation base, with the lineage recorded.
5. Start a baseline control solve from a supplied pulse: an entry's, an earlier run's, or the user's own file.
6. Do all of this through the CLI and the MCP server, except promotion, which stays on the operator CLI.

S3 has no Piccolo starting pulses, no pulse resampling, no entry withdrawal, and no trial count. Existing runs, checks, and investigations behave as before when no entry or starting pulse is named.

No step needs a live agent session: live refusals of bare `entanglix promote` under the documented rule are already retained, in the M3 conformance run and the S2 calibration session.

## 2. Build by step

```text
ADR-0008
    |
11-1 records and staging
    |
11-2 promotion
    |
11-3 device reuse
    |
11-4 starting pulse
    |
11-5 MCP, conduct, skills and references
    |
11-6 example and S3 evidence
```

Steps complete in order. 11-4 depends only on 11-1 and may move earlier.

| Step | Work | Evidence before advancing |
|---|---|---|
| 11-1 — Records and staging (ADR-0008 §2.1–§2.3, §2.5) | `_catalog/staging/` in the runs root, one immutable folder per candidate, published atomically; the device's files as the run froze them, with the system file written out by the proposal rewrite for a calibration solution, carrying every fitted device value; the pulse with its channel IDs; evidence referenced by runs-root-relative paths and digests; the claim note and parent entries; `register-result`, `list-catalog` and `read-entry` on the CLI | Candidates stage from a sweep point, a calibration solution, a control run (device, pulse, and both), a simulation's system, and files a script published. A device with a subsystem source and an operator array keeps both files with their digests. A staged device constructs the same model as its source, for a simulation with a parameter override, a sweep point, and a calibration solution. A calibration-solution device carries every fitted device value, lists the undetermined ones and references the identifiability report, and keeps the axis map as context. Evolution axes are recorded as context and not written. An incomplete run, a run without a design of the requested kind, and a missing selector are refused. The source run is unchanged. Concurrent staging gives distinct records, and no record changes after publication |
| 11-2 — Promotion (§2.4) | The operator command `promote`: evidence read through the source run's verification history, bound to the staged digests; the coverage rule for points, solutions and single results; the blocking rule and its exceptions; unusable evidence listed; the source-mismatch refusal; the entry record with the operator source and OS user; at most one promotion per staged candidate | An entry with no checks lists the relevant checks not run. Each blocking check's counted failure refuses promotion by name. A failure at another sweep point is not counted. A violated declared expectation, a missed holdout coverage expectation, and an unmet target are recorded and promotion proceeds. A failure followed by a passing attempt still refuses. A check bound to an earlier result is listed as unusable, not as a check that did not run. An artifact changed after staging and a missing source run are refused. A refused promotion writes nothing, and a second promotion of one candidate is refused |
| 11-3 — Device reuse (§2.6, §2.5) | An entry field in the simulation, control and calibration specs, exclusive with the system path, and reached by sweeps through their base spec; freezing the entry's files; `source_entry_ids` and the source run in `parent_run_ids`; an investigation base naming an entry, checked against the complete digest map at drafting; `read-entry` listing reuses and reporting missing sources | A sweep on an entry's device freezes its files and records the entry and its source run. An entry and a system path together, an unknown entry, and a staged candidate that was never promoted are refused. A rerun needs no catalog. A base whose referenced file differs from the entry's is refused at drafting, and launches under a matching base record the entry. `read-entry` lists the reusing runs and bases, and reports a removed source run as unavailable while the design stays readable |
| 11-4 — Starting pulse (§2.7) | A starting-pulse field in the control spec, exclusive with `init` and `seed`, naming an entry, a control run, or a file; matching by channel ID, knot count, the step bracket and the bounds under the constraints check's rules; the amplitudes used knot for knot on the solver's grid; freezing beside the unchanged ProblemSpec, with its digest in the launch record and its source in the manifest; the `initial_pulse` capability read from the solver that will run, after any fallback; the optional argument to `validate`, at launch and in the job, and to `solve`; baseline support with its refusal of a pulse outside its bounds; the Piccolo refusal; the trace record | The baseline's first evaluation is at the supplied amplitudes, including for a pulse on a slightly shifted permitted grid, and the returned times are the solver's. A supporting test solver that refuses a particular pulse in its preflight leaves no attempt. Channels given in another order are reordered by ID. A missing or extra channel, a different knot count, times outside the bracket, and amplitudes or derivatives beyond the bounds are refused, never clipped or resampled. A retained Piccolo pulse's knot times are accepted. Piccolo is refused at launch, also as a fallback. A test solver without the capability never receives the argument. The generated ProblemSpec's digest is unchanged by a starting pulse. The source is recorded for an entry, a run and a file |
| 11-5 — MCP, conduct, skills and references (§2.8) | The `register_result`, `list_catalog` and `read_entry` tools; the conduct's staging and citation rules; the reuse steps in the design-sweep and control skills; the catalog reference, and the protocol, control, investigation and MCP references updated | A scripted MCP client matches the CLI for staging and reads. No tool promotes, and a refused request leaves no record. The skill and guidance tests pass, and each reference stands without this guide |
| 11-6 — Example and S3 evidence (§3) | The catalog example and its recorded summaries; Block 11's checklist ticks | The example runs end to end through the CLI and through a scripted MCP client, with promotion run by the owner. The entry shows its check scope, the sweep and the warm start record their lineage, and Block 11's checklist items pass |

Each step reruns the existing acceptance studies whose plans reach the code it changes, decided from its actual diff. The catalog code stays outside `src/entanglix/corrector/`, so the policy digest does not change. 11-4 changes the baseline solver and is the step likeliest to reach the control studies.

## 3. The example

The example is recorded in its own directory and starts from the [co-design example](../examples/transmon-codesign/README.md)'s chosen candidate.

| Item | Choice |
|---|---|
| Source | A retained, completed control run of the co-design example's chosen candidate, with its recorded checks (`replay`, `constraints`, `uncertainty`, `uncertainty_refinement`). A check is rerun only when a code change since that run reaches it. When no such run is retained, one standalone control run re-solves the chosen capacitance with its best seed, with the same checks |
| Candidate | The device and the pulse together, with a claim note naming the co-design comparison |
| Promotion | The driver stages the candidate and prints the bare `entanglix promote` command. The owner runs it |
| Device reuse | A spectrum sweep over `EJ` around its design value on the entry's device, through a base spectrum spec that names the entry and declares only `EJ`, at the entry's value, since a sweep axis must be declared in its base. `Csh` is left undeclared, so the entry's value holds |
| Warm start | The same gate on the entry's device, starting from the entry's pulse with the baseline, and the same checks requested |
| Ending | `read-entry` shows the check scope, the sweep and the warm start as reuses, and the source run's lineage. The report states that the entry's standing rests on the checks requested |

## 4. Conventions to settle with the implementation

- **Field names.** The record fields, the spec's entry field, the starting-pulse field and the base field are named in 11-1, 11-3 and 11-4, and documented in the references.
- **IDs.** Staged and entry IDs follow the run ID form: a UTC timestamp and a random suffix.
- **Publication.** A folder is written under a temporary name inside `_catalog/` and renamed into place. One promotion per staged candidate is enforced at publication, so concurrent promotions cannot both succeed.
- **Coverage.** Whether a check covered a point or a solution is read from its recorded report: point rows for sweep checks, grid-scope checks for all points, and the named solution for calibration checks.
- **Tolerances.** Starting-pulse matching reuses the constraints check's functions and policy tolerances, read at launch. No new threshold is added.
- **Doc alignment.** The roadmap, checklist, contract, ADR-0003 and ADR-0005 edits of ADR-0008 §4 are made with this guide's commit. The operational references follow their steps.
- **Conduct and skills.** The conduct's version advances in 11-5. The design-sweep and control skills change; no new skill is added.
- **Live sessions.** A scripted MCP client is the required evidence for 11-5 and 11-6. A live session runs only on the owner's word, after its share of the usage window is estimated.

## 5. Layout and validation

Suggested locations, created as code arrives:

```text
src/entanglix/catalog.py             # records, staging, promotion, reads (11-1, 11-2, 11-3)
src/entanglix/protocols/*.py         # entry field in specs (11-3); starting pulse in control (11-4)
src/entanglix/investigations.py      # a base naming an entry (11-3)
src/entanglix/solvers/               # capability, optional argument, baseline support (11-4)
src/entanglix/cli.py, mcp.py         # commands and tools (11-1 to 11-5)
examples/<catalog>/                  # driver, recorded summaries
docs/operations/catalog.md           # candidates, staging, promotion, check scope, reads
skills/design-sweep/, skills/control/    # reuse steps
```

Validation commands, as in S2:

```sh
uv run --locked --exact pytest -m "not gpu"
uv run --locked --exact --extra mcp pytest -m "not gpu" -k "mcp"
uv run --locked --exact ruff check .
uv run --locked --exact ruff format --check .
```

Each step runs the tests its changes can reach and adds its tests to the existing CI jobs.

## 6. Exit evidence and handoff

[Checklist S3](checklist.md#s3--results-catalog-block-11) is the sole acceptance list; this guide adds no completion gates. The example's runs, checks and catalog records are Entanglix records. Their summaries are kept in the repository and linked from the example; the run directories and the catalog stay in the local runs root.

Later work builds on the same records:
- interoperability shares a catalogued pulse or ProblemSpec;
- Piccolo starting pulses, pulse resampling and entry withdrawal follow the workflows that need them.

Before retiring this guide, retain consequential design rationale in ADR-0008 and necessary conventions and instructions in operational docs.
