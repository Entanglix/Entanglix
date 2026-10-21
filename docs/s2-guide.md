# S2 Guide — Calibration

**Status:** Planned. The decisions are recorded in [ADR-0007](adr/0007-calibration-from-spectroscopy.md); completion and acceptance evidence belong to the checklist.
**Scope:** S2, Block 10 (protocol `calibrate`).
**Authority:** [roadmap](roadmap.md#block-10--protocol-calibrate), [ADR-0007](adr/0007-calibration-from-spectroscopy.md), [ADR-0003](adr/0003-runs-verification-and-dispatch.md), [ADR-0004](adr/0004-investigations-and-agent-sessions.md), and [harness contract](../HARNESS_CONTRACT.md#3-verification). [The checklist](checklist.md#s2--calibration-block-10) owns completion status and acceptance evidence.
**Audience:** the owner and implementation agents. This is a planning document; operational docs and production agents must work without it.

## 1. Outcome and boundary

S2 adds calibration from spectroscopy. At completion, an expert user or their agent can:

1. Convert a measurement into the uniform format, split off holdout data before fitting, and have each dataset validated and frozen with its provenance.
2. Bind a measured bias axis to a named flux parameter through a fitted axis map, or fit fixed-bias data without one.
3. Run `calibrate` on raw maps or on points, with supplied labels or with automatic assignment from a declared candidate family, an optional anchor, matching settings and starts. The run returns its assignment, fit, outcome and uncertainty report.
4. Request the `holdout`, `identifiability` and `refinement` checks, supplying the holdout dataset and naming the solution when several remain.
5. Write a device-update proposal from a named solution, and apply it as a human action.
6. Do all of this through the CLI and the MCP server, standalone or inside an investigation.

S2 covers transition-frequency fitting on the CPU, with one bias axis driving one flux.

It has no dynamics-based calibration, gate-charge axes, multi-loop crosstalk, profile likelihood, posterior sampling, laboratory converters, or interactive tools. Control and co-design are unchanged; they read a device model a human has updated.

No step needs a live agent session. A live session is the owner's call (§4).

## 2. Build by step

```text
Probe (done) --> ADR-0007
    |
10-1 measurement format and intake
    |
10-2 flux parameters and axis binding
    |
10-3 extraction
    |
10-4 calibrate: assignment, fit, outcomes and report
    |
10-5 checks: holdout, identifiability, refinement
    |
10-6 proposals and apply
    |
10-7 investigations, MCP and skill
    |
10-8 example and S2 evidence
```

Steps complete in order. 10-2 is independent of 10-1 and may move earlier.

| Step | Work | Evidence before advancing |
|---|---|---|
| 10-1 — Measurement format (ADR-0007 §2.2) | The NPZ-plus-TOML dataset, with map and point levels, an optional bias axis, and optional point linewidth, amplitude and assignment; the declared conditions and provenance; validation at intake; freezing with its digest; converter provenance recorded as unverified; a raw-file digest checked when the raw file is supplied; the split by a declared contiguous bias range into a holdout part and the remainder, each recording its parent's digest and selected bias indices | Datasets with a map, points, or both validate, including points with only a centre and its uncertainty, and datasets without a bias axis. Pickled or object arrays, axes that do not match their arrays, and missing units are refused with diagnostics. A frozen dataset's digest matches its source. A mismatched raw-file digest is refused. A split's parts are disjoint, together cover the parent, and record its digest and their selected indices. An interior range leaves a remainder of two intervals, recorded exactly. The report states that the conversion is unverified |
| 10-2 — Flux parameters and axis binding (§2.8) | `[[model.external_fluxes]]` values accepting `{ parameter = "name" }`; the axis map from bias to flux, with its offset and period as nuisance parameters of the calibration spec; the symmetry conventions | A circuit with a flux parameter constructs the same spectrum as the literal value. A sweep over the flux parameter reproduces separately built models. Literal fluxes and the fluxonium constructor behave as before. `validate` shows the bound parameter |
| 10-3 — Extraction (§2.3) | Declared background removal, I/Q projection, detection threshold and line shape, frozen with the run; the projection learned from training data only; per-peak centre, centre uncertainty, linewidth and amplitude; failures per column | On a fixed-seed regression set of synthetic maps, centre coverage at one and two reported standard uncertainties, with failed columns kept in the denominator, matches its recorded values (§4). Columns with no peak, failed fits, and unresolved overlaps are reported. Learned settings come only from the dataset being extracted |
| 10-4 — `calibrate` (§2.1, §2.4–§2.6, §2.8) | The calibration spec: system, training datasets, fitted parameters with priors, scales, nuisance parameters, candidate family, the optional anchor, matching range, cutoff, resolution, iteration limits, starts, conventions, merge tolerance, and an optional refinement. The runner job runs extraction, assignment (supplied labels, or automatic with the optional anchor) and the bounded weighted least-squares fit on full construction. The shared diagnostic calculation (rank, undetermined combinations, covariance, correlations, prior-width ratios) with the protocol's tolerances, recorded in the result. The result carries the outcome, assignment, fit statistics, diagnostics, bounds hit, distinct solutions and their largest marginal separation | A fixed-seed regression set of clean datasets reproduces its recorded outcomes and parameter coverage (§4). Fully labelled points keep their labels and fit without automatic assignment. Fixed-bias data fits without an axis map. The following are reported as specified: a single-bias, two-parameter case (insufficient evidence, its undetermined combination, no finite uncertainty, reduced χ² unavailable); an anchor failure, with its counts; non-convergence at the iteration limit; and the equivalents removed by the conventions. A fitted name that is not in the system file's parameter table or enters no construction, a label outside the family, a partly labelled input, and a point missing a field its procedure uses are refused at launch. The result records the tolerances used |
| 10-5 — Checks (§2.7) | `holdout`, `identifiability` and `refinement` in the corrector, on request; the check request supplying the holdout dataset and naming the solution, frozen in the verification run and bound to the source run; holdout matching in prediction-uncertainty units; an optional declared coverage expectation; identifiability on the corrector policy's tolerances | The fitting job's frozen inputs contain no holdout data, and the holdout check consumes a 10-1 split part or a separate measurement. A holdout dataset that is a fitting input, its parent, or an overlapping part is refused, and so is a partly labelled holdout dataset. Holdout uses only the procedure its source recorded: maps after a fit without extraction settings, and unlabelled points after a labelled-point fit, are inapplicable; labelled holdout points are evaluated after a labelled-point fit. A check on a run with several solutions is refused unless one is named. Each check is inapplicable without its input or without fitted parameters. The holdout check reports every count and both coverages, and a deliberately displaced model lowers coverage over extracted peaks. Identifiability agrees with the fit's report and records both sets of tolerances. No outcome is pass or fail without a declared expectation |
| 10-6 — Proposals (§2.9) | `propose_device_update` next to the run; the solution named when several remain; the binding to every frozen model input; the operator command that applies a proposal | A proposal lists device parameters only, marks undetermined ones as not proposed, and records the checks that ran. Applying it writes value and source into the system file and leaves priors unchanged. Apply is refused after the system file, a subsystem source or an array file changes, and there is no apply on the research surface |
| 10-7 — Investigations, MCP and skill (§2.10) | `calibrate` as a permitted operation with exact bases; reruns; the `calibrate` and `propose_device_update` MCP tools; the apply command restricted like the other human actions; the `calibrate` skill | An exact base admits the job, and any changed input is refused with the existing scope messages. A scripted MCP client matches the CLI for calibrate, check and propose, and a refused request leaves no attempt. The skill tests pass, and the operational reference stands without this guide |
| 10-8 — Example and S2 evidence (§3) | The calibration example: generator script, converted training and holdout datasets, system, calibration spec, run, checks, proposal, and apply on the example's own system copy; the doc alignment edits of §4 | The example runs end to end through the CLI and through a scripted MCP client. The report gives each known parameter's error in units of its reported uncertainty (§4), the holdout data is absent from the fitting inputs, and the report and proposal are inspectable. Block 10's checklist items pass |

Each step reruns the existing acceptance studies whose plans reach the code it changes, decided from its actual diff. 10-2 changes circuit construction and is the step likeliest to reach them.

## 3. The example

The example is recorded in its own directory. Its values follow the [calibration probe](../tests/probes/s2_calibration_results.json).

| Item | Choice |
|---|---|
| System | A two-junction transmon circuit with a shunt capacitor and a flux parameter on one junction's loop. `EJ1`, `EJ2` and `Csh` are fitted with uniform priors. The junction charging energy is fixed |
| Data | A generator script writes synthetic two-tone maps from known parameters: f01, f12 and the two-photon f02/2, with declared amplitudes, linewidth, background and noise. It converts them into the uniform format with recorded provenance. A separate holdout measurement covers a different bias range |
| Axis map | Coil bias to flux, with offset and period fitted as nuisance parameters, a positive period, and an offset window |
| Family and anchor | Transitions from the ground and first excited states up to the third, at photon orders 1 and 2. The anchor is f01 |
| Search | Declared starts a few percent from the generating values, as a fabricated device would be. The matching range, cutoff and resolution are those of the probe |
| Requested checks | `holdout` on the separate holdout measurement, `identifiability`, and `refinement` with a declared refinement |
| Ending | A device-update proposal applied to the example's own copy of the system file. The report states the anchor assumption, the prior-width ratios and any limitation |

## 4. Conventions to settle with the implementation

- **Field names.** The dataset, calibration-spec and proposal fields are named in 10-1, 10-4 and 10-6, and documented in the calibration reference.
- **Tolerances.** The numerical-rank and undetermined-direction tolerances belong to the calibration protocol and arrive with the shared diagnostic calculation in 10-4; each fit records the values it used. The identifiability check (10-5) takes its tolerances, and its agreement tolerance, from the corrector policy, and reports both sets. The matching settings, starts, scales and merge tolerance are declared by the user in the spec. The harness sets no defaults for them, and the reference documents the probe's values as examples.
- **Forward model.** Full construction at every evaluation is the first implementation. Reuse across bias points and parameter steps is added only if measured production cost requires it, with a test against full construction then. The probe results hold the cost of full construction per bias point.
- **Recovery evidence.** Recovery is judged statistically, never value by value: coverage at one and two reported standard uncertainties across declared noise realisations, with failures kept in the denominators. The probe supplies that evidence. Tests use a small fixed-seed regression set, chosen before the code is written, and compare its outcomes and coverage with recorded values. The example is one realisation, so it reports standardised errors and asserts no interval.
- **Generator.** The example's generator is example code, not a harness feature.
- **Doc alignment.** The roadmap, checklist and contract edits of ADR-0007 §4 are made with this guide's commit. The operational references follow their steps.
- **Skill.** `calibrate` is a new skill. The conduct changes only if the tests show a need.
- **Live sessions.** A scripted MCP client is the required evidence for 10-7 and 10-8. A live session runs only on the owner's word, after its share of the usage window is estimated, and a full rerun of one is not planned.

## 5. Layout and validation

Suggested locations, created as code arrives:

```text
src/entanglix/measurements.py        # uniform format, validation, freezing (10-1)
src/entanglix/models.py              # flux parameter references (10-2)
src/entanglix/protocols/calibrate.py # spec, extraction, assignment, fit, diagnostics, report (10-3, 10-4)
src/entanglix/corrector/calibration.py   # holdout, identifiability, refinement (10-5)
src/entanglix/proposals.py           # propose and apply (10-6)
src/entanglix/investigations.py, cli.py, mcp.py   # operation, commands, tools (10-7)
examples/<calibration>/              # generator, datasets, system, spec, recorded results
docs/operations/calibration.md       # format, spec, report, checks, proposals
skills/calibrate/SKILL.md            # procedure
```

Validation commands, as in M5:

```sh
uv run --locked --exact pytest -m "not gpu"
uv run --locked --exact --extra mcp pytest -m "not gpu" -k "mcp"
uv run --locked --exact ruff check .
uv run --locked --exact ruff format --check .
```

Each step runs the tests its changes can reach and adds its tests to the existing CI jobs.

## 6. Exit evidence and handoff

[Checklist S2](checklist.md#s2--calibration-block-10) is the sole acceptance list; this guide adds no completion gates. The example's runs, checks and proposal are Entanglix records. Their summaries are kept in the repository and linked from the example; the run directories stay in the local runs root.

Later work builds on the same records:
- co-design uses a device model updated from a proposal;
- the catalog promotes a calibration with its lineage;
- interoperability shares the measurement schema;
- dynamics-based calibration, gate-charge axes and multi-loop crosstalk follow the workflows that need them.

Before retiring this guide, retain consequential design rationale in ADR-0007 and necessary conventions and instructions in operational docs.
