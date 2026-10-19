# M5 Guide — Device–Control Co-design

**Status:** Planned. The decisions are recorded in [ADR-0006](adr/0006-device-control-co-design.md); completion and acceptance evidence belong to the checklist.
**Scope:** M5, Block 8 (device–control co-design).
**Authority:** [roadmap](roadmap.md#block-8--devicecontrol-co-design), [ADR-0006](adr/0006-device-control-co-design.md), [ADR-0005](adr/0005-control-problems-and-solvers.md), and [harness contract](../HARNESS_CONTRACT.md#4-bounded-investigations). [The checklist](checklist.md#m5--devicecontrol-co-design-block-8) owns completion status and acceptance evidence.
**Audience:** the owner and implementation agents. This is a planning document; operational docs and production agents must work without it.

## 1. Outcome and boundary

M5 makes co-design an investigation over M4's control runs and M3's investigations. At completion, an expert user or their agent can:

1. Approve one control base over a device-parameter region and an allowed seed set, with the required seeds declared in the evidence requirements.
2. Declare the frame as a named transition, with carriers following it and bounds as a fixed nominal Rabi-rate bound, resolved for each candidate at launch.
3. Solve candidates inside that base without further approvals, linked to their screening runs.
4. Request an uncertainty diagnostic on a returned pulse, over the samples declared in the base.
5. Compare candidates in a finding over the investigation report, following the comparison guidance in the operational reference.
6. Do all of this through the CLI and through the MCP server.

M5 covers closed-system evaluation, on CPU, of candidates that one approved base describes. There is no co-design protocol, device optimiser, sub-investigation, sweep-point binding, comparison view, or open-system evaluation. Standalone sweep/search and control stay unchanged. A private solver plugin attaches as in M4.

No step needs a live agent session; a live session is the owner's call (§4).

## 2. Build by step

```text
Probe (done) --> ADR-0006 (accepted)
    |
8-1 control parameters and seed variation
    |
8-2 named transition and Rabi-rate bounds
    |
8-3 lineage through MCP
    |
8-4 uncertainty diagnostic
    |
8-5 report, guidance and skill
    |
8-6 example and M5 evidence
```

Steps complete in order; 8-3 is independent and may move earlier.

| Step | Work | Evidence before advancing |
|---|---|---|
| 8-1 — Control parameters and seeds (ADR-0006 §2.2) | An optional `[parameters]` table in `control.toml`, applied and frozen as simulate's; control bases that vary `parameters.<name>` and `pulse.seed`; candidate values passed at launch as simulate's varied values are, through Python, `--vary` on the CLI, and `values` on the MCP `control` tool, with the authored base unchanged and the existing scope check; the resolved inputs and the seed frozen in each run | A control run's frozen `system.toml` holds the overridden values. A request inside the region is admitted without a revision; one outside it, or one changing solver settings or any other input, is refused with the existing scope messages. Each run records its seed. `rerun_control` preserves the candidate values and seed and checks them against the effective revision. Exact control bases and standalone control behave as before |
| 8-2 — Named transition and Rabi-rate bounds (§2.3) | A frame frequency declared as the transition between two named levels of a subsystem's isolated eigenstates; channels that follow the frame; a fixed nominal Rabi-rate bound for amplitude and both derivatives; the common-derivative-bound rule and its tolerance; resolved values in the frozen inputs and the launch listing | The resolved frequency equals the level difference of the constructed model with overrides applied. Declaring the first gate by transition and a Rabi-rate bound reproduces its explicit frame and raw bounds within the stated tolerance. A zero matrix element is rejected. Differing derived derivative bounds are refused, naming each channel's value. Explicit numbers still work, and the generated ProblemSpec carries only resolved numbers |
| 8-3 — Lineage through MCP (§2.5) | `parent_run_ids` on the `simulate`, `sweep`, `control`, and `run_script` tools, passed through and recorded as on the CLI | A scripted MCP client records the same lineage as the CLI for each tool, and a refused request leaves no attempt |
| 8-4 — Uncertainty diagnostic (§2.4) | The evaluation table in `control.toml`, excluded from the generated ProblemSpec; samples over `parameters.<name>` and circuit offset charges, marked drifting or calibrated; the requested diagnostic in the corrector; requested refinements over the whole sample set; relevance in the launch listing and `read_run` | The nominal sample reproduces the replay check's fidelity. A drifting-only table evaluates at explicit frequencies. A calibrated sample is evaluated at the frequencies of its calibration model, matching an independent replay at hand-computed frequencies. A combined sample is recalibrated for its calibrated change only. Calibrated samples without a named transition, and unsupported state correspondence, report the diagnostic as inapplicable with a reason. A requested refinement covers every sample and reports per-sample and minimum changes. No outcome is pass or fail. Corrector acceptance is re-run once, at the end of this step |
| 8-5 — Report, guidance and skill (§2.6, §2.7) | Varied values, seed, derived transition, and termination in the investigation report's attempt table; the comparison guidance and the budget policy in the control reference; the co-design procedure in the `control` skill, with its version raised | The report shows those fields where available on admitted attempts, including failed and interrupted ones. Refusals create no attempt and keep their existing reporting. The guidance and skill tests pass with the new version, and the operational reference stands without this guide |
| 8-6 — Example and M5 evidence (§2.1, §3) | The co-design example: system, investigation, control base, evaluation table, screening sweep, solves, diagnostics, and the comparison finding; the doc alignment edits of §4 | The example runs end to end through the CLI and through a scripted MCP client, choosing at least one candidate from evidence inside the approved base. It ends in a recommendation or a stated reason to stop. Block 8's checklist items pass |

All corrector changes belong to 8-4.

## 3. The example

Recorded in an example directory of its own. The probe's values come from the [co-design probe results](../tests/probes/m5_codesign_results.json).

| Item | Choice |
|---|---|
| System | The first gate's junction-shunt circuit, with three retained levels and a charge drive |
| Device region | A `Csh` range at fixed `EJ` and `ECJ`, screened first with the existing spectrum tools. The probe covered 10, 30, and 60 fF |
| Frame and channels | The 0–1 transition, with both quadratures following it |
| Bound policy | A fixed nominal 0–1 Rabi-rate bound for amplitude and both derivatives. The two quadratures share one matrix element magnitude, so the derived derivative bounds are common |
| Solver and seeds | `grape`, with an allowed seed set in the base. The evidence requirements declare the required seeds per candidate and how additional attempts are allocated. The probe used three seeds per candidate. Piccolo results, if requested, are a separate solver comparison |
| Evaluation table | Offset charge over half a period, drifting; `EJ` at a small relative offset each way, calibrated; and their combinations, as an explicit small grid. All samples are declared in the base before the comparison. Refinements of retained levels and the charge cutoff |
| Requested checks | Replay and constraints on every solve, and the uncertainty diagnostic on the pulses being compared. Target is not requested |
| Comparison quantity | Chosen before the duration or the bound is changed. Nominal fidelity saturated in the probe, while the response to offset charge differed by orders of magnitude between candidates |
| Duration and bound | Chosen in 8-6 by a bounded pilot. The probe's 20 ns and 0.06 GHz bound saturate nominal fidelity |
| Ending | A recommendation, or a stated reason to stop. An unresolved comparison that the evidence supports is a valid endpoint; a pilot need not produce a resolved ranking |

## 4. Conventions to settle with the implementation

- **Field names.** The `[parameters]`, transition, Rabi-rate bound, and evaluation-table fields are named in 8-1, 8-2, and 8-4 and documented in the control reference.
- **Tolerances.** The relative tolerance for a common derivative bound belongs to control validation (8-2). The overlap thresholds and supported cases for state correspondence belong to the corrector policy (8-4).
- **Requested refinements.** How a check request selects declared refinements follows the contract's rule for refinement declarations, settled in 8-4.
- **Doc alignment.** Two tracked doc changes wait for the owner's go-ahead in 8-6. ADR-0006 §4 replaces M5's "verified" wording in [the roadmap](roadmap.md) at lines 201, 204, and 285, and in [the checklist](checklist.md) at lines 275 and 282, with the checks requested and their scope reported. It also adds the uncertainty diagnostic to the contract's §3 `control` row.
- **Skill.** The co-design procedure extends the existing `control` skill rather than adding a skill. The conduct changes only if the tests show a need.
- **Live sessions.** A scripted MCP client is the required evidence for 8-3 and 8-6. A live session runs only on the owner's word, after its share of the usage window is estimated, and a full rerun of one is not planned.

## 5. Layout and validation

Suggested locations, created as code arrives:

```text
src/entanglix/protocols/control.py   # parameters, named transition, Rabi-rate bounds, evaluation table
src/entanglix/investigations.py      # control variations over parameters and seeds
src/entanglix/mcp.py                 # parent_run_ids on the job tools
src/entanglix/corrector/             # the uncertainty diagnostic (8-4 only)
src/entanglix/investigation_report.py, mcp_views.py   # attempt fields
examples/<co-design>/                # system, investigation, control base, evaluation table, recorded results
docs/operations/control.md           # inputs, diagnostic, comparison guidance
skills/control/SKILL.md              # co-design procedure
```

Validation commands, as in M4:

```sh
uv run --locked --exact pytest -m "not gpu"
uv run --locked --exact --extra differentiable pytest -m "not gpu" -k "control"
uv run --locked --exact --extra mcp pytest -m "not gpu" -k "mcp"
uv run --locked --exact ruff check .
uv run --locked --exact ruff format --check .
```

Each step adds its tests to the existing CI jobs.

## 6. Exit evidence and handoff

[Checklist M5](checklist.md#m5--devicecontrol-co-design-block-8) is the sole acceptance list; this guide adds no completion gates. The example's runs, diagnostics, and finding are Entanglix records. Their summaries are retained in the repository and linked from the example; the run directories stay in the local runs root.

Later work builds on the same records: open-system evaluation once a device-dependent relaxation model exists, calibration supplying model updates, and the catalog promoting a recommended candidate with its lineage.

Before retiring this guide, retain consequential design rationale in ADR-0006 and necessary conventions and instructions in operational docs.
