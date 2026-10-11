# M4 Guide — Control

**Status:** Implemented through the control spec and generated ProblemSpec, the open baseline, the Piccolo adapter, requested control checks, investigation admission and the MCP tools. Agent integration rests on a scripted MCP client; no live control session was run. Completion and acceptance evidence belong to the checklist. The decisions are recorded in [ADR-0005](adr/0005-control-problems-and-solvers.md).
**Scope:** M4, Block 7 (Protocol `control`).
**Authority:** [roadmap](roadmap.md#block-7--protocol-control), [ADR-0005](adr/0005-control-problems-and-solvers.md), and [harness contract](../HARNESS_CONTRACT.md#1-objects-and-interfaces). [The checklist](checklist.md#m4--control-block-7) owns completion status and acceptance evidence.
**Audience:** the owner and implementation agents. This is a planning document; operational docs and production agents must work without it.

## 1. Outcome and boundary

M4 adds the `control` protocol to M3's runs and investigations. At completion, an expert user can:

1. Write one `control` spec for an existing system: the frame and channels, the gate, the pulse, the bounds, an optional fidelity target, and the solver.
2. Run it through the CLI with the open baseline or with Piccolo. Both receive the same generated ProblemSpec file.
3. Read the solver's pulse in a neutral form, with the solver's own report labelled as such.
4. Request checks on the candidate: replay on another engine, the declared target, and the declared constraints.
5. Do the same inside an investigation and through the MCP server.

M4 covers closed-system gates on the shared problem subset of ADR-0005 §2.3, on CPU. Open-system control, spline pulses, free duration, GPU solves, cluster execution, and pulse warm-starts belong to later work. A private solver plugin attaches through the same entry point and is not part of this repository.

No step needs a live agent session until 7-6, and that one is the owner's call (§4).

## 2. Build by step

```text
Probe (done) --> ADR-0005 (accepted)
    |
7-1 control spec and generated ProblemSpec
    |
7-2 solver interface and the open baseline
    |
7-3 Piccolo adapter
    |
7-4 control checks
    |
7-5 investigations and M4 evidence
    |
7-6 MCP tool and skill
```

Steps complete in order.

| Step | Work | Evidence before advancing |
|---|---|---|
| 7-1 — Control spec and generated ProblemSpec | The `control` spec and its validation; the control model built by the existing frame transformation from the declared frame and each channel's carrier; the export; the generated TOML frozen with its digest; a launch listing that names dissipators left out. No solver | The generated file validates against the copied ProblemSpec OSS schema (§4). A fixed pulse propagated from the exported matrices agrees with the reference engine on the control model. The two quadrature operators match the probe's, apart from the drive's own scale. The same inputs give the same digest. The first gate's bounds are restated in the drive's own units (§3) |
| 7-2 — Solver interface and the open baseline | `SolverBackend`, entry-point discovery, a declared fallback, the missing-solver error; the baseline on the differentiable engine with SciPy's constrained optimiser; the neutral pulse; a result with solver-reported fields and a termination report; `control` as a runner job from the CLI | The baseline solves the first gate; CI runs a short solve, and the first-gate solve is retained workstation evidence. Amplitude and derivative bounds are explicit constraints: under an active derivative bound, the bound recomputed from the neutral pulse holds. A field outside its support is refused by name, and ignored guidance fields are listed. A solver the registry lacks fails before a run exists, and a declared fallback is recorded. No private import. Cost per iteration is measured, first call and later calls |
| 7-3 — Piccolo adapter | A pinned Julia project; a Julia entry that runs Piccolo's ProblemSpec runner with the iteration cap and tolerance as arguments and writes the neutral pulse; Julia version and manifest digest in the run | Piccolo solves the first gate from a file with the same digest as the baseline's. The cap and the tolerance take effect as arguments. A stop ends the solve with a terminal record and no surviving process. The duration is held. Under an active derivative bound, the bound recomputed from the neutral pulse holds by the definition of ADR-0005 §2.3. A request without Julia is a configuration error. Default tests pass without Julia; the Julia tests' report from the workstation is retained |
| 7-4 — Control checks | Replay, target, and constraints checks, and the leakage report, in the corrector; relevant checks in the launch listing and `read_run`; thresholds in the corrector policy | Each check passes and fails on a constructed pulse. An accurately reported pulse below its target fails the target check only. The constraints check passes each open solver's pulse under an active derivative bound. A problem outside the subset reports the checks as inapplicable. Corrector acceptance is re-run once, at the end of this step |
| 7-5 — Investigations and M4 evidence | `control` as a permitted operation with an exact-match base; usage, limits, stop, and deadline; `rerun_control`; the first-gate example with both solvers | `control` is admitted, refused out of scope, counted, stopped, and rerun inside an investigation. A test-only plugin is discovered through the entry point and its candidate passes the same replay. Both solvers' first-gate candidates pass the checks requested, in retained run directories |
| 7-6 — MCP tool and skill | `control` and `rerun_control` tools in the form of the existing job tools; the `control` skill; the conduct change, restoring "request run detail only when needed" for `read_run` | A scripted MCP client gets the CLI's results, and a refused request leaves no attempt. The conduct and skill tests pass with the new version. A live session only as in §4. Block 7's checklist items pass |

All corrector changes belong to 7-4.

## 3. The first gate

Recorded in an example directory of its own. The values come from the [control probe](../tests/probes/m4_control_results.json).

| Item | Choice |
|---|---|
| System | The junction-shunt circuit with three retained levels and a charge drive |
| Frame | Rotating at the 0–1 transition frequency computed from the model, approximately 5.482 GHz, with the declared rotating-wave approximation. The example declares the computed value |
| Channels | Two on the charge drive, a quarter cycle apart |
| Gate | X on the two lowest levels; the third level is leakage |
| Duration | 20 ns in 20 equal steps |
| Amplitude bound | 0.06 GHz on each quadrature, as probed on normalised operators |
| Derivative bounds | 0.1 GHz/ns and 0.1 GHz/ns², as probed |
| Fidelity target | 0.999 |
| Requested checks | Replay, target, and constraints, for each solver's candidate |

The probe normalised the quadrature operators by the 0–1 charge matrix element. The export keeps the drive's own scale, so step 7-1 restates the bounds in those units.

With the duration held, Piccolo reported 1 − 9e-10 in 100 iterations under these bounds, and the independent replay on the declared grid gave 1 − 4e-11. The open baseline loop and its measured cost are recorded with the [transmon example](../examples/transmon-control/README.md#open-baseline). 0.999 is the acceptance target, and each run reports the fidelity it attained.

## 4. Conventions to settle with the implementation

- **End points.** The [neutral pulse reference](operations/control.md#neutral-pulse-and-candidate-record) settles the N-knot/N−1-interval convention, including differences to the terminal knot, zero endpoint amplitudes for both adapters and no invented endpoint derivatives. The Piccolo adapter tests exercise that convention on its returned pulse.
- **Derivative definition.** The discrete derivative of ADR-0005 §2.3 comes from Piccolo's source. Step 7-3 tests it directly on a Piccolo pulse under an active bound, without the corrector. The constraints check applies the same definition in 7-4.
- **Schema copy.** amicode's `problemspec.oss.schema.json` is copied into the test data with its amicode commit, digest, and an attribution line (Apache-2.0). The conformance test validates generated files against it.
- **Extras.** The baseline needs the differentiable extra and the Piccolo adapter needs Julia. A request without either is a configuration error that names what is missing.
- **Live sessions.** A scripted MCP client is the required evidence for 7-6. A live session is run only on the owner's word, after its share of the usage window is estimated, and a full rerun of one is not planned.

## 5. Layout and validation

Suggested locations, created as code arrives:

```text
src/entanglix/protocols/control.py   # control spec, control model, export, generated ProblemSpec
src/entanglix/solvers/               # interface, discovery, the baseline, the Piccolo adapter and its Julia project
src/entanglix/corrector/             # control checks (7-4 only)
examples/<first-gate>/               # system, control specs for both solvers, recorded results
docs/operations/control.md           # the spec, the pulse form, solver options, checks
tests/probes/m4_control.py           # probe and its retained results
```

Validation commands after the corresponding capabilities exist:

```sh
uv run --locked --exact pytest -m "not gpu"
uv run --locked --exact --extra differentiable pytest -m "not gpu" -k "control"   # from 7-2
uv run --locked --exact --extra mcp pytest -m "not gpu" -k "mcp"                  # from 7-6
uv run --locked --exact ruff check .
uv run --locked --exact ruff format --check .
```

The default environment has neither extra, so the baseline and MCP tests are skipped there. Each step adds its tests to the existing CI jobs. The differentiable job selects by a file list and a keyword, and both are extended. The Julia tests are opt-in on the workstation; their command is in the [Piccolo acceptance reference](environment.md#piccolo-adapter-acceptance).

## 6. Exit evidence and handoff

[Checklist M4](checklist.md#m4--control-block-7) is the sole acceptance list; this guide adds no completion gates. The first-gate runs and their check reports are Entanglix records. Their summaries are retained in the repository and linked from the example; the run directories stay in the local runs root.

Later blocks build on the same records: co-design links device and control runs under one investigation, the catalog warm-starts from a promoted pulse, and interoperability shows `amico-run` accepting a generated ProblemSpec.

Before retiring this guide, retain consequential design rationale in ADR-0005 and necessary conventions and instructions in operational docs.
