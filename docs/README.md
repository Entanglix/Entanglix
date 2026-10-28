# Documentation index

Operational entry points are the root README, the harness contract, and the research-agent conduct. Decision records keep the decisions behind them.

## Operational harness

- [../README.md](../README.md) — User entry point: what Entanglix does, installation, project setup, and release notes.
- [research-agent.md](research-agent.md) — Versioned conduct for research sessions: formulation, recovery, human decisions, checks on request, assessments, and stopping. Supplied in initial context and by the MCP server; it is separate from repository development guidance.
- [../skills/](../skills/) — Versioned `system-modeler`, `simulate`, `design-sweep`, `control`, `calibrate`, and `gpu-budget` procedures, usable through runtime skill discovery or as plain text.
- [../HARNESS_CONTRACT.md](../HARNESS_CONTRACT.md) — Shared interfaces, permissions, checks, and execution guarantees.
- [operations/system-spec.md](operations/system-spec.md) — Implemented system syntax, units, operators, evolution, drive gradients, numerical checks, and engine selection. Other usage guides may be added under `operations/` as capabilities arrive.
- [operations/runs.md](operations/runs.md) — Authored CPU jobs, listing and inspection, frozen inputs, environment and code provenance, record ownership, and execution outcomes.
- [operations/protocols.md](operations/protocols.md) — Simulation and serial sweep specifications, metrics, CPU/CUDA dispatch, requested checks and coverage, and frozen-input replay.
- [operations/control.md](operations/control.md) — CPU control jobs, closed rotating-frame models, shared ProblemSpec export, solver selection, the pinned Piccolo/Julia adapter, neutral pulses, starting pulses, requested checks, investigation admission, linked reruns and MCP usage.
- [operations/investigations.md](operations/investigations.md) — Investigation declarations, immutable revisions and findings, human starts and approvals, scoped admission, optional limits, checkpoints, named reviews, stop/resume, usage, and record inspection.
- [operations/mcp.md](operations/mcp.md) — Stdio research tools, project setup and runtime settings, explicit server configuration, bounded waiting, launch recovery, and connection/call audit records.
- [operations/calibration.md](operations/calibration.md) — Calibration from spectroscopy: the uniform measurement format, validation, digests, raw-file checks, holdout splits, peak extraction, the calibrate protocol, the holdout, identifiability and refinement checks, device-update proposals and their application, and calibration in investigations and research tools.
- [operations/catalog.md](operations/catalog.md) — Results catalog: the designs a completed run can supply, staging with `register-result`, the staged record and its evidence digests, operator promotion with its check scope, device reuse, starting pulses, reads with source lineage and reuses, and the research tools.
- [environment.md](environment.md) — Observed GPU compatibility, tested package versions, limitations, and reproduction commands.

## Decision records

- [adr/](adr/) — Public decision records, kept current and retained during operation.
- [adr/0001-mandate.md](adr/0001-mandate.md) — Accepted foundation mandate and constraints. Future decisions add records as implementation proceeds.
- [adr/0002-system-model-representation.md](adr/0002-system-model-representation.md) — Accepted system model representation: model kinds, units, circuit coordinates, cutoffs, biases, operator names, and composite basis.
- [adr/0003-runs-verification-and-dispatch.md](adr/0003-runs-verification-and-dispatch.md) — Accepted run records, verification checks on request, engine dispatch by declared needs, and drives and frames.
- [adr/0004-investigations-and-agent-sessions.md](adr/0004-investigations-and-agent-sessions.md) — Accepted investigation records, revision authority and user decisions, admission and stopping, checkpoints and reviews, the evaluation protocol, and the user's own agent runtime.
- [adr/0005-control-problems-and-solvers.md](adr/0005-control-problems-and-solvers.md) — Accepted control spec and generated ProblemSpec, shared problem subset, solver interface, direct Piccolo adapter, and control checks on request.
- [adr/0006-device-control-co-design.md](adr/0006-device-control-co-design.md) — Accepted device–control co-design: composition of existing operations, device and seed variation within one base, the named transition and Rabi-rate bounds, the uncertainty diagnostic, lineage, comparison guidance, budget policy, and first-version scope.
- [adr/0007-calibration-from-spectroscopy.md](adr/0007-calibration-from-spectroscopy.md) — Accepted calibration from spectroscopy: the uniform measurement format, extraction, assignment, the fit and its uncertainty report, holdout and the calibration checks, flux binding, device-update proposals, and first-version scope.
- [adr/0008-results-catalog.md](adr/0008-results-catalog.md) — Accepted results catalog: candidates as design objects, catalog records, staging and promotion with check scope, lineage, device reuse, control starting pulses, and first-version scope.
- [adr/0009-published-formats.md](adr/0009-published-formats.md) — Accepted record compatibility and deferred format publication: supported earlier record versions stay readable, implicit version 1 for historically unversioned records, the legacy pulse reading, and standalone schemas deferred until a consumer needs them.
