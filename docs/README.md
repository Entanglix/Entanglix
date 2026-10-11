# Planning and documentation index

This index supports project planning and implementation. Operational entry points are the root README, harness contract, and research-agent conduct. Root auto-load files guide repository development.

## Documentation boundary

| Class | Examples | Lifecycle |
|---|---|---|
| Planning/development | Roadmap, checklist, phase guides, development instructions | Internal implementation guidance; plans are removable after building |
| Operational harness | Root README, harness contract, research-agent conduct, skills | Retained and maintained for production users and agents |
| Decision records | ADRs | Retained public decisions, rewritten in place when a decision changes, with earlier text in git; consulted when relevant, not automatically loaded into runtime context |

- Operational docs stand alone: they never link to or depend on planning/development docs. Planning docs may link to anything.
- Before a planning doc is removed, move lasting decisions into an ADR and operating guidance into an operational doc.
- Repository auto-load files are development entry points. Research sessions use a separate workspace with a copy of the shipped conduct in initial context, the same conduct as MCP instructions, and plain-text skills. Nothing is generated. Avoid `AGENTS.md` or `CLAUDE.md` filenames under `docs/`.

## Planning and development

- [roadmap.md](roadmap.md) — Research direction, scope, milestones, dependencies, quality criteria, and open choices.
- [checklist.md](checklist.md) — Completion status and links to acceptance evidence; progress narratives stay in conversation.
- [m1-guide.md](m1-guide.md) — Completed implementation guide for M1 (Blocks 1–2): model representation, engine integration, scientific fixtures, and acceptance evidence.
- [m2-guide.md](m2-guide.md) — Implementation guide for M2 (Blocks 3–5): run records, verification, drives and frames, and the first simulation and sweep.
- [m3-guide.md](m3-guide.md) — Implementation guide for M3 (Blocks 6a–6b): the first investigation and its evaluation, investigation records, admission, checkpoints and reviews, and agent integration.
- [m4-guide.md](m4-guide.md) — Planned implementation guide for M4 (Block 7): the control spec and generated ProblemSpec, the open baseline, the Piccolo adapter, control checks, and the first gate.
- [development.md](development.md) — Maintainer and coding-agent instructions. Root [AGENTS.md](../AGENTS.md) and [CLAUDE.md](../CLAUDE.md) are one-line pointers here; they are excluded from research-runtime guidance.
- Phase guides may be added as needed, including before code exists.

## Operational harness

- [../README.md](../README.md) — User entry point, available commands, and compatibility.
- [research-agent.md](research-agent.md) — Versioned conduct for research sessions: formulation, recovery, human decisions, checks on request, assessments, and stopping. Supplied in initial context and by the MCP server; it is separate from repository development guidance.
- [../skills/](../skills/) — Versioned `system-modeler`, `simulate`, `design-sweep`, and `gpu-budget` procedures, usable through runtime skill discovery or as plain text.
- [../HARNESS_CONTRACT.md](../HARNESS_CONTRACT.md) — Shared interfaces, permissions, checks, and execution guarantees.
- [operations/system-spec.md](operations/system-spec.md) — Implemented system syntax, units, operators, evolution, drive gradients, numerical checks, and engine selection. Other usage guides may be added under `operations/` as capabilities arrive.
- [operations/runs.md](operations/runs.md) — Authored CPU jobs, listing and inspection, frozen inputs, environment and code provenance, record ownership, and execution outcomes.
- [operations/protocols.md](operations/protocols.md) — Simulation and serial sweep specifications, metrics, CPU/CUDA dispatch, requested checks and coverage, and frozen-input replay.
- [operations/control.md](operations/control.md) — CPU control jobs, closed rotating-frame models, shared ProblemSpec export, solver selection, neutral pulses, and numerical evidence.
- [operations/investigations.md](operations/investigations.md) — Investigation declarations, immutable revisions and findings, human starts and approvals, scoped admission, optional limits, checkpoints, named reviews, stop/resume, usage, and record inspection.
- [operations/mcp.md](operations/mcp.md) — Optional stdio research tools, workspace setup and runtime settings, explicit server configuration, bounded waiting, launch recovery, and connection/call audit records.
- [environment.md](environment.md) — Observed GPU compatibility, tested package versions, limitations, and reproduction commands.

## Decision records

- [adr/](adr/) — Public decision records, kept current and retained during operation.
- [adr/0001-mandate.md](adr/0001-mandate.md) — Accepted foundation mandate and constraints. Future decisions add records as implementation proceeds.
- [adr/0002-system-model-representation.md](adr/0002-system-model-representation.md) — Accepted system model representation: model kinds, units, circuit coordinates, cutoffs, biases, operator names, and composite basis.
- [adr/0003-runs-verification-and-dispatch.md](adr/0003-runs-verification-and-dispatch.md) — Accepted run records, verification checks on request, engine dispatch by declared needs, and drives and frames.
- [adr/0004-investigations-and-agent-sessions.md](adr/0004-investigations-and-agent-sessions.md) — Accepted investigation records, revision authority and user decisions, admission and stopping, checkpoints and reviews, the evaluation protocol, and the user's own agent runtime.
- [adr/0005-control-problems-and-solvers.md](adr/0005-control-problems-and-solvers.md) — Accepted control spec and generated ProblemSpec, shared problem subset, solver interface, direct Piccolo adapter, and control checks on request.

## Sharing rule

This repository remains private until the verified shared ProblemSpec and open-solver artifacts required by roadmap §5 are available and the audience/license decision permits sharing. These artifacts normally arrive at M4; standalone control may provide them earlier. Agent integration is not a prerequisite. Before any file leaves the owner's machines:

- Remove `roadmap.md` Appendix A. It names a private solver.
- Any file matching `*-private.md` stays.
- Open-repo text says "private solver plugins", never a product name.
