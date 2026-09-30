# Planning and documentation index

This index supports project planning and implementation. Operational entry points are the root README and harness contract; research-agent guidance is a draft until runtime integration. Root auto-load files guide repository development.

## Documentation boundary

| Class | Examples | Lifecycle |
|---|---|---|
| Planning/development | Roadmap, checklist, phase guides, development instructions | Internal implementation guidance; plans are removable after building |
| Operational harness | Root README, harness contract, research-agent guidance, skills | Retained and maintained for production users and agents; draft guidance awaits integration |
| Decision records | ADRs | Retained public history; consulted when relevant, not automatically loaded into runtime context |

- Operational docs stand alone: they never link to or depend on planning/development docs. Planning docs may link to anything.
- Before a planning doc is removed, move lasting decisions into an ADR and operating guidance into an operational doc.
- Repository auto-load files are development entry points. Research work may use a separate workspace; MCP instructions, skills, or an Entanglix-generated workspace instruction file will deliver research guidance once that mechanism is selected and tested. Avoid `AGENTS.md` or `CLAUDE.md` filenames under `docs/`.

## Planning and development

- [roadmap.md](roadmap.md) — Research direction, scope, milestones, dependencies, quality criteria, and open choices.
- [checklist.md](checklist.md) — Completion status and links to acceptance evidence; progress narratives stay in conversation.
- [m1-guide.md](m1-guide.md) — Completed implementation guide for M1 (Blocks 1–2): model representation, engine integration, scientific fixtures, and acceptance evidence.
- [m2-guide.md](m2-guide.md) — Draft implementation guide for M2 (Blocks 3–5): run records, verification, drives and frames, and the first simulation and sweep.
- [development.md](development.md) — Maintainer and coding-agent instructions. Root [AGENTS.md](../AGENTS.md) and [CLAUDE.md](../CLAUDE.md) are one-line pointers here; they are excluded from research-runtime guidance.
- Phase guides may be added as needed, including before code exists.

## Operational harness

- [../README.md](../README.md) — User entry point, available commands, and compatibility.
- [research-agent.md](research-agent.md) — Draft research conduct for user agents; instruction delivery is decided before Block 6b. It avoids an auto-load filename and is not repository development guidance.
- [../HARNESS_CONTRACT.md](../HARNESS_CONTRACT.md) — Shared interfaces, permissions, evidence requirements, and execution guarantees.
- [operations/system-spec.md](operations/system-spec.md) — Implemented system syntax, units, operators, evolution, drive gradients, numerical checks, and engine-selection limits. Other usage guides may be added under `operations/` as capabilities arrive. Runtime skills will be introduced in the root `skills/` directory with agent integration.
- [operations/runs.md](operations/runs.md) — Run-record API, frozen inputs, environment and code provenance, record ownership, and execution outcome conventions; process execution follows separately.
- [environment.md](environment.md) — Observed GPU compatibility, tested package versions, limitations, and reproduction commands.

## Decision records

- [adr/](adr/) — Public decision history, retained during operation.
- [adr/0001-mandate.md](adr/0001-mandate.md) — Accepted foundation mandate and constraints. Future decisions add records as implementation proceeds.
- [adr/0002-system-model-representation.md](adr/0002-system-model-representation.md) — Accepted system model representation: model kinds, units, circuit coordinates, cutoffs, biases, operator names, and composite basis.
- [adr/0003-runs-verification-and-dispatch.md](adr/0003-runs-verification-and-dispatch.md) — Accepted run records, scoped verification, engine dispatch by declared needs, and drives and frames.

## Sharing rule

This repository remains private until the verified shared ProblemSpec and open-solver artifacts required by roadmap §5 are available and the audience/license decision permits sharing. These artifacts normally arrive at M4; standalone control may provide them earlier. Agent integration is not a prerequisite. Before any file leaves the owner's machines:

- Remove `roadmap.md` Appendix A. It names a private solver.
- Any file matching `*-private.md` stays.
- Open-repo text says "private solver plugins", never a product name.
