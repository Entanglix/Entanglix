# Planning and documentation index

This index supports project planning and implementation. Operational entry points are the root README, AGENTS.md, and harness contract.

## Documentation boundary

| Class | Examples | Lifecycle |
|---|---|---|
| Planning | Roadmap, checklist, phase guides | Written freely to guide high-quality implementation; removable afterward |
| Operational harness | Root README, `AGENTS.md`, harness contract, skills | Retained and maintained for production users and agents |
| Decision records | ADRs | Retained public history; consulted when relevant, not automatically loaded into runtime context |

- Operational docs stand alone: they never link to or depend on planning docs. Planning docs may link to anything.
- Before a planning doc is removed, move lasting decisions into an ADR and operating guidance into an operational doc.

## Planning

- [roadmap.md](roadmap.md) — Research direction, scope, milestones, dependencies, quality criteria, and open choices.
- [checklist.md](checklist.md) — Implementation status and acceptance evidence.
- Phase guides may be added as needed, including before code exists.

## Operational harness

- [../README.md](../README.md) — User entry point, available commands, and compatibility.
- [../AGENTS.md](../AGENTS.md) — Research-agent entry instructions and navigation; [../CLAUDE.md](../CLAUDE.md) points to it.
- [../HARNESS_CONTRACT.md](../HARNESS_CONTRACT.md) — Shared interfaces, permissions, evidence requirements, and execution guarantees.
- [../schemas/environment.schema.json](../schemas/environment.schema.json) — Machine-readable environment format reference.
- Runtime skills belong in the root `skills/` directory; it currently contains no skills. Additional usage guides may live under `docs/operations/` when needed.
- `environment.md` will document validated environment pins when GPU support is available; it should serve as a retained operational reference, independently of validation plans.

## Decision records

- [adr/](adr/) — Public decision history, retained during operation.
- [adr/0001-mandate.md](adr/0001-mandate.md) — Accepted foundation mandate and constraints. Future decisions add records as implementation proceeds.

## Sharing rule

This repository remains private until the verified shared ProblemSpec and open-solver artifacts required by roadmap §5 are available and the audience/license decision permits sharing. These artifacts normally arrive at M4; standalone control may provide them earlier. Agent integration is not a prerequisite. Before any file leaves the owner's machines:

- Remove `roadmap.md` Appendix A. It names a private solver.
- Any file matching `*-private.md` stays.
- Open-repo text says "private solver plugins", never a product name.
