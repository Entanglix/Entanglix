# Entanglix Documentation Index

- [roadmap.md](roadmap.md) — Pinned research direction, scope, milestones, build dependencies, and quality criteria.
- [checklist.md](checklist.md) — Aligned build status and acceptance evidence for core and supporting milestones.
- [../HARNESS_CONTRACT.md](../HARNESS_CONTRACT.md) — Required interfaces, execution guarantees, verification, grants, resource limits, permissions, lineage, and conformance. Normative once Decision 0 is accepted, as each capability is introduced.
- [adr/](adr/) — Architecture Decision Records. [0001-mandate.md](adr/0001-mandate.md) is the Draft agenda for foundation decisions in Decision 0. Later decisions are scheduled in roadmap §7.
- `environment.md` — Pinned GPU, driver, CUDA, cuQuantum, and package versions (to be written with hardware access).
- `AGENTS.md` — Short coding-agent working instructions, navigation, and commands (planned with the repository foundation).
- `skills/` — Procedures for research workflows, added with the corresponding capabilities; execution guarantees remain in the contract.
- Phase guides are written one phase ahead of the code, once there is code to link.

## Sharing rule

This repository remains private until the verified shared ProblemSpec and open-solver artifacts required by roadmap §5 are available and the audience/license decision permits sharing. These artifacts normally arrive at M4; standalone control may provide them earlier. Agent integration is not a prerequisite. Before any file leaves the owner's machines:

- Remove `roadmap.md` Appendix A. It names a private solver.
- Any file matching `*-private.md` stays.
- Open-repo text says "private solver plugins", never a product name.
