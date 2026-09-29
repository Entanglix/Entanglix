# Entanglix Documentation Index

- [roadmap.md](roadmap.md) — Authoritative implementation roadmap: scope, block order, done criteria, milestones.
- [checklist.md](checklist.md) — Build status and checkpoint checklist.
- [../HARNESS_CONTRACT.md](../HARNESS_CONTRACT.md) — Normative invariants: objects, engines, run directory, solver protocols, corrector, tools, prohibitions, audit.
- [adr/](adr/) — Architecture Decision Records. `0001-mandate.md` is Decision 0.
- `environment.md` — Pinned GPU, driver, CUDA, cuQuantum, and package versions (to be written with hardware access).
- Phase guides are written one phase ahead of the code, once there is code to link.

## Sharing rule

This repository is private until milestone M4. Before any file leaves the owner's machines:

- Remove `roadmap.md` Appendix A. It names a private solver.
- Any file matching `*-private.md` stays.
- Open-repo text says "private solver plugins", never a product name.
