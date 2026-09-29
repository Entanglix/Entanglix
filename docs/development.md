# Development instructions

For maintainers and coding agents working on Entanglix. Root `AGENTS.md` and `CLAUDE.md` point here; research-agent guidance lives separately.

## Document ownership

- [roadmap.md](roadmap.md) owns scope, dependencies, quality criteria, and open choices; [checklist.md](checklist.md) owns completion and acceptance evidence.
- [HARNESS_CONTRACT.md](../HARNESS_CONTRACT.md) owns shared interfaces and execution guarantees; [ADRs](adr/) retain decisions and rationale. Draft recommendations remain proposals.
- [README.md](../README.md) owns supported usage and release notes; [research-agent.md](research-agent.md) is a draft operational reference pending runtime integration. Follow the [documentation boundary and sharing rule](README.md).

## Checks

Run `uv sync --locked`, `uv run --locked pytest`, `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, and `git diff --check` as appropriate to the change.

## Working rules

- Never name or directly import the private solver outside the existing roadmap Appendix A; preserve that appendix during unrelated edits. Private plugins attach through entry points.
- Operational docs never link to planning/development docs. Keep contributor instructions out of research-agent guidance; [Block 6b](roadmap.md#block-6b--agent-integration) will decide how that guidance reaches user agents.
- Keep core imports and CPU tests usable without GPU or private packages. Preserve [ADR-0001's numerical policy](adr/0001-mandate.md#33-units-time-grid-and-truncation-defaults).
- Lean by default: reuse existing facilities and implement capabilities when needed.
- ADRs hold design decisions only, stated as final. Document organization belongs in the docs index, and edit history in git. A substantive change to an accepted decision gets a new ADR that supersedes the old one.
- Record acceptance evidence in the checklist; distinguish implemented, validated, and planned behavior.
- Commit or push only with the user's go-ahead. A request to "save" means stash.

## Reviews

- Review the requested target against its purpose. Report only changes it should carry, marked blocking or not. List anything outside it separately and briefly, labeled when an earlier review missed it; do not apply or record it.
- Follow-up reviews check the applied changes and their effects, not the whole target, unless asked.
- Planning docs hold what the next step needs to start. Details settled during implementation belong in code, tests, and operational docs.
