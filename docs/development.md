# Development instructions

For maintainers and coding agents working on Entanglix. Root `AGENTS.md` and `CLAUDE.md` point here; research-agent guidance lives separately.

## Document ownership

- [roadmap.md](roadmap.md) owns scope, dependencies, quality criteria, and open choices; [checklist.md](checklist.md) owns completion status and links to acceptance evidence.
- [HARNESS_CONTRACT.md](../HARNESS_CONTRACT.md) owns shared interfaces and execution guarantees; [ADRs](adr/) retain decisions and rationale. Draft recommendations remain proposals.
- [README.md](../README.md) owns supported usage and release notes; [research-agent.md](research-agent.md) owns operational research conduct, supplied separately from development instructions. The [documentation index](README.md) lists the operational docs and ADRs.

## Documentation boundary

| Class | Examples | Lifecycle |
|---|---|---|
| Planning/development | Roadmap, checklist, phase guides, development instructions | Internal implementation guidance; plans are removable after building; omitted from releases |
| Operational harness | Root README, harness contract, research-agent conduct, skills | Retained and maintained for production users and agents |
| Decision records | ADRs | Retained public decisions, rewritten in place when a decision changes, with earlier text in git; consulted when relevant, not automatically loaded into runtime context |

- Operational docs and ADRs stand alone: they never link to or depend on planning/development docs. Planning docs may link to anything.
- Before a planning doc is removed, move lasting decisions into an ADR and operating guidance into an operational doc.
- Repository auto-load files are development entry points. Research sessions use a separate workspace with a copy of the shipped conduct in initial context, the same conduct as MCP instructions, and plain-text skills. Nothing is generated. Avoid `AGENTS.md` or `CLAUDE.md` filenames under `docs/`.

### Planning docs

- [roadmap.md](roadmap.md) — Research direction, scope, milestones, dependencies, quality criteria, and open choices.
- [checklist.md](checklist.md) — Completion status and links to acceptance evidence; progress narratives stay in conversation.
- [m1-guide.md](m1-guide.md) — Completed implementation guide for M1 (Blocks 1–2): model representation, engine integration, scientific fixtures, and acceptance evidence.
- [m2-guide.md](m2-guide.md) — Implementation guide for M2 (Blocks 3–5): run records, verification, drives and frames, and the first simulation and sweep.
- [m3-guide.md](m3-guide.md) — Implementation guide for M3 (Blocks 6a–6b): the first investigation and its evaluation, investigation records, admission, checkpoints and reviews, and agent integration.
- [m4-guide.md](m4-guide.md) — Completed implementation guide for M4 (Block 7): the control spec and generated ProblemSpec, the open baseline, the Piccolo adapter, control checks, and the first gate.
- [m5-guide.md](m5-guide.md) — Implementation guide for M5 (Block 8): control parameters and seed variation, named transitions and Rabi-rate bounds, lineage through MCP, the uncertainty diagnostic, comparison guidance, and the co-design example.
- [s2-guide.md](s2-guide.md) — Implementation guide for S2 (Block 10): the measurement format, flux parameters and axis binding, extraction, the calibrate protocol, the calibration checks, device-update proposals, investigations and tools, and the calibration example.
- [s3-guide.md](s3-guide.md) — Implementation guide for S3 (Block 11): catalog records and staging, promotion with its check scope, device reuse in jobs and investigation bases, control starting pulses, tools, and the catalog example.
- Phase guides may be added as needed, including before code exists.

## Checks

Run `uv sync --locked --exact`, `uv run --locked --exact pytest`, `uv run --locked --exact ruff check .`, `uv run --locked --exact ruff format --check .`, and `git diff --check` as appropriate to the change. Add an optional extra only for checks that need it. Checks run locally; the repository has no CI.

## Working rules

- Never name a private solver or private project in repository text, and never import a private solver directly; private plugins attach through entry points ([roadmap Appendix A](roadmap.md#appendix-a--private-solver-plugins)).
- Keep contributor instructions out of research-agent guidance; research sessions receive their conduct and skills through a separate workspace and the MCP server.
- Keep core imports and CPU tests usable without GPU or private packages. Preserve [ADR-0001's numerical policy](adr/0001-mandate.md#33-units-time-grid-and-truncation-defaults).
- Lean by default: reuse existing facilities and implement capabilities when needed.
- ADRs hold design decisions only, stated as final. Document organization belongs in the docs index, and edit history in git. A substantive change to an accepted decision rewrites that ADR in place with a revised date; earlier text stays in git.
- Link acceptance artifacts from the checklist; distinguish implemented, validated, and planned behavior. Keep progress narratives in conversation. Retain implementation details and scientific evidence in the appropriate operational docs, code, tests, or artifacts.
- Commit or push only with the user's go-ahead. A request to "save" means stash.

## Branches and releases

- All branches and release tags live in the public repository.
- `development` holds the full history and every document. Work and pull requests go here.
- `main` holds the release history without the development history; each release adds one commit. It omits `docs/roadmap.md`, `docs/checklist.md`, `docs/development.md`, and `docs/*-guide.md`, and its own `AGENTS.md` and `CLAUDE.md` point to setup and the research workspace.
- `checkpoint` marks the `development` commit of the latest release.
- Never merge `development` into `main`. Before a release, `pyproject.toml` and the README release notes name the version, and the worktree is clean. Release `X.Y.Z` from `development` commit `C`:

  ```
  git switch main                           # first release: git switch --orphan main
  git rm -rq --ignore-unmatch . && git checkout C -- .
  git rm -rfq --ignore-unmatch docs/roadmap.md docs/checklist.md docs/development.md 'docs/*-guide.md'
  git checkout HEAD -- AGENTS.md CLAUDE.md  # first release: write them instead
  git add AGENTS.md CLAUDE.md
  git commit -m "Entanglix X.Y.Z (development C)"
  git tag -a vX.Y.Z -m "Entanglix X.Y.Z"
  git branch -f checkpoint C
  ```

- The owner reviews and pushes `main`, `checkpoint`, and the tag.

## Reviews

- Review the requested target against its purpose. Report only changes it should carry, marked blocking or not. List anything outside it separately and briefly, labeled when an earlier review missed it; do not apply or record it.
- Follow-up reviews check the applied changes and their effects, not the whole target, unless asked.
- Planning docs hold what the next step needs to start. Details settled during implementation belong in code, tests, and operational docs.
