# Entanglix

Entanglix is a quantum system research harness for superconducting circuits.

**Available in this checkout:** system validation and construction, CPU evolution with optional GPU adapters, carrier drives, explicit rotating-wave frames, and differentiable envelopes. Authored CPU jobs and CPU/CUDA `simulate` and serial `sweep` runs retain frozen inputs, code/environment provenance, metrics, plots, live inspection, timeouts, cancellation, recovery, and linked reruns. Built-in checks run only on request through `check`.

CPU [control solves](docs/operations/control.md) generate a shared ProblemSpec and use an open GRAPE-style baseline with JAX gradients and SciPy constraints, or the optional pinned Piccolo/Julia adapter, through entry-point solver discovery. The [transmon example](examples/transmon-control/README.md) supplies the first gate. Each candidate retains a neutral pulse, solver objective, convergence report and trace; requested control checks replay the pulse on QuTiP, test the declared fidelity target and pulse constraints, and report leakage. Control jobs can use exact approved investigation bases, inherited limits and stops, and linked `rerun-control` attempts.

Investigation records support drafts, human starts, proposed revisions and approvals, findings, retained history, scoped job admission, checkpoint assessments, named reviews, cancellation, and operator resumes. Optional limits cover job count, total execution time (`job_seconds`), and deadlines; GPU-seconds remain recorded usage. The [readout client](examples/readout/README.md#scripted-investigation-baseline) demonstrates scoped bisection, lab-frame confirmation, a precision handoff, and continuation across sessions.

An optional [stdio MCP server](docs/operations/mcp.md) exposes the same research operations, returns job admission under independent supervision, and records connection and tool-call audits. Its [workspace setup](docs/operations/mcp.md#research-workspace) supplies versioned research conduct and four focused skills without loading repository development instructions.

Shared execution guarantees are in [HARNESS_CONTRACT.md](HARNESS_CONTRACT.md); [research-agent conduct](docs/research-agent.md) guides user research sessions. [ADRs](docs/adr/) retain decisions and rationale for readers who need that history.

## Setup and CLI

Use Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then enter your authorized local clone. This checkout supports Linux and Python 3.13; `.python-version` pins 3.13.12. The current modeling and CPU reference path requires no GPU dependencies or device access.

```sh
uv sync --locked --no-dev
uv run --locked --exact --no-dev entanglix --help
uv run --locked --exact --no-dev entanglix --version
uv run --locked --exact --no-dev entanglix validate examples/two-level/system.toml
uv run --locked --exact --no-dev entanglix validate examples/junction-shunt/system.toml
uv run --locked --exact --no-dev entanglix validate examples/qubit-resonator/system.toml
uv run --locked --exact --no-dev python examples/cpu_reference.py
uv run --locked --exact --no-dev python examples/dynamics_reference.py
uv run --locked --exact entanglix run-script examples/recorded_decay.py --system examples/two-level/system.toml
uv run --locked --exact entanglix list-runs
uv run --locked --exact entanglix report-investigation PATH --output report.md
```

Invalid arguments return exit code `64` with a diagnostic on stderr.
The [system specification guide](docs/operations/system-spec.md) describes the accepted syntax, operators, units, and current modeling limits.
The [simulation and sweep reference](docs/operations/protocols.md) explains protocol inputs, metrics, GPU configuration, and replay. The [run reference](docs/operations/runs.md) explains authored jobs, result publication, requested checks, capabilities, inspection, limits, cancellation, and recovery.

For GPU adapter use on a supported NVIDIA workstation, install the relevant extra with `uv sync --locked --exact --extra cuquantum` or `uv sync --locked --exact --extra differentiable-cuda12`. Run the engines in separate processes when using both stacks, with `XLA_PYTHON_CLIENT_PREALLOCATE=false` for JAX. The CPU dynamiqs path uses `--extra differentiable`; try `JAX_PLATFORMS=cpu uv run --locked --exact --extra differentiable python examples/differentiable_reference.py`. See [environment observations and test commands](docs/environment.md), the [Python evolution interface](docs/operations/system-spec.md#explicit-operators-and-evolution), and [engine selection](docs/operations/system-spec.md#engine-selection-policy).

The [investigation reference](docs/operations/investigations.md) describes declarations, the operator decision command, findings, and the available record operations.

## Changes and compatibility

Unreleased: add exact-base control investigation admission, usage and limit accounting, stop/deadline enforcement and linked control reruns. Add requested control replay, target and constraint checks, independent QuTiP fidelity and ungraded leakage, with linked reports and frozen source binding. Add the direct Piccolo adapter with a pinned Julia project, native and neutral pulse artifacts, recorded runtime provenance and process-group cancellation. Add CPU control jobs, entry-point solver discovery with declared fallback, a constrained open GRAPE-style baseline, neutral pulse artifacts and solver termination/timing records. Add Python control-spec validation and shared ProblemSpec export with closed rotating-frame models, channel order, explicit units, and frozen input digests. Add circuit and composite modeling, explicit units and operator diagnostics, CPU/factorized exports, QuTiP/cuDensityMat/dynamiqs evolution, carrier drives, rotating-wave frames, and JAX drive gradients. Add authored CPU jobs and CPU/CUDA simulations with frozen inputs, atomic records, metrics, plots, process-group cancellation and timeouts, inherited GPU ownership, reconciliation, and linked replay. Add serial static/dynamic sweeps with frozen point plans, retained partial results, linked full-grid replay, metric plots, endpoint coherence and dressed dispersive shifts. Named checks now run only on request: integrity, local same-frame refined segments, diagnostics, spectral refinement, and declared sweep expectations. Reports carry per-check outcomes and coverage, with no run-level verdict or automatic reference solve. Refinement is optional and can be replaced for a check; dispatch and refined solves have no harness-imposed dimension cap or pre-launch memory estimate. Allocation failures and unexplained signal deaths retain distinct diagnostics. Add immutable investigation revisions and events, draft/read/list/finding commands, and a separate operator command for starts and revision approvals. Add investigation-linked simulation, sweep, script, check, and replay admission with atomic resource commitments, executor deadlines, and runner-derived usage and finding summaries. Add a scripted readout investigation baseline with normal and early-stop acceptance, checkpoint assessments, and recovery of completed work across sessions. Busy CUDA launches are refused before admission without creating attempts or using investigation allowance. Job commands can return admission with `--detach`; their supervisors continue across a lost caller. Launch recovery reports admission, and `read-run --wait` waits for an outcome without affecting supervision. Add the optional stdio MCP server with read-only annotations, bounded concurrent waits, independent job supervision and immutable connection/call audits. The optional extra changes the lockfile digest and recorded environment hashes; existing numerical package pins are unchanged. Ship versioned research conduct and four plain-text skills, with manual research-workspace setup and operator-declared restriction mode in connection audits. MCP investigation reads now separate status and attention from paged attempt/event tables; run reads retain detail on request. Add an on-request Markdown investigation report. Finding writes return acknowledgments, and tool replies carry one JSON text payload.

Version 0.1.0 established the installable package and CLI foundation.

Before 1.0, minor versions may break interfaces; changes are recorded in these release notes.

Access is limited to the owner and explicitly permitted collaborators. Collaborator access does not authorize onward distribution.
