# Entanglix

Entanglix is a quantum system research harness for superconducting circuits.

**Available in this checkout:** system validation and construction, CPU evolution with optional GPU adapters, carrier drives, explicit rotating-wave frames, and differentiable envelopes. Authored CPU jobs and CPU/CUDA `simulate` and serial `sweep` runs retain frozen inputs, code/environment provenance, metrics, plots, live inspection, timeouts, cancellation, recovery, and linked reruns. Built-in checks run only on request through `check`. Users can add their own engines and solvers as [plugin distributions](docs/operations/runs.md#plugins-in-the-runner-environment) in the runner environment; every run records the installed distributions.

CPU [control solves](docs/operations/control.md) generate a shared ProblemSpec and use an open GRAPE-style baseline with JAX gradients and SciPy constraints, or the optional pinned Piccolo/Julia adapter, through entry-point solver discovery. The [transmon example](examples/transmon-control/README.md) supplies the first gate. Each candidate retains a neutral pulse, solver objective, convergence report and trace; requested control checks replay the pulse on QuTiP, test the declared fidelity target and pulse constraints, and report leakage. Control jobs can use approved investigation bases, with device-parameter and seed variation, inherited limits and stops, and linked reruns.

Spectroscopy measurements for calibration use one [dataset format](docs/operations/calibration.md), with validation, raw-file digest checks, and holdout splits through `validate-dataset` and `split-dataset`. `calibrate` fits system parameters to maps or points by bounded weighted least squares on full model construction, with supplied labels or automatic one-to-one assignment from a declared candidate family, and reports the outcome, uncertainty along the determined directions, and the undetermined combinations. Its `holdout`, `identifiability` and `refinement` checks run on request through `check`, with holdout data supplied with the request. `propose-device-update` writes a proposal from a solution next to the run, and the operator command `apply-device-update` writes it into the system file only while every model input is unchanged. Calibration runs join investigations by exact base, rerun with `rerun`, and are available to agents through the MCP tools, which stage proposals but cannot apply them. The [transmon calibration example](examples/transmon-calibration/README.md) runs a synthetic round trip from spectroscopy maps to an applied proposal.

`register-result` stages a design from a completed run in the [results catalog](docs/operations/catalog.md): a device with every file its system references, a pulse with its channel names, or both, copied into `_catalog/staging/` with the digests of the evidence they came from. The operator command `promote` publishes an immutable entry with its source-bound check scope, promoter and lineage. `list-catalog` and `read-entry` read staged candidates and promoted entries. Simulation, sweep, control and calibration jobs can reuse a catalogued device through `system_entry`; investigation bases can name the same entry, and reruns use the frozen files. A baseline control solve can start from a catalogued pulse, an earlier control run's pulse or a pulse file through `starting_pulse`; an incompatible pulse or a solver without starting-pulse support is refused before an attempt exists. Research tools can stage and read catalog designs; promotion remains an operator action. The [catalog reuse example](examples/catalog/README.md) repeats a spectrum sweep and a pulse solve on a promoted co-design candidate.

Investigation records support drafts, human starts, proposed revisions and approvals, findings, retained history, scoped job admission, checkpoint assessments, named reviews, cancellation, and operator resumes. Optional limits cover job count, total execution time (`job_seconds`), and deadlines; GPU-seconds remain recorded usage. The [readout client](examples/readout/README.md#scripted-investigation-baseline) demonstrates scoped bisection, lab-frame confirmation, a precision handoff, and continuation across sessions.

An optional [stdio MCP server](docs/operations/mcp.md) exposes the same research operations, returns job admission under independent supervision, and records connection and tool-call audits. Its [workspace setup](docs/operations/mcp.md#research-workspace) supplies versioned research conduct and six focused skills without loading repository development instructions. Human decisions remain operator commands, which a user may let the agent run after approving each one in the runtime's prompt.

Shared execution guarantees are in [HARNESS_CONTRACT.md](HARNESS_CONTRACT.md); [research-agent conduct](docs/research-agent.md) guides user research sessions. [ADRs](docs/adr/) retain decisions and rationale for readers who need that history.

## Setup and CLI

Use Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository and enter it. This checkout supports Linux and Python 3.13; `.python-version` pins 3.13.12. The current modeling and CPU reference path requires no GPU dependencies or device access.

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

0.1.0, the first public release: the modeling, simulation, sweep, control, calibration, catalog and investigation workflows above; user-installed engine and solver plugins; the optional MCP server with versioned research conduct and skills; and operator commands for human decisions. Records written by earlier development builds remain readable under the [compatibility rule](docs/operations/runs.md#record-fields-and-ownership).

Before 1.0, minor versions may break interfaces; changes are recorded in these release notes.

## License

Copyright 2026 Lambert Lin. Entanglix is licensed under the [Apache License 2.0](LICENSE).
