# Entanglix

Entanglix is a quantum system research harness for superconducting circuits.

**Available in this checkout:** system validation and construction, CPU evolution with optional GPU adapters, carrier drives, explicit rotating-wave frames, and differentiable envelopes. Authored CPU jobs and CPU/CUDA `simulate` runs retain frozen inputs, code/environment provenance, metrics, plots, live inspection, timeouts, cancellation, recovery, and linked reruns. Built-in checks run only on request through `check`. Grid sweeps follow separately.

Shared execution guarantees are in [HARNESS_CONTRACT.md](HARNESS_CONTRACT.md); [research-agent guidance](docs/research-agent.md) is a draft pending runtime integration. [ADRs](docs/adr/) retain decisions and rationale for readers who need that history.

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
```

Invalid arguments return exit code `64` with a diagnostic on stderr.
The [system specification guide](docs/operations/system-spec.md) describes the accepted syntax, operators, units, and current modeling limits.
The [simulation reference](docs/operations/protocols.md) explains protocol inputs, metrics, GPU configuration, and replay. The [run reference](docs/operations/runs.md) explains authored jobs, result publication, requested checks, capabilities, inspection, limits, cancellation, and recovery.

For GPU adapter use on a supported NVIDIA workstation, install the relevant extra with `uv sync --locked --exact --extra cuquantum` or `uv sync --locked --exact --extra differentiable-cuda12`. Run the engines in separate processes when using both stacks, with `XLA_PYTHON_CLIENT_PREALLOCATE=false` for JAX. The CPU dynamiqs path uses `--extra differentiable`; try `JAX_PLATFORMS=cpu uv run --locked --exact --extra differentiable python examples/differentiable_reference.py`. See [environment observations and test commands](docs/environment.md), the [Python evolution interface](docs/operations/system-spec.md#explicit-operators-and-evolution), and [engine selection](docs/operations/system-spec.md#engine-selection-policy).

## Changes and compatibility

Unreleased: add circuit and composite modeling, explicit units and operator diagnostics, CPU/factorized exports, QuTiP/cuDensityMat/dynamiqs evolution, carrier drives, rotating-wave frames, and JAX drive gradients. Add authored CPU jobs and CPU/CUDA simulations with frozen inputs, atomic records, metrics, plots, process-group cancellation and timeouts, inherited GPU ownership, reconciliation, and linked replay. Named checks now run only on request: integrity, local same-frame refined segments, diagnostics, and spectral refinement. Reports carry per-check outcomes and coverage, with no run-level verdict or automatic reference solve. Refinement is optional and can be replaced for a check; dispatch and refined solves have no harness-imposed dimension cap or pre-launch memory estimate. Allocation failures and unexplained signal deaths retain distinct diagnostics.

Version 0.1.0 established the installable package and CLI foundation.

Before 1.0, minor versions may break interfaces; changes are recorded in these release notes.

Access is limited to the owner and explicitly permitted collaborators. Collaborator access does not authorize onward distribution.
