# Entanglix

Entanglix is a quantum system research harness for superconducting circuits.

**Available in this checkout:** CLI help/version, validation and CPU construction for component circuits, coupled subsystems, and explicit operators, plus verified reference-size evolution, differentiable drives, and an engine-selection policy through the Python API. Research workflows will be available in later releases.

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
```

Invalid arguments return exit code `64` with a diagnostic on stderr.
The [system specification guide](docs/operations/system-spec.md) describes the accepted syntax, operators, units, and current modeling limits.

For GPU adapter use on a supported NVIDIA workstation, install the relevant extra with `uv sync --locked --exact --extra cuquantum` or `uv sync --locked --exact --extra differentiable-cuda12`. Run the engines in separate processes when using both stacks, with `XLA_PYTHON_CLIENT_PREALLOCATE=false` for JAX. The CPU dynamiqs path uses `--extra differentiable`; try `JAX_PLATFORMS=cpu uv run --locked --exact --extra differentiable python examples/differentiable_reference.py`. See [environment observations and test commands](docs/environment.md), the [Python evolution interface](docs/operations/system-spec.md#explicit-operators-and-evolution), and [selection limits](docs/operations/system-spec.md#engine-selection-policy). Runner dispatch and retained research runs are still in development.

## Changes and compatibility

Unreleased: add system models for component circuits and composites, validation with transformation, offset, oriented-loop, and operator-meaning diagnostics, CPU exports including factorized drive and dissipator operators, verified driven/dissipative evolution across QuTiP and optional cuDensityMat/dynamiqs adapters, JAX drive gradients, a versioned selection policy, and worked examples.

Version 0.1.0 established the installable package and CLI foundation.

Before 1.0, minor versions may break interfaces; changes are recorded in these release notes.

Access is limited to the owner and explicitly permitted collaborators. Collaborator access does not authorize onward distribution.
