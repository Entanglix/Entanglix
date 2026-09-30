# Entanglix

Entanglix is a quantum system research harness for superconducting circuits.

**Available in this checkout:** CLI help/version, validation and CPU construction for component circuits, coupled subsystems, and explicit operators, plus reference-size evolution through the Python API using QuTiP or optional cuDensityMat/dynamiqs adapters. Research workflows will be available in later releases.

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
```

Invalid arguments return exit code `64` with a diagnostic on stderr.
The [system specification guide](docs/operations/system-spec.md) describes the accepted syntax, operators, units, and current modeling limits.

For GPU adapter use on a supported NVIDIA workstation, install the relevant extra with `uv sync --locked --exact --extra cuquantum` or `uv sync --locked --exact --extra differentiable-cuda12`. Run the engines in separate processes when using both stacks, with `XLA_PYTHON_CLIENT_PREALLOCATE=false` for JAX. The CPU dynamiqs path uses `--extra differentiable`. See [environment observations and test commands](docs/environment.md) and the [Python evolution interface](docs/operations/system-spec.md#explicit-operators-and-evolution). These adapters are reference-size integrations; automated engine selection and gradient-backed results are still in development.

## Changes and compatibility

Unreleased: add system models for component circuits and composites, validation with transformation, offset, oriented-loop, and operator-meaning diagnostics, CPU exports including factorized drive and dissipator operators, driven and dissipative QuTiP reference evolution, optional cuDensityMat/dynamiqs evolution adapters, and worked examples.

Version 0.1.0 established the installable package and CLI foundation.

Before 1.0, minor versions may break interfaces; changes are recorded in these release notes.

Access is limited to the owner and explicitly permitted collaborators. Collaborator access does not authorize onward distribution.
