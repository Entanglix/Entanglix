# Entanglix

Entanglix is a quantum system research harness for superconducting circuits.

**Available in version 0.1.0:** CLI help and version reporting. Research workflows will be available in later releases.

Shared execution guarantees are in [HARNESS_CONTRACT.md](HARNESS_CONTRACT.md); [research-agent guidance](docs/research-agent.md) is a draft pending runtime integration. [ADRs](docs/adr/) retain decisions and rationale for readers who need that history.

## Setup and CLI

Use Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then enter your authorized local clone. This release supports Linux and Python 3.13; `.python-version` pins 3.13.12. It requires no GPU dependencies or device access.

```sh
uv sync --locked --no-dev
uv run --locked --no-dev entanglix --help
uv run --locked --no-dev entanglix --version
```

Invalid arguments return exit code `64` with a diagnostic on stderr.

## Changes and compatibility

Version 0.1.0 establishes the installable package and CLI foundation.

Before 1.0, minor versions may break interfaces; changes are recorded in these release notes.

Access is limited to the owner and explicitly permitted collaborators. Collaborator access does not authorize onward distribution.
