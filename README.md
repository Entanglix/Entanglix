# Entanglix

Entanglix is a research harness for superconducting quantum circuits, driven by your own AI agent. You describe a device in a small TOML file. Entanglix builds its model, simulates it on CPU or an NVIDIA GPU, solves control pulses, and fits parameters to spectroscopy. It keeps every run with its frozen inputs, code and environment, so any result can be inspected, checked and rerun. Your agent (Claude Code, Codex or another MCP runtime) works through Entanglix's MCP server and research skills.

## What you can do

- Model transmons, resonators and coupled circuits, and sweep their spectra over flux or a design parameter.
- Simulate driven, lossy dynamics, such as [dispersive readout](examples/readout/README.md).
- Solve a gate pulse with the open GRAPE solver or Piccolo, and check it by independent replay ([first gate](examples/transmon-control/README.md)).
- Calibrate a device model against measured spectroscopy, then apply the fit to the system file after review ([calibration](examples/transmon-calibration/README.md)).
- Search device parameters and pulses together within agreed limits ([co-design](examples/transmon-codesign/README.md)), and reuse promoted designs later ([catalog](examples/catalog/README.md)).

## Who does what

You decide what to study and approve the terms. Your agent writes the system and protocol files, launches runs, reads the results and records findings. Three commands stay yours:

- `investigation-decision` starts or approves an investigation;
- `apply-device-update` writes a calibration into a system file;
- `promote` publishes a catalog design.

Project setup configures the agent's runtime to refuse those commands, or, with `--agent-command-privilege on-request`, to ask you before each one. Setup does not show that a runtime enforces this. Refusal is claimed only for configurations a retained test covers ([MCP reference](docs/operations/mcp.md#codex-example)); in other runtimes the rule rests on the agent's conduct. Your inputs, runs and records stay in your project directory.

## Requirements

Linux on x86_64 or aarch64, with git, curl, tar and sha256sum. No root and no Python of your own: the installer brings its own uv and Python. GPU engines need an NVIDIA driver.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/Entanglix/Entanglix/v0.1.0/install.sh | bash
```

This installs Entanglix 0.1.0 in `~/.local/share/entanglix/0.1.0`, with CPU engines and the GRAPE solver, and puts the `entanglix` command in `~/.local/bin`. It edits no shell profile; if that directory is not on your `PATH`, it says so and shows the line to add. Two components are added on request:

- `entanglix components add gpu`: the cuDensityMat and dynamiqs GPU engines, about 7 GB, mostly NVIDIA libraries;
- `entanglix components add piccolo`: the Piccolo solver with its own Julia, about 3.5 GB.

`entanglix components` shows what is installed. A newer version installs beside the old one and shares its package cache; `entanglix remove VERSION` deletes a version nothing is using.

## Start a project

```sh
entanglix setup-project <project-dir> --runtime claude
```

`<project-dir>` is the path of your project's directory, new or existing; setup creates it if needed. `--runtime` takes `claude`, `codex`, `other`, or a list such as `claude,codex`. Setup copies the research conduct, skills and references into the directory and writes the runtime's MCP server entry and command rules. If a file it would write already exists, it writes nothing. For `other`, it prints the entry to add by hand. With the GPU component and one GPU, the project computes on that GPU; choose with `--device cpu` or `--device UUID`.

Then start a fresh session there, as setup prints:

```sh
cd <project-dir> && CUDA_VISIBLE_DEVICES="" claude
```

A first prompt might be: "Model a flux-tunable transmon with EJ = 20 GHz and EC = 0.25 GHz, sweep its 0–1 transition over half a flux quantum, and tell me how far it tunes."

A project stays on the version that set it up. After installing a newer one, move it with `entanglix setup-project <project-dir> --refresh`, which refuses if you edited a file setup wrote.

To have your agent do the setup, paste this into it:

> Install Entanglix with `curl -fsSL https://raw.githubusercontent.com/Entanglix/Entanglix/v0.1.0/install.sh | bash`, run `~/.local/bin/entanglix setup-project <project-dir> --runtime <runtime>`, then tell me the launch instruction it prints.

Then start the fresh session yourself.

## Reference

The [documentation index](docs/README.md) lists the operational references: system files, simulations and sweeps, runs, control, calibration, the catalog, investigations, and the [MCP server](docs/operations/mcp.md), which also covers working from a source checkout. Shared execution guarantees are in [HARNESS_CONTRACT.md](HARNESS_CONTRACT.md), and [ADRs](docs/adr/) keep the decisions behind them.

## Changes and compatibility

0.1.0, the first public release: the modeling, simulation, sweep, control, calibration, catalog and investigation workflows; one-command installation with GPU and Piccolo components and project setup for agent runtimes; user-installed engine and solver plugins; the MCP server with versioned research conduct and skills; and operator commands for human decisions. Records written by earlier development builds remain readable under the [compatibility rule](docs/operations/runs.md#record-fields-and-ownership).

Before 1.0, minor versions may break interfaces; changes are recorded in these release notes.

## License

Copyright 2026 Lambert Lin. Entanglix is licensed under the [Apache License 2.0](LICENSE).
