# Entanglix

Entanglix is a research harness for superconducting quantum circuits, driven by your own AI agent. You describe a device in a small TOML file. Entanglix builds its model, simulates it on CPU or an NVIDIA GPU, solves control pulses, and fits parameters to spectroscopy. It keeps every run with its frozen inputs, code and environment, so any result can be inspected, checked and rerun. Your agent (Claude Code, Codex, Cursor or another MCP runtime) works through Entanglix's MCP server and research skills.

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

Linux on x86_64 or aarch64, with git, curl, tar, sha256sum and flock, or macOS on Apple Silicon, with git. No root and no Python of your own: the installer brings its own uv and Python. GPU engines need Linux and an NVIDIA driver.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/Entanglix/Entanglix/v0.1.0/install.sh | bash
```

This installs Entanglix 0.1.0 in `~/.local/share/entanglix/0.1.0`, with CPU engines and the GRAPE solver, and puts the `entanglix` command in `~/.local/bin`. It edits no shell profile; if that directory is not on your `PATH`, it says so and shows the line to add. Two components are added on request:

- `entanglix components add gpu`: the cuDensityMat and dynamiqs GPU engines and CUDA support for GRAPE, about 7 GB installed, mostly NVIDIA libraries;
- `entanglix components add piccolo`: the Piccolo solver with its own Julia, about 3.5 GB installed; setup can take several minutes.

`entanglix components` shows what is installed. Your own engines and solvers attach as [plugins](docs/operations/runs.md#plugins-in-the-runner-environment), added with `entanglix plugins add PATH`. A newer version installs beside the old one and shares its package cache; `entanglix remove VERSION` deletes a version nothing is using, and removing the last one also deletes the cache and the `entanglix` command.

These commands show readable summaries in a terminal and JSON when their output is captured or piped, including through an agent's shell tool.

## Start a project

```sh
entanglix setup-project <project-dir> --runtime claude
```

`<project-dir>` is the path of your project's directory, new or existing; setup creates it if needed. `--runtime` takes `claude`, `codex`, `cursor`, `other`, or a list such as `claude,codex`. Setup copies the research conduct, skills and references into the directory and writes the runtime's MCP server entry and command rules. If a file it would write already exists, it writes nothing. For `other`, it prints the entry for you or your agent to add to the runtime's configuration. With the GPU component and one GPU, the project computes on that GPU; choose with `--device cpu` or `--device UUID`.

Then start a fresh session there, as setup prints:

```sh
cd <project-dir> && claude
```

A first prompt might be: "Model a flux-tunable transmon with EJ = 20 GHz and EC = 0.25 GHz, sweep its 0–1 transition over half a flux quantum, and tell me how far it tunes."

Accept the runtime's project trust prompt so it loads the settings. Setup hides CUDA from Claude Code's and Codex's session commands; jobs still use the project's configured device. If Claude Code asks whether to use the Entanglix server, choose “Use this MCP server.” Setup's settings pre-approve Entanglix's tools in Claude Code and Codex, so the runtime need not ask before each call. Cursor keeps tool approval and the session's environment in your own settings, so setup writes neither there and prints what to do. By its conduct your agent launches a job only when you have requested or agreed to it, and the three commands above stay yours. Other runtimes use the launch instructions setup prints.

A project stays on the version that set it up. After installing a newer one, move it with `entanglix setup-project <project-dir> --refresh`, which refuses if you edited a file setup wrote. Start a fresh session after refreshing.

To have your agent do the setup, paste this into it:

> Install Entanglix with `curl -fsSL https://raw.githubusercontent.com/Entanglix/Entanglix/v0.1.0/install.sh | bash`, run `~/.local/bin/entanglix setup-project <project-dir> --runtime <runtime>`, then tell me the launch instruction it prints.

Then start the fresh session yourself.

## Reference

The [documentation index](docs/README.md) lists the operational references: system files, simulations and sweeps, runs, control, calibration, the catalog, investigations, and the [MCP server](docs/operations/mcp.md), which also covers working from a source checkout. Shared execution guarantees are in [HARNESS_CONTRACT.md](HARNESS_CONTRACT.md), and [ADRs](docs/adr/) keep the decisions behind them.

## Changes and compatibility

Unreleased: quieter installation and component setup, with readable component and plugin summaries in a terminal and JSON when output is captured or piped. Project setup hides CUDA from Claude Code's and Codex's session commands and prints simpler startup instructions, including Claude Code's MCP server choice. Research guidance clarifies authorization for jobs and permits CPU modeling in the session while identifying results without a frozen runner record. It also asks for lean code in the session and in authored job scripts. The references no longer say that grid sweeps are unavailable, and they name the module of `publish_toml`. The spectral refinement check compares excitation energies only, no longer the dispersive-shift value as well. Project setup pre-approves Entanglix's tools in Claude Code, as it already did in Codex. It adds a `cursor` runtime, and for `other` it now links the skills at `.agents/skills`. Installation supports macOS on Apple Silicon, for CPU engines and both control solvers. On macOS, installation and the example control jobs are checked by hand; the test suite runs on Linux. `entanglix plugins add` also reinstalls the dependencies a plugin declares as local folders, and the control reference and the `transmon-control` example show how to attach your own solver. Control specs can request CUDA for GRAPE and supporting solver plugins. Multi-node circuits are diagonalized hierarchically, in declared groups of coordinates; they need system `format_version = 2`, and a version 1 multi-node file is refused with the partition to adopt. Refinement checks also step each group's levels. Multi-node circuits may have extended coordinates, alone or mixed with periodic ones, and export the phase of each node whose phase involves only extended coordinates. Composites are built through scqubits' `HilbertSpace`. Circuit and constructor operators change at rounding level, so a fresh check of an earlier control candidate can report that its frozen ProblemSpec disagrees; check such candidates with the Entanglix version that made them. A loaded composite or hierarchical circuit provides its scqubits `HilbertSpace` through `hilbert_space()`, for dressed-state labels.

0.1.0, the first public release: the modeling, simulation, sweep, control, calibration, catalog and investigation workflows; one-command installation with GPU and Piccolo components and project setup for agent runtimes; user-installed engine and solver plugins; the MCP server with versioned research conduct and skills; and operator commands for human decisions. Records written by earlier development builds remain readable under the [compatibility rule](docs/operations/runs.md#record-fields-and-ownership).

Before 1.0, minor versions may break interfaces; changes are recorded in these release notes.

## License

Copyright 2026 Lambert Lin. Entanglix is licensed under the [Apache License 2.0](LICENSE).
