# Transmon calibration example

This example calibrates a flux-tunable transmon's junction energies and shunt capacitance from two-tone spectroscopy, as a standalone calibration with no control or investigation. It runs through the CLI or a scripted MCP client and ends with a device-update proposal, which the operator applies to a copy of the system file. Known parameters generate the data, so the fitted values can be compared with them. It is one noise realisation: it reports each parameter's error in units of its reported uncertainty and asserts no interval. Coverage over many realisations is in the [calibration probe](../../tests/probes/s2_calibration_results.json).

## Inputs

- [system.toml](system.toml): a two-junction circuit with a shunt capacitor.
  - The flux parameter `phi` sits on J2's loop.
  - `EJ1`, `EJ2` and `Csh` carry uniform priors (2–25 GHz, 2–25 GHz and 30–120 fF) and design values of 12.4 GHz, 7.7 GHz and 72 fF.
  - The junction charging energy `ECJ` is fixed at 4 GHz.
  - Charge cutoff 30, six retained levels.
- [generate.py](generate.py): example code that writes the two datasets.
  - **Known values:** `EJ1 = 12`, `EJ2 = 8`, `Csh = 70`, with coil offset 0.1 mA and period 1.0 mA.
  - **Lines:** f01, f12 and the two-photon f02/2, with amplitudes 1, 0.3 and 0.2 V, each computed through this system file at every bias.
  - **Line shape and noise:** Lorentzian lines of 2 MHz half width, a rotated I/Q background with a slope, and 0.02 V Gaussian noise per quadrature, from a fixed seed.
  - **Provenance:** the generator is recorded as the converter. Synthetic data has no laboratory raw file, so no raw-file digest is recorded.
- [training.toml](training.toml): a map of 21 coil biases from −0.25 to 0.38 mA, 3.5–6.5 GHz at 1 MHz.
- [holdout.toml](holdout.toml): a separate measurement of 8 biases from 0.40 to 0.65 mA, 2.0–5.0 GHz, nearer half flux, where the lines bend most. No fit reads it.
- [calibrate.toml](calibrate.toml): the training map only, with these settings:
  - **Fitted parameters:** `EJ1`, `EJ2` and `Csh`, with an axis map whose offset and period are fitted as nuisance parameters.
  - **Family:** transitions from levels 0 and 1 up to level 3 at one and two photons.
  - **Anchor:** each column's strongest peak is taken to be 0-1 for the first fit.
  - **Extraction and matching:** the settings of the probe.
  - **Conventions:** the offset window and the `EJ1`/`EJ2` exchange.
  - **Refinement:** a declaration (two more levels, ten more charge units), used only by the refinement check.
  - **Starts:** two. The first is the design values with a nominal coil offset and period; the second lies a few percent the other way.
- [calibration.py](calibration.py): runs the calibration, the checks and the proposal through the CLI or a scripted MCP client. Both paths make the same calls.

## Run

Use the runner profile already configured for the checkout. Each path's fit takes about 12 minutes on the CPU:

```sh
uv run --locked --exact python examples/transmon-calibration/calibration.py run --via cli --runs-root research_runs/calibration-cli --output research_runs/calibration-cli.json
uv run --locked --exact --extra mcp python examples/transmon-calibration/calibration.py run --via mcp --runs-root research_runs/calibration-mcp --output research_runs/calibration-mcp.json
```

Each run prints the operator's command. Applying the proposal is the operator's action, on a copy outside the runs root; the tracked system file keeps its design values. Afterwards `applied` adds the application record and the updated copy to the summary:

```sh
cp examples/transmon-calibration/system.toml research_runs/device-cli.toml
uv run --locked --exact entanglix apply-device-update <run path>/proposals/1.toml research_runs/device-cli.toml
uv run --locked --exact python examples/transmon-calibration/calibration.py applied --summary research_runs/calibration-cli.json --device research_runs/device-cli.toml
```

Each step the script takes is an ordinary command; `entanglix validate-dataset` reports a dataset before any fit:

```sh
entanglix calibrate examples/transmon-calibration/calibrate.toml --runs-root R
entanglix check <run> holdout identifiability refinement --holdout examples/transmon-calibration/holdout.toml
entanglix propose-device-update <run>
```

To regenerate the datasets, run `generate.py --output DIR`.

## Results

Both paths give the same numbers ([CLI](../../tests/probes/s2_example_cli.json), [MCP](../../tests/probes/s2_example_mcp.json)):
- **Outcome:** `solution`. Both starts reached it and merged.
- **Assignment:** each start's anchor phase assigned the strongest peak in all 21 columns. The full family then assigned all 63 extracted peaks, with none unexplained or unresolved.
- **Fit:** χ² is 75.3 over 63 peaks and 5 parameters, a reduced χ² of 1.30. The rank is 5, and no bound was hit.
- **Conventions:** neither convention changed a solution.
- **Missing lines:** 23. These are family lines predicted inside the map's window in a column where no peak was assigned: 1-3/2 in all 21 columns and 0-3/2 in two. The generator does not draw them.

| Parameter | Known | Fitted | Uncertainty | Error / uncertainty | Uncertainty / prior width |
|---|---:|---:|---:|---:|---:|
| `EJ1` | 12 GHz | 11.99803 GHz | 0.0014 GHz | −1.42 | 6.0e-5 |
| `EJ2` | 8 GHz | 7.99974 GHz | 0.0012 GHz | −0.22 | 5.2e-5 |
| `Csh` | 70 fF | 69.9904 fF | 0.0061 fF | −1.58 | 6.7e-5 |
| offset | 0.1 mA | 0.0999973 mA | 1.3e-6 mA | −2.09 | 1.3e-6 |
| period | 1.0 mA | 1.000022 mA | 3.6e-5 mA | 0.62 | 2.4e-5 |

The uncertainties are local, along the determined directions, and conditional on the assignment and on the extracted centres' uncertainties. `EJ1` and `Csh` are positively correlated (0.60), and the period is strongly correlated with each junction energy (−0.83 and 0.85). The offset lies 2.1 uncertainties from its known value. This is one realisation, and the [probe](../../tests/probes/s2_calibration_results.json) gives coverage over many.

**Checks**, requested together as one verification run with the holdout measurement:

| Check | Result |
|---|---|
| `holdout` | 24 extracted peaks, all assigned and scored: 17 within one prediction uncertainty (0.71) and all 24 within two. The mean squared standardised residual is 0.90. Descriptive, as no coverage expectation was declared |
| `identifiability` | Recomputed with the corrector's tolerances, the outcome, rank, determined parameters, uncertainties, correlations and χ² agree with the fit's report |
| `refinement` | At 8 levels and charge cutoff 40, the assigned predictions move by at most 2.1e-13 GHz, 6.2e-9 of a centre uncertainty |

The fitting run froze `calibrate.toml`, `system.toml`, `training.toml` and `training.npz` only. The verification run froze the holdout measurement under `holdout/`.

**Proposal** `proposals/1.toml`, from solution 0:

| Parameter | Old | New | Uncertainty |
|---|---:|---:|---:|
| `EJ1` | 12.4 GHz | 11.998029038 GHz | 0.0014 GHz |
| `EJ2` | 7.7 GHz | 7.999738704 GHz | 0.0012 GHz |
| `Csh` | 72 fF | 69.990438551 fF | 0.0061 fF |

Nothing is listed as not proposed. The offset and the period stay in the run's result. The proposal records the three checks with solution 0, and is bound to the system file's digest.

**Application.** For each path, the operator copied the system file outside the runs root and applied that run's proposal to the copy:
- The copy's digest before apply matched the tracked file.
- Each parameter's value and source changed; its unit and prior did not. The source names the proposal, the solution and the proposal's digest.
- The updated copy validates.
- The record is `proposals/1.applied.toml` beside the proposal.

The tracked `system.toml` keeps its design values.

## What this example does not show

- **Laboratory data.** The data were generated with the same model and family that fit them, with a known line shape and white noise. It does not test a line the family omits, a defect, a drifting background, or a conversion from a laboratory format. The probe's failure-mode cases cover a line-shape mismatch, defect lines and an unmodelled line.
- **That the anchor is right.** f01 is the strongest line by construction here. An anchor that held in every column does not establish that it was right on a real device.
- **Fast iteration.** Full construction at every evaluation takes about 0.3 s for the 21 columns, and the anchor phase fits all five parameters to the single 0-1 line, which leaves a nearly flat direction; the recorded fit took about 350 s per start.
