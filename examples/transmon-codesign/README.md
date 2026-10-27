# Transmon co-design example

This example runs one device–control co-design investigation over the shunt capacitance `Csh` of the [first gate's transmon](../transmon-control/README.md). It exercises the workflow. Its conclusions hold only for the model, samples, constraints and search policy declared here; they are not general device-design conclusions.

## Inputs

- [system.toml](system.toml): the junction–shunt circuit with `EJ = 15 GHz` and `ECJ = 2 GHz`, a declared `Csh` (60 fF unless varied), offset charge 0.23, five retained levels, charge cutoff 30 and a charge drive.
- [spectrum.toml](spectrum.toml) and [screen.toml](screen.toml): a spectrum sweep over `Csh` = 10, 20, 30, 45, 60, 75 and 90 fF.
- [control.toml](control.toml): the base control input.
  - An X gate on levels 1 and 2, in 6 ns with 21 knots.
  - The frame on the named 0–1 transition, with both quadratures following it.
  - Rabi-rate bounds of 0.15 GHz, 0.33 GHz/ns and 1.1 GHz/ns².
  - GRAPE with `max_iter = 100`.
  - An evaluation table of nine samples: offset charge 0, 0.25 and 0.5 (drifting) crossed with `EJ` = 14.985, 15 and 15.015 GHz (calibrated). Its refinement adds two retained levels and ten charge-cutoff units.
- [codesign.py](codesign.py): drafts the investigation terms, then runs the investigation through the CLI or a scripted MCP client. Both paths make the same calls.

## Declared conditions

- **Compared quantity:** the worst fidelity over the nine samples, taken from a candidate's best pulse that passes replay and constraints. It was fixed before the duration or bound was varied.
- **Regime:** chosen by an exploratory [pilot](../../tests/probes/m5_pilot.py) ([results](../../tests/probes/m5_pilot_results.json)), which solved six capacitances with three seeds in three regimes, at three and five levels. The 6 ns, 0.15 GHz regime was selected because it exercises the workflow with distinguishable outcomes. It is not a preferred-regime finding: the order of the leading candidates differs between regimes.

  | Best worst-sample fidelity, five levels | 10 fF | 20 fF | 30 fF | 45 fF | 60 fF | 90 fF |
  |---|---|---|---|---|---|---|
  | 20 ns, 0.06 GHz | 0.809 | 0.990 | 0.986 | 0.9999985 | 0.9999998 | 0.9999994 |
  | 10 ns, 0.10 GHz | 0.786 | 0.997 | 0.99985 | 0.9999996 | 0.9999998 | 0.9999998 |
  | 6 ns, 0.15 GHz | 0.922 | 0.998 | 0.99993 | 0.9999989 | 0.9999999 | 0.9999999 |

- **Truncation:**
  - Pulses optimised on three levels are truncation-sensitive. Refinement to five changes their worst-sample fidelity by up to 0.074.
  - With five levels at 6 ns, the change is at most 1.3e-7.
  - This supports five levels for the declared pulses and samples. It does not prove convergence generally.
- **Search policy:** the terms frozen at drafting.
  - **Approved base:** `Csh` from 10 to 90 fF, and seeds 7, 19 and 43.
  - **Seeds:** every candidate is solved with all three seeds, with no additional attempts.
  - **Initial candidates:** 10, 30 and 60 fF.
  - **Next candidate:** if the best-scoring candidate is the largest or smallest solved and the range extends beyond it, the range end is solved. Otherwise, the midpoint between it and its better neighbour.
  - **Stop:** when the best candidate and its better neighbour differ by no more than the larger of their seed spreads, or after five candidates. This decides only not to spend further; it leaves the comparison unresolved.
- **Checks:** replay, constraints, `uncertainty` and `uncertainty_refinement` are standing-requested on the `gate` base and issued explicitly after every solve.
- **Checkpoint:** every six finished attempts. An assessment is recorded before each new candidate.
- **Not modelled or sampled:**
  - relaxation and other device-dependent decoherence;
  - time-dependent noise (the samples are quasistatic offsets);
  - calibration error (a calibrated sample assumes perfect recalibration of the named transition);
  - fabrication spread in `Csh` or `ECJ`;
  - flux noise;
  - crosstalk;
  - effects beyond the rotating-wave approximation.

  Equal Rabi-rate bounds do not imply equal hardware drive voltage or power.

## Run

Use the runner profile already configured for the checkout. Draft the terms, review them, and start the investigation as the operator:

```sh
uv run --locked --exact python examples/transmon-codesign/codesign.py draft --via cli --runs-root research_runs/codesign-cli
uv run --locked --exact entanglix investigation-decision <investigation path> start --revision 1
uv run --locked --exact python examples/transmon-codesign/codesign.py run --via cli --runs-root research_runs/codesign-cli --investigation <investigation path> --output research_runs/codesign-cli.json
```

For the scripted MCP client, run the same steps with the `mcp` extra:

```sh
uv run --locked --exact --extra mcp python examples/transmon-codesign/codesign.py draft --via mcp --runs-root research_runs/codesign-mcp
uv run --locked --exact --extra mcp entanglix investigation-decision <investigation path> start --revision 1
uv run --locked --exact --extra mcp python examples/transmon-codesign/codesign.py run --via mcp --runs-root research_runs/codesign-mcp --investigation <investigation path> --output research_runs/codesign-mcp.json
```

Each step the script takes is an ordinary command:

```sh
entanglix sweep examples/transmon-codesign/screen.toml --runs-root R --investigation ID --base screen
entanglix control examples/transmon-codesign/control.toml --runs-root R --investigation ID --base gate --vary parameters.Csh=60.0 --vary pulse.seed=19 --parent-run <screen run ID>
entanglix check <control run> replay constraints uncertainty uncertainty_refinement
entanglix record-finding <investigation> finding.toml
entanglix report-investigation <investigation> --output report.md
```

## Results

Both paths give the same numbers ([CLI](../../tests/probes/m5_example_cli.json), [MCP](../../tests/probes/m5_example_mcp.json)). Each used 25 jobs and about 32 job-seconds. Every seed converged; replay and constraints pass on all 12 pulses; every control run's parent is the screening sweep.

| `Csh` | f01 | Anharmonicity | Best worst-sample fidelity (seed) | Seed spread | Refinement change |
|---|---|---|---|---|---|
| 10 fF | 9.76 GHz | −1.44 GHz | 0.92249 (19) | 4.9e-3 | 1e-11 |
| 30 fF | 7.13 GHz | −0.59 GHz | 0.999927 (43) | 6.5e-5 | 7e-10 |
| 60 fF | 5.48 GHz | −0.32 GHz | 0.99999985 (19) | 2.9e-7 | 1.2e-8 |
| 90 fF | 4.63 GHz | −0.22 GHz | 0.99999989 (19) | 2.2e-7 | 1.3e-7 |

- **How the search proceeded:** after the three initial candidates, 60 fF scored best and was the largest solved, so the rule chose the range end, 90 fF.
- **Why it stopped:** 60 and 90 fF differ by 3.3e-8, within the larger seed spread of 2.9e-7.
- **Report:** retain the evaluated 60 and 90 fF candidates; their ranking is unresolved. Refinement reverses their order. The selected pulses' worst-sample fidelities are 0.9999998545 and 0.9999998872, and 0.9999998545 and 0.9999998348 after refinement. Resolving the pair needs attempts the terms do not allocate, such as more seeds or another solver.
- **What the scores mean:** each is a lower bound from the pulses found. Finite search shows no upper bound, and no device between the evaluated candidates was evaluated.

## What this example does not show

- **General device-design conclusions.** The uncertainty diagnostic currently supports a single periodic circuit coordinate. For fluxonium, multi-node and composite models it is inapplicable, with its reason, though those models can still be modelled, solved and checked.
- **A preferred device or regime.** The leading candidates' order depends on the regime and on refinement.
