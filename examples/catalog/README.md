# Reuse a catalogued device and pulse

This example stages one retained candidate from the [transmon co-design example](../transmon-codesign/README.md), waits for the operator to promote it, and reuses its device and pulse through either the CLI or a scripted MCP client. It performs no promotion itself.

The chosen source is the 60 fF candidate with seed 19. The co-design comparison retained both 60 and 90 fF with their ranking unresolved; choosing 60 fF here demonstrates reuse and does not resolve that comparison.

After promotion, the client:

1. Sweeps `EJ` over the entry's value ±0.5 GHz, reporting the lowest three levels and deriving `f01`. Only `EJ` is declared in the base, so the entry's `Csh` holds. At the nominal `EJ = 15 GHz`, `f01` should reproduce the co-design candidate's approximately 5.48 GHz.
2. Repeats the source's 6 ns, 21-knot X-gate problem on the copied device, starting from the catalogued pulse. The source's frozen gate and solver settings are retained; `init` and `seed` are removed because the supplied pulse replaces them.
3. Requests `replay`, `constraints`, `uncertainty` and `uncertainty_refinement` on the new pulse, as for the source, then reads the entry's lineage and reuses.

The gate is a closed rotating-wave model, and its uncertainty samples are the source's declared quasistatic offsets. Checks cover that model and those samples; they establish neither hardware performance nor lab-frame accuracy. Promotion adds no checks, and source checks are not checks of the new runs. There is no run-level verdict.

## Recorded results

On 2026-10-25, the operator promoted a copy of the 60 fF candidate in each runs root. The [CLI summary](../../tests/probes/s3_catalog_cli.json) and [scripted MCP summary](../../tests/probes/s3_catalog_mcp.json) record the promoted entries, the promoter, the comparison claim, source checks and ancestry, and each entry's sweep and warm-start reuse.

Both paths preserved `Csh = 60 fF` and returned identical spectra:

| EJ (GHz) | f01 (GHz) |
|---|---|
| 14.5 | 5.384683356 |
| 15.0 | 5.482065689 |
| 15.5 | 5.577826405 |

Each baseline solve used the entry's pulse exactly, reported SLSQP success after one iteration and one objective evaluation, and returned unchanged amplitudes. Independent replay gave nominal fidelity `0.9999999866` and leakage `1.34e-8`; its disagreement with the solver was `2.22e-16`, below the `1e-7` replay limit. Replay and constraints passed. Across all nine declared samples, the minimum fidelity was `0.999999854530`, changing to `0.999999854546` at the declared refinement; the largest per-sample change was `3.12e-10`. The uncertainty diagnostics are descriptive, and no target check was requested.

The sweep and warm-start manifests name the entry and its source run; the new check names the warm-start run as its parent. The new checks use policy `efae45abf26599051af3fc463fa9c30ab1bf391adf75817d0d2c9a4043688868`, while promotion retains the source checks' older policy. These results demonstrate reuse within the model and sample limits above; the 60-versus-90 fF ranking remains unresolved.

## Run

Use the checkout's existing runner extras profile, for example:

```sh
export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12
export OPENBLAS_NUM_THREADS=1
. .venv/bin/activate
```

Each path's source is its 60 fF, seed 19 control run from the [co-design example](../transmon-codesign/README.md#run). Without one, produce a standalone source from the co-design model and control input with `Csh = 60` and `pulse.seed = 19`, with the same gate, bounds, solver settings, evaluation samples and four requested checks. There is no need to repeat the complete search.

Stage the CLI source:

```sh
uv run --locked --exact python examples/catalog/reuse.py stage --via cli \
  --source research_runs/codesign-cli/junction_with_shunt/<run_id>
```

Inspect the staged candidate with `entanglix read-entry STAGED_ID --runs-root research_runs/codesign-cli`, then run the printed bare `entanglix promote ...` command in the operator terminal. Continue with the returned entry ID:

```sh
uv run --locked --exact python examples/catalog/reuse.py reuse --via cli \
  --entry ENTRY_ID --runs-root research_runs/codesign-cli \
  --workspace research_runs/catalog-cli-inputs --output research_runs/catalog-cli.json
```

For the scripted MCP path, stage its source and repeat the operator promotion:

```sh
uv run --locked --exact --extra mcp python examples/catalog/reuse.py stage --via mcp \
  --source research_runs/codesign-mcp/junction_with_shunt/<run_id>
uv run --locked --exact --extra mcp python examples/catalog/reuse.py reuse --via mcp \
  --entry ENTRY_ID --runs-root research_runs/codesign-mcp \
  --workspace research_runs/catalog-mcp-inputs --output research_runs/catalog-mcp.json
```

The MCP server is CPU-only and receives the existing runner extras; this launches no agent runtime session. Each path creates three ordinary jobs: a sweep, a warm-start control solve and its checks. Inputs are written in the supplied workspace and frozen by the runner. The summary keeps promotion scope, run and entry lineage, spectral values, fidelity and refinement diagnostics, and execution usage; full reports stay in the runs root.

The client reads the source's frozen control input to repeat its gate. The catalog independently retains the device and pulse, but those designs do not by themselves declare a gate problem. Rerunning the client creates new attempts rather than continuing an earlier run.
