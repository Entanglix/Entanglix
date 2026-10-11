# Transmon control input

[system.toml](system.toml) is the junction-shunt circuit with three retained eigenstates and its dimensionless charge drive. [control.toml](control.toml) declares an X gate on basis indices 1 and 2, with the third level available for leakage, 20 ns in 20 equal intervals, and target fidelity 0.999.

The rotating frequency, computed from the system's 0–1 transition, is `5.482065688844835 GHz`. The channels use the same charge operator, with phases zero and `π/2`. The magnitude of the 0–1 charge matrix element is `1.1099964850544073`. To preserve the normalized probe's physical controls, each channel's amplitude bound is `0.05405422522311776 GHz`; the first and second discrete derivative bounds are `0.09009037537186294 GHz/ns` and `0.09009037537186294 GHz/ns²`.

The export removes the scalar energy origin `−12.183560513666507 GHz` from the drift while preserving the source charge operators. Both candidates below use ProblemSpec digest `716bdda08b61e182e4cea833a69282136bbb558ab36d492c042042a016daa3fa`. The [tests](../../tests/test_control.py) compare both quadratures, including the sign of the imaginary channel, with the adjacent-transition construction. A fixed complex-state pulse agrees with the QuTiP reference in the control frame within `1e-9` absolute state-entry error. This establishes export conventions; the baseline candidate is described below.

Validate and inspect the generated problem through Python:

```sh
uv run --locked --exact python - <<'PY'
from pathlib import Path
from entanglix.protocols.control import load_control

loaded = load_control(Path("examples/transmon-control/control.toml"))
print(loaded.listing)
PY
```

The [control reference](../../docs/operations/control.md) documents the input, units, frame, subspace indices, and frozen files. The declared model is closed and uses a rotating-wave approximation; the input makes no claim about dissipative or lab-frame gate accuracy.


## Open baseline

Run the gate through the CPU runner, with the `differentiable` extra in its established environment:

```sh
ENTANGLIX_RUNNER_EXTRAS=differentiable uv run --locked --exact entanglix control examples/transmon-control/control.toml --runs-root research_runs/m4-control
```

For the documented GPU workstation use its existing `cuquantum,differentiable-cuda12` runner profile instead. This solve still uses CPU only. The [control reference](../../docs/operations/control.md#launch-and-solver-selection) describes the environment, solver options and neutral pulse. [Baseline evidence](../../tests/probes/m4_baseline_results.json) retains the run, convergence and first/later objective-call costs. The candidate's objective is solver-reported; no independent control check is claimed.

The retained `grape` candidate reports fidelity **0.9996150467**, against the declared 0.999 target, after 30 SLSQP iterations and 39 objective/gradient calls. It reports successful convergence and took 16.2 s of job time. The first call took 12.4 s with tracing/compilation included; the second took 0.079 s. Subsequent calls had a median of 0.061 s, and subsequent iterations 0.074 s. These are CPU workstation measurements; the solve ran before the regression checks.

The neutral pulse lasts exactly 20 ns. Both channels reach their `0.0540542252 GHz` amplitude bound; the largest first difference is `0.0780027637 GHz/ns`, and the second difference reaches its `0.0900903754 GHz/ns²` bound within floating-point roundoff. Q and R are listed as ignored guidance by the pure-fidelity baseline. Random initialization uses the declared seed with 10% of the amplitude bounds. A preliminary full-bound initialization converged to only 0.97468 on this problem; convergence alone therefore does not establish the target. This candidate remains a closed-model RWA result, awaiting an independently requested control replay and target check.

## Piccolo candidate

[piccolo.toml](piccolo.toml) changes only the solver selection. It generates the same ProblemSpec digest as the baseline. Instantiate the pinned Julia project as described in the [control reference](../../docs/operations/control.md#piccolo), then run:

```sh
OPENBLAS_NUM_THREADS=1 uv run --locked --exact entanglix control examples/transmon-control/piccolo.toml --runs-root research_runs/m4-control
```

Keep the workstation's established runner extras profile; Piccolo itself needs no Python extra. [Retained evidence](../../tests/probes/m4_piccolo_results.json) records Julia 1.13.0, Piccolo 2.1.0, the pinned project digests, native pulse and neutral pulse, and both solver settings as arguments. The candidate reports fidelity **0.9999999922** against the 0.999 target, with native objective **0.0000104719795** including regularisation. It reached the declared 100-iteration cap at tolerance `1e-8`, so `converged = false`. The native solve took 18.0 s; the job took 26.8 s including Julia startup and compilation, with no concurrent regression work. Removing the scalar drift offset avoids the slow solve in the earlier export; these workstation timings are not a solver speed comparison.

The returned duration is `19.999999999995616 ns`. Channel maxima are `0.0515972200` and `0.0502446944 GHz`; the largest first and second differences are `0.0598031177 GHz/ns` and `0.0721359481 GHz/ns²`, below the declared bounds. These are quantities recomputed from the neutral pulse, not built-in check outcomes. As with the baseline, this is a closed-model RWA candidate awaiting a requested independent replay and target check.
