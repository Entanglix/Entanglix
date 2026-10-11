# Transmon control input

[system.toml](system.toml) is the junction-shunt circuit with three retained eigenstates and its dimensionless charge drive. [control.toml](control.toml) declares an X gate on basis indices 1 and 2, with the third level available for leakage, 20 ns in 20 equal intervals, and target fidelity 0.999.

The rotating frequency, computed from the system's 0–1 transition, is `5.482065688844835 GHz`. The channels use the same charge operator, with phases zero and `π/2`. The magnitude of the 0–1 charge matrix element is `1.1099964850544073`. To preserve the normalized probe's physical controls, each channel's amplitude bound is `0.05405422522311776 GHz`; the first and second discrete derivative bounds are `0.09009037537186294 GHz/ns` and `0.09009037537186294 GHz/ns²`.

The source charge operators are preserved in the export. The [tests](../../tests/test_control.py) compare both quadratures, including the sign of the imaginary channel, with the adjacent-transition construction. A fixed complex-state pulse agrees with the QuTiP reference in the control frame within `1e-9` absolute state-entry error. This is export evidence; no solver has run on this input.

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
