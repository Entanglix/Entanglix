# Weak-readout example

[system.toml](system.toml) is a validated transmon–resonator model. Carrier drives,
rotating frames, and public simulation/sweep commands are not yet available through
the harness. The [CPU feasibility probe](../../tests/probes/m2_feasibility.py) evaluates
these cases directly in QuTiP; its [retained measurements](../../tests/probes/m2_feasibility_results.json)
include solver times, accepted steps, physicality, spectra, and convergence comparisons.

```sh
uv run --locked --exact entanglix validate examples/readout/system.toml
MPLCONFIGDIR=/tmp/entanglix-m2-matplotlib uv run --locked --exact python tests/probes/m2_feasibility.py --output tests/probes/m2_feasibility_results.json
```

## Model and quantities

The qubit has `EC = 0.25 GHz`, `ng = 0`, and `EJ = 12, 13.5, 15 GHz`.
The resonator frequency is `7 GHz`, its loss rate is `0.03/ns`, and the fixed
charge–position coupling coefficient is `0.075 GHz`. The resulting qubit transition
coupling is `g01 = coefficient × |⟨0|charge|1⟩| = 0.081–0.086 GHz`; the coefficient
itself is not `g01`. Production retains 5 qubit × 6 resonator levels (dimension 30)
with charge cutoff 30. Reference refinement retains 6 × 8 (dimension 48) and cutoff 32.
Reference preparation aligns local eigenvector phases and embeds the same initial
state into the refined basis.

Dressed energies are assigned to `(q,r) = (0,0), (0,1), (1,0), (1,1)` by their
largest bare-product overlap; the assignments are unique and overlaps exceed 0.99
in the selected region. Define `fr,q = E(q,1) − E(q,0)` and
`χ = (fr,1 − fr,0)/2`. The static companion reports this `χ` over the three `EJ`
values, using at least the lowest five energies. It ranges from approximately
`−0.435` to `−0.734 MHz`.

## Readout and Purcell cases

Readout starts in `(|0⟩ + |1⟩)/√2` with an empty resonator. The metric is normalized
endpoint coherence `C(T) = 2|ρq,01(T)|`, where `ρq = Tr_resonator ρ` and `C(0) = 1`.
It is a coherence proxy, not a fitted pure-dephasing rate: relaxation and coherent
mixing can also affect it. The zero-drive row supplies the undriven baseline. Use a
100 ns window, 1 ns output spacing, and the 3 × 3 grid of the listed `EJ` values
and drive amplitudes `A = 0, 0.001, 0.002 GHz`. At fixed `EJ`, decreasing coherence
with increasing nonnegative amplitude is the candidate monotonicity check for this
grid and window; no monotonicity across `EJ` or outside this region is assumed.

Doubling the amplitude from `0.001` to `0.002 GHz` increases the endpoint coherence
loss relative to the zero-drive baseline by `4.80`, `2.93`, and `3.15` times at
`EJ = 12, 13.5, 15 GHz`, respectively. Together with increasing `|χ|` and increasing
maximum-drive loss across these `EJ` values, this supports the qualitative dispersive
trend. It does not establish an exact quadratic law or isolate a pure-dephasing rate.

The readout frequency follows `(fr,0 + fr,1)/2` at each production grid point
(approximately `7.00175–7.00276 GHz`). Freeze that frequency for its reference solve.
The lab-frame drive is `A cos(2π fd t) (a + a†)` in GHz, with zero phase. The probe's
common rotating frame uses `fd` on both subsystems' excitation-number operators,
retains excitation-conserving interactions, and retains half the cosine amplitude.
It is an explicit rotating-wave approximation (RWA). Maximum observed photon
occupation is below 0.121, versus critical photon numbers above 108. This is weak
readout, not typical measurement photon loading. Coherence decreases from the
zero-drive baseline by about 0.004–0.017 at the strongest drive.

The Purcell example uses `EJ = 15 GHz`, the excited qubit, empty resonator, no drive,
and a 1600 ns window with 1 ns spacing. Its final excited population is about 0.888
in the lab frame and 0.892 with the RWA. The leading dispersive estimate
`γP = κ(g01/Δ)²` gives `γP T ≈ 0.11`; `exp(−γP T) ≈ 0.896` is a trend comparison,
not an exact reference or a `1e-6` acceptance condition.

## Numerical evidence and candidate frame bounds

The probe checks all nine rotating-frame readout points. It compares the four grid
corners and the central `EJ` at maximum drive in both frames and at refined dimensions.
All reported coherence, population, and photon-number refinement residuals are below
`1e-6`; the largest is approximately `9.63e-7` in the Purcell photon number, leaving
only `3.68e-8` headroom (about 3.7% of the limit). The lab-frame Purcell run reaches
`3.72e-6` population in the highest retained qubit level, triggering the `1e-6`
truncation warning; refinement reduces that population to about `5.03e-8`.
Tightening
readout integration from `rtol/atol = 1e-10/1e-12` to `1e-11/1e-13` changes normalized
coherence by less than `9e-9`. Halving output spacing agrees at common timestamps.
These are same-engine QuTiP refinement checks at the sampled evidence points, not
independent-engine acceptance of the grid or arbitrary models and windows. Before
public protocols rely on the lab-frame 5 × 6 fixture, including the Purcell window,
it must pass comparison with an independent CPU dynamiqs reference at tightened
tolerances and 6 × 8 levels. Lab-frame references remain required.

The RWA is a separate model approximation. Maximum combined rotating-production
versus refined-lab residuals are approximately `0.00193` in normalized coherence,
`0.00390` in excited population, and `0.00551` in photon number. Candidate absolute
trajectory diagnostic bounds are respectively `0.003`, `0.005`, and `0.008`.
For the reported readout metric `C(100 ns)`, use a separate candidate absolute bound
of `5e-4`: the largest sampled endpoint error is `3.96e-4`. These candidates require
independent-engine acceptance before becoming protocol verification rules. The
`0.001 GHz` amplitude points have no lab-frame reference, so this is not an accepted
bound across the whole grid. The current corrector's `1e-6` numerical comparison
limit remains unchanged; these RWA cases would not pass that limit as frame comparisons.

Two endpoints with absolute error bound `5e-4` give a conservative `1e-3` uncertainty
on their difference. The measured adjacent endpoint decreases are:

| `EJ` (GHz) | `A: 0 → 0.001 GHz` | `A: 0.001 → 0.002 GHz` |
| --- | --- | --- |
| 12 | `8.22e-4` | `3.12e-3` |
| 13.5 | `4.03e-3` | `7.77e-3` |
| 15 | `5.49e-3` | `1.18e-2` |

The first `EJ = 12` step is unresolved at this candidate frame resolution, and the
second is only about 3.1 times the difference bound. Keep the 100 ns window and grid
for a qualitative study; do not claim precise low-drive effects there. At maximum
drive, the rotating-frame loss from baseline differs from the refined-lab result
by about 13% at `EJ = 12`, versus 0.64% at `EJ = 15`. A future `VERIFIED` verdict
within declared metric bounds would not imply that every sweep increment is resolved.

Use `vern9`, production `rtol/atol = 1e-10/1e-12`, refined `1e-11/1e-13`, and
100,000 maximum steps per output interval. Step counts in the probe exclude rejected
trials and dense-output interpolation; its counted trajectories agree with ordinary
`vern9` in both short validation cases.

On the measured workstation, rotating readout solves take about 0.26–0.44 s;
full-window refined lab references take about 10–21 s. Their combined solver cost
is about 10–22 s per checked readout point. The Purcell rotating solve takes about
1.6 s and its refined lab reference about 110 s. Times include the probe's step-count
instrumentation and exclude model construction, environment setup, and runner overhead.
These are CPU feasibility measurements;
they make no GPU performance claim. Reference cost dominates and must be included
in later run estimates and reused when matching retained evidence is available.
