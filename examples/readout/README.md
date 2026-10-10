# Weak-readout example

[system.toml](system.toml) is a validated transmon–resonator model. Carrier drives
and explicit rotating-wave frames are available through the
[Python evolution interface](../../docs/operations/system-spec.md#carriers-and-rotating-frames).
Public simulation and serial static/dynamic grid sweeps are available. The
[CPU feasibility probe](../../tests/probes/m2_feasibility.py) evaluates
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

The [CPU Purcell protocol](purcell.toml) uses the lab frame. The
[CUDA protocol](purcell-cuda.toml) declares cuDensityMat and the RWA studied below.
The common frame frequency `7.002760675881172 GHz` is the dressed readout midpoint
at `EJ = 15 GHz` from the frame study. With no drive, the RWA depends on excitation
conservation rather than this particular frequency. Both protocols use the same
model, initial state, 1600 ns window, and refinement to 6 × 8 retained levels.
No check runs automatically. Request `integrity`, `segments`, or `diagnostics` after
completion; segment tolerances default to `1e-6` and do not bound the RWA error.

```sh
export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12
uv run --locked --exact entanglix simulate examples/readout/purcell.toml
uv run --locked --exact entanglix simulate examples/readout/purcell-cuda.toml
```

See the [protocol reference](../../docs/operations/protocols.md) for results, plots,
requested checks, cancellation, and replay. CUDA execution makes no performance
claim; the engine's crossover and workspace costs remain unmeasured.

The [retained CPU simulation/replay](../../tests/probes/m2_simulation_cpu.json)
and [CUDA simulation/replay](../../tests/probes/m2_simulation_cuda.json) record
requested integrity, local segments and diagnostics, with identical numerical
outputs on replay. The CPU summary also includes a 6 × 8, dimension-48 `simulate`
and its requested integrity check as a more-levels comparison. Reports retain
coverage and work time. Lab CPU and RWA CUDA outputs are two different models:
passing local checks establishes neither their whole-window error nor RWA accuracy.
The [acceptance command](../../docs/environment.md#simulation-acceptance) regenerates
the records and summaries.

## Closed-system Floquet quasienergies

The [authored script](floquet.py) uses this model at its default `EJ = 13.5 GHz`
and 5 × 6 retained levels, with its declared readout operator and
`A cos(2π f t) (a + a†)`. It deliberately omits the `resonator_loss = 0.03/ns`
dissipator and records that omission. The frequency `f = 6.9 GHz` is detuned
below the roughly 7 GHz readout resonance to avoid strong mode mixing and large
highest-resonator-level occupation in this six-level basis; the period is
`1/f ≈ 0.144928 ns`.
This is full lab-frame periodic driving, using QuTiP `FloquetBasis` with `vern9`
at `rtol/atol = 1e-10/1e-12` and a 100,000-step ceiling.

```sh
uv run --locked --exact entanglix run-script examples/readout/floquet.py --system examples/readout/system.toml
uv run --locked --exact entanglix read-run research_runs/readout/<run_id>
uv run --locked --exact entanglix rerun-script research_runs/readout/<run_id>
```

Use the same [runner extras profile](../../docs/operations/runs.md#authored-cpu-jobs)
as other jobs in the checkout. The script, model and settings are frozen with the
run. `artifacts/quasienergies.npz` retains all 30 quasienergies at each amplitude,
label assignments, overlaps and margins, phase sampling times, both subsystem
population diagnostics, and the transition shift. Quasienergies are
in GHz, folded into `[-f/2, f/2)`, after conversion from radians/ns.

The two static dressed states are assigned to `(q,r) = (0,0), (1,0)`
by squared bare-product overlaps. Their Floquet labels use squared overlaps with
those static eigenstates at `t = 0`, at zero carrier phase. Assignments must be
distinct, with overlap at least `0.9`; otherwise the script fails with the
offending assignments and overlaps. Sorted quasienergy positions do not define the
labels. The `r = 0` metric is
`wrap_f[(ε(1,0;A) − ε(0,0;A)) − (ε(1,0;0) − ε(0,0;0))]`, choosing the shift nearest
zero modulo `f`. This convention describes the small shifts in this sampled region;
it does not select all physical transition sidebands or track branches through crossings.

| Amplitude (GHz) | Qubit shift, `r = 0` (kHz) |
| --- | --- |
| 0 | 0 |
| 0.005 | −0.736031 |
| 0.01 | −2.944122 |
| 0.02 | −11.776446 |

The script's numerical evidence is authored data in `result.toml`. Across this
grid, zero-drive agreement with all static eigenvalues modulo `f` has maximum
residual `9.64e-11 GHz`. Tightening to `1e-11/1e-13` changes the two labelled
quasienergies by at most `7.38e-14 GHz`; both comparisons require `1e-8 GHz`.
The one-period unitarity residual `||U†U − I||_F` is at most `1.32e-9`, below
the `1e-8` limit. The smallest static overlap is `0.99829`; the smallest Floquet
label overlap is `0.99024`, above the `0.9` assignment threshold. Among these two
labelled modes, the maximum sampled highest-qubit-level population is `1.20e-7`
(already `1.01e-7` at zero drive), and the highest-resonator-level population is
`5.58e-12`. Both are below the `1e-6` truncation-warning level used in this example.
The populations are sampled at 32 uniformly spaced, distinct phases over one
period; unitarity is evaluated separately at `T`. They are diagnostics of the
fixed basis, not evidence of basis convergence.

These results claim quasienergies under strictly periodic driving only. Crossings
alone do not establish transition probabilities or dissipative accuracy. Integration
refinement leaves the construction cutoff and retained levels fixed. Authored scripts
receive no built-in checks; the [CPU runner test](../../tests/test_floquet_example.py)
checks the numerical evidence, label assignments, artifact and pinned shift values.

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
independent-engine acceptance of the grid or arbitrary models and windows. The
[independent-engine study below](#independent-engine-frame-acceptance) passes the
lab-frame 5 × 6 fixture, including Purcell, against CPU dynamiqs at tightened
tolerances and 6 × 8 levels. Another model or window needs its own lab-frame
comparison run; no built-in check supplies one.

The RWA is a separate model approximation. Maximum combined rotating-production
versus refined-lab residuals are approximately `0.00193` in normalized coherence,
`0.00390` in excited population, and `0.00551` in photon number. Candidate absolute
trajectory diagnostic bounds are respectively `0.003`, `0.005`, and `0.008`.
For the reported readout metric `C(100 ns)`, use a separate candidate absolute bound
of `5e-4`: the largest sampled endpoint error is `3.96e-4`. The
[independent-engine study below](#independent-engine-frame-acceptance) accepts these
bounds at its sampled points; they are not segment tolerances or a built-in RWA accuracy check.
It additionally covers `EJ = 12, A = 0.001 GHz`; the other intermediate-drive
points remain unreferenced, so this is not an accepted bound across the whole grid.
These differences are far above the `1e-6` default segment tolerance, which applies
only to local comparisons in the run's own frame.

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
by about 13% at `EJ = 12`, versus 0.64% at `EJ = 15`. Agreement within a stated metric bound would not imply that every sweep increment is resolved.

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
in later run estimates when requesting comparison solves. No full-window reference is run automatically.

## Independent-engine frame acceptance

The [shared-interface study](../../tests/probes/m2_frames.py) and
[retained results](../../tests/probes/m2_frames_results.json) compare QuTiP at
5 × 6 levels with CPU dynamiqs at 6 × 8, `rtol/atol = 1e-11/1e-13`, and the same
lab-frame carrier frequency, initial-state embedding, and complete output window.
It covers the four readout corners, `EJ = 13.5` at maximum drive, the unresolved
`EJ = 12` intermediate-drive point, and the 1600 ns Purcell case. Every sampled
lab-production observable agrees within `1e-6`; the largest residual is about
`9.63e-7` in Purcell photon number. Physicality also passes. The truncation warning
and narrow numerical margin above remain applicable.

The shared RWA implementation passes the trajectory diagnostic bounds and the
`5e-4` endpoint-coherence bound against these independent lab references. Maximum
readout endpoint error remains about `3.96e-4`; at the additional `EJ = 12` low-drive
point it is `8.89e-5`. This acceptance covers those sampled points and windows,
preserving the unresolved-increment qualification above. These measurements do not establish accuracy for an untested run.

On this run, independent dynamiqs readout references took about 45–77 s, including
JAX compilation and device synchronization; the Purcell reference took about 385 s.
These costs replace the earlier QuTiP refinement timings when estimating independent
verification, and depend on host load and compilation reuse. The study uses a
`2_000_000` integration-step ceiling so dynamiqs can cover the full lab-frame solve;
its ceiling counts per solve, unlike QuTiP's per-output-interval limit.

```sh
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/entanglix-m2-matplotlib uv run --locked --exact --extra differentiable python tests/probes/m2_frames.py --output tests/probes/m2_frames_results.json
```

## Serial readout and static sweeps

[dephasing.toml](dephasing.toml) runs the declared 3 × 3 `EJ`/amplitude grid through
[readout.toml](readout.toml), with its complex initial state in
[readout-state.npz](readout-state.npz). It uses the 100 ns rotating-wave computation
and the endpoint metric `C(T) = 2|ρq,01(T)|`. The carrier and both frame frequencies
are paired explicitly with `EJ` in the sweep: `7.0017536727417085`,
`7.002206406997239`, and `7.002760675881172 GHz`. These are the current model's
four-label dressed midpoints and stay fixed during truncation refinement. The
[CPU convention test](../../tests/test_sweep.py) checks coherence normalization,
dressed assignment and the midpoint at the default `EJ`.

On the workstation, preserve the shared runner extras profile:

```sh
export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12
uv run --locked --exact entanglix sweep examples/readout/dephasing.toml --dry-run
uv run --locked --exact entanglix sweep examples/readout/dephasing.toml
uv run --locked --exact entanglix read-run research_runs/readout/<run_id>
uv run --locked --exact entanglix check research_runs/readout/<run_id> integrity segments diagnostics declared_expectation
uv run --locked --exact entanglix rerun-sweep research_runs/readout/<run_id>
```

For CPU dynamics, copy `system.toml`, `readout.toml`, `readout-state.npz` and
`dephasing.toml` beside each other and change `[needs].device` in `readout.toml`
to `"cpu"`. Omitting the frame gives a lab-frame sweep; preserve the paired carrier
frequencies and explicitly compare the resulting metrics when frame accuracy matters.
Local segments do not check the rotating-wave approximation. The earlier frame study's
low-drive resolution limits still apply.

The static companion [dispersive.toml](dispersive.toml) uses
[spectrum.toml](spectrum.toml) to report `χ` over `EJ`, with five reported energies
and four uniquely assigned dressed labels. It requires CPU and has no dynamics:

```sh
uv run --locked --exact entanglix sweep examples/readout/dispersive.toml
uv run --locked --exact entanglix check research_runs/readout/<run_id> spectral_refinement
```

Each point retains its states/observables or spectral arrays, solver settings, and
metrics. Aggregate line/heatmap plots and extrema help choose a finer region. To
refine the grid, copy the files, retain `EJ = 13.5`, reduce every `linked_values`
list to its matching frequency, and set amplitudes to `[0.001, 0.0015, 0.002]`.
Launch the new sweep with `--parent-run <earlier_run_id>`. A replay instead recomputes
the unchanged frozen grid, including points completed before interruption.

The [acceptance probe](../../tests/probes/m2_sweep.py) retains CPU/CUDA grid replay,
requested checks, a static companion and a linked narrower CPU grid. Its
[CPU summary](../../tests/probes/m2_sweep_cpu.json) and
[CUDA summary](../../tests/probes/m2_sweep_cuda.json) record results and per-point
coverage; [commands](../../docs/environment.md#sweep-acceptance) regenerate them.
CUDA demonstrates the execution path and carries no performance claim. Passing
segments applies only to selected points and sampled local windows; no run-level
verdict, whole-window error bound, or RWA certification is produced.

The retained CPU endpoint grid is:

| `EJ` (GHz) | `A = 0 GHz` | `A = 0.001 GHz` | `A = 0.002 GHz` |
|---|---:|---:|---:|
| 12 | 0.997271 | 0.996449 | 0.993327 |
| 13.5 | 0.995635 | 0.991608 | 0.983839 |
| 15 | 0.993890 | 0.988401 | 0.976614 |

CUDA endpoints agree within `4.82e-8` at these points. This compares two engines
for the same RWA declaration. All requested checks pass; local segments sample
points `0, 2, 6, 7, 8` and three 2 ns windows per point. Integrity, diagnostics
and the declared expectation cover all nine points. CPU replay is numerically
identical; CUDA replay differences across stored states and
observables remain below `1e-7`. Both replays match their parents' code and environment.
Whole-job execution takes about `5 s` on CPU and `8 min` on CUDA in these
retained runs, excluding validation and checks. These timings describe this small
fixture and backend, not a general crossover. The static `χ` values are
`−0.434875`, `−0.559116`, and `−0.733591 MHz`; requested refinement passes.

## Threshold characterization

This study asks, at each of the three `EJ` values, which readout drive amplitude
produces an additional endpoint-coherence loss of `0.01` at 100 ns. The loss is
`ΔC(A) = C(0) − C(A)` for the endpoint coherence defined above. It characterizes this
model; coherence alone does not establish readout quality or an optimal setting.

The [scripted client](investigate.py) runs this question as a human-started CPU
investigation with scoped admission, assessments, and a retained report. The
[retained probe](../../tests/probes/m3_readout.py) ran the plan on CPU with the
existing `sweep`, `simulate`, and `check` operations, and its
[results](../../tests/probes/m3_readout_results.json) name every run.
[Commands](../../docs/environment.md#threshold-characterization-probe) regenerate them.

- **Fixed:** the model, initial state, 100 ns window, and tolerances above, and
  `EJ = 12, 13.5, 15 GHz`, each with its paired carrier and frame frequency.
- **Varied:** the drive amplitude in `[0, 0.002] GHz`.
- **Search:** the 3 × 3 grid above as the seed, then midpoint bisection in the
  rotating frame between the seed amplitudes that bracket the target.
- **Confirmation:** lab-frame runs at zero amplitude and at both bracket ends, one at
  5 × 6 levels and one refined at 6 × 8 levels, cutoff 32, and tolerances
  `1e-11/1e-13`. The refined initial state is embedded in the refined basis, so its
  array differs between `EJ` values.
- **Claim rule:** a bracket is confirmed when the refined lab-frame loss is below the
  target at the lower end and above it at the upper end, each by more than the
  observed change between the two lab-frame runs.
- **Requested checks:** integrity, diagnostics, and the declared expectation on the
  seed sweep; integrity, segments, and diagnostics at the rotating-frame bracket ends;
  integrity on the refined lab-frame runs. All checks with pass/fail outcomes passed;
  diagnostics were recorded.

| `EJ` (GHz) | Width (GHz) | Rotating-frame bracket (GHz) | Refined lab minus rotating-frame loss | Lab-frame confirmation |
|---|---|---|---|---|
| 13.5 | `1.25e-4` | 0.001750–0.001875 | `7.4e-5` to `9.5e-5` | Confirmed |
| 13.5 | `1.5625e-5` | 0.001796875–0.001812500 | `8.5e-5` to `8.7e-5` | Not confirmed: the lower end is `3.6e-6` above the target |
| 15 | `1.25e-4` | 0.001375–0.001500 | `2.8e-5` to `5.1e-5` | Confirmed |
| 15 | `1.5625e-5` | 0.001437500–0.001453125 | `3.7e-5` to `4.0e-5` | Confirmed |

At `EJ = 12` the largest sampled loss is `0.0039`, so no crossing lies in range under
the monotonic assumption. The two lab-frame runs differ in loss by at most `2.4e-7`.

Job time is about 1.1 s for a rotating-frame point, 6.6 s for a lab-frame run, 20 s
for a refined lab-frame run, and 0.8 s for a check run. The probe's full run, both
widths at both crossing `EJ` values, is 52 jobs and 41 evaluated points in 301 s of
job time within 329 s elapsed. The agreed width alone accounts for 30 jobs, 27 points,
and 176 s of job time. Establishing both agreed-width brackets and then stopping at
the first failed tighter confirmation accounts for 41 jobs, 34 points, and 240 s.
The last two figures are sums over the probe's runs; the probe did not run those
shorter sequences separately.

The stated limits are:

- Coherence is assumed to decrease with amplitude at fixed `EJ`, which gives one
  crossing per line. The declared expectation checks this at sampled amplitudes only.
- The bracket width describes the rotating-frame crossing. Lab-frame confirmation is
  reported separately.
- The change between the two lab-frame runs is observed convergence evidence, not an
  error bound. No check bounds whole-window error.
- The failed confirmation at `EJ = 13.5` rests on a `3.6e-6` margin: 19 times the
  observed refinement change and about four times the `1e-6` cross-engine agreement
  above.
- When confirmation fails, more rotating-frame bisection cannot settle the bracket,
  and a lab-frame search is the next experiment. A lab-frame run costs six times a
  rotating-frame run here, so the failure does not show that the search would be
  uneconomic.

### Scripted investigation baseline

The client drafts five bases: rotating frame, lab frame, and one refined lab-frame
base per `EJ`, binding each embedded state to that value. It waits for an operator
start; the client never approves, starts, releases a review, or resumes an
investigation itself. Declare the checkpoint and, optionally, a job-count limit.
On the workstation, preserve the shared runner extras profile:

```sh
export ENTANGLIX_RUNNER_EXTRAS=cuquantum,differentiable-cuda12
uv run --locked --exact python examples/readout/investigate.py draft research_runs/threshold-client --runs-root research_runs/threshold --case normal --checkpoint 10 --job-count 60
INVESTIGATION=$(uv run --locked --exact python -c 'import json; print(json.load(open("research_runs/threshold-client/client.json"))["investigation"])')
uv run --locked --exact entanglix read-investigation "$INVESTIGATION"
uv run --locked --exact entanglix investigation-decision "$INVESTIGATION" start --revision 1
uv run --locked --exact python examples/readout/investigate.py run research_runs/threshold-client
```

See the [protocol reference](../../docs/operations/protocols.md#gpu-configuration-and-checks)
for runner configuration.
Omit `--job-count` for no investigation limit. Use a new workspace and runs root
with `--case early-stop --job-count 100` for the tighter request. The checkpoint of
10 and limits of 60/100 are these evaluations' explicit declarations, not harness
defaults.

The draft terms state the normal `1.25e-4 GHz` resolution and, for the tighter
case, the subsequent `1.5625e-5 GHz` request. The selection policy runs the seed,
bisects each crossing to `1.25e-4 GHz`, and
confirms both normal-width brackets before any tighter search. The tighter case
then bisects in ascending `EJ` order to `1.5625e-5 GHz`, stopping at the first failed
lab confirmation. It preserves the confirmed normal results, opens the named
`precision` review, proposes a lab-frame search, and marks remaining tighter work
unattempted. It does not extend the range or consume the remaining allowance.
Bayesian search adds nothing to this one-dimensional midpoint policy.

Check requests on the seed and final rotating-frame endpoints are conditional on
their role in this procedure, recorded in the evidence requirements. Integrity is
also a standing request on every refined lab base. Intermediate rotating-frame
points and coarse lab runs are retained without additional checks; findings list
relevant checks not run. Loss differences and brackets are client interpretations,
not trusted metrics: reports retain baseline and endpoint values with source IDs.

[Normal acceptance](../../tests/probes/m3_investigation_normal.json) and
[early-stop acceptance](../../tests/probes/m3_investigation_early_stop.json) record
the real investigation IDs, source runs, checks, job time and elapsed time. The
[acceptance driver](../../tests/probes/m3_investigation.py) acts as the operator for
these declared fixtures; [commands](../../docs/environment.md#scripted-investigation-acceptance)
regenerate the summaries.

| Case | Outcome | Jobs / points | Job time / elapsed | Assessments | Decisions after start |
|---|---|---|---|---|---|
| Normal | Both normal brackets confirmed | 30 / 27 | 178 s / 195 s | 2 | 0 |
| Early stop | Precision review at `EJ = 13.5` | 41 / 34 | 243 s / 267 s | 4 | 0 |

These are separately executed baselines. Elapsed time runs from the operator start
to final report preparation; summaries separate computation, checks, and time
outside jobs. Both reproduce the probe's normal brackets. The tighter `EJ = 13.5`
confirmation fails because the refined lab loss at the lower end is `3.6e-6` above
the target; `EJ = 15`'s tighter case stays unattempted. All requested checks with
pass/fail outcomes passed, and diagnostics are retained. The 100-job limit still
has 59 jobs available at handoff.

Continue with the same `run` command after a session ends. The workspace stores
base files and a locator; progress is recovered from retained member runs and
findings. Completed simulations and matching checks under the current policy are
reused. A failed or aborted attempt is retried under a new linked ID and counts
again. An outstanding attempt must first finish, or be aborted and reconciled
through the [run operations](../../docs/operations/runs.md). An investigation stop
requires the operator to release its reviews and resume it. The
[session tests](../../tests/test_readout_investigation.py) cover a killed client
between jobs and an interrupt during a solve. Jobs are blocking in this client;
it does not supply connection-independent supervision.

## Agent investigation comparison

The [evaluation driver](../../tests/probes/m3_evaluation.py) runs fresh research
sessions with the shipped conduct, procedures and syntax references. Formulation
runs as a separate session. Its prompt asks for a bounded next investigation and
the needed operator decisions, forbids computation, and explicitly cautions against
inferring readout quality from coherence alone. The session adds a proposed
fixed-`EJ = 13.5` comparison of three amplitudes across RWA, lab and larger-basis lab
inputs, with nine simulations, a proposed stability target and bounded resource
and review declarations. It launches no jobs. The retained proposal is included
in the results; this is a prompted formulation example, not an independently
discovered research objective.

The comparison terms are fixed in advance, independently of formulation. Each agent
drafts exactly the baseline's terms; the driver approves that fixture through the
operator CLI. Inputs and runs are separate from the baseline, and every solve,
check, assessment, report and stop goes through the real MCP server. Raw
transcripts retain the initial guidance and shell reads; the
[results](../../tests/probes/m3_evaluation_results.json) record the client/model,
source identities, source-bound claims, checks, decisions and stopping rationales.
The execution brief prescribes midpoint search, both bracket widths, which runs
to check, the lab confirmation rule, and the order of normal and tighter cases.
Numerical search is fixed by those agreed terms. Agent reasoning appears in the
`EJ = 12` no-crossing report, checkpoint assessments, stop decision and handoff;
this comparison does not measure the invention of a search algorithm.

| Case | Outcome | Jobs / points | Job time / elapsed | Assessments | Input tokens | 5-hour / weekly used % |
|---|---|---|---|---|---|---|
| Formulation | Bounded proposal; no compute | 0 / 0 | — | — | 188,317 | 32→34 / 6→6 |
| Fresh normal baseline | Both normal brackets confirmed | 30 / 27 | 184 s / 202 s | 2 | — | — |
| Normal agent 1 | Both normal brackets confirmed | 30 / 27 | 190 s / 330 s | 3 | 2,779,389 | 35→47 / 6→8 |
| Normal agent 2 | Both normal brackets confirmed | 30 / 27 | 187 s / 391 s | 3 | 2,735,366 | 47→61 / 8→10 |
| Normal agent 3 | Both normal brackets confirmed | 30 / 27 | 189 s / 349 s | 3 | 2,164,533 | 61→72 / 10→12 |
| Fresh early-stop baseline | Precision handoff at `EJ = 13.5` | 41 / 34 | 251 s / 277 s | 4 | — | — |
| Early-stop agent 1 | Precision handoff at `EJ = 13.5` | 41 / 34 | 254 s / 511 s | 4 | 3,407,468 | 72→85 / 12→14 |
| Early-stop agent 2 | Precision handoff at `EJ = 13.5` | 41 / 34 | 253 s / 464 s | 4 | 2,068,986 | 86→97 / 14→16 |
| Early-stop agent 3 | Runtime usage limit before compute | 0 / 0 | 0 s / unavailable | 0 | 304,891 | 97→100 / 16→16 |

The five completed agent investigations reproduce the baseline's source-supported
brackets and handoff. They retain the no-crossing finding at `EJ = 12`, pass the
requested checks, preserve both normal-width results, and leave tighter `EJ = 15`
unattempted after failed lab confirmation. All have zero operator decisions after
start. The third early-stop session exhausted its Codex account allowance after
drafting; its execution request also failed, without admitting a job. It has no
scientific outcome or final report. The owner-approved acceptance rule requires
successful formulation and at least two accepted sessions per case. Other sessions
must be reported runtime failures before any admitted job; a failure after admission
still fails the evaluation. This comparison passes with three accepted normal
sessions, two accepted early-stop sessions and one reported pre-compute runtime
failure.

Together, formulation, drafting and execution used 13,648,950 input tokens
(12,990,976 cached), 81,361 output tokens and 177 model requests. Resumed phases
subtract earlier thread totals, so drafting is counted once. The 5-hour window
already had 32% used at the first request and reached 100%: a 68-percentage-point
observed increase; the weekly window went from 6% to 16%. These are rounded
runtime-reported account windows, not a price estimate or an isolated attribution
of shared-account usage.

These measurements use non-interactive Codex 0.160.0, `gpt-6.1-sol` at high effort,
on this workstation. Completed agents use about the same job time and more elapsed
time than the scripted policy. Agents batch independent CPU jobs while the baseline
is serial; summed job time therefore differs from occupied elapsed time. The
summary measures time outside jobs by subtracting the union of execution intervals,
without counting overlapping jobs twice. It does not identify that entire interval
as model latency. Elapsed time covers operator start through investigation stop;
formulation and drafting are recorded separately. This fixture demonstrates
evidence-based decisions and reporting, with no acceleration claim or general
runtime success-rate claim.

A separate [pick-up check](../../tests/probes/m3_pickup_results.json) on 2026-10-09
killed the scripted normal client with SIGKILL between jobs after 12 attempts.
One fresh Codex 0.160.1 session (`gpt-6.1-sol`, high effort) read the retained
records and completed 18 more jobs through MCP `simulate` and `check`, reaching
30 jobs and 27 points without repeated points or checks. It first issued the
missing check on an existing refined run, confirmed the same brackets, cited all
30 runs, and stopped with `evidence_completion`. The session used 1,172,667 input
tokens (1,095,680 cached), 18 model requests and about 250 seconds; its 5-hour
window went from 0% to 4%. Six requests and about one window percentage point
were spent before requesting its first new job. This is one session interrupted
between jobs, with earlier records written by a script and a prompt that named
those records and prescribed the method. It does not demonstrate recovery during
an executing job or discovery of unmentioned earlier work. The
[summary probe](../../tests/probes/m3_pickup.py) rebuilds this evidence without
launching a session.

Four further [pick-up checks](../../tests/probes/m3_pickup_claude_results.json) on
2026-10-10 used Claude Code 2.1.296 with `claude-sonnet-5-5`. The scripted normal
client was killed during its 25th job, a lab run at `EJ = 15`, which left that
attempt interrupted beside 24 finished jobs. Each session was asked only to
continue the investigation and given the method brief; the prompt named neither
the earlier session nor the interrupted attempt.

| Session | Conduct | Interrupted attempt | Repeated work | Report cites | Requests | 5-hour used % |
|---|---|---|---|---|---|---|
| First | 0.1.1 | Not reconciled; point rerun with no link | One check | 28 of 32 attempts | 39 | 19→23 |
| Second | 0.1.2 | Reconciled; rerun linked to it | None | 42 of 42 attempts | 52 | 27→32 |
| Third | 0.1.4 | Reconciled; rerun linked to it | None | 31 of 31 attempts, in a finding of the wrong kind | 38 | 9→12 |
| Fourth | 0.1.6 | Reconciled; rerun linked to it | None | 31 of 31 attempts | 34 | 25→28 |

All four sessions confirmed the baseline's two normal brackets and launched every
new job through MCP. The first skipped recovery: the conduct did not say how to
settle an interrupted attempt, and its first read asked for the full record,
which then lacked attempt metrics and check links. Both were changed before the
second session, which reconciled the attempt, reran it with `rerun_simulation`,
and recorded an assessment after a check was refused at a due checkpoint. It then
continued into the tighter stage, which the normal terms do not request, and
ended with the precision handoff at `EJ = 13.5` after 17 new jobs where six were
needed; the brief it received describes both cases. Claude Code also rejected its
full investigation read of 54,724 characters as over the tool-output limit, and
accepted the default summary. The brief was then written per case and the single
investigation read replaced by a status with an attention list and paged tables.
The third session found the interrupted attempt on that list, ran the six missing
jobs and no others, and stopped with `evidence_completion`; no reply was rejected.
It recorded its final report as an `interpretation` finding, so the summary's
report criterion is not met; the conduct then did not name the kind to use. With
the kind named, the fourth session did the same six jobs, recorded a `report`
finding, and met every criterion. These are four sessions on one runtime and one
cut-off point, with no shell tool granted. The account windows are shared with
other use. The [summary probe](../../tests/probes/m3_pickup_claude.py) rebuilds
all four without launching a session.
