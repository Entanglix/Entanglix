# Calibration

Calibration fits device parameters to measured spectroscopy. This reference describes measurement datasets, with the format every measurement is converted to, validation, digests, raw-file checks, and holdout splits; peak extraction from maps; the `calibrate` protocol: its spec, assignment, fit, outcomes and report; the checks a calibration run takes on request; device-update proposals and their application; and calibration in investigations, reruns and research tools.

## Measurement datasets

A dataset is a TOML file and a numeric NPZ archive beside it:

```toml
format_version = 1
id = "two-tone-cooldown-3"
kind = "spectroscopy"
array_file = "two-tone.npz"

[conditions]
drive_power = { value = -30, unit = "dBm" }
readout_frequency = { value = 7.12, unit = "GHz" }
averages = 1000
temperature = { value = 12, unit = "mK" }

[provenance]
device = "chip-A"
cooldown = "CD-3"
measured_at = 2026-10-01T12:00:00Z
raw_file_sha256 = "<sha256 of the laboratory file>"
converter = { name = "lab-to-entanglix", version = "0.2" }

[map]
signal = { key = "signal", unit = "V" }
frequency = { key = "frequency", unit = "GHz" }
bias = { key = "bias", unit = "mA" }

[points]
centre = { key = "centre", unit = "GHz" }
centre_uncertainty = { key = "centre_sigma", unit = "GHz" }
bias = { key = "point_bias", unit = "mA" }
linewidth = { key = "hwhm", unit = "GHz" }
amplitude = { key = "amplitude", unit = "V" }
assignment = { key = "label" }
```

`kind = "spectroscopy"` is the only measurement kind. `[conditions]` names the conditions that decide which lines appear, such as drive and readout power and frequency, averaging, and temperature. Each entry is a `{ value, unit }` quantity, an integer count, or text; a bare decimal number is refused because it has no unit. `[provenance]` records the device, an optional cooldown, the measurement time as a TOML datetime with a UTC offset, the optional SHA-256 of the raw laboratory file, and the converter's name and version.

A dataset holds a map, points, or both:
- **Map:** `signal` is a real or complex array with shape `(bias, frequency)`, or `(frequency,)` for data at one fixed bias. `frequency` is in GHz and strictly increasing. The optional `bias` axis is strictly increasing in its declared unit.
- **Points:** one-dimensional arrays of equal length. `centre` and `centre_uncertainty` are required, in GHz, and the uncertainty is positive. `bias`, `linewidth` (half width at half maximum, GHz, positive), `amplitude`, and `assignment` are optional. Assignments are text labels, with an empty label for an unlabelled point; whether a calibration can use a dataset's fields and labels is decided when the calibration launches.

The map and the points share one bias axis with one unit, or neither has one. Every array is finite. The archive loads without pickle, so object arrays are refused. Each role names its own NPZ key, and every array in the archive must be referenced.

Conversion from a laboratory format happens outside Entanglix and is mechanical: axes, signal, units, and metadata. Peak selection and other scientific preprocessing belong to the calibration's declared settings, never to a converter. Entanglix records the converter's provenance but cannot verify the conversion.

## Validation and digests

Run `entanglix validate-dataset path/to/dataset.toml`. Success prints a JSON report with the dataset ID, kind, digest, bias unit, provenance, the statement that the conversion is recorded by its converter and not verified by Entanglix, the raw-file status, the map shape, signal type and frequency range, the point count, the number of labelled points and the fields present, and any split record. Invalid input prints a diagnostic to stderr and exits with code `64`.

With `--raw-file path/to/raw`, the file's SHA-256 must equal `provenance.raw_file_sha256`. A mismatch, or a raw file supplied for a dataset that declares no digest, is refused. Without `--raw-file`, the report states that no raw file was supplied.

The dataset digest is the SHA-256 of the length-prefixed TOML bytes followed by the length-prefixed NPZ bytes, so copies with identical bytes share it whatever their file names. The Python operation `entanglix.measurements.load_dataset(path, raw_file=None)` returns a `Dataset` with the validated specification, the arrays by role, the exact file bytes that were validated, and the digest; `dataset_report(dataset)` returns the report.

## Holdout splits

Split holdout data off before any fitting, so that a later holdout evaluation uses data no fit has seen:

```sh
entanglix split-dataset two-tone.toml --lower 0.2 --upper 0.5 --holdout holdout.toml --remainder training.toml
```

The range is closed and in the bias unit. Map columns and points whose bias lies inside it form the holdout part; the rest form the remainder. Each part is a new dataset, written with an NPZ archive of the same name stem, an ID of `<id>-holdout` or `<id>-remainder`, and the parent's conditions and provenance. Its `[split]` table records the parent's digest, the part, the range, and the kept map columns and points as `[start, stop)` index ranges in the parent. An interior range leaves a remainder of two intervals, recorded exactly.

A level with no entries inside a part, such as sparse points outside the range, is left out of that part. A split is refused for a dataset without a bias axis, for a part that would be empty, when an output file exists or its directory is missing, and for a dataset that is itself a split part. Both parts are written or neither: files written before a failure are removed. Loading a part checks that its recorded index counts match its arrays; it does not compare the values with the parent's.

## Peak extraction

Extraction turns a map into peaks under declared settings. Every setting is required:
- `background_window_ghz`: the width of the running median along frequency that is subtracted from each column's I and Q.
- `threshold`: the height and prominence a maximum needs, in units of its column's noise.
- `min_separation_ghz`: detected maxima closer than this are an unresolved overlap.
- `fit_half_window_ghz`: the half width of each line-fit window.
- `linewidth_bounds_ghz = [lower, upper]`: bounds on the fitted half width at half maximum.
- `line_shape`: `"lorentzian"` or `"gaussian"`.

The map needs a uniform frequency grid of at least three points. The projection is the principal axis of the background-removed I/Q signal, signed so that the largest excursion is positive. It is learned from the map being extracted unless one is given, so a projection learned from training data can be applied to other data. Each column's noise is 1.4826 times the median absolute deviation of its projected signal. A maximum is detected when its height and prominence both reach the threshold.

Each detected maximum that is separated from its neighbours is fitted with every detected line in its window and a constant offset, with residuals in units of the column's noise, the centre bounded to the window, and a non-negative amplitude. The centre's uncertainty comes from the fit's Jacobian, so it is conditional on the noise estimate and the declared line shape; a fit whose Jacobian does not determine every parameter in its window reports no uncertainty. Each peak reports its centre, the centre's uncertainty, its linewidth, and its amplitude.

Failures are reported per column: `no noise estimate` for a constant column, `no peak`, `unresolved overlap` with the frequencies of the maxima involved, and `line fit failed` with the maximum's frequency and a detail: a rank-deficient fit, an optimizer that did not converge, or a non-finite uncertainty.

The Python operation `entanglix.extraction.extract(dataset, settings, projection=None)` returns an `Extraction` with the projection used, the peaks as arrays (`column`, `bias` when the map has a bias axis, `centre`, `centre_uncertainty`, `linewidth`, `amplitude`), the failures, and each column's noise. `learn_projection(dataset, settings)` returns the projection alone.

## The calibrate protocol

`entanglix calibrate calibrate.toml` fits the named system parameters to training datasets as one CPU job. `--dry-run` validates every input and prints the preview without creating an attempt; `--runs-root`, `--parent-run` and `--detach` behave as for the other jobs. A refused launch prints its reason, exits with code `64`, and creates no attempt.

```toml
format_version = 1
system = "system.toml"
datasets = ["training.toml"]
merge_tolerance = 1e-3
timeout_seconds = 3600          # optional

[parameters]
EJ1 = { scale = 1.0 }
EJ2 = { scale = 1.0 }
Csh = { scale = 10.0 }

[axis_map]
flux = "phi"
offset = { lower = -0.5, upper = 0.5, scale = 0.01 }
period = { lower = 0.5, upper = 2.0, scale = 0.01 }

[family]
initial = [0, 1]
max_level = 3
photons = [1, 2]

[matching]
anchor = "0-1"
range_ghz = 0.3
cutoff = 5.0
resolution = 2.0
range_iterations = 20
cutoff_iterations = 20

[extraction]                    # maps only; see Peak extraction
background_window_ghz = 0.1
threshold = 6.0
min_separation_ghz = 0.01
fit_half_window_ghz = 0.02
linewidth_bounds_ghz = [2e-4, 0.02]
line_shape = "lorentzian"

[conventions]
offset_window = -0.5
exchange = [["EJ1", "EJ2"]]

[[starts]]
EJ1 = 12.4
EJ2 = 7.7
Csh = 72.0
offset = 0.13
period = 1.03
```

The system and the datasets are files beside the spec. Alternatively, replace `system` with `system_entry` to use a promoted [catalog device](catalog.md#reusing-a-device) from the configured runs root; declare exactly one. Its model files are frozen with the run, while the datasets remain beside the spec. An optional `[refinement.<subsystem>]` table, in the simulation refinement form (`retained_levels_step`, `cutoff_steps`), is frozen with the run; the fit does not use it. The harness sets no defaults for the matching settings, scales, starts or merge tolerance; the values above are examples from the synthetic two-junction transmon of the [calibration example](../../examples/transmon-calibration/README.md).

### Parameters and the axis map

Each key of `[parameters]` names a parameter in the system file's `[parameters]` table and gives its `scale`, in the parameter's unit. The parameter must carry a uniform prior, whose bounds bound the fit, and must enter construction: a name that no field of `[model]` references is refused, so operator models have no fittable parameters.

`[axis_map]` binds the datasets' bias axis to the flux parameter `flux`, which must be in `Phi0`, enter construction, and not be fitted directly. At bias `V` the flux is `(V - offset) / period`. The offset and the period are nuisance parameters with declared bounds and scales, in the bias unit; they are fitted and reported but are not device parameters. The period's lower bound must be positive, which removes the negative-period equivalent. Every dataset with a bias axis needs the axis map, and they share one bias unit; an axis map without such a dataset is refused. A dataset without a bias axis is evaluated at the system's own flux value, and a flux parameter may then be fitted directly.

### Candidate family

The family lists every transition the fit may predict. `initial`, with `final` or `max_level`, generates each transition from an initial level to a higher final level, at each of the `photons` orders (default `[1]`). A generated member is labelled `i-f`, or `i-f/k` for `k` photons, and its frequency is `(E_f - E_i) / k`, with levels numbered by energy order at each bias. `[family.lines.<name>]` adds a named line with `initial`, `final` and an optional `photons`: levels, or bare product states such as `[1, 0]`, one level per subsystem. Bare states are matched one to one to the dressed states they overlap most at each evaluation. Names must not take the generated form, and members must be distinct transitions. Every level must lie within the system's dimension.

### Procedure

With `[extraction]`, every training dataset's map is extracted under the declared settings, with the projection learned from that map; points stored in the dataset are not used. Without it, every dataset's points are used.

When every training point carries an assignment, the labels are used as given: each must name a family member, and the spec must declare no `[matching]`. A partly labelled input is refused. Otherwise assignment is automatic and needs `[matching]`, each point's linewidth, and, with an anchor, each point's amplitude; extracted peaks carry both.

Automatic assignment matches the peaks of each bias column one to one with the family's lines. Matching a peak costs its squared distance in units of its centre uncertainty; an unmatched peak costs the phase's penalty. A peak within `resolution` linewidths of two lines is unresolved: it is left out of the fit and costs the same penalty. The total cost ranks assignments and solutions; it is not guaranteed to fall from one iteration to the next. From each start:

1. **Anchor**, when `anchor` names a member: each column's largest-amplitude peak is taken to be that line, and it alone is matched and fitted through the two phases below.
2. **Range:** the family is matched with a penalty of `range_ghz` in units of each peak's uncertainty, and refitted until the assignment stops changing, at most `range_iterations` fits.
3. **Cutoff:** the same with a penalty of `cutoff` uncertainties, at most `cutoff_iterations` fits.

The fit is bounded weighted least squares (SciPy's trust-region reflective method) on the assigned peaks, with residuals in units of each peak's centre uncertainty, the declared scales as step scaling, and finite-difference derivatives. Every evaluation constructs the full system at each bias column through the same construction as `validate`. With supplied labels there is one fit per start.

A start ends without a solution, and with its reason, when an anchor phase assigns no peak (`anchor initialization failed under the declared search`, with the anchor's counts), when a later phase assigns none, when a phase reaches its iteration limit (`non-converged`), when the optimizer does not converge, or when construction fails.

### Conventions and solutions

Declared conventions map each solution to one representative: `exchange` orders each pair so that the first is not below the second, and `offset_window = w` wraps the offset into `[w * period, (w + 1) * period)`. An exchanged pair needs equal units, priors and scales. A convention is applied to a converged point only when the model, at the mapped values, reproduces every candidate line of every bias column with peaks within `equivalence_rtol` of the largest line; the assignment and cost are then recomputed at the mapped values. Otherwise the point keeps its fitted values and the convention is reported as not applied, with the largest change it would make to a line. This is a check at each solution, not symmetry discovery. Solutions from different starts merge when their assignments are identical and their values agree within `merge_tolerance` in units of the scales.

### Outcomes

- `solution`: one distinct solution, with the parameters numerically determined.
- `several distinct solutions`: more than one, ordered by cost.
- `insufficient evidence`: every solution has fewer assigned peaks than fitted parameters, or a rank-deficient Jacobian.
- `anchor initialization failed under the declared search`: every start failed in its anchor phase.
- `no fit`: no start returned a solution.

None of these is a pass or a fail. A search finds only the solutions reachable from its starts; a line outside the family that coincides with a predicted line is not recognised; and the absence of an anchor failure does not establish that the anchor was right.

### Report

`result.toml` records the outcome, the procedure, the fitted and nuisance parameters, the protocol's tolerances, each dataset's digest and peak count (with the projection and the number of extraction failures for maps), each start's status, cycles per phase, anchor counts, bounds hit, the equivalents removed and the conventions not applied, and each distinct solution:
- `values`, `objective` (the cost at the published values), the `starts` that reached it, `equivalents_removed`, `conventions_not_applied` and `bounds_hit`: the parameters that end within one Jacobian step of a bound;
- `counts` of assigned, unexplained and unresolved peaks and of missing lines, and the peaks per line;
- `statistics`: `chi2`, the number of assigned peaks, the number of fitted parameters, and `reduced_chi2`, absent when there are no more peaks than parameters;
- `rank` and `singular_values` of the weighted Jacobian in scaled coordinates (each parameter divided by its scale), and `undetermined_combinations` as unit vectors in those coordinates;
- `determined` parameters, with `uncertainty`, `uncertainty_over_prior_width`, `covariance` and `correlation` for those only. A parameter with a component above the undetermined-direction tolerance along an undetermined combination has no uncertainty;
- `largest_marginal_separation` from the best solution, for the others: the largest difference of a parameter over the two solutions' combined uncertainties.

The covariance is unscaled, a local linear approximation conditional on the assignment and on the centre uncertainties. The ratio to the prior width is descriptive; a large ratio means the parameter is weakly constrained over its declared range. The tolerances are `rank_rtol = 1e-6` (singular values below this fraction of the largest are not counted in the rank), `undetermined_tolerance = 1e-3`, `jacobian_step = 1e-5` (the finite-difference step, in units of the smaller of each parameter's scale and the width of its bounds), `equivalence_rtol = 1e-8` and the declared `merge_tolerance`; every result records them. The Jacobian is sampled strictly inside the bounds: central differences where both neighbours fit, otherwise second-order one-sided differences toward the farther bound. A step is at most `jacobian_step` of its interval, so the bounds themselves are never sampled.

`artifacts/assignment.toml` lists the bias columns, every peak with its column, centre, uncertainty, and linewidth and amplitude when known, the extraction failures, and for each solution the assignment of every peak (a member label, `unexplained` or `unresolved`) and the missing lines: family lines predicted inside a map's frequency range in a column where no peak was assigned to them. Points have no frequency range, so no line is reported missing for them.

### Refused at launch

A fitted name that is not in the system's parameter table, has no prior, or enters no construction; an axis-map flux that is fitted, not in `Phi0`, or unused; a dataset whose level the procedure needs is absent; a map without a uniform frequency grid of at least three points, when it is to be extracted; a bias axis without an axis map, or the reverse; bias units that differ; a label outside the family; a partly labelled input; `[matching]` with supplied labels, or none without them; an anchor outside the family; a point missing a field its procedure uses; a named line called `unexplained` or `unresolved`; a level beyond the system's dimension; a start without exactly the fitted and nuisance names or outside the bounds; and inconsistent conventions.

## Checks

A `calibrate` launch, its `--dry-run` preview, and `check RUN` or `read-run` on the run list `holdout`, `identifiability` and `refinement` with their availability and the holdout forms the run's procedure can read; a completed run's listing adds the number of solutions. Availability annotates the listing and never refuses the calibration: refinement is listed as unavailable without a declaration, or with one that does not meet the rules below. Named checks run as one verification attempt linked to the run, as for any run; see [run records](runs.md#corrector-policy-and-verification-records). Each check reports on one solution. When the run returned several, `--solution N` names one by its index in the result's `solutions` list, counting from 0, and a request without it is refused. Every check is inapplicable when the run returned no solution. None has a pass or fail outcome unless the user declares one, as below.

```sh
entanglix check RUN holdout identifiability --holdout holdout.toml --expect-coverage 2 0.9
entanglix check RUN refinement --solution 1 --refinement refinement.toml
```

### Holdout

The holdout check evaluates withheld data at the fitted parameters. `--holdout` supplies the dataset: a holdout part from `split-dataset`, or a separate measurement. The verification attempt freezes it under `holdout/` with its digest; the fitting run never holds it. Without `--holdout` the check is inapplicable.

The check refuses a dataset that is one of the run's training datasets, the parent of one, a split part of one, or a part of the same parent whose kept indices overlap one. It also refuses a partly labelled dataset, a label outside the family, a bias axis without the run's axis map or in another unit, a map to be extracted without a uniform frequency grid, and unlabelled points without linewidths.

The check uses only the procedure the run recorded:
- **A map,** when the run declared `[extraction]`, is extracted under the run's settings, with the projection the run learned from its training map; points stored beside it are not used. When several training maps learned different projections, the check is inapplicable.
- **Labelled points** keep their labels.
- **Unlabelled points,** when the run used automatic assignment, are matched one to one with the family's lines in each column. The run's `cutoff` is the unmatched penalty and its `resolution` decides unresolved peaks, with distances in units of each line's prediction uncertainty. The anchor is not used.

A map after a run without extraction settings, and unlabelled points after a fit to supplied labels, are inapplicable.

Nothing is refitted, and nothing is learned from the holdout data. A line's prediction uncertainty combines the peak's centre uncertainty with the parameter covariance propagated linearly through the prediction, axis-map parameters included. The covariance is recomputed from the training residuals at the fitted values, along the determined directions, with the corrector's tolerances. A prediction that moves along an undetermined combination is not determined: its peak counts as assigned but is not scored.

The report gives:
- the dataset's file, ID and digest, and the form read;
- `counts`: peaks, assigned, `not_determined`, unresolved, unexplained and scored, with `extraction_failures` for a map;
- `coverage_over_peaks` and `coverage_over_scored`: the fractions of all peaks and of scored peaks within one and two prediction uncertainties (`within_1`, `within_2`). Unexplained, unresolved and undetermined peaks count against coverage over peaks;
- `mean_squared_standardised_residual` of the scored peaks;
- the bias columns, and each peak with its line, prediction, prediction uncertainty and standardised residual.

`--expect-coverage WITHIN MINIMUM` declares an expectation: coverage over all peaks within `WITHIN` prediction uncertainties, 1 or 2, is at least `MINIMUM`, a fraction in (0, 1]. The check then passes or fails, with the coverage as `residual` and the minimum as `tolerance`.

### Identifiability

The identifiability check recomputes, at the solution's values, the fit statistics, rank, singular values, undetermined combinations, uncertainties, prior-width ratios, covariance and correlations. It uses the frozen training data, the run's final assignment and the corrector's tolerances. The training peaks are rebuilt from the frozen datasets and must equal the run's recorded peaks; otherwise the check attempt fails.

The report gives both sets of tolerances and the agreement with the fit's own report:
- whether the outcome, the rank and the determined parameters match;
- the largest relative difference of an uncertainty, the largest difference of a correlation, and the difference in χ²;
- for a rank-deficient fit, `undetermined_difference`: the sine of the largest angle between the two sets of undetermined combinations.

`agrees` is true when all three match and every difference is within `agreement_rtol`; the χ² difference is taken relative to the fit's χ², or absolutely when that is below 1. The check is descriptive.

### Refinement

The refinement check predicts the assigned training peaks at a refined truncation, at the solution's values. The declaration is the run's `[refinement.<subsystem>]` tables, or a replacement given with `--refinement FILE`. It must cover exactly the truncated subsystems and meet the policy's minimum steps, as for simulations; without a declaration, or with one that does not meet these rules, the check is inapplicable.

The report gives the dimensions before and after, the number of assigned peaks, the largest change of a prediction in GHz and in units of its peak's centre uncertainty, the largest standardised change for each line, and χ² at both truncations. It is descriptive: truncation error that the fit absorbed into its parameters shows as a change against the data's own uncertainty.

### Corrector tolerances

The corrector policy's `[calibration]` table holds `rank_rtol = 1e-6`, `undetermined_tolerance = 1e-3` and `jacobian_step = 1e-5`, used by the identifiability recomputation and by the holdout prediction uncertainty, and `agreement_rtol = 1e-6`. The fit records its own tolerances, so a policy change explains a difference between the two reports.

## Device-update proposals

`propose-device-update RUN` writes a proposal from one solution of a completed calibrate run. When the run returned several solutions, `--solution N` names one, as for checks. A run with no solution, and a solution that determines no fitted parameter, have nothing to propose and are refused. The proposal is published read-only as `proposals/<n>.toml` in the run's directory, numbered from 1; a later proposal from the same run, for example after further checks, takes the next number.

```sh
entanglix propose-device-update RUN --solution 0
entanglix apply-device-update RUN/proposals/1.toml system.toml
```

The proposal holds:
- `source_run_id`, `system_id`, `solution`, its `solution_outcome` and `bounds_hit`, and `source_result_sha256`;
- `model_inputs`: the digest of every model input the run froze, namely the system file, each subsystem source and each array file;
- `[parameters.<name>]` for each fitted parameter the solution determines: `unit`, `old` (the value in the run's frozen system file), `new`, `uncertainty` and `uncertainty_over_prior_width`, as the fit reported them;
- `[not_proposed.<name>]` for each fitted parameter the solution does not determine, with its `reason`;
- `[[verification]]`: each check attempt on the run when the proposal was written, with its `run_id`, `status`, `policy_version`, `source_matches`, and each check's `name`, `solution` and `outcome`. The outcome is `pass`, `fail` or `inapplicable`, or `descriptive` for a check that reports values without a verdict.

Only fitted device parameters from the system file's parameter table are proposed. The axis map's offset, period and flux parameter are not. Nuisance parameters, the covariance and the correlations stay in the run's result.

### Applying a proposal

`apply-device-update PROPOSAL SYSTEM` is an operator action. It writes each proposed parameter's `new` number into its `value`, and the provenance into its `source`, in the authored system file. Units, priors and every other field are unchanged. It refuses:
- a system file, or a subsystem source or array file beside it, that is missing or whose digest differs from `model_inputs`, so an edit these checks find is never overwritten;
- a proposed parameter whose value or unit in the run's frozen system file differs from the proposal's `old` and `unit`;
- a system file inside the runs root, such as the run's frozen copy;
- a proposal outside its run's `proposals/` directory, or one already applied;
- an apply while another apply holds the system file's folder or the proposal's folder.

Every model input is checked again after the update is validated and immediately before the file is replaced, so an edit made during validation is refused rather than overwritten. Apply's lock does not stop other programs: an edit made between this last check and the replacement is not detected.

The file is rewritten as canonical TOML: comments and the original layout are not kept, and the parsed content changes only in the applied values and sources. The source reads `calibration proposal <system_id>/<run_id>/proposals/<n>.toml, solution <N>, sha256 <proposal digest>`. Apply then publishes `proposals/<n>.applied.toml` with the time, `source = { kind = "operator-cli", os_user = ... }`, the proposal's digest, the system file's path and its digests before and after, and the names of the parameters applied.

The new content is built from the run's frozen system file, so it is the same on every attempt. If the file was replaced but the application record could not be published, apply reports it. Applying the same proposal to the same file again then publishes the record with `recovered = true`, provided the file still holds exactly the applied content.

Research tools can stage proposals but never apply them; the research tool surface has no apply operation.

## Investigations and reruns

`calibrate` is an investigation operation. Launched with `--investigation ID --base NAME`, a calibration is admitted only when its inputs match the base exactly: the spec under the fixed name `calibrate.toml`, the system file as `system.toml` with its subsystem sources and array files, and each training dataset and its array file under its own name. Calibration takes no varied values, and a changed input is refused with the base's digest differences. The run also freezes the authored files under `spec/base/`. Checks on the run belong to the investigation; a holdout dataset supplied with a check is not a base input.

`rerun-calibration RUN` re-fits the frozen inputs as a new attempt linked to `RUN`. An investigation rerun uses the authored base files, keeps membership, and passes admission under the current approved revision.

```sh
entanglix calibrate fit.toml --runs-root research_runs --investigation <investigation_id> --base fit
entanglix rerun-calibration research_runs/<system_id>/<run_id>
```

Research tools reach calibration through the MCP `calibrate`, `rerun_calibration`, `check` and `propose_device_update` tools; see the [MCP reference](mcp.md#tool-surface). `apply-device-update` is a human action: it is absent from those tools, and agent runtimes are configured to deny it like the other operator commands.
