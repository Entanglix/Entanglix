# Calibration

Calibration fits device parameters to measured spectroscopy. This reference describes measurement datasets, with the format every measurement is converted to, validation, digests, raw-file checks, and holdout splits, and peak extraction from maps.

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
