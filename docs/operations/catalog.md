# Results catalog

The catalog keeps designs taken from completed runs: a device, a pulse, or both. This reference describes the candidates a run can supply, staging one with `register-result`, the staged record and the evidence digests it holds, and reading candidates with their lineage.

## Candidates

A candidate is one of three designs:
- **Device:** a system file and every file it references, namely subsystem sources and array files.
- **Pulse:** a pulse in the neutral form, with the name of each channel.
- **Device and pulse:** both, from a control run, whose pulse was solved for its device.

One completed run supplies the design:

| Source run | Device | Pulse |
|---|---|---|
| `simulate` | The frozen `system.toml`, which holds the spec's parameter overrides | None |
| `sweep` | `--point N`: the point's frozen system, which holds the point's values | None |
| `control` | The frozen system, which holds the spec's `[parameters]` values | `artifacts/pulse.npz`, with channel names from the frozen control spec, in order |
| `calibrate` | The frozen system with every fitted parameter of the solution written; `--solution N` names it among several | None |
| `script` | The frozen system, or `--artifact NAME`: a system file the script published | `--artifact NAME`: a neutral pulse the script published |

Any other completed run supplies its frozen system as a device. A verification run holds no design. Each selector belongs to one protocol, and a selector given for another is refused.

**Sweep points.** A device is the point's frozen system. A sweep axis on an evolution setting, such as a drive amplitude, carrier or frame frequency, is a protocol setting and not part of the device. The point's index, coordinate and every varied value are recorded as context.

**Calibration solutions.** Every fitted parameter's value from the solution is written into its `value` in the system file's parameter table, determined or not, so the device is the one the solution's checks evaluate. Each written parameter's `source` reads `calibration run <system_id>/<run_id>, solution <N>, result sha256 <digest>`. Units, priors and every other field are unchanged, and the file is rewritten as canonical TOML. The record lists the parameters the solution does not determine and names the run's `result.toml`, which holds the solution's uncertainty report. The axis map's flux parameter is not written; its name and the fitted offset and period are recorded as context. A device-update proposal carries only determined parameters; see the [calibration reference](calibration.md#device-update-proposals).

**Script files.** A published system file must validate, and the files it references must lie beside it in `artifacts/`. A published pulse holds exactly the arrays below.

**Neutral pulse.** A staged `pulse.npz` holds:
- `kind = "zero_order"`;
- `times_ns`: N ≥ 2 knots, starting at zero and strictly increasing;
- `amplitudes_ghz`: finite, one row of N values per channel;
- `channels`: one distinct, nonempty name per row.

These are the arrays of the [control pulse](control.md#neutral-pulse-and-candidate-record), with the channel names added. A pulse that does not meet them is refused.

## Staging

```sh
entanglix register-result RUN --design device --point 1 --claim "f0 = 0.3 point of the spectrum sweep"
entanglix register-result RUN --design device_and_pulse --claim "X gate at 60 fF"
entanglix register-result RUN --design device --solution 0 --claim "Fit to cooldown 3"
```

`--design` is `device`, `pulse` or `device_and_pulse`. `--claim` states what the candidate is offered as, and is required. The output is the published record as JSON, with the folder's path.

Staging refuses:
- a run that is not completed;
- a run outside `<runs_root>/<system_id>/<run_id>`, or one whose frozen inputs no longer match their recorded digests;
- a run that holds no design of the requested kind, such as a pulse from a simulation;
- a sweep without `--point`, a point outside the grid, and a calibration with several solutions without `--solution`, or with no solution;
- a pulse not in the neutral form, or a published system file that does not validate;
- an empty claim.

A refused request writes nothing, and staging never changes the source run.

```text
<runs_root>/_catalog/staging/<staged_id>/
  record.toml
  device/system.toml
  device/<each referenced file>
  pulse.npz
```

Staged IDs have the form `s20261025T153012123456Z-<8 hexadecimal digits>`: a UTC timestamp with microseconds and a random suffix. Each folder is written under a temporary name inside `_catalog/` and renamed into place, so a reader sees a complete folder or none; concurrent requests publish separate folders. The files are read-only, and no command changes or removes a staged folder.

## The staged record

Every path in `record.toml` is relative to the runs root.

| Field | Meaning |
|---|---|
| `record_version`, `kind` | `1` and `staged_candidate` |
| `staged_id`, `created_at`, `design`, `claim` | The candidate's ID, UTC creation time, design and claim note |
| `[source]` | `run` as `<system_id>/<run_id>`, `protocol`, `investigation_id` when the run belongs to one, and `point`, `solution` or `artifact` when one was selected |
| `parent_entry_ids` | The catalog entries the source run's manifest names; empty when it names none |
| `[evidence]` | `result_sha256`, the digest of the source `result.toml`; `artifact_sha256`, the digest of the primary artifact that built-in checks bind (`output`, `pulse` or `assignment`), absent for a script; `source_files`, the digest of every source file the design was copied from |
| `[device]` | `path` of the staged device folder, `files` with each file's digest by name, and for a calibration solution `written` values and the `undetermined` parameters |
| `[pulse]` | `path`, `sha256`, `channels`, `knots` and `duration_ns` |
| `[context]` | `sweep`: `point`, `coordinate` and `values`; `calibration`: `solution`, `outcome`, `fit_report` and, with an axis map, `axis_map` |

Check outcomes are not copied. They stay in the source run's verification history, which `read-run` lists.

## Reading the catalog

```sh
entanglix list-catalog --runs-root research_runs
entanglix read-entry s20261025T153012123456Z-1a2b3c4d --runs-root research_runs
```

`list-catalog` lists staged candidates in creation order, with ID, status, path, design, source run, claim and creation time. A record that cannot be read is listed as `invalid` with a diagnostic.

`read-entry` returns the record with a `lineage` table:
- `source_run`: the run's path, ID, protocol, status and `parent_run_ids`;
- `ancestors`: each run reached through `parent_run_ids`, in the same form.

A run whose directory no longer exists is reported as `unavailable`; the staged design stays readable. Failed and unstaged runs remain readable through `list-runs` and `read-run` without the catalog.
