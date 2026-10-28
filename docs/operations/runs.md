# Run records

`entanglix.runner` launches authored CPU scripts with optional timeouts, cancellation, live inspection, recovery, and linked reruns. It freezes inputs and code provenance and publishes records atomically. The [simulation protocol](protocols.md) runs built-in CPU/CUDA computations with linked checks on request. Per-device GPU allocation locks protect scientific dispatch; serial static/dynamic sweeps retain per-point results. Reconciliation checks ownership and recorded process-group liveness itself.

## Commands and examples

Commands in these references are the bare `entanglix` of an installation, run from the project directory, with input files in the project. From a source checkout, developers prefix them with `uv run --locked --exact`; commands that only a checkout uses are labelled.

The examples ship in the installation directory that `entanglix components` reports as `path`, such as `~/.local/share/entanglix/0.1.0/examples/`. Copy an example's whole directory into the project, so its files stay beside one another, and run it there; the example paths below assume that copy:

```sh
cp -R ~/.local/share/entanglix/0.1.0/examples/two-level ~/.local/share/entanglix/0.1.0/examples/recorded_decay.py .
```

Copied system, protocol, dataset and job-script files run this way. The examples' driver programs, such as `codesign.py`, `calibration.py`, `reuse.py` and `investigate.py`, are checkout only: they import Entanglix and read the repository layout. An example that builds on another, such as the catalog example, names it in its README.

## Authored CPU jobs

```sh
entanglix run-script recorded_decay.py --system two-level/system.toml
entanglix capabilities
entanglix list-runs
entanglix read-run research_runs/<system_id>/<run_id>
entanglix abort-run research_runs/<system_id>/<run_id>
entanglix rerun research_runs/<system_id>/<run_id>
```

Commands print JSON. `run-script` waits for the job and returns its exit code. Use `--runs-root PATH` to choose the output root, repeated `--input NAME=PATH` to freeze additional files under `spec/`, and repeated `--parent-run ID` to record lineage. The corresponding Python operations are `run_script(system, script, ...)`, `list_capabilities()`, `list_runs(root)`, and `read_run(path)`. Capabilities inspect package metadata in the runner environment without creating or synchronizing it; before its first installation, no installed engines are reported. Engine names match the adapters, with supported `devices` reported separately; installation does not certify hardware availability. `plugin_engines` lists engines that installed plugins register, with their distribution and version; their devices and capabilities are resolved at admission. `engine_defaults` shows the user's per-device defaults ([engine selection](system-spec.md#engine-selection-policy)).

Before admission, a subprocess in the locked, synchronized runner environment constructs the CPU model, checks script syntax, checks the environment's consistency, and records its identity. Referenced subsystem specs and numeric archives are collected automatically; the standalone script is frozen as `spec/script.py`. Additional Python inputs receive the same syntax check. Relative imports are unsupported for standalone scripts. Changed inputs between validation and freezing fail preparation before computation.

Jobs sharing one installation or checkout use the same extras set. An installation uses the set its components record: `differentiable` for the base, adding `cuquantum` and `differentiable-cuda12` with the GPU component (`entanglix components add gpu`). A checkout uses `[tool.entanglix.runner]` `extras = [...]` in `pyproject.toml`; the default is empty. The workstation-local `ENTANGLIX_RUNNER_EXTRAS` comma-separated list overrides either for all jobs, including CPU checks. Each launch resolves the set, so a server started earlier uses a component added since. Each check and job uses `uv run --locked --inexact` with that set and an absolute `UV_PROJECT_ENVIRONMENT` pointing to the ignored `.venv-runner/`; synchronizing it leaves the server's `.venv` alone. An installation's uv calls use its own uv and pinned Python and ignore uv configuration files; inherited variables that would select another environment, interpreter or cache, or skip synchronization, are removed. CPU jobs receive `CUDA_VISIBLE_DEVICES=""` and `JAX_PLATFORMS=cpu`. Keep the checkout and configuration stable while jobs run; use a separate checkout when reconstructing an earlier code revision.

Each launch holds a shared lock on the job environment from before it resolves the set until its job and any surviving children end. Adding a component takes that lock exclusively without waiting: `components add` is refused while any launch or job runs, and launches are refused while a component is being added. A change that fails leaves the environment marked incomplete; launches are refused, naming the command, until that change is rerun and succeeds. A request for a bundled engine or solver that is not installed is refused before an attempt exists, with the component that adds it.

The ignored `.venv-runner-extras.toml` records the pinned extras as one sorted,
duplicate-free list, for example `extras = ["cuquantum", "differentiable-cuda12"]`.
The runner claims this profile atomically before the first synchronization; every
later launch must request the same set. A mismatch is rejected before `uv` can
change packages, including while jobs are active. Runtime versions are still
observed in the job environment, rather than inferred from this profile.
An existing environment without a profile is rejected: stop its jobs and confirm
a sync with the intended extras before recording that list, or recreate the
environment and let its first launch claim the profile. In an installation,
`components add gpu` changes the profile itself. In a checkout, to change profiles,
use a separate checkout or stop all jobs before removing both the environment and
its profile.

The job runs noninteractively in its own process group, with `artifacts/` as its working directory so ordinary relative outputs land in the job's area. It inherits the owner's descriptor and waits until `runner/started.toml` has been published before executing the frozen script. Output and diagnostics go to `run.log`. `ENTANGLIX_RUN_DIR` names the attempt and `ENTANGLIX_SYSTEM_SPEC` its frozen system spec; additional data is under `<ENTANGLIX_RUN_DIR>/spec/`.

Publish `<ENTANGLIX_RUN_DIR>/result.toml` atomically with `publish_toml(path, result, replace=True)`. An authored result requires an actual `engine` name and a `metrics` table; solver settings belong in the result when applicable. Optional `[artifacts.<name>]` entries supply `path` and `sha256`; paths are relative to the attempt directory, must stay inside `artifacts/`, and their digests are checked before completion. See [the recorded decay example](../../examples/recorded_decay.py). A missing or malformed result or a digest mismatch fails orchestration with code `64`. Additional authored fields, including a `checks` table, are data and do not create trusted check outcomes. Nonzero job exits retain their code and logs; [execution outcomes](#execution-outcomes) covers allocation failures and signal deaths.

An arbitrary script has no structured evolution request for engine selection. Its manifest uses `engine = "authored-script"`; the result names the engine actually used. Scripts have no trusted computation definition for built-in checks. Structured simulations use declared needs for engine selection.

Inspection is read-only. `read_run` includes records and the last 8 KiB of `run.log`; it does not determine scientific verification. Malformed results appear as `result_error`; listing retains malformed runner records with a diagnostic. If the current code cannot load a frozen plan or construct its model, inspection still returns the retained records and check history; `check_listing` contains no checks and explains why it is unavailable. Requesting checks still requires a valid plan. Without `FINISHED`, ownership or a live recorded group means `active`, absence of both means `interrupted`, and indeterminate group liveness means `unknown`. After the main process exits, the supervisor retains ownership and waits for the whole process group to disappear before validating the result and writing `FINISHED`.

Linked [investigation jobs](investigations.md#usage-and-optional-limits) also retain their dispatch revision, base, resource commitment, and admitted limits. Their absolute deadline is checked before start and during execution; checks and reruns count as separate jobs in that investigation.

### Plugins in the runner environment

Plugins are Python distributions that register engines under `entanglix.integrators` ([engine selection](system-spec.md#engine-selection-policy)) or solvers under `entanglix.solvers` ([control](control.md#launch-and-solver-selection)). An installation adds them to its job environment, and changes them only while no launch or job uses it:

```bash
entanglix plugins add PATH       # a project directory with a pyproject.toml, or a wheel
entanglix plugins                # installed plugins and the engines and solvers they register
entanglix plugins remove NAME
```

`plugins add` installs a project or wheel that registers an engine or a solver, declared in a project's `pyproject.toml`, with the installation's uv, which resolves its dependencies. Repeat it after editing the plugin's code: it reinstalls the plugin even when its version is unchanged. A plugin need not be a whole upstream codebase. A small project can hold the adapter and the modules it imports, with its code in `src/my_integrator/`:

```toml
[project]
name = "my-integrator"
version = "0.1.0"
dependencies = ["numpy"]

[project.entry-points."entanglix.integrators"]
my-integrator = "my_integrator:Engine"

[build-system]
requires = ["uv_build>=0.12,<0.13"]
build-backend = "uv_build"
```

Every module a plugin imports must be in its distribution or in its dependencies: the run records installed distributions, so code imported from elsewhere leaves no record. In a development checkout, install plugins with `uv pip install --python .venv-runner/bin/python PATH` after the first launch has created `.venv-runner`.

Each launch and `components add gpu` synchronize the locked dependencies and keep the other installed distributions. Before admission, the runner refuses an environment that `uv pip check` reports as inconsistent, an editable distribution other than the project itself, and a distribution without an install record. A plugin requiring another version of a locked package is undone by the next synchronization, which restores the locked version; the consistency check then refuses launches until the plugin is changed. Every run records the installed distributions ([environment provenance](#environment-and-code-provenance)).

## Computation records

Simulations appear as `protocol = "simulate"`. A simulation's `spec/` contains `system.toml`, `simulate.toml`, its compiled `plan.toml`, referenced model files, and numeric inputs. The manifest records the computation kind, plan digest, selected engine, environment, and code provenance. The plan digest covers normalized content and model/numeric input digests; separate `spec_sha256` entries preserve raw-file integrity. The plan format is internal and is not a public authoring interface.

The result records the actual engine, method, dimensions, device, and solver diagnostics. `artifacts/output.npz` holds `states`, `times_ns`, and `observable.<name>` arrays for dynamics, or `energies_ghz` for spectra; its digest is checked before completion. Computation completion describes execution only; checks run separately on request.

CPU dynamiqs jobs need the `differentiable` extra in the runner extras set; an extra installed only in the development environment does not reach jobs, and CPU jobs need no CUDA extra.

Sweeps appear as `protocol = "sweep"`. Their frozen inputs include the base protocol,
`sweep.toml`, an internal sweep manifest, and each point's effective model/plan
under `spec/points/<index>/`. Point indices follow grid order with the last axis
varying fastest. `artifacts/points/<index>/` contains `output.npz` and an atomic
`result.toml` with its plan digest, metrics, dimensions, engine, solver and diagnostics.
A point result is published only after its archive is complete. `read_run.sweep_progress`
shows completed points and archive errors even after interruption. The final
`artifacts/output.npz` contains the metric grids; aggregate `result.toml` links point
artifacts and metric plots. The [protocol reference](protocols.md#serial-grid-sweeps)
describes sweep checks, coverage and full-grid replay.

## Corrector policy and verification records

The corrector ships with [versioned thresholds](../../src/entanglix/corrector/thresholds.toml).
Its policy version hashes the code and thresholds; a change between admission and
load fails execution without check outcomes. There is no dimension cap or pre-launch
memory estimate; an allocation failure is reported as out of memory
([execution outcomes](#execution-outcomes)).

```sh
entanglix check research_runs/<system_id>/<run_id>
entanglix check research_runs/<system_id>/<run_id> integrity segments diagnostics --timeout 60
entanglix read-run research_runs/<system_id>/<run_id>
```

With no names, `check` lists relevant checks and computes nothing. Named checks run
as one linked verification attempt with per-check pass, fail, or inapplicable outcomes;
diagnostics have no pass/fail outcome. There is no run-level verdict. Code `0` means
execution completed, even when a check failed. Authored scripts have no built-in
checks. The [protocol reference](protocols.md#requested-checks-and-segment-tolerances)
describes each check, prerequisites, segment tolerances, and replacement refinement.

Verification attempts identify the source as a parent and retain its relative
location, result/artifact digests, and policy version. They freeze the plan, model
inputs, requested names, and effective refinement. Original results and arrays stay
in the source attempt; missing or changed evidence fails the check execution. A
replacement supplied with `--refinement FILE` leaves the source and its plan digest
unchanged. Reports record its origin; a check attempt whose frozen inputs were
altered fails without outcomes.
Calibration checks also take a holdout dataset, a solution and a coverage expectation;
see the [calibration reference](calibration.md#checks).

Reports identify scopes, tolerances, residuals, sampled windows and coverage, refined
engine/dimensions/settings, and inapplicable reasons. `--timeout SECONDS` and
`abort-run` bound execution. Failed, timed-out, aborted or interrupted check attempts
report no outcomes. A passing segment establishes only local agreement: no built-in
check measures whole-window error or the rotating-wave approximation.

`read-run` lists linked verification attempts with status, policy version, source
identity match, and completed check reports. Earlier reports remain history under
their recorded policy. A newer failure does not erase earlier outcomes or substitute
them for its missing report.

MCP reads summarize check and sweep details and page run listings; see the
[MCP response reference](mcp.md#investigation-views-and-tables). CLI and Python reads
are complete.

## Limits, interruption, and recovery

`run-script --timeout SECONDS` limits execution from publication of the start record, including lingering processes after the main job exits. The default has no timeout; preparation is outside this timer. The timeout must be finite and positive. Termination has a fixed two-second grace period. The supervisor polls at 10 ms intervals; scheduling can delay observation. At expiry it sends `SIGTERM` to the job group and, after the grace interval, `SIGKILL` if needed. It waits for the group to disappear, then records `aborted/130`, actual wall time including termination, and the reason in `run.log`.

`abort-run PATH` (Python: `abort_run(path)`) publishes `runner/abort.toml`, signals only that attempt's recorded process group, and uses the same grace before escalation. Repeated requests retain the first event. The supervisor writes the terminal record; the abort command reports a request, not confirmed completion. Inspect the run to confirm its outcome. Ctrl-C (`SIGINT`), `SIGTERM`, or `SIGHUP` sent to the supervising CLI command also stops the job group and records `aborted/130`. The supervisor watches the abort request so it can finish cancellation if the requesting command is interrupted. Abrupt runner death, such as `SIGKILL`, leaves no terminal record and may leave the job running.

For an interrupted attempt:

1. Inspect it with `read-run`; retained records and the log tail show what was observed.
2. If its orphaned job group is still active, use `abort-run` and inspect again. Do not reconcile a live or unknown group. A paused live supervisor also retains ownership.
3. Once the owner and group are gone, run `entanglix reconcile-run PATH --state "last observed state and recovery action"`. This calls `record_reconciliation`, retains uncertain usage, and never creates `FINISHED`.
4. Use `rerun PATH` to launch the frozen script, system, referenced assets, and additional inputs under a new ID linked to the earlier attempt. Every frozen input digest is checked before launch. The old attempt is preserved; partial outputs are not resumed.

Script reruns retain the earlier timeout; `--timeout SECONDS` overrides it and `--runs-root PATH` changes the destination. `rerun` replays any simulation, sweep, control or calibration run the same way, by the protocol its record names; a protocol rerun keeps its frozen spec's timeout and refuses `--timeout`. Python callers use `rerun(path, ...)`. Rerunning an active or unknown attempt is refused. Reconstruct the recorded code and environment separately as described below: reruns use the current checkout and configuration and record their actual provenance. Attempts without recorded `[launch]` metadata require manual replay of `spec/script.py`, `spec/system.toml`, and additional `--input` files with `--parent-run ID`.

### Local GPU ownership

For a CUDA simulation or sweep, the runner resolves the single visible device through `nvidia-smi`, then takes a nonblocking file lock before its launch check initializes CUDA and before admission creates an attempt. The lock lives under `$XDG_STATE_HOME/entanglix/gpu/` (default `~/.local/state/entanglix/gpu/`), shared by runner instances using that OS account and state directory, independent of their runs roots. A held lock, a live recorded process group, or an allocation awaiting reconciliation refuses the launch without an attempt, investigation charge, or checkpoint progress. A dry run takes the same lock for its check and releases it on return; it reserves no future launch.

The stable `<UUID>.lock` file is never unlinked. Once the attempt exists, `<UUID>.toml` records its absolute run path and, once started, process group. A failure of the launch check, admission, or manifest creation releases ownership. A failure after manifest publication retains a failed attempt.

The launch check and then the job hold the device lock continuously through `FINISHED`, so a job that survives its runner blocks reuse. Once the lock is free, a live or indeterminate recorded group still blocks reuse, and an allocation without `FINISHED` needs reconciliation even after its group disappears; missing history also blocks reuse. This is cooperative ownership, not device containment or a scheduler.

## Location and identity

The default runs root is `research_runs/` relative to the caller's working directory; `--runs-root` (Python: `runs_root=`) changes it. An attempt directory holds:

```text
<runs_root>/<system_id>/<run_id>/
  run.toml
  runner/started.toml
  runner/abort.toml
  runner/reconciled.toml
  spec/
    system.toml
    code/changes.patch
    code/untracked/
  artifacts/
  run.log
  result.toml
  FINISHED
  proposals/
```

Run IDs have the form `r20260930T153012123456Z-<8 hexadecimal digits>`: a UTC timestamp with microseconds and a random suffix. Creation is exclusive; an existing directory is never reused. System IDs must start with an ASCII letter or digit and contain only letters, digits, `.`, `_`, and `-`. Parent run IDs preserve lineage; a rerun gets a new ID.

Every research run, including an authored script, requires a validated system spec frozen as `spec/system.toml`. The launch gate validates the model and collects the files it references; additional inputs keep the names given with `--input NAME=PATH`. Names must be UTF-8 encodable and relative, without `..`, must not alias another file or use a file as a directory, and cannot use `spec/code/`, which is reserved for provenance. Files are copied, hashed, and made read-only, so later source edits do not change the attempt.

## Record fields and ownership

Runner records are published once and made read-only. Jobs own `result.toml` and `artifacts/`; runner paths are reserved. `proposals/` holds a completed calibrate run's [device-update proposals](calibration.md#device-update-proposals). This is a cooperative interface, not containment of arbitrary code running as the same OS user.

| Record | Fields and meaning |
|---|---|
| `run.toml` | `record_version = 1`; `run_id`, `system_id`, `protocol`, UTC `created_at`, `parent_run_ids`, declared `needs`, selected `engine`, `device` (`cpu` or `cuda`), `executor = "local"`, `spec_sha256`, `code`, `env_hash`, and `environment`. Script `[launch]` records explicit additional `input_names` and optional `timeout_seconds`; absence means no timeout |
| `runner/started.toml` | `record_version`, UTC `started_at`, positive `process_group`, and `gpu_devices` (device identifiers; empty for CPU). Published before computation starts |
| `runner/abort.toml` | UTC `requested_at`, published once by the abort operation; a request, not a terminal outcome |
| `FINISHED` | `record_version`, `status`, `exit_code`, UTC `ended_at`, and `usage`: nonnegative finite `wall_seconds`, `gpu_seconds`, and `uncertain = false` |
| `runner/reconciled.toml` | `record_version`, UTC `observed_at`, `last_observed_state`, and `usage.uncertain = true`. When a start record exists, includes its `started_at`, `process_group`, and `gpu_devices`; otherwise these fields are absent. Unknown usage is not replaced with zero |
| `result.toml` | A protocol-specific TOML table containing metrics, actual engine/solver information, and artifact references with digests. Trusted check outcomes come only from linked verification results, which carry the bound per-check report |

**Versions and compatibility.** A new release reads supported earlier versions of the records and inputs that earlier releases wrote. Every version written so far is supported, and a version leaves support only with a release-note entry. Reading means inspecting a record's retained contents; compiling, reusing, replaying or checking an earlier run is not promised. Historical records are never rewritten. Protocol results, sweep point results, the calibration assignment, launch attempts and abort requests carry no `record_version` and are version 1; their first incompatible revision carries the field. Records that carry `record_version` keep requiring it.

`run.toml` is the first published file, and inputs are frozen before computation starts. Input names, missing inputs, duplicates and file/directory conflicts are checked before the attempt directory exists. For handled preparation failures: one before `run.toml` is published removes the directory, because no run was admitted; one after publication records `FINISHED` with `failed/64` and zero job usage, and the error names the attempt path. If storage prevents that write, the attempt stays incomplete and can be reconciled without a start record.

Records are published atomically and durably, and a runner record is never replaced, even by a concurrent caller. A job replaces its own result with `publish_toml(path, result, replace=True)`; a failed serialization leaves the previous result intact.

The runner owns each attempt through an exclusive lock, taken before `run.toml` is published, and passes the lock to the job, whose child processes can inherit it. Ownership is held by the processes that hold a copy of the lock and is released when all of them have closed it, so a job that survives its runner keeps the attempt. This lock covers CPU and GPU runs; the per-device GPU lock is separate.

`reconcile-run` refuses while the runner or a surviving job owns the attempt, including after the job exits but before its runner publishes `FINISHED`. It also refuses a finished or unadmitted attempt, and a recorded process group that is live or whose liveness cannot be determined. Only when ownership and the process check both permit it does it publish its record. An attempt without a start record can be reconciled once ownership has been released.

## Environment and code provenance

Each run's `environment` records the lockfile SHA-256, Python version, OS, architecture, libc name and version, the sorted extras, GPU observations, and the installed distributions; kernel releases are excluded. The `gpu` fields are `driver`, `cuda` (the observed CUDA runtime version) and `cuquantum` (the loaded native cuDensityMat library version); a field not observed, as on every CPU run, is `"unobserved"`. Installed versions do not establish runtime versions or GPU availability.

`distributions` lists every distribution in the runner environment's `site-packages`, keyed by normalized name, with its `version` and `record_sha256`: the SHA-256 of its install record (`RECORD`), in which the installer lists each installed file with its hash. A package rebuilt and reinstalled at the same version therefore changes the record; files edited after installation do not. Recording refuses an environment that `uv pip check` reports as inconsistent, an editable distribution other than `entanglix`, and a distribution without `RECORD`. The project's own editable source is covered by code provenance.

`env_hash` is the SHA-256 of the environment table before `env_hash` is added, serialized as UTF-8 JSON with sorted keys and compact separators, so every field above participates in the identity.

Before a CUDA launch, the runner checks the selected GPU in the job environment. For every engine it confirms NVIDIA-SMI's canonical UUID and observes the driver; device ownership and the refusal of a busy device are [shared](#local-gpu-ownership). Only for `cudensitymat` does it import CuPy and cuQuantum (the `cuquantum` extra), match a CUDA-visible device, observe the loaded CUDA and native cuDensityMat versions, and initialize the native library on that device; other engines record `cuda` and `cuquantum` as `"unobserved"` and need no cuQuantum stack. Failures explain device visibility, driver access, and, for cuDensityMat, the extra. The device UUID and index are dispatch data, separate from `env_hash`. CPU runs neither run this check nor import GPU packages.

Code provenance requires a Git repository with a committed `HEAD`. The `code` table records `revision`, `dirty`, `patch_sha256`, and `untracked_sha256`. `changes.patch` contains the combined staged and unstaged diff from `HEAD`, including binary changes, with color disabled and fixed `a/` and `b/` prefixes regardless of Git configuration; `code/untracked/` retains all non-ignored untracked files. Untracked symlinks are rejected rather than silently copied as regular files. Ignored scripts and data must be supplied explicitly as inputs. Capture a stable working tree before dispatch.

The patch and untracked contents have a combined default limit of 16 MiB, configurable through `max_snapshot_bytes`. The limit is checked from file sizes before untracked files are read. Exceeding it fails with a diagnostic instead of omitting files. Keep unrelated generated data ignored, supply research data explicitly, or raise the limit for a larger source snapshot.

To reconstruct code, check out the recorded revision in a separate checkout, apply `spec/code/changes.patch` with `git apply`, and restore the untracked files to their recorded relative paths. Use the frozen specs and data and an environment with matching identity. Reproduction comparisons use determinism where supported and declared numerical tolerances otherwise.

## Execution outcomes

Records distinguish pre-start failure, observed job outcomes, and reconciliation:

- Invalid arguments or a failed launch gate return `64` with a diagnostic before `run.toml` exists, and create no run; Python operations raise an exception.
- After creation, known failures are recorded and the attempt is kept. Before a start record exists, an attempt ends `failed/64`, or `aborted/130` if cancelled, with zero job wall time; any GPU allocation time is still charged, including a failed spawn. Job outcomes require a start record.
- Observed successful execution uses `completed` and code `0`; a job exiting `130`, a timeout, or runner-triggered cancellation is `aborted/130`. Other job failures preserve exit codes in `0–255`. A signal death outside a requested cancellation records `128 + signal` (for example, `SIGKILL` becomes `137`).
- An observed allocation failure is identified as out of memory in the diagnostic, including a launch-gate failure before an attempt exists. A signal death the runner did not request is reported as that signal with its cause unknown; for `SIGKILL` the diagnostic names exhausted memory as a possible cause only. A positive shell-encoded status alone cannot distinguish a signal from an explicit job exit with that code, and the diagnostic states that uncertainty. If the supervisor also dies, the interruption and reconciliation rules apply.
- `wall_seconds` is elapsed job time from publication of the start record until the supervisor observes the whole process group's termination, measured once with a monotonic clock. Input preparation and result validation are excluded; a known pre-start preparation failure or cancellation records zero. Allocated `gpu_seconds` means allocated devices multiplied by allocation duration, including the launch check and failed preparation for an admitted attempt; CPU jobs record zero.
- Runner loss leaves no `FINISHED`, and a surviving job may still own the attempt and GPU. Reconciliation, described above, records uncertainty without fabricating completion.

Without a valid `FINISHED`, artifacts cannot substantiate a completed scientific result. With `FINISHED`, successful execution still does not establish verification. A sweep is one run with per-point artifacts; interrupted partial results are retained, and a new linked rerun recomputes the grid.

The [harness contract](../../HARNESS_CONTRACT.md) states the execution and verification guarantees.

## Detached launches and waiting

Job commands accept `--detach`: `simulate`, `sweep`, `control`, `calibrate`, `run-script`, `rerun`, and named `check` requests. The command returns the admitted run ID and path; a separate supervisor retains ownership, enforces the job timeout and investigation deadline, and writes the terminal record. Closing the requesting terminal or losing its response does not stop that supervisor. Use `abort-run` to cancel an admitted job. Dry-run previews and check listings remain synchronous and do not accept `--detach`.

```sh
entanglix simulate readout/readout.toml --runs-root research_runs --detach
entanglix read-run research_runs/SYSTEM_ID/RUN_ID --wait 60
```

`read-run --wait SECONDS` returns when the attempt completes, fails, aborts, loses ownership, or the requested wait expires. Seconds must be finite and non-negative; the command line applies no maximum. An expired or interrupted wait leaves the job running. Reads and cancellation can run concurrently; waits poll at intervals of at most 0.25 seconds. Ctrl-C returns exit 130 without a traceback, including while a detached launch awaits admission.

Before starting its supervisor, a detached request is recorded under `<runs_root>/_launches/<launch_id>/request.toml`. The record holds the operation, absolute input paths, and investigation ID when applicable. The launch directory's inherited lock tracks its owner. The admitted run's manifest carries `launch_id`; `attempt.toml` in the launch directory records the intended relative run path before the manifest is published; the link alone does not establish admission. `supervisor.log` retains launch diagnostics, and a refused request can carry `refused.toml` with its reason. Refusals create no attempt or investigation usage.

If an admission response is lost, use `list-runs` to see admitted attempts and unresolved launches. `read-run <launch_directory>` recovers one request as `pending`, `admitted` (with its run ID), or `refused`. With `--wait`, it waits for admission resolution rather than job completion. Recovery checks ownership and the linked manifest without scanning other attempts; it never submits another job. A supervisor lost after publication leaves an interrupted attempt for the normal reconciliation path. A request whose owner disappears before publication is definitively non-admitted.

`read-investigation` lists admitted attempts with their base, varied values, revision, and status, along with its unresolved launches. Pending requests consume no allowance; accounting starts at manifest publication.
