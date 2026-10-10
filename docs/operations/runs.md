# Run records

`entanglix.runner` launches authored CPU scripts with optional timeouts, cancellation, live inspection, recovery, and linked reruns. It freezes inputs and code provenance and publishes records atomically. The [simulation protocol](protocols.md) runs built-in CPU/CUDA computations with linked checks on request. Per-device GPU allocation locks protect scientific dispatch; serial static/dynamic sweeps retain per-point results. Reconciliation checks ownership and recorded process-group liveness itself.

## Authored CPU jobs

```sh
uv run --locked --exact entanglix run-script examples/recorded_decay.py --system examples/two-level/system.toml
uv run --locked --exact entanglix capabilities
uv run --locked --exact entanglix list-runs
uv run --locked --exact entanglix read-run research_runs/<system_id>/<run_id>
uv run --locked --exact entanglix abort-run research_runs/<system_id>/<run_id>
uv run --locked --exact entanglix rerun-script research_runs/<system_id>/<run_id>
```

Commands print JSON. `run-script` waits for the job and returns its exit code. Use `--runs-root PATH` to choose the output root, repeated `--input NAME=PATH` to freeze additional files under `spec/`, and repeated `--parent-run ID` to record lineage. The corresponding Python operations are `run_script(system, script, ...)`, `list_capabilities()`, `list_runs(root)`, and `read_run(path)`. Capabilities inspect package metadata in the runner environment without creating or synchronizing it; before its first installation, no installed engines are reported. Engine names match the adapters, with supported `devices` reported separately; installation does not certify hardware availability.

Before admission, a subprocess in the locked, exactly synced environment constructs the CPU model, checks script syntax and imports, and records the environment identity. Referenced subsystem specs and numeric archives are collected automatically; the standalone script is frozen as `spec/script.py`. Additional Python inputs receive the same static import check. Relative imports are unsupported for standalone scripts. The allowlist lives in the project's import-boundary configuration; scanning is preflight under the cooperative trust model, not containment. Changed inputs between validation and freezing fail preparation before computation.

Jobs sharing the checkout use the same extras set, configured once in `pyproject.toml` as `[tool.entanglix.runner]` with `extras = [...]`; the default is empty. The workstation-local `ENTANGLIX_RUNNER_EXTRAS` comma-separated list overrides this setting for all jobs, including CPU checks. Each check and job uses `uv run --locked --exact` with that set and an absolute `UV_PROJECT_ENVIRONMENT` pointing to the ignored `.venv-runner/` in the checkout. The first launch installs this environment; exact synchronization leaves the development `.venv` and its extras alone. CPU jobs receive `CUDA_VISIBLE_DEVICES=""` and `JAX_PLATFORMS=cpu`. Keep the checkout and configuration stable while jobs run; use a separate checkout when reconstructing an earlier code revision.

The ignored `.venv-runner-extras.toml` records the pinned extras as one sorted,
duplicate-free list, for example `extras = ["cuquantum", "differentiable-cuda12"]`.
The runner claims this profile atomically before the first synchronization; every
later launch must request the same set. A mismatch is rejected before `uv` can
change packages, including while jobs are active. Runtime versions are still
observed in the exact job environment, rather than inferred from this profile.
An existing environment without a profile is rejected: stop its jobs and confirm
an exact sync with the intended extras before recording that list, or recreate the
environment and let its first launch claim the profile. To change profiles, use a
separate checkout or stop all jobs before removing both the environment and its
profile. Keep one extras set in the shell override or project configuration.

The job runs noninteractively in its own process group, with `artifacts/` as its working directory so ordinary relative outputs land in the job's area. It inherits the owner's descriptor and waits until `runner/started.toml` has been published before executing the frozen script. Output and diagnostics go to `run.log`. `ENTANGLIX_RUN_DIR` names the attempt and `ENTANGLIX_SYSTEM_SPEC` its frozen system spec; additional data is under `<ENTANGLIX_RUN_DIR>/spec/`.

Publish `<ENTANGLIX_RUN_DIR>/result.toml` atomically with `publish_toml(path, result, replace=True)`. An authored result requires an actual `engine` name and a `metrics` table; solver settings belong in the result when applicable. Optional `[artifacts.<name>]` entries supply `path` and `sha256`; paths are relative to the attempt directory, must stay inside `artifacts/`, and their digests are checked before completion. See [the recorded decay example](../../examples/recorded_decay.py). A missing or malformed result or a digest mismatch fails orchestration with code `64`. Additional authored fields, including a `checks` table, are data and do not create trusted check outcomes. Nonzero job exits retain their code and logs; [execution outcomes](#execution-outcomes) covers allocation failures and signal deaths.

An arbitrary script has no structured evolution request for engine selection. Its manifest uses `engine = "authored-script"`; the result names the engine actually used. Scripts have no trusted computation definition for built-in checks. Structured simulations use declared needs for engine selection.

Inspection is read-only. `read_run` includes records and the last 8 KiB of `run.log`; it does not determine scientific verification. Malformed results appear as `result_error`; listing retains malformed runner records with a diagnostic. If the current code cannot load a frozen plan or construct its model, inspection still returns the retained records and check history; `check_listing` contains no checks and explains why it is unavailable. Requesting checks still requires a valid plan. Without `FINISHED`, ownership or a live recorded group means `active`, absence of both means `interrupted`, and indeterminate group liveness means `unknown`. After the main process exits, the supervisor retains ownership and waits for the whole process group to disappear before validating the result and writing `FINISHED`.

Linked [investigation jobs](investigations.md#usage-and-optional-limits) also retain their dispatch revision, base, resource commitment, and admitted limits. Their absolute deadline is checked before start and during execution; checks and reruns count as separate jobs in that investigation.

## Computation records

Simulations appear as `protocol = "simulate"`; historical `"plan"` and `"reference"` records remain readable. A simulation's `spec/` contains `system.toml`, `simulate.toml`, its compiled `plan.toml`, referenced model files, and numeric inputs. The manifest records the computation kind, plan digest, selected engine, environment, and code provenance. The plan digest covers normalized content and model/numeric input digests; separate `spec_sha256` entries preserve raw-file integrity. The plan format is internal and is not a public authoring interface.

The result records the actual engine, method, dimensions, device, and solver diagnostics. `artifacts/output.npz` holds `states`, `times_ns`, and `observable.<name>` arrays for dynamics, or `energies_ghz` for spectra; its digest is checked before completion. Computation completion describes execution only; checks run separately on request.

CPU dynamiqs jobs require `extras = ["differentiable"]` in `[tool.entanglix.runner]` in `pyproject.toml`. Installing an extra only in the development environment does not configure the job environment. Keep one extras set for all jobs sharing `.venv-runner/`; no CUDA extra is needed for CPU jobs.

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
load fails execution without check outcomes. There is no harness-imposed dimension
cap. The [dimension-48 exact regression](../../tests/test_corrector.py) and
[dimension-64 cross-engine case](../../tests/test_engine_acceptance.py) are engine
acceptance evidence, not ceilings.

```sh
uv run --locked --exact entanglix check research_runs/<system_id>/<run_id>
uv run --locked --exact entanglix check research_runs/<system_id>/<run_id> integrity segments diagnostics --timeout 60
uv run --locked --exact entanglix read-run research_runs/<system_id>/<run_id>
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

Reports identify scopes, tolerances, residuals, sampled windows and coverage, refined
engine/dimensions/settings, and inapplicable reasons. `--timeout SECONDS` and
`abort-run` bound execution. Failed, timed-out, aborted or interrupted check attempts
report no outcomes. A passing segment establishes only local agreement: no built-in
check measures whole-window error or the rotating-wave approximation.

`read-run` lists linked verification attempts with status, policy version, source
identity match, and completed check reports. Earlier reports remain history under
their recorded policy. A newer failure does not erase earlier outcomes or substitute
them for its missing report.

MCP reads summarize nested check details while retaining outcomes, scopes,
coverage, policy versions and source identity matches. Sweep points retain their
coordinates, metrics and any recorded error. Log tails remain complete. Manifests
name omitted provenance fields in `detail_omitted`; oversized metrics are also
marked as omitted. Use `read_run(path, detail=True)` through MCP
for full reports and point details; CLI and Python backing reads remain complete.
Large full check or sweep reads can exceed a runtime tool-reply limit. The complete
report is retained in the run directory: read `result.toml`, point results under
`artifacts/`, and their archives with the session's file tools. MCP `list_runs` uses
the same paged table form as investigation history; CLI listing stays complete.
See the [MCP response reference](mcp.md#investigation-views-and-tables).

## Limits, interruption, and recovery

`run-script --timeout SECONDS` limits execution from publication of the start record, including lingering processes after the main job exits. The default has no timeout; preparation is outside this timer. The timeout must be finite and positive. Termination has a fixed two-second grace period. The supervisor polls at 10 ms intervals; scheduling can delay observation. At expiry it sends `SIGTERM` to the job group and, after the grace interval, `SIGKILL` if needed. It waits for the group to disappear, then records `aborted/130`, actual wall time including termination, and the reason in `run.log`.

`abort-run PATH` (Python: `abort_run(path)`) publishes `runner/abort.toml`, signals only that attempt's recorded process group, and uses the same grace before escalation. Repeated requests retain the first event. The supervisor writes the terminal record; the abort command reports a request, not confirmed completion. Inspect the run to confirm its outcome. Ctrl-C (`SIGINT`), `SIGTERM`, or `SIGHUP` sent to the supervising CLI command also stops the job group and records `aborted/130`. The supervisor watches the abort request so it can finish cancellation if the requesting command is interrupted. Abrupt runner death, such as `SIGKILL`, leaves no terminal record and may leave the job running.

For an interrupted attempt:

1. Inspect it with `read-run`; retained records and the log tail show what was observed.
2. If its orphaned job group is still active, use `abort-run` and inspect again. Do not reconcile a live or unknown group. A paused live supervisor also retains ownership.
3. Once the owner and group are gone, run `entanglix reconcile-run PATH --state "last observed state and recovery action"`. This calls `record_reconciliation`, retains uncertain usage, and never creates `FINISHED`.
4. Use `rerun-script PATH` to launch the frozen script, system, referenced assets, and additional inputs under a new ID linked to the earlier attempt. Every frozen input digest is checked before launch. The old attempt is preserved; partial outputs are not resumed.

Reruns retain the earlier timeout; `--timeout SECONDS` overrides the timeout and `--runs-root PATH` changes the destination. Python callers use `rerun_script(path, ...)`. Rerunning an active or unknown attempt is refused. Reconstruct the recorded code and environment separately as described below: reruns use the current checkout and configuration and record their actual provenance. Attempts without recorded `[launch]` metadata require manual replay of `spec/script.py`, `spec/system.toml`, and additional `--input` files with `--parent-run ID`.

### Local GPU ownership

For a CUDA simulation or sweep, the runner resolves the single visible device through `nvidia-smi`, then takes a nonblocking file lock before its launch check initializes CUDA and before admission creates an attempt. The lock lives under `$XDG_STATE_HOME/entanglix/gpu/` (default `~/.local/state/entanglix/gpu/`), shared by runner instances using that OS account and state directory, independent of their runs roots. A held lock, a live recorded process group, or an allocation awaiting reconciliation refuses the launch without an attempt, investigation charge, or checkpoint progress. A dry run takes the same lock for its check and releases it on return; it reserves no future launch.

The stable `<UUID>.lock` file is never unlinked. Once the attempt exists, `<UUID>.toml` records its absolute run path and, once started, process group. A failure of the launch check, admission, or manifest creation releases ownership. A failure after manifest publication retains a failed attempt.

The launch-check child inherits the GPU descriptor. The local executor then passes both the attempt and GPU descriptors to its job, retaining device ownership continuously through `FINISHED` and closing without explicit unlocking. A surviving inheriting child therefore blocks reuse after runner death. Once the lock becomes free, a live or indeterminate recorded group still blocks reuse. An allocation without `FINISHED` requires explicit reconciliation even after the group disappears; missing history also blocks reuse. Completed allocation metadata is removed when the lease closes; reconciled metadata can be replaced by a subsequent owner. Lock files remain. This is cooperative ownership, not device containment or a scheduler.

## Location and identity

The default runs root is `research_runs/` relative to the caller's working directory. `create_run(runs_root=...)` overrides it explicitly and returns a `RunHandle`; its `.path` names the attempt directory:

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
```

Run IDs have the form `r20260930T153012123456Z-<8 hexadecimal digits>`: a UTC timestamp with microseconds and a random suffix. Creation is exclusive; an existing directory is never reused. System IDs must start with an ASCII letter or digit and contain only letters, digits, `.`, `_`, and `-`. Parent run IDs preserve lineage; a rerun gets a new ID.

Every research run, including an authored script, requires a validated system spec frozen as `spec/system.toml`. The caller supplies all referenced specs, numeric assets, scripts, and data as a mapping of relative destination names to source paths. Names must be UTF-8 encodable, cannot be absolute or contain `..`, and must not alias another file or use a file as a directory; `spec/code/` is reserved for provenance. Files are copied, hashed, and made read-only, so subsequent source edits do not change the attempt. Validation of the scientific model and collection of its referenced files belong to the launch gate.

## Record fields and ownership

Runner records are published once and made read-only. Jobs own `result.toml` and `artifacts/`; runner paths are reserved. This is a cooperative interface, not containment of arbitrary code running as the same OS user.

| Record | Fields and meaning |
|---|---|
| `run.toml` | `record_version = 1`; `run_id`, `system_id`, `protocol`, UTC `created_at`, `parent_run_ids`, declared `needs`, selected `engine`, `device` (`cpu` or `cuda`), `executor = "local"`, `spec_sha256`, `code`, `env_hash`, and `environment`. Script `[launch]` records explicit additional `input_names` and optional `timeout_seconds`; absence means no timeout |
| `runner/started.toml` | `record_version`, UTC `started_at`, positive `process_group`, and `gpu_devices` (device identifiers; empty for CPU). Published before computation starts |
| `runner/abort.toml` | UTC `requested_at`, published once by the abort operation; a request, not a terminal outcome |
| `FINISHED` | `record_version`, `status`, `exit_code`, UTC `ended_at`, and `usage`: nonnegative finite `wall_seconds`, `gpu_seconds`, and `uncertain = false` |
| `runner/reconciled.toml` | `record_version`, UTC `observed_at`, `last_observed_state`, and `usage.uncertain = true`. When a start record exists, includes its `started_at`, `process_group`, and `gpu_devices`; otherwise these fields are absent. Unknown usage is not replaced with zero |
| `result.toml` | A protocol-specific TOML table containing metrics, actual engine/solver information, and artifact references with digests. Trusted check outcomes come only from linked verification results, which carry the bound per-check report |

`run.toml` is the first published file. Inputs are then frozen before the caller may start computation. UTF-8 serialization, missing-input errors, normalized-path duplicates, and file/directory conflicts are checked before creating the attempt directory. On a handled failure before manifest publication, the exclusively created attempt directory is removed because no run was admitted. If preparation fails after publication, `create_run` publishes `FINISHED` with `failed/64` and zero job usage while holding ownership, then raises `RunCreationError` carrying the attempt path in `.run`. If storage prevents that terminal write, the original error retains the path and an explanatory note; the attempt remains incomplete and can be reconciled without a start record. Creation failures close the owner handle before raising.

`publish_toml` serializes before writing, flushes the temporary file, publishes atomically, and flushes the containing directory. Its default refuses to replace any existing record, including concurrent publication by another caller. A job publishes its result with `publish_toml(path, result, replace=True)`, using temporary-file rename; a failed serialization leaves the previous result intact.

`create_run` takes an exclusive `flock` on the attempt directory immediately after creating it and before publishing `run.toml`. Its returned handle retains that ownership. `record_start(handle, ...)` and `record_finish(handle, ...)` require the open handle and do not reacquire the lock. The supervising caller keeps it open through publication of `FINISHED`, then closes it, or uses it as a context manager. Closing the handle only closes its descriptor; it does not issue an explicit unlock that could release an inheriting job's ownership.

For process launch, pass `handle.fileno()` to the job through `subprocess.Popen(..., pass_fds=(handle.fileno(),))`. The shared lock remains held until the runner and every inheriting process close their copies. This attempt lock applies to CPU and GPU runs; the per-device GPU allocation lock is separate.

`record_reconciliation(path, ...)` tries the directory lock without waiting. It refuses while the runner or an inheriting job holds ownership, including when the job has exited but its runner has not published `FINISHED`. After obtaining ownership, it refuses a finished or unadmitted directory and checks any recorded process group with `os.killpg(group, 0)`. A live group, or permission failure that prevents determining liveness, blocks reconciliation. Only when both ownership and process checks permit it does reconciliation publish its record while holding the lock. No start record is required; such an attempt can be reconciled only after its owner handle closes. Direct calls to the generic publisher do not enforce these ownership rules.

## Environment and code provenance

`capture_environment(lockfile, extras=..., gpu=...)` records the lockfile SHA-256, Python version, OS, architecture, libc name/version, sorted declared extras, and supplied GPU observations. Kernel releases do not participate in the identity. The function does not import GPU packages or probe the driver. The `gpu` fields are `driver`, `cuda` (the observed CUDA runtime version), and `cuquantum` (the observed native library version); unknown keys are rejected and unavailable observations are explicitly `"unobserved"`. Missing GPU observations do not prevent CPU record creation. Installed package versions are pinned by the lockfile; they do not establish runtime versions or GPU availability.

`env_hash` is SHA-256 of the environment table before the hash is added, serialized as UTF-8 JSON with sorted keys and compact separators. Lockfile, Python, platform, extras, and GPU observations therefore participate in the identity. Supported execution must use `uv run --locked --exact` with the same extras set for all jobs sharing an environment; the record API itself does not sync or certify that environment.

`check_gpu_environment(device_uuid)` checks a selected GPU in the caller's exact environment with the `cuquantum` extra. It matches NVIDIA-SMI's canonical UUID to a CUDA-visible device, observes the driver and loaded CUDA/native cuDensityMat versions, and initializes the native library on that device. Failures explain device visibility, driver or extra requirements. CPU execution does not call this function or import GPU packages.

Pass the returned `gpu` table to `capture_environment(..., gpu=observations["gpu"])` and retain the resulting identity in the run manifest. The `cuquantum` identity field names the loaded native cuDensityMat library version. The returned device UUID/index are dispatch data, separate from `env_hash`. There is no pre-launch memory estimate. Simulation dispatch connects this check to inherited allocation ownership; CPU checks run only on request after completed execution.

`capture_code(repository)` requires a Git repository with a committed `HEAD`. The code table records `revision`, `dirty`, `patch_sha256`, and `untracked_sha256`. `changes.patch` contains the combined staged and unstaged diff from `HEAD`, including binary changes, with color disabled and fixed `a/` and `b/` prefixes regardless of Git configuration; `code/untracked/` retains all non-ignored untracked files. Untracked symlinks are rejected rather than silently copied as regular files. Ignored scripts and data must be supplied explicitly as inputs. Capture a stable working tree before dispatch.

The patch and untracked contents have a combined default limit of 16 MiB, configurable through `max_snapshot_bytes`. The limit is checked from file sizes before untracked files are read. Exceeding it fails with a diagnostic instead of omitting files. Keep unrelated generated data ignored, supply research data explicitly, or raise the limit for a larger source snapshot.

To reconstruct code, check out the recorded revision in a separate checkout, apply `spec/code/changes.patch` with `git apply`, and restore the untracked files to their recorded relative paths. Use the frozen specs and data and an environment with matching identity. Reproduction comparisons use determinism where supported and declared numerical tolerances otherwise.

## Execution outcomes

The record API and CPU launcher distinguish pre-start failure, observed job outcomes, and reconciliation:

- The launcher returns `64` with a diagnostic for invalid arguments or a failed pre-launch gate before `run.toml` exists, without creating a run. The Python API raises exceptions for invalid inputs.
- After creation, a supervising caller records known failures rather than deleting the attempt. Before a start record exists, `failed/64` or cancellation as `aborted/130` with zero job wall time is valid; any GPU allocation time is still charged, including a failed spawn. Job outcomes require a start record.
- Observed successful execution uses `completed` and code `0`; a job exiting `130`, a timeout, or runner-triggered cancellation is `aborted/130`. Other job failures preserve exit codes in `0–255`. For signal death outside a requested cancellation, the launcher converts Python's negative subprocess return code to `128 + signal` (for example, `SIGKILL` becomes `137`); `record_finish` requires the normalized code and rejects inconsistent status/code pairs.
- An observed allocation failure is identified as out of memory in the diagnostic, including a launch-gate failure before an attempt exists. A signal death the runner did not request is reported as that signal with its cause unknown; for `SIGKILL` the diagnostic names exhausted memory as a possible cause only. A positive shell-encoded status alone cannot distinguish a signal from an explicit job exit with that code, and the diagnostic states that uncertainty. If the supervisor also dies, the interruption and reconciliation rules apply.
- `wall_seconds` is elapsed job time from publication of the start record until the supervisor observes the whole process group's termination, measured once with a monotonic clock. Input preparation and result validation are excluded; a known pre-start preparation failure or cancellation records zero. Allocated `gpu_seconds` means allocated devices multiplied by allocation duration, including the launch check and failed preparation for an admitted attempt; CPU jobs record zero. The caller supplies these observations; the record API does not measure them.
- Runner loss leaves no `FINISHED`, and a surviving job may still own the attempt and GPU. Reconciliation, described above, records uncertainty without fabricating completion.

Without a valid `FINISHED`, artifacts cannot substantiate a completed scientific result. With `FINISHED`, successful execution still does not establish verification. A sweep is one run with per-point artifacts; interrupted partial results are retained, and a new linked rerun recomputes the grid.

See the [harness contract](../../HARNESS_CONTRACT.md) for execution and verification guarantees as the corresponding capabilities become available.

## Detached launches and waiting

Job commands accept `--detach`: `simulate`, `sweep`, `run-script`, their rerun commands, and named `check` requests. The command returns the admitted run ID and path; a separate supervisor retains ownership, enforces the job timeout and investigation deadline, and writes the terminal record. Closing the requesting terminal or losing its response does not stop that supervisor. Use `abort-run` to cancel an admitted job. Dry-run previews and check listings remain synchronous and do not accept `--detach`.

```sh
uv run --locked --exact entanglix simulate examples/readout/readout.toml --runs-root research_runs --detach
uv run --locked --exact entanglix read-run research_runs/SYSTEM_ID/RUN_ID --wait 60
```

`read-run --wait SECONDS` returns when the attempt completes, fails, aborts, loses ownership, or the requested wait expires. Seconds must be finite and non-negative; the command line applies no maximum. An expired or interrupted wait leaves the job running. Reads and cancellation can run concurrently; waits poll at intervals of at most 0.25 seconds. Ctrl-C returns exit 130 without a traceback, including while a detached launch awaits admission.

Before starting its supervisor, a detached request is recorded under `<runs_root>/_launches/<launch_id>/request.toml`. The record holds the operation, absolute input paths, and investigation ID when applicable. The launch directory's inherited lock tracks its owner. The admitted run's manifest carries `launch_id`; `attempt.toml` in the launch directory records the intended relative run path before the manifest is published; the link alone does not establish admission. `supervisor.log` retains launch diagnostics, and a refused request can carry `refused.toml` with its reason. Refusals create no attempt or investigation usage.

If an admission response is lost, use `list-runs` to see admitted attempts and unresolved launches. `read-run <launch_directory>` recovers one request as `pending`, `admitted` (with its run ID), or `refused`. With `--wait`, it waits for admission resolution rather than job completion. Recovery checks ownership and the linked manifest without scanning other attempts; it never submits another job. A supervisor lost after publication leaves an interrupted attempt for the normal reconciliation path. A request whose owner disappears before publication is definitively non-admitted.

`read-investigation` lists admitted attempts with their base, varied values, revision, and status, along with its unresolved launches. Pending requests consume no allowance; accounting starts at manifest publication.
