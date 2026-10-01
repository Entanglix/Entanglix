# Run records

`entanglix.runner` launches authored CPU scripts with optional timeouts, cancellation, inspection, recovery, and linked reruns. It freezes inputs and code provenance and publishes records atomically. GPU allocation locks are available to local executor code; scientific GPU dispatch, built-in protocols, and independent verification are not available yet. Reconciliation checks ownership and recorded process-group liveness itself.

## Authored CPU jobs

```sh
uv run --locked --exact entanglix run-script examples/recorded_decay.py --system examples/two-level/system.toml
uv run --locked --exact entanglix capabilities
uv run --locked --exact entanglix list-runs
uv run --locked --exact entanglix read-run research_runs/<system_id>/<run_id>
uv run --locked --exact entanglix abort-run research_runs/<system_id>/<run_id>
uv run --locked --exact entanglix rerun-script research_runs/<system_id>/<run_id>
```

Commands print JSON. `run-script` waits for the job and returns its exit code. Use `--runs-root PATH` to choose the output root, repeated `--input NAME=PATH` to freeze additional files under `spec/`, and repeated `--parent-run ID` to record lineage. The corresponding Python operations are `run_script(system, script, ...)`, `list_capabilities()`, `list_runs(root)`, and `read_run(path)`. Capabilities inspect package metadata in the runner environment without creating or synchronizing it; before its first installation, no installed engines are reported. Engine names match the adapters, with `device` reported separately.

Before admission, a subprocess in the locked, exactly synced environment constructs the CPU model, checks script syntax and imports, and records the environment identity. Referenced subsystem specs and numeric archives are collected automatically; the standalone script is frozen as `spec/script.py`. Additional Python inputs receive the same static import check. Relative imports are unsupported for standalone scripts. The allowlist lives in the project's import-boundary configuration; scanning is preflight under the cooperative trust model, not containment. Changed inputs between validation and freezing fail preparation before computation.

Jobs sharing the checkout use the same extras set, configured once in `pyproject.toml` as `[tool.entanglix.runner]` with `extras = [...]`; the default is empty. Each check and job uses `uv run --locked --exact` with that set and an absolute `UV_PROJECT_ENVIRONMENT` pointing to the ignored `.venv-runner/` in the checkout. The first launch installs this environment; exact synchronization leaves the development `.venv` and its extras alone. CPU jobs receive `CUDA_VISIBLE_DEVICES=""` and `JAX_PLATFORMS=cpu`. Keep the checkout and configuration stable while jobs run; use a separate checkout when reconstructing an earlier code revision.

The job runs noninteractively in its own process group, with `artifacts/` as its working directory so ordinary relative outputs land in the job's area. It inherits the owner's descriptor and waits until `runner/started.toml` has been published before executing the frozen script. Output and diagnostics go to `run.log`. `ENTANGLIX_RUN_DIR` names the attempt and `ENTANGLIX_SYSTEM_SPEC` its frozen system spec; additional data is under `<ENTANGLIX_RUN_DIR>/spec/`.

Publish `<ENTANGLIX_RUN_DIR>/result.toml` atomically with `publish_toml(path, result, replace=True)`. An authored result requires an actual `engine` name and a `metrics` table; solver settings belong in the result when applicable. Optional `[artifacts.<name>]` entries supply `path` and `sha256`; paths are relative to the attempt directory, must stay inside `artifacts/`, and their digests are checked before completion. See [the recorded decay example](../../examples/recorded_decay.py). A missing or malformed result, a digest mismatch, or a script-written `verdict` fails orchestration with code `64`. Nonzero job exits retain their code and logs.

An arbitrary script has no structured evolution request for engine selection. Its manifest uses `engine = "authored-script"`; the result names the engine actually used. Scripts have no trusted verification definition and cannot produce verified evidence. Structured engine dispatch accompanies built-in computation plans when available.

Inspection is read-only. `read_run` includes records and the last 8 KiB of `run.log`; it does not determine scientific verification. Malformed results appear as `result_error`; listing retains malformed runner records with a diagnostic. Without `FINISHED`, ownership or a live recorded group means `active`, absence of both means `interrupted`, and indeterminate group liveness means `unknown`. After the main process exits, the supervisor retains ownership and waits for the whole process group to disappear before validating the result and writing `FINISHED`.

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

`acquire_gpu(device_uuid, owner)` returns a `GpuLease` for a canonical `GPU-...` identifier. It takes a nonblocking file lock under `$XDG_STATE_HOME/entanglix/gpu/` (default `~/.local/state/entanglix/gpu/`), shared by runner instances using that OS account and state directory, independent of their runs roots. The stable `<UUID>.lock` file is never unlinked; `<UUID>.toml` records the allocation's absolute run path and, after `bind_process(group)`, process group.

The local executor passes both the attempt and GPU descriptors to its job, retains them through `FINISHED`, and closes them without explicit unlocking. A surviving inheriting job therefore blocks reuse after runner death. Once the lock becomes free, a live or indeterminate recorded group still blocks reuse. An allocation without `FINISHED` requires explicit reconciliation even after the group disappears; missing history also blocks reuse. Completed allocation metadata is removed when the lease closes; reconciled metadata can be replaced by a subsequent owner. Lock files remain. This is cooperative ownership, not device containment or a scheduler.

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
| `result.toml` | A protocol-specific TOML table containing metrics, actual engine/solver information, and artifact references with digests. Protocol results carry no verification verdict; independent verification results may carry one when verification is available |

`run.toml` is the first published file. Inputs are then frozen before the caller may start computation. UTF-8 serialization, missing-input errors, normalized-path duplicates, and file/directory conflicts are checked before creating the attempt directory. On a handled failure before manifest publication, the exclusively created attempt directory is removed because no run was admitted. If preparation fails after publication, `create_run` publishes `FINISHED` with `failed/64` and zero job usage while holding ownership, then raises `RunCreationError` carrying the attempt path in `.run`. If storage prevents that terminal write, the original error retains the path and an explanatory note; the attempt remains incomplete and can be reconciled without a start record. Creation failures close the owner handle before raising.

`publish_toml` serializes before writing, flushes the temporary file, publishes atomically, and flushes the containing directory. Its default refuses to replace any existing record, including concurrent publication by another caller. A job publishes its result with `publish_toml(path, result, replace=True)`, using temporary-file rename; a failed serialization leaves the previous result intact.

`create_run` takes an exclusive `flock` on the attempt directory immediately after creating it and before publishing `run.toml`. Its returned handle retains that ownership. `record_start(handle, ...)` and `record_finish(handle, ...)` require the open handle and do not reacquire the lock. The supervising caller keeps it open through publication of `FINISHED`, then closes it, or uses it as a context manager. Closing the handle only closes its descriptor; it does not issue an explicit unlock that could release an inheriting job's ownership.

For process launch, pass `handle.fileno()` to the job through `subprocess.Popen(..., pass_fds=(handle.fileno(),))`. The shared lock remains held until the runner and every inheriting process close their copies. This attempt lock applies to CPU and GPU runs; the per-device GPU allocation lock is separate.

`record_reconciliation(path, ...)` tries the directory lock without waiting. It refuses while the runner or an inheriting job holds ownership, including when the job has exited but its runner has not published `FINISHED`. After obtaining ownership, it refuses a finished or unadmitted directory and checks any recorded process group with `os.killpg(group, 0)`. A live group, or permission failure that prevents determining liveness, blocks reconciliation. Only when both ownership and process checks permit it does reconciliation publish its record while holding the lock. No start record is required; such an attempt can be reconciled only after its owner handle closes. Direct calls to the generic publisher do not enforce these ownership rules.

## Environment and code provenance

`capture_environment(lockfile, extras=..., gpu=...)` records the lockfile SHA-256, Python version, OS, architecture, libc name/version, sorted declared extras, and supplied GPU observations. Kernel releases do not participate in the identity. The function does not import GPU packages or probe the driver. The `gpu` fields are `driver`, `cuda` (the observed CUDA runtime version), and `cuquantum` (the observed native library version); unknown keys are rejected and unavailable observations are explicitly `"unobserved"`. Missing GPU observations do not prevent CPU record creation. Installed package versions are pinned by the lockfile; they do not establish runtime versions or GPU availability.

`env_hash` is SHA-256 of the environment table before the hash is added, serialized as UTF-8 JSON with sorted keys and compact separators. Lockfile, Python, platform, extras, and GPU observations therefore participate in the identity. Supported execution must use `uv run --locked --exact` with the same extras set for all jobs sharing an environment; the record API itself does not sync or certify that environment.

`capture_code(repository)` requires a Git repository with a committed `HEAD`. The code table records `revision`, `dirty`, `patch_sha256`, and `untracked_sha256`. `changes.patch` contains the combined staged and unstaged diff from `HEAD`, including binary changes, with color disabled and fixed `a/` and `b/` prefixes regardless of Git configuration; `code/untracked/` retains all non-ignored untracked files. Untracked symlinks are rejected rather than silently copied as regular files. Ignored scripts and data must be supplied explicitly as inputs. Capture a stable working tree before dispatch.

The patch and untracked contents have a combined default limit of 16 MiB, configurable through `max_snapshot_bytes`. The limit is checked from file sizes before untracked files are read. Exceeding it fails with a diagnostic instead of omitting files. Keep unrelated generated data ignored, supply research data explicitly, or raise the limit for a larger source snapshot.

To reconstruct code, check out the recorded revision in a separate checkout, apply `spec/code/changes.patch` with `git apply`, and restore the untracked files to their recorded relative paths. Use the frozen specs and data and an environment with matching identity. Reproduction comparisons use determinism where supported and declared numerical tolerances otherwise.

## Execution outcomes

The record API and CPU launcher distinguish pre-start failure, observed job outcomes, and reconciliation:

- The launcher returns `64` with a diagnostic for invalid arguments or a failed pre-launch gate before `run.toml` exists, without creating a run. The Python API raises exceptions for invalid inputs.
- After creation, a supervising caller records known failures rather than deleting the attempt. Before a start record exists, only `failed/64` with zero job wall time is valid; any GPU allocation time is still charged, including a failed spawn. Job outcomes require a start record.
- Observed successful execution uses `completed` and code `0`; a job exiting `130`, a timeout, or runner-triggered cancellation is `aborted/130`. Other job failures preserve exit codes in `0–255`. For signal death outside a requested cancellation, the launcher converts Python's negative subprocess return code to `128 + signal` (for example, `SIGKILL` becomes `137`); `record_finish` requires the normalized code and rejects inconsistent status/code pairs.
- `wall_seconds` is elapsed job time from publication of the start record until the supervisor observes the whole process group's termination, measured once with a monotonic clock. Input preparation and result validation are excluded; a known pre-start preparation failure records zero. Allocated `gpu_seconds` means allocated devices multiplied by allocation duration, including failed work; CPU jobs record zero. The caller supplies these observations; the record API does not measure them.
- Runner loss leaves no `FINISHED`, and a surviving job may still own the attempt and GPU. Reconciliation, described above, records uncertainty without fabricating completion.

Without a valid `FINISHED`, artifacts cannot substantiate a completed scientific result. With `FINISHED`, successful execution still does not establish verification. A sweep is one run with per-point artifacts; interrupted partial results are retained, and a new linked rerun recomputes the grid.

See the [harness contract](../../HARNESS_CONTRACT.md) for execution and verification guarantees as the corresponding capabilities become available.
