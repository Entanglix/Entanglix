# Stdio research tools

The optional `entanglix.mcp` module exposes the runner's existing research operations over MCP stdio. A client may use any runtime that supports that transport. Human decisions remain in the operator CLI; approving a runtime tool call does not start an investigation or approve proposed terms.

## Server process and configuration

Give the server a separate environment so an exact sync for an operator command cannot remove its SDK while it is running. From the checkout:

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.venv-mcp" uv sync --locked --exact --extra mcp
.venv-mcp/bin/python -m entanglix.mcp \
  --runs-root "$HOME/Projects/entanglix-research/runs" \
  --device "" \
  --runner-extras
```

Use the environment's absolute Python path when configuring a client started elsewhere. The command communicates on stdin/stdout; diagnostics go to stderr. Only stdio is provided.

The runs root, device and runner extras are explicit. The state-directory override is optional:

| Setting | Meaning |
|---|---|
| `--runs-root` | Persistent directory for attempts, investigations, launch records and audit records; use the same directory in operator commands |
| `--device` | Physical GPU UUID or `nvidia-smi` index for runner-launched CUDA jobs; an empty string supplies no GPU |
| `--runner-extras` | Space-separated extras for the separate `.venv-runner` job environment; the bare flag selects none |
| `--state-directory` | Optional shared state directory; defaults to the runner's `XDG_STATE_HOME`, or `~/.local/state` when unset. GPU locks live under its `entanglix/gpu/` directory |

For CUDA jobs, name the device and pass `--runner-extras cuquantum differentiable-cuda12`. The server explicitly sets the supervisor's device, extras and state directory, even when the client process inherits `CUDA_VISIBLE_DEVICES=""`. Operator commands must use the same runner extras and state directory. When the operator shell uses a custom `XDG_STATE_HOME` that the runtime does not pass to the server, set `--state-directory` to that directory in the server entry. The existing runner environment refuses a different extras set, rather than removing packages underneath jobs. See [protocol dispatch](protocols.md) and [GPU ownership](runs.md#local-gpu-ownership).

The resolved runs root and GPU lock directory are recorded in each connection's audit. This exposes configuration differences; it does not enforce a common location across independently configured processes.

## Tool surface

Tool arguments and result objects use the same backing operations as the CLI. Unknown request fields, including approval claims or policy overrides, are refused. The server configuration supplies the repository, runs root, runner extras and physical device; these are not tool arguments. Attempt, launch and investigation paths must resolve inside the configured runs root, including symlink targets. Relative record paths resolve from that root. Authored input paths (system, protocol, script, additional inputs and replacement refinement) may live elsewhere.

| Tools | Effect |
|---|---|
| `validate_system`, `list_capabilities`, `list_runs`, `read_run`, `list_investigations`, `read_investigation` | Read-only; validation includes operator meanings and model provenance |
| `simulate`, `sweep`, `run_script`, `rerun_simulation`, `rerun_sweep`, `rerun_script`, `check` | Admit jobs through the runner, returning the launch and run IDs once admission is established |
| `abort_run`, `stop_investigation` | Cancel work using the existing process-group and investigation rules |
| `create_investigation`, `revise_investigation`, `record_finding`, `reconcile_run` | Draft terms, propose revisions, append findings, or reconcile uncertain attempts |

Only the first row carries the MCP read-only annotation. Jobs and writes may require runtime approval. Unavailable catalog, control and calibration operations, promotion, and investigation start/approval/resume/review release are absent. A proposed revision never changes the effective terms. See [investigations](investigations.md) for the operator decision path.

Job tools do not expose dry runs. Use the CLI's `--dry-run` for protocol previews where the required device is visible; an agent session with no CUDA device has no CUDA preview through this server. `check` requires named checks; `read_run` includes the relevant-check listing without launching a check. A list result is carried in MCP structured content under `result`; object results are carried directly.

## Waiting and recovery

`read_run(run, wait_seconds=...)` accepts an attempt or launch directory. It returns on a terminal or interrupted state, or when the requested wait expires. Each MCP wait is capped at 30 seconds; the CLI's `read-run --wait` has no cap. Other calls, including `abort_run`, can proceed during a wait. Cancelling or abandoning the wait does not abort the job.

A job's supervisor owns its attempt and any GPU lease independently of the connection, enforces its recorded timeout/deadline, and publishes its terminal record. Closing a connection does not cancel the job.

Before starting a supervisor, a job call retains a launch path in its audit request. Reading that path distinguishes `pending`, `admitted`, and `refused`, without resubmission. A pending launch appears in `list_runs` and, when linked, `read_investigation`. Ownership also covers server startup of the supervisor: losing the server before that handoff leaves definitive non-admission once ownership ends. Losing a response after admission leaves the same admitted run recoverable. If the supervisor itself dies, use the ordinary [attempt recovery](runs.md) rules.

## Audit records

Audit files are separate from job outputs under `<runs_root>/_audit/`:

- `sessions/<session-id>.toml`: one immutable connection event, with the client name/version labelled `client-reported`, observed server configuration, supplied conduct/skill versions, and unknown model, loaded skills and transcript location.
- `calls/<call-id>/request.toml`: an immutable event written before invoking the tool or starting a supervisor, with timestamp, connection ID, tool, permission class, argument hash, and client request metadata labelled `client-reported`. Job call IDs also identify their launch directories.
- `calls/<call-id>/outcome.toml`: a separate immutable event with the result hash and returned/error/interrupted outcome, plus linked runs/revisions when known. The `links` table includes run and investigation/revision identifiers when known, and the recovered launch path/status for job calls.

The request event contains no outcome or result hash; those belong to the outcome event. The two call events form one logical call record. Neither is overwritten. An absent outcome means the server did not retain a final outcome; it does not mean a job was never admitted. Recover the recorded launch path before trying another launch. `returned` means the server produced a result, not that a client received it, and an admitted job can still be executing.

Hashes are SHA-256 of canonical JSON (sorted keys, compact separators) for arguments and MCP results. If a call is interrupted without a result, its result hash is `unknown`. Client request metadata belongs to the call and never rewrites the connection entry. It is provenance, not authenticated identity.

The server reads conduct from `docs/research-agent.md` and skill metadata from `skills/*/SKILL.md` when those files have front matter declaring `name`, `description`, and `version`. Versioned conduct is sent as server instructions. Draft or absent files are not supplied; the audit currently records `none supplied`. Loaded skills and transcript location stay unknown because the server cannot observe them. Files and runtime instruction delivery are separate from this transport.
