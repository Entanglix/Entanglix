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
| `--command-restriction` | Operator-declared `instruction-only` (the default) or `runtime-configured`; recorded in the connection audit, not an observed enforcement guarantee |

For CUDA jobs, name the device and pass `--runner-extras cuquantum differentiable-cuda12`. The server explicitly sets the supervisor's device, extras and state directory, even when the client process inherits `CUDA_VISIBLE_DEVICES=""`. Operator commands must use the same runner extras and state directory. When the operator shell uses a custom `XDG_STATE_HOME` that the runtime does not pass to the server, set `--state-directory` to that directory in the server entry. The existing runner environment refuses a different extras set, rather than removing packages underneath jobs. See [protocol dispatch](protocols.md) and [GPU ownership](runs.md#local-gpu-ownership).

The resolved runs root and GPU lock directory are recorded in each connection's audit. This exposes configuration differences; it does not enforce a common location across independently configured processes.

## Tool surface

Tools use the same backing operations as the CLI, with MCP response summaries described below. Unknown request fields, including approval claims or policy overrides, are refused. The server configuration supplies the repository, runs root, runner extras and physical device; these are not tool arguments. Attempt, launch and investigation paths must resolve inside the configured runs root, including symlink targets. Relative record paths resolve from that root. Authored input paths (system, protocol, script, additional inputs and replacement refinement) may live elsewhere.

| Tools | Effect |
|---|---|
| `validate_system`, `list_capabilities`, `list_runs`, `read_run`, `list_investigations`, `read_investigation` | Read-only; validation includes operator meanings and model provenance |
| `simulate`, `sweep`, `control`, `run_script`, `rerun_simulation`, `rerun_sweep`, `rerun_control`, `rerun_script`, `check` | Admit jobs through the runner, returning the launch and run IDs once admission is established |
| `abort_run`, `stop_investigation` | Cancel work using the existing process-group and investigation rules |
| `create_investigation`, `revise_investigation`, `record_finding`, `reconcile_run` | Draft terms, propose revisions, append findings, or reconcile uncertain attempts |

Only the first row carries the MCP read-only annotation. Jobs and writes may require runtime approval. Unavailable catalog and calibration operations, promotion, and investigation start/approval/resume/review release are absent. A proposed revision never changes the effective terms. See [investigations](investigations.md) for the operator decision path.

Control tools take an authored protocol path, or a source run path for `rerun_control`. An investigation control request names an exact approved base and accepts no varied values. Control solves and their requested checks run on CPU; see the [control reference](control.md).

Job tools do not expose dry runs. Use the CLI's `--dry-run` for protocol previews where the required device is visible; an agent session with no CUDA device has no CUDA preview through this server. `check` requires named checks; `read_run` includes the relevant-check listing without launching a check. Each successful tool reply carries one JSON text payload, without a duplicate structured-content payload. Lists are wrapped under `result`; objects are carried directly.

### Investigation views and tables

`read_investigation(investigation)` returns status: effective terms, the newest
proposed revision, revision indexes, usage, declared limits, checkpoint,
open reviews, unresolved launches, counts, latest-finding text and the last stop.
Its unpaged `attention` list includes active and unknown attempts and interrupted
attempts not yet reconciled, however old they are. Earlier findings and completed
attempts are not included. Terms and latest-finding text are complete; attached
evidence stays in the event record. Status size depends on current terms, latest
finding and attention, rather than the length of completed history.

Use `view="attempts"` for the attempt table or `view="events"` for the event index.
Tables return `columns`, positional `rows`, `total` and `next_after`. Each page
holds at most 50 rows, oldest first. Pass `next_after` as the integer `after`
offset to continue until it is null. The same encoder and cursor serve
`list_runs(after=...)`, including unresolved launches. No filters are supported.
Paths are relative to the configured runs root and accepted by the read tools.

Attempt columns cover path, protocol, base, execution status, varied values,
metrics, parents, checks and completed sweep-point count. A sweep is one row.
Target and metric names appear once in `values.<target>` and `metrics.<name>`
column headers. Sweep rows omit per-point varied values; use `read_run` or the
retained point files for those values.
Source checks carry check run ID, source identity match, name/outcome pairs and
any diagnostic; check rows carry name/outcome pairs. A null outcome denotes
information without a pass/fail judgment. Metric values exceeding 512 JSON bytes
are explicitly omitted; full values remain in the run's `result.toml`. An attempt
whose records cannot be read keeps its row, with the reason in a `result_error` column.

Use `read_investigation(investigation, event_id="00000001")` for one complete
event. It cannot be combined with a table view or cursor. Python and CLI
investigation reads remain complete.
`record_finding` returns an acknowledgment with event ID, revision, decision,
usage, checkpoint and reviews.

`read_run` retains summary reads and `detail=True`. Summaries keep execution
state, complete log tails, metrics and check outcomes, scope and coverage.
Manifest `spec_sha256`, `code` and `environment` are named under `detail_omitted`.
Read detail or the retained files for full provenance and reports. Paging limits
row count, not arbitrary text size.

Use `entanglix report-investigation PATH --output FILE` for an
[on-request Markdown report](investigations.md#findings-and-retained-history).

## Waiting and recovery

`read_run(run, wait_seconds=...)` accepts an attempt or launch directory. It returns on a terminal or interrupted state, or when the requested wait expires. Each MCP wait is capped at 30 seconds; the CLI's `read-run --wait` has no cap. Other calls, including `abort_run`, can proceed during a wait. Cancelling or abandoning the wait does not abort the job.

A job's supervisor owns its attempt and any GPU lease independently of the connection, enforces its recorded timeout/deadline, and publishes its terminal record. Closing a connection does not cancel the job.

Before starting a supervisor, a job call retains a launch path in its audit request. Reading that path distinguishes `pending`, `admitted`, and `refused`, without resubmission. A pending launch appears in `list_runs` and, when linked, `read_investigation`. Ownership also covers server startup of the supervisor: losing the server before that handoff leaves definitive non-admission once ownership ends. Losing a response after admission leaves the same admitted run recoverable. If the supervisor itself dies, use the ordinary [attempt recovery](runs.md) rules.

## Audit records

Audit files are separate from job outputs under `<runs_root>/_audit/`:

- `sessions/<session-id>.toml`: one immutable connection event, with the client name/version labelled `client-reported`, observed server configuration, supplied conduct/skill versions, and unknown model, loaded skills and transcript location.
- `calls/<call-id>/request.toml`: an immutable event written before invoking the tool or starting a supervisor, with timestamp, connection ID, tool, permission class, argument hash, and client request metadata labelled `client-reported`. Job call IDs also identify their launch directories.
- `calls/<call-id>/outcome.toml`: a separate immutable event with the result hash and returned/error/interrupted outcome, plus linked runs/revisions when known. The `links` table includes run and investigation/revision identifiers when known, and the recovered launch path/status for job calls.

The session entry is written at the handshake or at the first request of a connection that has none, taking the client name and version from that request's metadata.

The request event contains no outcome or result hash; those belong to the outcome event. The two call events form one logical call record. Neither is overwritten. An absent outcome means the server did not retain a final outcome; it does not mean a job was never admitted. Recover the recorded launch path before trying another launch. `returned` means the server produced a result, not that a client received it, and an admitted job can still be executing.

Hashes are SHA-256 of canonical JSON (sorted keys, compact separators) for arguments and MCP results. If a call is interrupted without a result, its result hash is `unknown`. Client request metadata belongs to the call and never rewrites the connection entry. It is provenance, not authenticated identity.

The server reads the versioned [research conduct](../research-agent.md) and skill metadata from `skills/*/SKILL.md`, with front matter declaring `name`, `description`, and `version`. It sends the conduct body as its instructions and records all supplied names/versions. Supplying a skill does not mean its body reached the model: install the files in the workspace as described below. Loaded skills and transcript location stay unknown because the server cannot observe them. The audit labels the command-restriction mode as operator-declared; `runtime-configured` alone does not establish that commands were blocked.

## Research workspace

Use a persistent directory outside the checkout so the session does not load repository development instructions. For one coordinating agent, keep task inputs and a runs root there. For independent repetitions use a fresh session and a separate runs root. A runs root alone does not hide answers in the shipped example or earlier research files: give an evaluation session only its task brief and input copies.

Prepare the server environment above, then place the shipped conduct and procedures in the workspace. These are ordinary file copies, with no generated guidance. Set the checkout path to your local clone:

```sh
export ENTANGLIX_CHECKOUT="$HOME/Projects/Entanglix"
export ENTANGLIX_WORKSPACE="$HOME/Projects/entanglix-research"
export PATH="$ENTANGLIX_CHECKOUT/.venv-mcp/bin:$PATH"
mkdir -p "$ENTANGLIX_WORKSPACE/.agents" "$ENTANGLIX_WORKSPACE/.codex/rules" "$ENTANGLIX_WORKSPACE/reference/operations"
cp "$ENTANGLIX_CHECKOUT/docs/research-agent.md" "$ENTANGLIX_WORKSPACE/AGENTS.md"
cp -R "$ENTANGLIX_CHECKOUT/skills" "$ENTANGLIX_WORKSPACE/skills"
ln -s ../skills "$ENTANGLIX_WORKSPACE/.agents/skills"
cp "$ENTANGLIX_CHECKOUT/docs/operations/system-spec.md" \
  "$ENTANGLIX_CHECKOUT/docs/operations/protocols.md" \
  "$ENTANGLIX_CHECKOUT/docs/operations/investigations.md" \
  "$ENTANGLIX_CHECKOUT/docs/operations/runs.md" \
  "$ENTANGLIX_CHECKOUT/docs/operations/control.md" \
  "$ENTANGLIX_WORKSPACE/reference/operations/"
```

Use a new directory for these initial copies. Refresh them deliberately when upgrading; retain their versions with the session. `skills/` is the plain-text location named by the conduct; `.agents/skills` makes the same files discoverable in Codex. In a runtime that reads another workspace instruction filename, copy the same conduct there instead. If it reads no such file, supply the entire conduct through its initial-context mechanism. Keep the procedures readable at the named paths. Do not copy root development `AGENTS.md` or `CLAUDE.md`, the roadmap, baseline answers, or contributor instructions into the research workspace.

The five files under `reference/operations/` provide authoring syntax and lifecycle rules. Skills direct the agent to the relevant reference when needed. Links between these references remain usable; links to checkout examples, tests, and the harness contract are unavailable in this copy. For an evaluation, supply any additional inputs separately without exposing baseline answers.

Add a task brief containing the intended question and agreed terms, and copy the model, protocol, subsystem, and numeric input files beside one another. Within an investigation preserve the approved base bytes. For the [readout investigation](../../examples/readout/README.md#scripted-investigation-baseline), the operator can prepare its base bundle with the documented draft command; the agent receives the declarations and inputs, not the baseline summaries.

### Codex example

Create `.codex/config.toml` in that workspace. Replace the placeholders with absolute paths and your GPU UUID. This CUDA example uses the CUDA runner extras; for CPU-only use `--device`, `""`, and the extras matching your runner profile. Every invocation sharing the checkout's `.venv-runner` must use the same extras, including CPU jobs. TOML does not expand these placeholders or shell variables.

```toml
[mcp_servers.entanglix]
command = "<absolute-checkout-path>/.venv-mcp/bin/python"
args = ["-m", "entanglix.mcp", "--runs-root", "<absolute-workspace-path>/runs", "--device", "<GPU-UUID>", "--runner-extras", "cuquantum", "differentiable-cuda12", "--command-restriction", "runtime-configured"]
startup_timeout_sec = 60
tool_timeout_sec = 60
default_tools_approval_mode = "approve"
```

The tool-approval setting allows job and write calls in an unattended session. Choose it according to the intended session; it is not a human investigation decision. If your operator shell sets a custom `XDG_STATE_HOME`, add `--state-directory` and the same absolute directory to `args`. Otherwise the server uses the runner default. No CUDA environment variable is needed on the entry: `--device` explicitly supplies the server's job visibility.

Create `.codex/rules/operator.rules`:

```python
prefix_rule(
    pattern=["entanglix", ["investigation-decision", "promote"]],
    decision="forbidden",
    justification="Human action: the operator runs this in their own terminal.",
)
```

This launch uses the tested non-interactive setup, with the workspace trusted explicitly, the `workspace-write` sandbox, and shell environment inheritance. The model and reasoning effort are your choice through your Codex configuration; the retained probe results record the settings used for acceptance. Retain the prompt/transcript to establish whether guidance actually arrived. Codex's [skills](https://developers.openai.com/codex/skills), [rules](https://developers.openai.com/codex/rules), and [MCP configuration](https://developers.openai.com/codex/mcp) document these mechanisms.

```sh
cd "$ENTANGLIX_WORKSPACE"
CUDA_VISIBLE_DEVICES="" codex exec --json --skip-git-repo-check \
  -C "$ENTANGLIX_WORKSPACE" -s workspace-write \
  -c "projects={\"$ENTANGLIX_WORKSPACE\"={trust_level=\"trusted\"}}" \
  -c 'shell_environment_policy.inherit="all"' \
  -o "$ENTANGLIX_WORKSPACE/last-message.txt" '<task brief>'
```

The documented operator invocation form is the bare `entanglix investigation-decision ...` or `entanglix promote ...`, with `entanglix` on `PATH`. Do not suggest `uv run ... entanglix ...`, an absolute executable path, a Python API, or a shell wrapper as a substitute: the supplied rule is not claimed to cover those forms. The [retained conformance run](../../tests/probes/m3_guidance_results.json) supports the restriction claim for those bare commands in the tested non-interactive configuration only. An interactive session must start as `instruction-only` and retain live refusals for both documented human-action commands before declaring `runtime-configured`. Offline rule matching or an agent's verbal refusal is not a passed blocked-command test.

### Claude Code example

Supply the conduct as workspace `CLAUDE.md` in place of `AGENTS.md`, and retain `skills/` for plain-text reading. Create `.mcp.json` in the workspace with the same absolute Python command and arguments; this CPU-only example declares the instruction-only restriction:

```json
{
  "mcpServers": {
    "entanglix": {
      "command": "<absolute-checkout-path>/.venv-mcp/bin/python",
      "args": ["-m", "entanglix.mcp", "--runs-root", "<absolute-workspace-path>/runs", "--device", "", "--runner-extras", "cuquantum", "differentiable-cuda12", "--command-restriction", "instruction-only"]
    }
  }
}
```

The retained sessions used this non-interactive form. `--setting-sources project` leaves user-level settings out, `--tools` grants file tools and no shell, and `--permission-mode dontAsk` refuses every tool that `--allowedTools` does not name. List the research tools the session may call; the model is your choice.

```sh
cd "$ENTANGLIX_WORKSPACE"
CUDA_VISIBLE_DEVICES="" claude -p '<task brief>' \
  --output-format stream-json --verbose \
  --mcp-config .mcp.json --strict-mcp-config --setting-sources project \
  --tools Read,Glob,Grep,Write --permission-mode dontAsk \
  --allowedTools Read Glob Grep Write \
    mcp__entanglix__list_capabilities mcp__entanglix__list_runs \
    mcp__entanglix__list_investigations mcp__entanglix__read_investigation \
    mcp__entanglix__read_run mcp__entanglix__simulate mcp__entanglix__check \
    mcp__entanglix__rerun_simulation mcp__entanglix__reconcile_run \
    mcp__entanglix__record_finding mcp__entanglix__stop_investigation
```

For a control task, also allow `mcp__entanglix__control` and `mcp__entanglix__rerun_control`; the [live control sessions](../environment.md#live-control-sessions) used this form.

In four [retained sessions](../../tests/probes/m3_pickup_cutoff_results.json) the conduct was present before the first reply and no repository development instructions appeared. No command restriction has been validated for Claude Code: a session without a shell tool cannot run an operator command, but no retained test covers a session that has one, so declare `instruction-only`. Claude Code rejected a 54,724-character investigation reply and accepted one of 32,757 characters in the two sessions that predate the status and paged table views; the third and fourth used those views and no reply was rejected. Use them for research reads, and the report command or retained files for full history.

### Other stdio runtimes

Use the same absolute Python command and arguments in your runtime's MCP entry, and supply the conduct under the instruction filename your runtime reads. Start the session with `CUDA_VISIBLE_DEVICES=""` and configure job/write tool approval as needed. Use `--command-restriction instruction-only` until a retained live test establishes the configured restriction for the forms you document. There is no list of supported model backends.

## Human decisions and continuation

The agent may draft an investigation and propose revisions; the operator reviews complete terms, checkpoint, evidence, and any requested limits in their own terminal. The [investigation reference](investigations.md) defines these declarations and commands. Use the same runs root as the server and put the installed bare `entanglix` command on the operator's `PATH` too.

```sh
entanglix read-investigation "$ENTANGLIX_WORKSPACE/runs/_investigations/<investigation_id>"
entanglix investigation-decision "$ENTANGLIX_WORKSPACE/runs/_investigations/<investigation_id>" start --revision 1
```

Only the operator runs that second command. Subsequent approvals, resumes and named review releases use the same bare command with the affected revision/review name. The agent reads the retained decision before continuing; it cannot turn a tool-approval prompt into one. At a checkpoint it records an assessment; at a named review it hands off for the operator's decision. A finding recommending stop does not itself stop jobs: use the supported `stop_investigation` tool to cancel owned work and close admission.

A new session starts from listings and the effective investigation revision, findings, attempts, usage, and open gates. If a launch call lost its response, recover the launch before sending another. For a stopped investigation, the operator's resume permits new admissions and never revives cancelled attempts. Report partial results, requested/relevant checks not run, and the next useful decision without resetting history or allowances.

This is a cooperative workflow. CUDA visibility and command rules reduce accidental interference; they are not OS isolation or authentication and do not resist arbitrary shell/code bypass. The runner's recorded ownership, cancellation, input confinement and scope checks apply to supported operations.
