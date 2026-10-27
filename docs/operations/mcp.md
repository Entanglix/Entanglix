# Stdio research tools

The optional `entanglix.mcp` module exposes the runner's existing research operations over MCP stdio. A client may use any runtime that supports that transport. Human decisions remain operator CLI commands ([human decisions](#human-decisions-and-continuation)).

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

The runs root, device and runner extras are explicit. The other settings are optional:

| Setting | Meaning |
|---|---|
| `--runs-root` | Persistent directory for attempts, investigations, catalog, launch records and audit records; use the same directory in operator commands |
| `--device` | Physical GPU UUID or `nvidia-smi` index for runner-launched CUDA jobs; an empty string supplies no GPU |
| `--runner-extras` | Space-separated extras for the separate `.venv-runner` job environment; the bare flag selects none |
| `--state-directory` | Optional shared state directory; defaults to the runner's `XDG_STATE_HOME`, or `~/.local/state` when unset. GPU locks live under its `entanglix/gpu/` directory |
| `--max-wait` | Longest `read_run` wait in seconds, finite and positive, 30 by default; recorded in the connection audit. Keep the runtime's per-call tool timeout above it |

For CUDA jobs, name the device and pass `--runner-extras cuquantum differentiable-cuda12`. The server explicitly sets the supervisor's device, extras and state directory, even when the client process inherits `CUDA_VISIBLE_DEVICES=""`. Operator commands must use the same runner extras and state directory. When the operator shell uses a custom `XDG_STATE_HOME` that the runtime does not pass to the server, set `--state-directory` to that directory in the server entry. The existing runner environment refuses a different extras set, rather than removing packages underneath jobs. See [protocol dispatch](protocols.md) and [GPU ownership](runs.md#local-gpu-ownership).

Engine defaults come from `ENTANGLIX_ENGINES` in the server's environment, for example `cpu=my-engine`; set it in the client's server entry, such as the `env` table of a Codex entry. `list_capabilities` reports them as `engine_defaults` and lists installed plugin engines ([engine selection](system-spec.md#engine-selection-policy)).

The resolved runs root and GPU lock directory are recorded in each connection's audit. This exposes configuration differences; it does not enforce a common location across independently configured processes.

## Tool surface

Tools use the same backing operations as the CLI, with MCP response summaries described below. Unknown request fields, including approval claims or policy overrides, are refused. The server configuration supplies the repository, runs root, runner extras and physical device; these are not tool arguments. Attempt, launch and investigation paths must resolve inside the configured runs root, including symlink targets. Relative record paths resolve from that root, then from the server's working directory. Authored input paths (system, protocol, script, additional inputs and replacement refinement) may live elsewhere.

| Tools | Effect |
|---|---|
| `validate_system`, `list_capabilities`, `list_runs`, `read_run`, `list_investigations`, `read_investigation`, `list_catalog`, `read_entry` | Read-only; validation includes operator meanings and model provenance; catalog reads include staged candidates, promoted designs and lineage |
| `simulate`, `sweep`, `control`, `calibrate`, `run_script`, `rerun`, `check` | Admit jobs through the runner, returning the launch and run IDs once admission is established |
| `abort_run`, `stop_investigation` | Cancel work using the existing process-group and investigation rules |
| `create_investigation`, `revise_investigation`, `record_finding`, `reconcile_run`, `propose_device_update`, `register_result` | Draft terms, propose revisions, append findings, reconcile uncertain attempts, or stage a device-update proposal or catalog candidate |

Only the first row carries the MCP read-only annotation. Jobs and writes may require runtime approval. Promotion, applying a device-update proposal, and investigation start/approval/resume/review release are absent. A proposed revision never changes the effective terms. See [investigations](investigations.md) for the operator decision path.

`register_result`, `list_catalog` and `read_entry` stage and read catalog designs; see the [catalog reference](catalog.md#research-tools) for arguments, source-run location and reply fields.

Control tools take an authored protocol path, or a source run path for `rerun`. An investigation control request names an approved base and passes varied values in `values`, as `simulate` does. Control solves and their requested checks run on CPU; see the [control reference](control.md).

`calibrate` takes an authored calibration spec, which lists training datasets only, and `rerun` a source run path; an investigation calibration names an approved base and takes no varied values. For a calibration source, `check` also takes `holdout`, the authored path of the withheld dataset, `solution`, a solution index, and `expect_coverage` as `[within, minimum]`, as the CLI's `--holdout`, `--solution` and `--expect-coverage` do. `propose_device_update` takes a completed calibrate run path and an optional `solution`, and returns the proposal it published in the run's `proposals/`. Applying it is the operator's `apply-device-update`; see the [calibration reference](calibration.md#device-update-proposals).

`simulate`, `sweep`, `control` and `run_script` accept an optional `parent_run_ids` list, recorded in the new run's `parent_run_ids` in the given order, as the CLI's repeated `--parent-run ID` is. IDs are recorded as given; pass the `run_id` values returned by earlier job tools. Reruns and checks set their own parent. For example, a candidate solve linked to the screening sweep that chose it:

```json
{"tool": "sweep", "arguments": {"protocol": "co-design/sweep.toml"}}
{"tool": "control", "arguments": {"protocol": "co-design/control.toml", "parent_run_ids": ["<run_id returned by sweep>"]}}
```

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
Paths in these tables and `list_investigations` are relative to the configured runs root and accepted by the read tools.

Attempt columns cover path, protocol, base, execution status, varied values,
metrics, parents, checks and completed sweep-point count. A sweep is one row.
Control rows add `seed`, the seed the solver reported using or, for an attempt
without a result, the declared seed; `transition`, each named frame transition's
levels and resolved frequency; and `termination`, the solver's termination record.
Calibrate rows add `outcome`, the fit's outcome.
Target and metric names appear once in `values.<target>` and `metrics.<name>`
column headers. Sweep rows omit per-point varied values; use `read_run` or the
retained point files for those values.
Source checks carry check run ID, source identity match, name/outcome pairs and
any diagnostic; check rows carry name/outcome pairs. A calibration check adds the
index of the solution it checked as a third element. A null outcome denotes
information without a pass/fail judgment. Metric values exceeding 512 JSON bytes
are explicitly omitted here and in `read_run` summaries; full values remain in the run's `result.toml`. An attempt
whose records cannot be read keeps its row, with the reason in a `result_error` column.

Use `read_investigation(investigation, event_id="00000001")` for one complete
event. It cannot be combined with a table view or cursor. Python and CLI
investigation reads remain complete.
`record_finding` returns an acknowledgment with event ID, revision, decision,
usage, checkpoint and reviews.

`read_run` retains summary reads and `detail=True`. Summaries keep execution
state, complete log tails, metrics and check outcomes, scope and coverage, with a calibration check's solution index,
and each completed sweep point's coordinates, metrics and any error. Manifest `spec_sha256`, `code` and `environment`,
and further check fields, are named under `detail_omitted`.
Read detail or the retained files for full provenance and reports. Paging limits
row count, not arbitrary text size.

Use `entanglix report-investigation PATH --output FILE` for an
[on-request Markdown report](investigations.md#findings-and-retained-history).

## Waiting and recovery

`read_run(run, wait_seconds=...)` accepts an attempt or launch directory. It returns on a terminal or interrupted state, or when the requested wait expires. The server's `--max-wait` caps one wait, and the tool description states the cap; the CLI's `read-run --wait` has no cap. Other calls, including `abort_run`, can proceed during a wait. Cancelling or abandoning the wait does not abort the job.

A runtime that waits one call at a time spends a model request each time a wait returns, whatever its length; a runtime that keeps working during a call can differ. Set `--max-wait` to cover the jobs you run, and the runtime's per-call tool timeout above it. The server sends nothing while it waits, so any idle timeout of the runtime must exceed the cap too. A runtime that abandons a call first returns an error to the agent, and the job continues.

A session need not stay open through a job of hours. The agent can end its turn with the run path, and the operator resumes the session after the job, from the workspace and with the first launch's configuration overrides:

```sh
entanglix read-run runs/<run> --wait <seconds>
codex exec resume <session-id> 'The run has finished; continue.'
```

Check the printed status before resuming: a wait that expires returns while the job is still active.

`codex exec --json` reports the session ID in its `thread.started` event. Claude Code resumes with `claude -p --resume <session-id>`, taking the ID from the `stream-json` init event. The command-line wait costs no model request; the resumed request carries the whole conversation, probably uncached after a long gap. No retained session covers this resume.

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

The server reads the versioned [research conduct](../research-agent.md) and skill metadata from `skills/*/SKILL.md`, with front matter declaring `name`, `description`, and `version`. It sends the conduct body as its instructions and records all supplied names/versions. Supplying a skill does not mean its body reached the model: install the files in the workspace as described below. Loaded skills and transcript location stay unknown because the server cannot observe them.

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
  "$ENTANGLIX_CHECKOUT/docs/operations/calibration.md" \
  "$ENTANGLIX_CHECKOUT/docs/operations/catalog.md" \
  "$ENTANGLIX_WORKSPACE/reference/operations/"
```

Use a new directory for these initial copies. Refresh them deliberately when upgrading; retain their versions with the session. `skills/` is the plain-text location named by the conduct; `.agents/skills` makes the same files discoverable in Codex. In a runtime that reads another workspace instruction filename, copy the same conduct there instead. If it reads no such file, supply the entire conduct through its initial-context mechanism. Keep the procedures readable at the named paths. Do not copy root development `AGENTS.md` or `CLAUDE.md`, planning documents, baseline answers, or contributor instructions into the research workspace.

The seven files under `reference/operations/` provide authoring syntax and lifecycle rules. Skills direct the agent to the relevant reference when needed. Links between these references remain usable; links to checkout examples, tests, and the harness contract are unavailable in this copy. For an evaluation, supply any additional inputs separately without exposing baseline answers.

Add a task brief containing the intended question and agreed terms, and copy the model, protocol, subsystem, and numeric input files beside one another. Within an investigation preserve the approved base bytes. For the [readout investigation](../../examples/readout/README.md#scripted-investigation-baseline), the operator can prepare its base bundle with the documented draft command; the agent receives the declarations and inputs, not the baseline summaries.

### Codex example

Create `.codex/config.toml` in that workspace. Replace the placeholders with absolute paths and your GPU UUID. This CUDA example uses the CUDA runner extras; for CPU-only use `--device`, `""`, and the extras matching your runner profile. Every invocation sharing the checkout's `.venv-runner` must use the same extras, including CPU jobs. TOML does not expand these placeholders or shell variables.

```toml
[mcp_servers.entanglix]
command = "<absolute-checkout-path>/.venv-mcp/bin/python"
args = ["-m", "entanglix.mcp", "--runs-root", "<absolute-workspace-path>/runs", "--device", "<GPU-UUID>", "--runner-extras", "cuquantum", "differentiable-cuda12", "--max-wait", "3600"]
startup_timeout_sec = 60
tool_timeout_sec = 3660
default_tools_approval_mode = "approve"
```

`--max-wait 3600` lets one wait cover a job of up to an hour, and `tool_timeout_sec` stays above it so Codex does not abandon the wait first. The tool-approval setting allows job and write calls in an unattended session. Choose it according to the intended session; it is not a human investigation decision. If your operator shell sets a custom `XDG_STATE_HOME`, add `--state-directory` and the same absolute directory to `args`. Otherwise the server uses the runner default. No CUDA environment variable is needed on the entry: `--device` explicitly supplies the server's job visibility.

Create `.codex/rules/operator.rules`:

```python
prefix_rule(
    pattern=["entanglix", ["investigation-decision", "apply-device-update", "promote"]],
    decision="forbidden",
    justification="Human action: the operator runs this in their own terminal.",
)
```

This launch uses the tested non-interactive setup, with the workspace trusted explicitly, the `workspace-write` sandbox, and shell environment inheritance. The model and reasoning effort are your choice through your Codex configuration. Retain the prompt/transcript to establish whether guidance actually arrived. Codex's [skills](https://developers.openai.com/codex/skills), [rules](https://developers.openai.com/codex/rules), and [MCP configuration](https://developers.openai.com/codex/mcp) document these mechanisms.

```sh
cd "$ENTANGLIX_WORKSPACE"
CUDA_VISIBLE_DEVICES="" codex exec --json --skip-git-repo-check \
  -C "$ENTANGLIX_WORKSPACE" -s workspace-write \
  -c "projects={\"$ENTANGLIX_WORKSPACE\"={trust_level=\"trusted\"}}" \
  -c 'shell_environment_policy.inherit="all"' \
  -o "$ENTANGLIX_WORKSPACE/last-message.txt" '<task brief>'
```

The documented operator invocation form is the bare `entanglix investigation-decision ...`, `entanglix apply-device-update ...` or `entanglix promote ...`, with `entanglix` on `PATH`. Do not suggest `uv run ... entanglix ...`, an absolute executable path, a Python API, or a shell wrapper as a substitute: the supplied rule is not claimed to cover those forms. The refusal is claimed for the three bare commands in the tested non-interactive configuration only ([conformance run](../../tests/probes/m3_guidance_results.json), [live calibration session](../../tests/probes/s2_sessions_results.json)); an interactive session is not covered. Offline rule matching or an agent's verbal refusal is not a passed blocked-command test.

To approve each human-action command instead of refusing it, set `decision="prompt"` in that rule, in an interactive session only: an unattended session cannot answer the prompt, so keep `forbidden` there. Approve each invocation on its own, never with an option that approves later commands. No retained session tests this setting.

### Claude Code example

Supply the conduct as workspace `CLAUDE.md` in place of `AGENTS.md`, and retain `skills/` for plain-text reading. Create `.mcp.json` in the workspace with the same absolute Python command and arguments, here for CPU only:

```json
{
  "mcpServers": {
    "entanglix": {
      "command": "<absolute-checkout-path>/.venv-mcp/bin/python",
      "args": ["-m", "entanglix.mcp", "--runs-root", "<absolute-workspace-path>/runs", "--device", "", "--runner-extras", "cuquantum", "differentiable-cuda12", "--max-wait", "3600"],
      "timeout": 3660000
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
    mcp__entanglix__rerun mcp__entanglix__reconcile_run \
    mcp__entanglix__record_finding mcp__entanglix__stop_investigation
```

The entry's `timeout` is Claude Code's per-call limit for this server, in milliseconds; keep it above `--max-wait`. It also sets the floor of Claude Code's idle check, which otherwise aborts a stdio call after 30 silent minutes. In an interactive session, Claude Code moves a call still running after two minutes to a background task; the non-interactive form does not. For a control task, also allow `mcp__entanglix__control`. For a calibration task, also allow `mcp__entanglix__calibrate` and `mcp__entanglix__propose_device_update`.

In four [retained sessions](../../tests/probes/m3_pickup_cutoff_results.json) the conduct was present before the first reply and no repository development instructions appeared. No command setting has been validated for Claude Code: a session without a shell tool cannot run an operator command, but no retained test covers a session that has one, so there the rule rests on the conduct. Claude Code can reject a large tool reply as over its output limit: use the status and paged table views for research reads, and the report command or retained files for full history.

In an interactive session with a shell tool, a project `.claude/settings.json` sets how Claude Code handles the human-action commands: listed under `deny` they are refused, under `ask` Claude Code asks before each one.

```json
{
  "permissions": {
    "deny": [
      "Bash(entanglix investigation-decision:*)",
      "Bash(entanglix apply-device-update:*)",
      "Bash(entanglix promote:*)"
    ]
  }
}
```

Approve each invocation on its own; a permission mode that skips prompts runs commands without asking and is not approval. No retained session tests either setting.

### Other stdio runtimes

Use the same absolute Python command and arguments in your runtime's MCP entry, and supply the conduct under the instruction filename your runtime reads. Start the session with `CUDA_VISIBLE_DEVICES=""` and configure job/write tool approval as needed. Keep the runtime's per-call tool timeout, and any idle timeout, above `--max-wait`. Claim that the runtime refuses or asks before the human-action commands only after a retained live test shows it for the forms you document; until then the rule rests on the conduct. There is no list of supported model backends.

## Human decisions and continuation

The agent may draft an investigation and propose revisions; the operator reviews complete terms, checkpoint, evidence, and any requested limits. The [investigation reference](investigations.md) defines these declarations and commands. Use the same runs root as the server and put the installed bare `entanglix` command on the operator's `PATH` too.

```sh
entanglix read-investigation "$ENTANGLIX_WORKSPACE/runs/_investigations/<investigation_id>"
entanglix investigation-decision "$ENTANGLIX_WORKSPACE/runs/_investigations/<investigation_id>" start --revision 1
```

That second command is the user's decision. The user runs it, or, where the runtime is set to ask, approves the agent's run of that exact command in the runtime's prompt; the decision record is the same either way. Subsequent approvals, resumes and named review releases use the same bare command with the affected revision/review name. The agent reads the retained decision before continuing; approving a research-tool call does not make one. At a checkpoint it records an assessment; at a named review it hands off for the operator's decision. A finding recommending stop does not itself stop jobs: use the supported `stop_investigation` tool to cancel owned work and close admission.

When you tell the agent which investigation to work on, name it by its ID or as `_investigations/<investigation_id>`; the agent's tools resolve it as described under [tool surface](#tool-surface). The operator commands above take a path from your shell.

A new session starts from listings and the effective investigation revision, findings, attempts, usage, and open gates. If a launch call lost its response, recover the launch before sending another. For a stopped investigation, the operator's resume permits new admissions and never revives cancelled attempts. Report partial results, requested/relevant checks not run, and the next useful decision without resetting history or allowances.

This is a cooperative workflow. CUDA visibility and command rules reduce accidental interference; they are not OS isolation or authentication and do not resist arbitrary shell/code bypass. The runner's recorded ownership, cancellation, input confinement and scope checks apply to supported operations.
