# ADR-0003: Run Records, Verification, and Engine Dispatch

**Date:** 2026-09-30, revised 2026-10-28
**Status:** Accepted
**Purpose:** fix how runs are recorded, how a user checks completed runs, which engine runs a computation and how it reaches the GPU without a performance claim, and how driven evolution is represented in the lab and rotating frames. Field lists, formats, and threshold values are documented with the implementation.

---

## 1. Context

The [harness contract](../../HARNESS_CONTRACT.md) requires a run directory with `run.toml` first and `FINISHED` last (§2), an independent corrector whose checks cover physicality and convergence (§3), an append-only history of every attempt (§6), and GPU exclusivity, cancellation, and reconciliation of unknown process state on the workstation (§7). [ADR-0001](0001-mandate.md) sets a cooperative trust model and a lean design that reuses process and platform facilities rather than building services.

Engine integration established agreement across QuTiP CPU, cuDensityMat, and dynamiqs on reference fixtures, a validated dispatch cap of dimension 30, and a selection policy that sends every non-gradient request to CPU. No measurement yet shows where the GPU is faster. Production runs may take hours, so verification that repeats a run on several engines or refines every sweep point would multiply its cost. On the first example, a refined full-window reference cost 12× to about 210× the run it checked, and the cheaper checks that could follow every run could not certify a result. Drives are constant within each output interval and evolution uses the lab frame, which makes resonant GHz drives impractical.

amicode's run-directory contract supplies the layout, terminal marker, atomic result, and exit codes. Its frozen validators also require a Julia table and control fields (`fidelity`, `iterations`) that do not describe Python simulations or sweeps.

Users may also supply their own engines and solvers as installed plugins.

## 2. Decisions

### 2.1 Run directories are the attempt history

- Runs follow amicode's layout and terminal semantics: `run.toml` written first, `FINISHED` written last with `status` and `exit_code`, `result.toml` published by temporary-file rename, exit codes `0`, `130`, `64`, or the job's code, and run IDs of the form `r<UTC-timestamp>Z-<hex>`. amicode's Julia table, control result fields, iteration log lines, iteration plot names, `index`, and `latest` are not adopted.
- The runner writes each applicable record once: `run.toml` when the attempt is created (identity, parent runs, declared needs, spec digests, code provenance, `env_hash`, engine, device); `runner/started.toml` before the job computes (start time, GPU allocation, process group); `FINISHED` when the supervising runner observes the end, with status `completed`, `failed`, or `aborted` (exit code, end time, usage); and `runner/reconciled.toml` only after runner loss, in place of `FINISHED` (last observed state, usage marked uncertain).
- There is no separate ledger store or index; listing scans the runs root. An attempt without a valid `FINISHED` is incomplete. Reconciliation never writes `FINISHED`, so a result left by an interrupted attempt cannot substantiate completion. Reruns receive new IDs linked to the earlier attempt.
- One `rerun` operation replays a supported retained computation or script attempt by the protocol its manifest records, and refuses an active or uncertain attempt or a changed frozen input. A script rerun may override its recorded timeout; a protocol rerun keeps its frozen spec's limit.
- Jobs write only `result.toml` and `artifacts/`. Runner records use separate, read-only paths.
- A sweep is one run. Its points (static spectral points, or dynamic points that are undriven or driven, in either frame) run serially inside the job. Completed points stay in `artifacts/` if the run is interrupted. A rerun is a new linked run that recomputes the grid, and verification treats the sweep as one source run with per-point coverage.

**Rationale:** each attempt already has a unique directory and immutable runner-written files, which provide atomic per-attempt events and remove contention between concurrent runs without a ledger service. Recording start, allocation, and process ownership before computation keeps usage and ownership available when the runner dies and `FINISHED` never appears. Resuming a sweep from its completed points waits for a protocol with restart support.

### 2.2 Execution environment and process ownership

- Every job runs through `uv run --locked --inexact` with the same declared extras set for all jobs sharing the environment. Locked dependencies are synchronized at each launch. Distributions the user installs in the runner environment, such as plugins, persist; Entanglix neither installs nor resolves them, and they are installed or updated only while no job uses the environment.
- The extras set is an explicit setting where the user gives one, otherwise the installation's recorded set, otherwise the project default. Each launch resolves it, so a server that started earlier uses a component added since.
- Each launch takes a shared lock on the environment before it resolves the set, synchronizes the environment, or runs the launch gate, and its job inherits the lock until the job and any surviving children end. Adding a component takes the lock exclusively without waiting, so it is refused while any launch or job holds it. It marks the environment incomplete before synchronizing and clears the mark on success; launches are refused while the mark stands.
- An installation's uv calls use its own uv by absolute path, without configuration files or development dependencies. They remove inherited variables that could select another environment, interpreter, or cache, or skip synchronization, and set the installation's own. A development checkout uses the uv on `PATH` and removes the variables that would skip synchronization or redirect imports.
- Before admission, the runner synchronizes the environment, checks its dependency consistency with uv, and refuses an inconsistent environment. A distribution without an install record, or an editable distribution other than the project itself, refuses admission.
- The run records every installed distribution with its version and a digest of its install record, the manifest in which the installer lists each installed file with its hash. `env_hash` covers that record, the lockfile digest, Python and platform, the installed extras, and observed driver, CUDA, and cuQuantum versions.
- Code provenance records the Git revision and dirty state and freezes a patch and untracked source files for uncommitted changes. Authored scripts and data are copied into `spec/` with digests.
- The runner process supervises its job in a new process group; there is no daemon or queue. Separate invocations read and abort through the run records.
- The GPU lock is a per-device file lock outside any runs root. The job inherits it, and the lock record names the holding run and process group. Reconciliation checks that process group before the GPU is reused.

**Rationale:** a sync with a different extras set would reshape the environment between jobs. An `flock` lock is held while any process sharing its open file description keeps it open, so an inherited lock stays held when the runner dies and its job survives; a lock held only by the runner would be released while the job still uses the GPU. Users install private plugins themselves, and an exact sync would remove them at each launch. Recording every installed distribution, rather than those absent from the lockfile, also covers a locked package that a plugin installs at another version. A version does not identify a locally rebuilt package, but its install record lists the hash of every file it installed, so digesting the records identifies content without reading the environment at each launch. An editable install's record does not cover its source. Under the cooperative trust model, files edited after installation are not detected. Incompatible requirements do not always fail at import, so consistency is checked before admission. A launch that resolved the set before a component change and synchronized after it would run in an environment it did not resolve; holding the lock from before the read until the job ends closes that window. A failed synchronization must not serve a job, so it leaves a mark rather than a partly changed environment. Resolving the set at each launch avoids restarting servers after a component is added. User configuration and inherited variables can redirect uv to another environment, interpreter, or cache, or skip synchronization, without an error.

### 2.3 Verification runs and checks on request

- The harness runs what the user requests. A protocol run computes and stores its results, and no check follows unless the user asks. A request may be standing, such as named checks for every run of an investigation.
- Verification is a set of built-in checks that the user requests on a completed run through `check`. Each check targets one error mode and reports its own outcome: pass, fail, or inapplicable with a reason. There is no run-level verdict.
- A verification run is its own run whose parent is the source run. Its `result.toml` holds a report bound to the source artifact digests and the corrector policy version. The report lists each requested check's outcome, scope, tolerance, residual, and coverage. Protocol results carry no check outcomes.
- The corrector is a subpackage of the same distribution, with versioned thresholds as package data outside run directories and authored-script workspaces. The corrector policy version is the digest of the corrector code and thresholds; it is recorded at launch and recomputed when the corrector loads, and a mismatch fails the verification run without a report.
- The checks relevant to a run derive from trusted definitions keyed by computation kind (Lindblad evolution, spectrum, parameter series) and from the validated plan's content, not from labels in specs or results. A launch, its dry-run preview, and `read_run` list them with the error mode each one catches, so the user can choose. A check requested where it does not apply is reported as inapplicable. Runs without a trusted definition, including authored scripts, have no built-in checks.
- A check's prerequisites never restrict a run. A refinement declaration is optional, and the evidence-point bound limits checks, not sweep grids.
  - A malformed declaration, such as a wrong type or a nonpositive step, is rejected as a spec error.
  - A declaration that is missing, leaves out a truncated subsystem, or is below the policy minimum does not stop the run, and neither does a grid whose corners exceed the bound. The listing names the checks that cannot run and why, before computation on a real launch as well as in the dry run. Those checks are reported as inapplicable with that reason.
  - A check request may carry a replacement refinement declaration. It is frozen in the verification run with the requested check names, under the runner's input digests, and recorded in the report with its origin. The source run is unchanged.
- Reading a run lists its verification runs with their status, their policy version, and each check's outcome. A verification run that crashes, is aborted, or is stopped by the runner's timeout keeps its execution status and reports no outcome. Reports under an earlier policy remain history labelled with that policy.
- A passing check establishes only what its scope states.

**Rationale:**
- Checks that run unrequested cost time on every run. For dynamics they could not certify a result without a reference solve costing more than the run, so their default outcome told the user little. The user knows which error modes matter for a run; the harness states which checks apply.
- A verdict compresses checks with different scopes into one word. Per-check outcomes with their scope state what was established and what was not.
- A refinement declaration serves checks only. Letting the check request replace it means a user corrects one without repeating the run.
- A separate run leaves the source run unchanged after `FINISHED`, keeps execution outcome distinct from checking, and lets a check be repeated under a new policy. A terminated verification run may have found a failure it never published, so it reports nothing.

### 2.4 Built-in checks

- **Integrity** (dynamics): trace, Hermiticity, and minimum eigenvalue of every stored state, agreement of the first stored state with the declared initial state in the lab basis, and consistency of reported values with the stored states. It catches corrupted or unphysical output and a run started from the wrong state. It cannot detect a fault that affects the declared and the stored state alike.
- **Segments** (dynamics): short windows restarted from stored states in the source run's frame, at tightened tolerances and with each truncated subsystem refined by its declared step. They run on CPU with the run's own built-in engine; a GPU run, or a run on a plugin engine, uses QuTiP. They catch local integration and truncation error.
  - Windows cover the start, middle, and end of the run and its drive changes, up to a policy maximum, spread across the drive changes when more exist than fit. Drive changes left uncovered are reported.
  - A segment that disagrees beyond its tolerance fails. The tolerance is a policy default with an explicit override per observable or metric, in that quantity's units; the effective value is recorded.
  - Passing segments show local agreement over the reported coverage. A restart inherits any error already accumulated, so they are never extrapolated to the whole window.
  - A segment on the run's own engine can share a systematic error with the run. It is a convergence comparison, not independent validation of the engine.
  - A segment on QuTiP for a plugin-engine run also compares the engines, and its scope says so.
  - Before the user requests segments, the listing shows the source and refined dimensions, the CPU engine, and the selected windows.
- **Diagnostics** (dynamics): top retained-level population, and static dressed-state shifts and weights at a refined cutoff or between full and rotating-wave Hamiltonians. They are reported and have no pass or fail outcome.
- **Spectral refinement** (spectra): the reported quantities recomputed at refined cutoffs and retained levels. It catches a truncation too small for the reported energies.
- **Declared expectation** (sweeps): the user declares an expected trend in the sweep spec, such as a metric's direction along an axis, and the check compares the results with it. A failure is reported as "declared expectation violated", because the expectation itself may be wrong.
- **Stated limit:** no built-in check measures error accumulated over the whole window or the accuracy of a rotating-wave approximation. The check listing and each dynamics report say so. A user who needs that evidence requests another solve, for example in the lab frame or with more levels, and compares the two runs.
- **GPU runs:** no scientific check precedes a launch. Checks on a GPU run execute on CPU.
- **Sweeps:**
  - Integrity and diagnostics cover every point whose states the sweep retained.
  - Segments and spectral refinement run at a deterministic, bounded set of evidence points selected after the sweep.
  - Reports mark each point's coverage and say nothing about unsampled points. Warnings are all reported; when the evidence points cannot cover them within the bound, the report says so.
- Refined solves run on CPU, with no dimension limit of their own (§2.5). A plugin-engine run's refined solves use QuTiP, so no check needs the plugin.
- Further checks are added when a need is identified.

**Rationale:**
- The checks follow how researchers test convergence: physicality first, then spot checks at tighter tolerance and more levels.
- A refined full-window reference repeats the computation that truncation and the rotating frame exist to avoid. It is another solve, so a user who doubts a result requests one as an ordinary run.
- Re-solving a spectrum with a second eigensolver, or against an independently constructed model, tests the solver and not the physics. The solver is the user's choice, and those comparisons stay in the acceptance tests of the model and engine layers.
- A segment tests tolerance and truncation, not the engine, so it uses the run's own engine where that engine is built in and runs on CPU. Engine agreement rests on the engine acceptance tests. A plugin engine has no acceptance tests here, and no check depends on private code, so its segments run on QuTiP.
- A check input that blocked a launch would restrict runs the user never asked to check.
- Segments see only the intervals they sample. In the probe a delayed pulse escaped start, middle, and end selection, and local residuals did not predict full-window errors.
- Lab-frame evolution restarted from a rotating-wave state shows local differences between the full and RWA dynamics, not the approximation's accumulated error. On the readout cases those differences exceeded the full-window error, so segments stay in the source run's frame.
- Static quantities without a demonstrated metric conversion missed every driven failure in the probe, so they are diagnostics.
- Refined solves on CPU never wait for the GPU lock.

Measurements: [verification feasibility probe](../../tests/probes/m2_verification_results.json).

### 2.5 Engine dispatch

- The engine is the one the spec names; otherwise the user's default for the declared device; otherwise the built-in rule on declared needs (`gradients` and `device`): without gradients, a declared `device = "cuda"` selects cuDensityMat and an undeclared device selects QuTiP CPU; gradients select dynamiqs. The selection table records needs, not performance.
- Engines are built in or discovered through the `entanglix.integrators` entry-point group. Each implements the `Integrator` interface and declares its capabilities.
- The user sets per-device defaults in the environment of the process that prepares runs. Research agents name an engine only at the user's request.
- A chosen engine that is not installed, or whose capabilities do not meet the declared needs, refuses the run before admission; no other engine replaces it. Plugin engines serve runs without gradients or batching. Spectrum runs use the QuTiP CPU eigensolver, ignore defaults, and refuse any other named engine.
- A refusal for a missing built-in engine names the installation component that provides it. Components describe bundled engines, not permission to use one: a plugin engine, including a CUDA one, runs without them.
- Plans store the engine name. Availability and capabilities are resolved at admission, so a run can be read and checked where its engine is not installed. The preview and run records name the engine, its distribution version and digest, and the source of the choice.
- A rerun uses its source run's engine and is refused if that engine is unavailable.
- Every CUDA launch passes the shared device checks: ownership, identity, and refusal of a busy device. Engine-specific initialization runs only for the engine that needs it.
- The harness imposes no dimension cap, on dispatch or on the corrector's refined solves, and no memory estimate precedes a launch. Tested sizes are acceptance evidence, not a ceiling.
- A run that exhausts memory fails:
  - An observed allocation failure is reported as out of memory, including one during model construction at the launch gate, before a run exists.
  - A job killed by a signal the runner did not send is reported as that signal, with exhausted memory as a possible cause only.
  - If the runner dies as well, the interruption and reconciliation rules of §2.1 and §2.2 apply.
- A launch shows the model's dimensions and output count. These are known inputs, not an estimate.
- Automatic performance-based dispatch requires measured CPU/GPU crossover and peak memory, together with a review of cuDensityMat open-system usage.

**Rationale:** a declared device exercises the GPU path without an unmeasured performance claim, and dispatch remains a recorded property of the run. The earlier caps, dimension 30 for dispatch and 48 for refined solves, marked how far exact references had been tested. With no reference check, they would only restrict what a user can model. A memory estimate refuses a run on a prediction; the failure itself is reported when it happens. A default switches engines for all new work without editing specs, and a named engine records an explicit request. Substituting an engine would report a computation the user did not choose. A plan that required its engine would make history unreadable once a plugin is removed. Pinning a rerun to its source engine keeps a changed default from changing what the replay computes. Shared device checks keep a CUDA plugin from requiring the cuQuantum stack. Naming the component gives the user the fix, and a component gate would make a user's own CUDA engine depend on the bundled GPU stack.

### 2.6 Drives and frames

- Lab-frame carrier drives are an envelope times `cos(2π f t + φ)`, with `f` in GHz, `t` in ns, and `φ` in radians, evaluated continuously through each engine's time-dependent coefficient path.
- Rotating frames are declared in the evolution input as frame frequencies applied to excitation-number operators in each subsystem's retained eigenbasis; explicit-operator models declare their own. The transformation is an explicit numeric operation on the constructed operators. A declared rotating-wave approximation removes counter-rotating terms, and provenance records the frame and the removed terms. Comparisons across frames use frame-invariant observables or states transformed back to the lab frame.
- Evolution uses the pinned adaptive solvers; no stiff or specialized integrators are added.

**Rationale:** a frame is a choice of computation, not a device property, so it belongs to the evolution input rather than the system specification. The rotating frame reduces cost; §2.4 states that no built-in check measures its approximation.

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| Separate append-only ledger store, `index`, and `latest` | Duplicate the runner records; `index` and `latest` serve the Run Inspector |
| amicode's frozen validators in full | Julia and control fields do not describe Python simulations or sweeps |
| Check outcomes in the protocol-written `result.toml` | Research code would report on its own results, and a check after `FINISHED` would change a finished run |
| Verification inside the source run before `FINISHED` | `FINISHED` would wait on refined solves, check failures would mix with execution outcome, and repeating a check would edit a finished run |
| Automatic verification after every run | Runs work the user did not request; without a reference solve it could not certify a dynamics result |
| Run-level verdicts over evidence requirements (`VERIFIED`, `UNVERIFIED`, `INCONCLUSIVE`) | One word hides which error modes were checked, and every clean dynamics run without a reference was `INCONCLUSIVE` |
| Waivers and a verification allowance | Nothing runs unrequested, so there is nothing to waive or to bound |
| Agreement across all three engines for every run | Multiplies the cost of long runs; the engine acceptance fixtures establish three-engine agreement |
| Convergence refinement at every sweep point | Repeats the production computation at larger dimension |
| A built-in full-window refined reference | Costs 12× to about 210× the run on the first example and repeats the computation that truncation avoids; a user who doubts a result requests another solve |
| Spectral comparison with a second eigensolver or an independently constructed model | Tests the solver, not the physics; the solver is the user's choice |
| Segments of a built-in CPU engine's run on a different engine | A segment tests tolerance and truncation; engine agreement is established by the engine acceptance tests |
| Declared per-observable and per-metric rotating-wave bounds | Each bound needed its own frame study and a lab-frame reference to check it |
| General linkage and multi-photon analysis as a static screen | A research method rather than a check; diagonalization covers static linkage within retained and refined levels, and drive-induced linkage needs dynamics |
| Lab-frame segments as approximation evidence for rotating-wave runs | They show local differences between the full and RWA dynamics, not accumulated error, and failed readout cases whose full-window error was within target |
| Passing or extrapolated segments as whole-window evidence | A restart inherits accumulated error, and sampled intervals are not representative of the window |
| An exact step-propagator bound on numerical error at stored times | Applies only where the generator is constant in the run's frame, up to about dimension 30; a time-dependent generator needs a time-ordered propagator, which is another numerical solve |
| Static dressed-rate screens as approximation or truncation evidence | The demonstrated case covers only undriven runs with resonator loss |
| A CPU reference or static screen before every GPU launch | The reference costs more than the run, and no screen applies generally |
| Top retained-level population as a pass/fail check | Rejects exact truncations, such as decay from the top retained Fock state |
| Expected metric trends built into the corrector | The direction is physics of one model and parameter region; the researcher declares it per sweep |
| Evidence points added for every truncation warning | Widespread warnings would recreate an unbounded set of refined solves |
| Refinement declarations required, and sweep grids limited by the evidence-point bound, at launch | Restricts runs the user did not ask to check |
| A refinement declaration checked against the corrector policy at launch, or fixed by the source run | Ties launches and replays to the corrector's thresholds, and correcting a declaration would mean repeating the run |
| A dimension cap on dispatch or on refined solves | The caps marked how far exact references had been tested; with no reference check they only restrict what a user can model |
| A memory estimate checked before launch | Refuses a run on a prediction that counts only the arrays known in advance; the failure is reported when it happens |
| GPU lock held only by the runner | Released when the runner dies while its job still uses the GPU |
| Automatic GPU selection | No measured crossover supports it |
| Rotating frames in the system specification | A frame is a computation choice, and one device model serves both frames |
| A separate corrector distribution | The policy digest provides the needed tamper evidence under the cooperative trust model |
| Exact sync of the runner environment | Removes user-installed plugins at each launch |
| Plugins installed by Entanglix, through `uv run --with` or a requirement list | Entanglix would carry plugin requirements; users install plugins themselves |
| Plugins added to a local `pyproject.toml` and lock | Every run records a dirty checkout, and private names enter recorded patches |
| Recording only distributions absent from the lockfile | Misses a locked package installed at another version |
| Selection only from declared needs | A user could not run their own engine |
| A built-in engine in place of a chosen engine that cannot run | Reports a computation the user did not choose |
| Segments and refined solves on a plugin engine | A check would depend on private code |
| Reruns re-selecting the engine | A changed default would change what a replay computes |
| One rerun per protocol | Five commands and tools expose one action already identified by the manifest |
| cuDensityMat initialization at every CUDA launch | A CUDA plugin would require the cuQuantum stack |
| Requiring every user to choose runner extras before the first job | Users choose functionality, not packaging; components map to extras, and an explicit setting still overrides them |
| The environment lock taken once the job runs | A component change between the launch gate and the job would reshape the environment under it |
| The extras set captured when a server starts | A component added later would need every server restarted |
| An installation using the uv, settings, and variables it inherits | An incompatible uv, a user setting, or an inherited variable could select another environment or skip synchronization |
| Engines usable only through their installation component | A user's own CUDA engine would need the bundled GPU stack |

## 4. Consequences

- The contract changes accordingly:
  - §1 and §2 describe results without check outcomes, and limit amicode compatibility to layout and terminal semantics, with the runner records and the declared device need.
  - §2 no longer refuses a request for its size, and states how an out-of-memory failure and an unexplained signal death are reported.
  - §3 replaces three-engine agreement and verdicts with built-in checks that run on request and report their own outcomes.
  - §5 lists `check` in place of `compare_reference`, and one `rerun`.
  - §6 identifies the runner records as the run ledger. Promotion shows the checks run on a result and their outcomes, and refuses a result with a failed check reporting numerical disagreement or a violated pulse constraint; a violated declared expectation or an unmet target is recorded ([ADR-0008 §2.4](0008-results-catalog.md#24-promotion)).
  - §7's launch checks include no scientific check.
- Tests cover runner loss while the job survives, lock inheritance, reconciliation, the evidence-point bound, and refinement-step minimums. They also cover a run above dimension 30 launching and completing, an allocation failure reported as out of memory, a signal death reported as its signal, engine agreement at dimension 64, a run completing with no verification run and with no refinement declared, a malformed refinement rejected, a replacement refinement on a check request and a tampered one rejected, per-check outcomes and inapplicable requests, a terminated verification run reporting no outcome, a wrong initial state failing integrity, segment failures on the run's own engine and selection at drive changes, and the top-Fock diagnostic case.
- A result's standing rests on the checks its user requested. Findings and promoted entries state which checks ran, their outcomes, and which relevant checks did not run.
- The checks planned for `control` and `calibrate` follow the same rule when those protocols are built.
- A research agent may request either device through a supported protocol; within an investigation the device belongs to a base the user approved ([ADR-0004 §2.8](0004-investigations-and-agent-sessions.md#28-agent-sessions)).
- The selection table and the corrector policy drop their dimension caps, and engine selection and the launch gate drop their memory estimates.
- Batching and performance-based dispatch follow the cuDensityMat usage study. Tolerances and segment limits are set with the implementation.
- The contract's §1 states that private implementations are distributions the user installs in the runner environment, and §2 states the engine order, the user default, the refusal of a chosen engine that cannot run, rerun pinning, and the installed-distribution record.
- The research conduct asks agents to leave `engine` unset unless the user asks for one.
- Tests cover engine precedence, refusals without fallback, spectrum handling, a plugin run read and checked without its plugin with segments on QuTiP, reruns keeping their engine, a local plugin distribution recorded with version and digest, the consistency and editable refusals, and CUDA admission without cuQuantum for another engine with the busy-device refusal unchanged.
- The contract's §2 states that an environment's extras set changes only while no launch or job uses it.
- Conformance must cover extras-set precedence and resolution at each launch, the lock held from before the read until the job ends, a component change refused meanwhile, the incomplete mark, plugins surviving a component change, the controlled uv environment against inherited variables, refusals naming the component, and a CUDA plugin engine running without the GPU component.
