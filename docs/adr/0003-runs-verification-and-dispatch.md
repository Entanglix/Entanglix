# ADR-0003: Run Records, Verification, and Engine Dispatch

**Date:** 2026-09-30, revised 2026-10-02
**Status:** Accepted
**Purpose:** fix how runs are recorded, how completed runs are verified at bounded cost, how a run reaches the GPU without a performance claim, and how driven evolution is represented in the lab and rotating frames. Field lists, formats, and threshold values are documented with the implementation.

---

## 1. Context

The [harness contract](../../HARNESS_CONTRACT.md) requires a run directory with `run.toml` first and `FINISHED` last (§2), an independent corrector with dynamics agreement, convergence, and physicality checks (§3), an append-only history of every attempt (§6), and GPU exclusivity, cancellation, and reconciliation of unknown process state on the workstation (§7). [ADR-0001](0001-mandate.md) sets a cooperative trust model and a lean design that reuses process and platform facilities rather than building services.

M1 established agreement across QuTiP CPU, cuDensityMat, and dynamiqs on reference fixtures, a validated dispatch cap of dimension 30, and a selection policy that sends every non-gradient request to CPU. No measurement yet shows where the GPU is faster. Production runs may take hours, so verification that repeats a run on several engines or refines every sweep point would multiply its cost. On the first example, a refined full-window reference cost 12× to about 210× the run it checked. Drives are constant within each output interval and evolution uses the lab frame, which makes resonant GHz drives impractical.

amicode's run-directory contract supplies the layout, terminal marker, atomic result, and exit codes. Its frozen validators also require a Julia table and control fields (`fidelity`, `iterations`) that do not describe Python simulations or sweeps.

## 2. Decisions

### 2.1 Run directories are the attempt history

- Runs follow amicode's layout and terminal semantics: `run.toml` written first, `FINISHED` written last with `status` and `exit_code`, `result.toml` published by temporary-file rename, exit codes `0`, `130`, `64`, or the job's code, and run IDs of the form `r<UTC-timestamp>Z-<hex>`. amicode's Julia table, control result fields, iteration log lines, iteration plot names, `index`, and `latest` are not adopted.
- The runner writes each applicable record once: `run.toml` when the attempt is created (identity, parent runs, declared needs, spec digests, code provenance, `env_hash`, engine, device); `runner/started.toml` before the job computes (start time, GPU allocation, process group); `FINISHED` when the supervising runner observes the end, with status `completed`, `failed`, or `aborted` (exit code, end time, usage); and `runner/reconciled.toml` only after runner loss, in place of `FINISHED` (last observed state, usage marked uncertain).
- There is no separate ledger store or index; listing scans the runs root. An attempt without a valid `FINISHED` is incomplete. Reconciliation never writes `FINISHED`, so a result left by an interrupted attempt cannot substantiate completion. Reruns receive new IDs linked to the earlier attempt.
- Jobs write only `result.toml` and `artifacts/`. Runner records use separate, read-only paths.
- A sweep is one run. Its points (static spectral points, or dynamic points that are undriven or driven, in either frame) run serially inside the job. Completed points stay in `artifacts/` if the run is interrupted. A rerun is a new linked run that recomputes the grid, and verification treats the sweep as one source run with per-point coverage.

**Rationale:** each attempt already has a unique directory and immutable runner-written files, which provide atomic per-attempt events and remove contention between concurrent runs without a ledger service. Recording start, allocation, and process ownership before computation keeps usage and ownership available when the runner dies and `FINISHED` never appears. Resuming a sweep from its completed points waits for a protocol with restart support.

### 2.2 Execution environment and process ownership

- Every job runs through `uv run --locked --exact` with the same declared extras set for all jobs sharing the environment. `env_hash` covers the lockfile digest, Python and platform, the installed extras, and observed driver, CUDA, and cuQuantum versions.
- Code provenance records the Git revision and dirty state and freezes a patch and untracked source files for uncommitted changes. Authored scripts and data are copied into `spec/` with digests.
- The runner process supervises its job in a new process group; there is no daemon or queue. Separate invocations read and abort through the run records.
- The GPU lock is a per-device file lock outside any runs root. The job inherits it, and the lock record names the holding run and process group. Reconciliation checks that process group before the GPU is reused.

**Rationale:** an exact sync with a different extras set would reshape the environment between jobs. An `flock` lock is held while any process sharing its open file description keeps it open, so an inherited lock stays held when the runner dies and its job survives; a lock held only by the runner would be released while the job still uses the GPU.

### 2.3 Verification runs and verdicts

- Verification is its own run whose parent is the source run. Its `result.toml` holds the verdict, `VERIFIED`, `UNVERIFIED`, or `INCONCLUSIVE`, and a report bound to the source artifact digests and the verification policy version. Protocol results carry no verdict.
- The corrector is a subpackage of the same distribution, with versioned thresholds as package data outside run directories and authored-script workspaces. The verification policy version is the digest of the corrector code and thresholds; it is recorded at launch and recomputed when the corrector loads, and a mismatch fails the verification run without a verdict.
- The runner starts verification after each completed `simulate` or `sweep` run; `compare_reference` starts it on demand. Evidence is reused when plan digests and the policy version match.
- Reading a run shows the latest verification attempt's status and whether a verdict exists for the requested policy, by default the current one. Earlier verdicts remain historical evidence labeled with their policy and source digests and never substitute for a missing current verdict.
- Evidence requirements derive from trusted definitions keyed by computation kind (Lindblad evolution, spectrum, parameter series) and from the validated plan's content: integrity of stored states, numerical accuracy, truncation, and, for rotating-wave runs, the approximation. Each requirement names the checks that can satisfy it; an inapplicable check creates no requirement.
- The verdict follows the requirements, with failure taking precedence:
  - `UNVERIFIED`: a check fails. This stands even if other work later runs out of allowance.
  - `VERIFIED`: every requirement is satisfied.
  - `INCONCLUSIVE`: otherwise. The report lists each unresolved requirement with its reason: waived, no applicable check, or allowance exhausted.
- Labels in specs or results cannot remove checks. A spec may declare expectations, such as a metric's direction along a sweep axis; a declared expectation adds a check and never removes one or satisfies a requirement. A violated expectation is a failed check even when a reference comparison passes; the report names it "declared expectation violated", because the expectation itself may be wrong. Refinement steps below the policy minimum are rejected. Runs without a trusted definition, including authored scripts, cannot verify.
- Only a human action waives a check. The waived check does not run and its requirement stays unresolved. The waiver records who applied it, when and why, and appears wherever the verdict appears. Re-verifying without the waiver needs no new simulation.
- Verification bounds its automatic work by a policy allowance. On exhaustion it starts no new work and reports the completed checks and unresolved requirements. A verifier that crashes, is aborted, or is stopped by the runner's timeout keeps its execution status and has no verdict.

**Rationale:** a separate run leaves the source run unchanged after `FINISHED`, keeps execution outcome distinct from verification, and allows verification to be repeated under a new policy. Showing the latest attempt prevents an older success from masking a newer failure. A third verdict separates missing evidence from evidence against a result, so neither a waiver nor an exhausted allowance reads as success or failure. A controlled stop can report what it established; a terminated verifier may have found a failure it never published, so it reports nothing.

### 2.4 Requirement-based dynamics verification

- **Integrity** at every stored state: trace, Hermiticity, minimum eigenvalue, and consistency of reported values with stored states. Integrity checks cannot be waived.
- **Diagnostics:** top retained-level population, and static dressed-state shifts and weights at a refined cutoff or between full and rotating-wave Hamiltonians. They are reported and never pass or fail a run.
- **Segments:** short windows restarted from stored states on an independent CPU engine (QuTiP, or dynamiqs when the run used QuTiP), in the source run's frame, at tightened tolerances and with each truncated subsystem refined by its declared step.
  - They run by default within the verification allowance, at the start, middle, and end of the window and at drive changes. Drive changes left uncovered are reported.
  - A segment that disagrees beyond the metric's accuracy target is a failed check.
  - Passing segments show local agreement over the reported coverage. A restart inherits any error already accumulated, so they satisfy no whole-window requirement and are never extrapolated.
- **Whole-window requirements:** numerical accuracy, truncation, and, for rotating-wave runs, the approximation are satisfied only by the full-window reference. Without it they stay unresolved, so a dynamics run with no failed check is `INCONCLUSIVE`. Spectral verification keeps its reference and refinement checks and needs no time-evolution reference.
- **Full-window reference, on request:** `compare_reference` solves the whole window on an independent CPU engine at tightened tolerances and refined truncation, in the lab frame for rotating-wave runs, and compares the reported observables and metrics at every stored time. It detects disagreement from engine, integration, truncation, and approximation together without separating them, and satisfies each requirement it covers.
- **Metric accuracy:** each metric has an accuracy target: a policy default, with an explicit override in the metric's units. The effective target is recorded, endpoint and trajectory accuracy are distinct, and segment and reference comparisons use it.
- **GPU runs** are verified on CPU after they complete. No reference or other scientific check precedes launch.
- **Sweeps:** requirements apply to the sweep as a whole.
  - Integrity and diagnostics run at every point.
  - Segments, and a requested reference, run at a deterministic, bounded set of evidence points selected after the sweep. A whole-window requirement is satisfied when the reference passes at every selected point.
  - Reports mark each point's coverage, and the verdict certifies nothing about unsampled points.
  - Warnings are all reported. When required evidence cannot be covered within the bound, the requirement stays unresolved and the report recommends a refinement.
- Reference and segment solves run on CPU engines within a reference dimension limit, separate from the dispatch cap and validated for both CPU reference engines by an exact-Liouvillian regression at that limit.

**Rationale:**
- Verification must cost a small fraction of an expensive run. A refined full-window reference repeats the computation that truncation and the rotating frame exist to avoid, so it runs only when the user asks for it.
- Automatic checks follow how researchers test convergence: integrity, then spot checks at tighter tolerance and more levels. They find failures cheaply but cannot certify a whole window, so a dynamics run without a failed check is `INCONCLUSIVE`, not `VERIFIED`.
- Segments see only the intervals they sample. In the probe a delayed pulse escaped start, middle, and end selection, and local residuals did not predict full-window errors.
- Lab-frame evolution restarted from a rotating-wave state shows local differences between the full and RWA dynamics, not the approximation's accumulated error. On the readout cases those differences exceeded both the target and the full-window error, so segments stay in the source run's frame.
- Static quantities without a demonstrated metric conversion missed every driven failure in the probe, so they are diagnostics.
- CPU references never wait for the GPU lock and can refine beyond the dispatch cap.

Measurements: [verification feasibility probe](../../tests/probes/m2_verification_results.json).

### 2.5 Engine dispatch

- Declared needs are `gradients` and `device`. Without gradients, a declared `device = "cuda"` selects cuDensityMat, and an undeclared device selects QuTiP CPU; gradients select dynamiqs. The selection table records needs, not performance, and the dispatch cap stays at dimension 30.
- Automatic performance-based dispatch requires measured CPU/GPU crossover and peak memory, together with a review of cuDensityMat open-system usage.

**Rationale:** a declared device exercises the GPU path without an unmeasured performance claim, and dispatch remains a recorded property of the run.

### 2.6 Drives and frames

- Lab-frame carrier drives are an envelope times `cos(2π f t + φ)`, with `f` in GHz, `t` in ns, and `φ` in radians, evaluated continuously through each engine's time-dependent coefficient path.
- Rotating frames are declared in the evolution input as frame frequencies applied to excitation-number operators in each subsystem's retained eigenbasis; explicit-operator models declare their own. The transformation is an explicit numeric operation on the constructed operators. A declared rotating-wave approximation removes counter-rotating terms, and provenance records the frame and the removed terms. Comparisons across frames use frame-invariant observables or states transformed back to the lab frame.
- Evolution uses the pinned adaptive solvers; no stiff or specialized integrators are added.

**Rationale:** a frame is a choice of computation, not a device property, so it belongs to the evolution input rather than the system specification. The rotating frame reduces cost; §2.4 states how its approximation is checked.

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| Separate append-only ledger store, `index`, and `latest` | Duplicate the runner records; `index` and `latest` serve the Run Inspector |
| amicode's frozen validators in full | Julia and control fields do not describe Python simulations or sweeps |
| Verdict in the protocol-written `result.toml` | Research code would write its own verdict, and verification after `FINISHED` would change a finished run |
| Verification inside the source run before `FINISHED` | `FINISHED` would wait on reference solves, verification failures would mix with execution outcome, and re-verification would edit a finished run |
| Agreement across all three engines for every run | Multiplies the cost of long runs; M1 fixtures establish three-engine agreement |
| Convergence refinement at every sweep point | Repeats the production computation at larger dimension |
| A full-window refined reference for every run or evidence point | Costs more than the run it checks and repeats the computation that truncation avoids; kept on request |
| Declared per-observable and per-metric rotating-wave bounds | Each bound needed its own frame study and a lab-frame reference to check it; metric accuracy targets replace them |
| General linkage and multi-photon analysis as a static screen | A research method rather than a check; diagonalization covers static linkage within retained and refined levels, and drive-induced linkage needs dynamics |
| Lab-frame segments as approximation evidence for rotating-wave runs | They show local differences between the full and RWA dynamics, not accumulated error, and rejected readout runs that met their target |
| Passing or extrapolated segments as whole-window evidence | A restart inherits accumulated error, and sampled intervals are not representative of the window |
| An exact step-propagator bound on numerical error at stored times | Applies only where the generator is constant in the run's frame, up to about dimension 30; a time-dependent generator needs a time-ordered propagator, which is another numerical solve |
| Static dressed-rate screens as approximation or truncation evidence | The demonstrated case covers only undriven runs with resonator loss |
| A CPU reference or static screen before every GPU launch | The reference costs more than the run, and no screen applies generally |
| Top retained-level population as a pass/fail check | Rejects exact truncations, such as decay from the top retained Fock state |
| Waivers through research tools or spec labels | Agents could remove the checks that judge their own results |
| Expected metric trends built into the corrector | The direction is physics of one model and parameter region; the researcher declares it per sweep |
| Evidence points added for every truncation warning | Widespread warnings would recreate an unbounded set of segment or reference solves |
| Production problems sized so refinement fits the dispatch cap | Restricts examples to fit a policy bound meant for dispatch |
| GPU lock held only by the runner | Released when the runner dies while its job still uses the GPU |
| Automatic GPU selection within the M1 cap | No measured crossover supports it |
| Rotating frames in the system specification | A frame is a computation choice, and one device model serves both frames |
| A separate corrector distribution | The policy digest provides the needed tamper evidence under the cooperative trust model |

## 4. Consequences

- The contract changes accordingly:
  - §2 describes the runner records, removes the verdict from `result.toml`, limits amicode compatibility to layout and terminal semantics, and adds the declared device need.
  - §3 replaces three-engine agreement with requirement-based verdicts, `INCONCLUSIVE`, and human waivers, using an independent reference engine.
  - §5 lists the waiver as a human action.
  - §6 identifies the runner records as the run ledger and lets a human promote an `INCONCLUSIVE` result that carries its unresolved scope.
  - §7's launch checks no longer include a reference-size check.
- The roadmap and checklist move `index` and `latest` to Block 12 and the cuDensityMat usage items out of Block 5, and add the verification step before the first sweep.
- Tests cover runner loss while the job survives, lock inheritance, reconciliation, verdict presentation after a failed newer attempt, evidence reuse, the evidence-point bound, label spoofing, and refinement-step minimums. They also cover verdict precedence, allowance exhaustion against verifier termination, recorded human waivers, segment failures and selection at drive changes, and the top-Fock diagnostic case.
- Without a requested reference, a dynamics run with no failed check is `INCONCLUSIVE`. The first example's acceptance shows both that outcome and `VERIFIED` with the reference.
- Whether research agents may declare a device is decided with the Block 6b permissions.
- Raising the dispatch cap, batching, reduced reference problems for runs above the reference limit, and performance-based dispatch follow the cuDensityMat usage study. Accuracy targets and allowances are set with the implementation.
