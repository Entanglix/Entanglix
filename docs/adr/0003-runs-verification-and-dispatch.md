# ADR-0003: Run Records, Verification, and Engine Dispatch

**Date:** 2026-09-30
**Status:** Accepted
**Purpose:** fix how runs are recorded, how completed runs are verified at bounded cost, how a run reaches the GPU without a performance claim, and how driven evolution is represented in the lab and rotating frames. Field lists, formats, and threshold values are documented with the implementation.

---

## 1. Context

The [harness contract](../../HARNESS_CONTRACT.md) requires a run directory with `run.toml` first and `FINISHED` last (§2), an independent corrector with dynamics agreement, convergence, and physicality checks (§3), an append-only history of every attempt (§6), and GPU exclusivity, cancellation, and reconciliation of unknown process state on the workstation (§7). [ADR-0001](0001-mandate.md) sets a cooperative trust model and a lean design that reuses process and platform facilities rather than building services.

M1 established agreement across QuTiP CPU, cuDensityMat, and dynamiqs on reference fixtures, a validated dispatch cap of dimension 30, and a selection policy that sends every non-gradient request to CPU. No measurement yet shows where the GPU is faster. Production runs may take hours, so verification that repeats a run on several engines or refines every sweep point would multiply its cost. Drives are constant within each output interval and evolution uses the lab frame, which makes resonant GHz drives impractical.

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

- Verification is its own run whose parent is the source run. Its `result.toml` holds the verdict, `VERIFIED` or `UNVERIFIED`, and a report bound to the source artifact digests and the verification policy version. Protocol results carry no verdict.
- The corrector is a subpackage of the same distribution, with versioned thresholds as package data outside run directories and authored-script workspaces. The verification policy version is the digest of the corrector code and thresholds; it is recorded at launch and recomputed when the corrector loads, and a mismatch fails the verification run without a verdict.
- The runner starts verification after each completed `simulate` or `sweep` run; `compare_reference` starts it on demand. Evidence is reused when plan digests and the policy version match.
- Reading a run shows the latest verification attempt's status and whether a verdict exists for the requested policy, by default the current one. Earlier verdicts remain historical evidence labeled with their policy and source digests and never substitute for a missing current verdict.
- Applicable checks derive from trusted definitions keyed by computation kind (Lindblad evolution, spectrum, parameter series) and from the validated plan's content. Labels cannot remove checks, refinement steps below the policy minimum are rejected, and runs without a trusted definition, including authored scripts, cannot verify.

**Rationale:** a separate run leaves the source run unchanged after `FINISHED`, keeps execution outcome distinct from verification, and allows verification to be repeated under a new policy. Showing the latest attempt prevents an older success from masking a newer failure.

### 2.4 Scoped dynamics verification

- Every point of a run receives output checks: trace, Hermiticity, and minimum eigenvalue of stored states. Top retained-level population is reported as a truncation warning, not as convergence evidence.
- Each evidence point receives one combined reference solve: an independent CPU engine (QuTiP, or dynamiqs when the run used QuTiP) at tightened tolerances, with each truncated subsystem refined by its declared step, compared on the reported observables and metrics. Rotating-frame runs use a lab-frame reference.
- A single run is its own evidence point, so `VERIFIED` requires its combined solve. For a GPU run that solve runs before launch as a linked run, satisfies the contract's pre-launch reference check, and is reused by verification.
- For sweeps, the verification policy fixes a deterministic selection rule and a maximum evidence-point count, including pre-launch corners. Grids whose corners exceed the bound are rejected before launch. Warnings are all reported, but only the bounded worst cases receive solves. When required evidence cannot be covered within the bound, the verdict is `UNVERIFIED` with a refinement recommendation.
- `VERIFIED` means every output check and every evidence-point comparison passed. It certifies nothing about convergence at output-checked points; reports mark each point's coverage.
- Reference solves run on CPU engines and are bounded by a reference dimension limit, separate from the dispatch cap and validated for both CPU reference engines by an exact-Liouvillian regression at that limit.

**Rationale:** the expensive production computation runs once, and verification cost is bounded by the number of evidence points rather than by grid size or engine count. The count does not bound each solve's duration: a standalone simulation still needs a full-window refined CPU solve unless matching evidence exists. One independent engine preserves the independence the corrector needs. A combined solve bounds engine, integration, and truncation differences together at the cost of not localizing a failure, which a diagnostic rerun can do. CPU references never wait for the GPU lock and can refine beyond the dispatch cap.

### 2.5 Engine dispatch

- Declared needs are `gradients` and `device`. Without gradients, a declared `device = "cuda"` selects cuDensityMat, and an undeclared device selects QuTiP CPU; gradients select dynamiqs. The selection table records needs, not performance, and the dispatch cap stays at dimension 30.
- Automatic performance-based dispatch requires measured CPU/GPU crossover and peak memory, together with a review of cuDensityMat open-system usage.

**Rationale:** a declared device exercises the GPU path without an unmeasured performance claim, and dispatch remains a recorded property of the run.

### 2.6 Drives and frames

- Lab-frame carrier drives are an envelope times `cos(2π f t + φ)`, with `f` in GHz, `t` in ns, and `φ` in radians, evaluated continuously through each engine's time-dependent coefficient path.
- Rotating frames are declared in the evolution input as frame frequencies applied to excitation-number operators in each subsystem's retained eigenbasis; explicit-operator models declare their own. The transformation is an explicit numeric operation on the constructed operators. A declared rotating-wave approximation removes counter-rotating terms, and provenance records the frame and the removed terms. Comparisons across frames use frame-invariant observables or states transformed back to the lab frame.
- Evolution uses the pinned adaptive solvers; no stiff or specialized integrators are added.

**Rationale:** a frame is a choice of computation, not a device property, so it belongs to the evolution input rather than the system specification. The rotating frame reduces cost, and the lab-frame reference in §2.4 bounds its approximation.

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| Separate append-only ledger store, `index`, and `latest` | Duplicate the runner records; `index` and `latest` serve the Run Inspector |
| amicode's frozen validators in full | Julia and control fields do not describe Python simulations or sweeps |
| Verdict in the protocol-written `result.toml` | Research code would write its own verdict, and verification after `FINISHED` would change a finished run |
| Verification inside the source run before `FINISHED` | `FINISHED` would wait on reference solves, verification failures would mix with execution outcome, and re-verification would edit a finished run |
| Agreement across all three engines for every run | Multiplies the cost of long runs; M1 fixtures establish three-engine agreement |
| Convergence refinement at every sweep point | Repeats the production computation at larger dimension |
| Evidence points added for every truncation warning | Widespread warnings would recreate an unbounded set of reference solves |
| Production problems sized so refinement fits the dispatch cap | Restricts examples to fit a policy bound meant for dispatch |
| GPU lock held only by the runner | Released when the runner dies while its job still uses the GPU |
| Automatic GPU selection within the M1 cap | No measured crossover supports it |
| Rotating frames in the system specification | A frame is a computation choice, and one device model serves both frames |
| A separate corrector distribution | The policy digest provides the needed tamper evidence under the cooperative trust model |

## 4. Consequences

- The contract changes accordingly: §2 describes the runner records, removes the verdict from `result.toml`, limits amicode compatibility to layout and terminal semantics, and adds the declared device need; §3 replaces three-engine agreement with scoped coverage and an independent reference engine; §6 identifies the runner records as the run ledger. The roadmap and checklist move `index` and `latest` to Block 12 and the cuDensityMat usage items out of Block 5.
- Tests cover runner loss while the job survives, lock inheritance, reconciliation, verdict presentation after a failed newer attempt, evidence reuse, the evidence-point bound, label spoofing, and refinement-step minimums.
- Whether research agents may declare a device is decided with the Block 6b permissions.
- Raising the dispatch cap, batching, reduced reference problems for runs above the reference limit, and performance-based dispatch follow the cuDensityMat usage study. The first sweep's exact metric and evidence thresholds are chosen from the Block 5 feasibility probe.
- A change to a decision above requires a new ADR that supersedes this one.
