# ADR-0007: Calibration from Spectroscopy

**Date:** 2026-10-20, revised 2026-10-27
**Status:** Accepted
**Purpose:** fix what the first `calibrate` workflow fits and from which data, the measurement format, how peaks are extracted and assigned, how parameters and their uncertainty are reported, how holdout data is kept out of the fit and evaluated, how a measured bias reaches the model, how a device-model update is proposed and applied, and what the first version leaves out. Field lists, formats, and command names are documented with the implementation.

---

## 1. Context

The [harness contract](../../HARNESS_CONTRACT.md) fixes these points for calibration:
- **Device model (§1):** the device model holds the current device parameters, and humans update it from calibration proposals.
- **Checks (§3):** fitting receives training data only. The corrector evaluates holdout data without refitting, and provides an identifiability report using covariance or profile likelihood, on request like every other check.
- **Tools (§5):** `calibrate` is a controlled job, and `propose_device_update` is a controlled write next to the calibration run.
- **Write-back (§5, §6):** applying a proposal is a separate human action, and research tools expose no device-model write-back.

Calibration also needs a data contract with point-in-time provenance.

Before this record:
- **Parameters:** a system parameter has a value, a unit, an optional source, and an optional uniform prior. Construction uses the value and never samples the prior.
- **Spectra:** the `spectrum` mode reports a system's lowest energies. One metric assigns dressed states by overlap with bare product states.
- **Flux:** a circuit's external flux is a literal number, not a parameter, and sweeps cannot vary it. The fluxonium constructor's flux accepts a parameter.
- **Operator models** load fixed matrices, so no parameter enters them.
- **Gradients:** the differentiable engine differentiates evolution on a constructed model. It does not differentiate model construction.
- **Calibration:** there is no calibration protocol and no measurement format.

Interactive tools already serve a person fitting spectra at a screen. Entanglix serves automated workflows that the user or their agent drives, so the first workflow takes raw measurements without required human preprocessing.

The [calibration probe](../../tests/probes/s2_calibration.py) exercised the pipeline below on synthetic two-tone maps of a two-junction transmon. The harness does only what the user requests.

## 2. Decisions

### 2.1 Spectroscopy first

- **What is fitted:** `calibrate` fits model parameters to measured transition frequencies. The forward model is the spectrum of any system the harness builds, evaluated through the harness's own model construction.
- **Parameters:**
  - Fitted parameters are named from the system file's parameter table, as sweeps and control vary them. A fitted parameter must enter construction: a name that no construction field references is refused at launch, so operator models offer no fittable parameters.
  - Each fitted device parameter needs a declared uniform prior, whose bounds bound the fit.
- **Deferred:** calibration from dynamics (time-domain traces, gate-error tables) waits for a workflow that needs it.

**Rationale:** spectroscopy carries most of a superconducting circuit's Hamiltonian information. Its forward model needs only the constructed spectrum, so every model the harness builds is in scope without a new engine path. The differentiable engine does not differentiate construction, so a dynamics-based lane would need new machinery before any workflow asks for it. Prior bounds keep a fit inside the region the user declared plausible.

### 2.2 One measurement format

- **Uniform format:** every dataset is converted to one Entanglix format before any protocol reads it. The format is NPZ arrays, with pickle loading refused, plus a TOML sidecar.
- **What a dataset declares:**
  - its format version and measurement kind;
  - named axes with units;
  - the complex signal;
  - the conditions that decide which lines appear, such as drive and readout power and frequency, averaging, and temperature;
  - its provenance: device, cooldown, measurement time, the digest of the raw laboratory file, and the converter's name and version.
- **Two levels:** a dataset holds a raw map, extracted points, or both. A point has its centre and centre uncertainty, its bias coordinate when the dataset has a bias axis, and optionally its linewidth, amplitude, and assignment. Points from any source enter at this level. Intake checks types and units; whether a calibration has the fields its procedure uses is checked at its launch (§2.4).
- **Split:** a dataset can be split by a declared contiguous bias range into a holdout part and the remainder. Each part records its parent's digest and the bias indices it keeps, so a remainder of two intervals is described exactly. The split is mechanical and precedes any run.
- **Conversion** is mechanical: axes, signal, units, and metadata. It runs outside the harness, in the user's or agent's environment. Peak selection and scientific preprocessing belong to the declared pipeline, never to a converter.
- **Intake:** the harness validates a dataset when a run takes it in, and freezes it with its digest. It records the converter's provenance but cannot verify the conversion, and the report says so. When the raw file is supplied alongside, its digest is checked.
- **Laboratory converters** are added when a real dataset needs one, and still produce the uniform format. Project files of other tools are never loaded.

**Rationale:** one format means protocols read one thing and per-laboratory interfacing stays outside the harness. The runner's locked environment does not carry laboratory libraries. Converters vary between laboratories, so any scientific choice made inside one would be opaque. The same format already carries numeric model data.

### 2.3 Extraction is separate from fitting

- **Extraction** turns a map into points with declared settings, frozen with the run: background removal, the I/Q projection, the detection threshold, and the line shape. These settings encode assumptions; none is chosen silently.
- **Settings learned from data,** such as the projection direction, are learned from training data only.
- **Each peak reports** its centre, the uncertainty of the centre, its linewidth and its amplitude. The fit weights by the centre's uncertainty, not the linewidth. That uncertainty is conditional on the noise and line-shape model.
- **Failures** are reported per bias column: no peak, a failed line fit, or overlapping peaks that could not be separated.
- **Point-level datasets** skip extraction and use the same fitting path.

**Rationale:** separate stages leave inspectable intermediate data, and let points from any source share one fit. A precise centre can still be biased when the line shape is wrong, so the shape is declared and reported.

### 2.4 Assignment

- **A declared candidate family:** initial states, final states or a maximum excitation, photon orders, and named lines. Transitions are dressed transitions of the constructed model, by energy order at each bias, or bare labels by overlap where the family declares them. Nothing outside the family is predicted.
- **Supplied assignments:** when every training point carries an assignment, automatic assignment is skipped. Each label must name a member of the candidate family. The labels are validated at launch and kept fixed, and no anchor, matching setting, or resolution applies. A partly labelled input is refused at launch.
- **Required fields:** automatic assignment needs each point's linewidth, for the resolution test, and its amplitude when an anchor is declared. Supplied assignments need neither. A point missing a field its procedure uses is refused at launch.
- **One-to-one matching in each bias column:**
  - Matching a peak to a line costs its squared standardised distance.
  - Leaving a peak unmatched costs a declared penalty.
  - A peak within a declared resolution, in linewidths, of two candidate lines is unresolved. It is left out of the fit and costs the same penalty.
  - The total ranks assignments and solutions across starts. It carries no monotonicity guarantee.
- **Initialisation:**
  1. **Anchor,** when one is declared: the anchor transition is taken to be each column's largest-amplitude peak, and fitted first.
  2. **Matching range:** the full family is matched within a declared range in frequency units, and refitted until the assignment stops changing.
  3. **Final cutoff:** matching repeats within a declared cutoff in units of each peak's uncertainty, until the assignment stops changing.

  Reaching a declared iteration limit is reported as non-convergence.
- **Reported:** every peak as assigned, unexplained, or unresolved, and every predicted line in the window as matched or missing. Missing lines are not penalised, because line visibility is not modelled.

**Rationale:** nearest-line matching captures the true assignment only from starts whose predictions already lie within half the spacing of the candidate lines, and fabricated devices start further away than that. One line per column removes that ambiguity, and fitting it first brings the other lines within reach. One-to-one matching stops several peaks from claiming one line. The penalties make every excluded peak contribute to the ranking score, though they do not guarantee that excluding a peak cannot improve it. The anchor is an assumption: it is declared, and its failure is reported. Supplied labels are another source's assignment, so they are validated and kept as given; mixing them with automatic matching would need a rule of its own.

### 2.5 Fit

- **Method:** bounded weighted least squares, with finite-difference derivatives, from declared starting points.
- **Assignment:** the final fit uses the fixed final assignment.
- **Scales:** declared parameter scales set the step scaling and the numerical tolerances.
- **Forward model:** full construction at every evaluation. Construction may be reused across bias points and steps only when measured cost requires it, its structure is unchanged, and the reused evaluation agrees with full construction.

**Rationale:** the residuals are standardised by each peak's uncertainty, so this is the maximum-likelihood fit under independent Gaussian errors, and a more precisely estimated centre carries more weight. Finite differences work through every construction path. Analytic eigenvalue derivatives remain a later optimisation, because the Hamiltonian's parameter derivatives and its basis are not available through every construction.

### 2.6 Outcomes and uncertainty

- **Outcomes:**
  - a solution;
  - several distinct solutions found under the declared search;
  - insufficient evidence, when fewer peaks are assigned than parameters are fitted, or the Jacobian is rank-deficient;
  - "anchor initialization failed under the declared search", with its matching counts;
  - no fit.

  None of these is a pass or a fail.
- **Fit statistics:** χ², the number of assigned peaks, the number of fitted parameters, and the reduced χ², which is unavailable when the denominator is not positive.
- **Rank:** the numerical rank of the weighted Jacobian, scaled by the declared parameter scales, and the parameter combinations below it. The rank and undetermined-direction tolerances are the calibration protocol's own, and each fit records the values it used.
- **Covariance:**
  - It is reported only along the determined directions. A parameter along an undetermined direction has no finite uncertainty.
  - It is unscaled, using the declared uncertainties.
  - It is described as a local linear approximation, conditional on the assignment and on the noise model.
- **Also reported:**
  - correlations, and parameters that end on a bound;
  - each uncertainty divided by its prior width, as a descriptive quantity with no classification threshold. A large ratio is described as weakly constrained over the declared range.
- **Solution identity:** solutions merge numerically when their assignments are identical and their parameters agree within a declared scaled tolerance. The largest marginal separation between distinct solutions is reported under that name.
- **Not guaranteed:**
  - a search finds only the solutions reachable from its declared starts;
  - an unmodelled line that coincides with a predicted line is not recognised;
  - an anchor failure is observed, not diagnosed;
  - the absence of a failure does not establish a correct anchor.

**Rationale:** the report states what the data constrain, locally and under declared assumptions, and leaves the judgement to the user. A local Gaussian uncertainty that reaches beyond bounded parameter space is itself unreliable, so its ratio to the prior is shown, not turned into a verdict. Statistical separation along a nonlinear flat direction is not established by a marginal measure, so the measure keeps its name.

### 2.7 Holdout and checks

- **Holdout data is separated before the fitting launch.** It is either a separate measurement under comparable conditions, or one part of a split (§2.2). The fitting job never receives it, not even masked, so no setting is learned from it.
- **The user requests checks through `check`**, as for any run:
  - **Inputs:** the check request supplies the holdout dataset, and names the solution when several remain. The verification run freezes these inputs with their digests and binds them to the source run, as it already does for a replacement refinement. Every calibration check is inapplicable when the source run returned no fitted parameters.
  - **Investigations:** a calibration check is admitted by the `check` operation, like every check. The holdout dataset is not matched against a base; its frozen digest records what was evaluated.
  - **`holdout`:**
    - It is inapplicable without a holdout dataset. A dataset that is one of the source run's fitting inputs, or that shares bias points with one through a recorded split, as its parent or an overlapping part, is refused. A partly labelled holdout dataset is refused.
    - It uses only the procedure its source run recorded. Holdout maps need the source's frozen extraction settings, and unlabelled holdout points need its matching settings; without them the check is inapplicable. Fully labelled holdout points need neither, so a fit from labelled points can evaluate labelled holdout points.
    - The corrector extracts peaks from holdout maps with the frozen extraction settings, fitting the declared line shape to each peak. Fully labelled holdout points keep their labels, which must name members of the candidate family. Unlabelled peaks are matched one-to-one within the candidate family under the fitted parameters, with the source's cutoff and resolution. Distances are in units of the prediction uncertainty: the centre uncertainty combined with the parameter covariance propagated through the prediction, including the axis-map parameters.
    - It refits no model or axis-map parameter and learns no setting: the projection, the matching settings, and the fitted parameters stay as the source run left them.
    - It reports the counts (extracted, assigned, those assigned peaks whose prediction is not determined, unresolved, and unexplained), together with coverage over all extracted peaks and over scored peaks within one and two prediction uncertainties, and the mean squared standardised residual.
    - There is no pass or fail outcome unless the user declares an expectation on coverage over extracted peaks. That expectation is then checked as declared.
  - **`identifiability`:** the corrector recomputes, independently of the fitting job, the rank, the undetermined combinations, the covariance, the correlations, and the prior-width ratios at the fitted parameters. It uses the frozen training data, the final assignment, and the corrector policy's tolerances. It reports agreement with the fit's own report, with the tolerances each used, and is descriptive.
  - **`refinement`:** the predicted frequencies of the assigned peaks at a declared refined truncation, standardised by each peak's uncertainty. It is descriptive, and inapplicable without a declaration in the calibration spec or the check request.

**Rationale:**
- **Holdout split:** a mask still lets shared preprocessing see the holdout data, so the split precedes the fitting launch.
- **Holdout route:** the check request carries the holdout data, so the fitting job's inputs cannot contain it, and the verification record shows what was evaluated against which solution.
- **Recorded tolerances:** the fit and the check each record their tolerances, so a later policy change explains a difference between them.
- **Prediction uncertainty:** extrapolating into withheld data carries the fitted parameters' uncertainty. Matching in units of the centre uncertainty alone discards correct predictions as unexplained.
- **Coverage over all extracted peaks:** unmatched peaks count against this figure, so an expectation cannot pass when little matched.
- **Independent recomputation:** check reports are the corrector's, so the identifiability report is recomputed, as replay recomputes a solver's fidelity.
- **Refinement:** truncation error can be absorbed into fitted parameters. The refinement check shows it against the data's own uncertainty.

### 2.8 Flux binding and symmetries

- **Flux as a parameter:** a circuit's external flux may be a parameter reference, as the fluxonium constructor's flux already is. Ordinary sweeps over flux then work through parameters.
- **Axis map:**
  - A dataset's bias axis binds to a named flux parameter through a declared affine map from bias to flux.
  - Its offset and period are nuisance parameters declared in the calibration spec, with bounds and scales.
  - They are fitted and reported in the calibration record, and never proposed as device updates.
  - A dataset without a bias axis needs no axis map. It is evaluated at the system's own flux values, and a flux parameter there may be fitted directly.
- **Symmetries:** conventions declared in the spec remove known equivalent solutions: a positive period, an offset within one declared window, and a declared ordering of parameters that the model makes exchangeable. The report lists the equivalents removed. There is no general symmetry discovery, and an exchange convention applies only where the declared model has the symmetry.
- **First version:** one bias axis drives one flux.

**Rationale:** parameter references reuse the existing parameter mechanism. The axis map belongs to the measurement setup, not to the device. Without conventions, aliases of one solution would appear as distinct solutions.

### 2.9 Device-update proposals

- **`propose_device_update`** writes a proposal next to the calibration run. For each proposed parameter it lists:
  - the old value and the new value;
  - the uncertainty and the prior-width ratio as reported by the fit;
  - the source run and the solution.
- **Contents:**
  - It contains only device parameters from the system file's parameter table.
  - Parameters not numerically determined are listed as not proposed, with the reason.
  - Nuisance parameters and the joint covariance stay in the calibration record.
  - The proposal records which checks ran on its source run and their outcomes.
- **Choosing a solution:** when several solutions remain, the caller names the one that supplies the proposal.
- **Binding:** the proposal records the digest of every model input its source run froze: the system file, subsystem sources, and array files. The human apply command refuses when any of them has changed.
- **Applying** writes the new numbers into each parameter's value and the provenance into its source, in the system file. Priors are unchanged.

**Rationale:** a silent change to the device model would change every later control and co-design result, so a human applies it. A digest binding over every model input stops a proposal from overwriting later edits, including edits to a referenced subsystem. Write-back stays on the parameter surface that sweeps and control already vary. The joint uncertainty has no home in the device model yet, so it stays where it was computed.

### 2.10 Investigations and tools

- **Investigations:** `calibrate` is a permitted operation of an investigation. A calibration job is in scope when its frozen inputs — system, calibration spec, and datasets — match a permitted base exactly. Usage, limits, the deadline, and stops apply as to any job.
- **Reruns:** a rerun re-fits frozen inputs as a new linked attempt.
- **MCP:** the server offers `calibrate` in the form of the existing job tools, and `propose_device_update` as a controlled write. Applying a proposal is a human action on the operator CLI, restricted like the other human actions.
- **Skill:** a `calibrate` skill gives the procedure.

**Rationale:** exact matching uses the digests that admission already checks, and the starting points are part of the spec. The tool and permission classes are the contract's.

### 2.11 Scope of the first version

- **In:** transition frequencies from spectroscopy maps or points, with one bias axis driving one flux, on the CPU.
- **Holdout:** a separate measurement, or a bias range. Holding out a transition waits for a workflow that needs it, because its pixels can share columns with training lines.
- **Out:**
  - dynamics-based calibration;
  - gate-charge axes, and several loops driven from one coil;
  - partly labelled inputs;
  - profile likelihood and posterior sampling;
  - laboratory converters before a real dataset needs one;
  - interactive tools.

**Rationale:** each of these is a design of its own, scheduled with the workflow that needs it.

Measurements: [calibration probe results](../../tests/probes/s2_calibration_results.json).

## 3. Alternatives not adopted

| Alternative | Reason |
|---|---|
| A dynamics-based gradient lane first | No first workflow needs it, and construction is not differentiable |
| A simplified forward model in place of the harness's construction | Would restrict calibration to the models it reimplements |
| Analytic eigenvalue derivatives first | Parameter derivatives and the basis are not available through every construction |
| Fitting the map's pixels directly | Needs line-amplitude, broadening, and readout-contrast models with many nuisance parameters |
| Unweighted errors | Treats every peak as equally precise |
| Bayesian optimisation | Gives a best point without parameter uncertainty, and a spectrum evaluation is cheap |
| Posterior sampling first | Costs far more evaluations and a new dependency without an established need |
| Nearest-line matching without one-to-one matching or penalties | Several peaks can claim one line, and discarding peaks can lower the score |
| Successive strongest-line assumptions to remove every initialisation failure | Each adds an assumption; visible failures are a documented limitation |
| Required human labels | Automation is the purpose; labels are an optional input |
| Readers for laboratory formats inside the runner environment | Brings laboratory libraries into the locked environment |
| Loading another tool's project files | They can execute code |
| A holdout mask over a shared map | Shared preprocessing would see the holdout data |
| Holdout matched in units of centre uncertainty | Discards correct predictions as unexplained |
| A classification threshold on uncertainty relative to the prior | A local uncertainty beyond the bounds is itself unreliable |
| Scaled covariance | Hides whether the declared uncertainties describe the residuals |
| A pass or fail on holdout without a declared expectation | Checks report their scope; a verdict is the user's |
| Proposing axis-map parameters | They describe the measurement setup, not the device |
| Applying a proposal through a research tool | The contract keeps device-model write-back a human action |

## 4. Consequences

- **Contract §3:** the `calibrate` row names the `holdout`, `identifiability`, and `refinement` checks, with the scope stated above.
- **System spec:** a circuit's external flux accepts a parameter reference.
- **Operational references:** a new calibration reference documents the measurement format, the calibration spec, the outcomes and report, the checks, and proposals. The system-spec, investigation, and MCP references document flux parameters, calibration bases, and the new tools.
- **Example:** a calibration example records a synthetic round trip in its own directory.
- **Tests cover:**
  - **Data and inputs:** dataset validation and refusals, including pickled content; points with only a centre and its uncertainty, and datasets without a bias axis; a split's parts disjoint and recording their parent and selected indices, including an interior range; a parameter that enters no construction refused; flux parameter references in circuits and in sweeps.
  - **Extraction and assignment:** extraction settings frozen and the projection learned from training data only; supplied labels kept fixed, and a label outside the family, a partly labelled input, or a missing field the procedure uses refused at launch; one-to-one matching with unexplained and unresolved peaks penalised; the anchor phase, and its failure reported with counts; non-convergence at the iteration limit.
  - **Fit and report:** fixed-bias data fitted without an axis map; a rank-deficient case reported with its undetermined combination and no finite uncertainty; reduced χ² unavailable when the denominator is not positive; the tolerances recorded; symmetry conventions and the equivalents they remove; reused construction, if added, agreeing with full construction.
  - **Checks:** holdout data absent from the fitting job's inputs, supplied with the check request, frozen, and bound to the source run; a holdout dataset that is a fitting input or overlaps one refused; a partly labelled holdout dataset refused; holdout inapplicable when the source recorded no procedure for the holdout's form, and labelled holdout points evaluated after a labelled-point fit; each check inapplicable without its input or without fitted parameters; holdout counts and both coverages, with matching in prediction-uncertainty units; the identifiability check agreeing with the fit's report; refinement inapplicable without a declaration.
  - **Proposals and tools:** a proposal or check refused for a solution not named among several; apply refused after any frozen model input changed; apply writing value and source into the system file; `calibrate` admission against an exact base; MCP and CLI parity, with no write-back tool on the research surface.
