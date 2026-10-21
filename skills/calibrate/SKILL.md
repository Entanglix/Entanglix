---
name: calibrate
description: Fit Entanglix device parameters to spectroscopy, request calibration checks with withheld data, and stage a device-update proposal for the operator to apply.
version: 0.1.0
---

# Calibration from spectroscopy

Read workspace `reference/operations/calibration.md` for the dataset format,
holdout splits, the calibration spec, outcomes, checks and proposals. Use
`reference/operations/investigations.md` for approved terms. Read
`reference/operations/runs.md` only when an attempt is interrupted or its state
is unknown.

Measurements arrive converted to the dataset format, with withheld data split off
before any fit: a holdout part from `split-dataset` or a separate measurement. The
calibration spec lists training datasets only. Never list a holdout part, its parent
or an overlapping part there, and never fit withheld data. If no withheld data was
supplied, report that the fit has no holdout evaluation.

Call `calibrate` with the authored spec. In an investigation read effective terms
first and name an approved base: the spec, system and datasets must match it
exactly, and any change needs another approved base. The tool returns admission,
not a fit. Pass the returned path to `read_run`, optionally with a bounded wait.

Read the run's `outcome` before any value. `solution` and `several distinct
solutions` carry values; `insufficient evidence` names the undetermined
combinations, and a parameter along one has no uncertainty; an anchor failure and
`no fit` carry no solution. Read each start's status for non-convergence. None of
these outcomes is a pass or a fail. Uncertainties are local, along the
determined directions, conditional on the assignment. Report each parameter's
uncertainty and its ratio to the prior width; a large ratio means the data barely
constrain it. Report bounds hit and conventions not applied. With several
solutions, report their largest marginal separation and name one by index with
`solution` for every check and proposal; do not choose one silently.

Tell the user which checks are available: `holdout` scores withheld data at the
fitted values without refitting; `identifiability` recomputes the diagnostics with
the corrector's tolerances; `refinement` predicts the assigned peaks at a refined
truncation. Issue `check` for explicitly or standing-requested names. Pass the
withheld dataset's path in `holdout`, and `expect_coverage` as `[within, minimum]`
only when the user declares that expectation; otherwise the checks are
descriptive. Read the check run with `read_run(detail=True)` for holdout counts and
coverage, the identifiability agreement and the refinement shifts, which the
default summary lists under `detail_omitted`. Coverage over all peaks counts
unexplained, unresolved and undetermined peaks against the fit.

Call `propose_device_update` with the run, and `solution` when several remain, only
when the user wants an update. It writes a proposal beside the run and changes
nothing else. Report its proposed and not-proposed parameters, uncertainties,
bounds hit and the checks it records. Applying is the operator's action: give the
exact bare command `entanglix apply-device-update <run path>/proposals/<n>.toml
<system file>` with the user's authored system file, and never run it, script it or
edit the system file yourself. If apply reports a changed model input, tell the user
which one so they can inspect it; even a comment-only edit changes its digest.

Settle an interrupted attempt once ownership and group-liveness checks permit
reconciliation. Use `rerun_calibration`, not a fresh fit of the same inputs, to
retain lineage and investigation membership under current approved terms. Checks
and proposals belong to the run they name, so a rerun has none until they are
requested. Record findings with source run
IDs, outcome, determined parameters, holdout coverage, unresolved limitations and
the next decision.
