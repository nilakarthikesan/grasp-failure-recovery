# Completed experiments and evidence

Updated 13 October 2026. This index reports saved simulation results.
The [research protocol](HUG_MONITOR_RESEARCH_PLAN.md) separates the recorded
research question and pending comparisons from completed experiments.
The [working paper](paper/working_draft.tex) draws on the evidence below.

## Original pilot and baseline

- **Collection:** 1,040 attempts, with 444 successes and 596 task failures.
  The data contains cube and apple geometry, four observation families and
  sixteen HUG proposals. [Collection and provenance](MULTI_SCENE_COLLECTION.md).
- **Inputs and splits:** closing histories through 2.96 seconds, before lift
  begins at 3.00 seconds. Observation-grouped partitions contain 520 training,
  260 validation and 260 test attempts. [Split contract](DATA_SPLITS.md).
- **Baseline:** the state-and-command logistic model has failure-ranking
  AUROC 0.8523 on the reserved apple observation, versus 0.6419 for commands
  alone. At the original 0.5 warning threshold it detects 187/202 failures,
  misses fifteen, and falsely warns on 22/58 successes.
  [Model guide](PREDICTION_BASELINE.md) and
  [saved baseline](../research_artifacts/task_outcome_baseline_v1/README.md).

This split was declared after collection outcomes had been inspected. Both
geometries appear in training, and the test contains one observation family.
It is a retrospective development result rather than an unseen-object benchmark.

## Frozen evaluation on fresh objects and scenes

A subsequent batch contains 2,080 attempts: 659 successes and 1,421 task failures.
The original model, normalization and warning threshold were frozen.
Failure-ranking AUROC is 0.9173 on fresh cube/apple scenes and 0.5588 across
new milk carton/mug attempts.

All original mug attempts fail during approach because the retargeted hand
strikes the table. That batch cannot measure false alarms on successful mugs.
[Evaluation guide](FRESH_OBJECT_EVALUATION.md) and
[portable evidence](../research_artifacts/fresh_object_scene_eval_v1/README.md).

## Mug executor correction and frozen-model scoring

A separately versioned executor corrects the forearm/wrist clearance and adds
a runtime guard. Twenty-six checks complete without table contact; one
high-noise closure stops before contact and remains outside binary metrics.
[Correction and checks](MUG_TABLE_CLEARANCE.md).

The unchanged predictor scores all 26 completed checks. The mug subset contains
twenty successes and three task failures, and all 23 trigger warnings.
The stopped attempt has no outcome label or prediction.
[Scored recordings](../research_artifacts/corrected_mug_model_diagnostic_v1/README.md).

This establishes persistent false alarms on corrected successful mugs.
It does not demonstrate a training remedy.

## Wrist input and paired diagnostics

Both wrist commands in all twenty successful corrected mugs lie outside the
original training ranges. Reconstructed wrist feature terms push their saved
scores toward failure. This is linear score attribution, not isolated physical
causation. [Input diagnostic](../research_artifacts/mug_wrist_input_diagnostic_v1/README.md).

None of the eight observed training wrist-command pairs meets the declared
static table-clearance gate for the eight mug proposals. The later comparison
therefore uses two unfamiliar, feasible configurations per proposal.
[Feasibility](../research_artifacts/wrist_pose_feasibility_v1/README.md) and
[frozen selection](../research_artifacts/wrist_pair_selection_v1/README.md).

All eight zero-noise pairs complete: sixteen attempts, thirteen successes and
three task failures. Every attempt triggers a warning; six pairs succeed at
both wrist settings. One changed label crosses the 14 cm lift cutoff by a
submillimeter amount and is not evidence of a drop.
[First pair](../research_artifacts/wrist_pair_diagnostic_v1/README.md) and
[remaining seven pairs](../research_artifacts/wrist_pair_remaining_v1/README.md).

The corrected checks and paired diagnostics reuse development families and
have distinct executor versions. They must be reported separately.

## Partner provider foundation and replay gate

PRs 21 through 24 add interchangeable grasp proposal sources, a unified
evaluation command, explicit pose/geometry translation checks and official
DexGraspNet sample replay. All are merged into main through `b503c2f` and
integrated into the ongoing working branch.

The recorded ten-sample mug gate has seven approach collisions, one failed
acquisition and two null-label joint-limit rejections, with no successes.
This is a dataset-replay execution check, not a trained predictor evaluation.
[Replay and translation scope](DEXGRASPNET_REPLAY.md) and
[gate evidence](../research_artifacts/dexgraspnet_replay_gate_v1/README.md).

The generic v2 output lacks velocity and commanded-target histories required
by the existing predictor exporter. The fresh DexGrasp Anything worker and
compatible prediction-data recording remain implementation work. No model improvement is
established by importing these proposals.

ContactGen MANO and UniDexGrasp Shadow adapters and isolated inference workers
are also implemented on the working branch. Their unit fixtures are synthetic;
fresh inference, model-output acceptance and simulation checks have not run.
No evaluation data was generated for this implementation.
[Implementation scope and contracts](CONTACTGEN_AND_UNIDEXGRASP.md).

## One fresh HUG CLI execution check

One newly authorized HUG inference and MuJoCo case completes successfully through
`grasp-eval`: 0.151335 m maximum lift, 2.0 s hold and no hand/table contact in
163 saved samples. The existing cube observation and seed 0 reproduce the
earlier native proposal and saved-replay trajectory exactly. This is a
development reproducibility check, not an independent benchmark. The frozen
predictor was not scored. [Run declaration, results and runtime](HUG_SINGLE_EVAL_2026-10-13.md).

## Claim boundaries and paper status

The saved results establish a simulation warning-reliability problem.
They do not yet establish improved prediction from added coverage, calibrated
real-robot failure probabilities, slip warning lead time or recovery benefit.
The larger coverage comparison and calibration/representation fits remain
unexecuted in the recorded evidence.

The manuscript remains an editable working draft. Submission preparation is
tracked in [the submission checklist](SUBMISSION_CHECKLIST.md).

The detailed milestone and exploration chronology is preserved in
[the archived progress record](archive/PAPER_PROGRESS_HISTORY_2026-10-12.md).
[Dated decisions](RESEARCH_DECISIONS.md) retain why the experiment design changed.
