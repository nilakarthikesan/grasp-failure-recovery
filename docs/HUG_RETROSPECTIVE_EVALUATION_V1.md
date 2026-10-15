# Retrospective HUG object holdouts v1

Declared 15 October 2026 before this new fit/evaluation. This is development
analysis of previously inspected outcomes, not a prospective independent test.
The historical frozen-model evaluation remains unchanged. No new simulation,
label, geometry, hand, controller or collection is authorized by this analysis.

## Frozen inputs and scope

Use only `research_artifacts/object_scene_pilot_v1` (1,040 attempts) and
`research_artifacts/fresh_object_scene_eval_v1` (2,080 attempts). Require identical
input contract, feature columns, label rule and scorer identity before joining
their derived prefixes. The fresh batch's frozen execution reference binds the
original protocol. Do not include corrected-mug or paired-wrist results: those
use distinct executor versions. Preserve the original files and prior metrics.
All source payloads/manifests, this protocol and analysis code are hashed in the
report. Episode IDs are namespaced by source dataset to avoid collisions.

## Membership and models fixed before fitting

Hold out each entire geometry in turn. For each remaining geometry, assign its
lexicographically last observation content ID to validation; all its other
observations train. Thus each proposal and all repeated perturbations remain
in one partition. Assignment never consults labels, scores or noise magnitude.
This rule is deterministic; its arbitrariness and previously inspected data
limit interpretation. These four geometries are examples, not a population.

Fit training prevalence, commands-only summary logistic regression and the
existing state-plus-command summary logistic regression. Preserve the existing
20 samples through 2.96 s, 24 joints, four summary statistics, fixed L2 grid
0.001/0.01/0.1/1.0, training-only normalization and validation log-loss selection.
Use a fixed failure-warning threshold of 0.5. No test-based choice, calibration
fit, extended feature search or new architecture is performed.

## Metrics and failure handling

Report AUROC, failure average precision, Brier score, log loss, fixed ten-bin
success-probability calibration, false warnings and missed failures together.
Use 1,000 proposal-family bootstrap resamples, seed 20261015, including every
episode of a sampled family. Percentile intervals are conditional on the one
held-out geometry and do not measure uncertainty across unseen objects. Report
family/capture counts alongside attempts. Undefined single-class AUROC and
false-alarm rate stay null; a held-out mug cohort with no successes cannot
measure false warnings on successful mugs. No resampling repairs class balance.
If training/validation cannot fit a model, retain the error rather than tuning
membership or criteria. Leave-one-failure-mode-out is deferred: pre-cutoff
collisions require a separate risk-set design before such an analysis is valid.

## Reproduce

```sh
PYTHONPATH=src /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  -m grasp_failure_prediction.prediction.retrospective_holdout \
  research_artifacts/object_scene_pilot_v1 \
  research_artifacts/fresh_object_scene_eval_v1 \
  --protocol docs/HUG_RETROSPECTIVE_EVALUATION_V1.md \
  --output research_artifacts/hug_retrospective_object_holdout_2026-10-15_v1
```

Freeze this declaration and analysis implementation in Git before that command.
Do not interpret these development fits as improving the original frozen model
or certifying the pending fresh-mapping/collection pipeline.
