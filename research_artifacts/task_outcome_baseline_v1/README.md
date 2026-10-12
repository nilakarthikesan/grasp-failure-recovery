# Pre-lift task-outcome baselines, version 1

These are the actual CPU-trained baselines for the
[published 1,040-attempt dataset](../object_scene_pilot_v1/).
See [the Step 6 explanation](../../docs/PREDICTION_BASELINE.md) and
[working manuscript](../../docs/paper/working_draft.tex).

- `proprioception_and_commands.npz`: logistic coefficients and training-only
  normalization for the 288 joint-state/command summary features.
- `commands_only.npz`: the 96-feature command comparison.
- `report.json`: frozen input/source hashes, selection candidates and all
  partition metrics; both models selected regularization 0.1 on validation.
- `predictions.csv`: success/failure probabilities for every attempt and all
  three baselines, including the constant training-prevalence reference.
- `stratified_metrics.json`: post-hoc breakdown by scene, proposal, noise level
  and outcome category, using unchanged predictions and verified source metadata.

The full model's reserved apple-scene ROC AUC is 0.8523, compared with 0.6419
for commands only and 0.5000 for the constant baseline. At the fixed 0.5 failure
threshold it detects 187/202 failures and falsely alerts on 22/58 successes.

Inputs end at 2.96 s, before lifting. These outputs predict eventual simulated
task outcome; they do not measure slip onset, warning lead time, recovery,
calibration in deployment, unseen-geometry generalization or physical transfer.
The held-out scene is a retrospective development split of previously inspected
raw outcomes. Complete grouped memberships and limitations remain in the report.
