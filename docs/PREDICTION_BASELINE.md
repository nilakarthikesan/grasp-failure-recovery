# Step 6: a first prediction before lifting

The first model is trained and evaluated on the
[frozen Step 5 dataset](DATA_SPLITS.md). It answers: **given how the hand has
closed so far, will this attempt meet our final grasp/lift/hold task rule?**
It does not estimate when a slip begins or select a recovery action.

## What the model sees

At **2.96 s**, before lifting starts at 3.00 s, it reads twenty closing samples
of measured joint positions, measured velocities and applied command targets.
For each channel it computes the last value, mean, population standard
deviation and end-minus-start change. This gives 288 features. A commands-only
comparison uses 96 features; a constant baseline always predicts the training
success fraction, 231/520.

The classifier is L2-regularized logistic regression. Scaling uses training
data only; channels with training standard deviation at most 1e-8 are removed.
The full model retains 269 channels and the commands-only model retains 77.
For each model, the fixed regularization grid is 0.001, 0.01, 0.1 and 1.0.
All candidates converged. Validation log loss selects **0.1** for both models,
with 25 and 15 optimization iterations respectively. Test outcomes do not
select features, scaling, regularization or the alert threshold.

Training has 520 attempts, validation 260 cube-B attempts and test 260 apple-B
attempts. Complete observations/proposals and all noisy variants stay together.
The split was declared after inspecting the original collection; it is a
retrospective development holdout. Both geometries occur in training.

## What happened on the reserved apple scene

- **Joint states + commands:** failure-ranking ROC AUC **0.8523**, log loss
  **0.3771**, Brier score **0.1170**, balanced accuracy **0.7732**.
- **Commands only:** ROC AUC **0.6419**, log loss **0.5244**, Brier score **0.1711**,
  balanced accuracy **0.5270**.
- **Training prevalence:** ROC AUC **0.5000**, log loss **0.6374**, Brier score
  **0.2222**, balanced accuracy **0.5000**.

Lower log loss and Brier score are better; higher ranking AUC and balanced
accuracy are better. Ordinary accuracy alone is misleading here: 202/260 test
attempts failed, so flagging every attempt already gets 77.69% accuracy.

At the predeclared rule **alert if p(failure) ≥ 0.5**, the full model catches
**187/202 failures**, misses **15**, correctly clears **36/58 successes**, and
falsely alerts on **22/58 successes**. Its failure recall is 92.57% and success
false-alarm rate is 37.93%. Improved probability scores do not by themselves
establish calibration or real-world failure probabilities.

The [saved report](../research_artifacts/task_outcome_baseline_v1/report.json)
contains train, validation and test metrics, every validation candidate, model
hashes and the input/source contract. The
[per-attempt predictions](../research_artifacts/task_outcome_baseline_v1/predictions.csv)
include all three baselines. The
[descriptive breakdown](../research_artifacts/task_outcome_baseline_v1/stratified_metrics.json)
uses those unchanged predictions, with no fitting or threshold changes.
For example, test AUC is 0.9232 at 0.15 rad and 0.8082 at 0.6 rad; these are
128 related attempts at each level, spanning only four proposals in one scene.

## Walk through one attempt

The first test failure, `episode_000390`, has label 0. From its closing history
alone the full model gives p(failure) = **0.6045**, so it raises an alert. The
first test success, `episode_000395`, has label 1 and p(failure) = **0.1825**,
so it clears the attempt. These examples were selected as the first test row
of each class, rather than as the most confident predictions.

We reproduced both probabilities from directories containing only
`trajectory.npz` and `actions.npz`, without metadata, result or event files.
They agree with predictions from the portable inputs to within 1e-14.
This checks that inference can run without opening the outcome label.

An honest error example is the existing replay `episode_000534`: it later drops,
but at 2.96 s the model gives p(failure) = **0.4815** and clears it. This episode
belongs to training, so it illustrates an error rather than test performance.
Only four episodes in the entire collection are task-scored drops; this is
primarily a failed-acquisition dataset, not evidence of specialized slip detection.

## Run it

The compact published dataset and coefficients run without HUG or a renderer:

```sh
pip install -e '.[prediction]'

# Train in a new output directory; existing output is never overwritten.
python -m grasp_failure_prediction.prediction.model \
  research_artifacts/object_scene_pilot_v1 \
  --output /path/to/new_baseline

# Predict from one original local recording, without opening its outcome files.
python -m grasp_failure_prediction.prediction.model \
  --model research_artifacts/task_outcome_baseline_v1/proprioception_and_commands.npz \
  --episode /path/to/collection_1040/episode_000390

# Optional descriptive analysis requires the original metadata for grouping.
python scripts/analyze_task_outcome.py \
  research_artifacts/object_scene_pilot_v1 \
  research_artifacts/task_outcome_baseline_v1 \
  --runs-root /path/to/collection_1040 \
  --output /path/to/new_stratified_metrics.json
```

The recorded CPU training runtime is Python 3.12.13, NumPy 2.2.6 and SciPy
1.17.1. Model files contain coefficients and training normalization, loaded
with `allow_pickle=False`. The report pins the training implementation hash.

## Evidence and next experiment

The complete suite passes **332 tests**, with five optional checks skipped and
one upstream `hppfcl` deprecation warning. The forty focused dataset/model checks
also pass with runtime warnings treated as errors. They cover causal input
whitelisting, grouped splits, training-only scaling, validation-only selection,
nonconverged-candidate exclusion, probability direction, metrics and saved-model
round trips. A test that changes only test inputs and labels leaves fitted
weights, normalization and selected regularization unchanged.

A separate read-only audit matched all 1,040 exported inputs to the original
recordings, reproduced all 3,120 saved probabilities exactly, recomputed every
partition's metrics independently, and confirmed training-only normalization
and validation-only selection. Both selected optimization gradients are below
1e-6. The analysis script also verifies original metadata hashes and reproduces
the frozen report metrics before writing its descriptive breakdown.

This completes the first implementation of Step 6. The useful finding is that
observable closing behavior carries task-outcome information beyond commanded
targets in this particular development split. Four observation families cannot
support broad statistical or transfer claims. Our next experiment should freeze
this protocol before generating fresh observation families, more geometries and
more post-acquisition failures. We can then test generalization, review the task
cutoff on separate development data, and collect real grasp/failure examples
before claiming physical slip warnings or recovery performance.
