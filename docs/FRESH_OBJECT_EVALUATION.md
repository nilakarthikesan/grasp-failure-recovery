# Testing the saved model on fresh objects and scenes

The question is: **does the existing model predict the final grasp, lift and
hold outcome on attempts it has not seen before?** We freeze the model before
collecting this evaluation. This experiment does not train a new model.

There are two separate models in the pipeline. The released HUG model proposes
how the hand should grasp the object from the initial image and depth. Our
saved logistic prediction model reads the resulting hand motion and commands,
then predicts whether the attempt will meet the final task rule. Neither set
of weights is trained during this experiment, and the controller does not use
the prediction to change its actions.

## Step 1: freeze what makes the prediction

The original published model bundle, normalization, feature summaries and
failure alert threshold of 0.5 stay unchanged. The model reads twenty closing
samples ending at 2.96 seconds, before lifting begins at 3.00 seconds. Its
inputs are measured hand joint positions, velocities and actual command targets.
Outcomes, object truth, noise settings and future samples remain outside those
inputs. The commands only model and constant training prevalence are unchanged
comparisons.

The full recording is retained for labeling and replay, but only this closing
prefix enters the predictor. Initial images serve HUG; we do not feed a video
or the object outcome into the current prediction model.

Predictions are computed offline from the saved recordings. The time 2.96
seconds describes the latest sample the model is allowed to use, not a live
intervention during collection. This tests what could be predicted from the
available closing history.

The saved plan records hashes of the original data, model files, training
implementation, scorer and execution protocol before new proposals or labeled
trials. The plan is in `setup/frozen_plan.json` under the experiment directory.

## Step 2: change the scene and object, keeping the task fixed

The declared batch has two fresh position and camera scenes for each of four
objects: the familiar cube and apple, plus a new milk carton and mug. Keeping
familiar objects gives a comparison for scene changes. Adding two objects tests
conditions outside the original geometry collection.

The milk carton is from `val/medium_5/milk_box` and the mug from
`val/medium_5/mug` in [HUG / 1M HuGs](https://huggingface.co/datasets/kevinywu/1m-hugs),
revision `56ea3f100b1a4ec38bb35e986c0dd14d38977283`. The source dataset declares
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). We use its MJCF and
referenced meshes, not its grasp annotations or recorded failure examples.
Download URLs, byte hashes and attribution are saved in `assets/download_manifest.json`.

The milk carton retains its native mass of 0.3705 kg and the mug 0.4253 kg.
The cube stays at 0.18 kg and apple at 0.2304 kg. Geometry and native mass both
vary in the new object comparison, so this is not an isolated test of shape.
Friction, controller, embodiment, timing and task scorer retain their original
settings. Each object must pass the existing stationary capture and calibration
checks before HUG inference. A setup rejection is recorded as an error, not a
failed grasp.

The first declared alternative was a salt shaker from `val/small_4/salt_shaker`.
It failed the unchanged stationary capture criterion in all six tested resting
orientations. We preserved the rejection and pose audit, then recorded a plan
amendment replacing it with the milk carton before generating any labeled grasp
attempts. No grasp outcome or model probability selected the replacement.

## Step 3: generate and record new attempts

Each of eight scenes supplies a fresh RGB and depth capture. HUG generates four
proposals with seeds 0 to 3 from each capture. The fixed Shadow Hand controller
executes every proposal with one baseline without noise and 32 repetitions at
each of the original bounds, 0.15 and 0.6 radians. The declared total is
**8 scenes × 4 proposals × 65 attempts = 2,080 attempts**, using noise master
seed 10400. These are eight observation families, not 2,080 independent scenes.

Noise is one random bias on each of 22 finger targets, held constant within an
attempt. The two wrist targets receive no bias. Joint limits can clip requested
targets, so requested offsets, actual commands and measured movement are saved
separately. Each attempt saves its trajectory, actions, noise, provenance,
scorer result and replay state.

One proposal per object is first executed at the three noise levels as a
separate pipeline check. Its outcomes do not select the objects, proposals,
noise or model. These smoke attempts are excluded from the final evaluation.

## Step 4: compare predictions with outcomes

The original scorer assigns 1 for success and 0 for completed failure. Success
requires qualifying contact, at least 14 cm peak lift, a two second hold at
least 12 cm above the initial height, at most 2 cm height loss from peak to final,
and the collision and timeout checks. Invalid attempts keep a null label.

The fresh evaluation command validates source contracts and regenerates labels,
rejects overlap with the original observation or proposal sources, extracts the
same causal prefix and uses the saved normalization and coefficients. It does
not fit weights or select a threshold. We report failure ranking, probability
errors and the fixed threshold confusion counts, including missed failures and
false alarms among successful attempts. Results are broken down by object,
observation and noise condition.

After seeing these results, any model improvement uses separate development
data. A later test must then use another untouched collection. This batch remains
an evaluation of eventual task outcome in simulation, not physical slip onset,
warning lead time or recovery.

The scorer receives an execution reference copied from an original recorded
attempt. It verifies the environment, controller parameters, physics settings
and registry hashes against that reference. It also verifies exact agreement
between the plan, complete index, episode metadata and final summary, so a
partial collection cannot be presented as the declared evaluation.

The frozen scoring command is:

```sh
python scripts/evaluate_fresh_collection.py \
  --runs-root /path/to/fresh_object_scene_eval_2026-10-11/collection_2080 \
  --project-root /path/to/the/local/asset/root \
  --original-dataset research_artifacts/object_scene_pilot_v1 \
  --models research_artifacts/task_outcome_baseline_v1 \
  --execution-reference /path/to/fresh_object_scene_eval_2026-10-11/setup/frozen_execution_reference.json \
  --output /path/to/a/new/evaluation
```

Run this using the existing combined HUG and simulation dependency environment
documented in the collection guide. The asset root resolves the original
project relative capture, proposal and mesh references. The output must be a
new directory, separate from the raw collection.

## Completed result

The declared batch completed with **2,080 attempts: 659 successes, 1,421 failures
and zero collection errors**. No attempt was excluded from evaluation. Each
object has 520 attempts; eight observation families supply 32 proposals.
All original frozen data, model and scoring files remain unchanged.

The result supports Enoch's concern about unfamiliar objects:

- **Familiar cube and apple in fresh scenes:** failure ranking AUROC 0.9173,
  553 of 626 failures flagged and 88 false alarms among 414 successes.
- **New milk carton and mug:** pooled AUROC 0.5588, 582 of 795 failures flagged
  and 185 false alarms among 245 successes.
- **Milk carton:** 245 successes and 275 failures, AUROC 0.8121; the unchanged
  threshold falsely flags 185 successful attempts and misses eight failures.
- **Mug:** all 520 attempts fail through hand collisions with the table during
  approach; the model flags 315 and misses 205. With no
  successful mugs, mug ranking AUROC and success false alarm rate are undefined.

The full model's overall AUROC is 0.7737, compared with 0.5524 for commands
alone and 0.5 for the unchanged prevalence baseline. Overall log loss is
0.6645 for the full model and 0.6584 for prevalence; improved ranking does not
mean every probability metric improves. The pooled new object score also mixes
object conditions and is not a population or geometry only transfer guarantee.

Mug approach collisions happen before the 2.96 second closing cutoff. They are
valid final task failures, but predicting those labels is not a warning before
a future failure. The original training collection had no approach collision
failures. This new batch exposes a new failure category as well as new shapes
and masses. Do not describe the result as slip generalization.

The portable [evaluation bundle](../research_artifacts/fresh_object_scene_eval_v1/)
contains all inputs, 6,240 saved predictions from the three frozen comparisons,
provenance and the [complete report](../research_artifacts/fresh_object_scene_eval_v1/report.json).
Its protocol directory preserves the declared plan, amendment and source assets.
Eight small [replay videos](../research_artifacts/fresh_object_scene_eval_v1/replays/)
are also included so the partner can watch examples without the full raw run.
There is no model retraining in this bundle.

All **77 focused model, dataset, fresh evaluator and viewer tests passed** with
runtime warnings treated as errors. The broader suite passed 338 tests with 34
checks skipped and one upstream deprecation warning. The actual
evaluation also revalidated all source bindings and independently regenerated
every task label before predicting.

An additional independent audit reconstructed all raw inputs and labels,
reproduced all 6,240 probabilities exactly from saved coefficients and
normalization without using prediction helpers, and recomputed all 87 grouped
metric records. It verified every model, controller and source binding hash.

## See one attempt

The [local fresh collection viewer](http://127.0.0.1:8770/) shows all attempts
and eight rendered examples first. Choose **Videos only**, then select an
attempt. Its detail view shows the actual outcome, model failure probability
using data through 2.96 seconds, the fixed alert and whether it was a missed
failure, false alarm or correct decision. Each detail includes five plain steps.

For example, `episode_001040` is the first successful milk carton attempt;
`episode_001042` is its first recorded failure. The mug examples are its first
baseline, `episode_001560`, and first 0.6 rad attempt, `episode_001562`, because
the object has no successful trials. Examples follow the saved selection rule,
not model confidence.

In `episode_001040`, the model assigns failure probability 0.6085 from the
closing history, so its unchanged 0.5 threshold raises a false alarm even though
the attempt succeeds. The video and outcome therefore demonstrate a model
error, not a failed attempt. For the first milk carton failure,
`episode_001042`, failure probability is 0.9923 and the alert is correct.

Predictions are computed offline from the recorded closing prefixes. Videos
replay saved poses at 25 frames per second and are not extra trials. Rendering
and video registration precede evaluation, so the report binds the final result
files. The viewer runs locally and is not a public URL.

```sh
python -m grasp_failure_prediction.evaluation.result_viewer \
  --runs-root /path/to/fresh_object_scene_eval_2026-10-11/collection_2080 \
  --evaluation-root /path/to/fresh_object_scene_eval_2026-10-11/evaluation \
  --port 8770 --no-open
```

## Where the data lives

The local experiment directory is
`/Users/nilakarthikesan/grasp-failure-prediction/runs/fresh_object_scene_eval_2026-10-11/`.
It contains `setup/`, `assets/`, `inputs/`, the separate smoke collection, the
final collection and the frozen evaluation output. Full rollouts, HUG weights
and mesh assets remain local. The final raw collection is `collection_2080/`
and its frozen evaluation is `evaluation/`. The raw collection occupies about
849 MiB after sharing byte identical immutable simulator model files. The
portable derived dataset and predictions are tracked in
`research_artifacts/fresh_object_scene_eval_v1/`; the NPZ input payload is
9,983,549 bytes.

## Enoch's questions and how this experiment answers them

**Where did the first evaluation data come from?** We rendered our own MuJoCo
observations using the existing cube and a HUG apple asset. HUG supplied fresh
grasp proposals. The fixed controller, finger target perturbations and task
scorer generated and labeled 1,040 simulated attempts. The original
[collection guide](MULTI_SCENE_COLLECTION.md) records the exact sources and method.

**Could the current model already be good enough?** The original result was
promising, but represented one apple test observation and flagged 22 of 58
successful attempts. This new collection tests the unchanged model beyond those
observations. Usefulness depends on both missed failures and false alarms, not
just one ranking score.

**Can improving the model replace the other proposed work?** Model improvements
can wait until this test. Predicting final task outcome does not by itself
establish slip timing or recovery. Those claims need appropriate event labels
and their own evaluations if we pursue them later.
