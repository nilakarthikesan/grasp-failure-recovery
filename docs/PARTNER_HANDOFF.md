# Grasp failure project handoff

**Current working branch: `codex/project-work`.** All pushed implementation,
demo, model and paper work is consolidated there. [PR 19](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/19)
merged the earlier snapshot into `main`; later diagnostics and paper work
continue on `codex/project-work` and are not automatically merged. Use
[WORKING_BRANCH.md](WORKING_BRANCH.md) for the active checkout and branch archive.

**Updated 12 October 2026.** We have a working simulation and labeling pipeline,
an original trained predictor, a fresh object/scene evaluation, a corrected mug
executor and audits of repeated mug false alarms. We are diagnosing those
alarms before changing the model. We have not demonstrated a training remedy,
physical slip prediction, warning lead time, recovery or real robot transfer.

**Working paper title: Evaluating Simulation Diversity for Grasp Failure Warnings.**

**Accepted question:** For a simulated dexterous hand, does training with more
diverse successful wrist configurations improve prediction of grasp, lift and
hold outcomes on unfamiliar objects and wrist configurations more than
additional finger-command noise, when both approaches use the same amount of
training data, the same success/failure proportions and the same model
architecture?

Our immediate task is smaller: compare safe wrist settings for the same intended
mug grasp at zero noise with the saved model unchanged. The first pair is now
complete: both grasps succeed and both trigger false warnings. Seven remaining
declared pairs and the later training comparison remain pending.

The first static check found that none of the eight observed training wrist
command pairs met the nominal table gate across the eight declared mug proposals
(0 of 64 combinations). The corrected poses still meet that static gate.
Those stored vectors cannot supply a safe familiar setting in these cases.
The static check did not step physics or generate new grasps; the later paired
test explicitly uses two unfamiliar safe settings instead.
See the [static feasibility record](../research_artifacts/wrist_pose_feasibility_v1/README.md)
for every candidate and source binding.
The [first paired diagnostic](../research_artifacts/wrist_pair_diagnostic_v1/README.md)
uses two safe unfamiliar settings selected without new scores or outcomes.
It does not claim a familiar-versus-unfamiliar contrast.

## Original pilot completed on 11 October

1. **Reviewed HUG inputs and benchmark assets.** Selected the existing cube and
   a pinned HUG apple development asset. Documented geometry, camera calibration,
   retargeting and attribution. This is a project simulation experiment rather
   than an official HUG benchmark reproduction.
2. **Implemented and explained bounded noise.** Executed a fixed approach,
   close, lift and hold sequence with one random finger-target bias per episode.
   Requested bounds are 0, 0.15 and 0.6 radians; wrist targets are unchanged.
   The requested bias, clipped commands and actual joint motion are logged
   separately. This currently tests command perturbations, not obstacle,
   sensor, mass or friction variation.
3. **Collected and inspected 1,040 attempts.** Two geometries, four observation
   families and sixteen HUG proposal families produced **444 successes,
   596 failures and zero runtime/validation errors**. Saved trajectories,
   actions, provenance and replay state support inspection. The local results
   viewer includes six rendered replay examples. The interactive teaching
   website and earlier literature notes are separate PRs.
4. **Audited binary task labels.** **1 means success, 0 means completed task
   failure, and errors remain null.** The frozen scorer requires qualifying
   contact, 14 cm peak lift, a two-second hold at least 12 cm above the initial
   height, at most 2 cm peak-to-final loss, and collision/timeout checks.
   Boundary cases and failure-category precedence are documented.
5. **Published causal inputs and grouped splits.** Each row contains twenty
   closing samples through **2.96 seconds**, before lifting starts at 3.00
   seconds: measured joint positions, velocities and actual commands. Outcomes,
   future samples, object truth, contact diagnostics, noise settings and group
   IDs remain outside the model input. The split is **520 training, 260
   validation and 260 test attempts**; observation/proposal families and all
   related noisy attempts stay together.
6. **Trained and evaluated the first model.** Compared joint states plus
   commands, commands alone, and a constant training-prevalence reference.
   Training-only scaling and validation-only selection choose regularization
   0.1. The full model's apple-scene failure-ranking AUC is **0.8523**, versus
   **0.6419** for commands alone and **0.5000** for the constant reference.
   At the fixed 0.5 failure threshold it flags **187 of 202 failures** and
   **22 of 58 successes**, missing 15 failures. The model predicts eventual
   task outcome from closing behavior; slip timing and recovery remain future work.

At the original handoff, the implementation was tested with **332 tests passing and five
skipped**. The forty focused dataset/model tests also passed with runtime
warnings treated as errors. That handoff's read-only artifact check
verified all hashes, all 3,120 saved predictions and every reported partition
metric. These are historical validation results, not a new test run for this
documentation update. The [working manuscript](paper/working_draft.tex) now
contains the new question and diagnostic evidence and compiles successfully in
the desktop editor. Workshop template compliance and author review are pending.

## What happened after the pilot

1. **Tested the frozen model on 2,080 fresh attempts.** New cube/apple scenes
   and milk carton/mug geometries gave 659 successes and 1,421 failures with
   zero runtime errors. All 520 v2 mug attempts hit the table before the model
   cutoff. They could not tell us whether the predictor recognizes successful
   unfamiliar mugs. See [FRESH_OBJECT_EVALUATION.md](FRESH_OBJECT_EVALUATION.md).
2. **Corrected the mug executor separately.** Changed wrist settings and
   compensated carrier placement preserve the intended world palm and finger
   targets while clearing the forearm from the table. A runtime clearance guard
   stops invalid attempts. Of 27 declared checks, 26 completed and one stopped.
   The completed mugs include 20 successes and three failed acquisitions.
   This is an execution correction, not model triggered recovery. See
   [MUG_TABLE_CLEARANCE.md](MUG_TABLE_CLEARANCE.md).
3. **Scored those existing recordings without changing the model.** All 23
   completed mugs triggered a failure alert, including every successful mug.
   The false warnings repeat at all three finger noise bounds, including zero.
   The three actual mug failures were also flagged, so warning on everything
   is the problem. The stopped attempt has no label or prediction. See
   [the scoring evidence](../research_artifacts/corrected_mug_model_diagnostic_v1/README.md).
4. **Compared wrist inputs with the original training data.** Both last wrist
   commands in all 20 successful mugs lie outside the training ranges. Their
   wrist feature contributions push the existing linear scores toward failure.
   This arithmetic is a contributor, not proof that posture alone caused the
   errors or that removing wrist features fixes them. See
   [the wrist audit](../research_artifacts/mug_wrist_input_diagnostic_v1/README.md).

## Start from the correct branch

Use **`codex/project-work`**, the single ongoing branch containing the latest
diagnostic evidence and plan. The old milestone branch and PR names are
historical references, not instructions to create another branch. On Nila's
machine, use `git worktree list` and work in the existing checkout holding this
branch. Preserve dirty files, runtime assets and all experiment directories.

From an existing clean partner clone, fetch and use the existing branch:

```sh
git fetch origin
git switch codex/project-work
```

Optional reproduction of the original fit writes a separate output bundle;
it does not replace the saved model used in the next diagnostic:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[prediction,dev]'
python -m grasp_failure_prediction.prediction.model \
  research_artifacts/object_scene_pilot_v1 \
  --output runs/partner_baseline_reproduction
```

Use a new output directory. The recorded training runtime is Python 3.12.13,
NumPy 2.2.6 and SciPy 1.17.1; cross-runtime numeric results may differ slightly.
Reproducing training needs the published derived data, without HUG inference
or graphics. The [model guide](PREDICTION_BASELINE.md) explains commands and metrics.

## Files available to the partner

- [Paper and six-step evidence notes](PAPER_PROGRESS.md).
- [Collection and simulation instructions](MULTI_SCENE_COLLECTION.md).
- [Noise and event definitions](NOISY_ROLLOUT_COLLECTION.md).
- [Input and split contract](DATA_SPLITS.md).
- [Model explanation and reproduction](PREDICTION_BASELINE.md).
- [Frozen corrected mug predictions](../research_artifacts/corrected_mug_model_diagnostic_v1/README.md).
- [Training wrist range and score attribution audit](../research_artifacts/mug_wrist_input_diagnostic_v1/README.md).
- [Planned wrist pair comparison](../research_artifacts/wrist_comparison_plan_v1/protocol.json).
- [Dated research decisions](RESEARCH_DECISIONS.md).
- [Portable dataset](../research_artifacts/object_scene_pilot_v1/): inputs,
  task labels, provenance and saved memberships, approximately 6.55 MB.
- [Saved model bundle](../research_artifacts/task_outcome_baseline_v1/):
  coefficients, normalization, selection candidates, full report, predictions
  and descriptive breakdowns.

The full original experiment remains on Nila's Mac at
`/Users/nilakarthikesan/grasp-failure-prediction/runs/object_scene_pilot_2026-10-10/`.
Its `collection_1040/` contains the raw rollouts; `inputs_v3/` and
`inputs_v3_extra/` contain captures/proposals. HUG checkpoints, source runtime
and geometry assets occupy other ignored `runs/` directories documented in the
collection guide. These files and replay videos are not included in the clone.
Arrange access or recreate the runtime before running new HUG inference.
The evaluation extras alone do not supply HUG/MANO assets and weights.

After the raw recordings are available locally, the viewer can be launched with:

```sh
python -m grasp_failure_prediction.evaluation.result_viewer \
  --runs-root /path/to/collection_1040 --port 8769
```

`http://127.0.0.1:8769/` refers to the machine running that server; it is not a
public website or a shareable link to Nila's recordings.

## Next work in order

1. **Understand the first completed pair.** Two safe unfamiliar wrist settings
   execute the same intended grasp with zero noise. Both lift and hold the mug;
   both receive a failure warning. Inspect the replay, causal inputs, labels and
   [independent checks](../research_artifacts/wrist_pair_diagnostic_v1/README.md).
   The unchanged model has two false alarms here. No failures occur, so this
   pair cannot estimate failure recall.
2. **Check the remaining seven declared pairs for repeatability.** The
   [frozen pose manifest](../research_artifacts/wrist_pair_selection_v1/pose_manifest.json)
   already selects A/B settings without new predictions or outcomes. The v4
   executor honors those exact poses and rejects substitution. Retain all
   outcomes and guard stops; use the same zero noise, causal cutoff and frozen
   model. Both settings are unfamiliar, so qualify these as sensitivity checks.
3. **Review the full diagnostic before a training change.** Compare both-success
   pairs separately from pairs whose outcomes differ. Measure actual palm/finger
   tracking, logits and wrist/nonwrist terms. The first pair alone cannot
   demonstrate improved prediction, generalization or a training remedy.
4. **Decide on the equal budget training experiment from the diagnosis.**
   Compare the same original base plus equal numbers of finger noise or wrist
   coverage examples, with matched success/failure mix and base object/proposal
   identities. Keep the same model architecture, feature fields and validation
   selection rule. This is planned work, not a demonstrated wrist coverage gain.
   Declare cohort selection before fitting; retain every generated, selected
   and excluded attempt count, since equal training counts need not cost equal
   collection effort.
5. **Measure improvement on one fresh common test set.** Keep entire test
   object/posture/observation/proposal groups outside training and validation,
   and freeze both models before looking at test outcomes. At the same 0.5
   threshold, improvement means fewer false warnings on successes without more
   missed failures. Report counts and group uncertainty; a few failures cannot
   prove broad reliability. Lower scores alone do not meet the criterion.
   Also score the original frozen model as a reference on this same test; its
   smaller training budget does not isolate the augmentation strategy.

Continue on `codex/project-work`; do not create another milestone branch or PR.
Preserve every scientific protocol version and original artifact. A warning
change on two successful paired grasps can diagnose sensitivity, but cannot
establish improved prediction because the model is unchanged.

For the workshop, review the evidence together, verify references and claims,
complete the four page CoRL formatted anonymous paper, and ensure every author
has an active OpenReview profile. The
[Sim2Real2Field call](https://sim2real2field-corl26.github.io/) gives the earlier
deadline as **15 October 2026 at 11:59 PM CDT**. Aim to submit during the Eastern
daytime. The current working draft is not a completed submission or evidence
of acceptance.

## How to describe the result

The phrase "retrospective development results from one test observation" means
that the original collection outcomes had already been inspected before the
scene split was declared. The model fitting did exclude its test rows, but
that split is still exploratory and contains only one apple observation family.
The probability scores have not established calibrated real-world failure rates.
The pilot showed that measured closing behavior added outcome information beyond
commands in that specific split. Later tests show that the fixed threshold does
not distinguish successful corrected mugs from failed mugs. Shifted wrist inputs
are a measured contributor to the scores; their causal role and the proposed
training remedy remain untested. This is an informative negative result, not a
claim that the model already generalizes.

Enoch's request to test new conditions before replacing the model is consistent
with this sequence. His agreement with the narrowed question and schedule is
still unknown; review the question and one proposed comparison with him before
expanding the implementation.
