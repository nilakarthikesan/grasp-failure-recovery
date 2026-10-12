# Grasp failure project handoff

**Current working branch: `codex/project-work`.** All pushed implementation,
demo, model and paper work is consolidated there. The earlier handoff below
records the original baseline; subsequent fresh object tests and mug correction
are documented in [PAPER_PROGRESS.md](PAPER_PROGRESS.md). Use
[WORKING_BRANCH.md](WORKING_BRANCH.md) for the active checkout and branch archive.

**Confirmed on 11 October 2026.** The original six-step plan now has a working
simulation pilot, a portable labeled dataset, and a first trained model.
Building on the existing HUG-to-Shadow execution work, this pilot adds controlled
noise collection, audited outcomes, grouped data splits, model comparisons and
paper documentation. The next experiment is to evaluate the saved model on fresh
observation families before changing it.

## Completed work

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

The completed implementation was tested with **332 tests passing and five
skipped**. The forty focused dataset/model tests also passed with runtime
warnings treated as errors. A fresh read-only artifact check for this handoff
verified all hashes, all 3,120 saved predictions and every reported partition
metric. The [working manuscript](paper/working_draft.tex) compiles and includes
primary-source citations, methods, actual results and limitations.

## Start from the correct branch

Use **`codex/task-outcome-baseline`**, the latest combined execution,
collection, split and model branch. The verified pre-handoff code/results
snapshot is commit `ef0f4d84bac6f6ce13931d911cc27f9c2ad54ed9`.
The latest work remains in open draft PRs and has not merged into `main`.

The dependency order is [execution #12](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/12)
→ [collector #10](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/10)
→ [multi-scene collection #11](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/11)
→ [paper #13](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/13)
→ [splits #14](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/14)
→ [model #15](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/15).
[Teaching website #8](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/8)
and [earlier literature notes #9](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/9)
target `main` separately; their additional files are not in the combined model
branch. Review and merge with those dependencies in mind.

From an existing clean clone, create a separate branch for the next experiment:

```sh
git fetch origin
git switch --no-track -c codex/fresh-scene-evaluation origin/codex/task-outcome-baseline
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

1. **Reproduce the saved baseline.** Confirm 1,040 rows, the 520/260/260
   memberships, and the comparison metrics. Preserve the published dataset,
   coefficients and task rule as version 1.
2. **Declare a fresh evaluation before collecting it.** First vary object
   positions and cameras for cube/apple while retaining the controller,
   label rule, noise schedule and saved models. A suggested first batch is
   eight fresh observation families, four per geometry, with four proposals
   per observation. The existing schedule gives **2,080 planned attempts**:
   32 proposals × (one baseline + 32 trials at each of two noise levels).
   Run a small collection smoke check before scaling. More noisy repeats of
   identical captures do not create new observation families.
3. **Add evaluation for that new batch.** The current dataset exporter binds
   the original four scene seeds, 9100–9103; it is not a general exporter for
   arbitrary new splits. Add a separately versioned fresh-evaluation command
   or manifest that validates labels/provenance and rejects overlap with the
   original observation/proposal groups. Reuse the frozen 2.96-second observable
   prefix and saved normalization/weights. Keep outcomes separate and exclude
   invalid attempts rather than labeling setup errors as failures.
4. **Score the fixed model before changing it.** Report all three comparisons,
   failure ranking, log loss/Brier score, recall, success false alarms, and
   counts per observation/proposal/noise level. Publish the new manifest,
   predictions and metrics. Do not adjust weights or the threshold using the
   new test outcomes. If tuning follows, reserve different development data
   and another untouched evaluation.
5. **Expand the research question after that check.** Add validated new
   geometries and more supported post-acquisition losses. The existing failure
   mix is 590 acquisition failures, two missing-contact failures and only four
   task-scored drops. A future-slip model needs separately versioned temporal
   labels and warning-time evaluation. Audit the friend's real recordings for
   available sensors, actions, synchronization and outcomes before combining
   them with simulation. Recovery and synthetic human video remain later work.

For the first new PR, deliver the fresh-evaluation plan/command, group-overlap
and causal-input checks, a small smoke run, and instructions to scale it.
For the following PR, deliver the completed batch, unchanged-model comparison
and a paper update with all observations and exclusions accounted for.

## How to describe the result

The phrase "retrospective development results from one test observation" means
that the original collection outcomes had already been inspected before the
scene split was declared. The model fitting did exclude its test rows, but
that split is still exploratory and contains only one apple observation family.
The probability scores have not established calibrated real-world failure rates.
The strongest present finding is that measured closing behavior added task-outcome
information beyond commands in this specific simulation split. Fresh evaluation
is the next test of whether that finding holds beyond this pilot.
