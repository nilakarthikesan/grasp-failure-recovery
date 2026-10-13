# Partner handoff

## Start here

Use [the completed experiment summary](PAPER_PROGRESS.md),
[the research protocol](HUG_MONITOR_RESEARCH_PLAN.md) and
[the working paper](paper/working_draft.tex). They separate measured results
from comparisons that remain unexecuted in the saved evidence.

Active development is on `codex/project-work`. PR 19 merged an earlier snapshot
into `main`; later diagnostics and documentation continue on the working branch.

From an existing clean clone:

```bash
git fetch origin
git switch codex/project-work
```

On Nila's machine, use `git worktree list` to locate the existing checkout
holding that branch. See [WORKING_BRANCH.md](WORKING_BRANCH.md) for local assets.

## Completed work

- Fixed HUG-to-Shadow simulation with force-limited execution, recorded
  trajectories, declared finger perturbations and task labels.
- A 1,040-attempt cube/apple pilot, observation-grouped splits and a saved
  pre-lift logistic outcome predictor.
- A 2,080-attempt frozen evaluation on fresh scenes and objects.
- A mug table-clearance correction, 26 completed controller checks and one
  separately recorded stop.
- Frozen scoring of those recordings: all twenty successful mugs and all
  three failed mugs trigger warnings.
- Wrist-input attribution, static feasibility checks and eight zero-noise
  paired comparisons. All thirteen paired successes trigger false warnings.

The evidence summary links each result to its versioned artifacts.
The current model has ranking information in the original development split,
but the mug checks expose an unreliable warning threshold under changed
conditions. Wrist contributions alone do not establish the cause or remedy.

## Reproduce and inspect

The included derived dataset and trained coefficients are sufficient to
reproduce the original baseline without HUG inference or graphics:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[prediction,dev]'
python -m grasp_failure_prediction.prediction.model \
  research_artifacts/object_scene_pilot_v1 \
  --output runs/partner_baseline_reproduction
```

Use a new output directory. The recorded training runtime is Python 3.12.13,
NumPy 2.2.6 and SciPy 1.17.1; cross-runtime arithmetic may differ slightly.
See [the model guide](PREDICTION_BASELINE.md).

Raw pilot recordings remain on Nila's Mac at
`/Users/nilakarthikesan/grasp-failure-prediction/runs/object_scene_pilot_2026-10-10/`.
A clone does not include all raw replays, HUG checkpoints or licensed MANO assets.
See [the collection guide](MULTI_SCENE_COLLECTION.md) for provenance and paths,
and [the running guide](RUNNING_THE_PROJECT.md) for setup and replay commands.

## Recorded next work

The latest stored protocol calls for a development check of calibration and
joint representations, followed by an equal-count comparison of broader HUG
situation coverage against noisy repetitions of fewer grasps.

Before fitting, declare feature formulas, data memberships, sampling quotas
and validation rules. Reserve one common fresh evaluation set and preserve
whole observation/proposal families. Evaluate false warnings and missed
failures together. The existing diagnostics remain development evidence.
Details and claim boundaries are in [the protocol](HUG_MONITOR_RESEARCH_PLAN.md).

Paper preparation tasks are listed in
[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).
The archive records earlier exploratory directions and is not the active
implementation plan.
