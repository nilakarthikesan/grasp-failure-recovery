# Grasp Failure Prediction

Research tools for predicting grasp outcomes from robot joint measurements and
commands before lifting. [HUG](https://grasping.io/) proposes human-hand grasps,
Dex Retargeting maps them to a Shadow Hand, and MuJoCo executes and records
the attempts.

## Pipeline

1. Capture an RGB-D object observation and generate a HUG grasp proposal.
2. Retarget the proposal and execute a fixed approach, close, lift and hold.
3. Save joint states, commands, applied perturbations and task outcomes.
4. Train and evaluate a separate predictor using the closing history.

The repository includes a grouped simulation dataset, a trained logistic
baseline, fresh-object evaluations and corrected mug diagnostics. The evaluated
target is grasp, lift and hold completion. Results are simulation-only.

## Quick start

Requires Python 3.10 or later. From a repository checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[prediction]'
python -m grasp_failure_prediction.prediction.model \
  research_artifacts/object_scene_pilot_v1 \
  --output runs/baseline_reproduction
```

Use a new output directory. This reproduces the original baseline using the
included derived dataset; HUG inference and rendering are not required.
[The model guide](docs/PREDICTION_BASELINE.md) explains the inputs, splits and
reported metrics.

For simulation setup, HUG asset requirements, collection and replay commands,
see [Running the project](docs/RUNNING_THE_PROJECT.md).

## Grasp proposal sources

HUG remains the reference generator. The shared proposal contract also supports
ContactGen through MANO retargeting and UniDexGrasp through verified Shadow
translation, with isolated inference workers. Those new workers have not yet
run an inference or simulation acceptance check. See
[provider integration and status](docs/CONTACTGEN_AND_UNIDEXGRASP.md).
Trusted saved Shadow poses and pinned DexGraspNet sample replay remain available.
The [unified evaluation command](docs/GRASP_EVAL_CLI.md) validates, runs, resumes
and inspects declared experiments. See [DexGraspNet replay](docs/DEXGRASPNET_REPLAY.md)
for the imported dataset and its execution limits.

The existing predictor uses the original closing-history dataset. Generic
provider runs do not yet record every state/action field required by that model.

## Documentation and results

- [Documentation index](docs/README.md)
- [Research protocol](docs/HUG_MONITOR_RESEARCH_PLAN.md)
- [Completed experiments and results](docs/PAPER_PROGRESS.md)
- [Working paper](docs/paper/working_draft.tex)
- [Partner handoff](docs/PARTNER_HANDOFF.md)
- [Recorded noise demo](docs/demos/grasp-noise/README.md)

Shared development uses
[`project-work`](https://github.com/nilakarthikesan/grasp-failure-recovery/tree/project-work).
The current Phase 1 work uses `phase1-scene-preparation` in an isolated checkout.
See [the contributor workflow](docs/WORKING_BRANCH.md) for checkout instructions.

## Attribution and license

Built with [HUG](https://github.com/KevinyWu/hug),
[Dex Retargeting](https://github.com/dexsuite/dex-retargeting) and
[MuJoCo](https://mujoco.org/). Third-party assets retain their own licenses;
HUG checkpoints and licensed MANO assets are supplied separately.

This repository is licensed under [MIT](LICENSE).
