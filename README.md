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

## Documentation and results

- [Documentation index](docs/README.md)
- [Research protocol](docs/HUG_MONITOR_RESEARCH_PLAN.md)
- [Completed experiments and results](docs/PAPER_PROGRESS.md)
- [Working paper](docs/paper/working_draft.tex)
- [Partner handoff](docs/PARTNER_HANDOFF.md)
- [Recorded noise demo](docs/demos/grasp-noise/README.md)

Active development uses
[`codex/project-work`](https://github.com/nilakarthikesan/grasp-failure-recovery/tree/codex/project-work).
See [the contributor workflow](docs/WORKING_BRANCH.md) for checkout instructions.

## Attribution and license

Built with [HUG](https://github.com/KevinyWu/hug),
[Dex Retargeting](https://github.com/dexsuite/dex-retargeting) and
[MuJoCo](https://mujoco.org/). Third-party assets retain their own licenses;
HUG checkpoints and licensed MANO assets are supplied separately.

This repository is licensed under [MIT](LICENSE).
