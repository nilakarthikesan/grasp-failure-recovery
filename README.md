# Grasp Failure Prediction and Recovery

MuJoCo research infrastructure for executing dexterous grasps and studying
whether object loss can be predicted early enough for a useful recovery action.

The primary path uses [HUG](https://github.com/KevinyWu/hug) human-hand grasp
predictions, [Dex Retargeting](https://github.com/dexsuite/dex-retargeting), and
a Shadow Hand model. The project separates grasp execution, future-failure
prediction, and recovery evaluation so each stage can be assessed independently.

## Current status

This is an active research prototype. The repository includes:

- HUG prediction loading and validation, coordinate-frame adapters, and
  RGB-D observation checks.
- Dex Retargeting integration with explicit MuJoCo joint-name mapping and
  numerical pose validation.
- A one-case HUG evaluation command with versioned environment and execution
  protocol registries, configuration hashes, trajectory artifacts, and outcome
  scoring.
- Separate experimental CPU inference, alignment, and actuated Shadow Hand
  diagnostic scripts. The [simulation pilot](docs/HUG_SIMULATION_PILOT.md)
  records results from one cube scene and their limitations.
- Supporting robosuite/Panda demonstration collection, synchronized logging,
  HDF5-to-LeRobot export, ACT training, and MuJoCo state snapshots.

The registered `fixed_grasp_lift_v1` case runner prescribes wrist and hand motion
kinematically. It places the palm horizontally above the object using protocol
parameters, rather than executing the source HUG wrist pose unchanged. Its grip
command is normalized position, not calibrated force. The experimental actuated
runner described in the pilot is a separate execution path.

The one-case interface currently supports `object01`, `nominal_v1` friction,
and `nominal_lift_v1` motion. Batch holdouts, future-failure labeling, trained
failure predictors, intervention branching, and learned recovery remain work
in progress. Neither the registered case runner nor the one-cube pilot
establishes general grasp reliability or physical-robot performance.

The supporting Panda pipeline also lacks its `part1.evaluate` module. Its
combined runner saves a trained checkpoint and skips evaluation; the declared
`eval-act` entry point is unavailable. This limitation is separate from the
implemented HUG case runner.

## Research plan

### Part I: Execute HUG-conditioned grasps

Generate multiple MANO hand poses from RGB-D observations, retarget them to
Shadow Hand joint targets, and execute a fixed grasp/lift protocol. Freeze the
observation, retargeting, controller, and outcome definitions before expanding
to batch evaluation.

The planned initial batch fixes the hand embodiment and nominal friction, then
varies grasp samples, objects, mass, initial pose, and motion. The current
one-case interface supports a narrower set of assets and profiles. Transport
and placement are extensions beyond the present fixed lift-and-hold protocol.

### Part II: Predict future object loss

Freeze the execution system and train a temporal model conditioned on
observations and actions. Labels use object loss within a specified future
horizon. Split complete episodes, objects, and physical conditions before
creating temporal windows.

Planned metrics include failure recall at a fixed false-alarm budget, warning
lead time, probability calibration, and performance under held-out conditions.
Simulator mass, friction, and contact truth can support analysis and oracle
baselines; they must not be confused with realistic tactile sensing.

### Part III: Compare recovery actions

Restore matched states and compare bounded grip adjustments, motion changes,
return to a support surface, and continued execution under specified response
delay. Evaluate task completion, object-preserving aborts, and harmful
interventions separately before training a recovery selector.

The research design is in [System design](docs/SYSTEM_DESIGN.md). The hand
embodiment and external-method scope are in
[Research foundations](docs/RESEARCH_FOUNDATIONS.md).

## Installation

Requires Python 3.10 or later. Initialize the pinned `dex-urdf` submodule and
install the evaluation extra:

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
```

HUG inference requires additional external code, model weights, and licensed
MANO assets. It is not installed by the evaluation extra. See
[HUG simulation pilot](docs/HUG_SIMULATION_PILOT.md) for that separate runtime.

For the supporting Panda training stack, the macOS ARM64 constraints record
the project's local dependency versions:

```bash
pip install -e ".[dev,robot]" -c constraints/part1-macos-arm64.txt
pip install "lerobot[dataset]" -c constraints/part1-macos-arm64.txt
```

On macOS, use `MUJOCO_GL=cgl` for offscreen rendering. The Panda
`validate-stack` command checks dependencies, a short simulator rollout, and
the LeRobot/ACT data path.

## Workflows

### Inspect an environment or HUG prediction

```bash
inspect-adroit
inspect-hug-prediction /path/to/grasp_pred/example.pkl
view-retargeted-hand /path/to/grasp_pred/example.pkl
```

The Adroit inspector resets one episode without stepping it and reports the
observation/action layout, timing, object physics, and simulator contacts.
Adroit's default observations do not include tactile or contact measurements.

For visual inspection of a retargeted hand on macOS:

```bash
MUJOCO_GL=glfw mjpython -m grasp_failure_prediction.evaluation.view_pose \
  /path/to/grasp_pred/example.pkl --viewer
```

Pickle files can execute arbitrary code during loading. Use predictions you
generated or obtained from a trusted source.

### Run one HUG evaluation case

Place a trusted prediction at the repository-relative path referenced by the
case. Use a new or empty output directory:

```bash
run-hug-eval-case \
  eval_cases/hug_case_001/case.yaml \
  --output runs/hug_case_001
```

For live playback on macOS:

```bash
MUJOCO_GL=glfw mjpython -m grasp_failure_prediction.evaluation.case_runner \
  eval_cases/hug_case_001/case.yaml \
  --output runs/hug_case_001_viewer \
  --viewer
```

The command resolves the environment and protocol registries, validates the
case and supported assets, and checks supplied expected configuration hashes.
It writes `resolved_case.json`, `trajectory.npz`, `final_state.npz`, and
`result.json`. The final state contains joint positions and velocities; it
does not constitute a complete controller/environment replay snapshot.

The `acquired_object` flag means hand-object contact during manipulation.
Use lift and hold measurements to assess task success. Coarse task-failure
labels must be validated before they supply future-drop training targets.

The executable input schema and provenance contract are documented in
[Evaluation manifest](docs/EVALUATION_MANIFEST.md).

### Collect Panda demonstrations and train ACT

These commands exercise the supporting parallel-jaw pipeline:

```bash
MUJOCO_GL=cgl validate-stack
MUJOCO_GL=cgl python scripts/inspect_container_task.py
MUJOCO_GL=cgl collect-demos runs/demos.hdf5 --split train
MUJOCO_GL=cgl export-demos runs/demos.hdf5 runs/lerobot
MUJOCO_GL=cgl train-act runs/lerobot runs/policy --smoke
```

The combined smoke runner collects, exports, and trains, then skips the missing
Panda evaluation stage:

```bash
MUJOCO_GL=cgl python scripts/run_part1_pipeline.py --workdir runs/part1_smoke
```

A small smoke-trained model checks the data path; it does not demonstrate a
competent manipulation policy. ACT is not used by the initial HUG execution
protocol.

To view the scripted Panda on macOS after installing the robot extra:

```bash
MUJOCO_GL=glfw mjpython -u scripts/watch_container_task.py
```

### Run tests

```bash
pytest
```

Tests for unavailable optional integrations may skip. Review the skip summary
when assessing what was exercised.

## Repository layout

```text
src/grasp_failure_prediction/
  environments/             # Adroit inspection
  integrations/             # HUG output, frames, RNG, and RGB-D validation
  evaluation/               # Retargeting, registries, case runner, and scoring
    configs/                # Versioned environment and protocol definitions
  part1/                    # Supporting Panda simulation and ACT training
eval_cases/                 # Example case inputs; prediction assets are external
scripts/                    # Inspection and separate simulation diagnostics
tests/                      # Unit tests and optional simulation regressions
constraints/                # Local simulation/training dependency versions
docs/                       # Design, contracts, and diagnostic records
```

## Documentation

- [System design](docs/SYSTEM_DESIGN.md): research questions and staged metrics.
- [Research foundations](docs/RESEARCH_FOUNDATIONS.md): HUG, Dex Retargeting,
  tactile adaptation, and embodiment choices.
- [Evaluation manifest](docs/EVALUATION_MANIFEST.md): case schema and provenance.
- [Hardware audit](docs/HARDWARE_AUDIT.md): requirements for physical validation.
- [HUG observation preflight](docs/HUG_OBSERVATION_PREFLIGHT.md): RGB-D checks.
- [HUG simulation pilot](docs/HUG_SIMULATION_PILOT.md): integration commands,
  dependencies, and limits.
- [Alignment investigation](docs/HUG_ALIGNMENT_DEBUG.md): coordinate frames,
  contact diagnostics, controller ablations, and historical results.

## Attribution and license

This project uses the released output format of
[Human Universal Grasping (HUG)](https://github.com/KevinyWu/hug), distributed
under the MIT License, and integrates
[Dex Retargeting](https://github.com/dexsuite/dex-retargeting). HUG code, model
weights, and MANO assets are not redistributed here. The pinned `dex-urdf`
submodule and separately obtained models, data, and assets retain their
respective licenses.

[See to Touch](https://see-to-touch.github.io/) informs the later tactile
adaptation direction; its code and data are not redistributed here.

Project code is available under the [MIT License](LICENSE).
