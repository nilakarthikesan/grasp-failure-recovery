# Learning, Anticipating, and Recovering from Grasp Failures

**A three-part robot-learning research program under hidden physical
variation**

This project studies one connected question:

> How can a robot learn to grasp an object, recognize when its current behavior
> is likely to fail, and choose an effective recovery while recovery is still
> possible?

The work progresses from **learning actions**, to **predicting outcomes**, to
**choosing interventions**. These are separately trained and evaluated learning
problems that share the same robot, task, objects, logging infrastructure, and
controlled physical variations.

This is one research program, not a commitment to produce three papers. The
initial goal is one coherent, reproducible result. A component should become a
separate paper only if it develops an independent question, contribution, and
body of evidence.

---

## Research program

### Part I — Learn the nominal manipulation skill

Train a policy to grasp one rigid object from a table, lift it, transport it a
short distance, and place it in a target region. The first policy will learn
high-level action decisions while existing kinematics, planning, and low-level
control execute them.

The selected Part I platform is a simulated Panda arm with a two-finger
gripper in **robosuite/MuJoCo**, trained by imitation learning with ACT as the
first reproducible baseline (via LeRobot). ManiSkill is the documented backup.
Reinforcement learning is a later comparison, not a prerequisite.

**Deliverable:** a trained grasp-and-transport policy, a reproducible training
pipeline, held-out closed-loop evaluations, and saved successful and failed
executions.

The detailed Part I contract is in
[`docs/PART_I_TRAINING_SPEC.md`](docs/PART_I_TRAINING_SPEC.md).

### Part II — Anticipate future failure

Freeze the Part I policy and execute it across controlled physical conditions.
Train a separate temporal model to estimate:

> Given the observations and actions so far, how likely is the object to be
> lost within a future horizon if the robot continues its current behavior?

The predictor is conditioned on behavior because the same grasp may survive a
slow movement and fail under a more aggressive motion. Data is split by complete
episodes, objects, and physical conditions—not neighboring frames.

**Deliverable:** a calibrated future-failure predictor with measured warning
lead time and evaluation under held-out mass, friction, and mass–friction
combinations.

### Part III — Estimate recoverability and intervene

At selected simulator states, compare multiple possible interventions from
matched starting conditions. Initially, the actions are fixed, bounded recovery
behaviors such as:

- adjust grip within hardware limits;
- slow or modify the transport motion;
- return the object to a support surface and regrasp; or
- continue the nominal policy.

Train an intervention-outcome model to estimate which available behavior is
most likely to recover successfully after realistic response delay. A learned
recovery controller comes only after this comparison is understood.

**Deliverable:** an evaluated recovery selector/controller, measurements of
when recovery remains possible, and explicit separation between task
completion, object-preserving aborts, and harmful interventions.

---

## End-to-end architecture

```
demonstrations
    │
    ▼
Part I: train grasp-and-transport policy
    │
    ├── successful and failed executions under controlled physics
    ▼
Part II: train future-failure predictor
    │
    ├── warning states + matched simulator intervention trials
    ▼
Part III: train intervention-outcome model and recovery selector
    │
    ▼
compare base, reactive, anticipatory, and recovery-aware systems
```

The components are trained and frozen in stages so improvements remain
interpretable. Joint training may be studied later.

## Controlled physical variation

The core evaluation varies object **mass**, surface **friction**, grasp
placement, and transport motion. Training and testing distinguish:

- unseen mass with familiar friction;
- unseen friction with familiar mass;
- unseen mass–friction combinations;
- extrapolation beyond the training range; and
- eventually, unseen objects under unseen physical conditions.

Hidden simulator parameters generate controlled trials and evaluation groups,
but they are not exposed to a sensor-based predictor.

## Current repository status

Part I (the nominal grasp-and-transport skill) is implemented and validated in
simulation; Parts II and III remain to be built:

- **Part I pipeline (implemented):** a weighted-container pick/lift/transport/
  place task on robosuite/Panda, with canonical synchronized logging, exact
  MuJoCo state snapshots, a scripted privileged demonstrator, LeRobot export,
  ACT training, and held-out closed-loop evaluation. See
  [`docs/PART_I_TRAINING_SPEC.md`](docs/PART_I_TRAINING_SPEC.md) and the
  `src/grasp_failure_prediction/part1/` package. All results are simulation-only
  until reproduced on a physical robot (see
  [`docs/HARDWARE_AUDIT.md`](docs/HARDWARE_AUDIT.md)).
- `AdroitHandRelocate-v1` inspection exposes observations, actions, timing,
  object mass/friction, and MuJoCo contacts.
- A validated adapter reads released
  [HUG](https://github.com/KevinyWu/hug) grasp predictions without depending on
  HUG's CUDA runtime.
- Future-failure labels (Part II), intervention branching, and recovery learning
  (Part III) remain to be implemented. The Part I logs already contain the
  histories, physics metadata, and restorable snapshots those parts need.

The default Adroit observation contains no tactile or contact measurements.
MuJoCo does maintain contact information internally, so a first simulation
study can surface contact-derived features through a wrapper. Those features
must not be described as realistic tactile sensing.

Adroit is useful for physics and contact inspection, but its dexterous
relocation task is not the Part I learning sandbox. Part I uses a simulated
Panda arm with a two-finger gripper in **robosuite/MuJoCo**, adapted from the
`PickPlace` task. MuJoCo is chosen over PhysX-based stacks because it uniquely
documents a complete integration state that restores to identical forward
dynamics, which the Part III recovery-branching study requires. ManiSkill is
the documented backup. The final simulated embodiment should still be chosen
with the available physical robot in mind.

The physical robot is not required for Part I. Three hardware discovery tracks
run in parallel while the simulation policy is built:

- **Track A — simulation now:** build and train the robosuite/Panda policy
  immediately; this is the critical path.
- **Track B — friend-built robot:** evaluate a written specification (DOF,
  payload, repeatability, control/telemetry rates, gripper force interface,
  URDF/simulation model, API, emergency stop, BOM, and publishability) before
  authorizing any build.
- **Track C — lab or funded hardware:** investigate lab access requirements and
  funding-dependent options (for example an xArm6 or FR3-class arm). No purchase
  or custom build is authorized yet.

At a single hardware gate the project selects one target embodiment. Large-scale
scientific data collection waits until then; a portable Cartesian end-effector
plus gripper interface keeps the policy transferable across these options.

## Data sources

- **Project demonstrations:** the primary source for training Part I. Every
  episode must contain synchronized observations, robot state, actions, and
  outcomes.
- **Policy executions:** successful and unsuccessful closed-loop runs from Part
  I provide Part II's future-outcome data.
- **Matched intervention trials:** simulator state branching produces the
  counterfactual comparisons needed by Part III.
- **A collaborator's VR-collected dataset:** usable for the central experiments
  only after auditing provenance, embodiment, synchronization, sensors, actions,
  outcomes, licensing, and physical-property labels. Data without failures or
  measured physical conditions may still support policy or representation
  pretraining.
- **HUG:** an optional later source of diverse static grasp hypotheses and
  benchmark objects, not a dependency for the first learned policy.

## Evaluation

The four primary system comparisons are:

1. Base policy alone.
2. Reactive detector plus a fixed recovery controller.
3. Future-failure predictor plus the same fixed controller.
4. Predictor plus learned intervention selection or recovery.

Part I reports closed-loop task success and failure modes. Part II reports
failure recall at a fixed false-alarm budget, warning lead time, probability
calibration, and held-out-condition performance. Part III reports recovery
success, task completion, safe aborts, harmful interventions, and sensitivity
to response delay.

When an intervention prevents a drop, the warning is not automatically a false
positive. Predictor evaluation therefore needs a non-intervention comparison
from simulator branching or closely matched physical trials.

---

## Installation

Requires Python ≥ 3.10. The lightweight Adroit/HUG baseline needs only the
default dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The Part I robot-training stack (robosuite, robomimic, LeRobot, ACT, PyTorch) is
an optional extra. Install it against the validated version set:

```bash
pip install -e ".[dev,robot]" -c constraints/part1-macos-arm64.txt
pip install "lerobot[dataset]" -c constraints/part1-macos-arm64.txt
```

On macOS, offscreen rendering uses CGL, so prefix simulation commands with
`MUJOCO_GL=cgl`. Validate the stack with `MUJOCO_GL=cgl validate-stack`.

---

## Workflows

### 1. Inspect the Adroit environment (runs on Mac)

Resets one episode, never calls `env.step(...)`, and prints the observation
layout, action/actuator layout, timing (0.01 s control step), object mass and
friction, and the MuJoCo contacts present at reset.

```bash
python scripts/inspect_adroit.py
# or, after installation:
inspect-adroit
```

### 2. Inspect a HUG prediction (optional, no HUG runtime needed)

Reads a `grasp_pred/*.pkl` you already generated with HUG's released inference
code and prints its usable grasp fields (MANO pose, wrist transform, translation,
landmarks, mesh vertices, camera metadata).

```bash
python scripts/inspect_hug_prediction.py /path/to/grasp_pred/example.pkl
# or, after installation:
inspect-hug-prediction /path/to/grasp_pred/example.pkl
```

> **Security note:** `*.pkl` files can execute arbitrary code when unpickled.
> Only run this on prediction files you generated or trust. The test suite uses
> synthetic in-memory dictionaries only — never opaque downloaded pickles.

### 3. Part I: grasp-and-transport pipeline (robosuite, simulation-only)

Requires the `robot` extra. Validate the stack, inspect the task, or run the
whole collect → export → train → evaluate loop:

```bash
MUJOCO_GL=cgl validate-stack                       # versions + live sim + LeRobot round-trip
MUJOCO_GL=cgl python scripts/inspect_container_task.py   # one demo, phases, HDF5, exact snapshot restore
MUJOCO_GL=cgl python scripts/run_part1_pipeline.py --workdir runs/part1_smoke  # end-to-end smoke
```

Individual stages are also exposed as entry points:

```bash
MUJOCO_GL=cgl collect-demos runs/demos.hdf5 --split train
MUJOCO_GL=cgl export-demos runs/demos.hdf5 runs/lerobot
MUJOCO_GL=cgl train-act runs/lerobot runs/policy          # add --smoke for a fast check
MUJOCO_GL=cgl eval-act runs/policy                        # held-out closed-loop metrics
```

The scripted demonstrator reliably solves the task; a short smoke-trained ACT
policy will not, by design. Reaching a competent policy needs the full-scale run
(more demonstrations, a larger model, and many optimizer steps). All Part I
numbers are simulation-only.

### 4. Run the tests

```bash
pytest
```

---

## Implementation roadmap

1. **Research and data contract:** freeze the task, embodiment, observation and
   action interfaces, failure taxonomy, logging schema, physics ranges, and
   held-out splits.
2. **Part I baseline:** validate the environment, collect demonstrations, train
   ACT, and evaluate closed-loop grasp-and-transport behavior.
3. **Part II benchmark:** freeze the policy, collect balanced future-failure
   windows under controlled physics, and train calibrated temporal baselines.
4. **Part III branching:** restore matched simulator states, test bounded
   interventions and delays, then train the intervention-outcome model.
5. **Physical validation:** reproduce the important findings using the sensing
   and actuation available on the selected robot.
6. **Optional extensions:** richer tactile sensing, HUG-generated grasps,
   learned recovery control, reinforcement learning, and joint training.

---

## Repository layout

```
src/grasp_failure_prediction/
  environments/adroit.py      # Adroit/MuJoCo inspection (runnable baseline)
  integrations/hug.py         # HUG prediction-pickle adapter (optional path)
  part1/                      # Part I grasp-and-transport pipeline (robot extra)
    config.py                 #   frozen task/physics/split contract
    environment.py            #   weighted-container task (robosuite/Panda wrapper)
    snapshot.py               #   exact MuJoCo state capture/restore (Part III primitive)
    record.py                 #   canonical episode schema + RoboMimic-style HDF5
    scripted.py               #   privileged scripted demonstrator (teacher)
    collect.py                #   demonstration collection
    lerobot_export.py         #   HDF5 -> LeRobot dataset for ACT
    train_act.py              #   ACT training loop
    evaluate.py               #   held-out closed-loop evaluation
    stack.py                  #   training-stack validator
scripts/
  inspect_adroit.py           # entry point for the Adroit inspection
  inspect_hug_prediction.py   # entry point for the HUG-output inspection
  validate_stack.py           # validate the Part I training stack
  inspect_container_task.py   # inspect the weighted-container task end to end
  run_part1_pipeline.py       # collect -> export -> train -> evaluate
constraints/
  part1-macos-arm64.txt       # validated Part I stack versions (macOS arm64)
tests/
  test_hug_adapter.py         # synthetic-dictionary tests for the HUG adapter
  test_part1_config.py        # observation contract, splits, phases
  test_part1_record.py        # episode/HDF5/snapshot round-trips, leak detection
  test_part1_pipeline.py      # export/eval pure-logic tests
```

---

The evolving end-to-end specification is in
[`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md).

## Attribution

This project builds on ideas and the released output format of **HUG (Human
Universal Grasping)**:

- Repository: <https://github.com/KevinyWu/hug>
- HUG is distributed under the MIT License.

HUG's authors retain all rights to their work; this repository only consumes
HUG's released prediction format and does not redistribute HUG code, weights, or
the MANO assets it depends on.

## License

MIT — see [`LICENSE`](LICENSE).
