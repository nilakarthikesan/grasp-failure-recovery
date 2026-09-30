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

### Part I — Execute HUG grasps with a dexterous hand

Given an RGB-D observation and a selected object, use HUG to generate multiple
MANO human-hand grasp samples. Retarget each sample to an articulated dexterous
hand with [Dex Retargeting](https://github.com/dexsuite/dex-retargeting),
execute the grasp in MuJoCo, then lift and transport the object under controlled
conditions.

The initial executable embodiment is the Adroit/Shadow Hand in MuJoCo. The
repository's Panda parallel-jaw pipeline remains useful generic manipulation
infrastructure, but it cannot execute a MANO dexterous grasp and is not the
primary experimental embodiment. A Panda or another arm may later carry a
dexterous hand.

**Deliverable:** a reproducible RGB-D → HUG → Dex Retargeting → Shadow Hand →
grasp/lift/transport pipeline with saved successful and failed executions.

The corrected HUG-centered scope is in
[`docs/RESEARCH_FOUNDATIONS.md`](docs/RESEARCH_FOUNDATIONS.md); the executable
Part I contract will be frozen after the retargeting prototype is validated.

### Part II — Anticipate future failure

Freeze the HUG-conditioned grasp execution system and run it across controlled
grasp, object, and motion conditions. Train a separate temporal model to estimate:

> Given the observations and actions so far, how likely is the object to be
> lost within a future horizon if the robot continues its current behavior?

The predictor is conditioned on behavior because the same grasp may survive a
slow movement and fail under a more aggressive motion. Data is split by complete
episodes, objects, and physical conditions—not neighboring frames.

**Deliverable:** a calibrated future-failure predictor with measured warning
lead time and evaluation under held-out HUG grasp samples, objects, masses, and
motion conditions.

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
RGB-D object observation
    │
    ▼
HUG: generate MANO human-hand grasp samples
    │
    ▼
Part I: Dex Retargeting → Shadow Hand → grasp/lift/transport
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

The initial controlled evaluation fixes one Adroit/Shadow Hand embodiment and
one nominal friction value. It varies the **HUG grasp sample**, object identity
and shape, object mass, initial object pose, and lift/transport motion. Friction
remains supported and recorded but is not swept in the first batch.

The experiment schema includes a hand `embodiment_id`. The initial value is
`shadow_hand_right`; `mano_human_reference` identifies the source HUG grasp
rather than an actuated embodiment. See
[`docs/RESEARCH_FOUNDATIONS.md`](docs/RESEARCH_FOUNDATIONS.md) for the staged
embodiment and HUG integration plan.

Every case also carries the versioned environment ID
`adroit_shadow_tabletop_v1`. It resolves through an explicit registry to the
HUG/Shadow-Hand MuJoCo runner and fixes the scene, robot model, controller,
timestep, sensor/action contracts, cameras, and outcome definitions. Unknown
IDs and environment configuration-hash mismatches are hard validation errors.
See [`docs/EVALUATION_MANIFEST.md`](docs/EVALUATION_MANIFEST.md) for the portable
case and provenance contract.

The initial case also fixes `execution_protocol_id` to
`fixed_grasp_lift_v1`: reset, load, retarget, pre-grasp, approach, close, lift,
hold, and score. This is a deterministic scripted protocol. The initial HUG
evaluation does not train or invoke a manipulation policy, which keeps learned
controller failures out of the grasp-quality measurement. Results save the
resolved protocol parameters and configuration hash.

Hidden simulator parameters generate controlled trials and evaluation groups,
but they are not exposed to a sensor-based predictor.

## Current repository status

The repository has useful components, but the HUG-centered execution pipeline
and Parts II and III remain to be built:

- `AdroitHandRelocate-v1` inspection exposes the articulated arm, wrist, and
  Shadow Hand action space, object physics, and MuJoCo contacts.
- A validated adapter reads released
  [HUG](https://github.com/KevinyWu/hug) grasp predictions without depending on
  HUG's CUDA runtime.
- The robosuite/Panda package provides generic collection, logging, snapshot,
  LeRobot export, ACT smoke training, and a visible MuJoCo example. It does not
  meaningfully evaluate HUG MANO grasps.
- The HUG-to-Dex-Retargeting coordinate adapter, MuJoCo joint-name mapping,
  HUG-conditioned execution, batch evaluation, future-failure labels,
  intervention branching, and recovery learning remain to be implemented.

The default Adroit observation contains no tactile or contact measurements.
MuJoCo does maintain contact information internally, so a first simulation
study can surface contact-derived features through a wrapper. Those features
must not be described as realistic tactile sensing.

Adroit/Shadow Hand is the initial executable embodiment because its articulated
fingers can represent a retargeted HUG grasp. Its stock relocation task still
needs to be adapted to accept object assets and HUG grasp targets and to expose
the project's grasp/lift/transport outcomes. MuJoCo remains the physics engine
for controlled execution and later recovery branching.

Integration testing found that the stock Gymnasium Adroit MJCF does not use the
same Shadow joint names and axes as Dex Retargeting's official URDF. The runner
must therefore embed the pinned `dex-urdf` Shadow model or use a separately
validated kinematic conversion. Numeric suffix remapping is rejected because
it produces incorrect finger and thumb poses.

The physical robot is not required for Part I. Three hardware discovery tracks
run in parallel while the simulation policy is built:

- **Track A — simulation now:** integrate Dex Retargeting with its supplied
  Shadow Hand configuration and execute one HUG grasp in Adroit/MuJoCo; this is
  the critical path.
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

- **HUG predictions:** MANO grasp samples, wrist transforms, landmarks, and
  object-conditioned grasp geometry provide the Part I starting hypotheses.
- **Dexterous-hand executions:** successful and unsuccessful retargeted HUG
  rollouts provide Part II's future-outcome data. Every episode must contain
  synchronized observations, hand/arm state, actions, contacts, and outcomes.
- **Matched intervention trials:** simulator state branching produces the
  counterfactual comparisons needed by Part III.
- **A collaborator's VR-collected dataset:** usable for the central experiments
  only after auditing provenance, embodiment, synchronization, sensors, actions,
  outcomes, licensing, and physical-property labels. Data without failures or
  measured physical conditions may still support policy or representation
  pretraining.
- **HUG:** the foundational source of human grasp hypotheses, MANO hand poses,
  and the HUG-Bench object taxonomy. Predictions enter through the validated
  adapter and pass through Dex Retargeting before Shadow Hand execution.
- **Dex Retargeting:** the MIT-licensed, AnyTeleop-derived retargeting library
  used to convert HUG's 21 hand landmarks into Shadow Hand joint targets. The
  project integrates its supplied Shadow Hand configuration instead of building
  a new retargeting optimizer.
- **See to Touch:** a research precedent for combining visual demonstrations
  with tactile policy adaptation. It motivates a later tactile recovery track,
  not the first simulation batch.

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
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
```

The pinned `dex-urdf` submodule supplies the official Shadow Hand URDF required
by Dex Retargeting. The repository does not copy those third-party assets into
its own source tree.

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

### 2. Inspect a HUG prediction (no HUG runtime needed)

Reads a `grasp_pred/*.pkl` you already generated with HUG's released inference
code and prints its usable grasp fields (MANO pose, wrist transform, translation,
landmarks, mesh vertices, camera metadata).

```bash
python scripts/inspect_hug_prediction.py /path/to/grasp_pred/example.pkl
# or, after installation:
inspect-hug-prediction /path/to/grasp_pred/example.pkl
```

Retarget and numerically validate the pose in the official Shadow model loaded
by MuJoCo; add `--viewer` under `mjpython` for visual inspection:

```bash
view-retargeted-hand /path/to/grasp_pred/example.pkl
MUJOCO_GL=glfw mjpython -m grasp_failure_prediction.evaluation.view_pose \
  /path/to/grasp_pred/example.pkl --viewer
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

To watch the scripted Panda perform one successful episode in a native MuJoCo
window on macOS:

```bash
MUJOCO_GL=glfw .venv-part1/bin/mjpython -u scripts/watch_container_task.py
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

These commands exercise the supporting Panda pipeline. ACT is not part of the
initial HUG/Shadow-Hand evaluation protocol.

### 4. Run the tests

```bash
pytest
```

### 5. Run one HUG evaluation case

Place the trusted HUG prediction at the repository-relative path referenced by
the case, then run:

```bash
run-hug-eval-case \
  eval_cases/hug_case_001/case.yaml \
  --output runs/hug_case_001
```

For real-time playback on macOS:

```bash
MUJOCO_GL=glfw mjpython -m grasp_failure_prediction.evaluation.case_runner \
  eval_cases/hug_case_001/case.yaml \
  --output runs/hug_case_001_viewer \
  --viewer
```

The command validates both registry references and hashes before simulation,
then writes `resolved_case.json`, `trajectory.npz`, `final_state.npz`, and
`result.json`.

---

## Implementation roadmap

1. **Research and data contract:** freeze the task, embodiment, observation and
   action interfaces, failure taxonomy, logging schema, physics ranges, and
   held-out splits.
2. **Part I baseline:** retarget HUG MANO grasps to Shadow Hand, execute one
   grasp end to end, implement the versioned environment registry and eval
   command, then evaluate grasp/lift/transport behavior in batches.
3. **Part II benchmark:** freeze HUG, retargeting, and execution versions;
   collect balanced future-failure windows and train calibrated temporal
   baselines.
4. **Part III branching:** restore matched simulator states, test bounded
   interventions and delays, then train the intervention-outcome model.
5. **Physical validation:** reproduce the important findings using the sensing
   and actuation available on the selected robot.
6. **Optional extensions:** richer tactile sensing, additional dexterous-hand
   embodiments, learned recovery control, reinforcement learning, and joint
   training.

---

## Repository layout

```
src/grasp_failure_prediction/
  environments/adroit.py      # Adroit/MuJoCo inspection (runnable baseline)
  integrations/hug.py         # validator for primary HUG grasp inputs
  part1/                      # supporting Panda manipulation infrastructure
    config.py                 #   frozen task/physics/split contract
    environment.py            #   weighted-container task (robosuite/Panda wrapper)
    snapshot.py               #   exact MuJoCo state capture/restore (Part III primitive)
    record.py                 #   canonical episode schema + RoboMimic-style HDF5
    scripted.py               #   privileged scripted demonstrator (teacher)
    collect.py                #   demonstration collection
    lerobot_export.py         #   HDF5 -> LeRobot dataset for ACT
    train_act.py              #   ACT training loop
    stack.py                  #   training-stack validator
scripts/
  inspect_adroit.py           # entry point for the Adroit inspection
  inspect_hug_prediction.py   # entry point for the HUG-output inspection
  watch_container_task.py     # visible supporting Panda simulation
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
The relationship to HUG, HUG-Bench, MANO hand poses, and See to Touch is in
[`docs/RESEARCH_FOUNDATIONS.md`](docs/RESEARCH_FOUNDATIONS.md).

## Attribution

This project builds on ideas and the released output format of **HUG (Human
Universal Grasping)**:

- Repository: <https://github.com/KevinyWu/hug>
- HUG is distributed under the MIT License.

HUG's authors retain all rights to their work; this repository only consumes
HUG's released prediction format and does not redistribute HUG code, weights, or
the MANO assets it depends on.

The recovery and tactile-adaptation direction is also informed by
[See to Touch](https://see-to-touch.github.io/); no See to Touch code or data is
redistributed here.

## License

MIT — see [`LICENSE`](LICENSE).
