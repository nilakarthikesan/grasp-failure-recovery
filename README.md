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

The recommended starting point is imitation learning from synchronized
demonstrations, with ACT as the first reproducible baseline. Reinforcement
learning is a later comparison, not a prerequisite.

**Deliverable:** a trained grasp-and-transport policy, a reproducible training
pipeline, held-out closed-loop evaluations, and saved successful and failed
executions.

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

The repository is currently an early scaffold, not a completed learning system:

- `AdroitHandRelocate-v1` inspection exposes observations, actions, timing,
  object mass/friction, and MuJoCo contacts.
- A validated adapter reads released
  [HUG](https://github.com/KevinyWu/hug) grasp predictions without depending on
  HUG's CUDA runtime.
- Rollout collection, policy training, future-failure labels, intervention
  branching, and recovery learning remain to be implemented.

The default Adroit observation contains no tactile or contact measurements.
MuJoCo does maintain contact information internally, so a first simulation
study can surface contact-derived features through a wrapper. Those features
must not be described as realistic tactile sensing.

Adroit is useful for physics and contact inspection, but its dexterous
relocation task may not be the simplest Part I learning sandbox. The system
design therefore treats a simpler MuJoCo pick-and-place environment, such as
ManiSkill PickCube, as the leading initial policy-training candidate. The final
simulated embodiment should be chosen with the available physical robot in mind.

## Data sources

- **Project demonstrations:** the primary source for training Part I. Every
  episode must contain synchronized observations, robot state, actions, and
  outcomes.
- **Policy executions:** successful and unsuccessful closed-loop runs from Part
  I provide Part II's future-outcome data.
- **Matched intervention trials:** simulator state branching produces the
  counterfactual comparisons needed by Part III.
- **Shivam's VR-collected data:** usable for the central experiments only after
  auditing provenance, embodiment, synchronization, sensors, actions, outcomes,
  licensing, and physical-property labels. Data without failures or measured
  physical conditions may still support policy or representation pretraining.
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

Requires Python ≥ 3.10. Using [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install -e ".[dev]"
```

Or with standard tooling:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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

### 3. Run the tests

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
scripts/
  inspect_adroit.py           # entry point for the Adroit inspection
  inspect_hug_prediction.py   # entry point for the HUG-output inspection
tests/
  test_hug_adapter.py         # synthetic-dictionary tests for the HUG adapter
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
