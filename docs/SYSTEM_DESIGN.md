# Learning, Anticipating, and Recovering from Grasp Failures

## Under Hidden Physical Variation

## System Design Document

**Status:** Working draft  
**Project repository:** `grasp-failure-recovery`

This document records the research question, experimental design, system
architecture, and decisions for the project. It is expected to change as we
review the literature, inspect available robot data, and test the simulator and
physical robot.

## 1. Thesis

Robots that manipulate objects in the real world will encounter physical
conditions that were not represented in their training data. An object's mass,
surface friction, center of mass, and contact geometry can make an apparently
successful grasp unstable during lifting and transport.

This project asks:

> How can a robot learn to grasp an object, recognize when its current behavior
> is likely to fail, and choose an effective recovery while recovery is still
> possible?

The project is one connected research program with three separately trained and
evaluated components:

1. **Grasp learning:** What action should the robot take?
2. **Failure prediction:** What will happen if it continues its current
   behavior?
3. **Recoverability and recovery:** Could another action still save the
   attempt, and which available action should it choose?

The components share the robot, objects, task, safety limits, data format, and
logging infrastructure. Their objectives, datasets, labels, and evaluations
remain separate so that improvements can be attributed to the correct part of
the system.

The initial publication goal is one coherent paper. The project does not assume
that each component will produce an independent paper. A component should be
separated only if it develops its own research question, contribution, and
substantial evidence.

## 2. Why This Problem Matters

The same underlying grasp-stability problem appears in several settings:

- **Homes and assistive care:** A robot may need to retrieve medicine, carry a
  glass, move dishes, or pick an object up from the floor. A dropped object can
  spill, break, or injure someone. Studies of older adults find that fetching,
  transporting, holding, and manipulating objects are among the forms of
  physical assistance people want from home robots.
- **Warehouses and manufacturing:** Robots encounter packages with different
  weights, surface materials, deformability, and centers of mass. A grasp that
  works for one product may slip when applied to a heavier or smoother product.
- **Disaster response and hazardous-object handling:** Remote robots may
  manipulate dangerous objects while their operators have limited force or
  tactile feedback. Detecting instability before an object is dropped is more
  useful than detecting it after the drop has begun.

These applications motivate the research, but the first experiment will remain
narrow and controlled. A controlled experiment is necessary to determine
whether mass and friction caused the failure rather than an unrelated change in
the object, controller, or environment.

## 3. Robot Task

The robot must:

1. Approach and grasp an object.
2. Lift the object to at least height **H** above its starting surface.
3. Transport it over a short, specified path.
4. Place it inside a target region.

The initial task uses one rigid object on a table without clutter, deformable
objects, or complex in-hand manipulation. These restrictions make it possible
to understand the learning and contact behavior before introducing additional
sources of failure.

An attempt fails when the object:

- slips or drops from the hand;
- rotates or moves out of the permitted stable region;
- never reaches the required lift height; or
- is not placed in the target region within the time limit.

Collisions, excessive squeezing, and policy or hardware faults are important
safety failures, but they may need separate labels. Combining every failure
into one label would make it difficult to determine whether the predictor
learned grasp physics or merely recognized an unrelated controller error.

A fixed-duration hold test remains useful as a controlled diagnostic for Part
II. It is not the complete Part I manipulation task.

## 4. Prediction Task

The failure predictor receives only information available up to the current
time. It must not see observations recorded after the event it is supposed to
predict.

Three related targets must not be conflated:

1. Will a proposed grasp succeed before contact?
2. Is slipping occurring at the current instant?
3. Will the object be lost in the future if the robot continues its current
   behavior?

The main study uses the third target:

```text
p_fail(history, policy, horizon)
  = P(object loss within the horizon
      | observations and actions so far, continue current policy)
```

The prediction is conditioned on behavior because a grasp can survive a slow
movement and fail during a faster or more abrupt motion. The predictor must
therefore receive relevant action history or intended motion, not only an image
of the object.

Several fixed horizons, such as 0.25, 0.5, and 1.0 seconds, will measure whether
the system provides an actionable warning. Eventual task failure can be
reported as a secondary target.

For a prediction window ending at time **t**, the label is based only on object
loss during the following horizon. Frames early in an ultimately failed episode
must not all be labeled as immediately pre-failure. Windows without complete
future follow-up are excluded or explicitly treated as censored.

## 5. Initial Definitions

### Observation

Information available to a model at one instant. Depending on the experiment,
this may include joint positions, joint velocities, commanded actions, object
motion, contact forces, or tactile measurements.

### Manipulation policy

The controller or learned model that chooses the robot's next action. It is
responsible for grasping, lifting, and holding the object.

### Failure predictor

A separate model that estimates the risk of future grasp failure. It does not
initially control the robot.

### Physical condition

The physical parameters under which a grasp is attempted. The first study will
focus on object mass and contact friction while holding other factors as
constant as possible.

### Unseen condition

A physical value or combination intentionally excluded from training and used
only for validation or testing. This must be divided into distinct cases:

- unseen mass with familiar friction;
- unseen friction with familiar mass;
- an unseen mass--friction combination;
- extrapolation beyond the training range; and
- eventually, an unseen object combined with unseen physical conditions.

### Incipient slip

Localized motion at part of the contact region that begins before the object
undergoes gross sliding. Detecting incipient slip may provide enough warning for
a robot to adjust its grasp before a drop occurs.

### Calibration

The agreement between predicted probabilities and observed outcomes. If a model
assigns approximately 80% failure probability to many attempts, approximately
80% of those attempts should fail. This is important because a safety system
must communicate reliable risk, not only a binary answer.

### Demonstration

A synchronized episode showing the observations, robot states, and actions used
to perform the task. Successful demonstrations can teach a nominal skill, but
they do not by themselves show what precedes failure or which alternative
action would have prevented it.

### Intervention

A bounded recovery behavior available to the robot, such as increasing grip
within calibrated limits, slowing transport, changing motion, or returning the
object to a support surface.

### Recoverable

An attempt is recoverable only relative to a specified robot, action set,
response delay, time budget, and force limits. In the first study, recovery
means returning to a stable grasp and then completing the task within those
limits.

Safely placing the object down and abandoning the task is recorded separately
as an **object-preserving abort**. It is useful, but it is not successful
completion of the original task.

## 6. End-to-End System

The system is trained and frozen in stages:

```text
demonstrations
    |
    v
Part I: train nominal grasp-and-transport policy
    |
    +-- collect successful and failed closed-loop executions
    v
Part II: train future-failure predictor
    |
    +-- select warning states and branch matched intervention trials
    v
Part III: train intervention-outcome model and recovery selector
    |
    v
evaluate the integrated system against matched baselines
```

Changing all components simultaneously would make it difficult to determine
why performance changed. Joint training is therefore deferred until the staged
system is understood.

### 6.1 Part I — Grasp learning

The first learned policy answers:

> What should the robot do to grasp, lift, transport, and place this object?

The learned policy does not need to replace every part of the manipulation
stack. Existing inverse kinematics, motion planning, actuator interfaces, and
feedback controllers may execute the commands selected by the learned model.

The first baseline will use imitation learning. ACT is the leading candidate
because it predicts short action chunks from camera observations and robot
state, has a documented LeRobot training path, and gives the project a concrete
baseline without inventing a new policy architecture.

Part I proceeds in this order:

1. Validate the environment with a scripted or demonstration-driven controller.
2. Verify coordinate frames, action scaling, gripper commands, contacts,
   success labels, and safety limits.
3. Collect demonstrations using exactly the observations the learned policy
   will receive.
4. Train the policy and evaluate it in closed-loop execution from held-out
   initial conditions and objects.
5. Plot performance as demonstrations are added rather than assuming a fixed
   dataset size is sufficient.

An optional reinforcement-learning policy may later provide an educational and
scientific comparison. It is not required before Part II.

Part I reports acquisition, lift, transport, and placement success separately,
along with object loss, collisions, and excessive-force events where those can
be measured.

### 6.2 Part II — Future-failure prediction

A competent Part I policy is frozen and run across controlled variations in:

- object mass and surface friction;
- grasp placement;
- object position and orientation;
- transport speed, acceleration, and path; and
- separately identified disturbances introduced during execution.

The policy should not be deliberately weakened to manufacture failures.
Instead, experiments should cover progressively difficult but meaningful
conditions.

The first predictor will be a compact temporal model over recent observations
and actions. The key initial comparison is between information sources:

1. robot state and action history;
2. vision plus robot state and actions; and
3. vision, robot state, actions, and available contact measurements.

This tests whether contact history provides earlier or more reliable warning
than non-contact observations. It does not assume the answer.

Data is split by complete episode and object. Held-out physical conditions are
kept out of training. Adjacent windows from one trajectory must never be
randomly divided across train and test sets. Hidden simulator properties define
experimental conditions and labels but are not model inputs.

Part II reports failure-event recall at a fixed false-alarm budget, warning lead
time, AUROC and AUPRC, Brier score, negative log likelihood, calibration under
each shift, and risk–coverage behavior.

### 6.3 Part III — Recoverability and recovery

A high failure probability does not establish that a useful alternative action
exists. Part III first learns an **intervention-outcome model** and only later a
more flexible recovery policy.

For an observation history **h**, intervention **u**, and response delay
**delta**, the model estimates:

```text
Q_recover(h, u, delta)
  = P(successful recovery within limits
      | execute intervention u after delay delta)
```

The initial intervention set contains:

- continue the nominal policy;
- make a bounded grip adjustment supported by the hardware;
- slow or change the transport motion; and
- return to a support surface for regrasping.

Simulation provides the counterfactual labels that a normal trajectory cannot:
at selected points, save the complete simulator and controller state, restore
it, and execute each intervention under matched conditions and realistic
delays. The resulting record is:

```text
history + intervention + delay -> recovery outcome
```

Failure of every tested intervention does not prove physical impossibility. It
only establishes that the attempt was not recovered by the tested action set
under the tested limits. The model must be allowed to express uncertainty.

On physical hardware, exact replay is not assumed. Closely matched repeated
trials are used and their variability is reported.

After fixed interventions are understood, a bounded learned correction policy
may be trained from successful recovery demonstrations or controlled simulator
interaction. It must be compared against a strong fixed controller with the
same limits.

### 6.4 Integrated comparisons

The evaluation isolates the value of each stage:

1. Base Part I policy alone.
2. Current-slip or reactive detector plus a fixed recovery controller.
3. Future-failure predictor plus the same fixed recovery controller.
4. Future-failure predictor plus learned intervention selection or recovery.

The base policy, action limits, physical test conditions, and intervention
budget remain matched. A system must not appear better merely because it grips
harder or abandons more attempts.

If a warning triggers a successful recovery, the absence of a later drop does
not make the original warning a false positive. Simulator branching or a
matched no-intervention trial provides the comparison needed to determine what
would otherwise have happened.

### 6.5 External data and physical robot

**Shivam's VR-collected data** enters only after a dataset audit covering:
provenance and licensing; physical versus simulated collection; robot
embodiment; sensors and actions; coordinate frames and units; timestamps and
synchronization; successes and failures; object identity; and measured physical
conditions.

- Synchronized demonstrations can support Part I.
- Failed executions with sufficient future context can support Part II.
- Single observed trajectories generally cannot supply Part III's alternative
  intervention outcomes.
- Data without failures or physical-property labels may still support
  representation or policy pretraining.

The **physical robot** validates important findings under real sensing noise,
contact dynamics, delay, and actuation limits. Before integration, document its
arm and gripper, control modes, rates, payload, URDF, cameras, tactile or
force-torque sensors, calibration, timestamping, and safety limits.

Simulation-only results using ideal contact information remain explicitly
simulation results until tested with measurements available on the robot.

## 7. Current Technical Starting Point

The repository currently provides:

- inspection of `AdroitHandRelocate-v1`, a MuJoCo environment with a Shadow
  Hand and arm;
- access to robot state, object position, mass, friction, and simulator contact
  information; and
- an adapter for validating grasp predictions produced by Human Universal
  Grasping (HUG).

It does not yet train a policy, collect rollouts, create future-horizon labels,
branch simulator states, or train failure and recovery models.

The default Adroit observation does not contain tactile measurements. MuJoCo
does maintain contact information internally, so the first simulation study can
use contact-derived features as a simulation proxy. These features must not be
described as realistic tactile sensing.

The current Adroit environment is a useful physics and dexterous-contact
inspection baseline, but it is not automatically the best Part I environment.
Its Shadow Hand differs from the Panda-based ManiSkill PickCube sandbox and may
differ from the available physical robot.

Before collecting a large dataset, the project must select the simulated
embodiment using these criteria:

- similarity to the available physical arm and gripper;
- support for pick, transport, and place;
- demonstrations or a reliable scripted controller;
- complete state save and restore for Part III;
- controllable mass, inertia, center of mass, and friction; and
- access to contacts without exposing privileged values to learned models.

ManiSkill PickCube is the leading simple Part I candidate. Adroit remains useful
for dexterous experiments if the research and available hardware justify the
additional complexity.

HUG may later provide diverse human-like grasp hypotheses, but it is not
required for the first controlled experiment. Retargeting its MANO hand poses
to any selected robot is a separate engineering problem.

## 8. Foundational Resources

- [ACT](https://arxiv.org/abs/2304.13705) provides the initial
  action-chunking imitation-learning baseline for Part I.
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) is an established
  receding-horizon action-sequence baseline to consider after ACT.
- [The Feeling of Success](https://arxiv.org/abs/1710.05512) studies whether
  vision and touch can predict grasp outcomes.
- [Maintaining Grasps within Slipping Bound](https://arxiv.org/abs/1810.13381)
  studies incipient slip as a warning signal for grasp instability.
- [Tactile Sensors for Friction Estimation and Incipient Slip
  Detection](https://doi.org/10.3390/s20010221) reviews the connection between
  friction, tactile sensing, and grip security.
- [See to Touch](https://see-to-touch.github.io/) demonstrates tactile-based
  adaptation for dexterous manipulation.
- [Human Universal Grasping](https://grasping.io/) provides human-like grasp
  hypotheses and a benchmark of previously unseen objects.
- [SlipSense](https://arxiv.org/abs/2609.15910) predicts current tactile slip
  classes with measured detection latency; it is a reactive baseline, not the
  same target as future object-loss prediction.
- [FeelWorld](https://arxiv.org/abs/2607.24267) predicts action-conditioned
  future visual, contact, force-related, and slip states for planning. It is a
  central novelty comparison for any future world-model extension.
- [Foundational World Models Accurately Detect Bimanual Manipulator
  Failures](https://arxiv.org/abs/2603.06987) provides a compact probabilistic
  world-model and conformal failure-monitoring baseline.
- [Recovery RL](https://arxiv.org/abs/2010.15920) separates task execution from
  learned recovery for constraint satisfaction and must be included in the
  recovery novelty comparison.
- [TF-Gripper and RETAF](https://arxiv.org/abs/2602.10013) separates
  high-frequency force adaptation from lower-frequency arm-pose decisions,
  demonstrating why gripper position commands and calibrated force control
  cannot be treated as equivalent.
- [OopsieVerse](https://robin-lab.cs.utexas.edu/oopsieverse/) motivates
  evaluating physical damage and unsafe execution separately from task success.
- [Older Adults' Task Preferences for Robot Assistance in the
  Home](https://arxiv.org/abs/2302.12686) provides evidence that people want
  assistance with retrieving, transporting, holding, and manipulating objects.

## 9. Build and Research Phases

### Phase 0 — Freeze the research and data contract

Before collecting a large dataset:

- select the simulator and robot embodiment;
- define the observation and action interfaces;
- define success, each failure type, safe abort, and harmful intervention;
- identify what the gripper actually commands: position, velocity, effort, or
  calibrated force;
- define the logging schema and synchronization requirements;
- define mass, inertia, center-of-mass, and friction interventions;
- freeze episode-, object-, and physical-condition-level splits;
- define prediction horizons and intervention delays; and
- audit Shivam's sample data and the available physical robot.

### Phase 1 — Build the learned manipulation baseline

- Validate the task with a scripted or demonstration controller.
- Collect a small end-to-end dataset with complete logging.
- Train ACT on the project's own demonstrations.
- Evaluate closed-loop behavior on held-out initial conditions and objects.
- Save successes and naturally occurring failures in the canonical format.
- Confirm that simulator states can be restored for future intervention trials.

**Exit criterion:** a reproducible trained policy can complete the simple task
often enough to be meaningful while still producing failures in controlled,
increasingly difficult conditions.

### Phase 2 — Build the future-failure benchmark

- Freeze a Part I checkpoint.
- Collect balanced rollouts across controlled physical and motion conditions.
- Generate leakage-safe fixed-horizon windows and censored-window metadata.
- Train heuristic, static, compact temporal, and calibrated ensemble baselines.
- Evaluate modality ablations and held-out physical conditions.

**Exit criterion:** the predictor gives calibrated warnings earlier than a
reactive detector at a matched false-alarm budget.

### Phase 3 — Build intervention comparisons

- Select warning states spanning failure modes and warning lead times.
- Branch complete simulator states into matched intervention trials.
- Sweep intervention delay and bounded action magnitude.
- Train and evaluate the intervention-outcome model.
- Compare fixed recovery, learned selection, safe abort, and continuation.

**Exit criterion:** action-conditioned intervention selection improves recovery
over reactive and fixed-controller baselines under matched limits.

### Phase 4 — Validate on physical hardware

- Map only supported observations and actions onto the physical robot.
- Reproduce a small controlled mass/friction matrix.
- Use repeated matched trials rather than claiming exact counterfactual replay.
- Report sensing, actuation, calibration, synchronization, and safety limits.

## 10. Candidate Paper

**Working title:** *Learning, Anticipating, and Recovering from Grasp Failures
Under Hidden Physical Variation*

A candidate central thesis is:

> Action-conditioned future-failure prediction improves intervention decisions
> over reactive failure alarms, particularly when physical properties and
> response delays differ from training.

This is a hypothesis to test, not a result to claim in advance. Training an
established policy on the project task is necessary infrastructure, but it is
not by itself a novel grasp-learning contribution. Likewise, adding a detector,
world model, or recovery layer is not sufficient novelty without a specific,
well-controlled finding.

## 11. Immediate Next Deliverable

The first concrete deliverable is a complete learned grasp-and-transport
baseline with excellent logging:

- a validated simulation environment;
- a demonstration collection pipeline;
- one policy trained by this project;
- held-out closed-loop evaluation;
- stored successful and failed executions; and
- logs that already preserve the histories needed by Part II and the simulator
  states needed by Part III.
