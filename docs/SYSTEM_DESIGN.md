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

The physical and grasp parameters under which an attempt is executed. The first
batch will vary HUG grasp sample, object identity and shape, object mass,
initial pose, and lift/transport motion while holding the Adroit/Shadow Hand
embodiment and nominal contact friction fixed.

### Unseen condition

A physical value or combination intentionally excluded from training and used
only for validation or testing. This must be divided into distinct cases:

- unseen mass under nominal friction;
- unseen grasp placements and transport motions;
- extrapolation beyond the training mass range; and
- later, unseen object categories, friction values, and embodiments.

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

The system is built and frozen in stages:

```text
HUG grasp samples
    |
    v
Part I: execute fixed grasp/lift protocol with Shadow Hand
    |
    +-- collect successful and failed executions
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

### 6.1 Part I — Deterministic HUG-grasp execution

The first experiment answers:

> Does this HUG-generated grasp succeed under a fixed execution protocol?

The initial experiment does not train a manipulation policy. It uses the
versioned deterministic `fixed_grasp_lift_v1` protocol so grasp failures are
not confounded with controller-learning failures. Existing inverse kinematics,
motion planning, actuator interfaces, and feedback controllers execute its
fixed sequence.

Part I proceeds in this order:

1. Validate the registered environment and fixed execution protocol.
2. Verify coordinate frames, action scaling, gripper commands, contacts,
   success labels, and safety limits.
3. Retarget one HUG grasp and visualize the resulting Shadow Hand pose.
4. Run the fixed pre-grasp, approach, close, lift, and hold sequence.
5. Expand the validated one-case command into a versioned evaluation manifest.

A learned execution policy may be studied later as a separately versioned
protocol. It is not part of the initial grasp-quality evaluation.

Part I reports acquisition, lift, transport, and placement success separately,
along with object loss, collisions, and excessive-force events where those can
be measured.

### 6.2 Part II — Future-failure prediction

A validated Part I environment, retargeter, and fixed execution protocol are
frozen and run across controlled variations in:

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

1. Base Part I execution system alone.
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

**A collaborator's VR-collected dataset** enters only after a dataset audit covering:
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

The current Adroit environment is the starting point for Part I because its
Shadow Hand exposes the articulated fingers needed for a retargeted MANO grasp.
The stock relocation task is not yet the final evaluation environment.

**Selected initial experimental embodiment:** Adroit/Shadow Hand in MuJoCo.
HUG produces the source MANO grasp; the existing
[Dex Retargeting](https://github.com/dexsuite/dex-retargeting) library maps its
21 landmarks to Shadow Hand joint targets before grasp, lift, and transport
execution. The integration uses the library's supplied Shadow Hand
configuration and explicitly maps its output to MuJoCo joints by name. The
stock Adroit relocation environment is only a starting model and must be
adapted for HUG grasp targets, benchmark objects, logging, and controlled
outcomes.

The existing Panda parallel-jaw pipeline is retained as generic manipulation,
logging, snapshot, and policy infrastructure. A two-finger gripper cannot
execute the finger configuration expressed by a MANO grasp, so Panda results
are not the primary HUG experiment.

Integration testing found that Gymnasium's stock Adroit MJCF and the official
Shadow URDF used by Dex Retargeting differ in joint naming and axis conventions.
The evaluation runner must use the pinned official Shadow kinematics or a
separately validated conversion. It must not infer a mapping by decrementing
joint-number suffixes.

The selection criteria were:

- an articulated multi-finger hand capable of representing a retargeted MANO
  grasp;
- support for pick, transport, and place;
- demonstrations or a reliable scripted controller;
- complete state save and restore for Part III;
- controllable mass, inertia, center of mass, and friction; and
- access to contacts without exposing privileged values to learned models.

The deciding factor was recovery branching. MuJoCo is the only examined engine
that documents a complete integration state (`mjSTATE_INTEGRATION`) whose
restoration reproduces identical forward dynamics; PhysX-based stacks expose
scene-state restoration without an equivalent full-engine guarantee. A correct
implementation must additionally snapshot environment counters, controller
state, RNG, observation-delay buffers, changed model parameters (mass, inertia,
friction), and any controller or policy action-history state, since stock
robosuite state helpers capture only time, `qpos`, and `qvel`.

**Hardware tracks.** The physical robot is not required for Part I. Track A
(simulation) is the critical path now. Track B evaluates a friend-built robot
only against a written specification and acceptance tests. Track C investigates
lab access and funding-dependent hardware (for example xArm6 or FR3-class arms).
A single hardware gate selects one target embodiment before large-scale
scientific data collection; a portable Cartesian-plus-gripper action interface
keeps the policy transferable. A low-cost arm such as SO-101 may validate
software plumbing but is not assumed sufficient for quantitative force/recovery
claims. See [PART_I_TRAINING_SPEC.md](PART_I_TRAINING_SPEC.md).

HUG is the source of grasp hypotheses for the primary experiment. Its RGB-D
model predicts wrist translation, wrist rotation, and a MANO hand pose, and its
grasp can be retargeted to robot hands. Retargeting a MANO pose does not by
itself create a simulated human hand: execution requires an articulated hand
model, collision geometry, actuators, a controller, and an embodiment adapter.
The Adroit/Shadow Hand supplies the first executable target. See
[RESEARCH_FOUNDATIONS.md](RESEARCH_FOUNDATIONS.md).

The planned batch-evaluation schema therefore records an `embodiment_id`. The
initial value is `shadow_hand_right`; `mano_human_reference` identifies the
non-actuated source grasp. Metrics are stratified by embodiment.

Every evaluation case also records a versioned `environment_id`. The initial
entry, `adroit_shadow_tabletop_v1`, resolves through the evaluation platform's
registry to the HUG/Shadow-Hand MuJoCo runner. It fixes the simulator scene,
table, arm and hand, controller, timestep, observation/action contracts,
cameras, and success/failure definitions. Objects, HUG grasps, mass, starting
conditions, motion profiles, and seeds vary at the case level. Each resolved
run records the environment configuration hash, MuJoCo version, and code
commit. Unknown environment IDs and configuration-hash mismatches fail before
execution. The complete contract is in
[EVALUATION_MANIFEST.md](EVALUATION_MANIFEST.md).

Each case also records `execution_protocol_id`. The initial
`fixed_grasp_lift_v1` registry entry defines the deterministic reset, load,
retarget, pre-grasp, approach, close, lift, hold, and scoring sequence plus its
timing, distance, force, and height parameters. The resolved protocol parameters
and hash are stored with every case and result. Unknown IDs or hash mismatches
fail validation.

## 8. Foundational Resources

- [ACT](https://arxiv.org/abs/2304.13705) is an optional later learned-execution
  protocol after the deterministic HUG-grasp benchmark is established.
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) is another optional
  learned action-sequence protocol for that later comparison.
- [The Feeling of Success](https://arxiv.org/abs/1710.05512) studies whether
  vision and touch can predict grasp outcomes.
- [Maintaining Grasps within Slipping Bound](https://arxiv.org/abs/1810.13381)
  studies incipient slip as a warning signal for grasp instability.
- [Tactile Sensors for Friction Estimation and Incipient Slip
  Detection](https://doi.org/10.3390/s20010221) reviews the connection between
  friction, tactile sensing, and grip security.
- [See to Touch](https://see-to-touch.github.io/) learns tactile dexterity from
  vision-derived rewards and motivates a later tactile residual/recovery track.
- [Human Universal Grasping](https://grasping.io/) predicts human grasps from
  RGB-D in MANO form, retargets them to robot hands, and provides HUG-Bench
  objects across five geometry categories and three size ranges.
- [Dex Retargeting](https://github.com/dexsuite/dex-retargeting) supplies the
  AnyTeleop-derived landmark-to-robot optimization, scaling, joint constraints,
  and a ready-made Shadow Hand configuration used by Part I.
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
- audit the offered external dataset and the available physical robot.

### Phase 1 — Build HUG-conditioned dexterous execution

- Generate and validate HUG MANO grasp samples from RGB-D inputs.
- Integrate and validate Dex Retargeting with its supplied Shadow Hand
  configuration; do not implement a new optimizer unless a documented test
  shows that the library fails this use case.
- Convert HUG camera coordinates and units to the Shadow palm frame, map output
  by MuJoCo joint name, and validate fingertip alignment, joint limits, and
  collisions.
- Execute one grasp, lift, and transport attempt end to end in MuJoCo.
- Register `adroit_shadow_tabletop_v1` and implement strict manifest validation
  before expanding beyond the one-case command.
- Register `fixed_grasp_lift_v1`, save its resolved parameters and hash, and
  reject unknown or mismatched protocol configurations.
- Collect a small HUG-conditioned rollout set with complete logging.
- Evaluate behavior across held-out grasp samples, objects, and initial poses.
- Save successes and naturally occurring failures in the canonical format.
- Confirm that simulator states can be restored for future intervention trials.

**Exit criterion:** a reproducible HUG-conditioned dexterous-hand pipeline can
execute grasps often enough to produce meaningful successes and controlled
failure cases.

### Phase 2 — Build the future-failure benchmark

- Freeze the HUG version, retargeting method, dexterous-hand controller, and
  execution protocol.
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
- Reproduce the initial controlled mass sweep at nominal friction, then add a
  small friction robustness sweep if the hardware supports it.
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

The first concrete deliverable is a complete HUG-conditioned grasp execution
baseline with excellent logging:

- a validated Adroit/Shadow Hand simulation environment;
- a thin HUG-to-Dex-Retargeting integration using the supplied Shadow config;
- one saved HUG grasp displayed in MuJoCo with fingertip alignment error;
- that grasp executed through pre-grasp, close, lift, and transport;
- a portable evaluation manifest;
- held-out closed-loop evaluation;
- stored successful and failed executions; and
- logs that already preserve the histories needed by Part II and the simulator
  states needed by Part III.
