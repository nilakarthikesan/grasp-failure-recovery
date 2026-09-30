# Part I: Grasp Execution and Failure-Data Collection — Proposed Revision

Status: proposal for review, not a frozen contract or completed benchmark.

## Research decision

Use existing HUG grasp proposals and a fixed Shadow Hand execution controller
as the primary data-collection approach. The scientific question is whether
observation/action history predicts future object loss under held-out physical
conditions. Learning a nominal manipulation policy is no longer a prerequisite.

This is a better fit for that question because multiple grasp proposals can be
compared under the same execution protocol. It reduces confounding from changes
in policy training, while retaining variation in grasp geometry and physics.
It does not eliminate controller or retargeting errors; these need separate labels.
More proposals alone do not establish a better dataset or generalization.

The existing Panda/ACT pipeline remains a separate learning baseline. Its scoring
module does not implement the proposed Shadow Hand experiment.

## 1. Proposed execution loop

1. Obtain HUG proposals for an object and record model/checkpoint provenance.
2. Retarget each proposal to the Shadow Hand; validate joint limits, reachability,
   initial collision state, and coordinate/scale conventions.
3. Reset the scene with the selected object, mass, friction profile, pose, and seed.
4. Execute a fixed approach, finger closure, lift, and hold sequence.
5. Record synchronized observations, commands, contacts, phase transitions, and outcomes.
6. Repeat across proposals and controlled physical conditions.
7. Train a temporal future-loss predictor on designated training conditions.
8. Evaluate it on complete held-out episodes and physical/object/proposal groups.

Initial protocol candidate: lift 0.15 m above the initial object reference height
and hold for 2 s. These are provisional; validate reachability and sensitivity
before freezing them. A normalized closure setting is not calibrated grip force.
Friction is nominal in the first runnable pilot; mass/friction grids follow only
after the pilot and labels are trustworthy.

## 2. Outcome taxonomy

Record distinct outcomes rather than one undifferentiated failure label:

- Invalid proposal/retargeting: failed pre-execution validation.
- Acquisition failure: no established hold before the acquisition deadline.
- Lift failure: acquired but did not reach the required height.
- Post-acquisition object loss: unintentional departure from an established grasp.
- Success: acquired, reached the height, and retained the object through the hold.
- Timeout/incomplete follow-up: insufficient time to determine the intended outcome.
- Collision, excessive contact force, controller fault: separate safety/error flags.

An object dropped onto the table still counts as object loss. Below-table position
alone is insufficient. Define acquisition and loss using validated combinations
of contact, hand-relative motion, support contact, and persistence thresholds.
Slip is a possible precursor; it is not automatically object loss.
All thresholds and the event timestamps need manual trajectory review and
sensitivity checks. Report rejected proposals and acquisition failures separately
so filtering cannot hide poor grasp proposals.

## 3. Prediction target and information boundary

Predict P(object loss within horizon | history so far, continue fixed protocol).
Use windows after validated acquisition for the primary future-loss target.
Evaluate acquisition prediction separately if added. Never label every frame of
an eventually failed episode as imminent failure. Exclude windows with incomplete
future follow-up or use an explicit censoring-aware method.

Candidate inputs: available camera history, measured robot/hand state, issued
actions, and protocol phase/intended motion. Contact-derived features are an
explicit simulation-proxy ablation, not a claim of real tactile sensing.
Mass, friction, ground-truth object pose, future frames, and outcome labels stay
outside the sensor-based predictor. Log privileged values for labels/analysis.

## 4. Dataset and held-out tests

Collect successful and unsuccessful executions of a fixed versioned controller.
Define train/validation/test condition grids before collection. Predictor training
must contain controlled physical variation; tests reserve specified values or
combinations. A fixed controller does not imply nominal-only predictor training.

Separate tests for unseen proposals, mass values, friction values, mass/friction
combinations, and objects. Distinguish interpolation from extrapolation. Split
by complete episodes, group related proposal/condition repeats, and prevent
neighboring windows or equivalent grasps from crossing a claimed holdout.

Initially, "unseen environment" means held-out physical conditions in the same
simulated embodiment. New robots, rooms, and real hardware require separate evidence.
Record seeds, object/grasp identifiers, sampling method, condition allocation,
controller/retargeting versions, camera/physics configuration, and code commit.
Do not inflate statistical confidence by treating correlated windows as independent.

## 5. Evaluation and prospective findings

Execution metrics: proposal validity, acquisition success, lift success, retention,
post-acquisition object-loss rate, timeouts, and separate safety events.
Prediction metrics: event recall at a fixed false-alarm budget, warning lead-time
distribution, probability calibration, and per-shift performance. Include simple
state/motion and reactive-slip baselines; compare information sources fairly.
Use episode/event-level counts and confidence intervals, with all denominators explicit.

Possible findings: mass/friction sensitivity of proposals; signals that precede
loss; predictor robustness or miscalibration on physical holdouts. These are
questions, not observed results. Pre-drop warning is not evidence of recoverability;
actionability requires later matched intervention/delay experiments.

## 6. Implementation status and exit gate

Implemented separately: Panda simulation, scripted demonstrations, dataset export,
and an ACT smoke-training pipeline. This is not a competent trained policy claim.
The evaluation-infrastructure branch adds schemas, registries, and retargeting
utilities; the complete HUG/Shadow Hand execution runner remains to be built.

Next gate: one valid proposal executes end to end with reviewed labels and saved
trajectory/provenance. Then run a small mixed-outcome pilot before scaling.
Replay validation must include controller/task state and repeated-action future
trajectories; exact qpos restoration alone does not establish identical execution.

## 7. Three-part roadmap

Part I: establish fixed grasp execution and collect trustworthy trajectories.
Part II: learn and evaluate future-loss prediction under held-out physics.
Part III: compare delayed interventions from matched states and learn recovery selection.

Existing Panda/ACT design retained below for historical comparison; it is not the
primary proposed experiment and its claims are subject to review.

<details>
<summary>Earlier Panda/ACT proposal</summary>

# Part I Training Specification

**Learn the nominal grasp-and-transport policy.**

This document is the frozen contract for Part I. It defines the task, the
observation and action interfaces, the nominal training physics, the episode
record, the data splits, and the closed-loop evaluation. Parts II and III
depend on every logged field defined here, so changes must be deliberate.

## 1. Platform

- Simulator: robosuite on MuJoCo.
- Robot: Panda arm with a two-finger parallel-jaw gripper.
- Task base: robosuite `PickPlace`, adapted to a single container and a target
  placement region.
- Imitation learning: ACT, trained through LeRobot on the exported dataset.
- Backup platform: ManiSkill (PickCube-derived task) if robosuite proves
  unworkable on the available machine.

MuJoCo is required, not incidental: Part III restores complete simulator state
to branch matched recovery interventions, and MuJoCo's `mjSTATE_INTEGRATION`
provides the only documented complete-state guarantee among the candidates.

## 2. Task and phases

The robot must grasp one rigid container from the table, lift it to at least a
target height, transport it a short distance, and place it in a target region.

Phases, logged per timestep as a discrete label:

1. `reach` — approach the container before contact.
2. `grasp` — close the gripper and establish a hold.
3. `lift` — raise the container to at least height `H` above the table.
4. `transport` — carry it toward the target region.
5. `place` — lower and release inside the target region.
6. `done` — terminal success or failure.

The initial task deliberately excludes clutter, deformable objects, and
in-hand reorientation.

## 3. Success and failure

- **Success:** the container reaches at least height `H`, is transported to the
  target region, and is released stably inside that region within the time
  limit.
- **Failure:** the container is dropped or slips from the gripper; is never
  lifted to `H`; leaves the permitted workspace; or is not placed in the target
  region before the time limit.

Collisions, excessive gripper force, and controller faults are logged as
**separate** flags, not merged into the primary success/failure label. This
separation lets Part II distinguish grasp-physics failures from unrelated
controller errors.

`H`, the target region tolerance, and the episode time limit are recorded as
explicit constants in the task configuration and echoed into every episode's
metadata.

## 4. Observation interface

The learned policy receives only these observations:

- `rgb_front` — fixed scene camera image.
- `rgb_wrist` — wrist-mounted camera image.
- `joint_pos`, `joint_vel` — arm joint state.
- `eef_pose` — end-effector position and orientation.
- `gripper_state` — commanded width and measured opening.

The policy does **not** receive object mass, friction, object pose, or
simulator contact truth. Those are privileged metadata (Section 7).

## 5. Action interface

- Action: Cartesian end-effector motion (delta position and orientation) plus a
  gripper open/close command.
- An existing operational-space / low-level controller executes motor commands.
- ACT predicts short action chunks; execution uses receding-horizon replanning.

This portable Cartesian-plus-gripper interface is the same contract every
candidate physical robot must satisfy, so the policy remains transferable.

## 6. Nominal training physics

Part I trains under a single nominal physical condition:

- fixed nominal container mass;
- fixed nominal surface and contact friction.

Randomize only what should not leak the Part II study:

- container starting position and yaw within the workspace;
- optionally, minor lighting/visual variation.

Do **not** randomize mass or friction during Part I training. Those shifts are
reserved for the frozen policy's Part II evaluation, so training them away here
would destroy the later generalization question.

## 7. Episode record (canonical schema)

Every episode is logged with synchronized, timestamped fields:

- `t` — timestamp and control-step index.
- `phase` — phase label from Section 2.
- observations: `rgb_front`, `rgb_wrist`, `joint_pos`, `joint_vel`,
  `eef_pose`, `gripper_state`.
- `action` — commanded Cartesian delta and gripper command.
- outcome: terminal `success`/`failure` plus separate `collision`,
  `excess_force`, and `controller_fault` flags.
- privileged metadata (labels/analysis only, never policy input):
  - `object_pose`;
  - `object_mass`, `object_friction`;
  - `contacts` — MuJoCo contact pairs and forces;
- reproducibility: RNG `seed`, task configuration (`H`, tolerances, time
  limit), and a complete simulator/controller `state_snapshot` sufficient to
  restore and replay the step deterministically.

The `state_snapshot` must include MuJoCo `mjSTATE_INTEGRATION`, changed
`mjModel` parameters (mass, inertia, friction), environment counters,
controller goals/filters/integrators, RNG state, observation-delay buffers, and
the ACT action-queue/temporal-ensemble state.

## 8. Data splits

- Split by **complete episode**, never by neighboring timesteps.
- Hold out a set of initial container poses for closed-loop generalization.
- Reserve object-identity and physical-condition holdouts for Part II; Part I
  itself trains and evaluates under the nominal condition only.

## 9. Evaluation

Report closed-loop results from execution, not training loss:

- per-phase success: acquisition, lift, transport, placement;
- object-loss rate;
- collision and excess-force rates;
- a learning curve as the number of demonstrations increases.

A decreasing imitation-learning loss is not evidence of a competent policy;
only closed-loop rollouts from held-out starting conditions are.

## 10. Data sources

1. **RoboMimic `Can`** — validate the loading, ACT-conversion, training, and
   closed-loop evaluation pipeline. Not the scientific training set.
2. **Project scripted demonstrations** — the actual Part I training data for the
   weighted-container task, using the exact observation/action contract above.
   Teleoperation is added only after the task and labels are verified.
3. **A collaborator's VR dataset** — used only if a schema/embodiment audit
   passes; may otherwise serve as pretraining/reference. Not a dependency.
4. **DROID / RH20T / BridgeData** — optional later pretraining if the eventual
   physical arm matches their embodiment.

## 11. Exit criteria

- A reproducible ACT policy trained on project demonstrations completes the task
  from held-out starting conditions.
- Demonstrations and evaluations share this documented schema.
- Closed-loop results establish competence.
- Logs already contain the histories, physical-condition metadata, and
  restorable states that Parts II and III require.
- The selected physical-hardware path is documented before large-scale
  scientific data collection.

</details>
