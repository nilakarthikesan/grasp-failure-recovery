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
