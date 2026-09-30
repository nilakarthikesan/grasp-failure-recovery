# Part I Hardware Audit and Decision Gate

Part I runs in simulation now (Adroit/Shadow Hand in MuJoCo). A physical robot
is **not** required to produce the first result. This document audits the parallel
hardware-discovery tracks and defines a single decision gate that must pass
before any purchase, custom build, or large-scale physical data collection.

No hardware is bought and no custom build is authorized by this document. Each
track below is an information-gathering activity.

## Why three tracks in parallel

We do not yet have enough information to commit to one robot, so all three
options advance simultaneously and feed one decision gate. Whichever track
first meets the acceptance criteria at acceptable cost/risk becomes the Part I
physical target; the others become fallbacks.

## Track A — Simulation now (critical path)

* Status: **active**. Adroit inspection and HUG prediction validation run on
  this machine; the Dex Retargeting coordinate/joint adapter and execution
  sequence remain to be built.
* Deliverable: HUG-conditioned dexterous grasp executions, closed-loop
  evaluations, and the logs/snapshots that Parts II and III consume.
* Blocking on physical hardware: **no**. Simulation results are labeled
  simulation-only until reproduced on a real robot.

## Track B — Friend-built robot

A collaborator offered to build a robot. Before authorizing a build, obtain a
written specification and evaluate it against acceptance criteria.

Specification to request:

- degrees of freedom, reach, and payload (must comfortably exceed container mass);
- repeatability / positional accuracy;
- control interface and rate for arm, wrist, and individual dexterous-hand
  joints at the Part I control rate;
- telemetry: joint position/velocity, wrist pose, per-finger state, and —
  ideally — force/torque or tactile signals for Parts II/III;
- hand type, degrees of freedom, joint limits, and whether it exposes position,
  effort, calibrated force, or tactile feedback;
- a URDF / MJCF or other simulation model for sim-to-real alignment;
- a documented software API (Python preferred) and emergency-stop behavior;
- bill of materials, build time, and total cost;
- whether the design and results are publishable (no restrictive IP).

Acceptance criteria (all must hold):

1. can execute a documented retargeting from HUG/MANO grasps;
2. streams the observations in the Part I contract at a usable rate;
3. has a usable simulation model for alignment;
4. repeatability is sufficient for a tabletop grasp-and-place task;
5. cost, build time, and safety are acceptable and it is publishable.

## Track C — Lab or funded hardware

The author knows people at robotics labs and could request access; access is not
guaranteed, and lab robots may not be lendable. Separately, a commercial arm
(for example an xArm6 or FR3-class arm) could be purchased if funding is secured.

Actions:

- enumerate reachable labs/contacts and, for each, what robot is available,
  access rules, supervision requirements, and scheduling;
- clarify whether results collected there are publishable by the author;
- for the funded option, scope one concrete arm + dexterous hand, its API, its
  simulation model, and total cost; treat purchase as **funding-dependent**
  (pursued only if a grant / lab budget / sponsor is obtained).

Acceptance criteria:

1. confirmed, scheduleable access (or secured funding) for enough time to
   collect the planned physical trials;
2. the robot meets the same interface/telemetry criteria as Track B;
3. publishability of the collected data is confirmed.

A low-cost teleoperable arm (e.g. SO-101-class) may validate the software
plumbing and LeRobot capture path, but is not assumed sufficient for the
quantitative force/recovery claims of Parts II–III.

## External VR dataset audit

Independent of the robot choice, audit the offered VR-collected dataset before
relying on it for the central experiments. Check:

- provenance and embodiment (what robot/hand; does it match our task?);
- synchronization of observations, robot state, actions, and timestamps;
- available sensors (RGB, depth, tactile/force?) and their calibration;
- action representation and control rate;
- presence of **failures** and any recorded **physical-condition** labels
  (mass, friction) — without these it cannot supply Part II failure labels;
- licensing / permission to use and publish.

Outcome tiers:

- **Central use:** only if embodiment, synchronization, actions, outcomes, and
  physical-condition labels all pass.
- **Pretraining / reference use:** if it is well-formed but lacks failures or
  physics labels.
- **Not used:** if provenance, licensing, or synchronization cannot be verified.

## Decision gate

Choose exactly one physical target when, and only when:

1. Track A has executed retargeted HUG grasps with logged trajectories,
   snapshots, and closed-loop evaluations; and
2. at least one of Tracks B/C meets all its acceptance criteria at acceptable
   cost and risk; and
3. the chosen platform exposes the required arm, wrist, dexterous-hand, and
   observation interfaces so the retargeting and evaluation pipeline can be
   transferred with documented changes.

Until the gate passes, work stays in simulation and all quantitative results are
reported as simulation-only.
