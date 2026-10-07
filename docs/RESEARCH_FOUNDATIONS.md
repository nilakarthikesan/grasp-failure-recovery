# Research Foundations and Embodiment Scope

This project studies failure prediction and recovery for grasps proposed or
executed across different hand embodiments. It is informed directly by these
projects and released resources:

- [Human Universal Grasping (HUG)](https://grasping.io/) and its
  [released code](https://github.com/KevinyWu/hug) provide the central grasp
  representation and evaluation precedent. HUG predicts a human grasp from
  RGB-D as wrist translation, wrist rotation, and a MANO hand pose, then
  retargets that grasp to robot hands. HUG-Bench contributes a useful object
  taxonomy: five geometry categories, three size ranges, and metric-scale
  meshes for paired simulation and physical evaluation.
- [See to Touch (TAVI)](https://see-to-touch.github.io/) demonstrates how
  visual demonstrations and tactile feedback can be combined to adapt
  dexterous policies. It motivates a later tactile recovery track; it is not a
  dependency of the initial simulation baseline.
- [Dex Retargeting](https://github.com/dexsuite/dex-retargeting) is the
  MIT-licensed, AnyTeleop-derived library selected for mapping human hand
  landmarks to robot-hand joint targets. Its supplied
  [Shadow Hand configuration](https://github.com/dexsuite/dex-retargeting/blob/main/src/dex_retargeting/configs/teleop/shadow_hand_right.yml)
  already defines the relevant human landmark and Shadow fingertip/middle-link
  correspondences, scaling, optimization, and joint constraints.

The project is not a reimplementation of any of these systems. HUG supplies grasp
hypotheses and a human-hand reference space; this project asks whether an
executed grasp is likely to fail soon and whether an intervention can still
save it.

## Hand embodiment as an experiment parameter

The experiment schema should include an explicit `embodiment_id` rather than
assuming every episode uses the same hand:

| `embodiment_id` | Role | Initial status |
| --- | --- | --- |
| `mano_human_reference` | HUG-predicted human grasp pose and geometry | Prediction adapter only |
| `shadow_hand_right` | Initial articulated hand receiving a retargeted HUG grasp | Selected; Dex Retargeting integration pending |
| `panda_parallel_jaw` | Generic manipulation and logging infrastructure | Supporting only |

`mano_human_reference` is not yet a simulated actuator. A MANO prediction is a
kinematic hand pose and mesh. Turning it into an executable MuJoCo embodiment
requires an articulated hand model, collision geometry, joint limits,
actuators, a controller, and a retargeting map from MANO joints to the chosen
hand. Each executable embodiment also needs its own observation/action adapter
and policy or controller.

For that reason, embodiment will be a supported parameter in the batch
experiment metadata, but it is held fixed to `shadow_hand_right` in the first
batch. Results from different hands are stratified by embodiment rather than
pooled as if their action spaces and grasp mechanics were interchangeable.

## Retargeting implementation decision

Do not build a MANO-to-Shadow-Hand optimizer from scratch. Use Dex Retargeting
through a thin project adapter:

```text
HUG 21×3 landmarks + wrist transform
    → convert camera coordinates and units to the Shadow palm frame
    → Dex Retargeting with the supplied right Shadow Hand configuration
    → reorder output by the MuJoCo model's joint names
    → validate joint limits, collisions, and pose quality
    → execute pre-grasp, close, lift, and transport
```

The project-specific implementation is limited to:

1. coordinate-frame and unit conversion;
2. the HUG landmark adapter;
3. MuJoCo joint-name ordering;
4. wrist placement relative to the object;
5. collision and pose-quality checks; and
6. the pre-grasp and grasp-execution sequence.

Dex Retargeting's output order must be mapped explicitly by joint name because
simulators can use different joint orderings. A new optimization API is only
justified if a recorded integration test demonstrates that the existing
library cannot produce an acceptable Shadow Hand pose.

## Staged experiment plan

### Initial controlled batch

- Fix the executable embodiment to Adroit/Shadow Hand.
- Generate multiple MANO grasp samples per object with HUG and retarget them to
  Shadow Hand joint targets through Dex Retargeting.
- Sweep HUG grasp sample, object identity and shape, object mass, initial object
  pose, and lift/transport motion.
- Support friction in the schema and simulator, but hold it at one recorded
  nominal value.
- Record complete outcomes and failure reasons for every episode.

### HUG-conditioned execution path

- Import validated HUG predictions using the existing
  `integrations/hug.py` adapter.
- Treat the MANO pose and wrist transform as the source grasp hypothesis.
- Convert each hypothesis to Shadow palm coordinates and retarget it with Dex
  Retargeting's supplied Shadow Hand configuration.
- Evaluate multiple HUG grasp samples per object rather than treating one pose
  as ground truth.
- Use HUG-Bench geometry and size categories to define object-level holdouts.

The Panda parallel-jaw environment remains available for testing generic
logging, policy, and evaluation infrastructure. It is not used to claim that a
HUG MANO grasp has been executed.

## First executable milestone

Take one saved HUG prediction, convert its 21 landmarks and wrist transform to
the Shadow palm frame, run Dex Retargeting, display the resulting hand pose in
MuJoCo, and report fingertip alignment error. Only after this single case is
visually and numerically validated should the repository add the reusable eval
command and expand it into a batch manifest. That command must resolve the
versioned `adroit_shadow_tabletop_v1` registry entry and enforce the strict
validation contract in [EVALUATION_MANIFEST.md](EVALUATION_MANIFEST.md). It
must also resolve `fixed_grasp_lift_v1`; no learned manipulation policy is used
in this initial evaluation.

### Tactile adaptation and recovery

- Add signals available on the selected physical hand or gripper.
- Compare vision-only, robot-state, contact-derived simulation features, and
  real tactile observations without calling simulator contact truth tactile
  sensing.
- Use the See-to-Touch result as motivation for learning a tactile residual or
  recovery policy after the failure-prediction benchmark is established.

## Predictor boundary

Simulator parameters such as true mass and friction may be used by an oracle
baseline and for constructing controlled evaluation groups. The deployable
failure predictor receives only observations available to the executing robot,
plus its action history and an embodiment identifier. This preserves the main
question: can failure be anticipated from the behavior observed so far rather
than from hidden simulator truth?
