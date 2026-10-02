# Alignment investigation

The original integration's no-contact result was reproduced. Controlled variants
used the same saved observation, model prediction, initial cube state, and physics.
These checks do not establish a successful grasp or validate a failure dataset.

## Observation and model

RGB, depth, mask, intrinsics, and the selected pixel passed preflight. The cube's
backprojected surface agreed with its simulated geometry within 0.012 mm before
millimeter depth quantization. The CV camera uses right/down/forward axes and its
camera-to-world transform was checked against the known cube surface.

Independent CPU inference runs with the same seed produced identical input tensor
hashes and proposal hashes. Checkpoint loading rejects missing learned weights;
excluded frozen image-encoder and MANO buffers come from their separate assets.
This establishes repeatability and basic loading, not grasp quality or robustness
to synthetic images. Projecting a grasp onto an RGB image alone cannot validate
its depth, contacts, or ability to resist gravity.

## Conversion and motion

1. Original hand conversion: mean fingertip residual 158.7 mm, no cube contact.
2. More optimization alone: residual 115.3 mm, still no contact.
3. MANO-to-Dex axis conversion: residual 19.7 mm; retaining the old global palm
   orientation instead produced a table collision.
4. Consistently converting the vectors and composing the URDF palm orientation:
   cube contact during closure, no table collision, but no successful lift.
5. Full target closure rather than the protocol's 0.6 fraction: more contact,
   still no successful lift.
6. Removing Dex's 1.2 target-vector scaling: residual 4.2 mm mean, 8.9 mm maximum.
   Contact occurred in 12 closing frames and 6 lifting frames, none during hold.
   The cube still stayed on the table.

The axis conversion is a proper rotation derived from Dex's right-hand
`OPERATOR2MANO_RIGHT`. Vector conversion and pose conversion must be paired;
unit tests verify they preserve world-coordinate landmark geometry. These are
experimental changes in our scripts, not edits to the partner's branch.

The approach reached its commanded palm position within floating-point precision
in all variants. The original visible gap therefore was not an unfinished
approach; the commanded robot geometry was inconsistent with the human grasp.

For the unscaled variant, sampled normal contact force peaked at approximately
1.22 N for an individual contact and penetration reached approximately 2.75 mm.
This is a simulator contact proxy at recorded steps, not calibrated tactile
data. An individual peak force does not establish opposing contacts, frictional
support, or a force-closure grasp.

## Remaining questions

- The mapping still identifies the human wrist origin with the robot palm origin.
  Review physical origin correspondence and full-hand geometry, not only tips.
- Inspect thumb/finger opposing contact pairs and contact forces throughout all
  physics steps. The current contacts do not persist into a held grasp.
- Check human mesh versus robot collision geometry, joint limits, and whether the
  predicted human grasp is feasible for this robot. HUG has not been exonerated
  as a source of possible grasp-quality errors.
- Validate the fixed executor using a known feasible robot grasp before blaming
  predicted grasps. It directly sets hand qpos and zeros hand velocities rather
  than using an actuator/force controller. Test closure, approach direction, lift
  speed, mass, friction, and solver settings separately after geometry is sound.
- The scorer's `acquired_object` currently means any hand-object contact during
  manipulation. That label can be true even while the cube stays on the table;
  it should not be communicated as a secured grasp.
- The old viewer reconstructed poses from partial logs and did not preserve
  object orientation. New ablation traces contain full qpos for exact geometry
  replay. Replay is not a new physics rollout and cannot validate contact forces.

## Reproduce

Run `scripts/debug_sim_alignment.py` with `--observation`, `--urdf-root`, and a
fresh `--output` directory in the isolated combined runtime described in
`HUG_SIMULATION_PILOT.md`. It emits ablation scores, contact summaries, full-qpos
traces, and previews. Use `scripts/view_sim_attempt.py --trace /path/to/trace.npz`
with `mjpython` to inspect a recorded variant interactively.

The next gate is a verified opposing-contact grasp that actually lifts and holds
the cube. Alignment corrections alone are not that gate.

## Contact and physics follow-up

The corrected, unscaled HUG proposal was compared under original zero-hand-
velocity execution, interpolated motion with consistent hand velocities, and
consistent velocities plus elliptic contacts/no-slip iterations. None lifted
the cube. At sampled control steps, force-bearing contacts belonged to middle
and ring finger bodies; none belonged to the thumb. No pairs with opposing
normals (dot product below -0.5, individual normal force above 0.01 N) were
observed. This sampling does not rule out very brief contacts between samples
and is not a full wrench-feasibility test.

A finite engineering search added finger curling and small palm offsets to the
corrected proposal. None of 45 candidates completed the task, although one
briefly lifted the cube 40.6 mm. A later audit found that this search clipped Dex
joint arrays against MuJoCo limits without first matching joint names. Those
results are historical diagnostics, not valid evidence that the correctly
bounded configurations are infeasible. The search now matches limits by joint
name. These modified grasps are not raw HUG proposals and must not enter an
unchanged-proposal evaluation dataset.

An independent two-pad positive control used position actuators with 8 N force
limits, the same 180 g, 5 cm cube, gravity 9.81 m/s², and sliding friction 1.
With an elliptic cone and 10 no-slip iterations, closed pads lifted approximately
149.7 mm and maintained at least 148.7 mm during the two-second hold. Open pads
did not lift the cube. Earlier configurations with default solver settings
lifted but drifted below the 140 mm threshold; increasing position gain alone
did not meet the threshold. This demonstrates solver sensitivity in that pinch
setup, not a universal requirement for ten iterations.

This is a positive control for the physics engine and frictional pickup. It has
different hand geometry, a dynamic welded carrier, actuators, and solver settings
from the Shadow scene. It does not establish a known successful Shadow grasp.
MuJoCo documents its additional no-slip solver as addressing frictional drift:
https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-noslip_iterations

The simple quasi-static vertical pinch condition is
`mu_left * N_left + mu_right * N_right >= m * g`, with opposing horizontal
forces and balanced moments also required. For ideal equal opposing contacts
with `mu = 1` and this cube, each normal force must be at least about 0.883 N.
This is a necessary friction-capacity calculation for that geometry, not a
general dexterous-hand success criterion. During upward acceleration, replace
`m*g` by `m*(g+a)`; contact location and torque balance still matter.

Scripts `check_contact_execution.py`, `search_contact_baseline.py`, and
`check_physics_pinch.py` save the corresponding reports and traces under a fresh
`--output` directory. The pinch regression test compares closed and open pads.
The next implementation gate remains a validated Shadow Hand grasp controller,
followed by unchanged HUG proposal executions with distinct acquisition, lift,
hold/slip, and drop labels. Keep solver/controller versions fixed across physical
holdouts and exclude integration-error cases from physical-failure training.

## Observation and collision compatibility audit

`audit_sim_compatibility.py` checked the actual saved input consumed by HUG.
The metric array and millimeter depth PNG agreed within 0.00049945 m, which is
normal half-millimeter rounding. Official input preparation preserved the depth
and calibration exactly at the pilot's existing 224px resolution. Dataset and
official interactive-inference RGB normalization matched; selected pixel
`[111, 111]` and depth approximately 0.501 m matched the interactive contract.
The predicted wrist landmark matched the wrist-transform translation exactly.
The human mesh remained above the table; its minimum world height was 20.64 mm.
These checks do not establish generalization from real images to rendered scenes.

Compiled simulator dimensions were confirmed to be 50 x 50 x 50 mm and mass
180 g. The Shadow model has no actuators. At the end of closing in the corrected
trace, `mj_geomDistance` found a thumb collision-surface gap of 8.70 mm, index
finger distal gap 5.33 mm, and middle distal penetration 2.88 mm. The observed
hand-object contact is therefore not an opposing thumb/finger grip. Collision
geometry and landmark geometry are distinct; small average landmark residual
alone is insufficient for retargeting acceptance.

Inference now rejects inconsistent meter-array versus depth-PNG encodings before
loading the model and records hashes of the consumed PNG and prepared sample.
Tests cover rounding, wrong units/stale depth, and invalid-pixel preservation.
The adapter documentation's transform direction was also corrected: despite
the name `T_camera_wrist`, HUG exports a wrist-to-camera transform.

The next design work should establish an actuated Shadow Hand grasp/controller
with a known feasible contact configuration, and validate physical surface
contacts when mapping MANO proposals. Do not collect physical-failure training
labels yet from a pipeline whose nominal Shadow grasp has never succeeded.

## Actuated Shadow baseline: successful pickup (2026-10-02)

The known-feasible Shadow grasp gate above is now met for one engineered cube
grasp. This supersedes the earlier statement that no Shadow pickup had
succeeded. It does not establish unchanged HUG grasp success.

Changes made in a separate diagnostic runner:

- Added 24 force-limited position servos. After initialization, joint motion is
  integrated by MuJoCo; the controller updates targets instead of overwriting
  hand joint positions and zeroing velocities at every control step.
- Used an ideal arm carrier constrained to the forearm, never to the object.
  This represents commanded arm support, not a calibrated physical arm.
- Kept all original collision shapes and the 50 mm, 180 g cube unchanged.
- Used a partially shaped approach rather than fully straight fingers, followed
  by 0.8 seconds of settling after closure. Straight fingers extending from the
  bent-grasp wrist pose could collide with the table during approach.
- Used implicit-fast integration, elliptic friction, and ten no-slip iterations.
  Wrist gains are 50, finger gains 10, and damping 0.3. Finger torque limits are
  at most 1 Nm; wrist limits use the imported model limits. These settings are
  experimental and are not hardware calibration.
- Matched Dex/MuJoCo joint names before clipping or optimizing joint limits.
- Fit the index, middle, and thumb collision surfaces to opposing cube faces,
  with small bounded palm offsets, starting from the corrected HUG retargeting.
  This is an engineered geometry correction using privileged simulator state.

With the final controller, the unchanged corrected HUG proposal still failed
acquisition: maximum lift was 3.09 mm and there was no force-bearing opposition.
The engineered contact fit lifted 144.65 mm and held the object for 2.0 seconds.
There was no approach collision. Force-bearing contacts involved the thumb,
index, and middle fingers, and opposition was observed throughout the hold at
physics-step resolution. This opposition diagnostic uses normal-force threshold
0.01 N and normal dot product below -0.5; it is not a full grasp-wrench test.

Three reruns from the same captured state all succeeded with identical full-qpos
traces. This checks deterministic repeatability, not robustness across objects,
seeds, masses, or friction. Opening the thumb with everything else fixed failed
acquisition, lifted only 0.48 mm, and produced no opposition. The peak measured
actuator force was at most 92.61% of each actuator's configured bound across the
successful runs. Torque and contact normal force are different measurements.

The baseline regression also succeeds from a fresh static cube initialization
using the saved JSON joint target, without loading HUG or MANO. Its open-thumb
negative control fails. A separate regression checks that adding actuators does
not change collision geometry or attach the cube to a constraint. These tests
require the optional partner evaluation package and downloaded Shadow assets.

The existing `acquired_object` flag still denotes any contact, so it can be true
in the failed open-thumb trial. Use lift and hold success, plus contact diagnostics,
when describing this result; do not equate that flag with a secured grasp.

### Reproduce the successful baseline

Use the isolated combined runtime described in `HUG_SIMULATION_PILOT.md`:

```sh
python scripts/fit_shadow_contacts.py \
  --observation /path/to/cube_observation \
  --urdf-root /path/to/dex-urdf/robots/hands \
  --axis x --output /path/to/contact_fit

python scripts/validate_shadow_baseline.py \
  --observation /path/to/cube_observation \
  --urdf-root /path/to/dex-urdf/robots/hands \
  --target /path/to/contact_fit/engineered_target.npz \
  --output /path/to/baseline_validation

mjpython scripts/view_saved_rollout.py \
  --scene /path/to/baseline_validation/scene.xml \
  --trace /path/to/baseline_validation/repeat_1_trace.npz
```

The viewer replays the full joint/object poses recorded during a physics rollout;
it does not recompute dynamics or inference. Space pauses/resumes and R restarts.

Keep original HUG proposals, retargeting corrections, and engineered positive
controls labeled separately. The next gate is evaluating fresh HUG proposals
with a fixed, validated conversion/controller and deciding which geometry
corrections belong in the declared execution protocol. One successful engineered
grasp is a simulator positive control, not a failure-prediction training dataset.
