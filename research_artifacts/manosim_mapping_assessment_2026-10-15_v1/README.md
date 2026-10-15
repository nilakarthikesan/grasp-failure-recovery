# MANOSim native HUG transfer: source-only assessment

15 October 2026. Prepared without executing upstream modules, MANO forwards,
MuJoCo compilation/forward/steps, HUG inference, geometry queries, optimization,
training, installation, or repository mutations. This is an implementation plan,
not a mapping certificate or measured validation result.

## Pinned sources and observed local assets

MANOSim is pinned to
[`c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac`](https://github.com/KevinyWu/manosim/commit/c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac)
(29 July 2026 UTC), read from the parent's source-only clone at
`/private/tmp/manosim-c96e2d5-2026-10-15`. `source_inventory.json` records SHA-256
hashes and literal XML inventories. No model was compiled to obtain dimensions.

The [README](https://github.com/KevinyWu/manosim/blob/c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac/README.md)
identifies this as the HUG simulation hand and identifies `myhand` as HUG's hand
shape. This claim has an independently checked local asset link: the pinned
`assets/mano_rhand_shape.npy` and local HUG file at
`/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/assets/mano_rhand_shape.npy`
are byte-identical. Existing saved source-FK evidence reports the same one shape
for all 32 development proposals; that report was read, not regenerated here.

Local HUG source is pinned at `8d1c52d4c24bfae5a369e32e3f134f5601a02630`.
Its MANO_RIGHT.pkl is already present in the local licensed MANO assets; no new
model download is required to plan a shape-specific reconstruction. The
source-only MANOSim clone already contains prebuilt right/left `myhand` and
`zerohand` XMLs and per-bone OBJ meshes, so compiling its right `myhand` model
does not require re-running the mesh builder. Licensed MANO assets must not be
silently redistributed with the MIT code.

## Why a direct native transfer is feasible

HUG's `src/models/mano.py:26-98,111-163` uses right MANO with `center_idx=0`,
`use_pca=False`, `flat_hand_mean=True`, stores fifteen local axis-angle rotations
as `(1,15,3)`, ten shape coefficients, and the wrist's camera-frame rigid
transform. Translation is metres. The saved camera-frame mesh/landmarks already
contain the root rotation and translation.

MANOSim's
[`mano_build.py:20-38,155-186`](https://github.com/KevinyWu/manosim/blob/c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac/src/mano_build.py#L20)
uses the same MANO setup and hierarchy:
`(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)`.
Bones 1-3 are index, 4-6 middle, 7-9 little, 10-12 ring, 13-15 thumb.
Its source maps these bones to MANO21 indices
`(0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3)`.

The literal prebuilt right XMLs each have 16 bodies, 3 slides and 16 ball joints,
51 position actuators. XML-inferred dimensions are `nq=67,nv=51,nu=51`.
[`build_mjcf.py:80-127`](https://github.com/KevinyWu/manosim/blob/c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac/src/build_mjcf.py#L80)
places `mano_body_0` at the wrist origin and compensates the palm mesh centroid.
Fingers have no axis restriction or 24-coordinate Shadow morphology bottleneck.

The proposed transfer is therefore:

1. Require exact native file, shape, MANO asset, source, XML/mesh/config and camera
   calibration identities. Refuse an unmatched shape; do not default to zero
   shape. Keep source raw rotations and conversion residuals in the receipt.
2. Set `wrist_tx/ty/tz` from the translation of
   `T_world_camera @ T_camera_wrist`. Convert that transform's rotation to the
   `wrist_rot` quaternion. Use the standalone right model with identity root
   body placement, so the slides express world XYZ.
3. Map saved `pose[0,i-1]` directly to the quaternion of named `mano_joint_i`,
   `i=1..15`. Use joint name/address lookup, never assumed qpos offsets.
4. Treat control and configuration separately. Ball qpos has four quaternion
   entries; its three XYZ position-actuator targets are the principal
   axis-angle-vector components. The slide targets are metric translation.
   MuJoCo 3.3.7 documents the gear-axis/axis-angle transmission and wrapping at
   pi in its [XML reference](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#actuator-general-joint).
   Predeclare the branch policy; retain conversion error and near-pi cases.
5. Do not apply Shadow's MANO-to-operator basis, a second root rotation on saved
   vertices, or the viewer's display-only R_RIGHT/left-right spacing.

This removes the need for an IK retargeter in the native-hand path. It does not
establish physical tracking or contact fidelity merely by assigning qpos.

## What remains approximate

The [builder](https://github.com/KevinyWu/manosim/blob/c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac/src/mano_build.py#L58)
partitions the neutral mesh by maximum skinning weight, discards cross-bone
faces, and creates one convex hull per bone. Mesh-hand collision is 16 hulls;
capsule-hand collision is 15 capsules plus the palm hull. Capsules use the 50th
percentile radial distance, with thumb-metacarpal override 20th percentile.
These are authored collision approximations, not the posed 778-vertex MANO
skin. OBJ coordinates use six decimal places (metres); XML coordinates use six
significant digits. Quantization error must be measured explicitly without
reclassifying the existing scientific tolerance.

The viewer's [overlay](https://github.com/KevinyWu/manosim/blob/c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac/src/visualize_hand.py#L296)
skins a zero-pose mesh with rigid-body transforms and weights. It does not
recompute nonzero-pose MANO corrective blend shapes. Its regressed skin
landmarks are also not an independent bone-anchor reference. Existing local
source-FK evidence already finds exact articulated neutral tip sites differ
from full pose-corrective skinned tips by up to 1.551407 mm on 32 development
proposals. That historical value is not a new MANOSim result.

The `myhand` config supplies mass/inertia/contact/gain choices. README statements
about human-demonstration correlation are upstream reports, not results validated
locally. Their soft-contact penetration caveat does not justify relaxing the
mapping acceptance test. The viewer disables the MuJoCo warning callback at
import, so the validation harness must not import that viewer or hide warnings.

## Exact next validation boundary, before any collection

Freeze a separate native-MANOSim protocol after peer review. Use saved native
files only, no new HUG outputs, task labels or optimizer. A minimal first static
qualification could use the existing original 32 as development readback, with
additional previously unused saved native files reserved prospectively if their
identities and availability are established. The 32 are not unseen validation.

First qualify kinematics independently: zero pose, one-joint basis rotations,
nonzero multi-joint and wrist rotations, rigid camera/world covariance, and
literal saved-pose reloading. Compare all 16 joint anchors and orientations to
independent MANO rest-skeleton FK; record numeric float32/quaternion/XML errors
and keep skin-derived fingertip metrics separate. Assert exact named assignment,
source state immutability, no integration and finite transforms. Read qpos and
actuator lengths back independently; do not treat ctrl assignment as tracking.

Then measure contact geometry under the original unchanged same-part protocol:
saved full MANO skin versus mesh hulls versus capsules, each separately;
complete full-field proximity inventory, exact eligible orientation witnesses,
and penetration diagnostics at the same frozen object/world pose. Preserve all
failures and unresolved regions. Agreement of the skeleton, or success of a
native-to-itself identity control, cannot promote HUG-to-Shadow fidelity.

The existing fail-closed fidelity admission gate has no supported positive
native-MANOSim certifier. A separately implemented, source-bound native-hand
capability would need its own evidence definition before task execution.

Native MANOSim must also receive a new hand/controller/scorer/data-cohort and
feature contract. The frozen Shadow 24-coordinate prefix/model input contract
cannot silently accept the MANOSim 51-dimensional tangent/command state. Keep
prior Shadow results historical; new data and subsequent models must be versioned.

## Recommended priority

Qualify this native bridge before spending further attempts optimizing the
anatomically different Shadow retargeter. Preserve the static-v2/full-field repair
code and its honest negative evidence. MANOSim offers a directly motivated native
simulation baseline; it does not retroactively repair Shadow geometry or prove
real-robot transfer.
