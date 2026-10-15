# Saved HUG camera projection and frame consistency — 15 October 2026

All **32 original saved proposals pass the frozen numerical checks**. The run uses 32 static `mj_forward` calls, zero physics steps, no inference, no retargeting, no fitting, no new proposals and no outcome labels. Every bound input/source remains unchanged. This is an additional engineering check for replaying the saved-command pipeline, not validation of native contact preservation or dynamic grasp execution.

## Selection and prior declaration

The protocol selects all 32 cases from the existing inventory without filtering by outcome, object-mask support or image placement. Source, seven manufactured controls and protocol were committed at `cd2e9de` before measurement. `protocol.json` has SHA-256 `2ee4a847d274e890a1ae50aa19ec6db250500033d2fadd5c8331125f78eb3c3f`. It binds the native files, source inventory, original RGB/depth/mask/intrinsics/camera transform, capture model/state, observation/inference/scene contracts, implementation and tests. Original input files remain external and are not copied or changed.

Before measurement, the numerical allowances were fixed at 1e-8 pixels for stored-2D and projection round trips, 1e-12 metres for world coordinate round trips, 1e-10 for the compiled camera transform, 1e-9 pixels for compiled intrinsics, 1e-6 for independently decoded float32 wrist rotations and 1e-12 metres for wrist translation. These distinguish float64 arithmetic from stored float32 rotation precision; they are engineering arithmetic allowances, not physical calibration or retargeting-error budgets. No allowance was changed after seeing the measurements.

## What the projection check establishes

Every original pickle contains 21 saved `landmarks_2d`, 21 predicted `landmarks_3d`, 778 predicted camera-space mesh vertices, native articulation/placement parameters and a wrist transform. Intrinsics/image dimensions are absent from the pickle itself, so the audit obtains them from the hash-matched original observation. All images are 224 by 224 pixels. Saved camera-space vertices/landmarks already include wrist rotation and translation; applying the wrist transform again would be wrong.

Pinned upstream `src/models/mano.py` projects using `proj = points_3d @ K.T`, followed by `proj[:, :2] / (proj[:, 2:3] + 1e-8)`. All saved 3D landmarks are float32; all matching intrinsics are float64; the matrix product, denominator and saved 2D points are float64. The independent source-order calculation and independent scalar calculation each reproduce every stored 2D point exactly (maximum discrepancy 0 pixels, 32/32 cases). Therefore the concern that epsilon was added to an unpromoted float32 depth does not arise for these inputs. There is no implementation repair or tolerance change.

A standard pinhole projection without HUG's epsilon is reported separately. Its maximum difference from HUG export convention is 3.552341e-6 pixels for landmarks and 4.243272e-6 pixels for mesh vertices. This is an intentional source-convention difference, not a camera-frame error.

**The saved 2D points are projections of generated 3D predictions, not independent observed hand labels.** These RGB-D captures precede the generated hand. Exact saved-export agreement is algebraic/export consistency; it does not prove that HUG predicted an observed hand, camera sensor calibration is physically exact, or the native grasp is good.

## Static frame results

The original captured MuJoCo model is restored from its bound binary and qpos/qvel. Static camera readback converts its OpenGL camera axes to OpenCV x-right/y-down/z-forward, then compares the result to saved `T_world_camera.npy`. Maximum transform discrepancy is 3.330669e-16; fovy/image-derived intrinsics match saved intrinsics exactly. This checks the captured simulator camera, not a newly rendered image or real camera.

Independent world-to-camera solving returns original points within 1.942891e-16 metres for landmarks and 2.289835e-16 metres for mesh vertices. Their projection round trips differ by at most 1.136868e-13 and 1.278977e-13 pixels respectively. Independent interleaved 6D Gram-Schmidt decoding differs from saved float32 wrist rotation by at most 1.133060e-7; wrist translation agrees exactly.

The frozen v1 helper accepts affine transforms for algebra and the v1 gate checks agreement to the compiled camera. At the coordinator's explicit request, the independent readback additionally checks rigid camera orthogonality and determinant, separately from the unchanged v1 gate, using its existing 1e-10 camera allowance. All 32 pass: maximum orthogonality residual 4.440892e-16, maximum absolute determinant-minus-one 3.330669e-16. This extra check does not alter the original result or imply general frame validation beyond the saved inputs.

## Image and scene-depth support (descriptive)

All **672/672 landmarks** and **24,896/24,896 vertices** have positive camera depth, fall within the declared pixel-centre window `0 <= u <= 223`, `0 <= v <= 223`, and have a valid nearest scene-depth pixel. No points are clipped or discarded. Of those points, 305/672 landmarks and 10,821/24,896 vertices project inside the object mask. Mask membership is not an acceptance gate: hands naturally extend beyond an object's mask and poor proposals are retained.

At the frozen descriptive 2 mm depth allowance, 176/672 landmarks and 6,321/24,896 vertices are behind the nearest scene-depth sample; 491/672 landmarks and 18,204/24,896 vertices are in front. The remaining 5 landmarks and 371 vertices lie within the allowance. The depth image contains the scene without the generated hand; nearest-pixel sampling, hand/object overlap, background and occlusion all affect these counts. They are neither observed hand-depth errors nor continuous occlusion/penetration certificates, and do not change any proposal or label.

## Files and reproduction

- `protocol.json`: premeasurement selection, tolerances and complete source/input hashes.
- `results.json`: every case's numerical checks and descriptive support counts.
- `readback_saved_projection.py` and `independent_readback.json`: separate scalar/source-dtype projection, rigid-camera checks and aggregate measurements; no simulator import or forwards.
- `manifest.json`: hashes of this evidence bundle.

From the declared project runtime, run the seven manufactured controls with `PYTHONPATH=src /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python -m pytest -q tests/test_hug_projection.py`. They cover known projections, epsilon semantics, behind-camera support, nontrivial frame inversion, double transformation, interleaved rotation columns, degenerate rotations, outside-image points, descriptive scene-depth/mask sampling and invalid values.

The measured output refuses overwrite. For reproduction, run `scripts/audit_hug_camera_projection.py prepare --output /path/to/fresh/output` and then its `run` action with the same fresh output, in the same declared runtime. The independent readback similarly refuses overwriting its result. Changes to a bound source/input require a separately frozen version. No inference, simulation integration, fitting or label generation is part of either command.

## Gate for the next pilot

This check supports using the declared existing32 camera/native-export inputs in a small, separately declared saved-command recording pilot. It is not permission to merge executor versions or to claim independent observed 2D validation, native-to-Shadow contact equivalence, future proposal generalization, successful grasps or improved prediction. The pilot must retain proposal/mapper/scene/controller provenance, null stopped/rejected labels and existing physical task outcomes separately.
