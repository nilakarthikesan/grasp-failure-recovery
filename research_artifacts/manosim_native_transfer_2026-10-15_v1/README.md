# Native HUG to upstream MANOSim static transfer

This separately frozen diagnostic uses the authors' right-hand MANOSim assets
at c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac. Their shape file is byte-identical to
the existing HUG shape file. No Shadow operator-frame rotation or 24-joint
retargeter belongs in this native path.

Two already inspected development natives (20 and 28) are reconstructed with
the original CPU parametric MANO layer. This is saved-pose forward kinematics,
not new HUG generation. Explicit source bone order, local pose convention,
right-hand shape, metres, camera transform and wrist origin must agree. The
adapter resolves joints by name and verifies anchors, not body centroids.

Numerical pose comparisons are predeclared: 5 micrometres for reconstructed
vertices/landmarks and native-to-compiled joint anchors, 1e-6 for rotation matrix
elements. These are arithmetic/asset-precision comparisons, never allowances
for contact or physical success. Float32 source rotations and their explicit
float64 parameterization residuals are preserved. No fitted transform or
silent pose normalization is allowed.

Both mesh-hull and capsule variants receive the same pose. Their rigid collision
surfaces are compared against the original HUG skinned mesh using the unchanged
saved whole-cell proximity fields and bands. A hull, capsule or rigid tip cannot
be assumed to equal pose blend shapes and linear blend skinning. Full orientation
is not measured here, so this program cannot issue native surface fidelity PASS.
Kinematic PASS is a distinct, narrower verdict.

Per case: at most 600 seconds, 200 static forwards, two parametric MANO forwards
and two complete collider-variant scans. Task integration, new grasp generation,
retargeting, physical labels and predictor training are prohibited. The old
scene contributes only a frozen object-frame reference; its hand is never queried.
Every output array and pose is indexed by its creation-time hash. Source, assets,
runtime and Git HEAD remain fixed until independent saved-array readback.
The decoded MANO skin reconstruction stays local and ignored, following the
repository's existing third-party asset policy. Its hash remains in the tracked
receipt; raw source models and generator weights are not copied into the package.

MANOSim's 67 qpos entries, 51 velocities and 51 actuator commands require a
distinct controller/recording/data protocol. Ball-joint qpos quaternions are
different from the axis-angle values used by its position actuators. This
static qualification does not test tracking, contact compliance, timestep
convergence, real-world success or existing Shadow predictor validity.

Source: https://github.com/KevinyWu/manosim/tree/c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac.
