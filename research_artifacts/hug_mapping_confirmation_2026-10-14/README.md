# HUG individual mapping confirmation

14 October 2026. The active generator is HUG; other providers are not advanced
to confirmed status. Read [the HUG walkthrough](../../docs/HUG_MAPPING_CONFIRMATION.md).

These are static diagnostics of the existing saved cube/seed-0 output, not
training/evaluation data. No new HUG sampling, physics integration, task cases,
labels or model fitting occurred. MANO forward kinematics reconstructs an
existing encoded pose. The adapter, v2 executor and frozen predictor are unchanged.

`hug_native_reconstruction.json` verifies the native vertices/landmarks/closed
faces and projection. `hug_boundary_checks.json` records manufactured rigid
coordinate/quaternion/joint-order checks, not new native grasps.
`hug_unit_test_result.json` records 18 passed existing tests with physics
stepping prohibited. `hug_surface_confirmation.json` records actual native
triangles/Shadow collision primitives and per-geom hulls at the same initial
object state. Its occupied-or-near cube footprint IoU is a grid/band-dependent
geometric diagnostic including penetration, not contact-patch identity or a
predictor score. Normal comparisons use nearest surfaces at common queries;
skinning labels approximate anatomy. The two figures use shared axes/camera/scale.

Snapshot scripts use exact local paths and dependencies in their reports.
To reproduce, copy scripts to a temporary directory before running; do not
overwrite these recorded reports. Some script outputs follow their own script
directory, while the surface/native scripts use fixed temporary output roots.
The existing MANO asset is read locally and identified by hash, not bundled.
`no_rollouts_pytest_plugin.py` snapshots the guard used for the existing tests.

Manifest hashes preserve the inspected evidence. Current status is partial:
native representation and frame/joint conventions confirmed within tested
scope; detailed native-to-Shadow contact fidelity remains unconfirmed.
