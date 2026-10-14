# Saved HUG native representation contract — 14 October 2026

This extends the original representative-cube parameter reconstruction to all
32 already inspected distinct native proposals, four objects/eight observation
groups/one MANO shape. It reads saved pose/shape parameters and uses the pinned
MANO CPU forward-kinematics wrapper. It does not load/sample the HUG grasp
generator, optimize a new grasp, import a simulator, create evaluation data,
label an outcome or train a model. The original licensed MANO asset is hashed
and retained locally, without copying it into Git.

All 778 vertices, 21 landmarks and wrist rotation elements reproduce exactly
for every native proposal; all 1,552 closed face indices match. The original
float32 representation allowances (5 micrometers vertices/landmarks, 1e-6
rotation elements) were frozen before execution, reused rather than fitted.
Actual maximum errors are zero. This confirms decoding/export consistency for
these saved inputs; it does not validate human-to-Shadow contact preservation,
source inference accuracy or physical anatomy. Camera reprojection is still
only checked in the separate original representative-cube audit.

The frozen protocol binds all native file hashes, upstream wrapper and asset,
prior inventory and script. Production/model/manuscript hashes remain unchanged.
Simulator imports are blocked. Exact native reconstruction is a numerical
contract, separate from strict proximity/orientation and outcome claims.

Reproduction requires the same original saved paths, licensed asset and
preserved Python/PYTHONPATH runtime documented by preceding HUG bundles. Copy
protocol.json and check_native_contract.py into a fresh temporary directory
and run the copied script there with PYTHONDONTWRITEBYTECODE=1. It writes its
report beside the copy, without overwriting published evidence. Frozen native
cases are reused, not sampled. No package installation or network is needed.
