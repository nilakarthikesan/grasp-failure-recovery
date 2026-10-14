# HUG mapping validation continuation

14 October 2026, existing codex/project-work source at 7a06603. This bundle
records static diagnostics of already generated HUG proposals. It contains no
new evaluation episodes, labels, model samples or fitted predictors. HUG remains
the active individual mapping case; other generators have not been confirmed.

## Evidence and interpretation

- `saved_proposals/existing_hug_inventory.json` binds 32 distinct existing native
  poses to saved original Shadow commands across cube, apple, milk box and mug:
  two observation groups per object and four seeds per group. The sources share
  one MANO shape. Source and mapping hashes are checked; copied Oct 13 cube
  records are duplicates, not additional cases.
- `saved_proposals/static_saved_hug_fk.json` forwards the original 24 named
  targets through their captured scene models, at the initial object pose.
  It selects the original closure endpoint at 2.16 s, before force-close and
  without applied command noise. No optimization or integration occurs.
- `surface/protocol.json` was frozen before running the saved-cube refinement.
  `surface/hug_surface_refinement.json` separates occupied-or-near footprints,
  unsigned anatomical-part surface proximity and conditional same-label normals.
  The three per-grid reports are snapshots of its detailed fields.
- `hug_mapping_validation_progress.png` summarizes point residuals and geometric
  footprint stability. These are distinct metrics, neither predictor accuracy
  nor a percentage of mapping correctness.

Overall fingertip mean is 6.428 mm across the 32 stored poses, with per-pose
means 5.704--8.305 mm and worst tip 17.518 mm. Fixed-URDF and captured-scene task
links agree within 7.845e-8 m; object pose and zero simulation time are preserved;
no named target violates the captured model's limits. This supports numerical
consistency of the implemented transform/joint policy, not anatomical equality.
The carrier quaternion includes the fixed URDF base offset and must not be
compared directly with the native operator basis.

The cube footprint difference persists over 2, 1 and 0.5 mm grids. At a 2 mm
band global IoU is 0.582946, 0.580406 and 0.578557. Grid stability does not prove
continuum convergence or contact fidelity. At the finest grid, little-finger
unsigned proximity on the top cube face is nearly lost, not strictly absent:
native 57 mm^2, Shadow 0.5 mm^2, shared 0.25 mm^2. The old coarse-grid absence
statement remains correct for that recorded grid, but must not be generalized.

Same-label geometric normal comparisons are conditional subsets. Of 748 fine
2 mm-band matched exterior queries, 548 have both nearest hand points inside the
cube, 100 one inside, and 100 neither inside. Their all-subset median normal
angle is 25.56 degrees; the neither-inside subset has median 59.99 degrees.
These are geometric surface directions, not paired physical contact forces.
Native skinning labels are anatomical proxies; open finger triangle subsets
are never assigned enclosed volume. Reports retain unmatched regions and label
ambiguities. No complete native-to-Shadow contact-equivalence claim is made.

## Reproduction and preservation

All source/scene/command paths, hashes and runtimes are recorded in the reports.
Licensed assets are read locally and never copied. See the surface README for
its frozen protocol and exact import path. Copy scripts and protocols to a new
temporary directory before reproduction; preserve the recorded files here.
The inventory/FK and plotting snapshots retain their original temporary output
paths; change their OUT/index/source constants to a new temporary directory in
those copies before running. Do not execute them to overwrite this bundle.
All FK scripts guard mj_step, mj_step1 and mj_step2; the inventory only reads.
The surface script reuses the hash-pinned setup-only prefix of the prior audit.
The plot script only reads completed reports.

`unchanged_components.json` binds the production mapper, executor, frozen model
and working manuscript before/after this phase. Parent manifest hashes bind the
published diagnostics. Historical source families were already inspected in
development and are not an independent held-out validation set. These audit
reports must not be counted as additional training or test episodes.

Current result: native representation confirmed for the original reconstruction
case; numerical transform/joint consistency extended to 32 stored poses;
native-to-Shadow grasp/contact fidelity remains approximate and unconfirmed.
See ../../docs/HUG_MAPPING_CONFIRMATION.md for the argument and next check.
