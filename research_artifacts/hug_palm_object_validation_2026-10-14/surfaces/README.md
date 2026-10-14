# Existing HUG object surface diagnostic, 14 October 2026

Completed static geometry comparisons for **six existing seed-0 HUG proposals**:
the two preexisting observation groups for apple, milk box and mug, selected in
inventory source order before metrics. Each retains its saved native MANO mesh,
original 24 nominal Shadow targets recovered from stored commands before
force-close, captured root pose and initial object state. No generator inference,
retarget optimization, physics rollout, evaluation-data collection, predictor
fit, calibration, pose adjustment or source-quality exclusion occurred.

Read `highlights.json` for compact figures, `summary.json` for domain-separated
per-finger band and density comparisons, and the full reports for provenance,
collision masks, individual geometry metadata and conditional normals. These
are descriptive diagnostic bands, not acceptance gates or physical contact area.

## Two distinct geometric reference surfaces

- **Final collision-union family v3:** `protocol_v3.json`,
  `audit_other_surfaces.py`, `hug_other_object_surfaces.json`. Actual saved
  compiled object colliders: apple 2, milk box 7, mug 15 convex mesh geoms. Each
  also has one disabled visual mesh. Geom masks, body exclusions and explicit
  pairs are checked; all 45 active hand geoms are object compatible, with zero
  explicit pairs and zero exclusions. Meshes use each geom's actual compiled
  vertex convex hull. Sphere, box and cylinder target queries are analytic.
- **Actual visual-surface family v4:**
  `visual_reference_surfaces/protocol_v4.json`, `audit_visual_surfaces.py`,
  `visual_reference_surfaces/hug_visual_reference_surfaces.json`. Query the
  disabled visual mesh's actual compiled triangle table, without converting the
  object to a convex hull. All six identical hand poses are checked separately.
  The two families are never pooled.

Both families use midpoint subdivisions until child triangle area is at most
4 mm² for all six cases and 1 mm² for the first source-order group of each
object. Queries are child centroids with exact triangle-area weights. This is
an **area density** parameter, not uniform spatial spacing: actual sample
weights and child edge distributions are reported. Compare unsigned per-finger
proximity at 1, 2 and 5 mm, including shared, lost native and new Shadow regions.
Open native anatomical triangle subsets are never artificially closed. Whole
native winding occupancy and any-target-collider occupancy stay separate.

Shadow unsigned queries minimize the absolute signed distance to each individual
compatible geom. Inside overlapping hand colliders, nearest per-collider surfaces
may be buried inside the hand union; this is **per-collider surface proximity**,
not exact exterior union-boundary distance. Any-collider occupancy remains a
union-membership measurement. The globally exterior normal mask avoids that
interior-interface ambiguity.

Normal comparisons use globally exterior, separated queries near both hands
and require matching nearest anatomical labels. Different labels and unmatched
queries remain explicit. Closest points inside the object **collision union**
are classified separately, including in the visual-reference family. Native
skinning labels and robot body-prefix labels are geometric proxies.

## Boundary correction and object approximation

The preserved `v2/` pilot removed strictly interior overlap points and duplicate
same-oriented faces but retained exactly abutting opposite-normal interfaces.
A geometry-only outward-probe check detected 546.49 mm² of such sampled seam
area on the milk box and 1,429.60 mm² on the mug at coarse density. Apple had
zero occluded outward probes at both densities. **V2 is superseded for final
collision-union metrics.** Its frozen protocols, exact executed script bytes,
reports and the detection script/report remain intact in `v2/`.

V3 was frozen before recomputation. It additionally removes any query whose
0.1 micrometer outward probe enters another collider, with 1e-10 m halfspace
tolerance. All retained probes are checked exterior. Same-oriented coincident
faces are assigned to the lower geom ID. Centroid overlap removal still gives
density-dependent exposed area rather than exact polygon-clipped union area.

The actual collision union and disabled visual object can have distinct
boundaries. Thus even corrected collision-union queries are a different
measurement domain. At coarse density, collision sample area is 36,499 mm² for
apple, 47,658 mm² for milk box and 89,080 mm² for mug; corresponding visual areas
are 24,304, 35,322 and 86,473 mm². On one fixed group/object, area-weighted
collision-to-visual distance means are 4.42, 2.76 and 2.12 mm, respectively.
Reverse-distance maxima are 36.47, 27.35 and 9.99 mm. These discrepancies show
why visual and collision boundaries cannot be treated as interchangeable.
They establish a geometric difference, not its cause: narrow decomposition
gaps, hidden cavity walls or other specific mechanisms were not independently
confirmed. The visual-to-union one-way reference diagnostic remains optional
local scratch; it is not part of the selected publication files.

## Measured behavior and limits

At the 2 mm band and coarse density, whole-hand **unsigned** proximity IoU is
0.195–0.290 on collision-union queries and 0.243–0.273 on visual queries across
six cases. Occupied-or-near IoU is higher (0.380–0.449 and 0.441–0.512); occupancy
must not be conflated with unsigned surface correspondence. Finer first-group
visual IoU changes are -0.00798 (apple), +0.000000605 (milk box) and +0.000145
(mug). This is stability across two tested densities, not proven convergence.

Per-finger differences persist on the source visual surface. For example, the
first mug group has native index 1,601.7 mm² near area, Shadow 1,444.8 mm² and
shared 275.1 mm² at the 2 mm band. Its native little finger 660.3 mm² and target
1,124.4 mm² share 160.7 mm². Apple little-finger shared area is about 39 mm² in
both groups despite native 203/214 mm² and target 305/311 mm². These are changed
geometric relationships, not paired physical contacts or evidence that the
original native grasp was good. Poor proposals and penetration are retained.

The six cases come from existing development families with one shared MANO
shape. They do not establish generalization, physical transfer, original HUG
system equivalence, grasp-family preservation or an acceptance criterion.

## Verification and figure

`geometry_sanity.json` verifies analytic primitive distances and closest points
under a nonidentity rigid transform, exact box-hull distance agreement, closed
cube winding, and opposite edge orientation for every saved native mesh. The
accelerated exact triangle query matches the previous NumPy routine within
numerical roundoff. `cube_baseline_regression.json` reproduces the old 3,750
2 mm cube-face grid's 2/5 mm occupied-or-near values exactly.

Every phase guards `mj_step`, `mj_step1`, `mj_step2` before model loads. Simulation
time stays zero; initial object qpos/world pose and all nominal qpos stay
unchanged. Source native, command, scene, frame, licensed MANO, prior inventory
and FK hashes are verified. Licensed assets are read locally and never copied.
Each full report's script SHA matches the exact executed file bytes; v4 also
records its unchanged v3 geometry-query library SHA. The original published
cube reports, mapping code, labels, controller and historical runtime artifacts
were not modified.

`hug_other_objects_native_shadow.png` is a static scientific figure of the
first fixed group per object. Each native/target pair has identical object axes,
metric scale, orthographic camera and limits. Gray objects show actual active
convex hulls; rendering individual hulls is not an exact clipped exterior union.
Forearm/wrist carriers are hidden in the figure and included in measurements.
`figure_metadata.json` records its provenance and display limits.

## Reproduction

Use the preserved Python/runtime import path documented in
`docs/HUG_SINGLE_EVAL_2026-10-13.md`. No package install, model inference or network
is needed. Copy each phase's script and frozen protocol into a **fresh declared
output directory** before execution; preserve existing report files. The v3
script supports `--out` (the destination must contain `protocol_v3.json`). V4
expects the v3 query library beside it and its protocol below
`visual_reference_surfaces/`. The figure script expects the v3 protocol beside
it. Keep v2 phase files distinct rather than rerunning them over final results.

The compressed query arrays are optional diagnostic derivatives, not newly
collected evaluation episodes; full JSON reports and source hashes are enough
to assess results and reproduce them. Cache and scratch files are excluded from
`manifest.json`. Parent review selects final publication files.
