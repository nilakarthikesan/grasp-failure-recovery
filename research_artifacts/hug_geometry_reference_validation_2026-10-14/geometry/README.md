# HUG object geometry cause investigation, 14 October 2026

This static diagnostic investigates the **three existing unique apple, milk-box
and mug assets**, and verifies their placements in both previously captured
observation groups. It reads the original source OBJ/MJCF files and saved compiled
MuJoCo models. No proposal inference, pose optimization, physics integration,
evaluation episode/data/labels, predictor fit, asset rebuilding or production
changes occurred. Grasp quality is not used to select or reject any input.

## What the discrepancy means

The source OBJ meshes and saved compiled meshes agree after applying their actual
geom frames. The worst bidirectional nearest-vertex errors are **4.15 nm for
apple, 7.36 nm for milk box and 7.69 nm for mug**. All declared object mesh scales
are one. Both observation groups' compiled per-geom object-frame vertices agree
within 1e-12 m. Original visual and collision parts remain in the same local
frame; MuJoCo's internal mesh-centering and principal-axis transforms are accounted
for by the actual compiled geom frames. A scale, frame or placement bug in the
tested object import was not established.

All 36 sim-asset files for these three objects match their preexisting download
manifests, pinned to HUG / 1M-HuGs dataset revision
`56ea3f100b1a4ec38bb35e986c0dd14d38977283`. The source properties declare convex
decomposition build parameters (`coacd_threshold=0.01`, `max_convex_hull=15`,
`alpha_pct=2.0`, `offset_pct=0.05`). Those metadata establish the intended
approximation, but the exact builder implementation/version has not been checked
and is not assigned responsibility for a particular measured gap.

The **collision union boundary includes internal slit walls** absent from the
visual object. On the fixed 2 mm area-density query set:

- Apple's worst collider-to-visual distance is 36.47 mm. Its query lies inside
  the closed, oriented visual mesh (winding approximately 1). Its outward ray
  enters the next compiled collider after **3.90862 micrometers**; the same query
  and ray enter the corresponding raw-source hull after **3.90928 micrometers**.
- Milk box's worst distance is 27.35 mm, also inside the visual mesh. The next
  compiled/raw hull entry is **3.51432/3.51777 micrometers**.
- Mug's worst distance is 9.99 mm. This query is outside the visual object
  (winding approximately zero), and its outward ray does not hit another piece.
  The slice shows an actual convex approximation of the mug's lower interior
  profile. Other mug regions also contain narrow inter-piece gaps.

Thus the apple/milk maxima **do not mean the external silhouette is displaced by
27–36 mm**. These witnesses sit on local interfaces separated by micrometer gaps
already present in the original published asset pieces. The source/compiled
rounding differences are nanometers. Internal piece surfaces and visible object
surfaces are distinct measurement domains.

The old v3 outward probe is 0.1 micrometer. It correctly excludes abutting
interfaces under its declared rule, while retaining boundaries of genuine gaps
larger than that probe. At a 10 micrometer probe, 12,365.6 / 13,130.5 / 4,271.9 mm²
of retained apple/milk/mug sampled area enters another collider. Most of these
areas are also inside the closed visual object. This is **point-probe sensitivity**,
not an instruction to enlarge the exclusion threshold or change physical geometry.
Queries can leave a thin neighboring collider again, so finite endpoint probe
membership is not guaranteed monotone with probe length.

Repeating at 1 mm area density reproduces mean collider-to-visual distances to
within 0.008 mm and retains the same interpretation. Density bounds child
triangle area, not edge spacing. Full reports retain weights, inside/outside
characterization, all 12 worst-distance witnesses per object/density, raw/source
correspondences and per-geom transforms.

## What remains unconfirmed

Closed and consistently oriented exact-welded triangle topology permits the
reported winding test, but exhaustive geometric self-intersection checks were
not performed. Global outside-air connectivity and accessibility of gaps to
finite hand collision geoms are **not** established by local ray re-entry or
outward probes. Neither are contact forces, outcome sensitivity, mapping
acceptance or equivalence to the original HUG system.

The visual-to-per-collider distance is separately labeled. For points inside
overlapping solid pieces it need not equal distance to the exterior union
boundary. Do not treat it as a two-sided Hausdorff distance, combine it with
visual-to-union **occupied volume** distances, or infer inside/outside from its
unsigned magnitude. The large reverse-boundary discrepancies remain genuine
geometric measurements with qualified meaning.

Aggregate `inside_visual_area_mm2` includes shallow near-surface insets as well
as internal slit walls. It must not be called exclusively gap-wall area.
`collision_query_inside_raw_source_union_*` uses 1e-10 m halfspace tolerance and
includes the corresponding own raw source hull. Because compiled/source vertex
rounding is 4–8 nm, those membership counts can reflect boundary rounding; they
are not evidence of deep penetration or a compiler error. The local gap claim
uses the far-from-visual witness winding and the positive raw/compiled next-hull
ray intervals, which are much larger than that rounding.

No asset correction is justified from this evidence alone. Faithful hand
retargeting should be measured on the common visible object reference, while
the simulator's collision approximation is reported separately. Previous
finger-surface discrepancies persisted on that visible reference and are not
explained away by the object decomposition finding.

## Verification and reproduction

`protocol.json` was frozen before metrics (SHA-256
`bcb4bf57eadea5b57a8476beceecebd1410503ccb4a4f6e5aacb3323f1f4228e`).
`geometry_cause_report.json` records the exact executed script and prior query
library hashes, all 63 protected inputs before/after, and zero physics steps,
zero integration attempts, zero simulation time and unchanged qpos/object poses.
Guards for `mj_step`, `mj_step1`, `mj_step2` were installed before model loads and
self-tested. `ray_sanity.json` verifies six analytic ray cases, including a
nonidentity rotation/translation. Download provenance uses a separate read-only
verifier and includes the exact manifest/script hashes.

`object_geometry_causes.png` is a static scientific figure, visually reviewed.
It shows deterministically subsampled boundary queries colored by distance,
slices through worst witnesses, and probe sensitivity. Slice lines are individual
piece facets, including buried facets, not a polygon-clipped exterior union.
The figure does not define metrics. `figure_metadata.json` records provenance,
display stride and slice bases.

Use `sh reproduce.sh /private/tmp/a-new-output-directory` with the preserved
runtime and local source assets. It creates fresh output and invokes only static
diagnostics. No package install/network is needed. Compressed mesh/query arrays
and caches remain local scratch and are excluded from the publication manifest;
reproduction regenerates them for the figure. No raw source or licensed MANO
assets are copied to this bundle. Original reports remain unchanged.
