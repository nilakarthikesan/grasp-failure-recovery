# HUG static sampling uncertainty on saved poses

14 October 2026. This phase measures sampling uncertainty on the **same six
previously selected seed-0 noncube development poses**: two apple, two milk-box,
two mug cases. It repeats the first observation group for each object at the
previously declared finer density. These are nine geometric quadrature reports,
not nine independent proposals or new evaluation episodes. The larger saved HUG
inventory has 32 distinct proposals/four objects, all with one native MANO shape;
this surface phase does not claim to cover all 32 poses or the cube continuum.

No new inference, optimizer, physics integration, outcome/label, dataset collection,
predictor training, source quality selection or production change occurred.
Original native files, saved nominal Shadow targets, object geometry, transforms,
licensed/runtime assets and 110 protected production/model/manuscript files were
verified unchanged. All measurements use time-zero forward kinematics.

## What was measured

The fixed query domain is the actual compiled disabled visual-mesh triangle table
in the unchanged object frame. This is distinct from the convex collision-piece
union and physical contact geometry. Original degenerate-facet exclusions are
retained and counted. For each source visual facet, the exact prior midpoint
subdivision order is rebuilt. Centroids and area weights reproduce the hash-bound
v4 cached arrays **exactly**; the child triangles cover the undeleted facet.

Each child triangle gets a radius equal to the maximum distance from its centroid
to its three vertices. Convexity makes this a valid coverage radius in real
arithmetic: every point of the triangle lies within that ball. Child area alone
cannot bound this radius. Distances are queried against the native anatomical
surface patches and current compatible Shadow collider boundary surfaces. These
are fixed geometric sets, so their unsigned distances are 1-Lipschitz.

For centroid distance `d`, radius `r`, band `tau`, and numerical allowance `eps`,
the distance throughout that cell is contained in
`[max(0, d-r-eps), d+r+eps]`, **conditional on eps bounding numerical error**.
A cell is definitely near if its upper distance is at most `tau`; it is possibly
near if its lower distance is at most `tau`. Summing complete definite/possible
cell areas gives lower/upper continuum proximity-area bounds. Shared, union,
lost and added masks use conservative set combinations, rather than comparing
only centroid decisions. IoU and retention ratios are bounded only when the
relevant area denominator is definitely nonzero; otherwise bounds remain null
with an explicit reason. Independent anatomical-part masks may overlap and their
areas must not be summed as a whole-hand partition.

The **explicit assumed** uniform numerical allowances are 1, 10 and 100 nm. The
primary report uses 10 nm. These are arithmetic sensitivity assumptions, **not a
formal global floating-point error certificate inferred from finite tests**.
The radius method supplies a mathematical geometry coverage bound; it does not
supply a missing arithmetic proof. Neither normal angles, winding/occupancy,
penetration preservation, collider-versus-visual discrepancies, anatomical
correspondence, actual contact or outcome labels inherit this distance bound.

Native forearm/wrist carrier reference surfaces are **unavailable**. Mathematical
empty-set distance entries are infinite, but scientifically these are not native
anatomical absence observations. Both carrier entries are excluded from the
six-part strict gate. Whole-hand diagnostics include all compatible mapped
colliders, including carriers, and disclose this larger domain. Mapped distances
are minima over per-collider boundary surfaces, including buried interfaces in
collider overlaps; they are not exact exterior-union signed distances.

## Findings

At the 2 mm band and coarse density, median cell radii range from 1.292 to
1.487 mm; maximum radii reach 5.627 mm. For the three finer repeats median radii
fall to 0.649–0.760 mm, while the maximum still reaches 4.551 mm. This confirms
why the prior “2 mm area density” was not a 2 mm edge-spacing or diameter guarantee.

At the 2 mm band, 8.611–16.821% of the entire coarse visual-query surface area has
an unresolved whole-hand native/mapped relationship under the primary allowance.
The three finer repeats reduce that proportion to 5.121–9.779%. Whole-hand IoU
centroid estimates are relatively stable (first-group fine-minus-coarse changes:
apple -0.007978, milk box +0.000000605, mug +0.000145), but conservative coarse
IoU upper bounds remain 1.0. **Stable estimates are not narrow continuum bounds.**
Finer whole-hand IoU intervals at 2 mm are apple [0.0924, 0.5973], milk box
[0.0633, 0.8005], mug [0.0783, 0.6432].

Despite the ambiguous area, there are definite same-part lost/added proximity
cells in **all six coarse cases and all three finer repeats** under the
independently frozen strict-preservation rule. At 2 mm, middle/ring/little have
resolved changes in all coarse cases; each mug case also has palm/index/thumb
changes. The finer milk-box repeat resolves a thumb change that coarse cells left
uncertain. All per-band sets of parts with definite changes are unchanged across
the 1/10/100 nm allowances. This is arithmetic-assumption sensitivity, not an
unconditional numerical theorem.

For example, a 1 mm apple-A middle-finger lost-proximity cell has area
1.038163 mm² and radius 0.922061 mm. Under 10 nm allowance the native distance
over that complete cell is at most 0.936781 mm, while the mapped distance is at
least 1.909627 mm. Its centroid, source face/cell indices, distance intervals,
area and source/protocol hashes are in `strict_preservation_witnesses.json`.
There are 178 selected witness cells: the largest-area definite cell for each
required part/band/relationship/case/density, with deterministic earliest-index
ties. These are auditable counterexamples, not optimized calibration targets.

The rule therefore reports **FAIL of strict declared-band anatomical relationship
preservation, conditional on the declared numerical error assumptions**. This
is not a numerical-conversion failure, a physical task failure or a statement
that every approximate retargeter is unusable. No unsupported approximate
distortion budget is introduced. Zero lower-bound loss for a part does not make
it pass: upper-bound mismatch and unresolved orientation/domain coverage remain.

## Protocol history and checks

`protocol.json` is operational V2, frozen before any complete case report.
`v1_aborted/` preserves the exact uncompleted pilot. That pilot incorrectly
required full new distances to match every finite old broad-phase cache value.
Old 5 mm broad-phase queries can leave an incomplete finite minimum farther than
5 mm when an analytic primitive was queried and a closer mesh was skipped. V2
compares distances where the old value is within 5 mm and requires identical
1/2/5 mm band decisions for every part/query; all pass. Full current distances
are necessary for cells whose centroid is outside a band while another cell
point might be inside it. No pose, band, selection or scientific gate changed.
The final `sampling_uncertainty.json` keeps stable report-format schema V1;
its authoritative protocol SHA points to operational V2.

Three integration guard self-tests pass; actual guarded-call count and physics
steps are zero. Native/source/FK/library/cache/protected hashes, nominal qpos,
initial object qpos/world pose and time-zero invariants pass. Six manufactured
logic tests pass, including a rotated skinny triangle, barycentric coverage,
fixed-point Lipschitz distances, analytic strip regions, empty denominators and
numerical-boundary ambiguity. Independent review checked 100 deterministic
triangle/strip cases by separate polygon clipping and found all six area
operations and ratios contained the analytic results (maximum rounding residual
8.89e-16 mm²). These finite tests validate implementation logic; they do not
establish a global arithmetic error bound or statistical confidence interval.

## Files and reproduction

`sampling_uncertainty.json` contains all per-case/per-density/per-part/per-band
area intervals and three numerical-allowance sensitivities. `summary.json` gives
primary intervals and density changes. `strict_preservation_witnesses.json` binds
the frozen independent acceptance rule and auditable cell witnesses.
`sampling_uncertainty.png` is a scientific diagnostic, with its report hashes
in `figure_metadata.json`. All publication files are listed in `manifest.json`.
No cache arrays, MANO assets, scene binaries or new eval records are published.

Run `reproduce.sh` to a new temporary output directory using the preserved local
runtime and original hash-bound visual query caches. If those caches are absent,
rebuild them with the original v4 `audit_visual_surfaces.py` and frozen library
and protocols in an isolated temporary directory; exact hashes are checked and
mismatches stop reproduction. The ignored original inputs/runtime assets must
remain available; the publication does not redistribute them. Reproduction
writes only static diagnostics under its output directory. Deterministic query
sites are quadrature, not independent statistical samples.
