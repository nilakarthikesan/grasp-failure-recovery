# Static signed HUG hand/object profiles, 14 October 2026

This phase reads six existing seed-0 apple, milk-box and mug proposals and the
already computed **stored current, shipped reference and metric-static reference**
target poses. It measures their hand geometry against the same actual compiled
disabled visual object triangles. No new proposal inference, retargeting,
optimization, physics, evaluation episodes/labels, training or production changes
occurred. Poor original proposals and all fixed cases remain included.

## Signed geometry and sampling

The separate object investigation verifies closed, consistently oriented
exact-welded visual topology for all three objects. Exact nearest visual triangle
distance is negative when whole-object absolute winding exceeds 0.5. Queries
within 0.1 micrometer unsigned distance become boundary zero. Queries strictly
outside the visual AABB cannot lie inside the closed triangle surface, so their
winding calculation is skipped; unsigned distance remains exact.

Native original mesh vertices are **unweighted count profiles**. Native
triangle-centroid profiles are separately weighted by source triangle area.
Anatomical labels reuse the previously verified MANO skinning groups and
triangle-majority/tie rule; labels are proxies. The supplied closed native mesh
includes its wrist cap, without separately closing finger subsets.

Target surfaces use the actual compiled per-geom convex hulls and exact boxes.
Spheres and cylinders have a declared 0.5 mm conservative chord approximation.
Their query points are projected back onto the exact analytic primitive surface;
their **weights still estimate area from planar tessellation facets**. Those
weights are not exact curved-surface area. Sphere facet-plane distances certify
the bound. This report does not substitute rendered robot meshes for colliders.

At all six cases, palm/finger centroid subdivisions bound triangle area at
16 mm² (4 mm area density). The first fixed group per object repeats all three
targets and native surface at 4 mm² (2 mm area density). This is an area bound,
not uniform spacing or continuum convergence. Carriers are sampled separately
at fixed **16 mm density**; primary native/target comparisons use native MANO
versus combined target palm/fingers. The heterogeneous full target hand and its
carriers remain explicit, rather than silently removing large geometry.

Target per-geom points inside another collider, outward-probe occluded points,
and same-oriented coincident duplicates are excluded and their source sampled
areas recorded per geom. Membership tolerance is 1e-8 m; outward probe length is
1e-7 m. This approximates the exposed union at finite sample sites, including
possible narrow gap/cavity walls. It does not certify global external-air access.

## Measurements and interpretation

`signed_hand_visual_profiles.json` contains whole-shape and part-specific signed
quantiles, binned count/area distributions, negative sample counts, negative area
fractions, conditional mean depth and **finite sampled maximum negative depth**.
Native vertex counts never become area fractions. Targets have different
morphology, so these are independent shape profiles, **not matched-point errors**.
The source domain retains all original MANO triangles, including wrist-cap and
any self-overlap surfaces; the target domain omits sampled buried collider
surfaces. Differences in profile fractions or miss counts therefore also depend
on these declared surface domains and cannot automatically rank mapping fidelity.

For each finger, unsigned-proximity miss means every retained sample is farther
than the declared 1/2/5 mm band in unsigned distance. Exterior-gap miss instead
requires every retained sample's signed distance to be positive and exceed the
band. Deep penetration without a near-boundary sample therefore remains distinct
from a separated exterior finger. Neither definition identifies a physical
contact or task outcome; an empty sample part is separately counted.

Finite sampled maxima can miss deeper penetration. Zero negative samples do
not prove absence of continuous intersection. The frozen protocol's lower-bound
phrase should be read only within its declared surface/sign assumptions:
native self-intersections are not exhaustively excluded, target union exposure
uses tolerances, and true exterior-union continuous maxima are not certified.
The supplied object topology permits winding, but exhaustive geometric
self-intersection checks on those surfaces were not performed either.

Low penetration, successful lifting or low miss count is **not an acceptance
criterion**. A faithful mapping can retain a poor source proposal. Comparing
profile changes is one part of the mapping argument; it does not by itself
establish grasp-family preservation, physical transfer or a valid training label.

Across the six 4 mm area-density native profiles, sampled maximum depth ranges
from **1.87 to 24.28 mm**, and negative native triangle-area fractions range from
**2.99% to 6.84%**. Thus penetration exists in the retained source proposals before
robot translation. For first apple, native/current/shipped/metric sampled depths
are **16.84 / 13.93 / 21.64 / 13.65 mm**, with corresponding independent negative
area fractions **3.89% / 4.17% / 11.45% / 2.99%**. First-group 2 mm repeats retain
these profile distinctions, but the shipped apple sampled maximum increases to
22.67 mm, illustrating finite sampling. For second milk-box, metric-static depth
is 27.37 mm versus native 24.28 mm; for mug all four profiles are around 2 mm.
These examples describe geometric changes and source imperfections, not a
mapping acceptance decision or an automatic ordering of retargeters.

## Frozen versions, bounds and verification

V1 is preserved in `v1/` as an **incomplete pilot**, with its exact script,
protocol and log. Its strict comparison mistakenly treated a rounded baseline
range field as full source authority and stopped before a report. The metric
target remained valid against the original URDF. V2 was frozen before rerunning
metrics (SHA-256 `1990bc9d74facf7a4cfec770150a19ab87621a47c7eeb6ba42df990bf6fd4c4a`).
Its amendment uses carriers at coarser density to bound runtime, exact AABB
winding rejection, conservative primitive chord certification, and **direct
full-precision source URDF XML joint ranges**. All target values/root poses stay
unchanged. Captured scene decimal-range excess remains reported separately;
these static configurations do not demonstrate production preflight rejection.

The completed report records script/protocol/library/input hashes before/after,
guard self-tests and zero actual integration calls. All models remain at time
zero with unchanged initial object qpos/world pose and original supplied target
qpos/root. `profile_sanity.json` checks known positive/negative/boundary box signs,
strict membership, same/opposite-normal interfaces, chord certificates and
weighted distributions. The original geometry evidence bundle remains unchanged.

`signed_profiles_comparison.png` is a scientific static figure. Filled markers
show all fixed 4 mm density cases; hollow markers show predeclared 2 mm repeats.
Carriers are excluded from this particular display but retained in the reports.
The figure neither defines metrics nor labels simulation outcomes.

## Reproduction and source dependencies

Use the preserved local Python/runtime import path from earlier HUG audit
bundles. Stage original geometry-cause and full reference comparison JSON bytes
at the paths recorded in `protocol.json`; when published as gzip, decompress
without altering content and verify recorded SHA-256. The baseline report is
consumed, not rerun or optimized. Local original scene/native/frame/MANO assets
are required and are read only. No package installation or network is needed.

Copy the preserved protocol and script into a fresh declared output directory,
then invoke `profile_signed_geometry.py --out <fresh-directory>`. Run
`check_profile_sanity.py` and `plot_signed_profiles.py` beside the new report.
Do not overwrite historical outputs. The full report may be stored as a
lossless gzip plus compact summary for publication; decompressed report bytes
must retain their recorded hash. No mesh/query arrays or licensed assets are
included in this phase's publication files.
