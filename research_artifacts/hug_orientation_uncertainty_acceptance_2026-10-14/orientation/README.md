# Existing HUG object-facing geometric orientation

14 October 2026. Frozen operational V2, six existing seed-0 noncube poses,
one MANO shape; the current original saved targets only. Nine resolutions:
2 mm triangle area-density for all six, plus 1 mm repeats on the first apple,
milk-box and mug observations. The cube's previous query family stays separate.

This closes the previously unimplemented object-facing normal measurement,
within the explicitly sampled scope. It does not certify continuous normal
preservation, acceptable physical contacts, grasp quality or success/failure.
No new HUG inference, optimization, evaluation episode, dynamics, labels or
training occurred. All static transforms and target joints retain their original
values. Protected production/model/manuscript hashes remain unchanged.

## Definitions and denominator rules

At the same visible-object centroid query, object-facing compatibility is
`-n_hand · n_object`. Positive is facing, negative is away. MANO and actual
visible-object geometric facet normals are globally oriented outward by signed
volume, conditional on closed consistently oriented source topology. Target
mesh normals come from actual per-geom convex hulls, and analytic primitives
use their exact SDF-query normal. These are encoded geometric normals, not
measured contact-force normals or tissue ground truth.

The primary paired region is native and mapped exterior proximity with the
same nearest anatomical proxy label. Queries inside either hand, lost/new
regions, different labels, unavailable carriers and nearest points inside or on
the object are preserved as separate denominators. Native and mapped independent
part areas are not summed as a whole-hand partition.

Point-level competing nearest normals, labels or locations invalidate paired
normal claims. The frozen 10 nm distance/point tolerance, 1e-6 normal-vector
comparison and dot deadzone, plus 0.1 micrometer visual sign boundary, are
numerical diagnostic safeguards, not scientific mapping acceptance margins.
Coplanar same-label same-point triangle ties keep a unique normal. Sharp mesh
facets, analytic edge/corner/rim/axis ambiguity, and target closest points buried
inside another collider are excluded with overlapping cause counts and a
separate union-excluded denominator. No excluded region becomes a perfect match.

Closest-point visual membership uses unsigned distance plus absolute winding,
with exact outside-AABB rejection. Closed oriented topology is verified in the
hash-bound prior geometry report; exhaustive self-intersection tests are not
available, so the inside interpretation remains conditional.

## Evidence and uncertainty

`summary.json` retains all band denominators, per-part orientation summaries,
strict sampled facing-sign witnesses and observed 2-to-1 mm density changes.
`orientation_report_full.json.gz` preserves the full exact report including
all confusion cells and exclusion categories; its decompressed hash is recorded
in the summary/verification. No raw query arrays, licensed hand assets,
scene binaries or duplicate per-case JSONs are published.

A reliable same-part sign reversal at a fixed sampled site, with both hand
closest points outside the visible object, can refute exact pointwise
orientation preservation on these encoded surfaces. Both facing→away and
away→facing changes matter: repairing an original poor relation is a change,
not faithful preservation. The absence of a sampled reversal does not certify
unsampled orientations. Density changes are observed sensitivity only.

Distance-field 1-Lipschitz bounds apply to proximity masks under the sibling
sampling protocol; they do not bound a discontinuous nearest-facet normal field.
This phase supplies no continuous normal-area or angular-error certificate.
A smaller mean angle, increased compatibility or a future successful lift
cannot by itself prove that all native contact relationships were retained.

## Measured findings

At the 2 mm proximity band and 2 mm area-density, the valid same-part region
with both closest points outside the visible object has area-weighted mean
native-to-mapped hand-normal angles of 19.537–28.720 degrees across the six cases.
These conditional means are measurements, not an acceptable-error threshold.
Their underlying eligible outside-object areas are 36.648–206.876 mm²;
all candidate/excluded, inside-object, lost/new and different-label denominators
remain in the full report.

The finer first-apple query reveals four reliable sampled facing→away sites
and six away→facing sites at that band, whereas its coarse query saw zero and
two. The finer first-milk query reveals one facing→away and three away→facing
sites. Source/target points and normals for strongest witnesses are retained.
Their centroid area weights do not establish continuous changed-normal area.
The first-mug repeats contain no sampled same-part sign flips at this band;
that absence is not a global orientation-preservation pass.

First-apple conditional mean angle changes from 20.426 to 25.677 degrees with
finer density; milk changes from 28.720 to 26.890; mug from 19.537 to 19.128.
This observed sensitivity and newly revealed witnesses prevent claiming normal
convergence from the coarse averages alone.

## Verification and history

`sanity_report.json` checks a rotated/translated manufactured box: distance,
parallel/antiparallel/orthogonal normal signs, coplanar and sharp-corner ties,
and visual inside/outside/boundary classification. The independent candidate
review checks the amended native-near/target-far candidate search. The independent
witness review replays the strongest fine-apple middle and ring sign witnesses
against original native and actual target geometry: all points, normals,
labels, distance, eligibility and outside-object classification agree. It uses
the same frozen query implementations, so it is an independent replay rather
than an independent closest-point algorithm.

V1 was frozen and analytically checked before saved-case metrics. Independent
review found its fixed 5 mm mesh tie-search AABB bound incomplete for sites near
only the native hand. V2 uses each site's exact mapped minimum plus tie tolerance,
without changing scientific bands, normal definitions or eligibility. V1 sources
and sanity results are preserved in `v1_before_metric_run/`. No saved-case
orientation metric was run under V1.

A first V2 run stopped when the sibling mug fine-resolution export was not yet
ready. The incomplete dependency run is preserved separately in
`v2_readiness_partial/`; the completed report reruns the same unchanged V2
protocol after all nine exports exist and does not pool partial outputs.

`reproduce.sh` requires original saved inputs/runtime and old v4 query caches,
then sibling sampling ephemeral exports. Those fields carry original cache,
native source, frozen sibling protocol/library and NPZ hashes. This script
independently regresses native distances and original mapped ≤5 mm distances,
all occupancy, original targets and time-zero/object invariants. Reproduction
outputs and timing may differ in serialization but definitions are frozen.
