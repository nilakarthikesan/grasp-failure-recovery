# Collision geometry contract review, 15 October 2026

The saved production scenes and historical collider extraction agree on the relevant geometry. No missing convexification, collision-mask exclusion, nonzero contact margin, or compiled-transform defect was found in the six existing HUG scenes. This result does not establish faithful HUG-to-Shadow surface mapping.

## Actual compiled scenes

Independent inspection of the six hash-bound MJB scenes used six static `mj_forward` calls and zero integration steps or new hand/object distance queries. Each scene has 45 collision-enabled Shadow geoms. Apple, milk box and mug use respectively 2, 7 and 15 object collision meshes, plus one disabled visual mesh. All hand/object pairs pass bitmask, exclusion, welded-body and direct-parent filtering. No explicit pairs, exclusions, flexes or plugins occur. Geom margins/gaps, override margin and option enable/disable flags are zero. Compiled geom poses agree with body transforms and local geom poses within 1e-12 m.

For every compiled mesh, full-vertex convex-hull extrema appear in MuJoCo's compiled support graph. The largest all-vertex residual against the support-graph hull is 2.78e-17 m; maximum hull volume difference is 3.05e-20 m³. The old SciPy hull extraction therefore has no identified reduced-hull mismatch in these scenes. Exact triangulations/normals may differ on coplanar faces without changing the convex solid.

The captured and installed runtime is MuJoCo 3.3.7. Its collision support code uses the compiled graph vertex indices, or all vertices when no graph exists. [Version-pinned implementation](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_convex.c).

MuJoCo treats mesh geoms as convex hulls for collision, even when their rendered triangles are nonconvex; collision filtering is pairwise and includes masks and body relationships. [MuJoCo 3.3.7 collision detection](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html#collision-detection).

Compilation recenters/reorients mesh assets and correspondingly adjusts geom transforms. The historical loader correctly starts from compiled vertices and actual geom transforms. In 3.3.7, contacts are detected below `margin` and force-active below `margin-gap`; these are both zero here. Current stable documentation uses different wording/semantics, so this audit pins the installed version. [MuJoCo 3.3.7 geom reference](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#body-geom).

## What the historical metric actually measures

The frozen sampling code explicitly labels its mapped set as the union of individual compatible collider boundary surfaces, including buried interfaces. Its distance is `min(abs(component signed distance))` within each anatomical proxy part. This is a valid distance to that declared fixed surface set and supports the existing conditional cell-radius inequalities. It is not generally distance to the exposed boundary of the solid union.

For a point strictly outside the whole union, the global minimum component distance is already the distance to the union boundary. Inside overlapping components, it can reach a buried interface first. Per-part queries can also select buried interfaces even when the query is outside the full hand, because the nearest component globally may belong to another part. Removing such interfaces changes the measured set and requires a separately versioned diagnostic; it does not retroactively invalidate or erase nine historical strict failures.

The latest frozen nine-comparison audit samples the disabled visual object mesh, using the same reference for native and mapped hands. It does not measure the original collision-union sampling surface. The historical orientation routine additionally requires the hand query point outside the hand, a unique normal/part, no buried chosen point, and the chosen hand point outside the visual object. A `closest_inside_or_on_visual` refusal is a pointwise eligibility decision, not proof that the collider was represented incorrectly. Mesh-edge ties and anatomical proxy-part identity are additional methodological assumptions, not universal physics requirements.

MuJoCo generates contacts between component geom pairs. An exposed-union boundary describes geometric occupancy/accessibility; it is not a reconstruction of every pairwise contact, especially during penetration. Contacts and local point normals should retain separate names and provenance.

## Requirements for the proposed exposure diagnostic

Use exact outside-whole-union minimum distance only where strict exterior is established. For inside-union distance, the maximum individual-component penetration depth is a conservative lower bound; exposed ray-exit witnesses give upper bounds, not generally the exact minimum. For an anatomical part, the old minimum component-boundary distance is a lower bound to the part's exposed subset; an upper witness must demonstrably belong to that part and remain on the whole-union boundary. Missing upper witnesses mean unavailable/unresolved, not infinity as a proven empty set or a fabricated pass.

Touching components need particular care: a point on two shared box faces can be interior to their union even though each component signed distance is zero. `any(signed_distance < 0)` alone does not certify strict union exterior or a zero exposed-boundary distance at such a seam. A finite outward probe can jump over a thin component or gap; analytic ray intervals and explicit coincidence/ambiguity labels are needed. Keep finite directions, numerical allowances and unresolved intervals visible. Do not claim global air connectivity, finite-hand accessibility, or simulator contact equivalence from ray witnesses.

The three object geometry identities are repeated across the paired observations; different seeds or observation groups are not unseen objects. The compiled receipt includes object, geometry, observation and inference-input hashes for leakage grouping.

Machine evidence is in `compiled_contract.json`; historical code/source hashes are recorded in `source_bindings.json`. No production source, saved proposal, historical metric or acceptance threshold was changed. Proposed query implementation/tests and execution protocol still require separate review before new six-case queries.
