# HUG static mapping acceptance definition — frozen scoped rule

14 October 2026. The operational rule was frozen at 12:45:42 UTC before reading
new orientation or cell-uncertainty reports. Its source is
`acceptance_rule.json`, SHA-256
`87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4`.
This explanation interprets that rule; the frozen JSON controls the applied
criterion. This definition does not generate evaluation data or authorize
simulation steps. Existing HUG poses are development cases, not a fresh
validation set. The earlier proposal remains separate protocol-design history.

## Applied result on existing development grasps

Numerical decoding/conversion passes within recorded scope. All 32 native 3D
exports reconstruct exactly; 32-camera reprojection is not claimed. Strict
declared-band same-part preservation FAILS conditionally in all six detailed
cases and three finer repeats, with 178 cell witnesses and reliable sampled
orientation reversals. Continuous physical-contact equivalence remains
UNRESOLVED; failure-label preservation, approximate predictor adequacy and
independent generalization remain NOT_TESTED. A mapping status is not a task
outcome label.

[The evidence
bundle](../research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/README.md)
and [machine-readable rule
application](../research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_application.json)
preserve source hashes, scopes, conditional numerical assumptions and all
case/part results. These measurements use existing saved grasps only. No new
eval data or production change occurs.

## What “acceptable” means here

Use three explicit claims, each with its own status:

1. **Numerically valid conversion:** we correctly decode the saved HUG proposal,
   place the declared hand anchor, associate the intended joints, and reproduce
   the mapped Shadow pose within justified numerical precision.
2. **Surface-relationship preservation at declared 1/2/5 mm bands:** native and
   mapped hands preserve the declared same-finger/palm proximity and
   object-facing direction states over a specified visible-object surface
   domain. Whole-hand occupancy and penetration receive separate reports and
   unresolved continuous-preservation statuses. This is a deliberately strict geometric contract;
   these inherited bands describe the claim, not a discovered universal contact
   tolerance. A hand with a different morphology can fail this contract despite
   correct numerical conversion.
3. **Failure-label preservation:** the translation does not change whether the
   native proposal would fail under a stated execution protocol. This remains
   NOT_TESTED in this phase; static geometry alone cannot establish it.

A mapping may therefore be numerically valid and have measured approximate
geometry, while failing a contact-relationship claim. Do not collapse those
statuses into one green “validated” badge. “Usable as a documented mapped-hand
simulation pipeline” is a different project claim from “faithfully predicts the
native generator's own failures.” This distinction must be retained in later
source-diversity experiments.

The evidence-state wording should be: **numerically accepted within checked
scope; implemented approximate mapped pipeline with measured distortion;
strict surface-relationship preservation PASS/FAIL/UNRESOLVED under the frozen
rule; failure-prediction adequacy and outcome equivalence untested**. “Approximate
pipeline” is an implementation description, not an adequacy certificate.

## Status rules

Each declared criterion returns PASS, FAIL or UNRESOLVED. Failure-label and
independent-generalization claims additionally use NOT_TESTED.

- PASS means all required evidence covers the entire declared domain and
  certifies the property at the stated resolution and numerical allowance.
- FAIL means at least one certified counterexample contradicts that property.
  A counterexample can refute a universal claim even if other regions are
  unresolved. Publish the affected case/part/band and a geometric witness.
- UNRESOLVED means no certified contradiction was found, but a required
  relationship is unavailable, ambiguous, partially sampled or not bounded.
- NOT_TESTED means the required experiment is outside this static phase.

For a conjunction: FAIL dominates; otherwise any UNRESOLVED prevents PASS.
Zero robust lost area or zero sampled normal reversals is insufficient for
PASS. Keep per-case statuses and every rejection; do not accept a source-wide
claim by averaging failures away.

## A. Hard numerical conversion contract

PASS requires all selected inputs to pass the existing declared checks for
source hashes, finite values, metric units, proper/invertible frame transforms,
exact native reconstruction within serialization precision, joint-name
correspondence, actual source-URDF limits, nominal target reproduction and
independent FK. Preserve native/raw/filtered/clamped/recorded states separately.
Existing per-check numerical tolerances remain those justified in their frozen
protocols; this note does not replace them with an arbitrary geometric margin.

Changing source proposal, wrist placement convention, hand/object assets,
retargeter configuration or q to pass the contract is a new mapping version.
Decimal scene-bound excess remains a representation audit distinct from actual
source physical limits. An unsupported solve is a mapping refusal, not a grasp
failure label.

## B. Spatial sampling uncertainty and same-part proximity contract

Use the same fixed visible-object surface cells for native and mapped hands.
Collision-object reference families receive a separate report and status; do
not pool them with the visible-object reference. State whether the measured
distance is to native open anatomical patches, closed native hand, individual
target collider surfaces or an exact exterior union boundary.

For a planar child triangle C with centroid c and vertices v_i, set

    r_C = max_i ||v_i - c||
    I_H,f(C) = [max(0, u_H,f(c) - r_C - epsilon_d),
               u_H,f(c) + r_C + epsilon_d]

Here u is distance to the declared fixed part surface and epsilon_d is a
separately justified closest-point numerical allowance. Every point of a
triangle lies within r_C of its centroid; distance to a nonempty fixed set is
1-Lipschitz. Triangle area alone is not a bound on spatial uncertainty. If
epsilon_d is only an engineering numerical allowance rather than rigorously
certified arithmetic, name the interval “conditional on epsilon_d,” not a
formal floating-point proof.

At tau in {1, 2, 5} mm:

- DEFINITE_NEAR if the upper interval endpoint is at most tau.
- DEFINITE_FAR if the lower interval endpoint is strictly greater than tau.
- AMBIGUOUS otherwise, including unavailable part geometry.

A native DEFINITE_NEAR/mapped DEFINITE_FAR cell is certified lost proximity;
the reverse is certified added proximity. Either is FAIL for strict same-part
band preservation. Count both directions: a new relationship can repair a bad
native grasp and still violate faithful proposal preservation. Matching
DEFINITE_NEAR or matching DEFINITE_FAR certifies same binary state over the
cell. Any remaining required cell is UNRESOLVED, even if its centroid states
match. PASS needs certified equal states throughout every required cell, part
and band. Individual part masks overlap and are not a partition; native-empty
denominators remain null with explicit counts.

This contract preserves binary states at the declared bands. It does not assert
equal distances within a band, equal surface areas, or contact-force/outcome
equivalence. Publish continuous distance changes and shared/lost/new areas as
additional diagnostics; IoU is not a gate.

Where a distance definition includes buried individual-collider interfaces,
the resulting claim is about that literal distance function. Do not label it
exterior contact preservation. Missing exact exterior-union geometry leaves
the stronger contact-boundary claim UNRESOLVED even if the literal criterion
passes.

## C. Occupancy and penetration contract

Only a valid closed whole-hand solid or declared closed collider union may
receive inside/outside labels. Native anatomical patches are open and stay
unsigned. A robust occupancy mismatch at a common object point or certified
cell would refute a separate whole-hand occupancy-preservation claim. It is not
an extra gate silently added to the frozen unsigned-proximity claim. Boundary/undefined solid
membership is UNRESOLVED. Signed-distance cell certificates require a valid
solid and a signed function with the stated spatial bound; winding on a
self-overlapping mesh is not automatically a proof of a unique physical solid.

The frozen native hand definition uses HUG's supplied 1,538 MANO triangles plus
14 wrist-cap triangles. The topology/orientation checks support the winding
diagnostic under its declared assumptions; they do not establish that every
posed surface is free of self-intersection. Native finger patches are never
closed independently. Target occupancy is a union of the actual declared closed
colliders; distance to the union of individual collider *surfaces* is a different
fixed-set function and can include buried interfaces.

Negative signed distances in a native proposal are retained. No absolute
“penetration must be zero” or “the hand must lift” criterion is allowed here.
Hand-to-object sampled depth distributions remain separate, noncorresponded
shape diagnostics. Sampling-density sensitivity and finite sampled maxima do
not certify equality or bound the true maximum unless the full cell domain
and sign assumptions support the stated bound.

## D. Object-facing surface orientation contract

For an eligible common object query q, use outward unit normals of the actual
object face and the relevant nearest same-part hand feature. Define

    a_H(q) = -n_H(q) dot n_object(q)

Positive means the two outward normals oppose each other (facing); negative
means they point into the same hemisphere (away); zero is tangent. Use a
numerically justified score allowance epsilon_a: FACING when a-epsilon_a>0,
AWAY when a+epsilon_a<0, and TANGENT_OR_UNCERTAIN otherwise. This uses a
directional geometric distinction, not a fitted 20/30/45-degree cutoff. Even
FACING can be very oblique and does not establish mechanically useful contact.

Eligibility requires valid nondegenerate object and hand normals, predeclared
same-part correspondence, required proximity, a verified exterior hand feature,
and the absence of unresolved nearest-feature/normal ties. Penetrating nearest
points, carrier parts without native counterparts, lost/new regions and
unavailable comparisons are reported rather than discarded from the overall
claim. An ineligible required comparison contributes UNRESOLVED, not PASS.

Bind the eligible-query definition to the frozen orientation measurement
protocol, including its shared-proximity band, anatomical matching and tie
rules. Report both the eligible conditional domain and the full required
proximity domain; do not enlarge a score by excluding lost/new or unavailable
relationships. A successful eligible-only comparison cannot turn incomplete
full-domain preservation into PASS.

A reliable native FACING/mapped AWAY query or the reverse is a certified
orientation-state counterexample and FAIL. A single reliable query can refute a
universal directional claim, but matching signs only certify the named query
sites. For a full surface-cell PASS, nearest-feature stability and normal
variation must be bounded separately. The 1-Lipschitz distance interval does
NOT bound normals: nearest features can switch, and mesh edges can change a
normal discontinuously. If those extra bounds are absent, retain query-level
results and mark full-domain orientation UNRESOLVED. Area weights attached to
query signs are quadrature estimates, not rigorous orientation-cell areas.

No assertion about physical contact forces follows from these normals.

## E. Decision at the end of this phase

Apply the pre-frozen rule to the new reports without threshold fitting. Report
A, B and D separately, preserving their domains and scope; C remains a separate
limitation/diagnostic rather than an added acceptance gate. Label the conjunction
as STRICT_SURFACE_RELATIONSHIP_PRESERVATION_AT_DECLARED_BANDS, not full native
grasp equivalence. Report
failure-label preservation and unseen-source/object generalization as
NOT_TESTED. Do not relax a criterion because current HUG fails; any later
approximate acceptance contract needs a new version with external rationale.

If strict relationship preservation fails, the next step is to identify the
affected fingers/regions and decide between a new mapping implementation and
a narrower declared pipeline claim. If unresolved uncertainty remains, improve
the measurement only where it can resolve the stated claim. More lift trials
cannot retroactively establish a static property. Independent mapping tests and
outcome-sensitivity experiments require separate later authorization.

## Why this is consistent with prior research

[HUG §5.2 and Appendix E](https://arxiv.org/html/2606.17054v1) reports geometry
and execution separately, discusses penetrations and misses, and studies
size-dependent effects of grasp displacement. Its simulation hand approximates
MANO and its robot embodiments have specific retargeting methods. This supports
separate evidence layers, not a universal HUG-to-Shadow tolerance.

[C2Dex §III-A and IV-A/D](https://arxiv.org/html/2608.07045v1) uses opposing
surface-normal consistency and evaluates contact position, alignment and
penetration on fixed reference hand/object inputs separately from trajectory
execution. Its objective can intentionally improve reconstructed contacts. We
borrow a measurement structure; its aggregate errors are neither an acceptance
budget nor ground truth for our saved proposals.

[DexPilot §VII-A](https://arxiv.org/pdf/1910.03135) handles distinct hand
morphologies with task-space vectors and deliberate pinch projection. A useful
deployment retargeter therefore need not preserve every original poor proposal.
Our criterion answers a different, stricter proposal-preservation question.

[MuJoCo 3.3.7 contact attributes](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#body-geom-margin)
distinguish detected contacts from active constraints through margin and gap.
These simulation parameters are not native-geometry accuracy allowances, and
our 1/2/5 mm bands do not become physical contact thresholds merely by being
reported. Collider approximation, solver settings and execution remain explicit
later protocol ingredients.

All acceptance definitions above are project choices derived from the declared
claim. The papers support their measurement rationale; they do not certify the
choice or require zero distortion for all research uses.
