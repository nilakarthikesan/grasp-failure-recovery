# HUG static mapping validation and acceptance

14 October 2026. Version 3 of this measurement/acceptance status note. Per-phase
frozen operational protocols and original reports remain immutable. Strict
encoded-surface acceptance V1 is frozen and applied; approximate predictor-use
adequacy, independent validation and outcome equivalence remain pending.
[The new static correction](HUG_SURFACE_REFINEMENT.md) optimizes mapping targets
on saved poses, with separate experimental APIs; it does not train a predictor.
No new HUG inference, eval data, physics, default mapper change or manuscript
edit occurs. Historical phase descriptions below retain their original scope.

## Current correction status, 14 October 2026

Native-relative surface fitting and guarded interpolation are implemented on six
existing noncube poses. The same six coarse/three finer checks still FAIL the
unchanged strict acceptance rule despite improved objective/whole unsigned IoU.
Tip/middle errors worsen and some added-region/direction witnesses regress.
The experimental APIs remain separate from the unchanged production mapper.
[Correction evidence](HUG_SURFACE_REFINEMENT.md) records settings, histories,
all cases, roots, checks, numerical assumptions, source hashes and next proposed
constrained/direction-aware solve. Its 10 mm engineering guard is not a newly
adopted geometric acceptance allowance. No current result certifies native
contact/failure-label equivalence or independent predictor adequacy.

## Completed checks and frozen acceptance, 14 October 2026

[The orientation/uncertainty
bundle](../research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/README.md)
implements object-facing sampled normals, valid/tied/buried/inside/lost/new
denominators and cell-radius Lipschitz proximity-area intervals for the six
fixed cases and three finer repeats. The all-32 native parameter reconstruction
matches every 3D vertex/landmark/rotation and face index exactly; all-32 camera
projection is not claimed.

[Acceptance V1](HUG_MAPPING_ACCEPTANCE.md) is now frozen for the strict
declared-band encoded-surface relationship claim. Numerical conversion passes
within checked scope; definite anatomical changes fail strict preservation in
all six cases and three finer repeats under explicit conditional numerical
allowances. Sampled normal sign reversals provide additional pointwise
witnesses. The choice is a definition of strict preservation, not a universal
approximate-retargeting tolerance or an imported paper standard. No
data-dependent tip/IoU/angle margin is adopted.

Conditional distance bounds cover the full retained visual triangle domain (zero
original degenerate faces excluded), but do not prove a formal arithmetic error
bound, continuous angular agreement, exact solid/intersection volume, physical
contact or outcome equivalence. Those stronger claims remain unresolved or
untested. Independent unseen-object validation and approximate predictor-use
distortion budgets still require separately justified protocols. No inference,
new eval data, dynamics, fit, production change or manuscript edit occurs. The
earlier sections describe measurement definitions and historical/proposed work;
this completed-status section supersedes their pending orientation/cell-bound
entries.

## Earlier geometry/reference measurement scope, 14 October 2026

[The current evidence
bundle](../research_artifacts/hug_geometry_reference_validation_2026-10-14/README.md)
implements fixed-reference point/FK checks on all 32 saved development poses and
detailed surface/signed profiles on the six previously selected seed-0 noncube
poses. Earlier bundles establish native reconstruction, frame algebra,
wrist/palm proxies and finer cube diagnostics. All use one MANO shape; no new
inference, eval episode/label, dynamics, fitting or production change occurs.

Implemented: source/compiled object scale-frame checks and local gap/exterior
witnesses; current/released/metric-static reference methods; whole/per-part
shared-lost-new proximity across three bands; continuous common-query distance
changes; conditional native/mapped hand-normal comparisons; separate signed
hand-to-visible-object profiles, native vertex counts and palm/finger versus
carrier domains; numerical/source-hash guards and preserved protocol amendments.
These methods report different denominators and uncertainty. Generic query-count
p95 values are not area-weighted p95.

At the preceding phase, object-facing normals and cell-radius bounds remained
pending; the completed-status section above now records those results. Exact
continuous intersection/volume, accessible union-boundary certificates,
approximate adequacy budgets, independent validation, execution/failure-label
sensitivity and physical transfer remain unresolved or untested. No arbitrary value is promoted to acceptance because these
development cases pass. The requirements below define the reporting framework
and future work; this document does not assert that every metric has been
implemented.

Original reference metadata mislabeled captured-scene bounds as physical limits;
the standalone correction preserves the historical report, verifies
full-precision source XML clamping and discloses decimal scene excess. Static
diagnostics do not authorize or certify a production rollout. Frozen operational
protocols and superseded pilots remain separate. Strict declared-band acceptance
is now frozen; approximate adequacy and independent execution protocols are not.

## What can be established now

Freeze measurement definitions and the fixed reference-method comparison before
running it. Do not declare an overall scientific pass threshold by inspecting
whether the current 32 development poses pass. These poses, including all four
objects and eight observation groups, have already informed the investigation;
they cannot become independent validation merely by splitting them now.

Keep three separate conclusions:

1. Native output reconstruction and numerical conversion are supported within
   explicitly tested scope.
2. Cross-hand geometric preservation is an approximation whose changes are
   measured, including changes to poor native proposals.
3. Execution success/failure and the impact of mapping changes on those outcomes
   remain later questions under a declared controller and physics protocol.

The currently implemented 10 mm mean tip gate is an engineering guard. Neither
that value, HUG's reported error numbers, nor a competing retargeter's reported
errors establishes an acceptable mapping error for our simulator or predictor.

## Evidence from primary sources

[HUG §5.2 and Appendix E](https://arxiv.org/html/2606.17054v1) reports fingertip
surface distances alongside execution and distinguishes penetrating proposals
from misses. Its physics hand approximates MANO with phalanx capsules and a
convex palm; the smooth MANO surface is therefore not the original simulator's
exact contact surface. Appendix E.2 examines displacement sensitivity and finds
size-dependent execution effects. Appendix F.2 uses AnyTeleop for Ability and a
WUJI-specific retargeter for WUJI. This supports paired geometry diagnostics and
separate execution evidence; it does not supply a HUG-to-Shadow certificate,
universal tolerance or measured contact-force ground truth for our saved outputs.

[DexPilot §VII-A](https://arxiv.org/pdf/1910.03135) motivates fingertip distance
and direction objectives because human and robot joint axes differ. It also
projects selected pinch distances to improve manipulation. Thus an optimizer
designed for executable contact can intentionally alter native geometry. Its
numbers are not proof that our proposal-preserving translation is accurate.

[C2Dex §IV-A/D](https://arxiv.org/html/2608.07045v1) isolates retargeting with fixed
reference human/object trajectories and reports contact-region displacement,
normal alignment and penetration separately from execution. This is a useful
comparison design. Our inferred proximity regions have weaker evidence than its
reference trajectories and must retain that distinction. We propose adopting
the reporting structure, not importing its acceptance thresholds.

These source statements motivate the protocol. The definitions below are
project proposals, not definitions certified by those papers.

## A. Hard numerical and provenance checks

Bind every selected proposal to native file hash, MANO shape/pose and face table,
camera/intrinsics/extrinsics, initial object pose and geometry, source/checkpoint,
processing stage, retargeter/version/configuration, Shadow URDF/collision assets,
and the recorded nominal joint targets. Keep raw/native/nominal/commanded/achieved
states distinct. Record explicit mapping refusals and causes for every attempted
proposal; never convert a mapping refusal into a physical failure label.

Require finite values, right-handed rigid transforms, declared metric units,
unique names and complete joint-name correspondence. Test identity and inverse
transforms, nonidentity object poses, global camera changes, named-joint
permutations and independent forward kinematics on the same Shadow asset.
Reconstruction should recover all native vertices/landmarks/faces and relevant
projection outputs, not merely the five tips. Floating-point tolerances should
be justified by serialization precision, independent implementation error and
accumulated transform operations; separate float32 and float64 paths. Preserve
existing exact reconstruction and roundoff measurements instead of replacing
them with a looser new threshold.

Require nominal robot joints within the actual physical limits. If the source
solver allows its documented ~1e-3 rad optimization margin and then clamps, record
the affected joint count and maximum change. Do not silently call clamped output
the unchanged optimizer result. This is an implementation policy, not a human-
to-robot joint-angle accuracy metric.

For a static comparison, protect mj_step/mj_step1/mj_step2, retain time zero and
initial object state, and verify hashes of production code, original targets,
inputs, model and manuscript before/after. A failed check blocks that case's
numerical-conversion certificate; it is not a grasp-quality judgment.

## B. Separate the object and hand geometry questions

Use actual compiled collision transforms/mesh scaling and active masks. The
object visual mesh and the exterior union of actual collision pieces are two
different reference families; never merge their metrics. Bind each to the same
object frame. Investigate bidirectional boundary distances, occupancy differences,
component transforms and retained cavity surfaces. Distance in one direction
alone does not prove equivalent boundaries.

Surface samples of a convex union must exclude buried boundaries, duplicates
and abutting opposite-normal internal interfaces. Preserve prior v2/v3/v4
protocol results as their actual versions. A sample's unsigned distance to the
surface of individual hand colliders is not exact distance to the exterior
boundary of their union when the query lies within the union. Retain that
limitation explicitly until the query method resolves it.

Use common object-side queries for native versus mapped hand comparisons within
each reference family. Report current square-millimeter child-area density as
area density, not a maximum edge-spacing guarantee. Include actual area weights,
triangle diameter/maximum-edge distributions and density sensitivity. Thin or
long triangles can have small area but large spatial coverage uncertainty.

## C. Point, direction and proximity measurements

For each source pose and each mapper, keep native and mapped geometry in the
same object frame with identical object geometry and root-placement convention.
Report each declared anatomical proxy separately: tip, middle-link, finger-base
and palm proxies. Record world/object-frame residual vectors, mean/median/tail/
maximum Euclidean error, fingertip-pair vector distance/direction changes and
finger-segment direction changes. Numerical implementation agreement of the
MANO wrist and selected Shadow palm anchor does not prove tissue equivalence.
No Procrustes alignment, source-specific translation, rescaling or per-case
reinitialization using the desired answer is allowed to remove actual placement
errors from these metrics. Diagnostic aligned variants must be separate.

Let q_j be a common object-surface sample with area weight w_j and let f be a
predeclared anatomical part. Define u_N,f(q_j) and u_M,f(q_j) as unsigned distances
to the native and mapped surfaces assigned to that part. Native labels use a
fixed, documented skinning-derived rule; these are anatomical proxies. Open
native finger patches are never assigned inside/outside signs or volumes.

At the existing diagnostic bands tau in {1, 2, 5} mm, define proximity masks
P_N,f = [u_N,f <= tau] and P_M,f = [u_M,f <= tau]. For every finger, palm and
whole-hand metric, report absolute native area A_N, mapped area A_M, shared area
A_both, lost area A_lost = A_N-A_both and new area A_new = A_M-A_both. Also report:

- Retention = A_both/A_N when A_N > 0.
- Mapped-region correspondence = A_both/A_M when A_M > 0.
- IoU = A_both/(A_N+A_M-A_both) when the union is nonempty.
- Native-absent/mapped-absent/both-absent cases and areas explicitly. Empty
  denominators are null with counts, not perfect scores, zero scores or omitted
  cases. A native-absent/mapped-present part is an introduced relationship,
  potentially a contact repair, even if useful for lifting.

These masks are geometric proximity, not observed physical contact. Independent
parts may overlap in area; do not sum them as a whole-hand area partition. A
single nearest-part labeling offers a complementary partition but must retain
different-label and tie categories rather than replacing independent part masks.
Penetration can appear near a surface and must be reported separately.

Keep continuous paired distance changes delta_u,f(q) = u_M,f(q)-u_N,f(q), with
area-weighted distributions and fixed source-proximity strata. This avoids
reducing all evidence to one chosen band. Report both loss and added proximity;
making delta_u negative is not automatically faithful for a bad native pose.

## D. Signed distances, penetration and normal interpretation

For watertight object geometry, sample each hand's valid surface and report signed
object distance with negative-inside convention. Report per-part native/mapped
profiles, cases with observed sampled penetration, maximum sampled depth,
penetrating surface area fraction and distance-to-object for cases with no
detected intersection. An exact intersection test and a negative signed-distance
sample are different levels of evidence; a finite sampler's absence of
penetration is not a proof of no intersection. Include zero sampled-penetration
cases in unconditional summaries and state intersecting-only denominators.
Intersection volume, if implemented, must name the integration
domain, method and resolution; a surface fraction is not intersection volume.

Native and mapped hand surface samples are not automatically point-corresponded.
Their signed-distance distributions are comparative anatomy profiles, not
matched residuals. For genuinely paired signed changes, use a common object-side
query and a valid closed whole-hand exterior-union signed-distance function;
per-part open patches must remain unsigned. Occupancy results can complement
these distances but must not be presented as exterior-union surface distance.

Where both methods have a same-part exterior proximity region, compare their
nearest hand-surface geometric normals in the common frame and the respective
object-facing compatibility (-n_hand dot n_object). Freeze matching/tie rules,
exclude degenerate normals with counts, and retain nearest points inside the
object, different labels, lost/new regions and unavailable comparisons in the
report. Show angular distributions and denominator areas/counts. Normals on
buried collider surfaces or penetrating points are not contact-force normals.
Good normals on the subset retained by both mappings cannot certify the regions
lost by either mapping. Do not discard those regions or native failures when
calculating the denominator.

## E. Numerical resolution and threshold boundaries

The 1/2/5 mm bands are inherited diagnostic bands, not acceptance values. Label
regions near a band boundary as sensitive to measurement resolution. Report the
actual distribution of triangle diameters and repeated-density changes; do not
treat quadrature queries as independent statistical samples.

For unsigned distance to a fixed surface, moving the query by at most r changes
distance by at most r (the distance function is 1-Lipschitz). With a valid local
cell radius and an established closest-point numerical error bound, this gives
a justified resolution interval for each query. It does not bound hand mapping
error, collider-versus-visual discrepancy, or mesh-normal error. Where the
interval crosses tau, report ambiguous proximity area. Do not claim a global
error bound from an observed finite-sample maximum or from child area alone.

A conservative robust-lost flag requires the native upper distance bound below
tau and mapped lower distance bound above tau; robust-new reverses them. Report
these flags and remaining ambiguous regions without converting zero robust
loss into an overall mapping-equivalence certificate.

## F. Fixed reference retargeter comparison

Compare the current recorded nominal targets with a fixed alternative objective
on the same native inputs, target Shadow asset, object pose and root convention.
Dex Retargeting 0.5.0's packaged DexPilot-style Shadow objective is a feasible
reference-method diagnostic. It is not a HUG-to-Shadow oracle and not the exact
original DexPilot reproduction: installed optimizer.py explicitly uses a Huber
loss in place of the original squared-distance loss. Pin its actual bytes.

Before execution freeze: input order, per-case fresh solver state, initialization,
joint limits/optimizer margin/clamping, filtering policy, iteration budget,
scaling, projection/escape distances and every other default. Packaged Shadow
teleoperation configurations scale vectors by 1.2 and filter frames; static
adaptations must be stated. If metric scale 1.0 and no filter are used to make
the comparison controlled, name it a project static adaptation, not the upstream
default. Report projection-triggered changes to pinch relationships. Declare
whether wrist joints are optimized and whether root pose is fixed for both.

Use independent per-case initialization, not unreported source-order warm starts.
Report objective residuals, termination status, bound saturation, repeated-run
reproducibility and all failures. A failed inverse solve remains a reference-
mapping refusal rather than a physical task failure. Replaying current mapper
with the same cold initialization can provide a separate reproducibility/control
metric because saved targets may have a different solver history. Do not replace
the recorded targets or present alternate static targets as fresh generator
proposals/evaluation data.

For each evaluable paired case, publish changes in landmark error, part-distance
profiles, lost/new regions, occupied/near areas and penetration, not just the
alternative's internal optimization objective. Summary means alone can hide
fingers that change in opposite directions. Neither lower landmark error nor
greater near-object area is sufficient evidence of faithful mapping.

## G. Reporting and eventual acceptance

Report all 32 native poses for available point/FK checks and the six previously
selected seed-0 noncube poses for currently available detailed surfaces. Give
every count: objects, observation groups, native proposals, reconstructable
inputs, attempted mappings, valid inverse solves, rejected mappings and metric-
available denominators. Do not imply six selected surface cases cover all 32
surface configurations. Keep the duplicate cube separate from distinct-pose
counts. State that one fixed native MANO shape is represented.

Use paired per-case differences, per-object/per-observation summaries, every
finger's tails/maxima, and source-poor versus changed-by-mapping descriptions.
These are descriptive development measurements. Confidence intervals or claims
of source-wide generalization are not established by treating thousands of
surface samples or repeated seeds as independent observations. Future intervals
should respect object/observation clusters; four existing objects give limited
support for population inference.

Freeze a later independent validation plan with unseen object geometry and
observations as groups, not random pose rows from the same object. Record shapes,
grasp types, scale and quality strata in advance. Keep calibration geometry,
mapper settings, threshold choice and future acceptance tests separate. A later
changing retargeter requires a new version and calibration record.

An overall acceptance margin must be justified from the intended claim and use:
resolution/numerical uncertainty, actual collider/reference approximation,
anatomical correspondence, simulator contact tolerance and later outcome-
sensitivity evidence. The current object approximation can exceed the existing
diagnostic bands; static measurements cannot establish that such differences
are outcome-irrelevant. A geometry-only claim can use externally justified
geometric margins without execution data, when such margins and their scope
are available. A claim that retargeting preserves the eventual task outcome
requires separately authorized outcome-sensitivity evidence. It is defensible
now to accept numerical conversion
checks and publish bounded geometric errors. It is not defensible now to certify
generator-independent contact or failure-label equivalence using an arbitrary
mean error, IoU, normal angle, contact count or successful lift.

Failure prediction trained on a declared mapped-hand/controller pipeline is a
different claim from predicting the native generator's own outcome. Retargeting
distortion can confound source-diversity comparisons and should remain recorded
and investigated. Preserve bad native misses/penetration; an unsupported or
unrepresentable translation has its own mapping status, not a fabricated task
label. No proposed later validation collection or outcome experiment is
authorized or executed by this static protocol note.
