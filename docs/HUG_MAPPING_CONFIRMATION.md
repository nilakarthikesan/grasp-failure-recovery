# HUG mapping confirmation

14 October 2026. HUG is the first individual generator under review. This
inspection uses the existing cube/seed-0 native output; no new grasp-generator
inference, physics rollout, evaluation dataset or predictor training occurred.

**Current conclusion:** The native representation, coordinate conversion and
joint conventions are confirmed for the checked boundaries. Human-to-Shadow
grasp preservation is approximate and has not earned full confirmation. Keep
HUG as the active mapping case before reviewing ContactGen, DexGrasp Anything
and UniDexGrasp individually.

## What HUG produces

RGB, registered depth, camera calibration and an object selection condition
HUG's prediction. The native result describes a right human MANO hand: shape,
15 finger-joint rotations, global wrist rotation/translation, 21 landmarks,
778 vertices and 1,552 faces. It does not directly supply Shadow joint targets.

For this check the source output is the already saved cube proposal, SHA-256
`573cb003458d9d7cfe85b5b4a7877e414257b3e67d5abda04a09bd6d687fa0c2`.
Object scale/frame, observation and source runtime remain those of
[the completed single HUG run](HUG_SINGLE_EVAL_2026-10-13.md).

## Check 1 Native representation

The pinned official HUG MANO wrapper reconstructs the saved hand from its
existing pose/shape parameters without loading or sampling the HUG generator.
All 778 vertices, 21 landmarks and wrist rotation match the saved export exactly;
camera reprojection also matches all 21 saved 2D landmarks exactly.

HUG exports the closed MANO face table: 1,538 original triangles and 14 wrist-cap
triangles. Those 1,552 face indices match exactly. The licensed asset was hashed
and read locally; it is not copied into the audit.

**Result:** Native representation confirmed for this saved output. This checks
that the displayed human hand is the one encoded by the generator output.

## Check 2 Coordinate and joint conversion

The adapter removes the global camera/wrist transform, applies the proper
MANO-to-operator rotation, and passes wrist-to-tip/middle-link vectors to Dex
Retargeting. The static configuration preserves metric dimensions, disables
temporal filtering, and runs 20 optimization iterations. Joint values are
reordered by name and limited to the actual Shadow model.

The Shadow palm is placed at the predicted MANO wrist using an explicit origin
convention. Object-relative palm pose then composes with the simulated object's
world pose. Correct transform algebra does not prove that this morphology
correspondence preserves surfaces.

Independent algebra checks pass for 64 nonidentity object transforms, 192
quaternion normalization cases and 128 permutations of the actual saved joint
values. Three invalid quaternions and a missing joint are rejected. Sixty-four
global changes of camera coordinates preserve wrist-local landmarks to numerical
roundoff. These manufactured coordinate changes are not new grasps or task cases.

Eighteen existing decoder/frame/retargeting/static-FK tests pass with all MuJoCo
stepping APIs prohibited. One unrelated stock-Adroit test is excluded. The
checks include agreement between Dex/Pinocchio and MuJoCo forward kinematics.

**Result:** Coordinate and joint conventions confirmed within tested scope.

## Check 3 Nominal grasp geometry and surfaces

Compare the native hand and the nominal Shadow target around the identical
50 mm cube before approach, closing or lifting. Do not adjust scale or placement
to improve the appearance of agreement.

Existing object-frame fingertip errors are thumb 14.125 mm, index 3.068 mm,
middle 4.852 mm, ring 3.724 mm and little 7.751 mm. Mean error 6.704 mm passes the
existing 10 mm mean gate, while three abduction joints are at their limits.
The gate is an implementation tolerance, not proof of contact preservation.

The native near-surface vertices and nominal Shadow collision contacts cover
the same three cube faces (-X, +Y, +Z). This is coarse agreement only. Native
vertex penetration reaches 15.213 mm and nominal Shadow collision penetration
11.196 mm; these are different measurements and cannot be subtracted as a
mapping-error score. The nominal Shadow thumb does not contact the object.

Native poor grasps must also be represented faithfully. Native penetration or
later task failure alone does not establish a mapping defect, and validation
must not silently repair or remove poor proposals. The relevant question is
which contact locations, fingers, normals or penetration patterns are changed
by conversion.

**Result:** Nominal pose is approximate. Full contact preservation remains
unconfirmed. The wrist-to-palm morphology convention is not independently
calibrated by the successful cube lift.

### Surface comparison at the same pose and scale

![Native HUG and translated Shadow surfaces](../research_artifacts/hug_mapping_confirmation_2026-10-14/hug_native_shadow_surfaces.png)

Left is HUG's exact saved MANO mesh, colored by skinning-derived anatomical
parts. Right is the nominal Shadow collision geometry. Both use identical
object axes, camera, limits and metric scale. Forearm/wrist carrier geometry
is omitted from visible panels so the fingers remain readable.

The target reconstruction uses actual analytic primitive shapes and each
mesh geom's own convex hull, consistent with
[MuJoCo 3.3.7 collision geometry](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html#collision-detection).
It does not substitute a single convex hull for the whole hand. The saved
native mesh is closed, consistently oriented and nondegenerate; the review
does not establish absence of native self-intersection.

The diagnostic queries 3,750 fixed cube-face cell centers, 2 mm apart. A query
belongs to a footprint if it is inside the native hand / a target collider or
lies within the chosen 2 mm or 5 mm proximity band. This measures occupied-or-near
geometry, including penetration; it is not actual physical contact area.

At 2 mm, the native footprint is 1,876 mm², Shadow 2,208 mm² and shared footprint
1,504 mm². Intersection over union is 0.583; at 5 mm it is 0.682. These are
resolution- and band-dependent diagnostics, not prediction accuracy or a
mapping pass threshold. The dominant index/middle (-X) and ring/little (+Y)
regions persist, while detailed regions shift or broaden. Native little-finger
coverage on the top face (+Z) is absent from the target's 2 mm footprint.

![Cube face geometric footprints](../research_artifacts/hug_mapping_confirmation_2026-10-14/hug_cube_face_surface_footprints.png)

The geometric nearest-surface normal differences on -X/+Y/+Z have medians
21°/72°/40° where both surfaces are outside and within 5 mm. They compare
nearest surfaces at common cube queries, not necessarily the same finger or
paired physical contact normals. Skinning labels are anatomical proxies.

**Interpretation:** Correct coordinate placement retains the gross grasp
arrangement, but does not preserve the detailed human-hand surface footprint
when converted to this robot. The finer evidence makes the remaining fidelity
gap explicit; it does not prove a frame defect or native grasp success/failure.

## Check 4 Execution changes

The current v2 executor starts near-open, approaches along world negative Z,
then adds bounded finger flexion before lifting. Its extra closing command
moves this case's fingertips by 18.644–32.212 mm in static FK. Actual physical
motion is constrained by contacts and actuator forces.

Those commands are separate from the nominal mapping. The existing successful
lift shows that one adapted grasp can execute; it does not establish fidelity
to native human contacts or to the original HUG evaluation system.

The current adapter, controller, predictor and historical labels remain
unchanged. No new execution experiment is part of this confirmation check.

## 14 October continuation: broadened static checks and finer surfaces

The static follow-up uses 32 distinct existing HUG native poses across cube,
apple, milk box and mug, in eight observation groups. Each object has two scenes
and four seeds. All share one MANO shape. The Oct 13 cube is a duplicate of one
of these sources. These are previously inspected development families, not a
new independent holdout or newly generated evaluation data.

Each native pose is bound by hashes and inference reports to the same recorded
HUG source/checkpoint. Saved mapping-code hashes match the current code. The
original Shadow target is recovered from each zero-noise episode's closing
endpoint at 2.16 s, before force-close: 24 named targets and a world carrier
pose. The cube's joint values match the current canonical proposal exactly.

Guarded static FK through each captured scene preserves its initial object
state and simulation time zero. Independent fixed-root URDF task-link positions
agree with the scene reconstruction within 7.845e-8 m. The reproduced mean-tip
error agrees with the original saved validation within 9.792e-9 m. Joint-limit
violations are zero. The floating carrier includes the URDF world-joint's
1.57079 rad fixed origin; comparing its quaternion directly with the operator
basis would be an incorrect frame comparison.

Across 160 fingertips, mean native-to-Shadow separation is 6.428 mm. Per-pose
means range 5.704--8.305 mm; the worst individual tip is 17.518 mm, the milk-box
scene-001/seed-0 thumb. Object-level means are cube 6.405 mm, apple 5.865 mm,
milk box 7.000 mm and mug 6.440 mm. These point residuals do not establish contact
preservation, and the existing 10 mm mean implementation gate remains separate
from scientific acceptance.

A protocol frozen before the cube refinement specifies 2, 1 and 0.5 mm grids
and 1, 2 and 5 mm diagnostic bands. The actual centered 50 mm box and all 45
hand/object collider compatibilities are verified, with no explicit pair or
body-exclusion overrides. The previous 2 mm grid's 2/5 mm global results
reproduce exactly. At the 2 mm band, occupied-or-near IoU is 0.582946, 0.580406
and 0.578557 across the three grids. At 5 mm it is 0.681668, 0.681060 and
0.681128. The main discrepancy persists; this is stability over tested grids,
not proof of continuum convergence.

Independent unsigned per-finger proximity uses open native anatomical triangle
subsets and actual target colliders. It does not invent enclosed finger volumes.
At the finest grid and 2 mm band, top-face little-finger proximity is nearly
lost: native 57 mm^2, Shadow 0.5 mm^2, shared 0.25 mm^2. Thus the earlier coarse
absence finding must be qualified at finer resolution. Native little-finger
+X coverage and target index -Y coverage remain unmatched in the 5 mm sweep.

Normal comparisons now require globally exterior near queries with matching
anatomical labels and retain unmatched/mismatched queries separately. The fine
2 mm band contains 748 matching and 84 different-label common queries. Among
matching pairs, 548 have both closest hand points inside the cube, 100 one
inside, and 100 neither inside. Median angle is 25.56 degrees over the matching
subset, versus 59.99 degrees when neither closest point is inside. This shows
why a small conditional closest-point separation does not establish valid
physical contact correspondence. Skinning labels, facet normals and deterministic
grid areas remain geometric proxies; they are not physical contact annotations.

![HUG static validation continuation](../research_artifacts/hug_mapping_refinement_2026-10-14/hug_mapping_validation_progress.png)

### Argument supported by these checks

Let T_AB map coordinates in frame B into frame A; W is world, O object, C camera,
M native MANO and P target palm. The native object-relative pose is
`T_OM = inverse(T_WO) * T_WC * T_CM`. Execution composes the saved target as
`T_WP = T_WO * T_OP`. A common rigid change of world coordinates cancels in the
object-relative expression. For the proper MANO/operator rotation A,
`v_operator = A * v_MANO` and `R_Woperator = R_WMANO * transpose(A)` imply
`R_Woperator * v_operator = R_WMANO * v_MANO`. This explains why coordinated axis
conversion preserves represented world geometry. Exact joint-name matching and
independent FK then check that the target model realizes the stored joint policy.

The numerical evidence supports decoding, placement and joint conventions in
the tested scope. Choosing the Shadow palm as the MANO wrist anchor and fitting
robot fingers to human vectors are additional morphology assumptions. The
surface discrepancies demonstrate why those assumptions cannot be justified by
transform algebra or a successful lift alone. They need separate evidence before
claiming preservation of the original grasp's contact relationships.

All measurements, frozen protocol, scripts and hashes are in
[the continuation bundle](../research_artifacts/hug_mapping_refinement_2026-10-14/README.md).
No generator inference, physics rollout, evaluation-data collection, predictor
fit, calibration or adapter/controller change accompanied this continuation.

## Confirmation boundary and next step

Use **native representation confirmed**, **coordinate/joint conversion
confirmed**, and **grasp/contact fidelity unconfirmed** as separate statements.
The overall HUG mapping is partially confirmed, not signed off as equivalent.

The finer-grid and anatomically matched surface check is complete. Numerical
transform/joint consistency now covers the 32 saved poses; full contact fidelity
remains unconfirmed. Next separate the wrist-to-palm anchor assumption from
finger morphology and characterize per-finger object-relative surfaces on
additional existing objects. Any proposed calibration must be versioned and
use different cases from its validation; a criterion must be justified before
acceptance testing, not selected because this cube passes. Preserve poor native
proposals and report changed/lost relationships rather than silently repairing
or removing them. Outcome sensitivity and physical transfer require separate
future execution evidence. HUG remains active before the next provider.

See [the initial four-source audit](GRASP_MAPPING_FIDELITY.md) for source links
and existing static measurements. HUG-only test reports and reproducible
diagnostics belong under `research_artifacts/hug_mapping_confirmation_2026-10-14/`.
