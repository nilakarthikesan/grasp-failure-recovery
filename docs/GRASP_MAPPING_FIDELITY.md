# Grasp generator mapping fidelity audit

Audited 14 October 2026 against `project-work` at `7b14f26`.
This is a static source, transform and forward-kinematics audit. It reuses the
saved HUG cube output and synthetic joint configurations; it contains no new
generator inference, physics rollout, task labels, evaluation collection or
predictor fitting. Existing controllers and historical results are unchanged.

## What we can currently claim

The common proposal format makes grasps executable by one Shadow Hand, but does
not establish that all four native grasps preserve their original contacts.
HUG's saved cube has verified coordinate composition and measured approximate
landmark fidelity. ContactGen has reviewed transform algebra without a real
native sample. UniDexGrasp has a verified reference kinematic bridge without
its actual released source-hand asset/native sample. DexGrasp Anything has no
implemented mapping; its official 24-joint URDF gives a promising static bridge
which differs from the older 22-joint handoff assumption.

Human MANO and Shadow have different shapes and joint structures. Their fidelity
must be expressed as measured pose/contact errors, rather than hand identity.
Even Shadow variants require separate checks of joint frames, surfaces and
collision representations.

## The common simulator boundary

Each accepted proposal carries object-relative palm position/orientation in
meters, exactly 24 named Shadow targets in radians, and input/model provenance.
The evaluator composes `T_world_palm = T_world_object @ T_object_palm`.
[Composition](../src/grasp_failure_prediction/proposals/execution.py#L85).

The current evaluator checks object geometry identity, joint names/limits,
protocol compatibility and a maximum palm distance. It records static hand
self-contact, but this is not a gate proving collision-free acquisition or
native hand/object contact preservation.
[Checks](../src/grasp_failure_prediction/proposals/execution.py#L100).

Separate three quantities in every future mapping audit:

1. Native pose versus nominal translated simulator pose, before any motion.
2. Nominal pose versus the common controller's requested commands.
3. Requested commands versus achieved physical hand/object states.

The v2 controller approaches along world negative Z from 8 cm away, closes from
a near-open posture, then adds up to 0.2 rad flexion before a 15 cm lift and
2 second hold. Its joint limits can clip those added commands. The generator
does not provide that trajectory. This is a declared execution choice, not a
coordinate conversion defect.
[Executor](../src/grasp_failure_prediction/evaluation/runner.py#L660),
[v2 parameters](../src/grasp_failure_prediction/evaluation/configs/execution_protocols.yaml).

## HUG

The subsequent [individual HUG confirmation](HUG_MAPPING_CONFIRMATION.md)
adds exact native reconstruction, boundary tests and detailed surface evidence.
Its status remains partial; the next generators are not advanced automatically.

**Native process:** RGB-D, camera calibration and an object selection condition
the model. Its output includes right MANO camera-frame landmarks, mesh vertices
and wrist pose. The adapter removes the global wrist pose, rotates local axes
for Dex Retargeting, fits Shadow landmark vectors, then places the Shadow palm
at the predicted MANO wrist origin. This wrist-to-palm origin convention is
explicit; it does not equate their surfaces.
[Pinned source](https://github.com/KevinyWu/hug/tree/8d1c52d4c24bfae5a369e32e3f134f5601a02630),
[retargeter](../src/grasp_failure_prediction/evaluation/retargeting.py),
[placement](../src/grasp_failure_prediction/evaluation/case_runner.py#L219).

**Measured from an existing real model output:** The native file is the saved
cube/seed-0 proposal with SHA-256
`573cb003458d9d7cfe85b5b4a7877e414257b3e67d5abda04a09bd6d687fa0c2`.
Axis conversion is orthogonal with determinant 1. Camera/wrist/object
round-trip discrepancy is at most 1.11e-8 m; the saved proposal and independent
composed Shadow landmark placement differ by at most 6.62e-9 m.

Nominal fingertip errors in the shared object frame are thumb 14.125 mm,
index 3.068 mm, middle 4.852 mm, ring 3.724 mm and little 7.751 mm. Mean error
is 6.704 mm. The existing 10 mm mean gate passes despite the larger thumb error.
The palm-relative diagnostic agrees with full object/world errors to 6.30e-9 m
under the implemented placement convention; no additional hidden frame offset
was found in this case. Three abduction targets reach joint limits.

![Saved HUG native and translated landmark comparison](../research_artifacts/grasp_mapping_audit_2026-10-14/hug_native_vs_shadow_object_frame.png)

Solid lines show MANO skeleton; dashed lines connect the Shadow task landmarks,
not its full collision surfaces. Circles/crosses are corresponding fingertips.
The object is the unchanged 50 mm cube; all panels use its shared frame.

The declared extra closing command moves nominal fingertips by 18.644 mm
(thumb), 30.935 mm (index), 31.637 mm (middle), 32.212 mm (ring), and 28.645 mm
(little) in static FK. Mean separation from native fingertips becomes 30.623 mm.
These are commanded static configurations, not achieved physical positions:
object contact and force-limited actuators affect actual motion.

![Controller-induced static displacement](../research_artifacts/grasp_mapping_audit_2026-10-14/hug_shadow_force_close_static_displacement.png)

Static reconstruction of the existing final simulator state produces six
contacts, matching the saved contact count. Reconstructed pairs include four
object contacts on thumb/index/middle/ring distal bodies and two hand
self-contacts; original pair identities were not saved in the trajectory. Palm placement matches the saved final trace
within 2.14e-10 m. This verifies the saved Shadow contact reconstruction, not
equivalence to the native MANO contact pattern. Forces were not reconstructed
because the saved final state omits control/mocap/actuator state.

The native MANO mesh itself has 116 saved vertices inside the fixed cube and
15.213 mm maximum vertex-to-box penetration. This measures the saved native
mesh, not mapping error; a successful later lift does not validate its native
contacts. Landmark distances do not substitute for surface contact tests.

**Nominal contact-region inspection:** Reconstructing the nominal translated
pose at the original captured cube state, before approach/closing/lift, leaves
the object state unchanged and gives FK time zero. Native mesh vertices near
the object surface (2 mm and 5 mm bands), penetrating native vertices, and
compiled nominal Shadow contacts all cover the same three faces: -X, +Y, +Z.
There are six nominal Shadow/object contacts, no thumb/object contact, zero
hand/table contacts and one hand self-contact. Nominal Shadow collision
penetration reaches 11.196 mm. This and native vertex penetration are different
measurements, so they must not be interpreted as a direct penetration residual.

Matching the three faces supports coarse geometric coverage in this one case,
not equality of contact patches, normals, forces or physical success. The
nominal inspection and existing after-lift reconstruction are different states;
their finger contacts should not be conflated.

**Verdict:** Coordinate composition is verified for this case. Human-to-robot
pose preservation is approximate and quantified. Contact-region and physical
equivalence remain unverified; one successful HUG lift establishes execution
reproducibility, not general mapping fidelity.

## ContactGen

**Native process:** An object mesh is centered by its vertex mean. ContactGen
samples contact locations, hand parts and directions, then its official
`optimize_pose` refines a right MANO hand. Our worker retains the refined
vertices and 16 joint frames. The adapter restores centering, applies the
mesh-to-object transform, constructs 21 thumb-first landmarks and uses the
same MANO-to-Shadow retargeting path as HUG.
[Pinned demo](https://github.com/stevenlsw/contactgen/blob/1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a/demo.py),
[worker](../scripts/infer_contactgen_object.py#L137),
[adapter](../src/grasp_failure_prediction/proposals/contactgen.py#L203).

**Verified scope:** Source review found no confirmed centering, root-origin,
joint-order or axis defect. The wrist comes from frame 0, including the shaped
MANO root, rather than optimized translation alone. Upstream metric coordinates
are not multiplied by an extra millimeter conversion. Independent static FK
composition at two wrist configurations matches intended object-frame vectors
within 1.21e-16 m. This is transform algebra, not a measured ContactGen grasp.

**Missing evidence:** There is no real ContactGen native output locally for a
native-to-Shadow fidelity check. Tests mock retargeting/validation boundaries.
Input mesh/hash/vertex-mean checks do not independently establish that the
transformed mesh surface equals the MuJoCo collision geometry. Optimized MANO
parameters are not regenerated against the saved vertices/frames. The worker
does not retain sampled input points or predicted contact/part/direction maps,
so the original contact intention cannot yet be compared directly.

**Verdict:** Plausible reviewed implementation, awaiting real native-output
and contact evidence. HUG's measured residual must not be assigned to ContactGen.

## DexGrasp Anything

**Native process:** Point-cloud geometry and normals condition diffusion; an
optional optimizer enables physics-guided sampling. The pinned DexGraspNet
configuration normalizes translation and 24 joint values. The released
visualizer undoes those normalizations and random object rotations, exporting
translation, 6D rotation and 24 joints (33 values).
[Pinned visualizer](https://github.com/4DVLab/DexGrasp-Anything/blob/86353d710ca71d0f547f0d8b15cc8ca959c78163/models/visualizer.py#L119),
[native hand model](https://github.com/4DVLab/DexGrasp-Anything/blob/86353d710ca71d0f547f0d8b15cc8ca959c78163/utils/handmodel.py#L225).

**Concrete handoff discrepancy:** The existing handoff asks for a free palm
and 22 named joints using the DexGraspNet MJCF mapping. The official DGA
generator uses a different 24-joint URDF rooted at the forearm, including both
wrists. At zero joints its forearm-to-palm displacement is
`[0, -0.010, 0.247]` m and changes with wrist configuration. The worker must
compose the actual palm frame and retain all wrist articulations, rather than
silently discard them or treat root translation as palm translation.
[Current handoff](DEXGRASP_ANYTHING_CLOUD_HANDOFF.md),
[official URDF](https://github.com/4DVLab/DexGrasp-Anything/blob/86353d710ca71d0f547f0d8b15cc8ca959c78163/assets/urdf/sr_grasp_description/urdf/shadowhand.urdf).

**Measured static bridge candidate:** Pure URDF FK, preserving the computed
source palm pose and matching all 24 joints by name, was checked on zero plus
64 deterministic random configurations inside shared joint limits. Maximum
fingertip separation was 0.100 mm (little); thumb was 0.080 mm; the other three
tips agree to numerical precision. Joint origins/axes and collision assets are
not byte-identical, so this is evidence for a dedicated mapping candidate,
not permission to reuse another source identity or a registered acceptance.

The documented smoke command also inherits upstream `dataset.active: real_dex`
unless overridden, despite our pinned checkpoint being DexGraspNet. It omits
`OPT`, so physics-guided sampling is disabled. Dataset/normalizer, object scale,
normals, stage and optimizer configuration must be explicit in the eventual
worker; the upstream visualizer's dataset-average scaling cannot replace our
registered object's exact physical dimensions.

**Verdict:** Native format and a promising kinematic bridge are audited. Actual
mapping remains `unsupported_pending_official_mapping` in the registry; no
fresh DGA sample or implemented decoder has been validated.

## UniDexGrasp

**Native process:** Object/table point clouds feed IPDF rotation sampling and
Glow translation/joint generation. Official ContactNet-based test-time
adaptation (TTA) can refine the pose; a separate learned policy executes it.
Our worker implements the non-joint IPDF+Glow stage before TTA. The adapter
undoes canonical rotation, composes input-to-object and actual source palm
offsets, verifies a compiled palm/finger kinematic certificate and maps 22
source finger joints into 24 target joints with fixed target wrists.
[Pinned source](https://github.com/PKU-EPIC/UniDexGrasp/tree/36c9bfcf7cedc987dd15e2384c548e51d5aadc21),
[worker](../scripts/infer_unidexgrasp_proposals.py),
[adapter](../src/grasp_failure_prediction/proposals/unidexgrasp.py).

**Measured reference bridge:** The exact registered DexGraspNet reference asset
matches the certificate. Across 112 synthetic static joint configurations,
maximum tip error is 1.751 mm (little), thumb 0.422 mm and the other tips
approximately numerical zero. Two all-bound corner configurations slightly
exceed target limits and would be rejected by execution preflight rather than
silently clipped.
This tests the shared reference bridge, not a real UniDexGrasp-generated pose
or its unavailable released generation hand asset.

The certificate covers body/joint/tip kinematics, not collision geometry,
friction or inertia. Source capsule/box surfaces and target URDF collision
surfaces differ. Matching tips within the maximum-per-tip 2 mm gate does not
establish equivalent contact patches or stability. Comparison must identify
pre-TTA versus refined TTA outputs; replacing the native learned policy with
our fixed executor changes execution semantics.

The certificate's scope was checked explicitly: doubling a source proximal
capsule radius and changing palm inertial mass leave the kinematic certificate
unchanged, while changing a fingertip-site offset changes it. Raw asset hashes
change in all three cases. These controlled mutations occur only in temporary
copies and demonstrate why source-asset identity and kinematics cannot be used
as a surface/dynamics equivalence certificate.

Pure upstream FK parser and MuJoCo agree on joint order and body positions to
71.5 nanometers across 18 synthetic poses each with default and nonidentity
root placement. This checks parser/frame consistency on the reference proxy,
not availability of the actual released UniDex hand asset.

**Verdict:** Reference kinematic bridge has quantitative evidence. Actual
UniDex asset acceptance, generated pose fidelity and contact preservation
remain pending. The full original UniDexGrasp system is not reproduced.

For externally imported refined `tta_hand_pose`, the current contract requires
a ContactNet checkpoint hash but does not bind its full TTA configuration.
That provenance gap should be closed before using such imports in a comparison;
it does not affect the current worker's explicitly pre-TTA stage.

## Acceptance evidence still needed

For each actual native sample, compare the same object at its exact registered
scale/frame; retain source units, centering, canonical transforms, joint order,
stage and source asset identities. Verify declared transforms against actual
input surfaces rather than relying solely on copied geometry hashes.

Measure nominal palm position/orientation, every fingertip, intermediate finger
segments and contact-bearing surfaces in the shared object frame. Report native
and translated object distances, contact regions/normals, self-intersection and
penetration. MANO retargeting errors need to be judged relative to the object's
contact geometry; the current 10 mm mean gate is not a scientific contact
acceptance criterion. Report each rejection and its reason separately.

Record controller-induced changes separately and preserve v2 unchanged while
auditing nominal mappings. Any later executor change needs a new protocol and
comparison with the historical one. A static pose check cannot prove identical
physical grasp behavior across different hand surfaces and simulators.

The HUG-only finer-grid and anatomically matched surface follow-up is complete,
and numerical transform/joint consistency extends to 32 existing poses; see
[the continuation](HUG_MAPPING_CONFIRMATION.md). Contact fidelity remains
unconfirmed. Wrist/palm proxies on all 32 poses and surfaces on six existing
apple/milk-box/mug cases are now checked; see the new section in
[the HUG note](HUG_MAPPING_CONFIRMATION.md). A common translation does not
explain the residuals, and native/target surface proximity differs on both
collision and visual references. The object collision approximation itself
also differs from the visual mesh. Next audit that geometry and define a
contact-preservation objective and separate calibration/validation groups
before changing HUG's mapper. HUG remains active before advancing providers.
The proposed cross-generator prediction study remains unrun; no new evaluation
data or fitting accompanied these static checks.

## Audit artifacts and reproduction

Reports, the existing-HUG overlay and snapshot scripts are in
[the audit artifact directory](../research_artifacts/grasp_mapping_audit_2026-10-14/).
Scripts retain the exact local input/source paths used; they are historical
diagnostics, not provider inference or a new dataset exporter. Reports distinguish
real saved HUG output from synthetic FK tests and record no task labels.


## Original-generator evidence and proposed common validation framework

Reviewed 14 October 2026. The user explicitly agrees that faithful mappings
must preserve poor native proposals, and that mapping fidelity must be assessed
before task success/failure labeling. The following is a literature-backed
protocol proposal, not a completed four-source validation or a frozen scientific
acceptance threshold. HUG remains the active individual audit; reviewing other
sources here does not advance their runtime or mapping acceptance.

### What the original work establishes

Original papers define the native representation and measure generated-grasp
quality. Released code additionally fixes operational frames, assets, joint
order, normalization and output stage. Their evidence motivates our checks;
it does not certify this project's adapter, Shadow geometry or controller.
Grasp generation can admit several useful configurations for an object, so
pose distance to one reference grasp is not a universal generation-quality
criterion. Mapping compares each proposal against its own declared native
output. Generation quality and mapping fidelity stay separate.

**HUG:** [Section 5.2 and Appendices E/F](https://arxiv.org/html/2606.17054v1)
define a MANO-derived simulation hand, fingertip/object distance, penetration
and miss diagnostics, a recorded-human oracle and displacement sensitivity.
Robot retargeting is separately described for Ability/WUJI, not our Shadow.
For our argument, reproduce the native MANO exactly, validate camera/object
placement, and quantify how object-relative finger/palm geometry changes under
retargeting. Measure missing proximity as well as penetration; a low penetration
value alone can describe a hand that misses the object. The original oracle
and success numbers cannot establish our conversion. Our current HUG evidence
is geometric, not measured human contact-force ground truth.

**ContactGen:** [Sections 4, 6, 7.3--7.4 and 7.7](https://arxiv.org/html/2310.03740v1)
define contact/part/direction maps and the solver producing a MANO-compatible
hand. Native reconstruction is assessed with mesh EPE/AUC/F-scores; generated
quality uses contact, overlap, gravity displacement, diversity and human ratings.
Direction denotes a bone-local ray, not a surface normal. Our argument should
separate sampled contact-to-native solver fidelity from refined MANO-to-Shadow
fidelity. Check optimized shape/pose reconstruction and lost/new finger/object
regions. Native reconstruction distance thresholds and generation-quality
measurements are not cross-hand acceptance limits. Current inputs are geometric
proxies, not force annotations. Preserve touch-only, missing-contact and
penetrating proposals.

For ContactGen, pinned [demo.py](https://github.com/stevenlsw/contactgen/blob/1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a/demo.py)
and [contact_optimizer.py](https://github.com/stevenlsw/contactgen/blob/1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a/contactgen/contact/contact_optimizer.py)
govern the executed solver. The paper describes freezing global pose in stage
two; that code keeps global rotation/translation trainable. Document the pinned
behavior. Locally, centering, shaped wrist origin and metric units are explicit,
but sampled points/contact/part/direction maps are not exported. Add that export
before future generation if we intend to substantiate source contact-to-pose
fidelity. No export or solver change is implemented by this literature review.

**DexGrasp Anything:** [Sections 3.1--3.4, 4.1, 5.1 and Appendix C](https://arxiv.org/html/2503.08257v2)
define articulated Shadow proposals, proximity/penetration constraints,
physics-guided sampling and force/displacement quality evaluation. These motivate
paired native/translated geometry measurements; successful-grasp curation does
not define permissible mapping error. Preserve the chosen sampling stage and
report changes to inner-finger/object proximity, object penetration and hand
self-intersection. Root placement, wrist articulation and scale require the
actual released model, not the assumption that all Shadow variants are equal.

Its [pinned visualizer](https://github.com/4DVLab/DexGrasp-Anything/blob/86353d710ca71d0f547f0d8b15cc8ca959c78163/models/visualizer.py#L119)
and [hand model](https://github.com/4DVLab/DexGrasp-Anything/blob/86353d710ca71d0f547f0d8b15cc8ca959c78163/utils/handmodel.py#L225)
require denormalization and a 24-joint forearm-to-palm transform. The
[force test](https://github.com/4DVLab/DexGrasp-Anything/blob/86353d710ca71d0f547f0d8b15cc8ca959c78163/envs/tasks/grasp_test_force_shadowhand.py#L248)
preconditions finger angles and settles before force evaluation. Paper and
recommended test learning rates differ, as explained in the
[official README](https://github.com/4DVLab/DexGrasp-Anything/blob/86353d710ca71d0f547f0d8b15cc8ca959c78163/README.md).
Keep generation, preconditioning and execution separate. Our source-specific
decoder/mapping remains pending; no force test is run here.

**UniDexGrasp:** [Sections 3.1--3.2, 4.2 and Appendices A/C.3](https://arxiv.org/html/2303.00938v2)
define canonicalization, IPDF/Glow proposals, ContactNet adaptation and a separate
execution policy. Proposal quality uses analytic wrench, penetration and
diversity metrics; dataset curation tests physical stability, and execution
separately measures configuration following through FK. Our
argument must identify pre-TTA versus refined output, invert canonical rotation,
and verify actual source-hand kinematics and surfaces. Success-conditioned pose
following does not characterize all proposals; our static fidelity audit should
cover every evaluable selected output. Its quality filters are not mapping gates.

The pinned [eval.py](https://github.com/PKU-EPIC/UniDexGrasp/blob/36c9bfcf7cedc987dd15e2384c548e51d5aadc21/dexgrasp_generation/network/eval.py#L100)
retains raw and adapted pose fields. Our worker deliberately exports pre-TTA
IPDF/Glow output. The source asset, canonical transforms, joint order and
refinement configuration therefore remain part of the identity of a proposal.
The existing reference certificate checks kinematics, not collision surfaces;
actual native-output acceptance remains pending.

### Recommendation: shared questions, source-specific tests

Use one reporting framework across four sources, with a source-specific
contract explaining each representation and stage:

1. **Recover the actual proposal.** Bind native files, assets, input geometry,
   source/checkpoint/configuration, units, frames and stage; reconstruct native
   geometry independently where possible. Retain raw and refined variants as
   distinct records whenever the selected pipeline produces both.
2. **Verify the conversion.** Check transforms, root/palm convention, joint
   names/signs/limits and independent FK. Require numerical agreement where
   identical models permit it; state the approximation where hands differ.
3. **Measure geometric changes.** Compare the same proposal and exact object
   in a shared object frame: per-finger/palm points, orientations, signed
   distance/miss profiles, preserved/lost/new proximity regions and penetration
   changes. Label inferred contact as geometric proximity; retain denominators,
   unmatched regions and sampling sensitivity. Audit visual/collision geometry
   separately. Individual collider-surface queries inside overlaps are not
   exact exterior-union distance or physical contacts.
4. **Evaluate execution separately, later.** Keep nominal conversion,
   controller commands and achieved state distinct. Eventual outcomes describe
   the mapped grasp under the declared simulator hand/controller/physics.
   Source-specific quality numbers from different original test systems are
   not pooled as a common outcome comparison.

A shared framework does not imply identical numerical tolerances or identical
source hands. Justify acceptance criteria from geometry and intended use,
freeze them before independent validation, and separate calibration groups.
Do not select thresholds because currently inspected cases pass. Existing HUG
cases are development evidence; later independent validation remains proposed.
No scientific threshold, collection count or outcome-sensitivity experiment
is frozen by this note.

Keep a record of mapping status separate from eventual task outcome. A source
thumb that misses the object can be faithfully mapped and subsequently fail.
A mapping that silently moves it into contact has altered the proposal even
if the grasp improves. Malformed, out-of-limit or unrepresentable proposals
must retain explicit rejection reasons; an adapter rejection is not a simulated
physical failure label. Quality improvement is permitted only as a declared,
versioned refinement stage, not hidden within coordinate conversion.

**Immediate next step:** Apply this proposal to HUG's existing static evidence:
declare contact/proximity correspondences and comparison baselines, resolve
the collision approximation question, then justify an independent validation
plan. Other generators await their individual audits. No new inference,
simulation rollout, task label, evaluation dataset, model fit, production-code
change or manuscript edit accompanies this review.


## 14 October 2026 — HUG object/reference measurement update

[The HUG continuation](HUG_MAPPING_CONFIRMATION.md) now tests source/compiled
object geometry, released/static reference retargeting on all 32 saved poses,
and surface/signed-profile comparisons on the six fixed noncube cases. No tested
scale/frame bug is found; inherited internal gap walls and exterior convex
approximation remain distinct. Better reference tip residuals do not establish
better finger surfaces: current whole unsigned 2 mm proximity is higher in all
six pairs, with per-finger tradeoffs. All native area profiles already have
sampled penetration. Poor proposals remain included.

[The HUG measurement framework](HUG_STATIC_VALIDATION_PROTOCOL.md) separates
implemented measurements from proposed margins/independent validation. Numerical
conversion is supported; cross-hand contact and failure-label equivalence remain
unconfirmed. No other generator acceptance, inference, physics, eval dataset,
model fit or production/manuscript change occurred.


## 14 October 2026 — HUG orientation/uncertainty acceptance result

[The HUG continuation](HUG_MAPPING_CONFIRMATION.md) completes all-32 exact
native 3D decoding and object-facing normal/cell-radius proximity checks on the
six fixed noncube cases with three finer repeats. [Strict acceptance
V1](HUG_MAPPING_ACCEPTANCE.md) was frozen before new metrics: numerical
conversion passes within scope; definite same-part lost/added relationships
conditionally FAIL the strict declared-band claim in all nine resolutions.
Reliable apple/milk sampled facing-state reversals add point witnesses; missing
flips or stable means cannot certify continuous orientation.

This is an implemented approximate HUG-to-Shadow mapping with quantified
distortion, not validated native-contact or failure-label equivalence. The rule
defines a strict project claim and does not establish that every useful
retargeter requires zero distortion. Approximate predictor-use adequacy and
independent validation remain untested. No other provider acceptance, new
inference, dynamics, eval data/labels, fitting or production/manuscript change
occurred.


## 14 October 2026 — HUG surface correction remains a development candidate

[The concrete correction](HUG_SURFACE_REFINEMENT.md) adds native-relative surface
fitting and a guarded target selection on six existing HUG poses. Geometric
objective and whole unsigned proximity improve, but fingertip/middle residuals
worsen, some added regions/directions regress, and unchanged strict preservation
still FAILS on all six cases and three finer repeats. Correct frames/joints/FK
remain distinct from human-to-Shadow morphology preservation.

The new APIs are explicit static research tools; the default mapper is unchanged.
No native poor proposal is deleted or repaired toward a success target. No new
eval data, inference, physics, task labels, predictor training or manuscript edit
occurs. HUG remains active, other generators await their individual audits, and
the multi-generator predictor claim still needs independent later evidence.
