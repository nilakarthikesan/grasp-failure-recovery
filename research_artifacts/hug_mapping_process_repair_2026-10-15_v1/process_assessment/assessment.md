# Mapping-process diagnosis, 15 October 2026

The original-tip nonworsening rule can restrict a surface correction, but the available results do not establish that it is the sole blocker. The current implementation also omitted an independent root-rotation search degree. A bounded joint plus SE(3) placement method is a concrete next capability to test under the unchanged scientific criteria, rather than repeating another27D fit.

## What the fingertip rule actually means

For native tip target t_i and mapped tip p_i(x), the rule is ||p_i(x)-t_i|| <= c_i, where c_i is the original mapper's error. Each mapped virtual tip must stay inside its original error ball around the native landmark. This is a baseline-relative regression guard, not an independent accuracy tolerance derived from HUG. It does not force the tip to stay at its original incorrect coordinate: movement toward the native target remains permitted. It can nevertheless reject a useful surface tradeoff because a Shadow virtual tip, the native anatomical landmark and an actual contact pad are different correspondences.

All five constraints begin active at the original pose. For nonzero error e_i, the local allowable directions obey e_i^T J_i delta <=0; reducing one tip does not compensate for increasing another. This can conflict with a surface descent direction. It does not prove the full feasible set is a point, and first-order tangent directions alone do not prove finite-step feasibility. Physical joint limits, the local q0±0.35 domain and fixed root rotation impose additional restrictions.

The six cases' individual caps range from0.907775 to17.517526mm. Their original means range5.844763–8.189192mm. Consequently, for these six, all five cap inequalities already imply mean<=10mm; the latter does not explain their refusals. The unequal cap sizes are inherited errors, not evidence that thumb distortion is scientifically permitted to be seventeen times larger than another finger's distortion.

Concrete evidence rules out the claim that the caps prevent all meaningful motion. The pooled Mug A pose moves its palm22.359525mm and changes joints by at most0.081071rad while satisfying every original cap. Twelve later projected endpoints also become exactly tip-feasible, but none passes the protected orientation/rank improvement gate. Ten lose protected orientation eligibility; six lose protected passes. Ineligible orientation rows alone do not establish reversed normals. A separately checked eligible Milk B sample does reverse its native-relative sign at both restored endpoints. No exact pre-projection orientation measurement exists to prove that the projection itself caused those changes.

Thus neither removing the caps nor trying more iterations has an established solution guarantee. If a different tradeoff policy is ever wanted, it requires a prospective new method/claim; the old results must not be relabelled.

## The search and acceptance objectives have different roles

Production retargeting minimizes landmark-vector mismatch through Dex. The experimental surface solver minimizes the largest normalized retained-witness surrogate violation, with the fingertip inequalities enforced by SLSQP. Exact eligibility, buried-point, tie and visual-exterior conditions are assessed at feasible checkpoints, while some optimizer terms are only guidance. Its lexicographic diagnostic rank prioritizes complete row feasibility, then invalid orientation count, worst exact violation, failed row count, then scaled pose norm. This makes gaming eligibility unattractive, but can reject a surface improvement with a new invalid row; that is a declared conservative selection policy, not proof that the candidate has worse physical grasp behavior.

Only178 proximity and134 orientation records were directly constrained; they are capped examples, with no retained orientation row for Mug A. Passing those rows would not prove complete surface preservation. The existing full-grid criteria and the additional preserved-eligibility gate should not be confused with the solver's surrogate or a standard HUG metric.

## Missing independent rotation

The free-palm helper has24 joint coordinates and3 world palm translations. Every staging call fixes the floating-root quaternion to its original bytes. Direct-surface and endpoint-restoration experiments inherit that helper. The Shadow wrist has two revolute axes (WRJ2 and WRJ1); finger joints do not rotate the rigid palm. Two wrist joints span a restricted orientation family and do not provide arbitrary independent root orientation.

Earlier surface studies also retained the carrier rotation. Canonical calibration introduced one fixed template-derived transform, followed by vector retargeting; it did not jointly optimize per-grasp root rotation and translation. The fixed-anchor radial obstruction considers all palm orientations only at those specific anchor positions. It does not exclude a jointly translated and rotated pose. I found no previous joint full-SE(3) placement search among these reviewed studies; this is a statement about the inspected protocol history, not every possible untracked file.

The new isolated capability uses30 coordinates: the same24 joints and3 translations, plus3 world rotation-vector coordinates. Root rotation is Exp(omega)R_original, and root translation is recomputed so the actual q-dependent palm stays at the requested anchor. Zero rotation delegates the exact old stage. The additional tip Jacobian is -skew(p_tip-p_palm)J_left(omega), with the coordinate scale applied. The root-orientation admission check must now verify the declared optimized transform; it cannot continue requiring the historical quaternion. This is an explicit new search-domain version, not a relaxation of tip or surface thresholds.

A±15degree box on rotation-vector components is a prospective engineering search limit, not a15degree geodesic limit: its maximum vector norm is sqrt(3)×15degrees. Retain the original translation reach enclosure, physical/joint-trust limits, native targets/caps and literal scientific metrics. Freeze a finite optimizer budget and protected direction predicates before fitting. Audit only a genuinely selected, independently verified candidate; keep unsuccessful outcomes visible. A failure of this bounded method is not global infeasibility.

## What the prediction thesis needs

If the intended label is failure of the executed Shadow grasp, exact native MANO anatomical-surface preservation is not logically necessary to define that target. The required identity is that the simulator executes the specified, documented Shadow retarget and the model predicts that executed pipeline's outcome. This does not license a claim about native HUG success/failure, universal retargeting fidelity, or sim-to-real accuracy.

Native distortion still matters: it can confound generator comparisons and change proposal diversity. Preserve every original proposal and adapter version, report distortion and mapping refusals, and later compare mapper/generator conditions without filtering for successful grasps. An approximate native-fidelity claim requires separately justified distortion allowances and independent evidence. Neither the original HUG paper nor our source review supplies universal Shadow thresholds. Keep this research-target distinction separate from continued attempts under the existing frozen strict criteria.

No fitting, new native generation, physics integration or evaluation episodes were performed for this assessment. The associated SE(3) implementation is a separate development artifact with explicitly authorized kinematic controls.
