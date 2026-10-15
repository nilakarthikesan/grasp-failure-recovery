# HUG canonical palm calibration: completed development diagnostic

15 October 2026, America/New_York. **This calibration is not an accepted mapping.** All six retargets pass numerical reconstruction and the 10 mm mean-tip guard, but 21 of 30 original individual fingertip caps fail. A separate whole-cell argument also refutes strict palm preservation at both saved mug anchors for any palm orientation. Production and scientific acceptance remain unchanged. No new native HUG proposal, physics, eval data, label or training was produced.

## Why test a new placement

The [previous guarded study and fixed-anchor certificate](HUG_GUARDED_FIT_AND_REACHABILITY.md) show that more finger/wrist fitting cannot repair one saved mug palm at the original native-wrist-to-Shadow-palm position. A [read-only source audit](../research_artifacts/hug_anchor_convention_audit_2026-10-14_v1/README.md) finds that this origin correspondence is intentional. The source URDF's 34 mm wrist-to-palm joint offset is already composed by the runner; no missing 34 mm transform is demonstrated. The earlier 32-case MCP residual-subtraction diagnostic was never applied and worsened hypothetical tip errors; it was not a successful calibration.

We therefore tested a new source-only anatomical mapping hypothesis, preserving all original results. It uses one fixed rigid transform shared by all six existing development grasps, with no object geometry, failure witness, contact objective or per-case offset used to select that transform. The anatomical proxies are explicit assumptions, rather than known true correspondence between human skin and robot colliders.

## Frozen construction and implementation

Use the exact HUG right MANO wrapper/manotorch layer, centered at wrist 0, flat mean, no PCA, CPU float32, explicit zero 48D axis-angle pose and zero 10D shape. This creates one neutral geometric template for calibration; it does not call HUG's neural grasp generator or create a new native proposal. Metric scale remains1.0 meters. Verify MANO21 MCP joints 5/9/13 against raw regressor joints 1/4/10. Target proxies are rigid palm-parent FFJ4/MFJ4/RFJ4 origins; the movable little metacarpal is excluded.

For each hand define a frame at the three-MCP centroid, with longitudinal axis from wrist/palm origin to centroid, lateral index-minus-ring direction projected orthogonally, and the remaining axis completing a proper right-handed basis. Freeze the signs, neutral pose, shape, 1 micrometer conditioning limits and no-scale rule before calculation. With frames F_M and F_P, C = F_M inverse(F_P) maps **neutral palm coordinates into MANO coordinates**. It is identical for all cases.

The fixed-URDF base B and neutral palm P0 have rotation R_BP0. For native wrist-frame row points L_M:

    reference_B = (L_M - C.translation) @ C.rotation @ R_BP0.T
    world_from_fixed_base_R = R_world_MANO @ C.rotation @ R_BP0.T
    palm_anchor = wrist_world + R_world_MANO @ C.translation

Let T_BF be the source fixed-base-to-forearm transform. With R_WB = world_from_fixed_base_R, T_WB.translation = palm_anchor - R_WB @ p_Bpalm(q), and the floating forearm pose is T_WF = T_WB @ T_BF. Root translation therefore compensates the actual q-dependent root-to-palm FK and retains the source world_joint transform. Actual palm rotation remains R_WB @ R_BP(q); world_MANO @ C describes neutral placement, not the actual palm orientation for all wrist angles. Neither a second MANO/operator rotation nor subtracting the human wrist again is permitted. Original native WORLD landmark targets remain unchanged.

Use one fresh existing Dex vector retargeter per case, metric scaling1.0, no temporal filter, its initial joint-limit-midpoint state and exactly 20 repeated calls. The new direct retarget uses full physical joint domains; it does not claim the previous q0 +/-0.35 correction restriction. Preserve all 120 raw vectors/statuses, existing numerical physical-limit clamping policy, all old name/value pairs and original anatomical fingertip caps. These six experimental robot retargets are new configurations of six existing native grasps, not new evaluation episodes.

## Measured outcome

The canonical translation has norm **8.668706 mm**. All 120 NLopt calls return supported FTOL status. All six pass physical-domain, original native-world reconstruction, source-URDF versus captured-scene FK, calibrated-anchor and strict 10 mm mean-tip checks. Their mean errors in inventory order 8/12/16/20/24/28 are **6.546, 6.498, 6.837, 9.634, 7.266 and 7.182 mm**.

Only **9/30 individual original tip caps pass; 21/30 fail**, so **0/6 candidates are eligible**. The 21 failed-cap increases range from 0.023897 to 5.044260 mm; caps are never rebased to the new placement. No case is discarded or called a physical grasp failure. The maximum new-vs-old joint change is 0.546381 rad, explicitly belonging to this new full-domain mapping experiment. The largest final physical clamp is about0.001000007 rad, within the unchanged documented 0.00101 rad raw-optimizer numerical margin.

Native-world reference reconstruction errors are at most 1.257e-8 m, fixed-URDF versus captured-scene task errors at most 6.339e-8 m, and calibrated anchor errors at most 3.104e-17 m. These are engineering/numerical reconstruction facts, not surface-preservation tolerances. The separate frozen surface-distance arithmetic assumptions are not enlarged by these admission gates.

A separately frozen independent verification reconstructs C from bound neutral descriptors, source-URDF frames, original/new captured-scene FK, original native WORLD targets, named joint permutations, exact original scene hashes/caps, q-dependent palm orientation and floating-root compensation. It independently reproduces all six numerical passes and all 21 cap refusals. It inherits the primary MANO neutral descriptor; it does **not** claim another independent MANO neutral forward or optimizer replay. No new full-grid surface audit runs on these refused poses.

## New-anchor palm contradiction

After C and the six outputs are fixed, a separate protocol applies the unchanged radial-shell argument to all seven existing palm cells and the same eight actual box colliders. It neither recalculates nor tunes C. All six cap refusals remain visible. Positive results refute only the new fixed anchors; a nonpositive row cannot establish PASS.

For Mug A/index 24, three of five old cells give positive whole-cell witnesses. The strongest is the existing 1 mm-grid child 16928: its native distance upper bound is 0.971673 mm at the 1 mm band, but every orientation of the palm at the new anchor remains at least 8.135419 mm away. Its cell area is 0.9748735 mm². For Mug B/index 28, both old cells are positive; child 17802 remains native within 4.745913 mm at the 5 mm band, while every rotated new-anchor palm stays at least 15.091851 mm away. Its area is 3.9636778 mm². The maximum strict band excesses are **7.135419 and 10.091852 mm**, respectively, after the same declared allowances.

A separate independent 90-digit replay passes **1,093 checks**, retaining all seven cells, all nonpositive margins and every refusal. Its maximum primary-versus-independent margin difference is4.71e-17 m. These five positive rows include repeats across sampling resolutions; they are not five independent grasps or disjoint physical contact areas.

As before, the result is conditional engineering arithmetic for literal individual palm-box surfaces, including buried interfaces, under the declared part correspondence and cell/radius assumptions. It is not formal outward-rounded floating-point proof, an exterior-contact proof, a native failure label, or a generalization result. It refutes these two new fixed-anchor formulations for the two saved mug grasps. Arbitrary other translations, different geometry/correspondence or all HUG proposals are **not** ruled out. Additional finger fitting at these same anchors cannot move those palm boxes and cannot repair this obstruction.

## Preserved runtime history and next research choice

V1 stops after computing the neutral template/C in memory, before exporting C or retargeting a case: the fixed-URDF compiler fuses the forearm body, defeating the original body lookup. V2 derives the already declared transform from the source world_joint and verifies the neutral joint chain, then exports C; it stops before any case retarget because saved and Dex joint orders differ. V3 explicitly maps the 24 unique joint names, preserves historical pairs, and requires float64-identical C to V2 before any fit. All C definitions, assets, objectives, caps and tolerances remain fixed across these runtime fixes. Both aborted histories and pre-execution reviews are retained.

Current status: the original mapping still has its recorded nine strict FAIL results; the new canonical experiment has zero eligible retargets and two separately refuted fixed mug anchors. No new nine-resolution full-grid count is asserted for the canonical experiment. It is neither promoted nor silently reinterpreted as native failure preservation.

The next implementation depends on the intended prediction target. Keeping a common Shadow embodiment requires an explicitly versioned per-grasp placement/retargeting policy, with recorded wrist displacement, original native targets and unchanged fidelity measurements; its label describes the executed Shadow grasp. Preserving HUG's native MANO hand instead requires a separate native-hand simulator representation and collision/dynamics validation. Neither option is implemented or accepted by this report. The target choice has been requested from the user; dependent implementation remains pending. New physics/eval data remains outside the present authorization.

## Evidence

- [Primary calibration and all runtime histories](../research_artifacts/hug_canonical_mcp_calibration_2026-10-14_v1/README.md). V3 protocol `68d229d7620bb5aa4b1abef3e1a158c4430065f4908dd2277456b1a5186715ca`; calibration `2c07c84ec8699e9e4214311a431401b0f19724070d27aeeb1220f6270bf2eed5`; six-output report `84496804c00267024af56f3364457943696db8481b9c1853bca2c93c6c824b9d`.
- [Independent pre-calculation review](../research_artifacts/hug_canonical_calibration_preflight_review_2026-10-15_v1/README.md).
- [Independent source/C/FK verification](../research_artifacts/hug_canonical_mcp_independent_verification_2026-10-14_v1/README.md), protocol `af2756ff035ffa2583eb30dc6c9e9270f7984933f2b26c16b955e80a82d40050`, report `8b79219b1b83f8ff75ab390010bec7bf6a84c7e7f144d3e59b33712bcaefa5b8`.
- [New-anchor radial control](../research_artifacts/hug_calibrated_anchor_radial_control_2026-10-15_v1/README.md), protocol `e02f6ef5e4715968631aa7ec01f10d6d3afe0fb99991bafbda9cdaf71411df74`, report `6e908767f32877bb8dbad3cb4794fd938fb2090b3d0f357eb942e8279b1dbd55`.
- [Independent new-anchor radial replay](../research_artifacts/hug_calibrated_anchor_radial_independent_verification_2026-10-15_v1/README.md), protocol `fd8e4d367ce8012a8485cffb119557c1e8e8ac02af7ef7fcc809a6e423894f54`, report `c902579ebf52ad00bea474856885d121e481934565398a442c61a61de7b853ea`.

Some artifact names retain the 14 October investigation-series date; their actual new phase/freeze timestamps are 15 October. They do not merge older protocols or rewrite historical results. Source-only calibration, actual numerical reconstruction and independent radial verification have separate scopes. No training, held-out validity or native outcome equivalence is claimed.
