# HUG wrist-to-palm convention: read-only source audit

15 October 2026, America/New_York. This note documents the source inspection
requested during the ongoing HUG validation work. The artifact name retains
the preceding 14 October evidence-series identifier. **No new frame/origin bug
is demonstrated by these inspected bytes.** That conclusion is deliberately
narrow: an explicit convention can still fail anatomical/surface preservation.
No geometry query, FK/model execution, fitting, inference, physics, eval-data
collection or shared-file edit occurred in this audit.

## What the inspected path actually defines

HUG centers MANO at joint 0 (`center_idx=0`) and adds the predicted metric wrist
translation to its joints/vertices; that same translation enters
`T_camera_wrist` [S1, lines 38 and 141–146]. Its native wrist therefore is an
explicit model origin, not an inferred Shadow body.

Dex's packaged Shadow vector configuration explicitly pairs source joint 0
with robot `palm` as every vector origin [S2, lines 6–12]. The project removes
the saved native wrist transform, converts axes and builds those source
origin-to-task vectors [S3, lines 44–75 and 148–157]. The paired vector/pose
rotation is internally coherent [S4, lines 25–27]; this is an algebraic frame
property, not anatomical ground truth. The current placement deliberately
returns native wrist translation as the Shadow palm target, with an explicit
comment that the origin correspondence is versioned [S5, lines 209–222].

The source URDF defines `WRJ1` from the body named `wrist` to `palm`, with an
origin offset `(0,0,0.034)` meters [S7, lines 182–188]. Consequently the body
named `wrist` is 34 mm from the declared palm anchor. The runner evaluates
actual root-to-palm FK and solves root rotation/translation to realize the
requested palm target [S6, lines 476–492]. The 34-mm joint origin is already
inside that composition. A body-name difference does not demonstrate a missing
34-mm translation; adding it would introduce a new anatomical correspondence.

The three proposed rigid nonthumb base proxies have explicit palm-parent
origins: FFJ4 `(0.033,0,0.095)`, MFJ4 `(0.011,0,0.099)` and RFJ4
`(-0.011,0,0.095)` meters [S7, lines 214–220, 346–353, 479–486]. The little
finger has an additional movable metacarpal joint LFJ5 [S7, lines 613–621].
These source facts motivate declared proxies; names/locations do not certify
homology with MANO's anatomical joints.

## What the earlier MCP diagnostic did not establish

The earlier report covers 32 saved native poses, one native hand shape and
original nominal targets. It measured the implemented palm/native-wrist
agreement (maximum 1.4288057044e-16 m) and the 34-mm body separation. Its
hypothetical subtraction of the mean MCP-proxy residual increased mean tip
error from **6.4276257175 mm to 10.3707654227 mm**. It explicitly says this was
pure subtraction in saved residual arrays: geometry was never moved and no
calibration was applied [S8, lines 333–344 and 365–401; H1 JSON fields below].
It was neither a successful placement calibration nor the proposed canonical
construction. Reusing that observed offset as a supposedly validated correction
would misstate the evidence.

## Proposed source-only calibration and independent verification plan

This is a **new mapping hypothesis**, not a known true anatomical correction.
Derive canonical neutral frames from predeclared homologous proxies, e.g.
MANO21 MCP indices 5/9/13 and Shadow's rigid index/middle/ring base origins.
Pin MANO assets, shape, neutral-pose/centering and joint ordering, robot neutral
q, frame-axis rules, units, degeneracy/conditioning refusal rules and weights.
Derive and freeze the transform without an object, mug witness, contact
objective, observed failure residual or per-case offset. Different hand shapes
and proxy choices can yield different calibrations; no single true offset is
identified here.

Use the resolved fixed-carrier/free-WRJ convention. Let C = F_MANO times
inverse(F_palm) map NEUTRAL palm coordinates P0 into native MANO coordinates.
Let R_BP0 be neutral robot-palm rotation in the fixed source-URDF base B.
For native row-vector task points L_M, use

    v_B = (L_M - C.t) @ C.R @ R_BP0.T
    R_carrier = R_world_MANO @ C.R @ R_BP0.T
    p_anchor = t_world_MANO + R_world_MANO @ C.t
    R_actual_palm(q) = R_carrier @ R_Bpalm(q)

Thus actual palm orientation remains q dependent. `world_MANO @ C` labels the
neutral-palm placement, not actual palm rotation for every wrist q. No second
MANO/operator axis transform is added. Subtracting a transformed human joint 0
again would cancel C.t and lose the intended translation.

Before any independent candidate surface audit, separately freeze the method,
C direction/provenance and validation protocol. Independently reconstruct C
from its canonical source descriptors and R_BP0 from source-URDF neutral FK.
Keep native WORLD task targets exactly the original saved camera landmarks
composed with `T_world_camera`; never apply C to scientific ground truth.
Reconstruct source and floating-model FK independently, including root
compensation that realizes p_anchor and R_actual_palm(q). Do not assume the
floating-root frame equals the fixed source-URDF base.

The new alignment gate must compare actual palm placement to this independently
reconstructed calibrated target, and separately record departure from the old
wrist-equals-palm target. Deleting the old equality without a replacement would
not validate placement. Preserve old native-relative fingertip caps separately
from calibrated-q0 errors; freeze their eligibility semantics explicitly and
never silently rebase them. Retain every case/refusal and all original proposals.

If all six cases meet the predeclared independent source/frame/pose/engineering
gates, apply the unchanged six coarse plus three fine scientific comparisons:
same object queries, part definitions, 1/2/5-mm bands, numerical allowances and
normal eligibility. Changed root pose means changed geometry even if all 24
joint values match; that cannot justify reuse of original surface reports.
Lower fitting error or disappearance of sampled reversals is not full-domain
surface/contact equivalence. Failure-label preservation and unseen-object
validity remain separate untested claims. **This independent calibration
verification plan has not been executed in this source audit.**

## SHA-bound source references

`source_evidence.json` stores exact one-based line excerpts at these hashes.

S1: [HUG_wrist_definition](/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/src/models/mano.py:34)

Path: `/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/src/models/mano.py`. Lines: 34–43, 133–146. SHA256: `ebc9113402846d3fd0773e7740e9ff6a70c2e59d317241b576d042b5dd683ed3`.

S2: [Dex_explicit_origin_correspondence](/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/dex_retargeting/configs/teleop/shadow_hand_right.yml:1)

Path: `/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/dex_retargeting/configs/teleop/shadow_hand_right.yml`. Lines: 1–15. SHA256: `e12e9aa5c8f51010793bfa04f3d92ea998b51b8503e5e61330ca8db194f7bf0e`.

S3: [project_vector_transform](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src/grasp_failure_prediction/evaluation/retargeting.py:44)

Path: `/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src/grasp_failure_prediction/evaluation/retargeting.py`. Lines: 44–63, 65–75, 148–163. SHA256: `a6631a0f0d20f6ef94c389529b7d6d677a8a8cd9f8e923efff19ccfee26aa4ff`.

S4: [project_paired_axis_transform](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src/grasp_failure_prediction/integrations/hug_frames.py:4)

Path: `/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src/grasp_failure_prediction/integrations/hug_frames.py`. Lines: 4–12, 23–28. SHA256: `cd7ad9b8df1d5d6adad70e949a9c7f21ad448a25710a54f47e1ffa95fa28562a`.

S5: [project_explicit_palm_placement](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src/grasp_failure_prediction/evaluation/case_runner.py:195)

Path: `/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src/grasp_failure_prediction/evaluation/case_runner.py`. Lines: 195–222. SHA256: `61aeef7a7125fa7698549cfbfc440a4f8374daf289bdddd87deaafe8ce108b11`.

S6: [runner_actual_root_compensation](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src/grasp_failure_prediction/evaluation/runner.py:476)

Path: `/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src/grasp_failure_prediction/evaluation/runner.py`. Lines: 476–492. SHA256: `4d819711660c346938ce468b06b207aaed9d0315ab1ee3dffd9d4c56e1f417e6`.

S7: [URDF_palm_wrist_and_rigid_MCP_origins](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf:102)

Path: `/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf`. Lines: 102–110, 182–188, 214–220, 346–353, 479–486, 613–621. SHA256: `29a50063cced2fba81dfe3d208ba54499fc12635c9189d0822f7caa2771c36de`.

S8: [earlier_proxy_diagnostic_source](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_palm_object_validation_2026-10-14/palm/diagnose_palm_anchor.py:50)

Path: `/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_palm_object_validation_2026-10-14/palm/diagnose_palm_anchor.py`. Lines: 50–51, 78–95, 365–401. SHA256: `7b90948bc4a144417434cdfc68606cb20607a6eb8dbea7f2a89a6153c6f103eb`.

H1: [Historical 32-case proxy report](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_palm_object_validation_2026-10-14/palm/palm_anchor_report.json)

Path: `/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_palm_object_validation_2026-10-14/palm/palm_anchor_report.json`. SHA256: `1e9a605161c8ba542110aa05596496f4272132907eb34e05086646700f70933d`. Mean-tip JSON fields: `global_translation_diagnostics.tips5.original_point_error_m.mean` and `cross_point_family_translation_diagnostic.fingertip_errors_if_MCP_mean_algebraically_removed_m.mean`. No licensed MANO asset is copied.
