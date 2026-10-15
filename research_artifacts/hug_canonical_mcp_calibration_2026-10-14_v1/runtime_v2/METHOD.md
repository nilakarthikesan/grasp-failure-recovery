# Source-only canonical MCP-frame calibration diagnostic

15 October2026 America/New_York. This new mapping experiment uses a single declared anatomical transform and the same six existing saved HUG grasps. It is not a source-frame bug fix, new HUG inference, new grasp, physics run, failure label, eval data or training. Shadow geometry and strict same-part surface acceptance are unchanged. Production is untouched. No object/contact witness informs calibration or retargeting. Existing inspected cases are development data.

## Explicit source template and frame signs

Use the frozen original HUG MANO wrapper and its existing manotorch ManoLayer, side=right, rot_mode=axisang, center_idx=0, use_pca=False, flat_hand_mean=True, CPUfloat32. Pass explicit ten ZERO betas and48 ZERO axis angles, including global rotation. Axis-angle zero is identity; no all-zero6D rotation is used. This is a new neutral geometric template solely for global anatomical calibration, not a generator grasp or eval input. There is no smplx substitution and no neural generator/checkpoint is loaded. The asset/native wrapper outputs meters at scale1.0; do not multiply/divide by1000. Layer and wrapper apply the same asset-metric output convention. Verify wrist0 is centered and MANO21 MCP rows5/9/13 equal raw regressor joints1/4/10 minus shaped wrist0 within the predeclared source numeric allowance.

Source landmarks are neutral MANO MCP index5, middle9 and ring13. Target homologues are sourceURDF FFJ4/MFJ4/RFJ4 joint origins, each rigidly attached to palm with declared zero RPY; compile-neutral FK checks this correspondence. No object mesh or observed case is used to derive this transform.

For BOTH hands place frame origin at that hand's three-MCP centroid mu. The longitudinal z axis is normalized mu−wrist/palm-origin. Lateral x is index−ring, projected orthogonal to z and normalized. y=z×x. Require the declared minimum1µm longitudinal and projected-lateral lengths and proper determinant+1; there is no Kabsch rank3 requirement or fitted scale. These signs, points and neutral pose are fixed before calculation.

Let F_M map the common anatomical frame to MANO wrist coordinates, and F_P map it to the neutral Shadow palm body coordinates. Define C_MANO_from_neutral_palm=F_M inv(F_P). Thus C.R=R_M R_P.T and C.t=mu_M−C.R mu_P. It maps neutral palm-model coordinates into source MANO wrist coordinates. A single C is shared by all six cases, regardless of object, pose or actual exported shape. Record all neutral MCP residuals and shape mismatch without tuning the points/frame or C.

## Fixed-base, free-wrist placement family

B denotes the fixed sourceURDF base used by Dex/independent MuJoCo FK; it is NOT assumed identical to the forearm free-root frame in stored scenes. P0 is the actual neutral palm body frame in B, with R_BP0. T_BF is the source fixed forearm pose, including the existing world_joint rotation. For each decoded native wrist-frame row L_M:

    target_P0 = (L_M−C.t) @ C.R
    target_B  = target_P0 @ R_BP0.T
    R_WB_raw = R_WM C.R R_BP0.T
    a_W = p_WM + R_WM C.t
    t_WB(q) = a_W−R_WB p_Bpalm(q)
    T_WF(q) = T_WB(q) T_BF

R_WB is the proper-rotation normalization of R_WB_raw using the existing SciPy convention. Record the normalization delta and retain the predeclared native-transform reconstruction tolerance; never alter a tolerance after seeing outputs. W_M C is the NEUTRAL palm placement. With free wrist joints, actual palm rotation is R_WB R_BP(q), and must be checked against that q-dependent expression, not forced to W_M C.R. Source fixed-base→forearm compensation is explicit. A final translation-only scene-FK compensation removes source-versus-compiled decimal-position differences while holding the same a_W; record its value and independently verify it.

The native wrist remains a separate landmark; it is not relabeled as the new calibrated palm origin. Do not call reference_vectors on transformed landmarks with native wrist index0 as origin: doing so cancels C.t and recreates the old mismatch. Pass the explicitly constructed palm-origin vectors to the existing Dex vector optimizer. Do not apply MANO_TO_OPERATOR_RIGHT again; C already specifies the complete basis conversion. Reconstruct original native world targets directly from T_world_camera and saved camera landmarks for an independent algebra check.

## Retargeting and source gate

Use one fresh existing ShadowHandRetargeter per case. Preserve its packaged vector config and optimizer implementation, disable its temporal filter, use metric scaling1.0, initial default joint-limit midpoint state and exactly20 sequential retarget calls. Retain all20 raw joint vectors, NLopt result codes/values, final physical clamping deltas, source physical bounds and deltas from the old q. Require every one of20 NLopt return codes to be in the predeclared supported positive set{1,2,3,4,5,6}, with finite recorded last objective. Negative/zero/unknown codes or a nonfinite last objective refuse the entire source gate; the library can otherwise silently return its previous q after RuntimeError. Positive finite-budget codes do not establish convergence/global optimality. Do not tune/reseed/retry based on this experiment's results. This new direct retarget version uses full physical joint limits; it does not claim to stay in the old q0±.35 repair domain. It is not a capped-refit/restoration stage.

Load all five authoritative ORIGINAL per-tip caps from the frozen earlier actual-scene baseline, independently reconstruct that original scene pose and record its agreement. Never replace those caps with the new placement's errors. Mandatory eligibility requires each new computed sqrt(einsum) tip norm≤its exact original cap, mean tip error≤10mm, source physical bounds, existing documented raw optimizer bound margin, native-target reconstruction and independent fixed-source-vs-captured-scene FK/root alignment. The gate adds no extra allowance to an old tip cap. Report every new pose/error even if refused; a constructed calibration is distinct from an eligible surface-audit candidate. Preserve every original target/root/result.

No full surface audit runs in this driver. Any refused case prevents blanket all-six go-ahead. Case-level audit decisions remain separate. A new anchor/pose requires genuinely fresh unchanged surface measurement; original-pose audit reuse is invalid even if24 joint numbers happen to match. No calibration or numeric source-gate PASS promotes strict mapping acceptance.

A separately frozen new-anchor radial control is REQUIRED before any full-surface-audit decision after C is fixed. Source-gate eligibility and full-surface eligibility are separate fields: this driver records the radial control as NOT_RUN and full-surface eligibility as false, even when its source gate passes. It cannot choose, tune or reject-and-recalculate C. The current driver never reads object/witness geometry or contact scores for any optimization/calibration decision; only saved scene state and frame transforms are used for placing and independently verifying the new numeric pose.

## Runtime V2 provenance

V1 executed the authorized neutral MANO forward and computed the deterministic source-only C in memory, then stopped because fixed MuJoCo URDF compilation fuses the static forearm body. No calibration file, numerical C value, case retarget fit, or case output was produced. V2 derives the same fixed T_BF directly from the hash-bound source URDF world_joint origin XYZ/RPY; parent/child/type are asserted and its rotation is crosschecked against neutral compiled palm rotation. This is a runtime access fix. Model/shape/pose/landmark/frame signs, C construction, optimizer, budgets, source inputs and gates are unchanged. Its protocol is frozen after that V1 abort and before the retry/first completed calibration and six-case output, not before every calibration calculation globally. The unobserved in-memory C did not tune any choice.
