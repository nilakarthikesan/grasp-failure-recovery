# HUG retargeter implementation repair — 15 October 2026

Two actual implementation defects were reproduced and fixed in a new explicit static profile: proposal-to-proposal state leakage and a scalar/gradient inconsistency in the installed Dex runtime. The repaired profile is deterministic on the bounded saved-input test. It is not yet an acceptable replacement mapping under the project's original per-tip nonworsening criteria. The legacy profile remains the default and all historical outputs are preserved.

## Evidence and causal scope

`ShadowHandRetargeter` disables filtering but previously reused Dex's `last_qpos` between independent calls. `SeqRetargeting.retarget` both initializes from and updates that state; its public `reset()` restores the joint-limit midpoint. Twenty repeated solves within a call do not remove dependence on the preceding grasp.

The installed `VectorOptimizer` returns only its Huber geometry loss, while its supplied gradient additionally contains `2 * norm_delta * (x - last_qpos)` with `norm_delta = 0.004`. The scalar does not include the corresponding quadratic term. A fixed interior-state central-difference check reproduced a maximum discrepancy of **0.00167551608**. Subtracting that unmatched term leaves **1.08e-11** residual. Setting `norm_delta = 0` at the same fixed state leaves the scalar unchanged and matches its finite-difference gradient to **1.08e-11**. This is an implementation consistency finding, not merely a proposal to tune a loss weight.

The A–B–A probe used only saved HUG proposals at indices 8 (Apple A) and 28 (Mug B). Exactly five mappings were run: fresh A, fresh B, and A–B–A on one reused instance. Fresh A and the first reused A agree bit for bit. After B, the same A differs by up to **0.192416504 rad / 11.0247°**; B after A differs from fresh B by **0.149019539 rad / 8.5382°**. All five terminal NLopt codes were positive, demonstrating that a positive termination code alone did not prevent either issue.

Current `run_case` and `translate_mano_with_official_retargeter` create a fresh instance for each proposal. Therefore the reuse defect is a real reusable-API bug, but this experiment does not show that it caused the six existing saved mappings to fail. The gradient inconsistency is present even for fresh instances.

## Implemented versioned repair

Only `src/grasp_failure_prediction/evaluation/retargeting.py` and `tests/test_retargeting.py` were changed.

- `legacy_shadow_vector_v1` remains the default, preserving its previous numerical path.
- Opt-in `independent_static_shadow_vector_v2` calls Dex's public reset once before each independent proposal and sets `norm_delta = 0`. Repeated solves within that proposal still warm-start from each other; the existing twenty-call budget, unit scaling, disabled filter, geometry objective, link correspondence and joint-limit handling are unchanged.
- The static profile requires readable optimizer termination status after every inner solve and refuses nonpositive, invalid or unavailable status. It records all twenty status codes. Positive budget-limited codes remain visible and are not represented as scientific acceptance or proven convergence.
- Optional frozen pose provenance records profile ID, configuration path/hash, effective vector scale/filter/temporal term, reset policy, repeated-call count and status codes. Existing pose construction and provider APIs remain compatible. This is effective-settings provenance, not a complete dependency-source attestation or a convergence certificate.

The installed Dex package and third-party assets were not edited. Source preimages are retained as `*.before`; proposed/applied files and their hashes are retained beside them. The first probe's original production source hash is resolved by its saved preimage after the authorized source edit. Its frozen protocol and results were not rewritten.

## Corrected saved-input check

A second plan was frozen before running the same five calls once with the new profile. Fresh/reused A and fresh/reused B are **bitwise identical**, with maximum joint difference exactly zero for all comparisons. All 100 inner solver statuses were positive. Ten additional source-URDF Pinocchio FK calls measured the five corrected outputs and the five already-recorded legacy outputs; there were no new legacy mappings or simulator surface queries.

| Saved proposal | Legacy mean fingertip error | Static-profile mean fingertip error | Fingertips that worsened |
|---|---:|---:|---:|
| Apple A, index 8 | 5.844768 mm | 5.913460 mm | 3 / 5 |
| Mug B, index 28 | 6.608605 mm | 6.581097 mm | 1 / 5 |

These are palm-relative, source-URDF kinematic errors, not the independent simulator full-surface audit. Both means remain below 10 mm, but four of ten individual fingertips worsen relative to the corresponding fresh legacy outputs. This does not justify selecting the new profile as a scientifically accepted mapping. Numerical correctness and reproducibility are necessary foundation repairs; they do not by themselves solve cross-morphology geometry preservation.

## Convention checks and remaining modeling choices

The MANO-to-operator matrix is exactly the transpose of the installed Dex right-hand operator-to-MANO matrix. In all six saved native files, landmark 0 equals `T_camera_wrist` translation exactly. No hidden wrist-origin offset was found in those inputs.

The stock config explicitly maps human landmark 0 to Shadow `palm`, and five fingertip plus five intermediate landmark vectors to ten robot links. The project intentionally overrides stock scaling 1.2 to 1.0. These are modeling choices; unit scaling does not turn Shadow anatomy into MANO anatomy. The current objective has no object geometry, contact preservation, palm-surface or normal-direction terms. Its fixed wrist-to-palm anchor is a separate assumption. Changing that correspondence, adding rotation freedom or preserving object-relative surfaces requires an independently specified mapping method, not relabeling these reliability repairs as complete validation.

## Verification and retained evidence

Sixteen focused retargeting/pose-validation tests passed in 3.02 seconds. They include real-runtime synthetic A–B–A invariance, a real scalar-gradient derivative check, legacy-default settings, and finite stale-pose refusal for failed/zero status, unavailable introspection, and positive budget-status recording. One existing dependency deprecation warning occurred.

The original five-call probe took 0.265 seconds after runtime import; the corrected probe including source FK metrics took 1.915 seconds. Both have pre-execution source-bound protocols. Neither produced model inference, physics steps, new evaluation data, surface queries, training data or a new default selection. This report makes no surface-fidelity or mapping-signoff claim.

Concurrent-edit scope: root reported a subsequent edit to `pose_validation.py` by another thread. These saved probes use Dex source-URDF FK and do not import that module. The sixteen-test log is retained as the result at its execution time; it does not validate later concurrent edits, and the validator dependency was not separately hash-frozen in the test receipt. No probe or test was rerun and no historical source binding was silently replaced.

Independent final source/result review passed40 hash, array and provenance checks. Receipt SHA-256: `a059c85b76188a101eb0fc5789b9ea9be5dad40aa34108c9c8d9acf270ce13cb`; the bound receipt is included in this bundle.
