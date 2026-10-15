# Independent collision-contract and exposed-geometry review

Review completed 15 October 2026. All review stages pass within the scope below. No production code, native proposals, historical criteria or scientific results were overwritten. This folder is ready to archive as a complete unit.

The compiled-geometry inspection found no missing convexification, collision-mask exclusion, nonzero contact margin, or transform defect in the six saved scenes. It used six static `mj_forward` calls, with no new hand/object queries or simulation steps. The historical metric correctly measures its declared union of individual collider surfaces, including buried interfaces; it is not generally the exposed boundary of the occupied union. See `contract_findings.md` and `compiled_contract.json` for version-pinned sources and evidence.

The reviewed new diagnostic uses no rays, CSG reconstruction, fitting or additional object grids. Its twelve fixed poses are the original and previously pooled pose for each of six existing HUG grasps. All 178 saved proximity witnesses and 134 saved orientation witnesses remain in source order, including duplicates. It evaluates 624 row-pose records through 84,240 component-point evaluations. The actual diagnostic used 36 static `mj_forward` calls and zero integration steps. No new eval data or inference was generated.

`diagnostic_preflight_review.json` binds the final draft, exact native/source rows/states and unchanged acceptance hash. Fifteen primary manufactured controls and ten independent controls on the final classifier pass. The earlier ten-control receipt is preserved separately; it is a previous-code check, not ten additional final tests. `independent_union_controls.json` contains six analytical examples explaining union semantics.

`diagnostic_output_review.json` independently replays source pointers, frozen values, caps, saved FK agreement, finite counts, exposure/bound arithmetic, persistence and lifecycle hashes. Its 15,648 assertions are bookkeeping/arithmetic checks, not independent scientific samples. This output replay performs no new geometry or FK calls. The complete component closest-point oracle is not recomputed; geometric correctness is supported separately by the compiled contract and manufactured controls. A reviewer-only quaternion schema-key correction is disclosed in `output_review_correction_history.json`; no measured value changed.

| Existing grasp | Original persistent witnesses | Pooled persistent witnesses | Retained proximity witnesses |
|---|---:|---:|---:|
| Apple A (8) | 31 | 31 | 33 |
| Apple B (12) | 15 | 10 | 15 |
| Milk A (16) | 35 | 35 | 37 |
| Milk B (20) | 12 | 6 | 14 |
| Mug A (24) | 46 | 19 | 57 |
| Mug B (28) | 17 | 13 | 22 |
| Total | 156 | 114 | 178 |

All six original and pooled poses retain conditional whole-cell counterexamples against the historical native reference. Original retained witnesses preserve 98 lost-near and 58 added-near counterexamples; pooled witnesses preserve 71 and 43. Across both roles, 575 requested-part nearest points are classified separated/exposed and 49 buried. No oracle inconsistency, exterior-burial contradiction or bound-order conflict occurs. All 134 pooled orientation records have exposed requested-part nearest points, but no new facing-sign or normal-uniqueness claim was made.

The 156-to-114 counts are **not an improvement score**. They concern capped, duplicated source witnesses, and candidates may produce intervals crossing thresholds or lack exposed upper witnesses. Failure to certify persistence is not repair. Total mismatch area and independent generalization are not measured here. `aggregate_review.json` independently matches the descriptive totals and supplies one deterministic illustrative surviving witness per pose.

All bounds are conditional on the reconstructed geometry, signed-distance oracle and floating arithmetic; they are not interval-arithmetic proofs. Native MANO skin is not certified as an exposed non-self-intersecting solid. An occupied-union boundary is not equivalent to MuJoCo's pairwise contact solver, force closure, finite-object accessibility or grasp success. The nine previous strict surface failures remain unchanged. This diagnostic does not certify complete HUG-to-Shadow mapping.

`final_receipt.json` binds the final review reports. `bundle_manifest.json` lists every file in this folder except itself. No further writes are planned after manifest creation.
