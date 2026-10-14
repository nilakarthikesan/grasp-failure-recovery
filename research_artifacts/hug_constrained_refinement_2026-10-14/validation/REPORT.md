# HUG constrained refinement: independent static audit

Date: 2026-10-14. Scope: the same six saved HUG development grasps on apple, milk box and mug, using their original native predictions, meshes, frames, commands and query cells. This artifact contains static geometry checks. It generates no evaluation episodes, generator proposals, outcomes, labels or training data.

The constrained solver returns 6 exact-original fallbacks and 0 accepted surface corrections. A lower raw fit cost does not satisfy the fingertip constraints and is not deployed. The original production mapping and its surface issue remain unchanged.

The final source preflight passes for all six cases: every fingertip error is at most its original error plus the predeclared 10 nm arithmetic allowance, and the actual mean fingertip error is at most 10 mm. The unchanged strict mapping rule returns **0 PASS, 9 FAIL and 0 UNRESOLVED** across six original coarse and three fine resolutions. Engineering preflight and mapping acceptance are separate claims.

## Frozen method and what is independently checked

The joint fit starts from each original 24-joint target, uses source URDF limits and a ±0.35 rad trust region, and fits the same preselected six-part unsigned distance and closest-point vector targets. It additionally fits the native-relative direction score `-dot(native hand outward normal, object outward normal)` on fixed original-valid area quadrature. Negative native away scores remain negative targets. Each available part has 64 deterministic area-quantile bins with mass 1/6; duplicate bins are intentional, and missing part mass is not redistributed. An invalid candidate normal comparison retains a fixed nonzero residual of 2 times the square root of its fixed weight. These fit terms are proxies, not scientific acceptance gates.

The solver records major iterations and its raw final result. It selects the lowest recorded eligible objective, retaining the earliest major iterate on an exact tie; the raw final result replaces it only on strictly lower cost. The independent audit checks this selection from the recorded q, fingertip vectors and objective trace, and independently recomputes original and final FK errors. It does not recompute all trace poses or certify a global optimum. Convergence, feasibility and strict mapping validity are reported separately.

The palm position stays at the native MANO wrist. Palm orientation follows `R_world_operator @ R_base_palm(q)` and can change with WRJ1/2; root translation compensates only the base-to-palm offset. Physical limits, scene precision and independent fixed-URDF versus scene FK are checked. No arbitrary hand shift or scale is introduced.

The full audit reconstructs fresh compiled colliders and uses the literal prior distance, orientation eligibility, conditional cell-bound and strict-status functions. Bands remain 1, 2 and 5 mm; cell radii, source normals, visual meshes, numerical allowances, signed-facing rules and every denominator remain unchanged. Original cells are reused byte-for-byte. The additional unit-normal sanity threshold in the fitting helper is an API numerical check, not a relaxed angular or sign gate.

## Case results

- **hug_apple_scene_002_seed_0**: mean TIP 5.845 → 5.845 mm; maximum TIP 13.768 → 13.768 mm; mean middle 7.675 → 7.675 mm. Same-stratum distance MAE 4.345 → 4.345 mm and closest-point mean error 7.461 → 7.461 mm. Fixed-bin invalid normals 0 → 0 of 256; selected original_fallback, 0 eligible major iterates, solver success False, budget exhausted False. The returned exact-original pose preserves palm orientation, with position delta 0 m.
  Rejected raw fit: reported objective reduction 29.3%; largest reported TIP worsening 1.421 mm; 3/5 individual caps fail. These are recorded callback values, not an independent full-grid audit of the rejected pose.
- **hug_apple_scene_003_seed_0**: mean TIP 5.866 → 5.866 mm; maximum TIP 13.794 → 13.794 mm; mean middle 7.220 → 7.220 mm. Same-stratum distance MAE 4.455 → 4.455 mm and closest-point mean error 7.524 → 7.524 mm. Fixed-bin invalid normals 0 → 0 of 256; selected original_fallback, 0 eligible major iterates, solver success False, budget exhausted False. The returned exact-original pose preserves palm orientation, with position delta 0 m.
  Rejected raw fit: reported objective reduction 18.4%; largest reported TIP worsening 0.224 mm; 3/5 individual caps fail. These are recorded callback values, not an independent full-grid audit of the rejected pose.
- **hug_milk_box_scene_000_seed_0**: mean TIP 6.647 → 6.647 mm; maximum TIP 15.163 → 15.163 mm; mean middle 6.409 → 6.409 mm. Same-stratum distance MAE 4.920 → 4.920 mm and closest-point mean error 8.409 → 8.409 mm. Fixed-bin invalid normals 0 → 0 of 256; selected original_fallback, 0 eligible major iterates, solver success False, budget exhausted True. The returned exact-original pose preserves palm orientation, with position delta 0 m.
  Rejected raw fit: reported objective reduction 23.3%; largest reported TIP worsening 1.840 mm; 3/5 individual caps fail. These are recorded callback values, not an independent full-grid audit of the rejected pose.
- **hug_milk_box_scene_001_seed_0**: mean TIP 8.189 → 8.189 mm; maximum TIP 17.518 → 17.518 mm; mean middle 6.039 → 6.039 mm. Same-stratum distance MAE 6.260 → 6.260 mm and closest-point mean error 9.635 → 9.635 mm. Fixed-bin invalid normals 0 → 0 of 256; selected original_fallback, 0 eligible major iterates, solver success False, budget exhausted True. The returned exact-original pose preserves palm orientation, with position delta 0 m.
  Rejected raw fit: reported objective reduction 38.1%; largest reported TIP worsening 0.001 mm; 1/5 individual caps fail. These are recorded callback values, not an independent full-grid audit of the rejected pose.
- **hug_mug_scene_002_seed_0**: mean TIP 6.609 → 6.609 mm; maximum TIP 15.878 → 15.878 mm; mean middle 7.677 → 7.677 mm. Same-stratum distance MAE 4.882 → 4.882 mm and closest-point mean error 6.986 → 6.986 mm. Fixed-bin invalid normals 0 → 0 of 320; selected original_fallback, 0 eligible major iterates, solver success False, budget exhausted True. The returned exact-original pose preserves palm orientation, with position delta 0 m.
  Rejected raw fit: reported objective reduction 10.6%; largest reported TIP worsening 0.460 mm; 4/5 individual caps fail. These are recorded callback values, not an independent full-grid audit of the rejected pose.
- **hug_mug_scene_003_seed_0**: mean TIP 6.609 → 6.609 mm; maximum TIP 15.945 → 15.945 mm; mean middle 7.474 → 7.474 mm. Same-stratum distance MAE 4.997 → 4.997 mm and closest-point mean error 6.961 → 6.961 mm. Fixed-bin invalid normals 0 → 0 of 320; selected original_fallback, 0 eligible major iterates, solver success False, budget exhausted True. The returned exact-original pose preserves palm orientation, with position delta 0 m.
  Rejected raw fit: reported objective reduction 4.1%; largest reported TIP worsening 0.837 mm; 5/5 individual caps fail. These are recorded callback values, not an independent full-grid audit of the rejected pose.

All five TIP errors and all five middle errors, in thumb/index/middle/ring/little order, are retained in `summary.json`. Raw final refusal reasons, selection refusals, convergence messages and actual callback counts are preserved; no unsuccessful original proposal is removed.
The unrounded rotation-angle diagnostic in `summary.json` is an arccos readout after removing the common operator serialization. Mug B has a roughly 1.7 × 10⁻⁶ degree floating-point residual even though its returned q/root and both WRJ targets are exactly original. This residual is numerical roundoff, not a physical palm-orientation change.

## Full original-grid preservation measurements

The comparisons below use the 2 mm proximity band on original cells with maximum triangle area 4 mm². Lost and added areas are sums of six separate part-area lower bounds; parts can overlap, so these are not unique hand area or physical contact area. Direction comparisons use each mapping's actual exterior, unique, same-part, unburied, outside-visual eligible domain; changes in raw counts alone are not evidence of improved preservation.

- **hug_apple_scene_002_seed_0**: unsigned centroid IoU 0.256 → 0.256; definite LOST 116.343 → 116.343 mm²; definite ADDED 174.361 → 174.361 mm²; reliable direction reversals 2/28 → 2/28 eligible queries.
- **hug_apple_scene_003_seed_0**: unsigned centroid IoU 0.273 → 0.273; definite LOST 103.135 → 103.135 mm²; definite ADDED 158.302 → 158.302 mm²; reliable direction reversals 0/38 → 0/38 eligible queries.
- **hug_milk_box_scene_000_seed_0**: unsigned centroid IoU 0.267 → 0.267; definite LOST 65.077 → 65.077 mm²; definite ADDED 84.280 → 84.280 mm²; reliable direction reversals 3/39 → 3/39 eligible queries.
- **hug_milk_box_scene_001_seed_0**: unsigned centroid IoU 0.246 → 0.246; definite LOST 89.790 → 89.790 mm²; definite ADDED 142.306 → 142.306 mm²; reliable direction reversals 1/18 → 1/18 eligible queries.
- **hug_mug_scene_002_seed_0**: unsigned centroid IoU 0.245 → 0.245; definite LOST 608.959 → 608.959 mm²; definite ADDED 551.879 → 551.879 mm²; reliable direction reversals 0/93 → 0/93 eligible queries.
- **hug_mug_scene_003_seed_0**: unsigned centroid IoU 0.243 → 0.243; definite LOST 595.512 → 595.512 mm²; definite ADDED 385.826 → 385.826 mm²; reliable direction reversals 0/101 → 0/101 eligible queries.

### Controlling remaining discrepancies

- **hug_apple_scene_002_seed_0**: largest coarse-grid definite discrepancy is little at 5 mm: LOST 86.552 mm² and ADDED 280.159 mm² lower bounds. This localizes a remaining surface mismatch; it does not prove that correction is globally infeasible.
  A retained exact-grid signed witness is ring / native_away_mapped_facing at band 5 mm, object-local point [-10.612, -17.782, -43.100] mm: native score -0.09108, mapped score 0.35126. Original exterior/unique/same-part/unburied/outside-visual eligibility remains satisfied at this witness.
- **hug_apple_scene_003_seed_0**: largest coarse-grid definite discrepancy is little at 5 mm: LOST 85.809 mm² and ADDED 296.752 mm² lower bounds. This localizes a remaining surface mismatch; it does not prove that correction is globally infeasible.
  A retained exact-grid signed witness is ring / native_away_mapped_facing at band 5 mm, object-local point [-13.435, -17.548, -41.177] mm: native score -0.03285, mapped score 0.36236. Original exterior/unique/same-part/unburied/outside-visual eligibility remains satisfied at this witness.
- **hug_milk_box_scene_000_seed_0**: largest coarse-grid definite discrepancy is little at 5 mm: LOST 203.086 mm² and ADDED 206.578 mm² lower bounds. This localizes a remaining surface mismatch; it does not prove that correction is globally infeasible.
  A retained exact-grid signed witness is little / native_away_mapped_facing at band 5 mm, object-local point [4.614, 28.384, 70.910] mm: native score -0.42957, mapped score 0.29916. Original exterior/unique/same-part/unburied/outside-visual eligibility remains satisfied at this witness.
- **hug_milk_box_scene_001_seed_0**: largest coarse-grid definite discrepancy is little at 5 mm: LOST 173.408 mm² and ADDED 325.153 mm² lower bounds. This localizes a remaining surface mismatch; it does not prove that correction is globally infeasible.
  A retained exact-grid signed witness is middle / native_facing_mapped_away at band 5 mm, object-local point [-24.768, 23.570, 67.516] mm: native score 0.18436, mapped score -0.35679. Original exterior/unique/same-part/unburied/outside-visual eligibility remains satisfied at this witness.
- **hug_mug_scene_002_seed_0**: largest coarse-grid definite discrepancy is index at 5 mm: LOST 1093.179 mm² and ADDED 561.250 mm² lower bounds. This localizes a remaining surface mismatch; it does not prove that correction is globally infeasible.
- **hug_mug_scene_003_seed_0**: largest coarse-grid definite discrepancy is index at 5 mm: LOST 1556.262 mm² and ADDED 294.656 mm² lower bounds. This localizes a remaining surface mismatch; it does not prove that correction is globally infeasible.
  A retained exact-grid signed witness is index / native_away_mapped_facing at band 5 mm, object-local point [4.474, 55.401, 66.628] mm: native score -0.24813, mapped score 0.01386. Original exterior/unique/same-part/unburied/outside-visual eligibility remains satisfied at this witness.

The complete 1/2/5 mm per-part lower/upper bounds, whole-hand intervals and candidate/valid/exterior orientation denominators are retained for all nine resolutions in the compressed independent reports and `summary.json`. Centroid IoU is an estimate; positive definite lost/added lower bounds or reliable sign reversals reject strict preservation under the original rule. An absence of sampled reversals remains UNRESOLVED for continuous orientation, not PASS.

## Scope and remaining claims

These are reused development cases, so fitting and checking them establishes only this scoped before/after geometry result. Full continuous physical contact equivalence remains UNRESOLVED. Failure-label preservation, unseen-object validation and predictor generalization remain NOT_TESTED. Native MANO and simulator colliders are different physical surface representations; no anatomical equivalence claim is made. A lower fit objective or passing the fingertip constraints does not establish strict mapping acceptance, successful grasping, or failure prediction accuracy.

All 114 protected source, API/test, assets, input and paper hashes match the pre-run snapshot. Actual physics integration attempts: 0. Inference, evaluation data and training: false.

## Evidence bindings

- Candidate protocol SHA256: `e42b74b787a8a1a5b5a661596466fe66b24c14eea39db25ad039fda2a462b8ff`.
- Independent wrapper protocol SHA256: `9f0e0833c80a5b36f0ad21f629ecb84cf399f0e2e7669436eb3b93e561faf8f5`.
- Summary script SHA256: `a4a7633c4983bb843974c49b9dd12963070f802bcd8d5bb6714b68858ebbe8a9`.
- `candidate_validation_8,12,16.json.gz` and `candidate_validation_20,24,28.json.gz`: complete independent reports.
- `exact_direction.py`, manufactured tests and `original_direction_eligibility_parity.json`: fixed normal eligibility parity and invalid-site behavior.
- `constrained_candidate_tradeoffs.png` and `constrained_fingertip_errors.png`: scientific diagnostic figures, bound to the summary by `figure_metadata.json`.
- `protected_hashes.json`: the pre-run 114-file snapshot. Prior protocol reports remain separate.
