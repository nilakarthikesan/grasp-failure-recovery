# Current Shadow representation: fixed HUG witness check

The corrected current representation still contains definite native-reference surface mismatches. This check used the same six saved HUG proposals and the same retained witness rows. It did not run a generator, optimizer, physics step, evaluation episode, training run or new full grid.

The six original poses used the actual current native-to-palm helper and current floating-root inversion, checked independently against the immutable native transform and source XML FK. The two latest experimental poses (indices 20 and 28, from `hug_feasible_eligibility_2026-10-15_v1`) retained their literal saved joint/root commands. All hand surfaces came from the current registered `ActuatedShadowRunner`, including all 45 active colliders and the forearm/wrist carriers used for whole-hand eligibility. The current source includes the reviewed object-inertia precision correction.

The native signs, anatomical labels, object-frame queries, bands, cell radii and numerical allowances remained frozen. Historical compiled object visual triangles served only as the unchanged geometric reference for object-membership predicates; no historical hand queries were rerun. Original comparisons include current frame and geometry changes. Experimental comparisons hold the saved root fixed.

## Results

These are overlapping retained rows, including duplicates across bands and prior audits. They are not independent proposals, area totals, success rates or held-out evidence.

| Pose | Proximity preserved | Definite lost / added | Proximity unresolved | Direction preserved | Valid opposite direction | Direction ineligible | Literal old tip caps |
|---|---:|---:|---:|---:|---:|---:|---|
| Original 8 | 0/33 | 17 / 16 | 0 | 0/54 | 54 | 0 | Refused |
| Original 12 | 0/15 | 8 / 7 | 0 | 2/11 | 9 | 0 | Pass |
| Original 16 | 0/37 | 20 / 17 | 0 | 0/52 | 49 | 3 | Pass |
| Original 20 | 0/14 | 8 / 6 | 0 | 0/15 | 10 | 5 | Refused |
| Original 24 | 0/57 | 32 / 25 | 0 | Unavailable | — | — | Refused |
| Original 28 | 0/22 | 13 / 9 | 0 | 0/2 | 1 | 1 | Refused |
| Experimental 20 | 2/14 | 6 / 3 | 3 | 3/15 | 12 | 0 | Pass |
| Experimental 28 | 1/22 | 8 / 5 | 8 | 1/2 | 1 | 0 | Pass |

All **178** retained original proximity counterexamples persist under the unchanged interval rules: 98 lost-near and 80 added-near rows. Of 134 original direction records, 123 remain valid opposite directions, two preserve the native sign, and nine have invalid eligibility. Index 24 has no retained reliable direction witness; it receives no orientation-validation credit.

The two experimental poses preserve three of 36 proximity rows. Twenty-two remain definite mismatches; eleven are unresolved, rather than proven mismatches. They preserve four of 17 direction rows and retain 13 valid opposite directions. Both experimental poses still satisfy every literal historical fingertip cap and the inclusive 10 mm mean gate on the current representation. Their mean tip errors are 7.139035 mm and 5.322369 mm respectively.

A subsequent pure source-XML readback confirms that all eight already evaluated joint vectors satisfy all 24 literal source joint limits, including experimental 20 and 28, with zero tolerance. This check used no new FK or geometry and changed no commands (`source_limit_readback.json`, SHA-256 `0e710e06b4bd6286145db684a8c4f5dfe71b5baee7ac3f42268ad191b58c995d`).

Original mean tip errors range from 5.844762 to 8.189182 mm. Four original poses fail at least one literal historical per-tip cap after current replay. The maximum excess is 10.172097 nm; original indices 8 and 20 exceed only at approximately 1e-16 m arithmetic scale. Those literal refusals are retained without rebasing or adding tolerance. They do not by themselves establish a practically meaningful anatomical change.

The largest measured change in a retained proximity distance relative to its historical reference is 11.268084 µm. This is a sampled per-part distance difference, not a global surface bound. The historical 6.9446 mm symmetric vertex-set discrepancy is not a hull-shape error; a separate source audit numerically computed the relevant two-sided convex-solid Hausdorff distance as 24.512339 µm, without a directed-rounding certificate. Neither quantity is a penetration measurement.

## How definite and uncertain outcomes differ

A retained native-near cell is definitely changed when the current lower distance bound exceeds its band; a native-far cell is definitely changed when the current upper bound is at or below its band. Other failures of the sufficient preservation predicate remain unresolved. The summary uses the exact stored interval bounds and unchanged strict/inclusive comparisons.

A reported opposite direction requires valid exact eligibility, membership in the original source band and a robust sign opposite the native direction. Ties, burial, wrong nearest part, inside status and other eligibility failures are reported separately. Invalidating a witness never counts as preserving its direction. None of the evaluated valid-in-band rows fell in the sign deadzone.

These saved counterexamples refute complete preservation under the historical surface criteria for these poses. They do not prove that no different Shadow pose could improve the mapping. No full current grid or generalization claim is made, and historical full-grid results remain attached to their original version.

## Reproducibility and scope

- Frozen protocol: `protocol.json`, SHA-256 `2ceaa74a5c9c59d552f8bbd0aad1cf103a3a34155f677bdf479268fe887fd61d`.
- Primary result: `results.json`, SHA-256 `47611467b6291d589b8c090ff22f48e309bf9ae738fb35b93d3c41c7c6f4303f`.
- Pure saved-output summary: `saved_output_summary.json`, SHA-256 `b1a8bf32e245578968b890e7a3138553236d6b0494b5c436741607fa0cb88b58`.
- Eight poses, 365 retained row evaluations, 32 static forward calls, zero integration/retargeting calls. Bound current sources were rechecked before each output and at completion.
- Twelve manufactured witness controls and three independent classification-partition examples passed. Import-only checks verified the current loader and function identity, with NumPy 2.2.6, MuJoCo 3.3.7 and two Numba threads.
- `protocol.pre_object_inertia_fix.draft.json` preserves the reviewed pre-correction draft. A pure preparation attempt initially decoded a gzipped cap report as text; it stopped before models or queries and is disclosed in `preparation_history.json`.
- No production mapper or acceptance criteria were changed by this diagnostic. The outputs are static mapping evidence, not evaluation data for failure-predictor training.
