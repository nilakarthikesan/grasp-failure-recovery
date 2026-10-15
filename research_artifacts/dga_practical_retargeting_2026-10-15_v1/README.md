# DGA practical retargeting: completed static pilot

15 October 2026. **STATIC_PILOT_ADMITTED_TESTED_SCOPE** for the pinned DGA MultiDex plus explicit retargeting pipeline on the unchanged Shadow hand, limited to these saved cube outputs and the conditional numerical assumptions below. Provider execution remains disabled. Native-hand fidelity and physical adequacy are not accepted by this result.

## Actual result

| Saved native cohort | Inputs | Source-valid static mappings | Native-invalid refusals |
| --- | ---: | ---: | ---: |
| Recent MultiDex regression | 16 | 11 | 5 |
| Original four MultiDex regression | 4 | 2 | 2 |
| Total | 20 | 13 | 7 |

All outputs are previously inspected development/regression material on the same 50 mm cube. None is an untouched native holdout. Original DexGraspNet outputs remain a separate lineage and were not substituted into this cohort; the inventory records 18 unique original-checkpoint outputs with native-domain failures.

Of 65 finger branches across the thirteen valid poses, 42 preserve reference target geometry exactly. Twenty-three branches propose a correction. One passes independent geometry admission; 22 remain unresolved at the frozen finest resolution and fall back to the explicit legal reference. “Unresolved” does not prove regression or impossibility. Every refusal and fallback remains in the accounting.

The admitted correction is recent row 4's thumb. Under the declared 1 nm per-distance arithmetic assumption, its continuous worst native-relative distance residual over the entire cube surface lies in **[1.678521, 2.117103] mm**, compared with reference **[2.508736, 2.947318] mm**. The candidate upper bound is 0.391633 mm below the reference lower bound. Thumb maximum landmark error falls from 0.852864 to 0.584096 mm; thumb SSE decreases from 1.348861 to 1.013983 mm². The little-finger proposal does not pass and is reverted, so the final whole-pose maximum is **0.605692 mm**, not the provisional candidate's 0.584096 mm. All other branches satisfy their own unchanged/nonregression gates.

Across all thirteen final poses, maximum native-to-target landmark error remains **6.075600 mm**. Maximum per-pose RMS is 1.778242 mm; pooled RMS over 13 × 28 sites decreases only from 1.179927 to 1.179537 mm. These are distortion measurements, not experimentally calibrated acceptable error thresholds. Prior palm overlap remains: final root/wrist identities carry forward six recent-row strict palm failures. The whole-surface test controls each finger's worst distance residual; it does not establish pointwise nonregression, preserved near/contact regions, signed occupancy, normals, grasp success or failure-label equivalence.

Independent literal XML/SciPy FK and compiled MuJoCo kinematics agree across final poses to 2.483e−16 m and 7.094e−16 rad; target limits match exactly. These numbers concern numerical realization of the prescribed target pose, not native-to-target accuracy. Only static kinematics is used; simulation/forward/inverse/integration entry points are blocked in the checker.

## Prospective criteria and repairs

See the [protocol](../../docs/DGA_PRACTICAL_RETARGETING_PROTOCOL.md). The criteria SHA is `6eb351dc097daffb4d08f04117615714de2120d3395367f900e4cefb444d4d34`; proposal method v3 SHA is `44c2c1e12fc51d912090c7b2987af8ec0631832127c1d764215972c4826ecf6e`; validator policy v5 SHA is `a37227a4ee6702ae2005f4be1159b148b9a98fa5dfe9b4fbfecfbc79a5fa18df`. Each was frozen before its associated run. Earlier versions remain preserved, including the incomplete v2 validation.

Source-invalid data are rejected before retargeting. The reference is explicit Euclidean finger-joint projection into literal target intervals, with native palm and wrists fixed. Already legal branches cannot change. Projection minimizes angular displacement only; it neither repairs the native output nor proves geometric optimality. Optional proposals need per-finger maximum landmark and SSE nonregression plus independently bounded uncapped worst-distance nonregression and a strict improvement. A lower fitting loss alone cannot pass.

Repairs address solver exceptions after an improved trial, the capped-vs-uncapped objective mismatch, unused vertices in mesh search bounds, the previously unsupported 1 mm target thumb visual box (the old reader failed closed) and conservative validation runtime. The final triangle search uses face-referenced vertex bounds, radius buckets and sphere/AABB pruning. Adaptive refinement preserves all leaf upper bounds and verifies full cube coverage on the original shifted 1, 0.5 and 0.25 mm grids. None of these changes relaxes the frozen admission inequalities.

Static repository tests passed (126 provider tests, 10 practical tests, 18 independent adversarial tests). Separate solver-exception, proposer and independent geometry controls are retained with their receipts. The fitting stage's numerical warnings are recorded rather than erased; final acceptance is independently recomputed with warnings treated as errors. Legacy proposer dispatch counts omit uncapped mesh work and are not total computation counts.

## Authoritative evidence and replay

- [Decision summary](final_summary_v1.json).
- [All-20 independent summary](validation/authoritative_all20_summary_v1.json), including per-pose and per-finger errors, counts and final compiled checks.
- [Final selected poses](validation/authoritative_all20_final_static_poses.npz), with original/recent cohort keys; these are static mapping artifacts, not evaluation episodes or outcome data.
- `validation/recent16_authoritative_v3_part_a/`, `validation/recent16_authoritative_v3_part_b/`, `validation/original4_authoritative_v3/`: full queries, leaf-cover ledgers, row reports and compiled snapshots.
- `protocol/*complete_geometry_recount_v1.json` and the combined independent recount: independent saved-evidence recomputation of coverage, bounds, adoption and final mixed poses.
- `mapper/implementation_v3/` and `mapper/results_v3/`; `mapper/results_original_four_regression_v3/` is kept separate. These contain **provisional** results, superseded for admission by the independent selected-pose archive.
- `validation/final_palm_diagnostic_identity_v2.json`: unchanged palm evidence, including inherited failures.
- `dependency_index.json`: original-path/hash-to-repository-file resolution, including dependencies already preserved by the prior committed study. Historical reports keep their original paths and hashes; resolve those paths through this index and the `mapper/`, `protocol/`, `validation/` directory mapping when replaying elsewhere. Do not edit a frozen method in place or mistake historical candidate files for final admission.
- `paper/`: exact compiled working manuscript and desktop compiler receipt. Only the DGA section was edited; concurrent unrelated manuscript work was preserved.
- `evidence_manifest.json`: final file hashes and preservation receipt.

The source APIs deliberately report geometry/physics as not assessed when called without the independent study gates. No provider registry promotion occurs. The new static protocol treats DGA plus the named mapper as a distinct generator pipeline; future outcome labels must describe the actual Shadow attempt and retain mapper identity, distortion, native-invalid refusals and the full input denominator. Additional native object coverage and physical/predictor adequacy remain separate future checks. No physics, new native inference, evaluation episodes/labels or training ran here. Historical strict-preservation studies and the unchanged Shadow target remain intact.


Final independent review verifies 385 aggregate checks and 2,159,607 retained leaf bounds across disjoint partitions, with a separate 41-check positive-result audit using independently coded scalar FK. [All-output review](protocol/authoritative_all20_independent_review_v2.json) and [positive thumb review](protocol/positive_row4_thumb_independent_review_v1.json) bind the final summaries and selected poses. [Exact replay commands and environment notes](mapper/control/reproducibility_commands_v1.txt) and [source/dependency audit](mapper/control/source_dependency_audit_v1.json) describe replay prerequisites. One inherited prose sentence in the original-four method supplement's output-schema description mentions 16/11; its actual scope, loader, arrays and recorded accounting consistently contain four inputs and two mapped outputs. That frozen sentence is retained transparently, not silently edited.

The initial failed JSON serialization is preserved under `protocol/history/failed_json_serialization_v1/` with a failure record. The successful part-A v2 receipt was already valid; additive v3 and combined-review v2 recompute the same result with serialization completed before output files are opened. [Receipt integrity audit](protocol/receipt_integrity_audit_v1.json) verifies all 118 combined bindings. No scientific result or criterion changed.
