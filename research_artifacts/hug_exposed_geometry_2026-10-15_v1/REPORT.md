# Exposed Shadow collision-geometry diagnostic — 15 October 2026

Buried component interfaces do occur in the retained witnesses, but they do not explain away the mapping failures. At least one whole-cell proximity counterexample remains in each of the six original and each of the six previously fitted poses, even when the mapped reference is restricted to the same anatomical part's exposed subset. This is a separate, conditional geometry diagnostic; it changes no historical acceptance rule or result and does not validate the complete HUG mapping.

## Scope and method

The study used the same six saved native HUG grasps, their original Shadow poses and the already-selected pooled poses. It queried the existing 178 proximity records and 134 orientation records at each pose: 624 row-pose records in total. Records repeat across source audits/bands and are capped examples, not independent grasps, exhaustive failures, or a new evaluation dataset. There are 221 distinct requested-part/coordinate sites per pose cohort; the two cohorts share those sites.

All 45 compiled, object-compatible Shadow colliders participated in whole-hand occupancy and burial checks, including wrist and forearm carriers. The independent compiled-scene review found no missing convexification, mask/filter, nonzero margin, or transform defect in these scenes. The geometric union reference describes occupied geometry and accessibility; MuJoCo still generates pairwise component contacts. They are different concepts.

The native reference remains the literal MANO anatomical skin triangles used by the prior audit. This study does not assert that every native skin patch is an exposed occupied-solid boundary, and does not inspect native self-intersections. Object query sites remain the frozen visual-object surface sites. Saved visual-membership/orientation diagnostics are carried through with their source identities; no new object grid or orientation-normal test was run.

For a query strictly outside the whole hand, minimum component distance equals distance to the whole occupied-union boundary. For inside queries, maximum component penetration supplies a lower bound; an available exposed point supplies an upper bound. For a requested anatomical part, the nearest individual-component surface distance is a lower bound to that part's exposed subset. A nearest point separated by more than the inherited 10 nm diagnostic tolerance from every other collider supplies a conditional exposed-point certificate. Strictly buried points are identified; coincident/shared-face or inconsistent-oracle cases remain unresolved. No CSG surface, ray search, inferred empty part, unique-normal assertion, or global inside nearest-boundary solution is constructed.

The component oracle, all source poses/rows and budgets were frozen before measurement. Fifteen primary and ten independent manufactured controls cover overlapping/contained solids, shared internal box faces, coincident and distinct ties, carrier burial, malformed closest-point oracles, and contradictory exterior burial. The bounds are statements about the reconstructed geometry conditional on the trusted oracle and floating arithmetic; they are not interval-certified numerical proofs.

## Results

| Saved grasp | Original proximity counterexamples retained | Pooled-pose proximity counterexamples retained |
|---|---:|---:|
| Apple A (8) | 31 / 33 | 31 / 33 |
| Apple B (12) | 15 / 15 | 10 / 15 |
| Milk box A (16) | 35 / 37 | 35 / 37 |
| Milk box B (20) | 12 / 14 | 6 / 14 |
| Mug A (24) | 46 / 57 | 19 / 57 |
| Mug B (28) | 17 / 22 | 13 / 22 |
| Total | **156 / 178** | **114 / 178** |

These are conditional whole-cell counterexamples supported by the new distance bounds, with an additional conservative numerical widening. An unproven counterexample is **not** a repaired or accepted row. The difference between 156 and 114 is not a new optimization improvement score: both sets of poses were fixed before this study, and fewer certified counterexamples can also mean uncertainty. Apple B illustrates the distinction: all 26 retained nearest-part points are exposed in both poses, while only 10 of its 15 pooled proximity rows remain definite under the whole-cell test.

For the original poses, all **98 / 98 lost-native-near** records remain counterexamples. Of the 80 added/new-mapped-near records, 58 remain counterexamples and 22 have buried nearest component points. Removing buried interfaces restricts the target surface set and can only increase distance to the same-part surface. It cannot repair a definite lost-near mismatch. Added-near records can become unresolved; this study does not assert that the 22 buried cases are actually repaired.

Across all 624 records, **575 nearest requested-part points are strictly separated/exposed** and **49 are strictly buried**. There are no coincident/unresolved exposure classifications, oracle inconsistencies, exterior-burial contradictions, or bound-order failures in this fixed sample. All detected burying components belong to the same anatomical part as the selected point; carriers were checked and did not supply a burying witness here. This finding concerns only these sampled points.

Of the original cohort's 134 orientation-query records, 132 have exposed nearest-part points and two have buried points. All 134 pooled orientation-query records have exposed nearest-part points. Exposure does not establish a unique normal, correct native-facing sign, preserved eligibility, or orientation acceptance. No new sign claims were computed. Mug A still has zero retained orientation witnesses and therefore no orientation evidence from this subset.

Occupancy is independently recorded. Of the 312 records per cohort, 54 original and 58 pooled queries are strictly inside at least one hand component. Their positive unsigned component distances must not be described as free-space clearance. The remaining 258 original and 254 pooled records are strictly exterior in this sample.

## Verification and computation

The measured individual-component distances replay the previous saved distances within 1e-12 m. Original native hashes, declared source rows, original/pooled normalized states, exact original fingertip caps, inclusive 10 mm mean, fixed root quaternion, physical/joint-correction limits and translation AABB all passed. All implementation/input/review hashes were checked before execution and before each output write. Existing output files were refused.

The bounded run completed all twelve poses in 9.814 seconds with 36 static `mj_forward` calls, 28,080 component-query point evaluations and 56,160 closest-candidate point evaluations: 84,240 total, as frozen. Numba was reset after inherited imports and remained at two threads. There were zero physics steps, inference calls, optimizers, new grids, generated grasps, evaluation labels, training runs, or production changes.

Protocol SHA-256: `3697aff0c70d1be747f355606fb3c0179f282c661ebd3ee289e48de09c337a6a`.
The pre-execution review is bound by the protocol; per-case raw distances, closest points, exposure/occupancy evidence, source pointers and all unresolved fields are preserved in `case_*.json`. `summary.json` contains saved-output aggregates, and the cohort file binds the case hashes.

## Implication

The practical next step is to keep exposed geometry and literal source skin explicitly separate while deciding which retargeting distortions are scientifically acceptable. Where the nearest requested-part point is demonstrably exposed, its distance remains a usable local residual; it still must preserve source context and be checked independently after any fit. Where a point is buried, this diagnostic supplies bounds and uncertainty rather than a replacement normal or fabricated fit target. No new optimizer or acceptance threshold was selected by these results.

Independent saved-output source/bound/persistence/count review passed. Receipt SHA-256: `a236551300b54de167e65673de4e4162ccaeb29b575536fe94d98f6bc120d90a`. The post-measurement aggregation script reproduces `summary.json` exactly without geometry queries.
