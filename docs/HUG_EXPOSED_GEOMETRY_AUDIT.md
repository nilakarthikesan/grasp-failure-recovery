# HUG exposed-geometry audit and current validation scope

15 October 2026. Shadow remains the execution hand. This audit reuses saved HUG proposals and poses; it does not generate evaluation episodes, outcome labels, training data or new native grasps. The production mapper and original acceptance rule are unchanged.

## What is now established

The saved native decoding report reconstructs all 32 well-bound native exports exactly. The saved nominal-pose report covers the same 32 proposals and all 768 joint-limit comparisons, with no violation. Independent URDF-versus-scene landmark FK differs by at most 0.00007845 mm. These establish the recorded conversion and pose conventions within that scope; they do not establish anatomical wrist/palm equivalence. All 32 meet the original 10 mm mean fingertip guard, but the largest individual fingertip discrepancy is 17.518 mm. A passing average does not hide that individual error.

The new independent compiled-scene audit covers the six detailed development cases. Each has 45 object-compatible Shadow colliders. Apple, milk box and mug have 2, 7 and 15 object collision meshes respectively. Transforms, convex collision hulls and collision filters match the audited representation; margins/gaps are zero under the installed MuJoCo 3.3.7. No importer, convexification or collision-filter correction was identified. This checks the modeled geometry, not its physical realism or dynamic tracking.

[Current claim manifest](../research_artifacts/hug_practical_validation_design_2026-10-15_v1/current_claim_manifest.json) binds the historical and new checks separately. [Independent collision review](../research_artifacts/hug_collision_contract_review_2026-10-15_v1/contract_findings.md) includes version-pinned simulator sources.

## A separate surface diagnostic

The historical strict metric measures the boundaries of individual collision components, including buried interfaces. The new diagnostic asks about the boundary of their occupied union, retaining forearm and wrist carriers. These are different geometric sets; neither is automatically the simulator's force-bearing contact set.

For an object query strictly outside the whole hand, the closest component globally gives the distance to the union boundary. For an interior point, the nearest component boundary may be buried, so it cannot be used as an exact union distance. The diagnostic uses conservative bounds or returns unresolved. It also checks whether the nearest surface of the requested anatomical part is exposed; a neighboring finger cannot silently substitute for that part.

Coincident/shared faces, numerical boundary cases and inconsistent closest-point calculations do not receive an invented exposure certificate. Analytic controls cover overlap, shared interfaces, rigid transforms and invalid-oracle refusal. Distances are conditional floating-point diagnostics, not formal interval-arithmetic proofs; no new surface-normal agreement certificate is issued.

Removing buried interfaces can only increase distance to the corresponding exposed part. A definite lost-near relationship therefore survives this change of surface set; an added-near relationship can become unresolved. Failure to prove persistence is not evidence of a repaired mapping.

The frozen diagnostic examined 624 row/pose records: 178 retained proximity witnesses and 134 retained orientation-site records, each at the original and previously selected poses. These records include duplicates and are not independent grasps or complete surface coverage. Of the 624 nearest requested-part points, 575 are strictly separated exposed points and 49 are buried. No oracle inconsistency or exterior/burial contradiction was detected. Fifteen primary manufactured controls and ten independent controls pass.

Every one of the six grasps retains definite proximity counterexamples at both saved poses, conditional on the stated geometric and numerical assumptions:

| Existing grasp | Retained proximity rows | Original: counterexamples persist | Previously selected: counterexamples persist |
| --- | ---: | ---: | ---: |
| Apple A | 33 | 31 | 31 |
| Apple B | 15 | 15 | 10 |
| Milk box A | 37 | 35 | 35 |
| Milk box B | 14 | 12 | 6 |
| Mug A | 57 | 46 | 19 |
| Mug B | 22 | 17 | 13 |
| Total | 178 | 156 | 114 |

The total difference is not a new improvement score. These are already selected historical poses; bounds can become inconclusive when a cell straddles a threshold or no exposed upper witness is available. The diagnostic never counts an unproven counterexample as a repaired region or a mapping pass.

For a concrete surviving example, milk-box B's little-finger region lies within 1.691 mm of the native patch throughout the saved object cell, whereas its corresponding exposed Shadow subset is at least 8.144 mm away in the previously selected pose. For mug B, a native palm region is within 4.746 mm while the corresponding Shadow subset is at least 8.899 mm away. Both nearest Shadow points are exposed; buried interfaces do not explain these examples. These are object-relative geometric discrepancies, not measured lift failures.

The run used 36 static forward calls, 84,240 component-point computations and zero integration steps, new grid points, optimizations or new orientation claims. [Frozen diagnostic and results](../research_artifacts/hug_exposed_geometry_2026-10-15_v1/REPORT.md). All nine historical full-grid strict FAILs remain unchanged; they were not rerun or replaced.

## What acceptance means for this project

The predictor's target is the executed retargeted Shadow grasp. Correct source decoding and realization of a declared Shadow pose can be checked independently of a universal native-to-robot geometry tolerance. This narrower claim does not establish faithful preservation of HUG's native grasp surfaces or usefulness for prediction.

The old nine strict failures remain failures under their original rule. An approximate native-fidelity claim would need a justified prospective allowance; none is assigned here. A later prediction study needs actual executed labels and controls for mapper distortion, object geometry, source coverage and failures. Poor native proposals and mapping refusals must remain visible, and successful lifting is not a universal mapping acceptance test.

All 32 well-bound saved proposals are development evidence across four geometries. The other 25 cases beyond the six repeatedly fitted non-cube cases and one detailed cube case can broaden development checks, but cannot become an untouched object holdout. The [prospective protocol](../research_artifacts/hug_practical_validation_design_2026-10-15_v1/REPORT.md) records this boundary and separates implementation correctness, native fidelity and prediction adequacy. Its schema tests check record consistency, not scientific acceptance.

## Next technical step

Use the surviving exposed-region discrepancies to define a fixed, continuous object-relative retargeting objective, preserving source poor proposals and the original fingertip/domain safeguards. Freeze the objective, correspondence, finite solve budget and baseline comparison before a new fit; compare all saved development inputs and retain every refusal. Do not optimize an aggregate contact-success score or rename old failures as passes. Any improvement remains an experimental mapper until independently checked.

The present diagnostic supplies a more defensible measurement target, not a successful new pose correction. Broader held-out geometry and dynamic/prediction adequacy remain unestablished and would require later authorized data. No blanket “HUG fully validated” claim is supported.
