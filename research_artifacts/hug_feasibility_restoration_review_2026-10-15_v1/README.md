# Independent review of saved HUG endpoint feasibility restoration

All twelve fixed saved V2 terminal states yielded an observed checkpoint that satisfies the exact historical individual fingertip caps, inclusive 10 mm mean, unchanged joint/domain bounds, and endpoint trust box. Independent reconstruction from the saved scene and source URDF verifies those twelve restored poses as well as the six final references. No candidate passed the frozen improvement gate; all six final poses retain the previously audited pooled reference. This is not complete mapping validation.

`final_output_review.json` records 15,133 source/trace/kinematic assertions and 964 recorded-geometry arithmetic assertions. These are replayed checks, not independent scientific samples. `merge_and_outcomes_review.json` adds 340 saved-record identity, outcome and count checks. The independent reviewer did no new surface queries, inference, physics, or evaluation episodes. It used static `mj_forward` to reconstruct the twelve restored endpoints and six references, then replayed retained-surface arithmetic from existing records.

Pre-execution receipts include 18 independent synthetic controls, an independent rerun of the primary 14 manufactured controls, 116 endpoint/source checks, and 320 draft-protocol/input-binding checks. These validate the bounded method implementation and provenance, not global feasibility or convergence.

## Frozen experiment and result

The experiment projected only the twelve hash-bound V2 terminal states, minimizing squared scaled distance from each endpoint. The trust box was the original domain intersected with endpoint ±0.25 in each scaled coordinate; its translation allowance is 1.25 mm per coordinate, not a Euclidean 1.25 mm sphere. Selection was the nearest observed exact-feasible endpoint/major/terminal checkpoint with stable ties. Raw line-search probes were not eligible. This is not a certificate of globally nearest projection.

All recorded probes, checkpoints, source identities, complete 27-coordinate keys, exact norm arithmetic, fresh selected FK, fixed trust boxes, optimizer terminal provenance and callback counts replay. All twelve restored poses satisfy the literal original caps without acceptance slack. Independent source-URDF/scene positional discrepancies stay below the inherited 1.1e-7 m threshold. Root quaternion and object pose remain unchanged.

All twelve restored candidates have a strictly worse five-component exact diagnostic rank than their pooled reference. Ten invalidate previously eligible orientation rows; six lose a row passing in the original or reference pose. These losses compare the restored candidates with original/reference evidence. The old infeasible endpoints did not receive exact surface checks, so these results do not isolate an effect caused by projection itself. Invalidated rows are not counted as reliable direction reversals. Apple B and Mug B protected-pass losses are invalidations; each Milk B endpoint additionally has one still-eligible facing reversal, with native score +0.2792811 and mapped scores −0.0400841 / −0.0438120. Mug A endpoint 0 passes 11/57 retained proximity rows versus the reference's 10/57, but its worst normalized violation worsens from 5 to 5.4453426; endpoint 1 is also worse. Mug A has no retained reliable orientation witness, so the missing coverage is not a passing orientation result.

The primary run used 1,101 tip-FK callbacks and 161 separately counted Jacobian staging calls, for 2,524 projection `mj_forward` calls. Each callback stages twice. Maximum per endpoint was 798 versus the frozen 920 bound. There were 24 post-projection retained-surface checks in total; zero surface calls occurred inside projection. Three optimizers reached the 40-iteration limit; their observed feasible checkpoints still passed independent replay. Post-import Numba thread masks were recorded as two per worker.

The frozen gate therefore authorizes no coarse follow-up grid. Previously measured nine full-grid failures remain the applicable scientific evidence; this study does not replace or repeat them. No production mapper was promoted.

## Reviewer corrections

All failed reviewer logs are retained in `reviewer_corrections.json` and the named attempt logs. The first endpoint reconstruction used algebraically equivalent root-zero compensation, which differed from the frozen original-root staging by 2.78e-17 m and failed a reviewer exact-equality assertion. The independent reconstruction now follows the frozen operation order without importing primary staging code; exact vector/cap comparisons remain unchanged. The merge helper then corrected two reviewer-only schema assumptions: cohort bindings are a list and empty orientation masks require explicit Boolean dtype. No experiment, data, thresholds, rank or selection was changed.

## Reproduction and file roles

- `verify_restoration_outputs.py`: independent source/scene FK and full trace/arithmetic replay; imports the hash-bound earlier reviewer FK helpers and local `constraint_arithmetic.py`, not primary engine/core/solver.
- `verify_merge_and_outcomes.py`: JSON-only merged/per-case/per-endpoint equality, reference identity, primary descriptive scalar replay and count/eligibility summaries.
- `preflight_review.json`, `source_endpoint_review.json`, `independent_core_controls.json`: immutable pre-execution prerequisites.
- `final_review.json`: consolidated PASS receipt and bindings.
- `bundle_manifest.json`: SHA-256 for every other reviewer file; verify after archive copying.

The local Python/runtime paths are recorded in the source files and source bindings. Primary artifacts remain under `/private/tmp/hug_feasibility_restoration_2026-10-15_v1`. The final candidate SHA-256 is `b0a6993006daf7aba9c1ee4d8898892a99dac19a8906f76e7fb1b85295832757`.
