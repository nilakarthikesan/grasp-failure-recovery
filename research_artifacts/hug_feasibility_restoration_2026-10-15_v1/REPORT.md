# Saved-endpoint fingertip feasibility restoration

15 October 2026. **12 restored endpoints; 0 accepted improvements; all six prior pooled poses retained unchanged.** No fresh coarse or fine grids were warranted or run by this method.

## Frozen question and method

The preceding V2 surface search rejected all twelve terminal states for small individual fingertip-cap excesses. This study asks whether a bounded, FK-only projection can remove those excesses while exposing a useful retained-surface improvement. All twelve source endpoints, their complete 27-coordinate states and exact trace provenance were frozen before any new geometry or projection.

One projection per endpoint minimizes normalized squared distance from that endpoint within a fixed per-coordinate trust box: ±0.00625 rad for each joint and ±1.25 mm for each world-palm coordinate, intersected with the original physical/correction/reach bounds. It is not a 1.25 mm Euclidean sphere. Selection uses the closest of the observed exact-feasible endpoint, major-iterate and terminal checkpoints, with stable ties; it does not claim a globally nearest feasible pose. Raw line-search probes are never eligible. Nine optimization runs report convergence; three reach their 40-iteration limit but already contain a fresh-verified eligible checkpoint.

Every returned projection satisfies the original literal individual fingertip caps, inclusive 10 mm mean gate, native-world targets, floating-root quaternion, original domain and endpoint trust box. The 1 nm inward optimization target is not an acceptance allowance. The strict surface-equivalence and original-cap nonworsening gates are this project's historical criteria, not thresholds prescribed by HUG's published evaluation.

## Recorded results

All twelve saved endpoints acquire an exact-tip-feasible checkpoint. The palm correction from each saved endpoint is only 0.000114–0.039256 mm. These small restoration corrections must not be confused with the endpoint's larger previously optimized displacement from the native anchor.

No restored checkpoint strictly improves the existing exact five-component retained-witness rank while also preserving every orientation pass and every eligible orientation row present in either the original or pooled reference pose. Ten endpoints invalidate protected orientation evidence. The other two are Mug A, which has no saved reliable orientation witness and remains worse by the exact rank.

Counts below retain duplicate bands/studies. They are not independent contacts, rates or surface areas. “Lost eligible / lost passing” compares each restored endpoint against the protected original-or-pooled orientation sets.

| Existing grasp | Endpoint 0: lost eligible / lost passing | Endpoint 1: lost eligible / lost passing | Proximity passes, endpoints 0 and 1 |
|---|---:|---:|---:|
| Apple A | 39 / 0 | 54 / 0 | 10/33, 10/33 |
| Apple B | 11 / 2 | 11 / 2 | 4/15, 2/15 |
| Milk box A | 50 / 0 | 34 / 0 | 8/37, 8/37 |
| Milk box B | 6 / 2 | 6 / 2 | 0/14, 0/14 |
| Mug A | 0 / 0 | 0 / 0 | 11/57, 8/57 |
| Mug B | 1 / 1 | 1 / 1 | 0/22, 3/22 |

Mug A endpoint 0 satisfies 11/57 proximity witnesses instead of the pooled reference's 10/57, but its maximum normalized violation worsens from 5.0 to 5.445343. Endpoint 1 satisfies 8/57 and has maximum violation 5.289485. The frozen ranking correctly refuses both; an extra passing example cannot compensate for a worse worst violation. No orientation conclusion is available for Mug A's empty retained orientation set.

Common invalidity reasons include closest hand points inside/on the visual object, object queries inside the hand, ambiguous mesh facets, buried closest points, source-band loss and nearest-part changes. These reasons overlap. Exact orientation was not measured at the old tip-infeasible endpoints, so the comparison does **not** establish that the tiny restoration itself caused this invalidity. It reveals that the restored configurations still lack admissible preserved surface evidence.

## Work and verification accounting

There are exactly **1,101 tip-FK callback executions + 161 analytic-Jacobian callback executions = 2,524 projection `mj_forward` calls**, because each callback stages two static forwards. Cache hits do not consume these callback budgets. Scene/source checks and final-source FK calls are separately counted per case. There are **24 exact retained-surface checks**, four per case, below the frozen maximum of five. All declared per-endpoint budgets are respected. Numba is set and asserted to two threads after all inherited imports.

The method performs no physics integration, HUG inference, new native grasp generation, evaluation-label generation, prediction-model training or production promotion. All six returned reference poses remain subject to their previously recorded surface failures. Restoring fingertip feasibility is not a mapping-validity certificate, and these bounded failures do not prove that no acceptable Shadow mapping exists.

## Evidence

- `protocol.json`: frozen source states, source/code hashes, bounds, budgets and review receipt.
- `candidate_targets.json`: all six unchanged returned references and all twelve restoration results.
- `restoration_summary.json`: compact endpoint selection, rank, guard and execution details.
- `endpoint_outcome_diagnostics.json`: saved geometry reasons and correction sizes.
- `traces/*.jsonl.gz`: every FK/Jacobian/cache/checkpoint/refusal event.
- `bundle_manifest.json`: SHA-256 of this local bundle, checked again after writing.

Independent final numerical review is maintained in the separate reviewer bundle; its result is not inferred from optimizer convergence. No full-grid acceptance result is claimed here.
