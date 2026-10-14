# Independent HUG guarded-refinement validation — development evidence

14 October 2026. **All six existing poses and all three finer repeats still FAIL the original strict relationship-preservation rule.** No new inference, eval episode, physics rollout, task label or training occurred. The new guarded mapping targets remain development candidates; an improved fitting objective or engineering gate is not scientific acceptance.

## Checks and unchanged scope

The original 32 native exports and their previous numerical reconstruction evidence are unchanged. This derivative checks six existing seed-0 apple/milk/mug poses on the exact original visual-object centroid grids, plus the original first-pose finer repeat for each object. Cube remains a distinct earlier query family. Source URDF physical limits, captured-scene representation differences, 24 named joints, native wrist-to-palm position, source-to-scene FK, all six candidate fingertip guards and exact declared backtracking interpolation pass. Actual physics integration attempts are zero. All **110 protected original production/model/paper hashes** remain unchanged.

Palm **POSITION** stays at the native MANO wrist. Palm **ORIENTATION** follows `R_world_operator @ R_base_palm(q)` and changes when WRJ1/2 change. The carrier translation compensates the wrist-induced palm offset; the native/operator frame, object geometry, units and scale are unchanged. Declared palm rotation changes are 0.501–1.355 degrees; position deltas are exactly zero. These are target-frame comparisons, not anatomical equivalence.

The raw V2 candidates exceeded the existing 10 mm mean-fingertip engineering gate on all six poses (11.650–13.204 mm), and remain recorded refusals. V3 uses a predeclared shared list of alpha values and the same whole-pose baseline objective; alpha is 0.5 in five cases and 0.25 for Milk B. All guarded candidates satisfy that existing gate (8.458–9.290 mm), but fingertip alignment worsens relative to the original 5.845–8.189 mm on every pose. The compound objective is recomputed consistently on six anatomical part strata with mass 1/6, original all-10 link anchors and original full-24 joint regularization. Its implementation was independently reviewed; this wrapper does not independently recompute every scalar objective value.

## Measured tradeoffs on the exact original grids

Whole-hand 2 mm unsigned centroid IoU improves in all six coarse cases, from 0.243–0.273 to 0.294–0.395. The six separate part-area lower-bound sum for lost proximity decreases in every case. Added proximity decreases in five; **Milk A increases from 84.28 to 102.14 mm²**. These summed part areas can overlap; they are not unique object area, physical contact area or task-quality scores. All 1/2/5 mm part intervals and finer-repeat results remain in `summary.json` and the two compressed complete reports.

Below, L/A are sums of the six independent anatomical lost/added **lower** area bounds in mm²; original → candidate. F/N is reliable sampled direction reversals / actual eligible outside-visual, same-part query count. N changes with the mapping, so comparisons of flip counts alone cannot establish better or worse direction preservation. Matching sampled signs never establishes continuous orientation preservation.

### hug_apple_scene_002_seed_0

Strict status: **FAIL**. Guarded mean fingertip error 8.458 mm; declared palm orientation change 1.013°.

- 1 mm: L/A 2.08/1.07 → 2.08/0.00; F/N 0/7 → 0/7.
- 2 mm: L/A 116.34/174.36 → 24.18/113.62; F/N 2/28 → 3/34.
- 5 mm: L/A 159.15/408.52 → 16.29/298.35; F/N 9/112 → 11/177.

Largest remaining 2 mm part discrepancies: ring lost 12.92 / added 56.78 mm², little lost 11.26 / added 51.12 mm².

### hug_apple_scene_003_seed_0

Strict status: **FAIL**. Guarded mean fingertip error 8.559 mm; declared palm orientation change 1.128°.

- 1 mm: L/A 1.10/0.00 → 1.10/1.05; F/N 0/10 → 0/12.
- 2 mm: L/A 103.13/158.30 → 32.69/102.88; F/N 0/38 → 0/52.
- 5 mm: L/A 148.41/435.99 → 11.19/321.89; F/N 4/175 → 6/231.

Largest remaining 2 mm part discrepancies: little lost 12.52 / added 55.76 mm², ring lost 20.16 / added 47.12 mm².

### hug_milk_box_scene_000_seed_0

Strict status: **FAIL**. Guarded mean fingertip error 9.290 mm; declared palm orientation change 0.591°.

- 1 mm: L/A 1.32/1.36 → 1.20/0.96; F/N 0/2 → 3/26.
- 2 mm: L/A 65.08/84.28 → 32.50/102.14; F/N 3/39 → 6/77.
- 5 mm: L/A 245.20/322.13 → 38.67/330.74; F/N 14/225 → 19/319.

Largest remaining 2 mm part discrepancies: ring lost 17.11 / added 55.08 mm², little lost 15.38 / added 47.06 mm².

### hug_milk_box_scene_001_seed_0

Strict status: **FAIL**. Guarded mean fingertip error 8.918 mm; declared palm orientation change 0.501°.

- 1 mm: L/A 0.59/0.00 → 0.59/2.23; F/N 0/3 → 0/7.
- 2 mm: L/A 89.79/142.31 → 58.75/102.52; F/N 1/18 → 2/36.
- 5 mm: L/A 221.71/370.35 → 52.12/273.78; F/N 11/111 → 16/150.

Largest remaining 2 mm part discrepancies: little lost 49.45 / added 73.44 mm², middle lost 1.99 / added 17.82 mm².

### hug_mug_scene_002_seed_0

Strict status: **FAIL**. Guarded mean fingertip error 8.564 mm; declared palm orientation change 1.301°.

- 1 mm: L/A 1.18/5.87 → 0.15/1.11; F/N 0/31 → 0/33.
- 2 mm: L/A 608.96/551.88 → 430.06/425.22; F/N 0/93 → 0/117.
- 5 mm: L/A 3116.25/1339.15 → 2239.21/653.27; F/N 0/352 → 4/484.

Largest remaining 2 mm part discrepancies: index lost 165.01 / added 185.53 mm², little lost 39.31 / added 121.32 mm².

### hug_mug_scene_003_seed_0

Strict status: **FAIL**. Guarded mean fingertip error 8.970 mm; declared palm orientation change 1.355°.

- 1 mm: L/A 1.05/2.27 → 1.05/3.36; F/N 0/31 → 0/43.
- 2 mm: L/A 595.51/385.83 → 390.05/309.92; F/N 0/101 → 0/142.
- 5 mm: L/A 3014.81/982.07 → 2033.61/424.40; F/N 1/469 → 2/682.

Largest remaining 2 mm part discrepancies: index lost 254.35 / added 121.97 mm², little lost 35.89 / added 88.66 mm².

## Concrete remaining failures and next technical work

Rechecking all **178 pre-existing cell witnesses** under the guarded targets leaves **65 definite counterexamples** (38 lost, 27 added), 100 unresolved cells and 13 definite matching cells. A witness ceasing to be definite does not mean it is repaired. This fixed old-witness subset is diagnostic; it does not enumerate every new candidate counterexample or replace the full-grid rule. No new cells were selected.

For Apple A, the same middle-finger child cell centered at object coordinates (−23.207, −12.615, −36.166) mm has native distance interval [0, 0.937] mm and guarded interval [1.226, 3.070] mm. It remains definitely lost at the 1 mm band despite improvement from the old mapped interval [1.910, 3.754] mm. Radius and 10 nm query-error assumptions are unchanged.

Apple/milk failures concentrate in ring/little added regions, along with retained middle-finger losses. Mug failures still include index lost/added regions and substantial 5 mm palm loss (912.26 and 648.79 mm²), despite the small permitted wrist rotation. Reliable direction-state reversals remain or appear on the enlarged eligible domain; examples and exact exterior/tie/buried flags remain in the complete orientation reports.

The next candidate should constrain both directions of band preservation at these fixed existing witnesses and preserve the source relative facing/away states at stable, eligible features, while retaining the original fingertip gate. Minimize distortion rather than reward contact quality or erase bad native proposals. Inspect mug index/palm collider extent and local reachable surface positions under the same physical limits and anchor convention before changing shape or placement. This evidence does **not** prove the differing morphology makes preservation impossible. Any new solver/objective/proxy version must be declared separately before its measurements and reevaluated on all cases, including regressions.

## Limits and decision

The rule remains SHA-256 `87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4`. Distances to open native anatomical patches and individual mapped collider surfaces preserve the prior literal definition; buried target features are excluded from eligible normals but may contribute to unsigned distances. Conditional distance intervals assume the stated uniform numerical allowance. A formal global arithmetic certificate, exact accessible union boundary, continuous normal-cell proof, physical contact equivalence, failure-label equivalence and independent unseen-object validation remain unavailable/not tested.

**Decision:** numerical and engineering checks pass within this derivative scope; guarded static proximity overlap improves, with explicit anatomical and direction tradeoffs; strict relationship preservation still FAILS on all nine resolution checks. No source-wide accurate-mapping badge or automatic production/default promotion follows.
