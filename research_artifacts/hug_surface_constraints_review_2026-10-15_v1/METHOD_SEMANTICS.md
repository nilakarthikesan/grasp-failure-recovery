# Independent review: explicit surface relationships

15 October 2026. Work only on the six existing HUG native grasps and preserved Shadow geometry. This review does not change scientific criteria, run inference, step physics or create evaluation episodes.

## Sufficient whole-cell constraints

Use the original object-cell centroid, radius, anatomical part, band and numerical allowance. Compute the literal inherited arithmetic: `error = radius + allowance`, `upper = distance + error`, and `lower = max(0, distance - error)`.

- A saved lost-proximity witness certifies the native cell is near: its native `upper <= band`. Require the target's `upper <= band`, so the entire covered cell retains that near relationship under the same assumptions.
- A saved added-proximity witness certifies the native cell is far: its native `lower > band`. Require the target's `lower > band`, so the entire covered cell retains that far relationship. The final comparison is strictly greater, not a nonnegative inequality with numerical slack.

Making a definite violation merely ambiguous does not satisfy these constraints. All native/world source values and every saved row remain fixed. Distances are in meters; optimizer scaling may divide a length residual by a declared 1 mm scale without changing feasibility. A declared inward optimizer buffer can be stricter than the final predicate, but never enlarges its acceptance set. Final acceptance uses the literal frozen arithmetic order, not an algebraically rearranged expression at a floating-point boundary.

The currently identified source set contains 178 capped original whole-cell examples. It is not all failing cells or complete field coverage. The saved orientation source set contains 134 rows, retaining duplicates and nested-band provenance. If these are the method's rows, the protocol must say so explicitly and must not imply complete coverage.

## Orientation and invalidation

Preserve the sign of native `-dot(hand_normal, object_normal)`, including negative native-away values. A target sign within the frozen deadzone is not established agreement. Retain the source witness's relevant band, nearest anatomical part, unique point/normal conditions, query exterior status, outside-object closest point, and buried/tie/degeneracy exclusions. Losing eligibility is a constraint violation or unresolved refusal, never a way to make a witness disappear.

Cheap raw geometric/sign residuals can guide the inner search, but they are surrogate feasibility only. Exact frozen eligibility must be measured for every accepted full-constraint iterate and for each final diagnostic, with invalid rows retained explicitly. No candidate may be called fully feasible based on raw normals alone. A final diagnostic ranking must account for unresolved/invalid rows rather than reward disappearance, and must be frozen before new full-grid outcomes are seen.

Nearest geometry, facet and part changes make distance/normal functions nonsmooth. Bounded finite differences and a finite optimizer are numerical search devices, not convergence or infeasibility certificates. Source-URDF joints, original q ± 0.35 rad, original root quaternion, q-dependent palm compensation, translation domain, native world targets and all literal original fingertip caps remain unchanged.

## Coverage limitation of the frozen interval estimator

Let native definite/possible near masks be `ND` and `NP`, and mapped masks be `MD` and `MP`, with `MD` implying `MP`. The inherited upper mismatch masks are `NP and not MD` (lost) and `MP and not ND` (added).

For a native-ambiguous positive-area cell, `NP=true` and `ND=false`. Zero lost upper would require `MD=true`, hence `MP=true`, hence positive added upper. Every possible target classification therefore leaves at least one upper mismatch for that native-ambiguous cell. The accompanying manufactured identity control evaluates the literal frozen functions and shows positive upper bounds even with identical native and target distances.

This is a limitation of independent interval bounds, not a physical mapping impossibility or a definite failure. Repairing selected definite counterexamples can still leave the scientific result UNRESOLVED. Full sign-off needs stronger coverage evidence under the same strict relationship criterion, in addition to continuous orientation coverage. This review does not implement an alternate certificate or modify the frozen evaluator.
