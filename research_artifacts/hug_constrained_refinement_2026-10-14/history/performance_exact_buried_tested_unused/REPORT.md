# Exact buried-feature work elimination — static performance verification

14 October 2026. This separately versioned helper omits unused nearest-triangle
work in the original buried-feature predicate. Existing frozen helpers, source
files, solver settings and ongoing fits remain unchanged. No constrained solve,
new inference, physics step, eval record, task label or training ran here.

## Why the branch preserves the declared predicate

The old mesh query computes complete local coordinates and the literal inside
predicate:

```python
local = (points - geom_position) @ geom_rotation
inside = max(local @ plane_normals.T + plane_offsets, axis=1) <= 1e-10
```

It computes exact triangle distance for every point, then assigns negative
distance only when `inside` is true. The buried predicate is signed distance
less than minus the existing tie tolerance. Therefore an old `inside=False`
point cannot be buried: its signed distance is nonnegative. That point's
triangle distance, nearest point and normal were unused by the predicate.

The new helper evaluates the **same complete local-point and halfspace matrix**
before selecting rows. It performs no AABB rejection, halfspace-distance
approximation, changed inside threshold or proximity threshold. For inside
rows it uses the original `nearest_parts` kernel, identical coordinates,
triangle order and zero part labels, and tests `distance > tie_tolerance`.
Point-wise triangle arithmetic is unchanged; only other independent query rows
are omitted. Primitive geometry retains the literal original `geom_query`.
The rest of `mapped_direction_fields` remains unchanged.

## Observed equivalence

The original 13 manufactured invalidity cases pass unchanged, with full-field
comparison on all 20 original fixture calls. A twenty-first call explicitly
tests a mesh feature buried inside another mesh, exercising the retained exact
interior branch. Separate mixed interior/boundary/exterior point vectors also
match the original negative-depth predicate.

All six original HUG poses plus three predeclared static FK perturbations pass:
**nine records**. All returned array values/dtypes, reason masks, ties, scores,
eligibility, nested diagnostics and matching NaN positions equal the original.
Every original eligible query is included beyond the fixed quadrature bins.
Native reference sites, targets, weights, available parts, coverage and
denominators remain identical. Object poses remain unchanged and time is zero.

This is exact observed value equivalence on the declared domain, with a
point-wise arithmetic argument. It is not a global floating-point certificate
or a claim that every complete nonlinear optimizer trajectory is bit-identical.
It is not mapping fidelity acceptance or physical-contact equivalence. A new
runtime version, if adopted, still needs its own source-bound independent audit.

## Performance and retained work

Two timing repetitions with both variants at NumBa two threads observed
**1.40–1.55× normal-callback speedup**; the ratio of summed median latencies is
**1.48×**. Measurements occurred under ongoing V2.1 host load. They are neither
confidence intervals nor a guarantee of complete-fit speedup.

All timed mesh buried-query points in these nine records were exterior under
the original inside predicate, so their nearest-triangle calls were skipped.
Exact work remains mandatory for every interior row; the manufactured
mesh-in-mesh case verifies that path. Query and work counts are preserved in
`summary.json`. Primary hand-surface queries, ties, mesh-facet checks, visual
membership, primitives and other callback work remain in place.

Protected/source hashes pass before and after. Three integration guard
self-tests pass with zero actual integration attempts. No scientific budget,
weight, constraint, candidate-selection rule or strict acceptance check changes.
This benchmark did not authorize or perform a restart.

Primary evidence: `benchmark.json`; manufactured evidence:
`manufactured_buried_parity.json`; declared bindings/settings: `protocol.json`;
derived timing/work counts: `summary.json`. New implementation:
`exact_direction_buried.py`. Publication selection excludes duplicate partial
reports, licensed assets, scene binaries and raw point arrays.
