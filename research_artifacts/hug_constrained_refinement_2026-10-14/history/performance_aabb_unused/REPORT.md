# AABB membership pilot — preserved, not activated in runtime V2

14 October 2026. This separate static pilot tests skipping visual-mesh distance
queries when distance to the mesh AABB already guarantees outside/nonboundary
classification. A conservative arithmetic fallback shell sends borderline
points through the original routine. Every remaining point uses the original
exact path, and the original flags/diagnostic schema remain unchanged. The
fallback shell controls shortcut eligibility; it does not alter the declared
object boundary threshold.

Manufactured concave, interior, boundary, near-boundary, far-outside and empty
fixtures pass. All six original HUG cases plus three fixed FK perturbations
also retain exact normal-field array values, reason masks, scores, eligibility
and nested diagnostics, including matching NaN positions. Fixed native
reference sites/weights/coverage and every original eligible query are preserved.
One- and four-thread variants are included in the separate pilot timing report.

The AABB shortcut alone gave small or noisy full normal-callback changes on
these cases. Runtime V2 therefore uses the **unchanged original membership
helper** and separately verified two-thread execution. This prototype was not
activated in that runtime or in any constrained fit, default mapper or eval
path. The pilot remains independent version history.

The AABB distance argument holds in real arithmetic; the finite tests establish
observed equivalence within the measured domain, not a global floating-point
certificate. No solver weight, constraint, candidate selection or scientific
acceptance rule was changed. Protected/source hashes remain unchanged,
integration attempts are zero, object state/time remain fixed, and no inference,
eval data or model training occurs.

`benchmark.json` contains the nine records and manufactured checks;
`protocol.json` declares source bindings and timing settings. The implementation
is `visual_membership_aabb.py`, and `benchmark_acceleration.py` reproduces the
pilot using existing local assets/caches. The publication selection excludes
licensed assets, scene binaries and raw point arrays.
