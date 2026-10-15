# Independent seed-stage arithmetic and binding review (2026-10-14)

Result: **PASS** (214 checks; 0 failures).

All 30 native-relative fingertip caps and all six strict 10 mm means were
recomputed from the stored ORIGINAL/final 3D error vectors using declared
sqrt(einsum) arithmetic. All six returned poses differ from their originals,
respect independently parsed source-URDF and original ±0.35 rad bounds, and
improve the frozen objective. Old raw seeds, native identities, callback/event
counts, fresh terminal calls, source/protocol/code hashes and shared objective
ASTs were checked. Source fixed-URDF/scene and palm-convention checks are
recorded below their declared limits; this review did not rerun FK/geometry.

There were **18 actual full surface calls**, three per case. Cheap projections
converged for **2/6** cases; **4/6 reached their 40-iteration budget** while
returning exactly feasible improving points. Selection is legal under the new
method, but this result is not six converged optimizations or a global claim.

Candidate SHA256: `90fd9186fb72c174644fd806b1af26f49e372d066c36295430a37636b3c9d2c5`.
Protocol SHA256: `b0e095b39897c6c2d337f53b0c26d82ef5bd6b7632d3c21f60dd9d8e574a98c6`.
Full check details and hashes: `independent_seed_numerical_review.json`.

Scientific surface mapping acceptance remains **pending** the independent
unchanged audit. No full geometry calls, physics steps, proposals, evaluation
episodes or training occurred in this review.
