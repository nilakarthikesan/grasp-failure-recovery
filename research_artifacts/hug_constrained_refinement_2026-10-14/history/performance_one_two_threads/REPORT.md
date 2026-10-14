# Unchanged HUG normal helper: one-versus-two-thread verification

14 October 2026. This is a separate static runtime benchmark. It did not run
the constrained solver, modify ongoing V1 fits, generate a proposal, step
physics, create eval data or train a model.

The original `exact_direction.py` remains byte-identical:
`c20380677f727206830e3e87beebd91948009ffb83dc341c53dad2fc5df9baae`.
Only the NumBa thread count changes between timed variants. BLAS/OpenMP thread
environment settings are held at one in this benchmark. The AABB shortcut was
neither imported nor activated.

## Evidence and performance scope

All six original saved seed-0 poses are covered, plus three predeclared static
FK perturbations on the first apple, milk-box and mug cases: **nine records**.
The perturbation adds 0.02 rad with alternating signs across the 24 joints,
clipped to actual URDF limits. Carrier rotation and the existing palm-position
compensation convention remain unchanged. These are deterministic geometry
benchmarks, not generated inference proposals or independent held-out grasps.

For each record, one- and two-thread `mapped_direction_fields` results have
exact array-value equality, matching dtypes and NaN positions, and equal nested
diagnostics. This includes distances, closest points, normals, actual scores,
part labels, tie counts, reason masks, eligibility and visual membership.
The comparison also covers every original eligible query, beyond the fixed
quadrature bins. `freeze_direction_sites` retains identical native targets,
query/weight order, available-part coverage and all denominators on all six
original cases.

Consequently the unchanged deterministic elementwise fixed-bin residual helper
has identical inputs. Residual equality follows from those equal inputs; it is
not a separately recomputed residual measurement. No loss, constraint, source
normal target, scientific threshold or acceptance rule changes.

Two timing repetitions per variant observed **1.42–2.01×** normal-callback
speedup. The ratio of summed one-thread to summed two-thread median latencies
is **1.74×**. These are measurements under concurrent V1 host load, not
confidence intervals or a guaranteed speedup for complete fits. Other geometry,
FK, optimization and process scheduling costs remain outside this timing claim.

## Checks and bounds of the claim

All protected and source hashes pass before and after; all object poses remain
unchanged and time remains zero. The three integration guards pass their
self-tests, with **zero actual integration attempts**. Native inputs, solver
settings and existing V1 source files are preserved.

The evidence establishes exact observed helper output parity on this declared
domain. It is not a global floating-point proof for every possible input or a
mapping fidelity sign-off. The independently frozen scientific checks still
control candidate validation. This benchmark supports a separately frozen
runtime V2; it does not overwrite V1 experiment history.

Primary evidence: `benchmark.json`; declared settings/source hashes:
`protocol.json`; concise derived timing/scope checks: `summary.json`.
`benchmark_two_threads.py` reproduces the verification from the existing local
assets and saved query caches. No licensed asset, scene binary or raw point
array is included in the publication selection.
