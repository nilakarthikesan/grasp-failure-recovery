# HUG guarded-fit independent static validation (14 October 2026)

This checks the same six saved development proposals after the separately frozen
guarded fitting study. It creates no new generator proposal, simulation episode,
outcome label, evaluation dataset or predictor training example.

The actual all-six independent source/FK preflight passes. Native wrist/palm
placement, original and final source-URDF versus captured-scene FK, physical
joint bounds, the original-relative correction domain, all 30 exact computed
fingertip caps, strict 10-mm means, source hashes, object pose and zero time pass.
These numerical checks are not a full-surface fidelity certificate.

All six selected **complete scene qpos arrays, including free-root translation
and quaternion**, equal the original arrays bit for bit. Every source, asset,
protocol and query-cache binding remains unchanged. Consequently all nine
original coarse/fine surface reports are **reused, not rerun**, as explicitly
recorded in every resolution's audit provenance. There are zero newly measured
full-grid proximity/orientation comparisons in this bundle. The nine-resolution
coverage is six coarse cases and the same three fine repeats.

Applying the unchanged rule gives strict relationship preservation **0 PASS /
9 FAIL / 0 UNRESOLVED** and proximity **0 PASS / 9 FAIL / 0 UNRESOLVED**.
Pointwise orientation retains the original **0 PASS / 7 FAIL / 2 UNRESOLVED**.
Full-domain continuous orientation and physical-contact equivalence remain
UNRESOLVED; native failure-label preservation and unseen-object generalization
are NOT_TESTED. The previous selected feasible poses' nine orientation failures
remain a different historical result and are not overwritten.

The independent verifier reconstructs recorded residual costs, native direction
targets and weights, each part's SSE/invalidated bins, guard decisions, selection
and fallback arithmetic, callback budgets and fresh terminal exact-q metadata.
It does not replay the entire trace's geometry. Five meaningful copied-record
corruptions are refused; the authentic control passes. The read-only report
closure passes 369 checks. No disappearance of finite witnesses would establish
continuous orientation or native outcome equivalence, and a finite unsuccessful
search does not establish global infeasibility.

`protocol.json` was frozen before independent source/FK or surface execution;
the draft and freeze binding are retained. `full_fields`, `status` and the whole
new-pose density measurement block are byte-identical to the previous wrapper.
All 89 protected numeric dependencies and 23 fixed bindings match. Step guards
record three manufactured self-tests and zero actual integration attempts per
run. The initial import-only attempt lacked supplemental dependency paths and
ended before FK; `preflight_import_failure.log` is retained. The successful run
uses the existing frozen HUG dependencies, with no installation.

Start with `summary.json`, `independent_consistency_review.json`,
`trace_tamper_refusals.json`, `wrapper_parity.json` and `manifest.json`.
Per-case JSON files are losslessly gzip-compressed for the archive; their
original and compressed hashes are recorded. Canonical combined report hashes
and the original temporary execution paths remain preserved.
