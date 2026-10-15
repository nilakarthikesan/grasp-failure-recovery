# Independent pre-execution review of canonical source-only calibration

15 October 2026, America/New_York. This reviewer did not run the MANO source
template, retargeting, object-to-hand geometry queries, inference, physics or
eval data generation. The review inspected source/dataflow and exercised only
manufactured frame algebra before the separately authorized source run.

The initial final-draft review verifies 100 source/implementation hashes and
215 checks. Its manufactured example deliberately has a nonzero calibrated
palm offset while the native wrist stays zero. Nonidentity world, neutral palm
and fixed forearm transforms must reproduce native task points through BOTH
the fixed-base and forearm world-placement expressions. Reusing the native
wrist as the optimizer vector origin is explicitly shown to break this
roundtrip. Free wrist joints retain the declared q-dependent palm rotation.

Manual review confirms the exact existing right-hand HUG/manotorch wrapper,
explicit zero-beta/zero-axis-angle flat neutral pose, wrist centering, MCP21
indices 5/9/13 and raw joint correspondence 1/4/10, proper deterministic frame
signs, centroid origins, no scale fit and one C shared across all cases. The
canonical function takes source-model/URDF/settings inputs only and writes C
before reading observed cases. No object/witness score can choose or retune C.
This is a new mapping implementation, not an old frame bug fix or rewrite.

Before final freeze the driver gained a required solver-status gate: unsupported
or nonpositive NLopt results and nonfinite objective values cannot produce an
eligible source pose. Every raw attempt is retained. New-anchor radial analysis
is required before a full surface audit; a numeric/calibration PASS cannot
promote surface validity or waive exact original tip caps.

Two narrowly scoped runtime repairs were reviewed separately in
`limited_runtime_reviews.json`: source-URDF extraction of the forearm transform
when fixed MuJoCo compilation fuses that body (including the explicit neutral
joint-chain check), then exact name-based reordering of all 24 joint values.
Neither changes the canonical transform rule, tolerances or anatomical tip-cap
order. The final runtime requires C to remain byte-identical to its already
exported parent before any new case fit. Earlier source-run aborts belong to
the author's separate preserved bundle, not to this manufactured review.

`aborted_metadata_race/` retains a reviewer attempt that stopped immediately
when the draft metadata hash changed during review. It computed no controls or
source results. The final receipt binds the updated draft; no science setting
was changed to make a result pass.

These are pre-execution implementation checks, not evidence that the new
mapping preserves hand surfaces, failure labels, held-out objects or all HUG
proposals. Original surface acceptance and all source-gate refusals remain
separate measured evidence.
