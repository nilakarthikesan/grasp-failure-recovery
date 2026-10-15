# Literal saved-shape native transfer audit, v2

This is the second, separately frozen stage. It consumes the completed
active `build_protocol_v2.json`/`build_run_v1` transaction and its independent saved-array
readback. It does not build assets implicitly. The unchanged v1 shape refusal
remains historical evidence.

Only saved development natives 20 and 28 are admitted, once each. The complete
two-variant generated asset inventory is fixed by the prior builder receipt:
one exact float32 shape file, two right-hand XML files and sixteen OBJ files.
Every producer artifact is hash-verified, including neutral geometry, body
inertias/masses and unchanged public source snapshots. Licensed MANO data is
referenced and hashed locally without copying it into the evidence bundle.
Generated raw files are hash-bound inputs; source/protocol files alone receive
Git tracking checks. The raw outputs remain under the repository's scoped
ignore rule.

`native_decode.py` and its manufactured controls are byte copies of v1. The
case bridge changes only asset selection/provenance: exact generated shape,
generated variant XMLs, and explicit expected shape SHA plus builder receipt
passed to the static adapter. `manosim_root` still denotes the pinned public
upstream source. Nominal upstream collider XML/OBJ files are not selected.

The source MANO decoder performs two CPU parametric forwards per case (saved
pose and neutral reference). No HUG grasp inference, Shadow retargeting,
physics integration, task labels or predictor fitting is allowed. The parent
worker enforces 600 seconds and 200 total static MuJoCo forwards per case.
Exactly two complete field scans (mesh and capsule) reuse the frozen original
query grids, cell radii and native distances. All relationships remain in the
denominator, including unresolved and definite failures.

The numerical pose comparisons remain 5 micrometres for reconstruction/joint
anchors and 1e-6 for rotation elements. Float64 source-formula root
Gram–Schmidt and Exp of observed mean-adjusted local rotations retain their
measured residuals to source float32 rotations. These limits are numerical
comparisons, never surface/contact allowances. Literal named-quaternion
readback and initial conversion precision are saved separately.

A numerical pose PASS cannot issue native surface fidelity PASS. Same-part
surface fields use unchanged predicates and the full orientation gate remains
NOT_RUN. Every result retains `native_fidelity_accepted=false`,
`production_promotion=false` and `new_collection_or_training_allowed=false`.
Rigid hulls/capsules, neutral LBS geometry and posed MANO skin remain distinct.
Physical gains, damping, masses and inertias are upstream assumptions and have
not been physically calibrated by this static audit.

The earlier `native_protocol.json` was frozen but never executed. A failure
logging correction now retains each creation-time partial artifact row and
verifies it individually; incomplete/unindexed array files remain explicitly
unverified. It does not modify any numerical rule or quota. The corrected
manufactured receipt is `native_controls_v2.json`.

After a successful independently reviewed build, a parent may prepare the
new declaration with `run_native_transfer.py prepare --protocol native_protocol_v2.json`.
Commit that declaration and all bound source/control files before invoking
`run_native_transfer.py run --protocol native_protocol_v2.json --freeze-commit
<exact-commit> --output <fresh-output>`. Preparation and execution are not part
of these source-only controls. Failed cases/partials are retained, no retries
or automatic replacements occur, and the second case remains NOT_RUN after
a hard stop.
