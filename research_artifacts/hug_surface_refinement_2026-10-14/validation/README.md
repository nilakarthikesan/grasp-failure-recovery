# Independent guarded HUG refinement audit

Read `REPORT.md` for the decision, `summary.json` for every part/band comparison,
and the two compressed reports for complete original/candidate diagnostics.
All six existing poses plus three finer repeats remain strict FAIL under the
unchanged rule. Native decoding, source bounds, placement/FK and existing
engineering fingertip checks have separate scopes and statuses. This contains
static research derivatives only: no new inference, evaluator episodes,
physics integration, labels or training.

The candidate comes from the sibling refinement implementation's guarded V3.1
protocol `59e04a7af29827c8c3b11e2c2278a553e2a0385c3016ed8bb20037b7042a8733`.
The independent full-grid protocol is
`0c293119f406a7f157c0e6f05a8c4b44e9fd2f0755d463b21aeb637f4e432eaf`.
Two disjoint cohorts contain all six cases and nine original grids. Gzip binding
JSON files record exact uncompressed-report and compressed-file SHA-256 values;
compression does not discard fields. Uncompressed per-case checkpoint reports
are intentionally omitted from the publication manifest because the final
compressed reports already contain them.

The wrappers reuse the exact prior geometry, cell and orientation libraries.
The strict proximity rule uses conditional radius-plus-numerical-allowance
intervals, and the direction rule retains actual proximity/exterior/tie/buried
eligibility denominators. Objective improvement and the 10 mm engineering gate
are never a replacement scientific mapping gate. The whole-pose global
objective implementation was reviewed, while its scalar values are explicitly
not independently recomputed by this wrapper. Native landmark errors, source
URDF FK, q interpolation, largest recorded acceptable alpha, limits and the
declared wrist/palm anchor policy are independently checked.

`existing_witness_recheck.json` applies the candidate to all 178 already selected
original counterexample cells: 65 remain definite, 100 unresolved, 13 definite
matching. It chooses no new queries and is not a complete candidate-witness
inventory. `palm_orientation_changes.json` uses already saved target matrices:
the palm position stays at the native wrist; wrist articulation changes its
orientation by 0.501–1.355°. Identical operator-frame serialization drift cancels
in the relative-matrix solve; no mapping transform is fitted or changed.

`protected_110_verification.json` confirms all 110 original protected
production/model/paper files at the completion of the static phase. The frozen
scripts also check input assets and original target/scene files. Case logs and
reports confirm zero time advancement and unchanged object state. The
independent wrapper review is code review, not an acceptance certificate.

## Reproduction

The retained licensed MANO asset, saved native grasps/scenes, source URDF,
original exact query caches and installed recorded Python runtime are required.
Their original absolute paths and hashes are preserved in protocols and prior
bundles; licensed assets and large query arrays are not copied here. Original
caches can be recreated using the prior frozen visual-surface scripts when the
same preserved inputs/runtime are available; their exact hashes must match.
Run `reproduce.sh` only in the recorded local workspace arrangement. Relocation
needs an explicit new binding protocol rather than pretending unchanged source
hashes are portable paths. This reproduction remains static and guarded.

Version directories preserve the initial unexecuted candidate wrappers and
binding amendments. Only V3.1-bound full-grid reports supply candidate results.
V1 manufactured/baseline regression and V2 original-pose anchor sanity are
separate checks, explicitly bound to their archived protocol/script versions.
No discarded candidate metric version is pooled into the final reports.
