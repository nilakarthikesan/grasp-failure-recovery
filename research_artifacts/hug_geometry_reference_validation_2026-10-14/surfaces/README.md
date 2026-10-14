# HUG reference surface comparison — 14 October 2026

Frozen V3 static diagnostics use six previously selected seed-0 development
poses (two groups per apple/milk-box/mug, one MANO shape). They reuse exactly
the hash-bound V4 visible-object query sites and native distances. The three
methods are stored current targets, Dex Retargeting 0.5.0 released Shadow
DexPilot-style targets, and its predeclared metric-scale/no-filter sensitivity.
The released adaptation is not the original Allegro DexPilot implementation or
an independently certified HUG mapping. No inference, dynamics, evaluation
label/episode, model fit or production change occurs.

## Results and scope

At the fixed 2 mm band and coarse area density, whole-hand unsigned proximity
IoU ranges 0.242634–0.272962 for current, 0.046728–0.161568 for the released
reference, and 0.132140–0.229001 for the metric/static sensitivity. Current is
higher in all six paired cases. The finer first-group repetitions retain that
ordering. Per-finger tradeoffs differ: mean native-region retention favors the
sensitivity for thumb and index, but current for middle, ring and little.
Thumb/index conditional means each cover 4/6 native-nonempty cases; other
fingers cover 6/6. All absolute shared/lost/new areas and empty denominators
remain in the report, with no success filtering. The figure shows these
separate summaries rather than presenting one accuracy or pass rate.

Bands 1/2/5 mm, occupied-or-near masks, independent anatomical regions,
conditional exterior nearest-label normals, unmatched/different-label regions
and continuous mapped-minus-native distances remain separate. Target distances
to individual collider surfaces may include buried interfaces for penetrated
queries; they are not exact exterior-union signed distances. Conditional normal
agreement is not object-facing contact compatibility or a paired physical
contact annotation. Whole target includes wrist/forearm carriers; native carrier
comparisons are undefined. Negative distance changes can introduce/repair a
relationship rather than preserve a bad native proposal.

The report's generic mean/median/p95/max are query-count statistics; separately
named area_weighted_mean/median use triangle-area weights. Do not call its p95
area weighted. Child-area density (4 mm², repeated at 1 mm²) is not uniform
spacing or a continuum-convergence proof. These inspected cases are development
evidence, not independent holdout or full-source coverage.

## Numerical checks and preserved history

All 27 method/density calculations finish with physical source-URDF bounds,
time zero, fixed object pose and fixed targets. Current bands/whole/part areas
reproduce previous V4 exactly (maximum difference 0). Three stepping-guard
selftests pass; actual attempted steps are zero. Protected production code,
predictor/manuscript and input hashes remain unchanged.

V1 was unexecuted. V2 added full finite per-collider queries for continuous
profiles, then stopped on a saved-scene decimal range difference. Its exact
script/protocol/status and scene-bound diagnostic are under v2_aborted;
V3 was frozen before its run. V3 checks source-URDF physical limits and permits
only the measured source-vs-scene bound representation difference plus 1e-9 rad
for static FK, without changing q or colliders; stricter scene excess remains
logged. The key strict_scene_1e9_tolerance_pass means a 1e-9 rad allowance, despite
its historical spelling. It is not a production rollout-preflight result.
The legacy reference report's limits.physical_lower_rad/upper_rad table actually
contains captured-scene bounds; see reference/corrections. This comparison
correctly parses the original source XML and is unaffected by that label.

## Reproduction

Run `sh reproduce.sh` with preserved original local inputs/assets/runtime. It
uses the preceding surface bundle's library and cached V4 query arrays, whose
hashes are in protocol_v3.json. Arrays are static derivatives, kept local and
not copied into Git. If missing, recreate them with the preceding bundle's
published V4 script and verify the required hashes before this run. The exact
reference report is restored from its gzip archive into a new temporary folder.
Measurements go to a separate temporary folder, leaving published evidence
untouched. Frozen protocol files are copied, not regenerated. Timing and script
path metadata can differ; geometry/configuration/hash bindings should agree.
No licensed native asset, checkpoint, scene binary or scratch array is bundled.
