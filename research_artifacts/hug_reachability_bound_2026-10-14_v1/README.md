# HUG palm reachability diagnostic — existing saved grasps only

14 October 2026, America/New_York. This is a separately frozen static
development diagnostic. No inference, new query site, physics integration,
evaluation episode, success/failure label or predictor training occurred.
The production mapper, fitter domain, objects and acceptance rule are unchanged.

## Measured result and scope

For saved Mug B (`hug_mug_scene_003_seed_0`, inventory 28), every wrist pose in
the current source-URDF-intersected original q +/- 0.35 rad rectangle has at least
one positive-area existing palm cell that is native DEFINITE_NEAR and mapped
DEFINITE_FAR at a frozen declared band. The deterministic local cover contains
149 certified leaves, 297 evaluated nodes and no unresolved leaf. Its minimum
strict leaf margin is 28.7562 micrometers after the declared allowances.

A separately frozen domain-only extension covers the complete physical
source-URDF WRJ1/WRJ2 rectangle: WRJ1 [-0.698131700798, 0.488692190558] rad,
WRJ2 [-0.523598775598, 0.174532925199] rad. It covers Mug B with 171 certified
leaves, 341 nodes and zero unresolved leaves. The minimum strict leaf margin
is 0.706247 micrometers after the allowances. The two already published witness
cells have positive areas 2.3558532 and 3.9636778 mm² at their original 2/5 mm
bands. A different witness can certify each wrist subrectangle; the conclusion
is that no single wrist setting can preserve ALL required palm relationships.
Repeated use of a cell is not summed into a physical area.

The hand root rotation and palm POSITION remain those of the existing mapping
convention. The eight exact box palm colliders attach rigidly to palm and have
only the two wrist hinges as moving ancestors. Finger constraints are ignored;
their 22 joint angles cannot move the palm surfaces. The certificate therefore
cannot be repaired by more finger fitting or by enlarging only the 0.35 rad
wrist correction cap. It does not prove impossibility for arbitrary free-root
hand placements, other mappings or embodiments, other objects, or all HUG
proposals. It does not show failure-label or predictor adequacy consequences.

For saved Mug A (`hug_mug_scene_002_seed_0`, inventory 24), the local cover hits
its unchanged 8,191-node budget and remains UNRESOLVED. It certifies only
67.6838% of that wrist rectangle, with 691 certified and 3,410 unresolved
leaves. This percentage is parameter-space coverage, not an object area,
grasp success rate, failure rate or evidence of global infeasibility.

## Conditional proof ingredients

`runtime_v3/METHOD.md` gives the local-domain derivation; `full_wrist_v4/METHOD.md`
records the separate full-wrist extension. Every box surface point is bounded
by its eight corners. The maximum palm-frame corner norm is 97.3498844 mm;
the radius receives 100 nm of declared arithmetic padding. Rotation telescoping
then bounds motion over a wrist rectangle by rho times the sum of its angular
halfwidths. Hausdorff distance and query-distance Lipschitz bounds combine this
motion with the existing cell centroid radius. All cell certificates use the
worst inherited 100 nm distance allowance plus a separate 100 nm transform
allowance. They remain conditional engineering arithmetic certificates,
not formally outward-rounded floating-point proofs.

Every leaf stores its rectangle, dyadic path, existing witness index, midpoint
distance, motion bound and strict margin. Complete dyadic coverage is checked
by reconstructing every rectangle, prefix freedom and exact integer binary
mass. Closed rectangles can share edges/corners; their interiors are disjoint.
No sampled maximum or local fit failure is used as a global certificate.

Exact original source/scene/command/frame files, native witness reports,
query caches, source URDF, historical geometry functions, cell library,
acceptance rule and implementation are hash-bound before new metrics. Raw
licensed assets, full query caches and new training recordings are not copied.

## Preserved versions

- `unexecuted_v1`: initial frozen method; never executed after review found a
  budget-check placement issue.
- `aborted_runtime_v2`: hard budget cap, output refusal and partition controls;
  failed closed before any bound metric because the actual palm uses boxes.
- `runtime_v3`: exact box support, geometry descriptors and radius witnesses;
  completed local-domain result for both saved Mugs.
- `full_wrist_v4`: separately frozen full physical wrist-domain extension for
  Mug B only, preserving V3 and all budgets, cells and allowances.

Compressed reports preserve exact uncompressed bytes, and manifests record
both forms. The 12.85-second V3 and 0.38-second V4 times describe these static
development diagnostics, not generator inference or full validation speed.

Independent review is reported separately: the recorded collaborator receipt
in V3 is a message transcript, not a claim that this author performed fresh
independent geometry validation. The project records the independent fresh
geometry audit and its own frozen protocol separately.

## Independent arithmetic and provenance review

The saved independent reviewer scripts and results are in
`independent_arithmetic_review/`. V3 passes 103,099 checks; V4 passes 3,581.
They verify every bound input hash, all native NEAR witness/cache provenance,
box corner/radius metadata, complete exact dyadic partitions and every leaf
margin, including 70-digit arithmetic recomputation. V4 retains a 0.7062470658
micrometer minimum positive margin; its double versus 70-digit motion-bound
difference is at most 3.47e-18 m. These read-only reviews do not rerun source
geometry and are distinct from the separately frozen independent fresh box/FK
audit. An initial V4 review adapter status-literal mismatch and its correction
are retained; neither modifies the bound method, data or result.

## Mutable documentation context checkpoint

`source_context_checkpoint/HUG_MAPPING_ACCEPTANCE.md` preserves the exact
explanatory document bytes bound by the original protocols (SHA d7261691...).
Later documentation updates can change the live file without rewriting these
historical inputs. The frozen numeric `acceptance_rule.json` remains separately
bound. Protocol absolute paths and bytes are unchanged; reproducing historical
hash checks requires the declared historical source context or an explicit
verified input overlay, rather than editing recorded hashes.
