# HUG-only static surface refinement

14 October 2026. This temporary evidence bundle refines the existing saved
cube/seed-0 HUG-to-Shadow comparison. No new grasp proposal, generator sampling,
physics rollout, task outcome, evaluation dataset, or fitted model is produced.

`protocol.json` was written before the measurement run. It specifies centered
cube-face cell grids at 2, 1, and 0.5 mm and diagnostic surface bands at 1, 2,
and 5 mm. These bands are not mapping acceptance thresholds.

`refine_hug_surfaces.py` reuses the setup-only prefix of the preserved previous
audit after verifying its manifest hashes. That prefix reads native meshes and
existing compiled assets and computes static forward kinematics. The previous
audit's measurements and output-writing section are never executed. No source
code, previous report, controller, retargeter, or predictor is modified.

## Distinct measurements

1. **Occupied-or-near global footprint:** whole closed MANO mesh winding
   determines native occupancy; actual compatible per-geom Shadow collider
   membership determines target occupancy. Globally exterior proximity also
   contributes. This reproduces the old metric, including penetrated cube-face
   queries. A negative minimum among primitive SDFs is used only for membership,
   not as exact penetration depth into the union of colliders.
2. **Independent per-part unsigned proximity:** triangles assigned to a native
   anatomical part form an open subset, never a fabricated closed mesh. Exact
   unsigned triangle distances are compared with nearest surfaces of actual
   colliders assigned to the corresponding Shadow part. Parts may overlap in
   these proximity footprints, so their areas must not be added as a partition.
3. **Conditional normal and closest-point comparison:** queries must be
   exterior to both whole-hand representations, near both, and have the same
   nearest anatomical label. Mismatched labels and native-only/target-only near
   sets are retained separately. Normal angle and closest-point separation are
   summaries of this selected subset, not whole-grasp error. Closest points are
   additionally classified by whether they lie strictly inside the cube. An
   exterior cube query can have its closest hand point inside the object.

Native triangle labels use a majority vote of dominant grouped skinning labels
at their three vertices, with an explicit mean-weight tie rule. Mixed and tied
triangle counts are reported. These are anatomical proxies, not verified contact
annotations. Forearm and wrist are distinct carrier categories on the target.

Normals are nearest native triangle or target hull-facet/analytic primitive
geometric normals. At edges or creases, the normal can be nonunique; the returned
facet/primitive branch determines the reported direction. The angles do not
represent contact forces or paired physical contact normals. The native mesh is
not established to be free of self-intersections.

## Reproduction

Preserve this recorded bundle. Copy the script and frozen protocol into another
temporary directory before rerunning. The copied script writes only next to
itself. It requires the existing licensed local MANO asset, which is read and
hashed but never bundled. Use the existing Python environment:

```sh
PYTHONPATH=/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages \
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python /absolute/temporary/copy/refine_hug_surfaces.py
```

Every MuJoCo stepping API is replaced with a throwing guard. The report asserts
zero simulation time, unchanged object state and unchanged static joint state.
It checks the actual object is a centered axis-aligned 50 mm box, examines
collision bitmask and body-exclusion compatibility, and verifies the original
2 mm grid's 2 and 5 mm global footprints. A five-case analytic single-triangle
check exercises face interiors, edge projection, vertices and on-surface points.

This remains one development grasp. Grid refinement can establish numerical
stability of these diagnostics; it cannot establish equivalence of human and
robot hands, physical contact preservation, or generalization across grasps.
HUG grasp/contact fidelity remains unconfirmed without justified tolerances,
representative independent cases and outcome-sensitivity evidence.

## Measured results

The complete run took 172.31 seconds, with 3,750, 15,000, and 60,000 cube-face
queries at the respective resolutions. Every state/hash invariant passed. Both
original 2 mm-grid global baselines reproduce exactly, including areas and IoU.

As grid spacing decreases from 2 to 1 to 0.5 mm, occupied-or-near IoU is:

- 1 mm band: 0.500000, 0.510619, 0.512000.
- 2 mm band: 0.582946, 0.580406, 0.578557.
- 5 mm band: 0.681668, 0.681060, 0.681128.

These tested resolution changes retain the geometric discrepancy. They do not
prove convergence to a continuum metric or validate a mapping tolerance.

At the 0.5 mm grid and 2 mm unsigned band, per-part IoUs are index 0.16534,
middle 0.55084, ring 0.22969, and little 0.19515. Palm, thumb, forearm carrier,
and wrist carrier footprints are empty in both representations, giving null
IoUs and conditional statistics. Their emptiness is specific to this pose and
band; it does not establish finger correctness or a successful native grasp.

The little-finger +Z proximity region is nearly lost rather than strictly
absent at the finer grid: native area 57 mm², target 0.5 mm², and shared 0.25 mm²
at the 2 mm band. At 5 mm, the native little-finger +X area is 39.5 mm² with no
corresponding target footprint. The target index -Y area is 132 mm² with no
corresponding native footprint.

For the fine grid's 2 mm band, native and target globally exterior near counts
are 2,598 and 3,316. They share 832 queries: 748 same-label and 84 different-label
queries. There are 1,766 native near queries without target exterior-near
correspondence (441.5 mm²), and 2,484 target near queries without native
exterior-near correspondence (621 mm²). These remain explicit denominators.

The same-label subset has median geometric normal angle 25.56° (95th percentile
84.86°) and median closest-point separation 0.866 mm. Both closest points lie
strictly inside the cube for 548 queries, exactly one does for 100, and neither
does for 100. On that last subset, median angle is 59.99° and median separation
1.086 mm. Near, label-matched conditional distances therefore cannot be read as
whole-grasp agreement or physical contact preservation.

There are 110 mixed-vertex-label native triangles and four three-way label ties
out of 1,552. The paired same-label near subsets do not select those mixed or
tied triangles at the declared bands. This narrows label ambiguity for those
subsets but does not make skinning-derived anatomy ground truth.
