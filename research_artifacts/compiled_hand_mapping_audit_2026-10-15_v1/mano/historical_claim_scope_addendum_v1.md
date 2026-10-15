# Addendum: compiled mesh identity defect and historical claim scope

15 October 2026. The earlier `saved_mano_conversion_audit_v1.json` PASS covers native landmark decoding, frame placement and independent XML link kinematics. It **does not cover collision-mesh file identity**. The subsequent shared-target investigation demonstrates a separate importer defect.

The pinned URDF deliberately assigns `meshes/visual/f_distal_pst.obj` to the index distal collision geom, but `meshes/collision/f_distal_pst.obj` to middle, ring and little distal collision geoms. The old MuJoCo URDF import reuses the first mesh by basename. As a result, those three finger links receive2691vertices/5374faces instead of the declared526vertices/1048faces. This is not a MANO landmark conversion error or a change requested to the target hand. A correct importer must preserve the existing URDF declarations.

`collision_mesh_alias_independent_v2.json` independently compares an original importer reproduction with isolated explicit-name MJCF imports of each declared mesh. Index geometry matches exactly. Middle/ring/little each fail identity. A maximum over every referenced actual-mesh vertex finds a3.829528mm point-to-declared-triangle-surface distance. The reverse sample maximum is0.041267mm. These concern original mesh triangle surfaces, not a guaranteed continuous maximum or runtime contact boundary.

Ordinary MuJoCo mesh contacts use convex geometry; separate independent convex-hull-boundary samples are0.024512mm actual→declared and0.019194mm declared→actual. They should not be confused with the larger triangle-surface discrepancy. `mesh_alias_witness_recount_v2.json` independently replays all12 extreme witnesses using plane projection, barycentric inclusion and every clamped edge segment against every target triangle. Maximum disagreement is7.996e-19m. Explicit scalar-sum vertex rotations and BLAS transforms agree to6.939e-18m, with all arrays finite.

The initial Trimesh meter-coordinate closest-point calculation had numerical discrepancies in the small reverse/hull cases. Its v1 report and failing independent recount remain preserved. V2 evaluates Trimesh in millimeter coordinates and converts outputs back to meters; all reported extreme points then match the separate meter-coordinate analytic oracle. The large3.829528mm triangle-surface witness agrees in both versions. These scratch measurement corrections do not change target geometry, poses or acceptance thresholds.

## What remains valid, and what requires a new check

- Original native MANO reconstruction receipts remain exact and unchanged.
- Current MANO landmark/frame adapter checks and independent link FK remain valid within their stated scope. Joint frames do not detect a mesh-content substitution.
- Existing HUG and ContactGen reports that derive collider geometry from `MjModel.mesh_vert`, `geom_dataid` and convex hulls measured the **historical imported target geometry**. Their saved arrays and scoped geometric counterexamples are not retroactively rewritten.
- Any broader interpretation that those tests proved the imported mesh identity matches every collision file declared in the pinned URDF is unsupported and now contradicted for these three distal links.
- Future mapping validation should use the corrected compiled target and bind both declared file identities and realized compiled geometry. Prior surface PASS/FAIL cannot simply be copied to that corrected import. A newly run static audit must state its model/version and reuse the unchanged scientific criteria.
- The ordinary mathematical MANO→Shadow morphology mismatch is still present. Fixing these import defects does not by itself establish scientific mapping acceptance or permit evaluation episodes.

Reviewed historical geometry path: `research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py` `geom_rep` constructs hulls of the actual compiled mesh vertices; its `load_case` loads the saved compiled scene. The ContactGen cohort helper copies this declared actual-compiled geometry path. Their triangle-distance functions are custom plane/edge routines, not the scale-sensitive Trimesh helper used in this scratch audit's initial attempt.
