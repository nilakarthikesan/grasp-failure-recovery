# Saved MANOSim construction ablation, v1

This is a fixed diagnostic on saved development cases20 and28, not a new hand
model or an acceptance procedure. No geometry/field execution is authorized by
these source files alone. The parent freezes the source, inputs, protocol,
runtime and output roots before a single bounded run.

Inputs are the completed literal-shape native transfer audit, its corrected
independent saved-array receipt, the producer's neutral geometry/weights/hull
faces, serialized OBJ/XML files, saved compiled body pose readbacks, and the
unchanged original query/native caches. The native decoder is never called.
The only licensed pickle read takes the original float64 `weights` array as
data: the frozen native face-label rule used these values, whereas the saved
producer tensor is float32. Group/face-label equality between the two versions
is measured separately. No MANO layer is constructed or evaluated.

The exact ordered scans are:

1. `native_reference`: saved posed778-vertex,1552-face closed skin, original
   anatomical group-sum/majority/mean-tie labels. Recomputed six-part distances
   must agree with the frozen native cache within1e-12m and all field masks
   exactly. Failure stops before generating any later diagnostic geometry.
   This control establishes evaluator consistency only.
2. `posed_uncapped`: remove only the14 faces absent from the upstream1538-face
   source topology. Membership is found by vertex tuple, without assuming face
   ordering. Retain original anatomical face labels.
3. `neutral_lbs_uncapped`: apply saved absolute bone transforms to the exact
   saved neutral vertices and original float32 producer weights in explicit
   float64 arithmetic, with a common world translation applied once. No weight
   normalization occurs. This surrogate combines omission of pose correctives
   with source-arithmetic precision effects; it cannot isolate posedirs alone.
4. `rigid_neutral_uncapped`: change only weighted binding to argmax-bone rigid
   binding, retaining the same1538 faces and original anatomical labels.
5. `rigid_dropped_fixedlabels`: remove cross-bone faces, retaining frozen
   anatomical face labels on surviving faces.
6. `rigid_dropped_bonelabels`: retain identical geometry/support but use the
   upstream owner's anatomical part, isolating ownership-label semantics.
7. `rigid_hulls_fullprecision`: historical stage ID for the saved **pre-OBJ
   producer precision**, not mathematical full precision. Use saved local
   hull faces/vertices without re-hulling. Report each dtype/hash and owner
   cloud/orphan support. This change combines convexification and inclusion
   of owner vertices unreferenced by surviving triangles, plus original
   producer centering precision. It is not a pure dropped-face repair.
8. `rigid_hulls_obj_decimal`: change only local coordinates to exact OBJ
   decimal values; retain the saved hull faces, membership and source frames.
9. `serialized_obj_xml_saved_frames`: replay literal OBJ/XML placement using
   the saved compiled BODY rotations/origins, including palm geometry offset
   and validated unit mesh scale. Unsupported transforms refuse explicitly.
   This change combines XML placement and source-to-proper-rotation
   representation. No compiler mesh frame is reconstructed or assumed.

The final stage is compared to the already saved actual compiled mesh distance
field without an additional scan. Actual compiled mesh triangles/geom mesh
frames were not saved, so a compiler-specific effect cannot be separated from
the final serialization/frame residual. Stage effects are conditional and
nonadditive. All coordinates, shape, anchors, rotations, labels/reference,
bands1/2/5mm,10nm allowance and query radii remain fixed as declared; no fitting
or adaptive variant/witness selection occurs.

`AblationCase(index,p,raw_case_dir,guard_reinstall=None,budget_guard=None)`
exposes `run()`, `.counts`, `.artifact_rows` and `.source_receipt`.
`source_inputs(p)` performs path/data reads only. Required declaration keys are
`native_protocol`, `native_run`, `builder_report`, `mano_asset_path`,
`geometry_base` and `field_rules`. The runner validates completed prior evidence
and all inherited byte hashes. It checks selected Git sources separately from
ignored raw generated data.

The case saves membership provenance, then each stage's complete geometry and
membership NPZ with a creation-time SHA/size before querying it. Field NPZs and
fixed witness results are similarly indexed. Predeclared witnesses include
milk5932/middle/5mm, milk543/ring/5mm, mug32591/index/2mm and
mug14847/palm/5mm. No witness changes alter the query domain or acceptance.

The fixed maximum is600s/case,9 scans,256 query points/chunk and
`query_count*18620` triangle-point tests. The injected guard receives
`('full_field_scan',stage_id)` and `('triangle_point_tests',integer_cost)`
before each operation. No MuJoCo/MANO model, FK engine, forward/integration,
new HUG inference, retargeting, physical task labels or model fitting is
allowed. All generated arrays remain under the fixed ignored
`ROOT/runs/manosim_geometry_ablation_2026-10-15_v1/run_v1/raw/{20,28}` roots;
tracked output consists of reports and immutable hash receipts.

Every stage/result keeps accepted mapping, native fidelity, production
promotion and collection/training permission false. Full orientation and
continuous surface equivalence remain unproved. Upstream masses/gains and
physical calibration are outside this saved-geometry diagnosis.
