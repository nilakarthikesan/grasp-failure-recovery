# Fixed-pose complete-grid orientation qualification, v1

Status: source and manufactured controls only. No saved native orientation queries have run under this declaration. Prepare only reads/hashes prior artifacts and array headers. Root must commit the source, controls and fresh protocol before the once-only run.

This separate diagnostic reuses the exact completed native full-surface v2 saved native/compiled triangles, original object visual triangles and the original case20/28 density2mm query grids (19,174/45,484), radii, six anatomical labels, area weights and 1/2/5mm bands. It does not compile or move a hand. It compares fixed-pose contact-facing geometry, not contact forces, finger articulation or task success. Collection and training remain refused.

The original orientation sign is `-dot(hand outward normal, object outward normal)`. The original numerical thresholds remain: distance ties 10nm; normal-vector difference 1e-6; tangent/facing deadzone 1e-6; hand-boundary 10nm; object-visual boundary 100nm. Every triangle is considered. Competing nearest normals, different closest points, competing anatomical parts, hand-interior query points, buried/on-object closest points, tangent normals and invalid normals are retained as explicit reasons. Degenerate geometry hard-stops rather than being dropped. Global outward orientation is conditional on the original closed, nonselfintersecting surface convention; volume alone does not prove that topology.

For every part, all facets within `d_part(q)+2*r+2*10nm` are inventoried as a conservative superset of possible nearest/tied facets anywhere in the original cell. This uses only the distance Lipschitz property, with two numerical error terms. It makes no angular Lipschitz assumption. Per-cell sign evidence additionally needs one stable interior facet, edge clearance greater than r+10nm, a stable nearest anatomical part, exterior clearance of the whole query cell, and exterior clearance of the closest-point image. Other cells remain unresolved. Facet normal dot extrema are saved even for ambiguous candidate sets. Possible-near cells include original proximity-ambiguous cells; they are not silently discarded.

The complete original-site all-tie sign comparison is the primary output. The conservative cell result is additional evidence. Point or cell reliable sign reversals produce FAIL. Agreement can produce only INCOMPLETE: continuous orientation over the full requested domain and object-boundary multivalued normal coverage are not proved. Areas attached to point counts are centroid quadrature, not certified area lower bounds. Overlapping bands are not independent contacts; point and cell reversal counts remain separate.

Bounded work: exactly two saved hand surfaces per case, 1552 triangles each, chunks of at most128 query points, 600seconds per case (child timeout610s). Each surface performs one complete point-triangle distance pass and one complete hand winding pass. Original object visual distance and winding checks have separate before-work caps of at most two full Q×Fobject passes per case. No model/data/forward/FK, physics, MANO/HUG inference, retargeting, fitting or task labels are permitted. NumPy2.2.6 and exact NumBa/LLVM/Python/runtime source bytes are frozen. The old pure closest-triangle/nearest-part/winding function definitions are AST-extracted without executing their engine-importing parent module.

All inherited3041 native input bindings, exact prior report and independent readback, new object operands and independent visual receipt, immutable original acceptance rule, original query cache, runtime packages and own source are bound. Selected source/protocol bytes must match committed HEAD before and after execution. Raw geometry/features/chunks are stored under ignored `runs/manosim_orientation_qualification_2026-10-15_v1/run_v1/raw`; scalar copies are tracked under this directory's `run_v1`. Creation-time NPZ byte/array hashes are rechecked, while any partially written unindexed files are described without inventing a creation-time proof. A STARTED marker prevents rerun/result selection. Failures retain attempted stage and partial artifacts; subsequent cases are NOT_RUN.

Commands, after root creates the controls receipt and commits the prepared protocol:

```
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python run_orientation.py prepare
# root commits source + controls.json + protocol.json
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python run_orientation.py run
```

Manufactured tests use only tiny synthetic arrays; they do not query the actual saved native/object geometry.
