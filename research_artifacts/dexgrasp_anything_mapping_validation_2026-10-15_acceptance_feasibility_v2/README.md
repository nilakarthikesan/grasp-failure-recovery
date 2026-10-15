# DGA acceptance feasibility audit — 15 October 2026

**Decision: NOT ACCEPTED. Target, original acceptance criteria, source outputs and declared input coverage remain unchanged.** No physics, new native inference, evaluation episodes/outcome labels or training occurred.

This continuation follows the shared compiler repair. It preserves the original DexGraspNet results, native MultiDex outputs and each separately named correction pipeline. It does not promote the DGA provider.

## Results

| Check | Result and scope |
|---|---|
| Saved raw cohort | All 16 retained: 11 source-valid surface diagnostics, 5 source-invalid rejections. Ten of the 11 raw mapped poses are target-illegal diagnostics. |
| Unchanged-angle raw row 6 | The sole source/target-legal raw pose FAILS index/ring visual preservation at all 0.5/1/2 mm bands. At 1 mm: lost-area lower bounds 50.25/66.9375 mm². Palm nearsets/occupancy pass through empty sets; direction unresolved. |
| Mesh/frame diagnosis | All eight index/ring source triangle sets reconstruct exactly. Nominal link frames agree within 2.50e-16; distal meshes really differ (970 vs 5,374 triangles, one local extent difference 2.171777 mm). No missing/doubled mesh rotation found. |
| Old development row 1 | Exact declared-math obstruction under original max/SSE guards: unavoidable added palm occupancy square ≥0.000104947 mm². |
| New native row 0 | Exact declared-math obstruction under original max/SSE guards: unavoidable added palm occupancy square ≥0.525625 mm². Stronger component-guard result remains separately labeled. |
| New band-focused correction of new row 1 | 2,238 trial states, two eligible improvements. Selected THJ3 +1e-5 rad; palm and maximum landmark error unchanged; maximum thumb fitting difference improves ~46.4 nm. Fresh independent three-grid visual validation FAILS index/ring/little/thumb at all three bands. |

The obstruction quantifies over every proper target palm placement and finger articulation satisfying the indicated original mathematical guards. It uses exact rational outward intervals for the saved robust-6D pose and source wrist FK, removing the old assumed 1 nm source-transform bridge. This is not a whole-generator impossibility theorem, a physical outcome claim, or a proof of arbitrary floating-point guard implementations. The required cohort nevertheless cannot all satisfy the unchanged declared-palm occupancy and landmark requirements. Additional optimizer strength cannot remove those two certified conflicts.

All 949 files in the previous component-study manifest rehash unchanged. The original static surface criteria still demand complete nearset equality, separate declared-palm occupancy preservation, direction evidence and independent native validation; a finite-grid improvement cannot establish them.

## Verification and evidence

- [Raw surfaces, corrected unit labels](surface/summary_v2.json); v1 remains as historical evidence. The v2 generator checks every m²→mm² leaf; no measurements were rerun for this reporting fix.
- [Authoritative exact certificate](feasibility/exact_palm_obstruction_v3.json), [2,082 controls](feasibility/feasibility_summary_v1.json), [independent review](contract_review/independent_exact_obstruction_review_v2.json). Earlier proof reports and their exact producer-source snapshots remain separate.
- [Source/target finger mesh reconstruction](contract_review/row6_index_ring_mesh_frame_review_v2.json) and [acceptance versus optimizer guard comparison](contract_review/landmark_guard_comparison_v1.json).
- [Frozen correction method](control/band_method_frozen_v1.json), [successful process receipt and aggregated warnings](control/band_process_receipt_v1.json), [independent adopted-state recount](contract_review/band_saved_candidate_independent_recount_v1.json).
- [Fresh candidate visual execution receipt](band_validation/execution_receipt_v1.json), [full visual result](band_validation/results_v1/report.json), and [machine-readable decision](control/final_summary_v1.json).

The raw audit uses 5,318,719 triangle and 85,095,000 polygon queries. The new candidate test uses 5,046,130 triangle queries at newly frozen fractions (13/37,19/41). These are static geometric validation queries, not evaluation episodes or grasp-failure labels. 51 raw-surface controls and nine band-proposal controls pass. Independent reviews recount original proof/area/guard facts without supplying fitting feedback.

The fitting run reaches its explicit 600-second wall boundary (600.5 seconds including completion) after two 800-state start budgets and a final 638-state partial start. This is bounded search, not proof of infeasibility for new row 1. All 221,562 captured NumPy matmul warnings and initial dependency-preflight failure are preserved. Finite-result guards and independent recomputation passed; the warning cause and universal directed-rounding soundness of surface computations remain unestablished. Surface-area statements retain the frozen conditional 1 nm arithmetic allowance; the exact palm theorem has its separately stated scope.

## Preservation and reproducibility

The top-level evidence manifest is finalized after documentation compilation. `dependency_archive.json` maps original absolute input/code paths to byte-identical content-addressed copies, preserving original policy text and hash identities. Reproduction needs the recorded compatible Python/NumPy/SciPy environment and path relocation; copying a source file alone is not an environment guarantee. Any superseded unavailable dependency version is listed explicitly rather than silently replaced. Runtime assets and earlier studies are not rewritten.

No target geometry, acceptance threshold, supported input domain, model checkpoint or controller was changed. No rejected mapping is relabeled as a physical grasp failure. A differently scoped target/fidelity claim/input domain would require a separate scientific decision; this study makes none.
