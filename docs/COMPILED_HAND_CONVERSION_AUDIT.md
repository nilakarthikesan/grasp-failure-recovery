# Compiled native-to-Shadow conversion audit

15 October 2026. **Two real conversion defects are fixed; generator-wide acceptance is not claimed.**

The shared importer silently substituted the index finger's visual fingertip mesh for the middle, ring and little fingers' declared collision meshes because their filenames shared a basename. It also rounded joint limits and frames while serializing/reloading compiled scenes. The repair gives distinct files distinct import identities and uses live MuJoCo specifications. Target URDF/mesh bytes, scientific criteria and control parameters remain unchanged.

Independent replay covers **373 saved pose representations across HUG, ContactGen, UniDexGrasp, DexGraspNet replay and DGA**, plus three static controls, on base/tabletop/actuated models. Corrected target FK agrees within 4.05e-16 m; joint limits and analytic sizes agree exactly. Five false compiled-limit rejections disappear. Twenty focused static regressions pass. Final model snapshots exactly match the independently audited models. No physics, native inference, retarget fitting, evaluation episodes/labels or training ran.

New scene exports use exact self-contained MJB snapshots; historical XML remains readable. The live-spec edit API replaces rebuilding an arbitrary compiled model through global XML state. The archived implementation and runtime hashes distinguish this correction from historical runs.

Original surface failures and source-invalid poses remain recorded. Earlier surface checks describe the historically imported model; corrected geometry needs its own unchanged provider-specific checks. DGA's real palm-geometry mismatch and scoped feasibility obstruction are not removed by fixing the three distal meshes. This establishes a reliable conversion boundary for further validation, not physical success or a revised acceptance rule.

[Full findings, generator-specific scope and evidence index](../research_artifacts/compiled_hand_mapping_audit_2026-10-15_v1/README.md). [Before/after measurements](../research_artifacts/compiled_hand_mapping_audit_2026-10-15_v1/control/compiled_pose_comparison_summary_v1.json). [Static regression results](../research_artifacts/compiled_hand_mapping_audit_2026-10-15_v1/control/final_static_tests_v2.txt).
