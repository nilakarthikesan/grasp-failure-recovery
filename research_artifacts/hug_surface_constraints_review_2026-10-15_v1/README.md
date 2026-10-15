# HUG explicit-surface-constraint V1 independent review

This directory records independent method, provenance, literal arithmetic and static FK checks for six existing HUG grasps. It contains no new generator inference, physics integration, evaluation episodes, failure labels or training data. A numerical review PASS is not scientific mapping acceptance.

- `preflight_review.json` binds the reviewed V1 draft, 172 source/code/start checks, 36 independently authored scalar controls, eight primary solver controls independently rerun, six baseline source/Jacobian gates, and documented method limitations.
- `saved_rows_review.json` records 5,156 provenance/native-label checks on all 178 capped whole-cell rows and 134 retained signed-orientation rows. These are not complete coverage: the original cache contains 44,558 overlapping failing part/band/direction/cell tuples.
- `identity_interval_control.json` and `METHOD_SEMANTICS.md` explain why the frozen independent interval upper bounds can remain unresolved even for identical geometry at a native-ambiguous cell.
- `source_preflight_arithmetic_review.json` records 253 independent arithmetic checks of six recorded original baselines.
- `cohort_8_16_24_output_review.json` and `cohort_12_20_28_output_review.json` independently reconstruct saved native targets, original caps, joint/domain bounds, q-dependent root translation, object/root orientation and source-versus-scene FK, and replay all saved raw/exact constraint values and exact-checkpoint selection.
- `final_output_review.json` binds the merged V1 candidate byte-for-value to those two reviewed cohorts: 16,706 source/trace checks, 159,112 constraint arithmetic checks, and 12 merge identities. All 30 individual original fingertip caps and six original inclusive 10 mm mean gates pass. This does not certify surface preservation; every selected pose still violates retained surface constraints.
- `manufactured_control_correction.md` discloses a reviewer-only expected-value correction at a floating-point addition boundary. No primary code or scientific threshold changed for that correction.

Reports bind absolute source paths and SHA-256 hashes. Full fit traces belong to the separately archived primary method; the reports verify their hashes without duplicating them here. V1 evidence is preserved separately from the later, isolated V2 far-guidance correction.
