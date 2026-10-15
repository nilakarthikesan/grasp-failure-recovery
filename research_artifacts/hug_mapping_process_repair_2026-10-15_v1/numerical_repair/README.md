# Retargeter state/gradient repair evidence

`protocol.json`, `run_probe.py`, `results.json` and `run.log` preserve the five-call legacy probe and fixed derivative check. `corrected_profile/` separately preserves the frozen five-call opt-in static-profile check and source-URDF FK comparisons. Neither protocol is merged with earlier surface studies.

`retargeting.py.before` and `test_retargeting.py.before` retain pre-edit source bytes. The original frozen probe's source hash resolves to that preimage after the authorized production edit. `*.proposed` are the applied final versions, with `implementation_change_plan.json` binding both versions. The production default remains `legacy_shadow_vector_v1`; the added opt-in profile is `independent_static_shadow_vector_v2`.

`implementation_tests_final.log` records sixteen passing focused tests. `concurrent_edit_scope.json` explains why this test outcome must not be represented as validation of a later concurrent validator edit. The frozen probes themselves use source-URDF Pinocchio FK and do not import that validator.

`REPORT.md` explains scope and negative geometry evidence. `PAPER_PARAGRAPH_PROPOSAL.md` is a suggested paragraph for the root-owned manuscript; this agent did not edit the paper. Runtime/bytecode caches are excluded from the bundle manifest.
