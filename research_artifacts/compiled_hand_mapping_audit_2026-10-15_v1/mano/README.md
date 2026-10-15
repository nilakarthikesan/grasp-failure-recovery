# MANO generator conversion versus compiled target — static audit

15 October 2026. This audit checks **32 original HUG and 13 original ContactGen saved native-to-target mappings**. It makes no scientific acceptance claim and does not replace later experimental mappings. Original ContactGen capsule/101 engineering refusal is retained. All source input hashes were unchanged after the audit. No learned-model inference, MANO forward, retarget optimization, time integration, evaluation episode, outcome label or training ran.

## Result

No new conversion defect was demonstrated in the existing HUG or ContactGen adapters.

- Independent XML kinematics versus fresh fixed-root MuJoCo compilation: maximum position-entry error **1.6653345369377348e-16 m**, rotation-entry error **6.661338147750939e-16** across the saved 45 joint states.
- HUG original compiled-scene task links reproduce the historical saved static FK **exactly**. Independent fresh XML-to-stored-scene task discrepancy is at most **7.844885658603291e-8 m**. This small compiled-scene serialization/import issue is reported to the common-target audit and is not attributed to MANO morphology or repaired by changing source proposals.
- HUG native float32 rotation round-trip error is at most **2.2629733198442636e-8 m**. Original rotation arithmetic is retained; no reorthogonalized source candidate is substituted.
- ContactGen explicit restoration of all 21 native landmarks and shaped-root wrist transform agrees with its adapter. Saved original palm poses reproduce from current adapter logic. Independent XML-to-recorded placed task links agree within **2.220446049250313e-16 m**.
- The official vector optimizer and validator use **fixed-base/operator vectors from palm origin**, not coordinates in the instantaneous palm axes. All 45 original poses have nonzero wrist angles and pass this contract recount. Applying inverse palm rotation only to achieved vectors would introduce a coordinate mismatch.

Remaining native-to-target fingertip errors are millimeter-scale: HUG original mean-tip errors range **5.704–8.305 mm** (worst individual tip **17.518 mm**); ContactGen original means range **3.086–12.397 mm** (worst tip **20.360 mm**, including the refused capsule). These describe original mapped geometry, not simulator conversion error, and do not supersede existing stricter per-case surface failures or later candidate studies.

## Native representation

`native_contract_receipt_review_v1.json` checks native hashes and original reconstruction receipts rather than rerunning either generator. All 32 HUG saved exports have exact original MANO reconstruction of vertices, landmarks, rotation and closed faces. All 13 ContactGen exports have exact float32 reconstruction of 778 vertices, all16 full frames and shaped wrist placement.

HUG uses right MANO, full rotations/no PCA, flat mean, shape10 and wrist center0. Its source translation is already the wrist origin. ContactGen uses right MANO PCA26/nonflat mean and retains the shaped-root offset; its adapter correctly restores processed-mesh vertex mean and uses frame0 rather than the translation parameter as the wrist. Its five fingertip vertex indices and thumb-first 21-joint order match the released source convention. Both outputs are metric. Native hands and target Shadow are different anatomical/geometric models.

## Files

- `inputs_frozen_v1.json`: all byte bindings for current successful audit.
- `audit_saved_mano_conversions.py` and `saved_mano_conversion_audit_v1.json`: per-case independent XML/FK and current adapter checks.
- `target_cases_v1.json`: parent-audit handoff, 45 named24-joint states and object-from-target-palm matrices with exact original native/command references.
- `check_operator_frame_contract.py`, `operator_frame_contract_v1.json`: official config/optimizer source proof and saved-state validation of the coordinate contract.
- `review_native_contract_receipts.py`, `native_contract_receipt_review_v1.json`: source-schema review and byte-bound historical native reconstruction evidence.
- `audit_saved_mano_conversions_attempt1.py`, `inputs_frozen_attempt1.json`, `attempt1_refusal.json`: preserved first-oracle refusal. Independent oracle initially retained raw float32 subtraction while adapter explicitly coerces to float64. The corrected oracle uses the adapter's documented float64 numerical contract; production, acceptance criteria and source geometry were unchanged.

Native licensed geometry remains in its original ignored runtime paths. This scratch output retains scalar reports, robot target poses, references and hashes, not the native MANO vertex/face/shape arrays.

## Subsequent shared-importer finding and review

A later independent check found a **shared MuJoCo URDF collision-mesh import error**, separate from these MANO adapters: middle/ring/little distal links received the index's distinct same-basename mesh. See `historical_claim_scope_addendum_v1.md` before interpreting the initial FK PASS. `collision_mesh_alias_independent_v2.json` and `mesh_alias_witness_recount_v2.json` bind the measured identity failure and independently checked surface-distance witnesses. The initial distance attempt and its discrepancy are retained, with correction scope stated.

The production importer now uses distinct virtual filenames for distinct declared paths; the live-spec scene builder avoids lossy XML serialization while persisting fixed-tip fused inertias. Read-only implementation review and independent3-cycle interleaving/recompilation test in `live_spec_implementation_review_v1.json` found no blocking issue in that scope. All original scene arrays were exactly retained when actuation was added, caller specs remained unchanged, and all fixed-tip masses were preserved. Serialization and physics APIs were blocked. This supports conversion correctness; it does not accept any generator's scientific mapping or assert controller dynamics equivalence.
