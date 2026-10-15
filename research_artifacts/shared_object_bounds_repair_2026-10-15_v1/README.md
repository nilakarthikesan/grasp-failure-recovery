# Shared object-bounds numerical repair — 15 October 2026

The existing object loader rejected apple and salt-shaker inputs when RuntimeWarnings were treated as errors: mixed float32-mesh/float64-rotation matrix multiplication reported divide-by-zero. This report does not establish the low-level backend cause. An explicit float64 `einsum` with `optimize=False` computes the same rigid transform without invoking that matrix-multiply path.

The applied change affects only the arithmetic used for local collision bounds. Source XML/mesh bytes, content hashes, collision selection, object frames, hand geometry, inertia, controller and criteria remain unchanged. Round-off reduction order can differ. This is numerical loader reliability, not grasp/physical acceptance.

Independent root reproduction on the same ten existing object inputs yields **8/10 before and 10/10 after**, with RuntimeWarnings raised as errors. Every passing output is checked against Python-scalar multiplication/addition; maximum final bounds discrepancy is **1.3877787807814457e-17 m**. All ten final geometry-content identities match the frozen input records. The scalar oracle shares compiled mesh vertices and MuJoCo quaternion conversion, so it independently checks arithmetic rather than the entire importer.

Targeted static repository verification: **47 passed, one optional asset test skipped, two free-fall tests excluded**. Integration APIs were guarded to raise; no physics steps, native inference, eval episodes, labels or training ran. The new manufactured test checks transformed compound-object bounds against the scalar oracle under floating-error traps. An initial checker invocation lacked the external Pydantic dependency and produced no case result; adding the existing dependency directories resolved setup before the recorded runs.

`before.json` / `after.json` bind applied source and each existing object identity. `check_objects.py` is the root verifier. `provider_*` retains the original independent proposal/evidence; its original source paths are historical. Before/after source snapshots preserve the exact implementation. Asset geometry itself is external and not redistributed. `test_receipt.json` and `pytest.log` bind current regressions. `paper_update.json` records the guarded common-section update; `paper_compile.json` records canonical compilation. `manifest.json` binds this archive.

Before source SHA-256: `66a39d05d65278c1da886b8816a2fd928b271462dc81454aa87f7a87bfc9594b`.
After source SHA-256: `1013f25080b4affe676376cdb34f251aa1b4e59bdcfa5d6e08f1f1beea70a3b3`.

Provider continuations must bind the new loader when repeating their own static preparation; no old scientific certificate is silently transferred. Existing HUG native/Shadow comparison results and original criteria remain historical and unchanged.
