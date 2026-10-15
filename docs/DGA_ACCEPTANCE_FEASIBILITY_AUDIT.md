# DGA acceptance feasibility audit

15 October 2026. **DGA remains unaccepted under the unchanged target and criteria. Two required poses have a verified mathematical incompatibility; this is more specific than an unsuccessful optimizer.**

The shared MuJoCo importer fixes remain intact. Independent reconstruction finds no additional rotation, joint-order, unit or source-mesh conversion error. Native and target fingertip meshes genuinely differ: the index/ring source distal mesh has 970 triangles and the target visual mesh has 5,374. Their local extents differ by at least 2.171777 mm in one direction; that is a geometric witness, not a whole-surface Hausdorff estimate.

All 16 saved native outputs remain accounted for. Eleven source-valid outputs received new static surface diagnostics; the five source-invalid outputs remain rejected. The only pose also legal on the target, raw row 6, fails index/ring visual proximity preservation at every 0.5/1/2 mm band without angle correction. At the 1 mm band, conservative lost-area lower bounds are 50.25 and 66.9375 mm² respectively. Its palm comparison passes only for empty nearsets and no occupancy; direction remains unresolved. The other ten raw poses remain target-illegal diagnostics.

Exact rational interval arithmetic now proves unavoidable added palm occupancy for old development row 1 and new native row 0 under their **original pose-specific landmark guards**. The minimum certified changed squares are 0.000104947 and 0.525625 mm² respectively. The proof covers every proper target palm placement and finger configuration satisfying those guards. It removes the previous assumed 1 nm source-transform bridge. Its scope is exact declared geometry and real-arithmetic guards with the recorded constants; arbitrary floating guard implementations, physical outcomes and other poses are not covered.

A separate 2,238-state, three-start correction test retained all tightened landmark and 18 component guards. It selected a 0.00001 rad THJ3 change with unchanged palm and maximum landmark error; the thumb maximum fitting error improved only about 46.4 nm. Fresh independent grids still FAIL strict visual preservation. No acceptance is inferred from fitting loss or the two eligible intermediate proposals.

2,082 exact-math controls, 51 raw-surface controls, nine band-proposal controls and independent certificate, mesh and saved-output reviews support the result. A discovered summary unit-label error was corrected in v2; original reports and measurements are preserved. NumPy matmul warnings remain disclosed, with finite outputs and independent arithmetic checks; no blanket numerical soundness claim is made.

**Decision:** no provider promotion or evaluation start. A mapping covering these required poses cannot satisfy the current exact declared-palm criterion and original landmark guards together. Repeating fitting on those obstructed domains cannot establish acceptance. This does not prove all DGA poses impossible or invalidate a separately scoped executed-Shadow prediction task. No target, criterion or supported input domain was changed.

No physics, new native inference, evaluation episodes/outcome labels, or training ran. All 949 files in the previous DGA component-study manifest rehash unchanged. Original DexGraspNet, native MultiDex and each correction study remain separate.

[Complete evidence and manifests](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_acceptance_feasibility_v2/README.md).
