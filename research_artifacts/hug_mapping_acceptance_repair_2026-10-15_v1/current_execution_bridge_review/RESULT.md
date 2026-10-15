# Static bridge outcome and independent review

All 32 saved legacy Shadow commands pass current nominal hand-command realization and literal authoritative XML joint limits. Historical task-point replay exactly matches the stored report. The primary run used 96 static forward calls, zero physics steps and zero new retargeting or proposals.

The reviewer independently reconstructed source XML FK using SciPy rotations and checked it against saved compiled body frames. Maximum body-position discrepancy was 3.7943e−16m in that independent reconstruction. The current mesh vertex-set comparison maximum was 1.01922e−8m, below its prospective1µm implementation allowance. Mesh geometry itself was not queried again by the reviewer; stored scalar predicates and source-bound implementation were checked.

Historical captured-model geometry does not pass the new source-precision comparison in any of the32 cases. The largest recorded mesh vertex-set discrepancy is6.944585mm at middle/ring/little distal colliders. Other old failures arise at much smaller primitive pose/size discrepancies under the strict numerical checks. This does not mean every failed old component is a millimeter alias error. Current versus historical task-point drift is at most6.33825e−8m, showing why point agreement alone did not establish collider correspondence. Historical results remain unchanged.

The supported acceptance claim is **static nominal hand-command representation for these32 saved legacy commands through the hash-bound current fixed/floating hand loader**. It does not include a freshly assembled current object/controller scene, physical command tracking or contact accuracy, strict MANO/Shadow fidelity, new mapper profiles/SE3 proposals, or unseen-case prediction performance. No new simulation/evaluation data were generated.

`output_review.json` records 23,208 saved-value, source-hash, arithmetic and FK comparisons. These are computational checks, not independent scientific trials. The reviewer made zero new MuJoCo calls.

The run preserved two early runtime preparation failures, both before model construction. The11 manufactured controls also passed in the declared runtime when repeated during execution, not before original approval. Successful-run NumPy matrix warnings are preserved and disclosed in `runtime_history_addendum.json`; their numerical cause was not established. All stored values used by this review were finite, but these checks are not rigorous interval certificates.
