# Independent review of HUG fixed-anchor radial obstruction

14 October 2026, America/New_York. This is a post-hoc independent arithmetic
verification of a separately frozen static control, not new grasp/eval data.

The corrected review passes 687 checks using 90-digit working precision. It
reconstructs the original native wrist-to-palm anchor from saved source camera
and native transforms, reconstructs the saved object transform, and checks
every actual compiled palm box's exact local pose, all eight corners and
anchor-to-surface minimum. It recomputes all seven inherited palm witness
margins, including negative results, and preserves the original shell padding,
distance allowance and transform allowance without alteration.

For saved Mug B, the existing positive-area cell at index 17802 (3.9636778 mm²)
has native upper distance 4.7459127486 mm at the frozen 5 mm band. Its entire
cell lies radially beyond every possible rotation of the same eight-box palm
by a further **1.645927796054253 mm after all declared budgets**. This is a
whole-cell certificate: the centroid-to-cell radius is explicitly subtracted,
and the independent fresh geometry audit's reconstructed radius/native-cell
evidence is hash-bound. It is not a centroid-only measurement. The maximum
original versus independent margin discrepancy is 3.35e-17 m. Mug A has no
positive witness under this radial control and remains undecided by it.

Rotations preserve the radius of every hand surface point about the fixed
anchor. Reverse triangle inequality bounds every point in the object triangle,
so this witness applies to **every SO(3) palm orientation at the fixed native
wrist-to-palm anchor**. It therefore exceeds the earlier physical wrist-range
claim. It excludes arbitrary anchor translations, changed hand geometry,
different mapping conventions, other proposals and conclusions about native
failure labels or predictor adequacy. The certificate remains conditional on
the declared engineering arithmetic budgets; 90-digit recalculation is not a
formally outward-rounded proof of all floating-point operations.

## Preserved review history

The initial review in this root folder independently reconstructed source URDF
RPY rotations but incorrectly demanded 1e-14 m equality with the actual compiled
scene's serialized local box rotations. It stopped before radial-margin
calculations. Its frozen protocol, script, trace and abort record are retained.

The corrected, separately frozen `runtime_v2/` uses the actual compiled box
poses for the certificate. Source URDF corners are compared separately; their
maximum difference from corresponding compiled corners is **8.05 nm**. The
separate source-representation comparison allowance is not added to the
scientific shell/distance/transform budgets and does not drive the obstruction
claim. No original control or bound result was edited or recomputed to obtain
acceptance.

The review imports no simulator/geometry library and issues no new object-to-hand
geometry/closest-point queries. Source assets remain read only and are not
copied. There is no inference, physics integration, new query site, simulation
episode, outcome label, training recording or production edit.

Original control: `hug_reachability_independent_review_2026-10-14_v1/radial_control_v1`
in the corresponding evidence bundle. The original protocol SHA is
`d5f7a4efad4b1b945dd2c95a6eb70fc24fa0345b64046bf9bde74796c381d9ef`;
its report SHA is
`db714454cc350f9c350794d10289706f2d316bd4a014b261938bae9057a3aa08`.
