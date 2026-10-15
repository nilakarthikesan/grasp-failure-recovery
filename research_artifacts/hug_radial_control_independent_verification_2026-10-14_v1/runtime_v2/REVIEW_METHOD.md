# Independent radial-shell control review

This is a post-hoc verification of the already frozen and completed radial
control, not a new scientific acceptance protocol or a search for new cells.
It imports no simulator or geometry library and issues no object-to-hand
distance/closest-point queries. Saved arrays and source URDF text are read only.

For every actual recorded box local pose and size, 90-digit Decimal arithmetic
independently reconstructs all eight corners and the anchor-to-box-surface
minimum. Inside boxes use the minimum face clearance; outside boxes use the
Euclidean norm of axis overflows. Thus every point of every literal box surface
has radius between the union's minimum and maximum; convexity bounds the upper
radius for each full surface by its corners. The exact same shell padding and
distance/transform budgets from the original radial control are applied.

Original source URDF RPY rotations are also independently reconstructed through
Decimal Taylor-series sin/cos. Their corners are compared with actual compiled
box poses in a separate representation audit. The first review mistakenly
required 1e-14 m equality between these different representations and stopped
before radial margins. That source/script/protocol/trace is preserved. The
corrected review uses actual compiled box poses for the certificate and a
separate inherited 1.1e-7 m source-representation check. That comparison
allowance is not added to the scientific shell/distance/transform budgets.

The native palm anchor is recomputed directly from original saved camera and
native wrist transforms. The object-frame anchor uses the saved object pose,
with a high-precision normalized unit-quaternion rotation. The original control
used the corresponding serialized floating-point quaternion matrix; tiny
normalization differences remain in the comparison, below its conditional
arithmetic budgets. All input and original result hashes are checked before
and after arithmetic.

For any R in SO(3), all palm points have the form a+Rv, hence preserve ||v||.
For every object cell point x, reverse triangle inequality gives
||x-a|| >= ||centroid-a|| - r_cell. Therefore its distance to ANY rotated palm
surface is at least max(0, ||centroid-a||-r_cell-Rmax). If this exceeds the
frozen band plus the original arithmetic allowances and the native upper cell
distance is within the band, every point in the positive-area triangle is a
lost-proximity witness. The cell radius is essential; a centroid-only test
would not prove the whole-cell claim. The independent fresh geometry audit
already reconstructed the two Mug B cells; its immutable result is hash-bound
and checked here alongside exact saved cache coordinates/areas/native distances.

All seven existing witness cells and all nonpositive results are retained.
The expected scope is one saved Mug B grasp, the same eight literal box
surfaces and fixed original wrist-to-palm anchor, with arbitrary orientation.
It excludes arbitrary anchor translations, different geometries, other HUG
proposals, outcome equivalence or predictor adequacy. These conditional numeric
checks are not formally outward-rounded floating-point proofs.
