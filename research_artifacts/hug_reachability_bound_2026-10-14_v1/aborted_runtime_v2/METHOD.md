# Conditional current-domain palm reachability bound

This study uses only seven previously published palm counterexample cells in
the two saved Mug development grasps. It creates static diagnostics, no new
grasp proposal, rollout, success/failure label, training data or new query site.

The engineering correction domain is each original joint q0 +/- 0.35 rad,
intersected with source URDF limits. Palm anchor POSITION and root orientation
stay at the original values, with root translation recomputed to keep the palm
anchor fixed. All 22 finger joints are allowed arbitrarily inside their full
engineering intervals; fingertip constraints are ignored, enlarging the
feasible domain. Each actual compiled active hand geom is classified by the
literal frozen anatomical rule. All and only palm-classified geom surfaces
must be attached directly to palm. Its moving ancestors must be exactly the
two declared unit-axis hinge wrist joints. These are checked before evaluation.

For each local palm surface point v, x(q) = anchor + R_palm(q) v. The maximum
radius rho bounds the norm of EVERY point on each convex compiled mesh hull:
the norm of a convex combination is no larger than the maximum vertex norm.
A positive separately declared padding covers the radius computation
conditionally. There is no sampled-radius argument. Primitive palm geoms are
unsupported and cause an assertion, rather than silently losing surfaces.

For two wrist vectors q and c, rotation telescoping and invariance of Euclidean
norm under fixed rotations give

    ||x(q)-x(c)|| <= rho (|q1-c1| + |q2-c2|).

Each elementary hinge term is bounded by
2*rho*|sin(delta/2)| <= rho*|delta|. Interposed constant rotations do not change
this bound. Translations before the palm disappear from x(q) because the
anchor is held fixed. Varying q does not change the palm shape in its own frame.
Thus the Hausdorff distance between the literal unions of palm collider
SURFACES is bounded by that same number; it is not a claim about exposed union
boundaries or physical contacts. Distance to a fixed set is 1-Lipschitz in the
query point, and changes by no more than the Hausdorff displacement when the
set moves. These two facts yield for a wrist rectangle B with midpoint c and
halfwidths h1,h2 and an existing object cell C with centroid z and radius r:

    distance(x, palm(q)) >= distance(z, palm(c))
                           - rho*(h1+h2) - r - epsilon_d - epsilon_transform

for every q in B and x in C, conditional on the named uniform arithmetic
allowances. A native DEFINITE_NEAR cell whose lower mapped bound strictly
exceeds its frozen 1/2/5 mm band is a positive-area lost-proximity witness for
EVERY wrist q in B. Its exact previously recorded cell area and native upper
interval are preserved. The rule uses the worst inherited 100 nm distance
allowance (also stronger than the primary 10 nm case), plus a separately
declared conservative 100 nm transform/radius-comparison arithmetic allowance.
These are conditional engineering budgets, NOT formally rounded interval
arithmetic or a floating-point proof.

Deterministic depth-first subdivision bisects the widest wrist interval,
choosing the first index on ties and visiting the left child first. Each node
selects the existing witness with the greatest positive strict margin; ties
choose first source order. A node with no certified witness subdivides until
the declared node/depth budget. Any remaining boxes retain UNRESOLVED. A full
partition covered by certified leaves would refute strict palm preservation
for this q0 +/- 0.35 engineering domain, even before fingertip constraints.
Partial cover provides no impossibility certificate for uncovered wrist poses.
No conclusion applies to all source-URDF poses, another wrist convention,
other embodiments, all six cases, or all HUG proposals. This does not demonstrate
failure-label changes or predictor adequacy.

Before the first bound metric, protocol.json records exact script/method/input
hashes, all limits, arithmetic assumptions, deterministic controls and finite
budgets. Manufacturing regressions and deterministic FK controls supplement the
derivation; empirical motion tests are not the source of the continuous bound.
