# Separately frozen new-canonical-anchor radial control

15 October 2026, America/New_York. This is a separate static development
diagnostic after a single source-only neutral anatomical calibration is fixed.
It does not select, tune, repeat or recompute that calibration. It uses the same
seven pre-existing palm witness cells in two saved Mug grasps, the same eight
literal compiled box surfaces, and the exact original radial-shell method and
arithmetic allowances. No new query site, inference, physics or eval data occurs.

The exact calibration JSON, its frozen protocol, the six-case pose/candidate
JSON and its script are bound before this control's first new-anchor margin.
All six source-gate/cap refusal records are retained. This control may evaluate
a refused candidate's declared anchor, because it is a diagnostic over every
possible orientation at that fixed anchor; it neither approves the candidate
nor waives any existing numerical or exact-original-tip requirement.

Let C map neutral Shadow palm coordinates into source MANO wrist coordinates.
For each saved case decode the original T_world_MANO directly from its saved
camera and native wrist transforms. The new anchor is

    a_new = T_world_MANO.translation + T_world_MANO.rotation @ C.translation.

Verify this against the immutable candidate's declared palm position. Object
pose is the same original saved contract. Convert a_new into object coordinates
without moving any object/hand geometry or issuing distance queries. A rigid
orientation of the same palm cannot change any surface point's radius about
that anchor, regardless of free wrist choices or forearm root orientation.

Compute the literal union-of-box-surfaces shell exactly as the old control:
Rmin is the smallest unsigned anchor-to-box-surface distance in palm local
coordinates, and Rmax is the largest norm of ALL eight corners of ALL boxes.
Convexity bounds every full box surface point. Lower Rmin and raise Rmax by
100 nm shell arithmetic padding. The original 100 nm inherited distance and
100 nm radial transform allowances are unchanged and separately retained.

For every existing triangular object cell, let rho be its centroid's distance
to a_new, r its recorded centroid-covering radius and tau its frozen band.
Reverse triangle inequality places every point's radial norm in
[max(0,rho-r),rho+r]. The all-orientation strict lost-proximity margins are

    inner = Rmin - (rho+r) - tau - eps_distance - eps_transform
    outer = (rho-r) - Rmax - tau - eps_distance - eps_transform.

An original native DEFINITE_NEAR positive-area cell with either positive margin
refutes strict same-part preservation at THIS NEW FIXED ANCHOR, for every SO(3)
palm orientation. The radius term makes this a whole-cell statement, not a
centroid-only test. Every nonpositive result is preserved. No positive existing
witness leaves the mapping status UNRESOLVED; it does not establish PASS.

This diagnostic remains conditional on named engineering arithmetic budgets,
not formally outward-rounded floating-point arithmetic. It does not establish
an impossibility under arbitrary translations, changed hand surfaces, other
mapping conventions or all HUG proposals. It does not establish native failure
labels, predictor adequacy, held-out generalization or a production promotion.
The earlier old-anchor certificates and all refused fits remain unchanged.
