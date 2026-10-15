# Independent HUG surface audit for per-grasp palm translation

This is a static mapping diagnostic for the six pre-existing saved HUG grasps (source indices 8, 12, 16, 20, 24, 28). It contains no new HUG inference, simulator integration, rollout, task outcome, failure label, or training example.

Each pose is chosen by a separate pre-grid selection protocol: a nonfallback guarded selection if available, otherwise a fully scored, original-tip-feasible, nonzero translated diagnostic selected by its unchanged objective across the two predeclared fit versions, or an exact original fallback if no eligible translation exists. A diagnostic refused by the optional per-part direction guard stays refused and is not promoted. Its role, guarded mapper fallback status, and direction-guard result are echoed in every audit record.

The audited pose is reconstructed from its joint values and full floating-forearm root position/quaternion. The original root quaternion is required to remain bitwise identical; the new palm anchor equals the original native wrist world position plus the declared per-grasp displacement. Independent fixed-source-URDF forward kinematics is compared with the saved captured-scene representation. The actual translated anchor must agree within 1e-12 m; the inherited source-representation position/rotation consistency bounds are engineering gates, not surface acceptance tolerances.

All five candidate fingertip errors must remain below both the frozen authoritative original caps and fresh independently reconstructed original caps, with zero extra allowance. The inherited strict mean error limit remains 10 mm. All six source/FK/cap preflights must pass before any full surface audit.

The six original coarse 2 mm grids and the three preselected fine 1 mm repeats (indices 8, 16, 24) are freshly measured for the candidate Shadow geometries, including exact original-pose fallbacks. Original hash-bound native query fields are reused because the native proposals and object geometry have not changed; full pose surface results are not reused. The query grid, cell radii, anatomical labels, 1/2/5 mm bands, 1/10/100 nm arithmetic allowances, normal eligibility rules, buried-surface exclusion, and sign deadzone remain unchanged. `wrapper_parity.json` binds the literal unchanged metric/status functions and density loop to the earlier audit source.

The immutable acceptance rule SHA is `87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4`.

Two declared audit cohorts may run concurrently, with two active Numba threads each, after fitting has completed. Every audit checks sources before/after execution, prohibits `mj_step`, `mj_step1`, and `mj_step2`, requires time zero and unchanged object pose, and records the audited full scene qpos. The source candidates, primary protocol, scientific sources, query cache, and wrapper are hash-bound before new geometry calculations.

`summary.json`, when present, is a separate arithmetic read-back: it recalculates proximity and pointwise-orientation statuses from the recorded bound intervals and counts, checks all original cap guards and root/FK evidence, and independently recomputes saved normal-witness dot products. It does not rerun the optimizer or historical candidate callback traces; those are the separate primary/independent-selection-review artifacts.

A FAIL is a mapping diagnostic, not a grasp-failure label. Absence of a sampled normal reversal remains UNRESOLVED for continuous orientation. Finite grid results on six development grasps do not establish unseen-object generalization or physical outcome equivalence. Literal individual collider surface distances may include buried interfaces; the same anatomical-part correspondence remains a stated modeling assumption.
