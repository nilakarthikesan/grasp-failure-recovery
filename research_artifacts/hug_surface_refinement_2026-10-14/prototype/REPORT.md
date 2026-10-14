# HUG surface mapping correction — static development candidate

14 October 2026. Six existing saved HUG grasps: two observations each of the
apple, milk box and mug, all seed 0. No new generator inference, evaluation
episode, physics integration, outcome label or predictor training occurred.
The original source proposals, original mapped targets, model, manuscript and
protected production files remain unchanged.

This correction is a versioned **development candidate**, not a scientific
mapping signoff. The separate independent audit applies the original frozen
strict proximity/orientation rule without relaxing it. Strict preservation still
fails in the independently inspected domains; the completed sibling report is
the authority for its full nine-resolution results. No default mapper promotion
is justified by this subbundle's engineering diagnostics.

## Implemented correction

The original mapper uses wrist-to-tip/middle landmark vectors. This candidate
adds the object-relative anatomical surface information that those landmarks
do not contain. It solves the actual compiled Shadow collision surfaces through
static FK while retaining the original native proposal as the reference.

For each of six anatomical parts, fixed object-side centroid cells come from
the existing 2 mm area-density visible-object query caches. The optimization
selects 192 deterministic area-quantile bins from the union of native or
original-mapped part proximity within 10 mm, plus 64 bins over the entire object
surface. The near/broad masses are 0.9/0.1. Repeated source indices represent
quadrature mass, not independent samples. Selection uses the same algorithm on
every case; it does not fit the previously reported failed witnesses.

The native anatomical patches remain open, so their distances are unsigned.
The objective matches both the native unsigned distance and the native closest
point offset vector, `closest_hand_point - object_query`, rather than minimizing
penetration or rewarding contact. The vector term records approach side as a
proxy. It is not an exact outward-normal comparison. Poor native relationships
are retained; the target is never an idealized successful grasp.

The frozen V2.1 settings are:

- Distance and closest-point vector residual scale: 2 mm; vector weight 1.0,
  with its three coordinates divided by square root of 3.
- Original task-link displacement anchor: weight 0.25, scale 10 mm.
- Original joint displacement regularizer: weight 0.05, scale 0.35 rad.
- Every joint stays within actual source-URDF physical limits and at most
  0.35 rad from its original target. There is no target clipping.
- One two-joint wrist stage with equal mass for all six parts, followed by
  thumb, index, middle, ring and little finger stages. Part joint sets do not
  overlap. There is no claim of a global optimization optimum.
- SciPy bounded least squares: soft-L1 loss, `f_scale=0.2`, at most 60 function
  evaluations per stage, finite-difference step `1e-4`, and `ftol`, `xtol`,
  `gtol` all `1e-5`.
- No normal reward is optimized. The independent audit measures actual reliable
  exterior normal states, including remaining or introduced reversals.

The wrist stage preserves **native wrist = Shadow palm position**. Wrist
articulation may change palm orientation according to the existing convention
`R_world_palm = R_world_operator @ R_base_palm(q)`. Carrier translation only
compensates the wrist-induced palm offset; it is not fitted to object contacts.
The original carrier rotation remains unchanged. No hand/object scaling,
arbitrary alignment shift or source-pose change is permitted.

## Engineering refusal and guarded derivative

All 36 raw stages reduce their frozen objective and return a successful solver
status, with no budget exhaustion. But **all six raw candidates exceed the
existing 10 mm mean fingertip error guard**, reaching 11.650–13.204 mm.
The raw targets and refusal information are preserved. A smaller surface
objective does not justify violating this existing pipeline requirement.

Before any derivative guard measurements, V3.1 froze a common backtracking
algorithm with `alpha = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0]`. It chooses
the first/largest joint interpolant from original to raw target satisfying both:

1. Actual mean fingertip error is at most the existing 10 mm engineering limit.
2. The **same full-pose** objective is lower than the original by more than
   `1e-12`.

The full-pose objective recomputes identical original/native fields, six equal
part masses, all ten original task-link anchors and all 24 original joint
regularizers for baseline and each interpolant. It never compares incompatible
per-stage costs. If no interpolant qualifies, the exact original q/root is
retained and the candidate refusal is recorded; no case is dropped or given a
grasp-failure label.

Five cases select alpha 0.5; the second milk-box case selects 0.25. All six
guarded targets pass the existing engineering guard, at 8.458–9.290 mm, while
the full geometric objective decreases 10.527–21.471%.

The tradeoff remains: **fingertip and middle landmark errors increase on every
case**. The guarded optimization quadrature's unsigned-distance MAE and
closest-point mean error decrease on every case. This says what the algorithm
changed; it does not establish full surface preservation or failure-label
equivalence. The 2/10 mm objective scales, 0.35 rad trust limit, geometric cost
and 10 mm engineering guard are not scientific mapping acceptance margins.

`summary.json` preserves every case's original/raw/guarded full cost, fingertip
and middle errors, quadrature diagnostics and selected alpha.
`refinement_tradeoff.png` displays the objective/anatomy tradeoff directly.

## Evidence, versions and source preservation

- V1 was an unexecuted draft with fixed wrist/palm orientation. It could not
  address rigid-palm proximity loss. Exact draft bytes remain under
  `v1_unexecuted`; no V1 metrics were computed.
- V2 was frozen but unexecuted before adding reusable API input checks,
  an existing fingertip-guard audit, and the tighter independent FK check.
  Its bytes remain under `v2_before_contract_validation`. Objective settings
  stayed unchanged before the first solve.
- V2.1 ran all six raw candidates: protocol SHA-256
  `8d2fd666757aee85a9f38be5404b3a5418b0521504509b2b6525c9c8648f6dfc`.
- The first V3 launch stopped at a raw-report binding-key typo **before case
  loading or metrics**. Exact script/protocol/log/status remain under
  `v3_guard_binding_abort`. V3.1 corrected that key with unchanged objective,
  alpha list, configuration and source inputs before guard measurements:
  protocol SHA-256
  `59e04a7af29827c8c3b11e2c2278a553e2a0385c3016ed8bb20037b7042a8733`.
- V2.1 raw report SHA-256:
  `fd9b891b888f07b1c1a4dd8a69000463c4e69487c3d301c774fc498f835a3fae`.
  V3.1 guarded report SHA-256:
  `dded9e0a571d921db0be515f5d5b91fdf965fee83e1aa9da442c1ff609898770`.

Raw/guarded reports may be stored as exact-byte gzip files. Decompression must
match these SHA-256 values. Runtime caches, source proposals, compiled scene
binaries and licensed MANO model arrays are excluded from the published bundle.
Derived closest-point witnesses in the raw report are measurements of existing
native proposals, not copied MANO model assets.

The raw solve took 256.033 seconds and the guarded derivative 12.754 seconds.
Each process self-tested its three blocked MuJoCo integration APIs, recorded
zero actual integration attempts, kept time zero and object q/pose fixed, and
checked original input hashes before/after. The 110 protected file hashes
remain unchanged. Independent fixed-URDF versus captured-scene FK error is at
most 70.229 nm for the guarded candidate. Source-URDF physical limits and
captured-scene decimal range differences are reported separately; the latter
are not invented production preflight rejections.

Manufactured moving-plane, deterministic quadrature, malformed-input,
physical/trust domain, largest-alpha selection and exact-fallback tests passed.
The sibling independent validator reconstructs final compiled geometry and
native fingertip errors, checks joint interpolation/placement independently,
and applies the original contact rule. A peer review record binds its exact
script/protocol and distinguishes code review from scientific acceptance.

## Proposed next technical step — not implemented here

A useful next version would solve a joint global surface objective with the
existing fingertip guard enforced as a constraint during optimization, rather
than first producing an unsafe raw target and then backtracking. A predeclared
native-versus-mapped facing-score term could then target the independently
confirmed reversals. It must compare against the native score, including native
nonfacing states, and explicitly handle normal ties, buried features, lost/new
proximity and unavailable comparisons. An eligibility change cannot erase a
required error from the validation claim.

Any such version needs a frozen algorithm and the same independent audit
before adoption. The current bounded solve does not prove that strict mapping
is globally impossible, nor that another set of joint targets can fix every
relationship. Different hand geometry is a constraint to investigate, not a
completed impossibility argument. Physical contact and outcome/failure-label
preservation remain separate, untested claims.
