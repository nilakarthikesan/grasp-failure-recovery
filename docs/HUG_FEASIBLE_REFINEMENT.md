# HUG feasible refinement: completed static development phase

14 October 2026 (America/New_York). This phase reuses the same six saved HUG
development grasps and previously rejected raw fits. It generates static
mapping diagnostics only, with no new inference, evaluation episodes, outcome
labels or predictor training. Historical protocols/results are unchanged.

## Why every earlier fit was refused

All 649 recorded major fingertip vectors replay exactly. The six original
five-cap Jacobians have rank five and admit nonzero inward steps within the
source bounds. There is no demonstrated frame, cache or constraint-sign bug
in these recorded checks. The earlier SLSQP surface search evaluates infeasible
trial poses and only tests final/major eligibility afterward; its first major
steps reach the 0.35-rad correction limit on 11–17 joints. None of its recorded
improving major/raw poses passes all original-relative fingertip constraints.

Cheap FK controls restore all six raw poses to the earlier engineering gate;
five converge with the earlier SLSQP settings and one reaches its iteration
limit with a 2.27-nm residual within the earlier declared 10-nm allowance.
Analytic-Jacobian projection converges in 4–13 iterations on all six. These
controls show feasible movement is possible; they are not surface fits.

A frozen 60-call objective stencil also measures a nearby Mug B direction
eligibility jump at two negative 1e-5-rad probes. Finer probes converge, and
the sampled original forward probes avoid the jump. Thus discontinuity is
documented but is not established as the sole cause of all six refusals.
The supported diagnosis is an unsuccessful coupled local search without
feasible-step restoration, rather than contradictory tip caps or global
infeasibility.

## Implemented repair and measured outcome

The isolated experimental API checks/restores cheap fingertip FK before any
new surface score. It supports finite-budget pattern search; that path has
manufactured tests only. The actual frozen study is seed-only: exactly one
immutable old raw pose per case, no pattern sweeps, unchanged quadrature,
distance/closest-point/facing/anchor/joint cost, and the same source URDF
bounds intersected with original q ±0.35 rad.

Positive original tip caps are tightened by a 1-nm restoration target; final
selection adds no arithmetic allowance. It uses declared sqrt(einsum) norms,
all five original-relative caps, a strict 10-mm five-tip mean, a lower frozen
cost, FK/full callback agreement and fresh terminal callbacks outside cache.
Original poor relationships, negative facing targets, invalid-site penalties
and missing direction mass remain in the objective. No lift-quality filter
is introduced. Callback errors/refusal return the exact original target.

All six new poses are selected and all 30 fingertip caps pass. The stage uses
18 full surface calls in 49.57 seconds, reusing the expensive old raw seeds;
this is not a from-scratch inference/fit speed claim. Two cheap projections
converge; four reach the 40-iteration budget while returning independently
checked feasible improving points. No full-convergence claim is made.

- Case 8 (hug_apple, scene_002_seed_0): cost 7.567033 → 5.735661 (24.20% lower); mean tip error 5.844763 → 5.410839 mm; all five caps pass.
- Case 12 (hug_apple, scene_003_seed_0): cost 8.037699 → 6.699976 (16.64% lower); mean tip error 5.866399 → 5.671855 mm; all five caps pass.
- Case 16 (hug_milk_box, scene_000_seed_0): cost 10.194887 → 9.330676 (8.48% lower); mean tip error 6.646875 → 5.565094 mm; all five caps pass.
- Case 20 (hug_milk_box, scene_001_seed_0): cost 12.449647 → 7.707918 (38.09% lower); mean tip error 8.189192 → 8.102360 mm; all five caps pass.
- Case 24 (hug_mug, scene_002_seed_0): cost 8.241253 → 7.438020 (9.75% lower); mean tip error 6.609030 → 6.530048 mm; all five caps pass.
- Case 28 (hug_mug, scene_003_seed_0): cost 8.410832 → 8.282651 (1.52% lower); mean tip error 6.608610 → 6.608609 mm; all five caps pass.

Independent original/final source-URDF versus captured-scene checks pass on
all six. Maximum selected link discrepancy is 7.122e-8 m, within the inherited
1.1e-7-m source representation check. Native wrist/palm placement, root
convention, physical bounds, object geometry/state and time-zero guards pass.
All 89 relevant protected numerical dependencies, inputs and the predictor
remain byte-identical; evolving paper and unrelated provider code are outside
this computational closure.

## Mapping acceptance remains failed

The independent wrapper preserves the literal previous surface/status functions
and complete query/cell/proximity/orientation measurement block. All six source
preflights finish before the unchanged six coarse and three fine comparisons.
The original 1/2/5-mm bands, numerical assumptions, anatomical labels and normal
eligibility remain unchanged.

The strict result is **0 PASS, 9 FAIL**. Pointwise orientation has
9 FAIL and 0 UNRESOLVED results.
Definite lost/added proximity regions remain. The nine resolutions measure
six grasps repeatedly; they are not nine physical grasp failures. Lower cost
and nonworsening tip error do not certify complete surface/contact preservation.
The caps retain existing absolute errors; some tips still exceed 10 mm, since
10 mm applies to the five-tip mean. Continuous physical contact, native failure
label preservation and held-out object/source generalization remain unconfirmed
or untested. The production mapper is unchanged because this fidelity gate
still fails.

The objective is not a monotone guarantee of the declared diagnostics: milk
carton B's coarse 2-mm summed added-area lower bound changes from 142.31 to
245.00 mm² while its fitting cost falls 38.09%. Mug A's coarse 5-mm check
finds seven reliable sampled sign reversals, versus none previously observed.
The previous absence was UNRESOLVED. Area sums are across anatomical parts,
not unique union/contact areas; lower bounds are not exact areas. Sample
eligibility changes with pose, so raw reversal counts are not comparable
failure rates. These facts refute certification from a lower average fit cost.

Next work is a separately frozen study of the remaining anatomical surface
witnesses with the repaired feasibility mechanism. Do not relax the current
acceptance rule, select only successful native grasps, or use these static
results as prediction-training labels. Neither an impossible global mapping
nor a complete HUG correction has been proved.

## Verification and evidence

- 146 installed new/prior surface tests pass, including 26 new manufactured
  feasible/refusal/curvature/cache/budget/callback cases.
- The independent seed review passes 214 arithmetic/provenance checks.
- The final independent report review passes 832 consistency checks, including
  all nine statuses, per-band summaries, source gates and protected hashes.
- The recorded selection verifier passes six authentic and ten deliberate
  tamper-rejection checks; these do not replace fresh geometry validation.
- Both scientific figures pass visual QA, include all six cases, and disclose
  Mug B's roughly 1-nm tip change and all four inner-budget exhaustions.
- All original scientific bundles are preserved. No licensed hand asset,
  raw proposal/scene/query cache or training recording is copied into this bundle.

[Portable evidence](../research_artifacts/hug_feasible_refinement_2026-10-14/README.md),
[implementation](../src/grasp_failure_prediction/evaluation/feasible_surface_refinement.py),
[tests](../tests/test_feasible_surface_refinement.py), and the existing
[manuscript](paper/working_draft.tex).
