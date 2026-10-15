# HUG mapping repair: numerical reliability and full palm placement

15 October 2026. Existing saved native grasps only. No new native inference, physics integration, evaluation episodes, outcome labels or predictor training.

## What was wrong

The production bridge solves ten wrist-to-fingertip/middle landmark vectors on Shadow. It receives no object geometry or native hand surface. Correct source decoding, axes, joint names and FK therefore establish correct realization of that declared landmark mapping; they do not guarantee preservation of palm/finger proximity or facing around the object. Our stricter scientific checks require those object-relative relationships. Treating the hand-only fit as sufficient for them was a modeling mismatch, rather than evidence that HUG itself generates invalid grasps.

The previous surface experiments allowed 24 joint coordinates and three palm-position coordinates while holding the floating root orientation fixed. Two wrist joints do not supply an independent three-dimensional root rotation. The fixed-anchor obstruction and failed local fits did not exclude jointly changing anchor and orientation. We have now implemented that missing search capability explicitly.

There were also two numerical defects in the inspected installed dependency/wrapper. Its vector objective adds a temporal term to the supplied gradient without adding that term to the scalar loss. The five-call A–B–A probe also showed that reusing one mapper leaks its previous proposal into the next initialization: identical A differed by 0.192417 rad (11.02 degrees) after B. Existing single-case paths construct fresh instances, so this reuse defect does not explain the six historical single-case surface failures.

## Implemented numerical repair

`ShadowHandRetargeter(..., profile=STATIC_RETARGETING_PROFILE)` explicitly selects `independent_static_shadow_vector_v2`. It resets the solver before each independent proposal, disables the inconsistent temporal gradient term for static inputs, preserves within-proposal refinement, refuses nonpositive or unavailable optimizer status, and records effective settings and all termination codes. Positive codes are not a guarantee of convergence or scientific acceptance. The historical profile remains the default and historical results are unchanged.

Sixteen focused tests pass, including actual dependency loss/gradient agreement, order independence and refusal of a swallowed failure that returns an old finite pose. The corrected saved-input five-call probe has bit-identical A/A/A and B/B results, with all 100 inner solver statuses positive. This verifies numerical reliability, not object-relative geometry. In source-URDF FK, three of five Apple-A fingertip errors and one of five Mug-B errors exceed their original values. Thus the repaired hand-only profile is not promoted as a geometry-accepted replacement.

## Joint palm translation and rotation

The experimental state now contains 24 joint coordinates, three world palm translations and three world rotation-vector coordinates. Rotation is left-composed onto the original root rotation; root translation compensates the joint-dependent palm offset. Zero extra rotation exactly reproduces the old staging path. The original fingertip caps, joint/translation domains and scientific surface predicates remain unchanged. The new rotation search box is +/-15 degrees per vector component, not an accuracy tolerance.

Thirteen manufactured kinematic controls pass. On saved case20, original and pooled zero-rotation full states/body poses match bitwise; the maximum analytic-versus-finite-difference Jacobian discrepancy is 1.59e-12 metres per scaled coordinate. Eight solver controls pass. These checks validate implementation, not native surface preservation.

A separately frozen two-case pilot uses existing milk-box B/index20 and mug B/index28, selected because earlier exposed-boundary checks retained clear finger and palm counterexamples. Each has exactly two starts, at most 1,800 raw callbacks and 60 exact checks, 25 iterations per start and a cooperative 180-second search stop. Selection preserves exact original fingertip caps, direction eligibility and previously passing retained records; a fresh final check is required. A one-micrometre inward fitting target is stricter than the original caps, not a relaxation. Raw search probes cannot become accepted poses.

The pilot completed both starts per case at the 25-iteration limit. Milk-box B used 1,661 raw callbacks and 17 exact checks; mug B used 1,623 and 6. Both returned the exact previously selected pooled pose, with zero additional root rotation: no eligible improvement was found. The selected retained records still fail 12 proximity and 13 orientation checks for milk-box B, and 22 proximity and one orientation check for mug B. These are diagnostic witness counts, not whole-field acceptance or independent grasps. The endpoints reduce parts of the continuous fitting penalty but exceed one or more original fingertip caps, so they cannot be accepted. The study used 6,780 static forward calls including reconstruction and zero integration steps. Failure of this finite local search is not proof of global infeasibility.

A concurrent authorized edit to the shared scene loader was detected before fitting. The pilot loaded the exact previously checked historical loader through a verified import overlay, preserving its original scene contract and leaving the other thread's edit untouched. This is an explicit version boundary; the pilot does not validate the newer loader. Both the stopped source check and the frozen replacement binding are preserved.

## Search constraints must match selection constraints

The completed pilot exposed a second modeling error in the fitting algorithm. Every surface margin entered the optimizer with the same epigraph slack. This allowed the search to trade away previously passing surface relationships and direction eligibility to reduce the worst remaining violation, although exact selection forbids that tradeoff. All 14 protected refusals for milk-box B lose passed proximity/orientation and eligibility; all three for mug B lose passed orientation and eligibility. These are repeated checked checkpoints, not independent samples.

The proposed small fingertip-only restoration was stopped before any execution: repairing tips alone cannot correct that surface tradeoff. Its unexecuted status is preserved. The next separately versioned fit instead gives zero slack to the previously passing pooled proximity/orientation constraints and the necessary eligibility guidance. Failing rows retain the minimax objective. The original inward search buffers are retained for hard rows, and the unchanged exact selector still checks strict boundary semantics and the additional nonsmooth facet, tie, buried-point and visual-membership conditions. The raw guidance does not become a substitute for those checks.

The final protected-row helper passed 16 manufactured tests; eight independent solver controls verified an actual constrained improvement without sacrificing a protected region, slack/Jacobian columns, refusal of invalid exact states and fresh fallback metadata. The first preparation test with removed buffers returned its reference. Its executed source and log are retained, but the temporary endpoint trace was deleted, so rounding is not asserted as the demonstrated cause. Before the real-case protocol was frozen, the original inward buffers were retained. No acceptance threshold was loosened.

The separate protected fit used one pooled start per case, again capped at 25 iterations. Milk-box B used 821 raw callbacks and eight exact checks; mug B used 811 and four. Both retained the old pooled poses. Stored terminal arrays show three fingertip caps and one hard proximity inequality still violated for milk-box B; mug B violates all five fingertip caps while its buffered hard surface inequalities are nonnegative. Both also retain small soft-constraint violations. Neither is a converged constrained solution. The guards correctly refuse them; a stopped finite optimization is not a global feasibility result. The protected fit used 3,372 static forward calls, zero integration and no full-grid rerun.

## Final bounded step check: improvement without sign-off

The final check introduces no optimizer or new seed. For each of the two saved protected-fit endpoints, it tests the fixed sequence `pooled + 2^-k * (endpoint - pooled)`, k=0…20, largest step first. Original fingertip/domain checks precede exact surface checks; only a strict improvement preserving all pooled passing/eligible rows may be selected. One independent fresh evaluation must reproduce the predicates and rank. The protocol is frozen separately; the earlier pilot results are not rewritten.

Both cases select k=4, a one-sixteenth step. Each uses five fingertip checks and five exact checks including the fresh recheck. All ten original fingertip caps pass, with minimum remaining slack 115.88 micrometres for milk-box B and 57.55 micrometres for mug B. No previously passing surface row or eligible direction row is lost.

| Existing grasp | Prior worst normalized violation | New violation | Remaining failed proximity / orientation rows |
|---|---:|---:|---:|
| Milk-box B, index20 | 9.283423 | 9.104737 | 12 / 13 |
| Mug B, index28 | 8.002982 | 7.573550 | 22 / 1 |

These mixed normalized diagnostic values are not universal millimetre errors or probabilities. No failed row crosses into PASS. This is a genuine, bounded improvement in severity while preserving the required safeguards, not a completed surface repair. The step check uses 44 static forward calls, zero integration, and about 7.82 seconds. The independent review recomputes saved selection/protection arithmetic; it is not an additional geometry experiment. No new full-grid acceptance run is claimed, and the nine historical strict failures remain unchanged.

## What can and cannot be signed off

The evidence supports the declared coordinate/FK realization on the saved cases and the repaired numerical behavior of the new explicit static profile. It does not support complete native-relative surface preservation. HUG's native MANO hand and Shadow's collision hand differ in shape and articulation; landmark agreement alone cannot establish matching object relationships. The strict project surface rules are not a published HUG-to-Shadow standard. They remain unchanged here, and neither local optimization failure nor these small improvements prove that every rule is feasible on Shadow.

The remaining engineering task is concrete: incorporate feasible, protected step acceptance into the object-aware fitter, so intermediate fingertip violations and damaged relationships cannot accumulate until an arbitrary iteration limit. The final segment check demonstrates this mechanism on two saved endpoints; it is not yet a general production solver or an independent-object validation. Further progress should preserve the original native sources/caps, keep passing relationships as hard constraints, and record distortion without selecting only successful grasps. A scientifically accepted replacement still needs every required retained relationship and the unchanged independent full-field/uncertainty checks; future untouched objects are needed to support generalization.

The numerical repair is opt-in production code. The surface/rotation/step experiments are saved-input research drivers, not a silently substituted live provider. The historical default is not promoted. Existing runtime and licensed assets are not copied into Git; archived scripts retain executed absolute source paths and need the recorded runtime/path bindings to replay.

Evidence: [versioned repair and bounded studies](../research_artifacts/hug_mapping_process_repair_2026-10-15_v1/). Original criteria: [criteria alignment](HUG_CRITERIA_ALIGNMENT.md). No new inference, eval episodes, outcome labels or training was generated.
