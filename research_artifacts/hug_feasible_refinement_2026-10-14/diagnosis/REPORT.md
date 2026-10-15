# HUG old constrained-fit diagnosis — 14 October 2026

All six original mapping poses are feasible under their own five tip caps. Every old recorded surface callback was ineligible, so the historical refusal is a search/convergence result. These diagnostics find no inconsistent constraint sign, cached tip vector, wrist/palm position convention or singular local cap Jacobian. They do not show that full surface preservation is possible.

## What was checked

All six immutable V2.1 old raw finals and complete callback traces were replayed through time-zero MuJoCo FK, with the literal q-dependent palm-position compensation. No full surface query, grasp generation, grasp-eval episode, physics integration, label or predictor fit ran. Each original, raw final and FK-restored pose was also compared with the independently compiled fixed Shadow URDF and its world-operator frame chain.

The original 1/2/5 mm surface rule is unchanged. Diagnostic inner optimizations produce static trial joint coordinates solely to investigate feasibility. A projected seed has no measured surface score in this report and is not an accepted grasp mapping.

## Recorded evidence

- Case 8 (hug_apple, scene_002_seed_0): nearest recorded callback still increased its worst tip by 69.8805 micrometres; analytic cheap restoration converged in 9 iterations, with maximum computed tip increase -1.431e-17 m; old-options quadratic control status 0 after 12 iterations, maximum increase 2.051e-11 m.
- Case 12 (hug_apple, scene_003_seed_0): nearest recorded callback still increased its worst tip by 197.152 micrometres; analytic cheap restoration converged in 6 iterations, with maximum computed tip increase -2.927e-17 m; old-options quadratic control status 0 after 4 iterations, maximum increase -1.556e-10 m.
- Case 16 (hug_milk_box, scene_000_seed_0): nearest recorded callback still increased its worst tip by 226.701 micrometres; analytic cheap restoration converged in 13 iterations, with maximum computed tip increase 1.483e-16 m; old-options quadratic control status 9 after 120 iterations, maximum increase 2.275e-09 m.
- Case 20 (hug_milk_box, scene_001_seed_0): nearest recorded callback still increased its worst tip by 0.04941 micrometres; analytic cheap restoration converged in 4 iterations, with maximum computed tip increase -1.388e-17 m; old-options quadratic control status 0 after 4 iterations, maximum increase -1.865e-14 m.
- Case 24 (hug_mug, scene_002_seed_0): nearest recorded callback still increased its worst tip by 59.2628 micrometres; analytic cheap restoration converged in 7 iterations, with maximum computed tip increase 3.08e-15 m; old-options quadratic control status 0 after 23 iterations, maximum increase 2.498e-13 m.
- Case 28 (hug_mug, scene_003_seed_0): nearest recorded callback still increased its worst tip by 4.1249 micrometres; analytic cheap restoration converged in 10 iterations, with maximum computed tip increase 2.255e-17 m; old-options quadratic control status 0 after 7 iterations, maximum increase -4.962e-12 m.

All 649 recorded callback tip-vector arrays replayed bit for bit. Recomputed normalized inequalities differed by at most 5.684e-14, attributable to the equivalent norm/square arithmetic order. Palm POSITION error was at most 8.784e-17 m. The independent fixed-URDF/scene comparison remained below 73 nm across original, raw and restored cases; its separately recorded rotation representation discrepancy remained below 1.01e-7 in matrix components. This confirms the existing convention in the inspected cases, not a native full-surface equivalence claim.

At original q, every individual-cap Jacobian has row rank 5 and every 15-coordinate fingertip Jacobian has rank 15 (absolute rank tolerance 1e-9). The smallest cap singular value ranges from 0.70 to 1.72, versus largest values 21.9–29.3. For every case, a bounded linear program finds a direction that decreases all five errors to first order. All 24 finite checks at step sizes 1e-5, 1e-4, 1e-3 and .01 rad preserve the source/trust joint bounds, all five exact computed caps and strict 10 mm mean guard.

The analytical world-point Jacobian subtracts the palm-position Jacobian to account for carrier translation compensation. It agrees with central differences at 1e-6 rad within 1.64e-8 normalized constraint units/rad. Literal bound-respecting old forward differences at 1e-5 rad differ by .00246–.00400 at q0, about 0.013–0.019% of the largest Jacobian component. This is a measured derivative discrepancy, not a demonstrated sole cause of the old failures.

A cheap SLSQP restoration minimizing squared distance to each immutable old raw final, with analytical constraint derivatives, converged on all six in 4–13 iterations. All six satisfy the old 10 nm arithmetic guard; three satisfy exact computed comparisons and the other three have positive norm differences no larger than 3.1e-15 m. This arithmetic observation motivates a stricter inward restoration target for the next method, not a relaxation of acceptance.

As a separate control, replacing only the surface objective with squared joint-coordinate distance while keeping the old finite-difference epsilon, ftol, 120-iteration budget and constraints yields guard-feasible nonzero restorations for all six. Five optimizer calls converge; first-milk reaches its budget but its largest increase is 2.28 nm, still within the old 10 nm guard. These controls show that the original constraints and finite-difference settings do not intrinsically force all candidates to be refused.

## Interpretation and next method

The old method starts with all five per-tip caps active, optimizes a piecewise surface objective using unconstrained intermediate trial steps, and checks final eligibility after each major callback. SLSQP does not guarantee nonlinear feasibility at every intermediate step. Here none of the recorded cost-improving callbacks or final results met every cap. Source surfaces/nearest parts can switch, and the frozen normal objective retains discontinuous penalties when a site becomes invalid; the presence of those operations follows from source code. A specific discontinuity as the causal event in each historical failure has not been proven by these FK-only diagnostics.

A concrete next correction is to restore each proposed seed using cheap FK and exact caps before evaluating its surface objective, preserve the native-relative direction targets, and reserve fresh final checks. For efficiency, score exactly one restored immutable old raw seed per case before any wider search. If it improves the frozen surface score and exact guards, then run the unchanged independent scientific audit. If it remains inadequate, report the counterexamples; do not tune the rule to pass.

## Provenance and limits

`diagnosis.json` binds 70 relevant source/input/model paths before and after the 5.7-second diagnostic. `controls/diagnosis.json` repeats the bindings and records the 13.8-second control. Guard counters are zero actual integration calls and three explicit self-tests per process. Licensed raw MANO/proposal/scene arrays are not copied into this report. Concurrent manuscript and unrelated provider edits are outside this numerical dependency closure.

The original JSON scope phrase “no solver fits” refers to no new **surface** fit; cheap FK-only optimization is explicitly recorded in its projection fields. This report clarifies that wording without rewriting the measured JSON or prior hashes.

Nothing here certifies full continuous contact geometry, task-success preservation, failure-label equivalence or held-out generalization. No production mapper change is authorized by these controls alone.
