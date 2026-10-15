# HUG per-grasp palm placement: static development study

**Result:** per-grasp palm translation has not produced a validated HUG-to-Shadow replacement. Both guarded fitting stages return all six original poses. Independently measured alternatives preserve the original fingertip limits but fail the unchanged surface rule. We completed nine initial comparisons and a separately frozen five-comparison coverage supplement on the same six native grasps. No new native grasp, physical evaluation data or training was generated.

15 October 2026, America/New_York. The user chose to retain the common Shadow Hand and explicitly authorized a per-grasp palm-position study on existing HUG grasps, with the original accuracy criteria and no new evaluation data. This resolves the embodiment question left pending in the preceding canonical-calibration report. The eventual task label concerns the executed retargeted Shadow grasp; static geometry does not establish equality with a native MANO outcome.

## Fixed inputs and new degrees of freedom

The six previously inspected seed-0 development grasps, their original native world targets, Shadow/object geometry, original floating-root quaternion, source-URDF physical bounds and original joint correction limits remain fixed. The new method adds three world-frame palm translation coordinates to the 24 joint coordinates. For each requested state it recomputes root-to-palm kinematics and compensates the forearm translation so that the palm reaches the original native wrist position plus the requested displacement. It retains actual palm, wrist and forearm movements separately. Fixing the floating-root quaternion does not freeze the palm orientation: the existing wrist-joint correction coordinates remain in their unchanged allowed domain.

The search uses normalized units of 0.025 rad per joint coordinate and 5 mm per translation coordinate. These are optimizer scales, not acceptance tolerances. The joints remain in source physical bounds intersected with original q +/-0.35 rad. A conservative translation box encloses the intersection of five balls centered on original native fingertip targets. Each radius is the source palm-to-tip joint-chain reach bound plus that fingertip's original error cap and a 1 micrometer outer-domain rounding pad. The pad expands only the search domain; it never relaxes a fingertip or surface criterion.

Twenty-six predetermined translation seeds comprise positive/negative world-axis movements of 2.5, 5 and 10 mm plus eight 5 mm diagonals. The method follows these with at most four normalized pattern sweeps. Each case has a declared budget of 128 full objective callbacks, 6,000 cheap fingertip callbacks, 6,000 analytic-Jacobian callbacks and 256 restorations. Analytic-Jacobian and preflight/terminal staging counts are reported separately; the cheap-callback limit is not misrepresented as a bound on every kinematic computation.

The analytic derivative is evaluated at each actual virtual fingertip, with the palm translation Jacobian subtracted to account for root compensation. Three world-translation columns are then appended. Manufactured off-origin fingertip controls and bounded finite differences check the implementation before fitting.

## Unchanged gates and independent measurements

Original individual fingertip caps are bound to historical evidence and compared with zero extra allowance; the strict 10 mm mean remains. The frozen native-relative distance, closest-point and signed direction objective retains the same quadrature, anatomical correspondence, native-away targets, invalid-site penalties and missing-mass treatment. The existing per-part direction-error guard remains an additional conservative candidate selector, not a replacement for scientific acceptance. Its refusal cannot establish global infeasibility.

Every callback/cache key includes all 27 state values. Complete full-callback records, selection/refusal traces and fresh terminal checks are retained. A fitted pose that passes these engineering gates still requires the independent unchanged full-grid audit. That audit comprises six coarse comparisons and three finer repeats on the original grids, with the same 1/2/5 mm proximity bands, arithmetic allowances, normal eligibility and PASS/FAIL/UNRESOLVED rules. There are six development grasps, not nine independent grasps.

No HUG model inference, physical stepping, rollout, outcome label, evaluation episode, predictor training, source-geometry change or production promotion is part of this study. All new computations concern static configurations of existing proposals.

## Fine coverage and diagnostic audit selection

After the initial bounded search returned all six original fallbacks and did not reach its finer planned pattern radii, we froze one separate fine-translation study. It starts from the same original poses and caps, uses 56 fixed translation seeds (six axes plus eight normalized cube diagonals at each of 1.25, 0.625, 0.3125 and 0.15625 mm), and performs no subsequent pattern sweeps. This follow-up is chosen after inspecting initial search coverage; it is development work, not independent validation or an extension of the original frozen protocol in place.

The full-grid audit must distinguish guarded mapper outputs from diagnostic alternatives. A separate frozen selection procedure prefers a nonfallback guarded output if available. Otherwise it chooses the lowest comparable fixed-objective, fully scored, nonzero-palm-displacement candidate satisfying all original fingertip, mean and domain gates. Objective comparability, complete 27-coordinate states, deterministic tie-breaking, trace hashes and independent source/FK reconstruction are required. The original fallback and every additional-direction-guard refusal remain explicit. A diagnostic candidate is not promoted to the mapper, even if a subsequent measured property improves. The original scientific criteria alone determine its reported geometric PASS/FAIL/UNRESOLVED status.

## Completed initial search

The first study makes 762 full objective calls across six grasps, including six original and six terminal calls. Its 841 proposal attempts include cache reuse; these are not 841 independent grasps. All six stop under the declared full-callback budget rule, using 127 of the 128 allocated calls per case (one terminal fallback reserve remains unused), after visiting normalized pattern radii 1 and 0.5; the smaller planned pattern radii were not reached. Fully scored feasible states achieve maximum palm displacements of 4.348–4.892 mm across the cases, so the search did exercise the added translation coordinates.

All six guarded results retain the exact original complete pose. The 822 fully assessed proposal attempts are refused by the extra per-part direction selector, including 72 cache reuses. Some candidates reduce total fitting cost but increase a part's direction error or invalidate an originally eligible direction site. This refusal is not itself the independent geometric acceptance result and is not proof that every possible translation fails.

The implementation passes 26 manufactured controls. Independent preflight includes 16 synthetic controls and 147 method/source checks. The final saved-record review passes 52,371 numerical/provenance checks covering all 762 callback records plus independent selected-pose source/FK reconstruction. Those records are not 762 independently rerun surface measurements. Independent fresh full-grid measurements of selected translated diagnostic poses are reported separately below.

## Completed fine study and diagnostic pose selection

The fine study completes all 56 declared seeds in every case, using 348 full callbacks. Each of its 336 nonbaseline assessed proposals is refused by the additional direction guard. All six returned poses are original fallbacks. No further fitting stage is run. Combined, the two studies use 1,110 full objective calls; 1,158 assessed nonbaseline proposal attempts include 72 cached repeats. These are static optimizer computations, not generated native grasps or simulated task attempts.

The independently frozen selection then verifies objective comparability, all complete trace records, original source/native inputs and actual selected scene poses. Its 98,184 checks pass. It selects six diagnostic alternatives, all from the initial study; none is a returned guarded replacement. Each alternative passes all five literal original-relative fingertip caps and the unchanged 10 mm mean gate in the declared computed arithmetic. The closest constraint is about 1 nm inside its original cap because the existing projection targets tighten inward; this is not a physical precision claim or an added acceptance allowance.

The fitting objective decreases in each chosen alternative. Three selected palm shifts are only 1.165–3.284 nm: these are joint-dominated changes, not meaningful tests of a moved palm in the full-grid audit. The other three shift the palm by 3.716–4.892 mm. Larger translations were scored during fitting for every case, but the fixed selection rule picks only one pose per case for the full-grid checks. There is no retrospective minimum-movement threshold or re-ranking after surface results.

| Existing grasp (inventory index) | Selected palm displacement | Mean fingertip error, original → diagnostic | Original individual caps | Extra direction selector |
| --- | ---: | ---: | --- | --- |
| Apple A (8) | 3.143 nm | 5.844763 → 5.829876 mm | 5/5 pass | Refused |
| Apple B (12) | 3.284 nm | 5.866399 → 5.851859 mm | 5/5 pass | Refused |
| Milk box A (16) | 1.165 nm | 6.646875 → 6.645969 mm | 5/5 pass | Refused |
| Milk box B (20) | 4.892 mm | 8.189192 → 7.240657 mm | 5/5 pass | Refused |
| Mug A (24) | 3.716 mm | 6.609030 → 6.193915 mm | 5/5 pass | Refused |
| Mug B (28) | 3.771 mm | 6.608610 → 6.181258 mm | 5/5 pass | Refused |

The original caps are original approximation errors, not externally justified scientific accuracy margins. Preserving them prevents a new regression; it does not establish exact native-hand geometry. The independent full-grid audit reconstructs these diagnostic poses afresh and passes all six source/FK gates, all 30 original caps and all six mean gates before any new grid query. It preserves the original object pose, floating-root quaternion and time zero.

## One explicit palm-movement coverage supplement

A read-only proposal-provenance review then shows that the first three diagnostic selections started as zero-translation thumb-joint proposals (THJ5 +0.025 rad). Their nanometer shifts arise in one restoration iteration that moves constraints 1 nm inward. This satisfies the original selection rule literally, but does not supply the intended full-grid coverage of deliberately moved palms for those three grasps.

We preserve that selection and its nine-grid audit. Before ranking any supplemental candidate, we freeze one additional selection protocol for indices 8, 12 and 16: require one of the previously frozen translation seeds with its exact seed index/state and original joint coordinates BEFORE restoration, retain all original feasibility gates, and use the same objective and deterministic tie rule. This excludes generic joint/tangent proposals; requested seed movements already span 0.15625–10 mm in the two declared studies. No new fit, movement threshold or scientific tolerance is introduced. The same existing coarse/fine grids for those three cases require five fresh comparisons. The other three originally selected candidates already originate in intentional translation proposals; their earlier four comparisons remain separately identified.

This is a post-hoc development coverage correction prompted by proposal provenance. It is not held-out confirmation, a rewrite of the original frozen selection, or a search repeated until PASS. There is no further candidate-selection or fitting loop in this study. The original nine and supplemental five measurements are reported separately.

The supplemental selector passes 44,320 numerical/provenance checks and its 14 manufactured policy controls; a separate replay passes 165 seed/proposal/trace checks. All three selected poses satisfy the original fingertip/mean gates and remain refused by the extra direction selector. Their measured requests and final positions are:

| Grasp | Requested seed displacement | Actual palm displacement after restoration | Mean fingertip error | Original cap excess, maximum |
| --- | ---: | ---: | ---: | ---: |
| Apple A | 0.156250 mm | 0.063565 mm | 5.838904 mm | -1.00000002e-09 m |
| Apple B | 5.000000 mm | 1.994976 mm | 5.708878 mm | -9.99998658e-10 m |
| Milk box A | 0.156250 mm | 0.031372 mm | 6.645773 mm | -9.99999986e-10 m |

The mean errors above are computed final-pose errors; no acceptance allowance is added. The best seed-origin diagnostics have slightly higher fitting costs than the original for Apple A (7.567033 → 7.577273) and Milk box A (10.194887 → 10.201555), and lower cost for Apple B (8.037699 → 7.966316). The frozen diagnostic rule did not require cost improvement; these are measured alternatives, not accepted repairs. The independently reconstructed supplemental poses pass all 15 original caps and all three mean gates before new grid measurements.

## Fresh surface results: initial nine comparisons

All nine new full-grid comparisons FAIL the unchanged proximity and strict relationship criteria. Pointwise orientation separately has seven FAIL and two UNRESOLVED outcomes. These are six existing native grasps with three density repeats, not nine independent grasps or physical failures. Zero complete-pose reports were reused: native query caches remain hash-bound while the target distance/orientation fields are freshly computed. Independent summary/witness arithmetic passes 1,586 checks; 44 saved witness rows across nested bands correspond to 39 distinct case/density/query/direction witnesses. They are capped examples, not the full population of violations.

At the coarse grid and 2 mm proximity band, the mug palm discrepancies persist despite millimeter-scale translation. The following are conditional whole-cell **area intervals** in mm² under the inherited 10 nm arithmetic allowance, not physical contact areas:

| Existing grasp / part / mismatch | Original interval | Diagnostic interval |
| --- | ---: | ---: |
| Mug A / palm / lost | [105.1615, 1155.7344] | [105.1615, 1155.7344] |
| Mug A / thumb / added | [0.0000, 1198.1706] | [4.2279, 1121.6473] |
| Mug B / palm / lost | [53.0901, 827.1070] | [53.0901, 827.1070] |
| Mug B / thumb / added | [0.0000, 1265.5941] | [5.7687, 1221.0864] |

A positive lower bound suffices to refute strict preservation for that pose. A zero lower bound does not establish absence, and changing lower bounds alone does not establish a change in total mismatch area. In particular, the thumb rows certify added-type proximity mismatch in the new poses, without proving that the original poses had none. The full per-part lower/upper bounds, ambiguity areas and orientation eligibility denominators are preserved in the audit summary. Overlapping part areas and resolutions are not summed as physical contact area.

## Fresh surface results: explicit-movement supplement

All five supplemental comparisons FAIL proximity, strict relationships and pointwise orientation. Independent arithmetic passes 1,478 checks; 42 saved witness rows correspond to 36 distinct case/density/query/direction witnesses. These five are fresh measurements of three alternative robot poses derived from existing native grasps. They neither overwrite the initial nine nor create five additional native grasps.

The table below identifies the measurements that actually cover deliberate translation-seed candidates for all six native grasps. It is a cross-reference across the two named audit protocols, not a newly pooled protocol or independent generalization cohort.

| Existing native grasp | Actual palm shift | Measurement source | Existing grid spacing | Strict surface result |
| --- | ---: | --- | --- | --- |
| Apple A | 0.063565 mm | Explicit-seed supplement | 2 mm and 1 mm | FAIL at both |
| Apple B | 1.994976 mm | Explicit-seed supplement | 2 mm | FAIL |
| Milk box A | 0.031372 mm | Explicit-seed supplement | 2 mm and 1 mm | FAIL at both |
| Milk box B | 4.891691 mm | Initial audit | 2 mm | FAIL |
| Mug A | 3.716480 mm | Initial audit | 2 mm and 1 mm | FAIL at both |
| Mug B | 3.770633 mm | Initial audit | 2 mm | FAIL |

All six deliberate-seed alternatives pass their original five fingertip caps and mean gate in independently reconstructed scene arithmetic. Their nine referenced grid comparisons have seven pointwise-orientation FAIL and two UNRESOLVED outcomes. Across the entire study there are **14 fresh full-grid comparisons of nine diagnostic robot poses derived from six existing native grasps**: initial nine plus supplemental five. This is not 14 independent grasps, physical failures or eval episodes.

## Decision, limits and next technical step

Do not promote these alternatives or change the production mapper. The user-selected common Shadow embodiment is retained, and the original scientific criteria remain frozen. The experiment establishes that these tested placements and joint adjustments do not preserve the declared native surface relationships. It does not prove that every possible translation or optimizer must fail, nor that a preserved original fingertip error is an adequate absolute error budget.

This remains development evidence from the previously inspected six cases. The earlier exact reconstruction of 32 native exports does not imply all-32 surface validation. Anatomical part correspondence, literal individual collider surfaces (including buried interfaces), native query coverage and assumed arithmetic allowances remain explicit limitations. These are conditional engineering measurements, not formal outward-rounded or physical-contact-equivalence proofs. Neither native MANO outcome equivalence nor unseen-object generalization follows.

**Proposed, not implemented:** use the already recorded whole-cell proximity and reliable direction counterexamples as explicit fitting constraints, retaining the same fingertip, joint and translation gates. An average objective must not compensate for a remaining violated relationship. Such a new development method needs a separately frozen constraint/selection policy, preservation of all failed constraints, and fresh full-grid validation beyond the fitted cells. Feasibility for the fixed Shadow geometry is unknown. Even eliminating every sampled reversal would leave continuous orientation coverage unresolved under the current rule; full sign-off also requires coverage bounds, not merely zero observed failures. No such new fitting stage or proof is claimed here.

No HUG inference, physics integration, lift rollout, evaluation episode/outcome label, predictor training, GPU provisioning or production promotion occurred in this study. `mj_forward` was used only to reconstruct static poses.

## Evidence and reproduction

All stages retain separate protocols, hashes, source bindings, failures and histories:

- [Initial 27-coordinate fit](../research_artifacts/hug_free_palm_fit_2026-10-15_v1/README.md): complete 762 callback traces, 26 manufactured controls, code, logs and protocol.
- [One fine-seed follow-up](../research_artifacts/hug_free_palm_fit_2026-10-15_v2/README.md): all 56 seeds per grasp, complete 348 callback traces and unchanged criteria.
- [Independent review and first diagnostic selection](../research_artifacts/hug_free_palm_review_2026-10-15_v1/README.md): complete-record arithmetic, source/FK verification and frozen policy.
- [Explicit-seed coverage selection](../research_artifacts/hug_explicit_palm_selection_2026-10-15_v1/README.md): provenance restriction frozen before ranking, 14 controls and independent seed replay.
- [Fresh independent surface audits](../research_artifacts/hug_free_palm_surface_audit_2026-10-15_v1/README.md): initial nine, separately versioned supplemental five, unchanged metric/status blocks, per-part intervals, eligibility counts and arithmetic checks.
- [Frozen scientific acceptance rule](../research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_rule.json), SHA-256 `87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4`.

The archives include reproduction scripts and environment/input hashes; source assets and licensed runtimes remain in their existing preserved locations. No reproducibility claim depends on a new model invocation. A separate metadata addendum repairs an optional display-field lookup in the initial audit summary; pose/callback hashes and scientific outputs are unchanged.
