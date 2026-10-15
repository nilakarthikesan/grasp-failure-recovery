# HUG surface objective coverage diagnosis, V1

15 October 2026 UTC / 14 October America/New_York. This is a post-hoc analysis of exactly six existing development fits and all nine existing audit resolutions. No model inference, geometry query, optimizer, simulator step, outcome label or training runs.

## Findings

The saved fits reduce the scalar fitting cost while worsening the fixed direction component on every case. Recovering the exact soft-L1 facing contribution from saved residual arrays confirms this is not an artifact of reporting SSE:

- hug_apple_scene_002_seed_0: facing cost 0.01206557 → 0.34063907; 66 selected direction bins invalid.
- hug_apple_scene_003_seed_0: facing cost 0.02811339 → 0.24744619; 49 selected direction bins invalid.
- hug_milk_box_scene_000_seed_0: facing cost 0.02493118 → 0.52350358; 104 selected direction bins invalid.
- hug_milk_box_scene_001_seed_0: facing cost 0.02914575 → 0.41567399; 76 selected direction bins invalid.
- hug_mug_scene_002_seed_0: facing cost 0.03923449 → 0.38043721; 68 selected direction bins invalid.
- hug_mug_scene_003_seed_0: facing cost 0.04065645 → 0.41717520; 77 selected direction bins invalid.

A lower overall average objective therefore permits direction regressions and invalid fitting sites in exchange for distance/closest-point improvements. The previous feasible solver did what its scalar objective permitted; fingertip feasibility alone cannot prevent this tradeoff.

The 43 unique saved coarse direction witnesses include 20 sites explicitly present in the fixed direction fitting bins, 3 further sites in the original eligible stratum but absent from its quadrature, and 20 sites outside that original stratum. Only 5 appear in the distance fitting bins for their matching anatomical part. These are capped saved witness examples, not all sign-reversal queries or independent grasps. Coverage alone does not explain all failures.

The frozen direction helper admits only the original native-and-baseline shared-valid, same-label exterior stratum, then freezes 64 equal-area bins per available part. New candidate-eligible sites outside this stratum have no direct fixed direction residual. Missing anatomical direction mass is not redistributed. Stored exclusion counts overlap and must not be added.

All three finer repeats are retained. Their query indices belong to a different quadrature: index membership is null. Exact numeric-coordinate equality and nearest selected-site distance are reported separately. Fine saved witnesses have 1/9 exact direction-coordinate matches for apple A, 5/17 for milk A and 0/4 for mug A; none of these coordinate matches certifies the same query normal, weight, cell or eligibility.

All six anatomical parts, all 1/2/5-mm bands and all three inherited numeric allowances appear per domain in analysis.json. Areas are not pooled across overlapping bands or parts. Source conditions and strict statuses are copied from the existing reports: all nine strict audits remain FAIL.

## Supported next change

Introduce a conservative selection safeguard that forbids a candidate from invalidating an originally valid fixed direction bin or increasing any available part's fixed native-relative direction cost, separately from fingertip caps. This blocks a documented scalar tradeoff; it does not repair the original mapping or prove that a suitable improving pose exists. A later version may investigate expanded native-reliable strata and non-averaged per-part/per-band proximity constraints. Neither new fitting nor a change of acceptance is performed here.

No normal lookup bug is demonstrated by this report. A human MANO surface and Shadow collider have different shape and articulation; this diagnosis does not establish global impossibility, physical contact equivalence or predictor adequacy.

## Reproduction and scope

protocol.json binds the analysis script and 18 read-only inputs, including six coarse caches. It was frozen before the expanded scripted analysis, but prior fit/audit results and preliminary counts were already inspected; this is explicitly not preregistered or held-out evidence. Original runtime caches are required for replay and are not copied here.

Run the existing project Python with analyze_saved_coverage.py --root /absolute/project-work/checkout --output /a/new/output.json. The tool refuses changed inputs, changed script and existing output paths. Run test_analyze_saved_coverage.py with unittest for nine manufactured math/identity checks.

Facing soft-L1 costs are reconstructed exactly from saved residuals within ordinary floating arithmetic. Separate distance and closest-point soft-L1 costs are unavailable because only their SSE totals, not every residual, were saved. No such costs are inferred.

The production mapper, acceptance rule, original proposals, predictor and all historical bundles are unchanged. No new eval data was generated.
