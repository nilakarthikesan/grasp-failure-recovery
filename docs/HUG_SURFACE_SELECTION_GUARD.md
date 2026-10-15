# HUG direction-preserving candidate selection

14 October 2026, America/New_York. This completed development step analyzes only the same six saved fits and nine stored audit resolutions. It does not produce a new grasp, geometry fit, simulation episode, outcome label or training example.

## What failed in the previous fitting objective

All six previously selected poses reduce the total soft-L1 cost while increasing the exact fixed-direction soft-L1 contribution. Each also invalidates 49–104 fixed direction bins. The total objective permitted distance/closest-point gains to pay for worse directions; fingertip caps cannot prevent this tradeoff.

The 43 unique saved coarse orientation witnesses contain 20 selected direction-fitting sites, three further originally eligible sites not selected in quadrature, and 20 sites outside the original native-and-baseline eligible stratum. Only five occur among the matching part's distance-fitting sites. These are capped saved witness examples, not all queries or independent grasps. Fine-grid indices are a different index space; coordinate equality and nearest-site distance are reported separately. Coverage gaps therefore matter, but coverage alone does not explain the failures.

## Implemented selector and saved replay

An optional experimental feasible-solver callback now separately refuses a candidate if any available anatomical part's fixed native-relative direction SSE increases or an originally valid fixed direction site becomes invalid. Fixed native targets, including negative away-facing scores, and positive weights are copied; unavailable parts retain missing mass, and nothing is renormalized. Invalid sites retain residual 2. The new selector has zero SSE increase allowance; it does not revise the historical objective or scientific rule.

Ordinary refusals skip the candidate while retaining earlier eligible improvements. Malformed evidence, callback mutation or errors use the safe exact-original fallback. A fresh terminal evaluation repeats the selector. With the optional hook absent, existing behavior and report fields remain unchanged. The production mapper is untouched.

Thirty-six new manufactured controls and all 182 installed new/prior surface API tests pass. A separately frozen replay on saved arrays accepts six baseline identity controls and refuses all six previously selected fits. All 30 saved fingertip caps still pass. Identity acceptance only checks selector consistency; the original poses still fail mapping validation. The replay does not create a replacement pose or fit.

Independent review reproduces both analysis reports byte for byte, verifies every bound input and recomputes every per-part SSE and invalidated-bin list. This is explicitly post-hoc development evidence. It is not held-out validation or a native failure label. All nine original strict mapping audits remain FAIL; no normal-correspondence implementation bug or global impossibility has been demonstrated.

## Next bounded geometry study

Wire the selector into a separately frozen geometry driver, retrieving authoritative direction metadata keyed by the exact joint vector rather than an unkeyed last-callback side channel. Preserve the current fingertip/domain guards and all fixed mass. Test a declared finite budget on the same six saved development grasps, retaining every refusal, then run the unchanged six-coarse/three-fine independent audit for any newly selected poses. No such new fit ran in this step. Expanded native-reliable coverage and per-part/per-band worst-region constraints remain proposed changes, not completed repairs. No acceptance threshold is relaxed and no successful-lift filter is used.

[Coverage and anatomy evidence](../research_artifacts/hug_surface_objective_coverage_2026-10-14_v1/README.md), [saved selector replay](../research_artifacts/hug_surface_objective_coverage_2026-10-14_v1/guard_replay_v1/README.md), and [implementation checkpoint](../research_artifacts/hug_direction_selector_implementation_2026-10-14_v1/).
