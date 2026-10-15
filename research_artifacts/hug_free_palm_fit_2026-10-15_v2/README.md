# Static per-grasp HUG palm-translation development study, version 2

Executed on 2026-10-15. This artifact contains only static translations/joint fits of the six existing HUG proposals (inventory indices 8, 12, 16, 20, 24, 28). No HUG inference, new native grasp proposal, physics integration, grasp-eval episode, outcome label, predictor training or production mapper change occurred.


This is a separately frozen exploratory fine-scale follow-up triggered by all six V1 guarded fallbacks. It uses only the four declared previously unvisited translation scales with axis and diagonal seeds, original q0/native caps, and no pattern sweeps. The prior protocol and complete results are hash-bound in `preceding_study`; this extension is run once and stops regardless of result.

## Frozen method

- Protocol SHA-256: `ebba4d832144a667d9320a8c8bfb2c37fae4ee8794eeebad749221e3f6ca5001`.
- Compact selected-pose JSON SHA-256: `fe5a785414cfc14eb07b9c091b99f1d9eedc8226506b7d571c28ed5267594a70`.
- State has 27 coordinates: `(q - original_q)/0.025 rad` for the 24 named joints and world palm displacement `/0.005 m` for XYZ. Exact complete state bytes key every cache and all native-normal evidence.
- Native world targets and the prior authoritative five fingertip caps are unchanged and reproduced exactly before search. Every returned state must satisfy each original cap with **zero extra allowance**, and the unchanged 10 mm mean criterion.
- Joint domain is physical URDF bounds intersect original q +/-0.35 rad. Normalized endpoint rounding can only shrink this domain. Translation is bounded conservatively by the intersection AABB of five source-chain tip reach plus original cap balls. The 1 micrometre outer-domain padding is **not** a fingertip or surface acceptance tolerance.
- The actual saved floating-forearm quaternion stays fixed. Forearm translation compensates changed wrist FK and requested palm translation. Original and selected actual Shadow wrist transforms, palm shifts, and forearm displacement are recorded separately.
- Analytic cheap projection Jacobian is evaluated at actual virtual-tip world points, subtracts the palm-origin Jacobian for the joint coordinates, and has scaled identity translation columns. A bounded central/one-sided difference check passes before each case fit. Baseline original geometry remains bitwise selected when no candidate is admitted.
- Seeds: For each distance1.25,.625,.3125,.15625mm: six XYZaxis +,- then eight normalized cube diagonals signs+,- lexicographic. All original joints and original anchor, clamp to original conservative domain.
- Per case max64fullcallbacks incl terminal reserves,6000cheapFK and6000analyticJac callbacks separately,256restorations. Pre-solver Jacobian gate/final independent FK counted separately.
- Original frozen quadrature, anatomy labels, unsigned distances, closest points, native-facing scores, away targets, missing mass, and invalid-site penalties are inherited without changes. The direction component guard remains an **additional conservative development selector**, not the scientific mapping-acceptance definition.

## Actual result

There were **348 actual full surface callbacks**, **4651 solver cheap-tip callbacks**, and **1923 separately counted solver analytic-Jacobian callbacks**. Pre-solver differential controls and terminal independent FK checks are separately reported in each case. Staging performed 14524 `mj_forward` calls; `mj_step`, `mj_step1` and `mj_step2` were guarded and never called outside the three manufactured guard controls per process.

The 336 fully assessed nonbaseline proposal attempts include cache hits; they must not be confused with actual full callbacks. There were 336 direction-guard-refused attempts. Actual visited pattern radii were []; the configured maximum does not imply all sweeps were completed.

| Existing case index | Actual full callbacks | Largest tip-feasible fully assessed palm displacement (mm) | Selected palm displacement (mm) | New guarded pose selected |
| --- | ---: | ---: | ---: | --- |
| 8 | 58 | 1.250000 | 0.000000 | False |
| 12 | 58 | 1.250000 | 0.000000 | False |
| 16 | 58 | 1.248080 | 0.000000 | False |
| 20 | 58 | 1.229265 | 0.000000 | False |
| 24 | 58 | 1.173297 | 0.000000 | False |
| 28 | 58 | 1.211608 | 0.000000 | False |

All numerical/preflight results are in `candidate_targets.json`, `fit_summary.json`, and `exploration_summary.json`; every refused proposal and reason is retained in the complete cohort files. These outcomes do not demonstrate global infeasibility or convergence. A returned original pose is **not** a geometric PASS. A changed pose refused by the development guard may still be separately audited diagnostically under the unchanged scientific rule, without promoting it or erasing its refusal.

## Evidence layout

- `protocol.json` is frozen before the first actual case measurements. `protocol.draft.json`, manufactured tests/log, and implementation files preserve the reviewed method. The protocol binds the independent review file and protected source/runtime inputs.
- `candidate_targets.json` contains compact selected poses, original caps, exact source checks, differential controls, and source bindings. Full solver proposal/event/reference records are losslessly preserved as `candidate_cohort_*.json.gz`; no raw large cohort duplication is required.
- `full_callback_traces_case_*.tar.gz` contain immutable deterministic gzip callback records. `callback_trace_archive_index.json` maps original paths to archive members and binds both compressed and uncompressed callback SHA-256. Complete callback residuals and five native-world tip-error vectors are retained alongside fixed native/candidate direction scores, weights and validity.
- Full frozen-grid proximity/orientation checks are **not** claimed by this fitting artifact. The independent audit is a separate artifact. All six cases are development cases already inspected; no held-out validation or physical failure-label preservation is established.
- `manifest.json` binds every included file. Large licensed/runtime assets are represented by immutable source hashes and are not copied into this artifact.
