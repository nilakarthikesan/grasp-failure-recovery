# Static per-grasp HUG palm-translation development study, version 1

Executed on 2026-10-15. This artifact contains only static translations/joint fits of the six existing HUG proposals (inventory indices 8, 12, 16, 20, 24, 28). No HUG inference, new native grasp proposal, physics integration, grasp-eval episode, outcome label, predictor training or production mapper change occurred.

## Frozen method

- Protocol SHA-256: `0468b2b63a86832a65b9664ffdbf66daea043fcd5041445fd97fdd812fca291a`.
- Compact selected-pose JSON SHA-256: `7c09699dbaf749a75efdd5f563aa6d18745daddfd18d6ce6ed007c64f7893772`.
- State has 27 coordinates: `(q - original_q)/0.025 rad` for the 24 named joints and world palm displacement `/0.005 m` for XYZ. Exact complete state bytes key every cache and all native-normal evidence.
- Native world targets and the prior authoritative five fingertip caps are unchanged and reproduced exactly before search. Every returned state must satisfy each original cap with **zero extra allowance**, and the unchanged 10 mm mean criterion.
- Joint domain is physical URDF bounds intersect original q +/-0.35 rad. Normalized endpoint rounding can only shrink this domain. Translation is bounded conservatively by the intersection AABB of five source-chain tip reach plus original cap balls. The 1 micrometre outer-domain padding is **not** a fingertip or surface acceptance tolerance.
- The actual saved floating-forearm quaternion stays fixed. Forearm translation compensates changed wrist FK and requested palm translation. Original and selected actual Shadow wrist transforms, palm shifts, and forearm displacement are recorded separately.
- Analytic cheap projection Jacobian is evaluated at actual virtual-tip world points, subtracts the palm-origin Jacobian for the joint coordinates, and has scaled identity translation columns. A bounded central/one-sided difference check passes before each case fit. Baseline original geometry remains bitwise selected when no candidate is admitted.
- Seeds: 18 seeds: distances2.5,5,10mm; XYZ axis; signs+,-; then8 length5mm cube diagonals with XYZ signs+,- lexicographically; all original joints, clamp normalized full state to frozen derived domain
- Per case max128full callbacks incl terminal reserves,6000cheap-tip callbacks incl terminal reserves,6000analyticJac callbacks,256restorations. Pre-solver bounded finite-difference gate and final source/FK verification outside solver counted separately. Each staging call performs exactly2mj_forward; no mj_step.
- Original frozen quadrature, anatomy labels, unsigned distances, closest points, native-facing scores, away targets, missing mass, and invalid-site penalties are inherited without changes. The direction component guard remains an **additional conservative development selector**, not the scientific mapping-acceptance definition.

## Actual result

There were **762 actual full surface callbacks**, **19611 solver cheap-tip callbacks**, and **6739 separately counted solver analytic-Jacobian callbacks**. Pre-solver differential controls and terminal independent FK checks are separately reported in each case. Staging performed 54904 `mj_forward` calls; `mj_step`, `mj_step1` and `mj_step2` were guarded and never called outside the three manufactured guard controls per process.

The 822 fully assessed nonbaseline proposal attempts include cache hits; they must not be confused with actual full callbacks. There were 822 direction-guard-refused attempts. Actual visited pattern radii were [1.0, 0.5]; the configured maximum does not imply all sweeps were completed.

| Existing case index | Actual full callbacks | Largest tip-feasible fully assessed palm displacement (mm) | Selected palm displacement (mm) | New guarded pose selected |
| --- | ---: | ---: | ---: | --- |
| 8 | 127 | 4.458866 | 0.000000 | False |
| 12 | 127 | 4.568963 | 0.000000 | False |
| 16 | 127 | 4.658589 | 0.000000 | False |
| 20 | 127 | 4.891691 | 0.000000 | False |
| 24 | 127 | 4.347863 | 0.000000 | False |
| 28 | 127 | 4.477273 | 0.000000 | False |

All numerical/preflight results are in `candidate_targets.json`, `fit_summary.json`, and `exploration_summary.json`; every refused proposal and reason is retained in the complete cohort files. These outcomes do not demonstrate global infeasibility or convergence. A returned original pose is **not** a geometric PASS. A changed pose refused by the development guard may still be separately audited diagnostically under the unchanged scientific rule, without promoting it or erasing its refusal.

## Evidence layout

- `protocol.json` is frozen before the first actual case measurements. `protocol.draft.json`, manufactured tests/log, and implementation files preserve the reviewed method. The protocol binds the independent review file and protected source/runtime inputs.
- `candidate_targets.json` contains compact selected poses, original caps, exact source checks, differential controls, and source bindings. Full solver proposal/event/reference records are losslessly preserved as `candidate_cohort_*.json.gz`; no raw large cohort duplication is required.
- `full_callback_traces_case_*.tar.gz` contain immutable deterministic gzip callback records. `callback_trace_archive_index.json` maps original paths to archive members and binds both compressed and uncompressed callback SHA-256. Complete callback residuals and five native-world tip-error vectors are retained alongside fixed native/candidate direction scores, weights and validity.
- Full frozen-grid proximity/orientation checks are **not** claimed by this fitting artifact. The independent audit is a separate artifact. All six cases are development cases already inspected; no held-out validation or physical failure-label preservation is established.
- `manifest.json` binds every included file. Large licensed/runtime assets are represented by immutable source hashes and are not copied into this artifact.
