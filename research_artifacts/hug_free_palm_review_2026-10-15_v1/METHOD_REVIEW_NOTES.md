# Independent free-palm review notes — 15 October 2026

Status: preparatory review, no actual candidate fit or new scientific measurement performed.

## Invariants to review before execution

- Six existing native proposals [8,12,16,20,24,28], unchanged native landmark WORLD targets, unchanged original five caps calculated with sqrt(einsum), zero added cap tolerance, strict 10 mm mean.
- Fixed original floating-root quaternion, source Shadow geometry, object transform, all 24 named joints and physical domains. Any old correction-domain change must be explicit before the fit; no changed criterion should be hidden as a runtime fix.
- A 27-vector has three declared-unit translation values and 24 joint angles. Every cache key and detached direction-evidence key includes all 27 float64 values, not just the joints. No component is copied from mutable last_extra after another callback.
- Each pose resets original scene state, installs q and fixed root quaternion, compensates actual q-dependent root-to-palm FK, and translates the palm to old native wrist anchor plus the declared world displacement. A zero-shift, old-q branch must reproduce exact original root bytes. Assert frozen object q, object transform, zero time, finite pose, and attained anchor.
- Translation domain must not be selected from a known failure witness. Conservative source reach bounds derive from all joint-origin translation norms along each palm-to-virtual-tip path, including fixed-tip offset. Any anchor satisfying the exact native-world tip cap is inside each ball centered on the corresponding native tip, radius chain reach plus cap. Their axis-aligned intersection gives a broad outer box; the actual exact five constraints remain authoritative. A box is not a proof of feasible reachable configurations.
- Analytic tip Jacobian at the actual virtual-tip world point uses its distal parent body. Compensated joint columns are J_tip_world - J_palm_world. Three world-translation columns are identity. Norm rows multiply by the unit tip error; at exactly zero error use a declared consistent handling. Compare central differences at synthetic/admissible control states before optimizing actual geometry.
- Native-relative unsigned distance, closest vectors, fixed original normal-site assignments, signed native direction scores and all per-part weights remain fixed. Native-away and penetrating proposals are not converted into success objectives. No dropping invalid sites, no weight redistribution, literal invalid residual 2, zero per-part direction-SSE increase allowance, no original-valid site invalidation.
- Candidate selection requires every original cap, mean, bounds, objective and direction guard; terminal fresh calculation must bypass cache. All refusals retained, original fallback deterministic. Feasibility or local search budget exhaustion is not global impossibility.
- Source closure frozen before actual fitting, including helpers, Python module versions, scene binaries, native records, frame contracts, source URDF/assets, query caches, acceptance rule, all inherited baseline records and process/thread settings.
- mj_step/mj_step1/mj_step2 forbidden. No HUG model inference, new native proposal, simulator rollout, eval dataset, outcome label, training, next-generator acceptance or production promotion.

## Independent output checks

Reconstruct original source/native WORLD task targets from native record and camera frame independently. Rebuild original and returned scenes using named q mapping and saved full states. Recompute exact original caps, actual virtual-tip errors, original-root quaternion identity, q-dependent palm placement and translation bounds. Compare independent source URDF FK to captured-scene result. Verify object unchanged and time zero. Check final 27-vector metadata keys, terminal evidence, guards, scope and no-integration counters. Count all failures/refusals.

Full unchanged surface audit remains a separate computation on selected terminal poses, using all six coarse and three fine existing grids. Source/FK PASS or an improved objective never substitutes for strict surface PASS. Published finite development measurements cannot establish general validity or native outcome equivalence.
