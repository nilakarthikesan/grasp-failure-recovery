# Static HUG reference retargeting comparison — 2026-10-14

This versioned development diagnostic compares the existing production vector
mapping with the **Dex Retargeting 0.5.0 DexPilot-style reference**, using the
identical 32 already saved native HUG proposals. These come from four objects,
eight observation groups and one MANO shape; they are not an independent test set.
No HUG model was loaded, new proposal inference run, time integration attempted,
controller executed, outcome label created, predictor fitted or production mapping
changed. The new files are static audit measurements/configurations, not new eval
episodes or training data.

The reference is the released Shadow/five-finger adaptation using Huber loss,
not the original Allegro implementation or a certified HUG mapping. Its official
Shadow YAML uses scale 1.2 and filtering alpha 0.2. A **separate sensitivity**, with
metric scale 1.0 and no filter, was declared before measurement. Both use the same
20 repeated static solves as the current wrapper, fresh per-case joint-limit
midpoint initialization, all 24 joints including wrists, and no cross-case state.
No source/object scale, point correspondence or hyperparameter was tuned.

The protocol was frozen before measurement, SHA-256
`f0dfd74d18972e98799869d127a7baae8b6bbc2c31c158871ce164547caa05de`.

## Findings

- Current mapper regenerated all 32 saved nominal joint vectors **exactly**.
  Original closing targets precede executor force-closing.
- Independent Pinocchio versus fixed-URDF MuJoCo task-link FK differs by at most
  2.23e-16 m; captured scene versus fixed-URDF task links by at most 1.01e-7 m.
  Existing stored-root and previous static-link outputs reproduce within 26.5 nm.
- Mean fingertip position error is **6.428 mm current**, **21.343 mm released
  scale-1.2 reference**, **2.353 mm metric-static sensitivity**. These are numerical
  geometric residuals, not physical contact or grasp-quality scores.
- Mean middle-link proxy error is **6.829 mm current**, **26.377 mm released
  reference**, **13.811 mm metric-static sensitivity**. Despite better tip points,
  the metric-static sensitivity increases overall ten-link error from **6.628 to
  8.082 mm**. This reveals an objective tradeoff; it does not justify replacing
  the current mapper from tip error alone.
- No projected pinch pairs activate for these saved cases. That is a property of
  this sample; the released objective can project pairs in other configurations.
- Each optimization method makes 640 solver calls, all NLopt return code 3
  (FTOL_REACHED); no optimizer exceptions/fallbacks or rejected target vectors.
  Solver convergence does not establish scientific mapping validity.
- Physical-limit clamping is within 1.000007e-3 rad, the released optimizer's
  documented numerical margin. Clamping changes task-link positions by at most
  **0.130 mm current**, **0.093 mm released reference**, **0.126 mm sensitivity**.
  These are disclosed numerical corrections, not contact-preservation calibration.
- At the 20th repeated solve, the as-shipped filter still differs from raw q by up
  to 0.02304 rad. Raw, filtered and physically clamped endpoints remain distinct.

## Placement policy and limits of the argument

The common policy places Shadow's palm origin at the native MANO wrist and uses
`R_world_palm = R_world_operator @ R_base_palm(q)` separately for each configuration.
The floating carrier pose is derived from that declared palm target and the actual
URDF joint chain. It is not assumed to equal the MANO/operator orientation, and the
absolute palm orientation is not forcibly held constant while wrist angles change.
The source native frames and captured object geometry/state stay fixed.

This controls the existing anchoring convention; it does not establish anatomical
wrist/palm identity. Both reference objectives ignore object surfaces. Middle-link
point proxies are also not complete finger surfaces. A mapping that improves tip
error can change finger-body proximity; the root's same-query visual-surface audit
must be read before interpreting contact fidelity. There is no success filtering,
scientific pass threshold, parameter selection or held-out validation in this report.

## Files and reproduction

`hug_reference_comparison_compact.json` retains all final q/root targets,
per-finger/world/object residuals, provenance, rejection records and solver summaries.
`hug_reference_comparison_full.json.gz` retains the exact complete report, including
all solver inputs/outputs and 20-iteration traces. Decompress with Python's gzip
module; its uncompressed SHA is recorded in the compact report and manifest.
`clipping_check.json` records the numerical-clamp FK effect.
`hug_reference_tradeoff.png` is a static scientific figure, visually inspected.
`summary.json` is a small metric extract. `protocol.json` is the frozen protocol;
`freeze_protocol.py` records how it was declared and is **not rerun** in reproduction.

With the current local saved inputs/assets and installed dependency runtime, run
`sh reproduce.sh`. It copies executable scripts plus the frozen protocol into
`/private/tmp/hug_reference_compare_reproduction_2026-10-14`, then reproduces only
static diagnostics there. Runtime seconds, timestamps and source-script paths are
not intended to be identical; geometry/configuration and bound-input hashes should
match. It requires the declared code/asset/config versions and aborts on hash drift.
The local saved sources are retained outside Git. No licensed raw MANO asset or
checkpoint is copied into this bundle.
