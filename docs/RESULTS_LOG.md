# HUG results log

Append-only evidence entries. Historical artifacts retain their original
protocols. Implemented checks, measured results and future work are distinct.

## Claim index — 15 October 2026

- Historical development model: AUROC 0.852253 versus commands-only 0.641943;
  260 reserved attempts. Immutable saved weights re-scored without optimization
  in `research_artifacts/hug_retrospective_object_holdout_2026-10-15_v1/historical_readback.json`.
- Historical frozen fresh model: familiar/new geometry AUROC 0.917319/0.558765;
  1,040 attempts per subset. Same read-back; original results unchanged.
- Exposed native boundary diagnosis: 32 saved poses, 65,536 geometric probes,
  1,461 added / 426 lost penetration flags. Final diagnostic; no physical pass.
- Retrospective object holdouts: 3,120 old attempts, four geometries, 12 captures,
  48 proposal families. Full AUROC apple0.823838/milk0.834486/cube0.872925;
  mug undefined. Final development analysis; not prospective validation.
- Recorder: 44 new static tests; 201 total focused checks pass, two optional
  skips/two free-fall deselections, zero integration calls in final check.
  Static implementation final; live pilot pending.

## 15 October 2026 — Phase 1 exposed collision diagnosis

**Source/gate:** protocol/source freeze `5fe44be`; measured outputs preserved in
`6f7f848`. Command and exact runtime are in
`research_artifacts/hug_exposed_boundary_2026-10-15_v1/verification.json`.
`python scripts/audit_hug_exposed_boundary.py run --output NEW_FROZEN_DIRECTORY`
consumes a separately prepared protocol with private saved inputs; never reuse
the populated archive as a new output.

**Counts/results:** 32 attempted/32 reconstructed/32 measured saved poses;
65 static forwards, zero integration/refits/inference. At 2,048 probes/case,
4,762 native-proxy interior probes; 1,461 added and 426 lost penetration flags
relative to resolved skin winding, three unknown skin occupancies. There are
1,931 buried component projections. Maximum sampled penetration bounds are
10.615–12.122 mm. Sixteen manufactured controls pass. See its README/results,
266 bound inputs and 43-file archive manifest.

**Verdict/claim:** diagnostic complete; native collision fidelity and physical
qualification NOT ACCEPTED. Supports exposed-union/occupancy diagnosis on the
original development inputs, not contact forces, injective skin, dynamic
adequacy, new-input acceptance or independent statistics from geometric probes.
The existing 32/32 saved-Shadow realization signoff retains its separate scope.

## 15 October 2026 — Phase 2 recorder foundation

**Source:** `c35b473`. New recording schema `hug_prediction_recording_v1`, opt-in
`RecordedActuatedShadowRunner` and `hug_recorded_prefix_dataset_v1` exporter.
No historical runner/default/schema was replaced. Test files and final command
selection/runtime are recorded in
`research_artifacts/hug_retrospective_object_holdout_2026-10-15_v1/final_static_tests.json`.

**Counts/results:** 44 new tests plus 28 unchanged historical dataset tests pass.
The complete focused increment passes 201, skips two optional-asset tests and
deselects two manufactured free-fall controls with integration APIs blocked.
Mapping refusals and all runtime errors retain null outcomes. Causal predictor
inputs contain only measured qpos/qvel/effective applied targets; diagnostics
are separate. Initial test-log capture failed before test execution due to a
missing isatty adapter and is preserved alongside the passing check. An earlier
broader static verification inadvertently ran two manufactured free-fall
controls (20 integration calls); it produced no episodes or labels and is
separate from all zero-step audit/final-check claims.

**Verdict/claim:** static foundation PASS; real20-attempt recording/tracking/
determinism/export acceptance NOT RUN. Does not claim collection is complete.
The next pilot will use only already-qualified saved Shadow commands, a frozen
controller and declared quota; it cannot qualify fresh solves or native physics.

## 15 October 2026 — Phase 3 retrospective geometry holdouts

**Source:** protocol+implementation freeze `7b41f14`; outputs/read-back `6f7f848`.
Run the exact command in `docs/HUG_RETROSPECTIVE_EVALUATION_V1.md` using a fresh
output. `scripts/verify_historical_prediction.py --output NEW_JSON` independently
re-scores existing saved weights without optimizer calls.

**Counts/results:** 3,120 attempts, 1,103 successes/2,017 failures, four geometries,
12 captures/48 proposal families. Full-model held-geometry results:
apple1040 AUROC0.823838, false warnings30/210, misses351/830;
milk520 AUROC0.834486, false warnings86/245, misses36/275;
cube1040 AUROC0.872925, false warnings471/648, misses15/392;
mug520 all failures, AUROC/false-warning rate null, warnings290/misses230.
All12 model/fold AUROCs and confusion counts were independently recomputed from
CSV. Training-only normalization and validation-only fixed-grid selection pass
review. Per-fold calibration, membership, model selection and prediction hashes
are in `research_artifacts/hug_retrospective_object_holdout_2026-10-15_v1/`.

**Verdict/claim:** retrospective object-holdout analysis COMPLETE with caveats;
phase3 overall PARTIAL (failure-mode risk-set design pending). Supports the
ranking-versus-warning diagnostic on inspected data, not prospective model
improvement. Original frozen scores reproduce. Mug collisions predate the
cutoff. Bootstrap resamples proposals nested in only2/4 captures; the field
`independent_groups` counts proposal families and does not make captures or
objects independent. Corrected-controller data are never pooled with v2.

## 15 October 2026 — Phases 4 and 5 scope

**Models:** extended histories/sequence architectures, an equal-budget new
training study and an independently frozen prospective evaluation are NOT RUN.
The phase3 refits use existing model classes and do not satisfy that future work.

**Manuscript:** the existing `docs/paper/working_draft.tex` is updated to an
anonymous HUG-only workshop paper, desktop compiler success. Prior source bytes
are retained in `docs/paper/archive/working_draft_before_hug_workshop_2026-10-15.tex`.
The verified export `output/pdf/hug_workshop_2026-10-15.pdf` has four body pages
plus one references page, all visually checked. Its source/PDF/style hashes are
in `docs/paper/workshop_verification.json`. Main results preserve original,
corrected-controller and retrospective protocols separately. No submission or
consent action occurred. Final author review remains necessary.


## 15 October 2026 — Phase 1 camera and fresh-instance gates

**Source:** frozen fresh-solve protocol `9d48405`, camera protocol `cd2e9de`;
completed evidence `9546ec4`. Commands are in each archive README.

**Camera:** 32/32 pass unchanged numerical gates; 672/672 saved 2D landmarks
reproduce exactly. All 672 landmarks and 24,896 native mesh vertices have
positive depth and image-window support. Camera-pose maximum discrepancy is
3.331e-16; intrinsics match exactly. Thirty-two static forwards, zero integration,
inference or retargeting. Source-dtype and independent scalar read-back confirm
float32 3D × float64 K yields float64 projection; no projection tolerance repair
was needed. Independent rigid-camera checks also pass. Evidence:
`research_artifacts/hug_camera_projection_2026-10-15_v1/`.

**Fresh solves:** original indices 0/8/16/24, both existing profiles, 38 solves /
760 refinements / 24 static forwards / zero integration. All four fresh-instance
gates pass per profile; legacy fresh commands match saved commands exactly.
Reused legacy fails order independence on all four (maximum 0.204672 rad);
stateless v2 remains bit-exact but changes fitting and stays unpromoted. Maximum
raw-to-legal-command clipping is 0.001000006773 rad, explicitly retained.
Independent source limits, frame/FK/readback and existing mean-tip guard pass.
Evidence: `research_artifacts/hug_fresh_mapping_qualification_2026-10-15_v1/`.

**Verdict/claim:** saved-input camera gate PASS; fresh-instance static behavior
PASS on four development inputs, not independent generalization. Existing
provider construction of a fresh legacy instance per proposal remains the
execution route. These results permit the separately frozen saved-command
recording pilot under the current Shadow controller; they do not promote native
physics or reused legacy state. Twenty-one additional static controls pass
with integration APIs blocked. No outcomes or new evaluation data were produced.
