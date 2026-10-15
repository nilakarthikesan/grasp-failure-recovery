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


## 15 October 2026 — Phase 2 live saved-command recording pilot PASS

**Protocol:** source/plan freeze `11cd155`, inspected authorization `6ccbd11`,
measured evidence `00b6816`. Run command:
`python scripts/run_hug_saved_recording_pilot.py run --output
research_artifacts/hug_saved_recording_pilot_2026-10-15_v1 --authorization
research_artifacts/hug_saved_recording_pilot_2026-10-15_v1/authorization.json`.
Do not rerun into the populated archive; its README gives fresh-output preparation.

**Counts:** exactly 20 planned/20 completed, 20 mapped/20 physically labeled,
zero errors/refusals/NOT_RUN; 65,200 physics steps under the declared hard cap.
Eight successes, seven failed acquisitions and five approach collisions. Four
objects/eight captures/16 unique saved commands plus four repeats. Unique
commands alone yield six successes/ten failures. Cube and milk each succeed
4/5, apple and mug 0/5; every mug fails during approach. No failed proposal was
replaced or repaired. Immediate warning/finiteness/time guards do not trigger.

**Recorder and model round-trip:** all four pairs have exact scored outcomes
and zero maximum difference across recorded full/hand positions, velocities,
requested/applied targets, timing and diagnostics. Pooled joint RMSE is
approach 0.009449, close 0.039463, lift 0.059766 and hold 0.057929 rad, report-only.
The new exporter produces 15 × 1,440 float32 inputs, eight successes/seven
failures. Five sampled mug collisions occur at 0.438–0.998 s and stay in physical
counts while excluded from the 2.96 s forecasting risk set. Both immutable
saved models accept their 288/96-feature summaries; no fitting/calibration runs.
Smoke metrics on these reused inputs are not a new model-performance claim.

**Independent evidence:** `independent_scalar_receipt.json` recomputes all
20 outcomes, X exactly, all repeated endpoint records and saved probabilities
within 1.12e-16. `post_run_verification.json` checks 454 frozen bindings and
80 raw artifact hashes. `historical_outcome_comparison.json` finds zero of
16 original success labels/failure types changed. These finite checks do not
prove outcome/trajectory equivalence in general. Final static regressions:
246 passed/two optional skips/two free-fall exclusions, zero real integration.

**Verdict/limits:** scoped engineering PASS for saved-command execution,
recording, labeling, deterministic endpoint replay and causal model-input
compatibility. No new HUG proposal, native-hand physical qualification,
prospective dataset, physics filter, general approach fix or trained improved
model is claimed. All raw scenes/full states remain ignored/local; portable
prefixes/scalars/manifests are committed separately from historical datasets.
The existing manuscript gains one measured-pilot paragraph and desktop
compilation succeeds; no new separate PDF is exported after that editor update.


## 15 October 2026 — Phase 1 canonical mapping provenance

**Source:** `3181301` (`phase1: preserve raw HUG mapping provenance and refusals`).
The HUG provider now writes `native/mapping_record.json` before accepting or
refusing a solve: all raw refinement commands, actual optimizer statuses,
clamped commands/limits, profile/settings, native hash and alignment metrics.
Missing/failed status refuses new proposals with no physical outcome. Existing
fresh legacy commands and solver settings are unchanged. Nonfinite refused
outputs remain explicit strings in valid JSON; they cannot qualify a pose.

**Command:** `.venv/bin/python scripts/check_hug_build.py --report
research_artifacts/hug_build_phase1_provenance_2026-10-15_v1/static_tests.json`
with the eight test files and selection recorded in that receipt. Use the
existing pinned base runtime plus dependency directories recorded by the script.
**Result:** 77 passed, zero integration calls; six new provenance/refusal tests.
Python3.12.13 / NumPy2.2.6 / MuJoCo3.3.7 / Pydantic2.13.5.

**Verdict:** provenance increment PASS. Does not yet establish new-input mapping
acceptance or complete the consolidated Phase1 gate. No inference, physical
collection, model fitting or manuscript change occurred.


## 15 October 2026 — Phase 1 consolidated mapping gate PASS_WITH_CAVEATS

**Source/protocol freeze:** `7d157dd`. Reproduce with:
`PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python scripts/validate_hug_mapping.py --hug-root /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug --output NEW_REPORT.json --prepare-protocol NEW_PROTOCOL.json saved`, commit that new protocol, then replace `--prepare-protocol` with `--protocol`. The frozen output path must match. Never overwrite this run.

**Results:** 32/32 current static input gates PASS, 96 static forwards and zero
integration attempts. Licensed MANO reconstruction, literal limits, independent
source-XML FK versus compiled hand, actual world-palm placement, unchanged mean
fingertip guard, camera projection and captured camera calibration pass. Scoped
contact distortion, source object/scene realization and dynamic tracking/repeats
are separately bound historical evidence, identified as such per case; they are
not new physical measurements or a new fidelity pass.

**Artifacts:** `research_artifacts/hug_consolidated_mapping_validation_2026-10-15_v1/`
contains the frozen protocol, complete per-case JSON and Markdown report.
Ninety-six focused static tests pass, including19 new validator controls; real
integration APIs were blocked.

**Verdict/claim:** Phase1 saved-input automation PASS_WITH_CAVEATS. Supports
re-runnable static acceptance of these32 commands with documented approximate
MANO-to-Shadow mapping. Does not establish exact contact equivalence or arbitrary
new-input acceptance. Every new proposal must pass the reusable validator before
execution; failed gates retain null physical labels. Phase2 bounded collection
can proceed with the validated Shadow hand, not the static native proxy.


## 15 October 2026 — Phase 2 system foundations

**Source:** `86f2480`. Explicit generic HUG recording mode now retains aligned
velocity/applied/requested commands, null refusal/error outcomes and guarded
artifact publication. `shadow_physics_filter_v1` implements six independent
gravity snapshots, per-window contact support, penetration/displacement codes,
finite quotas and unchanged source state. `hug_grasp_approach_v1` is a distinct
palm-preserving wrist/approach condition with fixed outcome-independent candidate
order and sampled table/proximal-object clearance. The sampler freezes identities,
physical variation and category folds before labels. Historical defaults remain
separate. Numerical mesh support uses an explicit scalar contraction to avoid
observed finite-input BLAS warnings; margins are unchanged.

**Command/result:** `python scripts/check_hug_build.py --report
research_artifacts/hug_build_phase2_foundation_2026-10-15_v1/static_tests.json`
with exact17test files/selection in the receipt:247passed,zero integration calls.
These static/manufactured controls do not establish live filter or approach
acceptance. New fresh collection/generation remains NOT_RUN at this increment.

**Reserve geometry:** `scripts/fetch_hug_object_assets.py --object-path
val/medium_5/OBJECT --output NEW_LOCAL_DIRECTORY`, pinned revision
56ea3f100b1a4ec38bb35e986c0dd14d38977283. Bell pepper:9files/577749bytes;
bear figure:19files/632526bytes. Only geometry files, no grasp labels. Portable
source/attribution/byte-hash manifests are beside the test receipt; raw assets
remain ignored/local. Neither object is capture-qualified or used for fitting.

**Verdict/claim:** Phase2 foundations PASS as software contracts. Unified new
inference/recording pilot, runtime filter results, same-grasp mug comparison and
declared collection remain pending. No system-level completion is inferred from
unit tests. No manuscript files were changed.


## 15 October 2026 — Phase 2 reserve stationary-capture gate

**Source/freeze:** `6c39f66`. Command:
`python scripts/qualify_hug_reserve_assets.py --runtime-project /Users/nilakarthikesan/grasp-failure-prediction run --protocol research_artifacts/hug_reserve_asset_qualification_2026-10-15_v1/protocol.json --freeze-commit 6c39f66 --output NEW_LOCAL_DIRECTORY`.

**Results:** four declared independent captures: bell pepper2/2PASS, bear figure
0/2PASS. Both bear calls exhaust the unchanged stationary-capture criterion and
retain CAPTURE_REJECTED with null task outcomes; no pose or tolerance was repaired.
Total settling budget at most8000steps. Zero grasp inference, task labels or
model fits. Prior static asset/import controls passed for both;82focused static
tests (11reserve tests) pass withzero integration attempts before capture.

**Artifacts:** the reserve qualification v1 archive now includes portable
capture_qualification.json and both bear rejection tracebacks. Raw images/scenes
remain ignored/local. Bell's ray/depth and calibration measurements are in the
report; this is camera/asset qualification only.

**Verdict:** bell PASS_STATIONARY_ONLY; bear stationary branch FAILED and excluded
from future grasp collection. The overall two-object reserve plan is INCOMPLETE.
An additional coffee-cup candidate will use a new predeclared asset protocol,
selected only after this setup failure and before any task outcome or fitting.
Its pinned geometry download is15files/651210bytes. No reserve task success or
predictor quality is established.

The capture emitted finite-input NumPy quaternion-dot RuntimeWarnings, plus the
renderer ARB_clip_control depth warning. The calibrated ray/depth gate still
passed for bell; bear failed stationarity. A subsequent explicit scalar
contraction repair preserves the gates and is tested separately before new
captures; this run is retained without replacement.


## 15 October 2026 — Phase 2 fresh-pipeline source/plan freeze

**Source:** `1115f91`. The driver connects fresh calibrated captures, pretrained
HUG inference, canonical-command mapping validation, nominal close-only filters,
scenario interventions, immediate2ms numerical guards, recorder, causal export
and exact scenario bindings for training. Its complete loaded model state hash
also binds encoder weights absent from the HUG checkpoint.

**Checks:** final321focused controls pass,zero real integration calls, receipt
`research_artifacts/hug_build_phase2_pipeline_2026-10-15_v1/final_static_tests.json`.
The preserved earlier receipt has316pass/1failure from an incorrect manufactured
manifest-key assertion; that test was corrected to the new schema. Independent
review found and fixed missing refusal messages, canonical versus mapping-record
semantic binding, source closure and late result-marker quarantine before freeze.

**Declared live plan:** `research_artifacts/hug_build_collection_2026-10-15_v1/`
contains preparation/source/input/config/asset bindings and generation authorization.
Eight new captures,32fresh proposals(seeds3000..3003),20pilot attempts, then a
separately authorized256attempt study. All filter accepted/rejected proposals
remain in schedules; mapping/setup/runtime errors have null physical outcomes.
Every new native mapping is checked. Stop conditions and exact per-stage step
quotas are in preparation.json. Scenario identities/folds are frozen after
generation and before task labels. No fresh task outcome exists at this freeze.

**Reserve update:** coffee-cup v2freeze `464f08f` passes2/2stationary/calibrated
camera captures with exact source commit recorded in its portable capture report.
Bell+coffee are now two qualified reserve geometries; bear's2failedcaptures stay
excluded without retries. No reserve HUG inference/task labels/model fits.
The unchanged ray/depth gate passes despite the renderer clip-control warning;
the repaired scalar quaternion contraction emits no prior NumPy matmul warning.

**Verdict:** Phase2 unified implementation/static contracts PASS; physical
fresh-pipeline acceptance still pending the declared run. No manuscript edits.


## 15 October 2026 — Phase 2 stopped generation and explicit runtime repair

The committed v1 preparation at e2ee3bd was executed once. It stopped before
native generation: ERROR1/NOT_RUN31, one calibrated capture/500settling steps,
zero native outputs, zero task outcomes, zero fits. The base Hub1.33.0 was
selected beside bundled Transformers4.57.1, whose supported Hub range is
>=0.34,<1. The preserved generation.draft.json records HARD_STOP; this is an
infrastructure error with null physical labels, never a grasp failure.

The repair selects only the declared dependency directories in the child,
prioritizing them after loading base NumPy/Torch. A real import probe resolves
NumPy2.2.6, Torch2.11.0, Transformers4.57.1 and bundled Hub0.36.2. Child logs now
persist with each native record and stale copied mapping/log files are cleared.
No requirement bypass or model/scientific gate changed. The affected provider,
core HUG mapping and collector controls pass110tests,1deselected,zero real
integration calls; receipt hug_build_phase2_runtime_repair_2026-10-15_v2/static_tests.json.
A preliminary test selection hit the static guard in an integration-only test;
its first step was blocked and it is excluded from the static rerun.

A new v2 preparation/run will retain the same declared32generation seeds,
20pilot slots and256study slots. V1 is retained without retries or replacement.


## 15 October 2026 — Phase 2 fresh generation v2 and pre-outcome analysis freeze

Source/protocol frozen at74eaeed. Fresh inference COMPLETED_BOUNDED_GENERATION:
32/32MAPPED across8new captures/four development objects; static native/frame/
calibration/FK/limits/canonical-command guards pass for each. One identical full
loaded model-state hash binds all32. Capture settling totaled4000steps; generation
elapsed153.751s. The renderer clip-control warning persists, but the unchanged
calibrated ray/depth gates pass. No task labels or real-data fits at this point.

The immutable generation.json now freezes exact pilot/study scenarios, category
folds and proposal identities before outcomes. The existing20slot pilot and
256slot study budget is unchanged. Both accepted and rejected physics filters
remain scheduled; numerical filter errors hard-stop instead of becoming labels.

Before task labels, freeze the prepared analysis/model specifications: cause-based
failure phases preserve nulls and reject mixed hand/controller/label-rule cohorts;
23manufactured checks pass. Five comparison kinds share causal close-prefix
positions, velocities and applied targets, fixedthreshold0.5, seeds0/1/2 and a
bounded validation-only tuning budget.25manufactured controls pass, including
tiny synthetic persistence/GRU checks. These are software results, not scientific
model performance. Their real-data execution remains later in Phase3/4. No
manuscript files changed.


## 15 October 2026 — Phase 2 bounded fresh pilot

Frozen source/protocol74eaeed; proposal/scenario freeze0ba5beb. Pilot completed
all20declared slots without integrity/runtime errors:16physical completions,
4approach refusals with null outcomes. Physical labels are8successes,
7failed_acquisition and1drop. Exactly52160task steps; no retries/replacements.
Nominal closed-state filter cohort32:11accepted,16rejected,5unreachable NOT_RUN;
120960close+filter steps, no numerical errors. Filter status never became a task
label and both accepted/rejected schedules were retained.

Strict causal export includes16prefixes(8success/8failure), excludes4refusals,
and binds four geometries/sixcaptures/twelveproposal families. Each X is20
endpoints from2.20..2.96s at25Hz,24joints×(qpos,qvel,appliedtarget),1440float32
features. Raw requested commands/contacts/fullstate/physics properties/provenance
are excluded from X. The complete observables/diagnostics remain local and hashed.

Per-object planned5each: apple2success/3failed_acquisition; milk1success/
3failed_acquisition/1drop; mug1failed_acquisition/4null plannerrefusals; cube5success.
These are pilot engineering measurements, not held-out model performance.
Source/protocol gates unchanged. The predeclared256slot study remains separately
authorized after integrity inspection, with no attempt to repair class balance.
