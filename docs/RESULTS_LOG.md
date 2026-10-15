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


## 15 October 2026 — Phase 2 paired fresh mug controller comparison

Frozen protocol507a5db43d88b089c768ee50e52afb5b3f2b1c990f38a04c3e4c45f1163e2f9b
at522083c, all8fresh mug proposals, two captures,16planned condition slots.
COMPLETED_PAIRED_QUOTA, zero runtime errors,35860physics steps,10.382s.
The unchanged vertical_v2 executes8/8:4approach collisions and0successes.
The new grasp_derived_v1 executes3/8:0approach collisions,1success and2other
physical failures;5static plannerrefusals keep null outcomes. Thus new-condition
physical success1/3 differs from planned-population success1/8. Exactly3pairs
complete bothconditions; one changes finalsuccess. Wrist posture and approach
both change, so this is a named-controller comparison, not a direction-only
causal effect. It demonstrates limited reachability and remains separate from
varied-physics training data and historical mug results. No rule/target/gate tuned.


## 15 October 2026 — Phase 2 predeclared 256-slot study

Generation/schedule freeze0ba5beb; pilot-integrity/authorization commit8b3d4d9.
COMPLETED_BOUNDED_QUOTA with zero runtime errors:256planned,213physical
completions,43static approachrefusals with null labels. Physical results:
126successes,84failed_acquisition,3no_contact. Exactly694380task steps,
141.326s. No failed filter proposal was removed and no candidate was replaced.

Strict recorded export has213eligible close-only prefixes:126success/87failure,
four geometry groups, seven included capture groups and28proposal families.
Attempt count is not independent geometry/proposal count. Each feature vector is
1440float32 values; contact/fullstate/object/seed/filter/outcome fields stay out.
All256attempts remain accounted for in the binding and manifest exclusions.

Per-object planned64each: apple36success/28failed_acquisition; milk34success/
23failed_acquisition/3no_contact/4nullrefusals; mug2success/23failed_acquisition/
39nullrefusals; cube54success/10failed_acquisition. Category folds, seeds, controls,
thresholds and model budget stay as frozen before these labels. Four objects
have both physical outcome classes, but only one geometry represents each named
category. Generalization conclusions remain limited to these held objects.

Phase2 collection/recording and causal export PASS integrity. The new approach
still has reachability refusals; it is not universal mug acquisition. Six-axis
filter screening is a separate nominal engineering diagnostic, not a success
certificate. Phase3 independent audit/category analysis and Phase4 fitting follow.


## 15 October 2026 — Phase 3 frozen category accounting and Phase 4 declaration

Analyzer source0dfa869 and31manufactured controls(0integration calls) preceded
the actual read-only audit. study_analysis.json is COMPLETED_READ_ONLY_ANALYSIS:
all256raw records,213physical/exported labels and43null outcomes bind exact
frozen scenarios/source/assets; all category/geometry/capture/proposal splits
pass the strict training adapter. No schedule/fold/label was changed. All4folds
have two-class training, validation and test partitions and are eligible.
The measured study has3pre_contact(no_contact),84acquisition failures and0hold
failures. The pilot drop is separate and is not pooled into this study. Thus
this dataset cannot support a claim of learned slip-onset/hold-failure prediction.

The structured Phase4 protocol now binds exact213×1440features/labels, immutable
folds/binding/manifest, all predictor dependency sources and runtime versions.
Five models×3seeds×4folds=60slots; at most4candidates each, fixed0.5warning
threshold, validation logloss selection only. GRU50epochs and logistic400iteration
caps remain as committed before task labels. No real fitting has occurred at
this declaration. Learned models must be frozen before reserve HUG/task outcomes.


## 15 October 2026 — Phase 4 real fixed model comparison

Frozen fitting declaration/HEAD064bfb3, protocolSHA4e6668494b0d4bd51e991ad6c7a5b50cf997c369bf1c1aa90148455b3fff1c2d.
All60model/fold/seed slots completed:5kinds×4heldcategories×3trainingseeds,
zero skips/errors,7.564s,zero integration attempts. Model/readback artifacts,
train-only normalizers, validation-only candidate selections and held predictions
are saved and hash-bound. Start/endGitcommit agree; no tuning/split/threshold
changes were made. Logistic/prevalence seed repetitions are deterministic, not
independent experiments.

Unweighted mean of the fourfold×three-seed failureAUROCs (descriptive; no
confidence interval or test-driven model selection): training_prevalence0.500000,
commands_only_logistic0.401080, summary_logistic0.593973,
full_history_logistic0.492607, small_gru0.399216. Corresponding logloss means:
0.855108,1.112681,1.369674,1.506852,0.873559. Summary-logistic fixedthreshold0.5
has mean failurerecall0.376702 and successfalsewarningrate0.327614. Its AUROC by
heldobject: apple0.745040,cube0.811111,milk0.254525,mug0.565217. These uneven
results do not establish reliable category transfer or calibrated deployment.

No test result selects the primary reserve model or threshold: summary_logistic
and0.5were fixed before reserve outcomes. All60completed bundles will be frozen
and evaluated as declared; no reserve-based refit/selection is allowed. The
learned target remains final task outcome from beforelift, not slip onset.


## 15 October 2026 — Conservative planner translation sensitivity diagnostic

Read-only static diagnostics under the frozen v2 engine identify all4milk planner
refusals as proposal_0023 at tableoffset0.02m. The same proposal is admitted at
zerooffset. All active hand/proximal/object geom positions translate together by
[0,0,0.02] within6.59e-17m, rotations/meshbytes/targets/direction are identical,
and tableclearance stays0.0864447551220233m. At pathalpha0.75 the forearm mesh
versus object_collision_001 distance changes0.18110005856385916m→0m. Other
pairs differ by at most1e-10m. This is measured static distance-query sensitivity,
not a missing palm/object/table shift. Zero integration calls;210staticforwards
in the fullgeometry diagnostic. Portable scripts/receipts bind the ignored full
query/geometry arrays. No source/protocol/data/refusal/label was changed.

The frozen controller conservatively refuses these cases; physical task labels
remain valid for the executed population. Table-offset nuisance invariance and
global physical infeasibility cannot be claimed. A future planner protocol must
address robust distance handling; the current result is retained as a limitation,
not silently repaired after comparison. Existing collected/model artifacts stand
under their exact declared controller condition.


## 15 October 2026 — Phase 4 prospective reserve declaration

All60learned models were committed at80bfb60 before reserve HUG/grasp outcomes.
The new reserve protocol binds every completed bundle, exact original fitted
source closure/Gitblobs at064bfb3, matching hand/controller/scorerlabel cohort,
input allowlist and fixed0.5threshold. summary_logistic stays primary; all other
bundles are diagnostic, never selected by reserve metrics.26manufactured tests
pass withzero integrationcalls, including source/cohort tampering and bootstrap.

Frozen assets: bellpepper2qualifiedcaptures+coffeecup2qualifiedcaptures; bear
remains excluded. NewHUGseeds9000/9001 on eachcapture give8proposals;2mass×2friction
conditions each give32task slots, maximum104320steps. No reserve task label,
model fit or reserve predictor score exists at this declaration. Stage generation
and task collection require separate exacthash/quotaaudit receipts and commits.

Remaining brief audits (historical score/source readback, failure-mode holdouts,
family-cluster uncertainty and earlier cutoffs) are being closed separately.
Historical source-bound LOO work already exists and is being independently
checked; its original chronology/results are preserved. These supplementary
analyses cannot change the frozen prospective models/primary/threshold.


## 15 October 2026 — Phase 4 reserve collection: no eligible physical prefixes

Prospective protocol/source freeze3717970; actual8/8reserve native grasps map
and generated32scenario identities were committed atc017250 before tasks.
All32bell/coffee slots receive conservative static approachrefusals, with null
physical outcomes; zero task physicssteps and zero retries/replacements.
Collection is COMPLETED_BOUNDED_QUOTA as ledger accounting, not physical success.
The reserve dataset status is NO_ELIGIBLE_PREFIXES. No physical success/failure
rate or predictor AUROC can be estimated for these unused objects from this run.

The fixed evaluator will be invoked once after this ledger commit to preserve
an explicit undefined result for every frozen model, rather than changing poses,
clearance thresholds, models or reserve objects after seeing this censoring.
This is an important prospective coverage failure and limits Phase4 acceptance.


## 15 October 2026 — Reserve evaluation and beginning-of-system evidence review

The one permitted prospective evaluator invocation yieldsNO_ELIGIBLE_PREFIXES:
all60frozen bundles explicitlySKIPPED, zero model-prediction calls, no fabricated
metrics. Eight fresh reservegrasps pass numerical mapping;32approachrefusals
prevent allphysicaldata. Phase4prospective generalization remainsINCOMPLETE.

The user asks whether mapping and simulator quality are proved. Independent
source/evidence review confirms32saved+40fresh scoped numerical passes, but
strictnative/Shadow surface acceptance remains failed in6detailedcases+3finer
repeats. Current-geometry replay retains178proximity-change witnesses and123
reliable directionreversals under that separate rule. New numerical passes do
not erase those failures or establish native physical label equivalence.

The source/compiled-hand bridge verifies geometry/FK/inertias, and213study tasks
pass finite/warning/time guards. An idealarm carrier, engineeringservo/contact
parameters, absent timestepconvergence/hardware calibration, and the measured
translation-sensitive query limit physicalfidelity claims. These distinctions
and each construction decision's reason/evidence/limit are now explained in
SYSTEM_BUILD_STATUS.md. No studyhand, labelrule, threshold, oldresult or paper
file changed in this review.


## 15 October 2026 — Supplementary historical readback and fixed cutoff replay

The later read-only historical audit performs zero fits/physics and reproduces
all four original saved-weight AUROCs: 0.852253328781154 / 0.6419426425401161,
and familiar/new 0.917318763408498 / 0.558765241945835. It verifies all 12 archived
LOO metric/calibration cells and group memberships over 3,120 historical attempts
(1,103 success, 2,017 failure; 4 geometries, 12 captures, 48 families). Archived
LOO coefficients were not saved, so that part is prediction/metric replay, not
coefficient readback. Old and new controller datasets remain separate.
Reproduction: original .venv Python, scripts/audit_hug_historical_prediction.py
--project CANONICAL_CHECKOUT audit --output NEW_JSON.

The four-cutoff declaration at 8d611c0 executes once without fits or physics:
2.96, 2.76, 2.56, 2.36 s; 213 eligible prefixes at each, 43 null exclusions.
All 60 frozen bundles score only their original held-test IDs. Baseline arrays,
labels, groups and probabilities reproduce the original comparison; all earlier
contracts are explicitly shifted-input diagnostics. Unweighted summary-logistic
mean failure AUROC is 0.593973 at 2.96 and 0.500000 at each earlier cutoff.
Fixed-threshold mean failure recall/false-warning rate changes 0.376702/0.327614
to 0.25/0.25. This does not establish useful early warning or slip onset. No
cutoff is selected using these results. Reproduction: scripts/audit_hug_cutoff_sensitivity.py
prepare with the documented original bindings, freeze, commit, then run --output
FROZEN_DECLARATION_DIRECTORY; new directories are required for a replay.

A final separate summary-only failure-family audit declaration now freezes all
six physical-cause folds, deterministic family validation, four L2 candidates,
fixed threshold 0.5 and 1,000 whole-family bootstrap draws per existing category/
model/seed. It does not change the original five-model comparison or prospective
models. Missing modes and single-class roles will skip. No real fitting or
bootstrap at this declaration; execution follows its source/protocol commit.


## 15 October 2026 — Final bounded audit results and explicit gate verdicts

Failure-family audit source/protocol freeze 29aa529 completed in 5.469 s with
four declared summary-logistic candidate fits and zero integration attempts.
Only the no-contact fold is eligible: 8 held-family variants, 5 successes and
3 failures. Failure AUROC/AP are 1.0, but the fixed 0.5 threshold warns on none:
failure recall 0/3 and successful-task false warnings 0/5. This tiny retrospective
risk set does not establish prospective unseen-mode detection. Acquisition skips
for single-class training; approach collision, slip, drop and timeout skip for
absence. No synthetic failures, balance repair or replacement fit is introduced.

All 60 original category/model/seed prediction cells receive the predeclared
1,000-draw whole-proposal-family bootstrap, without refitting or predicting again.
Intervals are conditional on the observed families within each held object.
They do not cover category/capture/controller/hardware uncertainty; repeated
training seeds are not independent datasets. Valid/undefined draw counts are
retained. A legacy field named independent_groups means family-cluster count,
not proof that families sharing captures are statistically independent.

Current build gate verdicts (distinct claims, no protocol merging):

- Phase 1 numerical conversion: PASS_WITH_CAVEATS, 32 saved and 40 fresh inputs.
  Saved mean/worst fingertip errors are 6.428/17.518 mm. The separate strict
  native/Shadow surface claim fails six detailed cases plus three finer repeats;
  native physical contact/outcome equivalence remains unverified.
- Phase 2 implementation/recording: PASS in the bounded cohort. Core study has
  256 planned / 213 physical / 43 null refusals; filters 11 accepted / 16 rejected /
  5 unreachable. The 2,000-attempt/six-physical-object collection target is NOT MET.
- Phase 3 auditing: four category folds and original historical score readbacks
  verified. Failure-mode coverage is PARTIAL: one eligible / five skipped modes;
  fixed family bootstrap is measured, with limited independent situation count.
- Phase 4 model comparison: all 60 slots complete without errors; summary-model
  mean held-object AUROC 0.594 and earlier-cutoff means 0.500. Prospective reserve
  generalization is INCOMPLETE: 32 planner refusals / 0 physical prefixes /
  60 explicit model skips. No real-robot, slip-onset or recovery claim is supported.

The user's latest focus is the beginning of the system. Decision/reason/evidence/
limitation is documented in SYSTEM_BUILD_STATUS.md, including the approximate
wrist-to-palm correspondence, ideal arm, uncalibrated physical parameters and
translation-sensitive query. No further scaling or model claim resolves those
foundation gaps. Manuscript files remain untouched.

Reproduction uses the committed protocol helpers and documented fresh-directory
commands in HUG_FAILURE_MODE_AUDIT_V1.md and HUG_CUTOFF_SENSITIVITY_V1.md.
Every input/source/fold/model binding must match before execution. The optional
historical-refit v2 declaration is preparation only; no new historical optimizer
run was needed or performed. Original historical artifacts remain unchanged.


## 15 October 2026 — Versioned solver repair and native-fidelity admission

The new mano_right_to_shadow_hand_right_static_v2 strategy selects the existing
independent-static profile: temporal-gradient inconsistency removed, state reset
per proposal, smoothing disabled and vector scale1.0. Legacy mapping identity
remains unchanged. Fresh translation calls now reject failed, missing or unknown
optimizer statuses; positive codes1–6 include budget stops and do not prove
convergence or contact preservation. No native-surface positive result is claimed.

New fidelity admission separates numerical realization from the unchanged native
surface rule. No real MANO-to-Shadow continuous positive verifier is implemented;
missing, sampled, capped-witness and self-declared PASS evidence cannot admit
collection. The bounded collector, filters/direct attempt entry, opt-in recorder,
default HUG proposal execution, legacy HUG case runner and saved pilot refuse
before physics when unqualified. Static diagnostics remain usable. Raw runner
and standalone approach-study APIs are not a universal physics firewall; none
are being executed. Refusals do not become negative task labels.

Final combined controls:226passed in5.70s, zero integration calls, in Python3.12.13,
NumPy2.2.6, MuJoCo3.3.7 and Pydantic2.13.5. Receipt:
research_artifacts/hug_mapping_admission_2026-10-15_v1/final_combined_checks.json.
An initial combined run had189passes/8failures because the old manufactured
saved-pilot tests expected post-admission execution. Their explicit admission
mocks preserve transaction tests; separate unmocked tests verify actual refusal,
including persisted not_executed/null-outcome metadata. Both receipts are kept.

A separate current-geometry full-field diagnostic is frozen before measurement:
cases20/28, four saved poses,19174+45484cells,341721+804343native-definite
relationships,600s and48staticforwards per case. It rechecks complete saved
fields, including baseline-passing regions, under unchanged bands/caps and0
physics/retarget/inference budget. Orientation full coverage remainsNOT_RUN,
so this audit cannot certifyPASS. A qualified current-scene30Dadapter and any
subsequent constraint-exchange repair require their own source/protocol freeze.
No new task data, inference, labels or predictor fits were produced. Paper files
remain untouched.


## 15 October 2026 — Complete-field current-geometry mapping counterexamples

Frozen commit354452e and protocol6b4b3004 execute the two-case diagnostic once:
four saved robot poses,10static forwards,0physics/retarget/inference calls.
Workers complete in27.208s(milk20) and64.481s(mug28), within the fixed budgets.
Raw arrays/logs and read-only predicate replay are preserved under
research_artifacts/hug_full_field_mapping_repair_2026-10-15_v1/run_v1/.

Milk: the original current pose has530definite lost/added proximity tuples;
the previous fitted pose has684, with321new definite violations and909lost
baseline-passing relationships. It repairs17old definite tuples;150become
unresolved and363remain definite failures. Mug: original3183, previous fit2874;
515new definite violations and2112lost baseline-passing relationships. It repairs
53old definite tuples;771become unresolved and2359remain definite failures.
Every original historical definite counterexample persists in the current
original poses. All original historical caps remain literal; the two experimental
poses pass them, while tiny current-original cap excesses remain explicit.

These are overlapping cell/part/band tuples from two development grasps, not
independent contacts, disjoint areas or task labels. Native-required relations
total341721 and804343; ambiguity is retained. Neither reducing a count nor
turning a definite failure into an ambiguous interval is an accepted repair.
Full orientation isNOT_RUN and continuous coverageNOT_PROVED; all four measured
poses have proximityFAIL under the declared geometry/arithmetic assumptions.

The historical kernel emits divide/overflow/invalid matmul warnings at halfspace
products despite finite output distances. A separate manufactured-operand probe
reproduces warning flags with bounded operands and agrees with scalar/einsum to
1.11e-16m. This does not clear actual operands: their independently frozen replay
is pending. Preserve the warning limitation with this diagnostic. No old result,
threshold, cap, dataset, controller or paper is rewritten. The previous capped-row
improvements cannot be promoted as a full-field faithful mapping. Current-scene
pose-adapter qualification and a separately bounded full-field exchange repair
are subsequent work; no production collection or predictor training resumes.


## 15 October 2026 — Actual halfspace arithmetic replay and pose-adapter controls

Frozen 8b1ddde executes the declared operand replay once in 19.24 s: four saved
poses, six static forwards, zero physics/inference/retarget/distance-query or
training calls. All 2,369,968,440 halfspace outputs across 5,710 matrices are
finite. Matmul and nonoptimized einsum differ by at most 5.55e-17 m; 91,360 fixed
scalar-reference samples agree within the same bound. No inside-mask or hull
predicate disagrees. Matmul emits each of its divide/overflow/invalid warnings
5,662 times; einsum emits none. The 217 bound files pass post-run readback.
Evidence: research_artifacts/hug_mapping_matmul_diagnosis_2026-10-15_v1/.

This supports replacing only the three diagnosed halfspace products in a new
version's mathematical kernel. It does not certify all distance fields, unknown
normal regions or mapping fidelity. The old source/run remain unchanged. The
new kernel's exact three substitutions and base/output hashes are declared in
the separate v2 geometry_patch.json; physical geometry and thresholds do not
change. Other operations retain their original arithmetic.

The current-scene SurfacePoseAdapter independently passes 17 controls in 1.04 s,
zero integration calls: current imported hand versus source XML FK, all 30
Jacobian columns at nonzero wrist/finger/root rotation, literal joint/domain
bounds, exact saved named-root replay, A–B–A and full non-hand state invariance.
Original native targets and historical caps are copied literally. A world-body
selection mistake in a strengthened test was corrected before the final peer
and root checks; the numerical tolerances were unchanged. Receipt:
research_artifacts/hug_mapping_admission_2026-10-15_v1/surface_pose_controls.json.

These controls qualify a numerical pose interface and scoped arithmetic repair,
not a faithful HUG-to-Shadow grasp. A separately frozen two-round full-field
constraint-exchange pilot is being prepared. It preserves all original and
newly passing measured relationships, literal caps and retained orientation
eligibility. Turning a definite failure into ambiguity is not labeled a repaired
relationship. No task rollout, new HUG generation or predictor training resumes.

## 15 October 2026 — Full-field exchange v2: no accepted surface improvement

The once-only pilot frozen at f4a616c completed both declared development cases.
Protocol SHA256: 93ef66966dc2b6b41751725733962f4225fb2cd6f56f6b6fb48fdc1eb036ab4b.
Evidence: research_artifacts/hug_full_field_mapping_repair_2026-10-15_v2/run_v1/;
report SHA256: 801a31bc6bd9f13f82f6d827005b42b43567563f3ee2fce482efb51598facef3.
The two workers use 710 static forwards, 349 raw callbacks and seven complete
field checkpoints in total; every counted integration/inference/retarget attempt
is zero. Worker elapsed times are 85.73 s and 86.24 s. Protocol completion is not
mapping acceptance.

Milk20 returns NO_ACCEPTED_SURFACE_IMPROVEMENT: both exchange candidates preserve
old passing relationships but introduce one new definite little-finger violation
at cell1111/band5mm. That relationship was originally unresolved and absent from
the worst-per-stratum active set. Both candidates are refused. The returned pose
restores literal tip caps but repairs zero relationships; all 530 original
definite proximity mismatches remain.

Mug28 returns REFUSED_NO_PROTECTED_START: the cap-restored candidate loses one
previously preserved native-far thumb relationship at cell16107/band2mm. Its
preservation margin changes from +3.552797912 nm to -5.339340325 nm, a distance
change of -8.892138237 nm. The candidate is refused and the literal original pose
is retained as an explicit diagnostic fallback, with its original strict cap
failures visible. All 3183 original definite mismatches remain; no feasible
mapping is admitted.

Independent receipt SHA256:
6ab84c12ca72b3ab20128e61fc5c272a71bedffd347718a86167e500051d469f.
It verifies 348 bound files before/after, all seven saved NPZ files, every saved
classification/margin, original-v1 field equality, exact retained-orientation
predicates and literal serialized tip/root/joint checks without new geometry or
optimization. Only baseline/final checkpoint digests were directly indexed by
the v2 result; intermediate hashes are inventoried in that independent receipt.
The unchanged legacy visual-mesh transform also emits NumPy matmul warnings;
the earlier halfspace replay does not clear that separate operation. Preserve
this scoped limitation. No threshold/cap was changed, no fidelity PASS or task
label was issued, and collection/training remain paused.

## 15 October 2026 — Protected exchange v3 and a native MANOSim reference

The once-only v3 run frozen at 792b576 completes both cases: 1200 static forwards,
594 raw callbacks, twelve complete field checkpoints, zero counted physics,
HUG-inference or official-retarget attempts. Worker times: 101.24 s and 229.27 s.
Active protocol SHA256: acd393ead3430eca81eaf053ae0b852fa9ff8bf0182fe07cbcc5c0f9711c53be.
Report SHA256: ba3b9ff8625d17ca0b0857bd6feb2b1e6a4bfc34038e4e6cdbc5235e1bc25f75.
Evidence: research_artifacts/hug_full_field_mapping_repair_2026-10-15_v3/run_v1/.

Joint restoration meets the literal original tip caps in both cases and retains
all previous full-field preservation. Counterexamples from rejected checkpoints
are retained as hard guards in later rounds. Final milk20 has 529 definite
mismatches versus 530 originally: one definite failure becomes unresolved, with
zero newly preserved relationships. Final mug28 preserves four previously
unresolved relationships; zero original definite failures become preserved and
all 3183 definite mismatches remain. Both final poses lose zero original passing
relationships and introduce zero new definite failures. The four gains are
overlapping cell/part/band tuples, not four independent contacts. Neither result
is a full strict mapping PASS, continuous coverage proof or physical outcome.

Independent receipt SHA256:
ad42952118059f54edaadd17261cc529c25570f0be14a98a3df3f6e7ca7c6257.
It verifies all 379 bindings before/after, twelve write-time checkpoint hashes,
four visual artifacts, exact saved predicates/margins, complete counterexample
unions, retained orientation predicates, caps, source joint limits and literal
root serialization without new FK/geometry. Whole-mesh scalar arithmetic agrees
with original visual transforms within 4.16e-17 m and 1.36e-19 m^3; original
warnings/outputs remain preserved. This clears that named arithmetic comparison,
not all geometry or physical realism.

The focused combined controls pass 110 tests with zero integration calls. An
initial plain pytest invocation could not import the pinned dependency bundle;
the project check wrapper supplies it. The first unexecuted preparation
declaration (dffea24a, archived as preparation_1_protocol.json) omitted the new
counterexample policy from its description. This was corrected before source
freeze and any native run; only the active acd393 declaration was executed.

The user then supplied https://github.com/KevinyWu/manosim. Source inspection is
pinned to c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac. Its shipped right-hand shape
file and the local HUG shape file have identical SHA256
3e9a85085ae67f36745ce647b904acb4e3620e8ac46378755498291fcf010ddf.
This supports inspecting a direct native MANO pose path. It is not a measured
pose-transfer or collision-surface PASS. The corrected Shadow translator's
separate audit source/tests are prepared but unexecuted while native MANOSim
qualification takes priority. No new collection or predictor training resumes.

## 15 October 2026 — Native MANOSim v1 refuses an exact shape mismatch

The once-only native qualification frozen at df3623c stops in case20 before any
model or geometry execution. Case28 is NOT_RUN. Every actual static-forward,
parametric-MANO-forward, collider query/scan, HUG-generator, retarget and physics
counter is zero. There are no pose/field arrays and no numerical fidelity verdict.
The error is `Native/upstream shape identity differs`.
Protocol: d93b5f5b65a5ec224ca4922b766ad90de2a6eb0502e67384e812019f6b28cad4.
Independent receipt e05ff183dfdd7b7c8333ca8f46033a80638b0f48d48878832b187e22e78b8d5a
verifies all443bindings, runtime, source freeze and failure artifacts without
native model execution. Evidence: research_artifacts/manosim_native_transfer_2026-10-15_v1/run_v1/.

A subsequent hash-checked data-only read of the two fixed native records finds
identical float32 beta values in both:
[-2.375,-1.25,-2.046875,-0.8515625,1.65625,-1.3515625,-1.8515625,-0.671875,-1.6875,-1.2109375].
They differ from the nominal shared shape file by at most0.005000114440917969
in beta space. All ten values exactly equal round-to-nearest-even BF16 rounding
of the nominal file, represented again as float32. This is an observed numerical
relationship, not proof of which checkpoint/export operation caused it. The
source inference code prefers a slim BF16 checkpoint and uses fixed_betas.

The shared upstream/HUG shape-file hash therefore did not prove identity of the
actual exported grasp shape. Preserve the refusal; do not substitute nominal
betas or widen the shape gate. Next build a distinct MANOSim hand using the exact
saved beta values through the pinned right-only upstream builder. Freeze and
verify those generated assets before a separate native pose/surface audit.
Upstream configuration values stay fixed; per-bone masses/inertias naturally
recompute from the generated shape. No native build or follow-up audit has run
at this entry. Collection/training remain paused; manuscript files are untouched.

### 2026-10-15 — Literal exported-shape MANOSim build preparation

Implemented a right-only upstream build transaction with one neutral MANO
forward and no MuJoCo model/forward/step, generation, retargeting or training.
107 combined manufactured checks pass (integration calls:0). Known cube/tetra
mass properties qualify the installed Trimesh5.1.1 API. The read-only declaration
`manosim_native_transfer_2026-10-15_v2/build_protocol.json` binds1832 source/input
files (SHA14285bf11df570b69d76a479f327a6a25d233382e4c735bad64bac6cf8b44787).
The first preparation exposed a nominalfloat64 versus nativefloat32 metadata
assumption; it wrote no protocol and performed no build. The corrected code
records the nominal conversion explicitly and keeps exported beta literal.
Actual shape build and independent readback remain pending at this freeze.

2026-10-15 follow-up: parent preflight at1380ebb refused Git tracking of ignored
historical geometry/provider inputs before creating an output directory or
launching a worker. Preserved the original declaration and explicit refusal.
Corrected only committed-code selection; all1832 input hashes remain required.
New `build_protocol_v2.json` and107-pass controls precede the first actual build.

### 2026-10-15 — Exact exported-shape MANOSim build and independent readback

Executed once at a8cc986 under active protocol fc93126db674f6653614a89269616fda65e816928db361a6c98f97ac4739406f:
COMPLETE in 2.55 s wall time, one neutral MANO forward, zero MuJoCo model/forward
or integration, zero generation/retarget attempts. Produced the literal shape,
two XMLs and sixteen OBJ assets. Builder report SHA178b66db3881b202ace1163cf8a3af1c957c811a04e16ddf00a1fe133cdceb31;
generated shape SHA4ffb005c8ce6636506c2bfee215d3c0e3dc3a561fc4206a8a9cbec27f3b33937.

Independent readback (SHAfda553cf728540c582ae5733cdd4d42e1281c062418c5d722f95c1aa9697e838)
verifies all 1,832 bindings before/after, 19 assets, 35 inventoried files and 91
saved arrays without model/FK/geometry execution. Native beta in both records
equals generated NPY and actual producer input; neutral pose/mean are zero.
Producer mass is exactly 0.4 kg, all raw/XML inertias pass positive-eigenvalue
and physical triangle checks. XML mass sums to 0.399999780 kg after serialization;
OBJ rounding is at most 0.499306 micrometres. These are provenance and basic
asset checks, not dynamic calibration. Pose/surface fidelity remain NOT_RUN.
Tracked receipts are under `manosim_native_transfer_2026-10-15_v2/build_run_v1`;
derived geometry stays ignored and byte-bound. Collection/training stay paused.

### 2026-10-15 — Freeze exact-shape native pose and collider-field audit

Independent source review and 135 manufactured checks pass, integration calls
zero. Native protocol SHA070d05931626b226e9e808ce44e0f407af64cb666316ebf70f290bb042af56d7
binds 1,879 inputs, including all 30 raw build artifacts and exact generated
shape/report provenance. Only saved development cases 20/28 are admitted, both
mesh and capsule variants, 600 seconds / 200 static forwards / two CPU MANO
forwards / two complete field scans per case. The first hard stop prevents the
second case. Original 5 micrometre reconstruction/anchor and 1e-6 rotation
comparison limits, cell grids, part labels and surface predicates are unchanged.
Collection, training and fidelity admission remain disabled. Actual measurements
and independent saved-array readback are pending at this freeze.

Before any native execution, review identified that a hard stop discarded
available creation-time artifact rows from the worker report. Corrected only
failure logging: preserve verified partial rows, original failure and any
unindexed interrupted files separately, without claiming complete coverage.
The earlier 070d0593 declaration remains unexecuted. New active
`native_protocol_v2.json` SHA c81cdd617ea98ad0a0edf1b37dddb9e5a78871b59cf2898b9d216fa847cc3207
binds the corrected source and 136 passing manufactured controls, integration
calls zero. All scientific rules and quotas remain unchanged.

### 2026-10-15 — Measured native pose PASS, strict collider fidelity FAIL

Active protocol c81cdd61 at freeze be46e622a5ddbdcf97539fd2689bccc917289146
completed both saved development cases. Each used five static MuJoCo forwards,
two CPU MANO forwards and two complete field scans, with zero physics,
generation or retarget attempts. All sixteen native joint anchors and body
rotations pass for both variants. Worst anchor error: 3.3261657e-7 m; worst
rotation element error: 2.1846116e-7. Source mesh reconstruction is within
3.0034251e-8 m. Numerical limits remain 5e-6 m / 1e-6 rotation elements.

On the unchanged six-part fields and 1/2/5 mm bands: milk has 341,721 required
relationships across 19,174 cells; mesh lost/added/unresolved = 5/11/540 and
capsule = 8/61/1,223. Mug has 804,343 required relationships across 45,484 cells;
mesh = 200/22/2,158 and capsule = 193/206/4,891. These are overlapping
cell/part/band tuples, not contacts or success rates. All four strict fields
fail; full orientation remains NOT_RUN and continuous equivalence NOT_PROVED.

Independent `run_v1/independent_readback_v2.json` (SHA
5cff858388ff1edb7ab34d4c17868907d8b701198c024a9bc2e70e12201ab13d) verifies all
1,879 bindings before/after, saved source parameters, sixteen-bone arithmetic,
literal quaternion/XML replay and all field masks/denominators without any
model or reference geometry calls. Initial compiled arrays were not separately
saved: in-run initial/replay equality is source-bound, while saved replay is
independently reconstructed. The first verifier's scope error (checking an
unused auxiliary cache column) remains preserved; the corrected verifier uses
exactly the frozen six anatomical columns. Experimental results did not change.

Saved-field diagnosis: 214/222 mug mesh discrepancies involve palm/index; the
index witness at cell 32,591 is 0.637 mm from native skin versus 9.244 mm from
the collider. Palm fields are identical between variants. Investigate neutral
rigid segmentation, posed skin deformation and convexification with the now
validated pose fixed. Pose-fitting and threshold relaxation are excluded.
Native report SHA af7e626eb96ec94d583fc79c7a1a4fb6d71397041e9dd11597de674330c0bd14.
This qualifies the tested native joint-transfer reference, not production
Shadow retargeting, collider equivalence or physical calibration. Collection,
training and manuscript work remain disabled in this stage.

### 2026-10-15 — Freeze saved-geometry collider construction diagnostic

Implemented nine ordered saved-geometry stages with a mandatory reference
reproduction gate (1e-12 m and exact field masks), before any construction
comparison. 63 manufactured controls pass, integration calls zero. Protocol
a64aeee90e924d6dd0ddf92f8a80d7354d1c1d83a63dd95af03eabe5fecc48b7 binds
1,914 files. Cases 20/28 each have a 600-second, nine-scan limit; pre-query
triangle-point work limits are 357,019,880 and 846,912,080. No model, FK-engine,
MANO, physics, generator, retargeting, training or task outcomes are permitted.
All original target fields and poses remain fixed. Source review and actual
measurements are pending at this freeze; all surfaces remain diagnostic only.

Final source review and 63 controls pass on the held source. The earlier
a64aeee9 preparation was never executed; final identity guards and partial
receipt publication are bound in active `protocol_v2.json` SHA
f5d80b06df40bb900861d4fa310e999e099ed468bb527e16af0838c8a671dbf7 (1,914
bindings). No geometry queries have run before this final declaration.

### 2026-10-15 — Construction diagnostic completed and independently verified

Executed active f5d80b06 at 790d0d596d62cfaedf4d2c81059345a02f8a2bc4.
Both cases complete: 13.79/25.77 s parent wall time; nine scans each;
220,961,176 / 524,157,616 pre-query triangle-point work units, within bounds.
All model, MANO import, FK, physics, generation, retarget and training counters
remain zero. Native reference distance error is exactly zero and all masks
match before subsequent stages run. The posed uncapped source also has zero
definite or unresolved required relationships in these query domains.

Ordered stage lost/added/unresolved counts (milk; mug):

- Native reference: 0/0/0; 0/0/0.
- Posed uncapped source: 0/0/0; 0/0/0.
- Neutral-LBS surrogate: 0/0/472; 0/3/1,093.
- Rigid neutral surrogate: 0/1/418; 0/1/957.
- Cross-bone faces removed, fixed target labels: 7/1/472; 256/1/1,264.
- Same geometry with upstream owner-part labels: 7/1/472; 256/1/1,264.
- Saved pre-OBJ owner-cloud hulls: 5/11/540; 200/22/2,159.
- OBJ decimal serialization: 5/11/540; 200/22/2,158.
- Literal XML plus saved compiled body frames: 5/11/540; 200/22/2,158.

Target denominators remain 341,721 / 804,343. These transitions are conditional,
nonadditive construction comparisons, not task scores. Fourteen closure faces
and 311 cross-bone faces are distinct operations. All owner vertices already
occur in retained triangles (zero orphan vertices); float64/float32 target face
labels agree. Owner-part relabeling changes some mug distances but neither
case's classification masks. Quantization affects one unresolved mug tuple;
it does not explain the definite failures.

Final source replay versus actual compiled mesh fields differs by at most
1.2025921e-9 m / 4.1157555e-9 m with identical masks. No compiler-specific
attribution is made. Independent saved-array verification checks all 1,914
bindings before/after, 60 Git blobs, 38 artifacts, raw/tracked scalar copies,
all masks/denominators/witnesses and construction arithmetic (max residual
5.55e-17 m), without model/FK/distance queries. Receipt
`run_v1/independent_readback.json` SHA964e55f90bdaea6af2f2b316ac36aa19e0e508bbdb038737b0fb3bc36d96c546;
report SHA2556931a8b5e60568ed2569eff57270103fb033be37350d5103cca5579fc1214.

The evidence supports a joint-spanning collider repair with fixed validated
poses. It does not admit these diagnostic surfaces as dynamic hands or release
collection/training. Production target remains unchanged while the user
clarifies native MANO versus Shadow; physical calibration is still pending.

### 2026-10-15 — Native MANO target selected; next repair scoped

The user selects HUG's native MANO hand in MANOSim with faithful transfer as the
priority. Existing Shadow results remain historical and are not pooled into the
new embodiment. Exact-shape native joint transfer has measured PASS; strict
surface fidelity remains FAIL. No new execution result follows from this choice.

In preparation: manufactured 3.3.7 triangle-flex radius/contact controls and a
full-face posed-skin reference with locked articulation. A static surface match
cannot qualify finger-motion deformation or articulated contact forces. A
distinct native recorder will preserve 67 position, 51 velocity and 51 command
channels with their actual units. No native collection/training or manuscript
work is enabled; source/protocol review precedes the next bounded run.

### 2026-10-15 — Native recording contract implemented with collection closed

Implemented `manosim_native_recording_v1` independently of the historical Shadow
format. The contract binds exact shape/assets/collider/controller/model hashes,
named 67/51/51 layouts, actual per-channel units and frames, and sampling policy.
The array-only recorder preserves raw wxyz quaternions, separate requested and
effective scalar controls, interval/phase clocks, and diagnostic isolation.
Mapping/setup refusals and runtime errors retain null physical outcomes.
Native loaders reject Shadow domains, width coercion, artifact/hash mismatches
and asserted positive admission. No supported native admission certifier exists.

Focused manufactured controls: 78 passed, zero integration attempts. Tests block
model/data construction, FK, forward and integration. This implements storage
and validation only: compiled-address gathering, pose-to-control conversion,
controller/runner integration, dataset export and prediction remain pending.
It creates no native task data, trained model or transfer/physics claim.

### 2026-10-15 — Freeze manufactured triangle-flex capability controls

35 manufactured source/protocol controls pass, integration attempts zero.
The new primitive-only declaration fixes 12 synthetic models, 12 data creations
and 84 explicit static forwards, with a 120-second worker limit. Tests cover
zero/1-micrometre radius, both sides, rigid frame changes, single/multiple body
attachments and a disconnected-triangle gap that must remain contact-free.
Exact topology/owners, analytical distances/points/normals and normal-row
Jacobians have explicit numerical limits. Multi-owner rotational differences
remain diagnostic; no MANO dynamics claim follows. Compilation rejection and
partial attempted work remain explicit. No licensed/native asset or task label
is read. Actual engine measurements are pending at this source freeze.

Before actual execution, source review found that an exactly touching contact
may have no active force-constraint row. Corrected the control to retain such
touch contacts descriptively; penetrating contacts must still have a row.
The original declaration at b0e9db9 remains unexecuted and unchanged. Active
`protocol_v2.json` binds the corrected source and 37 passing manufactured
controls. Scientific model cases, numerical limits and operation quotas remain
unchanged. No actual engine work occurred under the unused declaration.

### 2026-10-15 — Primitive geometry passes; contact derivatives refuse admission

Executed primitive v2 at 835b27e: 12 model compilations, 12 data constructions,
84 explicit static forwards, zero integration/native/training operations;
worker wall time 0.290 s. All geometry, both-sided contact distance/point/normal,
rigid-frame and disconnected-gap controls pass. All 12 independent normal-row
Jacobian checks fail, leaving both radius assessments NOT_QUALIFIED. A rigid
triangle owner translation coefficient is 1.688165034 in magnitude instead of
1. The sphere coefficient is 1. Report SHA
660abd835427660b4d3d9d03005bd3bf334b2b0a0d42f6fdecf8edb946a1f28c.

Pinned-source inspection identifies inverse-distance element weights normalized
by L2 norm in 3.3.7; they do not sum to one. Earlier engine-weight descriptions
as barycentric were incorrect. No threshold/reference or saved result changes.
The native full-surface wrapper must refuse this capability receipt. Independent
saved-array verification is pending. A separate 3.4.0 runtime has been installed
without dependencies because its source uses sum normalization; its actual
rigid-contact behavior is unmeasured and a new prospective protocol is pending.

Independent `run_v1/independent_readback.json` SHA
aac0602ae31286cc3af24332879f36469d6606817fe775543945ad83381d3369 verifies
977 bindings before/after, eight Git blobs and 36 creation-time artifacts.
Saved engine Jacobians match the independently reconstructed pinned L2-weight
formula within 4.44e-16; the physical reference rows reproduce within 2.22e-16.
Maximum vertex/contact-distance errors are 2.78e-17 / 2.54e-17 metres. Both
NOT_QUALIFIED verdicts and the helper's refusal are confirmed without another
model, FK, contact query or integration. The freeze is released.

The full-surface candidate and bounded wrapper now pass 56 manufactured controls
with zero integration calls. They preserve all 778 vertices/1,552 faces, literal
shape, fixed anatomical labels and exact saved named pose, with independent
compiled-readback/field guards. No native preparation/model/field execution has
occurred: the wrapper cannot select a radius from the failed capability receipt.

Final prepared 3.3.7 wrapper adds parent verification of child scalar/result
artifact identities; 57 manufactured controls pass, integration zero. Its
active receipt is `controls_v2.json`; the earlier 56-control receipt remains.
No actual native preparation or run has occurred under either source condition.

### 2026-10-15 — Freeze isolated 3.4.0 rigid capability experiment

48 manufactured controls and independent source review pass, integration zero.
The separate declaration preserves the exact 12 synthetic models, 84-forward
ceiling and all original numerical requirements. It retains the full six-case
assessment per radius and adds a prospective four-case single-owner verdict
for locked articulation only. Multi-owner failures cannot be omitted or turned
into articulated MANO admission. All 977 old study bindings are rechecked;
the base 3.3.7 runtime is unchanged. Source/runtime bytes, the independently
verified old failure and every new output identity are bound before execution.
Actual 3.4.0 measurements remain pending at this freeze.

### 2026-10-15 — Isolated 3.4.0 rigid contact capability passes

The separately frozen run at 21f07ef completes 12 model/data constructions and
84 static forwards in 0.578 s, zero physics/native/training operations. All
eight single-owner cases pass (four per radius), including the disconnected
gap, both sides, transformed frames and independent rigid normal derivatives.
Maximum rigid translation/rotation Jacobian errors are 3.33e-16 / 1.04e-17.
The four multi-owner translation checks still fail (max 0.108185); both full
six-case radius assessments remain NOT_QUALIFIED. Rigid-only capability cannot
admit finger deformation, native fidelity, physical tasks or model training.
Report SHA3728a2bd58c3375de78975980a97dfb6c06cff2f2885120cf90170630146dc1a.

Independent receipt SHA40f8ab4e2c6c918eea206a788b497fdf15253bc4f563ae8a298ed3a4652e8c4a
verifies 1,022 new bindings plus all 977 preserved bindings, 27 Git blobs and
36 artifacts. Source sum-normalized inverse-distance/common-point Jacobians
match all active saved rows within 3.33e-16; physical reference rows reproduce
within 2.22e-16. Capability selection returns only the locked-rigid scope and
retains the full NOT_QUALIFIED result. No engine calls or reruns are used for
readback. The freeze is released for a separately declared native snapshot test.

### 2026-10-15 — Freeze two native full-surface snapshots on isolated 3.4.0

64 manufactured controls pass, integration zero; independent source/test review
is clear. The separate native declaration fixes saved development cases 20/28,
all 778 original vertices/1,552 faces, original six-part labels, literal shape
and saved named joint pose. Prior skeleton/actuator content is retained while
16 old mesh colliders are disabled and the complete surface is attached to the
palm. Fingers stay fixed; this is a saved-pose reference, not articulated MANO.

Per case: one compilation/data creation, at most two static forwards, two full
field scans, 600 seconds and query_count*3,104 triangle-point work. Reference
fields must reproduce within 1e-12 m before the candidate field. All original
queries/radii, 1/2/5 mm predicates and numerical allowance remain unchanged;
compiled topology/vertices/oriented normals and original 5-micrometre native
kinematics are checked without adjustment. Full orientation/continuous contact,
native admission, physics, collection and training remain disabled. Actual
measurements are pending at this freeze.

### 2026-10-15 — Native snapshot v1 stops on a serialized shape-layout bug

At freeze af2ac9c, case 20 completes only the reference scan (29,758,048
triangle-point tests); its saved reference field reproduces the prior field
bytes. The candidate loader then refuses because it incorrectly expects a
flat ten-element shape file. The bound producer asset and decoded shape are
both literal float32 arrays of shape (1,10), with all values exactly equal.
No shape parameter, hash, pose, threshold or field predicate changes are needed.

All native model/data/forward/FK and physics counts remain zero. Case 28 is
NOT_RUN; physical outcome and measurement remain null. Both sealed partial
artifacts and the failure are preserved. The stopped version will remain
immutable; a separate v2 will correct only the exact row-layout contract,
retain the same field gates and bind this failure. Independent readback is
pending. No collection/training admission follows.

Independent receipt ac808d73b2c3921308f299f9f44a0a2be416256c4d39311211a0c419c209449f
verifies all 3,027 bindings before/after, 114 Git blobs, both partial artifact
hashes/arrays and the exact old reference-field bytes. All 341,721 required
reference relationships are preserved. Both serialized shape arrays are (1,10)
float32; the source's expected flat layout caused the refusal. No engine or
field query is rerun during independent verification; the freeze is released.

### 2026-10-15 — Freeze exact row-shape contract correction in native v2

68 manufactured controls pass, integration zero; independent source review is
clear. New v2 requires both literal shape arrays to be float32 (1,10), exactly
equal and bound to the unchanged original asset hash. Flat/wrong-row/wrong-type
or changed-value inputs refuse. No reshape, coercion, changed beta, geometry,
pose, field predicate or numerical limit is introduced. V1 source and the
verified hard stop remain immutable and enter the new declaration's lineage.
The same two saved cases, operation quotas and locked-articulation limits apply.
Actual v2 measurements remain pending at this freeze.

### 2026-10-15 — Complete native skin preserves both fixed-pose fields

Native snapshot v2 completes at 5d55bb1 under protocol
00b3ee6d13c3f2c9ed88fe1f07977905ea0f09de3cdea0ccb8327ff9f7246a76.
Case 20/28 parent times: 10.31/13.80 s. Each uses one model/data creation, two
static forwards and two full field scans; triangle-point work is 59,516,096 /
141,182,336, within the original per-case caps. All physics, native generation,
retargeting and training counts are zero; physical outcomes remain null.

All 778 vertices and 1,552 original faces retain their labels and oriented
topology. Max vertex component error against the rigidly re-expressed target:
2.78e-17 / 5.55e-17 m; direct comparison with original saved native vertices is
8.33e-17 m for both cases. Corresponding-normal component error <=1.44e-14.
Original reference fields reproduce exactly. Both
candidate fields preserve every required relationship: milk 341,721/341,721;
mug 804,343/804,343. Lost, added and unresolved-required counts are all zero.
Maximum distance delta is 6.94e-17 / 8.33e-17 m. Original native-ambiguous tuples
(3,411 / 14,369) remain reported separately and are not new required successes.

This resolves the measured near/far construction loss in two fixed poses, using
the whole skin attached to one palm. Finger articulation is prohibited; full
object-query orientation is NOT_RUN and the complete mapping verdict remains
INCOMPLETE. No native collection or training admission is issued. Report SHA
6c4f180aa2579c16a14f07a29bd046466c0a4cbdfebdea5042ee667424839d06.
Independent saved-array verification is pending; no further queries are needed
for that readback.

Independent receipt SHA50661b96420355a59f01a2893c0ec4ebb288d1862a37f6eb0d22e4dec4b8f1ca
verifies all 3,041 bindings before/after, 124 Git blobs, 18 creation-time
artifacts, raw/tracked result equality and every array hash. Literal 67/51/51
state inventory, native frames, exact shape, complete closed topology/parts and
unchanged skeleton/inertial/actuator XML are checked separately from the old
collider masks and added flex. All masks, denominators and operation counts
reproduce without engine, FK or distance queries. The verifier explicitly
distinguishes vertex comparisons with the rigidly re-expressed target from the
original saved native vertices. Global INCOMPLETE/no-admission flags remain.
The source freeze is released; collection and training stay paused.

### 2026-10-15 — Freeze motion, orientation and contact qualification sources

The user's next gate is finger movement, contact orientation and physical
contact behavior before collection/training. Three separate declarations now
have 172 passing manufactured controls: articulated skin/state tangent99,
saved full-grid orientation26, and stock rigid contact dynamics47. These are
source controls, with zero actual native evaluations/contact queries/physics.
Earlier synthetic arithmetic and arbitrary noncommutation-detector failures
are preserved separately; no scientific native gate was fitted to an outcome.

Articulated skin retains all shape/135pose-corrective/16bone blend terms and
uses51tangent coordinates. Two saved cases each have109 fixed perturbations,
an independent source32oracle, unchanged saved baseline, full51column
derivative/FD and VJP/virtual-work checks. Allowed engine work is one original
model compilation and109 mj_integratePos calls per case; Data/FK/forward and
physics are prohibited. This tests a skin function and state convention,
not yet an articulated collision surface or actuator response.

Orientation reuses both complete original grids, original triangles, bands,
parts and ambiguity thresholds. All tied normal-facing intervals remain
explicit; point agreement cannot grant continuous full-domain PASS. An unused
first preparation is preserved before conservative tie-interval hardening.

Stock contact fixtures predeclare14conditions/10650steps, start-force versus
endpoint-state clocks, analytic freefall/support/friction references, full3D
Jacobian, cone KKT residuals, momentum/reaction/work checks and timestep
comparisons. They qualify numerical behavior of this stock model only.
The separately implemented exact-skin J-transpose compliant-force kernel has
manufactured energy/virtual-work/reaction controls, including overflow refusal;
its native contact finder, callback and integration remain unqualified.

All actual results are pending at this source freeze. Each run must retain
failed/partial/NOT_RUN evidence and verify source/input/output hashes before
publication. Global mapping, native dynamics, collection and training remain
closed. Manuscript files are untouched.

### 2026-10-15 — Articulated skin passes; mechanics retains a sliding failure

At freeze eac84d987201739944b615ee14a7cb5779c7d876, skin protocol166a1d04
completes both109state cases (7.62/6.23s). All declared numerical checks pass:
source32/candidate32 coordinate residual<=2.98023224e-8m; continuous64/source32
<=5.95767092e-8m; baseline continuous64/original saved native<=7.30089949e-8m.
The independent original AA source reconstruction is exactly the saved target.
All51 J/FD columns pass (max4.11393050e-11); actual MuJoCo right-increment
residual<=1.11e-16. There are two model compilations/218position increments,
zero Data/FK/forwards/physics. All global admission flags remain closed:
running engine collision-surface articulation and contact forces are NOT_RUN.
Report44453ca5; independent saved-array receipt3bf9dbed verifies5235bindings,
138Gitblobs/17artifacts, all state/precision/derivative/work arithmetic without
a native skin, FK, engine utility or query replay.

Stock contact protocolfd69fde5 completes all14conditions/10650steps in5.55s.
Thirteen conditions pass. Strong sliding at1ms fails the declared acceleration
reference:1.57242002 versus2.4525m/s2. The0.5ms scalar check passes at2.54129105,
but the final velocity difference0.164840826m/s fails the unchanged timestep
comparison. Overall numerical qualification is NOT_QUALIFIED. Saved final100ms
shows intermittent contact at both steps:11/100 and19/200 active samples,
normal-force peaks26.2282/24.1732N, and airborne gaps up to670.5/487.3um.
KKT, impulse/work and action/reaction identities pass; those alone do not prove
stable sliding. Freefall, support, low-force creep and isolated impact pass;
support penetration is98.1um and low-force creep0.000613125m/s as declared.
Independent receiptca52e5b verifies1071bindings/50Gitblobs/42artifacts and
replays all10650saved endpoint metrics without engine calls. A separate finer
timestep declaration will retain the same model, gates and original failure.
These stock fixtures do not qualify the new skin force law.

Orientation protocolfd726019 hard-stops case20 before all query classes;
case28NOT_RUN. The source cache is(Q,8): six finite hand fields and two all+Inf
carrier fields; the canonical saved field is(Q,6). Both original first-six
fields match exactly. Independent receiptdc0b6f7b verifies3946bindings/140Git
blobs, zero query/engine work and the complete failed receipt. A distinct v2
requires that literal layout and compares the first six columns exactly;
31manufactured controls pass. No geometry, grid, target, mask, tolerance or work
quota changes. V1 remains immutable. All original source freezes are released
after independent readback; no collection or training has occurred.

### 2026-10-15 — Full-grid orientation executes; reliable signs agree

Corrected orientation v2 completes both saved cases at77d736c under protocol
317f2169. All19174milk/45484mug grid queries, original bands, parts and
uncertainty predicates remain. Reliable unique point counts are301/1759;
1/2/5mm band counts55/113/301 and405/757/1759 respectively. There are zero
reliable sign reversals among469/2921part-band observations. These overlapping
band observations are not independent grasps or contacts.

The conservative cell check certifies1matching milk cell of4914possible and
0mug cells of21512possible;4913/21512remain unresolved. Both full orientation
verdicts stay INCOMPLETE. Point agreement does not prove unique outward
normals, complete continuous orientation, articulated contact or task physics.
No unresolved cell is relabeled as success. Independent receipt13c759a4
verifies3958bindings/148Gitblobs,1018creation-time NPZs/chunks, all array hashes,
eligibility/tie intervals and every count/area summary, without geometry,
model, FK, field or physics replay. Source freeze is released; admission stays
closed. This corrects the loader, rather than changing scientific targets.

Next frozen checks are the actual state-to-skin-to-engine geometry bridge and
the same stock contact fixtures at0.25/0.125ms. Geometry-only contact/actuation
disable flags are explicit; stock flex must never apply the new skin's forces.
The separately versioned compliant law also needs its own integration fixtures.
Its proposed1/0.5ms steps were rejected before freezing or physics: the resting
smooth-friction linear damping implies an explicit-Euler local stability limit
of about0.408ms. Prospective0.25/0.125ms steps preserve the same material-model
assumptions and gates. This analytic bound is specific to that synthetic
resting-contact fixture and does not establish a native-hand timestep policy.

### 2026-10-15 — Freeze engine surface bridge and distinct physics repeats

Source/peer controls: engine geometry bridge62pass; stock finer-timestep
repeat52pass; separately versioned compliant point-contact integration32pass.
All have zero actual integration at this source declaration. Bridge test-tool
usage first refused an unsupported reporter option before any tests; corrected
usage and the earlier50control subset are preserved in its lineage. The stock
repeat's first manufactured pass had one path-mock fixture failure (51pass/1
fail), corrected without changing any scientific gate. Compliant proposal
30controls at1/0.5ms is archived NEVER_EXECUTED_NOT_PREPARED; analytic stability
then determines the prospective0.25/0.125ms pair, with32finalcontrols.

Bridge protocol123408f6 binds5262files and preserves the exact109state grids
on both native cases. Per case one model/data,109static forwards,109complete
skin evaluations and109direct flex updates; contact and actuation are disabled
as an explicit separate condition. Updater reads state and changes only local
flex vertices; all other model arrays/options and state/force buffers are
protected. Continuous64 engine readback and original source32/baseline normal
comparisons retain separate verdicts and unchanged gates. Zero physics steps.

Stock protocol6b22b6b5 binds1154files, retaining the original NOT_QUALIFIED
lineage and identical model/solver/material/gates. The14conditions have a
fixed0.25/0.125ms pair and42600steps. Compliant protocol13b2d8e4 binds1163files
and exact frozen force-law/control bytes. Its8conditions have27000steps/law
evaluations; engine collisions/constraints are prohibited. Analytic references
are mg/k penetration, the declared smooth-friction creep equilibrium and
sliding acceleration. Continuous dissipation and finite-step energy defects
are recorded separately; matching their algebra is not zero integration error.
Neither numerical contact model certifies actual MANO contact or real material
properties. All actual results remain pending; no collection/training admitted.

### 2026-10-15 — Compliant fixtures pass; stock convergence remains failed

At freeze5410cf8, stock refinement6b22b6b5 completes14conditions/42600steps in
19.17s. All individual analytic checks pass, including sliding meanacceleration
2.37188830/2.48972537m/s2 at250/125us. Overall NOT_QUALIFIED persists: endpoint
velocity difference is0.0133270m/s laterally and0.0266540m/s vertically, each
above the unchanged0.0122625m/s comparison gate. Both traces remain
intermittent bounce/traction. Independent receipt0700ba53 verifies1154bindings,
76selected Git sources,42artifacts/all42600intervals, every original assessment
and failed convergence. No further stock refinement or physics rerun occurs.

Compliant protocol13b2d8e4 completes8conditions/27000steps+lawcalls in14.92s.
All8analytic checks and4timestep comparisons pass. Support is1.962N/98.1um;
low-force smooth creep0.000258198890m/s; high-force acceleration2.45250203m/s2
at both steps. Max final velocity difference is0.0003435405m/s. Engine contact
counts/constraints remain zero; only explicit kernel loads act. Force/power,
common-point reaction, finite-step impulse/work and energy-defect identities
pass. Discrete integration defects are deliberately nonzero: support summed
defect+0.0033796/+0.0016148J at250/125us, with identity errors<3.7e-16J. This is
numeric qualification of the declared synthetic model, not material calibration
or full MANO dynamics.

Independent saved-only receipt e370d13f verifies1163bindings/86Gitblobs, all
27000intervals, original law operands/reactions/dissipation/energy and every
decision without invoking the engine or law. An earlier readback receipt
b5d7270b is preserved with verifier-only NumPy matmul warning observations;
the additive receipt uses explicit nonoptimized einsum, warnings-as-errors,
and records zero warnings. Producer sources/outputs and outcomes are unchanged.

Engine bridge123408f6 hard-stops case20 before every engine/skin operation:
the XML inventory incorrectly counts one default geom template among16
instantiated world geoms. No model, Data, forward, skin call, flex update or
artifact is produced; case28NOT_RUN. Independent receipt6f49b4ec verifies all
5262bindings/147Gitblobs and the exact16meshes/16bodies/16worldgeoms+1template
operand. Separate v2 will select only instantiated world geoms, require and
preserve the default template, and retain every numeric gate/budget/skin
operand. V1 source and its failure remain immutable. Global native contact,
mapping and collection/training admission stay closed.

### 2026-10-15 — Freeze exact XML inventory correction for engine bridge v2

65manufactured controls pass, including one default geom template plus16
world geoms and strict missing/extra template/geom refusals. Read/hash-only
prepare checks the actual bound XML inventory before freezing. New protocol
5164742d binds5277files and the independently verified zero-operation v1
failure. Only instantiated world geom masks are disabled; the original
default template is required and preserved. Skin, frames, faces, exactshape,
source32/continuous64/originalnormal gates and all109state quotas remain
unchanged. Actual v2 model/forward/skin results are pending at this freeze.

### 2026-10-15 — Actual engine finger surfaces match continuous skin; source normals fail

Engine bridge v2 (`5164742d`, source freeze `99af7b1`) completes both native
cases and all 109 declared static states per case. Each case uses one model,
one Data, 109 static forwards, 109 skin evaluations and 109 direct flex updates;
there are zero physics steps, object queries or actuator commands. Contact and
actuation remain explicitly disabled. This checks articulated geometry from
actual engine state, rather than dynamic tracking or grasp execution.

Actual engine vertices match the independently sealed continuous64 skin within
1.11e-16/1.67e-16 m (cases 20/28); corresponding normal-component errors are
4.82e-14/4.29e-14. Joint-anchor errors are 0.270/0.262 micrometers. Protected
state, model and force-buffer checks pass. The unchanged strict source32 normal
comparison FAILS: maximum component errors 2.5434e-5/1.0917e-5 exceed 1e-6.
The original saved baseline comparison also fails on normals, despite vertex
errors below 0.1 micrometers. These are distinct verdicts; engine placement
agreement does not convert the source-normal failure into a pass.

Report SHA256:
`ddd540216c5b3a32c6c8e2e41769e2675ed2c22ed857a376814669c93cfba387`.
Independent saved-only receipt
`research_artifacts/manosim_skin_engine_bridge_2026-10-15_v2/run_v1/independent_readback.json`
has SHA256 `2b554e501e8f22bfa3e7edb3923bb0ed2fda56a5bb9af992d46393aa0ea8c069`.
It verifies 5,277 bindings, 155 Git blobs and 15 creation-time artifacts, and
recomputes all saved vertex/normal comparisons without engine or skin replay.
Assertions available only from the bound producer, rather than persisted
operands, are identified separately in the receipt. The source freeze is released.

Saved-output diagnostics locate discrepancies in the already-sealed source
precision profiles, before the engine surface update. Read-only coordinate
algebra then identifies a concrete pipeline mismatch: the old root state uses
float64 Gram-Schmidt of the saved 6D root, while HUG evaluates its post-conversion
float32 axis-angle root in camera coordinates. Inverse recovery from the old
root differs by one float32 ULP in case 20's root Y and all three root entries
of case 28; all 45 local finger entries and all three camera translations
recover exactly. This algebra does not execute a new skin, model or physics run.

Next is a separately versioned source-initialized state/pipeline correction,
preserving the original HUG geometry, thresholds, old state grids and failures.
No geometry fitting, offsets or relaxed normal tolerance are permitted. A new
float32 surface must not silently inherit continuous64 contact derivatives.
Full mapping, oriented native contact, collection and training remain closed.

Saved-array illustration source:
`research_artifacts/manosim_saved_evidence_figures_2026-10-15_v1/render_engine_bridge.py`;
output: `runs/manosim_engine_bridge_evidence_2026-10-15_v1/engine_fingers_and_normal_failure.png`.
The figure shows actual engine meshes and the failed source-normal gate; it is
not a physics or actuator-motion demonstration.

### 2026-10-15 — Freeze the literal HUG camera pipeline in a distinct state domain

New source-only protocol `92b5f9bc2552b7a56cc0698aa981ae565f1701d665395095ae55ec6558c2ec9b`
binds 5,304 files. The 46 manufactured controls pass, with an independent
46-control peer pass and zero native/model/skin/physics operations. Peer review
corrects the equivalent-axis-angle branch at a 2-pi crossing and requires the
exact inherited 109 perturbation IDs/deltas before any actual work. Controls
also exercise byte-level baseline refusal and float32 translation ordering.

The new domain initializes root orientation from the original post-conversion
AA32 and original camera transform. It recovers AA32/translation32 from actual
named qpos, calls pinned ManoLayer in float32 camera coordinates, adds float32
translation there, and only then applies the original cameraWorld64 transform.
The original baseline inputs must recover byte-for-byte before any source or
model constructor. Baseline source geometry and the original 5-um/1e-6 gates
are unchanged. The old GS64 state grid, continuous64 profile and normal FAIL
remain historical results rather than new acceptance targets.

Per case: 600 seconds; one model, Data and source layer; 109 source forwards,
109 static engine forwards and 109 direct flex updates; zero steps, object
queries or actuator commands. All original masks/actuation are disabled for
this geometry condition; protected state, original inertials and model buffers
remain checked. Source/actual joint anchors and rotations provide an additional
coordinate check beyond placing source vertices in the engine. Actual results
are pending at this freeze. Native contact, collection and training remain closed.

Saved-only orientation plot added separately from this experiment:
`runs/manosim_saved_orientation_figure_2026-10-15_v1/orientation_evidence.png`.
Its write-once receipt verifies existing comparison-array hashes and recomputes
2,060 unique reliable sites, zero sign reversals, and the explicit unresolved
part-band cell counts. Rendering performs zero new geometry or physics queries.

### 2026-10-15 — Corrected source-native geometry passes all 218 static states

At source freeze `78f26d6deee192b83cb9409996a478c518e9f3a5`, protocol `92b5f9bc`
completes cases 20/28 in 8.75/7.30 seconds. All seven declared geometry and
coordinate checks pass in each case, with the exact operation counts and no
physics steps, object queries, actuator commands, task collection or training.
Both initial AA32 and translation32 payloads recover byte-for-byte before
source/model work. Each subsequent source call uses inputs recovered from the
actual engine state, with all 109 anchors/rotations checked independently.

Maximum engine/source vertex errors: 2.498e-16/1.110e-16 m. Corresponding
normal-component errors: 4.793e-14/4.802e-14. Against the unchanged original
saved baseline, vertices agree within 1.665e-16/1.110e-16 m and normals within
2.032e-14/1.971e-14. Maximum anatomical anchor errors are 0.3233/0.3163 um;
rotation-component errors are 3.405e-7/2.879e-7. All original 5-um/1e-6 limits
remain unchanged. The old GS64/world64 profile's FAIL remains preserved.

Report SHA256:
`5d5c4ed0b985fdbff3e525cf7c05709b02f6e160b2e6f583ce670dc1b5ff61ce`.
Independent receipt
`research_artifacts/manosim_native_source_pipeline_2026-10-15_v1/run_v1/independent_readback.json`
has SHA256 `9a5d3a551a29d45049aa360dd1012fab86c32a97f45b38c4477be77b32d5a13e`.
It checks 5,304 bindings before/after, 165 selected Git blobs and 19 creation-time
artifacts, and replays original input bytes, coordinate branches, float32
translation ordering, flex-local placement, oriented normals and every stored
decision without model, MANO, skin, FK, geometry-query or physics operations.
Verifier-development arithmetic-order/axis-reduction failures are embedded in
that receipt; producer outputs and scientific thresholds are unchanged.

Saved-array evidence plot:
`runs/manosim_native_source_evidence_2026-10-15_v1/faithful_finger_transfer.png`.
It shows actual engine baseline/coupled poses, displacement across the declared
grid and the original baseline normal-error correction. These are finite
static forwards, not an actuator trajectory. Contact orientation during motion,
native contact forces, dynamic tracking, collection and training remain
unqualified. The source freeze is released for the next separate qualification.

### 2026-10-15 — Freeze native static material-point force coupling

39 manufactured controls and an independent 39-control peer repeat pass with
zero native operations. New protocol
`3ae68dec49abb8e70c1b14903be8b01eff9955feafc3f3501a1a15eaaaef1641`
binds 5,474 files and 252 selected Git sources, including completed source-native
geometry and the separate synthetic compliant-contact benchmark. Preparation
reads/hashes saved labels to select original face IDs [0,28,142,127,1001,2]
for palm/index/middle/ring/little/thumb, each at fixed centroid barycentrics.

This distinct continuous64 physical profile evaluates the full MANO skin in
camera coordinates, then places it in world64. Its own JVP, central difference
and VJP use that same graph. Registered surface/Jacobian and state/profile
hashes refuse source32/old-world64 mixtures. Literal skin weights are preserved.
Source32 alignment keeps a separate original 5-um/1e-6 verdict; no alignment
failure can be hidden by a force-coupling pass.

Per case: 600 seconds; one model/Data, one baseline skin, 51 JVP and 102 FD
evaluations, one aggregate VJP, six material points, twelve fixed law/load
conditions, 24 paired static forwards and twelve full mass readouts. There are
zero steps, MANO source calls, object queries, inference, collection or training.
Known planes have fixed axis normals and 100-um penetration; they may load
backfaces and do not qualify original-object contact orientation or search.

Actual named 51-DOF addresses/types/tree, original inertials and unchanged
paired q/v/ctrl/time are required. Required checks are gap derivatives,
generalized virtual work, common-point force/moment reaction and native
M*(acceleration_loaded-acceleration_baseline)=tau. Translational/angular mass
residual limits are 1e-8 N/1e-10 Nm. Engine contacts and actuation are disabled;
all applied, bias, passive, constraint and actuator operands are saved. Literal
weight sums can differ from one, so rigid-root wrench equivalence is retained
only as a diagnostic. Actual results are pending; no native dynamics or data
admission follows from a static pass.

### 2026-10-15 — Native static force coupling passes; precision alignment remains failed

At source freeze `be15a1d2a0a6b65ace47194bfb0d25432d3d5b8f`, protocol `3ae68dec`
completes both native cases with exact quotas and zero integration, MANO source
calls, object queries, collection or training. All twelve fixed material-point
load conditions pass per case. The 51-column camera-graph derivative matches
central differences within 7.074e-11/8.327e-11; every plane-gap derivative passes.
Aggregate VJP/explicit transpose discrepancy is 4.441e-16 in each case.

Native M*(loaded acceleration-baseline acceleration) residuals are at most
2.665e-15/3.664e-15 N in translations and 2.220e-16/1.769e-16 Nm in local angular
coordinates. Common-point reaction, generalized power, original inertials,
actual named 51-DOF layout and paired state/force guards pass. These are static
known-plane force-application checks, including frictional load directions;
they do not execute a dynamic grasp or search the original object surface.

The distinct physical64/source32 alignment verdict FAILS on face normals:
1.80781e-5/2.44436e-5 versus the unchanged 1e-6 limit. Vertex differences
63.305/65.883 nm pass the original 5-um comparison. This failure does not undo
the exact source32 transfer geometry pass, and the force-coupling pass cannot
be used to hide it. A source32 derivative and full native contact remain
unqualified. Rigid-root wrench equivalence is only a stored diagnostic because
literal source weight sums are preserved rather than normalized.

Report SHA256:
`b484f256d8131c41f63af36538cdba2d01aba1c7748d643431eefc3a91337e0d`.
Independent saved-only receipt
`research_artifacts/manosim_native_point_contact_coupling_2026-10-15_v1/run_v1/independent_readback.json`
has SHA256 `58ef3edebe9d46fbe634a323e9428b48ed8a33499fa2b5d9a52d77a0a0b936c8`.
It verifies all 5,474 bindings before/after, 253 Git blobs including the
protocol, 43 creation-time artifacts, all point/derivative/law/work decisions
and every native mass-response condition. Each dense M is independently
reconstructed from the saved packed qM and DOF tree with exact equality, then
checked positive definite. No engine, model, MANO, skin, FK, geometry or physics
operation is replayed. Producer derivative evaluation and guard assertions are
distinguished from arithmetic that saved operands independently establish.

Saved evidence plot:
`runs/manosim_native_force_evidence_2026-10-15_v1/native_force_coupling_and_remaining_gap.png`.
All plots use saved operands only and retain failures. Native dynamic tracking,
contact search/orientation over motion, physical-model source alignment and
calibration remain open. Collection/training admission is false. Source/HEAD
freeze is released after independent verification.

### 2026-10-15 — External HUG/MANOSim implementation research; no new simulation

The user requested existing integrations and validation resources. A dated
primary-source audit is saved at
`research_artifacts/hug_manosim_external_resources_2026-10-15_v1/sources.json`
(SHA256 `f2e7f2fec6af28c9b85c70381b26d5ec81c4dfdc839a567bafedf61814826b50`).
It records pinned repositories, ten bounded public-text fetches with hashes,
benchmark directory metadata, observed mechanisms and explicit evidence limits.

Concrete independent examples: [Troy Robotics Lab's Genesis G1/Inspire integration](https://github.com/TroyRoboticsLab/Hug-G1-Sim/tree/8243b3b2d9f66d7283c5523cf76829b7030ff4f9)
contains camera/world transforms, heuristic finger retargeting and an execution
loop; [REVEL Robotics' WUJI integration](https://github.com/REVEL-Robotics/hug_grasp/tree/03ef1f0c5fd721f044ccae38cb763e90f6d227cd)
contains MuJoCo landmark-Jacobian retargeting and hardware command interfaces.
These use HUG with different hands, and do not establish native source-skin
contact fidelity. [Eric Heiden's Newton issue #3700](https://github.com/newton-physics/newton/issues/3700)
is a direct MANOSim import attempt, with a repair branch and regressions for
lost wrist translations/drive semantics. The issue is open at inspection;
related damping issue #3703 was resolved separately. Import repair is distinct
from a published grasp-validation result.

The most directly relevant physical baseline remains the original authors'
[MANOSim](https://github.com/KevinyWu/manosim/tree/c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac).
It uses capsule collision fingers and a convex palm, with a separate displayed
skin. Recorded-human replay motivates its compliant settings. Public HUG still
marks the complete HUG-Bench simulator as unreleased, so the public components
do not supply an exact ready-to-run benchmark reproduction. Related
DexHOI2Sim, ArtiGrasp, MS-MANO, GraspTTA, DexYCB and ContactPose examples are
classified separately in the audit rather than labeled direct adopters.

A read-only parameter comparison also finds that upstream `myhand`'s float64
shape file differs from our saved HUG float32 shape after explicit casting and
reshaping (maximum beta coefficient difference 0.00500011444). This is not a
measured surface error. Any new baseline must use the literal saved shape.
HUG-Bench's public metadata lists scene tar archives; internal oracle records
were not verified and archives were not downloaded.

No downloaded code was executed. There are zero new model constructions,
skin evaluations, physics steps, grasp generations, task data or training.
Next baseline/validation choices are prospective and recorded in
RESEARCH_DECISIONS.md; all existing source, physics failures and admission
flags are unchanged. No manuscript files were edited.
