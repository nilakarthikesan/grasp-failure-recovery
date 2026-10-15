# Completed experiments and evidence

Updated 14 October 2026. This index reports saved simulation results.
The [research protocol](HUG_MONITOR_RESEARCH_PLAN.md) separates the recorded
research question and pending comparisons from completed experiments.
The [working paper](paper/working_draft.tex) draws on the evidence below.

## Workshop writing plan

On 14 October the authors chose to aim for the Sim2Real2Field late-breaking
deadline with the completed diagnostics. The [paper structure and writing
plan](PAPER_STRUCTURE_PLAN.md) records verified format requirements, example
paper structures, a four-page outline and proposed starter prose. The central
submission story is warning reliability on successful unfamiliar grasps; the
matched diversity comparison remains unexecuted. The existing manuscript has
not yet been shortened or converted to the official template.

## Original pilot and baseline

- **Collection:** 1,040 attempts, with 444 successes and 596 task failures.
  The data contains cube and apple geometry, four observation families and
  sixteen HUG proposals. [Collection and provenance](MULTI_SCENE_COLLECTION.md).
- **Inputs and splits:** closing histories through 2.96 seconds, before lift
  begins at 3.00 seconds. Observation-grouped partitions contain 520 training,
  260 validation and 260 test attempts. [Split contract](DATA_SPLITS.md).
- **Baseline:** the state-and-command logistic model has failure-ranking
  AUROC 0.8523 on the reserved apple observation, versus 0.6419 for commands
  alone. At the original 0.5 warning threshold it detects 187/202 failures,
  misses fifteen, and falsely warns on 22/58 successes.
  [Model guide](PREDICTION_BASELINE.md) and
  [saved baseline](../research_artifacts/task_outcome_baseline_v1/README.md).

This split was declared after collection outcomes had been inspected. Both
geometries appear in training, and the test contains one observation family.
It is a retrospective development result rather than an unseen-object benchmark.

## Frozen evaluation on fresh objects and scenes

A subsequent batch contains 2,080 attempts: 659 successes and 1,421 task failures.
The original model, normalization and warning threshold were frozen.
Failure-ranking AUROC is 0.9173 on fresh cube/apple scenes and 0.5588 across
new milk carton/mug attempts.

All original mug attempts fail during approach because the retargeted hand
strikes the table. That batch cannot measure false alarms on successful mugs.
[Evaluation guide](FRESH_OBJECT_EVALUATION.md) and
[portable evidence](../research_artifacts/fresh_object_scene_eval_v1/README.md).

## Mug executor correction and frozen-model scoring

A separately versioned executor corrects the forearm/wrist clearance and adds
a runtime guard. Twenty-six checks complete without table contact; one
high-noise closure stops before contact and remains outside binary metrics.
[Correction and checks](MUG_TABLE_CLEARANCE.md).

The unchanged predictor scores all 26 completed checks. The mug subset contains
twenty successes and three task failures, and all 23 trigger warnings.
The stopped attempt has no outcome label or prediction.
[Scored recordings](../research_artifacts/corrected_mug_model_diagnostic_v1/README.md).

This establishes persistent false alarms on corrected successful mugs.
It does not demonstrate a training remedy.

## Wrist input and paired diagnostics

Both wrist commands in all twenty successful corrected mugs lie outside the
original training ranges. Reconstructed wrist feature terms push their saved
scores toward failure. This is linear score attribution, not isolated physical
causation. [Input diagnostic](../research_artifacts/mug_wrist_input_diagnostic_v1/README.md).

None of the eight observed training wrist-command pairs meets the declared
static table-clearance gate for the eight mug proposals. The later comparison
therefore uses two unfamiliar, feasible configurations per proposal.
[Feasibility](../research_artifacts/wrist_pose_feasibility_v1/README.md) and
[frozen selection](../research_artifacts/wrist_pair_selection_v1/README.md).

All eight zero-noise pairs complete: sixteen attempts, thirteen successes and
three task failures. Every attempt triggers a warning; six pairs succeed at
both wrist settings. One changed label crosses the 14 cm lift cutoff by a
submillimeter amount and is not evidence of a drop.
[First pair](../research_artifacts/wrist_pair_diagnostic_v1/README.md) and
[remaining seven pairs](../research_artifacts/wrist_pair_remaining_v1/README.md).

The corrected checks and paired diagnostics reuse development families and
have distinct executor versions. They must be reported separately.

## Partner provider foundation and replay gate

PRs 21 through 24 add interchangeable grasp proposal sources, a unified
evaluation command, explicit pose/geometry translation checks and official
DexGraspNet sample replay. All are merged into main through `b503c2f` and
integrated into the ongoing working branch.

The recorded ten-sample mug gate has seven approach collisions, one failed
acquisition and two null-label joint-limit rejections, with no successes.
This is a dataset-replay execution check, not a trained predictor evaluation.
[Replay and translation scope](DEXGRASPNET_REPLAY.md) and
[gate evidence](../research_artifacts/dexgraspnet_replay_gate_v1/README.md).

The generic v2 output lacks velocity and commanded-target histories required
by the existing predictor exporter. The fresh DexGrasp Anything worker and
compatible prediction-data recording remain implementation work. No model improvement is
established by importing these proposals.

ContactGen MANO and UniDexGrasp Shadow adapters and isolated inference workers
are also implemented on the working branch. Their unit fixtures are synthetic;
fresh inference, model-output acceptance and simulation checks have not run.
No evaluation data was generated for this implementation.
[Implementation scope and contracts](CONTACTGEN_AND_UNIDEXGRASP.md).

## One fresh HUG CLI execution check

One newly authorized HUG inference and MuJoCo case completes successfully through
`grasp-eval`: 0.151335 m maximum lift, 2.0 s hold and no hand/table contact in
163 saved samples. The existing cube observation and seed 0 reproduce the
earlier native proposal and saved-replay trajectory exactly. This is a
development reproducibility check, not an independent benchmark. The frozen
predictor was not scored. [Run declaration, results and runtime](HUG_SINGLE_EVAL_2026-10-13.md).

## Static mapping fidelity audit

Read-only source, frame and forward-kinematics checks reuse the existing HUG
cube and synthetic joint configurations. HUG's coordinate composition agrees
to numerical precision; nominal mean/max tip errors are 6.704/14.125 mm. Native near-surface vertices
and nominal Shadow contacts cover the same three cube faces, with penetration
in both representations; this is coarse coverage rather than contact equality.
Declared v2 extra closing moves commanded tips another 18.644–32.212 mm.
ContactGen has reviewed algebra but no measured real output. UniDex's shared
reference bridge has at most 1.751 mm tip error; its actual released hand asset
and contact fidelity are unverified. DGA's official 24-joint forearm-root format
needs a dedicated mapping instead of the handoff's 22-joint assumption.
These diagnostics establish bounded implementation facts, not four-generator
equivalence or prediction improvement. No new inference, physics, evaluation
collection, task labels or fitting occurred. [Audit and evidence](GRASP_MAPPING_FIDELITY.md).

## Individual HUG mapping confirmation

The user requested generator-by-generator confirmation, starting with HUG.
The saved native MANO export reconstructs exactly, 18 existing guarded tests
pass, and independent coordinate/quaternion/joint-order checks pass. Same-scale
surface comparison at the existing cube reveals detailed footprint changes:
occupied-or-near geometry IoU is 0.583 at a 2 mm diagnostic band and 0.682 at
5 mm. These include penetration and are not contact-preservation or prediction
scores. No acceptance threshold or full fidelity claim is established.
The HUG mapping remains partially confirmed; no new inference, physics,
evaluation collection, task labels or training occurred. [HUG case and evidence](HUG_MAPPING_CONFIRMATION.md).

## Claim boundaries and paper status

The saved results establish a simulation warning-reliability problem.
They do not yet establish improved prediction from added coverage, calibrated
real-robot failure probabilities, slip warning lead time or recovery benefit.
The larger coverage comparison and calibration/representation fits remain
unexecuted in the recorded evidence.

The manuscript remains an editable working draft. Submission preparation is
tracked in [the submission checklist](SUBMISSION_CHECKLIST.md).

The detailed milestone and exploration chronology is preserved in
[the archived progress record](archive/PAPER_PROGRESS_HISTORY_2026-10-12.md).
[Dated decisions](RESEARCH_DECISIONS.md) retain why the experiment design changed.


## HUG static mapping continuation, 14 October 2026

[The expanded HUG audit](HUG_MAPPING_CONFIRMATION.md) checks 32 existing poses
and refines the saved cube's surfaces across grid resolutions and anatomical
parts. Numerical transform/joint consistency is supported within tested scope;
contact fidelity remains approximate and unconfirmed. These are static
diagnostics, not new evaluation episodes, prediction results or a completed
four-generator validation. The working manuscript and frozen predictor remain
unchanged. The user prioritizes mapping validation before broader collection
and training; paper drafting is deferred while this validation proceeds.


## HUG wrist/palm and existing-object surfaces, 14 October 2026

[The static continuation](HUG_MAPPING_CONFIRMATION.md) checks wrist/palm anchors
on 32 saved poses and finger surfaces on six existing apple/milk-box/mug cases.
The declared anchor is realized; Shadow's separate wrist body is 34 mm away by
design. One shared translation explains 14.61% of tip squared residuals.
Native/target proximity differs on corrected collision-union and actual visual
surface queries. Object collision and visual boundaries also differ and require
separate review. These are geometric diagnostics with one MANO shape, not
physical contact equivalence or new independent evaluation/training data.

HUG remains partially confirmed and active. Acceptance criteria, collider
review and separate calibration/validation groups precede any revised mapper
or new collection. Production code, executor, frozen predictor and manuscript
are unchanged; no paper drafting or model fitting occurred. GPU rental remains
a future benchmark-based recommendation, not a provisioned run.


## Original-generator validation context, 14 October 2026

The four original papers and pinned implementations inform source-specific
mapping arguments in [the audit](GRASP_MAPPING_FIDELITY.md). A shared validation
framework is proposed; its tolerances and independent validation design are
not frozen. The user accepts preserving poor proposals and separating mapping
fidelity from task outcomes. This adds literature context, not new generated
evidence or a completed four-source comparison. HUG validation stays active;
the manuscript, production pipeline and predictor remain unchanged.


## HUG object/reference diagnostics, 14 October 2026

[The completed static phase](HUG_MAPPING_CONFIRMATION.md) separates inherited
object collision approximation from retargeter tradeoffs. Tested source/compiled
object frames agree; no placement correction is justified. A fixed released
reference and metric/static sensitivity show that better tip fitting can worsen
middle-finger geometry and object proximity. Current whole unsigned proximity is
higher on all six paired noncube development cases. Native signed area profiles
already penetrate the visible solid, so bad source proposals are retained rather
than repaired into successes.

[Measurement definitions and proposed
acceptance](HUG_STATIC_VALIDATION_PROTOCOL.md) distinguish numerical
correctness, approximate surfaces and untested execution/label equivalence.
Reference bound metadata is corrected without rewriting original results;
protocol pilots and final versions remain separate. Next
normal/sampling-uncertainty checks and an externally justified independent
validation design remain pending. No new inference, physics, eval data/labels,
training, production mapper change or manuscript edit occurs. This is supporting
mapping evidence, not a completed multi-generator predictor study.


## HUG scoped acceptance result, 14 October 2026

[Completed orientation and sampling checks](HUG_MAPPING_CONFIRMATION.md) reveal
preserved numerical decoding/conversion but changed native finger/object
relationships. All 32 saved native 3D exports reconstruct exactly. The six
detailed cases and three finer repeats have definite same-part proximity changes
under explicit conditional numerical allowances; some apple/milk query sites
also reverse object-facing orientation. These are static geometry witnesses, not
new eval outcomes or failure-prediction results.

[Acceptance V1](HUG_MAPPING_ACCEPTANCE.md) defines numerical PASS, strict
declared-band geometric PASS/FAIL/UNRESOLVED and separately untested outcome
adequacy. Current HUG fails the strict preservation claim in all nine detailed
resolutions while numerical conversion passes within scope. The rule is a scoped
project definition, not a universal paper standard; approximate retargeter
usefulness needs its own justified budget/independent validation. The existing
mapper and native poor proposals remain unchanged.

Next resolve the mapping-version/research-claim choice against these witnesses
before collection/training. Original manuscript and predictor are unchanged; no
drafting, new inference, rollout, eval dataset/label or model fit occurred.
Multi-generator predictor training and generalization remain the central
intended study, with this fidelity result supporting its validity boundary.


## HUG correction development, 14 October 2026

[The implemented static correction](HUG_SURFACE_REFINEMENT.md) adds native-relative
surface distances/closest-point vectors and a guarded joint correction. Six
existing apple/milk/mug grasps receive lower fitting costs and improved whole
unsigned proximity, but fingertip/middle accuracy worsens and some added-region
and direction witnesses regress. All six raw candidates violate the existing
fingertip limit; guarded derivatives respect it. Independent unchanged strict
acceptance still FAILS all six cases and three finer repeats.

The experimental API and 70 manufactured tests are implemented; the default
mapper is not replaced. Source/hand/controller/model/manuscript and old inputs
remain preserved. These are static development derivatives, not new generator
eval data, task labels, prediction fits or independent generalization evidence.
A constrained native-relative direction-aware solve remains proposed. The
central multi-generator failure-prediction study has not yet been trained or
validated by these mapping diagnostics.


## HUG constrained correction implementation, 14 October 2026 — results pending

[The joint constrained method](HUG_CONSTRAINED_REFINEMENT.md) is implemented in
an isolated API: five original-error fingertip constraints, strict mean guard,
actual native-relative direction targets and retained invalid-site penalties.
Fifty new manufactured tests pass; 120 installed new/prior API tests pass.
The six existing-grasp fits are still running under canonical runtime V2.1.
Independent source preflight and unchanged nine-resolution surface audit are
pending; no new geometry result or mapping sign-off is claimed yet.

Default production mapping, controller, original inputs, manuscript and frozen
predictor remain unchanged. No new generator inference, eval episode/label,
training or other-generator validation occurs. This is implementation progress
toward supporting mapping validity, not a completed multi-generator prediction
study or independent generalization result.


## 14 October 2026 — Joint fingertip/direction fitting and unchanged audit completed

All six saved-grasp fits are complete. The solver returned 0 eligible surface corrections and retained 6 exact original poses. Independent original/final source preflight passes all six; the unchanged six-coarse/three-fine strict mapping audit returns 0 PASS, 9 FAIL and 0 UNRESOLVED.

The isolated 24-joint constrained API and native-facing objective are
implemented. Each fingertip must retain at most its original native error
plus the separately declared 10 nm engineering arithmetic allowance; the
computed mean remains ≤10 mm. Invalid fixed direction sites retain penalties,
native-away targets are preserved, missing part mass is not redistributed,
and no successful-lift filter is used. Fifty new manufactured tests and all
120 installed new/prior surface API checks pass. These are implementation
checks, not full geometric equivalence. The default mapper is not promoted.

Final independent FK/source checks and the nine unchanged static surface
audits are complete. Optimizer nonconvergence/refusal is neither proof of
global infeasibility nor a native grasp-failure label. Preserve the frozen
1/2/5 mm rule, query cells, numerical assumptions, all source poor proposals
and each versioned result. Incomplete/import-aborted/unexecuted histories
and unused performance helpers remain separate. All 114 protected original
bytes, predictor and manuscript match; the new API/test are separately
bound. No new inference, physical eval data, outcome labels, training or
other-generator work occurred. [Completed evidence](HUG_CONSTRAINED_REFINEMENT.md).

Next technical work, if authorized, is a separately frozen constraint-feasible
refinement or a documented change of the mapping/embodiment claim followed
by appropriate validation. Do not loosen acceptance or call a fallback a
repair. Contact/failure-label equivalence, unseen-object validation and
predictor adequacy remain unresolved or untested.


## 14 October 2026 — Parallel provider validation and shared-paper writing authorized

**Previous plan and new user decision:** Individual audits had been sequenced
behind HUG. The user requests three handoffs for concurrent ContactGen,
DexGrasp Anything and UniDexGrasp validation, plus simultaneous updates to the
same manuscript, including abstract/introduction. This supersedes the earlier
sequential provider order. No new eval data is requested or generated.

**Evidence and coordination choice:** The completed HUG batch already used six
concurrent fits and two audit groups; expensive mesh queries share the host's
CPU. Separate chats advance provider-specific code/contracts and writing in
parallel, but do not add compute hardware. Shared MANO mapping affects HUG and
ContactGen; shared Shadow bridges, assets and measurement code can affect other
audits. Give each provider unique code/tests/evidence/report ownership, reserve
shared dependencies and Git operations for the coordinator, and bind each run's
actual scientific dependency closure. Evolving prose is recorded separately.

**Implemented:** Three ready-to-paste provider handoffs and a common coordination
guide are saved. The same existing paper now has marked source-specific blocks,
an updated multi-generator prediction abstract/introduction, supporting mapping
methods and a proposed one/two/three/four-source comparison. Its accepted title
is Evaluating Simulation Diversity for Grasp Failure Prediction. A lock/reread/
section-hash editor merges owned subsections without losing other writers'
changes; all 12 installed tests pass, including concurrent subprocess updates.
The standalone manuscript compiles successfully with the desktop compiler.
Historical HUG numerical result text is preserved exactly.

**Measured status and remaining work:** HUG accepted no correction and still
FAILS the unchanged strict claim at six coarse/three fine resolutions. These
are nine repeated static comparisons, not nine physical failures. ContactGen
and UniDexGrasp have synthetic/reference implementation evidence; DexGrasp
Anything needs its dedicated decoder/worker/registration. No actual-native
validation is claimed for those three. Missing authentic outputs/assets are
blockers to full native verification; synthetic fixtures cannot replace them.
Source diversity improving predictor generalization remains a hypothesis.

**Provenance:** Today's explicitly authorized manuscript revision follows the
completed HUG run and does not rewrite that run's old paper/source snapshot.
All prior scientific bundles and results remain immutable; shared model and
simulation inputs/code remain unchanged. No fresh inference, integration,
eval episodes, task labels or predictor fitting occurs in this writing phase.
[Parallel handoffs](PARALLEL_GENERATOR_VALIDATION_HANDOFF.md).


## 14 October 2026 — HUG feasible restoration completed; fidelity still fails

The old constraint failures are traced to the unsuccessful surface-search
trajectory, with feasible movement demonstrated on all six saved cases. A new
frozen seed-restoration stage returns six improving poses and passes all 30
original-relative fingertip caps plus independent source checks. It uses 18
surface calls in 49.57 s, reusing the old raw fits. Two cheap projections converge
and four reach their iteration budget; final feasibility is independently checked.
All 146 new/prior surface API tests pass.

The unchanged nine-resolution audit still returns 0 PASS/9 FAIL, now with nine
pointwise orientation FAIL results. This repairs constraint handling, not complete
grasp preservation. The default mapper, predictor and historical results remain
unchanged. No new eval data or training was generated. The existing paper's HUG
subsection is updated and compiles. [Evidence and remaining work](HUG_FEASIBLE_REFINEMENT.md).


## 14 October 2026 — DGA authentic native smoke and separate precision diagnosis

The owned dexgrasp_anything manuscript block now records one strict-loaded
794-key, unguided DDPM call producing two original cube proposals; both violate
source limits and remain rejected. Input/stage reconstruction PASS. The original
numerical FK/vertex FAIL is preserved; a separate saved-pose TF32-on replay is
bitwise identical and TF32-off static reconstruction passes unchanged gates.
This is static precision evidence, not a replacement inference or an accepted
mapping. The angular comparison is identified as a raw-matrix proxy.

The 150 provider/asset tests pass, and the existing manuscript compiles with the
desktop compiler. Other manuscript blocks and historical results are preserved.
The original surface audit remains blocked; contact adequacy is UNRESOLVED.
No physics, evaluation episodes/labels or training ran. The A40 Pod is stopped
and evidence is saved. [DGA report](parallel_validation/DEXGRASP_ANYTHING_VALIDATION_REPORT.md).


## 14 October 2026 — DGA fixed-cohort negative domain result

The owned DGA block now distinguishes the original two-output smoke from the
separately frozen16-output cohort. All16 originals violate sourceTHJ1 limits;
the negative value is already present in normalized output and exported unchanged.
Eight stage checks pass. Separate declared FP32 source references pass unchanged
numerical gates, while all defaultTF32 references fail. No source-and-target-valid
candidate is selected, so actual native mapping algebra and surface/contact tests
remain unestablished with zero surface queries. No samples were added.

Five manufactured cohort tests pass; the current manuscript compiles successfully
with the desktop compiler. All old results and other manuscript blocks are
preserved. No physics/evaluation labels/training ran; both DGA Pods are stopped.
[Measured report](parallel_validation/DEXGRASP_ANYTHING_VALIDATION_REPORT.md).


## 14 October 2026 — HUG scalar-objective gap diagnosed; direction selector tested

Saved-array analysis finds all six lower-cost feasible fits worsen the fixed facing objective and invalidate49–104 direction bins. Twenty of43 saved coarse orientation witnesses were already selected fitting sites; sparse coverage alone is insufficient. A conservative optional selector now separately preserves per-part direction SSE and original-valid bins. All 182 installed surface API tests pass (36 new manufactured controls), plus9 diagnostic math/identity controls. Independently reproduced saved replay accepts 6 baseline identities and refuses 6 previous selected fits; all 30 existing tip caps still pass.

No new geometry fit, query, inference, physics, eval episode/label or training ran. These are post-hoc development diagnostics and selector checks, not corrected poses or held-out evidence. All 9 strict HUG mapping audits remain FAIL; default mapper and acceptance rule are unchanged. The HUG manuscript block records this distinction. Next is a separately frozen bounded guarded geometry study. [Detailed note](HUG_SURFACE_SELECTION_GUARD.md).

The separately authorized replacement A40 is running with the saved DGA runtime restored and CUDA/import/hash controls passed; the original is kept stopped. This infrastructure recovery creates no scientific validation result. [Connection, cost and restore record](GPU_RUNTIME_RECOVERY_2026-10-14.md).


## 2026-10-14 — DGA preprocessing metadata and thumb-sign diagnosis

Completed a read-only CPU/public-source audit. Recovered checkpoint epoch60/step67762,794 serialized tensors and compatible100-step diffusion buffers; historical training config/data identity remain unverified. An author-posted22-to24-joint converter leaves original THJ0 unchanged in the DGA THJ1 slot despite opposite distal axes/limits. Actual published pose bytes (member SHA70bbacb7d7c53eb3de4526427bb10a756ecef0febc8b487b60bf297dd6afc23a) contain six negative and two positive final coordinates in the first eight consecutive records. Independent storage checks and a narrow relative-distal-frame matrix proof substantiate a released data/hand convention conflict, not complete checkpoint training lineage or whole-hand equivalence.

Updated only manuscript marker dexgrasp_anything using the section-hash/lock helper; compile receipt is stored with the metadata evidence. No scientific acceptance, angle repair, new inference, physics, evaluation episodes/task labels or training. All previous18 proposals/rejections,0/2 and0/16 source-valid counts, and numerical-reference policies are preserved; all304 old cohort files hash-match. Shared registration remains unsupported.

Next proposed gate is historical checkpoint convention binding followed by a separately frozen all-joint/frame semantic adapter and independent offline geometry/contact audit, using existing records before more GPU work. Evidence and detailed causal limits: research_artifacts/dexgrasp_anything_mapping_validation_2026-10-14_metadata_audit_v1/.


## 2026-10-14 — ContactGen native smoke manuscript update

The ContactGen marker block now reports the authentic one-cube/seed-42 A40 run,
exact native reconstruction, 4.44/8.52 mm mean/worst fingertip admission result,
corrected 60.97% 2 mm boundary-proximity retention (61.20% sensitivity), sampled
penetration and unresolved scientific/physical acceptance. Desktop compilation
succeeded on source `12a93e5b6456adf263e60fd5f667ed81fc652634a636dd8c1f93133cd5ea644d`.
Native outputs are private and hash-bound; both GPUs are stopped. No new evaluation
data or physical labels were produced. Evidence: `research_artifacts/contactgen_mapping_validation_2026-10-14_native_smoke_v1`.


## 2026-10-14 — DGA literal native geometry continuation

Updated only the owned DGA manuscript section with the separately frozen16-row mathematical-only audit: all24angles/rejections unchanged; palm residual≤5.648e-17m; native FP32 geometry reconstructed;15/16configurations have certified differences in encoded visual-triangle proximity at declared bands. Independent checks reproduce40,320area quantities and2,880interval overlaps. These are source-invalid mathematical configurations, not accepted grasps or physical contacts. The target-world/forearm output-label correction is additive and independently verified, with no metric rewrite.

All18original proposals and old0/2,0/16source-domain results remain; prior valid-row surface selection is still empty. The historical manifest gate is narrowed to intended legacy semantics, while operational pinned visualizer geometry can be audited independently. Full approximate exterior/contact adequacy and compiled collision/physical equivalence remain unestablished. No physics, new inference, task/evaluation data or labels, training, joint repair or shared registration change. A separate MultiDex checkpoint test is proposed only, pending the user's scope choice.

Evidence: research_artifacts/dexgrasp_anything_mapping_validation_2026-10-14_literal_geometry_v1/. Existing paper path and other provider sections preserved; desktop compilation receipt is included after the owned-section update.

## 2026-10-14 — ContactGen exact surface-depth and exposed-union assessment

Updated only the ContactGen marker in the existing working draft. Added the
strict-exterior 58.59% proximity result with its explicit changed denominator,
the exact 1.700186729 mm full-native-surface maximum and numerically exposed
3.794710139 mm robot witness (2.094523410 mm additional penetration). Retained
one-case, open-source-mesh, approximation-budget and no-new-evaluation limits.
Desktop compiler succeeded; paper source SHA256 `81bd8d68f05b9b600e5b08b1f5aa9a11d5d5d936365af6df15cedb6ef8c8e337` unchanged across
compile; all bytes outside the ContactGen marker were preserved during update.
Evidence: `research_artifacts/contactgen_mapping_completion_2026-10-14_static_v1/`.


## 14 October 2026 — HUG guarded search completed; fixed-anchor claim refuted

The HUG manuscript subsection now distinguishes actual guarded fitting from saved selector replay. All six fits fall back to exact original complete scene poses; all-six independent source/FK and 30 tip-cap checks pass. The nine original surface reports are reused explicitly (zero new full-grid measurements), retaining strict 0 PASS/9 FAIL and pointwise orientation 7 FAIL/2 UNRESOLVED. New driver and independent consistency controls are implementation/provenance evidence, not surface sign-off.

Separately frozen local/full-wrist covers and an all-SO(3) radial bound establish a conditional fixed-anchor palm obstruction for saved Mug B. Its existing positive-area cell is native within 4.746 mm at the 5 mm band and at least 6.646 mm from any rotated Shadow palm after allowances. Independent box/FK and high-precision arithmetic reviews pass. The paper retains the fixed-anchor/literal-surface/engineering-arithmetic limits and does not claim free-placement impossibility, physical contact, failure-label preservation or universal validation.

Production and frozen scientific criteria remain unchanged; no new eval data or training. A new source-only anatomical calibration is being prepared/reviewed, with no result yet asserted. [Evidence and next mapping work](HUG_GUARDED_FIT_AND_REACHABILITY.md).


## 15 October 2026 — HUG anatomical calibration and new-anchor proof completed

The HUG subsection now records the separately frozen canonical MCP-frame experiment: one source-only 8.668706 mm anchor calibration; all 6 numerical/mean-tip passes but 21/30 individual old caps refused, so 0/6 eligible. Independent source/C/FK/world-target reconstruction confirms the result. The same old positive-area cells refute both new Mug fixed anchors under arbitrary palm orientation; independent 90-digit replay passes 1,093 checks. No full-grid audit is falsely claimed for refused poses, no native failure label is inferred, and arbitrary other placements remain outside the proof.

Both runtime aborts, unchanged calibration bytes and all original results are preserved. Current strict native preservation is not accepted. The working title/main multi-generator prediction question remain unchanged; the target embodiment/failure-label meaning has been requested before another dependent mapping policy. No new native HUG inference, physics, eval data or training. [Completed calibration evidence](HUG_CANONICAL_PALM_CALIBRATION.md).


## 2026-10-15 — DGA separate MultiDex smoke and domain-transfer result

Completed the user-authorized four-output MultiDex arm and mirrored the frozen code, original outputs, runtime receipts and independent checks in [the distinct evidence bundle](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_multidex_smoke_v1/README.md). Two native poses satisfy source limits, none satisfies target limits; all four distal-thumb angles are valid in this bounded test. Independent inverse stages and an additive rejected-state algebra diagnostic pass, with all 24 values preserved. Default TF32 failures and separate FP32 reference passes remain distinct. No eligible surface case exists, and no physical/contact acceptance is claimed. Original DexGraspNet results and shared unsupported status are preserved. Updated only the dexgrasp_anything paper marker section; compiler receipt follows in the new bundle. No physics, evaluation labels or training ran.

<!-- contactgen-unseen-mapping-2026-10-15-v1 -->

## 2026-10-15 — ContactGen four-object native screen incorporated

Completed the authorized 12 native-grasp mapping cohort on four newly constructed
synthetic meshes (three fixed seeds each; training novelty UNKNOWN). The paper
retains all 12 scheduled attempts: 12/12 exact CUDA float32 native replays and 11/12
canonical engineering admissions. Capsule seed 101 fails the unchanged 10 mm mean-tip
gate at 12.39721 mm; its raw mapping stays saved. New whole-query 2 mm sampled
native proximity retention is 28.8937–61.6275% over 11 nonempty bands, with capped
quadrature and separate exterior denominators stated. No physical/contact
acceptance follows from these descriptors. New sampled penetration depths are
not continuous maxima; the original exhaustive cube proof remains separate.
Open-wrist/source-tie/population limitations and missing scientific fidelity
budget remain explicit.

Updated only the ContactGen marker through `scripts/update_paper_section.py`;
outside bytes matched. Existing paper path stayed open and compiled successfully
with the desktop compiler. Source SHA at compilation:
`6797b8fc35b53338395e6ed1e2acc1ada2ff68e27516614c32efccb2dcba78b1`.
Independent saved-array audit v3 passed 8,048 checks, preserving its earlier audit
versions. Evidence and scientific figure:
`research_artifacts/contactgen_unseen_mapping_2026-10-15_v1/`.
Private native outputs and arrays:
`/Users/nilakarthikesan/grasp-failure-prediction/runs/contactgen_unseen_mapping_2026-10-15_v1/`.
A40 `cn4nexx8e9fxjl` was stopped after verified download; estimated compute $0.26710
for this run, storage excluded. Twelve new native mapping samples were produced;
no physics rollout, outcome label, training or predictor evaluation episode ran.
Scientific mapping adequacy remains unresolved; no acceptance margin was invented.


## 2026-10-15 — DGA exact obstructions and retargeting development evidence

Added the exact joint-domain obstruction, fixed-palm fingertip lower bounds, separately frozen two-pose constrained retargeting results and visual-proximity diagnostic to the owned DGA manuscript block. New target-valid fitted poses remain distinct from original zero-eligible native outputs. The row3 4 mm² added visual near-region is conditional on the declared distance allowance and is not physical contact. Independent FK and saved-array checks pass; scientific approximation adequacy and independent validation remain unresolved. [Evidence](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_retarget_development_v1/README.md). No new inference, physics, labels or training. Existing manuscript path and historical results preserved; compilation receipt stored in this bundle.


## 2026-10-15 — ContactGen contact preservation and capsule audit
<!-- contactgen-contact-preservation-2026-10-15-v2 -->

Completed static follow-up on the twelve saved native grasps. All eleven originally admitted mappings have an exact positive-area counterexample to 2 mm proximity preservation against the exposed robot-union boundary. Capsule/101's native/object separation is at least223.516mm; two fixed80-call diagnostics do not rescue its10mm admission failure. Conditional adaptive area bounds and local orientation/node audits are retained with their limits. Independent exact, node and611,061-leaf checks passed; all271originalfiles unchanged.

The marked ContactGen manuscript section was updated and desktop compilation succeeded. Evidence: `research_artifacts/contactgen_contact_validation_2026-10-15_v2/README.md` and its hash manifest. No new native inference, GPU resume, physics or outcome labels. Approximate application suitability and physical outcomes remain unresolved; the recorded geometric failure is now established.


## 2026-10-15 — DGA free-palm tradeoffs and separate surface/collider evidence

Updated only the owned DGA manuscript block with the frozen two-pose free-palm development fit, independent FK and morphology findings, fresh/finer visual-area bounds, and declared palm collider-union comparisons. Row3maximum improvement is reported alongside palm motion and worse mean/site errors; no acceptance or native contact equivalence is claimed. Conditional visual and declaredcollider references remain distinct from compiled/physical behavior. Original histories and the active paper path are preserved. [Evidence](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_free_palm_v1/README.md). No new inference, physics, evaluation data/labels or training. Compilation receipt and final preservation checks are saved with the bundle.


<!-- hug-free-palm-2026-10-15-v1 -->

## 15 October 2026 — HUG palm-placement study completed, mapping still unvalidated

User chose common Shadow and authorized per-grasp translation with unchanged criteria. Both fitting stages (762 + 348 full callbacks) retain all six original guarded outputs. Independent tip-feasible diagnostic poses receive nine fresh strict surface FAIL results. A separate seed-origin coverage correction for three nanometer-shift selections adds five fresh FAIL results, preserving the first protocol. All six intentional-movement alternatives preserve original fingertip caps; actual palm shifts span 0.031372–4.891691 mm. Fourteen measurements concern nine robot poses from six existing native grasps, not fourteen grasps or physical failures.

The HUG section and abstract distinguish numerical admission, guard refusal and strict surface failure. The frozen criteria and production mapper are unchanged. Counterexample-constrained fitting is proposed only; continuous orientation coverage still requires evidence even after sampled failures are removed. No new native inference, physics, evaluation data or predictor training. See [study](HUG_FREE_PALM_MAPPING_STUDY.md) and its separately versioned evidence bundles.

The desktop LaTeX compiler reports success for both the canonical working paper (SHA-256 `906455622b1fe0bcb396a685341fafbfe1e39d70f464d0b67dc1f42881b52401`) and the exact selectively staged paper (`5bcf7db84de2a9d5ff0c86beed21e7324e99eeb4c13d7cf04c239167f310e092`). Other providers' uncommitted paper sections are preserved.


## 2026-10-15 — ContactGen repairs revalidated; strict preservation still fails
<!-- contactgen-mapping-repair-2026-10-15-v1 -->

Updated the owned ContactGen manuscript section with the conservative native-off-object preflight, exact derived-capsule source certificates, three contact-aware fitting schedules and independent final comparisons. All twelve final robot candidates pass both literal 10 mm fingertip gates and physical bounds, but all twelve retain exact positive-area 2 mm loss counterexamples. The capsule source repair is explicitly derived; the original failed native output and original 11/12 admission history remain unchanged. No research mapping is promoted.

The paper distinguishes unsigned proximity improvements from exposed-union preservation, adaptive development from blind validation, and fitting margins from scientific acceptance. The 71 focused software tests and cloud-kit isolated import pass; all 271 original files are unchanged. No new ContactGen inference, GPU resume, physics, outcome labels, predictor evaluation or training. See [repair evidence](../research_artifacts/contactgen_mapping_repair_2026-10-15_v1/README.md). Desktop compilation is recorded in that bundle after the section update.


## 2026-10-15 — DGA prospective surface-preservation experiment completed

Updated only the owned DGA paper marker with the new pre-test strict criteria, implemented surface-relative objective, all four guard refusals, unchanged baseline selection and fresh static validation. Both declared palm comparisons FAIL; visual row1 remains UNRESOLVED and row3FAIL. The manuscript distinguishes implementation/numerical verification from scientific acceptance, preserves conditional arithmetic/exterior/generalization limits, and does not claim a successful correction or global impossibility. Original results, source/production files and the working paper path are preserved. [Evidence](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_surface_preserving_v1/README.md). No inference, physics, evaluation data/labels or training; desktop compiler receipt is saved in this bundle.


## 2026-10-15 — DGA feasible-step correction and fresh validation recorded

Updated the owned DGA manuscript section with the prospectively frozen feasible-step method, unchanged scientific criteria, fitting-loss reductions and adverse landmark tradeoffs. Independent holdouts still fail combined native surface fidelity for both cases; mapping remains unaccepted. Original protocols/results, other provider sections and the working paper path are preserved. [Evidence and desktop compilation receipt](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_feasible_surface_v1/README.md). No new inference, GPU work, physics, evaluation data/labels or training. A source-faithful embodiment remains proposed, not implemented or accepted.

<!-- contactgen-counterexample-repair-2026-10-15-v1-stage1 -->
### 2026-10-15 — ContactGen continuous and self-intersection checks

Updated only the ContactGen paper section with the exact ≥8 mm³ prior capsule cross-digit intersection, the exposed-witness ellipsoid pilot, and its independently verified continuous lost patch despite zero losses detected by the fresh numerical audit. The pilot also has seven numerical middle–ring intersections and remains unadopted. The revised contour- and separation-constrained pilot is still in progress. Source/fingertip, one-way proximity, cross-digit geometry, application adequacy and physical outcomes remain separate claims.

Desktop LaTeX compilation succeeded for paper SHA `9aa8a56cab8bd607a5622cf8f2c161b73faf4b9641522ef779551ff222bdc62b`; the ContactGen section SHA is `f9e35ffed1bbead533331e2244b31bb68934f5979e386d63c5f40bee2f00c2ce`. Read-back confirmed all bytes outside that section unchanged. The original 271 manifest-bound files remain unchanged; no GPU, native inference, physics or outcome labels were added.

<!-- contactgen-counterexample-repair-2026-10-15-v1-stage2 -->
### 2026-10-15 — ContactGen ellipsoid certificate and native-surface scope

The ContactGen-only section now records the independently verified ellipsoid/10007 continuous 2 mm inclusion and all 359 cross-digit separation proofs, with original source identity and nonvacuity checked. It also records exact nonadjacent native-surface crossings in the ellipsoid and capsule, and explicitly limits the successful claim to the named static mapping properties. Corrected “fixed geometric obstruction” to the precise fixed-pose collision-envelope intersection wording.

Desktop compilation succeeded for paper SHA `44c0284f20cacf84410852163f7f78883c99fa906360ed0ecb86c97b4c48b533`; section SHA `2e1f79233bde892ec831a5dc85f361a096db14e92a69619ba3207aa86fc928d5`. All bytes outside ContactGen were preserved relative to the fresh preimage. Capsule repair and the rest of the cohort remain separate work in progress, not a completed generator-wide validation.


<!-- hug-direct-surface-constraints-2026-10-15-v1 -->

## 15 October 2026 — HUG direct-constraint results incorporated

Updated the existing HUG subsection and abstract with the completed two-version direct surface-constraint study. All six selected poses preserve original tip/mean gates, but all nine fresh full-grid comparisons FAIL strict preservation; pointwise orientation is 9 FAIL. The manuscript distinguishes Mug A's retained-row progress from increased thumb mismatches/new direction reversals, exact fingertip refusals from mean/domain acceptance, and certificate uncertainty from genuine mismatch. Search/callback budgets and the disclosed thread-count deviation remain explicit. No production promotion, new inference, physics, eval data or training. Other generator sections and the active paper path are preserved. See [completed study](HUG_SURFACE_CONSTRAINT_MAPPING_STUDY.md).

<!-- contactgen-counterexample-repair-2026-10-15-v1-stage3 -->
## 15 October 2026 — ContactGen capsule feasibility and exact continuation failure

The ContactGen marker now distinguishes the capsule's independently verified reference/separation seed from its unsuccessful contact continuation. The seed passes both 10 mm gates and all 359 cross-digit separation checks (minimum certified gap ≥0.423061 mm). The later candidate has an independently verified positive-area 2 mm loss (area ≥0.0753785 mm², native ≤1.637834618 mm, target ≥2.193980841 mm). Its failure is preserved. The remaining-ten screen and short-refresh retry are explicitly pending; no full-generator or physical acceptance is claimed.

Evidence is in the current counterexample-repair archive, especially `validation/capsule_feasibility_v4_scoped_receipt.json` and `validation/capsule_strict_v5_scoped_receipt.json`. Other provider sections and historical protocol results are preserved.

<!-- hug-criteria-alignment-restoration-2026-10-15-v1 -->

## 15 October 2026 — HUG evaluation scope clarified and restoration result recorded

The existing HUG subsection and abstract now distinguish our strict native-relative preservation contract from HUG's published generator benchmark. The 10 mm guard, original nonworsening caps and proximity bands are explicitly project choices. Practical adequacy for the executed Shadow prediction target requires a separately justified error budget and independent validation; none is claimed here.

The HUG subsection also records all twelve saved-endpoint fingertip restorations, zero eligible surface improvements, six unchanged reference poses, and the distinction between invalid direction evidence and actual reversals. The historical nine FAIL results remain prior measurements; no fresh grids or evaluation episodes ran. [Criteria explanation](HUG_CRITERIA_ALIGNMENT.md) links the review and bounded study. No other provider section was replaced.

The canonical open paper compiled successfully with the built-in desktop compiler at SHA-256 `750cf9d599afdb525d110375e1d323c7cabcc4e1721ccd0940a08c30f342423e`; HUG section SHA `6ae011b3e3af2b175b89f86fb74b93cd1b16265d53f6e7120666318938f11d4e`, abstract SHA `bd5a6b8ac68d4f57c5450e2290b07d2ad5e110ad91c8e9bf08760f2c2f1cb44d`. The user's existing editor and document path are preserved. No alternate document was compiled or opened.

<!-- dga-component-constraints-independent-native-2026-10-15-v1 -->
## 15 October 2026 — DGA component repair and independent native failures recorded

The existing DGA manuscript section now records the component-constrained method, the independently checked conditional palm-anchor obstruction for old development row 1, and the new sixteen-output MultiDex cohort. It distinguishes original versus tighter landmark domains, declared geometry versus physical contact, and unseen seeds versus population evidence. The first two preselected source-valid native outputs preserve their landmark guards but both fail fresh visual preservation; new row 0 also fails palm occupancy, while row 1 verifies empty palm exclusion without direction acceptance.

The manuscript retains all earlier DexGraspNet/MultiDex/correction histories and the same open paper path. Original target and scientific criteria remain unchanged. No provider promotion, physics, evaluation episodes/outcome labels or model training. [Frozen methods, raw outputs, independent checks and scoped decision](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_component_constraints_v1/README.md). Compiler confirmation is appended after the desktop check; editing alone is not compilation evidence.

DGA component-study compilation confirmed: the existing manuscript compiled successfully with the desktop compiler at whole-file SHA-256 `1090334760dbc6c899f0fac878418ddbbc8f665a3858f8d0a2116e44f1674441`, DGA section SHA `3368227c16a8a028a30fc4142d9baf2581b0327b20a6544d07c66e18b64aa21e`. The compiled source snapshot and tool receipt are retained in the study archive; unrelated sections and the open paper path were preserved.

<!-- hug-exposed-geometry-2026-10-15-v1 -->

## 15 October 2026 — HUG exposed-boundary diagnosis documented

HUG's subsection now records a separately frozen 624-record diagnostic on six existing grasps at original/previously selected poses. 575 nearest-part points are exposed and 49 buried; all six retain conditional proximity counterexamples (156 original,114 selected retained rows). The counts do not establish a new improvement, whole-surface agreement or practical PASS. Fifteen primary and ten independent controls pass; compiled scene geometry/transforms/filtering have no identified defect. No new eval data, inference, fits, physics or full grids.

The shared validation framework distinguishes realization of the declared Shadow pipeline from optional approximate native fidelity and empirical prediction adequacy. Historical nine strict failures remain unchanged. The new evidence manifest binds 32-case native/FK checks separately from six-case compiled geometry. All 32 are development evidence; no untouched object split is asserted. The canonical working paper remains open at its existing path and is compiled after this edit. Other provider sections remain preserved.


<!-- compiled-hand-conversion-2026-10-15-v1 -->
## 15 October 2026 — Shared compiled-hand conversion findings incorporated

The shared mapping section now records the independently verified fingertip mesh alias and serialization drift, their live-spec/asset-identity repairs, 373 saved representation checks plus three controls, and twenty passing static regressions. It distinguishes numerical target realization from native-surface or physical acceptance. Original generator subsections/results and the open paper path remain unchanged. All 11,037 pre-existing inventoried evidence/target files were preserved. No physics, inference, fitting, evaluation episodes/labels or training. [Audit and source snapshots](../research_artifacts/compiled_hand_mapping_audit_2026-10-15_v1/README.md). Desktop compilation succeeded for source SHA-256 `a04b22be3f1b65968ec8dbd2b60f4bfc39a488e4d8602b11146c30e86e10e990`; the source snapshot and confirmation are archived.

<!-- hug-mapping-process-repair-2026-10-15-v1 -->

## 15 October 2026 — HUG implementation diagnosis and two protected improvements

Updated the owned HUG subsection in the existing canonical paper to separate numerical reliability, object-aware retargeting and full surface acceptance. Added measured static-profile state/gradient repairs, missing root-rotation capability, protected-search mismatch and bounded negative fits. The final dyadic step check improves worst retained severity on saved milk-box B and mug B while preserving all original tip caps and previously passing/eligible rows. No failing row crosses into PASS; sign-off remains incomplete. No new full-grid or independent-object result is asserted. Original criteria/default/historical nine FAILs remain unchanged; no new eval data/training. [Report](HUG_MAPPING_PROCESS_REPAIR.md) and [evidence](../research_artifacts/hug_mapping_process_repair_2026-10-15_v1/). Other provider sections remain owned by their concurrent threads.


<!-- contactgen-counterexample-repair-2026-10-15-v1-final -->
### 2026-10-15 — ContactGen final static repair evidence

The completed retained cohort has **2 scoped geometric passes, 9 exact contact-preservation failures, and one original empty-source refusal** among twelve authentic native identities. The rigid-derived capsule is a separate additional contact failure. Across twelve nonempty target rows, 2 satisfy the combined scoped properties and 10 have independently verified positive-area 2 mm loss counterexamples. The historical cube is excluded. Diagnostic endpoints are not relabeled numerical acceptance candidates; cross-digit separation is `NOT_RUN` after a decisive contact failure.

The one-way property concerns the original conditioning surface and exposed boundary of the recorded robot-collider union. It does not establish reverse equality, anatomy, normals, graph-ray equivalence, physical success, or generator-wide generalization. The original capsule remains refused, and its rigid repair remains a different source lineage. The certified native surface crossings cannot be removed by target retargeting or a common invertible rigid source transform.

All ten cases in V8's separate screen have terminal outcomes. The earlier V6 stop (two contact blocks on one case, nine unrun) is preserved. Four V8 pre-fit dependency refusals caused by concurrent shared-source edits remain setup outcomes. A reviewed full sixteen-module package snapshot recovers the original byte identities for the remaining four cases and independent audits; fitting criteria, selection and budgets are unchanged. These results bind the frozen runtime, not later shared-code profiles. The shared project changes were preserved.

No new native inference, GPU action, physics step, outcome label, predictor evaluation or training was performed in this static stage. Native outputs, all failed and derived attempts, and private geometry remain retained separately from the sanitized public evidence. Final scalar evidence: `research_artifacts/contactgen_counterexample_repair_2026-10-15_v1/validation/current_cohort_scalar_summary_v6.json` (SHA-256 `84405dc9532ddc4f0f496ebab9abedc6eb3e74301a4609c083817868b06a43b6`).

The ContactGen manuscript section now records V7b, the completed V8 screen, explicit runtime recovery and the search limitation. Other paper sections are preserved by the section-hash guarded update. Compilation is recorded in the accompanying final paper receipt. Full generator validity remains unestablished.


<!-- dga-acceptance-feasibility-2026-10-15-v2 -->
## 15 October 2026 — DGA acceptance boundary resolved for two required poses

Updated only the owned DGA paper section with exact interval palm-occupancy obstructions under original guards, new unchanged-angle row 6 failures, verified distal mesh differences and the bounded band-fitting result. The paper distinguishes impossibility for two named poses from global DGA impossibility, empty-set palm passes from direction preservation, and real-arithmetic theorem scope from arbitrary floating implementations. DGA remains unaccepted; no target/criteria/domain change, physics, new native inference, evaluation data or training. [Evidence-backed report](DGA_ACCEPTANCE_FEASIBILITY_AUDIT.md). Compilation is recorded after the desktop compiler check.

DGA feasibility-audit compilation confirmed with the desktop compiler; whole-source SHA-256 `df888372e6f08bed01b376fe61a09cee2ebbff2ef909b70f0862ba404ba8ef61`. The exact source snapshot and receipt are preserved in the new study archive; unrelated manuscript sections and the open paper path remain unchanged.
