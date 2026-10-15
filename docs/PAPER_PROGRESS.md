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


## 14 October 2026 — HUG scalar-objective gap diagnosed; direction selector tested

Saved-array analysis finds all six lower-cost feasible fits worsen the fixed facing objective and invalidate49–104 direction bins. Twenty of43 saved coarse orientation witnesses were already selected fitting sites; sparse coverage alone is insufficient. A conservative optional selector now separately preserves per-part direction SSE and original-valid bins. All 182 installed surface API tests pass (36 new manufactured controls), plus9 diagnostic math/identity controls. Independently reproduced saved replay accepts 6 baseline identities and refuses 6 previous selected fits; all 30 existing tip caps still pass.

No new geometry fit, query, inference, physics, eval episode/label or training ran. These are post-hoc development diagnostics and selector checks, not corrected poses or held-out evidence. All 9 strict HUG mapping audits remain FAIL; default mapper and acceptance rule are unchanged. The HUG manuscript block records this distinction. Next is a separately frozen bounded guarded geometry study. [Detailed note](HUG_SURFACE_SELECTION_GUARD.md).

The separately authorized replacement A40 is running with the saved DGA runtime restored and CUDA/import/hash controls passed; the original is kept stopped. This infrastructure recovery creates no scientific validation result. [Connection, cost and restore record](GPU_RUNTIME_RECOVERY_2026-10-14.md).


## 14 October 2026 — HUG guarded search completed; fixed-anchor claim refuted

The HUG manuscript subsection now distinguishes actual guarded fitting from saved selector replay. All six fits fall back to exact original complete scene poses; all-six independent source/FK and 30 tip-cap checks pass. The nine original surface reports are reused explicitly (zero new full-grid measurements), retaining strict 0 PASS/9 FAIL and pointwise orientation 7 FAIL/2 UNRESOLVED. New driver and independent consistency controls are implementation/provenance evidence, not surface sign-off.

Separately frozen local/full-wrist covers and an all-SO(3) radial bound establish a conditional fixed-anchor palm obstruction for saved Mug B. Its existing positive-area cell is native within 4.746 mm at the 5 mm band and at least 6.646 mm from any rotated Shadow palm after allowances. Independent box/FK and high-precision arithmetic reviews pass. The paper retains the fixed-anchor/literal-surface/engineering-arithmetic limits and does not claim free-placement impossibility, physical contact, failure-label preservation or universal validation.

Production and frozen scientific criteria remain unchanged; no new eval data or training. A new source-only anatomical calibration is being prepared/reviewed, with no result yet asserted. [Evidence and next mapping work](HUG_GUARDED_FIT_AND_REACHABILITY.md).


## 15 October 2026 — HUG anatomical calibration and new-anchor proof completed

The HUG subsection now records the separately frozen canonical MCP-frame experiment: one source-only 8.668706 mm anchor calibration; all 6 numerical/mean-tip passes but 21/30 individual old caps refused, so 0/6 eligible. Independent source/C/FK/world-target reconstruction confirms the result. The same old positive-area cells refute both new Mug fixed anchors under arbitrary palm orientation; independent 90-digit replay passes 1,093 checks. No full-grid audit is falsely claimed for refused poses, no native failure label is inferred, and arbitrary other placements remain outside the proof.

Both runtime aborts, unchanged calibration bytes and all original results are preserved. Current strict native preservation is not accepted. The working title/main multi-generator prediction question remain unchanged; the target embodiment/failure-label meaning has been requested before another dependent mapping policy. No new native HUG inference, physics, eval data or training. [Completed calibration evidence](HUG_CANONICAL_PALM_CALIBRATION.md).


<!-- hug-free-palm-2026-10-15-v1 -->

## 15 October 2026 — HUG palm-placement study completed, mapping still unvalidated

User chose common Shadow and authorized per-grasp translation with unchanged criteria. Both fitting stages (762 + 348 full callbacks) retain all six original guarded outputs. Independent tip-feasible diagnostic poses receive nine fresh strict surface FAIL results. A separate seed-origin coverage correction for three nanometer-shift selections adds five fresh FAIL results, preserving the first protocol. All six intentional-movement alternatives preserve original fingertip caps; actual palm shifts span 0.031372–4.891691 mm. Fourteen measurements concern nine robot poses from six existing native grasps, not fourteen grasps or physical failures.

The HUG section and abstract distinguish numerical admission, guard refusal and strict surface failure. The frozen criteria and production mapper are unchanged. Counterexample-constrained fitting is proposed only; continuous orientation coverage still requires evidence even after sampled failures are removed. No new native inference, physics, evaluation data or predictor training. See [study](HUG_FREE_PALM_MAPPING_STUDY.md) and its separately versioned evidence bundles.

The desktop LaTeX compiler reports success for both the canonical working paper (SHA-256 `906455622b1fe0bcb396a685341fafbfe1e39d70f464d0b67dc1f42881b52401`) and the exact selectively staged paper (`5bcf7db84de2a9d5ff0c86beed21e7324e99eeb4c13d7cf04c239167f310e092`). Other providers' uncommitted paper sections are preserved.


<!-- hug-direct-surface-constraints-2026-10-15-v1 -->

## 15 October 2026 — HUG direct-constraint results incorporated

Updated the existing HUG subsection and abstract with the completed two-version direct surface-constraint study. All six selected poses preserve original tip/mean gates, but all nine fresh full-grid comparisons FAIL strict preservation; pointwise orientation is 9 FAIL. The manuscript distinguishes Mug A's retained-row progress from increased thumb mismatches/new direction reversals, exact fingertip refusals from mean/domain acceptance, and certificate uncertainty from genuine mismatch. Search/callback budgets and the disclosed thread-count deviation remain explicit. No production promotion, new inference, physics, eval data or training. Other generator sections and the active paper path are preserved. See [completed study](HUG_SURFACE_CONSTRAINT_MAPPING_STUDY.md).

<!-- hug-criteria-alignment-restoration-2026-10-15-v1 -->

## 15 October 2026 — HUG evaluation scope clarified and restoration result recorded

The existing HUG subsection and abstract now distinguish our strict native-relative preservation contract from HUG's published generator benchmark. The 10 mm guard, original nonworsening caps and proximity bands are explicitly project choices. Practical adequacy for the executed Shadow prediction target requires a separately justified error budget and independent validation; none is claimed here.

The HUG subsection also records all twelve saved-endpoint fingertip restorations, zero eligible surface improvements, six unchanged reference poses, and the distinction between invalid direction evidence and actual reversals. The historical nine FAIL results remain prior measurements; no fresh grids or evaluation episodes ran. [Criteria explanation](HUG_CRITERIA_ALIGNMENT.md) links the review and bounded study. No other provider section was replaced.

The canonical open paper compiled successfully with the built-in desktop compiler at SHA-256 `750cf9d599afdb525d110375e1d323c7cabcc4e1721ccd0940a08c30f342423e`; HUG section SHA `6ae011b3e3af2b175b89f86fb74b93cd1b16265d53f6e7120666318938f11d4e`, abstract SHA `bd5a6b8ac68d4f57c5450e2290b07d2ad5e110ad91c8e9bf08760f2c2f1cb44d`. The user's existing editor and document path are preserved. No alternate document was compiled or opened.

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

<!-- dga-acceptance-feasibility-2026-10-15-v2 -->
## 15 October 2026 — DGA acceptance boundary resolved for two required poses

Updated only the owned DGA paper section with exact interval palm-occupancy obstructions under original guards, new unchanged-angle row 6 failures, verified distal mesh differences and the bounded band-fitting result. The paper distinguishes impossibility for two named poses from global DGA impossibility, empty-set palm passes from direction preservation, and real-arithmetic theorem scope from arbitrary floating implementations. DGA remains unaccepted; no target/criteria/domain change, physics, new native inference, evaluation data or training. [Evidence-backed report](DGA_ACCEPTANCE_FEASIBILITY_AUDIT.md). Compilation is recorded after the desktop compiler check.

DGA feasibility-audit compilation confirmed with the desktop compiler; whole-source SHA-256 `df888372e6f08bed01b376fe61a09cee2ebbff2ef909b70f0862ba404ba8ef61`. The exact source snapshot and receipt are preserved in the new study archive; unrelated manuscript sections and the open paper path remain unchanged.

## 2026-10-15 — HUG current simulator-input signoff and preserved native failures

- Accepted scope: all 32 saved nominal HUG-derived Shadow commands pass the current complete static scene contract, including native frame conversion, independent source FK, literal source limits, hand/object geometry and declared controls. No dynamics or fresh-retargeter signoff.
- Repaired source-tensor precision after a preserved24/32 first result; corrected rerun32/32 under unchanged numerical gates. Applied static repository tests:23 pass,1 optional skip,2 physics tests excluded.
- Feasible-step fitting and exact eligibility guidance improve the two existing difficult candidates with original fingertip caps intact. Current retained witnesses still show definite native proximity/direction changes; neither a general replacement nor a strict native-equivalence pass is claimed.
- Diagnosed the old strict checker's missing positive-certification capability;25 manufactured identity controls now establish a narrowly supported positive route without changing old failures.
- HUG subsection and abstract updated in the canonical paper. Evidence/report: [HUG_MAPPING_ACCEPTANCE_REPAIR.md](HUG_MAPPING_ACCEPTANCE_REPAIR.md), `research_artifacts/hug_mapping_acceptance_repair_2026-10-15_v1/`. No new eval data, proposals, physics or training.


<!-- dga-practical-static-2026-10-15-v1 -->
## 15 October 2026 — Practical DGA static contract completed on saved outputs

The existing DGA manuscript section now distinguishes historical failed native preservation from the newly authorized practical Shadow pipeline. All 20 saved MultiDex outputs are accounted for: 13 native-valid static mappings, seven explicit native-invalid refusals, one independently admitted thumb correction and 22 unresolved branch proposals falling back to the fixed legal reference. The positive bound concerns each finger's worst uncapped distance residual over the complete cube surface, conditional on the declared arithmetic allowance; it does not prove pointwise contact preservation. Maximum landmark error remains 6.075600 mm, known palm overlap remains, and physical/native-fidelity acceptance is not asserted.

Status is `STATIC_PILOT_ADMITTED_TESTED_SCOPE` for these previously inspected cube outputs, with execution disabled. No new native inference, physics, evaluation episodes/labels or training. Historical protocols/results and other manuscript sections remain preserved. [Protocol and evidence](DGA_PRACTICAL_RETARGETING_PROTOCOL.md). The canonical paper is compiled after this edit; the source snapshot and compiler receipt bind the exact working file.

DGA practical-study compilation confirmed with the desktop compiler at whole-source SHA-256 `8d55b452eb31e1f978a27933d7deb2dc8075eb4127e777d287bcbe2d32da3c54`. The exact working-source snapshot and tool receipt are archived; unrelated sections and the open paper path were preserved.


<!-- hug-native-mano-comparison-2026-10-15-v1 -->
## 15 October 2026 — Native MANO static reference added to the HUG subsection

The existing canonical paper now distinguishes native skeleton import from collision approximation and cross-embodiment retargeting. On all 32 saved proposals, mean tip discrepancy is 0.453 mm native versus 6.428 mm Shadow; all 192 anatomical part/case surface means improve (balanced aggregate 1.463 versus 4.727 mm). Remaining proximity changes, including additional native 5 mm flags, prevent a complete contact-preservation claim. The paper records fixed-source numerical checks, 33 manufactured controls, scalar-only summary recovery, finite-sampling limits and the generator/hand confound. No new eval data, physics, inference, fits or training. Historical criteria/results and other provider sections remain preserved. [Report and evidence](HUG_NATIVE_MANO_COMPARISON.md). Desktop compilation is checked after the edit.

Desktop compilation succeeded for the canonical working source SHA-256 `e856dbb09e6b9e0f0b5157d3bd70aa44a9505d461f22535492223877f7e198a8`; the guarded HUG subsection and compiler receipt are archived with this comparison.


<!-- shared-object-bounds-repair-2026-10-15-v1 -->
## 15 October 2026 — Shared numerical loader repair recorded

The common mapping subsection now reports the equivalent bounds-contraction repair, 8/10 to 10/10 existing-object preparation and <=1.39e-17 m scalar-arithmetic disagreement. This remains an implementation reliability result, separate from anatomical/contact/physics validity. Forty-seven static tests pass; one optional asset test is skipped and two free-fall tests excluded. No new eval data. [Evidence](../research_artifacts/shared_object_bounds_repair_2026-10-15_v1/README.md).

The canonical working paper compiled successfully at source SHA-256 `67979b5ecf37d0f2b870af003f4fcd8e72d490834cea404c4eba4874d51fb9bf`; the compiler and guarded section receipts are archived.


<!-- hug-only-research-focus-2026-10-15-v1 -->
## 15 October 2026 — HUG-only continuation handoff prepared

The user selected HUG as the sole active generator for the next phase. The new handoff records accepted 32-command static realization, unresolved fresh-solve/native-fidelity scope, mapper/provenance and recorder/export gaps, historical prediction results, and conditional gates for later evaluation and model work. Other generators and their results remain preserved and deferred. A front-of-manuscript scope note marks the earlier multiple-generator plan as deferred; no historical findings or provider sections are rewritten. No new inference, physics, evaluation data or training ran. [Handoff](HUG_ONLY_MAPPING_AND_PREDICTION_HANDOFF.md). Desktop compilation is recorded separately in the handoff receipt.


<!-- hug-only-native-reference-clarification-2026-10-15 -->
## 15 October 2026 — HUG-only handoff and scope note reconciled

Clarified the handoff and its marked manuscript scope note: HUG-only is settled; Shadow remains the current execution default, while the latest native MANO static candidate remains in the fidelity investigation. The next bounded saved-cohort check and conditional later execution/data/model gates are explicit. No final-hand promotion or mapping pass is asserted. The historical multiple-generator manuscript and results remain preserved behind the prospective scope note. No experiments or new eval data.

The clarified scope note compiled successfully in the existing paper at SHA-256 `84fc5e1a640986b72427a11687c74b8c707c607c99d0702784dc0f47c885f353`. The handoff and compiler hashes are saved in `research_artifacts/hug_only_handoff_2026-10-15_v1/clarification_receipt.json`.
