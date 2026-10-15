# How the research plan changed

We are studying whether information available during a grasp can warn that the
attempt will fail if execution continues. The current implementation predicts
eventual grasp, lift and hold outcomes from recorded hand closing. Useful warning
time, performance on physical hardware and improved recovery remain questions
to test.

This record began on 11 October 2026. Earlier entries reconstruct the development
history from the existing evidence notes; they are not a claim that all choices
were preregistered. Each entry separates an implemented decision from a proposed
experiment. Later changes should add a dated entry explaining the observation,
the choice it changed, alternatives considered and what would change it again.
Preserve earlier reasoning and results rather than rewriting the project as
though the final plan was known from the start.

## 9 to 10 October 2026 Choosing the first experiment

The initial idea combined learning to grasp with predicting and recovering from
failure. That made it difficult to tell whether a failed attempt reflected a
poorly trained controller, a grasp hypothesis or the disturbance being studied.

**Implemented decision:** use HUG grasp proposals and a fixed controller first.
Record the resulting attempts and train the outcome predictor separately. The
larger policy learning and recovery ideas remain a research agenda.

Evidence: [the original plan changes](archive/PAPER_PROGRESS_HISTORY_2026-10-12.md#changes-to-the-original-plan)
and [research foundations](archive/RESEARCH_FOUNDATIONS.md).

## 10 October 2026 Making noise and labels precise

The word noise initially covered several possible changes. A reproducible
experiment needed one disturbance whose magnitude, application and actual
effect could be inspected.

**Implemented decision:** draw bounded offsets for 22 finger joint targets and
hold them constant during closing, lift and hold. Do not perturb the two wrist
targets. Pair random directions across amplitudes and save requested offsets,
clipped commands and measured motion separately.

**Implemented decision:** label final task success using the frozen grasp, lift
and hold rule. Keep setup rejection, runtime error, failed acquisition and
sampled height loss distinct. A failed task does not automatically mean a slip.

Evidence: [noise collection](NOISY_ROLLOUT_COLLECTION.md) and
[the multi scene collection](MULTI_SCENE_COLLECTION.md).

## 11 October 2026 Establishing a simple baseline

Many related trials come from the same observation and proposal. A random frame
split could make evaluation look stronger by exposing closely related data in
training and test. We also needed to check whether the robot's measured response
added information beyond the commands we issued.

**Implemented decisions:** split by observation families, retain each trajectory
and its related variants together, and compare measured state plus commands
against commands alone and training prevalence. Start with logistic regression
before choosing a larger architecture.

The full model's failure AUROC was 0.8523 on the held out apple observation,
compared with 0.6419 for commands alone. It nevertheless falsely alerted on
22 of 58 successful attempts. This motivates testing reliability, rather than
treating ranking performance as sufficient. The split was selected after the
collection was inspected and remains a retrospective development holdout.

Evidence: [grouped splits](DATA_SPLITS.md) and
[the trained baseline](PREDICTION_BASELINE.md).

## 11 October 2026 Testing unfamiliar objects before changing the model

Enoch raised the concern that a model trained on cube and apple attempts might
perform poorly on unfamiliar data. Adjusting the model first would hide the
answer to that question.

**Implemented decision:** freeze weights, normalization, prediction time and
threshold before testing fresh scenes and new milk carton and mug geometries.
The 2,080 attempt evaluation revealed substantial object dependent errors.
All 520 mug attempts were approach collisions, occurring before the 2.96 s
model input cutoff. Those attempts cannot establish advance warning before a
future collision or drop.

Evidence: [the frozen fresh evaluation](FRESH_OBJECT_EVALUATION.md). These
records remain unchanged even though later controller development followed.

## 11 October 2026 Investigating the mug instead of immediately retraining

Repeated mug failure could have been mistaken for a limitation of the predictor
or a need for more varied training examples. Physics replay identified a
different cause: the Shadow forearm intersected the table before noise began.

**Implemented decision:** correct execution feasibility before collecting more
data. Adjust the wrist and carrier while preserving the intended world palm and
finger targets. Add a 1 mm runtime table margin checked every 2 ms, and version
the corrected controller separately. A stopped attempt receives no binary label.

The small final regression completed 26 trials without table contact and stopped
one disturbed closure before contact. This validates the correction for those
checks; it does not improve or evaluate the prediction model. Historical v2 and
corrected v3 results must remain distinct.

Evidence: [the diagnosis and correction](MUG_TABLE_CLEARANCE.md),
[the independent replay audit](../research_artifacts/mug_table_clearance_v1/final_dynamic_replay_audit.json)
and [PR 17](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/17).

## 11 October 2026 Considering public data and existing detectors

The user questioned whether generating every trajectory ourselves was necessary.
Public data can reduce that work. NVIDIA's [DROID release on Hugging Face](https://huggingface.co/datasets/nvidia/Cosmos3-DROID)
provides successful and failed real robot trajectories. Its Franka arm and
gripper have different state channels from the Shadow Hand, and task failures
need inspection before being treated as grasp failures.

**Proposed experiment, not implemented:** inspect a small public grasp subset,
check sensing and outcome meaning, and build a separate compatible baseline if
the subset supports the question. No public data have been imported or mixed
into the frozen Shadow dataset, and no decision to replace the simulator has
been made. The simulator remains useful for controlled tests of our controller.

[Guardian's model card](https://huggingface.co/paulpacaud/guardian-thinking)
documents execution verification using images before and after a subtask, as
well as plan checking. Its execution verification results do not establish
advance warning before a future grasp failure.

**Proposed comparison, not implemented:** use Guardian as an outcome verification
reference. Comparing its after action images with our earlier state history
would compare different tasks and information. A forecasting experiment with
Guardian would need a separately specified prompt, causal image prefix and
evaluation; its existing verification score cannot be reused as that result.

## 11 October 2026 Narrowing what we are trying to prove

Prediction of robot failure already has relevant prior work, including
[SAFE](https://vla-safe.github.io/) and
[FIPER](https://tum-lsy.github.io/fiper_website/). The fact that our target occurs
earlier than Guardian's execution check does not by itself establish novelty.
We have not established a literature gap or an advantage over those methods.

**Proposed first research question:** does measured hand response before lifting
provide useful warnings of eventual grasp failure on unfamiliar objects, beyond
the commands alone, at a controlled rate of false alarms?

The current model processes saved closing history through 2.96 s and predicts
the eventual task outcome. This is an offline causal outcome baseline. Some
failures can already have happened by its cutoff. It does not estimate when a
slip begins or establish that a physical intervention still has time to work.

The next claims need separate evidence:

1. **Information:** compare state plus commands with commands alone on the same
   attempts and cutoff. Improvement would show that measured response carries
   predictive information in those conditions.
2. **Generalization:** evaluate frozen models on untouched object and scene
   families under the declared execution version. Report rejections, missed
   failures and false alarms separately. Lower errors would support transfer
   only to the tested conditions.
3. **Warning time:** define and validate a future failure event, including which
   attempts are still viable at the cutoff. Select the threshold and alarm
   budget using validation, then measure advance warning and false alarms on
   test trajectories. Include inference and stopping delays before calling an
   alert actionable. Missing an acquisition deadline is not a slip onset label.
4. **Intervention, later:** compare matched continuation and bounded correction
   attempts to determine whether responding to a warning actually helps. This
   requires an intervention experiment; a probability score alone cannot prove it.

These are hypotheses and planned tests, not completed contributions. Adding a
larger neural network, more noise types or more trajectories would be justified
by a demonstrated limitation, rather than used as a substitute for these tests.

## 11 October 2026 Simplifying the working branch

The milestone workflow created small reviewable PRs, but its stacked branches
made it hard for the user to identify the current project. Some demo and
literature files also remained outside the latest experiment branch.

**Implemented source decision:** consolidate the histories and unique files on
`codex/project-work`. Preserve historical designs and the optional container
runtime with clear status notes. Keep scientific v2 and v3 records distinct;
this is source organization, not new model training or a changed experiment.
The integrated suite passed 373 tests with 32 optional skips, and the frozen
source hashes and four published evidence bundles are unchanged.

**Ongoing workflow:** use one working branch and one consolidated PR. Create
another persistent branch only at the user's explicit request. Save original
tips as archive tags and a verified Git bundle before retiring their pointers.
The [branch guide](WORKING_BRANCH.md) and [archive](BRANCH_ARCHIVE.json) provide
the working location and recovery references. The open paper keeps its path.

## 12 October 2026 Publishing the consolidated snapshot

The user authorized merging PR 19 into `main` so the partner can read the
simulation, model, evidence and paper in one shared location. This publication
does not train a model or change an experimental protocol.

The Linux container image built successfully, but its public smoke harness
requested fresh inference by default without mounted model assets. A local
reproduction identified `--inference requires --hug-root and --checkpoint`.
The harness now explicitly uses the synthetic saved grasp with `--no-inference
--no-video` and exposes child stdout/stderr on failure. The corrected command
completed locally and produced all four required artifacts. The Linux rerun
is the next check before merging; public CI does not validate licensed HUG
inference or GPU rendering. Frozen simulation and model sources are unchanged.

The next research milestone remains collecting valid corrected mug attempts
and evaluating the fixed predictor by object. A better training score alone
would not establish transfer, warning time or recovery benefit.

## 12 October 2026 Walking through one corrected mug prediction

The next step was to inspect one feasible grasp before expanding data collection
or changing the model. We selected the first saved mug proposal before seeing
its new score and ran it with the v3 correction and zero finger angle noise.
This is a reused development scene, not an untouched generalization example.

**Measured result:** the hand lifted the mug 14.6077 cm and met the two second
sampled hold rule. The task label is 1. Independent replay checked 3,260 physics
steps with zero hand table contacts and reproduced every saved state bit for bit.
The original predictor, using only the closing prefix through 2.96 seconds,
assigned a failure score of 0.9999999785 and raised a false alarm. The model,
normalization, threshold and prediction source hashes are unchanged.

**Evidence behind our interpretation:** eight wrist features are more than five
training standard deviations from the original training means. Wrist inputs
account for 80.0% of absolute linear contributions to this score. This supports
investigating input distribution shift after the controller correction. It does
not establish physical causation, aggregate performance or a benefit from
removing wrist features. No feature masking, fitting or tuning was performed.

**Next decision remains proposed:** declare a small corrected controller batch
with familiar controls and fresh scenes, then measure the frozen model's false
alarms and missed failures. Review candidate representations using development
data before freezing a separate evaluation. We drafted a request for Irmak's
advice; no message or expert recommendation is claimed here.

Evidence: [the walkthrough, causal input, replay and score explanation](../research_artifacts/mug_single_walkthrough_v1/README.md).
Full raw recordings remain in the local run directory recorded in its manifest.

## 12 October 2026 Separating command noise from data diversity

The user asked whether adding more mug noise would address transfer and what
would make this a research contribution. Our first collection varied finger
commands within cube and apple scenes. That supplied command variation, not
independent coverage of object geometry, scene conditions or safe wrist poses.
The corrected zero noise mug false alarm motivates testing those factors
separately before expanding the collection or changing the model.

Primary sources reviewed on this date inform the recommendation:

1. [The Feeling of Success, CoRL 2017](https://proceedings.mlr.press/v78/calandra17a/calandra17a.pdf)
   already predicts grasp outcomes before lifting, varies grasp position,
   orientation and force, and evaluates with objects separated between training
   and test. This supports deliberate grasp diversity and object grouping;
   prelift prediction alone is not our novelty.
2. [SAFE, NeurIPS 2025](https://arxiv.org/html/2506.09937v2) tests runtime VLA
   failure detection on held out tasks and compares alarm accuracy and timing.
   It motivates measuring false alarms and useful timing in addition to ranking.
3. [FIPER, NeurIPS 2025](https://arxiv.org/html/2510.09459v2) explicitly tests
   unfamiliar successful executions alongside actual failures. It uses generative
   policy observation embeddings and action uncertainty, so its implementation
   is not directly compatible with our deterministic controller. Its calibration
   assumptions do not guarantee performance after our controller change.
4. [Isaac Lab noise documentation](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html#action-and-observation-noise)
   separates command and measurement noise, including fixed episode biases and
   perstep disturbances. Its [transfer guidance](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/transfer_policies_between_physx_and_newton.html#introducing-domain-randomization)
   recommends plausible physics uncertainty and a deterministic nominal
   evaluation. These are representative engineering practices, not a universal
   recipe or evidence that our current disturbances match hardware uncertainty.

**Proposed next experiment, not implemented:** declare a small v3 diagnostic
batch with familiar object controls, multiple mug grasps and fresh scenes.
Compare zero noise with the existing bounded finger command bias levels, while
keeping model weights, input cutoff, task labels and alert threshold unchanged.
Record commanded versus applied offsets, guard rejections and objectwise false
alarms and missed failures. Group related proposal and scene families together.
Do not treat previously inspected mug or milk results as an untouched test set.

Object and safe wrist diversity are separate axes. Testing multiple compensated
wrist solutions for the same intended palm and finger grasp would require a
planned executor extension and its invariance and motion checks. It is not
equivalent to enabling random wrist offsets in the current finger noise path.
Introduce actuator, sensor or friction variation later when a specific
uncertainty question justifies it. Current synthetic stress is not calibrated
hardware noise, and physical grasp outcomes must be measured rather than
assumed invariant under nominal geometry preservation.

**Provisional research direction:** evaluate whether a lightweight hand state
monitor can retain useful outcome discrimination and fewer false alarms across
safe wrist configurations and unfamiliar objects. Matched execution controls
and representation comparisons could make a specific dexterous hand study.
This is a hypothesis, not an established literature gap or completed novelty.
It needs repeated evidence and comparisons on fresh object and pose groups.
Warning lead time, intervention benefit and real robot transfer require separate
tests. We should choose the question now and let the experiments refine it,
rather than claim that finishing a collection pipeline establishes a contribution.

## 12 October 2026 Reconciling the external context brief

The user supplied an external robotics context brief compiled without access to
this repository. We treated its handoff suggestions as background, rather than
as instructions to reset the project or as evidence of completed experiments.
Repository records resolve several of its unknowns: the Shadow Hand and MuJoCo
setup, state and command features, task labels, grouped splits, trained baseline
and frozen fresh object evaluation already exist. Slip timing, real correction
data and recovery remain unresolved.

The category holdout proposal is useful when the held factor and test groups
are declared explicitly. Our existing pilot already prevents frame and proposal
variant leakage, but both original objects occur in training. A new object
holdout using those data would require a separate fit and split, not relabeling
the existing coefficients. Poor held out performance identifies a failing test
condition; it does not isolate a cause when geometry, mass or controller
conditions change together. Acquisition failures and sparse task scored drops
do not establish physical slip onset or warning time.

**Smallest recommended diagnostic, not yet scored:** reuse the 27 corrected
controller checks already saved. All 26 completed cases have valid closing
prefixes and the original model's joint ordering: 23 mug attempts and three
familiar controls, totaling 22 successes and four acquisition failures. Compare
frozen predictions with labels by proposal and noise amplitude. Report the
single guard rejection separately with a null label; it has no complete prefix.
This requires no new simulation or training and is a development diagnostic,
not an untouched object generalization benchmark. A fresh evaluation can be
declared after this check and any development decisions.

The brief's [Tobin et al. reference](https://arxiv.org/abs/1703.06907) concerns
randomized rendered images for visual localization. It does not establish the
right joint command disturbance for our state based predictor. Its
[Chebotar et al. reference](https://arxiv.org/abs/1810.05687) adapts simulation
randomization using real robot rollouts; no such real calibration is present
here. These sources support investigating specific uncertainty models, not
assuming that arbitrarily wide randomization covers unseen contact physics.

## 12 October 2026 Scoring the 26 completed corrected controller checks

**Previous plan:** the context reconciliation proposed scoring all existing
corrected checks before collecting more data or changing the model. The user
authorized this diagnostic and asked whether mug false alarms repeat across
noise levels, with the stopped attempt kept separate.

**Implemented evidence:** all 26 completed recordings were scored with the
original model, normalization, closing prefix and 0.5 failure threshold.
Predictions used only measured joint positions, velocities and applied targets
through 2.96 seconds. Every completed mug was flagged: false alarms on all
20 successes and detection of all three acquisition failures. False alarms
were 7/7 successes at zero noise, 8/8 at 0.15 rad and 5/5 at 0.60 rad. Recall
and AUROC are undefined in the all successful 0.15 rad group. The guard rejection
has null label, score and alert and is excluded from outcome metrics.

The same seed2 scene002 proposal failed with zero noise and succeeded at both
noisy settings, while the model flagged all three. Other proposals failed only
at the largest bound. This confirms why physical outcomes, rather than noise
presence or amplitude, supply the binary labels. Across all 26 cases, the cube
control was correctly cleared, the apple failure detected and the successful
milk carton falsely flagged. No new motion or fit was generated.

**Interpretation and choice:** the successful mug false alarm repeats under
all three command noise conditions at the current threshold. Adding this same
noise does not establish transfer to corrected mug grasps. The earlier wrist
attribution remains a candidate explanation rather than an isolated cause;
these scores do not justify declaring a feature removal fix. Keep this evidence
as a retrospective development diagnostic, separately from historical v2 and
any future evaluation declared after development decisions.

**Verification and artifact:** the [scoring record](../research_artifacts/corrected_mug_model_diagnostic_v1/README.md)
preserves all 26 observable prefixes and scores, grouped results, separate
rejection and figure. Independent calculation reproduced probabilities within
1.11e-16, verified metrics and the null rejection, and checked all 238 raw source
files and four frozen model/report/prediction source hashes were unchanged.

**Next check, not implemented:** compare input distributions and safe wrist
coverage with matched development controls. If a representation or training
change is justified, evaluate that comparison on freshly declared object and
pose groups. The current task did not retrain, remove wrist features, tune a
threshold or launch a larger collection. Useful warning timing, recovery and
physical transfer remain separate questions.

## 12 October 2026 Explaining wrist inputs before changing the model

**Previous plan:** after all completed mugs triggered alerts, we proposed
examining input distributions rather than immediately widening noise or
retraining. The user requested the wrist comparison first and asked for a
plain language explanation and a possible conference target within one week.

**Implemented evidence:** reconstructed original normalization exactly from
the 520 training rows, excluding validation and test. Both last wrist command
angles are outside training ranges in all 20 successful mugs. Last measured
WRJ2 angles are outside in 19/20 and WRJ1 in 17/20. Each successful mug has
four to eight active wrist statistics outside their training ranges. Combined
wrist contributions are negative for the success score in all 20. Absolute
wrist contribution fraction has median 75.5%, exceeds half in 17/20, and ranges
from 31.0% to 86.9%. This describes the original linear score, not physical
causation or an improved model. The [diagnostic record](../research_artifacts/mug_wrist_input_diagnostic_v1/README.md)
contains exact contributions, training ranges and provenance. No model change,
hypothetical modified prediction, new motion or collection was performed.
Independent calculation verified all 26 episode rows, 624 wrist feature rows
and eight source hashes. Normalization matched exactly; grouped contributions
matched within 3.55e-15. The audit is included with the diagnostic bundle.

**Interpretation and proposed next check:** wrist input extrapolation is now
supported across the successful mugs, rather than only one example. Finger
noise variation did not supply this wrist coverage. Wrist terms do not dominate
every case, and geometry, proposals and controller conditions differ together.
We should design a matched comparison of safe postures for the same intended
grasp, with invariance and execution checks, before choosing a feature or fit
change. A later fresh evaluation must remain separate from these development
records. This comparison is proposed, not implemented.

**Conference candidate, not a submission decision:** official sources checked
12 October identify [CoRL 2026 Sim2Real2Field](https://sim2real2field-corl26.github.io/)
as an [accepted conference workshop](https://2026.corl.org/program/workshops).
Its non archival Late Breaking Track accepts up to four pages excluding
references and explicitly welcomes preliminary findings and negative results.
The deadline is **15 October 2026, 11:59 PM Central Time**, three days from this
check, rather than a full week. A narrow simulation input/controller shift
diagnosis might fit its transfer failure theme; this is our interpretation,
not confirmation from organizers or evidence of achieved physical transfer.
Every author needs an OpenReview profile and one author must be willing to
review at most two papers. The site warns that new profiles without an
institutional email can take up to two weeks to approve. No account was created,
organizer contacted, venue commitment made or paper submitted here.

The requested one week goal ends 19 October. A short preliminary report is a
plausible writing scope; neither scientific readiness for a full main conference
paper nor acceptance is established. Choosing a deadline should constrain
scope and claims rather than require an unsupported contribution. The current
working manuscript is preserved pending an explicit venue and revision choice.

## 12 October 2026 Selecting the workshop and an understandable scope

**Previous plan:** Sim2Real2Field was a candidate; wrist extrapolation had been
measured before changing the model or collecting more recordings. The user
selected its Late Breaking Track, requested submission steps and public
examples, and asked to understand each piece before accelerating development.

**Choice:** keep this turn focused on exploration and explanation. Start with
one stored grasp and the difference between physical outcome and model alert.
The original apple development test caught 187/202 failures with 22/58 success
false warnings; the corrected mug diagnostic caught 3/3 with 20/20 success
false warnings. Preserve their different scopes. Catching every mug failure
while warning on every success is not useful discrimination at the saved
threshold. It is a measured limitation, not proof of a successful new method.

**Public examples:** one [author repository](https://github.com/YTomar79/sim2real_budget)
reports a workshop submission of a controlled pendulum measurement versus
randomization study; its older arXiv version names an IROS workshop. Treat that
report separately from an officially accepted paper. [SAFE](https://arxiv.org/html/2506.09937v2)
explains why unfamiliar behavior can succeed and why evaluation needs successful
as well as failed executions. Neither establishes novelty for our wrist study
or an acceptance cutoff. The selected venue hides current submissions and
does not list a complete contributed paper roster. Public social searches do
not resolve that visibility limit.

**Submission preparation:** checked the call, linked template, OpenReview form
and account guidance. The website cutoff converts to 16 October 12:59 AM
Eastern, one hour earlier than the invitation timestamp. Plan an upload for
15 October daytime, using the earlier cutoff. The [preparation record](archive/PAPER_PROGRESS_HISTORY_2026-10-12.md#workshop-preparation-and-first-reading-12-october)
records format, profiles, review commitment, anonymity interpretation, sources
and a staged outline. The workshop's welcome for preliminary and negative
results supports a narrow diagnostic report; it does not guarantee selection
or justify unsupported physical transfer claims.

**Next check:** explain a recorded attempt with the user, agree on the precise
claim and comparison, then prepare the existing manuscript in the required
format and review it with both authors. A posture experiment remains proposed,
not run. No model, threshold or recording was changed; no rollout, account
creation, external message or paper submission was performed. Preserve the
manuscript while the user establishes the question and understands its evidence.

## 12 October 2026 Checking whether recovery supplies a new contribution

**Question:** the user asked whether recognizing genuine failures and correcting
them makes the original prediction goal distinctive, relevant to the field and
appropriate for the selected workshop.

**New evidence:** [Gilday et al. 2023](https://doi.org/10.1002/aisy.202200390)
combine grasp outcome prediction and physical wrist translation recovery.
They also discuss poor predictions on unfamiliar objects with different
successful grasp modes. [Nazari et al. 2022](https://arxiv.org/html/2209.06019)
predict future slip and proactively adapt trajectories, with reactive and no
control comparisons. These are measured interventions, unlike interpreting
SAFE's motivation to stop or ask for help as evaluated grasp recovery.

**Interpretation and choice:** prediction plus recovery remains a useful goal,
but its combination and the need to avoid nuisance warnings are established
research. Our distinct hand and joint measurements do not by themselves prove
novelty. The workshop's explicit interest in monitoring transfer failures
supports a scoped reliability study; suitability does not establish acceptance.
Do not append a recovery feature merely to make the abstract sound new.

**Missing evidence and next check:** separate object changes from safe posture
changes with matched execution, retaining the current model. Then any proposed
warning improvement needs a separate frozen fresh evaluation. Future warning
claims need validated event timing; recovery claims need an intervention
comparison that also measures harm to successful grasps and added time.
The existing table clearance correction fixes the executor and is not evidence
of model triggered recovery. The [progress note](archive/PAPER_PROGRESS_HISTORY_2026-10-12.md#prediction-and-recovery-prior-work-12-october)
records the staged plan and primary papers. This turn reviewed literature and
documented proposals; it did not collect data, train a model, implement recovery
or alter the manuscript. Preserve that scope in the imminent workshop draft.

## 12 October 2026 Exploring documented limitations before choosing the next fit

**Question:** the user asked whether a wrist specific focus was too narrow,
what established gaps fit the workshop, and whether to keep improving the
implementation while exploring or let exploration guide the next experiment.

**New evidence:** [SAFECAST](https://arxiv.org/html/2608.04246v1) already addresses
deployment shift with training and calibration perturbations. Its stated limits
include shifts outside the collected families, rollout cost and the absence
of restored formal guarantees under arbitrary shifts. It reports that pooling
more perturbations generally does not improve detection over focused sets,
with model/metric exceptions. [FIPER](https://arxiv.org/html/2510.09459v2) already
addresses benign unfamiliarity and discusses remaining alarm accuracy and
calibration limitations. [RAPT](https://arxiv.org/abs/2602.01515) rules out treating
efficient joint summary monitoring itself as a new broad concept.

**Interpretation:** reliability after training/deployment mismatch is a
documented active problem with partial solutions. It is not an unoccupied
topic. A dexterous grasp experiment needs a specific comparison and a measured
finding before claiming a distinct contribution. The full explanatory sentence
need not become the paper title; wrist variation can be one concrete test of
a broader question about reliable warnings or useful simulation diversity.

**Proposed workflow:** bound exploration to a reading and decision pass ending
in one question, comparison and improvement criterion. Continue checking and
explaining existing records; retain the frozen baseline. Avoid simultaneously
retraining, widening collection and adding recovery while the question remains
undecided. Candidate directions are simulation diversity at equal budgets,
warning reliability across execution changes and trustworthy decisions under
unfamiliarity. The [progress record](archive/PAPER_PROGRESS_HISTORY_2026-10-12.md#candidate-directions-and-a-bounded-exploration-phase-12-october)
states each candidate's evidence and implementation requirements. Current
diagnostics can support a scoped preliminary workshop account; broader
augmentation claims require new experiments. The user has not chosen a new
direction. This turn changed documentation only, without a fit or rollout.

## 12 October 2026 Declaring the chosen question and first comparison

**User decision:** the user chose simulation diversity as the broader question,
with the wrist comparison first, and asked to define the experiment and success
criterion before model changes or more data collection.

**Declared question:** at matched training data counts and success/failure
balance, does successful safe wrist posture coverage reduce false warnings
on fresh conditions more than additional finger command noise? This is a
testable proposal, not an established advantage or novelty claim.

**First diagnostic:** retain the original model, normalization, 2.96 second
closing cutoff and 0.5 alert. Use the same intended mug grasps at zero noise
with two geometrically feasible wrist settings and compensated carrier poses.
Declare the eight existing proposal families, up to 16 future executions.
Setting B comes from the recorded corrected configuration. Seek A among
original training wrist commands when feasible, otherwise a distinct feasible
candidate closest to that distribution, without consulting predictions or new
outcomes. Exact angles and an audited explicit selector remain pending.

**Interpretation:** nominal palm/finger geometry must be preserved, but actual
tracking and outcomes must still be measured. Both successful pairs with
different warnings support sensitivity in the tested setup. Outcome changing
pairs are not harmless posture changes. Infeasible pairs and runtime stops
remain in the inventory with null labels. Two unfamiliar safe settings do not
establish an in range versus out of range comparison. This is development
diagnosis, not improved model performance or an untouched test benchmark.

**Improvement criterion for the later model comparison:** fewer success false
warnings and no additional missed failures on the same fresh held out test
groups, at the same threshold, with equal training budgets and matched label
proportions. Report counts and grouped uncertainty. No failures means recall
is undefined, and very few cannot establish broad reliability. Do not optimize
against the test or call lower scores alone an improvement.

The [proposed protocol](../research_artifacts/wrist_comparison_plan_v1/protocol.json)
binds original source hashes and planned case identities. The [plain explanation](archive/PAPER_PROGRESS_HISTORY_2026-10-12.md#selected-question-and-proposed-wrist-comparison-12-october)
records completed work and direct next steps. This turn wrote the plan only:
no selector code, feasibility search, simulation, inference or fit was performed.
Keep the current manuscript and all existing scientific records unchanged.

## 12 October 2026 Accepting the paper question and evidence standard

**Previous plan:** the first six steps produced a collection pipeline and an
initial task outcome model. Later frozen evaluations exposed a mug collision,
then repeated false warnings after the executor correction. We explored
simulation diversity, posture sensitivity and alarm trust as possible directions,
then declared a wrist diagnostic before more fitting or collection.

**User decision:** use the working title **Evaluating Simulation Diversity for
Grasp Failure Warnings** and the exact question: **For a simulated dexterous
hand, does training with more diverse successful wrist configurations improve
prediction of grasp, lift and hold outcomes on unfamiliar objects and wrist
configurations more than additional finger-command noise, when both approaches
use the same amount of training data, the same success/failure proportions and
the same model architecture?** Update the paper and evidence notes around this
question, challenge weak comparisons and explain each next step before scaling.
The user accepts that novelty and workshop acceptance cannot be promised.

**Evidence motivating the choice:** the unchanged model warned on all 23
completed corrected mugs, including 20 successful grasps. It detected the three
failed acquisitions; the guard stopped attempt has no label or prediction.
The [scoring artifact](../research_artifacts/corrected_mug_model_diagnostic_v1/README.md)
shows the false alarms at zero, 0.15 and 0.60 rad requested finger noise bounds.
The [wrist audit](../research_artifacts/mug_wrist_input_diagnostic_v1/README.md)
finds both last wrist commands outside their original training ranges for all
20 successful mugs and negative net wrist contributions to their success logits.
This is evidence of shifted inputs in the saved arithmetic, not isolated
physical causation or proof of a training remedy.

**Chosen sequence:** first conduct static pair feasibility and add audited
explicit wrist selection. Preserve the same intended world palm/finger grasp
with compensated carrier placement, and declare exact settings before any
new scores or outcomes. Then perform the already declared zero noise paired
diagnostic with frozen weights, normalization, cutoff and threshold. The
diagnostic must include infeasible cases, changed outcomes and null label stops.
Two safe but unfamiliar settings cannot support a familiar versus unfamiliar
claim. Different alerts on both successful grasps indicate sensitivity in the
tested setup, not an improved model or broad causal explanation.

**Later comparison:** retain the same base data and compare equally sized
additional finger noise and successful wrist coverage cohorts. Match class
proportions and base object/proposal identities, architecture, input fields
and validation selection. Fit normalization from training data only. Freeze
both models before scoring the same fresh held out groups. The operational
improvement criterion is fewer successful false warnings with no additional
missed failures at the common 0.5 threshold. Report absolute counts, group
breakdowns and uncertainty; this is not a formal noninferiority guarantee,
especially with few observed failures. Lower scores alone do not meet it.
Declare cohort selection before fitting and retain all generated, selected and
excluded counts. Match the selected training budget without implying equal
collection cost. Score the original frozen model on the common fresh test as
a reference; its smaller training budget does not isolate the diversity effect.

**Challenge gates:** require feasible and comparable execution, an identifiable
comparison, matched data budgets and family/class composition, a causal input
cutoff, genuine failure accounting, unbiased fresh evaluation and accurate prior
work comparisons. Publish useful negative results and limitations rather than
adding model complexity to imply novelty. Physical slip timing, actionable
warning lead time, model triggered recovery and real robot transfer remain
separate untested claims.

**Implementation status:** the existing manuscript and notes are updated
around this question and completed diagnostic evidence. The manuscript compiled
successfully with the desktop editor compiler. The question and protocol
remain proposals about future training benefits. No new model fit, threshold
change or simulation result is introduced by this documentation revision.
The [protocol](../research_artifacts/wrist_comparison_plan_v1/protocol.json)
and [progress record](archive/PAPER_PROGRESS_HISTORY_2026-10-12.md#accepted-paper-question-and-evidence-gates-12-october)
define the next evidence required. Source remains on `codex/project-work`;
PR 19 is merged into `main`, but later working branch changes are not merged
by documenting them.

**Partner and workshop check:** this sequence responds to Enoch's stated wish
to test unfamiliar conditions before assuming a better model is necessary.
His agreement remains unknown. The Sim2Real2Field Late-Breaking Track fits the
question's simulation diversity/monitoring theme but does not establish novelty
or acceptance. A maximum four page CoRL formatted anonymous submission, verified
references and claims, author review and active author OpenReview profiles
remain pending. Use the earlier official site cutoff, 15 October at 11:59 PM
CDT, while preparing a daytime submission. The next technical check is whether
eligible safe wrist pairs exist without changing the intended grasp.

## 12 October 2026 Static geometry blocks the observed familiar wrist pairs

**Previous plan:** seek setting A first among last wrist command pairs actually
observed in the original training partition, while preserving each intended
mug grasp. Compare it with the existing corrected setting B only when both
meet the declared geometry gates.

**New evidence:** a read only kinematic check tested the eight unique command
pairs from 520 original training episodes across the eight declared mug
proposals. None of the 64 combinations met the nominal 10 mm table clearance
gate. Their clearances ranged from approximately −65.211 mm to +0.187 mm.
All eight stored corrected B settings revalidated at approximately 10.001 to
10.042 mm. Physics stepping was blocked; there was no new rollout or model
prediction.

**Interpretation and choice:** the observed training wrist vectors cannot be
used immediately as a safe familiar setting for these eight intended grasps.
This finite search does not prove every continuous pose inside the training
command ranges infeasible, or certify dynamic safety for the corrected poses.
Do not force unsafe poses or relabel two unfamiliar settings as familiar versus
unfamiliar. The full equal budget diversity question is unchanged, but the
immediate diagnostic needs a valid familiar setting or another controlled case;
otherwise it must be explicitly described as sensitivity between two safe
unfamiliar wrist setups.

**Status and next check:** no exact A/B feasibility manifest is frozen and no
paired execution occurred. Review the clearance result, then complete geometry
selection and explicit executor support before collecting new attempts. Keep
the original model, data and prior controller results unchanged.
The [static feasibility bundle](../research_artifacts/wrist_pose_feasibility_v1/README.md)
records all candidate checks, geometry invariance, source hashes and reproduction.

## 12 October 2026 First explicit safe wrist pair, with the model unchanged

**Previous plan:** no actually observed training wrist pair cleared the nominal
gate for the declared mug proposals. We needed a safe comparison with exact
selection before executing more attempts.

**Pre-execution choice:** freeze all eight mug A/B pairs using only geometry
and original training wrist-command distance. Use a deterministic bounded grid
and refinement search; require at least 0.02 rad Euclidean A/B separation so
the second setting is not merely a numerical copy. Keep B from the existing
corrected recording. A is the nearest eligible evaluated candidate, not a
globally optimal continuous pose. The separation gate was fixed before new
predictions or outcomes. All eight pairs qualify geometrically, but both
settings remain outside training ranges; A is nearer training in only three
cases. The first eligible case in the declared order is executed once per
posture; the seven other pairs remain pending review.

**Implementation:** add the opt-in `fixed_grasp_lift_v4_frozen_wrist_pair`
executor, preserving historical v2/v3 code and outcomes. It validates frozen
hashes, exact wrist settings, unchanged intended palm/finger geometry, zero
noise, task parameters and the existing runtime guard. It rejects an unsafe
requested wrist before motion if planning would substitute a different setting.
Twenty-one focused tests pass. Static selection reproduces byte-for-byte and
passes 15,654 independent checks. See
[the selection bundle](../research_artifacts/wrist_pair_selection_v1/README.md).

**Measured result:** `mug_scene002_seed0_noise0.00` A and B both succeed,
lifting 14.6208 and 14.6077 cm and satisfying the 2 s HOLD requirement. Both
trigger false warnings: failure scores 0.9999999871415051 and
0.9999999784968061, with success logits −18.169261 and −17.655064. The
B-minus-A logit change is +0.514197, comprising +0.529098 wrist and −0.014901
nonwrist contributions. No stops occur. A is slightly closer to training in
the declared command distance but has the higher failure score. That distance
is not the learned decision rule.

**Independent checks:** whitelisted inputs, training normalization and manual
scores reproduce. All 22 applied finger command histories match exactly.
Nominal palm/finger paths agree within 1.67e−16 m. Actual maximum paired palm
and finger geometry differences are 0.05890 and 0.35723 mm; identical contact
dynamics are not claimed. Replay reproduces every saved state across 6,520
physics steps without table contact. B reproduces prior v3 motion, apart from
signed-zero representation in disabled-noise arrays. The
[paired result bundle](../research_artifacts/wrist_pair_diagnostic_v1/README.md)
preserves reports, inputs, tracking, hashes and replay media. Full new recordings
remain in `runs/wrist_pair_diagnostic_2026-10-12/first_pair` in the original
runtime checkout.

**Audit implementation correction:** the parent froze the auditor hash before
its checks were finished. Later geometry/tracking checks and the already-declared
1e−12 command reproduction tolerance finalized the auditor. The original plan
and score report remain intact. An explicit auditor-only amendment records both
hashes, and the exact historical script bytes are archived. Model, executor,
scorer, poses, predictor inputs and recordings are unchanged. Do not claim that
every source is unchanged without disclosing this exception.

**Interpretation and next check:** both successful grasps still cause warnings;
wrist terms change scores without clearing alerts. Saturated probabilities
hide the logit change. One development pair does not establish a physical
cause, improvement, failure recall, useful warning time or generalization.
Review this example, then check the seven remaining frozen pairs for repeatability
before a training remedy. The equal-budget training comparison remains
unexecuted. Keep bound plans immutable and update current status in notes and
new evidence bundles instead of rewriting historical declarations.

## 12 October 2026 Completing the seven remaining frozen wrist comparisons

**Previous plan and authorization:** after the first pair succeeded physically
but produced two false warnings, the user requested the remaining seven pairs
with zero noise and no model changes. Retain the same exact pose manifest,
v4 executor, model, normalizer, input cutoff, task scorer and 0.5 threshold.

**Pre-execution choice:** freeze the ordered seven-pair inventory and all
finished execution/model source hashes before any new physics step. Run A then
B once in every case, without replacements or outcome-based selection. Preserve
the first pair separately. Bind completed scoring and audit implementations
before inference rather than including unfinished analysis code in the
execution declaration. The old audit amendment remains a historical first-pair
record; this extension requires no new code exception.

**Measured result:** all 14 new attempts complete without guard stops or hand/table
contacts. There are 11 successes and three scorer-defined failed acquisitions;
the model warns on all 14. Thus 11 successes are false alarms and the three
task failures are detected. Five new pairs succeed at both settings and warn
at both, one pair fails at both, and one fails in A but succeeds in B. Including
the preserved first pair gives 16 attempts, 13 successful false alarms and
three detected failures, with six both-success pairs. See
[the extension evidence](../research_artifacts/wrist_pair_remaining_v1/README.md).

**New label evidence:** the changed-outcome scene 003 seed 2 pair peaks at
13.996398 cm in A and 14.020459 cm in B. A is just 0.036021 mm below the fixed
14 cm cutoff; both satisfy qualifying contact and the sampled 2 s HOLD rule.
The peak difference is 0.240608 mm. Preserve labels 0/1, but describe a small
task-cutoff crossing rather than a slip, drop or dramatic physical recovery.
The other two failures peak at approximately 13.04 and 13.02 cm. All three
failures derive from two proposal families.

**Interpretation and challenge:** false warnings repeat across the declared
both-success pairs. At the fixed decision threshold the predictor behaves like
an always-warn rule here; 3/3 detections do not establish useful reliability.
Descriptive AUROC remains 0.9091 on the new batch and 0.8974 on all 16, so do
not claim that the scores contain no information. Wrist changes affect scores
and actual contact dynamics; both settings remain outside training ranges.
These related development cases do not establish causal wrist harm, general
object transfer, a training benefit or actionable slip timing.

**Verification and storage:** manual probabilities match within 1.39e−17;
original training normalization reproduces bitwise. Replay reproduces all
recorded states over 45,640 new physics substeps without hand/table contact.
Zero noise, exact wrist targets, identical paired finger histories and nominal
palm paths are verified. All seven B motions reproduce their older v3
references apart from signed-zero noise representation. A separate scoring
rerun reproduces its report, predictions, dataset and pre-outcome record
byte-for-byte. Model, bound sources, raw recordings and first-pair artifacts
remain unchanged. New raw data are in
`runs/wrist_pair_diagnostic_2026-10-12/remaining_pairs`; portable inputs and
audit/figure records are in `research_artifacts/wrist_pair_remaining_v1`.

**Next decision:** the declared diagnostic is complete. Before fitting the
matched wrist-coverage versus finger-noise cohorts, declare equal selected
training budgets, class mix, base identities, validation and a common fresh
holdout. A separate validation-fitted calibration reference would test whether
coverage training improves beyond simple score calibration. Do not select a
mapping or threshold from these eight diagnostic pairs, change labels to clear
errors, or alter the frozen result. This is a proposed comparison addition,
not a completed calibration or training experiment. Paper, current evidence
index and handoff now report the full diagnostic while historical bound plans
and first-pair notes keep their original bytes.

## 12 October 2026 Prioritizing understandable training coverage

**User priority:** understand the purpose of the experiment and address the
limited variety in training before expanding the project. The practical goal
remains useful grasp outcome warnings in unfamiliar conditions. The wrist versus
finger noise question is a proposed way to measure which additional examples
help; the existing diagnostic has not demonstrated a training remedy. More
diversity improving performance is not, by itself, a new research contribution.

**Evidence and interpretation:** the original training contains 520 episodes
from cube and apple proposals with narrow wrist command ranges. Successful mug
commands fall outside those ranges, and their wrist feature contributions move
scores toward failure. This makes wrist coverage a supported candidate for the
next training experiment, rather than proof that insufficient episode count is
the cause. See the [input audit](../research_artifacts/mug_wrist_input_diagnostic_v1/README.md)
and [completed wrist diagnostic](../research_artifacts/wrist_pair_remaining_v1/README.md).
Zero noise was a temporary diagnostic control; controlled finger noise remains
one source of outcome variety in the proposed training comparison.

**Recommended next collection, not executed:** begin with a small, separately
declared simulation batch using the same Shadow Hand, observable joint fields
and task rule. Check safe wrist coverage and fresh grasp/scene proposals first,
retain both successes and failures, and add bounded finger perturbations in
declared conditions. Predeclare the common untouched test families and separate
validation data before collection or fitting. Keep the training augmentations
at matched selected counts, class mix and base object/proposal identities so
the eventual comparison can attribute improvement. Broad object, material and
mass changes should be staged as separate axes, with implementation and physics
checks, rather than changed together and treated as wrist-only evidence.

The current explicit v4 executor is bound to the eight inspected diagnostic
pairs and rejects nonzero noise. It is not a general training collector. A new
versioned collection protocol and wrapper are needed before those additional
trials; preserve v4, its manifest and all frozen model/evaluation artifacts.
Current mug cases remain development evidence and cannot become an untouched
final test after training or choosing settings from them.

**Outside data:** DROID is relevant to diverse manipulation, but its
[documented platform](https://droid-dataset.github.io/) uses a Franka Panda arm.
Our unchanged predictor expects Shadow Hand joint histories, so DROID joint
trajectories are not direct replacement training inputs. Any outside source
must be checked for compatible robot measurements, action definitions, timing
and successful/failed grasp labels. Using different robot data could require a
new feature representation and an additional transfer question; it is optional
future work, not a prerequisite for the controlled simulation comparison.

**Status:** this records the user's priority and a recommended next scope. No
new trials, training, calibration, threshold or label changes were made in this
discussion. The accepted formal question and completed results are preserved.

## 12 October 2026 Broad failure goal and a focused first experiment

The user agrees to simulation collection first and asks whether wrist comparisons
are too narrow given the multiple ways a grasp can fail. The recommendation is
to retain broad grasp outcome prediction as the goal and use wrist coverage as
the first controlled experiment. This discussion does not replace the accepted
formal question or declare a demonstrated wrist training benefit.

Wrists became the focus because successful corrected mugs repeatedly triggered
warnings and unfamiliar wrist features pushed the model's score toward failure.
The immediate problem is recognizing successful unfamiliar configurations, not
an established claim that wrist movement is the physical failure mechanism.
Both low false alarms on successes and detection of failures matter. The zero
noise comparison diagnosed repeatability; it did not establish causation.

Broader collection should be staged: geometry and grasp placement, safe wrist
coverage, and separately declared finger perturbations first; material/mass and
later disturbances require additional implementation and a defined observable
question. Preserve outcome labels and record verified failure-event metadata
for evaluation, without feeding hidden simulator causes into the predictor.
The original pilot is dominated by failed acquisitions and has only four
scorer-defined drops, so comprehensive slip prediction is not demonstrated.

The current model uses joint positions, velocities and commands before lifting.
A cue absent from these inputs cannot be supplied merely by generating more
examples. Contact sensing is a possible future input, supported by the
[vision/touch grasp outcome study](https://arxiv.org/abs/1710.05512) and
[tactile contact-event study](https://arxiv.org/abs/1910.03973); neither result
establishes a required sensor or a performance guarantee for our system.

**Status:** rationale recorded, simulation-first preference confirmed. Broader
failure coverage remains a staged proposal. No collection, model change,
calibration or claim of predicting every failure mode was made in this turn.

## 12 October 2026 Grounding the direction in published predictor evidence

**User request:** review strong public prediction systems and their training
practices before assuming wrist coverage is important or collecting more data.
The user asks for a credible baseline and evidence about other causes of failure.
Simulation-first collection remains the expressed preference.

**Completed review:** the [public evidence guide](archive/GRASP_PREDICTION_EVIDENCE_REVIEW.md)
records 21 systems, papers and company announcements spanning grasp scoring,
contact/slip detection, runtime task monitors, post-action verification and
policy value models. It separates prediction metrics from executed task success,
pins paper versions where available, and marks preprints and commercial claims.
Public descriptions do not independently certify production reliability.

**Interpretation:** there is no demonstrated universal recipe across those
different tasks and sensor interfaces. Recurring practices are an explicit
target/time, informative measurements, representative data, controlled
comparisons, held-out groups and separate calibration. Targeted coverage is an
established method; no reviewed source identifies wrist diversity as the unique
remedy for our joint-only predictor. Our original model remains a measurable
pilot baseline, with limited retrospective evaluation and poor mug warnings.

**Revised next gate, not an executed experiment:** before selecting larger
collection, inspect whether the label and causal inputs support the desired
prediction, and declare appropriate calibration and feature baselines on
grouped development data. A wrist-feature ablation or command-tracking feature
comparison would require retraining under a new protocol, not editing frozen
weights or pretending an altered score is the original model. Reserve fresh
evaluation before fitting. Choose collection axes from the remaining evidence.
The draft wrist-versus-noise comparison is now a candidate under review; no
broader hypothesis or new research contribution has been adopted as a result.

**Documentation and status:** the guide, manuscript related work, progress index
and partner handoff reflect this gate. Original model, normalization, threshold,
labels, protocol artifacts and recordings remain unchanged. No new simulation,
training, calibration, recovery or real-world evaluation was performed.

## 12 October 2026 Seeking deployment relevance and understanding VLAs

The user values the concrete decisions supported by Amazon's deployment study
and the touch-sensing comparison, and is open to other data or simulators.
They propose exploring a VLA-based grasp-failure predictor and ask about
workshop significance. This reopens scope selection; it does not authorize
describing a proposed integration as a completed result.

The [VLA direction review](archive/VLA_DIRECTION_REVIEW.md) distinguishes a controller,
an image/text model and a separate failure monitor. Our fixed HUG executor is
not a VLA. Adopting one is not a prerequisite for simulation collection.
Current work already combines VLA representations, varied rollouts and
calibration; merely combining those components would not establish novelty.
Failure timing and grasp-specific labels also need explicit evidence.

Public SAFE source and archive previews were inspected without downloading data.
The expected formats include saved policy features that might support an offline
monitor baseline. Final task outcomes are not automatically grasp outcomes;
archive content, reuse terms and a suitable runtime still need checking.
The preferred next exploration is a small relevant recording/label audit
before building a new controller, followed by a compatible baseline.

The target remains a non-archival CoRL workshop welcoming preliminary and
negative findings, not a journal. The official call was rechecked. No universal
acceptance threshold or accepted contribution is inferred from the papers.
No simulation, controller, predictor, calibration or frozen evidence changed.

## 12 October 2026 Prioritizing a relevant VLA-monitoring research area

The user asks for a direct judgment about relevance before committing to a
research direction. Recent primary monitoring papers and company research
support VLM/VLA failure monitoring as an active area. The answer is yes:
prioritize grasp-specific monitoring alongside an existing VLA policy.
This selects an area of investigation, not a final hypothesis or a claim that
all robotics failures are now handled by one industry-standard architecture.
Public announcements do not establish startup formation or adoption counts.

The evidence review now includes 24 records, adding action-only and temporal
monitoring alternatives. The next gate is real-recording suitability, a
compatible baseline, and a taxonomy of missed grasp failures. Existing work
already addresses feature fusion, timestamp supervision, phase-aware checks
and varied rollouts, so a distinctive contribution requires a closer
comparison. The previous wrist-versus-noise study remains a preserved candidate,
not an assumed primary experiment.

No data archive, training run, VLA integration, model change or new experiment
was performed. The existing manuscript remains a record of completed pilot
evidence while a precise contribution is selected. The separate context-trust
and multimodal agent-evaluation idea was transferred to a new projectless chat
at the user's request; it is not part of the grasp study.

## 12 October 2026 Starting from SAFE real recordings and preserving our simulator

Previously, public loader formats suggested an offline VLA monitoring route,
but archive contents and monitor availability were unverified. The user now
asks to inspect a successful/failed grasp before selecting a contribution and
to consider reusing the existing simulator and wrist analysis.

We inventoried SAFE's public WidowX ZIP and extracted one same-task pair under
a selection rule fixed before viewing. Source episode 3 succeeds; episode 0
fails. Sampled video frames show the carrot carried to a plate versus an empty
raised gripper. Six extracted members passed ZIP CRC checks and have saved
SHA256 hashes. Passive pickle inspection confirmed metadata and hidden-state
records without executing deserialization. The [source audit](../research_artifacts/safe_recording_audit_v1/README.md)
stores provenance, format checks and qualitative observations; downloaded raw
files stay local and outside our training dataset.

No published trained SAFE checkpoint, calibration band or raw-pair score trace
was found in the inspected release. Thus we did not measure its misses on the
raw pair. A separate author-provided Franka demo eventually alerts during
repeated unsuccessful acquisitions. The sampled green-to-red transition
raises a timing question but is neither an undetected task failure nor a
validated physical detection delay. An empty acquisition can be retried;
event-level labels must remain distinct from eventual task outcome.

Choice: first verify alignment and inspect a small declared development
inventory, then reproduce a compatible baseline and check repeated errors.
The provisional objective is recognizing unsuccessful acquisitions sooner
without unnecessary warnings on successful attempts. Temporal boundaries,
visual task progress and benign novelty already have related work; no distinct
contribution is established by this example. Wrist variation remains a possible
benign stress test, not a proven SAFE weakness. Rotation commands in the new
CSVs are not measured wrist angles. Our fixed Shadow executor supplies no VLA
hidden states, so its trajectories and weights cannot be treated as SAFE input.

The [walkthrough](archive/SAFE_RECORDING_WALKTHROUGH.md) maps reuse of provenance,
recording, splits, replay and controlled simulation tests. Model fitting,
calibration changes, new simulation collection, interventions and sim-to-real
transfer remain unexecuted. Original scientific artifacts are unchanged.
The manuscript adds only a clearly separated qualitative audit paragraph;
its quantitative results remain the original simulation results.

## 12 October 2026 Retaining HUG while clarifying the visual-monitor route

The user favors retaining HUG and asks to understand the VLM/VLA distinction,
how a predictor would be trained and what HUG contributes before proceeding.
The previous VLA-monitoring discussion left two interfaces open: policy latent
features or a common visual monitor. The preferred route for exploration now
keeps the HUG-conditioned fixed executor and considers a visual monitor that
could also be evaluated on real VLA-controlled recordings. This does not train
or integrate a VLA. A VLM could verify images and task text; a smaller image
encoder with a learned classifier is another candidate.

The [learning guide](archive/HUG_VISUAL_MONITOR_LEARNING_GUIDE.md) and
[teaching artifact](../research_artifacts/hug_visual_monitor_teaching_v1/README.md)
bind an existing same-proposal cube success/failure pair and preserve its exact
source hashes. The failed case has contact and some lift, so it must not be
presented as an empty-hand event or slip-onset ground truth. The guide separates
predicting a future task outcome from detecting an already unsuccessful
acquisition. Real task success and our lift/hold rule cannot silently become
one target. Replay outcome/score overlays and future frames remain outside
model inputs.

HUG's sampling randomness creates candidate pose diversity; our original
finger-target bias is applied later during execution. These are distinct
sources of variety. HUG's published limitations include open-loop contact/lift
execution and retargeting issues; this motivates monitoring but does not prove
an untouched gap. The recent Guardian VLM verifier already uses simulated and
real data, and its published missed-plate example is a qualitative case, not
proof that a general empty-grasp weakness dominates. Any data-strategy claim
needs controlled, size-matched comparisons and reserved real evaluation.

No model fitting, new rollout, calibration, label revision or new controller
was performed. The original predictor remains unchanged. The next scientific
choice is the visual monitor's target and decision time; our learning artifact
does not adopt a new paper title or claim visual transfer. The working paper
adds only the explicitly proposed route and relevant source context.

## 12 October 2026 Returning to the HUG setup and testing collection coverage

The user explicitly chooses to continue the existing HUG simulation and
prediction setup, deferring VLM/VLA work. Earlier exploration remains recorded
but is not the active plan. The working question now compares broader HUG
observation/proposal coverage with more finger-noise repetitions at equal
training episode counts. Wrist coverage can be a subgroup of that strategy,
not a presumed sole cause of prediction failure.

The practical motivation is a pre-lift warning that catches unsuccessful tasks
without unnecessarily rejecting successful unfamiliar grasps. The original
model has outcome signal, but repeated corrected mug warnings expose unreliable
operating behavior. The 1,040 pilot trials contain only four observation and
sixteen proposal families. This supports checking coverage, while calibration,
feature encoding and label boundaries remain alternative explanations.

Close non-VLA prior includes See to Reach, Feel to Grasp's learned proprioceptive
control/readiness score. Thus hand-local command errors and pre-lift prediction
are not inherently new. Our candidate contribution is a controlled HUG
collection-strategy finding. A bounded search also found ContactGuard's
pre-contact world-model monitor, which is a different interface; no global
absence of prior HUG monitoring was established.

The [current plan](HUG_MONITOR_RESEARCH_PLAN.md) requires a development comparison
of calibration and representations before larger collection, followed by
matched cohorts and one fresh family-disjoint test under a corrected executor.
Keep the target as final grasp/lift/hold outcome from the 2.96-second prefix.
The user-facing goal of useful real-robot warnings remains an ambition; the
current joint model needs compatible robot measurements for real validation.

The manuscript question and plans now reflect this scope; completed results
and all original scientific artifacts are unchanged. No new training,
calibration, rollout or visual/VLA integration occurred. Collection quotas,
source IDs and exact baseline formulas must be declared before execution.

## Keeping this record current

For each material change, record the date, what we thought previously, the new
observation or source, the choice made, its implementation status, the supporting
artifact and the next check. Mark unsuccessful experiments and rejected inputs
explicitly. Distinguish measured results from interpretations and proposals.
Link the paper to the evidence used for each claim. This convention is the
ongoing documentation requirement.

## 12 October 2026 Curating documentation after research exploration

**Previous presentation:** the README, paper progress and partner handoff mixed
operational instructions and measured results with superseded research
questions, VLM/VLA teaching, workshop browsing and the original long-term
recovery program. The README contained both wrist-only and broader HUG coverage
questions.

**Evidence reviewed:** all 91 commits reachable through `6790b2d`, including
the 24 after the main snapshot `c95dd8d`, were reviewed for their implementation,
evidence, exploration or workflow roles. No source or experiment commit was
reverted. The [commit review](archive/COMMIT_REVIEW_2026-10-12.md) records each
disposition.

**Choice and implementation:** keep the README focused on the implemented
pipeline, reproduction and essential documentation. Archive earlier designs,
surveys and educational routes; retain the full previous progress record and
branch history. Current results, partner instructions and submission tasks
now have focused documents. Remove deferred visual-monitor ideas and the
qualitative SAFE recording walkthrough from the manuscript; retain literature
that directly informs the measured HUG outcome study. Archive moves require
relative-link maintenance, and two original guide paths remain forwarding
notices for immutable evidence bundles.

**Scientific status:** this is documentation curation, with no code, model,
data, label or controller change. The recorded question and quantitative
results are preserved. The user reports a partner-agreed thesis/testing plan;
its exact wording was requested rather than inferred. The larger coverage
comparison remains unexecuted in the current saved evidence. Any later agreed
protocol should be reconciled with this record and supporting artifacts.

## 13 October 2026 — Implement the evidence-backed translation boundary first

- **Previous plan:** Add DexGraspNet saved-record replay after hardening the
  provider boundary and geometry inputs.
- **New evidence:** The repository contains the pinned right-Shadow URDF, the
  existing tested HUG/Dex Retargeting conversion, and project-authored saved
  Shadow fixtures. It does not contain a licensed DexGraspNet record, an exact
  official saved-record contract, an audited object-scale binding, or an
  authoritative mapping from the released source hand to this evaluator's
  pinned Shadow asset.
- **Choice:** Implement the strict reusable boundary and deterministic geometry
  inputs, register the existing HUG strategy and exact same-asset Shadow mapping,
  and fail closed for unsupported source models. Register DexGraspNet as disabled
  `dataset_replay` with the missing evidence stated explicitly. Do not invent a
  transform or describe the project-authored synthetic fixture as DexGraspNet.
- **Implementation status:** Implemented on `codex/multi-provider-translation`.
  Provider classification and license status appear in resolved results;
  pre-rollout rejections carry a null task label; seeded object-frame point
  clouds and manifests are content-addressed. The environment, controller,
  scorer, and execution protocol were not changed.
- **Evidence:** Focused translation/provider/object tests pass, including exact
  named-joint mapping, changed-evidence rejection, deterministic point bytes,
  manifest sensitivity, disabled-provider metadata, and no-outcome rejection
  semantics. The full-suite result is recorded with the implementation commit.
- **Next check:** Obtain an authorized redistributable or externally mounted
  DexGraspNet record plus its exact official schema, source asset, coordinate and
  scale semantics, then add a provider-specific decoder and run the visual and
  physics acceptance gate before enabling it.

## 12 October 2026 Adding an interchangeable grasp-proposal boundary

The active evaluator previously coupled fresh inference, HUG pickle loading,
MANO conversion, Dex Retargeting and MuJoCo execution in one HUG-specific
command. The user wants to compare grasp generators while keeping the registered
Shadow Hand environment, fixed controller and outcome rule unchanged.

The implementation adds a strict `grasp_proposal_v1` JSON boundary. A proposal
binds a provider/configuration, object geometry, generation seed, exact named
Shadow joint targets and `T_object_palm`, where the `palm` link is expressed in
the registered MuJoCo object body frame. It also records separate geometry,
observation and generation-request groups. Portable execution validates the
canonical artifact and its content hash; provider-native outputs and checkpoints
remain optional deep-audit inputs so a cloud-generated proposal can run without
the generation environment.

Generation and execution are separate. Built-in HUG and project-authored saved
record adapters, an external-command contract and an HTTP JSON contract all
produce the same artifact. New `eval_case_v2` files execute through a generic
one-case or batch command. Historical `eval_case_v1` HUG commands remain
available and are not rewritten. Provider identity is provenance and analysis
metadata and is not an input to the existing failure predictor.

This is an infrastructure result, not evidence that any grasp model is better.
A deterministic saved proposal completed the common MuJoCo command and was
scored as `approach_collision`; that failure only verifies end-to-end dispatch.
A fresh simulator observation was then processed by the pinned HUG checkpoint
on CPU and executed successfully through the generic runner. Replaying the same
native HUG proposal through the legacy runner produced the same successful
outcome and matching phase/contact arrays. Maximum numeric trajectory deviation
was 5.1e-13 m for palm position, 8.2e-12 rad for hand position and 3.5e-9 N for
contact force, consistent with transform serialization roundoff. These ignored
runtime smoke artifacts are not research data and are not committed.

The next scientific check is paired HUG and non-HUG proposals on identical
objects and initial conditions, followed by grouped, held-out outcome analysis.

## 12 October 2026 — Use one versioned command for proposal-to-evaluation experiments

- **Previous plan:** Users invoked proposal generation, case materialization,
  batch execution, and the result viewer as separate commands. That exposed
  internal paths and made it easy to mix proposals, hashes, and output roots.
- **New evidence:** The interchangeable-provider implementation already has
  strict request, proposal, case, and result boundaries, so orchestration can
  be added without changing HUG, the MuJoCo environment, or the scorer.
- **Choice:** Add `grasp-eval` with `validate`, `run`, `status`, `resume`, and
  `view` subcommands over `grasp_eval_experiment_v1`. One experiment file maps
  provider requests and proposal sample indexes to controlled physical cases.
- **Implementation status:** Implemented on `codex/project-work`. Completed
  cases are immutable during resume; incomplete stage directories are retained
  before retry. Existing low-level commands remain available for debugging.
- **Next check:** Run a small multi-provider experiment and confirm that resume,
  videos, and the viewer behave correctly before adding dataset-wide
  Parquet/DuckDB export.
# DexGraspNet replay uses a measured kinematic mapping

- **Decision:** Enable the pinned official DexGraspNet sample as
  `dataset_replay`, with licensed inputs kept under ignored local `data/`.
- **Evidence:** Official commit `bd1b13d7248729af117e1d46aaa6266b147a3c7b`
  documents metric translation, static-XYZ Euler rotation, 22 named Shadow
  joints, and object scale. FK comparison against the canonical `dex-urdf`
  Shadow hand measured 0.429 mm mean and 1.722 mm maximum fingertip residual on
  the first record.
- **Boundary:** The source has a free palm while the canonical model has two
  wrist joints. Replay maps the source palm pose to `T_object_palm`, sets the
  canonical wrist joints to zero, and applies only the registered name/sign map.
- **Outcome:** Ten official mug records produced seven approach collisions, one
  failed acquisition, and two unlabeled joint-limit rejections. No proposal was
  clipped or repositioned.
- **Next check:** Run fresh DexGrasp Anything sampling on an Ubuntu/NVIDIA worker
  and feed its untouched native outputs through the same translation and FK
  gates.


## 13 October 2026 Integrating the partner's merged provider infrastructure

**Evidence:** Enoch merged PRs [21](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/21),
[22](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/22),
[23](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/23) and
[24](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/24) into main
through `b503c2f`. All four partner branch tips are ancestors of that snapshot.
The provider contract, unified CLI, geometry preparation, translation checks
and pinned DexGraspNet replay are implemented. Fresh DexGrasp Anything inference
remains a GPU-worker handoff rather than a completed integration.

**Source integration:** merge that main snapshot into the existing
`codex/project-work` branch, preserving the curated README and both decision
histories. The only merge conflicts are documentation. Partner implementation
files are incorporated without modifying them; existing models, datasets,
labels and corrected v3/v4 diagnostics retain their original provenance.

**Execution evidence and next gate:** the ten-record DexGraspNet gate contains
seven approach collisions, one failed acquisition and two null-label joint-limit
rejections, with no successes. Its free-space grasps use the original v2 tabletop
executor, so this is not an evaluation of the corrected mug protocol. Generic
provider output currently omits the per-frame velocities and command targets
required by the existing predictor exporter. Inspect tabletop suitability and
define a versioned compatible recorder/export path before treating new provider
rollouts as prediction data. No new model fit or research collection is part of
this integration.

**Integration verification:** the combined suite passes 416 tests with 33
skipped. Initial collection failed in the lightweight local environment because
Pydantic and Pinocchio were absent; missing dependencies were provisioned under
`/private/tmp` without modifying the recorded project environment. The passing
test runtime used Python 3.12.13, NumPy 2.2.6, SciPy 1.17.1, MuJoCo 3.3.7,
Pydantic 2.14.1, Dex Retargeting 0.5.0 and Pinocchio 4.1.0. It differs from
historical experiment runtimes and is an integration check, not a replay of
the research results. The suite reports 105 numerical warnings in existing
table-clearance and scene-calibration matrix operations; they remain disclosed.
The CLI help command succeeds, 376 local documentation links resolve, and
original model/pilot/corrected-mug/wrist artifacts and the manuscript have no
changes relative to the pre-integration branch.

## 13 October 2026 Running one HUG-derived cube through the shared evaluator

One declared cube case ran through `grasp-eval` at code snapshot `7c49494`,
using the original v2 executor and no extra command noise. The verified saved
HUG scene-000 seed-0 proposal was retargeted with the existing Dex adapter
(mean fingertip error 6.704 mm) and emitted as a project-authored saved Shadow
record. This is a HUG-derived development replay, not fresh generator inference
or an independent unseen-object test.

The physical task succeeds: maximum lift 0.151335 m, stable hold 2.0 s and
peak-to-final height loss 0.000760 m. Independent arithmetic reproduces the
result from all 163 saved samples, with no hand/table contact in those samples.
The original predictor hash remains `16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb`;
it was not scored because this generic recorder lacks required velocity/command
histories. No model, controller, source code or historical result was changed.

The earlier fresh-inference launch stopped before physics when the HUG runtime
reported missing dependencies. That setup attempt has no binary task label
and remains separate. Full inputs, declaration, verification hashes, trajectory
and replay are local under
`runs/partner_eval_2026-10-13_hug_cube_replay/` in the working checkout; the
incomplete launch is at `runs/partner_eval_2026-10-13_hug_cube/`. This check
establishes usable shared execution of an existing HUG-derived target, not
new predictor accuracy or generalization.


## 13 October 2026 — Add ContactGen and UniDexGrasp without collecting evaluations

**Previous plan:** The handoff identified source-translation checks and compatible
velocity/command recording before expanding provider evaluation. HUG is the
reference generator; DexGrasp Anything has a pinned external-worker handoff.

**New evidence and user direction:** The user specified four proposal sources:
HUG, DexGrasp Anything, ContactGen and UniDexGrasp, and explicitly deferred all
evaluation-data generation. Official ContactGen source at
`1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a` generates contact representations,
runs its MANO optimization solver and exports meshes. Official UniDexGrasp at
`36c9bfcf7cedc987dd15e2384c548e51d5aadc21` exposes IPDF/Glow proposal generation
separately from its learned execution policy. Source links and native contracts
are recorded in [the integration guide](CONTACTGEN_AND_UNIDEXGRASP.md).

**Choice and implemented status:** Add `contactgen_mano_shadow_v1` through the
existing MANO/Dex Retargeting path and `unidexgrasp_shadow_v1` through the existing
named Shadow mapping. Both have isolated, explicitly invoked model workers and
native-output import adapters. ContactGen accepts only refined right MANO output,
checks the original metric mesh/centering and preserves the existing wrist-origin
palm-placement policy. UniDexGrasp verifies the actual external source hand's
effective palm/finger kinematics against the pinned DexGraspNet reference before
reusing that mapping, composes the actual source palm offset and enforces the
2 mm fingertip FK gate. Its worker implements IPDF+Glow before TTA, not its
learned policy. Source, input, checkpoint, configuration and native-output
identities are preserved; pickle asset hashes are checked before model loading.

**Scientific status:** This is provider implementation, not a new experiment or
result. No fresh ContactGen/UniDexGrasp inference, MuJoCo rollout, acceptance
case, fitting or evaluation collection was run. Synthetic temporary unit fixtures
are not generator outputs or evaluation data. Existing HUG/DexGrasp Anything
specification hashes, frozen model, scientific protocol versions and historical
evidence are preserved. The generic predictor recorder/export gap remains.

**Implementation verification:** 72 focused tests pass; the two existing
rollout tests are explicitly excluded, and an autouse test-run guard fails any
MuJoCo stepping call. Tests use synthetic temporary fixtures, with generator and
pose translation boundaries mocked where appropriate. Both worker help paths
succeed without model loading, eight Python files parse, and the frozen model
hash remains `16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb`.
Existing provider hashes remain unchanged. Trimesh 4.12.2 is used only from a
temporary test dependency directory; the original recorded runtime is preserved.

**Next check:** When the user requests generation, supply compatible external
runtime/checkpoint/hand assets and declare one source/frame/geometry acceptance
check from each generator before collecting a larger batch. Fresh GPU execution
and physical suitability remain unverified. Reconcile the final agreed scientific
comparison separately; adding providers does not establish prediction improvement.


## 13 October 2026 — Run one newly authorized fresh HUG CLI evaluation

**Previous plan and authorization:** Evaluation generation was deferred while
ContactGen and UniDexGrasp adapters were implemented. The user then explicitly
requested one HUG evaluation using `grasp-eval`. This authorizes the single HUG
case; it does not begin a multi-provider collection or relax the remaining
providers' unrun acceptance status.

**New evidence:** The original HUG dependency trees, real compiled torch-cluster
CPU backend, MANO assets and cached DINO weights remain available. The earlier
missing-dependency launch omitted those directories from its import path. No
installation, runtime replacement, fake backend or checkpoint download is needed.

**Choice and implemented result:** Before inference, declare the existing
verified cube observation, HUG seed 0, one proposal, one nominal v2 execution and
no extra command noise. `grasp-eval validate`, `run` and `status` complete at
source `e1abfc3`; the actual HUG provider performs fresh 50-step CPU inference,
Dex Retargeting and one successful lift/hold. Independent trajectory arithmetic
verifies maximum lift 0.151335441733 m, hold 2.0 s and peak-to-final loss
0.000760347291 m. There are no hand/table contacts in 163 saved samples. The
video contains 163 frames at 25 fps. Full provenance is in
[the run note](HUG_SINGLE_EVAL_2026-10-13.md) and ignored local run bundle
`runs/hug_grasp_eval_2026-10-13_fresh_cube/`.

**Scientific status:** The native proposal and all saved trajectory arrays match
the earlier scene-000/seed-0 replay exactly. This is actual fresh inference and
execution reproducibility on an existing development situation, not a new
independent grasp/object/scene sample or model-accuracy result. The frozen
predictor is unchanged and unscored because generic output still lacks required
velocity/command histories. No ContactGen/UniDexGrasp generation, fitting, large
batch, controller change or historical relabeling was performed.

**Next check:** Preserve the completed video and hashes. Future HUG runs can use
the now-verified preserved import path. Compatible prediction recording and the
other generators' declared runtime/translation acceptance checks remain separate
work before scientific comparisons.


## 14 October 2026 — Audit grasp mapping fidelity before collection or training

**Previous plan:** HUG and new ContactGen/UniDexGrasp providers share a canonical
Shadow proposal boundary; DexGrasp Anything remains an external-worker handoff.
The user asked to verify that different native grasps are faithfully represented
before collecting data or training a cross-generator predictor.

**New evidence:** Static checks of the existing HUG cube show correct transform
composition but approximate pose fidelity: mean tip error 6.704 mm, thumb
14.125 mm. The declared v2 force-close command moves nominal tips by a further
18.644–32.212 mm in static FK; this is not measured achieved motion. At the
unchanged initial cube state, native vertex proximity and nominal Shadow
collision contacts cover -X/+Y/+Z; both intersect the object, and face coverage
does not establish matching contact patches. ContactGen
source/frame logic is consistent, without a real native sample. UniDex's
hash-pinned Dex reference bridge has maximum 1.751 mm tip error over 112
synthetic static configurations, but its kinematic certificate excludes surfaces
and dynamics. Actual UniDex source hand and model-output acceptance remain unrun.
DGA's pinned generator emits 24 joints rooted at a forearm, requiring explicit
root-to-palm composition; the 22-joint handoff recommendation is unsuitable as
an assumed native contract. A dedicated named-joint bridge candidate gives at
most 0.100 mm tip separation on 65 synthetic static URDF configurations.

**Choice:** Treat coordinate composition, nominal pose/contact preservation,
controller alterations and achieved physics as separate evidence. Use bounded
fidelity claims for MANO-to-Shadow retargeting. Keep real native artifacts,
reference proxies and synthetic algebra tests distinct. Four-source
generalization remains a proposed question, not an implemented result.

**Implementation status:** Added [the audit](GRASP_MAPPING_FIDELITY.md), static
reports, guarded diagnostic snapshots and saved-HUG figures under
`research_artifacts/grasp_mapping_audit_2026-10-14/`. No new model inference,
physics steps, eval data, task labels, fitting, adapter/controller changes or
historical relabeling occurred. The existing paper and frozen predictor are
unchanged. Existing dirty/untracked files and ignored runtimes remain preserved.

**Next check:** Compare finer native MANO/Shadow contact patches and normals
beyond the completed coarse HUG face coverage; obtain actual native outputs/source assets for the
remaining providers. Explicitly bind DGA dataset/normalizer, scale, native
24-joint/root semantics and sampling stage; bind full TTA config if importing
refined UniDex outputs. Proceed to collection only after declared mapping
acceptance evidence and a compatible predictor recorder are available.


## 14 October 2026 — Structure the late-breaking paper around completed diagnostics

**Previous plan:** The working manuscript motivates a matched HUG coverage
comparison but still uses generic article formatting and a title about evaluating
simulation diversity. The recorded protocol keeps that comparison unexecuted.

**New evidence:** The current [Sim2Real2Field call](https://sim2real2field-corl26.github.io/)
confirms four pages excluding references for preliminary work and a 15 October,
11:59 PM CDT deadline. The linked CoRL template is single-column with anonymous
initial-submission mode. The workshop does not explicitly exempt appendices or
provide a separate supplementary-upload field. Existing saved evidence reports
20 successful corrected v3 mugs with warnings and 13 successful paired v4
attempts with warnings; their shared families and protocol versions preclude
pooling them as independent trials.

**Choice:** After being offered the deadline-oriented and gradual-writing
options, the user chose to aim for this deadline with the completed diagnostics.
Plan a focused warning-reliability paper, with a title naming that finding,
methods and results prioritized, and a short limitations/next-experiment section.
Use full conference examples for argument structure, compressed to four pages.
This editorial scope does not replace the broader research protocol or assert
that calibration, representation or diversity remedies have been evaluated.

**Implementation status:** Added [PAPER_STRUCTURE_PLAN.md](PAPER_STRUCTURE_PLAN.md)
with verified requirements, primary-source examples, a page budget, display
plan and proposed opening/abstract. Updated the evidence index and submission
checklist. The working manuscript, models, labels, executor versions and
experiment artifacts remain unchanged; no fitting or new execution occurred.

**Next check:** Review the proposed title, contributions and starter prose with
both authors; shorten and format the existing manuscript in place, compile it
with the desktop compiler, inspect the PDF and verify the live submission form.
Keep essential evidence within four pages and use the earlier public deadline.


## 14 October 2026 — Confirm generators individually and keep HUG as the active case

**Previous plan:** A first four-source static audit identified correct boundaries
and unresolved contact/asset/worker evidence. The user clarified that each
generator must be tested and confirmed individually before moving on.

**Choice:** Start with the existing real HUG cube output. Separate exact native
representation, coordinate/joint conversion, nominal geometry/contact fidelity,
and executor changes. Preserve poor native proposals too: source penetration
is not itself a mapping failure or a reason to repair/remove a proposal.

**Implemented evidence:** The upstream parametric MANO wrapper reproduces all
778 vertices, 21 landmarks, wrist rotation, 1,552 closed face indices and saved
2D projection exactly without loading/sampling HUG's grasp generator. Eighteen
existing tests pass under a stepping guard. Additional manufactured checks of
64 object poses, 192 quaternion normalizations and 128 joint permutations pass;
64 global coordinate changes preserve local landmarks to numerical precision.
These are algebra checks rather than new grasps or evaluation cases.

Same-scale surface inspection uses the actual saved native mesh and Shadow's
analytic collision geoms/per-mesh convex hulls. On 3,750 deterministic cube-face
queries at 2 mm spacing, occupied-or-near footprint IoU is 0.583 at a 2 mm band
and 0.682 at 5 mm. Dominant finger/face regions persist, but patches shift or
broaden; native little-finger top-face coverage is missing from the 2 mm target
footprint. These diagnostics include penetration and are not physical contact
areas, a fidelity acceptance threshold or model accuracy. Skinning labels and
nearest-surface normal comparisons are geometric proxies.

**Status:** Native representation and coordinate/joint conventions are confirmed
within the tested scope. Contact fidelity is approximate and remains unconfirmed;
HUG has not received an overall fidelity sign-off. Other generators remain
pending individual checks. No new generator sampling, physics integration,
task labels, evaluation dataset, fitting, adapter/controller changes or
historical relabeling occurred. The paper and frozen predictor are unchanged.
[HUG walkthrough and artifacts](HUG_MAPPING_CONFIRMATION.md).

**Next check:** On this same existing cube, test grid/band sensitivity of the
per-finger footprint differences and compare normals only where anatomical
parts correspond. Relate any acceptance criterion to geometry/the intended
scientific claim before changing the mapping or moving to the next provider.


## 14 October 2026 — Keep failure prediction central and mapping validation foundational

**Previous framing:** A proposed paper opening placed grasp-to-simulator mapping
in the title and central question. The existing HUG research protocol instead
asks whether broader simulation coverage improves pre-lift outcome warnings
more than repeated finger perturbations at equal training-data budgets. Its
matched coverage comparison remains unexecuted.

**User clarification:** The prediction model and learning from varied simulated
grasps are the main research focus. Faithful translation of each generator into
the simulator is essential supporting methodology, rather than the paper's
primary research question.

**Choice:** Center the broader paper on failure prediction and simulation
training coverage. Treat four-generator mapping validation as evidence needed
to interpret the training examples and downstream comparisons. Prediction
outcomes describe adapted grasps under a declared robot, executor and simulator;
mapping fidelity alone does not guarantee predictive accuracy.

**Proposed extension, not a frozen experiment protocol:** At equal training-data
budgets, test whether broader coverage of simulated grasp situations, including
multiple grasp generators, improves warnings on unfamiliar objects and grasp
sources. Compare broader single-source coverage, multiple-source coverage and
additional perturbations of a narrower proposal set with common model, task and
execution settings. Independent object and held-out-generator evaluation would
test distinct generalization claims. Broader coverage is not assumed to help;
source-specific mapping discrepancies must remain available for interpretation.

**Status:** Recorded the user's framing clarification only. The earlier HUG
protocol, deadline-oriented diagnostic paper plan, working manuscript, historical
results and predictor are preserved. Four-source validation and cross-generator
prediction have not been completed. HUG remains the first individual mapping
case, with contact fidelity unconfirmed. No data generation, inference, physics
rollout, fitting or controller/adapter changes accompanied this entry.

**Next writing step:** Draft the abstract and introduction around failure
prediction and the training-coverage hypothesis, explain mapping as a validity
requirement in the method, and distinguish the intended broader study from the
completed HUG evidence. Finalize comparison conditions and prediction inputs
before future collection or fitting.


## 14 October 2026 — Accept the prediction title and make multiple-source training explicit

**User decision:** Adopt the broader working title, **Evaluating Simulation
Diversity for Grasp Failure Prediction**. The user clarified that proposals
from multiple grasp generators should be explicit in the central question,
rather than incidental wording inside a general coverage question.

**Proposed question for author review:** Does training a grasp-failure predictor
on simulated grasps from multiple generators improve generalization to
unfamiliar objects and grasp configurations compared with single-generator
training? Held-out-generator evaluation is a separate, stronger transfer check.

**Comparison rationale:** Equal selected training-episode counts are a methods
control to distinguish source mixing from simply adding more examples. Keep
predictor settings, input information, task labels and execution settings
consistent across the source comparison. Keep related proposals and their
perturbations together in splits. Comparing predictor architectures on matched
data addresses a different question and can complement the source comparison.
Published scores on different hands, sensors, tasks or test sets cannot establish
a direct performance advantage for our predictor.

**Primary-source evidence:** [FFHFlow, Sections 5.1--5.2](https://arxiv.org/html/2407.15161v4)
already uses a learned evaluator across several generative grasping baselines,
with heuristic-generated training data. This is a precedent for shared grasp
scoring, not evidence that our proposed mixed-source trajectory-prediction
comparison has been performed. [DexDiffuser/DexEvaluator](https://arxiv.org/html/2402.02989v3)
is a candidate static grasp-scoring comparison, with Allegro candidate-pose
inputs; [Temporal Visuo-Tactile Learning](https://arxiv.org/html/2610.10283v1)
is a pre-lift outcome-prediction precedent with a different hand and multimodal
inputs. Compatibility and reproduction remain to be assessed. This bounded
review does not establish novelty of multiple-source predictor training.

**Recommendation, not an added implementation commitment:** Complete and
validate the four current sources before expanding. They permit staged
single-source/mixed-source comparisons and train-on-three/test-on-one tests.
An additional source should broaden measured grasp coverage or provide an
external transfer test; source count alone is not evidence of generality.

**Status and next step:** Title accepted; exact comparative protocol remains
proposed. Record a prediction-centered opening, then specify compatible
predictor baselines and independent evaluation groups before any collection
or fitting. The existing manuscript and historical protocols/results are
preserved. HUG remains the active individual mapping check. This entry and
literature review generated no evaluation data, simulation rollout or model fit.


## 14 October 2026 — Resume actual HUG validation before drafting or fitting

**User direction:** Begin the multiple-generator prediction project by completing
mapping validation individually; paper drafting and training follow. The user
wants progressively better prediction on unfamiliar objects. Broader-source
training benefit remains a hypothesis, and iterative tuning must use validation
sets while independent object/source tests stay separate. No final accuracy
threshold or new collection protocol is frozen by this discussion.

**Implemented evidence:** A provenance inventory finds 32 distinct existing HUG
poses across four objects and eight observation groups, all with one MANO shape.
Original 24 named Shadow targets are recovered from each zero-noise closing
endpoint before force-close. Guarded FK reproduces stored validation and
independent fixed-root/captured-scene placement to numerical precision, with zero
joint-limit violations and unchanged object poses. Overall tip mean is 6.428 mm;
per-case means 5.704--8.305 mm; worst individual tip 17.518 mm. These historical
families are not new or independent held-out task cases.

A protocol frozen before measurement refines the existing cube on 2, 1 and
0.5 mm grids and 1/2/5 mm diagnostic bands. Original baselines reproduce exactly.
Global 2 mm-band occupied-or-near IoU is 0.582946/0.580406/0.578557, supporting
persistence of the geometry discrepancy over tested grids. At the finest grid,
little-finger top-face unsigned proximity is nearly lost, not strictly absent:
native 57 mm^2, target 0.5 mm^2, shared 0.25 mm^2. Conditional matched normals
retain lost/new/mismatched sets and object-interior nearest-point classifications;
548 of 748 matched fine-grid 2 mm-band pairs have both nearest points inside
the cube. They are not physical-contact correspondence evidence.

**Argument and status:** Proper coordinated frame rotations preserve world
geometry analytically; composition and named-joint/independent-FK checks support
the implemented mapping policy. Human-wrist/robot-palm anchoring and changed
finger geometry remain additional assumptions. HUG contact fidelity is
unconfirmed. The scripts/reports are preserved in
[the continuation bundle](../research_artifacts/hug_mapping_refinement_2026-10-14/README.md);
[the HUG note](HUG_MAPPING_CONFIRMATION.md) records the rationale and limitations.
No new HUG inference, physics, task labels, evaluation dataset, calibration,
retarget optimization, fitting, provider/controller change or paper edit occurred.
The frozen predictor SHA-256 stays
16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb.

**Next check:** Isolate wrist/palm anchoring from morphology and characterize
per-finger surfaces on additional existing objects. Any calibration and its
validation cases must remain separate and acceptance criteria must be justified
before testing. Keep HUG active before advancing providers; predictor collection
also needs the existing recorder gap resolved before future scoring/training.


## 14 October 2026 — HUG anchor and other-object surfaces; compute deferred

**Previous plan and user direction:** Separate HUG's wrist/palm assumption from
finger morphology and check the other existing objects before the next provider.
The user explicitly prohibits new eval data and asks whether renting GPU compute
would help reach larger research counts.

**Implemented evidence:** Guarded diagnostics on 32 saved poses reproduce the
MANO-wrist/Shadow-palm origin within 1.429e-16 m. The actual wrist body is 34 mm
away by URDF; native/Shadow palm-to-finger-base proxy distances are 95.351/93.889
mm. A shared translation explains 14.61% of fingertip squared residuals and
4.29% across ten task points. The MCP-proxy-derived mean increases mean tip
error to 10.371 mm. No offset is applied or claimed as calibrated.

Six fixed existing seed-0 cases cover both groups of apple, milk box and mug.
V1 maximum-edge sampling was not executed; v2 area-density sampling was run,
then an independent review found retained abutting internal collider seams.
A frozen v3 outward-probe correction removes those interfaces; v2 script,
protocol and report remain preserved as a superseded pilot. V4 adds separately
frozen actual visual-triangle queries. The final families are not pooled and
historical cube results remain unchanged. At a 2 mm proximity band, unsigned
IoU is 0.195--0.290 on collision queries and 0.243--0.273 on visual queries.
Actual lost/new per-finger regions persist on the visible surface. Shadow
unsigned distances query per-collider surfaces, potentially buried at penetrated
sites; occupancy and globally exterior matched normals stay distinct.

Reverse collision-to-visual area-weighted distances for first groups average
4.42/2.76/2.12 mm for apple/milk/mug. This proves the sampled geometric reference
surfaces differ, not that a specific decomposition-gap cause or outcome effect
is established. Sampling is area weighted, not a uniform spatial grid or a
convergence proof. The 32/six cases are inspected development evidence from
one MANO shape. No numerical score is adopted as a scientific acceptance gate.

**Choice and status:** The implemented placement policy is supported; exact
human-to-robot contact preservation is unconfirmed. Keep HUG active. Review
object collision approximation and declare anatomical/contact objectives with
separate calibration/validation groups before a versioned mapper change. Retain
poor native grasps; do not repair/filter them into successes. All reports,
scripts and protocol hashes are in
[the new bundle](../research_artifacts/hug_palm_object_validation_2026-10-14/README.md).
Primitive/triangle/winding checks, old cube regression and zero-time/input-hash
invariants pass. No inference, physics integration, task labels, new evaluation
dataset, fitting, collider/mapper/controller change or paper edit occurs.
Predictor hash remains
16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb.

**Compute recommendation, not an experiment:** GPU rental could help later
generator inference/refinement; the current CPU MuJoCo runner and NumPy/SciPy
logistic baseline do not automatically benefit. A GPU simulation backend would
need separate compatibility/numerical validation. Benchmark successful proposals
per minute, refinement time, peak VRAM and CPU rollouts before selecting hardware
after new collection is authorized. No rental, provisioning or spending occurs.
Independent objects/observations/generators and grouped held-out comparisons
matter more than repeated episode counts; larger counts or source variety do
not guarantee prediction improvement. Resolve the previously documented recorder
gap before future predictor collection/scoring.


## 14 October 2026 — Original-source evidence and mapping validity before outcomes

**User direction:** Ground each generator's mapping argument in its original
paper and implementation. The user explicitly agrees that faithful mappings
must preserve bad proposals and that mapping validity precedes task labeling.

**New evidence:** Read-only review of all four original papers and pinned code
finds useful native contracts and quality measurements, but no certification
of this project's translation. Source generation/refinement/test stages differ.
The latest section of [the mapping audit](GRASP_MAPPING_FIDELITY.md) records
paper sections, code links, distinctions and current export/asset gaps.

**Choice and proposed work:** Use a shared reporting framework with source-specific
contracts: native reconstruction, conversion/FK correctness, paired geometric
changes and separately evaluated execution. The detailed protocol and acceptance
tolerances remain proposed; no scientific pass threshold is frozen. Keep native
quality measurements separate from mapping-induced changes and eventual outcomes.
Do not silently repair/filter poor source proposals. Explicit mapping/feasibility
rejections retain their reasons and are not physical task-failure labels.

**Implemented scope and next check:** Literature review and documentation only;
HUG remains active before other providers. Apply source-grounded correspondences
and baselines to HUG while reviewing object collision approximation. ContactGen
contact-map export and actual provider assets/runtime evidence remain pending.
No new generator inference, dynamics, eval data, fit, mapper/controller/collider
change or manuscript edit occurs. Existing protected hashes remain unchanged.


## 14 October 2026 — HUG object causes, reference methods and signed profiles

**Authorized direction:** Complete the proposed HUG static investigation before
changing the mapper or moving providers. Preserve poor native proposals and do
not generate eval data.

**Implemented evidence:** Source OBJ/XML scale/transforms match compiled object
vertices within 4.146/7.365/7.690 nm for apple/milk/mug. The large apple/milk
reverse-distance witnesses lie inside visible solids at inherited micrometer
slits (next hull about 3.91/3.52 micrometers away), not equivalent large outer
placement errors. Mug's roughly 10 mm worst witness is an exterior convex
approximation. All 36 raw simulation asset files match pinned download
manifests. No tested scale/frame bug is established, and physical hulls/probes
are not altered. Local ray witnesses do not certify global air connectivity or
contact accessibility.

Current targets regenerate exactly on all 32 saved poses. A frozen released Dex
Retargeting 0.5.0 Shadow DexPilot-style reference and a separate
metric/no-filter sensitivity expose a tradeoff: mean tips 6.428/21.343/2.353 mm,
middle proxies 6.829/26.377/13.811 mm. The sensitivity's ten-link mean is 8.082
versus current 6.628 mm. Current whole unsigned 2 mm visual proximity IoU
exceeds both alternatives on all six selected cases and first-group finer
repeats; individual fingers differ. No projected pinch activates;
raw/filter/clamp stages remain distinct. This reference adaptation is neither
the original Allegro reproduction nor a HUG mapping ground truth.

Independent signed visible-object profiles show penetration in all six native
area samples (sampled maximum 1.87–24.28 mm, negative area 2.99%–6.84%). Native
vertex-count, native triangle-area and target approximate-area denominators
remain separate. Targets expose sampled union surfaces and coarser carriers
separately; the shapes are not point-corresponded. Finite maxima/misses do not
certify continuous intersection/contact or rank fidelity automatically. Bad
native proposals remain included, without task labels.

**Review correction and version history:** Original reference limit-table
metadata actually contains captured-scene decimal ranges; actual
optimization/clamping used full source URDF. A standalone correction verifies
unchanged final targets against the correct source bounds, preserving original
bytes. Surface V2 and signed V1 stop on this decimal representation issue; final
frozen surface V3/signed V2 parse original XML and log strict scene excess (up
to 3.878e-7 rad), without changing q or implying production preflight rejection.
Incomplete pilots remain separate.

**Choice and bounded claim:** Retain the production mapper; no measured evidence
justifies a replacement. Numerical conversion is supported in tested scope,
while cross-hand contact and failure-label equivalence remain unconfirmed. [HUG
findings](HUG_MAPPING_CONFIRMATION.md), [measurement/proposed-acceptance
framework](HUG_STATIC_VALIDATION_PROTOCOL.md) and [versioned
artifacts](../research_artifacts/hug_geometry_reference_validation_2026-10-14/README.md)
record the argument. Next address object-facing normals and spatial sampling
uncertainty, justify acceptance margins, and freeze an independent validation
design. These inspected development cases are not holdout evidence.
Outcome-sensitivity/physical tests require separate future authorization.

All completed source-hash/guard/time-zero/object-state checks pass; current
visual baselines reproduce exactly and reference reproduction matches
geometry/configuration. Production Python, frozen predictor and manuscript
hashes remain unchanged. No new HUG inference, physics rollout, eval
dataset/label, fit, production mapper/controller/collider change or manuscript
edit occurs. Predictor SHA remains
16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb.


## 14 October 2026 — HUG orientation, uncertainty and scoped acceptance completed

**Previous plan and authorization:** Finish surface orientation and sampling
uncertainty on existing HUG grasps, then define acceptable mapping. The user
explicitly prohibits new eval data. Keep poor source proposals and separate
mapping status from task failure.

**Measured evidence:** All 32 saved MANO 3D exports reconstruct exactly (778
vertices/21 landmarks/wrist rotation/1,552 capped faces); camera projection
remains the original representative-cube scope. Six previously selected seed-0
noncube cases plus three finer repeats receive object-facing normal checks and
full triangle-radius bounds. Eligible exterior/same-part normals change sign in
retained apple/milk sites, with independent source/current replay of middle/ring
witnesses. Mean angles are conditional geometric descriptions, not contact
forces or an acceptance margin. Coarse sampling misses some fine sign reversals;
no continuous angular certificate is supplied.

The old area density leaves maximum centroid radii 5.627 mm coarse/4.551 mm
fine. At a 2 mm band, unresolved whole-hand area is 8.611–16.821% coarse and
5.121–9.779% fine. Nevertheless positive-area same-part lost/added proximity
survives cell-radius accounting in all six cases and three finer repeats; 178
auditable cells bind source/protocol/rule hashes. Results are unchanged across
assumed 1/10/100 nm arithmetic allowances, which are conditional budgets rather
than formal floating-point proofs. Open anatomical surfaces and per-collider
boundary distances retain their literal scope; occupancy/normals/physical
contacts do not inherit these bounds. Carriers lack native reference and are
excluded from the six-part gate.

**Choice and definition:** Freeze [strict scoped acceptance
V1](HUG_MAPPING_ACCEPTANCE.md) before reading new reports (rule SHA
87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4). Numerical
guards must pass. Strict 1/2/5 mm same-part band preservation fails on a
definite positive-area loss/addition or reliable pointwise facing-state
reversal. PASS needs complete equal-state/domain evidence; unknowns remain
UNRESOLVED. This zero-change condition defines a strict preservation claim, not
a universal acceptable-error budget for approximate morphology conversion.
Papers motivate separate geometry/outcome reporting but supply no HUG-to-Shadow
certificate. An approximate predictor pipeline needs an externally justified
distortion budget and independent validation; none is invented from development
scores.

**Result and limits:** Numerical conversion passes in recorded scope; strict
relationship preservation FAILS conditionally in all nine detailed resolutions.
Full physical contact equivalence remains UNRESOLVED; failure-label equivalence,
independent validation and predictor adequacy remain NOT_TESTED. These statuses
are not eval outcome labels. Keep the existing production mapper and original
bad proposals unchanged while recording the failed strict claim. The checks and
definition are completed, rather than continuing to say all geometry is merely
unmeasured.

**Version history and verification:** Preserve sampling V1 incomplete far-cache
regression pilot and final V2; preserve orientation V1 pre-metric
nearest-candidate amendment and unchanged final V2 after a missing-export
partial run. No scientific threshold or source/q/geometry is tuned. Six
manufactured sampling tests, 100 independent analytic-strip checks,
sign/tie/member tests, orientation witness replay and all
source/protocol/cache/110 protected hashes pass; guards show zero actual
integration, time/object/nominal states unchanged. [Versioned
evidence](../research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/README.md)
includes selected reports, figures, scripts and frozen rules, without raw
licensed assets or scratch arrays.

**Next check:** Specify a new version of retargeting against retained
anatomical-region witnesses, or adopt an explicit narrower mapped-simulator
claim with a justified approximation contract. Do not loosen this strict rule or
use successful lifts to hide changed native relations. New independent
cases/outcome sensitivity require later scope; no new inference, physics, eval
records/labels, model fit, production mapper/collider/controller change or
manuscript edit occurred. Existing predictor and paper hashes remain unchanged.


## 14 October 2026 — Surface-aware HUG correction is implemented but not accepted

**Previous plan and authorization:** Address retained native/Shadow relationship
witnesses before moving to another generator. The user asks to fix HUG and keeps
new eval collection deferred. This authorizes static mapper development on saved
inputs; it does not authorize new inference, rollouts, task labels or fitting a
prediction model.

**Implemented correction and evidence:** Frozen V2.1 adds fixed native-relative
unsigned distances and closest-point vector residuals, near/broad area quadrature,
weak original landmark/joint regularization, source-URDF limits and a 0.35 rad
trust domain. Wrist then finger stages run actual compiled time-zero FK on the
same six saved grasps. Native poor relationships remain targets. All 36 stages
improve their objective but all six raw targets exceed the existing 10 mm mean
fingertip guard. Frozen V3.1 uses a common [1,.5,.25,.125,.0625,.03125,0] joint
backtracking rule and a consistent original-relative full-pose cost. Five cases
select .5 and second milk selects .25; guarded mean tips are 8.458–9.290 mm and
costs decrease 10.527–21.471%. Tip/middle errors still worsen on every case.

The declared MANO-wrist/Shadow-palm POSITION stays fixed. WRJ articulation changes
palm ORIENTATION 0.501–1.355 degrees under the existing q-dependent convention;
carrier translation only compensates this articulation. No arbitrary object/hand
shift, scaling, carrier-rotation fit or source edit occurs.

Independent unchanged six-coarse/three-fine audits verify actual final geometry,
source joints/root chain, interpolation and fingertip guard. All nine strict
surface statuses remain conditional FAIL. Whole 2 mm centroid IoU improves on
all six coarse cases (.243–.273 to .294–.395); six-part lost-area lower-bound sums
also decrease. First-milk added-area sum regresses 84.28→102.14 mm². Reliable
2 mm sign counters first apple 2/28→3/34, first milk 3/39→6/77, second milk 1/18→2/36
retain changing eligibility domains; counts are not comparable error rates.
178 original cell witnesses replay as 65 retained definite changes, 13 definite
equal states and 100 unresolved cells. Uncertainty is not a repair certificate.

**Choice and status:** Keep the default mapper/grasp-eval unchanged; preserve
this as a versioned development correction, not validated native-equivalent
geometry. Add separate explicit static APIs with validation and callback/audit
protection. Seventy manufactured tests pass, and eleven valid frozen/API
comparisons reproduce targets and all diagnostics exactly. Stored engineering
assessments reproduce all six guarded selections exactly; neither check claims
a six-geometry API rerun. A default local solver can improve loss while remaining
on the wrong side in a manufactured case; the test records this limitation.
[Report](HUG_SURFACE_REFINEMENT.md) and [evidence](../research_artifacts/hug_surface_refinement_2026-10-14/README.md)
retain the full case/part/resolution results and regressions.

The strict zero-change rule remains unchanged and is not a universal approximate
retargeter standard. Independent distance intervals remain ambiguous on some
band-boundary cells even with identical distance functions. Removing known
counterexamples would not automatically establish continuous full-domain PASS;
normal feature bounds and adequate evidence remain required. Current failure
does not prove that every possible Shadow target must fail. Native-contact
forces/outcomes, unseen-object validation and predictor adequacy are unresolved
or untested, rather than inferred from solver/engineering success.

**Versions and preservation:** Raw V2.1 protocol 8d2fd666… and guard V3.1
59e04a7a… bind unchanged source inputs; independent wrapper 0c293119… applies the
old strict rule 87da8d6c…. Preserve unexecuted V1/V2 and V3 binding-key abort before
case loading separately; no historical report is rewritten. Selected exact-byte
gzip reports, scripts, protocol manifests, source/root/witness checks and scientific
figures are published without raw licensed assets/checkpoints/scratch arrays.
110 original protected source/model/paper hashes and original inputs remain
unchanged. Integration API guards record zero actual steps; no new HUG inference,
eval data/labels, predictor training, default mapper/collider/controller change
or manuscript edit occurs.

**Proposed next check:** A global surface objective with the existing fingertip
condition constrained during solving and reliable native-relative orientation
features could target the remaining relationships. Freeze settings and preserve
bad native states, ties/unavailable domains and the same independent audit.
This next solver is not implemented or guaranteed here. No new collection or
four-generator sign-off follows from the measured development improvement.


## 14 October 2026 — Joint HUG fingertip/direction fitting implemented; real results pending

**Authorization and previous plan:** The user asks to implement fingertip
constraints and actual native-relative surface-direction fitting on existing
HUG grasps, then rerun mapping checks. New eval data and other-generator work
remain deferred. The previous staged/interpolated surface correction worsened
tip/middle errors and failed all nine strict audits.

**Implemented evidence:** The new isolated joint SLSQP API enforces each
fingertip's original-error bound plus the existing strict 10 mm mean guard;
final engineering checks name a 10 nm per-tip arithmetic allowance separately.
Actual native facing scores, including away states, are fixed targets. Newly
invalid fixed sites retain nonzero penalties and reasons. Source limits,
original±0.35 rad trust bounds, original-q fallback, best feasible major/raw
selection and nonconvergence bookkeeping are explicit. Fifty new manufactured
tests pass; the installed API and prior surface/guard tests pass all 120 checks.
The installed module is byte-identical to the frozen experiment solver.

**Runtime choice and status:** Canonical runtime V2.1 (e42b74b7…) runs all six
existing seed-0 apple/milk/mug cases from original q under the same method V1
settings. It uses two active Numba threads with pool four and BLAS/OpenMP one.
One/two-thread field/reference parity passes six baselines and three fixed
static poses. V1 incomplete searches and V2 import-only aborts remain separate;
the faster burial helper passed separate equality checks but its frozen V3
fitting preparation is unexecuted. Retain the active run rather than discard
over twenty minutes of progress for an uncertain whole-fit timing gain.

**Pending measured claims:** No completed six-case constrained result or new
independent mapping verdict exists at this checkpoint. Next finish all six
under fixed 120-iteration settings, recompute source/target preflight and apply
the unchanged six-coarse/three-fine rule. Do not promote from average fitting
improvement or engineering guard passage. Default mapper/controller, original
inputs, frozen predictor and manuscript remain unchanged. No inference,
integrated physics, eval episode/task label, prediction training or provider
advance occurs. [Implementation and scope](HUG_CONSTRAINED_REFINEMENT.md).


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


## 14 October 2026 — HUG feasible-step repair; full mapping remains failed

**Previous plan:** Six surface fits were refused because every eligible major/raw
candidate violated an original-relative fingertip cap. The user authorizes
continued HUG diagnosis/repair, concurrent provider work and ongoing paper updates,
with no new eval data.

**Measured diagnosis:** All 649 recorded major fingertip vectors replay exactly;
all six original cap Jacobians have rank five and admit inward feasible steps.
Cheap FK controls restore all six old raw poses. The first surface optimizer
takes large nonlinear steps and lacks feasible-step restoration; its refused
trajectory does not establish contradictory constraints or global infeasibility.
A bounded 60-call stencil observes nearby direction-eligibility jumps but does
not identify them as the sole cause of all six failures. No demonstrated frame,
cache or constraint-sign bug was found in these recorded checks.

**Implemented and frozen:** An isolated feasible-refinement API restores/checks
cheap FK before new full-surface scores, retains original caps/source bounds and
native poor relationships, and independently recomputes terminal results. The
actual study is seed-only on one immutable old raw pose per original case, no
pattern sweeps. Its 1-nm inward target tightens projection; computed final caps
have zero added allowance, and the five-tip mean remains strictly ≤10 mm. The
scientific 1/2/5-mm surface rule is unchanged. Twenty-six new manufactured and
120 prior surface checks pass (146 total).

**Completed result:** All six poses improve frozen cost by 1.52–38.09% while
passing all 30 original-relative fingertip caps and independent source/scene
checks. This restoration stage uses 18 full calls in 49.57 s, reusing previously
computed raw seeds; it is not a from-scratch speed claim. Two cheap projections
converge, four exhaust their 40-iteration budget while returning verified
feasible improving points. All six source preflights PASS. The unchanged six
coarse/three fine surface audits give 0 PASS/9 FAIL; pointwise orientation is
9 FAIL, versus 7 FAIL/2 UNRESOLVED on original fallback poses. Larger computed
added-area lower bounds and new reliable reversal witnesses show that the
fitting objective cannot certify contact preservation. Counts have pose-specific
eligibility and are not physical failure rates.

**Decision and scope:** Retain the feasibility repair as experimental. Do not
promote it to the default mapper, loosen acceptance, or call complete HUG mapping
validated. Study the remaining anatomical contact/surface witnesses separately
with this feasibility mechanism. The general pattern path has manufactured
evidence only. All 89 bound numerical dependencies, saved inputs and predictor
remain unchanged; mutable paper/unrelated provider code is outside that closure.
The same manuscript's HUG subsection is updated and compiles successfully.
No inference, integrated physics, eval episodes, outcome labels or training ran.
Original scientific bundles/protocols remain immutable.
[Completed phase](HUG_FEASIBLE_REFINEMENT.md).


## 14 October 2026 — DexGrasp Anything native smoke completed; mapping remains unaccepted

**Previous plan:** acquire the pinned DGA runtime/checkpoint and authentic native
outputs before claiming actual-output mapping, with no physics, evaluation
data/labels or training.

**Measured evidence:** one strict 794/794-key checkpoint load and one unguided
100-step DDPM call produced two unfiltered object01 outputs, seed 20261014.
Sampling alone took 0.927458506077528 s. Both original poses violate the source
URDF joint limits and remain REJECTED_SOURCE_LIMIT, with no physical-failure
labels. Original input/stage reconstruction PASS; original runtime FK/vertex
reconstruction FAILS its frozen 1e-5 m / 1e-4 rad engineering gates. A separately
frozen static TF32 control reproduces all original arrays bit for bit with flags
on; flags off passes the same gates on identical saved poses. This diagnoses
static FK precision only, not TF32-off generation equivalence.

**Choice and implementation status:** retain both original source rejections and
numerical FAIL alongside the separate static control PASS. Do not clip/reroll,
loosen gates or promote the shared registry. Original dependent surface auditing
remains blocked; contact adequacy UNRESOLVED, physical/predictor equivalence
NOT_TESTED. Native wrapper, actual-output validator and precision diagnostics
are implemented; 150 decoder/asset tests pass. The DGA-only manuscript section
is updated and compiles. The A40 Pod is stopped; original outputs and runtime
overlay are saved. No physics, evaluation episodes/labels or training ran.

**Next check:** declare a bounded source-valid cohort and any changed generator,
refinement or numerical-precision policy separately, preserving every rejection.
Establish justified geometry/contact evidence before accepting a mapping. This
two-output smoke is not success-rate or throughput evidence. Previous scientific
protocols and historical results remain unchanged.

[Provider report](parallel_validation/DEXGRASP_ANYTHING_VALIDATION_REPORT.md);
[evidence](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-14_native_smoke_v1/).


## 14 October 2026 — Fixed DGA native cohort: 0/16 domain-valid, no surface queries

**Previous plan:** obtain raw source-valid DGA proposals with a separately
declared bounded protocol, then audit at most the first two satisfying source
and target joint domains, without replacing later prerequisite failures.

**Measured evidence:** frozen protocol ccf24cd4... completed eight two-sample
runs, seeds20261015–20261022, with unchanged unguided DDPM generation and
eight strict794/794 checkpoint loads. All16 originals are retained; every
THJ1 is negative (-0.541857 to -0.00317943rad). Independent XML/domain/hash
accounting agrees with every saved rejection. All eight stage checks pass;
original defaultTF32 reference reconstruction fails16/16 and the separately
declared FP32 reference passes16/16 under unchanged engineering gates.

**Diagnosis:** the normalized final THJ1 component is already below-1 in all
rows; the official affine inverse and unchanged24-joint export reproduce it
exactly. No tested sign/order/denormalization defect was found. Available
release material does not establish historical training/preprocessing/joint-slot
and input-distribution compatibility, so the model-level cause is unproven.

**Choice and implementation:** stop at the declared16; no clipping, rerolls,
extra seeds, changed sampler or hidden refinement. Selection is empty, hence
zero actual source-target mapping candidates, surface cases and triangle tests.
Numerical source-FK reconstruction does not validate executable poses/contact.
Registration remains unsupported; scientific acceptance UNRESOLVED. No physics,
evaluation episodes/labels or training ran. Both DGA Pods are stopped. Five new
manufactured controller tests pass; the DGA manuscript block is updated and
compiles. Earlier protocols/results and unrelated provider changes are preserved.

**Next check:** obtain authoritative training/release preprocessing and24-slot
joint metadata. Any later constraint/refinement or changed sampler would be
a separately declared generator stage. The bounded single-object cohort is not
a general validity/success-rate estimate.
[Report](parallel_validation/DEXGRASP_ANYTHING_VALIDATION_REPORT.md);
[evidence](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-14_native_cohort_v1/).


## 14 October 2026 — A40 recovery and HUG direction-selection diagnosis

**Previous plan:** retain the original stopped A40, restore its verified runtime on another available A40 only after another failed restart, and continue HUG validation without new eval data.

**Operational result:** original 8ibam2z5n6ea5r again returned insufficient free GPUs on its host. Fresh Secure Cloud stock showed A40 Medium in CA-MTL-1 at $0.59/hour. One replacement 10fvywuiko09rk is running; the original remains stopped and preserved. Both backup hashes and 21,079 restored regular files/three symlinks verify. Python 3.8.12/Torch 1.11/CUDA 11.3, compiled imports, pip check, pinned source/input hashes and a fixed CUDA arithmetic control pass. No model/checkpoint load, generator call, physics or eval data was produced. Compute remains billable while running; retained disks also bill. This is runtime recovery, not HUG GPU acceleration or scientific acceptance.

**New measured HUG evidence:** saved residual analysis shows the total fitting objective falls while its exact facing soft-L1 contribution rises in every one of six fits. Each invalidates 49–104 fixed direction sites. Of 43 unique saved coarse direction witnesses, 20 are fixed direction-fitting sites, 3 are originally eligible but unselected and 20 are outside the original eligible stratum. Only 5 match part-specific distance-fitting sites. Fine indices/coordinate proximity are distinguished; all six parts/three bands/three numeric allowances are retained per domain without area pooling. No normal-lookup bug or global impossibility is proved.

**Implemented decision:** add an optional conservative candidate selector to the isolated feasible solver: no per-part fixed direction-SSE increase and no original-valid site invalidation; fixed targets, positive mass and invalid residual 2 remain. Ordinary refusal skips a candidate; malformed/error/mutation fails safely and fresh terminal evidence is rechecked. Thirty-six new manufactured controls and 182 installed surface API tests pass. Post-hoc, separately frozen replay accepts 6 baseline identity controls and refuses 6 saved regressing fits while all 30 saved tip caps still pass. Independent report replay, input/hash checks and all per-part arithmetic agree. No new fit or geometry query ran; baseline selector acceptance is not mapping acceptance. All 9 strict mapping comparisons still FAIL. Production mapping, original proposals, acceptance rule, predictor and historical scientific bundles are preserved.

**Next check:** a separately frozen bounded geometry driver must key normal metadata by q, use the new selector during search and reapply the unchanged independent audit to any selected new pose. Expanded strata and per-part/per-band constraints are proposed. HUG surface/contact and failure-label equivalence remain unconfirmed; no training or new evaluation collection is begun. [HUG report](HUG_SURFACE_SELECTION_GUARD.md); [runtime handoff](GPU_RUNTIME_RECOVERY_2026-10-14.md).


## 2026-10-14 — DGA checkpoint metadata recovery and distal-thumb convention conflict

Previous plan: obtain authoritative checkpoint training/preprocessing metadata before more native sampling or mapping acceptance after the separately frozen two- and sixteen-output protocols rejected every proposal.

New evidence (completed CPU/public-source audit): symbolic checkpoint metadata exposes epoch60, step67762 and794 tensors with compatible100-step diffusion buffers, but no historical run/data/joint-sign manifest. The author-posted converter preserves original negative-range THJ0 in final24-slot23; the DGA URDF has positive-range THJ1 and opposite parent-frame axis. Independent manufactured distal rotation checks support v=-q within2.052e-10, without establishing whole-hand geometry. The published processed pose member at HF dataset revision5931d702918388094a00a3954ec952ca4b8a4e1f has verified size/CRC/SHA70bbacb7d7c53eb3de4526427bb10a756ecef0febc8b487b60bf297dd6afc23a; six of its first eight consecutive raw final coordinates are negative and two positive. All eight are retained, without a prevalence estimate. Independent raw storage/CRC/SHA checks agree. Full outer-archive hash and checkpoint-to-training-file lineage remain unverified.

Decision: classify the released processed-data/current-hand convention conflict as verified and a strong explanation consistent with the eighteen native thumb rejections; do not promote that to proven historical checkpoint causality. Author conversion code is retrospective and qualified. No wrapper indexing/affine/joint-tail defect or schedule-buffer mismatch was found. Preserve the separate terminal-prior and nonunit-normal-scaling concerns as hypotheses/source traits, not thumb-specific causal proof.

Implementation status: metadata evidence/report and only the owned DGA manuscript subsection updated; no semantic adapter, bounds/angle repair, mapper/registry/controller/FK/evaluator change, model call, physics, evaluation-data/label creation or training. Prior0/2 and0/16 source-valid counts and both precision policies remain unchanged; all304 original cohort files still hash-match. Provider stays unsupported_pending_official_mapping. No staging, commit or push.

Next check (proposed, not run): bind checkpoint SHA f7675d8eb93b3978b2c2d72ec9f54ead6b936d1dd307e07605daa0c02bcac14f /epoch60/step67762 to a historical data/converter/hand convention, then separately freeze and independently audit an all24-joint/frame adapter using existing authentic records and saved outputs before mapping/contact acceptance. This offline diagnostic requires no extra GPU. No author outreach sent.

Evidence: research_artifacts/dexgrasp_anything_mapping_validation_2026-10-14_metadata_audit_v1/metadata_recovery_report_v1.md and evidence_manifest.json; primary author conversion: https://github.com/4DVLab/DexGrasp-Anything/issues/27#issuecomment-3550557965. Historical scientific protocols remain distinct.


## 2026-10-14 — ContactGen authentic native smoke, bounded mapping evidence

Previous plan: assets/runtime were prepared while authentic native reconstruction
and mapping remained untested. Explicit authorization enabled one seed-42
50 mm object01 development case on private Runpod A40 compute. Native execution
and exact float32 replay of all 778 vertices/16 frames pass. The unchanged
10 mm mean-tip admission gate passes at 4.44 mm mean/8.52 mm worst tip; the
canonical importer exports one proposal using the saved authentic retarget.
Corrected static geometry retains 60.97% of native 2 mm boundary-proximity area
(61.20% finer sampling), with sampled penetration present. One post-observation
forearm-label audit bug was corrected from cached arrays; original v1 evidence
is preserved and no pose, threshold or physical protocol changed.

Decision/implemented: retain the native outputs and nominal mapping, distinguish
engineering admission from unresolved scientific contact/mapping acceptance,
and update only the ContactGen paper section with these bounded results. Desktop
LaTeX compilation passes. One native sample and one retarget total; zero physics,
evaluation episodes, new labels, training or evaluation dataset. Both ContactGen
Pods are verified stopped; retained disks bill. The older empty Pod was not
deleted because automatic approval review required explicit deletion consent.

Evidence: `research_artifacts/contactgen_mapping_validation_2026-10-14_native_smoke_v1` and
`docs/parallel_validation/CONTACTGEN_VALIDATION_REPORT.md`. Next check/proposed:
justify source-specific scientific acceptance and assess an independent cohort
under separately authorized scope. No broader generator, physical outcome or
predictor validity is established, and no historical protocol/result was merged.


## 2026-10-14 — DGA operational geometry audit on preserved rejected outputs

Previous plan: recover training/preprocessing semantics before attempting further mapping checks. Refinement: unpublished training logs are not logically required to audit the pinned visualizer's explicit operational geometry. A checkpoint-bound encoding specification remains necessary for claims about intended legacy semantics or historical training causality. No automatic thumb reinterpretation is justified.

Completed: all24-joint/frame semantic audit and independently reviewed mathematical-only diagnostic on all16 saved cohort outputs, under a new premeasurement policy SHA56606b0e1e6708c8a4e00d03705a0f641225b4e5d92805536883ce8fe04579ca. All24 angles and strict old rejections remain unchanged. Palm translation residual≤5.648e-17m; independent FK discrepancy≤1.388e-16. Five manufactured guard/transform tests passed; all16 FP32 source references reconstruct within old engineering gates. Target FK is rooted at world, not forearm; a frozen-v1 metadata field was misnamed. An additive frame-label correction, independently checked on all16, preserves all numerical results.

Visual evidence: exact bounded triangle queries on1mm/0.5mm cube grids show positive lower bounds on changed common-coverage near regions in15/16 configurations at each0.5/1/2mm band. At1mm, maxima111.0mm2 source-only and76.25mm2 target-only. These are encoded visual-triangle observations outside source joint domains; overlapping walls remain, so no physical-contact loss/exterior-union normal claim. Independent arithmetic reproduces40,320areas and2,880coarse/fine overlaps. Exact encoded nearset equality is disproved for witnessed rows; approximate exterior/contact adequacy remains UNRESOLVED without a justified criterion and reliable surface contract.

Official eval source has the same24joint domains, no hidden thumb sign fix/active Python clamp, a separate one-Adam target stage and different visual/collision declarations; literal eval XML has duplicate closing robot tags. No optimizer/simulator was executed. DGA→target preserves angle signs; original→DGA has additional little/thumb geometry differences beyond distal sign.

Decision/status: coordinate algebra now has actual-output mathematical evidence, while original0/2 and0/16 source-valid counts remain. Shared registration stays unsupported_pending_official_mapping. No new inference, GPU start, clipping/repair, physics, evaluation episodes/labels or training; no shared mapper/controller/FK/evaluator edits and no Git staging/commit/push. Alternative official MultiDex arm is only a pinned feasibility/plan pending the user's scope choice; it cannot replace old results or guarantee contact fidelity.

Evidence: research_artifacts/dexgrasp_anything_mapping_validation_2026-10-14_literal_geometry_v1/continuation_findings_v1.md. Next accepted mapping needs an explicit chosen generator/stage contract, authentic domain-valid examples, and source-specific geometry/contact criteria. Historical protocols are preserved separately.

## 2026-10-14 — ContactGen exact surface-depth and exposed-union assessment

Previous plan: complete checks on the saved native smoke output without new
evaluation data. Implemented: inventory confirms one distinct cube/seed42 native
grasp; coordinate/schema/scalar controls pass. New strict-exterior diagnostics
reuse the same 12,288 cube queries, retaining 58.59% of the native 2 mm proximity
area on the target-exterior domain (different denominator from prior 60.97%).
An exact rational proof covers all 1,538 unchanged MANO triangles and establishes
maximum cube depth 1.700186729 mm. A separately numerically verified exposed
middle-finger robot boundary point reaches 3.794710139 mm: 2.094523410 mm deeper.
The proof, source bindings and available 45-collider baseline provenance were
independently checked. The v1 zero-mean guard refusal and corrected center-aware
freeze are preserved. Exact preservation is contradicted; useful approximate
fidelity remains unresolved without a prospective application error budget.
Global MANO material occupancy, source nearest-face tie ambiguity, unseen-object
and physical/predictor equivalence remain uncertified or untested. No threshold
was selected to pass this case and no shared production mapper was changed.
No new inference, retarget, physics, labels, training or evaluation data ran.
Both ContactGen Pods remain stopped. Evidence:
`research_artifacts/contactgen_mapping_completion_2026-10-14_static_v1/`.
Next check: resolve whether additional native mapping grasps are within the
user's no-new-evaluation-data constraint; freeze claim/margins/cohort before any
new sampling. Previously inspected shared-mapper geometries are development,
not new holdouts. This entry does not authorize a new native cohort.


## 14 October 2026 — HUG guarded search and fixed-anchor palm obstruction

**Previous plan:** run the direction-preserving guard during actual bounded fitting on the six existing HUG grasps, then independently validate selected poses under the unchanged rule.

**Completed evidence:** 134 full surface callbacks refuse all 122 fully scored nonbaseline candidates; each case reaches 23,999 FK callbacks and retains its exact original complete scene pose. Nineteen new driver/cache controls pass. Fresh independent source/FK preflight passes all six and all 30 exact computed tip caps. Because complete poses and bound inputs are bitwise identical, the nine original surface reports are explicitly reused, with zero newly measured full-grid comparisons: strict/proximity 0 PASS/9 FAIL/0 UNRESOLVED; pointwise orientation 7 FAIL/2 UNRESOLVED. An independent 369-condition review and five authentic-record corruption refusals pass. This finite search alone does not prove infeasibility.

**New geometric argument:** separately frozen complete wrist-domain covers refute strict palm preservation for saved Mug B under the current anchor. A stronger radial-shell control refutes it for ANY palm orientation at that same anchor: an existing whole 3.9636778 mm² cell is native-near within 4.745913 mm at the 5 mm band, yet at least 6.645928 mm from every rotated eight-box Shadow palm after declared allowances. Independent 90-digit/provenance review passes 687 checks. Independent source/FK/analytic-box replay reproduces all 149 local and 171 full-wrist leaf distances. Mug A remains unresolved; source/compiled rounding differences and aborted/unexecuted histories are retained. These are conditional engineering arithmetic results for literal part collider surfaces, not formal directed-rounding, exterior-contact or native failure-label proofs. Arbitrary anchor translations, other embodiments, all proposals and predictor adequacy are not covered.

**Decision/status:** do not spend further joint-only fitting on this fixed-anchor obstruction, promote production, relax the 1/2/5 mm rule or substitute lift success. A source-only audit finds the native wrist0→Shadow palm origin correspondence intentional; no missing 34 mm wrist transform is demonstrated. Prepare a new, separately frozen anatomical calibration from canonical neutral MANO MCP5/9/13 and rigid Shadow FFJ4/MFJ4/RFJ4 landmarks, independent of the mug witness/object geometry. Placement and reference-vector origins must change consistently. This is a new mapping experiment, not a coordinate-bug correction or measured repair. The current version remains FAIL.

**Next check:** review/freeze the canonical shape/pose/frame definitions before calculation, independently verify the rigid transform and all-six numerical admission, then apply the unchanged surface rule to any eligible actual new poses. No calibration result is claimed in this entry. No new HUG generator inference, physics, eval episodes/data/labels or predictor training occurred. [Completed evidence and scope](HUG_GUARDED_FIT_AND_REACHABILITY.md).


## 15 October 2026 — HUG source-only anatomical calibration independently refused

**Previous plan:** derive one canonical anatomical placement without object/witness tuning, verify consistent translated reference vectors and root compensation, and admit no pose that worsens its original individual fingertip caps.

**Completed evidence:** fixed neutral MANO MCP 5/9/13 and rigid Shadow FFJ4/MFJ4/RFJ4 frames yield one proper rigid C, with translation norm 8.668706 mm. All six new retargets pass physical bounds, native-world/source/FK reconstruction and the strict 10 mm mean-tip gate; all 120 optimizer calls return FTOL. However 21/30 original exact tip caps fail, leaving 0/6 eligible. Independent C/source/floating-root/scene reconstruction reproduces all 6 numerical passes, all 21 refusals and exact old caps/scene hashes. It inherits the separately frozen neutral MANO descriptor, without claiming an independent neutral-model forward or optimizer replay.

**Separate geometric result:** after C/output hashes are fixed, the same 7 existing cells and 8 actual palm boxes refute palm preservation at BOTH saved Mug anchors for any SO(3) orientation. Three positive rows for Mug A and two for Mug B give maximum strict excesses 7.135419/10.091852 mm after unchanged allowances; rows across resolutions may overlap. Independent 90-digit replay passes 1,093 checks. These remain conditional literal-surface/part-correspondence engineering arguments; arbitrary other translations/embodiments and physical outcome equivalence are not ruled out.

**Implementation/history:** canonical calibration is an isolated new mapping experiment, never promoted. Two pre-retarget runtime stops are retained: fused fixed forearm lookup after in-memory C computation; then explicit old-vs-Dex joint-order refusal after C export. Source-joint-chain compensation and named permutation fixes leave C/criteria unchanged, with exact V2→V3 C bytes required. Six experimental retargeted robot poses exist, but no fresh native HUG proposal, physics, eval data/labels or training. No new nine-grid audit count is claimed for these refused poses.

**Next decision:** the user has been asked whether the failure label should describe an executed common-Shadow retarget or HUG's native MANO hand. The former requires an explicit per-grasp placement policy and measured distortion; the latter requires a separate native-hand simulator/collision/dynamics path. Neither is implemented/validated here. Dependent mapping design awaits that target choice; static evidence/paper documentation is complete. Original mapping retains its 9 strict FAIL results. [Calibration report](HUG_CANONICAL_PALM_CALIBRATION.md).


## 2026-10-15 — DGA MultiDex fixed smoke exposes target-domain obstruction

Previous plan: preserve the pinned DexGraspNet smoke/cohort and test a separately identified official MultiDex checkpoint with exactly four native outputs. The user explicitly authorized this arm; no physics, evaluation data or training was authorized or run.

Evidence: [the new MultiDex bundle](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_multidex_smoke_v1/README.md) records two fixed batch2 calls, 794 strict state keys and twelve exact pre-load/post-load schedule checks. Two of four poses satisfy the source limits; none satisfies the target limits. All four THJ1 values are valid, while source LFJ1/FFJ4 failures and narrower target finger/thumb domains cause rejection. Inverse-stage checks and a separate all-four literal algebra diagnostic pass; all four default TF32 references fail, while distinct fresh FP32 references pass unchanged engineering budgets. No row qualifies for the frozen surface selection, so zero surface queries ran.

Decision/implemented status: retain every output and strict domain rejection; keep MultiDex provenance separate from DexGraspNet and preserve all earlier scientific protocols/results. No production/shared mapping, registry, target joint limits or controller changed. The original 670 evidence/production files remain byte-identical. The temporary GPU was removed after verified download; elapsed-time compute estimate about 0.14 USD is not a reconciled bill. Scientific mapping acceptance remains UNRESOLVED and execution unsupported.

Next check: an accepted unchanged-angle mapping requires native examples in the common source/target domain plus source-specific surface/contact adequacy. Retargeting into narrower target domains would be a distinct stage and needs its own frozen fidelity protocol; it is not a coordinate correction or established result. These four outputs do not estimate a validity rate or establish the original checkpoint's historical thumb-error cause.

<!-- contactgen-unseen-mapping-2026-10-15-v1 -->

## 2026-10-15 — ContactGen additional native mapping cohort completed

Previous plan: retain the one-case cube counterexample and clarify whether new
native mapping samples were allowed. The user explicitly authorized more native
grasps on unseen objects for mapping only, prohibiting physics rollouts and
outcome labels. Implemented a frozen 4 × 3 cohort: new synthetic frustum, ellipsoid,
capsule and torus, seeds 101/1009/10007 each. These are four diagnostic object
groups; ContactGen training-set novelty is UNKNOWN. They are not population or
natural-object holdouts. All 12 scheduled native attempts completed without
replacement; all 778 vertices and all 16 full frames replayed exactly in the pinned CUDA float32
runtime. The original maps on all 2,048 solver points were saved before optimization.

One registered translation per attempt yielded 11/12 canonical admissions at the
unchanged 10 mm mean-tip gate. Capsule seed 101 failed at 12.39721 mm mean/20.35977 mm
worst; its raw nominal result and descriptors were preserved. Whole-query 2 mm
unsigned native proximity retention was 28.8937–61.6275% over 11 defined native
bands; capsule seed 101's native band was empty. Strict-exterior-domain values use a
different denominator, including a 100% case based on five queries / 11.7163 mm².
All quadratures were cap-limited under the 3000-query cap (actual 2998–3000); new sampled
penetration maxima supply no continuous native upper bound. Only torus seed 10007
passed the selected target exposure guard, depth 9.0789 mm; the other 11 candidate
exposures were refused. The original exhaustive cube proof is unchanged and
separate. Native open-wrist occupancy, source-face tie ambiguity and scientific
adequacy under an external fidelity budget remain unresolved or untested.

Choice: do not repair poses, weaken gates, retry failures or promote engineering
PASS to contact/physical equivalence. The independent saved-array audit passes
8,048 checks without inference/retarget/geometry/physics calls, with its versioned
corrections and limits retained. Update only the ContactGen paper marker and
archive all methods/scalars/synthetic inputs; licensed native arrays stay private.
The paper compiled successfully at SHA 6797b8fc35b53338395e6ed1e2acc1ada2ff68e27516614c32efccb2dcba78b1.
A40 `cn4nexx8e9fxjl` was verified stopped after 148 native files were downloaded and
hash-checked. This run's elapsed compute estimate is $0.26710 excluding storage;
workspace storage remains retained. Evidence:
`research_artifacts/contactgen_unseen_mapping_2026-10-15_v1/` and the associated
private ignored run. Twelve mapping-native samples are new; no physics rollout,
outcome label, training or predictor evaluation episode was generated.

Status: the authorized diagnostic cohort is completed. Interpretation:
representation/replay succeeds, admission is 11/12, and contact preservation is
not established. Proposed next scientific check: justify an application-specific
approximation budget and representative cohort before claiming mapping adequacy.
No such acceptance budget, production mapper edit or protocol merge occurred.


## 2026-10-15 — DGA target-domain obstruction and explicitly separate retargeting

Previous plan: continue acceptance work after the four-output MultiDex test produced two source-valid but zero target-valid poses. New evidence: exact interval witnesses refute a total unchanged-angle mapping; a fixed-palm analytic lower bound forces at least 4.975267 mm index-tip error on native row3. A separately frozen constrained development fit makes rows1/3 target-valid with maximum positional errors1.397779/4.975626 mm, while preserving native records, palm and wrists. Independent FK verifies it. A visual diagnostic records a conditional 4 mm² added thumb near-region for row3; a lack of row1 witnesses does not certify equality. [Full evidence](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_retarget_development_v1/README.md).

Choice and implementation: retain unsupported registration; preserve DexGraspNet, original MultiDex protocols and every rejection. The new fit is an explicitly named development stage, not an angle correction or scientific acceptance. No shared mapper/assets/limits/registry/controller changes, new inference, physics, evaluation data or training. The remaining decision is the claim to validate: native-fidelity preservation versus a distinct DGA-plus-retargeting pipeline. Approximate contact adequacy needs prospectively justified limits and independent evidence; no threshold was selected from these observed errors. No new GPU cost.


## 2026-10-15 — ContactGen proximity-preservation refuted on the saved cohort
<!-- contactgen-contact-preservation-2026-10-15-v2 -->

**Previous plan:** close the remaining contact-preservation and failed-capsule checks using saved native outputs, with no physics or outcome-label generation.

**Evidence:** independently verified exact positive-area 2 mm proximity loss for all eleven admitted mappings (ten outside all target solids, one inside the union but separated from its exposed boundary). Capsule/101 has an exactly certified native/object separation of at least 223.516 mm, so its native proximity denominator is empty. Two fixed 80-call optimizer diagnostics never pass 10 mm; restoring the verified missing proximal scalar term leaves every candidate vector unchanged. Adaptive area estimates retain their cap limits and conditional error assumptions; all611,061leaves and all24,576originalnodes passed separate saved-array read-back.

**Decision and implemented status:** reject the existing mapping as exactly preserving the frozen geometric 2 mm set for these eleven cases. Keep the capsule admission failure and empty denominator explicit. Preserve all original outputs and nominal mappings; do not replace production retargeting or treat these diagnostics as physical outcomes. Paper and validation report updated; public-safe evidence archived at `research_artifacts/contactgen_contact_validation_2026-10-15_v2` with private geometry retained under the corresponding ignored run. All271originalfiles rehashed unchanged. No GPU resume, new native inference, physics or outcome labels.

**Next check / limitation:** any approximate-use acceptance requires a prospective application-specific tolerance. A contact-aware replacement mapping or native-optimizer repair would be a distinct development protocol requiring fresh validation; current evidence does not prove kinematic impossibility, physical success/failure, or population generalization.


## 2026-10-15 — DGA free-palm correction improves maximum error but does not preserve native geometry

Previous plan: correct remaining DGA mapping defects following fixed-palm fits and an exact fixed-palm reachability obstruction. The user asked to continue toward accurate acceptance, including a distinct DGA-plus-retargeter stage, without relaxing criteria. Implemented: frozen two-start minimax/SSE-guarded fitting on only the existing two source-valid MultiDex development poses, preserving both wrists and target limits. Row1retainsbaseline; row 3maximum decreases4.975626→1.797795 mm while its palm moves1.797795 mm/1.796291degrees and22 of 28site errors increase. Independent FK/selection review passes. New evidence: thumbvisual reconstruction is correct within 33.1 nm, but literal distal shapes have disjoint certified diameter intervals; fresh/finer static surface checks detect positive lost/added nearregions, and an independent declared palm boxunion check detects added proximity/occupancy in bothrows. [Full evidence](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_free_palm_v1/README.md).

Choice/status: the candidate is useful development evidence but full mapping remains unaccepted and unsupported. Distinct pipeline identity does not waive native fidelity. No scientific DGA approximation budget is inferred from measured residuals. Next research check needs an explicit native-relative surface objective and independently justified prospective approximation claim before unseen-output validation. Preserve all original checkpoints/protocols/outputs/rejections/default TF32 failures and previous fits. No production mapper, hand, limits, registry or controller edits, inference/GPU, physics, evaluation data/labels or training. This supersedes the earlier pending scope question through the user's continued authorization, without claiming the remaining scientific choice is resolved.


<!-- hug-free-palm-2026-10-15-v1 -->

## 15 October 2026 — HUG per-grasp palm translation tested under original criteria

**Previous plan/authorization:** the user chose to keep the common Shadow Hand and test per-grasp palm placement on existing HUG grasps, retaining original accuracy criteria and prohibiting new eval data. This resolves the prior embodiment choice; the eventual task label describes the executed Shadow retarget, without claiming native MANO outcome equivalence.

**Completed evidence:** a frozen 27-coordinate bounded fit performs 762 full objective callbacks; one separately frozen fine study completes all 56 seeds per case with 348 callbacks. Both retain all six exact original guarded outputs. Independent review of all 1,110 records and selected poses passes 98,184 checks. Six diagnostic alternatives preserve all 30 original literal fingertip caps, but fresh unchanged full-grid measurement gives 9 strict/proximity FAIL and 7 pointwise-orientation FAIL/2 UNRESOLVED. Three selected shifts are only 1.165–3.284 nm from restoration of thumb-joint requests, so one separate seed-origin coverage supplement is frozen before ranking. Its actual shifts are 0.063565/1.994976/0.031372 mm; all 15 repeated original caps pass, but all 5 fresh surface comparisons FAIL. The other three deliberate-seed poses shift 3.716–4.892 mm and also fail. These are 14 comparisons of nine diagnostic robot poses derived from six existing native grasps, not new native proposals or physical labels. Both audit protocols and every original guard refusal are preserved.

**Decision/status:** retain Shadow and the original frozen scientific criteria; do not promote a candidate or alter production. These finite negative development studies do not prove that every possible placement fails. Positive whole-cell palm/finger discrepancies and reliable orientation witnesses remain; original cap preservation is no proof of absolute fidelity. Full continuous orientation coverage also remains unresolved.

**Proposed next check:** freeze a new development method that constrains the actual recorded surface counterexamples while retaining tip/domain gates, then independently audit the complete fields and remaining coverage uncertainty. This proposal is not implemented and feasibility is unknown. No HUG inference, physics integration, eval data/labels, training or GPU provisioning occurred. [Full study and evidence](HUG_FREE_PALM_MAPPING_STUDY.md).


## 2026-10-15 — ContactGen repair improves geometry but does not establish preservation
<!-- contactgen-mapping-repair-2026-10-15-v1 -->

Previous plan: repair the eleven contact-preservation failures and the failed capsule using saved native data, then revalidate without physics or outcome labels.

The off-object capsule is repaired as an explicitly derived rigid native candidate: exact verification establishes a positive-area native 2 mm proximity set and ≤1.001 mm penetration across all 1,538 native triangles. Its final robot mapping passes both literal 10 mm fingertip gates (9.9900 mm each). All twelve final mappings—eleven unchanged native identities and the derived capsule—pass both fingertip gates and physical joint bounds, but every one still has an independently verified positive-area lost-proximity patch. Exact 2 mm preservation remains FAIL; no research candidate replaces the canonical mapper.

For the eleven unchanged native hands, unsigned individual-boundary retention improves in all cases, from 28.84–60.59% (median 46.08%) to 59.04–96.31% (median 79.59%). This descriptor may include buried interfaces. The derived capsule has 78.61% on that domain and 71.88% common-exterior retention over a quadrature-estimated 206.00 mm² native-near denominator. Its exact lost patch is at least 0.0245042 mm², with native distance at most 1.251624 mm and robot-union distance at least 2.832153 mm. These patch lower bounds disprove universal preservation, not total-area change or global mapping feasibility.

Implemented: an exact rational AABB preflight rejects a provably empty native 2 mm band before retargeting; only the original capsule/101 is rejected among the twelve saved cohort inputs. Overlap remains inconclusive. The existing canonical mapping algorithm and 10 mm gate are unchanged; metadata explicitly declines contact-preservation validation. Focused guard/provider tests pass 71 cases. The private cloud-kit packager includes the new module, and its isolated declared-file import passes.

Research implementation: 72 retained contact-aware static fits across weighted, minimax and wider-domain schedules. The last schedule expands to physical finger limits and per-component palm corrections ±60 mm / ±0.7 rad; the original validation gates remain unchanged. Independent geometry checks use 107,973 query evaluations per stage at locations disjoint from fitting centroids, with the same object locations reused across seeds. This is adaptive development, not blind holdout validation. Literal fitting margins, failed solver terminations, 19 capsule source optimizer attempts and unadopted articulated candidates remain recorded. No separate self-collision or normal-preservation acceptance budget is enforced.

The selected capsule's source UV loss increases by 7.739e-7, within numerical tolerance but above its nominal bound, and its solver reports line-search failure. Exact geometric certificates do not erase that caveat or establish a physically valid grasp. All 148 original native-download and 123 static-output files remain byte-identical. The authentic native inventory remains thirteen; derived repairs are not new independent model samples. No GPU resume, ContactGen inference, physics, outcome labels, predictor evaluation or training occurred.

Private outputs: `runs/contactgen_mapping_repair_2026-10-15_v1/`. Public source, protocols, scalar comparisons, exact-proof receipts and inventories: [repair evidence](../research_artifacts/contactgen_mapping_repair_2026-10-15_v1/README.md). Licensed geometry, full poses, per-query arrays and exact private certificate coordinates remain excluded. The active ContactGen paper section records the improvements and remaining failures; historical protocols and other provider sections are preserved.

Decision and implementation status: retain the new conservative native rejection guard, retain all derived source/mapping candidates as research evidence, and decline promotion of the contact-aware mapper because all twelve final mappings still fail the frozen exact 2 mm criterion. The capsule source defect and its derived engineering admission are repaired, while contact preservation is not.

Next check (proposed only): a separately frozen solver could constrain actual exposed-union counterexample regions and certify continuous coverage if a feasible candidate is found. Such fitting would use development evidence and require new disjoint verification. Global infeasibility has not been proved. Approximate application use would additionally need a prospective justified fidelity budget; neither a relaxed criterion nor physical evaluation is implemented here.


## 2026-10-15 — DGA surface-preserving objective and prospective strict criteria tested

Previous plan: correct the free-palm candidate's native surface discrepancies. The user explicitly authorized implementation and criteria before testing. Implemented: a separately frozen, native-relative objective on seven true-triangle distance fields and signed declared palm-union distance, two fixed starts/two sweeps on existing rows1/3, immutable native poses, strict target joints/wrists and terminal landmark maximum/SSE nonregression. The prospective scientific rule tests exact same-part near-region equality at established0.5/1/2mm scales and palm occupancy equality; positive conservative lower differences FAIL, zero full upper differences/geometry proof PASS, remaining uncertainty UNRESOLVED. No data-derived nonzero tolerance or physical adequacy claim was introduced.

Evidence: all four terminal fits reduce loss37.08–60.65% but violate both landmark guards, retaining prior baselines. Fresh1/.5/.25mm grids yield row1visual UNRESOLVED and row3FAIL; both declared palm near-region/occupancy comparisons FAIL. Finest2mm newpalm-near bounds133.9375–167/424.875–535.5mm²; added occupancy65–82.5625/384.375–443.3125mm². Direction lacks complete coverage. Independent fitting/FK, saved-array recounts and frozen-dependency checks pass; all1184 historical evidence/production files unchanged. The all-triangle reader corrects tiny/degenerate distance handling additively without rewriting earlier measurements. [Complete evidence](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_surface_preserving_v1/README.md).

Choice/status: no accepted correction or provider promotion; original DexGraspNet/MultiDex and every failed attempt remain separate. Terminal-only landmark guards expose a method limitation, not proof that all constrained corrections are impossible. Next proposed check: prospectively freeze a method that maintains landmark feasibility throughout fitting and covers finer/uncapped native-relative fields, followed by fresh validation. Feasibility and independent native generalization remain unknown. No new inference, GPU operations, physics, evaluation data/outcome labels or model training occurred.


## 2026-10-15 — DGA feasible-step correction tested under unchanged native-fidelity criteria

Previous plan: terminal-only landmark guards rejected all surface-aware corrections. User decision: preserve native DGA grasp fidelity; keep the retargeter unaccepted while investigating alternatives. Implemented and frozen before fitting: a bounded coupled-step method that checks eleven positive fractions against strict domains, original wrists and true 28-site maximum/SSE guards before line-search surface queries. All feasible fractions compete with an exact cached zero step. Proposal workspaces are separately labeled and need not be feasible. All four attempts/48 blocks completed; 88 positive checks yielded 37 guard refusals, 51 field evaluations and eight feasible adopted steps.

Evidence: fitting loss decreases 5.75%/20.82%; selected maximum errors are 1.311337/4.906432 mm. The guard reference is the older fixed-palm fit, so row3 worsens from the later free-palm maximum 1.797795 mm (172.91% increase) and SSE rises 14.77%; row1 mean rises 68.00%. Fresh reserved visual and palm queries preserve the original scientific rule. Row1 visual equality remains UNRESOLVED; row3 FAIL. Both declared palm near-regions and occupancy FAIL; finest 2 mm added-near bounds are 129.5625–161.75/395.875–505.8125 mm², added occupancy 59.875–76.9375/322.75–377.75 mm². Direction remains UNRESOLVED. Independent fit/FK, saved-array recounts and frozen-dependency checks pass, confirming the reported failures. All 1,467 historical files are unchanged. [Complete evidence](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_feasible_surface_v1/README.md).

Choice/status: no accepted mapping or shared-provider promotion. No changed thresholds, source repairs or historical-result replacement. Known rigid thumb-shape noncongruence is distinct from unproved same-object near-region feasibility. Next proposed investigation: an explicitly separate representation preserving the pinned source geometry, kinematics, joint domains and raw output, with identity/correspondence proofs; it would be a different embodiment and cannot validate the existing fixed target by substitution. Lower fitting loss alone is insufficient. No new inference/GPU, physics, evaluation data/outcome labels or model training occurred.

<!-- contactgen-counterexample-repair-2026-10-15-v1-stage1 -->
### 2026-10-15 — ContactGen: exposed witnesses, continuous counterexamples, and cross-digit feasibility

**Previous plan.** Repair the remaining contact-preservation failures without changing the saved native outputs, the two 10 mm mean-tip gates, the 2 mm proximity property, or the prohibition on physics/outcome data. Earlier v3 contact-aware fits passed the engineering gates but all twelve retained exact lost patches.

**New measured evidence.** A hand-only static collision audit of twelve original and twelve previous repaired poses reports cross-digit intersections in 4/12 and 6/12 respectively. The constant wrist–forearm attachment overlap is classified separately. Independent exact affine arithmetic proves a 2 mm-side cube lies strictly inside both ring and little proximal cylinders of the prior derived-capsule target pose (intersection volume at least 8 mm³). This is collision-envelope geometry, not a physical outcome or an exact penetration-depth estimate.

An exposed-witness pilot for ellipsoid/10007 retains the original native identity and passes the unchanged engineering gates after 206 optimizer iterations. All 2,569 development points have numerical exposed witnesses within 2 mm, and an 8,997-query independent audit observes no 2 mm loss. Nevertheless, the continuous search finds a missed strip. A separate rational verifier certifies a positive-area lost patch ≥4.84712e−7 mm², with native distance ≤1.997889893 mm and target-union boundary distance ≥2.002050957 mm; four tampered certificates are rejected. The pilot also introduces seven numerically reported middle–ring overlaps, up to 0.224799 mm. It is not adopted.

**Decision and implementation status.** Treat sampled retention, exact one-way inclusion, physical joint limits, both engineering gates, and cross-digit geometry as separate properties. The exact continuous-cover builder and independent verifier are implemented and control-tested; the first bounded search is UNRESOLVED, while the exact lost patch independently establishes FAIL for that fixed pilot. Add source-band contour points and these proved counterexamples to a separately frozen development protocol. Add all 359 cross-digit support-gap constraints and independent exact pair certificates; the revised pilot is in progress. A native collision-distance API crashed during isolated tests, before optimization, so the replacement uses geometric support bounds rather than that API. No threshold was relaxed, no shared mapper was replaced, and no generator-wide or physical-validity claim is made.

**Preservation and next check.** All 148 native-download files and 123 original static-output files rehash unchanged. Keep the prior capsule source repair and every failed target pose distinct. Independently recheck the revised candidate on fresh queries, complete continuous surface coverage or produce a counterexample, and certify every cross-digit pair before considering it usable. The 2 mm one-way property is still a named geometric research condition; an externally justified application-error budget and generator-wide generalization remain unestablished. No GPU resume, ContactGen inference, physics steps, outcome labels, predictor evaluation or training occurred. The ContactGen-only paper update compiled successfully; other sections were byte-preserved.

Evidence: `research_artifacts/contactgen_counterexample_repair_2026-10-15_v1/` (public scalar/source archive); private geometry and raw certificate coordinates remain under the ignored run of the same name in the primary checkout. Earlier protocols and results remain unchanged.

<!-- contactgen-counterexample-repair-2026-10-15-v1-stage2 -->
### 2026-10-15 — First ContactGen case with independent continuous and cross-digit proofs

**Previous plan and measured result.** Add native-band boundary points and cross-digit separation constraints to the exposed-witness pilot. Ellipsoid/10007 v2 now passes the unchanged physical joint bounds and both 10 mm gates (5.897459984 mm object-frame mean; 9.990000 mm original operator-reference mean). The independent rational verifier checks a complete partition of all 1,280 object faces into 2,763 source-far and 1,049 target-ball leaves, with 562 exposed boundary witnesses and zero pending cells. A separate support verifier proves all 359 specified cross-digit collider pairs have clearance at least 0.010 mm. The previously certified positive native-near patch is reused only after exact source-array equality and identical object hashes, preventing a vacuous inclusion claim. Combined evidence: `validation/pilot_v2_scoped_property_receipt.json` in the new run/archive.

**Choice and implementation status.** This repairs the stated static properties for one unchanged native identity; it is not universal generator or physical validity. Keep the canonical mapper unchanged and test other saved cases separately. The derived capsule's six frozen extension rounds produced no eligible candidate (423 iterations; all local line-search failures). Static collision read-back still reports its ring/little overlap at every round. A separately frozen, controlled multi-start feasibility stage is now testing fingertip and separation constraints before contact fitting; no contact-preservation claim follows from feasibility alone. Thresholds and original data remain unchanged.

**Additional source evidence and next check.** Separate exact predicates and two independent verification implementations establish transverse intersections between nonadjacent native MANO triangles: an interior segment ≥1.315428 mm in both original and rigid-derived capsule sources, and ≥0.035722 mm in the original ellipsoid source. These are sheet-geometry descriptors, not material occupancy, physical failure labels, anatomy guarantees, or a new acceptance budget. They do not invalidate the scoped ellipsoid proximity/separation proofs, but preclude a claim that its source is self-intersection-free. Next: complete the capsule feasibility schedule, perform strict contact fitting only on an eligible predetermined seed, then independently verify the same fixed criteria. No GPU, new inference, physics, outcome labels or predictor work was used.


<!-- hug-direct-surface-constraints-2026-10-15-v1 -->

## 15 October 2026 — HUG direct surface constraints tested; strict mapping still fails

Previous plan: replace average-fit tradeoffs with direct constraints on failing palm/finger regions, keeping Shadow and the original criteria. Implemented two separately frozen bounded fits on the same six saved HUG grasps: 178 retained proximity examples and 134 signed-orientation records, with all original tip caps, mean guard, native targets and domains fixed. V2 removes only a flat far-distance optimizer residual. Complete traces preserve 20,526 raw executions and 112 exact checks; independent source/FK/arithmetic reviews pass. V2's twelve terminal states all exceed an original individual tip cap and are refused. Fitting used four Numba threads per worker rather than the declared two; this resource deviation is preserved, not rewritten away.

A predeclared, independently replayed selection retains V1 in all six cases. Mug A's new 22.359525 mm palm displacement satisfies 10/57 retained proximity rows, but the fresh nine-grid audit gives nine proximity/strict FAIL results and 9 FAIL for pointwise orientation. The mug's changed palm bound accompanies increased thumb mismatches and newly observed direction reversals. All thirty individual caps and six mean guards pass; numerical correctness is not surface acceptance. A separate literal identity control demonstrates uncertainty in the independent interval certificate, not actual mismatch or global infeasibility.

Choice: preserve the production mapper and frozen scientific criteria; no mapping sign-off. Next proposed work is feasibility-restored surface search and stronger continuous coverage evidence, not weaker acceptance limits. Those methods are not implemented by this study. No new native inference, physics rollout, eval data, outcome labels or training. [Completed report](HUG_SURFACE_CONSTRAINT_MAPPING_STUDY.md) and five separately versioned evidence bundles preserve successes, refusals and tradeoffs.

<!-- contactgen-counterexample-repair-2026-10-15-v1-stage3 -->
## 15 October 2026 — Capsule feasibility separated from contact preservation

Previous plan: repair the capsule's deep cross-digit overlap before contact fitting. Implemented: a frozen three-start feasibility search found a candidate satisfying physical joint bounds and both original 10 mm mean-tip gates (9.979997/9.980000 mm); independent exact checks prove all 359 cross-digit pairs separated by at least 0.423061 mm. The feasible reference/separation example disproves an inference of subproblem impossibility from the prior optimizer failures. It does not establish contact preservation or whole-hand/physical feasibility.

New evidence: six strict-contact continuation rounds (1,351 optimizer iterations) produced a predetermined diagnostic candidate that passes the original tip gates but has an exact positive-area proximity loss. An independently verified patch has area ≥0.0753785 mm², native distance ≤1.637834618 mm, and target-union-boundary distance ≥2.193980841 mm. The fixed rigid-derived native source and original operator references are independently rechecked. The previous capsule and all unsuccessful poses remain saved.

Decision: retain this exact FAIL and test a separately frozen short-refresh strategy; two alternate separating directions on an earlier fixed pose have independently verified gaps ≥1.943448 and ≥0.647143 mm, compared with approximately 0.010 mm under the old directions. This demonstrates conservative fixed-direction constraints at that pose, not global contact-map feasibility. The remaining ten original nonempty cohort cases are now running under a reviewed prospective route: original nominal eligibility, otherwise fixed three-start restoration, followed by strict contact fitting. Final acceptance still requires independent geometric proof; diagnostic endpoint selection is not acceptance.

Evidence: `research_artifacts/contactgen_counterexample_repair_2026-10-15_v1/validation/capsule_feasibility_v4_scoped_receipt.json`, `validation/capsule_strict_v5_scoped_receipt.json`, and method reviews. Source data, 10 mm engineering gates and 2 mm property are unchanged. This stage uses no GPU, fresh native inference, physics, outcome labels, predictor evaluation or training. Next check: complete per-case frozen fitting and independent proofs, then the versioned capsule retry.

<!-- contactgen-counterexample-repair-2026-10-15-v1-stage4 -->
## 15 October 2026 — Adaptive search revision before completing the cohort

Previous plan: run the remaining ten cases with nominal-first feasibility routing and six long contact blocks. New evidence from the first case, capsule/10007: two fully saved blocks consumed 480 optimizer iterations and 435.77 seconds, with maximum attached-witness distances 11.56966 then 8.91082 mm. Long within-block plateaus followed by improvement after refreshing witnesses/directions indicate inefficient conservative subproblems. The gate-only seed rule also selected a lower-tip-error seed while another restored seed satisfied the same guards; their contact scores had not been compared, so the alternative is not retrospectively claimed better.

Implemented decision: stop that worker and scheduler safely after all second-block outputs were saved and checked. V6 is explicitly EARLY_STOPPED_ADAPTIVE_METHOD_REVISION for its first case; the other nine are NOT_RUN. No skipped attempt becomes a PASS or a completed cohort. Its protocol, stopped outputs, restoration attempts, logs and stop receipt remain immutable.

Freeze V8 as a new adaptive development screen for all ten inputs, using the same original native identities, original references, joint limits and 10 mm/2 mm criteria. Compare a bounded pool of at most four feasible seeds on identical per-case development queries: nominal plus a declared feasible terminal from each of the same three restoration starts. Rank by contact loss/unresolved count, exposed-witness distance and excess, with fixed tie-breaks. Exclude the stopped V6 fitted poses. Refresh witnesses and separating directions every 40 iterations, with at most 12 blocks (480 iterations) per case. A completed-block checkpoint starts no further contact block once elapsed contact fitting reaches 300 seconds; the active block finishes, so 300 seconds is not a hard wall-time cap. Restoration and seed-scoring time are reported separately. All ten receive the same allocation.

Protocol hash: `f78fd337139d88c832718f33ee14ea0e16d853cdf4cde6b6b51317fde6285cc8`. Independent method review precedes execution. Numerical acceptance candidates are distinct from diagnostic endpoints, and final acceptance requires independent continuous geometric proof with the hardened certificate-file binder. A search cap or optimizer failure never proves infeasibility. A separately frozen capsule/101 short-refresh retry remains a distinct derived-source study. No fresh inference, GPU action, physics, outcome labels or training is introduced. Next check: retain actual per-case budgets and obtain exact positive certificates, exact counterexamples, or explicit unresolved statuses.

<!-- hug-criteria-alignment-restoration-2026-10-15-v1 -->

## 15 October 2026 — HUG criteria provenance and saved-endpoint feasibility restoration

**Previous plan.** Keep the Shadow Hand, preserve the original strict mapping criteria and improve the six existing HUG development grasps before any new evaluation data. The direct surface fit left twelve V2 terminal poses slightly outside individual fingertip limits. The user explicitly authorized committing and publishing the work to the existing GitHub branch; prior commit `31a7335737fca0e7ea5b5b707b781020d6ca8006` was pushed successfully.

**New evidence.** Independent local provenance and primary-source reviews distinguish HUG's generator evaluation from our mapping contract. The mean 10 mm native-target guard, original individual nonworsening caps, 1/2/5 mm bands and zero changed relationship criterion are project choices, not HUG-prescribed Shadow tolerances. The official source snapshot is pinned; no released benchmark evaluator was verified. Historical strict failures remain valid for their named claim and do not establish universal practical unusability of a Shadow retargeter.

A separately frozen bounded FK-only restoration reuses all twelve saved V2 endpoints. All twelve acquire exact-tip-feasible checkpoints. None improves the existing surface diagnostic rank; ten lose protected direction eligibility and six lose protected passing records. Mug A endpoint 0 improves proximity passes 10/57 to 11/57 but worsens the maximum normalized violation 5.000 to 5.445343. Invalid direction evidence is not a demonstrated reversal, and previously infeasible endpoint directions were not measured, so restoration is not identified as their cause. All six pooled reference poses are retained. There were 1,101 tip-FK and 161 analytic-Jacobian callbacks, 2,524 actual projection forwards and 24 retained-surface checks; zero fresh coarse/fine grids. The previous nine FAIL results are prior evidence for unchanged poses.

**Decision and implementation.** Keep the historical criteria and production mapper unchanged. Publish the criteria provenance, bounded result and explicit scope correction. The user-selected prediction target remains the executed retargeted Shadow grasp. A practical adequacy protocol is design-only: no new threshold, calibration cohort or acceptance decision has been chosen. Derive future distortion allowances from independent evidence rather than making these development cases pass. Preserve poor source grasps, refusals and mapper versions.

**Next check.** Verify the exposed geometry reference, then prospectively justify a distortion budget with separate calibration/sensitivity evidence under a fixed controller and later held-out testing. Improving literal surface preservation remains measurable work, but a stronger optimizer cannot itself justify a scientific acceptance tolerance. No inference, physics episodes, evaluation labels or training were generated. See [criteria alignment](HUG_CRITERIA_ALIGNMENT.md) and [restoration evidence](../research_artifacts/hug_feasibility_restoration_2026-10-15_v1/REPORT.md).

<!-- dga-component-constraints-independent-native-2026-10-15-v1 -->
## 15 October 2026 — DGA fixed-target component repair, feasibility certificate and independent native test

**Previous plan.** Improve the mapping to the agreed target Shadow hand without trading landmark accuracy for lower aggregate surface loss. Preserve the original scientific criteria, old outputs and the prohibition on physics/evaluation data/training. Test new native outputs before any acceptance decision.

**Implemented method.** A prospectively frozen corrector enforces eighteen separate nonregressing geometry metrics, strict target domains, unchanged wrists, independent frames and tighter fixed/free maximum/SSE plus free-mean landmark guards. Its 856 fitting queries include 256 historical witnesses classified as development data. Method, consumers, selection and scientific protocol were bound before unsealing a separate sixteen-output MultiDex cohort. No scientific tolerance or target change was introduced.

**New analytic evidence and choice.** Exact-rational rigid-anchor inequalities certify a sufficient declared-palm occupancy obstruction for old development row 1 under its original landmark guards, allowing arbitrary proper target palm placement and finger articulation. With the declared 1 nm anchor/geometry bridge and boundary cushion, a positive mismatch square is at least 0.000104920398 mm²; avoiding the implicated box requires total squared landmark error at least 3.737504620 mm² against 3.620809006 mm² allowed. Two independent implementations reproduce the proof. Old row 3 remains unresolved under its original guards; its tighter new domain has a separate obstruction. These are conditional declared-box/pose statements, not compiled/physical/global-generator impossibility. Further old-development fitting was NOT_RUN_ANALYTIC_OBSTRUCTION rather than an attempted optimization reported as failed.

**New measured results.** Eight native calls at seeds 20261025–20261032 produce sixteen complete finite traces: 11 source-valid, five source-invalid, ten source-valid/target-invalid, one unchanged-angle candidate. All default TF32 failures and separate FP32 passes remain explicit. The first two source-valid outputs in original order, new rows 0/1, are selected before all later gates with no backfill. Correction completes 24 blocks/471 trial field states/eight adoptions in 14.2 seconds. Landmark maxima stay 1.203120/1.498822 mm. Row 0 changes are subnanometre; row 1 index maximum field error improves 0.890514→0.743128 mm and landmark SSE falls about 7.95%. Neither palm changes.

Fresh independent 1/0.5/0.25 mm geometry tests at fractions (2/5,1/3) perform 18,415,400 triangle tests and 115,290,000 polygon queries. Strict visual preservation FAILS for both frozen corrected outputs. New row 0 also fails declared-palm preservation: finest 2 mm added-near 437.4375–553.6875 mm² and added occupancy 724.0625–786.625 mm². New row 1 has certified empty palm near/occupied sets on both sides but still fails thumb/finger visual preservation; palm directions remain unresolved for both. These new-pose failures do not prove all corrections impossible. Independent raw/FK/adoption reviews and 4,290 visual/519 palm numeric recounts pass. Preserved matmul warnings coexist with finite explicit-arithmetic transform agreement; the warning cause and complete directed-rounding bounds remain unestablished.

**Decision/status and next gate.** Keep the mapping unaccepted and shared registry unsupported. A map covering the obstructed old pose cannot satisfy both unchanged declared-palm occupancy equality and original landmark guards under the stated premises; repeating an acceptance-seeking fit there is not justified. Any future method needs a distinct prospective protocol and independent evidence, with no threshold inferred from these failures. The sixteen new seeds share one object and only two receive correction; no cohort-wide or unseen-object generalization is claimed. All 1,929 older files remain unchanged. The newly owned A40 was stopped then deleted; no physics, evaluation episodes/outcome labels or training ran. [Complete evidence](../research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_component_constraints_v1/README.md).

<!-- hug-exposed-geometry-2026-10-15-v1 -->

## 15 October 2026 — HUG compiled geometry verified; exposed-boundary counterexamples persist

**Previous plan.** Diagnose whether buried collision-component interfaces or an incorrect simulator representation explain the strict mapping failures before another fitting search. Preserve Shadow, original criteria, source poor proposals and the prohibition on new eval data.

**New evidence.** Independent MuJoCo 3.3.7 inspection of six saved scenes confirms 45 compatible hand colliders each, the compiled mesh hulls/transforms/filtering and zero contact margins/gaps; no importer or collider correction was identified. A separately frozen diagnostic (`3697aff0c70d1be747f355606fb3c0179f282c661ebd3ee289e48de09c337a6a`) checks existing 178 proximity and 134 orientation-site rows at original and pooled poses: 624 repeated records, 575 exposed nearest-part witnesses and 49 buried. All six have surviving conditional whole-cell proximity counterexamples at both poses (156 original,114 pooled). Those counts are not an improvement score or whole-surface coverage; unknown persistence is not repair. Selected milk-box B retains a little-finger cell with native upper distance 1.691 mm and exposed-Shadow lower distance 8.144 mm. Fifteen primary and ten independent controls pass; the independent output review replays hashes/poses/records/bounds without new geometry.

**Decision and scope.** The original nine full-grid FAILs remain intact and were not rerun. Removing buried surfaces cannot repair a definite lost-near relationship because the exposed same-part subset is smaller. Continue to distinguish correct decoding/declared pose realization from native surface preservation and empirical prediction adequacy. The user-selected executed Shadow target needs no universal morphology cutoff merely to define its outcome; any optional approximate-native-fidelity claim needs a justified prospective budget. No such budget or practical PASS is invented. All 32 well-bound saved proposals are development evidence, not an untouched object test set.

**Implemented.** Bounded diagnostic, collision-contract receipt, current scoped evidence manifest, prospective record/protocol design, manuscript and handoff update. 36 static forwards, 84,240 component-point computations, zero physics integration, new grids, fits, inference, eval episodes, outcome labels or training. Production and original acceptance unchanged. [Current report](HUG_EXPOSED_GEOMETRY_AUDIT.md), [diagnostic](../research_artifacts/hug_exposed_geometry_2026-10-15_v1/REPORT.md), [prospective design](../research_artifacts/hug_practical_validation_design_2026-10-15_v1/REPORT.md).

**Next check, proposed rather than completed.** Freeze a continuous, explicitly object-relative baseline/candidate measurement and finite retargeting objective using defensible exposed-region correspondences. Test on saved development inputs with original fingertip/domain guards and all refusals retained; avoid another unchanged broad search or post-hoc acceptance. Independent object generalization, dynamic tracking and prediction adequacy remain future evidence requiring later authorization for new data.


<!-- compiled-hand-conversion-2026-10-15-v1 -->
## 15 October 2026 — Declared hand versus compiled simulator conversion repaired

**Previous plan.** Preserve the agreed target and all scientific criteria while comparing each native hand/pose with the actual compiled MuJoCo model, before further retargeting or evaluation. Native/source failures and original results must remain separate.

**Evidence.** Independent source and target FK checks covered 32 HUG, 13 ContactGen, 32 UniDexGrasp raw plus 32 prior frame derivatives, 246 pinned official DexGraspNet replay records, and 16 DGA raw plus two saved corrections. These total 373 saved representations, not 373 independent native outputs. The independent compiled oracle added three static controls and checked 30 links/45 colliders in three model stages. It found a basename alias: middle/ring/little distal collision files (526 vertices) had been replaced by the index's visual mesh (2,691). It also measured joint-range rounding up to 3.67321e-6 rad and five false compiled-limit classifications. Native adapter sign/order/unit/root checks found no new adapter defect; different local thumb axes alone do not justify sign changes.

**Implemented choice.** Import unchanged mesh bytes under unambiguous virtual names; retain live MjSpec through scene/actuator construction, preserving fused-tip masses and declared full-precision parameters. New replay/render exports use exact MJB; legacy XML reading remains. No target asset, provider gate, scientific protocol or controller tuning changed. Exact model files and source snapshots distinguish the corrected implementation. The native source models and target morphology remain different.

**Verification/status.** Twenty static regression cases pass with physics and runner execution blocked. Across all saved cases and controls, corrected FK residual is <=4.05e-16 m; joint ranges and analytic sizes are exact, and all five false limit classifications disappear. Final production MJBs are byte-identical to the independently audited after snapshots. All 11,037 pre-existing inventoried evidence/target files remain unchanged. No physics, inference, fitting, evaluation data/labels or training ran. This accepts only the bounded conversion checks; no provider or surface-preservation failure is promoted.

**Scope correction/next check.** Earlier compiled-surface reports measured their historical imported geometries; they did not prove intended collision-file identity. Preserve those results and independently revalidate affected provider-specific geometry on the corrected compiler with unchanged criteria. DGA's palm difference and conditional landmark/occupancy obstruction remain distinct; evaluation is not yet justified by this conversion audit alone. [Complete evidence](../research_artifacts/compiled_hand_mapping_audit_2026-10-15_v1/README.md).

<!-- hug-mapping-process-repair-2026-10-15-v1 -->

## 15 October 2026 — HUG numerical and search-policy defects repaired; two protected steps improve severity

Previous plan: diagnose surviving exposed surface differences under the original HUG→Shadow rules, without generating eval data. The code trace now identifies the modeling mismatch: production fits ten hand vectors without object geometry, while acceptance requires object-relative surfaces. The earlier surface fitter also fixed root orientation. Two measured numerical defects are separately repaired in the opt-in `independent_static_shadow_vector_v2`: proposal-state leakage (saved A–B–A maximum 11.02 degrees) and a scalar/gradient inconsistency. Corrected repeats are bitwise identical,16 focused tests pass, and failure statuses are explicit. Four of ten source-FK fingertip errors worsen, so this reliable hand-only variant is not promoted as a surface-accepted replacement.

Implemented and measured: full 30D joint/palm-translation/root-rotation search with unchanged caps/domain/surface gates;13 kinematic and 8 solver controls pass. Its bounded case20/case28 pilot retains pooled poses. Source/trace review then identifies every surface margin being relaxed by epigraph slack although selection protects previous passing rows. A new hard-row policy fixes that search mismatch, retains inward search buffers and passes16 helper and 8 independent solver controls. A separate 25-iteration fit still ends infeasible and retains prior poses; this is not global infeasibility evidence.

The final fixed dyadic segment check selects 1/16 steps for both saved endpoints, satisfies all 10 original tip caps and preserves all previously passing/eligible retained rows. Worst normalized severity 9.283423→9.104737 and 8.002982→7.573550; no failed row becomes a pass. Remaining proximity/orientation failures 12/13 and22/1 prevent sign-off. Original historical nine strict FAILs are unchanged; no new full grids are asserted. A proposed four-endpoint projection was canceled before execution. A shared loader edit was isolated through an explicit historical-module overlay, preserving concurrent work and protocol boundaries.

Choice/next check: retain Shadow, original scientific criteria and historical default. Implement feasible protected step acceptance inside the object-aware solver rather than interpreting finite optimizer stopping as infeasibility or accepting altered thresholds. The bounded step improvement is development evidence, not a general algorithm or untouched-object validation. Native/Shadow outcome equivalence is not claimed. No inference, physics integration, eval episodes/labels or training. Evidence and scope: [HUG_MAPPING_PROCESS_REPAIR.md](HUG_MAPPING_PROCESS_REPAIR.md), [archived protocols/results/reviews](../research_artifacts/hug_mapping_process_repair_2026-10-15_v1/). User authorization to commit/publish on codex/project-work persists.


<!-- contactgen-counterexample-repair-2026-10-15-v1-final -->
### 2026-10-15 — Complete the bounded ContactGen contact and separation repair screen

**Previous plan.** Extend the first exactly certified ellipsoid repair to the remaining saved nonempty inputs, including the separately derived capsule, without changing the literal 10 mm gates or the exact 2 mm contact property.

**New evidence and implemented result.** The completed retained cohort has **2 scoped geometric passes, 9 exact contact-preservation failures, and one original empty-source refusal** among twelve authentic native identities. The rigid-derived capsule is a separate additional contact failure. Across twelve nonempty target rows, 2 satisfy the combined scoped properties and 10 have independently verified positive-area 2 mm loss counterexamples. The historical cube is excluded. Diagnostic endpoints are not relabeled numerical acceptance candidates; cross-digit separation is `NOT_RUN` after a decisive contact failure.

The one-way property concerns the original conditioning surface and exposed boundary of the recorded robot-collider union. It does not establish reverse equality, anatomy, normals, graph-ray equivalence, physical success, or generator-wide generalization. The original capsule remains refused, and its rigid repair remains a different source lineage. The certified native surface crossings cannot be removed by target retargeting or a common invertible rigid source transform.

All ten cases in V8's separate screen have terminal outcomes. The earlier V6 stop (two contact blocks on one case, nine unrun) is preserved. Four V8 pre-fit dependency refusals caused by concurrent shared-source edits remain setup outcomes. A reviewed full sixteen-module package snapshot recovers the original byte identities for the remaining four cases and independent audits; fitting criteria, selection and budgets are unchanged. These results bind the frozen runtime, not later shared-code profiles. The shared project changes were preserved.

No new native inference, GPU action, physics step, outcome label, predictor evaluation or training was performed in this static stage. Native outputs, all failed and derived attempts, and private geometry remain retained separately from the sanitized public evidence. Final scalar evidence: `research_artifacts/contactgen_counterexample_repair_2026-10-15_v1/validation/current_cohort_scalar_summary_v6.json` (SHA-256 `84405dc9532ddc4f0f496ebab9abedc6eb3e74301a4609c083817868b06a43b6`).

**Decision.** Retain the exactly certified per-pose results and all explicit failures. Do not declare the canonical mapper or learned generator verified, and do not adopt diagnostic candidates. The two proved cases remain research candidates with separate same-state certificate bundles.

**Next check (proposed, not implemented).** A new search should retain each seed as a baseline and declare an acceptance-aligned objective and selection rule; the current common-slack objective and count-first diagnostic ranking can disagree. A production path would also need certificate-checked output admission tied to source, object, reference, pose, geometry and actual proof bytes. Neither proposal is evidence that the remaining cases can pass; bounded failures do not prove global infeasibility. Additional anatomy, normal, ray and application budgets require explicit definitions and separate evidence.


<!-- dga-acceptance-feasibility-2026-10-15-v2 -->
## 15 October 2026 — DGA exact feasibility obstruction and unchanged-angle failures

**Previous plan.** Continue repairing and independently testing DGA on the corrected importer while preserving target, criteria and originals; examine the previously untested target-legal raw pose and resolve the palm proof's numerical premise.

**Implemented evidence.** All eleven source-valid saved outputs received static raw-pose surface diagnostics, with all sixteen ledger entries retained. Raw row 6 is source/target legal but fails index/ring proximity at all three bands. Independent mesh reconstruction finds exact agreement with saved source triangles and identifies genuine distal shape differences, not a missing/doubled transform. Its palm nearset/occupancy pass is an empty-set result; directions remain unresolved. Ten other tested raw poses remain target-illegal.

Exact outward rational intervals reconstruct saved robust-6D source poses and literal wrist FK, removing the old assumed 1 nm source-transform bridge. Old development row 1 and new native row 0 each have an unavoidable positive palm-occupancy mismatch under their ORIGINAL max/SSE guards: at least 0.000104947 and 0.525625 mm². These quantifiers cover proper target palm placement and arbitrary finger articulation in the stated mathematical guard domains. Independent review recounts 516 certificates, nine literal boxes and guard provenance; 2,082 controls pass. The tighter component-guard result remains separate. Arbitrary floating guard predicate soundness and physical outcomes are not certified.

A separately frozen simultaneous joint/palm band-fitting experiment explores 2,238 states in 600.5 seconds. Two trial states pass strict component improvement and all existing guards; selection changes only THJ3 by 0.00001 rad. Thumb maximum fitting difference improves about 46.4 nm, palm and maximum landmark error are unchanged. Fresh full visual grids at (13/37,19/41) still FAIL. Budget exhaustion is not an infeasibility proof for this other pose. Independent adoption review and saved-output recount pass. A reporting-unit label defect was fixed in a separate summary version without changing measurements. All warnings and unsuccessful attempts remain preserved.

**Decision and next boundary.** DGA remains unsupported/unaccepted. The required cohort cannot all meet the unchanged exact declared-palm and landmark requirements; more optimization cannot remove the two certified conflicts. Other poses remain case-specific questions, not established impossibility. Do not silently substitute a passing subset, change geometry or infer physical grasp failure from mapping rejection. Progress to evaluation under a different target/fidelity claim/declared input domain would be a separate scientific scope decision, not a code fix; none was made. All 949 prior-study files remain unchanged. No physics, new inference, eval episodes/labels or training. [Report](DGA_ACCEPTANCE_FEASIBILITY_AUDIT.md).


<!-- contactgen-acceptance-repair-2026-10-15-v1-plan -->
### 2026-10-15 — Revise the ContactGen repair objective and retain feasible incumbents

**Previous evidence.** The preceding archived study has two scoped geometric passes and ten exact contact-preservation failures across twelve nonempty targets. The original capsule/101 remains a separate empty-source refusal. Its public archive manifest is `55562baea2541a28f657ddce3851db4adcce6a8d9544c7e2330a582ed383ef7b`; all 1,011 listed files rehash unchanged. The prior common-slack fit held surface attachments fixed for up to forty iterations, while diagnostic selection counted lost queries. Saved endpoints show that the count and maximum-distance objectives can disagree and a starting seed can outperform later poses.

**Decision and current status.** In response to the user's request to continue repair, develop a separately frozen method that refreshes closest exposed surface candidates during optimization and retains its original feasible seeds and improving candidates under one explicit acceptance-aligned score. Dynamic closest points may remove tangential constraints introduced by fixed attachments; this is a hypothesis until a new pose passes the independent exact checks. The first pilot is ellipsoid/1009, whose saved endpoint has a smaller maximum exposed-witness distance and substantially more finger-separation clearance than ellipsoid/101. The fixed derived capsule follows. No new fitting result is claimed in this entry.

**Unchanged contract.** Keep original source/object/reference identities, a positive native 2 mm band, both literal 10 mm mean-tip gates, physical joint bounds, continuous one-way 2 mm proximity inclusion at the exposed robot-union boundary, and strict positive separation for all 359 specified cross-digit collider pairs. An unresolved nearest-exposed-point search is not an exact contact failure. A dynamic numerical objective is not a continuous certificate. Original capsule geometry cannot acquire a nonempty band through target retargeting; any successful capsule repair retains its explicit derived lineage and the original refusal. No anatomy, normal, ray or physical-success claim is added.

**Implementation and next check.** New private work is under `runs/contactgen_acceptance_repair_2026-10-15_v1`; the existing sixteen-module frozen runtime is reused and all earlier archives are immutable. Review numerical branches and distance gradients before optimizing, then independently verify candidate geometry and proof bindings. Reuse saved native outputs only, with no GPU action, physics steps, outcome labels, predictor evaluation or training. Run at most one heavy process in this thread tree.


<!-- contactgen-acceptance-repair-2026-10-15-v1-epigraph-plan -->
## 2026-10-15 — Equivalent epigraph contact objective after a slow numerical pilot

The first two completed ten-iteration blocks of the dynamic closest-exposed-point pilot improve the retained ellipsoid/1009 seed from 2.483451142 to 2.477913096 mm maximum exposed-witness upper distance on the identical 4,435-query development pool. Their 126 recorded evaluations take 194.95 s; both endpoints satisfy the literal fingertip, physical-bound and numerical separation guards. This is numerical search progress, not a continuous-preservation certificate. Failed intermediate trial poses and the seed remain retained.

Decision: stop the current search adaptively at a recorded boundary and prospectively test an epigraph formulation, rather than interpreting this bounded pilot as infeasibility. The original frozen code and partial execution history remain unchanged. For query excess e_i = max(d_i − 1.95 mm, 0), the current merit is mean(e_i²) + 0.25 max(e_i)². Introducing z ≥ 0 and z ≥ d_i − 1.95 mm, and minimizing mean(e_i²) + 0.25 z², has the same optimum on resolved witness states. It removes the outer maximum's single-winner derivative, but not changes of closest collider, ray owner or support feature. Actual recomputed contact merit, never an inflated auxiliary z, must rank and retain candidate poses.

Status: mathematical review complete; wrapper and manufactured controls are being prepared. No epigraph optimization result exists at this decision point. Its protocol, dependencies, starts and stop semantics will be saved before execution. Both original 10 mm mean-tip gates, physical bounds, positive separation of all 359 cross-digit pairs, source identity/nonvacuity, and exact continuous one-way 2 mm inclusion remain unchanged. The 1.95 mm fitting target is a search margin only. The original empty capsule remains refused; its fixed rigid-derived source is an explicit separate target. No native inference, GPU action, physics, outcome labels, predictor training or evaluation is authorized by this methodological change.
