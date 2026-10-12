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

Evidence: [the original plan changes](PAPER_PROGRESS.md#changes-to-the-original-plan)
and [research foundations](RESEARCH_FOUNDATIONS.md).

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
15 October daytime, using the earlier cutoff. The [preparation record](PAPER_PROGRESS.md#workshop-preparation-and-first-reading-12-october)
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
of model triggered recovery. The [progress note](PAPER_PROGRESS.md#prediction-and-recovery-prior-work-12-october)
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
unfamiliarity. The [progress record](PAPER_PROGRESS.md#candidate-directions-and-a-bounded-exploration-phase-12-october)
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
binds original source hashes and planned case identities. The [plain explanation](PAPER_PROGRESS.md#selected-question-and-proposed-wrist-comparison-12-october)
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
and [progress record](PAPER_PROGRESS.md#accepted-paper-question-and-evidence-gates-12-october)
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

## Keeping this record current

For each material change, record the date, what we thought previously, the new
observation or source, the choice made, its implementation status, the supporting
artifact and the next check. Mark unsuccessful experiments and rejected inputs
explicitly. Distinguish measured results from interpretations and proposals.
Link the paper to the evidence used for each claim. This convention is the
ongoing documentation requirement.
