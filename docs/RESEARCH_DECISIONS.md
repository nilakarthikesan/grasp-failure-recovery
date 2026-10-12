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

## Keeping this record current

For each material change, record the date, what we thought previously, the new
observation or source, the choice made, its implementation status, the supporting
artifact and the next check. Mark unsuccessful experiments and rejected inputs
explicitly. Distinguish measured results from interpretations and proposals.
Link the paper to the evidence used for each claim. This convention is the
ongoing documentation requirement; this turn did not launch new experiments.
