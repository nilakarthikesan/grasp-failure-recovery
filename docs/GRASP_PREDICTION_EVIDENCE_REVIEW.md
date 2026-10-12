# Public evidence for grasp failure prediction

Reviewed 12 October 2026. This is a bounded review of 24 published systems,
research releases and company announcements. It includes foundational studies
and recent 2025–2026 work. It is not an exhaustive company census, a benchmark
leaderboard or independent certification of production reliability. Paper
versions are pinned where available; preprints and company claims are marked.
No model, label, threshold, simulator or recording changed in this review.

The later [VLA direction review](VLA_DIRECTION_REVIEW.md) explains SAFE's
controller/monitor distinction, checks published real-rollout formats and
records remaining questions about useful warning timing and deployment relevance.

## Start with these three sources

1. [Amazon Robin](https://arxiv.org/html/2305.10272v1): a deployed pick-success
   predictor that tests a targeted change in training coverage.
2. [Feeling of Success](https://proceedings.mlr.press/v78/calandra17a/calandra17a.pdf):
   the closest foundational comparison to our after-closing, before-lift timing.
3. [SAFE](https://arxiv.org/html/2506.09937v1): a runtime monitor whose unseen-task
   results show why strong development results do not establish broad reliability.

## What is being predicted?

* **Before execution:** score a proposed grasp or action. Geometry, object-relative
  pose and learned action values are common inputs in the reviewed systems.
* **During contact or execution:** estimate grasp outcome, slip or task failure
  using the information already observed. An alarm after slip onset is detection,
  not advance slip prediction.
* **After an action:** verify success using resulting images. Guardian belongs
  primarily here; its execution-verification accuracy is not a pre-lift warning score.
* **Policy/readiness values:** guide execution or regrasping. High executed success
  does not establish low false alarms, calibrated probabilities or warning lead time.

Percentages from these tasks cannot be placed on a common accuracy leaderboard.

## Do we already have a baseline?

Yes, a frozen pilot baseline. It uses joint positions, velocities and applied
targets during closing, without camera or tactile inputs. It was trained on
520 cube/apple episodes. On one reserved apple observation family it detected
187 of 202 failed tasks and falsely warned on 22 of 58 successful tasks.
That split was declared after collection inspection and is retrospective
development evidence, with limited object and scene coverage.

The later eight wrist pairs include 13 successful mug attempts and three
scorer-defined failures; every attempt receives a warning. The original
model therefore supplies a measurable reference, but is not a validated
general grasp-warning system. Its task labels mostly measure failed
acquisition/lift, not physical slip onset. See the
[original baseline](../research_artifacts/task_outcome_baseline_v1/) and
[complete wrist diagnostic](../research_artifacts/wrist_pair_remaining_v1/README.md).

## Common practices in the reviewed work

These are our synthesis of published methods, not a universal industry standard.

1. Define the outcome and prediction time before choosing a model. Lift/hold
   completion, slip onset and whole-task failure are different labels.
2. Choose measurements that can reveal the desired failure. Vision/object-relative
   geometry, tactile contact, command tracking and policy representations answer
   different questions. Hidden future disturbances need not be predictable.
3. Collect varied successful and unsuccessful conditions, or explicitly evaluate
   a success-only detector. Variation includes objects, placements, contact
   regions, forces, actual policy mistakes and selected deployment shifts.
4. Compare sensing/representation and data-collection strategies with controls.
   Changing every factor at once can improve a system without revealing what helped.
5. Reserve whole objects/tasks/scenes as appropriate, and separate validation or
   calibration from final testing. Hundreds of related frames are not hundreds
   of independent grasps.
6. Report false alarms, missed failures and timing, together with coverage and
   exclusions. Executed success after rejecting difficult grasps needs a separate
   denominator. Training on the same objects used in evaluation is not an
   unseen-object test.

The supporting examples and qualifications follow. Methods based on frozen
learned-policy representations are not drop-in replacements for our fixed
controller. Tactile/image datasets do not directly match our current joint inputs.

## What this changes for our next decision

The user has asked to establish the field foundation before scaling collection.
The wrist-versus-finger-noise question remains a candidate experiment under
review. Out-of-range wrist inputs and their score contributions motivate it;
they do not establish wrist importance, a causal failure mechanism or a training
remedy. Amazon and DENSE already investigate coverage strategies in other
embodiments, so generic "more variety helps" is not a new research gap.

Recommended sequence, not implemented in this review:

1. Confirm that our label and available input history match the outcome we want
   to predict. Describe the present target as grasp/lift/hold outcome forecasting.
2. Assess simple alternatives before attributing errors to episode count:
   separately validation-fitted calibration, a retrained wrist-feature ablation,
   and physically motivated command-tracking features. These need a declared
   grouped development protocol and fresh evaluation; none has been fitted here.
3. Choose collection axes from the remaining error pattern. Our own simulation
   remains the user's preferred first source. Start with a small versioned
   feasibility batch rather than thousands of unexamined attempts.
4. Freeze fresh evaluation groups and compare selected strategies with matched
   training budgets and class mix. Preserve the original baseline and report
   both false alarms and missed failures. Use causal, observable features;
   hidden simulator metadata is for analysis, not an undeclared prediction input.

This does not establish novelty or workshop readiness. A contribution needs a
controlled comparison and useful findings that extend the cited work.

## Source records

Each record separates the task, collection, reported evidence and its limits.
Primary papers or official research/company releases are linked directly.

### Amazon Robotics: Robin pick-success predictor

2023 · RSS. Peer-reviewed production study. [Primary source](https://arxiv.org/html/2305.10272v1).

**Inputs and task:** RGB-D geometry, material and proposed suction-tool pose.
**Training variety:** >394k real picks; failures retained; about 41k broader pick locations replace center picks at equal size.
**Reported evidence:** Random-location evaluation ROC-AUC: 0.802 → 0.848.
**Limit:** Predicts pick success before execution; hidden weight and tool wear remain unknown.
**Use for our decision:** Test targeted coverage against an equal-size control.

### Berkeley / MIT: Feeling of Success

2017 · CoRL. Peer-reviewed. [Primary source](https://proceedings.mlr.press/v78/calandra17a/calandra17a.pdf).

**Inputs and task:** RGB and tactile images after closing, before lift.
**Training variety:** 9,269 real attempts; 106 objects; varied position, orientation, height and force.
**Reported evidence:** Object-split classification: vision+touch 77.8%; vision 68.8%.
**Limit:** 94% executed success is a separate selected-grasp experiment, not warning accuracy.
**Use for our decision:** Match prediction time and measure sensor contribution.

### Berkeley / MIT: More Than a Feeling

2018 · RA-L. Peer-reviewed. [Primary source](https://arxiv.org/html/1805.11085).

**Inputs and task:** Vision, touch and proposed grasp adjustment.
**Training variety:** 6,450 trials / 65 training objects; separate calibration on about 1,900 validation examples.
**Reported evidence:** 80.28% object-split cross-validation accuracy; separate robot evaluation on 22 unseen objects.
**Limit:** Recovery action selection differs from our unchanged pre-lift warning.
**Use for our decision:** Check calibration and action conditioning.

### Independent academic research: Let's DENSE

2025 · npj Robotics. Peer-reviewed. [Primary source](https://www.nature.com/articles/s44182-025-00055-y).

**Inputs and task:** Tactile contact signals.
**Training variety:** 2,100 total collected grasps; dense training variants use 210 grasps, one repetition per pose; seven objects.
**Reported evidence:** Held-out objects and grasp poses; collection strategies compared with MCC.
**Limit:** Slip detection, not forecast; reported relative improvements are condition-specific.
**Use for our decision:** Cover physical contact variety instead of only repeating poses.

### NVIDIA: GraspGen

2026 · ICRA / pinned 2025 preprint. Peer-reviewed; dataset count from pinned preprint. [Primary source](https://arxiv.org/html/2507.13097v1).

**Inputs and task:** Object point cloud and candidate grasp pose.
**Training variety:** 53.1M synthetic grasps across three grippers; parallel-jaw simulation and analytical suction labels; generated poses add labeled examples.
**Reported evidence:** Real grasping system reports 81.3% executed success.
**Limit:** Pre-execution discriminator and planner; executed success is not failure-warning accuracy.
**Use for our decision:** Collect mistakes produced by the actual grasp generator.

### Google: QT-Opt

2018 · CoRL. Peer-reviewed. [Primary source](https://proceedings.mlr.press/v87/kalashnikov18a/kalashnikov18a.pdf).

**Inputs and task:** Camera observations and proposed robot actions.
**Training variety:** 580k real grasp attempts; seven robots; over 1,000 objects.
**Reported evidence:** Grasp policy: 96% success in 700 unseen-object attempts.
**Limit:** A policy guided by a value function, not a separately evaluated alarm classifier.
**Use for our decision:** Real successes/failures and diverse objects inform collection.

### NVIDIA / Oxford: Grasp-MPC

2026 · ICRA. Peer-reviewed / pinned preprint. [Primary source](https://arxiv.org/html/2509.06201v1).

**Inputs and task:** Point cloud and object-relative hand pose.
**Training variety:** About 2.1M synthetic trajectories; thousands of shapes; translation/orientation perturbations.
**Reported evidence:** Simulation and novel-object hardware grasp-policy evaluations.
**Limit:** Training labels use feasible grasp targets; full trajectories lack physics validation.
**Use for our decision:** Geometry-relative inputs may help; audit what labels actually certify.

### Meta and collaborators: Sparsh / TacBench

2024 · CoRL. Peer-reviewed. [Primary source](https://arxiv.org/html/2410.24090v1).

**Inputs and task:** Tactile images and pretrained touch representations.
**Training variety:** 460k+ unlabeled tactile images; grasp task roughly 8k train / 1.3k eval examples.
**Reported evidence:** Grasp classification around 80%; six downstream tasks.
**Limit:** Grasp split includes all objects; aggregate relative gains are not grasp accuracy.
**Use for our decision:** Representation and sensor coverage matter alongside labeled data.

### Academic / Analog Devices collaborators: SlipSense

2026 · September preprint. Preprint. [Primary source](https://arxiv.org/html/2609.15910v1).

**Inputs and task:** Pressure arrays and accelerometers.
**Training variety:** 1.4M frames; 28 training / nine held-out objects; grip forces, speeds and contact regions varied.
**Reported evidence:** Held-out macro F1 95.75%; false-positive rate 1.57%.
**Limit:** Slip detection after onset; hand transfer degrades results.
**Use for our decision:** Report false alarms, timing and hardware transfer.

### Academic research: See to Reach, Feel to Grasp

2026 · September preprint. Preprint. [Primary source](https://arxiv.org/html/2609.31323v1).

**Inputs and task:** Five frames of joints and measured-minus-command residuals.
**Training variety:** Simulation varies object mass, shape/scale, pose, servo gains and disturbances.
**Reported evidence:** Simulation success; force-closure correlation; qualitative hardware examples.
**Limit:** Learned readiness/control score, not calibrated failure probability; hardware reliability unmeasured.
**Use for our decision:** Test physically meaningful joint features before scaling collection.

### Toronto / Toyota Research Institute: SAFE

2025 · NeurIPS. Peer-reviewed. [Primary source](https://arxiv.org/html/2506.09937v1).

**Inputs and task:** Frozen robot-policy representations.
**Training variety:** Successful and failed rollouts; complete tasks held out; thresholds calibrated separately.
**Reported evidence:** Real seen-task ROC-AUC 0.8676; unseen-task 0.6416.
**Limit:** Runtime task failures; some slip alarms follow slipping.
**Use for our decision:** Generalization remains difficult even for strong published detectors.

### Academic failure-monitor research: SAFECAST

2026 · August preprint. Preprint. [Primary source](https://arxiv.org/html/2608.04246v1).

**Inputs and task:** Policy representations with augmented deployment rollouts.
**Training variety:** Clutter, objects, arrangements, lighting and instructions; training and calibration tested separately.
**Reported evidence:** Real failure-monitor F1 improves in reported settings.
**Limit:** Benefits vary; arbitrary distribution shift breaks calibration assumptions.
**Use for our decision:** Separate coverage improvements from warning-threshold improvements.

### Toyota Research Institute / Woven: FAIL-Detect

2025 · RSS. Peer-reviewed / pinned preprint. [Primary source](https://arxiv.org/html/2503.08558v2).

**Inputs and task:** Sensor/action representations from a learned policy.
**Training variety:** Successful demonstrations; separate successful rollout calibration; camera and task perturbations.
**Reported evidence:** Four simulated tasks and two real bimanual tasks.
**Limit:** Unfamiliar successful behavior can resemble an anomaly.
**Use for our decision:** A success-only detector is an alternative, with false-alarm tradeoffs.

### Stanford / NVIDIA collaborators: Sentinel

2024 · CoRL. Peer-reviewed. [Primary source](https://arxiv.org/html/2410.04640v1).

**Inputs and task:** Fast action-consistency checks plus slower video task-progress checks.
**Training variety:** Rollouts with varied execution failures; released evaluation data.
**Reported evidence:** Combined detector finds more failures than either component.
**Limit:** VLM latency and task-level detection differ from pre-lift grasp forecasting.
**Use for our decision:** Different failure mechanisms may need complementary measurements.

### Inria / ENS / CNRS: Guardian

2025–2026 · pinned v3. Version-pinned research. [Primary source](https://arxiv.org/html/2512.01946v3).

**Inputs and task:** Task context and before/after images.
**Training variety:** About 30,876 examples; simulated failures plus real robot data.
**Reported evidence:** Execution verification: 86% RoboFail / 77% UR5-Fail.
**Limit:** Post-action verification is not demonstrated pre-slip prediction.
**Use for our decision:** Outcome checkers are useful but answer a different question.

### Physical Intelligence: RECAP / pi-star-0.6

2025 · November preprint. Preprint plus company release. [Primary source](https://arxiv.org/html/2511.14759v1).

**Inputs and task:** Vision-language task value / progress.
**Training variety:** Demonstrations, autonomous rollouts, successes/failures and expert corrections.
**Reported evidence:** Improved executed task success in evaluated settings.
**Limit:** No isolated grasp-alarm false-positive or lead-time benchmark.
**Use for our decision:** Train from deployment mistakes and measure intervention benefit.

### Figure: Helix 2.5 controlled pretraining comparison

2026 · September report. Company-reported. [Primary source](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization).

**Inputs and task:** Human-data-pretrained manipulation policy.
**Training variety:** Broad pretraining; same downstream task data/architecture; 30 unseen homes.
**Reported evidence:** Company-reported full-task success: 9% → 56%.
**Limit:** Policy performance; not grasp-warning accuracy or independent replication.
**Use for our decision:** Hold evaluation homes/objects out before experiments.

### OpenAI: Procedural object domain randomization

2017 · research. Published research / official summary. [Primary source](https://openai.com/index/domain-randomization-and-generative-models-for-robotic-grasping/).

**Inputs and task:** Scene observations for grasp planning.
**Training variety:** Millions of procedurally generated object shapes.
**Reported evidence:** Reported real executed grasp success: 80%.
**Limit:** Grasp planning, not a validated early failure-warning detector.
**Use for our decision:** Simulation geometry variety has long-standing precedent.

### Anthropic: Claude robotics interface evaluation

2026 · research report. Company research report. [Primary source](https://www.anthropic.com/research/claude-plays-robotics).

**Inputs and task:** Vision/force observations or supervision of pretrained policies.
**Training variety:** Robotics benchmark across control interfaces.
**Reported evidence:** Direct manipulation full-task success remains low; supervision helps some tasks.
**Limit:** No published dedicated grasp-warning training recipe in this report.
**Use for our decision:** Robot interface and usable observations affect results.

### Dexterity: Foresight

2026 · company announcement. Company claim. [Primary source](https://dexterity.ai/blog/foresight).

**Inputs and task:** Production-trained physical/world models.
**Training variety:** Company claims over 100M production actions.
**Reported evidence:** Describes predicting grasp hold, paths and placement stability.
**Limit:** Reviewed announcement lacks isolated detector accuracy, false alarms and reproducible training details.
**Use for our decision:** Treat commercial capability claims separately from measured predictor evidence.

### Covariant: RFM-1 announcement

2024 · company announcement. Company claim / access-limited. [Primary source](https://www.linkedin.com/posts/covariant-ai_introducing-rfm-1-our-robotics-foundation-activity-7172954447805894657-Sxyg).

**Inputs and task:** Multimodal world-model direction.
**Training variety:** Company describes robot experience plus internet data.
**Reported evidence:** Public announcement reviewed; old technical pages inaccessible.
**Limit:** No reproducible grasp-warning benchmark in reviewed material.
**Use for our decision:** A public survey cannot certify undisclosed production reliability.

### Tsinghua / collaborators: ActProbe

2026 · June preprint. Preprint. [Primary source](https://arxiv.org/html/2606.08508v1).

**Inputs and task:** Emitted action chunks; magnitude and temporal inconsistency.
**Training variety:** Released generative-policy rollout benchmarks; compact learned action probe.
**Reported evidence:** Early-detection comparisons; small real recorded-pick evaluation.
**Limit:** Offline real tests are limited; not universal physical-slip forecasting.
**Use for our decision:** Compare an action-only baseline with hidden-feature monitoring.

### Academic VLA-monitor research: Hide-and-Seek

2026 · May preprint. Preprint. [Primary source](https://arxiv.org/html/2605.30834v1).

**Inputs and task:** VLA representations with temporal contrastive training.
**Training variety:** Trajectory outcomes; no step-level annotation required.
**Reported evidence:** Simulation and real task-monitor evaluations.
**Limit:** Averaged embeddings can lose joint structure; perceptual failures remain difficult.
**Use for our decision:** Localized supervision and signal structure already have prior work.

### Academic VLA-monitor research: VLA-FAIL

2026 · June preprint. Preprint. [Primary source](https://arxiv.org/html/2606.21386v1).

**Inputs and task:** Internal feature shift and overlapping-action inconsistency.
**Training variety:** Policy fine-tuning data; failure examples not required.
**Reported evidence:** Runtime-monitoring studies of learned policies.
**Limit:** Can miss failures with consistent features and actions; requires compatible action chunks.
**Use for our decision:** Check physical task evidence beyond policy consistency.

## Review status and reproducibility

This review records public descriptions, not reproduced experiments. Source
claims remain limited to their tasks, sensors and evaluation conditions.
Private training recipes and production reliability were not inferred from
demos or deployment scale. The Covariant technical-page access limitation is
recorded instead of substituting third-party claims. New preprints require
extra scrutiny; older directly matched studies remain useful methodological
references.

The current manuscript, baseline, source-bound diagnostic artifacts and raw
recordings remain the scientific record. No published result here certifies
our simulator's contact physics, our chosen wrist augmentation, or real-world
transfer. The accepted draft question has not been replaced with a broader
claim; scope selection should follow review of the baseline and observable
failure target.
