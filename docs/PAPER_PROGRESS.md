# Paper progress: evaluating simulation diversity for grasp failure warnings

**Working branch: `codex/project-work`.** The earlier milestone branches were
consolidated through [PR 19](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/19),
which is merged into `main`. Later diagnostics and paper preparation continue on
this one working branch; they are not automatically part of `main`. Historical PR links below preserve the
development record; see [WORKING_BRANCH.md](WORKING_BRANCH.md) for the current
checkout and recovery references. Scientific protocol versions remain separate.

**Updated: 12 October 2026.** This is the evidence index for the
[standalone working paper](paper/working_draft.tex), now titled **Evaluating
Simulation Diversity for Grasp Failure Warnings**. The current draft question compares
successful wrist coverage with additional finger command noise at matched
training budgets. That proposed comparison is under review and not completed.
The [public evidence review](GRASP_PREDICTION_EVIDENCE_REVIEW.md) records training,
sensing and evaluation practices before we choose further collection. Completed evidence
comprises the pilot, frozen fresh evaluation, separate mug executor correction,
scoring of 26 completed checks, the wrist input audit and a static check of
observed training wrist settings, frozen selection of eight wrist pairs and
completion of all eight zero-noise comparisons. Future slip timing,
recovery and real world transfer remain untested.

**Immediate next step:** review the published evidence and confirm our prediction
target, observable inputs and baseline comparisons before choosing new data.
Any fitting still needs separate validation/calibration and a common fresh holdout.
All eight wrist pairs are complete: every success still triggers a warning.
See the
[current question and evidence gates](#accepted-paper-question-and-evidence-gates-12-october).
The latest [completed paired results](#seven-remaining-wrist-pairs-completed-12-october) supersede
earlier planning-status statements below; bound historical plans retain their
original bytes.

The authoritative completed collection is documented in
[MULTI_SCENE_COLLECTION.md](MULTI_SCENE_COLLECTION.md). Earlier documents retain
their historical scope and dates; this page does not reinterpret their smaller
verification batches as new independent data.

The earlier milestone documents preserve their dated findings. Reviewable sections are the
[interactive demo](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/8),
[earlier literature notes](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/9),
[execution dependency](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/12),
[collector](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/10),
and [expanded collection](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/11).

## 1. Review HUG datasets and benchmarks — initial review complete

HUG supplies grasp hypotheses; its RGB-D model predicts wrist placement and a
MANO pose that we retarget to the actuated right Shadow Hand. The released
HUG-Bench objects provide a useful taxonomy and metric simulation assets. Our
review is enough for this pilot, not a full benchmark reproduction or an audit
of every released dataset.

We selected the development apple asset `val/medium_3/apple` and the existing
cube. The apple's native MJCF/visual/two-part collision geometry, mass, content
hash, pinned revision, and CC BY 4.0 attribution are recorded in
[the geometry provenance section](MULTI_SCENE_COLLECTION.md#geometry-source-and-what-we-use).
Official benchmark grasp annotations remain unused: inputs are rendered
locally and proposals come from fresh HUG inference. See
[Research foundations](RESEARCH_FOUNDATIONS.md) for the embodiment and
retargeting decisions.

**Paper note:** HUG generates grasps; our contribution at this stage is the
verified execution/data contract around those hypotheses. Use
[the HUG paper](https://arxiv.org/abs/2606.17054) and
[official resources](https://grasping.io/) for source claims.

## 2. Generate varied simulation trials — controlled milestone complete

We froze `adroit_shadow_tabletop_v2`, `shadow_hand_right`, and
`fixed_grasp_lift_v2`. A fixed high-level approach/close/lift/hold sequence runs
through force-limited position servos. It does not learn or replan a policy.

The only current noise type is `uniform_episode_bias`: one random angle offset
per finger target, fixed through designated phases, with wrist offsets zero.
The schedule requests 0, 0.15, and 0.6 rad bounds. Nominal plus requested bias
is clipped at joint limits, and measured motion can differ from the target.
Requested offsets, actual clipped offsets, commands, and measured states are
saved separately. This is deliberate command stress, not obstacles, camera
noise, or a calibrated hardware-noise distribution. See
[the noise definition and example](NOISY_ROLLOUT_COLLECTION.md#what-the-noise-means).

**Paper note:** At least one target was clipped in 508/512 attempts at 0.15 rad
and 512/512 at 0.6 rad. Avoid claiming equal applied disturbances or that noise
always causes failure; 108 noisy apple attempts succeeded although all eight
apple baselines failed.

## 3. Collect trajectories — 1,040-attempt pilot complete

Two geometries × two observation scenes × four fresh proposals produced
**16 proposal families and four observation families**. One baseline plus 32
repetitions at each nonzero amplitude gives **1,040 attempts**: **444 successes,
596 failures, and zero errors**. There are 512 paired noise directions across
amplitudes. These attempts are not 1,040 independent scenes.

The local source is `runs/object_scene_pilot_2026-10-10/`: `inputs_v3/` and
`inputs_v3_extra/` hold captures/proposals; `collection_1040/` holds the frozen
plan, index, summary, and episode bundles; `collection_1040_audit.json` records
verification. These ignored raw files are not a public dataset release. The
[collection documentation](MULTI_SCENE_COLLECTION.md#fresh-results-and-verification)
records exact paths, counts, provenance, and reproduction commands.

Initial RGB-D is saved per scene. Robot/object state, commands, contacts,
outcomes, and replay states are logged per control interval; there is no online
camera image at every rollout frame. The
[local viewer](http://127.0.0.1:8769/) shows the collection and six videos
rendered later from saved states. It is available only while the local server
is running.

**Paper note:** All 1,040 scores/events regenerate exactly, all source bindings
and paired draws were checked, and seven representative complete sequences
reexecuted bitwise on the same runtime. The collection audit identifies source
`99f70a6`; the documented suite at `301ca1f` passed 292 tests with five skips.
Same-runtime replay is not cross-platform or real-robot validation.

## 4. Label trajectories — frozen task labels complete; target review needed

The existing scorer keeps **1 = task success**, **0 = completed physical task
failure**, and **null = validation/runtime error**. Success requires qualifying
contact, at least **14 cm peak lift**, **2 s sampled HOLD at least 12 cm above
initial height**, no more than **2 cm peak-to-final loss**, and the collision
and timeout checks. Contact alone is not a secured-grip label.

The failures are 590 `failed_acquisition`, two `no_contact`, and four `drop`.
The [label audit](MULTI_SCENE_COLLECTION.md#what-the-labels-mean-before-we-train)
found **21 attempts within 0.5 mm of the lift cutoff**. Apple
`episode_000465` meets the sampled hold requirement but falls 0.0308 mm short
of the peak-lift cutoff. Thirty additional attempts lose more than 2 cm yet
remain acquisition failures because of category precedence. Preserve these
labels; any different grasp/retention target needs its own explicit version.

`events.json` separately records lift/retention proxies, an acquisition deadline
interval, and observed post-proxy height-loss bounds. Unknown, unsupported, and
censored cases stay explicit. Forty-millisecond endpoints and the logged 2 ms
diagnostic offset do not give exact slip onset. See
[event definitions](NOISY_ROLLOUT_COLLECTION.md#final-labels-and-event-annotations).

**Paper note:** This collection mainly supports acquisition/task-outcome study.
Four task-scored drops do not establish a future-slip benchmark. Event labels
are analysis/supervision, never model features.

## 5. Separate training, validation, and test — exploratory pilot complete

The actual derived export is saved in
[`research_artifacts/object_scene_pilot_v1/`](../research_artifacts/object_scene_pilot_v1/).
It contains [`dataset.npz`](../research_artifacts/object_scene_pilot_v1/dataset.npz),
the [feature/provenance manifest](../research_artifacts/object_scene_pilot_v1/manifest.json),
and [split memberships](../research_artifacts/object_scene_pilot_v1/splits.json).
See [DATA_SPLITS.md](DATA_SPLITS.md) for the input contract, checks, and
reproduction command. It exports **1,040 × 1,440 float32 values**: the final
20 closing samples × 72 whitelisted position/velocity/command columns, ending
at 2.96 s before lifting. Outcomes and source identities are stored separately,
outside the model matrix; no privileged or future features are included.

Training fits model parameters; validation selects regularization; test measures
performance after that choice is fixed. Complete trajectories, proposal groups,
their noise variants, and observation groups remain in one partition.

The selected protocol is a **retrospective observation-disjoint split**:
scenes 0 + 2 train on 520 attempts (231 successes/289 failures); scene 1 validates
on 260 cube attempts (155/105); scene 3 tests on 260 apple attempts (58/202).
Training contains both geometries, while validation is cube-only and test is
apple-only. This asks about a held-out apple observation under the frozen task;
it does not establish unseen-geometry or broad object generalization. It was
chosen after collection, not preregistered. All shared observation/proposal/noise
variants stay together. Four observations and two geometries limit any claim.

Training contains eight proposal/two observation families; validation and test
each contain four proposal/one observation family. The export has zero excluded
attempts and preserves all 444 success/596 failure labels. Its loader checks
payload hashes, exact column order, source-label/scorer agreement, provenance,
group consistency, and the declared assignments. **28 focused exporter/loader
tests passed**, including causal/privileged-data mutation checks, group conflicts,
corruption, exclusions, and overwrite protection; the actual export also passed
the validated loader.

The dataset payload hash is
`sha256:29f773f7faf03cff5e1c694ef475d02adfd07c239314b349f2228f86db9d3aad`;
the split hash is
`sha256:27a9c894b0775a5a10e2ae83a18b6a70f78d4c504b80396086a80158b3b5aac7`.
These records complete step 5 for this exploratory pilot. A larger untouched
benchmark and unseen-geometry evaluation remain future work.

## 6. Train and evaluate a prediction model — first task-outcome baseline complete

The frozen first protocol predicts the unchanged eventual task label from the
**last 20 CLOSE_FINGERS samples, ending at 2.96 s**, before first LIFT at 3.00 s.
Whitelist 24 measured hand joint positions, 24 joint velocities, and 24 actual
commanded targets: a 20 × 72 window. Last/mean/population-standard-deviation/end-minus-start
summaries give 288 features. L2 logistic regression uses SciPy L-BFGS-B,
training-only scaling, and the fixed regularization grid **0.001, 0.01, 0.1, 1**,
selected by minimum validation log loss among converged fits. Both the full
model and command-only ablation selected **lambda = 0.1** and converged. Training
standard deviation ≤ 1e-8 excludes a feature: 269 full-model and 77 command-only
features remain active. The objective is mean binary cross entropy plus half
lambda times the squared weight norm, with unpenalized intercept.

We compared a command-only 96-feature ablation and a training-prevalence constant
baseline (success probability 231/520). The decision threshold remained fixed
at failure probability ≥ 0.5; test was scored only after validation-based
regularization selection, with no test tuning. A larger temporal model remains
a later comparison.

Exclude future samples, final outcomes, event metadata, noise vectors/settings/
seeds, source identities, simulator object truth, and full integration state from
predictor features. This first task is eventual-outcome prediction. Fixed-horizon
future loss and useful warning time require a separately validated event target
and more supported loss cases.

The unchanged split has **520 training attempts (231 success/289 failure)**,
**260 cube-B validation attempts (155/105)**, and **260 apple-B test attempts
(58/202)**. On the apple-B test partition:

- **Measured joint state + commands:** failure AUROC **0.8523**, failure average
  precision **0.9478**, log loss **0.3771**, Brier score **0.1170**. The fixed
  threshold flagged **187/202 failures** and incorrectly flagged **22/58
  successes**: 92.6% failure recall and 37.9% success false alarms.
- **Commands only:** failure AUROC **0.6419**, average precision **0.8665**,
  log loss **0.5244**, Brier score **0.1711**; flagged 192/202 failures and
  52/58 successes.
- **Training-prevalence constant:** failure AUROC **0.5000**, average precision
  **0.7769**, log loss **0.6374**, Brier score **0.2222**; flags every attempt.

Measured response improves this comparison beyond command summaries. These are
empirical metrics on one held-out apple observation, not established calibrated
deployment rates. The retrospective raw outcomes were previously inspected,
both geometries occur in training, and validation is cube-only. This is not an
untouched benchmark or evidence of unseen-geometry transfer, slip onset, warning
lead time, or drop detection. Only four task-scored drops occur in the entire
collection.

An illustrative CLI prediction for cube-A `episode_000534` returns failure
probability **0.4815** at 2.96 s and therefore predicts success, while the recorded
outcome is `drop`. This belongs to a training observation family, not an
additional held-out evaluation. The error helps demonstrate why this classifier
must not be described as a specialized drop detector.

The [baseline documentation](PREDICTION_BASELINE.md) records the method and
reproduction command. The completed
[artifact bundle](../research_artifacts/task_outcome_baseline_v1/) contains the
[full report](../research_artifacts/task_outcome_baseline_v1/report.json),
[state-and-command model](../research_artifacts/task_outcome_baseline_v1/proprioception_and_commands.npz),
[command-only model](../research_artifacts/task_outcome_baseline_v1/commands_only.npz),
and [per-attempt predictions](../research_artifacts/task_outcome_baseline_v1/predictions.csv).
The report pins input dataset/manifest/split and training-source hashes, fitted
models, all fixed-grid validation candidates, and prediction hashes. Its
runtime is Python 3.12.13, NumPy 2.2.6, and SciPy 1.17.1, separately recorded
from the simulation collection runtime.

**Verification:** The final combined suite passed **332 tests, with five skips**
and one upstream `hppfcl` deprecation warning. The **40 focused prediction tests**
also passed with runtime warnings treated as errors. The older 292-pass/five-skip
suite reported above remains the historical collection check. Model reload/
prediction consistency and causal input checks are separate from proof of
physical generalization.

**Published sections:** [paper and evidence notes (#13)](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/13),
[Step 5 dataset and grouped splits (#14)](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/14),
and [Step 6 trained baselines and results (#15)](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/15).
These were originally stacked draft PRs. Source is now consolidated on
`codex/project-work`; their historical links do not mean those changes have
automatically merged into `main`.

## Literature findings → implementation decisions

These primary sources were checked on 11 October 2026. The arrows below are our
design inferences; they do not mean the papers evaluated our pipeline.

- **[HUG](https://arxiv.org/abs/2606.17054), Wu et al., 2026:** RGB-D-conditioned
  human grasp hypotheses and metric benchmark assets → use pinned proposals,
  explicit retargeting, and asset/scene provenance; do not call this pilot an
  official benchmark score.
- **[The Feeling of Success](https://proceedings.mlr.press/v78/calandra17a.html),
  Calandra et al., 2017:** compares vision and GelSight touch for grasp-outcome
  prediction → preserve a future sensing-ablation question while distinguishing
  simulator diagnostics from physical tactile measurements.
- **[SlipSense](https://arxiv.org/html/2609.15910v1), Jian et al., 2026,
  §§3–5:** causal current-frame contact/slip classification and latency versus
  false-alarm evaluation → keep current slip, future loss, and acquisition failure
  separate; use only history available by the prediction time.
- **[FeelWorld](https://arxiv.org/html/2607.24267v1), Ma et al., 2026,
  §III:** action-conditioned visual/contact/force-related/slip prediction → save
  commands alongside observations and defer a large world model until the simple
  outcome target is understood.
- **[Foundational World Models Accurately Detect Bimanual Manipulator
  Failures](https://arxiv.org/html/2603.06987v1), Ward et al., 2026,
  §IV-C:** trajectory-level conformal calibration with exchangeability and shift
  caveats → freeze grouped validation and report empirical calibration; do not
  promise guarantees across novel objects or changed conditions.
- **[MuJoCo](https://www.roboti.us/lab/papers/TodorovIROS12.pdf),
  Todorov, Erez, and Tassa, 2012:** articulated dynamics and contact simulation
  for model-based control → use controlled execution and replay, while measuring
  contact fidelity on hardware before any transfer claim.
- **[AnyTeleop](https://yzqin.github.io/anyteleop/), Qin et al., 2023, and
  [Dex Retargeting](https://github.com/dexsuite/dex-retargeting):** the released
  retargeting library documents joint-order differences → use its supplied
  Shadow configuration and explicit joint-name mapping; preserve library and
  robot-asset provenance instead of implementing a new optimizer.

## Changes to the original plan

The [research decision history](RESEARCH_DECISIONS.md) records the evidence and
reasons behind these changes, including later mug debugging, consideration of
public data and the distinction between outcome verification and useful early
warnings. Implemented choices and proposed experiments are marked separately.
Append a dated entry after each material change instead of rewriting earlier
reasoning as though the final plan had been known from the start.

- **9–10 October:** Replaced the initial idea of training a grasp policy first
  with fixed HUG-conditioned execution, isolating grasp/command effects from
  controller-learning failures. The earlier
  [system design](SYSTEM_DESIGN.md) remains a research agenda, not a completion
  ledger.
- **10 October:** Replaced generic “noise” with one documented bounded episode
  bias, paired amplitude directions, and separate requested/applied/measured
  records. Distinguished task labels from conservative temporal proxies.
- **10 October:** Expanded one cube observation to verified cube/apple geometry,
  four byte-defined observations, and sixteen proposals. Repeated identical
  captures were grouped together. The upright apple setup was rejected; a
  documented pose-window settling gate accepted bounded inverted poses without
  changing physics. Setup rejection was never labeled grasp failure.
- **11 October:** Added this progress index and the scoped working paper. Kept
  original task labels after the lift-boundary audit, and made target review,
  grouped splitting, and a causal baseline the next evidence-producing steps.
- **11 October, step 5 update:** Exported the 1,040 causal closing prefixes and
  saved the observation-disjoint development split, source hashes, and strict
  loader checks. The retrospective limitation remains; step 6 training and
  evaluation were still pending at that point.
- **11 October, step 6 update:** Trained the regularized task-outcome model and
  two comparisons using the frozen prefix/splits. Validation selected lambda
  0.1 without test tuning. Recorded the actual apple-B metrics and its substantial
  success false-alarm rate; future-slip, drop-timing, recovery, and real-robot
  claims remain untested.

External VR failure data and synthetic human-video generation remain separate
exploratory tracks. The former needs a source/sensor/outcome audit; the latter
has no implemented method or measured benefit. Neither blocks expanding and
testing the controlled simulation baseline on new observations.

## Frozen evaluation on fresh objects and scenes, 11 October

The next experiment is now complete. Before collection we froze the saved
model, normalization, 2.96 second input cutoff, 0.5 failure alert, scorer and
original executed environment/controller configuration. Two new position and
camera scenes per familiar cube/apple and new HUG milk carton/mug supplied
eight observation families and 32 proposals. The original noise schedule gave
**2,080 completed attempts: 659 successes, 1,421 failures and zero errors**.
No new data selected model weights, regularization, features or threshold.

The unchanged full model has failure AUROC **0.9173 on fresh familiar geometry
scenes** and **0.5588 across the new geometries**. The milk carton has AUROC
0.8121 but 185 false alarms among 245 successes at the original threshold.
Every mug attempt is an approach collision; the model misses 205 of those
520 failures. With no mug successes, mug AUROC and success false alarm rate
are undefined. These approach failures happen before the input cutoff and do
not measure warning time before a future failure. The new object comparison
changes native mass and geometry together and mixes two different conditions.

Overall AUROC is 0.7737, log loss 0.6645 and Brier score 0.1865. The constant
training prevalence has log loss 0.6584; improved ranking/Brier does not imply
better log loss. Report both the aggregate and object breakdowns. These results
support studying transfer and threshold reliability under new conditions,
not claiming slip detection or reliable deployment.

The initially selected salt shaker failed stationary capture in all six
declared resting orientations. A recorded amendment replaced it with the milk
carton before grasp outcomes, without changing physics or acceptance rules.
That setup rejection is not a grasp label. The completed batch uses native HUG
geometry/masses and fresh inference rather than downloaded failure trajectories.

The [step by step guide](FRESH_OBJECT_EVALUATION.md) explains methods, examples,
Enoch's questions and storage. The [portable evaluation bundle](../research_artifacts/fresh_object_scene_eval_v1/)
contains the causal data, all 6,240 predictions, full/grouped results and frozen
protocol/source records. Full rollouts remain local; eight small replay videos
are included with the portable artifact.
The [fresh viewer](http://127.0.0.1:8770/) shows predictions beside outcomes.

Validation: all 77 focused model/data/evaluator/viewer checks passed, and the
full suite passed 338 tests with 34 skipped and one upstream deprecation warning.
An independent audit reconstructed every raw prefix and task label, reproduced
all 6,240 predictions exactly and recomputed all 87 group metric records. It
verified disjoint original source content and unchanged model/controller hashes.
The standalone manuscript currently retains its earlier pilot scope; these
separate evidence notes preserve the new experiment for a subsequent revision.

## Decision history and research question update 11 October

The [decision history](RESEARCH_DECISIONS.md) now connects the original pilot,
frozen fresh evaluation, separately versioned mug correction and public data
discussion. The standalone manuscript adds a scoped discussion of this evolution
and the proposed warning question; its quantitative pilot results remain the
earlier experiment. Guardian and public DROID data are potential comparisons,
with no import, training or evaluation completed here. Prior work already studies
runtime failure prediction, so earlier timing alone is not a novelty claim.
The next evidence should address causal information, unfamiliar conditions and
advance warning at a controlled false alarm rate. Recovery benefit requires a
separate intervention experiment.

## One corrected mug walkthrough, 12 October

One first listed development proposal was executed with the corrected v3
controller and zero noise. It succeeded with 14.6077 cm peak lift and two seconds
sampled hold, while the unchanged predictor raised a near 100% failure score
from the closing prefix through 2.96 seconds. This is a false alarm, with no
training or threshold change. Independent replay checked 3,260 physics steps,
found zero hand table contacts and reproduced the full saved states exactly.

Eight wrist features exceed five original training standard deviations; wrist
inputs dominate this particular linear model score. This motivates a controller
and input distribution shift check, not an immediate conclusion to remove wrist
inputs. The reused scene and changed execution protocol make this a development
demonstration. It supplies neither an aggregate unseen object score nor evidence
of warning time, calibration or recovery. The [walkthrough bundle](../research_artifacts/mug_single_walkthrough_v1/README.md)
preserves the actual input, score, outcome, audit and explanation. The manuscript
retains its earlier pilot scope; this note does not revise its quantitative claims.

## Frozen model diagnostic on all corrected checks, 12 October

Scored the 26 completed recordings from the existing corrected v3 checks,
without simulation, training, feature changes or threshold tuning. The saved
state and command model retained its original normalization, 20 closing samples
through 2.96 seconds and 0.5 failure alert threshold. All predictions were
computed from copied observable prefixes before loading task outcomes.

All 23 completed mugs were flagged: 20 successful grasps were false alarms and
three failed acquisitions were detected. By requested finger command noise
bound, false alarms were **7/7 successful mugs at 0.00 rad, 8/8 at 0.15 rad and
5/5 at 0.60 rad**. There are no failures at 0.15 rad, so its failure recall and
AUROC are undefined. The high noise condition has one additional guard rejection
with null label and prediction; it is excluded from all outcome metrics.

A paired proposal failed without noise and succeeded with both noisy settings,
yet the model warned in all three runs. Noise does not assign a failure label.
Controls comprise a correctly cleared successful cube, a detected apple
acquisition failure and a falsely flagged successful milk carton. Overall:
21 false alarms, four detected failures, one correctly cleared success and zero
missed failures. The model cannot clear a successful mug at its current threshold.

The [diagnostic bundle](../research_artifacts/corrected_mug_model_diagnostic_v1/README.md)
contains every score, original label, causal input, grouped metrics, separate
guard record and an exportable figure. Independent calculation reproduced all
26 probabilities within 1.11e-16 and checked metrics, prefixes and hashes.
All 238 raw files and four frozen model/report/prediction source files were
unchanged. These checks reuse two mug observation families and eight proposal
families under a changed controller; this is a retrospective development
diagnostic, not an untouched benchmark or slip warning experiment. It extends
the single false alarm observation without isolating wrist inputs as its cause.
The manuscript's quantitative pilot claims remain unchanged.

## Wrist input comparison, 12 October

Compared the 26 saved diagnostic inputs with the original 520 training rows
only. Both last wrist command angles are outside the training ranges for all
20 successful mugs; measured last angles are outside in 19/20 for WRJ2 and
17/20 for WRJ1. Original temporal statistics reproduce saved normalization
exactly. The combined wrist contributions push the unchanged linear success
score downward in all 20 successes. Their absolute contribution fraction has
median 75.5%, exceeding half in 17/20 cases; it is not a causal percentage.

The [wrist diagnostic](../research_artifacts/mug_wrist_input_diagnostic_v1/README.md)
preserves per episode and per feature attribution, training ranges and source
hashes. An independent calculation checked all 26 episode rows, 624 wrist
feature rows and eight original source hashes, reproducing the normalization
exactly and grouped contributions within 3.55e-15. This extends the input
distribution shift evidence without changing
the model or collecting data. It does not establish that removing wrist inputs
fixes transfer, or isolate controller change from object/proposal differences.
The stopped attempt remains excluded. The next comparison should separate safe
posture changes from object changes before further training. The manuscript's
earlier quantitative scope remains unchanged.

## Workshop preparation and first reading, 12 October

The user selected **CoRL 2026 Sim2Real2Field, Late Breaking Track**, and asked
for understanding and exploration before further experiments. This is a
preliminary, non archival workshop target. The paper has not been submitted.

### Does the predictor work?

In the original apple development test, it detected 187 of 202 failures, missed
15 and falsely warned on 22 of 58 successes. In the corrected mug diagnostic,
it warned on all three failures and all 20 successes. A predictor that always
warns would also catch those three failures. Failure detection alone therefore
does not establish useful discrimination or transfer to mugs.

HUG proposes a grasp from initial image and depth. A fixed controller executes
it. Our separate predictor reads joint angles, velocities and applied targets
during closing, then produces a failure score. Its original 0.5 threshold
turns that score into an alert. Later, the task scorer supplies the actual
grasp, lift and hold outcome. That outcome is not an input to the prediction.

In `mug_scene002_seed0_noise0.00`, the predictor read motion through 2.96
seconds and gave a failure score of 0.9999999785. The actual grasp succeeded.
This is a false warning, not a robot failure. The [saved score and input](../research_artifacts/corrected_mug_model_diagnostic_v1/README.md)
and [wrist comparison](../research_artifacts/mug_wrist_input_diagnostic_v1/README.md)
preserve the evidence. The near one score is not a calibrated physical failure
probability. The stopped attempt remains separate and unscored.

### Public examples and what to learn from them

The authors of [How Should a Simulation to Reality Transfer Budget Be Spent](https://github.com/YTomar79/sim2real_budget)
report in their current README that they submitted to Sim2Real2Field. This is
an author report, not acceptance evidence. Its June [arXiv version](https://arxiv.org/html/2606.22062v2)
still names an earlier IROS workshop. The paper compares system measurement
with randomization breadth in a controlled two parameter pendulum simulation,
varying two factors over ten seeds and reporting uncertainty and limits.
Its finding concerns that identifiable setting, not a general conclusion that
noise is bad. Our inferred lesson is to isolate one question and comparison;
the reported submission does not establish an acceptance standard.

[SAFE](https://arxiv.org/html/2506.09937v2) is related prior work, not a confirmed
submission to this workshop. It explains why an unfamiliar execution can
succeed, tests failure detection on unseen tasks and measures alarm performance
and timing. Its [project videos](https://vla-safe.github.io/) show successful
and failed executions. It uses internal features of large robot policies,
rather than our joint summaries. The lesson is to measure false warnings as
well as caught failures; it is not an evaluated baseline in our project.

The official program lists invited speakers rather than contributed papers.
The public venue configuration hides submissions. GitHub, arXiv, LinkedIn and
forum searches found the author report above and organizer announcements, but
did not establish a complete applicant or accepted paper list. We cannot infer
an acceptance rate or score cutoff.

### Submission requirements checked on 12 October

The [workshop call](https://sim2real2field-corl26.github.io/) welcomes preliminary
and negative results. This track permits four pages excluding references in
the linked [CoRL 2026 template](https://drive.google.com/file/d/1R9irIW0ImqeDHh5g8Ukm2XGy91sWxOsQ/view).
All authors need active OpenReview profiles, and one author must be willing to
review up to two papers. The workshop is in person in Austin on 12 November;
specific accepted author registration obligations are not explicit in the call.

The website cutoff is **15 October, 11:59 PM Central**, equivalent to
**16 October, 12:59 AM Eastern**. The public invitation timestamp is one hour
later. Use the earlier website cutoff; aim to submit during 15 October daytime
in New York. This target gives three days, not a full week.

Use the [workshop portal](https://openreview.net/group?id=robot-learning.org/CoRL/2026/Workshop/Sim2Real2Field)
and choose **Late-Breaking Track** in the shared form. It requires title,
author profiles, keywords, abstract, track and PDF, with a 50 MB PDF limit.
The current form has license, author email sharing and accepted paper public
release acknowledgments, but no supplementary upload field. See the
[public invitation](https://api2.openreview.net/invitations?id=robot-learning.org%2FCoRL%2F2026%2FWorkshop%2FSim2Real2Field%2F-%2FSubmission).

Prepare an anonymous initial PDF in the template's submission mode. This follows
the template and [standard CoRL reviewing instructions](https://www.corl.org/contributions/instruction-for-authors),
although the workshop does not separately spell out anonymity. Its four page
limit and actual form take precedence over main conference page and supplement
rules. Named GitHub links expose authorship and need a suitable anonymous
evidence presentation. Main CoRL asks authors to own content, verify citations
and disclose generative AI use; the workshop form currently has no separate
disclosure field. Prepare an accurate account of assistance for submission.

### Preparation sequence with understanding first

1. **Today:** Enoch creates and activates his profile, using an institutional
   email if available. Confirm authors, affiliations and the reviewer commitment.
   Use [OpenReview signup guidance](https://docs.openreview.net/getting-started/creating-an-openreview-profile/signing-up-for-openreview).
2. **Explain one stored attempt:** identify what the model measured, what its
   warning meant and how the actual task outcome was assigned. Understand
   this example before extending the system.
3. **Agree on one narrow claim:** ask whether a grasp outcome predictor gives
   false warnings when wrist settings leave the training range. Current
   attribution supports a mechanism in the scores, not isolated physical
   causation or a validated fix. A matched safe posture comparison is proposed,
   not run. Real transfer, slip warning and recovery remain untested.
4. **Prepare the four page paper:** introduce the question and prior work;
   explain execution, inputs and labels; present original evaluation, repeated
   mug false warnings and wrist evidence; state limits and the next comparison.
   Both authors inspect each claim and figure and can explain them before upload.
5. **Confirm submission:** upload the final reviewed PDF in the selected track,
   check the stored version and authors, and save its OpenReview identifier
   and receipt. A draft or account is not a confirmed submission. Selection
   follows review.

These are preparation steps. The existing manuscript, model and recordings
remain unchanged during this exploration.

## Prediction and recovery prior work, 12 October

The user asked whether recognizing genuine failures and correcting them would
make the project distinctive. Both remain important, but the combination has
been demonstrated before. It cannot serve as a novelty claim on its own.

[Gilday et al. 2023](https://doi.org/10.1002/aisy.202200390) predict grasp
outcomes with a sensorized soft hand and trigger a wrist translation recovery.
They compare intervention enabled and disabled on physical trials. Their
Section 3.6 also reports inaccurate predictions when novel objects succeed
through different grasp modes, suggesting insufficient diversity of successful
training examples. Their hand and tactile inputs differ from our Shadow Hand
simulation and joint summaries, but even unfamiliar successful grasps causing
poor predictions is established motivation.

[Nazari et al. 2022](https://arxiv.org/html/2209.06019) predict slip from tactile
and action histories and alter the manipulation trajectory proactively. They
compare proactive, reactive and no slip control, including novel objects.
This tests actual prevention, not only a proposed future recovery response.
As a more recent adjacent example, [FPC-VLA](https://fpcvla.github.io/) integrates
a vision and language supervisor with action correction and reports simulation
and robot experiments. None is a measured baseline in our project.

The [workshop's transfer questions](https://sim2real2field-corl26.github.io/)
explicitly include monitoring and surviving transfer failures using onboard
signals. Our possible fit is therefore a careful reliability diagnostic and
controlled comparison. A state only predictor is a different sensing choice,
but that difference alone does not establish a new method or literature gap.

### What our current experiment still lacks

1. **A controlled cause comparison.** The current mugs change geometry,
   proposals and wrist/controller conditions together. Weighted score
   attribution explains the classifier arithmetic but does not isolate the
   physical effect of posture. A declared comparison should keep object,
   scene and intended grasp fixed while testing feasible safe wrist solutions.
   Preserve world palm and intended finger geometry, validate execution and
   measure actual outcomes rather than assuming they are invariant.
2. **A validated warning improvement.** Use development evidence to select
   one representation or training change, then freeze it and evaluate on
   fresh declared object and posture groups. Measure false warnings and
   detected failures together. The current model clears no successful mug at
   its original threshold; neither more finger noise nor removing wrist inputs
   has been demonstrated to fix it. No change has been made here.
3. **A timed future event.** Final task failure is not slip onset. Establish
   the failure being forecast, when it occurs, whether the attempt is still
   viable at the warning, and the time available after inference and actuation
   delay. Acquisition failures cannot be relabeled as predicted slips.
4. **Measured recovery benefit.** Only then test a bounded correction using
   information available at the warning. Compare matched intervention and
   continuation conditions, including successful grasps that might be harmed
   by unnecessary correction. Report final success, false interventions,
   missed failures and added time. The table collision correction was an
   executor fix; it was not a recovery triggered by the prediction model.

The proposed next scientific step remains the small controlled wrist
comparison with the existing model frozen. Its planner extension and physics
checks need a declared protocol before collection. Reliable prediction and
successful recovery are separate accomplishments. With the workshop deadline
on 15 October, report completed diagnostic evidence and its limits rather than
promise a full recovery system by that date. No model change, rollout or
recovery experiment was performed during this literature review.

## Candidate directions and a bounded exploration phase, 12 October

The user asked for documented open problems and alternatives to a long wrist
specific question. The overall goal remains useful failure warning on new
conditions, with eventual correction. A short question can express that goal;
the wrist can be one experimental factor rather than the whole research topic.

Current primary work supports an active reliability problem, not an empty
research area. [FIPER](https://arxiv.org/html/2510.09459v2) already distinguishes
unfamiliar successful executions from failures. Its limitations discuss costly
false alarms, calibration data requirements and restricted policy modalities.
The August 2026 [SAFECAST preprint](https://arxiv.org/html/2608.04246v1) already
studies perturbed training and calibration. It reports that unseen perturbations
can still degrade detection, empirical gains do not restore guarantees under
arbitrary shifts, and pooling many additional perturbations generally fares
worse than its focused sets, with exceptions depending on model and metric.
Its visual/language VLA study does not supply a demonstrated solution for our
fixed controller and joint state predictor. [RAPT](https://arxiv.org/abs/2602.01515)
also studies efficient interpretable monitoring with joint summaries for
humanoid deployment; joint measurements or a lightweight monitor alone are
not novelty claims.

Three candidate questions remain proposals:

1. **Which simulation variations help grasp warnings transfer?** At equal
   collection/training budgets, compare more repetitions of finger noise
   with broader successful grasp and posture coverage. This directly addresses
   the workshop's simulation diversity question. It requires new grouped data,
   controlled training comparisons and fresh held out evaluation. Existing
   records show limited coverage, not a demonstrated augmentation benefit.
2. **Do warnings survive a controller or posture change?** Keep the current
   model fixed and separate execution factors in matched safe comparisons.
   Current false warnings and input attribution provide preliminary evidence;
   an isolated posture comparison requires explicit feasible pose selection,
   geometry invariance and dynamic execution checks before fresh rollouts.
3. **When should a warning be trusted?** Study a decision threshold or withheld
   decision policy under changed conditions. Distinguish unfamiliarity from
   actual failure and report missed failures, false warnings and withheld
   decisions together. Tune on development data only and evaluate separately.
   This would be an empirical study, not a universal safety guarantee.

Recommendation: finish one focused reading and decision pass before further
model or collection changes. Existing data verification and explanation can
continue while exploring. End the pass with **one question, one comparison and
one criterion for improvement**. Preserve the saved model as the baseline.
The existing diagnostic is the smallest immediate workshop scope; a data
diversity study is a broader follow up requiring new work. The user has not
selected a new direction, and no candidate's novelty is established here.

## Selected question and proposed wrist comparison, 12 October

The user chose useful simulation diversity as the broad question and a
controlled wrist comparison as the next diagnostic. The [proposed protocol](../research_artifacts/wrist_comparison_plan_v1/protocol.json)
records the case list, frozen hashes, comparisons and interpretation rules.
Its status is **planned, not executed**. No model or recording changed.

### The research question

**Does varied simulation data make grasp warnings more reliable in new conditions?**

Operationally: at the same training episode count and success/failure balance,
does adding coverage of successful safe wrist postures reduce false warnings
on new object/posture groups more than adding more finger command noise?
Both eventual training variants must use the same model family, feature fields,
validation selection rule and base object/proposal identities. The benefit of
posture coverage remains untested. This comparison follows the original noisy
data plan and asks which variation helps transfer.

### The first comparison with the model unchanged

For the same saved mug proposal, compare two feasible wrist settings with the
same object, scene, initial object state, finger targets, planned world palm
path, timings, physics and zero noise. Compensate carrier placement so the
nominal palm and finger grasp stays the same. The model, normalization,
closing inputs through 2.96 seconds and 0.5 alert threshold stay frozen.

Setting B is the case's recorded corrected wrist pose. First seek setting A
among wrist command pairs observed in the original training partition, using
geometry alone to test feasibility. If none fits, choose a distinct feasible
candidate closest to the training command distribution and record its distance.
Selection cannot use candidate prediction scores or new outcomes. A familiar
command does not imply all measured model inputs are familiar.

Use the eight existing mug proposal families from two scenes, with at most
two future executions per eligible case: **up to 16 diagnostic trials**.
These are development cases, not a fresh generalization benchmark. The current
executor selects one pose; explicit pair selection is not implemented yet.
The exact A/B angles must be declared in a feasibility manifest before any
rollout. A training range pose might not clear the table. Record missing pairs
instead of forcing unsafe settings or replacing cases based on their scores.

Both settings must satisfy the existing nominal geometry gates and the runtime
1 mm table margin. Record measured palm/finger tracking, scores, alerts and
outcomes. Nominal preservation does not imply identical forces or motion.
Report the entire pair outcome matrix, including changed outcomes and stops.
Rejected or stopped attempts have null labels and remain outside scored metrics.

The cleanest comparison is a pair in which **both grasps succeed**. If their
alert decisions differ, this shows that a successful intended grasp can receive
different warnings across the paired wrist setups. Report score, linear logit
and wrist/nonwrist contributions too. Repetition across at least two proposal
families is qualitative repeatability, not a statistical reliability guarantee.
If actual outcomes change, the pair cannot demonstrate a harmless posture
change. If both settings remain unfamiliar or scores saturate near one, no
alert change does not rule out wrist sensitivity.

### What counts as improvement

This first comparison is diagnosis; it cannot improve unchanged model weights.
Later improvement requires, at the same 0.5 threshold on a common freshly
declared test set, **fewer false warnings among successful grasps without more
missed failures**. Report counts, rates, object/posture breakdowns and uncertainty
that respects related proposal/scene families. Use matched training budgets
and label proportions so simply adding more success labels does not masquerade
as better posture coverage. Keep test groups outside training and validation.

Lower scores alone, a result on different test populations, an always warn
predictor or a never warn predictor do not meet this criterion. If the test
contains no failures, failure detection is undefined. Very few failures cannot
establish broad noninferiority. The three current mug failures do not resolve
that limitation. Useful lead time, intervention benefit and real transfer are
still separate untested claims.

### Direct next actions

1. Check whether two usable wrist settings exist, using geometry and the
   original training distribution before any new outcome or prediction.
2. Implement and audit explicit posture selection, preserving the existing
   controller records. Freeze exact angles, source hashes and invariance gates.
3. Run one eligible pair first and inspect its input, warning and physical
   outcome, then execute the remaining declared eligible pairs.
4. Review the diagnostic before choosing the size and implementation of the
   equal budget data diversity experiment or changing the model.

The completed foundation is the simulated execution and labeling pipeline,
the 1,040 cube/apple pilot and trained baseline, the 2,080 fresh object/scene
evaluation, the separate mug executor correction, frozen scoring of 26 completed
checks and the wrist training distribution audit. Paper evidence notes and a
working manuscript exist. The controlled pair comparison, data diversity fit,
useful warning timing, recovery and workshop submission are still pending.

## Accepted paper question and evidence gates, 12 October

**Working title: Evaluating Simulation Diversity for Grasp Failure Warnings.**

**Accepted research question:** For a simulated dexterous hand, does training
with more diverse successful wrist configurations improve prediction of grasp,
lift and hold outcomes on unfamiliar objects and wrist configurations more than
additional finger-command noise, when both approaches use the same amount of
training data, the same success/failure proportions and the same model
architecture?

In plain language: **which simulation variations make grasp warnings more
reliable in new conditions?** The wrist diagnostic comes first because we need
to understand the existing false alarms before testing a training remedy.
Neither the wrist coverage benefit nor its novelty has been established.

### What the completed evidence establishes

The unchanged predictor alerted on **all 23 completed corrected mug attempts**:
**20 successful grasps were false alarms and three failed acquisitions were
detected**. This repeated at all three finger noise bounds, including zero.
One guard stopped attempt remains separate with no binary label or prediction.
See the [frozen scoring evidence](../research_artifacts/corrected_mug_model_diagnostic_v1/README.md).

Both last wrist commands in all 20 successful mugs lie outside their respective
original training ranges. Wrist terms push their saved linear scores toward
failure, but object geometry, proposals and execution also changed. The
[wrist input audit](../research_artifacts/mug_wrist_input_diagnostic_v1/README.md)
is an attribution of existing scores, not an isolated physical cause or proof
that changing features or training data will fix the model.

The paper revision connects this completed evidence to the accepted question.
The original pilot and fresh v2 results retain their own controller versions,
counts and limitations. The v3 correction is an executor fix, not a successful
model triggered recovery experiment. This revision adds no training or rollout
result.

### What must be checked before the next claim

1. **Feasibility and comparable execution.** Select safe A/B wrist settings
   using geometry and the original training distribution, without new scores
   or outcomes. Freeze exact angles, compensated carrier placement, source
   hashes and geometry tolerances. Implement explicit selection before any
   rollout; then measure actual palm and finger tracking. If no safe familiar
   setting exists, report that instead of implying a familiar versus unfamiliar
   comparison.
2. **Interpretable frozen diagnosis.** Use zero noise, the unchanged model and
   the same intended grasp. Publish all declared pairs, changed outcomes and
   guard stops. A different warning when both attempts succeed is useful
   diagnostic evidence; it does not by itself show improved prediction.
3. **Controlled training comparison.** Later compare the same base collection
   plus equally sized additional finger noise or wrist coverage cohorts. Match
   success/failure proportions and base object/proposal identities; retain the
   architecture, feature contract and validation selection rule. Fit scaling
   on each training partition only. An infeasible match requires a declared
   protocol change, not an undisclosed substitute comparison.
   Declare cohort selection before fitting and publish generated, selected and
   excluded counts. Equal selected training counts do not mean equal collection
   effort. Score the original frozen model on the same fresh test as a reference;
   its smaller training budget does not isolate the two augmentation strategies.
4. **Fresh evaluation and useful improvement.** Freeze both fitted models
   before examining a common new test set whose object, posture, observation
   and proposal groups are kept outside fitting and selection. At the same
   0.5 threshold, seek fewer false warnings on successes without additional
   missed failures. Report counts, rates, group breakdowns and uncertainty.
   This criterion is not a statistical noninferiority guarantee; three mug
   failures do not establish broad reliability.
5. **Honest research scope.** Compare against relevant prior work, preserve
   negative findings and distinguish unfamiliar inputs from genuine failed
   grasps. Inputs must remain earlier than the scored outcome. Do not claim
   physical slip prediction, useful intervention time, recovery or real robot
   transfer from this task outcome experiment. A contribution depends on a
   convincing comparison and an informative finding, not on adding complexity.

The [declared protocol](../research_artifacts/wrist_comparison_plan_v1/protocol.json)
contains the eight development mug proposal families and exclusions. Exact
pair feasibility and executor support remain the first gates; up to 16 paired
executions are proposed, not collected. These reused cases cannot become an
untouched generalization benchmark by changing the paper title.

### First static feasibility result

A read only kinematic check tested the **eight unique last wrist command pairs
actually observed among the 520 training episodes** against the eight declared
mug proposals, preserving the intended world palm/finger geometry. **None of
these 64 combinations met the 10 mm nominal table clearance gate**. Clearances
ranged from approximately **−65.211 mm to +0.187 mm**. All eight stored corrected
setting B poses revalidated at approximately **10.001 to 10.042 mm** nominal
clearance. Physics stepping was blocked; no model scoring or rollout took place.

This finite check rules out immediately reusing those stored training command
pairs as safe familiar settings in these cases. It does not prove that no
continuous pose inside the training command ranges could work, and nominal
clearance does not establish dynamic safety. No A/B feasibility manifest is
frozen yet. The next design choice is to find another valid familiar setting
or controlled case, or explicitly qualify a comparison between two unfamiliar
safe settings as posture sensitivity. The full matched training diversity
question remains unchanged. Do not force unsafe settings to preserve a desired
comparison label.

The [static feasibility bundle](../research_artifacts/wrist_pose_feasibility_v1/README.md)
records every candidate, failed gate, compensated carrier pose and source hash.
The existing working manuscript was updated in place and compiled successfully
with the desktop editor compiler. This confirms source compilation; final
workshop template compliance and author review remain pending.

### Workshop and partner preparation

The target is the **Sim2Real2Field CoRL 2026 workshop Late-Breaking Track**,
a nonarchival research workshop. Its simulation diversity and deployment
monitoring themes fit this empirical question; fit and acceptance remain
unproven. The [official call](https://sim2real2field-corl26.github.io/) requires
at most four pages excluding references using the standard CoRL template.
Final condensation, template packaging, anonymization, reference and claim
review, and active OpenReview profiles for every author remain submission tasks.
Use the earlier official site deadline, **15 October 2026 at 11:59 PM CDT**
(16 October at 12:59 AM Eastern), and aim to finish during the Eastern daytime.
The working manuscript is not itself a completed submission.

Enoch asked for tests on new things before assuming a better model was needed.
The frozen diagnosis and subsequent controlled comparison address that concern
without first replacing the predictor. That alignment is an interpretation of
his messages, not confirmation that he agrees with the scope or timeline. The
[current partner handoff](PARTNER_HANDOFF.md) explains the actual data source,
completed results and first bounded task for discussion.

## First safe wrist pair, 12 October

**Completed:** freeze eight A/B wrist pairs using geometry and training distance
alone, implement an exact-pose v4 executor, then execute and score only the first
pair with zero noise and the original model. The seven other pairs remain
unexecuted. The matched training-diversity experiment is still planned.

**Simple result:** both actual grasps lift and hold the mug successfully, but
the model gives a failure warning for both. Each is a false alarm. Success is
the physical task label; the alert is a separate prediction from joint positions,
velocities and commands during closing, before lifting.

The [selection manifest](../research_artifacts/wrist_pair_selection_v1/README.md)
records every case, training-distance/range flag, angle, carrier pose, gate and
source hash. Both selected wrists remain unfamiliar in all eight cases.
The predeclared separation rule is at least 0.02 rad; the first pair differs
by 0.028686 rad. A is slightly nearer training in this pair; that is not true
for every selected case and does not mean the model must score it safer.

The first pair is `mug_scene002_seed0_noise0.00`. A commands
`[+0.1136282552, −0.2020766563]` rad at WRJ2/WRJ1, B
`[+0.0990838385, −0.2268021563]` rad. Nominal clearance is 10.062933 and
10.003215 mm. Actual peak lifts are 14.6208 and 14.6077 cm; both satisfy the
2 s HOLD requirement. Neither stops or contacts the table.

The unchanged model's failure scores are 0.9999999871415051 and
0.9999999784968061. Success logits are −18.169261 and −17.655064; wrist terms
change by +0.529098 and nonwrist terms by −0.014901 from A to B. Neither alert
changes. These are uncalibrated scores; a rounded display of 1.000000 does not
mean a measured 100% physical failure probability.

The [paired evidence and replay](../research_artifacts/wrist_pair_diagnostic_v1/README.md)
includes original prediction inputs, outcomes, reports and independent checks.
All 22 applied finger histories are identical. Nominal palm/finger paths agree
within 1.67e−16 m; actual maximum paired palm and finger geometry differences
are 0.05890 and 0.35723 mm. All 6,520 replayed physics steps reproduce states
bitwise without table contact. B also reproduces its earlier v3 motion, apart
from signed-zero noise metadata. Separate v4 results are not pooled with the
old 23 v3 mugs as an independent benchmark.

**Quality and audit status:** 21 focused executor tests passed and selection
reproduced byte-for-byte with 15,654 independent checks. The independent audit
hand reconstructs the causal prefixes, normalization and scores. Its source was
still being finalized when included in the execution declaration; an explicit
auditor-only amendment and exact historical source snapshot preserve that
correction. The original plan, scorer report, executed poses, model and
recordings remain unchanged. The paper includes this measured first pair and
compiles with the desktop compiler.

**What remains:** review this example, then check repeatability over the remaining
seven frozen pairs before any training change. Neither broad transfer nor a
training remedy is demonstrated. No failures occur in this pair, so failure
recall is undefined. The workshop template, page limit, anonymous packaging and
author review remain submission preparation tasks.

## Seven remaining wrist pairs completed, 12 October

**Execution:** all seven remaining declared pairs ran once per wrist setting,
with zero noise, unchanged poses, the same v4 executor and the original model.
All **14 new attempts completed**, with no guard stops or hand/table contacts.
The first pair and its evidence remain unchanged and were not run again.

**Predictions versus outcomes:** the model warned on all 14. **Eleven actual
successes were false alarms; three failed task labels were correctly flagged.**
Five new pairs succeeded in both A and B, and neither setting cleared the alert
in any of those pairs. One pair failed in both; one failed in A and succeeded
in B. Including the preserved first pair gives eight pairs and 16 attempts:
13 successful false alarms, three detected task failures, six both-success
pairs, one both-failure pair and one A-failure/B-success pair. The full
[paired report and figure](../research_artifacts/wrist_pair_remaining_v1/README.md)
keep these stages and outcomes separate.

**Label boundary:** in scene 003 seed 2, A peaks at 13.996398 cm, only
0.036021 mm below the 14 cm cutoff; B peaks at 14.020459 cm. Both satisfy
qualifying contact and the sampled 2 s HOLD rule. Labels remain 0 and 1;
this small cutoff crossing does not establish slipping, dropping or recovery.
The other two failed acquisitions peak at about 13.04 and 13.02 cm. These
three failures come from two proposal families, not three independent objects.

**Interpretation:** at the fixed 0.5 threshold, this batch behaves like an
always-warn rule. The false alarms repeat across six both-success proposal
pairs. Descriptive AUROC is 0.9091 for the new batch and 0.8974 for all 16,
so scores retain some ranking information; we do not conclude that the model
contains no useful information. Those tiny development counts do not establish
reliable detection, calibrated probabilities or justify tuning a threshold
on these cases. Both wrist settings remain outside original training ranges.
Actual motion also changes, so this is not a pure input intervention or an
untouched generalization benchmark.

**Verification:** the finished scorer and independent auditor were bound before
new inference. No new audit-source exception is needed. All 14 predictions
were saved before reading task-result files. Original training normalization
reproduces bitwise and manual probabilities agree within 1.39e−17. All 45,640
new physics substeps replay exact states with no hand/table contact; all 22
finger command histories match within pairs. Model, poses, source records and
the old first-pair bundle remain unchanged. Current pair tracking and causal
input exports are preserved in the new bundle. An actual separate scoring
rerun reproduced the report, predictions, dataset and pre-outcome prediction
record byte-for-byte.

**Next design decision:** the declared diagnostic is complete. The matched
wrist-coverage versus additional finger-noise training study remains unexecuted.
Define equal budgets/class mix/base identities, separate validation and a
fresh common test before fitting. A validation-fitted calibration reference
would check whether a simple score adjustment helps as much as new training
coverage. That comparison is proposed, not fitted; preserve the current 0.5
result and original labels. The paper now reports the full diagnostic and
compiles. Workshop formatting and author review remain pending.
