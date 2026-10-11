# Paper progress: controlled grasp collection pilot

**Updated: 11 October 2026.** This is the evidence index for the
[standalone working paper](paper/working_draft.tex). The draft's title and claims
describe a collection protocol and pilot. Future-slip prediction, recovery, and
real-world transfer are research goals, not completed results.

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

## 6. Train and evaluate a prediction model — planned

The frozen first protocol predicts the unchanged eventual task label from the
**last 20 CLOSE_FINGERS samples, ending at 2.96 s**, before first LIFT at 3.00 s.
Whitelist 24 measured hand joint positions, 24 joint velocities, and 24 actual
commanded targets: a 20 × 72 window. Last/mean/standard-deviation/end-minus-start
summaries give 288 features. Use L2 logistic regression with SciPy L-BFGS,
training-only scaling, and a fixed regularization grid selected by validation
log loss. The objective is mean binary cross entropy plus half lambda times
the squared weight norm, with unpenalized intercept.

Compare a command-only 96-feature ablation and a training-prevalence constant
baseline. Keep the decision threshold fixed at failure probability ≥ 0.5; score
test only after regularization selection. A larger temporal model remains a
later comparison.

Exclude future samples, final outcomes, event metadata, noise vectors/settings/
seeds, source identities, simulator object truth, and full integration state from
predictor features. This first task is eventual-outcome prediction. Fixed-horizon
future loss and useful warning time require a separately validated event target
and more supported loss cases. Planned measures include failure-class AUROC/
AUPRC, Brier score, log loss, and fixed-threshold results, with
group/object/noise breakdowns where feasible.

**Result record pending:** Add baseline source, feature whitelist, prefix cutoff,
split hash, metrics, regularization/threshold choices, and limitations here after
execution. No trained predictor or recovery result is currently reported.

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
  evaluation are still pending.

External VR failure data and synthetic human-video generation remain separate
exploratory tracks. The former needs a source/sensor/outcome audit; the latter
has no implemented method or measured benefit. Neither blocks completing and
evaluating this controlled simulation baseline.
