# Literature review for the noisy grasp pilot

Reviewed 10 October 2026. This note connects three papers to the project's six-step plan and records proposed refinements for the first predictor experiment. Keep the fixed controller and joint-angle bias as the collection baseline. Complete a small, reproducible prediction experiment before scaling collection or adding recovery.

The recommendations below are design decisions for the next pilot, not completed experimental results. They extend the existing [system design](SYSTEM_DESIGN.md) and [research foundations](RESEARCH_FOUNDATIONS.md); they do not establish a new research contribution by themselves.

## 1. AHA and FailGen: organize how failures are generated

FailGen perturbs successful simulated demonstrations at selected waypoints to produce named failure mechanisms. AHA recognizes and explains the current subtask outcome from image history; it does not establish advance prediction of grasp loss. Its official slip implementation deliberately releases and opens the gripper, rather than modeling naturally emerging friction-induced slip.

Our adaptation is to distinguish the injected mechanism from the observed outcome, keep perturbed successes as well as failures, and record disturbance time separately from failure time. We need only mechanisms relevant to our grasp/lift task.

Read [AHA, sections 3.1–3.2 and 4.1](https://arxiv.org/html/2410.00371v1), and the official [SlipFailure implementation](https://github.com/NVlabs/AHA/blob/main/aha/Data_Generation/rlbench-failgen/failgen/fail_slip.py).

## 2. Predictive Learning of Error Recovery: timing changes usefulness

Gilday and colleagues use real robot trials, pressure-sensor histories and an LSTM to predict grasp outcomes. Their perturbation is a constant planar shift of a replayed wrist trajectory. Predictions trigger a heuristic recovery; waiting for more confident predictions reduces recovery opportunities. Their binary convention is 1 for failure and 0 for success, opposite ours.

Our adaptation is to evaluate predictions using only the observed history and measure their timing. Their soft-hand sensing and recovery results do not establish performance for our Shadow Hand. Split original episodes before deriving training clips.

Read [sections 2.4–2.6 and 3.5–3.6](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202200390).

## 3. FAIL-Detect: test whether collecting failures adds value

FAIL-Detect learns anomaly scores from successful demonstrations and calibrates alarms using independent successful policy rollouts. Its reported first-alarm time starts at rollout initiation; that is not necessarily warning lead time before failure. Environmental shifts can also increase false alarms.

Our adaptation is a success-only observation baseline compared with the supervised predictor. Tune its threshold using reserved successful validation rollouts and evaluate both models on the same successful and failed test trials. This is an inspired baseline, not a reproduction or a claim of conformal guarantees. Stochastic-policy action-distribution methods do not directly fit our fixed controller.

Read [sections IV–VII and Appendix B](https://arxiv.org/html/2503.08558v3), with the [official implementation](https://github.com/CXU-TRI/FAIL-Detect).

## Refinements to the six steps

### Step 1: review datasets and benchmarks

Add this method review to the dataset audit. For any external recording, check what its label means, which observations are available, whether failure timing is identifiable, and whether its robot matches our inputs. Keep official benchmark test objects out of tuning. A compatible dataset is useful; a new publication date alone is not a selection criterion.

### Step 2: generate varied simulation trials

Retain uniform episode-level finger bias and the no-noise control. Document the injection rule, bounds, units, active phases and clipping. Add one relevant mechanism at a time only after validating it. Preserve nominal friction initially, as specified in the existing plan; later friction or mass studies need separately justified ranges and controls.

### Step 3: collect trajectories

Retain both outcomes, including noisy successes. Save synchronized measured states, commands, timestamps, source-grasp grouping and perturbation provenance. Keep sampler seeds, injected mechanism and hidden simulator properties as analysis metadata, not predictor features. Audit the logs before increasing volume. The current pilot collects no per-frame camera or tactile observations; initial RGB-D references exist. Scope the first model to logged commands and measured robot state.

### Step 4: label outcomes and events

Keep **1 = success and 0 = failure**, with invalid execution samples separate. Add an observed failure category and event time or acquisition deadline when definable. A contact flag alone does not establish a secured grasp. Define acquisition and subsequent loss before labeling drops. Mark unknown timing explicitly; disturbance time is not a substitute for an event label.

Final outcome supports eventual-outcome prediction from a trajectory prefix. Predicting loss within a future horizon requires a stated horizon and enough observed future context. Do not label every frame of an eventually failed episode as imminent failure. Keep observed misses and post-acquisition losses separate in the analysis.

### Step 5: create independent splits

Split complete episodes and source-grasp groups before creating windows or augmentations; keep paired noise variants together. Reserve unseen objects for final evaluation once multiple objects are supported. Use validation for model selection and a separately reserved calibration subset for alarm thresholds. Never tune on the test set; report group counts and class balance for each partition.

### Step 6: train and evaluate simple predictors

Compare a supervised prefix predictor with a current-state baseline and a success-only observation monitor. Use measured state/velocity and command-tracking features available by prediction time; simulator-only contact features need an explicitly separate oracle analysis. An initial simple observation baseline need not implement the full FAIL-Detect framework.

Report missed failures, false alarms per successful episode and warning lead time relative to the defined event. Include failures with no alarm; fast timing on detected cases alone can hide poor coverage. Exclude alarms after the event from advance-warning claims. Report inference latency; useful recovery claims additionally require intervention delay and outcome experiments.

## Decision record and completion checkpoint

- **Keep:** the six-step structure, fixed execution controller, angle-bias baseline and final binary labels.
- **Preserve and refine collection:** existing source groups, provenance, coarse failure categories and invalid handling; refine observed categories and add separate disturbance/event timestamps and unknown-event handling.
- **Add to evaluation:** current-state and success-only comparisons, grouped/object holdouts, false alarms and warning timing.
- **Defer:** more complex models, large-scale collection, recovery experiments and real-world transfer claims until the small pilot gives an honest comparison.

The local 9 October collection contains 50 one-cube trials: 40 successes and 10 failures. All 10 failures have the coarse `failed_acquisition` category. These results validate logging and label plumbing; they do not establish post-lift slip prediction. Evidence is in the locally generated `runs/noisy_pilot_2026-10-09/pilot_report.json`; raw trajectories are not redistributed in this documentation change. The existing local collector needs integration, publication and review; broader object support and event annotations still need implementation.

Finish this pilot when its recorded data, grouped splits, simple baselines and evaluation can be reproduced. A negative result is still a completed experiment. Use that evidence to choose whether to improve sensing/data, expand conditions or reconsider the target.

## How to use this in a future paper

The rationale can be described now; adopted implementations and experimental results must be reported only after they exist. A suitable design statement is:

> Our initial plan used bounded joint-target perturbations and final grasp outcomes. The literature review motivated separating disturbance mechanisms from observed failures, recording event timing, and comparing supervised prediction with a success-only monitor. We retain the original collection baseline and propose grouped holdouts and timing-based evaluation to test whether warnings are useful before failure.

Use the three citations above in the corresponding related-work and method sections. If experiments later adopt these changes, describe the exact protocols and report their comparisons. Do not claim that these papers validate our simulated contact physics, show our expected accuracy, or establish our novelty.
