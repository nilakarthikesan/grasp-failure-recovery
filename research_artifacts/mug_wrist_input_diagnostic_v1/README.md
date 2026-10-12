# Why did successful mugs trigger warnings?

12 October 2026. This compares the existing 26 scored recordings against the
original **520 training episodes only**. No model weights, input features,
normalization, thresholds, recordings or controllers were changed. No new
simulation, collection or training was performed.

## The experiment in plain language

1. **The simulator performs the grasp.** A fixed controller moves the hand.
   The prediction model does not control this motion. A successful recording
   means that the hand actually grasped, lifted and held the object according
   to the existing task scorer. Its label is 1. A completed failed task has
   label 0.
2. **The model makes a separate guess.** During closing, before lifting, it
   estimates whether the task will fail. A failure score of at least 0.5
   triggers an alert. "Flagged" means this alert, not that the object fell or
   that the simulator stopped the hand. A successful grasp with an alert is
   a false alarm.
3. **Its inputs are robot measurements and commands.** Each of 20 samples
   through 2.96 seconds contains 24 joint angles, 24 joint velocities and
   24 commanded joint angles. Two joints are the wrist: WRJ2 and WRJ1.
   Joint angle describes a joint's position; velocity describes its movement
   speed; a command is the target angle the controller asks it to reach.
   The model does not receive camera images, mug identity, outcome labels,
   object state, contacts or future lifting motion.
4. **We compared those wrist inputs with its training examples.** Both wrist
   commands in all 20 successful mugs are outside the original training ranges.
   Last measured angles are outside those ranges in 19/20 successes for WRJ2
   and 17/20 for WRJ1. Commands and actual measured angles are distinct.
5. **We examined the existing model's arithmetic.** It summarizes the closing
   samples, scales them using original training statistics, then adds weighted
   contributions. The combined wrist terms push its success score downward in
   all 20 successful mugs. Unfamiliar wrist settings are therefore a concrete
   contributor to these failure warnings.

## What was measured

Both last wrist command ranges are disjoint between training and successful mugs:

* **WRJ2:** training −0.2039 to −0.1764 rad; successful mugs −0.1464 to +0.1745 rad.
* **WRJ1:** training −0.0394 to −0.0146 rad; successful mugs −0.5312 to −0.0817 rad.

For intuition, the WRJ2 training command range is about −11.7 to −10.1 degrees;
successful mug commands span about −8.4 to +10.0 degrees. These signed values
use that joint's coordinate convention, not a camera viewing angle.

We reconstructed the 288 temporal statistics from the training partition and
matched the saved normalization exactly. The model has 269 active features,
including 20 wrist features. Every successful mug has four to eight active
wrist statistics outside their training ranges. Their net wrist contributions
to the existing success logit range from −25.7648 to −2.1716; negative values
push toward the failure side of the saved linear decision rule.

Wrist terms account for a median 75.5% of the absolute weighted feature
contributions among successful mugs, with a range of 31.0% to 86.9%. They
exceed half in 17/20 cases. This fraction excludes the intercept and measures
score arithmetic, not a percentage of physical causation or error. It would
be incorrect to say wrist features dominate every case. Finger measurements,
commands and the intercept also contribute.

## What this does and does not explain

The model extrapolates beyond a narrow set of training wrist commands. Its
coefficients are not an explicit "unfamiliar means failure" rule, but their
weighted contributions move these particular predictions toward failure.
The earlier noise experiment varied finger commands; it did not ensure coverage
of the successful mug wrist settings. More finger noise alone does not answer
this input coverage question.

This explains a shared mechanism in the saved scores. It does not isolate why
the wrist relationship was learned, prove that the controller correction alone
created the shift, establish physical wrist danger, or show that removing wrist
features will improve unseen object performance. Mug geometry, grasp proposals
and execution conditions differ together. The recordings are reused development
checks, so this is not a fresh independent benchmark.

**Proposed next check, not performed:** design a small comparison of matched
safe wrist postures for the same intended grasp, with invariance and execution
checks. That can separate posture sensitivity from object changes. Declare the
comparison before further collection or fitting; retain a separate fresh
evaluation for any later model change. No feature removal, replacement inputs,
hypothetical modified predictions or new data were generated here.

The one earlier guard rejection has no complete model input or outcome and
remains outside this analysis, as recorded in the
[scoring diagnostic](../corrected_mug_model_diagnostic_v1/README.md).

## Evidence

* [report.json](report.json): original source hashes, training comparisons and scope.
* [episode_attribution.csv](episode_attribution.csv): exact wrist/nonwrist contributions for all 26 scored recordings.
* [wrist_features.csv](wrist_features.csv): every wrist statistic, original training range, coefficient and contribution.
* [independent_audit.json](independent_audit.json): independent checks of the published values and unchanged sources.
* [manifest.json](manifest.json): packaged file hashes.

From an environment with this repository installed, use a new output directory:

```bash
python scripts/analyze_mug_wrist_inputs.py \
  --output /tmp/mug_wrist_input_diagnostic_reproduction
```

The script reads the original model and data, verifies original training source
hashes and normalization, and reconstructs the same existing scores. It does
not call a training or simulation routine.
