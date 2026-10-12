# SAFE real robot recording walkthrough

Inspected 12 October 2026. This is an offline source and recording audit, not
a newly trained monitor, simulation experiment or reproduced SAFE evaluation.
The original Shadow Hand predictor, weights, threshold and recordings remain
unchanged.

## Start with one grasp that works and one that does not

The [official SAFE repository](https://github.com/vla-safe/SAFE) links
[real WidowX/OpenVLA recordings](https://drive.google.com/file/d/1EwaccasZjnlM9L6SEYyWqTd7d6-BR9zp/view).
We inspected its ZIP directory and extracted only six members: a video, action
CSV and saved-feature record for each of two episodes. The selection was fixed
before watching: the first carrot collection in archive order, then the lowest
raw episode index for each recorded outcome. This is an illustrative pair,
not a representative accuracy estimate or a controlled wrist experiment.

Both episodes ask the robot to **put the carrot on the plate**:

1. **Success, raw episode 3:** the robot approaches the green end, closes around
   it, carries the carrot above the plate at frame 30 and releases it onto the
   plate by frame 49. The source filename and passive metadata inspection both
   report success.
2. **Failure, raw episode 0:** it approaches and closes near the carrot, but at
   frame 30 the raised gripper is empty and the carrot remains on the table.
   By frame 49 the robot has returned toward it; the plate is still empty. The
   source reports final task failure. An unsuccessful acquisition attempt is
   distinct from an irreversible failure: a retry could have succeeded.

These are Codex visual observations, not independently reviewed event labels
or contact-sensor measurements. Initial carrot pose and location differ, so
the pair cannot isolate wrist effects. Each video has 50 frames at 5 fps; its
ten seconds of encoded playback are not verified physical execution time.
Equal counts of video frames and CSV rows do not alone prove synchronization.

The source IDs, extraction checks and observations are recorded in
[the audit bundle](../research_artifacts/safe_recording_audit_v1/README.md).
Downloaded data remain local under
`/Users/nilakarthikesan/grasp-failure-prediction/runs/safe_recording_audit_2026-10-12/`.
They are not added to the portable training dataset or redistributed through Git.

## What does the existing monitor see, and what does it miss?

SAFE is a small learned monitor that reads saved internal representations of
a VLA controller. Its failure score is compared with a time-dependent
calibration band. It is separate from the robot controller and from our
joint-only Shadow Hand model. See [the SAFE paper](https://arxiv.org/html/2506.09937v1).

The two raw WidowX CSVs have 50 rows, seven action components and token
uncertainty statistics. Passive inspection of their PKLs identifies the
task/outcome fields and saved hidden-state tensor records. No monitor scores
or thresholds occur in the CSVs. The archive has 532 CSV/PKL/MP4 triplets;
no archive pathname identifies a checkpoint, calibration band, score file or
license. The inspected public Git release likewise provides training code,
but no trained monitor checkpoint was found. Archive filename checks do not
prove that every payload has been inspected.

Therefore **we have not scored these two raw episodes with the published SAFE
monitor and cannot report its false alarms or misses on them**. Token
uncertainty columns are not SAFE predictions. A faithful evaluation needs
the exact trained checkpoint/configuration, representation aggregation and
calibration artifacts, or a separately named reproduction trained with
declared development/calibration/test partitions. Our model's 0.5 threshold
must not be transferred to SAFE's potentially cumulative score.

There is useful qualitative evidence in the
[authors' demonstration page](https://vla-safe.github.io/). Its separate
Franka/pi0-FAST pair asks the robot to put both a carrot and a ball into a bowl:

* [Successful demo, displayed episode 45](https://vla-safe.github.io/static/videos/pi0-fast-real-2-succ.mp4):
  the carrot is visibly carried around 11 seconds, the ball is handled around
  17.5 seconds, and both are in the bowl by the inspected 33 and 50 second
  frames. The displayed score stays below the alert boundary in the inspected
  trace; no red warning border is visible at those sampled moments.
* [Failed demo, displayed episode 37](https://vla-safe.github.io/static/videos/pi0-fast-real-2-fail.mp4):
  repeated carrot acquisition attempts do not complete the task. At encoded
  playback 20.0 seconds the carrot is still on the table and the frame border
  is green; at approximately 25.25 seconds the border is red. The warning is
  also present at inspected later frames around 33 and 43.5 seconds.

This example is **an eventual detection, not a demonstrated missed task
failure**. The sampled transition suggests a timing question; it is not a
validated physical delay, a first-alert timestamp or a measured improvement.
These are author-selected demonstrations, not our raw WidowX episodes. Red
means an alert was raised; it does not establish a recovery intervention.

## How our existing work fits

Our simulator can provide controlled examples of acquiring an object versus
lifting an empty hand, with saved commands, states, outcomes and replay. The
existing provenance, grouped splits and false-alarm accounting are useful
infrastructure. Our wrist diagnostic motivates checking **harmless variation**:
a warning should not interrupt an otherwise successful grasp just because a
configuration is unfamiliar.

That is a candidate test, not evidence that SAFE has our wrist problem. The
current Shadow Hand executor is fixed and produces no VLA hidden states.
Its joint vectors and predictor cannot be applied to WidowX or Franka records.
The WidowX CSV's rotation commands are not measured absolute wrist angles.
Changing a recorded video does not change its already-saved VLA features.
A common visual check could be studied separately in both domains; a SAFE
latent-feature simulation experiment requires a compatible VLA and robot
environment or a new, validated integration.

## Next small evidence gate, before changing a model

The provisional practical objective is: **recognize an unsuccessful grasp
attempt sooner, while avoiding unnecessary warnings during successful grasps**.
This remains a candidate, not our final novelty claim. Earlier timing and
visual task-progress monitoring already have related work, including
[Hide-and-Seek](https://arxiv.org/html/2605.30834v1),
[FailureSpot](https://arxiv.org/html/2609.04277v1) and
[Sentinel](https://arxiv.org/html/2410.04640v1).

1. Declare a small development inventory by task and collection, keeping
   source IDs. Include successes, empty acquisitions, successful retries and
   later placement failures if available. Select without monitor scores.
2. Review acquisition attempt intervals and uncertainty without looking at
   scores. Keep final task outcomes separate; verify video/action/feature
   alignment before calculating warning timing.
3. Obtain the exact monitor/calibration artifacts or explicitly declare a
   reproduction. Check whether acquisition warnings are consistently delayed
   and whether successful unfamiliar cases warn unnecessarily.
4. Choose an additional observable signal and a simple comparison only if
   the repeated error supports it. Compare earlier detection at a matched
   calibration-selected episode false-alarm budget on a separate fresh test.
   Use our simulator to isolate relevant harmful and benign changes; keep
   real evaluation separate. No improvement or sim-to-real transfer is assumed.

No further simulation, new training, threshold adjustment or recovery was
performed during this audit. Raw data reuse terms remain unverified; the
website template license does not establish dataset redistribution rights.
