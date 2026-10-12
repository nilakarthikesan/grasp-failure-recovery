# Understanding HUG, VLAs and our proposed visual monitor

Prepared 12 October 2026 using existing recordings. No visual monitor training,
VLA integration or new simulation occurred. The original predictor is unchanged.

## Five components, five jobs

**A vision-language model (VLM) interprets images and text.** Given an image
and “did it pick up the cube?”, it answers. Its answer can be wrong.

**A vision-language-action model (VLA) controls a robot.** It takes an image
and an instruction such as “put the carrot on the plate” and outputs movement
commands. OpenVLA adapts a VLM to produce actions; a separate monitor can
watch for failures.
[OpenVLA, Figure 2 and Sections 3.1–3.2](https://arxiv.org/html/2406.09246v3).

**HUG proposes a grasp.** It takes RGB-D and a point on the target object,
then generates a human hand pose and wrist position/orientation. Its distinctive
approach learns from human grasp examples and retargets proposals to robot
hands. It can propose different plausible grasps for one scene. It does not
guarantee successful execution or model contact force.
[HUG architecture and failure modes](https://grasping.io/).

The generator's sampling noise produces different candidate grasp poses. Our
finger-angle noise is a separate disturbance applied after choosing a proposal.
These provide two different kinds of variation; they should not be conflated.

**MuJoCo is our simulated world.** Our fixed controller approaches, closes,
lifts and holds the object using a retargeted HUG proposal. HUG supplies the
starting grasp; MuJoCo computes what happens during execution.

**Our proposed visual monitor watches the attempt.** Images available so far
would enter a separate model that scores an outcome or event. It could watch
our HUG executor and a real VLA-controlled robot. This would be a visual
monitor evaluated on VLA rollouts, not a newly trained VLA or VLA-feature
monitor. Our existing predictor uses joint measurements and commands.

## Walk through our existing cube example

Look at the [initial RGB image](/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/cube_observation/rgb.png)
and [HUG proposal illustration](/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/cube_observation/hug_observation_overlay.png).
The illustration is for explanation, not a sensor image for training. Its pose
has not been verified as the shared seed-2 proposal used by the two recordings.

Both recordings use the same saved HUG seed-2 proposal, observation and
controller. They differ in fixed random finger-target offsets; wrists are unchanged.

1. [No added noise](/Users/nilakarthikesan/grasp-failure-prediction/runs/noisy_pilot_2026-10-09/batch/episode_000006/rollout.mp4):
   the completed trial lifted the cube **147.97 mm** and held it for two
   qualifying seconds. Its final task label is **1 = success**.
2. [Finger offsets bounded by ±0.6 radians](/Users/nilakarthikesan/grasp-failure-prediction/runs/noisy_pilot_2026-10-09/stress_batch/episode_000005/rollout.mp4):
   the completed trial reached **52.84 mm** of lift and no qualifying hold
   time. Its label is **0 = failure**. There was contact and some lift; this
   is a scorer-defined acquisition failure, not an empty grasp or a verified
   slip-onset example. The large offset is a stress condition, not a calibrated
   hardware-error distribution.

These final labels require contact, 14 cm peak lift, two seconds of qualifying
hold, at most 2 cm final drop from peak, no approach collision and no timeout.
Invalid or stopped attempts receive no binary label.

## What would the prediction model learn?

We must first choose between two different tasks:

- **Predict future task outcome:** use only images through a fixed cutoff
  before lifting, then predict the final success/failure label. Our original
  joint baseline uses CLOSE history ending at 2.96 seconds, before LIFT at
  3.00 seconds. Whether images at that point contain enough evidence is untested.
- **Detect an unsuccessful acquisition:** observe an attempted lift and judge
  whether the object came with the gripper. This needs acquisition-event
  annotations. An already empty lifted gripper supports detection, not a
  claim that we warned before acquisition failed.

A proposed training route: render clean causal images, extract features with
a fixed pretrained image encoder, then train a small classifier. A full
language model is optional. Whole episodes and related proposals stay together
in their split. Validation selects thresholds; separate real recordings test transfer.

Model inputs must exclude replay overlays, future frames, final labels, noise
settings and privileged simulator diagnostics. Stored states allow clean replay;
per-frame sensor RGB-D was not recorded. Shadow Hand and WidowX differ in
camera, robot and task. Real acquisition events need separate annotations;
we cannot assume that success at placing an object and our simulated lift/hold
label define the same target.

## What could distinguish our experiment?

A candidate comparison is **real development data alone**, versus the same
data plus **targeted simulated examples**, versus **an equally sized random
simulation addition** with comparable class balance. Targeted examples could
include genuine failed acquisitions and successful unfamiliar wrist settings.
This could test the value of choosing examples, beyond merely adding more.
No improvement, transfer or novelty has been established.

For reading next, [Guardian Sections 4.1 and 5.5](https://arxiv.org/html/2512.01946v4)
explain a VLM verifier using before/after images. Figure 7 includes a missed
plate grasp that the model calls successful. This is a useful failure example,
not proof that our method would fix it. Guardian already studies mixed real
and simulated training, so that combination alone is not our contribution.
[SAFE](https://arxiv.org/html/2506.09937v1) is an optional contrast: its monitor
reads a VLA's internal features. Our fixed HUG executor does not produce those.

The immediate decision is the monitor's timing and target. Understanding that
choice comes before fitting a visual model or collecting more data.
