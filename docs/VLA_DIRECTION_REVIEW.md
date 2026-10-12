# VLA failure monitoring: purpose, feasibility and remaining questions

Reviewed 12 October 2026. The user asks for a concrete robotics benefit and a
research basis before choosing a niche. They are open to other datasets and
simulators, and are considering a VLA-based grasp monitor. This records an
exploration, not adoption of a new controller or a claim of novelty.

**Current research priority:** grasp-specific failure monitoring alongside an
existing VLA policy. Public evidence establishes this as an active research
direction, so the next exploration will be based here. The exact contribution,
dataset and experimental question remain to be selected; no model migration
or new VLA result is implied.

## Evidence of relevance, rather than a universal standard

SAFE is followed by recent [ActProbe](https://arxiv.org/html/2606.08508v1),
[Hide-and-Seek](https://arxiv.org/html/2605.30834v1) and
[VLA-FAIL](https://arxiv.org/html/2606.21386v1) studies of lightweight runtime
monitors. These offer alternative signals and concrete limitations to compare.
Foundation-model activity also appears in Google's
[success-detection reasoning model](https://deepmind.google/blog/gemini-robotics-er-1-6/)
and Physical Intelligence's
[learned value function](https://arxiv.org/html/2511.14759v2).
These sources support technical and commercial relevance; they do not prove
one dominant industrial detector or comparative startup-formation rates.

The working area is now clear. Next inspect grasp-containing real recordings,
reproduce a suitable monitor and analyze which failures it misses. A candidate
question is whether observable grasp-state evidence can catch failed
acquisitions before transporting an unsecured object, at the same false-alarm
budget. Existing visual/phase-aware systems and gripper-feature ablations
require closer comparison before calling that question distinctive.

## What makes the comparison useful?

Amazon's study connects collected deployment data to a decision about which
picks to attempt. Predicting an outcome and inspecting important features do
not establish the physical cause of every failure. The touch study tests
whether additional measurements help; its practical value comes from the
controlled sensor comparison. Peer review, baseline controls, held-out
evaluation, released methods and limitations are evidence to examine.
We cannot infer precisely why reviewers accepted either paper or assume that
a large improvement alone establishes significance. The detailed sources
remain in [the evidence review](GRASP_PREDICTION_EVIDENCE_REVIEW.md).

## VLM, VLA and failure monitor are different components

A vision-language model interprets images and text. A vision-language-action
model additionally produces robot control actions. For example, an image and
the instruction "pick up the mug" lead to proposed movements and gripper commands.
Pretraining can supply useful visual representations, but a model still needs a
compatible robot interface and relevant evaluation. See [OpenVLA's official
implementation](https://github.com/openvla/openvla).

[SAFE](https://arxiv.org/html/2506.09937v1) attaches a small learned monitor to an
existing VLA. It trains on recorded successes and failures, reads internal
features, and applies a separately calibrated warning threshold. Its output
is a task-failure score, not a causal explanation. Timely task alarms do not
automatically establish advance warning of physical slip.

The simulator still executes actions and generates observations. The controller
may be scripted or learned. Introducing a VLA is not inherently necessary or
more efficient for generating labeled simulation trials. A VLM could assist
interpretation, but its descriptions should not silently replace verified
physics/task labels.

## Fit to this repository

Our current HUG-conditioned Shadow Hand executor follows a fixed sequence;
it is not a VLA. The original predictor observes 24 joint positions, velocities
and applied targets before lifting. It does not receive policy hidden states,
camera images or tactile readings. Recorded states can support a separately
specified visual experiment; that does not require changing the controller
first. Published OpenVLA action interfaces do not directly match the Shadow
Hand's joint targets. A VLA migration would be a new integration and study.

## A concrete route using actual robot recordings

The [official SAFE README](https://github.com/vla-safe/SAFE) links real Franka
pi0-FAST and WidowX OpenVLA rollouts. This could provide relevant deployment
evidence without owning a robot or running a VLA during offline detector training.

Public source-code inspection is pinned to
`b6036abe07b2b2bb9996afb2c07f13d6a9f507c0`:

* [Franka loader](https://github.com/vla-safe/SAFE/blob/b6036abe07b2b2bb9996afb2c07f13d6a9f507c0/failure_prob/data/pizero_fast_droid.py)
  expects environment outcomes/task metadata, saved policy representations,
  actions and a video path.
* [WidowX loader](https://github.com/vla-safe/SAFE/blob/b6036abe07b2b2bb9996afb2c07f13d6a9f507c0/failure_prob/data/openvla.py)
  expects associated action CSVs, hidden-state records, task/outcome metadata
  and videos.
* [Franka configuration](https://github.com/vla-safe/SAFE/blob/b6036abe07b2b2bb9996afb2c07f13d6a9f507c0/failure_prob/conf/dataset/pizero_fast_droid_0510.yaml)
  includes grasp-containing tasks and other manipulation tasks. Final task
  failure cannot automatically be relabeled as grasp failure.

Anonymous preview metadata names the archives `pi0fast_droid_0510_all.zip`
(approximately 27.37 GB) and `openvla_widowx.zip` (approximately 1.41 GB).
No archive was downloaded or inspected. Expected fields are not verified contents.
No explicit code/data license was found in the reviewed Git tree; archive terms
remain unverified. The published trainer hardcodes CUDA, so execution on this
Mac is not established. The next check is access/terms and a small relevant
recording inventory, then aligned features and grasp-event labels.

## Established problems, candidate experiments

These are documented pain points, not claims of untouched research gaps.

1. **Deployment changes can undermine warnings.**
   [SAFECAST](https://arxiv.org/html/2608.04246v1) already combines VLA features,
   varied rollouts and calibration, including sim-to-real experiments. It
   reports remaining sensitivity outside its perturbation families and
   collection cost. Therefore, "VLA plus varied simulated data" is prior art.
   A physical-contact variation study would need explicit conditions, budgets
   and comparisons with related methods before novelty could be assessed.
2. **Useful warning time needs a meaningful event definition.**
   [FailureSpot](https://arxiv.org/html/2609.04277v1), a September 2026 preprint,
   addresses timestamp supervision and annotation cost. A whole failed task
   does not mean every earlier moment was already failing. A grasp-specific
   event/lead-time study is a possible extension, but timestamp labeling alone
   has already been studied. Our near-lift-cutoff failures are not slip events.
   Final outcome labels are appropriate when the target is eventual task
   completion, as in our current fixed-prefix predictor. Detecting a current
   failure state and forecasting a future event are different targets; the
   timestamp critique does not automatically invalidate outcome prediction.
3. **An unfamiliar successful grasp must remain acceptable.**
   Our mug warnings illustrate this locally. A useful study would measure
   false alarms under harmless changes and missed real grasp failures under
   harmful changes. Distinguishing those conditions and choosing observable
   measurements matters more than selecting a fashionable model name.

## Workshop and significance

The [Sim2Real2Field call](https://sim2real2field-corl26.github.io/) is a
non-archival CoRL workshop, not a journal. Its four-page Late-Breaking track
welcomes preliminary findings, negative results and practical lessons.
The deadline is 15 October 2026, 11:59 PM Central. No improvement percentage
guarantees acceptance. Simulation evidence can support a bounded lesson;
real-world transfer and recovery claims require corresponding evaluation.

Our assessment framework is: a specific deployment problem, an identifiable
user of the result, a controlled comparison, and a conclusion supported by
the evaluated conditions. The workshop call supplies scope and format;
this framework is our research judgment, not a promised acceptance rubric.
A complete VLA migration is a larger undertaking than a three-day revision.

## Recommended next gate and status

Inspect the published real-rollout format and a relevant sample before adopting
a VLA controller. Reproduce a compatible monitor baseline if the data and
runtime support it. Then identify a grasp-specific limitation and decide which
additional simulation or real data would test it. Keep the current pilot as
preserved evidence while assessing that route.

No new controller, dataset import, archive extraction, model fitting,
calibration, label change or simulation was performed. The wrist comparison
remains a candidate under review; no VLA thesis or new title has been adopted.
