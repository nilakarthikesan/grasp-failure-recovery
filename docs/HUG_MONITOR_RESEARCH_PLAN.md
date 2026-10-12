# Current direction: useful pre-lift warnings for HUG execution

12 October 2026. The user chooses the existing HUG, fixed MuJoCo executor and
joint-based predictor. VLM/VLA integration and visual-monitor development are
deferred. Their notes remain historical exploration, not the active plan.

## The practical problem

A HUG user needs to catch unsuccessful grasps without rejecting successful
unfamiliar ones. HUG proposes a static pose; its authors identify execution and
retargeting failures. A proposal is not a guarantee of acquisition. See
[HUG Sections 5–7](https://arxiv.org/html/2606.17054). Our pre-lift score cannot
prevent an approach collision that has already happened or certify contact stability.

Our original apple-scene model has outcome signal: failure-ranking AUC 0.8523,
versus 0.6419 for commands alone. But it falsely warns on 22 of 58 successes.
All 20 successful corrected mug checks warn, as do all 13 successes in the
eight-pair wrist diagnostic. These are development findings, not independent
population estimates. The 1,040-attempt pilot contains **four observation
families and sixteen HUG proposals**; training uses 520 episodes from two of
those observation families. More noisy repetitions do not create independent
grasp situations. These facts motivate a coverage comparison, not proof of its remedy.

## Working question and prediction target

**At a fixed number of simulated training attempts, does broader coverage of
HUG-generated grasp situations improve pre-lift outcome warnings on unfamiliar
objects and scenes more than additional finger-command perturbations of a
smaller set of grasps?**

Keep the target: final grasp/lift/hold outcome from the closing prefix through
2.96 seconds, before lifting begins at 3.00 seconds. This is eventual task-outcome
prediction, not timestamped slip prediction. Original labels, weights,
normalization and the 0.5 warning rule stay unchanged as historical evidence.
New comparisons would write separate models.

## Proposed comparison

**Repeated-perturbation collection:** more finger-target perturbations around
a relatively small observation/proposal set.

**Broader-coverage collection:** allocate equally many selected training
episodes across more HUG observations and proposals. Include successful
unfamiliar configurations and failures. Safe wrist coverage can be part of
the strategy and a declared subgroup. More HUG seeds do not automatically
broaden joint/wrist coverage; measure the resulting distribution.

Use the same allowed training-object pool, task definitions, corrected executor
and model family. Match selected counts and declared class composition; retain
all generated, selected and excluded trials. Record clipping and actual applied
disturbances. Fit scaling only on training data. HUG inference and execution
costs may differ despite equal episode counts.

Changing several coverage axes tests a **collection strategy**, not isolated
wrist causality. Exact source identities, quotas and sampling rules must be
frozen before collection/fitting. No such new comparison has run.

## First check simpler model explanations

Before larger collection, declare a small comparison using existing development
data: original features with validation-only calibration/threshold choices,
a wrist-removed reference, and a target-relative representation using measured
joints, velocities and command-tracking errors. Freeze feature formulas,
memberships and selection rules before fitting. Already inspected families
remain retrospective development evidence.

These change representation, not sensors. Tracking error can reflect object
contact, joint limits, dynamics or table contact. It is not a guaranteed grip
measurement; subtracting positions and targets partly reuses information
already in the original input. Do not credit calibration or feature changes
to added data. A highly accurate baseline is not a prerequisite, but a fair
baseline comparison is.

Reserve one common **new** object/scene/proposal holdout for the coverage study.
Keep complete observation/proposal families and all noise variants in one split.
Both comparative collections and fresh evaluation use one frozen corrected
executor. Historical controller versions remain explicit. Invalid/guard stops
stay outside binary metrics; report lift-cutoff boundary cases separately.

## Useful findings and workshop scope

Report false warnings on successes and missed completed failures together,
plus ranking and probability error. Select operating rules on validation/
calibration only. Include the unchanged and always-warn references. Report
the trade-off, counts, subgroups and family-level uncertainty. Few failures
cannot establish statistical noninferiority or broad safety.

A positive result could tell a HUG user to collect more distinct grasp
situations rather than repeat command noise. A negative result could identify
representation, calibration or labels as the limiting factor. Neither is assumed.

Close prior includes [See to Reach, Feel to Grasp](https://arxiv.org/html/2609.31323v1),
a September 2026 proprioceptive control/readiness preprint, and
[The Feeling of Success](https://proceedings.mlr.press/v78/calandra17a.html),
which compares before-lift vision/touch across objects. We are testing HUG
collection/monitor reliability, not claiming the first pre-lift predictor or
a new sensor/control algorithm. A bounded search did not establish global novelty.

The [Sim2Real2Field call](https://sim2real2field-corl26.github.io/) welcomes
preliminary/negative/system lessons in its four-page Late-Breaking track.
Deadline: 15 October 2026, 11:59 PM Central. A useful submission needs completed,
interpretable evidence and precise scope; topic fit does not establish acceptance.

Current evidence is simulation-only. Our joint predictor cannot run on the
inspected WidowX/Franka videos with its existing inputs. Real warning improvement
requires compatible measurements and separate real evaluation. No new rollout,
fit, calibration, label or controller change occurred in selecting this direction.
