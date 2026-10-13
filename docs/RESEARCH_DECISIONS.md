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

## Keeping this record current

For each material change, record the date, what we thought previously, the new
observation or source, the choice made, its implementation status, the supporting
artifact and the next check. Mark unsuccessful experiments and rejected inputs
explicitly. Distinguish measured results from interpretations and proposals.
Link the paper to the evidence used for each claim. This convention is the
ongoing documentation requirement; this turn did not launch new experiments.

## 12 October 2026 Adding an interchangeable grasp-proposal boundary

The active evaluator previously coupled fresh inference, HUG pickle loading,
MANO conversion, Dex Retargeting and MuJoCo execution in one HUG-specific
command. The user wants to compare grasp generators while keeping the registered
Shadow Hand environment, fixed controller and outcome rule unchanged.

The implementation adds a strict `grasp_proposal_v1` JSON boundary. A proposal
binds a provider/configuration, object geometry, generation seed, exact named
Shadow joint targets and `T_object_palm`, where the `palm` link is expressed in
the registered MuJoCo object body frame. It also records separate geometry,
observation and generation-request groups. Portable execution validates the
canonical artifact and its content hash; provider-native outputs and checkpoints
remain optional deep-audit inputs so a cloud-generated proposal can run without
the generation environment.

Generation and execution are separate. Built-in HUG and project-authored saved
record adapters, an external-command contract and an HTTP JSON contract all
produce the same artifact. New `eval_case_v2` files execute through a generic
one-case or batch command. Historical `eval_case_v1` HUG commands remain
available and are not rewritten. Provider identity is provenance and analysis
metadata and is not an input to the existing failure predictor.

This is an infrastructure result, not evidence that any grasp model is better.
A deterministic saved proposal completed the common MuJoCo command and was
scored as `approach_collision`; that failure only verifies end-to-end dispatch.
A fresh simulator observation was then processed by the pinned HUG checkpoint
on CPU and executed successfully through the generic runner. Replaying the same
native HUG proposal through the legacy runner produced the same successful
outcome and matching phase/contact arrays. Maximum numeric trajectory deviation
was 5.1e-13 m for palm position, 8.2e-12 rad for hand position and 3.5e-9 N for
contact force, consistent with transform serialization roundoff. These ignored
runtime smoke artifacts are not research data and are not committed.

The next scientific check is paired HUG and non-HUG proposals on identical
objects and initial conditions, followed by grouped, held-out outcome analysis.

## 12 October 2026 — Use one versioned command for proposal-to-evaluation experiments

- **Previous plan:** Users invoked proposal generation, case materialization,
  batch execution, and the result viewer as separate commands. That exposed
  internal paths and made it easy to mix proposals, hashes, and output roots.
- **New evidence:** The interchangeable-provider implementation already has
  strict request, proposal, case, and result boundaries, so orchestration can
  be added without changing HUG, the MuJoCo environment, or the scorer.
- **Choice:** Add `grasp-eval` with `validate`, `run`, `status`, `resume`, and
  `view` subcommands over `grasp_eval_experiment_v1`. One experiment file maps
  provider requests and proposal sample indexes to controlled physical cases.
- **Implementation status:** Implemented on `codex/project-work`. Completed
  cases are immutable during resume; incomplete stage directories are retained
  before retry. Existing low-level commands remain available for debugging.
- **Next check:** Run a small multi-provider experiment and confirm that resume,
  videos, and the viewer behave correctly before adding dataset-wide
  Parquet/DuckDB export.
