# New-thread handoff: grasp failure project

Prepared 13 October 2026. Verified project state through `0bc3bf4`; this handoff
adds documentation only. Start by reading `AGENTS.md` and checking current Git
state rather than assuming these branch references have not changed.

## Working location and collaboration

- Repository: <https://github.com/nilakarthikesan/grasp-failure-recovery>.
- Continue on **`codex/project-work`**, using its existing checkout:
  `/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction`.
- `/Users/nilakarthikesan/grasp-failure-prediction` is the original checkout,
  on local main at `c95dd8d`; it also holds most raw recordings and assets.
  Do not reset it or replace its runtime to synchronize source.
- Enoch's PRs 21–24 are merged into remote main at `b503c2f` and were integrated
  into our working branch by `7c49494`. Our later diagnostics, cleaned docs and
  latest evaluation note are pushed on the working branch, not all on main.
- Preserve the untracked `MUJOCO_LOG.TXT` and historical
  `docs/GRASP_FAILURE_NEXT_STEPS.md`. Preserve ignored run directories.
- The existing paper is `docs/paper/working_draft.tex` in the working checkout.
  Edit it in place when requested; use the built-in LaTeX compiler afterward.

Nila wants short, simple, step-by-step explanations and an understanding of what
is being tested. Keep the README professional and concise. Earlier surveys,
brainstorming and teaching material are in `docs/archive/`; avoid bringing them
back into current summaries. Distinguish measured results from plans. Do not
create another branch or milestone PR. Do not message Enoch or other people
without explicit authorization.

## Research scope and unresolved agreement

The implemented system is **HUG proposals → Dex Retargeting → fixed Shadow Hand
execution in MuJoCo → a separate outcome predictor**. HUG means Human Universal
Grasping, not Hugging Face. We have not integrated a VLA or VLM.

The latest recorded question is:

> At a fixed number of simulated training attempts, does broader coverage of
> HUG-generated grasp situations improve pre-lift outcome warnings on unfamiliar
> objects and scenes more than additional finger-command perturbations of a
> smaller set of grasps?

The equal-count coverage comparison and proposed calibration/representation
fits have **not run in the saved evidence**. Nila said she and Enoch agreed on
their thesis/testing plan, but its exact final wording was not supplied. Confirm
any newer agreement before changing the research direction or asserting that
this proposed comparison is complete.

Working paper title: **Evaluating Simulation Diversity for Grasp Failure Warnings**.
Current results are simulation-only; slip warning lead time, model-driven
recovery and real-robot transfer have not been demonstrated.

## Saved predictor and completed research evidence

The frozen logistic model uses 20 closing samples, 2.20–2.96 seconds, before lift
starts at 3.00 seconds. Inputs are 24 joint positions, 24 velocities and 24
commanded targets. Last/mean/standard-deviation/change statistics give 288
features. It predicts final grasp/lift/hold completion, not physical slip onset.
Label 1 means task success, 0 completed task failure; errors/stops have null
labels. The historical failure-warning threshold is 0.5.

- **Pilot:** 1,040 attempts, 444 successes and 596 failures; cube/apple geometry,
  four observation families and sixteen HUG proposals. Splits are 520/260/260.
  Original noise is episode-constant uniform finger-target bias with bounds
  0, 0.15 and 0.6 rad, clipped to limits; wrists receive no injected bias.
- **Original test:** failure AUROC 0.8523; 187/202 failures detected,
  fifteen missed, and 22/58 successes falsely flagged. This is retrospective
  development evidence from one apple observation, not unseen-object proof.
- **Fresh evaluation:** 2,080 attempts, 659 successes and 1,421 failures.
  Original mugs collided during approach, exposing an executor problem.
- **Corrected checks:** 26 completed and one separate stop. The mug subset has
  twenty successes and three failures; the unchanged model warns on all 23.
- **Wrist diagnostic:** eight zero-noise pairs, sixteen completed attempts,
  thirteen successes and three failures; every attempt warns. Shifted wrist
  terms contribute to scores without proving the physical cause or remedy.
  One label change is a submillimeter lift-cutoff crossing, not a drop.

Keep the original v2 evaluation, corrected v3 checks and explicit v4 wrist
diagnostics separate. Related proposal families are not independent trials.

Frozen model: `research_artifacts/task_outcome_baseline_v1/proprioception_and_commands.npz`.
SHA-256: `16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb`.

## Enoch's recent implementation

- [PR 21](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/21):
  shared proposal contracts, interchangeable sources and
  `grasp-eval validate/run/status/resume/view`.
- [PR 22](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/22):
  removes the automatic Docker smoke workflow; Docker tooling remains.
- [PR 23](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/23):
  registered pose translations, provenance, deterministic geometry inputs and
  structured pre-rollout rejection.
- [PR 24](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/24):
  pinned official DexGraspNet saved-record replay. Ten mug records give seven
  approach collisions, one acquisition failure and two joint-limit rejections,
  with zero successes. This is replay, not fresh model inference.

Fresh DexGrasp Anything inference is a documented GPU-worker handoff, not an
implemented worker. Providers currently use `fixed_grasp_lift_v2`; our corrected
mug protocols are not automatically applied. Some planned scene/path checks
remain unimplemented.

**Prediction-data gap:** generic provider trajectories save joint positions
but omit per-step velocities and commanded targets, plus the collection
metadata required by the original exporter. Do not fabricate these fields or
claim these runs can already be scored by the frozen predictor.

## Latest completed evaluation

Nila requested one eval. One **saved HUG-derived cube grasp** was retargeted and
executed through the shared evaluator, using `shadow_saved_record_v1`, no extra
command noise and the unchanged v2 controller. It succeeded:

- Peak lift **0.151335 m**; stable hold **2.0 s**.
- Final loss from peak **0.000760 m**; no hand/table contact in 163 saved samples.
- Video: 163 frames at 25 fps, duration 6.52 s.
- Independently verified from the saved trajectory. Model hash unchanged;
  **model not scored**. This is a development execution check, not model accuracy.

Local run directory in the working checkout:
`runs/partner_eval_2026-10-13_hug_cube_replay/`.
It contains `plan.json`, `experiment.json`, `request.json`, `verification.json`
and a provenance-rich saved Shadow record. Result, trajectory and replay are at
`output/runs/saved_hug_cube_seed0_nominal/` within it.

An earlier fresh-inference launch stopped **before physics** because HUG runtime
dependencies were missing. It remains separate at
`runs/partner_eval_2026-10-13_hug_cube/`; it is not a failed grasp label.

Verified original HUG input:
`/Users/nilakarthikesan/grasp-failure-prediction/runs/fresh_object_scene_eval_2026-10-11/inputs/proposals/scene_000_seed_0`.
HUG checkout: `/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug`.
Checkpoint: `/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_cpu_audit/checkpoints/hug_full.safetensors`.
Checkpoint SHA-256: `515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae`.
Official DexGraspNet raw assets were not found locally; its committed gate is available.

## Runtime and verification

The base Python is
`/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python`.
It has numerical/MuJoCo packages but lacked Pydantic and Dex Retargeting.
Additional dependencies were isolated in temporary directories, preserving
the recorded runtime. These paths may disappear; verify them before reuse.

The combined suite passed **416 tests, 33 skipped**. Test versions: Python
3.12.13, NumPy 2.2.6, SciPy 1.17.1, MuJoCo 3.3.7, Pydantic 2.14.1,
Dex Retargeting 0.5.0 and Pinocchio 4.1.0. There were 105 numerical warnings in
existing matrix operations. This is integration testing, not historical replay.

From the working checkout, the complete temporary import path was:

```bash
export PYTHONPATH="src:/private/tmp/grasp-partner-integration-deps:/Users/nilakarthikesan/grasp-failure-prediction/.venv/lib/python3.12/site-packages:/private/tmp/grasp-partner-simulation-deps:/private/tmp/grasp-partner-simulation-deps/cmeel.prefix/lib/python3.12/site-packages"
```

Use `MUJOCO_GL=cgl` for offscreen rendering on this Mac. Fresh HUG inference
also lacks packages including `tyro`, `transformers`, `manotorch` and the
compiled `torch_cluster` backend. Restore its real dependencies before claiming
fresh inference; do not replace them with fake model outputs or patched stubs.

## Recommended continuation

1. Read the current protocol and confirm any newer partner-agreed plan.
2. If continuing provider evaluation, inspect one DexGraspNet pose/collision
   before scaling. Its free-space grasps are not automatically suitable for the
   tabletop approach.
3. Add a versioned, compatible velocity/command recorder and export path before
   evaluating the unchanged predictor on new provider trajectories.
4. Then run a small declared check before collecting a large batch or fitting
   a remedy. Preserve original weights, grouped memberships and protocol versions.

Start with [the documentation index](README.md),
[completed evidence](PAPER_PROGRESS.md), [recorded protocol](HUG_MONITOR_RESEARCH_PLAN.md),
[provider CLI](GRASP_EVAL_CLI.md), [DexGraspNet replay](DEXGRASPNET_REPLAY.md)
and [dated decisions](RESEARCH_DECISIONS.md).

The recorded target venue is the Sim2Real2Field CoRL 2026 Late-Breaking workshop:
four main-text pages, deadline 15 October 2026 at 11:59 PM CDT. Verify the live
call before acting. Nila has OpenReview; Enoch's activation is not confirmed.
See [submission tasks](SUBMISSION_CHECKLIST.md). No submission or acceptance has
been completed or verified.


## 13 October 2026 continuation — four-source integration, generation deferred

The user redirected the immediate implementation work to four proposal sources:
HUG, DexGrasp Anything (in progress), ContactGen and UniDexGrasp. ContactGen's
MANO adapter/isolated worker and UniDexGrasp's verified Shadow adapter/isolated
IPDF+Glow worker are implemented. Their native contract and runtime requirements
are in [the integration guide](CONTACTGEN_AND_UNIDEXGRASP.md).

The user explicitly deferred evaluation-data generation. No new generator
inference, physical evaluation, acceptance case or research dataset was run for
this implementation. Unit fixtures are synthetic and temporary. Fresh GPU
runtime acceptance, physical suitability and the predictor-recording gap remain
pending. Continue on the same branch and preserve the manuscript and historical
results. Do not interpret the implementation as a completed scientific comparison.
