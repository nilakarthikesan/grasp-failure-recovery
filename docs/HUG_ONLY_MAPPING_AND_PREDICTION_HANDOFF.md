# HUG only mapping and prediction handoff

Prepared 15 October 2026 for a new Codex chat. Evidence includes the DGA decision at `1370e18`, the native HUG reference at `74ae888`, and the collision-bound arithmetic repair at `a23db18` on `codex/project-work`; inspect current HEAD before continuing. This document records the user's latest scope decision and distinguishes completed results from proposed work. It does not report a new experiment.

## User objective and latest decision

Focus on **HUG as the sole grasp generator**. Continue fixing and validating the mapping to the existing Shadow hand in MuJoCo. Once the appropriate mapping gate is documented as satisfied, move into bounded evaluation, trustworthy data collection and failure-prediction model development. Do not make DGA, ContactGen, UniDexGrasp or a four-generator comparison prerequisites.

The user accepted the recommendation to prioritize one dependable execution pipeline and obtain diversity through objects, observations, poses and grasp configurations. Additional generators are deferred. The user wants autonomous progress on routine fixes and verification rather than repeated permission questions or an indefinite sequence of increasingly strong acceptance tests.

The latest request conditionally authorizes the later HUG evaluation and model phases **after their prerequisites are met**. It does not request immediate physics or training before mapping qualification. Earlier turns prohibited physics, evaluation data and training while the mapping investigation was underway; those historical restrictions and results must remain accurately described. Do not interpret the new request as permission to skip qualification, silently change the hand/controller, or launch an unbounded collection. Once the declared gates are satisfied, proceed within this HUG-only scope without asking the same permission again.

The current chat's task is to prepare this handoff. It has not started the later phases or created a new chat.

## Working directory and preservation rules

Use the user's single ongoing branch, `codex/project-work`. Run `git worktree list` first. Its current canonical checkout is:

```text
/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction
```

The original checkout below is on `main` and contains important ignored runtime assets, model files, environments and experiment outputs:

```text
/Users/nilakarthikesan/grasp-failure-prediction
```

Do not create another branch, worktree or PR for each step. Do not reset, clean, overwrite dirty work, move the existing paper, or delete ignored assets. Follow `.cursor/rules/project-workflow.mdc` and applicable AGENTS instructions. Append dated evidence-backed decisions to `docs/RESEARCH_DECISIONS.md`; keep `docs/PAPER_PROGRESS.md` and the existing paper consistent. Compile the existing paper with the desktop compiler after source edits.

Canonical paper:

```text
/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/paper/working_draft.tex
```

There are unrelated dirty ContactGen/UniDexGrasp files, untracked historical artifacts and concurrent documentation edits. Inspect Git status before changing or staging anything. Stage only owned changes; preserve other providers' work. Previously inspected open-PR listings were empty; recheck before updating a PR, and do not invent a new one. The handoff itself may be committed after the evidence baseline above, so read the current HEAD.

The primary local Python used in recent static work is `/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python`. Prior successful HUG static checks used Python 3.12.13, NumPy 2.2.6 and MuJoCo 3.3.7; individual evidence bundles pin their exact dependencies. Verify the runtime rather than assuming every shell or temporary dependency directory matches. Broad pytest selection includes physics tests: choose known static tests deliberately while the static-only gate is active.

`docs/HUG_SINGLE_EVAL_2026-10-13.md` records the existing HUG inference runtime and checkpoint bindings. HUG inference has already run on Mac CPU; do not assume a new cloud GPU is necessary to begin. Prior DGA work used Runpod, but that does not imply a currently running Pod, remaining balance or a requirement to install CUDA on the Mac. Reuse verified local assets where possible.

## Scientific claim to validate

The prediction target is failure of the **actual executed HUG-plus-retargeter-plus-controller Shadow attempt**. It is not the outcome HUG would obtain on native MANO, and it is not an official HUG benchmark score.

Keep three conclusions distinct:

| Claim | Required evidence | Current position |
| --- | --- | --- |
| Correct native decoding and realization of a prescribed Shadow command | Units, transforms, source identity, named joints, literal limits, geometry, independent FK and compiled-scene checks | Supported for the 32 saved HUG commands described below |
| Shadow preserves the native MANO surfaces, contacts or outcomes | Strong anatomical/contact and, for outcomes, execution-transfer evidence | Not established; strict historical surface tests fail |
| A predictor works on future executed Shadow attempts | Reliable recorded observations and labels, diverse independent situations, held-out prediction and calibration tests | Historical signal exists; broad reliability is unestablished |

Native and target hands have different geometry. Exact equality of their contact regions is not a prerequisite for predicting the recorded target-hand execution. This does not justify loose coordinate conversion or hiding distortion. Keep numerical correctness strict; measure native-to-target landmark and surface changes; state the pipeline being studied.

Do not invent an acceptable millimeter threshold by looking at which current examples would pass. The existing 10 mm mean fingertip gate is an engineering guard, not a calibrated physical-fidelity guarantee. A lower fitting loss or a successful lift does not prove mapping fidelity. A mapping refusal or runtime error is not a completed physical failure label.

## Authoritative HUG starting evidence

Read [HUG mapping acceptance repair](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/HUG_MAPPING_ACCEPTANCE_REPAIR.md) first. It supersedes earlier broad pending-status descriptions only within its stated scope; it does not erase their results.

Completed evidence:

- All 32 saved HUG-derived nominal Shadow commands pass the current complete static scene contract. Checks cover native camera/wrist-to-palm composition, independent XML FK/root inversion, exact named targets and source-precision limits, hand and object geometry, captured object pose/mass and declared control/table settings.
- Independent current compiled task-point positions agree within approximately `3.991e-16 m`. The complete-scene rerun has maximum palm discrepancy `1.093e-16 m` and object-inertia tensor discrepancy `6.506e-19 kg m²`. These quantify realization of the prescribed target scene, not native anatomical error.
- The initial scene bridge passed 24/32; eight cases failed the fixed inertia precision gate. A source-tensor-preserving principal-frame correction repaired the numerical issue without changing the intended inertia or tolerance. The corrected rerun passed 32/32. Both versions remain recorded.
- A visual/collision mesh basename alias was repaired. The older 6.9446 mm vertex-set discrepancy must not be described as collider-boundary displacement: a direct convex-solid comparison found a maximum 24.512339 micrometres. This importer issue does not explain the larger native/Shadow anatomical discrepancies.
- Earlier all-32 native MANO parameter reconstruction matched the saved 3D vertices, landmarks, rotations and faces exactly. All-32 camera projection was not thereby established.
- Thirty-two poses span four objects and eight observation groups, with one native MANO shape in the audited set. They are inspected development data, not a fresh holdout. Detailed surface coverage was six selected cases plus three finer repeats, not all 32 full surfaces.

The acceptance is **static realization of those saved commands**. It does not cover every fresh retargeting solve, dynamic servo tracking, unseen inputs, alternative assets or prediction utility.

Authoritative bundle: `research_artifacts/hug_mapping_acceptance_repair_2026-10-15_v1/`. Read its protocols/manifests for exact hashes and reproductions. Earlier conversion evidence is in `research_artifacts/compiled_hand_mapping_audit_2026-10-15_v1/`.

## Additional native MANO reference, not a target replacement

A parallel completed static investigation is recorded in [HUG native MANO comparison](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/HUG_NATIVE_MANO_COMPARISON.md) and `research_artifacts/hug_native_mano_comparison_2026-10-15_v1/` (commit `74ae888`). It reconstructs the same 32 saved proposals in an isolated articulated MANO reference with capsule phalanges and a convex palm. Skeleton import passes; mean fingertip discrepancy is 0.453 mm versus 6.428 mm for the literal saved Shadow commands, and the equally averaged 192 part/case surface-distance means are 1.463 versus 4.727 mm. These finite, one-directional measurements support an anatomical explanation for some distortion, not a best-possible Shadow bound or continuous contact certificate. Near-object changes remain in the native approximation too.

This reference has disabled gravity/contact masks, placeholder inertias and no actuators/controller or evaluation registration. It is **not a validated physical hand**, not an official reproduction of HUG's unpublished simulator, and not the target for this HUG-only Shadow continuation. Use it as diagnostic evidence; do not silently switch hands or start its separate proposed collision-development path. The saved aggregate was recovered after a JSON integer serialization failure without rerunning geometry; the evidence explains that recovery. Commit `a23db18` separately repairs numerical warnings in object collision-bound arithmetic; it is not a new mapping or physics acceptance.

## Retargeter profiles and remaining mapping issues

The live API is `ShadowHandRetargeter` in `src/grasp_failure_prediction/evaluation/retargeting.py`.

- `legacy_shadow_vector_v1` remains the historical/default profile. Saved-command acceptance must not be extended to arbitrary fresh legacy solves.
- `independent_static_shadow_vector_v2` addresses per-case initialization/reproducibility and objective/gradient/status issues. It is a separate reliability profile and changes geometry. It has not been promoted as a generally validated replacement.
- The feasible-step and eligibility-guided surface corrections are experimental per-case methods. Their improved fitting scores do not constitute a new general mapper or a native-fidelity pass.

The shared translation path currently calls `ShadowHandRetargeter()` without selecting v2, so HUG's provider still uses legacy. The canonical provider output retains mean/max fingertip error but does not yet propagate all `RetargetingProvenance` fields. Future accepted fresh outputs should record profile/settings, solver statuses, initialization/reset policy and raw versus physically clamped commands. The solver's optimization margin and later clipping must not be presented as unchanged native output.

The reliability audit observed a 0.192417 rad change in repeated A under reused legacy A–B–A solver order; the stateless profile repeated its inspected inputs bit-exactly. This motivates a fresh-solve reproducibility gate, not automatic geometric promotion of v2. A previous canonical palm calibration also failed; the source URDF's approximately 34 mm wrist-to-palm offset was already composed. Do not assume another missing offset explains the surface differences.

The original six coarse and three finer strict surface audits remain nine FAILs. The old checker also lacked a general positive-certification path: uncertain threshold-crossing cells and unresolved continuous orientation meant “optimize until this returns PASS” was an invalid stopping rule. Its definite counterexamples remain valid. A new identity certifier passes 25 manufactured controls, but its supported geometry is narrow and does not make MANO and Shadow identical.

On two difficult development cases, protected fitting improved the retained objective while preserving the specified fingertip guards, but surface failures remained. Final mean fingertip errors were about 7.139 mm for milk-box B and 5.322 mm for mug B. The mug's tightest guard had only approximately 8.6 nm slack; this is not a robust physical margin. Do not promote these candidates on objective improvement alone.

Immediate remaining mapping work should be bounded:

1. Freeze the exact profile, asset closure, initialization, solver settings and intended static claim before inspecting new validation outcomes.
2. Separate replaying the accepted 32 saved commands from running fresh solves. Verify repeated-run and input-order behavior for the selected fresh profile.
3. Check input decoding, camera/world/object/wrist/palm transforms, literal limits, units, named joints, geometry and independent compiled FK. Include nonidentity poses and manufactured failure controls.
4. Measure per-finger maxima/tails, surface proximity, introduced/lost near regions and penetration on predeclared new inputs. Preserve poor native proposals and all refusals. Do not convert lack of exact anatomical equality into an automatic blocker for the executed-Shadow research claim.
5. Record a scoped acceptance or a concrete remaining implementation defect. Do not repeatedly invent stronger tests after observing a pass, or tune on a nominal holdout and continue calling it unseen.

For source interpretation: HUG supplies right-MANO camera-frame predictions. `T_camera_wrist` is wrist-to-camera; its inverse removes the global wrist transform before local retargeting. The MANO-to-operator basis change and explicit wrist/palm placement convention must be retained and checked. The target hand has 24 named commands. Do not confuse a correct frame convention with equal anatomy.

## Gate from mapping into evaluation

After the selected HUG pipeline has documented static qualification, use a small predefined execution-and-recording pilot before large collection. The user's latest instruction makes this a conditional next phase; it has not run as part of this handoff.

Check:

- The actual imported scene and requested commands are the ones statically qualified, with source/checkpoint/preprocessing/mapper/controller/object/hand hashes saved.
- Placement, table clearance, joint limits, approach/close/lift/hold phases and control timing are explicit. Verify dynamic tracking rather than assuming a nominal pose is reached.
- Native pose, nominal mapped pose, requested controller command and measured achieved state remain separate fields. The existing controller adds closing motion, so these are not interchangeable.
- Repeated deterministic cases reproduce within declared tolerances; unexpected numerical/runtime failures are separate statuses. Review representative completed successes and failures, not only videos of successful lifts.
- Outcome labels can be independently recomputed from recordings. Preserve null labels for invalid/guarded/runtime failures and disclose the attempted-input denominator.

Do not blindly call scene capture during a static-only check: the existing capture workflow can settle objects using physics. Inspect the helper and phase authorization first. Reuse saved validated observations or a deliberately static preparation method where appropriate.

## Recorder compatibility must be closed before collection

The generic proposal evaluator currently writes position/contact summaries and final state, but its trajectory does not provide all time-aligned joint velocity and applied command histories needed by the existing predictor exporter. A successful `grasp-eval` run is therefore not automatically a prediction-ready recording.

Inspect these paths before scaling:

- `src/grasp_failure_prediction/proposals/execution.py` — `_write_artifacts` and generic proposal execution.
- `src/grasp_failure_prediction/evaluation/runner.py` — `ActuatedShadowRunner`, state/control progression and trace contents.
- `src/grasp_failure_prediction/evaluation/noisy_rollouts.py` — historical prediction-compatible collection.
- `src/grasp_failure_prediction/prediction/dataset.py` — actual exporter/input contract.
- `src/grasp_failure_prediction/evaluation/recording.py` — video recorder; video alone does not provide the needed numerical histories.

A new recorder bridge should preserve historical schemas and write a new version when necessary. Record timestamps/phase, joint names, measured positions and velocities, actually applied targets, command changes and relevant provenance. Keep full simulator state/contact forces for replay and diagnostics separate from the allowed predictor feature set. Test array alignment, missing fields, causal cutoffs, actual versus requested perturbations, and label isolation before collecting a large batch.

The historical exporter expects twenty closing samples at 0.04 s spacing, with 24 position, 24 velocity and 24 applied-command channels. A final velocity snapshot is not a velocity history. `evaluation/table_safe_execution.py` and `evaluation/paired_wrist_execution.py` also contain relevant historical recording/contract checks.

**The existing `prediction/dataset.py` hardcodes historical scene IDs 9100–9103 and split roles.** New data need a new versioned export/split protocol. Do not assign new scenes those old IDs, silently append to the old frozen dataset, or assume the old exporter handles arbitrary new collection correctly.

## Existing prediction work and what it means

The project is not starting model development from zero. Read [paper progress](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/PAPER_PROGRESS.md), [prediction baseline](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/PREDICTION_BASELINE.md), [data splits](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/DATA_SPLITS.md), and [HUG monitor plan](/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/HUG_MONITOR_RESEARCH_PLAN.md).

Historical results, to preserve separately from corrected future runs:

- 1,040 attempts: 444 successes, 596 task failures; two object geometries, four observation families and sixteen HUG proposals. Many perturbations are related trials, not independent grasp situations.
- Observation-grouped training/validation/test counts: 520/260/260. The split was declared after outcomes had been inspected; both geometries occur in training. This is retrospective development evidence, not an unseen-object benchmark.
- The original model uses closing histories through 2.96 s, before lift begins at 3.00 s. It predicts eventual grasp/lift/hold outcome, not a timestamped slip event or recovery action.
- Features summarize measured joint positions, velocities and applied targets. The L2 logistic baseline achieved failure-ranking AUROC 0.8523 versus 0.6419 for commands alone on the reserved apple observation. At the original 0.5 warning threshold, it caught 187/202 failures and falsely warned on 22/58 successes.
- A later frozen-model test had AUROC 0.9173 on fresh cube/apple scenes and 0.5588 on new milk/mug attempts. Original mug executions had table collisions. Separately corrected mug executions included twenty successes, all warned as failures. These diagnostics show a reliability/generalization problem; they do not establish its remedy.
- Wrist-range attribution and paired tests are diagnostic, not proof of isolated wrist causality. Historical controller versions and cutoff-sensitive examples remain separate.

The historical task rule includes contact, at least 14 cm maximum lift, a two-second hold, no more than 2 cm final drop from peak, and no approach collision/timeout. The authoritative code/configuration governs the exact label. A task-cutoff crossing is not automatically a slip. Freeze any revised task or prediction horizon prospectively and never relabel historical results silently.

Relevant existing portable evidence: `research_artifacts/object_scene_pilot_v1/`, `research_artifacts/task_outcome_baseline_v1/`, `research_artifacts/fresh_object_scene_eval_v1/`, and the corrected-mug/wrist diagnostic bundles linked from `docs/PAPER_PROGRESS.md`. Original larger runs live in the original checkout's ignored `runs/` tree; verify paths from manifests.

## Diversity and the next prediction study

Four generators are not required. Make variety in independent HUG situations the priority: more object shapes/sizes, observations, orientations and distinct grasp configurations. Extra inference seeds count as useful diversity only if they measurably change the coverage. Bounded perturbations can supplement that coverage; repeated noise around a small set must not be presented as broad diversity. Mass/friction/controller variation would be an explicit later protocol change, not a hidden augmentation.

Before collection, freeze object/observation/proposal identities, quotas, sampling, perturbations, task rules and splits. Retain all attempted inputs, refused mappings and executed outcomes. Avoid selecting only easy successes or manufacturing only obvious extreme failures. Do not use outcome-dependent repair to improve grasps and then attribute the results to unchanged HUG.

Keep all repetitions/noise variants from a proposal family in one split; use genuine new object/family holdouts for the claimed generalization test. Fit normalization on training only, choose thresholds/calibration on validation only, and keep test data out of debugging. Existing inspected families are development data even if they are repartitioned.

Start with the existing simple baselines before adding model complexity: prevalence/constant, commands-only and state-plus-command models. The existing research plan also proposes wrist-removed and target-relative representations as separately declared comparisons. Sensor-available observations and privileged simulation state must be distinguished. Future observations, outcome labels and outcome metadata cannot enter a pre-lift predictor.

Report false warnings on successful attempts and missed failures together, plus ranking, probability error/calibration and group-level uncertainty. A large episode count does not repair a tiny number of independent objects/proposals. Simulation-only success does not establish real-robot transfer. Learning-curve and held-out results, not a guessed universal dataset size, should guide expansion.

## Deferred generator context

The earlier project attempted HUG, DGA, ContactGen and UniDexGrasp on one Shadow target. Different native hand representations made exact cross-hand preservation a much stronger task than correct simulator input conversion. The user explicitly chose to defer this multi-generator dependency.

DGA's original exact-preservation audit found a conditional mathematical incompatibility for two required poses under the original palm and landmark constraints. A separately authorized practical protocol then accounted for twenty saved MultiDex outputs: thirteen native-valid static mappings and seven native-invalid refusals. One thumb correction passed its independent worst-distance/landmark gates; 22 proposed branches fell back to the legal reference. Maximum landmark error remained 6.0756 mm and native/target palm mismatch remained. That static cube-only result is preserved, not generalized or converted into HUG criteria.

DGA work was committed and pushed as `1370e18`. See `docs/DGA_ACCEPTANCE_FEASIBILITY_AUDIT.md`, `docs/DGA_PRACTICAL_RETARGETING_PROTOCOL.md`, and `research_artifacts/dga_practical_retargeting_2026-10-15_v1/README.md`. Its final evidence manifest SHA-256 is `e5dabac92e47d63fca0721b1ec063be5322b940b45c7f60d4984fc9597aae120`. A failed review JSON serialization is explicitly preserved as a partial historical artifact; valid additive receipts and the final integrity audit support the result.

Do not reopen DGA optimization, acquire other generators, pool their labels, copy their tolerances to HUG, or modify their dirty files as part of the new HUG-only chat. Their code/results remain useful history. The new focus is a prospective project decision, not a claim that earlier research never happened.

## First actions for the new chat

1. Read this handoff, applicable AGENTS/project workflow, and current Git status/worktree/HEAD. Read `HUG_MAPPING_ACCEPTANCE_REPAIR.md` before older mapping reports.
2. Produce a short current gap list separating accepted saved-command replay, fresh-solve qualification, dynamic tracking, recorder compatibility and predictor evaluation. Identify the exact profile and file hashes involved.
3. Freeze a bounded HUG-only static validation protocol with justified numerical gates and explicit anatomical diagnostics. Preserve the target and existing results. Fix demonstrated conversion/reproducibility defects; do not require native anatomical identity as the stopping rule for the executed-Shadow claim.
4. Qualify the chosen mapping on predeclared inputs with independent checks. State scope and all refusals. Use a clear go/no-go milestone rather than continuing optimization indefinitely.
5. Once the scoped mapping gate is satisfied, complete the recorder bridge and a bounded execution/label audit before large collection. Keep new runs/protocols versioned.
6. Freeze the HUG-only diversity study and independent split, collect appropriate data, then compare prediction baselines. Expand based on measured coverage and learning behavior.

No defensible deadline exists for eliminating every anatomical difference. The useful next milestone is one qualified HUG execution-and-recording pipeline, followed by a validated small pilot. Do not promise that all mappings will become exact or that more data automatically improves prediction.

## Code and test map

Paths in this table are relative to the canonical `codex/project-work` checkout above.

| Responsibility | Current files |
| --- | --- |
| Native HUG decoding and frame convention | `src/grasp_failure_prediction/integrations/hug.py`, `hug_frames.py` |
| Input preprocessing and random state | `src/grasp_failure_prediction/integrations/hug_preprocessing.py`, `hug_rng.py` |
| HUG inference and sampling | `scripts/infer_sim_observation.py`, `scripts/sample_hug_sim_proposals.py` |
| Native MANO reconstruction | `src/grasp_failure_prediction/evaluation/native_mano.py`, `scripts/audit_hug_native_mano.py` |
| Mapper and provenance | `src/grasp_failure_prediction/evaluation/retargeting.py` |
| Target import and static pose checks | `src/grasp_failure_prediction/evaluation/pose_validation.py` |
| Object geometry and inertia import | `src/grasp_failure_prediction/evaluation/object_assets.py` |
| Native world-palm composition | `src/grasp_failure_prediction/evaluation/case_runner.py`, especially `_hug_world_palm_pose` |
| Provider and profile selection | `src/grasp_failure_prediction/proposals/providers.py`, `translation.py`, `configs/providers.yaml` |
| Controller and execution | `src/grasp_failure_prediction/evaluation/runner.py`, `configs/execution_protocols.yaml` |
| Generic execution and output | `src/grasp_failure_prediction/proposals/execution.py`, `docs/GRASP_EVAL_CLI.md` |
| Prediction data/model/evaluation | `src/grasp_failure_prediction/prediction/dataset.py`, `model.py`, `fresh_evaluation.py` |

Candidate focused static tests, after inspecting the exact current definitions: `tests/test_hug_adapter.py`, `test_hug_frames.py`, `test_hug_preprocessing.py`, `test_hug_rng.py`, `test_retargeting.py`, `test_compiled_shadow_conversion.py`, and `test_object_geometry_schema.py`. Relevant native-MANO/pose tests can be selected from the current inventory. The compiled-conversion tests explicitly block physics APIs. Do not run `test_shadow_actuation.py`, a generic runner suite or the entire repository test suite under a static-only assumption; they include real closure/pickup/dynamics paths.

For historical source details, read `docs/HUG_MAPPING_PROCESS_REPAIR.md`, `docs/HUG_CANONICAL_PALM_CALIBRATION.md`, `docs/HUG_STATIC_VALIDATION_PROTOCOL.md`, `docs/HUG_MAPPING_ACCEPTANCE.md`, `docs/MULTI_SCENE_COLLECTION.md`, `docs/NOISY_ROLLOUT_COLLECTION.md`, `docs/MUG_TABLE_CLEARANCE.md` and the evidence links in the authoritative current acceptance report. Earlier planning headers can be stale; prefer the dated completed result and its immutable manifest, and keep each protocol's meaning separate.

## Paste into the new chat

```text
Read /Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/HUG_ONLY_MAPPING_AND_PREDICTION_HANDOFF.md and follow its evidence links.

We are focusing only on HUG as our grasp generator, using the existing Shadow hand and MuJoCo pipeline. Continue on codex/project-work in its existing worktree, preserving dirty files, assets, historical results and the open paper path.

First close the remaining HUG mapping/reproducibility and static validation gaps. Validate correct realization of the declared retargeted Shadow grasp; do not make exact MANO-to-Shadow surface equality a prerequisite for predicting actual Shadow outcomes. Keep distortion visible and do not weaken numerical conversion checks or silently promote experimental mapper profiles.

Once the scoped mapping gate is documented as satisfied, proceed to a small execution/recorder/label validation pilot, then HUG-only diverse data collection and prediction-model development under frozen protocols. Preserve all refusals and separate protocol versions. Routine authorized fixes do not need repeated confirmation. Do not resume other-generator work or launch large collection/training before the prerequisites pass.

Start by reading the current authoritative HUG acceptance report and inspecting code/state. Give me a concise gap list and begin the first bounded mapping-validation step.
```
