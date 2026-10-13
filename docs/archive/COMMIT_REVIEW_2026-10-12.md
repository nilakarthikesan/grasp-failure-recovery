# Commit documentation review, 12 October 2026

This audit reviews all **91 commits reachable through `6790b2d`**, including
the **24 commits after the published `main` snapshot `c95dd8d`**. The review
inspected commit file lists and documentation changes, with particular attention
to the post-main research discussion and experimental records.

The dispositions below concern **documentation presentation**, not removing
commits. No commit is reverted, no history is rewritten, and the complete source
lineage remains available. Implementation, tests, models, datasets, protocol
versions and measured evidence are retained. This audit does not establish a new
research direction or report a new experiment.

Roles:

- **Implementation/evidence:** retain executable functionality and reproducible
  experimental records; keep relevant technical instructions and measured claims.
- **Exploration:** archive provisional questions, teaching material and deferred
  directions; retain selected relevant reasoning and citations for the paper.
- **History:** preserve integration, milestone and workflow details as historical
  records rather than public project introduction.
- **Current guidance:** retain and reconcile concise operational guidance, the
  active research plan and the working manuscript with actual evidence.

## Foundation and connected execution

- `e55b7c3` — **Implementation/evidence.** Retain the initial package, HUG adapter
  and environment foundation; preserve the original broad thesis as design history.
- `6bdfbc5` — **Implementation/evidence.** Retain the supporting Panda collection
  and training pipeline; distinguish it from the current Shadow Hand experiment.
- `1803257` — **Exploration.** Archive the proposed Part I design and review
  questions; retain the supporting evaluation module.
- `0dbb5f4` — **Implementation/evidence.** Retain evaluation architecture,
  environment changes and manifest contracts; archive prospective broad scope.
- `d68cc4d` — **Implementation/evidence.** Retain strict evaluation schemas.
- `5a37681` — **Implementation/evidence.** Retain versioned environment and
  execution registries.
- `96036df` — **Implementation/evidence.** Retain HUG-to-Dex retargeting integration.
- `c315e18` — **Implementation/evidence.** Retain Shadow pose validation in MuJoCo.
- `c22df2b` — **Implementation/evidence.** Retain the deterministic grasp/lift runner.
- `07d4b81` — **Implementation/evidence.** Retain outcome scoring and result
  artifact persistence.
- `335b4f1` — **Implementation/evidence.** Retain the one-case evaluation command.
- `94ff0ec` — **Exploration.** Archive the earlier fixed-HUG execution proposal
  and its proposed status; retain the supporting Panda evaluator revision.
- `cf1bee9` — **Implementation/evidence.** Retain palm-centering and registry fixes.
- `0567050` — **Implementation/evidence.** Retain observation validation and seeded
  HUG preprocessing checks.
- `ec391c2` — **Implementation/evidence.** Retain simulated RGB-D capture, HUG
  inference and alignment tooling.
- `b9a4e05` — **Implementation/evidence.** Retain actuated pickup/contact
  diagnostics, positive and negative controls, and measured debugging evidence.
- `737c4ca` — **Implementation/evidence.** Retain saved-proposal execution with
  bounded closing pressure and reproducible CPU sampling.

## Integration, runtime and viewer

- `109fe2b` — **History.** Preserve the PR 3 merge milestone; retain all underlying
  evaluation implementation.
- `ef619c4` — **Implementation/evidence.** Retain patch-equivalent observation and
  preprocessing changes in the consolidated lineage.
- `99fb32c` — **History.** Preserve the PR 4 merge milestone.
- `a862512` — **Implementation/evidence.** Retain patch-equivalent RGB-D inference
  and alignment changes in the consolidated lineage.
- `dda5595` — **History.** Preserve the PR 5 merge milestone.
- `2c6d28c` — **Implementation/evidence.** Retain patch-equivalent actuated Shadow
  pickup diagnostics in the consolidated lineage.
- `96ef339` — **History.** Preserve the PR 6 merge milestone.
- `c473f0d` — **Implementation/evidence.** Retain patch-equivalent fixed closing
  and seeded proposal execution in the consolidated lineage.
- `6903400` — **History.** Preserve the PR 7 merge milestone.
- `dcb6420` — **Implementation/evidence.** Retain the OpenCV inference dependency.
- `cd0bb78` — **Implementation/evidence.** Retain optional Docker/CI runtime and
  its technical instructions.
- `34e9d70` — **Implementation/evidence.** Retain the local checkpoint ignore rule.
- `50b2357` — **Implementation/evidence.** Retain per-case HUG inference and
  provenance recording.
- `4f972f0` — **Implementation/evidence.** Retain offline HUG/Shadow retargeting fixes.
- `d5af08e` — **Implementation/evidence.** Retain actuated Shadow evaluation.
- `2a5c6b4` — **Implementation/evidence.** Retain rollout recording and rendering.
- `843fcfe` — **Implementation/evidence.** Retain the local evaluation results viewer.
- `71aa539` — **Current guidance.** Retain useful operational documentation;
  reconcile stale research scope with the current evidence and plan.

## Collection, baseline and unfamiliar-object evaluation

- `2bd06ef` — **Exploration.** Archive the long Step 1 educational walkthrough;
  keep it available as optional background rather than README narrative.
- `dafa350` — **Implementation/evidence.** Preserve recorded noise strengths and
  outcomes in the walkthrough and demo provenance.
- `372cb02` — **Exploration.** Archive noise-choice teaching explanations while
  retaining exact applied-noise definitions in technical collection guidance.
- `0486648` — **Implementation/evidence.** Preserve the offline noise demo and
  source manifest; retain a short README link rather than the entire tutorial.
- `ac26940` — **Exploration.** Archive the initial literature review and six-step
  refinements; use directly relevant findings in the paper's related work.
- `e5f12f0` — **Implementation/evidence.** Retain the reproducible noisy collector,
  event annotations and collection contract.
- `21d0fd4` — **Implementation/evidence.** Retain loaded-versus-installed
  Pinocchio version provenance.
- `e31b6e9` — **Implementation/evidence.** Retain paired-pilot verification and
  documented collection limitations.
- `9f6ac3f` — **Implementation/evidence.** Retain scene capture and collection
  across declared object scenes.
- `097dab9` — **Implementation/evidence.** Retain proposal-seed and exact-input
  scene binding, with reproducible collection instructions.
- `301ca1f` — **Implementation/evidence.** Retain bounded pose-window settling checks.
- `99f70a6` — **Implementation/evidence.** Retain consistent object rendering.
- `b8f8e33` — **Implementation/evidence.** Retain the verified 1,040-trial collection
  and its task-label limitations.
- `96396ce` — **Current guidance.** Retain the manuscript and evidence index;
  replace repeated milestone narrative with concise current progress.
- `6c85271` — **Implementation/evidence.** Retain portable closing-prefix data and
  observation-grouped development splits, including retrospective split limits.
- `0a262eb` — **Implementation/evidence.** Retain logistic baseline training.
- `11dccea` — **Implementation/evidence.** Retain trained models, predictions,
  metrics and analysis; preserve the manuscript's measured claims and limits.
- `ef0f4d8` — **History.** Preserve historical dataset/baseline PR references
  outside the project introduction.
- `3e8fc8f` — **Current guidance.** Retain a concise partner handoff; remove
  duplicated or obsolete prospective roadmap material.
- `b21b53b` — **Implementation/evidence.** Retain frozen fresh-object/scene
  evaluation and its reproducibility records.
- `af06d76` — **Implementation/evidence.** Retain mug forearm-clearance correction
  and unsafe-closure guards as a distinct executor version.
- `fc56464` — **Exploration.** Preserve dated research-question evolution in
  decision history; do not present exploratory conclusions as current results.

## Source consolidation and branch history

- `3066e4a` — **History.** Preserve the main-documentation consolidation merge.
- `5dc3c3a` — **History.** Preserve the recorded-demo/walkthrough consolidation merge.
- `89c0920` — **History.** Preserve the original-literature consolidation merge.
- `07860c2` — **History.** Preserve historical proposal and Panda evaluator
  consolidation; retain the evaluator implementation.
- `b4d1489` — **Implementation/evidence and history.** Retain runtime integration
  contract resolutions; keep the consolidation narrative in historical records.
- `844916c` — **History.** Preserve patch-equivalent checkpoint ancestry without
  treating it as a new implementation or scientific result.
- `5dd5595` — **History.** Preserve branch cleanup and recovery evidence; retain
  the one-working-branch instruction in contributor guidance.
- `c96bd17` — **History.** Preserve completed cleanup counts and historical PR links.
- `4d931fe` — **Implementation/evidence.** Retain the public saved-grasp container
  smoke-test correction.
- `c95dd8d` — **History.** Preserve the PR 19 main integration snapshot; describe
  branch status briefly rather than making integration history the README focus.

## Post-main diagnostics and research discussion

These are all 24 commits following `c95dd8d` through the reviewed tip.

- `f7b4373` — **Implementation/evidence.** Retain the successful corrected
  zero-noise mug and its frozen-model false alarm, with replay evidence.
- `5f3882a` — **Implementation/evidence.** Retain the Docker OpenMP dependency fix.
- `556fbe8` — **Exploration and evidence.** Archive provisional diversity
  direction discussion; retain verified Linux runtime status in the container guide.
- `9edf3fd` — **Exploration.** Archive external-context reconciliation; preserve
  relevant provenance and factual measurement clarifications.
- `3ab05ec` — **Implementation/evidence.** Retain frozen scores for 26 completed
  corrected checks, keeping the stopped attempt separate and unscored.
- `cf2cd6b` — **Implementation/evidence.** Retain wrist-input distributions,
  model contributions and the limitation that attribution is not causal proof.
- `436c78a` — **Exploration.** Archive workshop selection and reading-tour prose;
  keep current submission requirements in focused submission guidance.
- `6e2516a` — **Exploration.** Archive prediction/recovery novelty discussion;
  retain selected relevant prior-work citations in the manuscript.
- `6e0ffe3` — **Exploration.** Archive candidate-direction brainstorming; it
  reports no implemented training remedy.
- `21ada83` — **Evidence and exploration.** Preserve the frozen wrist-diagnostic
  protocol; keep the earlier wrist-training question in dated decision history.
- `7b4f835` — **Exploration.** Preserve earlier wrist-cohort matching assumptions
  as history, distinct from the later broader-coverage proposal.
- `9cbf164` — **Evidence and exploration.** Retain static wrist feasibility code
  and results; archive superseded wrist-only thesis and immediate-next-step prose.
- `affde66` — **Implementation/evidence.** Retain the first frozen safe wrist
  pair, opt-in executor and prediction, replay and independent audit records.
- `014c782` — **Implementation/evidence.** Retain seven remaining paired
  diagnostics, predictions, independent audits and the lift-cutoff boundary result.
- `ce5e8a5` — **Exploration.** Archive training-coverage versus outside-data
  discussion; preserve reasoning that directly informs the selected comparison.
- `22acf88` — **Exploration.** Preserve broad-goal versus focused-wrist-study
  reasoning as historical research discussion.
- `269f0d1` — **Exploration.** Archive the broad industry/academic survey; curate
  directly relevant comparisons for the current scientific argument.
- `7272e33` — **History.** Preserve formatting-only survey changes in Git history.
- `6ec1d6f` — **Exploration.** Archive VLA-monitor feasibility discussion; the
  proposed route is deferred.
- `9ee5b24` — **Exploration.** Archive the temporary VLA-monitor priority and
  additional readings, explicitly superseded by the return to HUG.
- `917d793` — **Exploration and evidence.** Archive the SAFE walkthrough while
  retaining source-audit artifacts; no SAFE monitor result was reproduced.
- `4d7ffe9` — **Exploration and evidence.** Archive the visual-monitor teaching
  route while preserving teaching-asset provenance and existing recording evidence.
- `81fcd56` — **Exploration.** Preserve HUG/controller/visual-feedback definitions
  as optional background; this commit does not add an experiment or implementation.
- `6790b2d` — **Current guidance.** Retain the latest broader HUG coverage question,
  calibration/feature diagnostic gate and simulation-only claim boundary. The
  training comparison remains proposed at this reviewed commit.

## Overall disposition

The implementation, collection, baseline and frozen diagnostic commits remain
necessary project evidence. Merge and branch-cleanup commits preserve authorship
and recoverable source history. Exploration helped choose the direction but
does not belong throughout the README or current handoff.

The documentation cleanup should leave a short project introduction, a single
current plan, measured results with links to their artifacts, focused operational
guides, the working paper and an explicitly historical exploration archive.
Completed diagnostics remain distinct from proposed comparisons and from
unmeasured recovery or real robot transfer.
