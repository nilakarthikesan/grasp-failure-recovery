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


## Latest authorized HUG evaluation — 13 October 2026

After provider implementation, the user requested one HUG `grasp-eval` run.
It completed real fresh HUG inference and one successful v2 cube execution:
maximum lift 0.151335 m, stable hold 2.0 s, no hand/table contact in 163 saved
samples, and a 6.52 s replay. The same existing observation/seed reproduces the
previous native proposal and trajectory exactly, so this is a development
reproducibility check. The predictor remains unchanged and unscored.

The earlier missing HUG dependencies are resolved by using the **existing**
`hug_integration_check/hug-deps`, `deps` and `deps/cmeel.prefix` import paths;
real compiled torch-cluster and cached model/MANO assets were present. No
dependency installation or stub was required. Use the full preserved runtime
from [the run note](HUG_SINGLE_EVAL_2026-10-13.md), which records actual versions
separately from the temporary integration-test runtime.

Local run: `runs/hug_grasp_eval_2026-10-13_fresh_cube/`. Result/video are under
`output/runs/hug_cube_seed0_nominal/`. ContactGen/UniDexGrasp evaluation generation
is still deferred; this later user request authorized only one HUG case.


## 14 October 2026 — Mapping audit is the immediate priority

The user requested proof of faithful native-to-simulator portrayal before
collection or training. [The static fidelity audit](GRASP_MAPPING_FIDELITY.md)
records what is measured and what remains pending. HUG coordinate composition
passes for the existing cube, but human-to-robot fingertips differ (mean 6.704 mm,
maximum 14.125 mm), and v2 extra closing deliberately changes commands. ContactGen
real-output contact fidelity and UniDex actual source-hand/native acceptance
are unverified. DGA's released output uses 24 joints and a forearm root, requiring
a dedicated native contract rather than the previous 22-joint assumption.

No generator inference, physics rollout, new evaluation dataset or fitting was
performed for this audit. Diagnostic plots/reports are separate from scientific
training data. Static nominal contacts and native near-surface vertices cover the same three
cube faces, but both intersect the object. Next compare finer patches/normals;
preserve the historical executor until any change is explicitly versioned.


## 14 October 2026 — Individual confirmation begins with HUG

The user clarified that generators must be tested and confirmed one at a time.
[HUG is the active case](HUG_MAPPING_CONFIRMATION.md). Exact native MANO
reconstruction, 18 guarded existing tests and independent coordinate/joint checks
pass. The detailed native/Shadow surface footprint changes despite coarse
finger/face agreement; occupied-or-near IoU 0.583 at a 2 mm band is a geometric diagnostic
including penetration, not a contact-fidelity pass threshold or predictor score.
HUG's overall status remains partial; ContactGen/DGA/UniDex have not advanced.

Next inspect per-finger grid/band sensitivity and anatomically matched normals
on the same existing cube. No new HUG sampling, physics/eval data or training
occurred. Existing adapters, controller, manuscript and predictor are unchanged.


## 14 October 2026 — Resume validation before paper writing and training

The user accepted the broader working title, **Evaluating Simulation Diversity
for Grasp Failure Prediction**, with multiple-generator training as the intended
comparison and mapping validity as supporting methodology. The latest direction
defers writing the paper and begins mapping validation before new collection
or fitting. Keep the earlier HUG diagnostic paper plan and historical protocols
separate from the proposed one/two/three/four-source training study.

[The HUG continuation](HUG_MAPPING_CONFIRMATION.md) now covers numerical
placement/joint consistency for 32 distinct existing native poses across four
objects, eight observation groups and one MANO shape. Mean tip residual is
6.428 mm, worst individual tip 17.518 mm. The finer-grid cube geometry mismatch
persists. Little-finger top-face coverage is nearly lost rather than absolutely
absent at the finest grid. Conditional same-label normals still do not represent
paired physical contacts. HUG contact fidelity remains unconfirmed.

Artifacts are in research_artifacts/hug_mapping_refinement_2026-10-14. All model
and mapper/executor/manuscript hashes are preserved; stepping was guarded and
zero time checked. No new inference, rollout, task label, dataset or fit occurred.
Next isolate wrist/palm anchoring from morphology and examine more existing
object surfaces before moving to the next generator. Future model tuning must
use development/validation data and preserve independent object/source tests.


## 14 October 2026 — HUG wrist/palm and other existing objects checked

Latest user direction: finish HUG's wrist-to-palm and finger surfaces on other
existing objects before advancing providers; do not generate new eval data.
[The new static bundle](../research_artifacts/hug_palm_object_validation_2026-10-14/README.md)
covers 32 anchor cases and six seed-0 apple/milk-box/mug surface cases. The
declared MANO-wrist/Shadow-palm placement matches exactly; the actual wrist
body is 34 mm away by URDF. A shared translation explains only 14.61% of tip
squared residuals and is not applied as calibration.

Changed finger proximity persists on both corrected collision-union v3 and
actual visual-triangle v4 references. V2 retained abutting internal interfaces
and is preserved as a superseded pilot, never pooled with final v3/v4. Actual
collision surfaces also differ from visual ones, so investigate that object
approximation before attributing all discrepancies to retargeting. Details,
protocols, hashes and scientific figures are in [the HUG note](HUG_MAPPING_CONFIRMATION.md).

HUG contact fidelity remains unconfirmed. Next declare the contact-preservation
objective and separate calibration/validation groups while reviewing collider
approximation. No inference, rollout, new task label/dataset, training, production
mapper/controller/collider change or manuscript edit occurred. Frozen predictor
hash remains unchanged. Optional static query arrays stay local and uncommitted.

GPU compute may help future generation; current MuJoCo mj_step and the logistic
predictor run on CPU. No compute was rented. Benchmark actual runtime/VRAM and
rollout bottlenecks after mapping validation and new-collection authorization;
independent object/source coverage matters more than repeated episode count.


## 14 October 2026 — Source-grounded shared validation framework proposed

The user explicitly accepts preserving bad native proposals and assessing
mapping fidelity before task success/failure. Read-only review of the four
original generator papers and pinned code is recorded in
[the mapping audit](GRASP_MAPPING_FIDELITY.md). Recommended common questions
cover native reconstruction, conversion/FK, geometric preservation and later
execution; source-specific contracts retain hand/frame/stage semantics.
Original quality filters are not conversion acceptance thresholds. Explicit
mapping rejections are not physical failure labels. No thresholds or new
collection protocol are frozen. HUG remains the active audit; no other provider
acceptance, new inference/eval data, training or paper edit occurs.


## 14 October 2026 — HUG object/reference static phase completed

[Latest HUG findings](HUG_MAPPING_CONFIRMATION.md) and [the versioned
geometry/reference
bundle](../research_artifacts/hug_geometry_reference_validation_2026-10-14/README.md)
complete source-vs-compiled object checks, all-32 reference point/FK checks, six
same-query surface comparisons and six signed hand/object profiles with finer
first-group repeats. No tested scale/frame bug is found. Apple/milk large
interior-wall distance witnesses reflect inherited micrometer slits; mug also
has exterior hull approximation. Do not enlarge probes or modify colliders to
improve diagnostic scores.

Metric-static reference tips improve (6.428 to 2.353 mm mean), but middle-link
proxies worsen (6.829 to 13.811 mm) and current whole unsigned 2 mm visual
proximity is higher for all six pairs. Per-finger tradeoffs remain explicit. All
six native area profiles already contain sampled penetration. Keep current
mapping and poor native proposals; lower penetration or a successful lift is not
a mapping gate.

Historical reference bound-table metadata is corrected separately, without
overwriting it: actual solver clamps use source URDF, legacy table lists rounded
captured-scene bounds. Final surface V3/signed V2 parse source XML and record
tiny scene excess; incomplete surface V2/signed V1 remain preserved. Do not
claim this static discrepancy caused a production preflight rejection.

[The measurement framework](HUG_STATIC_VALIDATION_PROTOCOL.md) clearly marks
implemented versus proposed criteria. Numerical native/coordinate/joint checks
are supported within scope, contact/outcome equivalence remains unconfirmed.
Next complete object-facing normals and spatial sampling uncertainty, justify
margins and freeze independent unseen-object/observation validation; inspected
poses are development evidence. HUG stays active before other providers. No new
inference, dynamics, eval data/labels, training, production change or manuscript
edit occurred. Protected model/code/paper and original inputs remain unchanged.


## 14 October 2026 — Orientation/uncertainty finished; strict HUG preservation fails

Latest authorized static phase is complete.
[Findings](HUG_MAPPING_CONFIRMATION.md), [acceptance
definition](HUG_MAPPING_ACCEPTANCE.md) and [evidence
bundle](../research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/README.md)
cover six fixed noncube cases/three finer repeats and all-32 native MANO
decoding. All native 3D vertices/landmarks/rotations and capped faces
reconstruct exactly; 32-camera reprojection is not claimed.

Frozen strict declared-band rule predates new metrics,
SHA87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4.
Positive-area definite same-part losses/additions cause conditional FAIL in all
six cases and three finer repeats; 178 cell witnesses remain auditable. Reliable
sampled facing/away reversals supply pointwise orientation failures. Numerical
conversion still passes in tested scope. Full continuous contact, failure-label
and predictor adequacy are not certified. No average IoU/angle/tip threshold or
lift-success gate is adopted; ambiguity cannot become a pass.

Sampling operational V2 d2103f7f… and orientation V2 22ef5c53… preserve
incomplete/pre-metric versions separately. Exact query/area order, band
regressions, zero-time/q/object guards, all sibling input bindings and 110
protected code/model/paper hashes pass. Conditional cell bounds use explicit
assumed 1/10/100 nm allowances, not formal arithmetic certificates. Distance
bounds do not extend to normals/occupancy/contact forces; carriers are
unavailable natively and outside the six-part gate. No new inference/eval
data/labels, physics, training, production mapper/collider/controller change or
manuscript edit occurred.

Next work is a versioned mapping improvement targeted at retained finger/region
differences, or an explicit narrower claim about a documented approximate
simulator pipeline with independent adequacy validation. Do not repeat completed
checks, loosen the frozen rule to pass HUG, advance other providers as accepted,
or begin collection/training from this static result. Preserve inspected
development cases as development evidence and all historical records.


## 14 October 2026 — Surface-aware HUG correction tested; no promotion

The user authorized repairing the failed mapping while retaining the no-new-eval
constraint. [The correction report](HUG_SURFACE_REFINEMENT.md) records a frozen
native-relative surface/closest-point refiner and a separately frozen guard.
All six raw candidates exceed the existing 10 mm mean fingertip limit; all six
guarded candidates satisfy it at 8.458–9.290 mm, with 10.527–21.471% lower fitting
cost. Fingertip/middle accuracy nevertheless worsens on every case.

Independent same-grid checks on six existing grasps plus three finer repeats
still FAIL the unchanged strict surface rule in all nine domains. Coarse whole
2 mm unsigned IoU and lost-area bounds improve, but first-milk added proximity
and several eligible direction counterexamples regress. Original witness replay
retains 65 definite changes of 178; uncertainty is not acceptance. Palm POSITION
is fixed; palm ORIENTATION follows the q-dependent source-hand convention and
changes 0.501–1.355 degrees. No free object/hand alignment or scaling is fitted.

New experimental APIs and 70 manufactured tests are implemented separately from
the unchanged default mapper/grasp-eval. Eleven frozen/API comparisons and all-six
stored engineering-assessment selection replay match exactly; this is not an
API rerun of six geometry solves. The versioned bundle preserves frozen modules,
raw refused and guarded targets, all nine audits, input/protocol hashes,
pre-metric/aborted histories and figures. All 110 original protected files,
model, manuscript and old input hashes remain unchanged; integration guards
record zero physics steps. No inference, eval episodes/labels or training ran.

HUG is not signed off as contact/failure-label equivalent and the next generator
is not accepted by this result. Proposed next work is a global surface solve
with fingertip constraints and reliable native-relative direction terms,
followed by the unchanged audit. Do not tune acceptance to these development
cases. No current evidence proves strict preservation globally impossible or
approximate predictor adequacy. Preserve the current paper path and existing
untracked/ignored files; continue on codex/project-work.


## 14 October 2026 — Fingertip/direction code checkpoint; HUG fits still running

Latest authorization: implement fingertip constraints and surface direction
fitting on existing HUG grasps, rerun mapping checks, generate no new eval data
and do not move to another generator. [The new method](HUG_CONSTRAINED_REFINEMENT.md)
and isolated API are implemented. Fifty new manufactured tests and all 120
installed new/prior surface API tests pass; this is not geometry sign-off.

Canonical joint method V1 runtime V2.1 is running all six existing seed-0
apple/milk/mug grasps from original q, fixed 120-iteration budget. Protocol
e42b74b787a8a1a5b5a661596466fe66b24c14eea39db25ad039fda2a462b8ff is at
/private/tmp/hug_constrained_refinement_runtime_v2_1_2026-10-14/protocol.json.
Complete fits, source preflight and unchanged six-coarse/three-fine audit are
PENDING at this checkpoint. Do not treat live callback probes as selected poses.

The final independent wrapper is prepared in
/private/tmp/hug_constrained_validation_runtime_v2_1_2026-10-14, protocol
9f0e0833c80a5b36f0ad21f629ecb84cf399f0e2e7669436eb3b93e561faf8f5.
Preserve incomplete V1, import-aborted V2 and unexecuted faster-helper V3 history
separately. Do not restart the active fits for further performance work. The
default mapper/controller/model/manuscript and 114 protected source bytes remain
unchanged. Continue the authorized static checks; no new inference, physical
eval data, training or another-generator advance is authorized.
