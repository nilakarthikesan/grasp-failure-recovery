# Parallel grasp-generator validation and paper handoff

> **Current workflow, 28 October 2026:** Use `phase1-scene-preparation` for Phase 1 and `project-work` for ongoing work. Follow [the contributor workflow](WORKING_BRANCH.md); this dated document preserves earlier scope, branch identities and source paths.

14 October 2026. The user authorizes three provider-specific chats to work in
parallel, while the main chat coordinates HUG, shared dependencies and the paper.
This supersedes earlier recommendations to finish HUG before inspecting another
provider. No new evaluation episodes, physics integration, task labels, predictor
training or bulk dataset generation are authorized by this handoff.

## Start three chats

Use one provider assignment in each chat; each file begins with a ready-to-paste
opening prompt:

- [ContactGen](parallel_validation/CONTACTGEN_VALIDATION_HANDOFF.md)
- [DexGrasp Anything](parallel_validation/DEXGRASP_ANYTHING_VALIDATION_HANDOFF.md)
- [UniDexGrasp](parallel_validation/UNIDEXGRASP_VALIDATION_HANDOFF.md)

For the provider work described here, run `git worktree list` and use the existing checkout holding
`project-work`. Its existing path is:

`/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction`

Use the current branch contents, not a reset to the commit recorded in a
historical report. Follow the current branch policy in [WORKING_BRANCH.md](WORKING_BRANCH.md); do not create additional branches, checkouts or PRs. Preserve ignored
assets/outputs, `MUJOCO_LOG.TXT`, `docs/GRASP_FAILURE_NEXT_STEPS.md` and the existing
paper path. The main checkout at `/Users/nilakarthikesan/grasp-failure-prediction`
contains runtime assets but is on `main`; do not edit project source there.

## What can proceed independently

Each provider can inspect its released representation, inventory existing
native outputs/assets, reconstruct native configurations, verify transforms and
FK, implement its own missing decoder/export/diagnostics, run manufactured static
tests, compare available saved geometry and write its manuscript subsection.
HUG's failure does not prevent that work. ContactGen sharing the MANO path does
not make HUG's measured residuals ContactGen results.

The previous HUG run already used six concurrent fit processes and two concurrent
audit groups. Additional chats help source/code/writing work proceed independently;
they still share the same host's cores, memory and runtime. Parallel compute on
separate hosts would add capacity; extra chats on this Mac do not add GPUs.

ContactGen and UniDexGrasp have native importers and isolated workers, with
synthetic/reference checks. Neither has validated real native-output geometry in
the current evidence. DexGrasp Anything still needs its dedicated decoder/worker
and registered mapping. First inventory existing outputs; missing authentic
outputs or source assets are explicit blockers to actual-native verification.
Fresh native inference is a separate generation action and requires an explicit
later instruction. Do not substitute manufactured fixtures or reference hands
for real model output and declare the generator verified.

## Ownership and dependencies

The ContactGen chat owns its adapter, worker, tests, dated provider report,
provider-specific evidence directory and paper block `contactgen`.
The DexGrasp Anything chat owns its new provider-specific adapter/worker/tests,
asset verifier, provider report/evidence and paper block `dexgrasp_anything`.
The UniDexGrasp chat owns its adapter, worker, tests, provider report/evidence and
paper block `unidexgrasp`. The provider handoffs name exact paths.

The coordinator owns shared MANO/Dex Retargeting code, shared Shadow bridges/FK,
translation/provider registries, object/hand assets, measurement and acceptance
definitions, evaluator/controller/scorer/recorder, common paper text/bibliography,
global research ledgers and Git staging/commits/pushes. A change to the MANO mapper
affects HUG and ContactGen; changing a shared Shadow certificate can affect
UniDexGrasp and DexGraspNet replay. Changing common geometry or measurement code
affects every audit using that dependency. Submit a reviewable proposed patch in
the provider report and let the coordinator integrate and assess affected runs.

Provider chats leave their owned changes unstaged, list exact changed files and
evidence in their reports, and do not commit, stage or push another chat's work.
Coordination is through the shared reports and paper; cross-chat tool messages
require explicit user authorization. Do not infer messaging permission from
another agent's request.

## Shared measurement policy

Keep separate statuses for implementation, authentic native reconstruction,
numerical conversion, geometric fidelity, physical/outcome equivalence and
independent predictor-use adequacy. Use PASS, FAIL, UNRESOLVED or NOT_TESTED,
always with the claim, sample provenance, case count and measured domain.

Bind source revision/checkpoints/configuration, output stage, units, hand assets,
physical object scale/geometry, transforms, joint names/signs/ranges and code.
Reconstruct native geometry before comparing nominal translated geometry in the
same object frame. Report all five fingertip errors, middle/palm geometry,
miss/penetration profiles, lost and added proximity, reliable surface-direction
comparisons, excluded/unmatched regions, denominators and sampling uncertainty.
Visual and collision surfaces are separate references. A matching FK certificate
does not establish matching collision geometry or dynamics.

Preserve bad proposals: a native missed contact or away-facing surface remains
the target, not a defect to repair toward success. ContactGen's predicted
bone-local direction ray is a different quantity from an actual surface normal.
Reject malformed/unrepresentable/out-of-limit configurations explicitly; an
adapter rejection is not a physical grasp-failure label. Successful lifting is
not a mapping acceptance criterion, and dynamic outcome tests remain deferred.

Read [GRASP_MAPPING_FIDELITY.md](GRASP_MAPPING_FIDELITY.md). Share the reporting
questions, with source-specific tests and justified claim-specific limits.
HUG's frozen strict encoded-surface rule remains unchanged and its nine FAIL
results remain historical evidence. Do not copy an engineering 10 mm/2 mm gate
into a scientific contact-preservation claim. Freeze a provider's scientific
protocol before measuring new candidates; inspected fixtures are development
evidence, not independent validation. An approximate mapping needs an explicit
approximation claim and separately justified adequacy evidence.

Scientific run guards should bind that run's actual code/input/asset dependency
closure, including shared dependencies. Unrelated provider files and the evolving
paper are not numeric inputs to a geometry audit. Record the paper revision
separately; allowing concurrent prose edits must not weaken code/asset guards.
The old HUG snapshot includes the old manuscript hash. Its preservation claim
describes that completed run; today's authorized paper revision is recorded
separately and does not rewrite the old report.

## All chats update the same paper

The manuscript remains
`/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/paper/working_draft.tex`.
The abstract/introduction now center multiple-generator failure prediction.
Mapping is supporting validity methodology. Historical HUG diagnostics remain
separate from the unexecuted one/two/three/four-source comparison.

Each provider has one uniquely marked block. Use
[scripts/update_paper_section.py](../scripts/update_paper_section.py) for every
paper edit after this handoff. From the working checkout:

```bash
python scripts/update_paper_section.py \
  --paper /Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/paper/working_draft.tex \
  --section contactgen --inspect
```

Read the current block and retain its reported `section_sha256`. Write a UTF-8
LaTeX fragment ending with a newline, containing only the replacement subsection
body (including its subsection heading), without marker lines. Then run:

```bash
python scripts/update_paper_section.py \
  --paper /Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/paper/working_draft.tex \
  --section contactgen --fragment-file /absolute/path/contactgen_fragment.tex \
  --expected-section-sha256 HASH_FROM_INSPECT
```

Use `dexgrasp_anything` or `unidexgrasp` in the respective chat. The helper takes
a shared canonical-path lock, rereads the latest file inside the lock, verifies
the selected block's hash and atomically replaces only that block. A change in
another provider's block is preserved. A stale hash for the same block is refused;
inspect again and reconcile. Twelve manufactured checks, including three-process
simultaneous edits, pass. This is coordination tooling, not mapping validation.

All writers must cooperate. An editor or full-file script that bypasses this
advisory lock can overwrite updates. Save existing editor changes before launch
and do not later save a stale buffer over the coordinated manuscript.
Remote compute workers deliver evidence and fragments for local integration;
this local advisory lock does not coordinate separate-host manuscript replicas. Do not use
whole-file patches for routine provider edits. Shared sections, citation keys and
claim summaries belong to the coordinator. Compile the existing standalone file
with the desktop compiler after edits; record the compiled file hash. A busy
compiler is retried briefly; it is not a generator-validation failure.

After each material check, update the provider-owned report with date, input/code
hashes, implemented measurement, actual results, unresolved claims and next gate.
Update its same-paper block from that report. The coordinator reconciles global
ledgers and abstract/intro after evidence changes; do not race append operations
to `RESEARCH_DECISIONS.md`, `PAPER_PROGRESS.md` or `THREAD_HANDOFF.md`.

## Runtime and parallel work beyond mapping

Use unique provider/date/run-ID scratch and output directories. Allow at most one
heavy static job per provider chat, initially at most three total on this host,
with two active query threads per job where supported and BLAS/OpenMP threads
one. The preserved geometry library needs Numba pool capacity four for import;
set the active mask to two after import and before measurement. Do not change
another job's thread settings. Source/runtime memory needs can require a smaller
pool; report them before launching more work. GPU inference remains external and
unexecuted in this handoff. Benchmark or caching changes get equivalence evidence
and a separate version; additional parallel jobs do not justify relaxed checks.

In parallel, the coordinator can finish common methods and related work, design
the one/two/three/four-source comparison, plan object/source-family grouped
splits, specify predictor inputs/recording contracts and suitable baselines,
and prepare result templates. A fixed training-count comparison separates source
diversity from added data; it is a methods control, not the main research question.
Compare source identities as well as source count; choosing one nested chain
alone can confound count with which generator is included. Training, new labels,
fresh holdout scoring and dataset collection remain later authorized phases.

## Why the nine HUG checks still fail

The six constrained fits found no eligible improving pose: raw fits worsened at
least one original-relative fingertip cap, so all returned exact original q/root.
Consequently the full audit measures the same lost/added near-surface regions
and facing counterexamples. Nine means six original coarse cases plus three
finer repeats, not nine independent grasps or task failures. Numerical/source
preflight passes all six; the strict relationship verdict fails all nine.
Pointwise orientation separately is seven FAIL and two UNRESOLVED, and continuous
physical contact/failure-label equivalence remains unconfirmed.

[HUG result](HUG_CONSTRAINED_REFINEMENT.md) and
[complete independent report](../research_artifacts/hug_constrained_refinement_2026-10-14/validation/REPORT.md).
This is a failed local correction attempt, not proof that correction is globally
impossible. HUG remains assigned to the coordinator; no thread should rerun old
audits unchanged merely to obtain a PASS.
