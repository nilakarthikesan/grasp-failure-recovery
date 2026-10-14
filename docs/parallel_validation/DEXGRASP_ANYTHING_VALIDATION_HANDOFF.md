# DexGrasp Anything mapping-validation thread handoff

Prepared 14 October 2026 from `codex/project-work` at `0db42ca`. This is a
parallel work assignment; DGA's native decoder/accepted mapping are not yet
implemented.

## Copy into the new thread

```text
Read docs/PARALLEL_GENERATOR_VALIDATION_HANDOFF.md and
docs/parallel_validation/DEXGRASP_ANYTHING_VALIDATION_HANDOFF.md in the existing
codex/project-work checkout. Use git worktree list to locate that checkout;
currently it is /Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction.
Own DexGrasp Anything's source/native contract, dedicated 24-joint mapping
implementation/audit, evidence and dexgrasp_anything subsection of
docs/paper/working_draft.tex. Work in parallel with the other provider threads.
First inventory existing assets/native outputs, then give a simple status and
next check. Do not generate new eval data, run physics, train a predictor, or
invoke fresh native inference unless I explicitly authorize that later. Do not
change shared mapper/simulator/registry files or commit/push without coordinating
with the main thread. Follow the handoff's paper updater and evidence rules.
```

## Actual starting point

Provider `dexgrasp_anything_shadow_v1` is registered as external, but its
translation strategy remains **`unsupported_pending_official_mapping`**.
There is an asset verifier, pinned cloud-runtime handoff and synthetic source
URDF bridge audit. There is no DGA-specific worker/decoder or accepted actual
native mapping. Actual-native evidence: **NOT_TESTED**. Full completion is
blocked until authentic existing output/assets are available or later native
inference is explicitly authorized.

Pinned official source:
[4DVLab/DexGrasp-Anything](https://github.com/4DVLab/DexGrasp-Anything/tree/86353d710ca71d0f547f0d8b15cc8ca959c78163),
commit `86353d710ca71d0f547f0d8b15cc8ca959c78163`.
Checkpoint `DexGraspNet.pth` from `GaussionZhong/Dexgraspanyting`, HF revision
`b24637ce366af4dd75cff723d2b853179155ec29`, SHA-256
`f7675d8eb93b3978b2c2d72ec9f54ead6b936d1dd307e07605daa0c02bcac14f`.
Check the authoritative `docs/DEXGRASP_ANYTHING_CLOUD_HANDOFF.md` for the exact
HF revision spelling and acquisition contract before any later asset action.

Read that handoff, `docs/GRASP_MAPPING_FIDELITY.md`, `docs/GRASP_PROVIDER_WORKERS.md`,
`scripts/verify_dexgrasp_anything_assets.py`, the registries and
`research_artifacts/grasp_mapping_audit_2026-10-14/{audit_dga_static.py,dga_static_fk.json}`.
The static bridge tested zero plus 64 deterministic shared-limit random states:
maximum tip discrepancy 0.100 mm little and 0.080 mm thumb, other tips near
numerical zero. This is a promising **synthetic candidate**, not accepted DGA
native mapping; origins/axes/limits/assets are not byte-identical.

## Critical source-contract correction

The old cloud handoff's 22-joint free-palm recommendation is superseded by its
dated correction and actual source audit. The pinned visualizer exports
translation, **6D rotation and 24 joints including both wrists**: 33 values.
The source URDF is rooted at the forearm. At zero articulation, root-to-palm
translation is `[0, -0.010, 0.247] m` and changes with wrist configuration.
Compose actual source-root-to-palm FK and preserve WRJ1/WRJ2; never drop wrists,
treat native root translation as palm translation, or route through the
22-joint DexGraspNet MJCF decoder as if source identity were the same.

Official point cloud/normals condition diffusion; physics-guided optimization is
an optional generation stage. Denormalization and random object rotation undo
must follow the chosen dataset/checkpoint exactly. The documented smoke command
inherits `dataset.active: real_dex` although the pinned checkpoint is DexGraspNet;
without `OPT` it disables physics-guided sampling. Record dataset, normalizer,
scale, source hand, stages and optimizer configuration explicitly. Dataset-average
visualizer scaling cannot replace the registered object's physical dimensions.
Generation, any upstream angle preconditioning/force tests, our nominal mapping
and common controller execution are separate states.

## Ordered work

1. Record current branch/HEAD/dirty files and hashes. Inventory existing pinned
   source URDF/dependencies, checkpoint, exact object mesh/cloud/normals and any
   authentic native outputs/configurations/logs. Preserve original bytes. The
   existing verifier validates pins before later checkpoint loading. No runtime
   install, checkpoint execution or fresh generation is required/authorized now.
2. Re-audit native tensor order, normalization/inverse transforms, 6D rotation
   convention, all 24 named joints, limits, axes, forearm root and actual palm
   link. Independently derive object/root/palm composition for nonzero wrists
   and nonidentity object transforms; keep units/frame/scale explicit.
3. Implement dedicated DGA provider-specific decoder/worker/export contract and
   static diagnostics as needed, with manufactured fixtures proving correct
   joint/root behavior and fail-closed malformed/hash/stage inputs. A worker
   entry point can be checked without CUDA loading; do not invoke its inference.
   Propose the dedicated strategy/registry/evidence registration to coordinator
   rather than editing shared registries concurrently.
4. Reproduce the dedicated synthetic bridge using independent source/target
   FK with declared joint ranges and every relevant palm/finger link. Audit
   visual/collision asset surfaces in addition to tips. Do not assume small tip
   discrepancy proves collider identity or physical equivalence. Preserve
   unsupported/unrepresentable/target-limit cases with explicit rejection.
5. If an authentic saved native record exists, reconstruct untouched native
   geometry and verify exact shared-object scale/frame and input-surface identity.
   Compare nominal palms/orientations, tips/middle links and matched exterior
   surfaces: signed/unsigned distances, lost/new proximity, penetration and
   reliable normal/facing witnesses. Separate source visual from collision
   references, interior overlapping collider walls, missing surface matches,
   tie ambiguities, area denominators and sampling uncertainty.
6. Justify source-specific acceptance claims/tolerances and freeze them before
   new validation metrics. Do not adopt HUG's zero-distortion claim universally,
   loosen criteria to pass examples or use lift success as the mapping gate.
   Preserve native misses and poor/penetrating proposals. A changed/refined
   generator stage must be explicit rather than hidden in coordinate mapping.
7. Rerun relevant static checks after provider-specific fixes and classify
   scoped claims PASS, FAIL, UNRESOLVED or NOT_TESTED. Report missing actual
   samples as a blocker to their dependent claims, while completing independent
   engineering/source work. Do not relabel synthetic bridge evidence as fresh
   checkpoint output or a verified complete mapping.
8. Append dated provider progress and update the owned paper subsection as
   evidence arrives; deliver exact files/results, proposed shared patches and
   next gate to coordinator. No new task labels/data or learned-executor comparison.

## File ownership and paper updates

Own new DGA-specific files, such as
`src/grasp_failure_prediction/proposals/dexgrasp_anything.py`,
`scripts/infer_dexgrasp_anything_proposals.py`, `tests/test_dexgrasp_anything_provider.py`,
and existing `scripts/verify_dexgrasp_anything_assets.py` where necessary. Own
`docs/parallel_validation/DEXGRASP_ANYTHING_VALIDATION_REPORT.md` and unique
`research_artifacts/dexgrasp_anything_mapping_validation_*` output bundles.
Preserve previous audit artifacts and bind every new report to input/code/assets,
configuration and zero-step evidence. Coordinate file names if another thread
already owns a DGA implementation.

The coordinator owns shared registries/translation schemas/geometry/FK helpers,
Shadow model, controller, evaluator, shared acceptance rules, abstract/introduction,
global ledgers and Git staging/commit/push. Send reviewable shared patches for
integration instead of concurrently changing those files. Append dated decisions
to your provider report; coordinator reconciles `docs/RESEARCH_DECISIONS.md` and
`PAPER_PROGRESS.md` without racing writes or merging historical scientific rules.

All threads contribute to the same existing manuscript
`/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/paper/working_draft.tex`.
Own only marker section `dexgrasp_anything`. Inspect using
`scripts/update_paper_section.py --inspect --paper ABSOLUTE_PAPER_PATH --section dexgrasp_anything`,
write a local TeX fragment, then update using
`scripts/update_paper_section.py --paper ABSOLUTE_PAPER_PATH --section dexgrasp_anything --fragment-file ABSOLUTE_FRAGMENT_PATH --expected-section-sha256 HASH_FROM_INSPECT`.
If helper/markers are absent, send the fragment to coordinator. Do not invent
markers, alter other blocks or rewrite the paper. Refresh section hash before
each update, honor helper lock and resolve stale-hash failures. Document native
representation, dedicated conversion, measured synthetic scope and limitations
now; pending real-native results remain explicit. Compile after paper edits via
desktop standalone compiler, retrying transient busy states as permitted. The
coordinator reviews claims and owns commits.

## Runtime boundary

At most one heavy job in this thread. Agree shared CPU capacity before costly
work; additional chats do not supply more machine resources. Existing geometry
imports require NumBa pool capacity 4; use active threads 2 after import when
applicable, BLAS/OpenMP1, and distinct scratch/output directories. Do not launch
six simultaneous solvers or change another thread's runtime. Official fresh
model runtime is external NVIDIA/Ubuntu with pinned CUDA dependencies/Isaac Gym;
that setup and native inference await later explicit authorization. No dynamics,
evaluation episodes, new task labels, data collection or predictor training.
