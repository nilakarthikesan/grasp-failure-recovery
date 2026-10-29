# ContactGen mapping-validation thread handoff

> **Current workflow, 28 October 2026:** Use `phase1-scene-preparation` for Phase 1 and `project-work` for ongoing work. Follow [the contributor workflow](../WORKING_BRANCH.md); this dated document preserves earlier scope, branch identities and source paths.

Prepared 14 October 2026 from `project-work` at `0db42ca`. This is a
parallel work assignment, not completed ContactGen validation.

## Copy into the new thread

```text
Read docs/PARALLEL_GENERATOR_VALIDATION_HANDOFF.md and
docs/parallel_validation/CONTACTGEN_VALIDATION_HANDOFF.md in the existing
project-work checkout for the provider work described here. Use git worktree list to locate that checkout;
currently it is /Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction.
Own ContactGen's mapping audit, provider-specific fixes, evidence and the
contactgen subsection of docs/paper/working_draft.tex. Work in parallel with
the other provider threads. First inventory existing assets and native outputs,
then give a simple status and next check. Do not generate new eval data, run
physics, train a predictor, or invoke fresh native inference unless I explicitly
authorize that later. Do not change shared mapping/simulator/registry files or
commit/push without coordinating with the main thread. Follow the handoff's
paper-section updater and evidence/status rules.
```

## Actual starting point

The native importer and isolated inference worker are implemented. They have
not been run on a real ContactGen output locally. Existing unit fixtures mock
retargeting/FK boundaries; they are synthetic implementation checks. Current
actual native-output mapping evidence: **NOT_TESTED**. Full native mapping
completion is blocked until an existing authentic output is found or later
native inference is explicitly authorized and its external assets are present.

Provider: `contactgen_mano_shadow_v1`. Translation:
`mano_right_to_shadow_hand_right_v1`, shared with HUG. The current mean-tip
10 mm gate is an engineering gate, not a scientific contact-preservation limit.
HUG's measured errors and failed surface tests must not be assigned to ContactGen.

Pinned official source:
[stevenlsw/contactgen](https://github.com/stevenlsw/contactgen/tree/1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a),
commit `1aea2b6ce498ec90c4af0dd4c5fbdd4a23db761a`.

Read `docs/CONTACTGEN_AND_UNIDEXGRASP.md`, `docs/GRASP_PROVIDER_WORKERS.md`,
`docs/GRASP_MAPPING_FIDELITY.md`, and the provider/translation registries. The
existing artifact `research_artifacts/grasp_mapping_audit_2026-10-14/contactgen_transform_review.json`
checks transform algebra at two prescribed wrist states with maximum residual
`1.21e-16 m`; it explicitly contains no actual grasp/contact evidence.

## Native contract to preserve

ContactGen centers the input mesh by its arithmetic vertex mean, samples contact
locations/parts/directions, and runs its released `optimize_pose` to produce a
right MANO hand. Our worker uses 200 global and 1000 pose iterations, right
PCA-26 with nonflat mean, retaining optimized parameters, 778 vertices and
16 frames in `contactgen_native_bundle_v1`, stage `refined_mano`.

Restore the exact center and declared mesh-to-object rigid transform. Frame 0
includes the shaped MANO wrist position: optimized translation alone is not
the wrist origin. The modified upstream MANO implementation is already metric;
do not apply an extra millimeter conversion. Build the existing 21 thumb-first
landmarks, fit Shadow and retain the declared HUG wrist-to-palm origin policy.
Do not infer that this makes human and robotic surfaces identical.

The current worker discards sampled input points/normals and predicted
contact/part/direction maps. Add hash-bound preservation before any future
inference if the paper will claim source contact-to-native solver fidelity.
ContactGen's predicted direction is a bone-local ray; compare it according to
the source definition. It is not interchangeable with outward surface normal,
object-facing score or physical contact force.

## Ordered work

1. Record current branch/HEAD, dirty files, code/assets/configuration hashes and
   the exact object/scene/translation identities. Preserve ignored runtime
   assets, historical evidence and the open manuscript path. Inventory existing
   ContactGen source/checkpoints, hand-SDF, licensed MANO and native bundles;
   read contents/provenance before labeling a bundle authentic. Do not launch
   inference or use a synthetic fixture as real model evidence.
2. Recheck pinned native source behavior and centering, units, shape-dependent
   root, MANO configuration, mesh-to-object transform and landmark ordering.
   Independently derive transforms with nonidentity placements. Check imports
   fail closed for incompatible stages, geometry/hash mismatches and malformed
   data. Fix confirmed provider-specific contract/export defects with focused
   manufactured tests and zero simulator steps.
3. If an authentic saved bundle is available, reconstruct its original MANO
   from parameters/assets and compare all vertices/frames to its saved native
   output before translating. Bind original input mesh and any contact maps.
   Verify that the input surface actually matches the declared simulator object
   at the same metric scale/frame; copied hashes alone cannot prove this.
4. Compare each native proposal to its own nominal translated Shadow geometry:
   wrist/palm position/orientation, every fingertip and middle segment, matched
   finger/palm surfaces, signed/unsigned distance profiles, lost/new proximity,
   penetration and reliable surface-facing states. Distinguish visual and
   collision references, exterior union from individual collider walls, and
   geometric proximity from physical contact. Report unmatched anatomy, tie or
   normal ambiguity, surface coverage, denominators and sampling sensitivity.
5. Justify and freeze source-specific acceptance claims/tolerances before new
   validation metrics. Do not copy HUG's strict zero-distortion claim as a
   universal requirement, loosen a threshold to pass inspected cases, or claim
   global validity from mean-tip error. Preserve misses, penetration and other
   poor native proposals; successful lifting is not a mapping criterion.
6. Rerun relevant static checks after confirmed fixes; retain original and
   candidate results separately. Report PASS, FAIL, UNRESOLVED or NOT_TESTED for
   each claim, evidence scope and assumptions. Missing native assets/samples
   block dependent claims while source/static engineering work can continue.
7. Update your owned report and paper block as evidence arrives. End with exact
   completed evidence, blocked claims, patch paths and the next gate; do not
   mark the generator verified solely because software tests pass.

## File ownership and paper updates

Own `src/grasp_failure_prediction/proposals/contactgen.py`,
`scripts/infer_contactgen_object.py`, `tests/test_contactgen_provider.py`,
`docs/parallel_validation/CONTACTGEN_VALIDATION_REPORT.md`, and unique
`research_artifacts/contactgen_mapping_validation_*` outputs. Preserve old
artifact bundles; new reports bind scripts/input hashes and record zero steps.

The coordinator owns shared MANO retargeting, frame helpers, translation registry,
schemas, Shadow model, controller, evaluator, geometry/acceptance helpers,
abstract/introduction, global ledgers and Git staging/commit/push. Propose changes
there as reviewable patches, not concurrent mutations. A shared MANO change
affects HUG as well as ContactGen and invalidates dependent evidence unless
rechecked. Append dated entries to your provider report; send the coordinator
the evidence-backed decision for `docs/RESEARCH_DECISIONS.md`/`PAPER_PROGRESS.md`.

All threads contribute to the same manuscript
`/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/paper/working_draft.tex`.
Own only the marker section `contactgen`. Inspect with
`scripts/update_paper_section.py --inspect --paper ABSOLUTE_PAPER_PATH --section contactgen`,
write a local TeX fragment, then update with
`scripts/update_paper_section.py --paper ABSOLUTE_PAPER_PATH --section contactgen --fragment-file ABSOLUTE_FRAGMENT_PATH --expected-section-sha256 HASH_FROM_INSPECT`.
If the helper/markers are absent, send a fragment to the coordinator; do not
invent marker formats or rewrite the paper. Refresh the section hash before each
update, use the helper's lock, and resolve a mismatch rather than overwrite it.
Record representation/conversion/measurement/limitations now; clearly label
pending results. Compile via the desktop standalone compiler after manuscript
edits, with transient busy retries as permitted; diagnostics are not evidence
of scientific acceptance. The coordinator reviews all paper claims and commits.

## Runtime boundary

At most one heavy job in this thread. Before a costly batch, agree CPU capacity
with the coordinator; parallel chats share the same CPU/RAM. Existing geometry
imports need a NumBa pool capacity of 4 while active threads may be set to 2
after import; keep BLAS/OpenMP at 1. Do not launch six fitting processes here or
change another thread's environment. Use distinct scratch/output directories.
No evaluation episodes, new task labels, dynamics, dataset collection or model
training in this assignment. Fresh native inference needs later explicit user
authorization; inventory/audit/diagnostics and provider-specific fixes proceed now.
