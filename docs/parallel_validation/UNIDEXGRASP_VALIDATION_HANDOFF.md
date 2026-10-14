# UniDexGrasp mapping-validation thread handoff

Prepared 14 October 2026 from `codex/project-work` at `0db42ca`. This is a
parallel work assignment, not completed actual UniDexGrasp mapping validation.

## Copy into the new thread

```text
Read docs/PARALLEL_GENERATOR_VALIDATION_HANDOFF.md and
docs/parallel_validation/UNIDEXGRASP_VALIDATION_HANDOFF.md in the existing
codex/project-work checkout. Use git worktree list to locate that checkout;
currently it is /Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction.
Own UniDexGrasp's mapping audit, provider-specific fixes, evidence and the
unidexgrasp subsection of docs/paper/working_draft.tex. Work in parallel with
the other provider threads. First inventory existing source-hand assets and
native outputs, then give a simple status and next check. Do not generate new
eval data, run physics, train a predictor, or invoke fresh native inference
unless I explicitly authorize that later. Do not change shared mapping/simulator/
registry files or commit/push without coordinating with the main thread. Follow
the handoff's paper-section updater and evidence/status rules.
```

## Actual starting point

The native importer and isolated IPDF-plus-Glow worker are implemented. Neither
the actual UniDexGrasp released generation hand nor actual generated native
proposals have been validated locally. Current actual-native mapping evidence:
**NOT_TESTED**. Full completion is blocked until authentic existing assets and
outputs are found or later explicitly authorized native inference supplies them.

Provider: `unidexgrasp_shadow_v1`. It checks an effective palm/finger kinematic
certificate and reuses `dexgraspnet_shadow_to_shadow_hand_right_v1`. The
max-per-tip 2 mm engineering FK gate is not a contact/surface/dynamics certificate.

Pinned official source:
[PKU-EPIC/UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp/tree/36c9bfcf7cedc987dd15e2384c548e51d5aadc21),
commit `36c9bfcf7cedc987dd15e2384c548e51d5aadc21`.

Read `docs/CONTACTGEN_AND_UNIDEXGRASP.md`, `docs/GRASP_PROVIDER_WORKERS.md`,
`docs/GRASP_MAPPING_FIDELITY.md`, and the provider/translation registries.
`research_artifacts/grasp_mapping_audit_2026-10-14/unidex_reference_static_fidelity.json`
is explicitly a DexGraspNet **reference proxy**, not the actual generation hand.
Its 112 synthetic poses have maximum tip difference 1.751 mm little and
0.422 mm thumb; other tips are approximately numerical zero. Two corner states
violate target joint limits and are rejected. Pure source-parser/MuJoCo body
positions agree within 71.5 nm on two root variants of 18 poses each. These
scope-limited results must remain separate from actual-native validation.

## Native contract to preserve

Our worker runs IPDF rotation plus non-joint GraspGlow translation/joint proposal
generation, **before TTA**, and does not run UniDexGrasp's learned execution
policy. Input binds metric point cloud, physical object scale, input frame,
input-to-object rigid transform and plane. It does not resample or rescale points.
Checkpoint architecture and effective configuration hashes must match the
selected pre-TTA stage.

Native `unidexgrasp_native_bundle_v1` binds numeric NPZ, inference report and
actual source MJCF. It contains translation, axis-angle rotation and 22 named
`robot0:` finger joints. Invert the sampled input-to-canonical rotation, compose
input-to-object and the actual source-root-to-palm pose, then map into 24 Shadow
targets with fixed target wrists. Never assume source palm root identity.
Preserve target-limit rejections rather than silently clip/repair proposals.

The current kinematic certificate excludes collider radii/shapes, mass and
friction: controlled capsule-radius and palm-mass mutations leave it unchanged.
Source capsule/box surfaces differ from the target URDF. An exact raw asset hash
and actual compiled surface comparison are required in addition to kinematics.
Imported `tta_hand_pose` requires a ContactNet hash but currently lacks binding
of the full TTA configuration; close that gap before such imports are used.
Current worker exports pre-TTA `hand_pose`, so do not mix these stages.

## Ordered work

1. Record current branch/HEAD, dirty files and hashes. Inventory authentic actual
   generation hand, IPDF/Glow configurations/checkpoints, exact existing object
   point clouds/plane/frame transforms and native bundles. Preserve original
   native arrays/report bytes. Do not use reference proxies or test fixtures as
   actual UniDex evidence; no fresh inference is authorized here.
2. Independently audit source canonicalization, rotation inversion, translation
   units/object scale, named joint order, actual source palm offset and output
   stage. Check nonidentity roots/transforms and fail-closed asset/config/input
   mismatches. Fix confirmed provider-specific defects with focused static tests.
3. With an actual source asset, verify its dependency closure, compiled inherited
   defaults/axes/limits/sites and certificate. Independently compare source and
   target FK over declared legal synthetic states before interpreting real
   outputs. A mismatch remains FAIL/UNRESOLVED; do not edit a certificate merely
   to accept an unknown hand. Send needed shared mapping changes to coordinator.
4. If authentic saved proposals exist, reconstruct native geometry at exact
   metric object scale/frame and compare each proposal's palm/orientation,
   fingertips and intermediate links. Verify point cloud/visual object/collision
   references actually describe the same declared geometry, not just equal
   copied identity strings.
5. Audit corresponding exterior finger/palm surfaces: signed/unsigned object
   distances, lost/new proximity, penetration and reliable normal/facing states.
   Separate overlapping collider interior walls, unmatched surfaces, visual vs
   collision geometry, tie ambiguity, coverage and sampling uncertainty. Tip
   agreement alone cannot prove contact patch or failure-label preservation.
6. Freeze justified source-specific claim/tolerances before new validation
   metrics, with calibration/validation scope explicit. Do not copy HUG's strict
   zero-distortion rule as universal or loosen criteria after inspecting results.
   Preserve native misses, poor/penetrating configurations and every rejection;
   nominal mapping fidelity is assessed before any eventual task success.
7. Rerun relevant static checks after fixes; classify each scoped claim PASS,
   FAIL, UNRESOLVED or NOT_TESTED. Keep reference synthetic, actual-native,
   nominal mapping and eventual controller/achieved execution evidence distinct.
   Document blocked claims if actual assets/outputs are absent, while completing
   independent source/contracts/diagnostics work.
8. Append dated provider progress and update the owned manuscript block as
   evidence arrives. Report exact results, provenance and remaining gate; passing
   tests or a reference certificate cannot sign off the full generator mapping.

## File ownership and paper updates

Own `src/grasp_failure_prediction/proposals/unidexgrasp.py`,
`scripts/infer_unidexgrasp_proposals.py`, `tests/test_unidexgrasp_provider.py`,
`docs/parallel_validation/UNIDEXGRASP_VALIDATION_REPORT.md`, and unique
`research_artifacts/unidexgrasp_mapping_validation_*` outputs. Do not overwrite
historical bundles. Bind scripts/assets/input hashes and zero-step guards.

The coordinator owns shared DexGraspNet translation/mapping/certificate policy,
registry/schema/FK helpers, simulator/Shadow/controller/acceptance helpers,
abstract/introduction, global ledgers and Git staging/commit/push. Shared bridge
changes also affect DexGraspNet replay; request review/integration and revalidation
of dependents. Append decisions to your report, then give the coordinator an
evidence-backed entry for `docs/RESEARCH_DECISIONS.md`/`PAPER_PROGRESS.md` rather
than simultaneously appending to shared ledgers.

All threads contribute to the existing manuscript
`/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/docs/paper/working_draft.tex`.
Own only marker section `unidexgrasp`. Inspect with
`scripts/update_paper_section.py --inspect --paper ABSOLUTE_PAPER_PATH --section unidexgrasp`,
write a local TeX fragment, then update with
`scripts/update_paper_section.py --paper ABSOLUTE_PAPER_PATH --section unidexgrasp --fragment-file ABSOLUTE_FRAGMENT_PATH --expected-section-sha256 HASH_FROM_INSPECT`.
If the helper/markers are absent, send a fragment to the coordinator. Do not
invent markers or rewrite the entire file. Refresh the section hash for each
update, use the helper lock and resolve stale-hash failures. Write the source
representation/conversion, verification scope and limitations now; pending
actual results remain explicit. Compile after edits with the desktop compiler;
transient busy states can be retried as permitted. The coordinator reviews the
paper and owns commits.

## Runtime boundary

At most one heavy job in this thread; agree CPU capacity with the coordinator
before costly work. Chats share CPU/RAM. Existing geometry imports need NumBa
pool capacity 4; set active threads to 2 after import when applicable and keep
BLAS/OpenMP at 1. Do not launch six simultaneous fits or mutate another thread's
runtime. Use distinct scratch/evidence directories. No physics, evaluation
episodes, new task labels, data collection or training. Fresh native inference
needs later explicit user authorization; source/assets audit, static diagnostics
and provider-specific fixes proceed now.
