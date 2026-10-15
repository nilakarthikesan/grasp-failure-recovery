# Compiled hand conversion audit — 15 October 2026

Two shared implementation errors were reproduced and corrected: MuJoCo's URDF import aliased distinct mesh files sharing a basename, and compiled-XML serialization rounded hand/scene parameters. The original target URDF, its mesh bytes, scientific acceptance criteria, provider configuration and control policy are unchanged. The corrected compiler now realizes the declared target more accurately. **This is a conversion result, not generator-wide mapping acceptance or permission to run evaluations.**

## Frozen inputs and scope

The audit reuses 373 saved pose representations: 18 DGA (16 raw MultiDex plus two saved corrections), 64 UniDexGrasp (32 raw plus 32 existing CPU64 frame derivatives), 246 official DexGraspNet replay records, 32 HUG, and 13 ContactGen. Derivatives and corrections are counted separately; these are not 373 independent native outputs. All original rejections remain in the records. Three manufactured nonzero-wrist/rotated-root cases are static controls, not evaluation cases.

The declared target is `shadow_hand_right.urdf`, SHA-256 `29a50063cced2fba81dfe3d208ba54499fc12635c9189d0822f7caa2771c36de`. Runtime: MuJoCo 3.3.7. The independent oracle parses the original URDF, uses independent SciPy transforms and compares 30 non-world link frames plus 45 collision geometries against base, tabletop and actuated compiled models. It uses `mj_kinematics` only and blocks dynamics/solver entry points. Native reconstruction, operator-frame contracts, source/target limits and file identities have separate receipts.

`control/static_protocol.json` freezes the engineering checks without changing any scientific acceptance rule. Exact parameter identity and floating-point FK consistency must not be substituted for continuous surface preservation or execution adequacy.

## Verified defects and repairs

1. **Wrong collision-mesh file identity.** The index distal link intentionally uses `meshes/visual/f_distal_pst.obj`. Middle/ring/little distal links instead declare `meshes/collision/f_distal_pst.obj`. MuJoCo 3.3.7's URDF parser keys meshes by stripped basename and scale even with `strippath=false`. Consequently all four previously used the 2,691-vertex/5,374-face visual file; the other three should use the 526-vertex/1,048-face collision file. The importer now supplies unique virtual filenames bound to resolved paths and the original file bytes. Seven distinct hand mesh assets compile instead of six. No mesh was remodeled, rescaled or replaced in the source target.
2. **Precision loss between validation and execution.** `mj_saveLastXML` was used twice while constructing a scene and actuators. Joint bounds shifted by up to 3.67321e-6 rad, link positions by up to 65.83 nm, and some legal boundary commands were clipped. Five saved ContactGen cases acquired false compiled-limit violations. Scene construction now retains a live `MjSpec` and compiles directly; it never uses the global last-XML serializer. Full-precision fused fixed-tip masses/inertias are explicitly retained before recompiling the mutated specification. Controller gains, force limits, weld parameters and solver settings are preserved; dynamics were not tested.

The basename defect is independently established by file identity, counts and isolated explicit-name MJCF imports. A vertex-cloud distance is not a contact-surface error. The independent triangle/hull witness study in `mano/` reports separate sampled distances; its first meter-scale Trimesh diagnostic was corrected by an explicit all-triangle implementation and retained as superseded evidence. The corrected triangle-mesh witness reaches 3.829528 mm, while the sampled convex-hull boundary differences are about 0.024512/0.019194 mm by direction. These are bounded sampled witnesses, not continuous maxima or measured physical contact gaps.

MuJoCo's source explains the basename behavior: [3.3.7 URDF mesh parser](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/xml/xml_urdf.cc#L545-L562). Its supported live model-editing API is documented in [model editing](https://mujoco.readthedocs.io/en/3.3.7/programming/modeledit.html).

## Independent results

| Check | Before | After |
|---|---:|---:|
| Middle/ring/little distal imported mesh vertices | 2,691, wrong file | 526, declared file |
| Maximum joint-range discrepancy | 3.67321e-6 rad | 0 |
| Maximum link-position discrepancy across scene builds | 65.83 nm | 4.05e-16 m |
| Maximum link-rotation discrepancy | 3.914e-7 rad | 1.41e-15 rad |
| Maximum collider-frame translation discrepancy | 65.10 nm | 3.93e-16 m |
| False limit classifications among saved cases | 5 | 0 |
| Analytic geometry size discrepancy | 0 | 0 |

All three corrected compiled models now reach the same target-FK accuracy. The remaining maximum imported mesh vertex-cloud residual is 10.192 nm, principally the forearm's float32 mesh storage; do not call this exact continuous surface equivalence. Quaternion representability/projection and native float32 reconstruction errors are reported separately.

Primary before/after receipts: `control/compiled_pose_comparison_summary_v1.json`, `control/pose_audit_before_v3.json`, `control/pose_audit_after_v1.json`. Deliberate wrong-root, joint-axis and collider-offset controls verify that the checks detect conversion errors. Independent reviews are in `shadow/independent_pose_oracle_review_v1.json` and `mano/live_spec_implementation_review_v1.json`.

## What the generator comparisons establish

| Generator | Saved coverage | Result and remaining scope |
|---|---|---|
| HUG | 32 original mappings | Native MANO configuration, wrist/operator placement and target FK are consistent. Original fitting/surface residuals remain; this does not accept native-to-Shadow surface preservation. |
| ContactGen | 13 original mappings | PCA26/nonflat-mean MANO, shaped wrist origin, mean restoration, units and landmarks agree. Five false compiled-limit violations disappear; one also has an independent original fitting-gate failure, which remains. |
| UniDexGrasp | 32 raw + 32 prior frame derivatives | Source Shadow joint ordering/signs, root/palm conversion and nominal tip mapping agree. Its previously accepted 12-proposal input subset belongs to its separate recorded protocol; no new acceptance or physics result is inferred here. |
| DexGraspNet replay | 246 pinned official saved records | Native Shadow/MJCF conversion and named-joint mapping agree. 172 are target-legal, 120 source-legal and 117 jointly legal under the diagnostic strict limits. Legality and sub-2-mm tip residuals alone do not prove surface fidelity. |
| DGA | 16 raw + two saved corrections | Decoding and compensated forearm/root/palm mapping agree. Thumb spatial axes agree despite different local frames; a sign flip is not justified. Original five source-limit rejects, ten target-limit rejects and one domain-valid raw candidate remain. Both saved corrections remain unaccepted under their existing surface results. |

The DGA palm has a real one-box-versus-eight-box source/target difference, and the source/target thumb base differs by 0.08 mm. These are declared asset differences, not newly discovered conversion bugs. The earlier conditional palm/landmark obstruction is not disproved by fixing distal mesh import.

## Historical and future use

Original native outputs, checkpoint studies, rejected corrections and historical evaluation outcomes are preserved. Earlier compiled-geometry results describe the historically imported model. They must not be relabeled as validation of the intended collision-file identities or of this corrected import. See `mano/historical_claim_scope_addendum_v1.md`.

The implementation also removes legacy compiled-XML dependence from affected scene export/replay paths. New scene snapshots use exact, self-contained MJB files; the renderer retains support for historical XML inputs. MJBs are MuJoCo-version-specific, so retain the runtime version and source/model hashes. A live specification is required for further scene editing; an arbitrary compiled model is not silently reconstructed from global parser state.

The next unchanged scientific gate is provider-specific landmark/surface validation on the corrected compiler, with original and corrected implementation results kept separate. Then only accepted pipeline variants and eligible proposals can enter a separately authorized evaluation. Failed native poses and genuine morphology distortions cannot be repaired by renaming coordinates or relaxing thresholds. No physics, new native inference, retarget fitting, evaluation episodes/outcome labels or training ran in this audit.

The evidence manifest binds all archived reports, source snapshots, tests and compiled before/after MJBs. `control/historical_inputs_before.json` and the preservation receipt bind the pre-existing evidence and target assets. Absolute historical paths in receipts remain provenance; `path_relocations.json` maps this audit's temporary prefix to its durable archive.

## Final implementation checks

Twenty focused static regression cases pass with stepping and runner execution blocked. They cover declared mesh identity/topology, exact hand and object parameters, legal joint boundaries, nonzero-wrist placement, global parser-state independence, fused-tip mass preservation, exact MJB roundtrips, legacy XML reading, mass recompilation, and the legacy pilot helper. No dynamics suite ran. The final production models are byte-identical to the independently audited after snapshots. All 11,037 pre-existing evidence/target files in the frozen inventory remain unchanged. See `control/final_static_tests_v2.txt`, `control/final_model_identity.json`, `control/historical_preservation.json` and `control/implementation_files.json`.
