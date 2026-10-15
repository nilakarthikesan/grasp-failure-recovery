# Static Shadow-native conversion and compiled-model repair audit

15 October 2026. This directory contains static diagnostic evidence only. No physics integration, native inference, optimization, evaluation episodes, labels or training were run.

## Native adapters

`audit_shadow_conversions.py` independently parses the native MJCF and target URDF, computes XML forward kinematics, checks them against direct MuJoCo compilation, and compares decoded root poses and named-joint mappings to the unchanged production adapters. It retains every original output, including illegal poses. `report.json` and `target_pose_records.json` record the results; `target_cases_v1.json` is the 310-case interface to the coordinator's common compiled-target audit.

- UniDexGrasp: the original 32 native rows and the separately preserved existing CPU64 frame derivatives were checked. Both retain seven source-legal and 12 target-legal rows. All nominal fingertip residuals are below the existing 2 mm gate, with maximum 1.731283 mm. This does not change the original raw canonical-frame failures or promote a contact-preservation claim. Prior independently authorized physical outcomes elsewhere in the project were neither rerun nor relabeled.
- DexGraspNet replay: the pinned official saved sample file was absent locally and was downloaded from the exact catalog revision. Its SHA256 was verified before loading the official pickle container. All 246 existing saved records were checked: 120 source-legal, 172 target-legal, 117 jointly legal; all nominal fingertip residuals below 2 mm, maximum 1.736501 mm. These static diagnoses do not constitute new accepted proposals or new evaluation data.
- Independent source XML/direct-compiled FK, target XML/direct-compiled FK, and independent adapter root reconstruction each agree to at most 6.67e-16 in the measured matrix components. No sign, ordering, radians/meters, Euler convention, root offset, or duplicated object-scale conversion defect was found in these two adapters.

The direct-import checks above concern kinematics. The coordinator's independent common audit subsequently found a collision-mesh filename alias defect and precision loss in scene/actuator construction. Neither is visible from fingertip checks alone.

## Shared compiled-model repair

`precise_spec_fix/` contains the first reviewed scratch implementation and original-file hashes. It introduces unique virtual filenames per resolved native mesh path while retaining exact asset bytes, then builds the unchanged scene and actuators through a live `MjSpec`, avoiding a precision-limited `mj_saveLastXML` reconstruction.

A new edge was measured while implementing this: initial URDF compilation fuses fixed fingertip masses but removes those bodies from the live specification. Recompiling the mutated spec otherwise loses the removed 1 g tip masses. The fix explicitly preserves first-compiled body mass/inertia/frames before any further compilation, and disables additional static fusion. New object visual geoms remain retained. The probe checks 24 hand fields bit-for-bit through direct/base/actuated models and 18 actuator/equality fields against the previous controller. The original compound-object fixture retains one visual and two collision geoms, with exact inertial/geom frames; a separate asymmetric mesh with native refpose/scale also agrees exactly. Physics entry points and XML serialization are blocked during these probes.

`export_integration_fix/` is the subsequent scratch patch set. It keeps each runner's actual actuated live specification and replaces compiled scene XML exports with self-contained exact MJB snapshots. Video rendering loads MJB and continues to support historical XML inputs. The patch removes the process-global XML-identity dependency and prevents mesh/parameter re-rounding during replay or rendering. Existing historical XML files are preserved. MJB snapshots are tied to the compatible MuJoCo runtime; this is not a universal interchange-format guarantee.

The export integration probe performs constructor/spec-copy compilation only, blocks serialization/export/render/physics entry points, and checks 31 arrays bitwise. No new rollout, video, observation or evaluation exports were executed. `legacy_callsite_fix/` preserves the original diagnostic pilot and its intermediate helper update; its final combined variant is in `export_integration_fix/after/scripts/actuated_shadow_pilot.py`.

## Scientific boundary

These fixes restore the specified target import and conversion behavior. They do not make differing source and target hand shapes identical, prove native contact equivalence, remove existing DGA surface-preservation failures, or authorize evaluation. The coordinator's before/after compiled-target report separately measures the repaired importer across all generators. Acceptance criteria and native output histories remain unchanged.
