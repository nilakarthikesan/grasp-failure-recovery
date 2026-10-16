# Compensated native capsule profile — source-only declaration

This is a separately named controller/model condition, `manosim_authored_capsule_pure_gravity_compensation_v1`. It keeps the exact saved native shape/targets, authored capsules/palm, masses/inertias, gains, damping and51 angle-axis position commands. The completed uncompensated capsule v3 baseline and its failures/predecessors remain immutable.

All16 hand bodies receive `gravcomp=1`, with `jnt_actgravcomp=0` on their19 joints. MuJoCo therefore applies the sum of `−m*g` at the actual inertial COMs through the passive force path. It does **not** cancel moving Coriolis/centrifugal bias, call a learned model, add feedforward applied forces, offset target poses, or set qpos during integration. Pinned official source is in `primary_sources/`; `engine_passive.c` implements COM forces and joint routing, and `engine_forward.c` combines passive minus velocity-dependent bias. The builtin compensation is the intentional new profile, not a correction to historical gravity-sag results.

## Fixed scope and budgets

Cases20 and28 are the two existing development HUG inputs. Each has2ms/1ms versions of three families:1.5s gravity motion/hold with all16 bone rotations and wrist tracking;0.75s simultaneous index-distal and middle-distal contact against separate analytic planes;1s root-only approach to unchanged q0 around a fixed sphere and table. Every model clears source/actuator, gravity, path/contact and principal-chart static checks before the first step of **any** fixture. All old dynamic/force/Jacobian/KKT/convergence tolerances remain unchanged. No task outcomes or collection/training permission are produced.

The two-plane normals derive from the original capsule axial+radial directions. A deterministic minimum-norm root translation solves both0.5mm indentation targets; normal-condition>1e6 or move>2mm refuses with no replacement, scaling or outcome selection. Per-pair contacts must be unique, elliptic3D and source-compliant. A positive simultaneous reaction is required throughout the final0.1s; summed51D force and virtual power use both independent ancestral point Jacobians. Static mathematical tangency allows0..2 contacts, while any present pair must have the realized plane frame, gap and normal.

The approach has51 fixed linear knots from q0−60mm worldX to q0, sphere center(0.3,0,0.1)m/radius20mm, table z=−0.1m, required clearance2mm. The fifteen capsules have analytic sphere/table separation. The convex palm uses exact table vertex support and a conservative enclosing-sphere bound against the sphere. All16 colliders and both obstacles are checked;1nm is the declared conservative arithmetic budget for these bounded (<10m) analytic operations, not contact/fidelity acceptance. No mesh/GJK oracle or object-query fitting is admitted. A1-Lipschitz mathematical segment bound is evaluated for the fixed rigid reference translation with the declared engineering arithmetic budget; no generic formal interval proof is claimed. Actual servo motion has once-per-step clearance and engine-contact checks; it has **no continuous swept-trajectory certificate**. Hand2/affinity1 versus obstacle1/affinity2 disables self-contact explicitly in this approach condition.

Hard total budget:12model compilations,12data,240static forwards (228before integration, including204path-knot forwards),9750steps,4500contact-force reads,600s worker/610s parent. Per case6models/data,120forwards,4875steps,2250force reads,300s dynamics. Any failure is retained; no retries/extensions or adaptive replacement. All dynamic geometry/force channels use pre-step clock, with endpoints recorded separately. Actual body COMs, anchors and rotations are saved to permit independent force-map algebra.

## Reusable scene/control API

`native_hand.py` contains pure wrappers without MuJoCo/model imports. `validate_hand_scene` accepts either67q/51v/51u/19j standalone or exactly74q/57v/51u/20j with one appended world object freejoint (id19,qadr67,dadr51), uncompensated and outside the hand. `read_hand_state` returns copies of the hand prefix. `command_from_hand_qpos` preserves the existing named principal-ball position-command rules. `write_hand_controls` validates the compiled signature then writes only ctrl51. Object state and applied forces stay untouched. Call `profile_math.compensated_model_receipt(model, contract.hand)` separately when using this compensated condition. Unsupported scene layouts refuse. The collector must independently qualify its object geometry/approach/task condition; these fixtures grant no task admission.

## Reproduction and evidence policy

`run_profile.py prepare --durable-report PATH --durable-report-sha256 SHA` is read/hash/XML-only. It requires `controls.json`, the durable reconstruction receipt, exact19 generated assets and builder receipts, completed uncompensated v3/readback, both historical capsule failures, live required inputs, selected Git sources, runtime files and pinned gravity/compiler/frame sources. Missing historical temporary-only closure remains disclosed in the durable audit; it is never called reverified. Imported v3 case/math helpers must match their original frozen Git bytes.

After parent review and protocol commit, the single allowed command is:

```
ORIGINAL_VENV/bin/python research_artifacts/manosim_native_physics_profile_2026-10-16_v1/run_profile.py run --protocol research_artifacts/manosim_native_physics_profile_2026-10-16_v1/protocol.json --freeze-commit EXACT40CHAR_SHA
```

The driver inserts the isolated MuJoCo3.4.0 runtime before imports and preserves NumPy2.2.6. Only registered model/data/forward/step/contact-force operations are exposed; warnings, nonfinite values, time/control mutations, unplanned loads, extra APIs and quota escapes hard-stop. Raw data under ignored `runs/manosim_native_physics_profile_2026-10-16_v1/run_v1/raw` retain creation-time hashes. Complete inventory is80artifacts; partial contact/live-state operands survive failures. The parent report keeps scoped qualification separate from every closed admission flag. No actual native fixture has run merely by preparing source/protocol.
