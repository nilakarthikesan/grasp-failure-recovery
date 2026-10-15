# Shadow closed-grasp physics filter v1 — prepared protocol

Prepared 15 October 2026. This implements the Phase 2b screening API and static
contract tests. **No six-axis physics experiment has been run or validated by
this implementation work.** Runtime execution follows Phase 1 signoff and a
separate frozen, bounded collection protocol. Root owns that authorization.

The source is `src/grasp_failure_prediction/proposals/physics_filter.py`; its
packaged configuration is `proposals/configs/physics_filter.yaml`. The identity
is `shadow_physics_filter_v1`. Freeze source/config hashes before using it. Any
threshold, timing, support-region or table policy change needs a new filter ID
and separate results. Historical task rules, labels and frozen weights remain
independent of this filter.

## API and state boundary

`filter_closed_grasp(model, closed_data, geometry, configuration=..., provenance=...)`
accepts a model and **already closed** MjData. The caller selects and records the
closed-state time by an independently declared approach/close protocol. This API
does not approach, close, stabilize, repair or refit the grasp and does not read
later task outcomes. Controls and mocap targets remain fixed at their supplied
values. Object mass, inertia, contact coefficients and hand/controller parameters
come from the supplied compiled model. Caller provenance must bind asset closure,
model/controller/protocol, proposal/command, scenario and recorder hashes.

Six separate model copies and fresh MjData are used for world axes +X, -X, +Y,
-Y, +Z, -Z. Every copy restores the same exact `mjSTATE_INTEGRATION` snapshot:
time, qpos, qvel, activation, solver warm start, ctrl, applied forces, active
equalities, mocap position/quaternion, userdata and plugin integration state.
Restoration is checked by exact array equality and its SHA-256 is recorded per
axis. The original gravity is copied before assigning each declared axis; source
gravity and collision masks are never changed. Source full integration state,
qacc, warnings, gravity and masks are compared exactly on exit. Concurrent source
mutation is unsupported and causes a hard integrity error.

The object must own one free joint, with no direct joint actuation, active
object equality, externally applied object force or gravity compensation.
Gravity must be enabled. All active object collision geoms must be identified.
`PhysicsFilterGeometry.from_shadow_model` implements the current runner's named
body contract; callers can provide explicit IDs for other qualified models.
This is intended for the qualified Shadow controller, not arbitrary force
plugins or unknown transmission mechanisms.

## Fixed engineering thresholds

Defaults are chosen before measuring any filter outcome:

- Hold each independent axis for 1.0 second at 9.81 m/s². With the current
  2 ms step this is 500 steps per axis and 3,000 attempted steps per proposal.
  The declared hard bounds are 1,000 per axis, 6,000 total and 60 seconds wall
  time, checked before each attempted step. Hold/contact windows must be integer
  timestep multiples; an oversized request is refused before integration.
- Reject any axis exceeding 20 mm maximum object-origin displacement in the
  current palm coordinate frame, relative to that axis's initial state. This
  uses the existing task's 20 mm loss scale as a screening engineering scale;
  it does not reuse or redefine task success. Relative rotation is report-only.
- Count the five distinct **distal finger** regions and palm separately. A
  contact must have an active constraint address and normal force strictly above
  0.01 N; `ncon` and margin-only proximity do not establish support. Multiple
  contacts/geoms on one finger count once. At least two fingers must support in
  90% or more of samples in **each 200 ms window**; no unsupported run may exceed
  40 ms. The force floor matches the runner's significance convention and 40 ms
  is one controller interval. These persistence settings screen chatter and
  missing support; they are not force-closure or universal grasp criteria. Palm
  support is reported and cannot substitute for the two fingers.
- Reject maximum negative MuJoCo hand/object contact distance above 5 mm over
  the initial and all integrated states. This cap is 10% of the nominal 50 mm
  cube edge, a declared broad quality-control scale. It is not a calibrated
  penetration tolerance for all objects, native MANO surfaces or soft objects.
  The collision mesh and MuJoCo soft-contact model determine this diagnostic.
  Penetration alone cannot establish grasp stability.
- Reject force above 0.01 N on an object/non-hand contact. This prevents external
  support from being mistaken for hand retention. All declared table geom masks
  are disabled on the model copies by default. Explicit table contact pairs are
  refused because collision-mask changes would not disable those pairs.
- Optional `minimum_table_clearance_m` defaults to null. If enabled, caller must
  declare table IDs and finite top-plane Z. Initial compiled hand geometry is
  checked with `evaluation.table_clearance._lowest_world_z`. This is a static
  infinite-plane gate for the supplied closed state, not an approach sweep or a
  certificate that a dynamic trajectory will clear the table. No wrist search
  or pose substitution happens inside the filter.

MuJoCo's `mj_step` forwards before integrating qpos. After every guarded step,
this protocol explicitly runs `mj_forward` to sample forces and geometry at the
integrated endpoint, including the last one. This forward-pass policy is part
of the version. It must not be changed silently between cohorts.

Warnings, qpos/qvel/qacc/ctrl/activation/mocap nonfiniteness, nonfinite force or
frame diagnostics, stalled/reset simulation time and quota overruns produce
`filter_status=error`. Warning counters and finite state are checked before and
immediately after every step, before the endpoint forward pass can obscure a
numerical fault. Time must advance by exactly one step within 1e-12 seconds.
Later axes are retained as `not_run` after a runtime error. Physical threshold
rejections still run all six axes for complete measurements.

## Metadata, retention and downstream use

Result schema `physics_filter_result_v1` includes all six axis records,
configuration, source state hash, caller provenance, attempted steps, elapsed
wall time, per-window support fractions, per-region contact fractions, maximum
displacement/rotation/penetration and external support. Codes include
`GRAVITY_FAIL_<axis>`, `INSUFFICIENT_CONTACT`, `CONTACT_DROPOUT`,
`PENETRATION_EXCEEDED`, `EXTERNAL_OBJECT_SUPPORT`,
`TABLE_COLLISION_PREDICTED` and `FILTER_RUNTIME_ERROR`. Each physical rejection
records measured value, threshold, units and comparison. Errors preserve the
exception and attempted step count; they do not become physical outcome labels.

This API returns metadata for one proposal. **The caller retains all proposals,
including rejections and runtime errors**, and freezes accepted/rejected rollout
quotas separately. `task_outcome_label` is null and
`predictor_feature_eligible` is false. Filter measurements/codes are diagnostics,
not inputs to the failure predictor. Accepted/rejected outcome rates can only
be measured after the independent task collection, without changing thresholds.

Runtime budgeting is expressed primarily as steps. A 16-command screen requests
48,000 steps; 32 commands request 96,000. Existing saved recording pilot timing
is not a benchmark for six-axis contact/forward/model-copy overhead. Run a
separately authorized, bounded engineering smoke before estimating full-batch
wall time; the 60-second per-proposal stop remains binding.

## Verification and limitations

`tests/test_physics_filter.py` blocks real `mj_step`, `mj_step1` and `mj_step2`.
It uses only model compilation/static forward calculations and manufactured
clock/contact callbacks. It checks six independent exact state restorations,
source immutability, full endpoint sampling, warning-reset and time guards,
quota refusal, measured rejection records, palm-relative invariance, inactive
contacts, distinct finger counting, per-window dropout and static table gating.
Passing these tests establishes software contracts only; it is not a measured
gravity-retention result or real-world validation.

Relevant engine semantics are documented by the pinned MuJoCo 3.3.7 primary
sources: [gravity, collision masks and contact parameters](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html),
[soft contact model](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html#soft-contact-model)
and [simulation state and stepping](https://mujoco.readthedocs.io/en/3.3.7/programming/simulation.html).
