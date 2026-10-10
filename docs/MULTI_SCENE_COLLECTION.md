# Completed collection: two geometries, four scenes, 1,040 attempts

This controlled collection milestone is complete. We captured four initial
object-position/camera scenes of the existing cube and a HUG apple, generated
four fresh HUG proposals per scene (seeds 0–3), and executed 1,040 attempts with
bounded finger-target noise. The result is 444 task successes, 596 task failures,
and zero validation/runtime errors. These are four simulation input families,
not 1,040 independent scenes or real-world recordings.

The fixed controller, bounded finger-target noise, final scorer, and conservative
event annotations remain those documented in
[Noisy rollout collection](NOISY_ROLLOUT_COLLECTION.md). This work depends on
[collector PR #10](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/10)
and the partner branch's actuated Shadow execution code. Before training, we
need to review the task's lift cutoff and what we want the model to predict.
Training splits, a predictor, and recovery remain future steps.

## Geometry source and what we use

The apple comes from the HUG / 1M-HuGs dataset's development asset
`val/medium_3/apple`, pinned to dataset revision
`56ea3f100b1a4ec38bb35e986c0dd14d38977283`. The local download manifest records
source URLs, sizes, file hashes, revision, and the dataset's CC BY 4.0 license.
Attribution belongs to [HUG / 1M-HuGs, published by kevinywu](https://huggingface.co/datasets/kevinywu/1m-hugs)
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This milestone uses only the MJCF geometry and its three referenced mesh files:

- `sim_assets/object.xml`
- `sim_assets/visual.obj`
- `sim_assets/collision/part_000.obj`
- `sim_assets/collision/part_001.obj`

The source provenance is recorded locally in
`runs/hug_benchmark_review_2026-10-09/download_manifest.json`. Raw assets, input
data, model checkpoints, and generated rollouts are not committed with this
change. Official benchmark grasp annotations remain unused. We render our own
RGB-D inputs, infer fresh HUG proposals, and label the resulting robot attempts.
This is not an official HUG-Bench reproduction or benchmark score.

The loader preserves the source's native mesh, geom, and inertial frames. It
imports both collision parts and the separate visual mesh; it does not replace
the apple with a box or reapply MuJoCo's compiled mesh-centering transforms.
The apple's declared native mass is 0.2304 kg. Use that mass for this milestone;
the existing cube remains 0.18 kg. No mass or friction augmentation is claimed.

## Declaring an object and binding its files

`ObjectCase.geometry` is an explicit `object_geometry_ref_v1` reference. Its
`mjcf_path` is project-relative and its `expected_content_hash` pins the complete
MJCF/mesh closure, not the XML alone. The hash is the canonical SHA-256 of a
mapping from relative asset filenames to their byte hashes. Directory location
does not change identity; changes to referenced bytes or relative asset names do.

For the inspected apple files, the geometry reference is:

```yaml
object:
  id: hug_apple
  mass_kg: 0.2304
  friction_profile_id: nominal_v1
  geometry:
    schema_version: object_geometry_ref_v1
    kind: mjcf_asset
    mjcf_path: runs/hug_benchmark_review_2026-10-09/apple/sim_assets/object.xml
    expected_content_hash: sha256:744ec02132a20129defe59048d1c3c6474e528ec276f717a5009ece8004d8c81
```

Add this object block to a portable evaluation case using
`adroit_shadow_tabletop_v2`, `shadow_hand_right`, and `fixed_grasp_lift_v2`.
The capture case declares a requested starting pose and a HUG grasp ID; the
preparation helper later fills the actual captured pose and generated proposal
paths. See [the evaluation manifest](EVALUATION_MANIFEST.md) for the remaining
case fields. Omitting `geometry` retains only the registered legacy cube path;
an arbitrary object ID cannot silently become a cube.

## Capture fresh inputs before generating proposals

Capture uses the declared geometry and pose, parks the hand away from the object,
and waits for sufficiently stationary object motion. Gravity may change the
requested pose during settling. The recorded position and orientation are the
actual settled pose, and prepared rollout cases start from that pose.

Capture must finish within 2,000 physics steps (4 s in this environment). Its
final one-second window must have a position-box diagonal at most 0.1 mm and a
conservative orientation diameter at most 0.002 rad, continuous table contact,
no hand contact, and object center inside the working table. Actual velocities
are saved and checked for finite values. Rollout reset uses the captured pose
and the protocol's standard zero initial velocities.

The apple's original upright setup rolled off the table and was rejected before
grasp generation. Inverted setups showed microscopic contact jitter that
exceeded the original instantaneous angular-speed threshold, despite staying
within the pose bounds. We therefore measure movement over a full second rather
than require 100 consecutive near-zero speed samples. The upright setup still
fails this check. No mesh, mass, friction, or in-run velocity edits were used.
The fresh apple templates declare a 180-degree rotation about X and initial
center height 0.04295394025198684 m, 0.1 mm above the rotated collision bounds;
the observed settled pose, not this requested height, enters each rollout.

The virtual camera renders registered 224-by-224 RGB, metric depth, and an object
mask covering the object's geoms. The capture records intrinsics and
`T_world_camera.npy` in the convention x right, y down, z forward. It checks
visible interior depth pixels against independent MuJoCo ray distances, with a
2 mm maximum allowed discrepancy. A valid selected object pixel and millimeter
depth encoding are checked before HUG preprocessing.

For a declared case, the individual capture CLI is:

```sh
MUJOCO_GL=cgl python -m grasp_failure_prediction.integrations.sim_observations \
  runs/multi_scene_inputs/cases/apple_a.yaml \
  --project-root . \
  --output runs/multi_scene_inputs/captures/apple_a \
  --camera-position 0.25 -0.35 0.35
```

The installed entry point is `capture-hug-scene`. On macOS, use the environment's
`mjpython` launcher if the renderer requires main-thread execution by passing
`--renderer-python /path/to/mjpython` to the preparation helper. By default the
helper uses the current Python interpreter; this milestone's offscreen CGL
capture worked with ordinary Python.

Each capture writes `scene_contract.json` using `source_scene_contract_v1`. It
binds the object ID, geometry closure hash, actual mass and settled pose,
environment configuration hash, camera hash, and all six RGB-D/calibration input
file hashes. The original contract is copied into each proposal directory.
Before labeling a rollout, the collector checks agreement between the proposal,
observation files, contract, and requested execution scene. A missing or
mismatched contract for new geometry is an error, not a binary grasp failure.
It also checks the original and copied observation-manifest bytes, selected
pixel/provenance, depth PNG hash, inference-report hash, and integer inference
seed against the declared seed. Saved metadata identifies the actual HUG code,
checkpoint, inference script, and runtime used for each proposal.
The contract is a content binding; it cannot establish that an external capture
was truthful.

## Prepare four scenes and sixteen fresh proposals

Create four capture-case YAML files: two for the existing cube and two for the
apple, each declaring its requested object pose. Then create a scene-spec JSON
file such as `runs/multi_scene_inputs/scenes.json`:

```json
[
  {"case_path": "runs/multi_scene_inputs/cases/cube_a.yaml", "camera_position_m": [0.25, -0.35, 0.35]},
  {"case_path": "runs/multi_scene_inputs/cases/cube_b.yaml", "camera_position_m": [0.30, -0.32, 0.38]},
  {"case_path": "runs/multi_scene_inputs/cases/apple_a.yaml", "camera_position_m": [0.25, -0.35, 0.35]},
  {"case_path": "runs/multi_scene_inputs/cases/apple_b.yaml", "camera_position_m": [0.30, -0.32, 0.38]}
]
```

The completed experiment requested these starting positions, in meters:

- Cube A: `[0, 0, 0.03]`; Cube B: `[-0.04, 0.025, 0.03]`, both with identity orientation.
- Apple A: `[0, 0, 0.04295394025198684]`; Apple B:
  `[0.04, -0.025, 0.04295394025198684]`, both with XYZW quaternion `[1, 0, 0, 0]`.

The settled poses in the capture reports, rather than these requested poses,
are the execution inputs. The original specifications are saved in
`runs/object_scene_pilot_2026-10-10/setup/scene_specs.json` and its case YAMLs.

Use the validated HUG CPU environment with its existing local HUG checkout,
MANO assets, and cached DINO/HUG weights. The repository's evaluation extras
alone do not replace that HUG runtime. Point `HF_HOME` at the existing cache to
keep model loading offline:

```sh
export HF_HOME="$PWD/runs/hug_integration_check/hf_cache"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=cgl \
  python scripts/prepare_scene_pilot.py \
  runs/multi_scene_inputs/scenes.json \
  --project-root . \
  --output runs/multi_scene_prepared \
  --hug-root runs/hug_integration_check/hug \
  --checkpoint runs/hug_cpu_audit/checkpoints/hug_full.safetensors \
  --proposal-seeds 0 1 2 3
```

The helper captures each scene once, copies its inputs into a fresh directory
for each seed, and runs real 50-step HUG CPU inference. It writes authored case
YAML files with the captured pose/mass and the generated proposal paths, then a
project-relative `cases.json` manifest. It does not select grasps using official
labels or optimize the grasp against object contacts. Different seeds and views
do not guarantee successful grasps; the usual retargeting and execution gates
still apply.

The completed run used two preparation batches: seeds 0–1 in `inputs_v3` and
seeds 2–3 in `inputs_v3_extra`. The eight capture directories contain four
byte-identical input identities across the batches. They therefore form four
observation groups, while the sixteen fresh proposals form sixteen proposal
groups. Repeating capture did not create extra independent scenes.

## Collect paired noise variants and inspect outcomes

The completed schedule used amplitudes 0, 0.15, and 0.6 rad, 32 repetitions per
proposal at each nonzero amplitude, one baseline per proposal, and master seed
9400. The saved run used `all_cases_v3.json`. After the preparation example
above, repeat the same schedule into a new output directory with its generated
manifest:

```sh
python -m grasp_failure_prediction.evaluation.noisy_rollouts \
  runs/multi_scene_prepared/cases.json \
  --project-root . \
  --output runs/object_scene_pilot_2026-10-10/collection_reproduction \
  --noise-amplitudes 0 0.15 0.6 \
  --repetitions 32 \
  --baseline-once \
  --seed 9400
```

`--baseline-once` retains one zero-noise execution per proposal instead of
repeating an identical deterministic baseline at every repetition. Noisy
repetitions still use distinct draws, paired across amplitudes: 16 × (1 + 32 +
32) = 1,040 attempts, with 512 paired noise directions.

Noise here means a requested random angle bias for each commanded finger joint,
sampled uniformly within ±0.15 or ±0.6 rad and held fixed for that episode after
approach. Wrist targets are unperturbed. The actual command is the nominal target
plus that bias, clipped to joint limits, then interpolated by the actuated
controller. The saved requested bias and actual applied offsets let us see what
the hand was commanded to do; the executed joint motion is saved separately.

At least one target was clipped in 508 of 512 trials at 0.15 rad and all 512 at
0.6 rad. These amplitudes describe requested disturbances, not an equally large
angle change at every joint. They are deliberate stress settings, not measured
hardware noise. Noise can improve a grasp as well as break one: all eight apple
baselines failed the task, while 108 noisy apple attempts succeeded.

The final labels remain `1 = task success`, `0 = completed task failure`, and
`null = validation/runtime error`. The same scorer requires the registered lift,
hold, final-drop, collision, and timeout conditions. These are task labels,
not secured-grip or physical-slip ground truth. `events.json` records
sampled lift/retention proxies, acquisition deadline intervals, and observed
height-loss bounds. Those bounds are not exact physical slip-onset labels.
Inspect `plan.json`, `index.jsonl`, `summary.json`, representative trajectories,
and capture/inference reports before scaling the collection.

The logged identities support later split decisions:

- `group_id`: one proposal and all of its noisy variants stay together.
- `observation_group_id`: proposals from one captured pose/camera/input scene
  stay together for an observation holdout.
- `geometry_group_id`: all scenes and proposals using the same geometry stay
  together for a geometry holdout, regardless of human-readable object IDs.

No splits are generated now. Four input families provide distinct simulation
inputs; two geometries remain too small to establish broad object generalization. Final
outcomes, event metadata, noise settings, and privileged simulator state remain
excluded from a future observable-input predictor.

## Fresh results and verification

The canonical raw experiment is `runs/object_scene_pilot_2026-10-10/` in the
original project checkout. Its important paths are:

- `all_cases_v3.json`: the sixteen prepared case references.
- `inputs_v3/` and `inputs_v3_extra/`: initial RGB-D/calibration files, capture
  contracts/reports, fresh HUG proposals, inference reports, and case YAMLs.
- `collection_1040/`: `plan.json`, `index.jsonl`, `summary.json`, and all 1,040
  episode directories with commands/noise, states, scores, events, and models.
- `collection_1040_audit.json`: the final checks and detailed outcome analysis.

We save initial RGB-D inputs for each scene, not camera frames at every timestep.
The rollout trajectories contain robot/object states, contacts, nominal and
perturbed commands, requested noise, and outcomes. The six demonstration videos
were rendered later from saved trajectories; they are not additional attempts.

The completed counts are:

- Overall: **444 successes, 596 failures, 0 errors** across 1,040 attempts.
- Cube: **336 successes / 520 attempts**; apple: **108 / 520**.
- Zero noise: 8 successes / 16; 0.15 rad: 294 / 512; 0.6 rad: 142 / 512.
- Failure categories: 590 `failed_acquisition`, 2 `no_contact`, and 4 `drop`.
  `no_contact` means no qualifying hand/object contact during the acquisition
  phases, rather than no physical contact at any time.

The two canonical small validation batches contain 48 attempts total: 18
successes, 30 failures, and no errors. Those checks are separate from the
1,040-attempt collection above and share its input families.

## What the labels mean before we train

The frozen task requires a peak lift of **at least 14 cm above the initial
object-center height**, two seconds in the hold phase at least 12 cm above that
initial height, and at most 2 cm peak-to-final height loss. Acquisition contact,
approach collision, and timeout checks also apply. Passing these conditions
produces label 1. Failing a condition produces label 0 and the scorer's first
applicable failure category.

The lift cutoff matters: **21 attempts lie within 0.5 mm of 14 cm** (14 successes
and 7 failures). Apple `episode_000465` stays held for two seconds but peaks only
**0.0308 mm below** the lift cutoff, so its stored task label is 0 with
`failed_acquisition`. This label does not establish an insecure grasp.

Failure-category precedence also matters. Thirty additional attempts lose more
than 2 cm from peak to final height but remain `failed_acquisition` because they
never reach the 14 cm lift cutoff; four attempts meet that cutoff and receive
`drop`. The category alone cannot identify every observed height-loss event or
its physical cause. Labels and outcomes remain unchanged. Before training, we
should review whether our target is this task outcome or a separately defined
grasp/retention outcome, and document any new label version explicitly.

## Verification and reproducibility

The final audit passed on collection source commit `99f70a6`. The full suite at
`301ca1f` passed **292 tests, with 5 skipped**; the subsequent renderer-only
change was verified with the six real replay videos. Capture checks kept the one-second pose
window within 0.1 mm / 0.002 rad; the largest independent ray/depth discrepancy
was **8.7665 × 10⁻⁶ m**, below the 2 mm limit. The upright apple was rejected
before proposal generation and has no grasp-outcome label.

All 1,040 task scores and temporal event files regenerate exactly. The audit
also verified command bounds/clipping, unperturbed wrist commands, phase/time
alignment, noise RNG records, all 512 paired amplitude directions, and all
sixteen proposal/capture provenance bindings. Seven representative complete
command sequences were reexecuted from reset with bitwise-identical states,
contact diagnostics, and forces. This verifies replay on the same runtime;
cross-platform bitwise identity has not been tested.

The effective runtime was Python 3.12.13, MuJoCo 3.3.7, NumPy 2.5.3,
dex-retargeting 0.5.0, and imported Pinocchio 3.8.0. HUG used 50-step CPU
inference from code commit `8d1c52d4c24bfae5a369e32e3f134f5601a02630` and
checkpoint SHA-256
`515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae`.
The exact script/input/report hashes and runtime provenance are saved with the
collection and audit. The original HUG asset attribution above still applies.

Storage deduplication preserved all 1,040 `scene.mjb` paths and their original
hashes while sharing two unique model files through hardlinks, saving
approximately **6.39 GB** of allocated model storage. Treat these shared model
bytes as immutable. The collection occupies roughly 420 MiB before the few
additional megabytes of videos. Adding video references did not change task
labels or outcomes.

## See the completed attempts

The local [collection viewer](http://127.0.0.1:8769/) serves all 1,040 results
and six representative videos. It runs on this machine and is not a hosted
public dataset. Each video contains 163 saved control frames played at 25 fps.

- `episode_000000`: successful cube baseline with zero noise.
- `episode_000533`: cube success at 0.15 rad.
- `episode_000534`: cube drop at 0.6 rad, paired with `episode_000533` using the
  same proposal and underlying noise direction.
- `episode_000467`: apple drop.
- `episode_000355`: apple missing qualifying acquisition contact.
- `episode_000465`: apple held below the task lift cutoff described above.

This completes the controlled collection milestone. The next decision is to
review the task cutoff and prediction target, then choose grouped training,
validation, and test sets. No splits, predictor training, recovery controller,
real-world transfer evaluation, or official benchmark score are completed yet.
