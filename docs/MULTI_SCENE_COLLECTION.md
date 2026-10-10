# Next milestone: fresh grasps across two geometries and four scenes

The next collection milestone replaces the single-observation cube pilot with
fresh observations of the existing cube and a HUG apple geometry. The planned
small batch has four declared object-position/camera configurations and two new
HUG proposal seeds per configuration: eight proposals before noise variants.
These are controlled simulation scenes, not independent real-world recordings.

The fixed controller, bounded finger-target noise, final scorer, and conservative
event annotations remain those documented in
[Noisy rollout collection](NOISY_ROLLOUT_COLLECTION.md). This work depends on
[collector PR #10](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/10)
and the partner branch's actuated Shadow execution code. It does not yet create
training splits, train a predictor, or add recovery.

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
  id: hug_apple_medium3
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
The contract is a content binding; it cannot establish that an external capture
was truthful.

## Prepare the four scenes and eight fresh proposals

Create four capture-case YAML files: two for the existing cube and two for the
apple, each declaring its requested object pose. Then create a scene-spec JSON
file such as `runs/multi_scene_inputs/scenes.json`:

```json
[
  {"case_path": "runs/multi_scene_inputs/cases/cube_a.yaml", "camera_position_m": [0.25, -0.35, 0.35]},
  {"case_path": "runs/multi_scene_inputs/cases/cube_b.yaml", "camera_position_m": [0.30, -0.30, 0.40]},
  {"case_path": "runs/multi_scene_inputs/cases/apple_a.yaml", "camera_position_m": [0.25, -0.35, 0.35]},
  {"case_path": "runs/multi_scene_inputs/cases/apple_b.yaml", "camera_position_m": [0.30, -0.30, 0.40]}
]
```

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
  --proposal-seeds 0 1
```

The helper captures each scene once, copies its inputs into a fresh directory
for each seed, and runs real 50-step HUG CPU inference. It writes authored case
YAML files with the captured pose/mass and the generated proposal paths, then a
project-relative `cases.json` manifest. It does not select grasps using official
labels or optimize the grasp against object contacts. Different seeds and views
do not guarantee successful grasps; the usual retargeting and execution gates
still apply.

## Collect paired noise variants and inspect outcomes

An example collection schedule is:

```sh
python -m grasp_failure_prediction.evaluation.noisy_rollouts \
  runs/multi_scene_prepared/cases.json \
  --project-root . \
  --output runs/multi_scene_rollouts \
  --noise-amplitudes 0 0.15 0.3 0.6 \
  --repetitions 2 \
  --baseline-once \
  --seed 9000
```

`--baseline-once` retains one zero-noise execution per proposal instead of
repeating an identical deterministic baseline at every repetition. Noisy
repetitions still use distinct draws, paired across amplitudes. The 0.3 and
0.6 rad settings are stress levels, not measured hardware noise.

The final labels remain `1 = success`, `0 = completed physical failure`, and
`null = validation/runtime error`. The same scorer requires the registered lift,
hold, final-drop, collision, and timeout conditions. `events.json` records
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

No splits are generated now. Four captures provide distinct simulation inputs;
two geometries remain too small to establish broad object generalization. Final
outcomes, event metadata, noise settings, and privileged simulator state remain
excluded from a future observable-input predictor.

## Fresh results and verification

Pending execution of this milestone. Record the exact four case/scene specs,
geometry closure hashes and source provenance, effective runtime versions,
capture ray errors and settled poses, proposal seeds and inference hashes,
collection schedule, success/failure/error counts, and any rejected cases.
Record the source commit and the tests run against this implementation. Keep
capture/preparation failures separate from attempted physical grasps. Do not
reuse earlier cube counts or test results as verification of this expansion.
