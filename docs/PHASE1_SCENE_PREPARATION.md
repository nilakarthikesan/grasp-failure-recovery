# Phase 1, Part 1: prepare a known-object scene

This step creates one hand-free MuJoCo scene and a matching RGB-D observation.
The user approved the cube and camera baseline on 28 October 2026. Work is on
`phase1-scene-preparation`; the source and prior research results remain separate.

## What we built

The [configuration](../configs/scene_preparation_cube.json) declares a 5 cm cube,
100 g mass and sliding friction 1.0 on a flat table. Gravity lets the cube settle
before capture. These are test assumptions, not measured real-object properties.

The camera sits at (0.25, -0.35, 0.35) m and looks at the actual settled cube center.
It renders 224 by 224 pixels with a 45 degree vertical field of view. The table
top is world z=0; the scene uses meters, kilograms and seconds.

The remaining values are explicit engineering defaults in the same file: a
1 m square table, 8 cm initial cube-center height, 2 ms simulation steps,
MuJoCo contact/solver settings, lighting and numerical validation tolerances.
Mass gives the uniform cube its analytical inertia. Nothing is fitted to a real
image or changed to make a grasp succeed. Velocities are never forced to zero
after settling. The capture retains the actual measured object pose.

The [implementation](../src/grasp_failure_prediction/integrations/scene_preparation.py)
uses MuJoCo, NumPy, Pillow and the existing observation-preflight helpers. It
does not load a hand, the HUG model, a retargeter or a training pipeline.

## Run it

From the Phase 1 checkout, with the existing local Python runtime:

```bash
MUJOCO_GL=cgl PYTHONPATH=src \
  /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  -m grasp_failure_prediction.integrations.scene_preparation \
  --config configs/scene_preparation_cube.json \
  --output /Users/nilakarthikesan/grasp-failure-prediction/runs/phase1_scene_new
```

Choose a new output directory each time. Existing directories are rejected
without overwriting their contents. `cgl` is the macOS rendering backend; other
platforms need an appropriate installed graphics backend.

The completed local capture is at
`/Users/nilakarthikesan/grasp-failure-prediction/runs/phase1_scene_preparation_2026-10-28_v1`.
A [small evidence bundle](../research_artifacts/phase1_scene_preparation_2026-10-28_v1/README.md)
contains the preview, report and source/configuration hashes.

## What the output files mean

- `scene.xml` is the editable, self-contained object/table/camera model. It
  includes a `capture` keyframe; there are no external mesh or licensed assets.
- `scene_state.npz` stores the complete MuJoCo integration state, including the
  measured velocities and solver warm-start state. `scene_state_metadata.json`
  binds it to the XML/configuration and the MuJoCo version.
- `rgb.png` is the color image; `object_mask.png` identifies the cube pixels.
- `depth_m.npy` stores optical-axis depth in meters. `depth.png` is its rounded
  uint16 millimeter encoding for the HUG input convention; zero means invalid.
- `intrinsics.npy` maps camera points to image pixels. `T_world_camera.npy`
  maps camera coordinates to world coordinates. The camera axes are right,
  down and forward; the convention change from MuJoCo is explicit.
- `observation_manifest.json` binds the image inputs and selected object pixel
  to the synthetic scene provenance. `scene_config.json` saves every setting.
- `validation_report.json` and `.txt` report measurements and scope. The
  `observation_checks` directory provides a mask overlay and depth preview.

For exact same-runtime restoration, use the supplied helper:

```python
from grasp_failure_prediction.integrations.scene_preparation import reload_saved_scene

model, data = reload_saved_scene("/path/to/capture")
```

Loading XML alone starts from its initial pose unless the capture keyframe is
selected; use the helper for the full saved integration state.

## How we tested this piece

The [focused tests](../tests/test_scene_preparation.py) check the compiled cube's
dimensions, mass, analytical inertia and contacts; gravity settling; known camera
projections; independent NumPy ray/box intersections and MuJoCo ray casts;
RGB-D export and complete state reload; malformed settings; insufficient
settling; existing-output protection; and rejection of changed saved artifacts.
The existing observation-preflight tests also pass.

```bash
MUJOCO_GL=cgl PYTHONPATH=src \
  /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  -m pytest -q tests/test_scene_preparation.py tests/test_observations.py
```

The final focused run passed **41 tests**. In the saved capture, the cube settled
after 727 steps (1.454 simulated seconds). Maximum cube/table surface-depth
discrepancies were 7.12 and 10.25 micrometers across 128 interior pixels each.
The engineering tolerance is 20 micrometers; this is not a physical calibration
claim. Full-state, RGB, depth and segmentation reloads matched exactly on the
recorded MuJoCo 3.3.7/macOS runtime.

MuJoCo's compliant contact permits about 0.216 mm support-plane penetration in
this setup; the measured pose and penetration are recorded, not silently fixed.
The macOS renderer warns that `ARB_clip_control` is unavailable. Depth remains
within the independently checked engineering tolerance on this machine.
Cross-platform or driver-independent bitwise identity is not established.

## Boundary of this milestone

This qualifies a synthetic scene-preparation fixture. It does not reconstruct
a real scene, validate a HUG proposal, retarget Wuji, label grasp success/failure
or train a predictor. Part 2 will consume these explicit image/depth/calibration
inputs; its technical choices remain a separate discussion with the user.
