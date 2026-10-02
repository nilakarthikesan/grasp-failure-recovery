# HUG observation preflight

This step checks an RGB-D observation before loading HUG. It does not train a
policy, generate a grasp, or establish simulation success.

An observation contains an RGB image, registered metric depth, camera intrinsics
at that resolution, an object mask, and an explicit selected pixel `[u, v]`.
Depth is floating-point meters; zero marks invalid depth. Registration and
calibration declarations still require visual and source-level verification.

## Offline input checks

Run `python -m grasp_failure_prediction.integrations.observations
observation_manifest.json --output checks` (on one line).

The JSON manifest uses `rgb_path`, `depth_m_path`, `intrinsics_path`, `mask_path`,
`selection_uv`, `depth_units: "meters"`, `registered_to_rgb: true`,
`intrinsics_at_rgb_resolution: true`, and `source`. File paths are relative to
the manifest. Intrinsics and depth are NumPy `.npy` arrays; RGB and mask are images.
The command saves validation results, source hashes, and visual previews.

## Official preprocessing check

Use a separate HUG checkout from https://github.com/KevinyWu/hug at revision
`8d1c52d4c24bfae5a369e32e3f134f5601a02630`. Run:

```sh
python scripts/check_hug_preprocessing.py \
  --official-root /path/to/hug/src \
  --sample /path/to/hug/data/hug_bench/medium_1.pkl \
  --output runs/hug_preflight \
  --preprocessing-seed 42
```

Only use trusted upstream pickle files. This script checks the upstream sample
format, not arbitrary manifests. It calls the official dataset preprocessing with
a narrow point-cloud sampling override that supplies a request-local RNG. It
checks tensor shapes, finite values, projection geometry, the selected object
pixel, crop bounds, and repeatability. It saves tensors and a provenance report.
The seed controls preprocessing only, not future model sampling.

The official `medium_1.pkl` observation passed locally: RGB `3 x 224 x 224`,
point cloud `4096 x 3`, and repeatable tensor contents at seed 42. Unit tests
cover input validation and the sampling adapter without network calls.

## Remaining work

Load HUG and MANO, check its inference-specific input path, generate a proposal,
and execute it in simulation. Observations captured from our own simulation
still need to pass these checks. The offline manifest path and upstream sample
test are not yet a single end-to-end inference interface.

MANO model files, checkpoints, upstream downloads, and generated results stay
outside version control. Local MANO assets belong in `assets/mano_models/`;
retain the downloaded `models/` directory and license files.
