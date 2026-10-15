# HUG reserve geometry qualification v1

The reserve objects are `hug_bell_pepper` and `hug_bear_figure`. They are reserved
for later fresh evaluation after the prediction model is frozen. Asset inspection
and stationary camera qualification produce no grasp labels, and do not qualify
HUG grasps, mappings, controllers, model performance or physical grasp success.
No grasp inference or model fitting is permitted in this asset protocol.

The original geometry comes from the official
[1M-HuGs pinned dataset revision](https://huggingface.co/datasets/kevinywu/1m-hugs/tree/56ea3f100b1a4ec38bb35e986c0dd14d38977283/val/medium_5).
The [primary dataset metadata](https://huggingface.co/api/datasets/kevinywu/1m-hugs/revision/56ea3f100b1a4ec38bb35e986c0dd14d38977283)
declares `cc-by-4.0` at revision
`56ea3f100b1a4ec38bb35e986c0dd14d38977283`; its identifying fields are saved
with the protocol. Attribution is to the 1M-HuGs / HUG authors, with each original
pinned file URL retained in `download_manifest.json`. The download contains only
`sim_assets` geometry and native properties, without grasp annotations. Assets
remain in the original checkout's ignored `runs/hug_build_heldout_assets_2026-10-15_v1`
directory. This document records the dataset's license declaration; original
file attribution remains attached to the downloaded manifests.

`scripts/qualify_hug_reserve_assets.py prepare` verifies every downloaded file's
length, digest and original path, plus the exact MJCF dependency closure using
the existing `load_object_geometry`. It statically compiles the original and
existing source-accurate imported rigid asset. The native source mass and center
of mass must survive import. Reconstructed imported inertia must match the XML
source tensor within the existing absolute `1e-12 kg m^2` regression tolerance;
the raw native compiler's tensor residual and properties-file rounding residual
are also reported. Positive principal moments and the rigid-body triangle
condition are required. No inertias, masses, mesh frames or physics are tuned.

Native XML masses are 0.1806 kg for bell pepper and 0.1164 kg for bear figure.
Each object has exactly one declared initial orientation, the source identity
body frame `[0,0,0,1]` in XYZW order. Initial XY is zero and body Z is
`-local_collision_bounds_min_z + 0.005 m`, placing the lowest collision vertex
5 mm above the unchanged tabletop. The protocol stores the calculated heights,
collision bounds, all geometry hashes, registry configuration hashes, source
hashes, hand asset hashes and runtime versions.

The finite capture schedule has four calls: both objects with cameras at
`[0.25,-0.35,0.35]` and `[-0.25,-0.35,0.35]` metres. Each starts independently
from the same native orientation and existing reset. There are no retries or
automatic replacement orientations/objects. Failed calls retain their status,
error and traceback; their task outcome remains null.

Only after root commits this declaration may the explicit `run` command invoke
the existing, unmodified `capture_scene`. Before any settling, it verifies that
the supplied freeze commit contains the exact protocol, cases and source bytes,
and rechecks native asset, hand and runtime bindings. Existing gates remain:
500 trailing states, position box diagonal at most 0.1 mm, quaternion diameter
bound at most 0.002 rad, continuous object/table contact, no hand/object contact,
finite velocities, workspace abs XY below 0.3 m and at most 2,000 physics steps
per call. Captures remain 224 by 224, 45 degree field of view, with the existing
2 mm calibrated ray/depth gate. There are at most 8,000 settling steps total,
16 simulated seconds at the registered 2 ms step. A stationary rejection is an
asset/capture rejection, never a completed grasp failure.

Preparation and manufactured tests block `mj_step`, `mj_step1` and `mj_step2`.
The prepared protocol marks stationary capture `NOT_RUN`; an asset-only static
pass does not imply that the object will settle or render successfully. A later
stationary-only pass preserves the reserve restriction until a separate model
freeze and explicit fresh evaluation protocol exist.

Run from the canonical project-work checkout, using the original environment:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python scripts/qualify_hug_reserve_assets.py --runtime-project /Users/nilakarthikesan/grasp-failure-prediction prepare --asset-root /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_build_heldout_assets_2026-10-15_v1 --hands-root /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/dex-urdf/robots/hands --evidence-dir research_artifacts/hug_reserve_asset_qualification_2026-10-15_v1
```

The separate capture command is intentionally explicit and requires the commit
that freezes those bytes:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python scripts/qualify_hug_reserve_assets.py --runtime-project /Users/nilakarthikesan/grasp-failure-prediction run --protocol research_artifacts/hug_reserve_asset_qualification_2026-10-15_v1/protocol.json --freeze-commit FREEZE_COMMIT --output /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_reserve_capture_qualification_2026-10-15_v1
```
