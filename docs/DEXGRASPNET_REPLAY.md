# DexGraspNet Replay

The first non-HUG provider is a saved-record replay from the official
DexGraspNet release. It is intentionally classified as `dataset_replay`; it is
not fresh checkpoint inference.

## Pinned source and license

- Repository: `https://github.com/PKU-EPIC/DexGraspNet`
- Commit: `bd1b13d7248729af117e1d46aaa6266b147a3c7b`
- Code, sample records, and object assets: CC BY-NC 4.0
- Initial object: `core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03`

Downloaded records and meshes remain under the ignored `data/` directory. The
repository stores their official hashes and attribution, not the licensed data.
The decoder opens the official object-array `.npy` only after its complete hash
matches the pinned catalog.

## Import and run

Clone the official source next to or outside this repository:

```bash
git clone https://github.com/PKU-EPIC/DexGraspNet.git ../DexGraspNet
git -C ../DexGraspNet checkout bd1b13d7248729af117e1d46aaa6266b147a3c7b
```

Import one bundled official record:

```bash
grasp-eval import-dexgraspnet \
  --source ../DexGraspNet \
  --record-index 0 \
  --output data/dexgraspnet/core_mug_record_0000
```

The command verifies the source commit and record, mesh, and hand-model hashes.
It copies the licensed inputs into ignored local storage, creates a
content-addressed scaled MuJoCo object, and writes a ready-to-run request and
experiment. Run it with:

```bash
grasp-eval run data/dexgraspnet/core_mug_record_0000/experiment.yaml --no-video
```

Omit `--no-video` only when an MP4 is needed. Video encoding is independent of
the physical result and is much slower than the headless rollout on the current
Mac.

## Translation evidence

The official record contains:

- palm translation `WRJTx`, `WRJTy`, `WRJTz` in meters;
- static-XYZ Euler angles `WRJRx`, `WRJRy`, `WRJRz`;
- 22 named articulated joints in radians; and
- the object mesh scale.

The DexGraspNet hand has a free palm and no articulated wrist. The canonical
hand has 24 joints, including `WRJ2` and `WRJ1`. Replay maps the official palm
pose directly into `T_object_palm`, sets the canonical wrist joints to zero,
and applies the registered name/sign mapping to the 22 finger joints.

This is a verified kinematic mapping rather than an exact asset identity. Across
the pinned official samples, the two released hand assets differ slightly in
the little-finger and thumb geometry. The first accepted record measures
0.429 mm mean and 1.722 mm maximum fingertip discrepancy. The registered gate
is 2 mm maximum. A proposal exceeding it fails translation.

## Initial ten-record gate

Records 0 through 9 were imported from the bundled official mug sample and run
through the unchanged `adroit_shadow_tabletop_v2` evaluator:

| Result | Count |
|---|---:|
| Completed: approach collision | 7 |
| Completed: failed acquisition | 1 |
| Rejected: canonical joint-limit violation | 2 |
| Completed success | 0 |

The dataset records are free-space grasps validated in Isaac Gym. They were not
generated for this project's fixed vertical tabletop approach, so the collision
rate is evidence of protocol incompatibility, not evidence that the coordinate
mapping is wrong. Records 5 and 8 exceed the canonical `THJ4` limit and remain
unlabeled pre-rollout rejections. The adapter does not clip or optimize them.

## GPU handoff

DexGrasp Anything fresh sampling remains the next milestone. It requires an
Ubuntu/NVIDIA environment, its official checkpoint, CUDA-specific PyTorch and
PyTorch3D, CSDF, and Isaac Gym. Its worker should emit the same native Shadow
record convention and pass through the mapping and FK gates documented here.
