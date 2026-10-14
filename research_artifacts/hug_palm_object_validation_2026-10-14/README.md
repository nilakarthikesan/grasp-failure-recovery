# HUG wrist/palm and existing-object checks — 14 October 2026

This bundle records static diagnostics on **previously saved** HUG outputs.
It generates no new grasp proposal, simulation rollout, evaluation episode,
task label, training dataset or model fit. HUG remains the active provider.

Read [the project explanation](../../docs/HUG_MAPPING_CONFIRMATION.md) first.
[Palm evidence](palm/README.md) covers all 32 existing poses; [object-surface
evidence](surfaces/README.md) covers both saved seed-0 observation groups for
apple, milk box and mug. Neither set is an independent holdout. All have one
MANO shape. The original nominal target is the saved closing endpoint before
force-close; the controller's subsequent changes remain separate.

![Anchor and residual diagnostic](hug_anchor_diagnostic_2026-10-14.png)

The native wrist equals the selected Shadow palm origin numerically. Shadow's
separate wrist body is 34 mm away by its URDF. Nonthumb finger-base proxies
support this placement convention more than matching body names does, but
they are approximate anatomical correspondences. A shared translation removes
only 14.61% of fingertip squared residuals; the MCP-proxy mean worsens tip
residuals. These are algebraic decompositions on the same saved cases, not an
applied calibration or a validated improvement.

The surface evidence keeps corrected per-geom collision-union queries (v3)
and actual visible-mesh triangle queries (v4) separate. Both show changed
native/target surface proximity. Shadow unsigned distances use individual
collider surfaces, including buried interfaces at penetrated sites; they are
not exact exterior hand-union distance. Occupancy and exterior matched normals
are reported separately. None is physical contact area or a pass threshold.

![Existing native and nominal Shadow poses](surfaces/hug_other_objects_native_shadow.png)

Gray objects in this figure are individual active convex collision hulls, not
an exact clipped union mesh. Each pair shares camera, axes, limits and scale.

The exposed collision query surface also differs from the actual visual
surface. Reverse-distance measurements establish that distinction, but do not
identify the sole cause or demonstrate task-outcome sensitivity. Investigate
the object's collision approximation before interpreting this difference as
human-to-robot retargeting error. No object or hand geometry was changed.

The v1 plan was unexecuted. The preserved v2 pilot retained some exactly
abutting internal interfaces and is superseded for collision metrics. V3
removes them with a declared outward probe; v4 adds a separate metric family.
Historical published cube reports remain unchanged. Protocol and exact
executed script hashes are retained. Optional query arrays and caches remain
local diagnostic derivatives and are excluded from this publication; see the
surface manifest for their inventory. No licensed MANO asset is copied.

Reproduction requires the preserved runtime and local source/input files:
follow each subfolder README and use temporary diagnostic output directories.
The anchor plot script reads the recorded temporary palm report and writes
its figure to /private/tmp. It does not run inference or dynamics.

## Compute recommendation, not a provisioned experiment

A GPU rental could accelerate future generator inference/refinement when
authorized. DGA's [official installation](https://github.com/4DVLab/DexGrasp-Anything/blob/main/README.md)
requires its GPU/Isaac Gym environment; UniDexGrasp's
[generation setup](https://github.com/PKU-EPIC/UniDexGrasp/blob/main/dexgrasp_generation/README.md#installation)
uses CUDA/PyTorch3D. Runtime compatibility and VRAM requirements need measurement.

The current evaluator uses CPU mujoco.mj_step, and the current 288-feature
predictor uses NumPy/SciPy logistic regression. Renting a GPU alone does not
change either execution path. GPU simulation needs a separately validated
batched backend; [MuJoCo Warp guidance](https://mujoco.readthedocs.io/en/latest/mjwarp/#when-to-use-mjwarp)
distinguishes throughput from single-world latency and documents compatibility
and numerical differences.

After mapping validation and authorization for new collection, benchmark
successful proposals/minute, refinement time, peak VRAM and CPU rollout time
before choosing rental hardware. More independent objects, observations and
grasp families matter more than repeated episode count alone. More samples
do not guarantee improved generalization. No compute was rented or provisioned.
