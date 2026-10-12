# Containerized evaluation runtime

**Optional runtime preserved during consolidation on 11 October 2026.** Docker
is unavailable on the current Mac. Linux image build and public smoke results
are recorded in the [container workflow](https://github.com/nilakarthikesan/grasp-failure-recovery/actions/workflows/container-eval.yml).
The connected inference script currently runs on CPU; a CUDA base image does
not establish that inference uses the GPU.

The 12 October Linux rerun built the image but failed while importing Pinocchio
because `libgomp.so.1` was missing. `libgomp1` is now included in the image's
Ubuntu packages. A subsequent Linux rebuild and runtime smoke test are pending;
the corrected mug walkthrough used the existing native Mac runtime.

`Dockerfile.eval` creates the pinned Linux x86-64 runtime for HUG inference,
Dex Retargeting, and MuJoCo evaluation. It pins HUG and Dex URDF revisions and
records the project commit as an OCI image label and runtime environment value.

The image excludes the HUG checkpoint, MANO files, Hugging Face model cache,
recorded datasets, demo videos and rollout results. Runtime assets are mounted
separately. Case templates and the synthetic smoke fixture remain part of the
image; local assets and saved prediction pickles are excluded from its context.

## Public build and saved-grasp smoke test

```bash
bash scripts/verify_container_image.sh grasp-eval:local
```

The smoke test imports the native runtime dependencies, checks the pinned HUG
source and Shadow Hand URDF, executes `hug_case_001` in MuJoCo, and requires a
completed result plus all four portable evaluation artifacts. It explicitly
uses `--no-inference --no-video`: the synthetic saved grasp checks execution
without licensed model assets or a GPU rendering context. This public check
does not validate HUG inference or video rendering. Child failures include their
stdout and stderr in the CI log. The case may
score as a grasp failure; that is a valid model outcome. The smoke test fails
only when the runtime, runner, or artifact contract fails.

## Licensed-asset gate

Download the released HUG checkpoint and obtain MANO through its official
licensed distribution. Then run:

```bash
docker run --rm --gpus all \
  -e MUJOCO_GL=egl \
  -v "$PWD/local-assets/mano_models:/opt/hug/assets/mano_models:ro" \
  -v "$PWD/local-assets/hug_full.safetensors:/models/hug_full.safetensors:ro" \
  -v "$PWD/local-assets/huggingface:/models/huggingface" \
  grasp-eval:local \
  python scripts/container_smoke_test.py --require-hug-assets
```

This additionally checks the published HUG checkpoint SHA-256, requires
`models/MANO_RIGHT.pkl` under the mounted MANO asset root (with the legacy
flat layout also accepted), hashes the local MANO file for provenance, and
imports HUG's dataset, inference, and MANO modules. DINOv2 downloads into the
mounted Hugging Face cache on first model load.

## Connected inference

Mount a captured observation directory at `/runs` and invoke the existing
pilot script:

```bash
docker run --rm --gpus all \
  -e MUJOCO_GL=egl \
  -v "$PWD/local-assets/mano_models:/opt/hug/assets/mano_models:ro" \
  -v "$PWD/local-assets/hug_full.safetensors:/models/hug_full.safetensors:ro" \
  -v "$PWD/local-assets/huggingface:/models/huggingface" \
  -v "$PWD/runs:/runs" \
  grasp-eval:local \
  python scripts/infer_sim_observation.py \
    --hug-root /opt/hug \
    --observation /runs/cube_observation \
    --checkpoint /models/hug_full.safetensors \
    --seed 42 \
    --steps 50
```

Use the immutable image digest, rather than a mutable tag, in the evaluation
registry. Each result should retain that digest together with `environment_id`,
the environment configuration hash, code commit, HUG commit, checkpoint hash,
MuJoCo version, and execution protocol hash.

## Platform notes

The image targets Linux x86-64. Run it directly on an NVIDIA Linux host or
through Docker Desktop with the WSL2 NVIDIA backend on Windows. Apple Silicon
can emulate the image for limited CPU checks, but it cannot provide the NVIDIA
GPU execution. CPU inference and image build compatibility must be checked on
the chosen Docker host.
