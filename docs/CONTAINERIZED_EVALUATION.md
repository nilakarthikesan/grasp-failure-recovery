# Containerized evaluation runtime

`Dockerfile.eval` creates the pinned Linux x86-64 runtime for HUG inference,
Dex Retargeting, and MuJoCo evaluation. It pins HUG and Dex URDF revisions and
records the project commit as an OCI image label and runtime environment value.

The image does not contain the HUG checkpoint, MANO files, Hugging Face model
cache, evaluation inputs, or results. Those assets are mounted at runtime so
licensed files and large generated artifacts cannot enter the public image.

## Public build and saved-grasp smoke test

```bash
bash scripts/verify_container_image.sh grasp-eval:local
```

The smoke test imports the native runtime dependencies, checks the pinned HUG
source and Shadow Hand URDF, executes `hug_case_001` in MuJoCo, and requires a
completed result plus all four portable evaluation artifacts. The case may
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
CUDA runtime required for representative HUG inference.
