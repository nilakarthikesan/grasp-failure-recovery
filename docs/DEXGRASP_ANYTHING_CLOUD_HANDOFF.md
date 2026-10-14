# DexGrasp Anything Cloud Handoff

Fresh DexGrasp Anything inference starts at this handoff. Source and checkpoint
identity are pinned; no GPU installation was attempted on the Mac.

## Pinned artifacts

| Artifact | Identity |
|---|---|
| Official source | `https://github.com/4DVLab/DexGrasp-Anything` |
| Source commit | `86353d710ca71d0f547f0d8b15cc8ca959c78163` |
| Source license | MIT |
| Hugging Face repository | `GaussionZhong/Dexgraspanyting` |
| HF revision | `b24637ce366af4dd75cff723d2b853179155ec29` |
| Checkpoint | `DexGraspNet.pth` |
| Checkpoint size | `110490637` bytes |
| Checkpoint SHA-256 | `f7675d8eb93b3978b2c2d72ec9f54ead6b936d1dd307e07605daa0c02bcac14f` |

The source repository is MIT. The Hugging Face model repository does not
publish a separate checkpoint license or model card, so checkpoint
redistribution remains unverified. The checkpoint trained on DexGraspNet is
treated as research-only pending clarification; DexGraspNet is CC BY-NC 4.0.

## Worker acquisition

On the Ubuntu/NVIDIA worker:

```bash
git clone https://github.com/4DVLab/DexGrasp-Anything.git
git -C DexGrasp-Anything checkout 86353d710ca71d0f547f0d8b15cc8ca959c78163

python -m pip install huggingface-hub
hf download GaussionZhong/Dexgraspanyting DexGraspNet.pth \
  --revision b24637ce366af4dd75cff723d2b853179155ec29 \
  --local-dir DexGrasp-Anything/checkpoints
```

Copy `scripts/verify_dexgrasp_anything_assets.py` from this repository and run:

```bash
python scripts/verify_dexgrasp_anything_assets.py \
  --source ../DexGrasp-Anything \
  --checkpoint ../DexGrasp-Anything/checkpoints/DexGraspNet.pth
```

Do not load the PyTorch pickle checkpoint until this verification passes.

## Official runtime requirements

The release documents:

- Python 3.8;
- PyTorch 1.11.0 and torchvision 0.12.0 built for CUDA 11.3;
- PyTorch3D 0.7.2 built with CUDA;
- CSDF;
- the pinned packages in the official `requirements.txt`; and
- NVIDIA Isaac Gym Preview 4.

Isaac Gym must be obtained from NVIDIA under its own terms. Place
`IsaacGym_Preview_4_Package.tar.gz` in the location specified by the official
README, then follow its installation commands. It is not downloaded or
redistributed by this repository.

The official checkpoint must be arranged as an experiment directory because
`sample.py` searches `${exp_dir}/ckpts/model.pth`. A minimal layout is:

```text
worker_experiment/
└── ckpts/
    └── model.pth   # verified DexGraspNet.pth
```

Then the official smoke command is:

```bash
bash scripts/grasp_gen_ur/sample.sh /absolute/path/to/worker_experiment
```

## Integration after the worker starts

The first worker implementation should:

1. accept the hashed object mesh and point cloud from our provider request;
2. construct the DGA dataset input without changing object scale or frame;
3. run one deterministic sampling request and retain its configuration/log;
4. export the untouched native free-palm pose and 22 named joints;
5. pass that record through `dexgraspnet_shadow_to_shadow_hand_right_v1`;
6. enforce the 2 mm FK residual and canonical joint-limit gates; and
7. return `grasp_proposal_v1` through the existing HTTP provider contract.

The worker must not use DGA's Isaac Gym success result as this project's task
label. MuJoCo with `fixed_grasp_lift_v2` remains the common evaluator.
