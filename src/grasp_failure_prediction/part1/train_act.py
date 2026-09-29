"""Train an ACT policy on the exported LeRobot dataset.

ACT (Action Chunking Transformer) is the first reproducible Part I baseline.
This module builds an ``ACTConfig``, lets LeRobot's ``make_policy`` attach the
dataset's normalization statistics, and runs a straightforward training loop.

Two scales are supported:

* ``--smoke`` (default small config, few steps) — proves the full
  collect -> export -> train loop executes end to end on this machine.
* a real run (larger ``--dim-model``, ``--chunk-size``, and ``--steps``) that
  can train a competent policy given enough demonstrations and time.

On this Apple Silicon host training runs on MPS; there is no CUDA. The vision
backbone is initialised from scratch (``pretrained_backbone_weights=None``) so
no weights are downloaded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    dataset_root: str
    output_dir: str
    repo_id: str = "grasp_failure_prediction/container_part1"
    fps: int = 20
    steps: int = 200
    batch_size: int = 8
    chunk_size: int = 20
    dim_model: int = 256
    n_heads: int = 8
    dim_feedforward: int = 512
    n_encoder_layers: int = 2
    n_decoder_layers: int = 1
    latent_dim: int = 32
    lr: float = 1e-4
    log_every: int = 25
    seed: int = 0


def _select_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_policy(dataset, cfg: TrainConfig, device):
    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy

    act_cfg = ACTConfig(
        input_features={},
        output_features={},
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        n_obs_steps=1,
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.chunk_size,
        dim_model=cfg.dim_model,
        n_heads=cfg.n_heads,
        dim_feedforward=cfg.dim_feedforward,
        n_encoder_layers=cfg.n_encoder_layers,
        n_decoder_layers=cfg.n_decoder_layers,
        latent_dim=cfg.latent_dim,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        optimizer_lr=cfg.lr,
        device=str(device),
        push_to_hub=False,
    )
    # Unused enum import guard (keeps FeatureType referenced for clarity).
    _ = FeatureType
    return make_policy(act_cfg, ds_meta=dataset.meta)


def train(cfg: TrainConfig) -> str:
    import torch
    from torch.utils.data import DataLoader

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.manual_seed(cfg.seed)
    device = _select_device()
    print(f"training on device: {device}")

    delta_timestamps = {"action": [i / cfg.fps for i in range(cfg.chunk_size)]}
    dataset = LeRobotDataset(
        cfg.repo_id, root=cfg.dataset_root, delta_timestamps=delta_timestamps
    )
    print(f"dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    policy = build_policy(dataset, cfg, device)
    policy.train()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    step = 0
    running = 0.0
    done = False
    while not done:
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running += float(loss.item())
            step += 1
            if step % cfg.log_every == 0:
                print(f"  step {step:>5}/{cfg.steps}  loss={running / cfg.log_every:.4f}")
                running = 0.0
            if step >= cfg.steps:
                done = True
                break

    os.makedirs(cfg.output_dir, exist_ok=True)
    policy.save_pretrained(cfg.output_dir)
    print(f"saved policy -> {cfg.output_dir}")
    return cfg.output_dir


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train ACT on the container dataset.")
    parser.add_argument("dataset_root", help="LeRobot dataset root (from export)")
    parser.add_argument("output_dir", help="where to save the trained policy")
    parser.add_argument("--repo-id", default="grasp_failure_prediction/container_part1")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny model + few steps to validate the pipeline end to end",
    )
    args = parser.parse_args(argv)

    cfg = TrainConfig(dataset_root=args.dataset_root, output_dir=args.output_dir, repo_id=args.repo_id)
    if args.smoke:
        cfg.steps = 4
        cfg.batch_size = 2
        cfg.chunk_size = 8
        cfg.dim_model = 64
        cfg.n_heads = 4
        cfg.dim_feedforward = 128
        cfg.n_encoder_layers = 1
        cfg.latent_dim = 16
        cfg.log_every = 1
    if args.steps is not None:
        cfg.steps = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
