#!/usr/bin/env python
"""Run the full Part I pipeline: collect -> export -> train ACT -> evaluate.

Smoke mode (default) uses a handful of demonstrations, a tiny ACT, and a few
optimizer steps to prove the pipeline runs end to end on this machine. A real
run raises the counts and model size.

Usage:
    MUJOCO_GL=cgl python scripts/run_part1_pipeline.py --workdir runs/part1_smoke
    MUJOCO_GL=cgl python scripts/run_part1_pipeline.py --full \
        --n-train 40 --n-val 10 --steps 20000 --workdir runs/part1_full
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "cgl")

    parser = argparse.ArgumentParser(description="Run the Part I pipeline end to end.")
    parser.add_argument("--workdir", default="runs/part1_smoke")
    parser.add_argument("--n-train", type=int, default=6)
    parser.add_argument("--n-val", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--full", action="store_true", help="use a real (non-smoke) ACT config")
    parser.add_argument("--mass", type=float, default=None)
    args = parser.parse_args()

    from grasp_failure_prediction.part1.collect import collect
    from grasp_failure_prediction.part1.config import default_config
    from grasp_failure_prediction.part1.lerobot_export import convert_hdf5
    from grasp_failure_prediction.part1.train_act import TrainConfig, train

    config = default_config()
    os.makedirs(args.workdir, exist_ok=True)
    hdf5_path = os.path.join(args.workdir, "demos.hdf5")
    ds_root = os.path.join(args.workdir, "lerobot")
    policy_dir = os.path.join(args.workdir, "policy")

    print("=" * 60)
    print("[1/4] collecting demonstrations")
    train_seeds = config.split.train_seeds()[: args.n_train]
    collect(hdf5_path, seeds=train_seeds, config=config, mass_kg=args.mass,
            render_images=True, capture_snapshots=False)

    print("=" * 60)
    print("[2/4] exporting to LeRobot format")
    ds = convert_hdf5(hdf5_path, repo_id="grasp_failure_prediction/container_part1", root=ds_root)
    print(f"  dataset: {ds.num_episodes} episodes, {ds.num_frames} frames")

    print("=" * 60)
    print("[3/4] training ACT")
    cfg = TrainConfig(dataset_root=ds_root, output_dir=policy_dir, steps=args.steps)
    if not args.full:
        cfg.steps = args.steps
        cfg.batch_size = 2
        cfg.chunk_size = 8
        cfg.dim_model = 64
        cfg.n_heads = 4
        cfg.dim_feedforward = 128
        cfg.n_encoder_layers = 1
        cfg.latent_dim = 16
        cfg.log_every = 1
    train(cfg)

    print("=" * 60)
    print("[4/4] closed-loop evaluation (held-out seeds, simulation-only)")
    try:
        from grasp_failure_prediction.part1.evaluate import evaluate
    except ImportError:
        print(
            "  evaluation metrics are under review in the design-review PR; "
            "skipping. (Trained policy saved at "
            f"{policy_dir}.)"
        )
        return 0
    val_seeds = config.split.val_seeds()[: args.n_val]
    result = evaluate(policy_dir, seeds=val_seeds, config=config, mass_kg=args.mass)
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
