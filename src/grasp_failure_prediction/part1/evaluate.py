"""Held-out closed-loop evaluation of a trained ACT policy.

Reports the Part I metrics from ``docs/PART_I_TRAINING_SPEC.md`` section 9:
per-phase reach, object-loss rate, collision / excess-force rates, and task
success — all measured from closed-loop rollouts on held-out starting
conditions (the validation seeds), not from training loss.

Results here are simulation-only.

This module evaluates the earlier Panda/ACT baseline only. It does not execute
or score the proposed HUG/Shadow Hand fixed-protocol study. Its outcome
definitions remain provisional; see docs/PART_I_TRAINING_SPEC.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from grasp_failure_prediction.part1.config import Part1Config, Phase, default_config
from grasp_failure_prediction.part1.lerobot_export import STATE_KEYS


@dataclass
class EvalResult:
    n_episodes: int = 0
    successes: int = 0
    object_losses: int = 0
    timeouts: int = 0
    collisions: int = 0
    excess_force: int = 0
    # Max task phase reached, excluding the terminal DONE marker.
    max_phase: List[int] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        n = max(self.n_episodes, 1)
        phases = np.array(self.max_phase) if self.max_phase else np.array([0])
        return {
            "episodes": self.n_episodes,
            "success_rate": self.successes / n,
            "object_loss_rate": self.object_losses / n,
            "timeout_rate": self.timeouts / n,
            "collision_rate": self.collisions / n,
            "excess_force_rate": self.excess_force / n,
            "reached_grasp_rate": float(np.mean(phases >= int(Phase.GRASP))),
            "reached_lift_rate": float(np.mean(phases >= int(Phase.LIFT))),
            "reached_place_rate": float(np.mean(phases >= int(Phase.PLACE))),
        }


def _obs_to_batch(obs, device):
    import torch

    def img(x):
        t = torch.from_numpy(np.asarray(x)).float() / 255.0  # HWC uint8 -> float
        return t.permute(2, 0, 1).unsqueeze(0).to(device)  # 1,C,H,W

    state = np.concatenate([np.asarray(obs[k]).ravel() for k in STATE_KEYS]).astype(
        np.float32
    )
    return {
        "observation.images.front": img(obs["rgb_front"]),
        "observation.images.wrist": img(obs["rgb_wrist"]),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
    }


def evaluate(
    policy_dir: str,
    seeds: Optional[List[int]] = None,
    config: Optional[Part1Config] = None,
    mass_kg: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> EvalResult:
    import torch

    from lerobot.policies.act.modeling_act import ACTPolicy

    from grasp_failure_prediction.part1.environment import WeightedContainerTask

    config = config or default_config()
    seeds = seeds if seeds is not None else config.split.val_seeds()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    policy = ACTPolicy.from_pretrained(policy_dir)
    policy.to(device)
    policy.eval()

    env = WeightedContainerTask(
        config=config, mass_kg=mass_kg, render_images=True, capture_snapshots=False
    )
    result = EvalResult()
    try:
        for seed in seeds:
            obs = env.reset(seed=seed)
            policy.reset()
            max_phase = 0  # excludes the terminal DONE marker
            lost = collided = forced = success = False
            for _ in range(max_steps or config.task.horizon):
                with torch.no_grad():
                    action = policy.select_action(_obs_to_batch(obs, device))
                action = action.squeeze(0).cpu().numpy()
                obs, step = env.step(action)
                if step.phase < int(Phase.DONE):
                    max_phase = max(max_phase, step.phase)
                collided = collided or step.collision
                forced = forced or step.excess_force
                lost = lost or step.object_lost
                if step.success:
                    success = True
                if step.success or step.failure:
                    break
            timeout = not success and not lost
            result.n_episodes += 1
            result.successes += int(success)
            result.object_losses += int(lost)
            result.timeouts += int(timeout)
            result.collisions += int(collided)
            result.excess_force += int(forced)
            result.max_phase.append(max_phase)
            print(
                f"  seed {seed}: success={success} max_phase={Phase(max_phase).label} "
                f"object_lost={lost} timeout={timeout} collision={collided}"
            )
    finally:
        env.close()
    return result


def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Closed-loop eval of a trained ACT policy.")
    parser.add_argument("policy_dir", help="directory saved by train_act")
    parser.add_argument("--n", type=int, default=None, help="limit number of val seeds")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--mass", type=float, default=None)
    args = parser.parse_args(argv)

    config = default_config()
    seeds = config.split.val_seeds()
    if args.n is not None:
        seeds = seeds[: args.n]
    result = evaluate(
        args.policy_dir, seeds=seeds, config=config, mass_kg=args.mass, max_steps=args.max_steps
    )
    print("closed-loop evaluation (simulation-only):")
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
