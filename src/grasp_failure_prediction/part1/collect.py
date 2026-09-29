"""Collect scripted demonstrations of the weighted-container task.

Rolls the scripted privileged demonstrator through :class:`WeightedContainerTask`
and assembles canonical :class:`EpisodeRecord` logs. Episodes are saved to a
RoboMimic-style HDF5 file that the LeRobot exporter and RoboMimic tooling can
both read.

The learned policy never sees privileged information; the demonstrator does,
which is legitimate for a teacher.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from grasp_failure_prediction.part1.config import Part1Config, default_config
from grasp_failure_prediction.part1.record import EpisodeRecord, save_episodes
from grasp_failure_prediction.part1.scripted import ScriptedDemonstrator


def _task_config_dict(env) -> dict:
    task = env.config.task
    return {
        "lift_height": task.lift_height,
        "place_tolerance": task.place_tolerance,
        "control_freq": float(task.control_freq),
        "horizon": float(task.horizon),
        "transport_distance": env.transport_distance,
        "table_top": env.table_top,
        "target_x": float(env.target_xy[0]),
        "target_y": float(env.target_xy[1]),
    }


def run_episode(env, seed: int, max_steps: Optional[int] = None) -> EpisodeRecord:
    """Run one scripted demonstration and return its canonical record."""
    obs = env.reset(seed)
    demo = ScriptedDemonstrator(env.table_top, env.cube_half_z)
    episode = EpisodeRecord(
        seed=seed,
        object_mass=env.object_mass,
        object_friction=env.object_friction,
        task_config=_task_config_dict(env),
    )
    cube_pos = env.container_position()
    horizon = max_steps or env.config.task.horizon
    for _ in range(horizon):
        eef_pos = obs["eef_pose"][:3]
        action = demo.act(eef_pos, cube_pos, env.target_xy)
        obs, step = env.step(action)
        episode.add(step)
        cube_pos = step.privileged["object_pose"][:3]
        if step.success or step.failure:
            break
    last = episode.steps[-1]
    episode.terminal_success = bool(last.success)
    episode.terminal_failure = bool(last.failure)
    return episode


def collect(
    out_path,
    seeds: List[int],
    config: Optional[Part1Config] = None,
    mass_kg: Optional[float] = None,
    friction: Optional[np.ndarray] = None,
    render_images: bool = True,
    capture_snapshots: bool = True,
) -> List[EpisodeRecord]:
    """Collect one episode per seed and save them to ``out_path`` (HDF5)."""
    from grasp_failure_prediction.part1.environment import WeightedContainerTask

    config = config or default_config()
    env = WeightedContainerTask(
        config=config,
        mass_kg=mass_kg,
        friction=friction,
        render_images=render_images,
        capture_snapshots=capture_snapshots,
    )
    episodes: List[EpisodeRecord] = []
    try:
        for seed in seeds:
            episode = run_episode(env, seed)
            episodes.append(episode)
            print(
                f"  seed {seed}: {episode.length} steps, "
                f"success={episode.terminal_success}, failure={episode.terminal_failure}"
            )
    finally:
        env.close()
    save_episodes(out_path, episodes)
    n_success = sum(e.terminal_success for e in episodes)
    print(f"saved {len(episodes)} episodes ({n_success} successful) -> {out_path}")
    return episodes


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Collect scripted container demos.")
    parser.add_argument("out_path", help="output HDF5 path")
    parser.add_argument("--n", type=int, default=None, help="number of episodes (train split by default)")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--mass", type=float, default=None, help="container mass (kg)")
    parser.add_argument("--no-images", action="store_true", help="skip camera rendering")
    parser.add_argument("--no-snapshots", action="store_true", help="skip per-step snapshots")
    args = parser.parse_args(argv)

    config = default_config()
    seeds = config.split.train_seeds() if args.split == "train" else config.split.val_seeds()
    if args.n is not None:
        seeds = seeds[: args.n]
    collect(
        args.out_path,
        seeds=seeds,
        config=config,
        mass_kg=args.mass,
        render_images=not args.no_images,
        capture_snapshots=not args.no_snapshots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
