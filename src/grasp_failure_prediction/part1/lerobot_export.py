"""Export canonical episode records to a LeRobot dataset for ACT training.

The learned policy sees only the observation contract, so the exported dataset
contains exactly:

* ``observation.images.front`` / ``observation.images.wrist`` (84x84x3),
* ``observation.state`` = [joint_pos(7), joint_vel(7), eef_pose(7), gripper(2)],
* ``action`` (7).

Privileged metadata (mass, friction, contacts, object pose) is deliberately
**not** exported here; it lives in the HDF5 records for Parts II and III.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from grasp_failure_prediction.part1.record import EpisodeRecord, load_episodes

STATE_KEYS = ("joint_pos", "joint_vel", "eef_pose", "gripper_state")
STATE_DIM = 7 + 7 + 7 + 2
DEFAULT_TASK = "grasp the container, lift it, transport it, and place it in the target"


def state_vector(obs) -> np.ndarray:
    """Concatenate the proprioceptive policy observation into one state vector."""
    return np.concatenate([np.asarray(obs[k]).ravel() for k in STATE_KEYS]).astype(
        np.float32
    )


def _features(camera_h: int, camera_w: int) -> dict:
    return {
        "observation.images.front": {
            "dtype": "image",
            "shape": (camera_h, camera_w, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (camera_h, camera_w, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["state"],
        },
        "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    }


def export_episodes(
    episodes: List[EpisodeRecord],
    repo_id: str,
    root,
    fps: int = 20,
    task: str = DEFAULT_TASK,
    use_videos: bool = False,
):
    """Write episodes to a LeRobot dataset at ``root`` and return it."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not episodes:
        raise ValueError("no episodes to export")
    front0 = episodes[0].steps[0].obs["rgb_front"]
    h, w = front0.shape[:2]
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        features=_features(h, w),
        use_videos=use_videos,
    )
    for episode in episodes:
        for step in episode.steps:
            ds.add_frame(
                {
                    "observation.images.front": np.asarray(
                        step.obs["rgb_front"], dtype=np.uint8
                    ),
                    "observation.images.wrist": np.asarray(
                        step.obs["rgb_wrist"], dtype=np.uint8
                    ),
                    "observation.state": state_vector(step.obs),
                    "action": np.asarray(step.action, dtype=np.float32),
                    "task": task,
                }
            )
        ds.save_episode()
    if hasattr(ds, "finalize"):
        ds.finalize()
    return ds


def convert_hdf5(in_path, repo_id: str, root, fps: int = 20, use_videos: bool = False):
    """Load a RoboMimic-style HDF5 of episodes and export a LeRobot dataset."""
    episodes = load_episodes(in_path)
    return export_episodes(episodes, repo_id=repo_id, root=root, fps=fps, use_videos=use_videos)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export HDF5 demos to a LeRobot dataset.")
    parser.add_argument("in_path", help="input HDF5 (from collect-demos)")
    parser.add_argument("root", help="output LeRobot dataset root directory")
    parser.add_argument("--repo-id", default="grasp_failure_prediction/container_part1")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args(argv)

    if os.path.exists(args.root) and os.listdir(args.root):
        raise SystemExit(f"refusing to write into non-empty dir: {args.root}")
    ds = convert_hdf5(args.in_path, repo_id=args.repo_id, root=args.root, fps=args.fps)
    print(f"exported {ds.num_episodes} episodes, {ds.num_frames} frames -> {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
