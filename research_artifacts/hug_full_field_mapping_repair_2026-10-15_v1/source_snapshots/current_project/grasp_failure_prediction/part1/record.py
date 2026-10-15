"""Canonical, synchronized episode records for Part I.

Every episode is logged with the fields defined in
``docs/PART_I_TRAINING_SPEC.md`` section 7. The same record feeds:

* Part I policy training (policy observations + actions),
* Part II future-failure labels (outcomes + histories + physics metadata), and
* Part III recovery branching (per-step restorable snapshots).

Records serialize to an HDF5 layout that mirrors RoboMimic's
``data/demo_*/{obs,actions,...}`` convention, so existing RoboMimic tooling can
read the demonstrations and our LeRobot exporter can consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from grasp_failure_prediction.part1.config import ObservationContract
from grasp_failure_prediction.part1.snapshot import SimSnapshot


@dataclass
class StepRecord:
    """One synchronized control step."""

    t: float
    step_index: int
    phase: int
    # Policy-visible observation (contract.policy_keys only).
    obs: Dict[str, np.ndarray]
    action: np.ndarray
    # Privileged metadata: labels/analysis only, never a policy input.
    privileged: Dict[str, np.ndarray]
    # Separate outcome flags (spec section 3).
    success: bool = False
    failure: bool = False
    # True object loss (dropped / left workspace), distinct from a horizon timeout.
    object_lost: bool = False
    collision: bool = False
    excess_force: bool = False
    controller_fault: bool = False
    # Optional restorable snapshot for Part III branching.
    snapshot: Optional[SimSnapshot] = None


@dataclass
class EpisodeRecord:
    """A complete episode with metadata and reproducibility fields."""

    seed: int
    object_mass: float
    object_friction: np.ndarray
    task_config: Dict[str, float]
    steps: List[StepRecord] = field(default_factory=list)
    terminal_success: bool = False
    terminal_failure: bool = False

    def add(self, step: StepRecord) -> None:
        self.steps.append(step)

    @property
    def length(self) -> int:
        return len(self.steps)

    def assert_no_privileged_leak(
        self, contract: Optional[ObservationContract] = None
    ) -> None:
        """Ensure no privileged key ever appears in the policy observation."""
        contract = contract or ObservationContract()
        for step in self.steps:
            contract.assert_no_leak(step.obs.keys())


def _stack(records: List[StepRecord], attr: str) -> Dict[str, np.ndarray]:
    """Stack a dict-valued attribute across steps into arrays."""
    if not records:
        return {}
    keys = getattr(records[0], attr).keys()
    return {
        key: np.stack([getattr(step, attr)[key] for step in records]) for key in keys
    }


def to_hdf5_group(group, episode: EpisodeRecord) -> None:
    """Write one episode into an open h5py group (e.g. ``data/demo_0``)."""
    episode.assert_no_privileged_leak()
    n = episode.length
    group.attrs["num_samples"] = n
    group.attrs["seed"] = episode.seed
    group.attrs["object_mass"] = episode.object_mass
    group.attrs["object_friction"] = np.asarray(episode.object_friction)
    group.attrs["terminal_success"] = episode.terminal_success
    group.attrs["terminal_failure"] = episode.terminal_failure
    for key, val in episode.task_config.items():
        group.attrs[f"task/{key}"] = val

    group.create_dataset("actions", data=np.stack([s.action for s in episode.steps]))
    group.create_dataset("t", data=np.array([s.t for s in episode.steps]))
    group.create_dataset("phase", data=np.array([s.phase for s in episode.steps]))
    for name in (
        "success",
        "failure",
        "object_lost",
        "collision",
        "excess_force",
        "controller_fault",
    ):
        group.create_dataset(
            name, data=np.array([getattr(s, name) for s in episode.steps], dtype=bool)
        )

    obs_grp = group.create_group("obs")
    for key, arr in _stack(episode.steps, "obs").items():
        obs_grp.create_dataset(key, data=arr)

    priv_grp = group.create_group("privileged")
    for key, arr in _stack(episode.steps, "privileged").items():
        priv_grp.create_dataset(key, data=arr)

    # Snapshots (optional). Stored as ragged per-step arrays under one group.
    if any(s.snapshot is not None for s in episode.steps):
        snap_grp = group.create_group("snapshots")
        for i, step in enumerate(episode.steps):
            if step.snapshot is None:
                continue
            step_grp = snap_grp.create_group(str(i))
            for key, arr in step.snapshot.to_arrays().items():
                step_grp.create_dataset(key, data=arr)


def from_hdf5_group(group) -> EpisodeRecord:
    """Read one episode back from an open h5py group."""
    task_config = {
        k.split("/", 1)[1]: float(v)
        for k, v in group.attrs.items()
        if isinstance(k, str) and k.startswith("task/")
    }
    episode = EpisodeRecord(
        seed=int(group.attrs["seed"]),
        object_mass=float(group.attrs["object_mass"]),
        object_friction=np.asarray(group.attrs["object_friction"]),
        task_config=task_config,
        terminal_success=bool(group.attrs["terminal_success"]),
        terminal_failure=bool(group.attrs["terminal_failure"]),
    )
    n = int(group.attrs["num_samples"])
    actions = group["actions"][:]
    t = group["t"][:]
    phase = group["phase"][:]
    flags = {
        name: group[name][:]
        for name in (
            "success",
            "failure",
            "object_lost",
            "collision",
            "excess_force",
            "controller_fault",
        )
    }
    obs = {k: group["obs"][k][:] for k in group["obs"].keys()}
    priv = {k: group["privileged"][k][:] for k in group["privileged"].keys()}
    snap_grp = group["snapshots"] if "snapshots" in group else None

    for i in range(n):
        snapshot = None
        if snap_grp is not None and str(i) in snap_grp:
            arrays = {k: snap_grp[str(i)][k][:] for k in snap_grp[str(i)].keys()}
            snapshot = SimSnapshot.from_arrays(arrays)
        episode.add(
            StepRecord(
                t=float(t[i]),
                step_index=i,
                phase=int(phase[i]),
                obs={k: obs[k][i] for k in obs},
                action=actions[i],
                privileged={k: priv[k][i] for k in priv},
                success=bool(flags["success"][i]),
                failure=bool(flags["failure"][i]),
                object_lost=bool(flags["object_lost"][i]),
                collision=bool(flags["collision"][i]),
                excess_force=bool(flags["excess_force"][i]),
                controller_fault=bool(flags["controller_fault"][i]),
                snapshot=snapshot,
            )
        )
    return episode


def save_episodes(path, episodes: List[EpisodeRecord]) -> None:
    """Write a list of episodes to a RoboMimic-style HDF5 file."""
    import h5py

    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        total = 0
        for i, ep in enumerate(episodes):
            to_hdf5_group(data.create_group(f"demo_{i}"), ep)
            total += ep.length
        data.attrs["total"] = total
        data.attrs["num_demos"] = len(episodes)


def load_episodes(path) -> List[EpisodeRecord]:
    """Read a RoboMimic-style HDF5 file written by :func:`save_episodes`."""
    import h5py

    episodes: List[EpisodeRecord] = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        demo_names = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
        for name in demo_names:
            episodes.append(from_hdf5_group(data[name]))
    return episodes
