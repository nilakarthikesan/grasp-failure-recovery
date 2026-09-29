"""Tests for canonical episode records and their HDF5 round-trip.

These use synthetic arrays (no simulator), so they run anywhere h5py is
available.
"""

import numpy as np
import pytest

from grasp_failure_prediction.part1.config import ObservationContract
from grasp_failure_prediction.part1.record import (
    EpisodeRecord,
    StepRecord,
    load_episodes,
    save_episodes,
)
from grasp_failure_prediction.part1.snapshot import SimSnapshot


def _make_step(i: int, with_snapshot: bool = False) -> StepRecord:
    obs = {
        "rgb_front": np.zeros((8, 8, 3), dtype=np.uint8),
        "rgb_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        "joint_pos": np.zeros(7, dtype=np.float32),
        "joint_vel": np.zeros(7, dtype=np.float32),
        "eef_pose": np.arange(7, dtype=np.float32),
        "gripper_state": np.zeros(2, dtype=np.float32),
    }
    privileged = {
        "object_pose": np.arange(7, dtype=np.float64),
        "object_mass": np.array([0.3]),
        "object_friction": np.array([1.0, 0.005, 1e-4]),
        "contacts": np.array([1.0, 2.0, 3.0, 0.0]),
    }
    snapshot = None
    if with_snapshot:
        snapshot = SimSnapshot(
            integration_state=np.arange(5, dtype=np.float64) + i,
            body_mass={25: 0.3},
            geom_friction={88: np.array([1.0, 0.005, 1e-4])},
        )
    return StepRecord(
        t=i * 0.05,
        step_index=i,
        phase=min(i, 5),
        obs=obs,
        action=np.full(7, i, dtype=np.float32),
        privileged=privileged,
        success=(i == 4),
        excess_force=(i == 2),
        snapshot=snapshot,
    )


def _make_episode(n: int = 5, with_snapshot: bool = False) -> EpisodeRecord:
    ep = EpisodeRecord(
        seed=1000,
        object_mass=0.3,
        object_friction=np.array([1.0, 0.005, 1e-4]),
        task_config={"lift_height": 0.04, "horizon": float(n)},
        terminal_success=True,
    )
    for i in range(n):
        ep.add(_make_step(i, with_snapshot))
    return ep


def test_episode_hdf5_roundtrip(tmp_path):
    ep = _make_episode(6)
    path = tmp_path / "demos.hdf5"
    save_episodes(path, [ep, _make_episode(4)])

    loaded = load_episodes(path)
    assert len(loaded) == 2
    first = loaded[0]
    assert first.length == 6
    assert first.seed == 1000
    assert first.object_mass == pytest.approx(0.3)
    assert first.terminal_success is True
    # Observations and actions preserved.
    assert np.allclose(first.steps[3].action, np.full(7, 3))
    assert first.steps[0].obs["rgb_front"].shape == (8, 8, 3)
    assert first.steps[2].excess_force is True
    assert first.steps[4].success is True
    # Task config attrs preserved.
    assert first.task_config["lift_height"] == pytest.approx(0.04)


def test_episode_snapshot_roundtrip(tmp_path):
    ep = _make_episode(4, with_snapshot=True)
    path = tmp_path / "demos_snap.hdf5"
    save_episodes(path, [ep])
    loaded = load_episodes(path)[0]
    snap = loaded.steps[2].snapshot
    assert snap is not None
    assert snap.body_mass[25] == pytest.approx(0.3)
    assert np.allclose(snap.geom_friction[88], [1.0, 0.005, 1e-4])
    assert np.allclose(snap.integration_state, np.arange(5) + 2)


def test_privileged_leak_detected_on_save(tmp_path):
    ep = _make_episode(3)
    # Inject a privileged key into a policy observation.
    ep.steps[1].obs["object_mass"] = np.array([0.3])
    with pytest.raises(ValueError):
        save_episodes(tmp_path / "leak.hdf5", [ep])


def test_snapshot_array_roundtrip():
    snap = SimSnapshot(
        integration_state=np.linspace(0, 1, 10),
        body_mass={3: 0.5, 7: 1.2},
        geom_friction={9: np.array([0.8, 0.005, 1e-4])},
    )
    restored = SimSnapshot.from_arrays(snap.to_arrays())
    assert np.allclose(restored.integration_state, snap.integration_state)
    assert restored.body_mass == snap.body_mass
    assert np.allclose(restored.geom_friction[9], snap.geom_friction[9])


def test_contract_default_keys_present():
    contract = ObservationContract()
    step = _make_step(0)
    # Every policy key in the record is an allowed contract key.
    assert set(step.obs.keys()) == set(contract.policy_keys)
