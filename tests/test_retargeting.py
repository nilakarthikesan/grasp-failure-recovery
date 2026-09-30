from __future__ import annotations

import numpy as np
import pytest

from grasp_failure_prediction.evaluation.retargeting import (
    ShadowHandRetargeter,
    camera_landmarks_to_wrist,
    reorder_for_mujoco,
)
from grasp_failure_prediction.integrations.hug import normalize_hug_prediction


def synthetic_landmarks() -> np.ndarray:
    """A metric right-hand skeleton in local MANO wrist coordinates."""

    points = np.zeros((21, 3), dtype=np.float64)
    bases = {
        1: (-0.030, -0.015, 0.010),
        5: (-0.020, 0.015, 0.000),
        9: (0.000, 0.020, 0.000),
        13: (0.020, 0.015, 0.000),
        17: (0.038, 0.005, 0.000),
    }
    lengths = {1: 0.025, 5: 0.030, 9: 0.034, 13: 0.031, 17: 0.026}
    for base_index, base in bases.items():
        points[base_index] = base
        for offset in range(1, 4):
            direction = np.array([-0.7, 0.7, 0.1]) if base_index == 1 else np.array([0, 1, 0])
            points[base_index + offset] = points[base_index] + direction * lengths[base_index] * offset
    return points


def synthetic_hug_prediction():
    local = synthetic_landmarks()
    angle = np.pi / 3
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    translation = np.array([0.2, -0.1, 0.7])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    camera = local @ rotation.T + translation
    return normalize_hug_prediction(
        {
            "pose": np.zeros((15, 3)),
            "shape": np.zeros(10),
            "t": translation,
            "T_camera_wrist": transform,
            "landmarks_3d": camera,
            "mesh_vertices": np.zeros((778, 3)),
        }
    ), local


def test_camera_to_wrist_recovers_hug_local_landmarks() -> None:
    grasp, expected = synthetic_hug_prediction()
    actual = camera_landmarks_to_wrist(grasp.landmarks_3d, grasp.T_camera_wrist)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_real_dex_shadow_retargeting_is_finite_and_within_limits() -> None:
    grasp, _ = synthetic_hug_prediction()
    retargeter = ShadowHandRetargeter()
    pose = retargeter.retarget(grasp)
    assert len(pose.joint_names) == 24
    assert pose.qpos.shape == (24,)
    assert np.all(np.isfinite(pose.qpos))
    limits = retargeter.joint_limits
    assert np.all(pose.qpos >= limits[:, 0] - 1e-6)
    assert np.all(pose.qpos <= limits[:, 1] + 1e-6)


def test_real_adroit_joint_set_is_fully_covered_by_name_mapping() -> None:
    gym = pytest.importorskip("gymnasium")
    gymnasium_robotics = pytest.importorskip("gymnasium_robotics")
    mujoco = pytest.importorskip("mujoco")
    gym.register_envs(gymnasium_robotics)
    env = gym.make("AdroitHandRelocate-v1")
    try:
        env.reset(seed=0)
        model = env.unwrapped.model
        target_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(6, 30)
        ]
        retargeter = ShadowHandRetargeter()
        mapped = reorder_for_mujoco(
            retargeter.joint_names, np.arange(24, dtype=float), target_names
        )
        assert mapped.shape == (24,)
        assert set(mapped) == set(np.arange(24, dtype=float))
    finally:
        env.close()
