from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from grasp_failure_prediction.evaluation.retargeting import (
    RetargetingError,
    LEGACY_RETARGETING_PROFILE,
    STATIC_RETARGETING_PROFILE,
    ShadowHandRetargeter,
    camera_landmarks_to_wrist,
    reorder_for_mujoco,
    reference_vectors,
)
from grasp_failure_prediction.integrations.hug import normalize_hug_prediction
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT


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


def test_offline_retargeting_converts_mano_axes_without_filter_or_scaling() -> None:
    grasp, local_mano = synthetic_hug_prediction()
    retargeter = ShadowHandRetargeter()
    pose = retargeter.retarget(grasp)

    expected_landmarks = local_mano @ MANO_TO_OPERATOR_RIGHT.T
    np.testing.assert_allclose(pose.landmarks_wrist_m, expected_landmarks, atol=1e-8)
    assert retargeter._retargeting.filter is None
    assert retargeter._retargeting.optimizer.scaling == 1.0
    assert retargeter.optimizer_iterations == 20


def test_stock_adroit_joint_convention_is_rejected() -> None:
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
        with pytest.raises(RetargetingError, match="MuJoCo joints missing"):
            reorder_for_mujoco(
                retargeter.joint_names, np.arange(24, dtype=float), target_names
            )
    finally:
        env.close()


def test_unknown_retargeting_profile_is_rejected_before_runtime_loading() -> None:
    with pytest.raises(RetargetingError, match="unknown retargeting profile"):
        ShadowHandRetargeter(profile="not-a-profile")


def test_default_profile_preserves_historical_solver_settings_and_reports_them() -> None:
    grasp, _ = synthetic_hug_prediction()
    retargeter = ShadowHandRetargeter()
    pose = retargeter.retarget(grasp)
    evidence = pose.retargeting_provenance
    assert evidence.profile_id == LEGACY_RETARGETING_PROFILE
    assert evidence.temporal_norm_delta == 4e-3
    assert not evidence.reset_before_proposal
    assert evidence.config_sha256 == hashlib.sha256(retargeter.config_path.read_bytes()).hexdigest()
    assert evidence.vector_scaling == 1.0
    assert not evidence.low_pass_filter_enabled
    assert evidence.refinement_calls == 20
    assert evidence.optimizer_status_codes is None


def test_static_profile_maps_a_b_a_independently_of_call_order() -> None:
    grasp_a, local = synthetic_hug_prediction()
    other_local = local.copy()
    # Change the finger configuration, not merely its camera-frame pose.
    for base in (1, 5, 9, 13, 17):
        for offset in range(1, 4):
            other_local[base + offset] += [0.002 * offset, -0.004 * offset, 0.006 * offset]
    transform = grasp_a.T_camera_wrist
    grasp_b = replace(grasp_a, landmarks_3d=other_local @ transform[:3, :3].T + transform[:3, 3])
    fresh_a = ShadowHandRetargeter(profile=STATIC_RETARGETING_PROFILE).retarget(grasp_a)
    fresh_b = ShadowHandRetargeter(profile=STATIC_RETARGETING_PROFILE).retarget(grasp_b)
    reused = ShadowHandRetargeter(profile=STATIC_RETARGETING_PROFILE)
    first_a = reused.retarget(grasp_a)
    after_a_b = reused.retarget(grasp_b)
    final_a = reused.retarget(grasp_a)
    np.testing.assert_array_equal(first_a.qpos, fresh_a.qpos)
    np.testing.assert_array_equal(after_a_b.qpos, fresh_b.qpos)
    np.testing.assert_array_equal(final_a.qpos, fresh_a.qpos)
    evidence = final_a.retargeting_provenance
    assert evidence.profile_id == STATIC_RETARGETING_PROFILE
    assert evidence.reset_before_proposal
    assert evidence.temporal_norm_delta == 0.0
    assert len(evidence.optimizer_status_codes) == reused.optimizer_iterations
    assert all(code > 0 for code in evidence.optimizer_status_codes)
    assert reused._retargeting.num_retargeting == reused.optimizer_iterations


def test_static_profile_actual_dex_gradient_matches_its_scalar_objective() -> None:
    grasp, _ = synthetic_hug_prediction()
    retargeter = ShadowHandRetargeter(profile=STATIC_RETARGETING_PROFILE)
    optimizer = retargeter._retargeting.optimizer
    landmarks = camera_landmarks_to_wrist(grasp.landmarks_3d, grasp.T_camera_wrist)
    landmarks = landmarks @ MANO_TO_OPERATOR_RIGHT.T
    vectors = reference_vectors(landmarks, optimizer.target_link_human_indices).astype(np.float32)
    limits = retargeter.joint_limits
    last = limits.mean(axis=1).astype(np.float32)
    direction = np.where(np.arange(len(last)) % 2 == 0, 1.0, -1.0)
    x = last.astype(float) + direction * 0.1 * (limits[:, 1] - limits[:, 0])
    objective = optimizer.get_objective_function(vectors, np.array([], dtype=np.float32), last)
    analytic = np.zeros_like(x)
    objective(x, analytic)
    step = 1e-5
    numerical = np.empty_like(x)
    for i in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[i] += step
        minus[i] -= step
        numerical[i] = (objective(plus, np.array([])) - objective(minus, np.array([]))) / (2 * step)
    np.testing.assert_allclose(analytic, numerical, atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("status", [-1, 0])
def test_static_profile_refuses_upstream_failed_status_even_if_pose_is_finite(monkeypatch, status) -> None:
    grasp, _ = synthetic_hug_prediction()
    retargeter = ShadowHandRetargeter(profile=STATIC_RETARGETING_PROFILE)
    calls = []
    def stale_pose(_):
        calls.append(True)
        return np.zeros(len(retargeter.joint_names))
    monkeypatch.setattr(retargeter._retargeting, "retarget", stale_pose)
    monkeypatch.setattr(retargeter._retargeting.optimizer, "opt", SimpleNamespace(last_optimize_result=lambda: status))
    with pytest.raises(RetargetingError, match="failed with termination status"):
        retargeter.retarget(grasp)
    assert len(calls) == 1


def test_static_profile_refuses_runtime_without_status_introspection(monkeypatch) -> None:
    grasp, _ = synthetic_hug_prediction()
    retargeter = ShadowHandRetargeter(profile=STATIC_RETARGETING_PROFILE)
    monkeypatch.setattr(retargeter._retargeting.optimizer, "opt", object())
    with pytest.raises(RetargetingError, match="requires optimizer termination status"):
        retargeter.retarget(grasp)


def test_static_profile_preserves_positive_budget_status_without_calling_it_convergence(monkeypatch) -> None:
    grasp, _ = synthetic_hug_prediction()
    retargeter = ShadowHandRetargeter(profile=STATIC_RETARGETING_PROFILE)
    monkeypatch.setattr(retargeter._retargeting, "retarget", lambda _: np.zeros(len(retargeter.joint_names)))
    monkeypatch.setattr(retargeter._retargeting.optimizer, "opt", SimpleNamespace(last_optimize_result=lambda: 5))
    pose = retargeter.retarget(grasp)
    assert pose.retargeting_provenance.optimizer_status_codes == (5,) * 20
