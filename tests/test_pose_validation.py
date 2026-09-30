from __future__ import annotations

import numpy as np

from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
from grasp_failure_prediction.evaluation.retargeting import (
    ShadowHandRetargeter,
    default_dex_urdf_root,
)
from test_retargeting import synthetic_hug_prediction


def test_mujoco_uses_exact_official_shadow_joint_names() -> None:
    validator = ShadowPoseValidator(default_dex_urdf_root())
    retargeter = ShadowHandRetargeter()
    assert set(validator.joint_names) == set(retargeter.joint_names)


def test_mujoco_forward_kinematics_matches_dex_pinocchio() -> None:
    grasp, _ = synthetic_hug_prediction()
    retargeter = ShadowHandRetargeter()
    pose = retargeter.retarget(grasp)
    validator = ShadowPoseValidator(default_dex_urdf_root())
    result = validator.validate(pose)

    optimizer = retargeter._retargeting.optimizer
    optimizer.robot.compute_forward_kinematics(pose.qpos)
    positions = np.asarray(
        [
            optimizer.robot.get_link_pose(index)[:3, 3]
            for index in optimizer.computed_link_indices
        ]
    )
    expected = (
        positions[optimizer.task_link_indices]
        - positions[optimizer.origin_link_indices]
    )
    np.testing.assert_allclose(result.achieved_vectors_m, expected, atol=1e-8)


def test_pose_validation_reports_each_target_link() -> None:
    grasp, _ = synthetic_hug_prediction()
    pose = ShadowHandRetargeter().retarget(grasp)
    result = ShadowPoseValidator(default_dex_urdf_root()).validate(pose)
    assert len(result.link_names) == 10
    assert result.errors_m.shape == (10,)
    assert result.fingertip_errors_m.shape == (5,)
    assert result.mean_fingertip_error_m == np.mean(result.errors_m[:5])
    assert np.isfinite(result.mean_error_m)
    assert result.maximum_error_m == np.max(result.errors_m)


def test_validated_pose_has_visual_geometry_for_mujoco_viewer() -> None:
    grasp, _ = synthetic_hug_prediction()
    pose = ShadowHandRetargeter().retarget(grasp)
    validator = ShadowPoseValidator(default_dex_urdf_root())
    validator.set_pose(pose)
    assert validator.model.ngeom > 0
    assert np.all(np.isfinite(validator.data.geom_xpos))
