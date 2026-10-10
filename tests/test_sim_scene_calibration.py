import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.integrations.sim_observations import (
    camera_calibration, object_segmentation_mask,
)


def test_camera_frame_points_forward_and_projects_target_to_center():
    position = np.array([.25,-.35,.35])
    target = np.array([-.03,.02,.05])
    quaternion, transform, intrinsic = camera_calibration(position, target)
    np.testing.assert_allclose(transform[:3,:3].T @ transform[:3,:3], np.eye(3), atol=1e-12)
    assert np.linalg.det(transform[:3,:3]) == pytest.approx(1.)
    camera_point = transform[:3,:3].T @ (target-position)
    assert camera_point[2] > 0
    projected = intrinsic @ camera_point
    np.testing.assert_allclose(projected[:2]/projected[2], [111.5,111.5])
    gl_rotation = np.empty(9)
    mujoco.mju_quat2Mat(gl_rotation, quaternion)
    np.testing.assert_allclose(gl_rotation.reshape(3,3) @ np.diag([1,-1,-1]), transform[:3,:3], atol=1e-12)


def test_compound_object_segmentation_includes_all_geoms_and_no_other_types():
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    segmentation = np.array([[[3,geom],[4,geom],[5,geom],[3,int(mujoco.mjtObj.mjOBJ_SITE)]]])
    np.testing.assert_array_equal(object_segmentation_mask(segmentation,{3,5}), [[True,False,True,False]])


@pytest.mark.parametrize('position', [[0,0,0], [0,0,1], [np.nan,0,1], [0,1]])
def test_invalid_or_degenerate_camera_is_rejected(position):
    with pytest.raises(ValueError):
        camera_calibration(position,[0,0,0])
