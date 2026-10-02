import numpy as np
import pytest
from grasp_failure_prediction.integrations.hug_frames import convert_hand_frame


def test_axis_conversion_preserves_world_landmark_vectors():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(21, 3))
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [.1, .2, .3]
    converted, target_pose = convert_hand_frame(vectors, pose)
    np.testing.assert_allclose(converted @ target_pose[:3, :3].T,
                               vectors @ pose[:3, :3].T, atol=1e-12)
    np.testing.assert_array_equal(target_pose[:3, 3], pose[:3, 3])


def test_reflected_axis_conversion_is_rejected():
    with pytest.raises(ValueError, match="proper rotation"):
        convert_hand_frame(np.zeros((21, 3)), np.eye(4), np.diag([1, 1, -1]))
