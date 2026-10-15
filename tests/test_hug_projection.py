import numpy as np
import pytest

from grasp_failure_prediction.integrations.hug_projection import (
    image_support, inverse_transform_points, project_camera,
    rotation_from_saved_six_d, transform_points,
)


def intrinsic():
    return np.array([[10., 0., 4.], [0., 20., 3.], [0., 0., 1.]])


def test_analytical_projection_and_hug_epsilon_are_distinct():
    xyz = np.array([[1., 2., 2.]])
    uv, positive = project_camera(xyz, intrinsic())
    np.testing.assert_array_equal(positive, [True])
    np.testing.assert_allclose(uv, [[9., 23.]], rtol=0, atol=1e-14)
    exported, _ = project_camera(xyz, intrinsic(), denominator_epsilon=1e-8)
    np.testing.assert_allclose(exported, [[18. / (2. + 1e-8), 46. / (2. + 1e-8)]])
    assert np.max(np.abs(exported - uv)) > 0


def test_nonpositive_depth_is_never_image_support():
    uv, positive = project_camera([[0, 0, -1], [0, 0, 0], [0, 0, 1]], intrinsic())
    np.testing.assert_array_equal(positive, [False, False, True])
    assert np.isnan(uv[:2]).all()
    assert image_support([[0, 0, -1]], intrinsic(), 9, 7)["within_pixel_center_window"] == 0
    export, _ = project_camera([[0, 0, -1]], intrinsic(), denominator_epsilon=1e-8, include_nonpositive=True)
    assert np.isfinite(export).all()  # Algebraic export consistency is separate.


def test_translated_rotated_camera_roundtrip_uses_world_inverse():
    t = np.array([[0., -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]])
    xyz = np.array([[1., 2., 3.], [-1., .5, 2.]])
    world = transform_points(xyz, t)
    np.testing.assert_allclose(world[0], [0., 4., 7.])
    np.testing.assert_allclose(inverse_transform_points(world, t), xyz)
    # Applying a wrist rotation again to already-camera-space vertices is wrong.
    assert np.max(np.abs(world - transform_points(world, t))) > 1


def test_interleaved_rotation_columns_and_degenerate_rejection():
    saved = np.array([0., -1., 1., 0., 0., 0.])
    np.testing.assert_allclose(rotation_from_saved_six_d(saved), [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for values in ([0.] * 6, [1, 2, 0, 0, 0, 0]):
        with pytest.raises(ValueError, match="degenerate"):
            rotation_from_saved_six_d(values)


def test_image_support_is_descriptive_and_preserves_outside_points():
    depth = np.full((7, 9), 2.)
    mask = np.zeros((7, 9), np.uint8)
    r = image_support([[0, 0, 1], [0, 0, 3], [100, 0, 1]], intrinsic(), 9, 7, depth=depth, mask=mask)
    assert r["within_pixel_center_window"] == 2
    assert r["outside_pixel_center_window"] == 1
    assert r["behind_scene_by_over_allowance"] == 1
    assert r["in_front_of_scene_by_over_allowance"] == 1
    assert r["supported_points_inside_object_mask"] == 0
    assert r["mask_is_not_acceptance_gate"] is True


@pytest.mark.parametrize("points,k", [([[0, 0, float('nan')]], intrinsic()), ([[0, 0, 1]], np.eye(2))])
def test_rejects_invalid_projection_inputs(points, k):
    with pytest.raises(ValueError):
        project_camera(points, k)
