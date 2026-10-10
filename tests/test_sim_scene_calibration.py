import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.integrations.sim_observations import (
    camera_calibration, evaluate_settling_window, object_segmentation_mask,
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


def stationary_states(samples=500):
    poses = np.zeros((samples, 7))
    poses[:, 2], poses[:, 3] = .05, 1.
    return poses, np.zeros((samples, 6)), np.ones(samples, dtype=bool), np.zeros(samples, dtype=bool)


def test_small_contact_jitter_passes_without_mutating_states_or_zeroing_velocities():
    poses, velocity, table, hand = stationary_states()
    phase = np.linspace(0, 12*np.pi, len(poses))
    poses[:, 0] = .00002*np.sin(phase)
    poses[:, 1] = .00001*np.cos(phase)
    angle = .0003*np.sin(phase)
    poses[:, 3], poses[:, 4] = np.cos(angle/2), np.sin(angle/2)
    velocity[:, 0], velocity[:, 3] = .0017*np.sin(phase), .0419*np.cos(phase)
    before = poses.copy(), velocity.copy()
    result = evaluate_settling_window(poses, velocity, table, hand)
    assert result['accepted'] is True
    assert 0 < result['position_box_diagonal_m'] < .0001
    assert 0 < result['quaternion_diameter_bound_rad'] < .002
    assert result['maximum_linear_speed_m_s'] > .001
    assert result['maximum_angular_speed_rad_s'] > .01
    np.testing.assert_array_equal(poses, before[0])
    np.testing.assert_array_equal(velocity, before[1])


def test_finite_speed_is_recorded_without_overriding_geometric_stationarity():
    poses, velocity, table, hand = stationary_states()
    velocity[:, :3] = [.003, .004, 0.]
    velocity[:, 3:] = [0., .06, .08]
    result = evaluate_settling_window(poses, velocity, table, hand)
    assert result['accepted'] is True
    assert result['maximum_linear_speed_m_s'] == pytest.approx(.005)
    assert result['maximum_angular_speed_rad_s'] == pytest.approx(.1)


@pytest.mark.parametrize('drift', ['position', 'orientation', 'two_axes'])
def test_slow_drift_fails_even_when_velocity_is_small(drift):
    poses, velocity, table, hand = stationary_states()
    if drift == 'position':
        poses[:, 0] = np.linspace(0, .00012, len(poses))
        velocity[:, 0] = .00012
    elif drift == 'two_axes':
        poses[:, :2] = np.linspace(0, .00008, len(poses))[:, None]
        velocity[:, :2] = .00008
    else:
        angle = np.linspace(0, .0011, len(poses))
        poses[:, 3], poses[:, 6] = np.cos(angle/2), np.sin(angle/2)
        velocity[:, 5] = .0011
    assert evaluate_settling_window(poses, velocity, table, hand)['accepted'] is False


def test_quaternion_signs_and_near_unit_rounding_do_not_create_apparent_motion():
    poses, velocity, table, hand = stationary_states()
    poses[:, 3:] = [np.cos(np.pi/3), np.sin(np.pi/3), 0., 0.]
    poses[::2, 3:] *= -1.
    poses[::3, 3:] *= 1. + 5e-7
    before = poses.copy()
    result = evaluate_settling_window(poses, velocity, table, hand)
    assert result['accepted'] is True
    assert result['quaternion_diameter_bound_rad'] < 1e-12
    np.testing.assert_array_equal(poses, before)


def test_short_window_cannot_pass_and_old_motion_outside_trailing_window_is_ignored():
    assert evaluate_settling_window(*stationary_states(499))['accepted'] is False
    poses, velocity, table, hand = stationary_states(520)
    poses[:20, 0] = .5
    result = evaluate_settling_window(poses, velocity, table, hand)
    assert result['accepted'] is True
    assert result['sample_count'] == 500


@pytest.mark.parametrize('contact', ['lost_table', 'hand_contact'])
def test_every_window_sample_must_have_table_contact_without_hand_contact(contact):
    poses, velocity, table, hand = stationary_states()
    if contact == 'lost_table':
        table[250] = False
    else:
        hand[250] = True
    assert evaluate_settling_window(poses, velocity, table, hand)['accepted'] is False


@pytest.mark.parametrize(('coordinate', 'accepted'), [(.2999, True), (.3, False), (-.3, False), (.31, False)])
def test_capture_pose_stays_strictly_inside_declared_workspace(coordinate, accepted):
    poses, velocity, table, hand = stationary_states()
    poses[:, 0] = coordinate
    assert evaluate_settling_window(poses, velocity, table, hand)['accepted'] is accepted


@pytest.mark.parametrize('invalid', ['pose_shape', 'velocity_shape', 'empty', 'position_nan',
                                    'quaternion_inf', 'velocity_nan', 'zero_quaternion',
                                    'nonunit_quaternion', 'contact_shape', 'contact_numbers'])
def test_malformed_and_nonfinite_settling_states_are_rejected(invalid):
    poses, velocity, table, hand = stationary_states()
    if invalid == 'pose_shape':
        poses = poses[:, :6]
    elif invalid == 'velocity_shape':
        velocity = velocity[:-1]
    elif invalid == 'empty':
        poses, velocity, table, hand = stationary_states(0)
    elif invalid == 'position_nan':
        poses[1, 0] = np.nan
    elif invalid == 'quaternion_inf':
        poses[1, 3] = np.inf
    elif invalid == 'velocity_nan':
        velocity[1, 0] = np.nan
    elif invalid == 'zero_quaternion':
        poses[1, 3:] = 0.
    elif invalid == 'nonunit_quaternion':
        poses[1, 3] = 2.
    elif invalid == 'contact_shape':
        table = table[:-1]
    else:
        table = table.astype(int)
    with pytest.raises(ValueError):
        evaluate_settling_window(poses, velocity, table, hand)
