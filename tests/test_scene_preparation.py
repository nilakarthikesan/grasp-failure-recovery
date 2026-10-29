"""Checks for the standalone hand-free scene preparation path.

The saved-scene tests intentionally render with the installed local MuJoCo
backend. On macOS they need graphics access (MUJOCO_GL=cgl), not new packages.
"""
import copy
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from PIL import Image
import pytest

from grasp_failure_prediction.integrations.observations import check_manifest, validate_depth_encoding
from grasp_failure_prediction.integrations.scene_preparation import (
    STATE_SPEC, _integration_state, camera_calibration, geometry_checks, load_config,
    main, prepare_scene, ray_box_depth, reload_saved_scene, scene_xml, settle_scene,
    validate_config,
)


CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'scene_preparation_cube.json'


@pytest.fixture
def config():
    return load_config(CONFIG)


def test_compiled_cube_dimensions_mass_inertia_and_contact_defaults(config):
    model = mujoco.MjModel.from_xml_string(scene_xml(config))
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'cube')
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table')
    body = model.geom_bodyid[cube]
    assert model.nbody == 2 and model.ngeom == 2 and model.nu == 0
    assert model.nq == 7 and model.nv == 6
    np.testing.assert_allclose(2 * model.geom_size[cube], [.05, .05, .05], atol=0, rtol=0)
    assert model.body_mass[body] == pytest.approx(.1)
    # Independent homogeneous-cube formula: I = m*(a*a + a*a)/12.
    np.testing.assert_allclose(model.body_inertia[body], np.full(3, .1 * (.05**2 + .05**2) / 12))
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
    np.testing.assert_allclose(model.geom_friction[[cube, table]], [[1, .005, .0001]] * 2)
    np.testing.assert_allclose(model.geom_solref[cube], config['contact']['solref'])
    np.testing.assert_allclose(model.geom_solimp[cube], config['contact']['solimp'])
    assert model.geom_condim[cube] == model.geom_condim[table] == 3
    assert model.opt.timestep == .002


def test_gravity_settling_preserves_measured_velocity(config):
    model = mujoco.MjModel.from_xml_string(scene_xml(config))
    data = mujoco.MjData(model)
    height = data.qpos[2]
    result = settle_scene(model, data, config)
    assert result['accepted'] and result['continuous_table_contact']
    assert result['sample_count'] == 500
    assert result['physics_steps'] <= config['settling']['maximum_steps']
    assert .024 < data.qpos[2] < .026 < height
    np.testing.assert_array_equal(data.qvel, result['velocity_at_capture_m_s_rad_s'])
    # The solver retains a tiny measured residual: no forced post-settling zero.
    assert np.any(data.qvel != 0)
    assert data.time == pytest.approx(result['physics_steps'] * .002)
    assert result['position_box_diagonal_m'] <= .0001
    assert result['maximum_linear_speed_m_s'] <= .001


def test_known_camera_axes_and_off_center_projections(config):
    camera = copy.deepcopy(config['camera'])
    camera['position_m'] = [0, -1, 0]
    quaternion, transform, intrinsic = camera_calibration(camera, [0, 0, 0])
    # Looking along world +Y: image right=+X, image down=-Z.
    np.testing.assert_allclose(transform[:3, :3], [[1, 0, 0], [0, 0, 1], [0, -1, 0]], atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(transform[:3, :3]), 1.)
    focal = 112 / np.tan(np.pi / 8)
    points = np.array([[0, 0, 0], [.1, 0, .2], [-.2, 1, -.4]])
    expected = np.array([[111.5, 111.5], [111.5 + .1*focal, 111.5 - .2*focal],
                         [111.5 - .1*focal, 111.5 + .2*focal]])
    camera_points = (points - camera['position_m']) @ transform[:3, :3]
    homogeneous = camera_points @ intrinsic.T
    np.testing.assert_allclose(homogeneous[:, :2] / homogeneous[:, 2:3], expected)
    gl_rotation = np.empty(9)
    mujoco.mju_quat2Mat(gl_rotation, quaternion)
    np.testing.assert_allclose(gl_rotation.reshape(3, 3) @ np.diag([1, -1, -1]), transform[:3, :3], atol=1e-15)


def test_analytic_box_rays_have_known_depth_misses_and_rotated_surface():
    rays = np.array([[0., 0, 1], [.2, 0, 1], [2, 0, 1], [0, 0, -1]])
    depth = ray_box_depth([0, 0, -1], rays, np.zeros(3), np.eye(3), np.full(3, .5))
    np.testing.assert_allclose(depth[:2], [.5, .5])
    assert np.isinf(depth[2:]).all()
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rotated = ray_box_depth([1, 0, 0], np.array([[-1., 0, 0]]), np.zeros(3), rotation, np.full(3, .5))
    np.testing.assert_allclose(rotated, [1 - np.sqrt(.5)])
    inside = ray_box_depth([0, 0, 0], rays[:1], np.zeros(3), np.eye(3), np.full(3, .5))
    np.testing.assert_allclose(inside, [.5])


@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('object', 'mass_kg', 0), ('object', 'mass_kg', True),
    ('object', 'mass_kg', 10**400),
    ('object', 'dimensions_m', [.05, .05, -.05]),
    ('object', 'dimensions_m', [.05, .06, .05]),
    ('object', 'initial_orientation_wxyz', [0, 0, 0, 0]),
    ('object', 'initial_position_m', [0, 0, .01]),
    ('object', 'initial_position_m', [1, 0, .08]),
    ('camera', 'vertical_fov_degrees', 180), ('camera', 'width', 0),
    ('camera', 'height', 224.5), ('camera', 'world_up', [0, 0, 0]),
    ('camera', 'position_m', [0, 0, 1]),
    ('camera', 'position_m', [np.nan, 0, 1]),
    ('camera', 'target', 'initial_object_center'),
    ('contact', 'friction', [1, -.005, 0]), ('contact', 'solimp', [1, .95, .001, .5, 2]),
    ('settling', 'maximum_steps', 100), ('simulator', 'gravity_m_s2', [0, 0, 0]),
    ('lighting', 'headlight_active', 1),
])
def test_malformed_configuration_is_rejected(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_config_rejects_unknown_missing_duplicate_and_nonfinite_json(config, tmp_path):
    for variant in ({**config, 'unexpected': 1}, {key: val for key, val in config.items() if key != 'contact'}):
        with pytest.raises(ValueError):
            validate_config(variant)
    path = tmp_path / 'bad.json'
    for payload in ('{"schema_version": 1, "schema_version": 2}', '{"bad": NaN}'):
        path.write_text(payload)
        with pytest.raises(ValueError):
            load_config(path)


def test_output_collision_and_bad_config_do_not_modify_existing_files(config, tmp_path):
    existing = tmp_path / 'existing'
    existing.mkdir()
    sentinel = existing / 'sentinel.txt'
    sentinel.write_text('keep me')
    with pytest.raises(FileExistsError):
        prepare_scene(CONFIG, existing)
    assert sentinel.read_text() == 'keep me'
    bad_config = tmp_path / 'bad.json'
    bad_config.write_text('{}')
    new_output = tmp_path / 'must_not_exist'
    assert main(['--config', str(bad_config), '--output', str(new_output)]) == 1
    assert not new_output.exists()


@pytest.fixture(scope='module')
def saved_scene(tmp_path_factory):
    directory = tmp_path_factory.mktemp('scene_prepare') / 'capture'
    report = prepare_scene(CONFIG, directory)
    return directory, report


def test_saved_scene_outputs_reload_and_observation_contract(saved_scene, tmp_path):
    directory, report = saved_scene
    assert report['scene_preparation_checks_passed']
    assert report['physics']['no_hand_or_actuators']
    assert all(value is False for value in report['scope'].values())
    expected = ('scene.xml', 'scene_config.json', 'scene_state.npz', 'scene_state_metadata.json',
                'rgb.png', 'depth.png', 'depth_m.npy', 'object_mask.png', 'intrinsics.npy',
                'T_world_camera.npy', 'observation_manifest.json', 'validation_report.json',
                'validation_report.txt')
    assert all((directory / name).is_file() for name in expected)
    depth = np.load(directory / 'depth_m.npy', allow_pickle=False)
    encoded = np.asarray(Image.open(directory / 'depth.png'))
    mask = np.asarray(Image.open(directory / 'object_mask.png'))
    rgb = np.asarray(Image.open(directory / 'rgb.png'))
    assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
    assert encoded.dtype == np.uint16 and depth.dtype == np.float32
    assert set(np.unique(mask)) == {0, 255} and np.all(depth[mask > 0] > 0)
    assert validate_depth_encoding(depth, encoded)['depth_encoding_checked']
    assert check_manifest(directory / 'observation_manifest.json', tmp_path / 'preflight')['preflight_passed']
    for geometry in report['geometry_checks'].values():
        assert geometry['sample_count'] == 128
        assert geometry['maximum_analytical_depth_error_m'] < .00002
        assert geometry['maximum_mujoco_ray_depth_error_m'] < .00002
        assert geometry['maximum_raw_camera_reprojection_error_pixels'] < 1e-8
    assert report['reload']['maximum_rgb_channel_error'] == 0
    assert report['reload']['maximum_depth_error_m'] == 0
    assert report['reload']['maximum_integration_state_error'] == 0
    # Portability: the saved XML has no external file references or absolute paths.
    moved = tmp_path / 'moved'
    shutil.copytree(directory, moved)
    model, data = reload_saved_scene(moved)
    with np.load(moved / 'scene_state.npz', allow_pickle=False) as archive:
        assert int(archive['state_spec']) == STATE_SPEC
        np.testing.assert_array_equal(_integration_state(model, data), archive['integration_state'])
    assert model.nkey == 1
    key_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, key_data, 0)
    np.testing.assert_allclose(key_data.qpos, data.qpos, atol=0, rtol=0)
    np.testing.assert_allclose(key_data.qvel, data.qvel, atol=0, rtol=0)
    # Independent reopened states also agree after continuing physics.
    second_model, second_data = reload_saved_scene(moved)
    for _ in range(10):
        mujoco.mj_step(model, data)
        mujoco.mj_step(second_model, second_data)
    np.testing.assert_array_equal(_integration_state(model, data), _integration_state(second_model, second_data))


def test_saved_scene_independent_geometry_rejects_wrong_depth_and_calibration(saved_scene):
    directory, _ = saved_scene
    config = load_config(directory / 'scene_config.json')
    model, data = reload_saved_scene(directory)
    depth = np.load(directory / 'depth_m.npy')
    transform = np.load(directory / 'T_world_camera.npy')
    intrinsic = np.load(directory / 'intrinsics.npy')
    # Both primitives have stable fixed names/ids in this independent fixture.
    from grasp_failure_prediction.integrations.scene_preparation import capture_arrays
    _, _, geometry = capture_arrays(model, data, config['camera'])
    wrong_depth = np.where(depth > 0, depth + .001, 0)
    with pytest.raises(ValueError, match='depth'):
        geometry_checks(model, data, wrong_depth, geometry, transform, intrinsic, config)
    wrong_intrinsic = intrinsic.copy()
    wrong_intrinsic[0, 0] *= .5
    wrong_intrinsic[1, 1] *= .5
    with pytest.raises(ValueError):
        geometry_checks(model, data, depth, geometry, transform, wrong_intrinsic, config)


@pytest.mark.parametrize('filename', ['scene.xml', 'scene_config.json', 'scene_state.npz'])
def test_saved_scene_rejects_tampered_reload_artifacts(saved_scene, tmp_path, filename):
    directory, _ = saved_scene
    copied = tmp_path / 'tampered'
    shutil.copytree(directory, copied)
    with (copied / filename).open('ab') as stream:
        stream.write(b'\n ')
    with pytest.raises(ValueError, match='saved artifact changed'):
        reload_saved_scene(copied)


def test_unsettled_cube_is_rejected_without_output(config, tmp_path):
    config['settling']['maximum_steps'] = config['settling']['window_samples']
    path = tmp_path / 'too_short.json'
    path.write_text(json.dumps(config))
    output = tmp_path / 'unsettled'
    with pytest.raises(ValueError, match='did not settle'):
        prepare_scene(path, output)
    assert not output.exists()
