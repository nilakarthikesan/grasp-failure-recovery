"""Prepare a hand-free synthetic cube scene and calibrated RGB-D artifacts.

Run ``python -m grasp_failure_prediction.integrations.scene_preparation
--config configs/scene_preparation_cube.json --output NEW_DIRECTORY``.
The editable scene.xml contains a capture keyframe. Exact same-runtime reloads
use reload_saved_scene(directory), which restores the full integration state.
All configured physics values and numerical tolerances are test assumptions;
passing these software checks is not evidence about physical grasp success.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from .observations import check_manifest, validate_arrays, validate_depth_encoding


STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
SCHEMA = {
    'schema_version': None, 'purpose': None, 'assumptions': None,
    'simulator': ('timestep_s', 'gravity_m_s2', 'integrator', 'solver', 'iterations',
                  'tolerance', 'cone', 'impratio'),
    'contact': ('friction', 'condim', 'solref', 'solimp', 'margin_m', 'gap_m',
                'priority', 'solmix'),
    'object': ('shape', 'dimensions_m', 'mass_kg', 'initial_position_m',
               'initial_orientation_wxyz', 'initial_velocity_m_s_rad_s', 'rgba'),
    'table': ('center_m', 'dimensions_m', 'rgba'),
    'camera': ('position_m', 'target', 'world_up', 'width', 'height',
               'vertical_fov_degrees', 'near_m', 'far_m'),
    'lighting': ('position_m', 'direction', 'ambient', 'diffuse', 'specular',
                 'cast_shadow', 'headlight_active', 'offscreen_samples'),
    'settling': ('window_samples', 'maximum_steps', 'maximum_position_box_diagonal_m',
                 'maximum_orientation_diameter_bound_rad', 'maximum_linear_speed_m_s',
                 'maximum_angular_speed_rad_s', 'maximum_support_penetration_m'),
    'validation': ('interior_erosion_pixels', 'maximum_sample_pixels_per_geometry',
                   'maximum_depth_error_m', 'maximum_reprojection_error_pixels',
                   'maximum_reload_state_error', 'maximum_reload_depth_error_m',
                   'maximum_reload_rgb_channel_error'),
}


def _number(value, name, *, minimum=None, strict=False, integer=False):
    types = (int,) if integer else (int, float)
    valid = not isinstance(value, bool) and isinstance(value, types)
    try:
        valid = valid and np.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f'{name} must be a finite {"integer" if integer else "number"}')
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ValueError(f'{name} must be {">" if strict else ">="} {minimum}')
    return value


def _vector(value, length, name, *, minimum=None, strict=False):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'{name} must be an explicit {length}-element list')
    return np.array([_number(item, name, minimum=minimum, strict=strict) for item in value], dtype=np.float64)


def validate_config(config):
    """Strict schema and geometry checks, with no implicit configuration defaults."""
    if not isinstance(config, dict) or set(config) != set(SCHEMA):
        raise ValueError('config must contain exactly the documented top-level fields')
    for section, keys in SCHEMA.items():
        if keys is not None and (not isinstance(config[section], dict)
                                  or set(config[section]) != set(keys)):
            raise ValueError(f'{section} must contain exactly {list(keys)}')
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    if config['purpose'] != 'synthetic_scene_preparation_only':
        raise ValueError('purpose must be synthetic_scene_preparation_only')
    assumptions = config['assumptions']
    if not isinstance(assumptions, list) or not assumptions or not all(
            isinstance(item, str) and item.strip() for item in assumptions):
        raise ValueError('assumptions must be a nonempty list of strings')
    sim, contact, obj, table, camera, light, settle, checks = (
        config[key] for key in ('simulator', 'contact', 'object', 'table', 'camera',
                               'lighting', 'settling', 'validation'))
    for key in ('timestep_s', 'tolerance', 'impratio'):
        _number(sim[key], f'simulator.{key}', minimum=0, strict=True)
    gravity = _vector(sim['gravity_m_s2'], 3, 'gravity_m_s2')
    if not (gravity[0] == gravity[1] == 0 and gravity[2] < 0):
        raise ValueError('this table scene requires downward vertical gravity')
    if sim['integrator'] not in ('Euler', 'RK4', 'implicit', 'implicitfast'):
        raise ValueError('unsupported integrator')
    if sim['solver'] not in ('PGS', 'CG', 'Newton') or sim['cone'] not in ('pyramidal', 'elliptic'):
        raise ValueError('unsupported solver or cone')
    _number(sim['iterations'], 'iterations', minimum=1, integer=True)
    _vector(contact['friction'], 3, 'contact.friction', minimum=0)
    _vector(contact['solref'], 2, 'contact.solref', minimum=0, strict=True)
    impedance = _vector(contact['solimp'], 5, 'contact.solimp', minimum=0)
    if not (0 < impedance[0] <= impedance[1] < 1 and impedance[2] > 0
            and 0 < impedance[3] < 1 and impedance[4] >= 1):
        raise ValueError('invalid contact.solimp')
    if type(contact['condim']) is not int or contact['condim'] not in (1, 3, 4, 6):
        raise ValueError('contact.condim must be 1, 3, 4 or 6')
    for key in ('margin_m', 'gap_m', 'solmix'):
        _number(contact[key], f'contact.{key}', minimum=0)
    _number(contact['priority'], 'contact.priority', minimum=0, integer=True)
    if contact['gap_m'] > contact['margin_m']:
        raise ValueError('contact gap cannot exceed margin')
    dimensions = _vector(obj['dimensions_m'], 3, 'object.dimensions_m', minimum=0, strict=True)
    if obj['shape'] != 'cube' or not np.allclose(dimensions, dimensions[0], rtol=0, atol=1e-12):
        raise ValueError('this synthetic configuration supports only a cube')
    _number(obj['mass_kg'], 'object.mass_kg', minimum=0, strict=True)
    position = _vector(obj['initial_position_m'], 3, 'object.initial_position_m')
    quaternion = _vector(obj['initial_orientation_wxyz'], 4, 'initial_orientation_wxyz')
    if not np.isclose(np.linalg.norm(quaternion), 1., rtol=0, atol=1e-10):
        raise ValueError('object orientation must be a unit wxyz quaternion')
    _vector(obj['initial_velocity_m_s_rad_s'], 6, 'initial_velocity_m_s_rad_s')
    center = _vector(table['center_m'], 3, 'table.center_m')
    table_dimensions = _vector(table['dimensions_m'], 3, 'table.dimensions_m', minimum=0, strict=True)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    extent = np.abs(rotation.reshape(3, 3)) @ (dimensions / 2)
    if np.any(np.abs(position[:2] - center[:2]) + extent[:2] >= table_dimensions[:2] / 2):
        raise ValueError('initial cube must be fully inside the table footprint')
    if position[2] - extent[2] <= center[2] + table_dimensions[2] / 2:
        raise ValueError('initial cube must start above, separated from the table')
    for section, value in (('object', obj), ('table', table)):
        rgba = _vector(value['rgba'], 4, f'{section}.rgba', minimum=0)
        if np.any(rgba > 1) or rgba[3] != 1:
            raise ValueError('RGBA channels must be in [0,1] with opaque alpha')
    for key in ('width', 'height'):
        _number(camera[key], f'camera.{key}', minimum=16, integer=True)
        if camera[key] > 4096:
            raise ValueError('camera image dimensions exceed the 4096 pixel resource guard')
    fov = _number(camera['vertical_fov_degrees'], 'vertical_fov_degrees', minimum=0, strict=True)
    if fov >= 179:
        raise ValueError('vertical FOV must be below 179 degrees')
    _number(camera['near_m'], 'camera.near_m', minimum=0, strict=True)
    _number(camera['far_m'], 'camera.far_m', minimum=camera['near_m'], strict=True)
    if camera['target'] != 'settled_object_center':
        raise ValueError('camera target must be settled_object_center')
    camera_calibration(camera, position)
    for key in ('position_m', 'direction', 'ambient', 'diffuse', 'specular'):
        vector = _vector(light[key], 3, f'lighting.{key}')
        if key in ('ambient', 'diffuse', 'specular') and (np.any(vector < 0) or np.any(vector > 1)):
            raise ValueError('lighting color channels must be in [0,1]')
    if np.linalg.norm(light['direction']) == 0:
        raise ValueError('lighting.direction must be nonzero')
    for key in ('cast_shadow', 'headlight_active'):
        if type(light[key]) is not bool:
            raise ValueError(f'lighting.{key} must be boolean')
    _number(light['offscreen_samples'], 'offscreen_samples', minimum=0, integer=True)
    if light['offscreen_samples'] > 16:
        raise ValueError('offscreen_samples cannot exceed 16')
    _number(settle['window_samples'], 'window_samples', minimum=2, integer=True)
    _number(settle['maximum_steps'], 'maximum_steps', minimum=settle['window_samples'], integer=True)
    for key in SCHEMA['settling'][2:]:
        _number(settle[key], f'settling.{key}', minimum=0, strict=True)
    _number(checks['interior_erosion_pixels'], 'interior_erosion_pixels', minimum=1, integer=True)
    _number(checks['maximum_sample_pixels_per_geometry'], 'maximum_sample_pixels_per_geometry', minimum=4, integer=True)
    for key in SCHEMA['validation'][2:-1]:
        _number(checks[key], f'validation.{key}', minimum=0, strict=True)
    _number(checks['maximum_reload_rgb_channel_error'], 'maximum_reload_rgb_channel_error', minimum=0, integer=True)
    if checks['maximum_reload_rgb_channel_error'] > 255:
        raise ValueError('maximum RGB error cannot exceed 255')
    return config


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON field: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(), object_pairs_hook=_unique_pairs,
                                      parse_constant=_reject_constant))


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _values(array):
    return ' '.join(format(float(item), '.17g') for item in np.asarray(array).ravel())


def camera_calibration(camera, target):
    """Return MuJoCo quaternion, camera-to-world CV transform and pinhole K.

    Pixel indices refer to pixel centers: principal point ((width-1)/2,
    (height-1)/2). CV axes are right/down/forward; MuJoCo uses right/up/back.
    """
    position = _vector(camera['position_m'], 3, 'camera.position_m')
    target = np.asarray(target, dtype=float)
    up = _vector(camera['world_up'], 3, 'camera.world_up')
    if target.shape != (3,) or not np.isfinite(target).all() or not np.isclose(
            np.linalg.norm(up), 1., rtol=0, atol=1e-10):
        raise ValueError('camera target must be finite and world_up must be unit length')
    forward = target - position
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError('camera must be separated from target')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError('camera view is parallel to world_up')
    right /= np.linalg.norm(right)
    rotation_gl = np.column_stack([right, np.cross(right, forward), -forward])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation_gl.reshape(9))
    transform = np.eye(4)
    transform[:3, :3] = rotation_gl @ np.diag([1., -1., -1.])
    transform[:3, 3] = position
    focal = camera['height'] / (2 * np.tan(np.deg2rad(camera['vertical_fov_degrees']) / 2))
    intrinsic = np.array([[focal, 0, (camera['width'] - 1) / 2],
                          [0, focal, (camera['height'] - 1) / 2], [0, 0, 1.]])
    return quaternion, transform, intrinsic


def scene_xml(config, *, target=None, capture=None):
    """Build self-contained MJCF with primitives only; no external assets."""
    validate_config(config)
    sim, contact, obj, table, camera, light = (
        config[key] for key in ('simulator', 'contact', 'object', 'table', 'camera', 'lighting'))
    root = ET.Element('mujoco', model='phase1_synthetic_cube_table')
    ET.SubElement(root, 'compiler', angle='radian', inertiafromgeom='true')
    ET.SubElement(root, 'option', timestep=str(sim['timestep_s']), gravity=_values(sim['gravity_m_s2']),
                  integrator=sim['integrator'], solver=sim['solver'], iterations=str(sim['iterations']),
                  tolerance=str(sim['tolerance']), cone=sim['cone'], impratio=str(sim['impratio']))
    # Extent=1 makes znear/zfar settings explicit metric distances.
    ET.SubElement(root, 'statistic', center='0 0 0', extent='1')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(camera['width']), offheight=str(camera['height']))
    ET.SubElement(visual, 'map', znear=str(camera['near_m']), zfar=str(camera['far_m']))
    ET.SubElement(visual, 'quality', offsamples=str(light['offscreen_samples']))
    ET.SubElement(visual, 'headlight', active=str(int(light['headlight_active'])))
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', friction=_values(contact['friction']), condim=str(contact['condim']),
                  solref=_values(contact['solref']), solimp=_values(contact['solimp']),
                  margin=str(contact['margin_m']), gap=str(contact['gap_m']),
                  priority=str(contact['priority']), solmix=str(contact['solmix']),
                  contype='1', conaffinity='1')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', name='scene_light', pos=_values(light['position_m']),
                  dir=_values(light['direction']), directional='true',
                  ambient=_values(light['ambient']), diffuse=_values(light['diffuse']),
                  specular=_values(light['specular']), castshadow=str(light['cast_shadow']).lower())
    ET.SubElement(world, 'geom', name='table', type='box', pos=_values(table['center_m']),
                  size=_values(np.array(table['dimensions_m']) / 2), rgba=_values(table['rgba']))
    body = ET.SubElement(world, 'body', name='cube', pos=_values(obj['initial_position_m']),
                         quat=_values(obj['initial_orientation_wxyz']))
    ET.SubElement(body, 'freejoint', name='cube_free')
    ET.SubElement(body, 'geom', name='cube', type='box',
                  size=_values(np.array(obj['dimensions_m']) / 2), mass=str(obj['mass_kg']),
                  rgba=_values(obj['rgba']))
    quaternion, _, _ = camera_calibration(camera, obj['initial_position_m'] if target is None else target)
    ET.SubElement(world, 'camera', name='capture', pos=_values(camera['position_m']),
                  quat=_values(quaternion), fovy=str(camera['vertical_fov_degrees']))
    if capture is not None:
        keys = ET.SubElement(root, 'keyframe')
        ET.SubElement(keys, 'key', name='capture', time=str(capture.time),
                      qpos=_values(capture.qpos), qvel=_values(capture.qvel))
    ET.indent(root)
    return ET.tostring(root, encoding='unicode') + '\n'


def _ids(model):
    return {name: mujoco.mj_name2id(model, kind, label) for name, kind, label in (
        ('cube', mujoco.mjtObj.mjOBJ_GEOM, 'cube'), ('table', mujoco.mjtObj.mjOBJ_GEOM, 'table'),
        ('body', mujoco.mjtObj.mjOBJ_BODY, 'cube'), ('camera', mujoco.mjtObj.mjOBJ_CAMERA, 'capture'))}


def _table_contact(data, ids):
    return any({int(item.geom1), int(item.geom2)} == {ids['cube'], ids['table']}
               for item in data.contact)


def settle_scene(model, data, config):
    """Gravity settling; never overwrite velocities, poses or forces after stepping."""
    criteria = config['settling']
    ids = _ids(model)
    history = deque(maxlen=criteria['window_samples'])
    data.qvel[:] = config['object']['initial_velocity_m_s_rad_s']
    mujoco.mj_forward(model, data)
    for step in range(criteria['maximum_steps']):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)  # Contact/pose diagnostics at the same time as qpos.
        if not all(np.isfinite(item).all() for item in (data.qpos, data.qvel, data.qacc)):
            raise ValueError('nonfinite state during gravity settling')
        if any(item.number for item in data.warning):
            raise ValueError('MuJoCo warning during gravity settling')
        history.append((data.qpos.copy(), data.qvel.copy(), _table_contact(data, ids)))
        if len(history) < criteria['window_samples']:
            continue
        poses = np.array([item[0] for item in history])
        velocity = np.array([item[1] for item in history])
        quats = poses[:, 3:7].copy()
        quats *= np.where(np.einsum('ij,j->i', quats, quats[0], optimize=False) < 0, -1., 1.)[:, None]
        angles = 4 * np.arctan2(np.linalg.norm(quats - quats[0], axis=1),
                               np.linalg.norm(quats + quats[0], axis=1))
        metrics = {
            'position_box_diagonal_m': float(np.linalg.norm(np.ptp(poses[:, :3], axis=0))),
            'orientation_diameter_bound_rad': float(2 * angles.max()),
            'maximum_linear_speed_m_s': float(np.linalg.norm(velocity[:, :3], axis=1).max()),
            'maximum_angular_speed_rad_s': float(np.linalg.norm(velocity[:, 3:], axis=1).max()),
            'continuous_table_contact': all(item[2] for item in history),
        }
        if (metrics['continuous_table_contact']
                and metrics['position_box_diagonal_m'] <= criteria['maximum_position_box_diagonal_m']
                and metrics['orientation_diameter_bound_rad'] <= criteria['maximum_orientation_diameter_bound_rad']
                and metrics['maximum_linear_speed_m_s'] <= criteria['maximum_linear_speed_m_s']
                and metrics['maximum_angular_speed_rad_s'] <= criteria['maximum_angular_speed_rad_s']):
            metrics.update(accepted=True, physics_steps=step + 1, sample_count=len(history),
                           sample_span_s=(len(history) - 1) * model.opt.timestep,
                           simulation_time_s=float(data.time),
                           velocity_at_capture_m_s_rad_s=data.qvel.tolist(),
                           velocity_policy='initial velocity declared; no post-step state or velocity edits')
            return metrics
    raise ValueError('cube did not settle within configured engineering tolerances')


def capture_arrays(model, data, camera):
    with mujoco.Renderer(model, height=camera['height'], width=camera['width']) as renderer:
        renderer.update_scene(data, camera='capture')
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera='capture')
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera='capture')
        segmentation = renderer.render().copy()
    geometry = np.where(segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
                        segmentation[:, :, 0], -1)
    # Background carries far-plane depth in the renderer; exported invalids are zero.
    depth = np.where((geometry >= 0) & np.isfinite(depth) & (depth > 0)
                     & (depth < camera['far_m']), depth, 0).astype(np.float32)
    return rgb, depth, geometry


def ray_box_depth(origin, directions, center, rotation, half_sizes):
    """Independent NumPy slab intersection, returning ray parameter t.

    Directions have camera z=1 before rotation, so t is optical-axis depth,
    not Euclidean range. Misses return infinity. This does not call mj_ray.
    """
    local_origin = (np.asarray(origin) - center) @ rotation
    local_direction = np.asarray(directions) @ rotation
    parallel = np.abs(local_direction) < 1e-14
    safe_direction = np.where(parallel, 1., local_direction)
    first = (-np.asarray(half_sizes) - local_origin) / safe_direction
    second = (np.asarray(half_sizes) - local_origin) / safe_direction
    lo, hi = np.minimum(first, second), np.maximum(first, second)
    lo = np.where(parallel, -np.inf, lo)
    hi = np.where(parallel, np.inf, hi)
    invalid = np.any(parallel & (np.abs(local_origin) > half_sizes), axis=1)
    entry, exit = lo.max(axis=1), hi.min(axis=1)
    distance = np.where(entry >= 0, entry, exit)
    return np.where((exit >= entry) & (exit >= 0) & ~invalid, distance, np.inf)


def _interior(mask, iterations):
    mask = mask.copy()
    for _ in range(iterations):
        reduced = np.zeros_like(mask)
        reduced[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                               & mask[1:-1, :-2] & mask[1:-1, 2:])
        mask = reduced
    return mask


def geometry_checks(model, data, depth, geometry, transform, intrinsic, config):
    """Compare rendered interiors to analytical boxes, engine rays and raw GL camera."""
    ids, tolerance = _ids(model), config['validation']
    checks = {}
    raw_gl_rotation = data.cam_xmat[ids['camera']].reshape(3, 3)
    raw_camera_position = data.cam_xpos[ids['camera']]
    camera = config['camera']
    focal = camera['height'] / (2 * np.tan(np.deg2rad(model.cam_fovy[ids['camera']]) / 2))
    for name in ('cube', 'table'):
        geom_id = ids[name]
        yy, xx = np.where(_interior((geometry == geom_id) & (depth > 0),
                                   tolerance['interior_erosion_pixels']))
        if len(xx) < 4:
            raise ValueError(f'{name} has insufficient visible interior pixels')
        samples = np.linspace(0, len(xx) - 1, min(len(xx), tolerance['maximum_sample_pixels_per_geometry']), dtype=int)
        xx, yy = xx[samples], yy[samples]
        pixels = np.column_stack([xx, yy, np.ones(len(xx))])
        camera_rays = np.linalg.solve(intrinsic, pixels.T).T
        rays = camera_rays @ transform[:3, :3].T
        analytic = ray_box_depth(transform[:3, 3], rays, data.geom_xpos[geom_id],
                                 data.geom_xmat[geom_id].reshape(3, 3), model.geom_size[geom_id])
        if not np.isfinite(analytic).all():
            raise ValueError(f'analytical rays miss rendered {name}')
        analytic_errors = np.abs(analytic - depth[yy, xx])
        engine_errors = []
        for direction, measured_depth in zip(rays, depth[yy, xx]):
            scale = np.linalg.norm(direction)
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, transform[:3, 3], direction / scale,
                                     None, True, -1, hit)
            if hit[0] != geom_id or distance < 0:
                raise ValueError(f'engine ray hit does not agree with {name} segmentation')
            engine_errors.append(abs(distance / scale - float(measured_depth)))
        world_hits = transform[:3, 3] + rays * analytic[:, None]
        gl_points = (world_hits - raw_camera_position) @ raw_gl_rotation
        reprojected = np.column_stack([
            focal * gl_points[:, 0] / -gl_points[:, 2] + (camera['width'] - 1) / 2,
            focal * -gl_points[:, 1] / -gl_points[:, 2] + (camera['height'] - 1) / 2])
        reprojection_error = np.linalg.norm(reprojected - pixels[:, :2], axis=1)
        checks[name] = {
            'sample_count': len(samples),
            'maximum_analytical_depth_error_m': float(analytic_errors.max()),
            'maximum_mujoco_ray_depth_error_m': float(max(engine_errors)),
            'maximum_raw_camera_reprojection_error_pixels': float(reprojection_error.max()),
        }
        if max(analytic_errors.max(), max(engine_errors)) > tolerance['maximum_depth_error_m']:
            raise ValueError(f'{name} depth does not agree with independent geometry/rays')
        if reprojection_error.max() > tolerance['maximum_reprojection_error_pixels']:
            raise ValueError(f'{name} projection does not agree with raw MuJoCo camera')
    return checks


def _integration_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def reload_saved_scene(directory):
    """Load editable XML plus verified complete integration state, without a hand.

    Reproducibility is verified for the same MuJoCo version and runtime. GPU
    rasterization and physics need not be bitwise identical across platforms.
    """
    root = Path(directory)
    metadata = json.loads((root / 'scene_state_metadata.json').read_text())
    if metadata['mujoco_version'] != mujoco.__version__:
        raise ValueError('saved state requires the recorded MuJoCo version')
    for name in ('scene.xml', 'scene_config.json', 'scene_state.npz'):
        if _hash(root / name) != metadata['sha256'][name]:
            raise ValueError(f'saved artifact changed: {name}')
    load_config(root / 'scene_config.json')
    model = mujoco.MjModel.from_xml_path(str(root / 'scene.xml'))
    data = mujoco.MjData(model)
    with np.load(root / 'scene_state.npz', allow_pickle=False) as saved:
        if set(saved.files) != {'integration_state', 'state_spec'} or saved['state_spec'].shape != ():
            raise ValueError('invalid integration-state archive')
        state = saved['integration_state']
        if (int(saved['state_spec']) != STATE_SPEC or state.dtype != np.float64
                or state.shape != (mujoco.mj_stateSize(model, STATE_SPEC),)
                or not np.isfinite(state).all()):
            raise ValueError('invalid saved integration state')
        mujoco.mj_setState(model, data, state, STATE_SPEC)
    mujoco.mj_forward(model, data)
    return model, data


def _physical_measurements(model, data, config):
    ids = _ids(model)
    cube_rotation = data.geom_xmat[ids['cube']].reshape(3, 3)
    extent = np.abs(cube_rotation) @ model.geom_size[ids['cube']]
    position = data.geom_xpos[ids['cube']]
    table_top = data.geom_xpos[ids['table'], 2] + model.geom_size[ids['table'], 2]
    support_gap = position[2] - extent[2] - table_top
    if abs(support_gap) > config['settling']['maximum_support_penetration_m']:
        raise ValueError('settled cube bottom does not agree with table support plane')
    if np.any(np.abs(position[:2] - data.geom_xpos[ids['table'], :2]) + extent[:2]
              >= model.geom_size[ids['table'], :2]):
        raise ValueError('settled cube left the table footprint')
    return {
        'cube_dimensions_m': (2 * model.geom_size[ids['cube']]).tolist(),
        'cube_mass_kg': float(model.body_mass[ids['body']]),
        'cube_inertia_kg_m2': model.body_inertia[ids['body']].tolist(),
        'cube_position_m': position.tolist(),
        'cube_orientation_wxyz': data.qpos[3:7].tolist(),
        'table_dimensions_m': (2 * model.geom_size[ids['table']]).tolist(),
        'table_top_z_m': float(table_top), 'cube_bottom_support_gap_m': float(support_gap),
        'gravity_m_s2': model.opt.gravity.tolist(), 'timestep_s': float(model.opt.timestep),
        'geometry_friction': {key: model.geom_friction[ids[key]].tolist() for key in ('cube', 'table')},
        'contacts': [{'distance_m': float(item.dist), 'friction': item.friction.tolist(),
                      'dimension': int(item.dim)} for item in data.contact
                     if {int(item.geom1), int(item.geom2)} == {ids['cube'], ids['table']}],
        'body_count': model.nbody, 'geometry_count': model.ngeom,
        'no_hand_or_actuators': bool(model.nbody == 2 and model.ngeom == 2 and model.nu == 0),
    }


def _write_readable_report(output, report):
    physics, settling = report['physics'], report['settling']
    lines = [
        'Phase 1 Part 1: synthetic scene preparation', '',
        'Status: software scene/observation checks passed.',
        'Scope: hand-free simulation only. No HUG execution, grasp trial, physical grasp, training or prediction claims.',
        '', 'Configured test assumptions:', *[f'- {item}' for item in report['assumptions']], '',
        f"Cube dimensions (m): {physics['cube_dimensions_m']}",
        f"Cube mass (kg): {physics['cube_mass_kg']}",
        f"Cube inertia (kg m^2): {physics['cube_inertia_kg_m2']}",
        f"Cube capture position (m): {physics['cube_position_m']}",
        f"Cube orientation (wxyz): {physics['cube_orientation_wxyz']}",
        f"Cube bottom minus table top (m): {physics['cube_bottom_support_gap_m']:.9g}",
        f"Cube/table friction [sliding, torsional, rolling]: {physics['geometry_friction']}",
        f"Gravity (m/s^2): {physics['gravity_m_s2']}",
        f"Settled after {settling['physics_steps']} steps ({settling['simulation_time_s']:.6g} s).",
        f"Trailing samples: {settling['sample_count']}; span {settling['sample_span_s']:.6g} s; continuous table contact: {settling['continuous_table_contact']}.",
        f"Position drift diagonal (m): {settling['position_box_diagonal_m']:.9g}",
        f"Orientation diameter bound (rad): {settling['orientation_diameter_bound_rad']:.9g}",
        f"Maximum trailing linear/angular speeds: {settling['maximum_linear_speed_m_s']:.9g} m/s, {settling['maximum_angular_speed_rad_s']:.9g} rad/s.",
        f"Capture velocity [linear, angular]: {settling['velocity_at_capture_m_s_rad_s']}",
        'Velocities are measured, never forced to zero after gravity settling.', '',
        'Camera: x right, y down, z forward; T_world_camera maps camera to world.',
        'Depth: optical-axis meters in depth_m.npy; rounded uint16 millimeters in depth.png; zero is invalid.',
        f"Resolution/FOV: {report['camera']['width']} x {report['camera']['height']}, {report['camera']['vertical_fov_degrees']} degrees vertical.",
        f"Camera position/target (m): {report['camera']['position_m']} / {report['camera']['actual_target_m']}",
        f"Projected actual cube center (pixels): {report['camera']['projected_target_uv']}", '',
        'Independent surface checks (interior pixels, excluding silhouette rasterization):',
    ]
    for name, checks in report['geometry_checks'].items():
        lines.append(f"- {name}: {checks['sample_count']} samples; analytical box depth error {checks['maximum_analytical_depth_error_m']:.9g} m; MuJoCo ray depth error {checks['maximum_mujoco_ray_depth_error_m']:.9g} m; raw-camera reprojection error {checks['maximum_raw_camera_reprojection_error_pixels']:.9g} pixels.")
    lines += ['', f"Saved XML/state reload checks: {json.dumps(report['reload'], sort_keys=True)}",
              'Reload scene.xml and full scene_state.npz through reload_saved_scene(output_directory).',
              'The XML also includes a capture keyframe for manual inspection; full integration state is required for exact continuation.',
              f"Runtime scope: MuJoCo {report['runtime']['mujoco_version']}; {report['runtime']['platform']}.",
              'Cross-platform/driver bitwise reproducibility is not asserted. Engineering tolerances are saved in scene_config.json.',
              'Files are synthetic and contain no external or licensed assets.', '']
    (output / 'validation_report.txt').write_text('\n'.join(lines))


def prepare_scene(config_path, output_directory):
    config = load_config(config_path)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError('scene output must be a new directory; existing output is never overwritten')
    model = mujoco.MjModel.from_xml_string(scene_xml(config))
    data = mujoco.MjData(model)
    settling = settle_scene(model, data, config)
    physics = _physical_measurements(model, data, config)
    ids = _ids(model)
    target = data.geom_xpos[ids['cube']].copy()
    quaternion, transform, intrinsic = camera_calibration(config['camera'], target)
    model.cam_quat[ids['camera']] = quaternion
    mujoco.mj_forward(model, data)
    rgb, depth, geometry = capture_arrays(model, data, config['camera'])
    mask = geometry == ids['cube']
    yy, xx = np.where(mask & (depth > 0))
    if not len(xx):
        raise ValueError('cube not visible with valid depth')
    selection_index = np.argmin((xx - xx.mean()) ** 2 + (yy - yy.mean()) ** 2)
    selection = [int(xx[selection_index]), int(yy[selection_index])]
    preflight = validate_arrays(rgb, depth, intrinsic, mask, selection,
                                 registered=True, calibration_matches=True)
    if not preflight['preflight_passed']:
        raise ValueError(str(preflight['errors']))
    if np.any(depth >= 65.5345):
        raise ValueError('metric depth cannot be represented by the uint16 millimeter encoding')
    encoded_depth = np.rint(depth * 1000).astype(np.uint16)
    if np.any((depth > 0) & (encoded_depth == 0)):
        raise ValueError('positive depth rounds to reserved invalid zero in millimeter encoding')
    encoding = validate_depth_encoding(depth, encoded_depth)
    independent_checks = geometry_checks(model, data, depth, geometry, transform, intrinsic, config)
    # No filesystem changes until configuration, gravity settling and sensor checks pass.
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'scene_config.json', config)
    (output / 'scene.xml').write_text(scene_xml(config, target=target, capture=data))
    state = _integration_state(model, data)
    np.savez_compressed(output / 'scene_state.npz', integration_state=state,
                        state_spec=np.array(STATE_SPEC, dtype=np.int32))
    _json(output / 'scene_state_metadata.json', {
        'schema_version': 1, 'mujoco_version': mujoco.__version__,
        'state_kind': 'mjSTATE_INTEGRATION',
        'sha256': {name: _hash(output / name) for name in ('scene.xml', 'scene_config.json', 'scene_state.npz')},
    })
    Image.fromarray(rgb).save(output / 'rgb.png')
    Image.fromarray(encoded_depth).save(output / 'depth.png')
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / 'object_mask.png')
    for name, array in (('depth_m.npy', depth), ('intrinsics.npy', intrinsic),
                         ('T_world_camera.npy', transform)):
        np.save(output / name, array)
    reloaded_model, reloaded_data = reload_saved_scene(output)
    reload_state_error = float(np.max(np.abs(_integration_state(reloaded_model, reloaded_data) - state)))
    reload_rgb, reload_depth, reload_geometry = capture_arrays(reloaded_model, reloaded_data, config['camera'])
    reload = {
        'maximum_integration_state_error': reload_state_error,
        'maximum_rgb_channel_error': int(np.abs(reload_rgb.astype(int) - rgb.astype(int)).max()),
        'maximum_depth_error_m': float(np.abs(reload_depth - depth).max()),
        'segmentation_identical': bool(np.array_equal(reload_geometry, geometry)),
        'reload_method': 'standalone scene.xml + mjSTATE_INTEGRATION state + mj_forward',
    }
    tolerances = config['validation']
    if (reload['maximum_integration_state_error'] > tolerances['maximum_reload_state_error']
            or reload['maximum_rgb_channel_error'] > tolerances['maximum_reload_rgb_channel_error']
            or reload['maximum_depth_error_m'] > tolerances['maximum_reload_depth_error_m']
            or not reload['segmentation_identical']):
        raise ValueError('saved scene/state failed independent reload consistency check')
    source = {'kind': 'synthetic_mujoco_cube', 'mujoco_version': mujoco.__version__,
              'scene_xml_sha256': _hash(output / 'scene.xml'),
              'scene_config_sha256': _hash(output / 'scene_config.json'),
              'assets': 'built-in primitives only', 'assumptions': config['assumptions']}
    manifest = {'rgb_path': 'rgb.png', 'depth_m_path': 'depth_m.npy', 'depth_path': 'depth.png',
                'intrinsics_path': 'intrinsics.npy', 'mask_path': 'object_mask.png',
                'T_world_camera_path': 'T_world_camera.npy', 'selection_uv': selection,
                'depth_units': 'meters', 'depth_png_units': 'millimeters',
                'depth_semantics': 'optical_axis_z', 'invalid_depth_value': 0,
                'registered_to_rgb': True, 'intrinsics_at_rgb_resolution': True, 'source': source}
    _json(output / 'observation_manifest.json', manifest)
    observation_checks = check_manifest(output / 'observation_manifest.json', output / 'observation_checks')
    if not observation_checks['preflight_passed']:
        raise ValueError(str(observation_checks['errors']))
    camera_point = transform[:3, :3].T @ (target - transform[:3, 3])
    projected = intrinsic @ camera_point
    report = {
        'schema_version': 1, 'scene_preparation_checks_passed': True,
        'assumptions': config['assumptions'], 'physics': physics, 'settling': settling,
        'camera': {**config['camera'], 'actual_target_m': target.tolist(),
                   'projected_target_uv': (projected[:2] / projected[2]).tolist(),
                   'camera_convention': 'x right, y down, z forward',
                   'T_world_camera_semantics': 'camera-to-world rigid transform'},
        'geometry_checks': independent_checks, 'reload': reload, 'depth_encoding': encoding,
        'observation_preflight_passed': observation_checks['preflight_passed'],
        'object_pixels': int(mask.sum()), 'selection_uv': selection,
        'runtime': {'mujoco_version': mujoco.__version__, 'numpy_version': np.__version__,
                    'platform': platform.platform()},
        'scope': {'hand_present': False, 'hug_executed': False, 'grasp_trials_executed': False,
                  'physical_grasp_tested': False, 'training_executed': False,
                  'prediction_executed': False, 'scientific_thresholds_validated': False},
        'artifact_sha256': {path.name: _hash(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    _json(output / 'validation_report.json', report)
    _write_readable_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare_scene(args.config, args.output)
    except (ValueError, KeyError, OSError) as exc:
        print(f'Scene preparation rejected: {exc}')
        return 1
    print(json.dumps({'scene_preparation_checks_passed': report['scene_preparation_checks_passed'],
                      'output': str(Path(args.output).resolve()), 'scope': report['scope']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
