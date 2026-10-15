"""Capture fresh calibrated HUG inputs from a declared object/scene case."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from grasp_failure_prediction.evaluation.case_runner import (
    _geometry_content_hash, _resolve_object_geometry, _runner_class,
    _validate_case_configuration, load_case,
)
from grasp_failure_prediction.evaluation.registry import resolve_case
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
from grasp_failure_prediction.evaluation.scene_contracts import build_scene_contract
from .observations import check_manifest, validate_depth_encoding


INPUT_FILES = ('rgb.png', 'depth.png', 'depth_m.npy', 'intrinsics.npy',
               'object_mask.png', 'T_world_camera.npy')
SETTLING_WINDOW_SAMPLES = 500
SETTLING_MAX_PHYSICS_STEPS = 2000
SETTLING_MAX_POSITION_DIAGONAL_M = .0001
SETTLING_MAX_ORIENTATION_DIAMETER_RAD = .002
SETTLING_WORKSPACE_HALF_WIDTH_M = .3


def _json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _hash(path):
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()


def camera_calibration(position, target, *, size=224, fovy_degrees=45.):
    position, target = np.asarray(position, dtype=float), np.asarray(target, dtype=float)
    if position.shape != (3,) or target.shape != (3,) or not np.isfinite([position, target]).all():
        raise ValueError('camera position and target must be finite three-vectors')
    forward = target - position
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError('camera must be separated from its target')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-8:
        raise ValueError('camera view must not be parallel to world up')
    right /= np.linalg.norm(right)
    rotation_gl = np.column_stack([right, np.cross(right, forward), -forward])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation_gl.reshape(9))
    transform = np.eye(4)
    transform[:3, :3] = rotation_gl @ np.diag([1., -1., -1.])
    transform[:3, 3] = position
    focal = size / (2 * np.tan(np.deg2rad(fovy_degrees) / 2))
    intrinsic = np.array([[focal, 0., (size - 1) / 2],
                          [0., focal, (size - 1) / 2], [0., 0., 1.]])
    return quaternion, transform, intrinsic


def object_segmentation_mask(segmentation, object_geom_ids):
    return (np.isin(segmentation[:, :, 0], list(object_geom_ids))
            & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))


def evaluate_settling_window(poses_wxyz, velocities, object_table_contact, hand_object_contact):
    """Measure the trailing 500 object states without changing their dynamics.

    Poses are [x,y,z,qw,qx,qy,qz], velocities are linear/angular six-vectors,
    and contacts are boolean vectors. Invalid states raise; short windows fail.
    At the registered 2 ms timestep, 500 samples cover one second. Geometric
    drift and continuous contact determine acceptance; speeds are recorded.
    Quaternion signs are equivalent. Near-unit quaternions are normalized only
    for measuring angles, never written to the simulator.
    """
    poses = np.asarray(poses_wxyz, dtype=np.float64)
    velocity = np.asarray(velocities, dtype=np.float64)
    table, hand = np.asarray(object_table_contact), np.asarray(hand_object_contact)
    if (poses.ndim != 2 or poses.shape[1] != 7 or not len(poses)
            or velocity.shape != (len(poses), 6)):
        raise ValueError('settling poses and velocities must be nonempty Nx7 and Nx6 arrays')
    if (table.shape != (len(poses),) or hand.shape != (len(poses),)
            or table.dtype != np.bool_ or hand.dtype != np.bool_):
        raise ValueError('settling contacts must be boolean vectors matching the states')
    if not np.isfinite(poses).all() or not np.isfinite(velocity).all():
        raise ValueError('settling states must be finite')
    norms = np.linalg.norm(poses[:, 3:], axis=1)
    if not np.isfinite(norms).all() or np.any(np.abs(norms - 1.) > 1e-6):
        raise ValueError('settling orientations must be valid unit quaternions')
    poses, velocity, norms, table, hand = (
        value[-SETTLING_WINDOW_SAMPLES:] for value in (poses, velocity, norms, table, hand))
    orientation = poses[:, 3:] / norms[:, None]
    first = orientation[0].copy()
    orientation *= np.where(orientation @ first < 0, -1., 1.)[:, None]
    angles = 4 * np.arctan2(np.linalg.norm(orientation - first, axis=1),
                           np.linalg.norm(orientation + first, axis=1))
    maximum_angle = float(angles.max())
    maxima = {
        'sample_count': len(poses),
        'maximum_linear_speed_m_s': float(np.linalg.norm(velocity[:, :3], axis=1).max()),
        'maximum_angular_speed_rad_s': float(np.linalg.norm(velocity[:, 3:], axis=1).max()),
        'position_box_diagonal_m': float(np.linalg.norm(np.ptp(poses[:, :3], axis=0))),
        'maximum_quaternion_angle_from_first_rad': maximum_angle,
        'quaternion_diameter_bound_rad': 2 * maximum_angle,
        'maximum_abs_workspace_xy_m': float(np.abs(poses[:, :2]).max()),
        'all_object_table_contact': bool(table.all()),
        'any_hand_object_contact': bool(hand.any()),
    }
    if not all(np.isfinite(value) for value in maxima.values()):
        raise ValueError('settling state measurements must be finite')
    maxima['accepted'] = bool(
        len(poses) == SETTLING_WINDOW_SAMPLES
        and maxima['position_box_diagonal_m'] <= SETTLING_MAX_POSITION_DIAGONAL_M
        and maxima['quaternion_diameter_bound_rad'] <= SETTLING_MAX_ORIENTATION_DIAMETER_RAD
        and maxima['maximum_abs_workspace_xy_m'] < SETTLING_WORKSPACE_HALF_WIDTH_M
        and maxima['all_object_table_contact'] and not maxima['any_hand_object_contact'])
    return maxima


def capture_scene(case_path, output_dir, *, project_root=None, urdf_root=None,
                  camera_position_m=(.25, -.35, .35)):
    project = Path(project_root or Path.cwd()).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('capture output must be a fresh directory')
    case = load_case(case_path)
    resolved = resolve_case(case)
    _validate_case_configuration(case)
    geometry = _resolve_object_geometry(case, project)
    runner = _runner_class(resolved)(
        resolved.execution_protocol, urdf_root or default_dex_urdf_root(),
        physics_timestep_s=resolved.environment.simulator.physics_timestep_s,
        control_timestep_s=resolved.environment.simulator.control_timestep_s,
        object_geometry=geometry,
    )
    runner.reset(seed=case.seed, object_mass_kg=case.object.mass_kg,
                 object_position_m=np.asarray(case.initial_condition.object_position_m),
                 object_orientation_wxyz=np.asarray(case.initial_condition.object_orientation_xyzw)[[3,0,1,2]])
    # Keep the hand and its ideal arm target above the table while the object settles.
    root = runner._root_qpos_address
    runner.data.qpos[root:root+3] = [0., 0., 1.]
    runner.data.qpos[root+3:root+7] = [1., 0., 0., 0.]
    if runner.model.nmocap:
        runner.data.mocap_pos[0] = [0., 0., 1.]
        runner.data.mocap_quat[0] = [1., 0., 0., 0.]
    mujoco.mj_forward(runner.model, runner.data)
    joint = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_JOINT, 'object_free')
    dof = int(runner.model.jnt_dofadr[joint])
    address = runner._object_qpos_address
    poses, velocities, table_contacts, hand_contacts = (
        deque(maxlen=SETTLING_WINDOW_SAMPLES) for _ in range(4))
    for settle_step in range(SETTLING_MAX_PHYSICS_STEPS):
        mujoco.mj_step(runner.model, runner.data)
        if not all(np.isfinite(value).all() for value in (runner.data.qpos, runner.data.qvel, runner.data.qacc)):
            raise ValueError('simulation produced nonfinite capture state while settling')
        if any(runner.data.warning[warning].number for warning in (
                mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC)):
            raise ValueError('simulation produced invalid capture state while settling')
        hand_object, _, object_table, _, _ = runner._contact_metrics()
        poses.append(runner.data.qpos[address:address+7].copy())
        velocities.append(runner.data.qvel[dof:dof+6].copy())
        table_contacts.append(object_table)
        hand_contacts.append(hand_object)
        settling_window = evaluate_settling_window(poses, velocities, table_contacts, hand_contacts)
        if settling_window['accepted']:
            break
    else:
        raise ValueError('object did not settle to a sufficiently stationary capture pose')
    mujoco.mj_forward(runner.model, runner.data)
    actual_pose = runner.data.qpos[address:address+7].copy()
    actual_mass = float(runner.model.body_mass[runner._object_body_id])
    camera_id = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_CAMERA, 'front')
    quaternion, transform, intrinsic = camera_calibration(camera_position_m, actual_pose[:3])
    runner.model.cam_pos[camera_id] = camera_position_m
    runner.model.cam_quat[camera_id] = quaternion
    runner.model.cam_fovy[camera_id] = 45.
    mujoco.mj_forward(runner.model, runner.data)
    object_geoms = np.flatnonzero(runner.model.geom_bodyid == runner._object_body_id)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1,1,1,0,0,0]
    with mujoco.Renderer(runner.model, height=224, width=224) as renderer:
        renderer.update_scene(runner.data, camera='front', scene_option=option)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(runner.data, camera='front', scene_option=option)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(runner.data, camera='front', scene_option=option)
        segmentation = renderer.render().copy()
    mask = object_segmentation_mask(segmentation, object_geoms)
    valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
    depth = np.where(valid, depth, 0.).astype(np.float32)
    yy, xx = np.where(mask & valid)
    if not len(xx):
        raise ValueError('object is not visible with valid depth')
    selected = np.argmin((xx-xx.mean())**2 + (yy-yy.mean())**2)
    selection = [int(xx[selected]), int(yy[selected])]
    # Check interior visible pixels against independent MuJoCo ray distances.
    interior = mask.copy()
    for _ in range(2):
        reduced = np.zeros_like(interior)
        reduced[1:-1,1:-1] = (interior[1:-1,1:-1] & interior[:-2,1:-1]
                              & interior[2:,1:-1] & interior[1:-1,:-2] & interior[1:-1,2:])
        interior = reduced
    iy, ix = np.where(interior & valid)
    if not len(ix):
        raise ValueError('object has no interior pixels for calibration validation')
    errors = []
    for index in np.linspace(0, len(ix)-1, min(128, len(ix)), dtype=int):
        pixel = np.array([ix[index], iy[index], 1.])
        point = np.linalg.solve(intrinsic, pixel) * float(depth[iy[index],ix[index]])
        direction = transform[:3,:3] @ point
        expected_distance = float(np.linalg.norm(direction))
        direction /= expected_distance
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(runner.model, runner.data, transform[:3,3], direction,
                                 option.geomgroup, True, -1, hit)
        if hit[0] not in object_geoms or distance < 0:
            raise ValueError('calibrated ray did not hit the rendered object')
        errors.append(abs(float(distance)-expected_distance))
    if max(errors) > .002:
        raise ValueError('calibrated depth differs from object ray surface by more than 2 mm')
    output.mkdir(parents=True)
    Image.fromarray(rgb).save(output / 'rgb.png')
    Image.fromarray(mask.astype(np.uint8)*255).save(output / 'object_mask.png')
    encoded = np.rint(depth*1000).astype(np.uint16)
    validate_depth_encoding(depth, encoded)
    Image.fromarray(encoded).save(output / 'depth.png')
    np.save(output / 'depth_m.npy', depth)
    np.save(output / 'intrinsics.npy', intrinsic)
    np.save(output / 'T_world_camera.npy', transform)
    mujoco.mj_saveLastXML(str(output / 'scene.xml'), runner.model)
    mujoco.mj_saveModel(runner.model, str(output / 'scene.mjb'))
    np.savez_compressed(output / 'scene_state.npz', qpos=runner.data.qpos, qvel=runner.data.qvel)
    source = {'kind':'mujoco_render', 'object':case.object.id, 'seed':case.seed,
              'geometry_content_hash':_geometry_content_hash(geometry),
              'mujoco_version':mujoco.__version__}
    manifest = {'rgb_path':'rgb.png', 'depth_m_path':'depth_m.npy', 'intrinsics_path':'intrinsics.npy',
                'mask_path':'object_mask.png', 'selection_uv':selection, 'depth_units':'meters',
                'registered_to_rgb':True, 'intrinsics_at_rgb_resolution':True, 'source':source}
    _json(output / 'observation_manifest.json', manifest)
    checks = check_manifest(output / 'observation_manifest.json', output / 'checks')
    if not checks['preflight_passed']:
        raise ValueError(checks['errors'])
    hashes = {name:_hash(output/name) for name in INPUT_FILES}
    contract = build_scene_contract(
        object_id=case.object.id, geometry_content_hash=_geometry_content_hash(geometry),
        mass_kg=actual_mass, object_position_m=actual_pose[:3].tolist(),
        object_orientation_xyzw=actual_pose[[4,5,6,3]].tolist(),
        environment_config_hash=resolved.environment_config_hash,
        camera_sha256=hashes['T_world_camera.npy'], observation_files_sha256=hashes,
    )
    _json(output / 'scene_contract.json', contract)
    report = {'input_checks_passed':True, 'object_pixels':len(xx), 'selection_uv':selection,
              'maximum_ray_surface_error_m':max(errors), 'surface_check_pixels':len(errors),
              'actual_object_position_m':actual_pose[:3].tolist(),
              'actual_object_orientation_xyzw':actual_pose[[4,5,6,3]].tolist(),
              'actual_object_mass_kg':actual_mass, 'settling_physics_steps':settle_step+1,
              'settling_criteria': {
                  'window_samples':SETTLING_WINDOW_SAMPLES,
                  'window_duration_s':SETTLING_WINDOW_SAMPLES*runner.physics_timestep_s,
                  'window_sample_span_s':(SETTLING_WINDOW_SAMPLES-1)*runner.physics_timestep_s,
                  'maximum_physics_steps':SETTLING_MAX_PHYSICS_STEPS,
                  'maximum_settling_duration_s':SETTLING_MAX_PHYSICS_STEPS*runner.physics_timestep_s,
                  'position_box_diagonal_at_most_m':SETTLING_MAX_POSITION_DIAGONAL_M,
                  'quaternion_diameter_bound_at_most_rad':SETTLING_MAX_ORIENTATION_DIAMETER_RAD,
                  'quaternion_bound_method':'2 * maximum sign-invariant angle from first sample',
                  'workspace_abs_xy_strictly_below_m':SETTLING_WORKSPACE_HALF_WIDTH_M,
                  'continuous_object_table_contact':True, 'no_hand_object_contact':True,
                  'velocity_policy':'finite velocities are recorded without speed thresholds',
                  'contact_diagnostic_lag_s':runner.physics_timestep_s,
              },
              'accepted_settling_window':settling_window,
              'object_velocity_at_capture':runner.data.qvel[dof:dof+6].tolist(),
              'camera_convention':'x right, y down, z forward',
              'scene_binary_sha256':_hash(output/'scene.mjb'),
              'scene_contract_sha256':_hash(output/'scene_contract.json')}
    _json(output / 'capture_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--output', required=True)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--camera-position', type=float, nargs=3, default=[.25,-.35,.35])
    args = parser.parse_args()
    print(json.dumps(capture_scene(args.case, args.output, project_root=args.project_root,
                                   camera_position_m=args.camera_position), indent=2))


if __name__ == '__main__':
    main()
