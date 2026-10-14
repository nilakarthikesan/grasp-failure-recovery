"""Static FK of 32 EXISTING saved nominal HUG translations; never step time.

No HUG learned model, Dex optimization, new proposals or evaluation executions.
The input index is a hash/provenance inventory of preexisting experiment files.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import pickle
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def forbid_integration(*args, **kwargs):
    raise RuntimeError('Saved-pose FK audit forbids MuJoCo time integration')


for forbidden_api in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, forbidden_api, forbid_integration)

# Guards above are installed before importing the project or loading any model.
from grasp_failure_prediction.evaluation.pose_validation import _link_positions, TASK_LINK_NAMES, ShadowPoseValidator, _body_id
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root, RetargetedHandPose
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT

OUT = Path('/private/tmp/grasp-hug-existing-inventory-2026-10-14')
INDEX_PATH = OUT / 'existing_hug_inventory.json'
INDICES = np.array([4, 8, 12, 16, 20, 2, 6, 10, 14, 18])
FINGERS = ['thumb', 'index', 'middle', 'ring', 'little']
URDF = default_dex_urdf_root() / 'shadow_hand/shadow_hand_right.urdf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ident(model, kind, name):
    result = int(mujoco.mj_name2id(model, kind, name))
    if result < 0:
        raise ValueError(f'Missing {kind}: {name}')
    return result


def main():
    index = json.loads(INDEX_PATH.read_text())
    assert index['summary']['all_candidate_binding_and_mapping_hash_checks_pass']
    assert index['summary']['unique_modern_native_hashes'] == 32
    # This fixed-root URDF preserves world_joint's declared 1.57079 rad origin.
    # A floating carrier quaternion cannot be treated as that base frame.
    validator = ShadowPoseValidator(default_dex_urdf_root())
    records = []
    for case in index['modern_cases']:
        native_path = Path(case['native']['path'])
        command = case['stored_shadow_command']
        assert sha(native_path) == case['native']['sha256']
        assert sha(command['path']) == command['sha256']
        assert sha(case['scene_mjb']['path']) == case['scene_mjb']['sha256']
        native = pickle.loads(native_path.read_bytes())
        contract = json.loads(Path(case['frame_files']['scene_contract.json']['path']).read_text())
        T_wc = np.load(case['frame_files']['T_world_camera.npy']['path'], allow_pickle=False)
        T_wm = T_wc @ np.asarray(native['T_camera_wrist'])
        R_operator = T_wm[:3, :3] @ MANO_TO_OPERATOR_RIGHT.T
        native_world = np.asarray(native['landmarks_3d']) @ T_wc[:3, :3].T + T_wc[:3, 3]
        with np.load(command['path'], allow_pickle=False) as actions:
            row = command['row_index']
            names = actions['joint_names'].tolist()
            target = actions['nominal_target_rad'][row].copy()
            root_pos = actions['root_target_position_m'][row].copy()
            root_quat = actions['root_target_quaternion_wxyz'][row].copy()
        model = mujoco.MjModel.from_binary_path(case['scene_mjb']['path'])
        data = mujoco.MjData(model)
        with np.load(case['frame_files']['scene_state.npz']['path'], allow_pickle=False) as state:
            assert state['qpos'].shape == data.qpos.shape
            data.qpos[:] = state['qpos']
        data.qvel[:] = 0.0
        root_id = ident(model, mujoco.mjtObj.mjOBJ_JOINT, 'world_joint')
        object_joint = ident(model, mujoco.mjtObj.mjOBJ_JOINT, 'object_free')
        root_address = int(model.jnt_qposadr[root_id])
        object_address = int(model.jnt_qposadr[object_joint])
        object_qpos_initial = data.qpos[object_address:object_address + 7].copy()
        data.qpos[root_address:root_address + 3] = root_pos
        data.qpos[root_address + 3:root_address + 7] = root_quat
        joint_limit_checks = []
        for name, value in zip(names, target, strict=True):
            joint = ident(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            limits = model.jnt_range[joint]
            data.qpos[int(model.jnt_qposadr[joint])] = value
            joint_limit_checks.append({'joint_name': name, 'value_rad': float(value),
                                       'limits_rad': limits.tolist(),
                                       'violation_rad': float(max(limits[0] - value, value - limits[1], 0.0))})
        if model.nmocap:
            data.mocap_pos[0] = root_pos
            data.mocap_quat[0] = root_quat
        mujoco.mj_forward(model, data)
        assert data.time == 0.0
        assert np.array_equal(data.qpos[object_address:object_address + 7], object_qpos_initial)
        palm = ident(model, mujoco.mjtObj.mjOBJ_BODY, 'palm')
        object_body = ident(model, mujoco.mjtObj.mjOBJ_BODY, 'object')
        shadow_world = _link_positions(model, data, URDF)
        palm_world = data.xpos[palm].copy()
        palm_rotation_world = data.xmat[palm].reshape(3, 3).copy()
        root_rotation_world = Rotation.from_quat(root_quat[[1, 2, 3, 0]]).as_matrix()
        object_position_world = data.xpos[object_body].copy()
        object_rotation_world = data.xmat[object_body].reshape(3, 3).copy()
        contract_rotation = Rotation.from_quat(contract['object_orientation_xyzw']).as_matrix()
        contract_position = np.asarray(contract['object_position_m'])
        position_delta = np.linalg.norm(object_position_world - contract_position)
        rotation_delta = np.max(np.abs(object_rotation_world - contract_rotation))
        assert position_delta < 1e-12 and rotation_delta < 1e-12
        shadow_object = (shadow_world - object_position_world) @ object_rotation_world
        native_object = (native_world - object_position_world) @ object_rotation_world
        errors_world = np.linalg.norm(shadow_world - native_world[INDICES], axis=1)
        errors_object = np.linalg.norm(shadow_object - native_object[INDICES], axis=1)
        native_operator = (np.asarray(native['landmarks_3d']) - np.asarray(native['T_camera_wrist'])[:3, 3]) @ np.asarray(native['T_camera_wrist'])[:3, :3] @ MANO_TO_OPERATOR_RIGHT.T
        target_vectors = native_operator[INDICES] - native_operator[0]
        # Express achieved world vectors in the native operator/fixed-URDF base,
        # not the carrier quaternion frame whose offset includes URDF origins.
        achieved_vectors = (shadow_world - palm_world) @ R_operator
        errors_body = np.linalg.norm(achieved_vectors - target_vectors, axis=1)
        pose = RetargetedHandPose(tuple(names), target, native_operator, target_vectors,
                                  np.asarray(native['T_camera_wrist']))
        validator.set_pose(pose)
        assert validator.data.time == 0.0
        local_palm = _body_id(validator.model, 'palm')
        local_palm_rotation = validator.data.xmat[local_palm].reshape(3, 3).copy()
        fixed_vectors = _link_positions(validator.model, validator.data, URDF) - validator.data.xpos[local_palm]
        fixed_errors = np.linalg.norm(fixed_vectors - target_vectors, axis=1)
        expected_shadow_world = fixed_vectors @ R_operator.T + T_wm[:3, 3]
        previous = json.loads((Path(case['representative_zero_noise_episode']) / 'result.json').read_text())['retargeting']
        links = []
        for i, name in enumerate(TASK_LINK_NAMES):
            links.append({'shadow_link': name, 'native_landmark_index': int(INDICES[i]),
                          'finger': FINGERS[i % 5], 'landmark_type': 'tip' if i < 5 else 'middle',
                          'native_world_m': native_world[INDICES[i]].tolist(), 'shadow_world_m': shadow_world[i].tolist(),
                          'native_initial_object_m': native_object[INDICES[i]].tolist(), 'shadow_initial_object_m': shadow_object[i].tolist(),
                          'world_residual_xyz_m': (shadow_world[i] - native_world[INDICES[i]]).tolist(),
                          'world_error_m': float(errors_world[i]), 'initial_object_error_m': float(errors_object[i]),
                          'operator_body_error_m': float(errors_body[i])})
        records.append({
            'native_proposal_path': str(native_path), 'native_sha256': case['native']['sha256'],
            'object_id': case['object_id'], 'generation_seed': case['native']['seed'],
            'observation_group_id': case['observation_group_id'], 'geometry_content_hash': case['geometry_content_hash'],
            'stored_shadow_command': command, 'scene_model_path': case['scene_mjb']['path'],
            'mean_tip_world_error_m': float(errors_world[:5].mean()), 'maximum_tip_world_error_m': float(errors_world[:5].max()),
            'mean_all10_world_error_m': float(errors_world.mean()), 'maximum_all10_world_error_m': float(errors_world.max()),
            'mean_tip_operator_body_error_m': float(errors_body[:5].mean()),
            'previous_saved_mean_tip_validation_error_m': previous['mean_fingertip_error_m'],
            'previous_saved_maximum_tip_validation_error_m': previous['maximum_fingertip_error_m'],
            'previous_vs_static_body_mean_tip_error_delta_m': float(abs(errors_body[:5].mean() - previous['mean_fingertip_error_m'])),
            'fixed_urdf_vs_stored_scene_link_position_max_error_m': float(np.max(np.linalg.norm(shadow_world - expected_shadow_world, axis=1))),
            'fixed_urdf_vs_world_link_error_max_delta_m': float(np.max(np.abs(fixed_errors - errors_world))),
            'palm_position_vs_native_wrist_translation_error_m': float(np.linalg.norm(palm_world - T_wm[:3, 3])),
            'palm_rotation_vs_native_operator_times_local_palm_rotation_max_abs_error': float(np.max(np.abs(palm_rotation_world - R_operator @ local_palm_rotation))),
            'stored_carrier_vs_native_operator_rotation_max_abs_difference': float(np.max(np.abs(root_rotation_world - R_operator))),
            'stored_carrier_rotation_difference_note': 'Carrier quaternion includes fixed-URDF base offset; not an anatomical/operator rotation-error metric',
            'native_landmark_wrist_vs_T_camera_wrist_translation_error_m': float(np.linalg.norm(np.asarray(native['landmarks_3d'])[0] - np.asarray(native['T_camera_wrist'])[:3, 3])),
            'world_vs_body_link_error_max_delta_m': float(np.max(np.abs(errors_world - errors_body))),
            'world_vs_object_link_error_max_delta_m': float(np.max(np.abs(errors_world - errors_object))),
            'initial_object_qpos_unchanged': True, 'time_after_fk_s': float(data.time),
            'object_position_vs_scene_contract_error_m': float(position_delta),
            'object_rotation_vs_scene_contract_max_abs_error': float(rotation_delta),
            'joint_limit_violation_max_rad': max(r['violation_rad'] for r in joint_limit_checks),
            'joint_limit_checks': joint_limit_checks, 'links': links,
        })
    grouped = defaultdict(list)
    for r in records:
        grouped[r['object_id']].append(r)
    by_object = {}
    for object_id, items in grouped.items():
        tip_values = np.array([link['world_error_m'] for r in items for link in r['links'][:5]])
        means = np.array([r['mean_tip_world_error_m'] for r in items])
        by_object[object_id] = {'existing_native_source_count': len(items),
                                'observation_group_count': len({r['observation_group_id'] for r in items}),
                                'mean_over_all_fingertips_m': float(tip_values.mean()),
                                'worst_individual_tip_m': float(tip_values.max()),
                                'per_source_mean_tip_min_m': float(means.min()),
                                'per_source_mean_tip_max_m': float(means.max())}
    runtime_command = "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages' /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python /private/tmp/grasp-hug-existing-inventory-2026-10-14/static_saved_hug_fk.py"
    report = {
        'scope': 'Static FK of 32 already-saved original nominal HUG translations in exact captured initial object frames; no new generator samples or evaluation executions',
        'guarded_integration_apis': ['mj_step', 'mj_step1', 'mj_step2'],
        'guard_installed_before_model_load': True, 'learned_HUG_model_loaded': False, 'retarget_optimizer_used': False,
        'input_inventory_path': str(INDEX_PATH), 'input_inventory_sha256': sha(INDEX_PATH),
        'script_sha256': sha(Path(__file__)), 'runtime_command': runtime_command,
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'mujoco': mujoco.__version__},
        'task_link_urdf_path': str(URDF), 'task_link_urdf_sha256': sha(URDF),
        'task_link_order': list(TASK_LINK_NAMES), 'corresponding_native_landmark_indices': INDICES.tolist(),
        'frame_policy': 'Read stored world root pose, 24 named pre-force-close targets, initial captured qpos and T_world_camera. No origin subtraction or rescaling applied to world/object error.',
        'wrist_palm_policy_limitation': 'Legacy translation locates Shadow palm at native MANO wrist using operator axes; numerical consistency checks this implemented policy, not anatomical wrist/palm equivalence.',
        'summary_by_object': by_object,
        'global_checks': {
            'existing_unique_native_sources': len(records), 'all_time_zero': all(r['time_after_fk_s'] == 0 for r in records),
            'all_initial_object_qpos_unchanged': all(r['initial_object_qpos_unchanged'] for r in records),
            'max_joint_limit_violation_rad': max(r['joint_limit_violation_max_rad'] for r in records),
            'max_palm_vs_native_wrist_translation_error_m': max(r['palm_position_vs_native_wrist_translation_error_m'] for r in records),
            'max_palm_rotation_vs_composed_fixed_urdf_rotation_abs_error': max(r['palm_rotation_vs_native_operator_times_local_palm_rotation_max_abs_error'] for r in records),
            'max_fixed_urdf_vs_stored_scene_link_position_error_m': max(r['fixed_urdf_vs_stored_scene_link_position_max_error_m'] for r in records),
            'max_world_body_link_error_delta_m': max(r['world_vs_body_link_error_max_delta_m'] for r in records),
            'max_world_object_link_error_delta_m': max(r['world_vs_object_link_error_max_delta_m'] for r in records),
            'max_previous_vs_static_body_mean_tip_error_delta_m': max(r['previous_vs_static_body_mean_tip_error_delta_m'] for r in records),
        },
        'cases': records,
        'limitations': [
            'Historical source families are not independent held-out validation.',
            'Task-link point residuals do not establish surface/contact-patch, normal or force preservation.',
            'This uses original nominal grasp endpoint only; later force-close, noise and dynamics are excluded.',
            'Native source quality, including any existing penetration, is not evaluated or repaired here.',
        ],
    }
    output = OUT / 'static_saved_hug_fk.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'summary_by_object': by_object, 'global_checks': report['global_checks']}, indent=2))


if __name__ == '__main__':
    main()
