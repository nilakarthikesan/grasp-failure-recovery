"""HUG boundary algebra using one existing saved proposal, never new grasps.

Manufactured rigid coordinate transformations and joint-list permutations test
the implementation only. No inference, execution case, physical integration,
task label, training/evaluation record, or repository file is produced.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import scipy
from scipy.spatial.transform import Rotation


DIRECTORY = Path(__file__).resolve().parent
PROJECT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
PROPOSAL = PROJECT / 'runs/hug_grasp_eval_2026-10-13_fresh_cube/output/proposals/hug_cube/sample_0000/proposal.json'
NATIVE = PROPOSAL.parent / 'native/proposal.pkl'
EXPECTED_NATIVE_SHA256 = '573cb003458d9d7cfe85b5b4a7877e414257b3e67d5abda04a09bd6d687fa0c2'
COUNT = 64
SEED = 2026101401
# Roundoff tolerance for double-precision algebra only, not contact fidelity.
ALGEBRA_TOLERANCE = 1e-12
STEP_ATTEMPTS = []


def forbid_integration(*args, **kwargs):
    STEP_ATTEMPTS.append(True)
    raise RuntimeError('HUG boundary confirmation forbids physics integration')


for guarded_name in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, guarded_name, forbid_integration)

from grasp_failure_prediction.evaluation.pose_validation import load_shadow_model
from grasp_failure_prediction.evaluation.retargeting import (
    camera_landmarks_to_wrist, default_dex_urdf_root, reorder_for_mujoco,
)
from grasp_failure_prediction.integrations.hug import load_hug_prediction
from grasp_failure_prediction.proposals.artifacts import load_proposal
from grasp_failure_prediction.proposals.execution import _matrix, _world_palm


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scipy_transform(position, quaternion_xyzw):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    matrix[:3, 3] = position
    return matrix


def main():
    if digest(NATIVE) != EXPECTED_NATIVE_SHA256:
        raise ValueError('Existing native HUG proposal does not match its audit pin')
    proposal = load_proposal(PROPOSAL)
    native = load_hug_prediction(NATIVE)
    object_palm = scipy_transform(
        proposal.object_to_palm.translation_m,
        proposal.object_to_palm.quaternion_xyzw,
    )
    rng = np.random.default_rng(SEED)
    positions = rng.uniform(-1.0, 1.0, (COUNT, 3))
    quaternions = Rotation.random(COUNT, random_state=rng).as_quat()
    if np.any(Rotation.from_quat(quaternions).magnitude() < 1e-3):
        raise RuntimeError('Test manufacture unexpectedly produced near-identity rotation')

    translation_errors = []
    rotation_errors = []
    quaternion_norm_errors = []
    sign_errors = []
    normalization_errors = []
    for position, quaternion in zip(positions, quaternions, strict=True):
        # Lightweight coordinate carrier, not an EvaluationCase/task case.
        coordinate = SimpleNamespace(initial_condition=SimpleNamespace(
            object_position_m=position,
            object_orientation_xyzw=quaternion,
        ))
        expected = scipy_transform(position, quaternion) @ object_palm
        actual_position, actual_wxyz = _world_palm(coordinate, proposal)
        actual_xyzw = actual_wxyz[[1, 2, 3, 0]]
        actual_rotation = Rotation.from_quat(actual_xyzw).as_matrix()
        translation_errors.append(float(np.max(np.abs(actual_position - expected[:3, 3]))))
        rotation_errors.append(float(np.max(np.abs(actual_rotation - expected[:3, :3]))))
        quaternion_norm_errors.append(abs(float(np.linalg.norm(actual_wxyz)) - 1.0))
        coordinate.initial_condition.object_orientation_xyzw = -quaternion
        sign_position, sign_wxyz = _world_palm(coordinate, proposal)
        sign_rotation = Rotation.from_quat(sign_wxyz[[1, 2, 3, 0]]).as_matrix()
        sign_errors.append(max(
            float(np.max(np.abs(sign_position - actual_position))),
            float(np.max(np.abs(sign_rotation - actual_rotation))),
        ))
        # This probes the helper's explicit normalization. Actual proposal/case
        # schemas still require unit quaternions; these are not valid cases.
        expected_object = scipy_transform(position, quaternion)
        for scale in (0.25, 3.5, -2.0):
            normalization_errors.append(float(np.max(np.abs(
                _matrix(position, quaternion * scale) - expected_object,
            ))))

    invalid_quaternion_rejections = 0
    for quaternion in (
        [0.0, 0.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0, 1.0],
        [0.0, np.inf, 0.0, 1.0],
    ):
        try:
            _matrix([0.0, 0.0, 0.0], quaternion)
        except ValueError:
            invalid_quaternion_rejections += 1

    # Compile only the existing hand, with no object/task/runner/forward call,
    # to obtain the real simulator joint order rather than assuming XML order.
    model = load_shadow_model(default_dex_urdf_root())
    joint_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                        for i in range(model.njnt))
    if len(joint_names) != 24 or set(joint_names) != set(proposal.joint_targets_rad):
        raise ValueError('Saved proposal and existing Shadow joint inventories differ')
    target_values = np.array([proposal.joint_targets_rad[name] for name in joint_names])
    permutations = 128
    exact_reorders = 0
    for _ in range(permutations):
        order = rng.permutation(24)
        result = reorder_for_mujoco(
            tuple(joint_names[i] for i in order), target_values[order], joint_names,
        )
        exact_reorders += int(np.array_equal(result, target_values))
    missing_joint_rejected = False
    try:
        reorder_for_mujoco(joint_names[:-1], target_values[:-1], joint_names)
    except ValueError:
        missing_joint_rejected = True

    local_native = camera_landmarks_to_wrist(native.landmarks_3d, native.T_camera_wrist)
    invariance_errors = []
    for position, quaternion in zip(positions, quaternions, strict=True):
        global_change = scipy_transform(position, quaternion)
        changed_landmarks = (
            np.asarray(native.landmarks_3d, dtype=np.float64) @ global_change[:3, :3].T
            + global_change[:3, 3]
        )
        changed_wrist = global_change @ native.T_camera_wrist
        actual_local = camera_landmarks_to_wrist(changed_landmarks, changed_wrist)
        invariance_errors.append(float(np.max(np.abs(actual_local - local_native))))

    checks = {
        'world_palm_independent_scipy_composition': {
            'manufactured_nonidentity_object_transform_count': COUNT,
            'maximum_translation_component_error_m': max(translation_errors),
            'maximum_rotation_matrix_element_error': max(rotation_errors),
            'maximum_output_wxyz_quaternion_norm_error': max(quaternion_norm_errors),
            'maximum_quaternion_sign_invariance_error': max(sign_errors),
            'passed': max(translation_errors + rotation_errors + quaternion_norm_errors + sign_errors) <= ALGEBRA_TOLERANCE,
        },
        'xyzw_to_wxyz_helper_normalization': {
            'manufactured_scaled_quaternion_count': len(normalization_errors),
            'maximum_transform_matrix_element_error': max(normalization_errors),
            'invalid_quaternion_count': 3,
            'invalid_quaternion_rejection_count': invalid_quaternion_rejections,
            'scope': 'helper normalization only; nonunit quaternions do not become valid evaluation cases',
            'passed': max(normalization_errors) <= ALGEBRA_TOLERANCE and invalid_quaternion_rejections == 3,
        },
        'exact_named_joint_reordering': {
            'joint_order_source': 'existing Shadow URDF compiled model introspection, without task or FK',
            'joint_names': joint_names,
            'manufactured_permutation_count': permutations,
            'exact_array_equal_count': exact_reorders,
            'missing_joint_rejected': missing_joint_rejected,
            'passed': exact_reorders == permutations and missing_joint_rejected,
        },
        'camera_to_wrist_global_rigid_transform_invariance': {
            'manufactured_global_coordinate_transform_count': COUNT,
            'landmark_source': 'same existing saved native HUG output, no new human pose',
            'maximum_local_landmark_component_error_m': max(invariance_errors),
            'passed': max(invariance_errors) <= ALGEBRA_TOLERANCE,
        },
    }
    report = {
        'schema_version': 'hug_boundary_algebra_audit_v1',
        'date': '2026-10-14',
        'classification': 'manufactured_coordinate_algebra_tests_on_existing_saved_HUG_not_new_grasps_or_eval_data',
        'source_code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        'random_seed': SEED,
        'algebra_roundoff_tolerance': ALGEBRA_TOLERANCE,
        'native_proposal': str(NATIVE),
        'native_proposal_sha256': digest(NATIVE),
        'canonical_proposal': str(PROPOSAL),
        'canonical_proposal_sha256': digest(PROPOSAL),
        'script_sha256': digest(__file__),
        'source_file_sha256': {
            str(path.relative_to(PROJECT)): digest(path) for path in (
                PROJECT / 'src/grasp_failure_prediction/proposals/execution.py',
                PROJECT / 'src/grasp_failure_prediction/evaluation/retargeting.py',
                PROJECT / 'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf',
            )
        },
        'runtime': {
            'executable': sys.executable,
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'scipy_version': scipy.__version__,
            'mujoco_version': mujoco.__version__,
            'pythonpath': os.environ.get('PYTHONPATH'),
        },
        'guarded_physics_apis': ['mj_step', 'mj_step1', 'mj_step2'],
        'physics_step_calls_attempted': len(STEP_ATTEMPTS),
        'new_inference': False,
        'task_cases_generated': False,
        'task_labels_generated': False,
        'repo_files_modified': False,
        'checks': checks,
        'passed': all(check['passed'] for check in checks.values()) and not STEP_ATTEMPTS,
        'limitations': 'Boundary algebra only; no new contact-fidelity or physical-success acceptance evidence.',
    }
    destination = DIRECTORY / 'hug_boundary_checks.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(destination), 'passed': report['passed'], 'checks': checks, 'runtime': report['runtime']}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
