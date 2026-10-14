"""Read-only UniDex fidelity audit: static synthetic FK, never model/eval data.

Run with project core plus existing integration/simulation dependency paths.
Uses the already fetched, hash-pinned Dex source XML as a REFERENCE PROXY;
the actual released UniDex generation hand asset is absent and is not tested.
Writes only aggregate audit JSON and temporary stripped skeletons in /tmp.
"""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import torch

from grasp_failure_prediction.proposals.unidexgrasp import (
    _prepare_source, _certificate, KINEMATICS_CERTIFICATE_SHA256,
)
from grasp_failure_prediction.proposals.artifacts import file_sha256
from grasp_failure_prediction.proposals.dexgraspnet import SOURCE_JOINT_NAMES, SOURCE_TIP_SITES, _derive_kinematic_mjcf
from grasp_failure_prediction.proposals.translation import NativeHandPose, translate_native_pose
from grasp_failure_prediction.proposals.schema import RigidTransform
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator, _link_positions
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root, RetargetedHandPose

SOURCE = Path('/private/tmp/gfp-dex-shadow-source.xml')
UPSTREAM = Path('/private/tmp/gfp-unidexgrasp-source')
ACTUAL_UNIDEX_ASSET = UPSTREAM / 'dexgrasp_generation/data/mjcf/shadow_hand.xml'
EXPECTED_RAW = 'sha256:39b1f1e1150c9c3821748ba70cfaeaefc2517100d45b5d6d5ef683a7d132de8a'
if file_sha256(SOURCE) != EXPECTED_RAW:
    raise ValueError('Registered reference source XML differs from the pinned asset')
def forbidden(*args, **kwargs):
    raise RuntimeError('This audit forbids all simulation stepping')
for name in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, name, forbidden)

report = {
    'classification': 'static_synthetic_fidelity_audit_not_evaluation_data',
    'reference_proxy': str(SOURCE), 'reference_raw_sha256': EXPECTED_RAW,
    'actual_unidex_generation_hand_present': ACTUAL_UNIDEX_ASSET.exists(),
    'actual_unidex_generation_hand_tested': False, 'simulation_steps': 0,
}
with tempfile.TemporaryDirectory(prefix='unidex-fidelity-audit-', dir='/private/tmp') as temp:
    temp = Path(temp)
    derived = temp / 'reference.xml'
    source_model = _prepare_source(SOURCE, derived, EXPECTED_RAW)
    source_data = mujoco.MjData(source_model)
    target = ShadowPoseValidator(default_dex_urdf_root())
    source_range = source_model.jnt_range
    qstates = [np.zeros(22), source_range.mean(axis=1), source_range[:, 0], source_range[:, 1]]
    for joint in range(22):
        for fraction in (.25, .75):
            q = np.zeros(22)
            q[joint] = source_range[joint, 0] + fraction * np.ptp(source_range[joint])
            qstates.append(q)
    qstates.extend(np.random.default_rng(293).uniform(source_range[:, 0], source_range[:, 1], size=(64, 22)))
    errors, bodyerrors, lfcenters, limit_rejections = [], [], [], []
    lf_body = mujoco.mj_name2id(target.model, mujoco.mjtObj.mjOBJ_BODY, 'lfmetacarpal')
    lf_geom = [i for i in range(target.model.ngeom) if target.model.geom_bodyid[i] == lf_body and target.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX][0]
    for index, q in enumerate(qstates):
        native = NativeHandPose('dexgraspnet_shadow_mjcf_bd1b13d', 'robot0:palm', RigidTransform(translation_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.)), dict(zip(SOURCE_JOINT_NAMES, map(float, q))))
        mapped = translate_native_pose(native, 'dexgraspnet_shadow_to_shadow_hand_right_v1')
        pose = RetargetedHandPose(joint_names=tuple(mapped.joint_targets_rad), qpos=np.asarray(tuple(mapped.joint_targets_rad.values())), landmarks_wrist_m=np.zeros((21, 3)), reference_vectors_m=np.zeros((10, 3)), T_camera_wrist=np.eye(4))
        target.set_pose(pose)
        source_data.qpos[:] = q
        mujoco.mj_forward(source_model, source_data)
        sp = source_data.body('robot0:palm').xpos
        sr = source_data.body('robot0:palm').xmat.reshape(3, 3)
        tp = target.data.body('palm').xpos
        tr = target.data.body('palm').xmat.reshape(3, 3)
        source_tips = np.asarray([sr.T @ (source_data.site(name).xpos - sp) for name in SOURCE_TIP_SITES])
        target_tips = np.asarray([tr.T @ (p - tp) for p in _link_positions(target.model, target.data, target.source_urdf)[:5]])
        errors.append(np.linalg.norm(source_tips - target_tips, axis=1))
        sbody = source_data.body('robot0:lfmetacarpal')
        source_lfcenter = sbody.xpos + sbody.xmat.reshape(3, 3) @ [-.0165, 0., .01]
        target_lfcenter = target.data.geom_xpos[lf_geom]
        lfcenters.append(float(np.linalg.norm(sr.T @ (source_lfcenter-sp) - tr.T @ (target_lfcenter-tp))))
        bodyerrors.append(float(np.linalg.norm(sr.T @ (sbody.xpos-sp) - tr.T @ (target.data.body('lfmetacarpal').xpos-tp))))
        targetq = np.asarray([mapped.joint_targets_rad[name] for name in target.joint_names])
        limited = target.model.jnt_limited.astype(bool)
        outside = (targetq < target.model.jnt_range[:, 0] - 1e-9) | (targetq > target.model.jnt_range[:, 1] + 1e-9)
        if np.any(outside & limited):
            limit_rejections.append({'synthetic_state_index': index, 'joints': [target.joint_names[i] for i in np.flatnonzero(outside & limited)]})
    e = np.asarray(errors)
    report['static_reference_to_target_fk'] = {
        'synthetic_pose_count': len(qstates), 'certificate_matches': _certificate(source_model) == KINEMATICS_CERTIFICATE_SHA256,
        'per_tip_max_error_m': dict(zip(['thumb', 'index', 'middle', 'ring', 'little'], map(float, e.max(axis=0)))),
        'overall_max_tip_error_m': float(e.max()), 'mean_all_tip_error_m': float(e.mean()),
        'max_lfmetacarpal_named_body_origin_difference_m': max(bodyerrors),
        'max_lfmetacarpal_collision_box_center_difference_m': max(lfcenters),
        'source_lfmetacarpal_box_half_size_m': [.0095, .0111, .025],
        'target_lfmetacarpal_box_half_size_m': list(map(float, target.model.geom_size[lf_geom])),
        'target_limit_rejections': limit_rejections,
    }
    blindspots = []
    for role in ('capsule_radius', 'palm_inertial_mass', 'fingertip_site_offset'):
        xml = ET.parse(SOURCE)
        if role == 'capsule_radius':
            geom = xml.getroot().find('.//geom[@name="robot0:C_ffproximal"]')
            geom.set('size', '.02 .0225')
        elif role == 'palm_inertial_mass':
            xml.getroot().find('.//body[@name="robot0:palm"]/inertial').set('mass', '30')
        else:
            xml.getroot().find('.//site[@name="robot0:S_fftip"]').set('pos', '0 0 .040')
        altered = temp / f'{role}.xml'
        xml.write(altered, encoding='utf-8')
        skeleton = temp / f'{role}_kin.xml'
        _derive_kinematic_mjcf(altered, skeleton)
        altered_model = mujoco.MjModel.from_xml_path(str(skeleton))
        blindspots.append({'mutation': role, 'raw_hash_changed': file_sha256(altered) != EXPECTED_RAW, 'certificate_unchanged': _certificate(altered_model) == KINEMATICS_CERTIFICATE_SHA256})
    report['certificate_blind_spot_measurement'] = blindspots

    # Import only the pinned pure-kinematics submodule. No model modules or
    # checkpoints are imported/loaded, and no learned inference is performed.
    pkg = UPSTREAM / 'dexgrasp_generation/thirdparty/pytorch_kinematics/pytorch_kinematics'
    spec = importlib.util.spec_from_file_location('pytorch_kinematics', pkg / '__init__.py', submodule_search_locations=[str(pkg)])
    sys.modules['pytorch_kinematics'] = importlib.util.module_from_spec(spec)
    upstream_mjcf = importlib.import_module('pytorch_kinematics.mjcf')
    comparison = []
    xml = ET.parse(derived)
    for variant, root_offset in (('reference', False), ('nonidentity_root_offset', True)):
        if root_offset:
            palm = xml.getroot().find('worldbody/body')
            palm.set('pos', '.1 .2 .3')
            palm.set('quat', '.7071067811865476 0 0 .7071067811865476')
        text = ET.tostring(xml.getroot(), encoding='unicode')
        chain = upstream_mjcf.build_chain_from_mjcf(text).to(dtype=torch.float32, device='cpu')
        model = mujoco.MjModel.from_xml_string(text)
        data = mujoco.MjData(model)
        qs = np.vstack([np.zeros(22), model.jnt_range.mean(axis=1), np.random.default_rng(284).uniform(model.jnt_range[:, 0], model.jnt_range[:, 1], size=(16, 22))])
        maxpos, maxrot = 0., 0.
        for q in qs:
            fk = chain.forward_kinematics(torch.tensor(q[None, :], dtype=torch.float32))
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            for body in range(1, model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
                pkname = name + '_child' if body in model.jnt_bodyid else name
                a = fk[pkname].get_matrix().numpy()[0]
                maxpos = max(maxpos, float(np.linalg.norm(a[:3, 3] - data.xpos[body])))
                maxrot = max(maxrot, float(np.max(np.abs(a[:3, :3] - data.xmat[body].reshape(3, 3)))))
        comparison.append({'variant': variant, 'synthetic_pose_count': len(qs), 'joint_order_matches': chain.get_joint_parameter_names() == list(SOURCE_JOINT_NAMES), 'max_body_position_error_m': maxpos, 'max_rotation_matrix_element_error': maxrot})
    report['upstream_pk_vs_mujoco_static_fk'] = comparison

destination = Path('/private/tmp/gfp-unidex-fidelity-audit.json')
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
