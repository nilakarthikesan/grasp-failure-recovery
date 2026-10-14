"""HUG-only wrist/palm anatomy-proxy and residual decomposition diagnostic.

Reads 32 existing saved native poses and original nominal Shadow targets.
No learned model, new grasp, retarget optimization, integration or calibration.
All diagnostic outputs are written beside this script. Licensed MANO weights
are read locally for an anatomical surface proxy; no asset or weights copied.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import pickle
import sys
import warnings
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

GUARD_CALLS = []


def forbidden(*args, **kwargs):
    GUARD_CALLS.append(True)
    raise RuntimeError('Palm-anchor diagnostic forbids MuJoCo integration')


for api in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, api, forbidden)

# Guards precede project imports and model loading.
from grasp_failure_prediction.evaluation.pose_validation import _link_positions, TASK_LINK_NAMES
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
import chumpy

OUT = Path(__file__).resolve().parent
ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
PREVIOUS = ROOT / 'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals'
INDEX = PREVIOUS / 'existing_hug_inventory.json'
PRIOR_FK = PREVIOUS / 'static_saved_hug_fk.json'
PRIOR_RECONSTRUCTION = ROOT / 'research_artifacts/hug_mapping_confirmation_2026-10-14/hug_native_reconstruction.json'
URDF = default_dex_urdf_root() / 'shadow_hand/shadow_hand_right.urdf'
ASSET = Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/assets/mano_models/models/MANO_RIGHT.pkl')
MANOLAYER = Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps/manotorch/manolayer.py')
HUG_WRAPPER = Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/src/models/mano.py')
FINGERS = ['thumb', 'index', 'middle', 'ring', 'little']
TASK_INDICES = np.array([4, 8, 12, 16, 20, 2, 6, 10, 14, 18])
MCP_INDICES = np.array([5, 9, 13, 17])
MCP_JOINTS = ['FFJ3', 'MFJ3', 'RFJ3', 'LFJ3']
GROUPS = {'palm': [0], 'index': [1, 2, 3], 'middle': [4, 5, 6],
          'ring': [10, 11, 12], 'little': [7, 8, 9], 'thumb': [13, 14, 15]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ident(model, kind, name):
    value = int(mujoco.mj_name2id(model, kind, name))
    assert value >= 0, name
    return value


def stats(values):
    a = np.asarray(values)
    return dict(min=float(a.min()), mean=float(a.mean()), median=float(np.median(a)),
                max=float(a.max()), std=float(a.std(ddof=0)))


def point_summary(points):
    p = np.asarray(points)
    return dict(min_m=p.min(axis=0).tolist(), max_m=p.max(axis=0).tolist(),
                extent_m=np.ptp(p, axis=0).tolist(), vertex_mean_m=p.mean(axis=0).tolist())


def proxy_frame(anchor, mcp):
    """Axes are directional geometry proxies; normal is not palmar/dorsal."""
    longitudinal = mcp.mean(axis=0) - anchor
    longitudinal /= np.linalg.norm(longitudinal)
    transverse = mcp[0] - mcp[3]  # little -> index, same names on both models
    transverse -= longitudinal * np.dot(transverse, longitudinal)
    transverse /= np.linalg.norm(transverse)
    normal = np.cross(transverse, longitudinal)
    frame = np.column_stack([transverse, longitudinal, normal])
    assert np.max(np.abs(frame.T @ frame - np.eye(3))) < 1e-10
    assert abs(np.linalg.det(frame) - 1) < 1e-10
    return frame


def decompose(residual):
    """Analytic point-residual diagnostic only; never updates model placement."""
    r = np.asarray(residual)
    mean = r.mean(axis=0)
    d = r - mean
    before_sse = float(np.sum(r * r))
    after_sse = float(np.sum(d * d))
    assert abs(before_sse - after_sse - len(r) * np.dot(mean, mean)) < 1e-13
    return dict(common_residual_operator_m=mean.tolist(),
                hypothetical_shadow_shift_operator_m=(-mean).tolist(),
                original_point_error_m=stats(np.linalg.norm(r, axis=1)),
                differential_point_error_m=stats(np.linalg.norm(d, axis=1)),
                original_rms_m=float(np.sqrt(np.mean(np.sum(r * r, axis=1)))),
                differential_rms_m=float(np.sqrt(np.mean(np.sum(d * d, axis=1)))),
                squared_error_common_fraction=float(1 - after_sse / before_sse),
                differential_residual_operator_m=d.tolist())


def common_across_sources(residuals):
    r = np.asarray(residuals)
    global_mean = r.mean(axis=(0, 1))
    hypothetical = r - global_mean
    means = r.mean(axis=1)
    differential = r - means[:, None]
    original_sse = np.sum(r * r)
    return dict(common_residual_operator_m=global_mean.tolist(),
                hypothetical_single_shadow_shift_operator_m=(-global_mean).tolist(),
                per_source_common_residual_operator_std_m=means.std(axis=0).tolist(),
                per_source_common_residual_norm_m=stats(np.linalg.norm(means, axis=1)),
                per_source_common_vector_distance_from_global_mean_m=stats(np.linalg.norm(means - global_mean, axis=1)),
                original_point_error_m=stats(np.linalg.norm(r.reshape(-1, 3), axis=1)),
                single_common_removed_point_error_m=stats(np.linalg.norm(hypothetical.reshape(-1, 3), axis=1)),
                per_source_common_removed_point_error_m=stats(np.linalg.norm(differential.reshape(-1, 3), axis=1)),
                original_rms_m=float(np.sqrt(np.mean(np.sum(r * r, axis=-1)))),
                single_common_removed_rms_m=float(np.sqrt(np.mean(np.sum(hypothetical * hypothetical, axis=-1)))),
                per_source_common_removed_rms_m=float(np.sqrt(np.mean(np.sum(differential * differential, axis=-1)))),
                single_common_squared_error_fraction=float(1 - np.sum(hypothetical * hypothetical) / original_sse),
                per_source_common_squared_error_fraction=float(1 - np.sum(differential * differential) / original_sse))


def urdf_joint_record(j):
    return dict(name=j.get('name'), type=j.get('type'), parent=j.find('parent').get('link'),
                child=j.find('child').get('link'), origin=dict(j.find('origin').attrib),
                axis=dict(j.find('axis').attrib) if j.find('axis') is not None else None)


def main():
    assert OUT == Path('/private/tmp/hug-palm-anchor-2026-10-14')
    guard_selftests = []
    for name in ('mj_step', 'mj_step1', 'mj_step2'):
        try:
            getattr(mujoco, name)(None, None)
        except RuntimeError as error:
            assert str(error) == 'Palm-anchor diagnostic forbids MuJoCo integration'
            guard_selftests.append(name)
        else:
            raise AssertionError(f'{name} guard did not reject the deliberate test')
    assert len(GUARD_CALLS) == 3
    GUARD_CALLS.clear()  # Deliberate guard tests are recorded separately.
    production_files = sorted(list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'scripts').rglob('*.py')))
    production_before = {str(p): sha(p) for p in production_files}
    inputs = {str(p): sha(p) for p in [INDEX, PRIOR_FK, PRIOR_RECONSTRUCTION, URDF, ASSET, MANOLAYER, HUG_WRAPPER]}
    index = json.loads(INDEX.read_text())
    prior = json.loads(PRIOR_FK.read_text())
    prior_reconstruction = json.loads(PRIOR_RECONSTRUCTION.read_text())
    assert sha(ASSET) == prior_reconstruction['mano_asset_sha256']
    assert sha(HUG_WRAPPER) == prior_reconstruction['upstream_mano_source_sha256']
    assert sha(URDF) == prior['task_link_urdf_sha256']
    assert index['summary']['all_candidate_binding_and_mapping_hash_checks_pass']
    assert index['summary']['unique_modern_native_hashes'] == 32
    prior_by_hash = {c['native_sha256']: c for c in prior['cases']}
    # Confirm index ordering from actual pinned wrapper, no guessed landmarks.
    source = MANOLAYER.read_text()
    assert 'joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]' in source
    assert 'center_idx=0' in HUG_WRAPPER.read_text()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        asset = pickle.loads(ASSET.read_bytes(), encoding='latin1')
    weights = np.asarray(asset['weights'])
    group_weights = np.column_stack([weights[:, ids].sum(axis=1) for ids in GROUPS.values()])
    vertex_labels = group_weights.argmax(axis=1)
    tree = ET.parse(URDF).getroot()
    joint_xml = {j.get('name'): j for j in tree.findall('joint')}
    chosen_joint_names = ['WRJ2', 'WRJ1', 'FFJ4', 'FFJ3', 'MFJ4', 'MFJ3', 'RFJ4', 'RFJ3', 'LFJ5', 'LFJ4', 'LFJ3', 'THJ5', 'THJ4', 'THJ3']
    records = []
    task_residuals = []
    mcp_residuals = []
    native_mcps = []
    shadow_mcps = []
    surface_extents = defaultdict(list)
    shape_hashes = set()
    axis_angles = []
    for case in index['modern_cases']:
        for spec in [case['native'], case['stored_shadow_command'], case['scene_mjb'], *case['frame_files'].values()]:
            assert sha(spec['path']) == spec['sha256']
            inputs[spec['path']] = spec['sha256']
        native = pickle.loads(Path(case['native']['path']).read_bytes())
        shape_hashes.add(hashlib.sha256(np.asarray(native['shape']).tobytes()).hexdigest())
        T_wc = np.load(case['frame_files']['T_world_camera.npy']['path'], allow_pickle=False)
        T_wm = T_wc @ np.asarray(native['T_camera_wrist'])
        R_op = T_wm[:3, :3] @ MANO_TO_OPERATOR_RIGHT.T
        native_world = np.asarray(native['landmarks_3d']) @ T_wc[:3, :3].T + T_wc[:3, 3]
        native_vertices_world = np.asarray(native['mesh_vertices']) @ T_wc[:3, :3].T + T_wc[:3, 3]
        command = case['stored_shadow_command']
        with np.load(command['path'], allow_pickle=False) as a:
            row = command['row_index']
            names = a['joint_names'].tolist()
            target = a['nominal_target_rad'][row].copy()
            root_pos = a['root_target_position_m'][row].copy()
            root_quat = a['root_target_quaternion_wxyz'][row].copy()
        model = mujoco.MjModel.from_binary_path(case['scene_mjb']['path'])
        data = mujoco.MjData(model)
        with np.load(case['frame_files']['scene_state.npz']['path'], allow_pickle=False) as state:
            data.qpos[:] = state['qpos']
        data.qvel[:] = 0
        jid = lambda name: ident(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        bid = lambda name: ident(model, mujoco.mjtObj.mjOBJ_BODY, name)
        ra = int(model.jnt_qposadr[jid('world_joint')])
        oa = int(model.jnt_qposadr[jid('object_free')])
        initial_object_qpos = data.qpos[oa:oa + 7].copy()
        data.qpos[ra:ra + 3] = root_pos
        data.qpos[ra + 3:ra + 7] = root_quat
        for name, value in zip(names, target, strict=True):
            data.qpos[int(model.jnt_qposadr[jid(name)])] = value
        if model.nmocap:
            data.mocap_pos[0] = root_pos
            data.mocap_quat[0] = root_quat
        nominal_qpos = data.qpos.copy()
        mujoco.mj_forward(model, data)
        assert data.time == 0
        assert np.array_equal(data.qpos, nominal_qpos)
        assert np.array_equal(data.qpos[oa:oa + 7], initial_object_qpos)
        palm = data.xpos[bid('palm')].copy()
        wrist = data.xpos[bid('wrist')].copy()
        palm_R = data.xmat[bid('palm')].reshape(3, 3).copy()
        wrist_R = data.xmat[bid('wrist')].reshape(3, 3).copy()
        shadow_world = _link_positions(model, data, URDF)
        old = prior_by_hash[case['native']['sha256']]
        prior_link_world = np.array([link['shadow_world_m'] for link in old['links']])
        prior_delta = float(np.max(np.linalg.norm(shadow_world - prior_link_world, axis=1)))
        assert prior_delta < 1e-12
        n_local = (native_world - T_wm[:3, 3]) @ R_op
        s_local = (shadow_world - T_wm[:3, 3]) @ R_op
        n_mcp = n_local[MCP_INDICES]
        s_mcp_world = np.array([data.xanchor[jid(name)].copy() for name in MCP_JOINTS])
        s_mcp = (s_mcp_world - T_wm[:3, 3]) @ R_op
        palm_local = (palm - T_wm[:3, 3]) @ R_op
        native_wrist = n_local[0]
        nf = proxy_frame(native_wrist, n_mcp)
        sf = proxy_frame(palm_local, s_mcp)
        angle = np.degrees(np.arccos(np.clip(np.diag(nf.T @ sf), -1, 1)))
        axis_angles.append(angle)
        native_mcps.append(n_mcp)
        shadow_mcps.append(s_mcp)
        r = s_local - n_local[TASK_INDICES]
        mr = s_mcp - n_mcp
        task_residuals.append(r)
        mcp_residuals.append(mr)
        # Native palm subset is an OPEN anatomical surface proxy, no volume.
        faces = np.asarray(native['mesh_faces'], dtype=int)
        labels = vertex_labels[faces]
        counts = np.column_stack([(labels == k).sum(axis=1) for k in range(len(GROUPS))])
        means = group_weights[faces].mean(axis=1)
        face_labels = np.argmax(np.where(counts == counts.max(axis=1)[:, None], means, -np.inf), axis=1)
        palm_vertex_indices = np.unique(faces[face_labels == 0].ravel())
        n_palm = (native_vertices_world[palm_vertex_indices] - T_wm[:3, 3]) @ R_op
        corners = []
        palm_geoms = []
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != bid('palm') or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            size = model.geom_size[g]
            v = np.array([[x, y, z] for x in [-size[0], size[0]] for y in [-size[1], size[1]] for z in [-size[2], size[2]]])
            v_world = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            corners.append((v_world - T_wm[:3, 3]) @ R_op)
            palm_geoms.append(dict(id=int(g), full_size_m=(2 * size).tolist(),
                                   center_palm_m=((data.geom_xpos[g] - palm) @ palm_R).tolist(),
                                   rotation_palm=(palm_R.T @ data.geom_xmat[g].reshape(3, 3)).tolist()))
        assert len(palm_geoms) == 8
        s_palm = np.concatenate(corners)
        native_palm_bounds = point_summary((n_palm - native_wrist) @ nf)
        shadow_palm_bounds = point_summary((s_palm - palm_local) @ sf)
        surface_extents['native'].append(native_palm_bounds['extent_m'])
        surface_extents['shadow'].append(shadow_palm_bounds['extent_m'])
        n_cmc = n_local[1]
        s_cmc = (data.xanchor[jid('THJ5')] - T_wm[:3, 3]) @ R_op
        body_anchor = (palm - wrist) @ wrist_R
        expected_body_anchor = np.fromstring(joint_xml['WRJ1'].find('origin').get('xyz'), sep=' ')
        assert np.max(np.abs(body_anchor - expected_body_anchor)) < 1e-12
        wrist_to_palm_angle = float(Rotation.from_matrix(wrist_R.T @ palm_R).magnitude())
        by_finger = {}
        tip_dec = decompose(r[:5])
        all_dec = decompose(r)
        for i, name in enumerate(FINGERS):
            by_finger[name] = dict(tip_residual_operator_m=r[i].tolist(),
                                   tip_error_m=float(np.linalg.norm(r[i])),
                                   tip_after_common_tip_mean_m=float(np.linalg.norm(r[i] - np.mean(r[:5], axis=0))),
                                   middle_residual_operator_m=r[i + 5].tolist(),
                                   middle_error_m=float(np.linalg.norm(r[i + 5])),
                                   tip_after_common_all10_mean_m=float(np.linalg.norm(r[i] - np.mean(r, axis=0))),
                                   middle_after_common_all10_mean_m=float(np.linalg.norm(r[i + 5] - np.mean(r, axis=0))))
        records.append(dict(native_sha256=case['native']['sha256'], object_id=case['object_id'],
            observation_group_id=case['observation_group_id'], generation_seed=case['native']['seed'],
            scene_model_path=case['scene_mjb']['path'], nominal_targets_rad=dict(zip(names, target.tolist(), strict=True)),
            software_anchor=dict(palm_origin_vs_native_wrist_m=float(np.linalg.norm(palm - native_world[0])),
                palm_origin_vs_T_world_MANO_translation_m=float(np.linalg.norm(palm - T_wm[:3, 3])),
                actual_shadow_wrist_vs_native_wrist_m=float(np.linalg.norm(wrist - native_world[0])),
                wrist_to_palm_vector_wrist_body_m=body_anchor.tolist(), wrist_to_palm_length_m=float(np.linalg.norm(palm - wrist)),
                wrist_to_palm_rotation_rad=wrist_to_palm_angle, WRJ1_rad=float(dict(zip(names, target))['WRJ1']),
                WRJ2_rad=float(dict(zip(names, target))['WRJ2'])),
            anatomical_proxies=dict(native_nonthumb_MCP_operator_m=n_mcp.tolist(), shadow_nonthumb_MCP_operator_m=s_mcp.tolist(),
                native_thumb_CMC_operator_m=n_cmc.tolist(), shadow_thumb_base_THJ5_operator_m=s_cmc.tolist(),
                thumb_CMC_vs_THJ5_error_m=float(np.linalg.norm(s_cmc - n_cmc)),
                native_MCP_centroid_from_wrist_m=(n_mcp.mean(axis=0) - native_wrist).tolist(),
                shadow_MCP_centroid_from_palm_m=(s_mcp.mean(axis=0) - palm_local).tolist(),
                native_wrist_to_MCP_centroid_length_m=float(np.linalg.norm(n_mcp.mean(axis=0) - native_wrist)),
                shadow_palm_to_MCP_centroid_length_m=float(np.linalg.norm(s_mcp.mean(axis=0) - palm_local)),
                shadow_actual_wrist_to_MCP_centroid_length_m=float(np.linalg.norm(s_mcp_world.mean(axis=0) - wrist)),
                native_index_to_little_MCP_span_m=float(np.linalg.norm(n_mcp[0] - n_mcp[3])),
                shadow_index_to_little_MCP_span_m=float(np.linalg.norm(s_mcp[0] - s_mcp[3])),
                native_proxy_axes_operator_columns=nf.tolist(), shadow_proxy_axes_operator_columns=sf.tolist(),
                transverse_longitudinal_signed_normal_direction_angles_deg=angle.tolist(),
                nonthumb_MCP_translation_decomposition=decompose(mr)),
            palm_shape_proxies=dict(native_palm_triangle_count=int((face_labels == 0).sum()),
                native_palm_surface_vertex_count=len(palm_vertex_indices), native_palm_open_subset_bounds_own_proxy_frame=native_palm_bounds,
                shadow_palm_8_collision_boxes_bounds_own_proxy_frame=shadow_palm_bounds, shadow_palm_collision_boxes=palm_geoms),
            task_point_diagnostic=dict(tips5=tip_dec, tips_and_middle10=all_dec, by_finger=by_finger),
            invariants=dict(time_s=float(data.time), initial_object_qpos_unchanged=True, nominal_qpos_unchanged_by_forward=True,
                            prior_saved_FK_task_link_max_difference_m=prior_delta)))
    task_residuals = np.asarray(task_residuals)
    mcp_residuals = np.asarray(mcp_residuals)
    by_object = {}
    for object_id in sorted({c['object_id'] for c in records}):
        select = np.array([c['object_id'] == object_id for c in records])
        by_object[object_id] = dict(source_count=int(select.sum()), tips5=common_across_sources(task_residuals[select, :5]),
                                    tips_and_middle10=common_across_sources(task_residuals[select]),
                                    nonthumb_MCP_proxies4=common_across_sources(mcp_residuals[select]))
    by_finger = {}
    global_tip_mean = task_residuals[:, :5].mean(axis=(0, 1))
    global_all_mean = task_residuals.mean(axis=(0, 1))
    global_mcp_mean = mcp_residuals.mean(axis=(0, 1))
    case_tip_means = task_residuals[:, :5].mean(axis=1)
    case_all_means = task_residuals.mean(axis=1)
    for i, finger in enumerate(FINGERS):
        by_finger[finger] = dict(tip_original_m=stats(np.linalg.norm(task_residuals[:, i], axis=1)),
            tip_after_per_case_tip_common_m=stats(np.linalg.norm(task_residuals[:, i] - case_tip_means, axis=1)),
            tip_after_single_global_tip_common_m=stats(np.linalg.norm(task_residuals[:, i] - global_tip_mean, axis=1)),
            middle_original_m=stats(np.linalg.norm(task_residuals[:, i + 5], axis=1)),
            middle_after_per_case_all10_common_m=stats(np.linalg.norm(task_residuals[:, i + 5] - case_all_means, axis=1)),
            middle_after_single_global_all10_common_m=stats(np.linalg.norm(task_residuals[:, i + 5] - global_all_mean, axis=1)),
            tip_residual_mean_operator_m=task_residuals[:, i].mean(axis=0).tolist(),
            middle_residual_mean_operator_m=task_residuals[:, i + 5].mean(axis=0).tolist())
    native_mcps = np.asarray(native_mcps)
    shadow_mcps = np.asarray(shadow_mcps)
    anatomy_summary = dict(proxy_axis_direction_angles_deg={name: stats(np.asarray(axis_angles)[:, i])
                          for i, name in enumerate(['transverse', 'longitudinal', 'signed_cross_product_normal'])},
        native_MCP_points_operator_std_m=native_mcps.std(axis=0).tolist(), shadow_MCP_points_operator_std_m=shadow_mcps.std(axis=0).tolist(),
        native_wrist_to_MCP_centroid_m=stats([c['anatomical_proxies']['native_wrist_to_MCP_centroid_length_m'] for c in records]),
        shadow_palm_to_MCP_centroid_m=stats([c['anatomical_proxies']['shadow_palm_to_MCP_centroid_length_m'] for c in records]),
        shadow_actual_wrist_to_MCP_centroid_m=stats([c['anatomical_proxies']['shadow_actual_wrist_to_MCP_centroid_length_m'] for c in records]),
        thumb_CMC_vs_THJ5_base_proxy_error_m=stats([c['anatomical_proxies']['thumb_CMC_vs_THJ5_error_m'] for c in records]),
        native_index_to_little_MCP_span_m=stats([c['anatomical_proxies']['native_index_to_little_MCP_span_m'] for c in records]),
        shadow_index_to_little_MCP_span_m=stats([c['anatomical_proxies']['shadow_index_to_little_MCP_span_m'] for c in records]),
        native_palm_open_subset_extent_own_proxy_frame_m={name: stats(np.asarray(surface_extents['native'])[:, i])
                         for i, name in enumerate(['transverse', 'longitudinal', 'signed_cross_product_normal'])},
        shadow_palm_boxes_extent_own_proxy_frame_m={name: stats(np.asarray(surface_extents['shadow'])[:, i])
                         for i, name in enumerate(['transverse', 'longitudinal', 'signed_cross_product_normal'])})
    assert not GUARD_CALLS
    assert len(shape_hashes) == 1
    assert all(sha(path) == digest for path, digest in production_before.items())
    assert all(sha(path) == digest for path, digest in inputs.items())
    report = dict(schema='hug_existing_wrist_palm_anatomical_proxy_diagnostic_v1', date='2026-10-14',
        scope='32 existing native poses, four existing object families, eight observation groups; static original nominal targets only',
        script_sha256=sha(__file__), input_sha256=inputs, production_sha256_before_and_after=production_before,
        runtime=dict(python=sys.version, numpy=np.__version__, mujoco=mujoco.__version__),
        guards=dict(apis=['mj_step', 'mj_step1', 'mj_step2'], installed_before_project_import_and_model_load=True,
                    attempted_guarded_calls=len(GUARD_CALLS), physics_steps=0, learned_generator_loaded=False,
                    deliberate_guard_rejection_selftests=guard_selftests,
                    unique_existing_native_shape_hashes=sorted(shape_hashes),
                    asset_wrapper_and_URDF_match_prior_confirmation_hashes=True,
                    retarget_optimizer_loaded_or_used=False, calibration_applied=False, production_files_unchanged=True),
        landmark_definition=dict(actual_MANO_to_21_reorder=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20],
            wrist=0, thumb_CMC=1, nonthumb_MCP=dict(zip(FINGERS[1:], MCP_INDICES.tolist())),
            corresponding_nonthumb_Shadow_MCP_joint_proxies=dict(zip(FINGERS[1:], MCP_JOINTS)),
            thumb_caveat='Native thumb CMC [1] compared only to THJ5 base as a CMC/base proxy; not called an MCP correspondence.',
            task_landmark_indices=TASK_INDICES.tolist(), task_link_names=list(TASK_LINK_NAMES)),
        selected_actual_URDF_joint_definitions=[urdf_joint_record(joint_xml[name]) for name in chosen_joint_names],
        measurement_policy=dict(common_frame='Native operator/fixed-URDF base directions R_world_MANO @ MANO_TO_OPERATOR_RIGHT.T; origin native saved wrist.',
            proxy_axes='Longitudinal = anchor to nonthumb MCP centroid. Transverse = little-to-index MCP span projected perpendicular to longitudinal. Signed normal = transverse cross longitudinal. No palmar/dorsal assignment.',
            anchor='Native wrist landmark [0] versus Shadow palm origin; actual Shadow wrist body measured separately.',
            translation='Residual r=Shadow-native; rbar is algebraic mean over named proxies. r-rbar decomposes point error. Hypothetical -rbar is never applied to geometry/model/control. Across-source common mean uses operator coordinates.',
            anatomical_pairs='Same anatomical names and base-chain roles define candidate proxies across different models, not certified identical anatomical locations.',
            surface='Native majority grouped-skinning palm triangle subset is open, includes assigned wrist-cap/mixed seam triangles and their boundary vertices. Its used-vertex bounding extents are compared with union bounds of actual eight Shadow palm collision boxes. Bounds/vertex means are geometric descriptions, not volumes or contact metrics.'),
        software_anchor_summary=dict(palm_origin_vs_native_wrist_m=stats([c['software_anchor']['palm_origin_vs_native_wrist_m'] for c in records]),
            actual_shadow_wrist_vs_native_wrist_m=stats([c['software_anchor']['actual_shadow_wrist_vs_native_wrist_m'] for c in records]),
            wrist_to_palm_length_m=stats([c['software_anchor']['wrist_to_palm_length_m'] for c in records]),
            WRJ1_rad=stats([c['software_anchor']['WRJ1_rad'] for c in records]), WRJ2_rad=stats([c['software_anchor']['WRJ2_rad'] for c in records])),
        anatomical_proxy_summary=anatomy_summary,
        global_translation_diagnostics=dict(tips5=common_across_sources(task_residuals[:, :5]),
            tips_and_middle10=common_across_sources(task_residuals), nonthumb_MCP_proxies4=common_across_sources(mcp_residuals)),
        cross_point_family_translation_diagnostic=dict(
            tip_mean_residual_vs_all10_mean_vector_distance_m=float(np.linalg.norm(global_tip_mean - global_all_mean)),
            tip_mean_residual_vs_MCP_mean_vector_distance_m=float(np.linalg.norm(global_tip_mean - global_mcp_mean)),
            fingertip_errors_if_MCP_mean_algebraically_removed_m=stats(np.linalg.norm((task_residuals[:, :5] - global_mcp_mean).reshape(-1, 3), axis=1)),
            MCP_proxy_errors_if_tip_mean_algebraically_removed_m=stats(np.linalg.norm((mcp_residuals - global_tip_mean).reshape(-1, 3), axis=1)),
            note='Pure subtraction in saved residual arrays. No geometry is moved; differing proxy families have no established shared calibration target.'),
        by_object=by_object, differential_errors_by_finger=by_finger, cases=records,
        limitations=['Origin agreement confirms the implemented anchor convention, not anatomical wrist/palm equivalence.',
            'Different models have no ground-truth anatomical landmark identity; thumb base and nonthumb MCP comparisons are declared proxies.',
            'Axis comparisons are standalone geometry diagnostics, not an identified replacement rotation or calibration.',
            'One MANO shape and reused development sources do not identify a general population calibration or independent validation.',
            'A small point residual or reduced algebraic SSE does not establish preservation of surfaces, contact normals, forces or grasp success.',
            'Palm surface bounds compare an open skinning-based subset with collision boxes; they are not equivalent anatomical tissue regions.',
            'The pointwise residual decomposition never changes original targets, root placement, object pose, poor native proposals or historical labels.'])
    (OUT / 'palm_anchor_report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (OUT / 'reproduce.sh').write_text("#!/bin/sh\nset -eu\ncd /private/tmp/hug-palm-anchor-2026-10-14\nPYTHONDONTWRITEBYTECODE=1 PYTHONPATH='/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages' /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python diagnose_palm_anchor.py\n")
    print(json.dumps(dict(report=str(OUT / 'palm_anchor_report.json'), software_anchor=report['software_anchor_summary'],
                         anatomy=anatomy_summary, translation=report['global_translation_diagnostics']), indent=2))


if __name__ == '__main__':
    main()
