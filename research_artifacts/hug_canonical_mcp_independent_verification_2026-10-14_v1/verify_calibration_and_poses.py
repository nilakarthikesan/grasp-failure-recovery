"""Independent C/frame/source/compiled-FK checks; no retargeting or surface audit.

The neutral MANO descriptor is inherited from the separately frozen primary
source-template forward. This script independently reconstructs its frames/C,
source-URDF correspondences and every original/new saved-case world pose.
"""
from pathlib import Path
import hashlib,json,gzip,pickle,time,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
OUT=Path(__file__).parent
COUNTS={'actual':0,'selftests':0};TEST=False
def forbid(*args,**kwargs):
    COUNTS['selftests' if TEST else 'actual']+=1
    raise RuntimeError('Independent canonical verification forbids physics integration')
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbid)
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator,_body_id,_link_positions
from grasp_failure_prediction.evaluation.retargeting import reorder_for_mujoco
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def norm_rows(value):
    x=np.asarray(value,float);return np.sqrt(np.einsum('ij,ij->i',x,x))
def array_sha(value):return hashlib.sha256(np.asarray(value,dtype='<f8').tobytes()).hexdigest()
def ident(m,kind,name):
    result=mujoco.mj_name2id(m,kind,name);assert result>=0,name;return result
def compare(actual,recorded,allowance,label):
    a=np.asarray(actual,float);r=np.asarray(recorded,float)
    assert a.shape==r.shape and np.isfinite(a).all() and np.isfinite(r).all(),label
    difference=float(np.max(np.abs(a-r)))
    assert difference<=allowance,(label,difference,allowance)
    return difference
def frame(points,origin,minimum_longitudinal,minimum_lateral):
    points=np.asarray(points,float);origin=np.asarray(origin,float)
    assert points.shape==(3,3) and origin.shape==(3,) and np.isfinite(points).all() and np.isfinite(origin).all()
    center=np.mean(points,axis=0);forward=center-origin;long_length=float(np.sqrt(forward@forward))
    assert long_length>minimum_longitudinal
    z=forward/long_length;across=points[0]-points[2];projected=across-(across@z)*z;lat_length=float(np.sqrt(projected@projected))
    assert lat_length>minimum_lateral
    x=projected/lat_length;y=np.cross(z,x);matrix=np.eye(4);matrix[:3,:3]=np.stack([x,y,z],axis=1);matrix[:3,3]=center
    assert abs(np.linalg.det(matrix[:3,:3])-1)<=1e-12 and np.max(np.abs(matrix[:3,:3].T@matrix[:3,:3]-np.eye(3)))<=1e-12
    return matrix,{'longitudinal_length_m':long_length,'projected_lateral_length_m':lat_length}

def source_link_zero_pose(xml,link,rigid_only=True):
    """Compose source joint origins at zero; optionally require a rigid chain."""
    by_child={j.find('child').get('link'):j for j in xml.findall('joint')}
    transforms=[];seen=set();body=link
    while body in by_child:
        assert body not in seen;seen.add(body);joint=by_child[body]
        assert joint.get('type') in (('fixed',) if rigid_only else ('fixed','revolute','continuous')),joint.get('name')
        origin=joint.find('origin');value=np.eye(4)
        if origin is not None:
            rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ');xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            assert rpy.shape==xyz.shape==(3,)
            value[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();value[:3,3]=xyz
        transforms.append(value);body=joint.find('parent').get('link')
    result=np.eye(4)
    for value in reversed(transforms):result=result@value
    return result

def main():
    global TEST
    started=time.perf_counter();p=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json')
    assert p['execution_authorized_by_root_review'] and sha(__file__)==p['script_sha256']
    assert not (OUT/'report.json').exists()
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    TEST=True
    for api in ('mj_step','mj_step1','mj_step2'):
        try:getattr(mujoco,api)(None,None)
        except RuntimeError:pass
        else:raise AssertionError('Static integration guard failed')
    TEST=False
    primary=read(p['primary_protocol']);cal=read(p['calibration_path']);candidate=read(p['candidate_path']);settings=primary['canonical_settings']
    assert primary['execution_authorized_by_root_and_independent_review']
    assert candidate['schema']=='hug_canonical_MCP_calibration_candidate_v1'
    assert cal['schema']=='hug_source_only_canonical_MCP_frame_v1'
    assert cal['protocol_sha256']==candidate['protocol_sha256']==sha(p['primary_protocol'])
    assert candidate['calibration_sha256']==sha(p['calibration_path']) and cal['settings']==settings
    assert not candidate['strict_surface_audit_run'] and not candidate['new_eval_data'] and candidate['physics_steps']==0
    assert not candidate['HUG_generator_loaded'] and not candidate['HUG_generator_sampled'] and not candidate['new_grasp_proposal'] and not candidate['training'] and not candidate['production_changed']
    assert candidate['strict_surface_acceptance_rule_unchanged']
    assert settings['MANO21_MCP_indices']==[5,9,13] and settings['MANO_raw_MCP_indices']==[1,4,10]
    assert settings['MANO_center_idx']==0 and settings['MANO_metric_scale']==1.0 and settings['MANO_side']=='right'
    assert settings['MANO_use_pca'] is False and settings['MANO_flat_hand_mean'] is True and settings['MANO_rot_mode']=='axisang'
    assert settings['MANO_dtype']=='float32' and settings['MANO_device']=='cpu'
    assert settings['MANO_betas']==[0.]*10 and settings['MANO_pose_axisangle']==[0.]*48
    assert primary['no_second_MANO_TO_OPERATOR_rotation'] and primary['old_tip_cap_acceptance_allowance_m']==0.0
    assert primary['task_human_indices']==[4,8,12,16,20,2,6,10,14,18]
    assert primary['task_link_names']==['thtip','fftip','mftip','rftip','lftip','thmiddle','ffmiddle','mfmiddle','rfmiddle','lfmiddle']
    assert primary['maximum_mean_fingertip_error_m']==.01 and primary['existing_Dex_optimizer_bound_margin_rad']==.00101
    assert primary['supported_NLopt_result_codes']==[1,2,3,4,5,6] and primary['require_finite_NLopt_last_value']
    assert primary['required_separate_new_anchor_radial_control_before_full_surface_audit']
    assert candidate['all_six_eligible_for_separate_full_surface_audit'] is False and candidate['required_new_anchor_radial_control']=='NOT_RUN'
    assert cal['source_only_no_object_or_witness_inputs'] and cal['calibration_is_new_mapping_not_frame_bugfix']
    parent_cal=read(primary['parent_exported_calibration_path'])
    published_C=np.asarray(cal['C_MANO_from_neutral_palm'],dtype=np.float64)
    parent_C=np.asarray(parent_cal['C_MANO_from_neutral_palm'],dtype=np.float64)
    assert published_C.tobytes()==parent_C.tobytes()
    assert hashlib.sha256(published_C.tobytes()).hexdigest()==primary['parent_exported_C_float64_sha256']==cal['C_float64_sha256']
    assert cal['parent_exported_C_float64_byte_equality_pass']
    numerical=p['field_reproduction_allowance_m']
    J=np.asarray(cal['neutral_MANO21_joints_m'],float);assert J.shape==(21,3) and np.isfinite(J).all() and np.array_equal(J[0],np.zeros(3))
    compare(J[[5,9,13]],cal['neutral_MANO_MCP_5_9_13_m'],0.,'canonical_MANO_descriptor_order')
    urdf=Path(settings['urdf']);xml=ET.parse(urdf).getroot();P=[]
    for name in ['FFJ4','MFJ4','RFJ4']:
        joint=next(j for j in xml.findall('joint') if j.get('name')==name)
        assert joint.find('parent').get('link')=='palm'
        assert np.array_equal(np.fromstring(joint.find('origin').get('rpy','0 0 0'),sep=' '),np.zeros(3))
        P.append(np.fromstring(joint.find('origin').get('xyz'),sep=' '))
    P=np.asarray(P);compare(P,cal['neutral_Shadow_MCP_FFJ4_MFJ4_RFJ4_palm_m'],0.,'source_URDF_MCP_origins')
    FM,cm=frame(J[[5,9,13]],J[0],settings['minimum_longitudinal_m'],settings['minimum_projected_lateral_m'])
    FP,cp=frame(P,np.zeros(3),settings['minimum_longitudinal_m'],settings['minimum_projected_lateral_m'])
    # Solve C*FP=FM independently; do not call the primary rigid-inverse helper.
    C=np.linalg.solve(FP.T,FM.T).T
    compare(FM,cal['F_MANO_from_MCP_frame'],numerical,'MANO_anatomical_frame')
    compare(FP,cal['F_palm_from_MCP_frame'],numerical,'palm_anatomical_frame')
    compare(C,cal['C_MANO_from_neutral_palm'],numerical,'C_MANO_from_neutral_palm')
    assert np.max(np.abs(C[3]-[0,0,0,1]))<=numerical and abs(np.linalg.det(C[:3,:3])-1)<=numerical
    compare(C@FP,FM,numerical,'C_direction_frame_identity')
    compare(P@C[:3,:3].T+C[:3,3]-J[[5,9,13]],cal['pointwise_neutral_MCP_residual_vectors_MANO_m'],numerical,'neutral_MCP_residuals')
    for own,recorded in [(cm,cal['neutral_MANO_conditioning']),(cp,cal['neutral_Shadow_conditioning'])]:
        for key,value in own.items():assert abs(value-recorded[key])<=numerical
    validator=ShadowPoseValidator(settings['urdf_root']);validator.data.qpos[:]=0;validator.data.qvel[:]=0;mujoco.mj_forward(validator.model,validator.data)
    palm=_body_id(validator.model,'palm')
    RP0=validator.data.xmat[palm].reshape(3,3).copy();pP0=validator.data.xpos[palm].copy()
    TBF=source_link_zero_pose(xml,'forearm');RF=TBF[:3,:3];pF=TBF[:3,3]
    TBP0=source_link_zero_pose(xml,'palm',rigid_only=False)
    compare(TBP0,cal['T_B_from_neutral_palm_from_source_joint_chain'],numerical,'source_neutral_joint_chain')
    neutral_chain_difference=max(float(np.max(np.abs(TBP0[:3,:3]-RP0))),float(np.max(np.abs(TBP0[:3,3]-pP0))))
    assert neutral_chain_difference<=numerical and abs(neutral_chain_difference-cal['source_joint_chain_vs_compiled_neutral_palm_maxabs'])<=numerical
    compare(RP0,cal['R_B_from_neutral_palm'],numerical,'source_neutral_palm_R_BP0')
    compare(pP0,cal['p_B_neutral_palm_m'],numerical,'source_neutral_palm_position')
    compare(TBF,cal['T_B_from_forearm'],numerical,'source_fixed_forearm_transform')
    compiled=np.asarray([(validator.data.xpos[_body_id(validator.model,n)]-pP0)@RP0 for n in ['ffknuckle','mfknuckle','rfknuckle']])
    compare(compiled,P,settings['source_vs_compiled_MCP_agreement_m'],'source_compiled_neutral_MCP_agreement')
    compare(compiled,cal['neutral_Shadow_compiled_MCP_palm_m'],numerical,'primary_compiled_neutral_MCP')
    inventory=read(primary['inventory_path'])['modern_cases'];old=read(primary['old_caps_candidate_path']);oldmap={c['source_inventory_index']:c for c in old['cases']}
    prior=json.loads(gzip.decompress(Path(p['prior_actual_preflight']).read_bytes()));prior_map={c['source_inventory_index']:c for c in prior['cases']}
    assert prior['protocol_sha256']==p['prior_actual_preflight_protocol_sha256']
    targets={c['source_inventory_index']:c for c in candidate['cases']};assert sorted(targets)==primary['case_indices']==p['case_indices']==[8,12,16,20,24,28]
    assert len(targets)==len(candidate['cases'])==len({c['native_sha256'] for c in candidate['cases']})==6
    physical={j.get('name'):[float(j.find('limit').get(k)) for k in ('lower','upper')] for j in xml.findall('joint') if j.find('limit') is not None}
    records=[];native_frame_sources={}
    for index in p['case_indices']:
        target=targets[index];source_case=inventory[index];old_target=oldmap[index]
        assert target['native_sha256']==source_case['native']['sha256']==old_target['native_sha256']
        assert target['object_id']==source_case['object_id'] and target['source_case']==Path(source_case['native']['path']).parent.name
        case_paths=[source_case['native']['path'],source_case['scene_mjb']['path'],source_case['stored_shadow_command']['path']]+[x['path'] for x in source_case['frame_files'].values()]
        assert target['input_sha256']=={path:primary['source_sha256'][path] for path in case_paths}
        assert target['new_eval_data'] is False and target['physics_steps']==0 and target['training'] is False
        assert target['calibration_sha256']==candidate['calibration_sha256']
        source=pickle.loads(Path(source_case['native']['path']).read_bytes());native_camera=np.asarray(source['landmarks_3d'],float);TC=np.asarray(source['T_camera_wrist'],float);TWC=np.load(source_case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
        L=(native_camera-TC[:3,3])@TC[:3,:3];WM=TWC@TC;native_world=native_camera@TWC[:3,:3].T+TWC[:3,3]
        tasks=np.asarray(primary['task_human_indices']);native_tasks=native_world[tasks]
        targetP=(L-C[:3,3])@C[:3,:3];reference=(targetP@RP0.T)[tasks]
        Rraw=WM[:3,:3]@C[:3,:3]@RP0.T;RWB=Rotation.from_matrix(Rraw).as_matrix();anchor=WM[:3,3]+WM[:3,:3]@C[:3,3]
        compare(L,target['native_landmarks_wrist_MANO_m'],numerical,f'{index}_original_MANO_landmarks')
        compare(native_tasks,target['native_task_world_m'],numerical,f'{index}_native_WORLD_groundtruth')
        compare(WM,target['T_world_MANO'],numerical,f'{index}_native_world_wrist_transform')
        compare(WM[:3,3],target['native_wrist_world_position_m'],numerical,f'{index}_native_wrist_not_relabelled')
        compare(targetP[tasks],target['reference_vectors_neutral_palm_m'],numerical,f'{index}_C_translation_kept_in_reference')
        compare(reference,target['reference_vectors_URDF_base_m'],numerical,f'{index}_target_base_vectors')
        compare(Rraw,target['T_world_URDF_base_rotation_before_SO3_normalization'],numerical,f'{index}_no_second_A')
        compare(RWB,target['T_world_URDF_base_rotation'],numerical,f'{index}_normalized_carrier_rotation')
        compare(WM@C,target['neutral_target_palm_world_transform'],numerical,f'{index}_NEUTRAL_palm_only')
        compare(anchor,target['target_palm_world_position_m'],numerical,f'{index}_new_calibrated_anchor')
        compare(anchor-WM[:3,3],target['calibrated_anchor_minus_native_wrist_world_m'],numerical,f'{index}_declared_old_anchor_delta')
        assert abs(float(np.linalg.norm(anchor-WM[:3,3]))-target['calibrated_anchor_displacement_norm_m'])<=numerical
        native_reconstruction=float(np.max(np.abs(reference@RWB.T+anchor-native_tasks)))
        compare(native_reconstruction,target['source_native_transform_reconstruction_maxabs_m'],numerical,f'{index}_native_reference_roundtrip')
        normalization=float(np.max(np.abs(RWB-Rraw)))
        assert abs(normalization-target['SO3_root_rotation_normalization_maxabs'])<=numerical
        assert abs(float(np.linalg.norm(RWB-Rraw))-target['SO3_root_rotation_normalization_frobenius'])<=numerical
        names=target['joint_names'];old_names=old_target['joint_names']
        assert len(names)==len(set(names))==len(old_names)==len(set(old_names))==24 and set(names)==set(old_names)
        old_by_name=dict(zip(old_names,old_target['original_qpos_rad'],strict=True));old_in_current_order=np.asarray([old_by_name[n] for n in names])
        assert target['old_original_joint_names_saved_source_order']==old_names
        compare(target['old_original_qpos_rad_saved_source_order'],old_target['original_qpos_rad'],0.,f'{index}_saved_old_name_value_pair_preserved')
        ranges=np.asarray([physical[n] for n in names]);q=np.asarray(target['qpos_rad'],float);raw=np.asarray(target['raw_candidate_qpos_rad'],float)
        assert q.shape==raw.shape==(24,) and np.isfinite(q).all() and np.isfinite(raw).all()
        compare(ranges,target['source_physical_joint_bounds_rad'],0.,f'{index}_physical_source_limits')
        assert np.array_equal(q,np.clip(raw,ranges[:,0],ranges[:,1]))
        compare(q-raw,target['clamping_delta_rad'],0.,f'{index}_raw_to_physical_clamp')
        overflow=np.maximum(np.maximum(ranges[:,0]-raw,raw-ranges[:,1]),0.)
        compare(overflow,target['raw_joint_overflow_rad'],0.,f'{index}_raw_optimizer_overflow')
        rawpass=bool(np.all(overflow<=primary['existing_Dex_optimizer_bound_margin_rad']));physicalpass=bool(np.all(q>=ranges[:,0]) and np.all(q<=ranges[:,1]))
        history=target['retarget_history'];assert len(history)==primary['optimizer_repeats']==20
        assert [h['repeat'] for h in history]==list(range(1,21)) and np.array_equal(raw,history[-1]['raw_qpos_rad'])
        assert all(np.asarray(h['raw_qpos_rad']).shape==(24,) and np.isfinite(h['raw_qpos_rad']).all() and (h['nlopt_last_value'] is None or np.isfinite(h['nlopt_last_value'])) for h in history)
        solverpass=all(h['nlopt_result_code'] in primary['supported_NLopt_result_codes'] and h['nlopt_last_value'] is not None for h in history)
        compare(np.asarray(target['original_qpos_rad']),old_in_current_order,0.,f'{index}_old_joint_identity_by_name')
        compare(target['original_root_target_position_m'],old_target['original_root_target_position_m'],0.,f'{index}_old_root_position_identity')
        compare(target['original_root_target_quaternion_wxyz'],old_target['original_root_target_quaternion_wxyz'],0.,f'{index}_old_root_quaternion_identity')
        compare(q-old_in_current_order,target['joint_delta_from_old_mapping_rad'],0.,f'{index}_declared_full_physical_new_domain')
        validator.data.qpos[:]=reorder_for_mujoco(names,q,validator.joint_names);validator.data.qvel[:]=0;mujoco.mj_forward(validator.model,validator.data)
        Btask=_link_positions(validator.model,validator.data,urdf);Bpalm=validator.data.xpos[palm].copy();BactualR=validator.data.xmat[palm].reshape(3,3).copy();tWB=anchor-RWB@Bpalm
        compare(tWB,target['T_world_URDF_base_translation_m'],numerical,f'{index}_q_dependent_base_translation')
        compare(RWB@BactualR,target['target_palm_world_rotation'],numerical,f'{index}_actual_q_dependent_palm_R')
        m=mujoco.MjModel.from_binary_path(source_case['scene_mjb']['path']);d=mujoco.MjData(m)
        with np.load(source_case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:d.qpos[:]=z['qpos']
        ja=np.asarray([m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names],int);ra=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);oa=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'object_free')]);scene_palm=_body_id(m,'palm');scene_obj=_body_id(m,'object')
        d.qpos[ja]=old_in_current_order;d.qpos[ra:ra+3]=old_target['original_root_target_position_m'];d.qpos[ra+3:ra+7]=old_target['original_root_target_quaternion_wxyz'];d.qvel[:]=0
        if m.nmocap:d.mocap_pos[0]=d.qpos[ra:ra+3];d.mocap_quat[0]=d.qpos[ra+3:ra+7]
        mujoco.mj_forward(m,d);old_complete=d.qpos.copy();objectq=d.qpos[oa:oa+7].copy();objectpos=d.xpos[scene_obj].copy();objectR=d.xmat[scene_obj].copy()
        assert array_sha(old_complete)==prior_map[index]['scene_state_binding']['original_qpos_float64_sha256']
        old_vectors=_link_positions(m,d,urdf)-native_tasks;old_actual_caps=norm_rows(old_vectors[:5]);old_saved_caps=np.asarray(old_target['authoritative_original_tip_caps_sqrt_einsum_m'])
        compare(old_actual_caps,old_saved_caps,primary['old_cap_reconstruction_allowance_m'],f'{index}_original_actual_scene_caps')
        compare(old_saved_caps,target['old_original_five_tip_caps_sqrt_einsum_m'],0.,f'{index}_old_caps_not_rebased')
        compare(old_actual_caps,target['old_actual_scene_tip_caps_m'],numerical,f'{index}_driver_old_actual_caps')
        compare(old_vectors[:5],target['old_actual_scene_tip_error_vectors_m'],numerical,f'{index}_driver_old_actual_vectors')
        assert target['old_original_tip_caps_never_rebased']
        # Independent floating-model root compensation, not a copy of T_WB*T_BF.
        d.qpos[ja]=q;d.qpos[ra:ra+3]=0.;d.qpos[ra+3:ra+7]=[1.,0.,0.,0.]
        if m.nmocap:d.mocap_pos[0]=0.;d.mocap_quat[0]=[1.,0.,0.,0.]
        mujoco.mj_forward(m,d);p_free_palm=d.xpos[scene_palm].copy();R_free_palm=d.xmat[scene_palm].reshape(3,3).copy()
        independently_required_root_R=(RWB@BactualR)@R_free_palm.T
        independently_required_root_p=anchor-independently_required_root_R@p_free_palm
        rootpos=np.asarray(target['root_target_position_m']);rootquat=np.asarray(target['root_target_quaternion_wxyz']);assert abs(np.linalg.norm(rootquat)-1)<=numerical
        recorded_root_R=Rotation.from_quat(rootquat,scalar_first=True).as_matrix()
        root_R_difference=float(np.max(np.abs(recorded_root_R-independently_required_root_R)))
        root_p_difference=float(np.linalg.norm(rootpos-independently_required_root_p))
        assert root_R_difference<=primary['rotation_matrix_allowance'] and root_p_difference<=primary['fixed_URDF_vs_scene_link_allowance_m']
        fixed_root_position=tWB+RWB@pF
        compare(fixed_root_position,target['root_target_fixed_URDF_prediction_position_m'],numerical,f'{index}_source_fixed_forearm_origin')
        compare(rootpos-fixed_root_position,target['root_translation_scene_compensation_delta_m'],numerical,f'{index}_reported_compiled_translation_compensation')
        d.qpos[ja]=q;d.qpos[ra:ra+3]=rootpos;d.qpos[ra+3:ra+7]=rootquat
        if m.nmocap:d.mocap_pos[0]=rootpos;d.mocap_quat[0]=rootquat
        mujoco.mj_forward(m,d);actual=_link_positions(m,d,urdf);predicted=Btask@RWB.T+tWB
        link_error=float(np.linalg.norm(actual-predicted,axis=1).max());anchor_error=float(np.linalg.norm(d.xpos[scene_palm]-anchor));rotation_error=float(np.max(np.abs(d.xmat[scene_palm].reshape(3,3)-RWB@BactualR)))
        compare(actual,target['achieved_task_world_m'],numerical,f'{index}_actual_scene_task_points')
        errors=actual-native_tasks;tipnorm=norm_rows(errors[:5]);caps=tipnorm<=old_saved_caps;mean=float(tipnorm.mean());meanpass=bool(mean<=primary['maximum_mean_fingertip_error_m'])
        assert abs(mean-target['mean_fingertip_error_m'])<=numerical and abs(float(old_saved_caps.mean())-target['old_original_mean_tip_m'])<=numerical
        compare(errors,target['task_link_error_vectors_m'],numerical,f'{index}_unchanged_WORLD_groundtruth_errors')
        compare(tipnorm,target['tip_error_norms_sqrt_einsum_m'],numerical,f'{index}_actual_tip_norms')
        compare(tipnorm-old_saved_caps,target['old_tip_error_increase_m'],numerical,f'{index}_literal_old_cap_deltas')
        compare(native_world[[5,9,13]],target['native_MCP_index_middle_ring_world_m'],numerical,f'{index}_unchanged_native_MCP_world_points')
        compare(np.asarray([d.xpos[_body_id(m,n)] for n in ['ffknuckle','mfknuckle','rfknuckle']]),target['achieved_MCP_index_middle_ring_world_m'],numerical,f'{index}_achieved_MCP_world_points')
        wrist_position=d.xpos[_body_id(m,'wrist')]
        compare(wrist_position,target['achieved_robot_wrist_body_world_m'],numerical,f'{index}_actual_robot_wrist_body')
        compare(wrist_position-WM[:3,3],target['robot_wrist_body_minus_native_wrist_world_m'],numerical,f'{index}_explicit_robot_wrist_to_native_delta')
        compare(link_error,target['fixed_URDF_vs_scene_max_task_error_m'],numerical,f'{index}_reported_source_scene_error')
        compare(anchor_error,target['calibrated_anchor_error_m'],numerical,f'{index}_reported_new_anchor_error')
        compare(rotation_error,target['calibrated_actual_palm_rotation_maxabs_error'],numerical,f'{index}_reported_q_dependent_rotation_error')
        assert target['old_exact_tip_cap_pass']==caps.tolist() and target['strict_mean_guard_pass']==meanpass
        assert target['raw_optimizer_margin_pass']==rawpass and target['bounded_output_source_physical_limits_pass']==physicalpass
        reconstructionpass=bool(native_reconstruction<=primary['native_transform_reconstruction_allowance_m'] and link_error<=primary['fixed_URDF_vs_scene_link_allowance_m'] and anchor_error<=primary['anchor_position_allowance_m'] and rotation_error<=primary['rotation_matrix_allowance'])
        eligible=bool(solverpass and rawpass and physicalpass and caps.all() and meanpass and reconstructionpass)
        assert target['supported_solver_status_pass']==solverpass
        assert target['numeric_reconstruction_pass']==reconstructionpass and target['source_gate_eligible_for_required_radial_control']==eligible
        assert target['eligible_for_separate_full_surface_audit'] is False and target['required_new_anchor_radial_control']=='NOT_RUN'
        assert target['strict_surface_mapping_acceptance']=='NOT_RUN'
        assert d.time==0 and np.array_equal(d.qpos[oa:oa+7],objectq) and np.array_equal(d.xpos[scene_obj],objectpos) and np.array_equal(d.xmat[scene_obj],objectR)
        reasons=[]
        if not solverpass:reasons.append('unsupported_NLopt_status_or_nonfinite_last_value')
        if not rawpass:reasons.append('raw_optimizer_margin')
        if not physicalpass:reasons.append('physical_limits')
        if not caps.all():reasons.append('one_or_more_original_fingertip_caps_worsened')
        if not meanpass:reasons.append('strict_10mm_mean')
        if not reconstructionpass:reasons.append('numeric_source_frame_or_FK_gate')
        records.append({'source_inventory_index':index,'native_sha256':target['native_sha256'],'old_complete_scene_qpos_float64_sha256':array_sha(old_complete),'new_complete_scene_qpos_float64_sha256':array_sha(d.qpos),
            'native_WORLD_targets_reconstructed_without_C':True,'C_translation_preserved_in_targets':True,'no_second_A':True,
            'calibrated_anchor_independently_reconstructed_m':anchor.tolist(),'old_wrist_anchor_displacement_m':(anchor-WM[:3,3]).tolist(),'old_wrist_anchor_displacement_norm_m':float(np.linalg.norm(anchor-WM[:3,3])),
            'actual_palm_rotation_is_q_dependent':True,'source_neutral_frame_not_forced_as_actual_palm_R':True,
            'native_reference_roundtrip_maxabs_m':native_reconstruction,'source_scene_task_max_error_m':link_error,'actual_calibrated_anchor_error_m':anchor_error,'actual_source_palm_rotation_maxabs':rotation_error,
            'floating_root_required_R_vs_recorded_maxabs':root_R_difference,'floating_root_required_position_vs_recorded_norm_m':root_p_difference,
            'original_caps_sqrt_einsum_m':old_saved_caps.tolist(),'new_tip_norms_sqrt_einsum_m':tipnorm.tolist(),'old_cap_pass':caps.tolist(),'old_cap_increase_m':(tipnorm-old_saved_caps).tolist(),'old_tip_caps_never_rebased':True,
            'mean_tip_error_m':mean,'strict_mean_pass':meanpass,'raw_margin_pass':rawpass,'physical_limits_pass':physicalpass,'numeric_reconstruction_pass':reconstructionpass,'supported_solver_status_pass':solverpass,
            'joint_name_bijection_and_old_name_value_pairs_verified':True,'current_joint_names':names,'old_saved_joint_names':old_names,
            'independent_source_gate_eligible_for_required_radial_control':eligible,'eligibility_refusal_reasons':reasons,'retarget_iterations_recorded_not_replayed':20,
            'eligible_for_full_surface_audit':False,'required_new_anchor_radial_control':'NOT_RUN_IN_PRIMARY_CANDIDATE',
            'time_s':float(d.time),'object_state_and_world_pose_exact_original':True,'strict_surface_audit':'NOT_RUN'})
    all_eligible=all(r['independent_source_gate_eligible_for_required_radial_control'] for r in records)
    assert all_eligible==candidate['all_six_source_gate_eligible_for_required_radial_control']
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    assert sha(OUT/'protocol.json')==ph and sha(__file__)==p['script_sha256'] and COUNTS['actual']==0
    report={'schema':'hug_canonical_MCP_independent_calibration_and_pose_verification_v1','verification_execution_completed':True,'primary_driver_checks_and_this_independent_result_are_distinct':True,
        'protocol_sha256':ph,'script_sha256':sha(__file__),'primary_protocol_sha256':sha(p['primary_protocol']),'calibration_sha256':sha(p['calibration_path']),'candidate_sha256':sha(p['candidate_path']),
        'C_independently_reconstructed_from_bound_neutral_descriptors':C.tolist(),'F_MANO_independent':FM.tolist(),'F_palm_independent':FP.tolist(),'R_BP0_independent_source_FK':RP0.tolist(),
        'neutral_MANO_forward_independently_repeated':False,'neutral_MANO_descriptor_provenance':'Inherited from the primary separately frozen MANO neutral forward; frame algebra, URDF correspondences and case FK/world targets independently reconstructed.',
        'case_count':len(records),'all_six_independently_source_gate_eligible_for_required_radial_control':all_eligible,'independent_source_gate_eligible_case_count':sum(r['independent_source_gate_eligible_for_required_radial_control'] for r in records),
        'full_surface_audit_eligibility':'NOT_AUTHORIZED_PENDING_SEPARATE_RADIAL_CONTROL_AND_ROOT_REVIEW',
        'numeric_reconstruction_PASS_count':sum(r['numeric_reconstruction_pass'] for r in records),'original_individual_tip_cap_PASS_count':sum(sum(r['old_cap_pass']) for r in records),'cases':records,
        'guards':COUNTS,'physics_steps':0,'new_inference':False,'new_eval_data':False,'new_surface_queries':0,'new_optimizer_runs':0,'training':False,'production_changed':False,
        'strict_surface_mapping_acceptance':'NOT_RUN','failure_label_preservation':'NOT_TESTED','runtime_seconds':time.perf_counter()-started,'input_sha256':p['input_sha256'],
        'limitations':p['limitations']}
    with (OUT/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ['case_count','numeric_reconstruction_PASS_count','original_individual_tip_cap_PASS_count','independent_source_gate_eligible_case_count','all_six_independently_source_gate_eligible_for_required_radial_control','runtime_seconds']},indent=2))

if __name__=='__main__':main()
