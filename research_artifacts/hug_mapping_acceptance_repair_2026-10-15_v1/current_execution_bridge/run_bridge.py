"""Frozen static execution bridge for all 32 saved nominal Shadow commands."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from xml_reference import (read_urdf,forward_xml,source_limit_rows,obj_vertices,
                           transform,quaternion_matrix_wxyz)
from bridge_model import assign_named,assign_carrier,compare_body_frames,compare_collisions,body_world,ident,joint_metadata_against_source

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x


def main():
    protocol_path=HERE/'protocol.json';protocol=json.loads(protocol_path.read_text());protocol_hash=sha(protocol_path)
    assert protocol['status']=='FROZEN_AUTHORIZED_STATIC_ONLY'
    authorization=json.loads((HERE/'authorization.json').read_text())
    assert authorization['protocol_sha256']==protocol_hash and authorization['authorized'] is True
    def verify():
        assert sha(protocol_path)==protocol_hash,'Protocol changed'
        for path,digest in protocol['bindings'].items():assert sha(path)==digest,path
    verify()
    output=HERE/'results.json'
    assert not output.exists() and not (HERE/'cases').exists(),'Never overwrite bridge evidence'
    import mujoco
    assert mujoco.__version__==protocol['runtime']['mujoco_version']
    assert np.__version__==protocol['runtime']['numpy_version']
    physics=[0];forwards=[0]
    def forbidden(*args,**kwargs):physics[0]+=1;raise RuntimeError('Bridge forbids all physics integration')
    for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden)
    original_forward=mujoco.mj_forward
    def forward(*args,**kwargs):forwards[0]+=1;return original_forward(*args,**kwargs)
    mujoco.mj_forward=forward
    from grasp_failure_prediction.evaluation import pose_validation as pv
    from grasp_failure_prediction.evaluation import retargeting as ret
    ret.ShadowHandRetargeter.retarget=forbidden
    assert Path(pv.__file__).resolve()==Path(protocol['current_pose_validation_source']).resolve()
    assert sha(pv.__file__)==protocol['bindings'][str(Path(pv.__file__).resolve())]
    source=read_urdf(protocol['source_urdf']);urdf_root=Path(protocol['source_urdf']).parent.parent
    source_meshes={(c['path'],tuple(c['scale'])):obj_vertices(c['path'],c['scale']) for c in source['collisions'] if c['kind']=='mesh'}
    historical=json.loads(Path(protocol['historical_static_report']).read_text());history={c['native_sha256']:c for c in historical['cases']}
    inventory=json.loads(Path(protocol['inventory_path']).read_text())
    fixed=pv.ShadowPoseValidator(urdf_root)
    floating=pv.load_shadow_model(urdf_root,floating_root=True);floating_data=mujoco.MjData(floating)
    assert set(fixed.joint_names)=={j['name'] for j in source['joints'] if j['kind']=='revolute'}
    tol=protocol['engineering_tolerances'];records=[];started=time.perf_counter();(HERE/'cases').mkdir()
    for frozen in protocol['cases']:
        index=frozen['source_inventory_index'];case=inventory['modern_cases'][index]
        assert case['native']['sha256']==frozen['native_sha256']
        old=history[frozen['native_sha256']];command=case['stored_shadow_command']
        assert sha(command['path'])==command['sha256']==frozen['command_sha256']
        with np.load(command['path'],allow_pickle=False) as z:
            row=command['row_index'];names=z['joint_names'].tolist();q=z['nominal_target_rad'][row].copy()
            position=z['root_target_position_m'][row].copy();quaternion=z['root_target_quaternion_wxyz'][row].copy()
        assert names==frozen['joint_names']
        for actual,key in [(q,'qpos_rad'),(position,'root_position_m'),(quaternion,'root_quaternion_wxyz')]:
            assert np.array_equal(actual,np.asarray(frozen[key])),key
        source_limits=source_limit_rows(source,names,q)
        reference_fixed=forward_xml(source,names,q)
        carrier=transform(position,quaternion_matrix_wxyz(quaternion));reference_world=forward_xml(source,names,q,carrier)
        # Context arrays are deliberately unused by set_pose; no native pose is
        # inferred or fabricated. Only original names/q enter this API call.
        command_pose=ret.RetargetedHandPose(tuple(names),q,np.zeros((21,3)),np.zeros((10,3)),np.eye(4))
        fixed.set_pose(command_pose)
        floating_data.qpos[:]=floating.qpos0;floating_data.qvel[:]=0.
        current_joint_rows=assign_named(floating,floating_data,names,q)
        root_address=assign_carrier(floating,floating_data,position,quaternion)
        mujoco.mj_forward(floating,floating_data)
        assert fixed.data.time==floating_data.time==0.
        named_roundtrip=all(floating_data.qpos[r['qpos_address']]==v for r,v in zip(current_joint_rows,q,strict=True))
        current_body_fixed=compare_body_frames(source,reference_fixed,fixed.model,fixed.data,tol['current_position_m'],tol['current_rotation_matrix'])
        current_body_world=compare_body_frames(source,reference_world,floating,floating_data,tol['current_position_m'],tol['current_rotation_matrix'])
        current_geom_fixed=compare_collisions(source,reference_fixed,fixed.model,fixed.data,source_meshes,tol)
        current_geom_world=compare_collisions(source,reference_world,floating,floating_data,source_meshes,tol)
        joint_metadata=joint_metadata_against_source(source,current_joint_rows,tol['source_compiled_joint_limit_rad'])
        # The production validator already wrote these same named values; this
        # readback reassignment must be bitwise inert and does not call forward.
        fixed_before=fixed.data.qpos.copy()
        fixed_joint_rows=assign_named(fixed.model,fixed.data,names,q)
        assert np.array_equal(fixed_before,fixed.data.qpos)
        fixed_joint_metadata=joint_metadata_against_source(source,fixed_joint_rows,tol['source_compiled_joint_limit_rad'])
        # Literal replay of the historical captured model. This is a distinct
        # measurement, not a loader overlay or substitute for current execution.
        assert sha(case['scene_mjb']['path'])==case['scene_mjb']['sha256']
        old_model=mujoco.MjModel.from_binary_path(case['scene_mjb']['path']);old_data=mujoco.MjData(old_model)
        with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:old_data.qpos[:]=z['qpos']
        object_joint=ident(old_model,mujoco.mjtObj.mjOBJ_JOINT,'object_free');oa=int(old_model.jnt_qposadr[object_joint]);object_q=old_data.qpos[oa:oa+7].copy()
        old_joint_rows=assign_named(old_model,old_data,names,q);assign_carrier(old_model,old_data,position,quaternion)
        old_data.qvel[:]=0.;mujoco.mj_forward(old_model,old_data)
        assert old_data.time==0. and np.array_equal(old_data.qpos[oa:oa+7],object_q)
        old_world=pv._link_positions(old_model,old_data,Path(protocol['source_urdf']))
        current_world=pv._link_positions(floating,floating_data,Path(protocol['source_urdf']))
        current_fixed=pv._link_positions(fixed.model,fixed.data,Path(protocol['source_urdf']))
        xml_world=np.array([reference_world[n][:3,3] for n in pv.TASK_LINK_NAMES])
        xml_fixed=np.array([reference_fixed[n][:3,3] for n in pv.TASK_LINK_NAMES])
        historical_world=np.array([r['shadow_world_m'] for r in old['links']])
        assert [r['shadow_link'] for r in old['links']]==list(pv.TASK_LINK_NAMES)
        saved_replay=float(np.max(np.linalg.norm(old_world-historical_world,axis=1)))
        current_xml=float(np.max(np.linalg.norm(current_world-xml_world,axis=1)))
        fixed_xml=float(np.max(np.linalg.norm(current_fixed-xml_fixed,axis=1)))
        current_old=float(np.max(np.linalg.norm(current_world-old_world,axis=1)))
        historical_geometry_vs_source=compare_collisions(source,reference_world,old_model,old_data,source_meshes,tol,model_is_hand_only=False)
        forearm=body_world(floating,floating_data,'forearm');palm=body_world(floating,floating_data,'palm')
        old_palm=body_world(old_model,old_data,'palm')
        carrier_position_error=float(np.linalg.norm(forearm[:3,3]-position));carrier_rotation_error=float(np.max(np.abs(forearm[:3,:3]-carrier[:3,:3])))
        source_targets_in_limits=all(r['literal_pass'] for r in source_limits)
        execution_pass=bool(named_roundtrip and current_body_fixed['passed'] and current_body_world['passed'] and current_geom_fixed['passed'] and current_geom_world['passed'] and
                            all(r['source_compiled_limits_match'] for r in joint_metadata+fixed_joint_metadata) and current_xml<=tol['current_position_m'] and fixed_xml<=tol['current_position_m'] and
                            carrier_position_error<=tol['current_position_m'] and carrier_rotation_error<=tol['current_rotation_matrix'])
        record={'schema':'hug_current_execution_bridge_case_v1','source_inventory_index':index,'native_sha256':frozen['native_sha256'],
                'protocol_sha256':protocol_hash,'literal_stored_command':frozen,'command_targets_unchanged':True,
                'current_named_joint_command_roundtrip_exact':named_roundtrip,'current_joint_metadata':joint_metadata,'current_fixed_joint_metadata':fixed_joint_metadata,
                'literal_source_joint_limit_rows':source_limits,'all_literal_source_limits_pass':source_targets_in_limits,
                'current_fixed_body_and_virtual_links':current_body_fixed,'current_world_body_and_virtual_links':current_body_world,
                'current_fixed_collision_geometry':current_geom_fixed,'current_world_collision_geometry':current_geom_world,
                'historical_geometry_against_current_source_reference':historical_geometry_vs_source,
                'historical_compiled_joint_rows':old_joint_rows,
                'task_links':list(pv.TASK_LINK_NAMES),'current_world_task_positions_m':current_world,'xml_world_task_positions_m':xml_world,
                'historical_replayed_task_positions_m':old_world,'immutable_historical_task_positions_m':historical_world,
                'current_vs_xml_max_task_position_error_m':current_xml,'current_fixed_vs_xml_max_task_position_error_m':fixed_xml,
                'current_vs_historical_max_task_position_change_m':current_old,'historical_replay_vs_immutable_max_error_m':saved_replay,
                'historical_replay_matches':bool(saved_replay<=tol['historical_replay_position_m']),
                'stored_carrier_quaternion_norm':float(np.linalg.norm(quaternion)),
                'current_carrier_position_error_m':carrier_position_error,'current_carrier_rotation_matrix_error':carrier_rotation_error,
                'current_world_palm_transform':palm,'xml_world_palm_transform':reference_world['palm'],
                'current_vs_historical_palm_position_change_m':float(np.linalg.norm(palm[:3,3]-old_palm[:3,3])),
                'command_realization_pass':execution_pass,'command_within_source_limits_and_realized':bool(execution_pass and source_targets_in_limits),
                'no_native_anatomical_equivalence_claim':True,'physics_steps':0,'new_eval_data':False,'new_retargeting_solves':0}
        for p,h in frozen['input_bindings'].items():assert sha(p)==h,p
        target=HERE/'cases'/f'case_{index:02d}.json';assert not target.exists();target.write_text(json.dumps(plain(record),indent=2,allow_nan=False)+'\n')
        records.append({'index':index,'path':str(target),'sha256':sha(target),'command_realization_pass':execution_pass,
                        'literal_source_limits_pass':source_targets_in_limits,'historical_replay_matches':record['historical_replay_matches'],
                        'max_current_vs_xml_task_error_m':current_xml,'max_current_vs_historical_task_change_m':current_old,
                        'current_collision_geometry_pass':current_geom_world['passed'],'historical_collision_geometry_matches_source':historical_geometry_vs_source['passed']})
        print(json.dumps(records[-1]),flush=True)
    verify();assert physics[0]==0 and forwards[0]==3*len(protocol['cases'])
    report={'schema':'hug_current_execution_bridge_v1','protocol_sha256':protocol_hash,'scope':protocol['scope'],
            'cases':records,'case_count':len(records),'all_current_command_realizations_pass':all(c['command_realization_pass'] for c in records),
            'all_literal_source_limits_pass':all(c['literal_source_limits_pass'] for c in records),
            'all_historical_static_replays_match':all(c['historical_replay_matches'] for c in records),
            'actual_static_forward_calls':forwards[0],'physics_steps':physics[0],'new_eval_data':False,'new_retargeting_solves':0,
            'current_source_bindings_unchanged_after_run':True,'runtime_seconds':time.perf_counter()-started,
            'mujoco_version':mujoco.__version__,'numpy_version':np.__version__,'python':sys.version,
            'does_not_validate_native_geometry_or_new_retargeter_profile_or_se3_candidates':True}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'summary':str(output),'sha256':sha(output),'counts':len(records),'forwards':forwards[0]}))

if __name__=='__main__':main()
