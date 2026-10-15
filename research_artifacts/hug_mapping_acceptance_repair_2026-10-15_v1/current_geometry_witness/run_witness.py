"""Current complete representation, fixed historical queries, eight static poses."""
from pathlib import Path
import gzip,hashlib,json,pickle,sys,time
import numpy as np
import mujoco
from witness_check import PARTS,anatomical_part,descendant,fixed_rows,evaluate_fixed
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def main():
    protocol_path=HERE/'protocol.json';p=read(protocol_path);ph=sha(protocol_path)
    assert p['status']=='FROZEN_AUTHORIZED_STATIC_ONLY'
    authorization=read(HERE/'authorization.json');assert authorization['authorized'] and authorization['protocol_sha256']==ph
    def verify():
        assert sha(protocol_path)==ph
        for path,h in p['bindings'].items():assert sha(path)==h,path
    verify();assert not (HERE/'cases').exists() and not (HERE/'results.json').exists()
    assert np.__version__==p['runtime']['numpy'] and mujoco.__version__==p['runtime']['mujoco']
    # Import only frozen mathematical kernels, never the historical Case or preload overlay.
    import exact_direction as direction
    import constraint_core as core
    from geometry_helpers import query_anatomical_part
    from scene_runtime import build_current_runner,reset_captured_scene,stage_named_root,stage_named_palm
    from grasp_failure_prediction.evaluation import pose_validation as pv,retargeting as ret,case_runner,runner as runner_module
    from grasp_failure_prediction.integrations.hug import load_hug_prediction
    from native_reference import expected_native_palm
    from xml_reference import read_urdf,forward_xml,transform,quaternion_matrix_wxyz
    import numba
    numba.set_num_threads(2);assert numba.get_num_threads()==2
    assert Path(pv.__file__).resolve()==Path(p['current_pose_source']).resolve()
    assert tuple(direction.a.PARTS)==PARTS
    a=direction.a;counts={'forwards':0,'steps':0,'retarget':0}
    def forbidden_step(*args,**kw):counts['steps']+=1;raise RuntimeError('No integration')
    def forbidden_retarget(*args,**kw):counts['retarget']+=1;raise RuntimeError('No retargeting solve')
    for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden_step)
    ret.ShadowHandRetargeter.retarget=forbidden_retarget
    case_runner._run_case_inference=forbidden_retarget;case_runner.run_case=forbidden_retarget
    for cls in (runner_module.AdroitShadowRunner,runner_module.ActuatedShadowRunner):
        cls.execute=forbidden_retarget;cls._advance=forbidden_retarget
    original_forward=mujoco.mj_forward
    def forward(*args,**kw):counts['forwards']+=1;return original_forward(*args,**kw)
    mujoco.mj_forward=forward
    saved=read(p['saved_witnesses']);inventory=read(p['inventory']);op=read(p['orientation_protocol'])
    xmlsource=read_urdf(p['source_urdf']);validator=pv.ShadowPoseValidator(Path(p['source_urdf']).parent.parent)
    (HERE/'cases').mkdir();records=[];started=time.perf_counter();contexts={}
    for target in p['poses']:
        verify();idx=target['source_inventory_index'];source=inventory['modern_cases'][idx]
        assert source['native']['sha256']==target['native_sha256']
        pi,oi,prox,ori=fixed_rows(saved,idx,target['native_sha256'])
        assert pi==target['proximity_source_indices'] and oi==target['orientation_source_indices']
        assert len(prox)==target['proximity_count'] and len(ori)==target['orientation_count']
        names=target['joint_names'];q=np.asarray(target['qpos_rad']);rp=np.asarray(target['root_position_m']);rq=np.asarray(target['root_quaternion_wxyz'])
        if target['role']=='original_current_nominal':
            with np.load(source['stored_shadow_command']['path'],allow_pickle=False) as z:
                row=source['stored_shadow_command']['row_index']
                assert names==z['joint_names'].tolist() and np.array_equal(q,z['nominal_target_rad'][row])
                assert np.array_equal(rp,z['root_target_position_m'][row]) and np.array_equal(rq,z['root_target_quaternion_wxyz'][row])
        else:
            candidate=read(target['candidate_path']);x=np.asarray(candidate['returned_state_scaled'])
            assert np.array_equal(q,np.asarray(target['original_qpos_rad'])+.025*x[:24])
            assert np.array_equal(rp,candidate['returned_exact']['root_target_position_m'])
            assert np.array_equal(rq,candidate['returned_exact']['root_target_quaternion_wxyz'])
        if idx not in contexts:
            # Historical binary is used ONLY to recover the immutable object visual reference.
            # No historical hand surface query, target correction or native refit is performed.
            old=mujoco.MjModel.from_binary_path(source['scene_mjb']['path']);od=mujoco.MjData(old)
            with np.load(source['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:od.qpos[:]=z['qpos']
            od.qvel[:]=0.;mujoco.mj_forward(old,od);ob=a.ident(old,mujoco.mjtObj.mjOBJ_BODY,'object')
            pos=od.xpos[ob].copy();R=od.xmat[ob].reshape(3,3).copy()
            visual=[{'id':g} for g in range(old.ngeom) if descendant(old.body_parentid,int(old.geom_bodyid[g]),ob) and not (old.geom_contype[g] or old.geom_conaffinity[g])]
            assert len(visual)==1 and int(old.geom_type[visual[0]['id']])==int(mujoco.mjtGeom.mjGEOM_MESH)
            visual_tri,_,visual_meta=direction.o.visual_mesh({'disabled_visual_geoms':visual},old,od,pos,R)
            saved_resolved=read(source['resolved_case_path']);payload=saved_resolved['case'];contract=read(source['frame_files']['scene_contract.json']['path'])
            runner,resolved,geometry=build_current_runner(payload,p['runtime_project_root'],Path(p['source_urdf']).parent.parent)
            assert resolved.environment.runner=='grasp_failure_prediction.evaluation.runner:ActuatedShadowRunner'
            assert resolved.environment_config_hash==saved_resolved['environment_config_hash']
            assert resolved.execution_protocol_config_hash==saved_resolved['execution_protocol_config_hash']
            reset_captured_scene(runner,payload,contract)
            native=pickle.loads(Path(source['native']['path']).read_bytes());T=np.load(source['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
            native_tasks=(np.asarray(native['landmarks_3d'])@T[:3,:3].T+T[:3,3])[[4,8,12,16,20,2,6,10,14,18]]
            contexts[idx]=(runner,pos,R,visual_tri,visual_meta,native_tasks)
            del old,od
        runner,pos,R,visual_tri,visual_meta,native_tasks=contexts[idx]
        m,d=runner.model,runner.data;historical_rp=rp.copy();historical_rq=rq.copy();placement_reference=None
        if target['role']=='original_current_nominal':
            nativepath=Path(source['native']['path']);grasp=load_hug_prediction(nativepath)
            raw=pickle.loads(nativepath.read_bytes());wrist=np.asarray(raw['T_camera_wrist'],float).reshape(4,4)
            camera=np.load(source['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
            assert np.array_equal(camera,np.load(nativepath.parent/'T_world_camera.npy',allow_pickle=False))
            fixed=forward_xml(xmlsource,names,q);expected=expected_native_palm(camera,wrist,fixed['palm'])
            pose=ret.RetargetedHandPose(tuple(names),q,np.zeros((21,3)),np.zeros((10,3)),wrist)
            palmpos,palmquat=case_runner._hug_world_palm_pose(grasp,pose,validator,nativepath.parent)
            actualpalm=transform(palmpos,quaternion_matrix_wxyz(palmquat))
            assert np.linalg.norm(actualpalm[:3,3]-expected['palm_world'][:3,3])<=p['frame_assertion_m']
            assert np.max(np.abs(actualpalm[:3,:3]-expected['palm_world'][:3,:3]))<=p['rotation_assertion']
            rp,rq,_=stage_named_palm(runner,names,q,actualpalm)
            local=np.linalg.inv(fixed['forearm'])@fixed['palm'];wantedR=expected['palm_world'][:3,:3]@local[:3,:3].T
            wantedp=expected['palm_world'][:3,3]-wantedR@local[:3,3]
            assert np.linalg.norm(rp-wantedp)<=p['frame_assertion_m']
            assert np.max(np.abs(quaternion_matrix_wxyz(rq)-wantedR))<=p['rotation_assertion']
            assert np.linalg.norm(d.xpos[runner._palm_body_id]-expected['palm_world'][:3,3])<=p['frame_assertion_m']
            assert np.max(np.abs(d.xmat[runner._palm_body_id].reshape(3,3)-expected['palm_world'][:3,:3]))<=p['rotation_assertion']
            placement_reference={'native_reference':expected,'actual_helper_palm':actualpalm,'source_XML_root_position_m':wantedp,'source_XML_root_rotation':wantedR}
        else:stage_named_root(runner,names,q,rp,rq)
        obj=int(runner._object_body_id);object_position_error=float(np.linalg.norm(d.xpos[obj]-pos));object_rotation_error=float(np.max(np.abs(d.xmat[obj].reshape(3,3)-R)))
        assert object_position_error<=p['frame_assertion_m'] and object_rotation_error<=p['rotation_assertion']
        assert d.time==0 and np.array_equal(d.qpos[runner._root_qpos_address:runner._root_qpos_address+3],rp)
        assert np.array_equal(d.qpos[runner._root_qpos_address+3:runner._root_qpos_address+7],rq)
        assert m.npair==0 and m.nplugin==0 and m.nflex==0
        assert len(names)==24 and len(set(names))==24
        for name,value in zip(names,q,strict=True):
            jid=a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,name)
            assert d.qpos[int(m.jnt_qposadr[jid])]==value
        forearm=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'forearm')
        objects=[g for g in range(m.ngeom) if descendant(m.body_parentid,int(m.geom_bodyid[g]),obj) and (m.geom_contype[g] or m.geom_conaffinity[g])]
        exclusions=[sorted([int(s)>>16,int(s)&65535]) for s in m.exclude_signature]
        hand=[];geom_metadata=[]
        for g in range(m.ngeom):
            body=int(m.geom_bodyid[g])
            if not descendant(m.body_parentid,body,forearm) or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            assert m.geom_margin[g]==0. and m.geom_gap[g]==0.
            assert objects and all(m.geom_margin[o]==0. and m.geom_gap[o]==0. for o in objects)
            assert objects and all(((int(m.geom_contype[g])&int(m.geom_conaffinity[o])) or (int(m.geom_contype[o])&int(m.geom_conaffinity[g]))) and sorted([body,int(m.geom_bodyid[o])]) not in exclusions for o in objects)
            rep=a.geom_rep(m,d,g,R,pos);rep['part']=anatomical_part(rep['body']);hand.append(rep)
            geom_metadata.append({k:rep[k] for k in ('id','body','part','kind','representation','mesh','contype','conaffinity')})
        assert len(hand)==p['expected_hand_collider_count'] and set(g['part'] for g in hand)==set(PARTS)
        assert len(objects)==target['expected_object_collider_count']
        result=evaluate_fixed(hand,visual_tri,prox,ori,op,a,query_anatomical_part,direction,core)
        caps=np.asarray(target['historical_tip_caps_m']);errors=pv._link_positions(m,d,Path(p['source_urdf']))-native_tasks
        tips=core.exact_tip_status(errors[:5],caps)
        assert np.array_equal(caps,read(target['historical_cap_report'])['engineering_guard_independent_audit']['original_tip_caps_sqrt_einsum_m'])
        historical=read(target['historical_witness_report'])
        if target['role']=='original_current_nominal':
            oldcase=next(c for c in historical['cases'] if c['source_inventory_index']==idx)
            oldexact=oldcase['baseline_constraint_diagnostics']
        else:oldexact=historical['returned_exact']
        assert len(oldexact['proximity']['passed'])==len(prox) and len(oldexact['exact_orientation']['passed'])==len(ori)
        record={'schema':'hug_current_geometry_fixed_witness_case_v1','pose_id':target['pose_id'],'protocol_sha256':ph,
            'source_inventory_index':idx,'native_sha256':target['native_sha256'],'frozen_pose':target,
            'current_hand_colliders':geom_metadata,'current_object_collider_ids':objects,
            'frozen_visual_reference':visual_meta,'object_frame_error_m':object_position_error,'object_rotation_matrix_error':object_rotation_error,
            'current_task_error_vectors_m':errors,'historical_tip_caps_m':caps,'current_tip_status_against_literal_historical_caps':tips,
            'historical_tip_cap_excess_m':tips['norms_m']-caps,'historical_caps_rebased':False,
            'current_palm_world_position_m':d.xpos[runner._palm_body_id].copy(),'current_palm_world_rotation':d.xmat[runner._palm_body_id].reshape(3,3).copy(),
            'current_witness':result,'historical_reference_witness':{'proximity_passed':oldexact['proximity']['passed'],'orientation_passed':oldexact['exact_orientation']['passed'],'orientation_valid':oldexact['exact_orientation_fields']['valid'],
                'proximity_distance_m':oldexact['part_distance_m'][:len(prox)],'orientation_distance_m':oldexact['exact_orientation_fields']['distance_m'],'orientation_score':oldexact['exact_orientation_fields']['score']},
            'actual_root_position_m':rp,'actual_root_quaternion_wxyz':rq,
            'root_position_change_from_saved_m':float(np.linalg.norm(rp-historical_rp)),
            'root_rotation_matrix_change_from_saved':float(np.max(np.abs(quaternion_matrix_wxyz(rq)-quaternion_matrix_wxyz(historical_rq)))),
            'current_nominal_placement_reference':placement_reference,
            'comparison_includes_current_frame_and_geometry_changes':target['role']=='original_current_nominal',
            'experimental_literal_root_preserved':target['role']!='original_current_nominal','retained_subset_only':True,'no_current_full_grid_mapping_acceptance':True,
            'physics_steps':counts['steps'],'new_eval_data':False,'new_proposals':False}
        verify();path=HERE/'cases'/(target['pose_id']+'.json');assert not path.exists();path.write_text(json.dumps(plain(record),indent=2,allow_nan=False)+'\n')
        records.append({'pose_id':target['pose_id'],'path':str(path),'sha256':sha(path),**result['summary'],'historical_tipcaps_pass':tips['passed']})
        print(json.dumps(records[-1]),flush=True)
    verify();assert counts['steps']==counts['retarget']==0 and numba.get_num_threads()==2
    assert len(records)==p['finite_budget']['poses'] and sum(c['proximity_rows']+c['orientation_rows'] for c in records)==p['finite_budget']['retained_row_evaluations']
    assert counts['forwards']<=p['finite_budget']['maximum_static_forwards']
    report={'schema':'hug_current_geometry_fixed_witness_v1','protocol_sha256':ph,'cases':records,'counts':counts,
        'source_closure_unchanged':True,'runtime_seconds':time.perf_counter()-started,'native_queries_recomputed':False,
        'thresholds_changed':False,'historical_caps_rebased':False,'no_new_eval_data':True,'full_current_mapping_status':'NOT_EVALUATED_BY_RETAINED_SUBSET'}
    (HERE/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'results_sha256':sha(HERE/'results.json'),'counts':counts}),flush=True)
if __name__=='__main__':main()
