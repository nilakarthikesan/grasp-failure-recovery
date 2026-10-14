"""Predeclared V3 engineering-guard derivative of unchanged V2 candidates.

Static FK only on six existing poses. No new proposal, episode or label.
"""
from pathlib import Path
import hashlib,json,pickle,time,xml.etree.ElementTree as ET,warnings
import numpy as np
import mujoco
import run_refinement as rt
from surface_refinement import RefinementConfig,select_sites
from guarded_refinement import GuardConfig,guarded_joint_selection
from grasp_failure_prediction.evaluation.pose_validation import _link_positions,ShadowPoseValidator,_body_id
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root,reorder_for_mujoco
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT

ROOT=rt.ROOT;OUT=Path(__file__).parent;a=rt.a
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    start=time.perf_counter();pp=OUT/'guard_protocol.json';ph=a.sha(pp);p=read(pp)
    for path,h in p['source_sha256'].items():assert a.sha(path)==h,path
    for path,h in p['implementation_sha256'].items():assert a.sha(path)==h,path
    protected=read(OUT/'protected_hashes.json')
    for path,h in protected.items():assert a.sha(path)==h,path
    raw=read(OUT/'candidate_targets.json');v2protocol=read(OUT/'protocol.json')
    assert raw['protocol_sha256']==p['raw_protocol_sha256']==a.sha(OUT/'protocol.json')
    assert a.sha(OUT/'candidate_targets.json')==p['raw_v2_report_sha256']
    assert len(raw['cases'])==6
    config=RefinementConfig(**v2protocol['configuration']);guard=GuardConfig(**p['guard_configuration'])
    validator=ShadowPoseValidator(default_dex_urdf_root())
    prior=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals'
    inventory=read(prior/'existing_hug_inventory.json');fk=read(prior/'static_saved_hug_fk.json');fkmap={c['native_sha256']:c for c in fk['cases']}
    physical={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))
        for j in ET.parse(a.URDF).getroot().findall('joint') if j.find('limit') is not None}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
    weights=np.asarray(asset['weights']);gw=np.column_stack([weights[:,ids].sum(1) for ids in a.GROUPS.values()])
    rt.GUARDS['testing']=True
    for api in ('mj_step','mj_step1','mj_step2'):
        try:getattr(mujoco,api)(None,None)
        except RuntimeError:pass
        else:raise AssertionError('No-time-integration guard failed')
    rt.GUARDS['testing']=False
    records=[]
    for rawcase in raw['cases']:
        case_start=time.perf_counter();index=rawcase['source_inventory_index'];case=inventory['modern_cases'][index]
        assert case['native']['sha256']==rawcase['native_sha256']
        meta,tri,labels,norm,hand,objects,m,d,initial,objadr,pos,R=a.load_case(case,gw,fkmap)
        nominal=d.qpos.copy();names=rawcase['joint_names'];q0=np.asarray(rawcase['original_qpos_rad']);qraw=np.asarray(rawcase['qpos_rad'])
        assert np.array_equal(q0,np.asarray([meta['recovered_original_targets'][n] for n in names]))
        adr=np.array([m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names],int)
        rootadr=int(m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')])
        palm=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');rootpos=np.asarray(meta['root_target_world_position_m']);rootquat=np.asarray(meta['root_target_world_quaternion_wxyz'])
        anchors0=_link_positions(m,d,a.URDF)
        source=pickle.loads(Path(case['native']['path']).read_bytes());T_wc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
        T_wm=T_wc@np.asarray(source['T_camera_wrist'],float);Rop=T_wm[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
        native_world=np.asarray(source['landmarks_3d'])@T_wc[:3,:3].T+T_wc[:3,3]
        native_tasks=native_world[np.array([4,8,12,16,20,2,6,10,14,18])]
        caseid=case['object_id']+'_'+Path(case['native']['path']).parent.name
        cache=Path(v2protocol['query_cache_root'])/f'{caseid}_density2mm_queries.npz';assert a.sha(cache)==v2protocol['query_cache_sha256'][str(cache)]
        with np.load(cache,allow_pickle=False) as arr:cached={k:arr[k].copy() for k in arr.files}
        fullpath=Path(v2protocol['original_full_distance_root'])/f'{caseid}_density2mm_full_current.npz';assert a.sha(fullpath)==v2protocol['full_cache_sha256'][str(fullpath)]
        with np.load(fullpath,allow_pickle=False) as arr:full={k:arr[k].copy() for k in arr.files}
        query=cached['query_object_m'];area=cached['area_mm2'];nd=cached['native_distance_by_part_m'];sd=full['current_distance_by_part_m']
        prepared={}
        for part in a.PARTS[:6]:
            pi=a.PARTS.index(part);sites,siteweights,sitemeta=select_sites(nd[:,pi],sd[:,pi],area,config)
            points=query[sites];ntri=tri[labels==pi]
            ndpart,ncp,nix=a.nearest_parts(points,ntri,np.zeros(len(ntri),np.int64),1)
            assert np.max(np.abs(ndpart[:,0]-nd[sites,pi]))<1e-12
            if part!='palm':
                assert np.array_equal(sites,rawcase['part_diagnostics'][part]['query_indices'])
                assert np.array_equal(siteweights,rawcase['part_diagnostics'][part]['query_weights'])
                assert np.max(np.abs(ncp-np.asarray(rawcase['part_diagnostics'][part]['native_closest_point_object_m'])))<1e-12
            prepared[part]={'sites':sites,'weights':siteweights,'meta':sitemeta,'points':points,
                'native_distance':nd[sites,pi],'native_closest':ncp,'geoms':[g for g in hand if g['part']==part]}
        joined_nd=np.concatenate([v['native_distance'] for v in prepared.values()])
        joined_cp=np.concatenate([v['native_closest'] for v in prepared.values()])
        joined_weights=np.concatenate([v['weights']/6 for v in prepared.values()]);sqrtw=np.sqrt(joined_weights)
        def set_pose(qfull):
            d.qpos[:]=nominal;d.qpos[adr]=qfull;d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat
            mujoco.mj_forward(m,d)
            # Literal original q/root preserved on fallback and baseline.
            corrected=rootpos.copy() if np.array_equal(qfull,q0) else rootpos-(d.xpos[palm]-T_wm[:3,3])
            d.qpos[rootadr:rootadr+3]=corrected
            if m.nmocap:d.mocap_pos[0]=corrected;d.mocap_quat[0]=rootquat
            mujoco.mj_forward(m,d)
            assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
            assert np.linalg.norm(d.xpos[palm]-T_wm[:3,3])<1e-12
            return corrected
        def assess(qfull):
            corrected=set_pose(qfull);ds=[];cp=[]
            for part,v in prepared.items():
                transformed=[rt.refresh_geom(g,m,d,pos,R) for g in v['geoms']]
                dd,cc,_=rt.query_part(transformed,v['points']);ds.append(dd);cp.append(cc)
            distances=np.concatenate(ds);closest=np.concatenate(cp);anchors=_link_positions(m,d,a.URDF)
            components={
                'distance':sqrtw*(distances-joined_nd)/config.distance_scale_m,
                'closest_vector':config.closest_vector_weight*sqrtw[:,None]*(closest-joined_cp)/config.distance_scale_m/np.sqrt(3),
                'anchor':config.anchor_weight*(anchors-anchors0)/config.anchor_scale_m/np.sqrt(anchors0.size),
                'joint':config.joint_weight*(qfull-q0)/config.joint_delta_limit_rad/np.sqrt(len(q0))}
            residual=np.concatenate([v.ravel() for v in components.values()])
            cost=float(config.f_scale**2*np.sum(np.sqrt(1+(residual/config.f_scale)**2)-1))
            errors=np.linalg.norm(anchors-native_tasks,axis=1)
            return {'objective':cost,'component_sse':{k:float(np.sum(v*v)) for k,v in components.items()},
                'mean_fingertip_error_m':float(errors[:5].mean()),'mean_middle_error_m':float(errors[5:].mean()),
                'mean_all10_error_m':float(errors.mean()),'task_link_error_m':errors.tolist(),
                'root_target_position_m':corrected.tolist(),'root_target_quaternion_wxyz':rootquat.tolist(),
                'fixed_quadrature_unsigned_distance_mae_m':float(np.dot(joined_weights,np.abs(distances-joined_nd))),
                'fixed_quadrature_closest_point_mean_error_m':float(np.dot(joined_weights,np.linalg.norm(closest-joined_cp,axis=1))),
                'palm_anchor_error_m':float(np.linalg.norm(d.xpos[palm]-T_wm[:3,3]))}
        guarded_q,detail=guarded_joint_selection(q0,qraw,np.asarray([physical[n] for n in names]),assess,guard)
        final=assess(guarded_q)
        validator.data.qpos[:]=reorder_for_mujoco(names,guarded_q,validator.joint_names);validator.data.qvel[:]=0;mujoco.mj_forward(validator.model,validator.data)
        fixedp=_body_id(validator.model,'palm');desiredR=Rop@validator.data.xmat[fixedp].reshape(3,3)
        rotation_error=float(np.max(np.abs(d.xmat[palm].reshape(3,3)-desiredR)));assert rotation_error<1e-6
        fixedlinks=_link_positions(validator.model,validator.data,a.URDF)-validator.data.xpos[fixedp]
        expected_world=fixedlinks@Rop.T+T_wm[:3,3];actual_world=_link_positions(m,d,a.URDF)
        link_error=float(np.linalg.norm(actual_world-expected_world,axis=1).max());assert link_error<1.1e-7
        record={'source_inventory_index':index,'native_sha256':rawcase['native_sha256'],'object_id':case['object_id'],
            'source_case':rawcase['source_case'],'joint_names':names,'qpos_rad':guarded_q.tolist(),
            'original_qpos_rad':q0.tolist(),'raw_v2_qpos_rad':qraw.tolist(),
            'root_target_position_m':final['root_target_position_m'],'root_target_quaternion_wxyz':final['root_target_quaternion_wxyz'],
            'original_root_target_position_m':rootpos.tolist(),'original_root_target_quaternion_wxyz':rootquat.tolist(),
            'target_palm_world_position_m':T_wm[:3,3].tolist(),'target_palm_world_rotation':desiredR.tolist(),
            'T_world_operator_rotation':Rop.tolist(),'engineering_guard':detail,'final_assessment':final,
            'original_source_inputs_sha256':meta['input_sha256'],'input_sha256':meta['input_sha256'],
            'fixed_URDF_vs_scene_max_link_error_m':link_error,'palm_rotation_convention_maxabs_error':rotation_error,
            'palm_anchor_position_fixed':True,'candidate_link_world_m':actual_world.tolist(),
            'query_indices_by_part':{part:v['sites'].tolist() for part,v in prepared.items()},
            'query_weights_by_part':{part:v['weights'].tolist() for part,v in prepared.items()},
            'maximum_abs_joint_delta_rad':float(np.max(np.abs(guarded_q-q0))),
            'runtime_seconds':time.perf_counter()-case_start,'new_eval_data':False,'physics_steps':0,
            'mapping_acceptance_claim':False,'outcome_labels':False}
        records.append(record);write(OUT/'guarded_candidate_targets.partial.json',{'partial':True,'cases':records})
        for path,h in meta['input_sha256'].items():assert a.sha(path)==h,path
        print(json.dumps({'case':caseid,'selected_alpha':detail['selected_alpha'],'original_objective':detail['baseline']['objective'],
            'guarded_objective':final['objective'],'original_mean_tip_mm':detail['baseline']['mean_fingertip_error_m']*1000,
            'guarded_mean_tip_mm':final['mean_fingertip_error_m']*1000,'seconds':record['runtime_seconds']}),flush=True)
    for path,h in p['source_sha256'].items():assert a.sha(path)==h,path
    for path,h in p['implementation_sha256'].items():assert a.sha(path)==h,path
    for path,h in protected.items():assert a.sha(path)==h,path
    assert a.sha(pp)==ph and rt.GUARDS['actual']==0
    report={'schema':'hug_surface_refinement_guarded_candidate_v3','protocol_sha256':ph,'script_sha256':a.sha(__file__),
        'guard_implementation_sha256':a.sha(OUT/'guarded_refinement.py'),'raw_v2_report_sha256':p['raw_v2_report_sha256'],
        'raw_v2_protocol_sha256':p['raw_protocol_sha256'],'cases':records,'runtime_seconds':time.perf_counter()-start,
        'guards':rt.GUARDS,'protected_hashes_unchanged':True,'source_hashes_unchanged':True,
        'new_eval_data':False,'new_generator_proposal':False,'physics_steps':0,'training':False,'production_changes':False,
        'strict_rule_unchanged_sha256':v2protocol['strict_acceptance_rule_unchanged_sha256'],
        'limitations':p['limitations'],'meaning':'Guarded development mapping targets; independent strict preservation audit required.'}
    write(OUT/'guarded_candidate_targets.json',report)
    print(json.dumps({'completed':True,'case_count':len(records),'seconds':report['runtime_seconds']}),flush=True)

if __name__=='__main__':main()
