"""Joint static HUG surface/direction correction on EXISTING saved proposals.

Frozen same-setting case cohorts are independent; no inference, controller,
physics integration, evaluation episode, outcome label or predictor training.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,pickle,time,warnings,os

# Runtime-only V2.1: bind thread controls before NumPy/SciPy/Numba imports.
RUNTIME_ENVIRONMENT={
    "OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
    "VECLIB_MAXIMUM_THREADS":"1","OMP_NUM_THREADS":"1",
    "NUMEXPR_NUM_THREADS":"1","NUMBA_NUM_THREADS":"4",
    "OMP_DYNAMIC":"FALSE","MKL_DYNAMIC":"FALSE"}
for variable,value in RUNTIME_ENVIRONMENT.items():os.environ[variable]=value
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

COUNTS={'actual':0,'selftests':0,'testing':False}
def forbidden(*args,**kwargs):
    COUNTS['selftests' if COUNTS['testing'] else 'actual']+=1
    raise RuntimeError('Joint static HUG correction forbids physics integration')
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden)

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT=Path(__file__).parent
OLD=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
direction=module('exact_direction_features',Path('/private/tmp/hug_constrained_validation_2026-10-14/exact_direction.py'))
a=direction.a
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden)
import numba
numba.set_num_threads(2)
from constrained_refinement import ConstrainedEvaluation,ConstrainedRefinementConfig,solve_constrained_refinement,native_direction_residual
from geometry_helpers import refresh_colliders,query_anatomical_part
from grasp_failure_prediction.evaluation.surface_refinement import RefinementConfig,select_sites
from grasp_failure_prediction.evaluation.pose_validation import _link_positions,ShadowPoseValidator,_body_id
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root,reorder_for_mujoco
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT

def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case-index',type=int,nargs='+',required=True)
    args=ap.parse_args();started=time.perf_counter();pp=OUT/'protocol.json';ph=a.sha(pp);p=read(pp)
    assert Path(direction.__file__).resolve()==Path(p['direction_helper_path']).resolve()
    indices=args.case_index
    plan_indices=[row['source_inventory_index'] for row in p['cases']]
    flattened=[index for cohort in p['allowed_cohorts'] for index in cohort]
    assert len(plan_indices)==len(set(plan_indices))==6
    assert len(flattened)==len(set(flattened))==6 and sorted(flattened)==sorted(plan_indices)
    assert len(p['allowed_cohorts'])==6 and all(len(cohort)==1 for cohort in p['allowed_cohorts'])
    assert p['runtime_protocol_version']==2 and p['runtime_amendment']==1
    assert p['numba_pool_capacity']==numba.config.NUMBA_NUM_THREADS==4
    assert p['numba_threads_per_process']==numba.get_num_threads()==2
    assert p['runtime_environment']==RUNTIME_ENVIRONMENT
    assert all(os.environ.get(variable)==value for variable,value in RUNTIME_ENVIRONMENT.items())
    if indices not in p['allowed_cohorts']:
        raise ValueError('Only the six predeclared singleton cohorts are permitted')
    suffix='_'.join(map(str,indices));report_path=OUT/f'candidate_cohort_{suffix}.json';partial_path=OUT/f'candidate_cohort_{suffix}.partial.json'
    if report_path.exists() or partial_path.exists():raise FileExistsError('Existing cohort evidence must not be overwritten')
    for path,h in p['source_sha256'].items():assert a.sha(path)==h,path
    for path,h in p['implementation_sha256'].items():assert a.sha(path)==h,path
    protected=read(OUT/'protected_hashes.json')
    assert a.sha(OUT/'protected_hashes.json')==p['protected_snapshot_sha256']
    for path,h in protected.items():assert a.sha(path)==h,path
    configuration=ConstrainedRefinementConfig(**p['solver_configuration'])
    surface_config=RefinementConfig(**p['surface_configuration'])
    orientation_p=read(OLD/'orientation/protocol.json')
    physical={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))
        for j in ET.parse(a.URDF).getroot().findall('joint') if j.find('limit') is not None}
    prior=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals'
    inventory=read(prior/'existing_hug_inventory.json');fkmap={c['native_sha256']:c for c in read(prior/'static_saved_hug_fk.json')['cases']}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
    weights=np.asarray(asset['weights']);grouped=np.column_stack([weights[:,ids].sum(1) for ids in a.GROUPS.values()])
    validator=ShadowPoseValidator(default_dex_urdf_root())
    COUNTS['testing']=True
    for api in ('mj_step','mj_step1','mj_step2'):
        try:getattr(mujoco,api)(None,None)
        except RuntimeError:pass
        else:raise AssertionError('Static integration guard failed')
    COUNTS['testing']=False
    planned={row['source_inventory_index']:row for row in p['cases']};records=[]
    for index in indices:
        tick=time.perf_counter();case=inventory['modern_cases'][index]
        assert case['native']['sha256']==planned[index]['native_sha256']
        meta,ntri,nlabels,nnorm,original_hand,objects,m,d,initial,objadr,opos,oR=a.load_case(case,grouped,fkmap)
        names=case['stored_shadow_command']['joint_names'];q0=np.array([meta['recovered_original_targets'][n] for n in names])
        assert len(names)==len(q0)==24
        limits=np.asarray([physical[n] for n in names]);assert np.all(q0>=limits[:,0]) and np.all(q0<=limits[:,1])
        nominal=d.qpos.copy();adr=np.array([m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names],int)
        palm=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');object_body=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'object')
        rootadr=int(m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')])
        rootpos=np.asarray(meta['root_target_world_position_m']);rootquat=np.asarray(meta['root_target_world_quaternion_wxyz'])
        source=pickle.loads(Path(case['native']['path']).read_bytes());T_wc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
        T_wrist=T_wc@np.asarray(source['T_camera_wrist'],float);Rop=T_wrist[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
        native_world=np.asarray(source['landmarks_3d'])@T_wc[:3,:3].T+T_wc[:3,3]
        native_tasks=native_world[np.array([4,8,12,16,20,2,6,10,14,18])]
        baseline_anchors=_link_positions(m,d,a.URDF).copy()
        visual_tri,visual_norm,visual_meta=direction.o.visual_mesh(meta,m,d,opos,oR)
        caseid=case['object_id']+'_'+Path(case['native']['path']).parent.name
        cache=Path(p['query_cache_root'])/f'{caseid}_density2mm_queries.npz';assert a.sha(cache)==p['query_cache_sha256'][str(cache)]
        with np.load(cache,allow_pickle=False) as f:cached={k:f[k].copy() for k in f.files}
        fullpath=Path(p['original_full_distance_root'])/f'{caseid}_density2mm_full_current.npz'
        assert a.sha(fullpath)==p['full_cache_sha256'][str(fullpath)]
        with np.load(fullpath,allow_pickle=False) as f:full={k:f[k].copy() for k in f.files}
        query,area,on=cached['query_object_m'],cached['area_mm2'],cached['visual_triangle_outward_normal']
        nd,sd=cached['native_distance_by_part_m'],full['current_distance_by_part_m']
        assert np.array_equal(full['query_object_m'],query)
        prepared={}
        for pi,part in enumerate(a.PARTS[:6]):
            sites,siteweights,sitemeta=select_sites(nd[:,pi],sd[:,pi],area,surface_config)
            subset=ntri[nlabels==pi];dist,ncp,nix=a.nearest_parts(query[sites],subset,np.zeros(len(subset),np.int64),1)
            assert np.max(np.abs(dist[:,0]-nd[sites,pi]))<1e-12
            prepared[part]={'query_indices':sites,'weights':siteweights,'selection':sitemeta,
                'native_distance_m':nd[sites,pi],'native_closest_object_m':ncp,'points':query[sites]}
        normal_reference,normal_coverage=direction.freeze_direction_sites(query,area,on,ntri,nlabels,nnorm,
            original_hand,visual_tri,orientation_p,nd,cached['native_inside'],cached['native_nearest_part'],full,
            bins_per_part=p['direction_configuration']['bins_per_part'])
        available=[part for part in a.PARTS[:6] if normal_reference[part]['available']]
        normal_sites=np.concatenate([normal_reference[part]['query_indices'] for part in available]) if available else np.empty(0,int)
        expected_parts=np.concatenate([np.full(len(normal_reference[part]['query_indices']),a.PARTS.index(part),int) for part in available]) if available else np.empty(0,int)
        native_distance=np.concatenate([v['native_distance_m'] for v in prepared.values()])
        native_cp=np.concatenate([v['native_closest_object_m'] for v in prepared.values()])
        quad_weights=np.concatenate([v['weights']/6 for v in prepared.values()]);sqrtw=np.sqrt(quad_weights)
        full_callback_count=0;tip_callback_count=0
        last_extra=None
        def set_pose(q):
            values=np.asarray(q,float);assert values.shape==(24,) and np.isfinite(values).all()
            d.qpos[:]=nominal;d.qpos[adr]=values;d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat
            mujoco.mj_forward(m,d)
            corrected=rootpos.copy() if np.array_equal(values,q0) else rootpos-(d.xpos[palm]-T_wrist[:3,3])
            d.qpos[rootadr:rootadr+3]=corrected
            if m.nmocap:d.mocap_pos[0]=corrected;d.mocap_quat[0]=rootquat
            mujoco.mj_forward(m,d)
            assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
            assert np.array_equal(d.xpos[object_body],opos) and np.array_equal(d.xmat[object_body].reshape(3,3),oR)
            assert np.linalg.norm(d.xpos[palm]-T_wrist[:3,3])<1e-12
            return corrected
        def evaluate(q,emit=True):
            nonlocal full_callback_count,last_extra
            full_callback_count+=1;corrected=set_pose(q)
            hand=refresh_colliders(original_hand,m,d,opos,oR)
            distance=[];closest=[]
            for part,v in prepared.items():
                pp=[g for g in hand if g['part']==part]
                ds,cp,_=query_anatomical_part(a,pp,v['points']);distance.append(ds);closest.append(cp)
            distances=np.concatenate(distance);points=np.concatenate(closest)
            normal_fields=None;normal_blocks=[];normal_diagnostics={};offset=0
            if len(normal_sites):
                normal_fields=direction.mapped_direction_fields(query[normal_sites],on[normal_sites],hand,visual_tri,
                    orientation_p,expected_part=expected_parts)
            for part in a.PARTS[:6]:
                ref=normal_reference[part]
                if not ref['available']:
                    normal_diagnostics[part]={'available':False,'fit_bin_count':0,'fixed_part_mass':0.,'reason':ref['reason']};continue
                count=len(ref['query_indices']);sl=slice(offset,offset+count);offset+=count
                fields={k:v[sl] if isinstance(v,np.ndarray) and v.ndim else v for k,v in normal_fields.items()}
                # The pure residual helper accepts normalized per-part weights;
                # fixed1/6 mass is applied separately, including empty parts.
                residual=native_direction_residual(ref['native_score'],fields['score'],ref['weights']*6,fields['valid'])/np.sqrt(6)
                legacy=direction.direction_residual(ref['native_score'],fields,ref['weights'],
                    p['direction_configuration']['score_weight'],p['direction_configuration']['invalid_penalty'])
                assert np.max(np.abs(residual-legacy))<1e-14
                normal_blocks.append(residual)
                normal_diagnostics[part]={'available':True,'fit_bin_count':count,'fixed_part_mass':1/6,
                    'valid_bin_count':int(fields['valid'].sum()),'invalid_bin_count':int((~fields['valid']).sum()),
                    'invalid_reason_bin_counts':{key:int(mask[sl].sum()) for key,mask in normal_fields['reason_masks'].items()},
                    'native_score':ref['native_score'].tolist(),'candidate_score':fields['score'].tolist(),
                    'candidate_valid':fields['valid'].tolist(),'native_negative_score_bin_count':int((ref['native_score']<0).sum()),
                    'valid_mapped_minus_native_score':(fields['score'][fields['valid']]-ref['native_score'][fields['valid']]).tolist(),
                    'residual':residual.tolist(),'invalid_sites_retained_at_nonzero_penalty':True,
                    'candidate_nearest_part':fields['part_index'].tolist(),'candidate_distance_m':fields['distance_m'].tolist(),
                    'candidate_outward_normal':fields['outward_normal'].tolist(),'candidate_closest_object_m':fields['closest_object_m'].tolist()}
            anchors=_link_positions(m,d,a.URDF)
            components={
                'distance':sqrtw*(distances-native_distance)/surface_config.distance_scale_m,
                'closest_vector':surface_config.closest_vector_weight*sqrtw[:,None]*(points-native_cp)/surface_config.distance_scale_m/np.sqrt(3),
                'facing':np.concatenate(normal_blocks) if normal_blocks else np.empty(0),
                'anchor':surface_config.anchor_weight*(anchors-baseline_anchors)/surface_config.anchor_scale_m/np.sqrt(baseline_anchors.size),
                'joint':surface_config.joint_weight*(np.asarray(q)-q0)/surface_config.joint_delta_limit_rad/np.sqrt(len(q0))}
            residuals=np.concatenate([v.ravel() for v in components.values()]);errors=anchors-native_tasks
            evaluation=ConstrainedEvaluation(residuals,errors[:5])
            cost=configuration.f_scale*np.sum(np.hypot(configuration.f_scale,residuals)-configuration.f_scale)
            last_extra={'objective':float(cost),'component_sse':{k:float(np.sum(v*v)) for k,v in components.items()},
                'task_link_error_vectors_m':errors.tolist(),'task_link_error_norm_m':np.linalg.norm(errors,axis=1).tolist(),
                'mean_fingertip_error_m':float(np.linalg.norm(errors[:5],axis=1).mean()),
                'mean_middle_error_m':float(np.linalg.norm(errors[5:],axis=1).mean()),
                'root_target_position_m':corrected.tolist(),'root_target_quaternion_wxyz':rootquat.tolist(),
                'normal_diagnostics':normal_diagnostics,'normal_bin_count':len(normal_sites),
                'normal_valid_bin_count':sum(v.get('valid_bin_count',0) for v in normal_diagnostics.values()),
                'normal_invalid_bin_count':sum(v.get('invalid_bin_count',0) for v in normal_diagnostics.values()),
                'fixed_quadrature_unsigned_distance_mae_m':float(np.dot(quad_weights,np.abs(distances-native_distance))),
                'fixed_quadrature_closest_point_mean_error_m':float(np.dot(quad_weights,np.linalg.norm(points-native_cp,axis=1))),
                'palm_anchor_position_error_m':float(np.linalg.norm(d.xpos[palm]-T_wrist[:3,3]))}
            if emit and full_callback_count%p['progress_full_evaluation_interval']==0:
                print(json.dumps({'progress_case':caseid,'full_surface_evaluations':full_callback_count,
                    'elapsed_seconds':time.perf_counter()-tick,'last_evaluated_objective':float(cost),
                    'last_evaluated_mean_tip_mm':last_extra['mean_fingertip_error_m']*1000,
                    'last_evaluated_invalid_direction_bins':last_extra['normal_invalid_bin_count'],
                    'not_necessarily_solver_major_iterate':True}),flush=True)
            return evaluation
        def evaluate_tips(q):
            nonlocal tip_callback_count
            tip_callback_count+=1;set_pose(q)
            return _link_positions(m,d,a.URDF)[:5]-native_tasks[:5]
        print(json.dumps({'starting':caseid,'source_inventory_index':index,'normal_available_parts':available,
            'normal_bins':len(normal_sites),'normal_source_coverage':normal_coverage['exact_original_shared_valid']}),flush=True)
        initial_evaluation=evaluate(q0,emit=False);baseline_fresh=jsonable(last_extra)
        returned,audit=solve_constrained_refinement(q0,limits,evaluate,configuration,evaluate_fingertips=evaluate_tips)
        # Cache hits do not imply MuJoCo side-state. Recompute final diagnostics.
        final_evaluation=evaluate(returned,emit=False);final_fresh=jsonable(last_extra)
        assert np.max(np.abs(final_evaluation.fingertip_error_vectors_m-np.asarray(audit['final_assessment']['tip_error_vectors_m'])))<1e-12
        assert abs(final_fresh['objective']-audit['final_assessment']['objective'])<1e-10
        assert np.max(np.abs(initial_evaluation.fingertip_error_vectors_m-np.asarray(audit['baseline']['tip_error_vectors_m'])))<1e-12
        assert abs(baseline_fresh['objective']-audit['baseline']['objective'])<1e-10
        assert final_fresh['mean_fingertip_error_m']<=configuration.maximum_mean_fingertip_error_m
        assert np.all(np.asarray(final_fresh['task_link_error_norm_m'][:5])<=
            np.asarray(baseline_fresh['task_link_error_norm_m'][:5])+configuration.feasibility_error_allowance_m)
        validator.data.qpos[:]=reorder_for_mujoco(names,returned,validator.joint_names);validator.data.qvel[:]=0
        mujoco.mj_forward(validator.model,validator.data);fixedp=_body_id(validator.model,'palm')
        desiredR=Rop@validator.data.xmat[fixedp].reshape(3,3)
        rotation_error=float(np.max(np.abs(d.xmat[palm].reshape(3,3)-desiredR)))
        fixedlinks=_link_positions(validator.model,validator.data,a.URDF)-validator.data.xpos[fixedp]
        expected_world=fixedlinks@Rop.T+T_wrist[:3,3];actual_world=_link_positions(m,d,a.URDF)
        scene_error=float(np.linalg.norm(actual_world-expected_world,axis=1).max())
        assert scene_error<1.1e-7 and rotation_error<1e-6
        ranges=[]
        for name,value in zip(names,returned,strict=True):
            low,high=physical[name];assert low<=value<=high
            jid=a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,name);sl,sh=m.jnt_range[jid];delta=max(abs(sl-low),abs(sh-high))
            overflow=max(sl-value,value-sh,0.);assert overflow<=delta+1e-9
            ranges.append({'joint':name,'source_URDF_range_rad':[low,high],'scene_range_rad':[float(sl),float(sh)],
                'scene_representation_difference_rad':float(delta),'strict_scene_overflow_rad':float(overflow)})
        record={'source_inventory_index':index,'native_sha256':case['native']['sha256'],'object_id':case['object_id'],
            'source_case':Path(case['native']['path']).parent.name,'joint_names':names,'qpos_rad':returned.tolist(),
            'original_qpos_rad':q0.tolist(),'raw_candidate_qpos_rad':audit['raw_candidate_qpos_rad'],
            'root_target_position_m':final_fresh['root_target_position_m'],'root_target_quaternion_wxyz':rootquat.tolist(),
            'original_root_target_position_m':rootpos.tolist(),'original_root_target_quaternion_wxyz':rootquat.tolist(),
            'target_palm_world_position_m':T_wrist[:3,3].tolist(),'target_palm_world_rotation':desiredR.tolist(),
            'T_world_operator_rotation':Rop.tolist(),'palm_anchor_position_fixed':True,'solver_audit':audit,
            'baseline_fresh_diagnostics':baseline_fresh,'selected_fresh_diagnostics':final_fresh,
            'original_task_link_error_m':baseline_fresh['task_link_error_norm_m'],
            'candidate_task_link_error_m':final_fresh['task_link_error_norm_m'],
            'source_preflight':{'original_computed_mean_tip_guard_pass':baseline_fresh['mean_fingertip_error_m']<=.010,
                'selected_computed_mean_tip_guard_pass':final_fresh['mean_fingertip_error_m']<=.010,
                'selected_individual_tip_nonworsening_with_10nm_arithmetic_allowance_pass':bool(np.all(
                    np.asarray(final_fresh['task_link_error_norm_m'][:5])<=
                    np.asarray(baseline_fresh['task_link_error_norm_m'][:5])+configuration.feasibility_error_allowance_m)),
                'surface_candidate_returned':audit['surface_candidate_returned'],
                'selected_candidate_source':audit['selected_candidate_source'],
                'selected_major_iteration':audit['selected_major_iteration'],
                'fallback_original_target_exact':bool(np.array_equal(returned,q0)),
                'not_scientific_mapping_acceptance':True},
            'surface_reference_by_part':jsonable(prepared),'direction_reference_by_part':jsonable(normal_reference),
            'direction_reference_coverage':normal_coverage,'palm_rotation_convention_maxabs_error':rotation_error,
            'independent_fixed_URDF_vs_scene_max_link_error_m':scene_error,'actual_source_ranges_audit':ranges,
            'input_sha256':meta['input_sha256'],'original_query_cache_sha256':a.sha(cache),
            'original_full_distance_cache_sha256':a.sha(fullpath),'visible_object_geometry':visual_meta,
            'full_callback_count_including_fresh_final':full_callback_count,'tip_callback_count':tip_callback_count,
            'final_state_freshly_recomputed_outside_solver_cache':True,'source_poor_relationships_not_replaced_by_quality_reward':True,
            'time_s':float(d.time),'object_state_and_world_pose_unchanged':True,'runtime_seconds':time.perf_counter()-tick,
            'new_eval_data':False,'new_generator_proposal':False,'physics_steps':0,'training':False,'mapping_acceptance':False}
        for path,h in meta['input_sha256'].items():assert a.sha(path)==h,path
        assert np.array_equal(d.qpos[objadr:objadr+7],initial) and d.time==0
        records.append(record)
        write(partial_path,{'schema':'hug_global_constrained_surface_candidate_cohort_v1','partial':True,
            'protocol_sha256':ph,'cohort_case_indices':indices,'cases':records})
        print(json.dumps({'completed_case':caseid,'selected_candidate':audit['surface_candidate_returned'],
            'selected_iteration':audit.get('selected_major_iteration'),'solver_success':audit['solver_success'],
            'solver_iterations':audit.get('iterations'),'objective_baseline':baseline_fresh['objective'],
            'objective_selected':final_fresh['objective'],'mean_tip_baseline_mm':baseline_fresh['mean_fingertip_error_m']*1000,
            'mean_tip_selected_mm':final_fresh['mean_fingertip_error_m']*1000,'seconds':record['runtime_seconds']}),flush=True)
    for path,h in p['source_sha256'].items():assert a.sha(path)==h,path
    for path,h in p['implementation_sha256'].items():assert a.sha(path)==h,path
    for path,h in protected.items():assert a.sha(path)==h,path
    assert a.sha(pp)==ph and COUNTS['actual']==0
    assert all(getattr(mujoco,api) is forbidden for api in ('mj_step','mj_step1','mj_step2'))
    report={'schema':'hug_global_constrained_surface_candidate_cohort_v1','protocol_sha256':ph,
        'script_sha256':a.sha(__file__),'solver_implementation_sha256':a.sha(OUT/'constrained_refinement.py'),
        'direction_implementation_sha256':a.sha(p['direction_helper_path']),'cohort_case_indices':indices,'cases':records,
        'runtime_protocol_version':2,'runtime_amendment':1,
        'numba_pool_capacity':numba.config.NUMBA_NUM_THREADS,'numba_threads_per_process':numba.get_num_threads(),
        'runtime_environment':RUNTIME_ENVIRONMENT,
        'runtime_seconds':time.perf_counter()-started,'protected_source_count':len(protected),'protected_hashes_unchanged':True,
        'original_inputs_and_implementations_unchanged':True,'guards':COUNTS,'new_eval_data':False,
        'new_generator_proposal':False,'physics_steps':0,'training':False,'production_changes':False,
        'strict_acceptance_rule_unchanged_sha256':p['strict_acceptance_rule_sha256'],
        'limitations':p['limitations'],'scope':'Original six saved development grasps only; independent frozen-rule audit still required.'}
    write(report_path,report)
    print(json.dumps({'completed_cohort':indices,'report':str(report_path),'seconds':report['runtime_seconds']}),flush=True)

if __name__=='__main__':main()
