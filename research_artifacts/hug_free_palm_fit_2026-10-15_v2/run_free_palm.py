"""Per-grasp translated-palm bounded pattern study on EXISTING saved HUG proposals.

Frozen same-setting case cohorts are independent; no inference, controller,
physics integration, evaluation episode, outcome label or predictor training.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,pickle,time,warnings,os,gzip

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
direction=module('exact_direction_features',OUT/'exact_direction.py')
a=direction.a
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden)
import numba
numba.set_num_threads(2)
from grasp_failure_prediction.evaluation.constrained_surface_refinement import ConstrainedEvaluation,native_direction_residual
from free_palm_solver import FreePalmConfig,solve_free_palm
from free_palm_kinematics import decode_state,urdf_tip_reaches,state_bounds,initial_seeds,virtual_tip_jacobian,verify_jacobian,stage_translated_palm
from geometry_helpers import refresh_colliders,query_anatomical_part
from surface_component_guard import freeze_direction_component_guard
from keyed_direction_evidence import KeyedDirectionEvidence
from grasp_failure_prediction.evaluation.surface_refinement import RefinementConfig,select_sites
from grasp_failure_prediction.evaluation.pose_validation import _link_positions,ShadowPoseValidator,_body_id,_fixed_tip_offsets,TASK_LINK_NAMES
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root,reorder_for_mujoco
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT

def read(path):
    path=Path(path);return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_text())
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
    assert sorted(set(flattened))==sorted(plan_indices)
    assert p['method_version']==2 and p['method_stage']=='free_palm_fine_translation_seed_v2'
    assert p['execution_authorized_by_root_review'], 'Frozen root execution review required'
    assert p['solver_configuration']['maximum_pattern_sweeps']==0
    assert p['solver_configuration']['maximum_full_geometry_evaluations']==64
    assert p['numba_pool_capacity']==numba.config.NUMBA_NUM_THREADS==4
    assert p['numba_threads_per_process']==numba.get_num_threads()==2
    assert p['runtime_environment']==RUNTIME_ENVIRONMENT
    assert all(os.environ.get(variable)==value for variable,value in RUNTIME_ENVIRONMENT.items())
    if indices not in p['allowed_cohorts']:
        raise ValueError('Only predeclared case cohorts are permitted')
    suffix='_'.join(map(str,indices));report_path=OUT/f'candidate_cohort_{suffix}.json';partial_path=OUT/f'candidate_cohort_{suffix}.partial.json'
    if report_path.exists() or partial_path.exists():raise FileExistsError('Existing cohort evidence must not be overwritten')
    for path,h in p['source_sha256'].items():assert a.sha(path)==h,path
    for path,h in p['implementation_sha256'].items():assert a.sha(path)==h,path
    protected=read(OUT/'protected_hashes.json')
    assert a.sha(OUT/'protected_hashes.json')==p['protected_snapshot_sha256']
    for path,h in protected.items():assert a.sha(path)==h,path
    configuration=FreePalmConfig(**p['solver_configuration'])
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
        state0=np.zeros(27)
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
        wrist_body=_body_id(m,'wrist')
        original_wrist_world_position=d.xpos[wrist_body].copy();original_wrist_world_rotation=d.xmat[wrist_body].reshape(3,3).copy()
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
        full_callback_count=0;tip_callback_count=0;analytic_callback_count=0;set_pose_calls=0
        dofs=np.asarray([m.jnt_dofadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names],int)
        tipspec=_fixed_tip_offsets(a.URDF)
        tipbodies=np.asarray([_body_id(m,tipspec[n][0]) for n in TASK_LINK_NAMES[:5]],int)
        last_extra=None
        def set_pose(state):
            nonlocal set_pose_calls
            set_pose_calls+=1
            _,shift=decode_state(state,q0)
            target_palm=T_wrist[:3,3]+shift
            corrected=stage_translated_palm(m,d,nominal,adr,rootadr,palm,rootpos,rootquat,T_wrist[:3,3],q0,state)
            assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
            assert np.array_equal(d.xpos[object_body],opos) and np.array_equal(d.xmat[object_body].reshape(3,3),oR)
            assert np.linalg.norm(d.xpos[palm]-target_palm)<1e-12
            assert np.array_equal(d.qpos[rootadr+3:rootadr+7],rootquat)
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
                'joint':surface_config.joint_weight*(decode_state(q,q0)[0]-q0)/surface_config.joint_delta_limit_rad/np.sqrt(len(q0))}
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
                'palm_anchor_position_error_m':float(np.linalg.norm(d.xpos[palm]-(T_wrist[:3,3]+decode_state(q,q0)[1]))),
                'palm_displacement_world_m':decode_state(q,q0)[1].tolist()}
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
        def evaluate_tip_jacobian(state):
            nonlocal analytic_callback_count
            analytic_callback_count+=1;set_pose(state)
            return virtual_tip_jacobian(m,d,palm,_link_positions(m,d,a.URDF)[:5],tipbodies,dofs)
        initial_errors=evaluate_tips(state0)
        original_caps=np.sqrt(np.einsum('ij,ij->i',initial_errors,initial_errors))
        cap_binding=p['prior_original_cap_bindings'][str(index)]
        assert a.sha(cap_binding['path'])==cap_binding['sha256']
        prior_cap_record=read(cap_binding['path'])
        assert prior_cap_record['source_inventory_index']==index and prior_cap_record['native_sha256']==case['native']['sha256']
        prior_caps=np.asarray(prior_cap_record['engineering_guard_independent_audit']['original_tip_caps_sqrt_einsum_m'])
        assert np.array_equal(original_caps,prior_caps), 'Original cap rebasing is forbidden'
        bounds,domain=state_bounds(q0,limits,native_tasks[:5],original_caps,T_wrist[:3,3],urdf_tip_reaches(a.URDF))
        seeds,seed_labels=initial_seeds(bounds)
        assert seeds.shape==(56,27)
        jacobian_gate=verify_jacobian(state0,bounds,evaluate_tips,evaluate_tip_jacobian,
            **p['analytic_jacobian_verification'])
        pre_solver_tip_count=tip_callback_count;pre_solver_analytic_count=analytic_callback_count
        print(json.dumps({'starting':caseid,'source_inventory_index':index,'normal_available_parts':available,
            'normal_bins':len(normal_sites),'normal_source_coverage':normal_coverage['exact_original_shared_valid'],
            'jacobian_maxabs_m_per_state':jacobian_gate['maxabs_difference_m_per_scaled_state']}),flush=True)
        baseline_holder={};terminal_holder={}
        evidence=KeyedDirectionEvidence(state0,normal_reference,normal_block_start=len(native_distance)*4,normal_block_count=len(normal_sites),trace_directory=OUT/'full_callback_traces'/str(index))
        def evaluate_for_solver(q):
            value=evaluate(q,emit=True)
            evidence.record(q,value,jsonable(last_extra))
            if np.array_equal(q,state0) and not baseline_holder:
                baseline_holder.update(evaluation=value,diagnostics=jsonable(last_extra))
            terminal_holder.update(qpos_rad=np.asarray(q).copy(),evaluation=value,diagnostics=jsonable(last_extra))
            return value
        returned_state,audit=solve_free_palm(state0,bounds,evaluate_for_solver,config=configuration,evaluate_fingertips=evaluate_tips,predeclared_seed_state_scaled=seeds,candidate_guard=evidence.assess,evaluate_fingertip_jacobian=evaluate_tip_jacobian)
        returned,selected_shift=decode_state(returned_state,q0)
        assert baseline_holder and np.array_equal(terminal_holder['qpos_rad'],returned_state)
        assert audit['final_callbacks_fresh_outside_cache'] and audit['counters']['full']==full_callback_count
        assert 2<=full_callback_count<=configuration.maximum_full_geometry_evaluations
        initial_evaluation=baseline_holder['evaluation'];baseline_fresh=baseline_holder['diagnostics']
        terminal_direction_guard=evidence.assess(returned_state,terminal_holder['evaluation'])
        assert terminal_direction_guard['accepted']
        assert terminal_direction_guard['metadata_callback_sequence']==full_callback_count
        assert evidence.callback_count==full_callback_count
        assert evidence.all_available_baseline_parts_valid
        final_evaluation=terminal_holder['evaluation'];final_fresh=terminal_holder['diagnostics']
        selected_source='guarded_'+('fine_translation_seed_after_FK_restore') if audit['surface_candidate_returned'] else 'original_fallback'
        audit['selected_candidate_source']=selected_source
        audit['selected_major_iteration']=None  # New method has seed/proposal sequences, no SLSQP surface major iterates.
        audit['seed_labels']=seed_labels
        assert audit['counters']['FK']==tip_callback_count-pre_solver_tip_count
        assert audit['counters']['analytic_Jacobians']==analytic_callback_count-pre_solver_analytic_count
        # Independently stage the returned pose with FK only after fresh terminal surface diagnostics.
        preflight_errors=evaluate_tips(returned_state)
        assert np.max(np.abs(preflight_errors-final_evaluation.fingertip_error_vectors_m))<1e-12
        assert np.max(np.abs(final_evaluation.fingertip_error_vectors_m-np.asarray(audit['final_assessment']['tip_error_vectors_m'])))<1e-12
        assert abs(final_fresh['objective']-audit['final_assessment']['objective'])<1e-10
        assert np.max(np.abs(initial_evaluation.fingertip_error_vectors_m-np.asarray(audit['baseline']['tip_error_vectors_m'])))<1e-12
        assert abs(baseline_fresh['objective']-audit['baseline']['objective'])<1e-10
        # Exact guard arithmetic is declared sqrt(einsum), consistent with new API.
        # np.linalg.norm task diagnostics remain separately recorded.
        original_cap_norms=np.sqrt(np.einsum('ij,ij->i',initial_evaluation.fingertip_error_vectors_m,initial_evaluation.fingertip_error_vectors_m))
        final_exact_norms=np.sqrt(np.einsum('ij,ij->i',preflight_errors,preflight_errors))
        assert np.array_equal(original_cap_norms,np.asarray(audit['baseline']['tip_error_norm_m']))
        assert np.array_equal(original_cap_norms,prior_caps)
        assert np.all(final_exact_norms<=original_cap_norms)
        assert final_exact_norms.mean()<=configuration.maximum_mean_fingertip_error_m
        validator.data.qpos[:]=reorder_for_mujoco(names,returned,validator.joint_names);validator.data.qvel[:]=0
        mujoco.mj_forward(validator.model,validator.data);fixedp=_body_id(validator.model,'palm')
        desiredR=Rop@validator.data.xmat[fixedp].reshape(3,3)
        rotation_error=float(np.max(np.abs(d.xmat[palm].reshape(3,3)-desiredR)))
        fixedlinks=_link_positions(validator.model,validator.data,a.URDF)-validator.data.xpos[fixedp]
        expected_world=fixedlinks@Rop.T+T_wrist[:3,3]+selected_shift;actual_world=_link_positions(m,d,a.URDF)
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
            'original_qpos_rad':q0.tolist(),'original_state_scaled':state0.tolist(),'selected_state_scaled':returned_state.tolist(),
            'palm_displacement_world_m':selected_shift.tolist(),'palm_displacement_norm_m':float(np.linalg.norm(selected_shift)),
            'original_shadow_wrist_world_position_m':original_wrist_world_position.tolist(),
            'selected_shadow_wrist_world_position_m':d.xpos[wrist_body].tolist(),
            'original_shadow_wrist_world_rotation':original_wrist_world_rotation.tolist(),
            'selected_shadow_wrist_world_rotation':d.xmat[wrist_body].reshape(3,3).tolist(),
            'shadow_wrist_world_displacement_m':(d.xpos[wrist_body]-original_wrist_world_position).tolist(),
            'floating_forearm_world_displacement_m':(np.asarray(final_fresh['root_target_position_m'])-rootpos).tolist(),
            'state_coordinates':p['state_coordinates'],'domain':domain,'analytic_jacobian_baseline_check':jacobian_gate,
            'root_target_position_m':final_fresh['root_target_position_m'],'root_target_quaternion_wxyz':rootquat.tolist(),
            'original_root_target_position_m':rootpos.tolist(),'original_root_target_quaternion_wxyz':rootquat.tolist(),
            'target_palm_world_position_m':(T_wrist[:3,3]+selected_shift).tolist(),'target_palm_world_rotation':d.xmat[palm].reshape(3,3).tolist(),'source_expected_palm_world_rotation':desiredR.tolist(),
            'T_world_operator_rotation':Rop.tolist(),'palm_anchor_position_fixed':False,'palm_orientation_source_convention_unchanged':True,'solver_audit':audit,
            'authoritative_original_tip_caps_sqrt_einsum_m':original_cap_norms.tolist(),
            'prior_original_cap_binding':cap_binding,'prior_original_caps_reproduced_exactly':True,
            'selected_tip_norms_sqrt_einsum_m':final_exact_norms.tolist(),
            'exact_guard_norm_definition':'sqrt(einsum(ij,ij->i, vectors, vectors)); linalg norm diagnostics remain separate',
            'baseline_fresh_diagnostics':baseline_fresh,'selected_fresh_diagnostics':final_fresh,
            'terminal_direction_guard':terminal_direction_guard,'keyed_direction_evidence':evidence.audit(),
            'original_task_link_error_m':baseline_fresh['task_link_error_norm_m'],
            'candidate_task_link_error_m':final_fresh['task_link_error_norm_m'],
            'source_preflight':{'original_computed_mean_tip_guard_pass':baseline_fresh['mean_fingertip_error_m']<=.010,
                'selected_computed_mean_tip_guard_pass':bool(final_exact_norms.mean()<=.010),
                'selected_individual_tip_nonworsening_exact_computed_pass':bool(np.all(final_exact_norms<=original_cap_norms)),
                'surface_candidate_returned':audit['surface_candidate_returned'],
                'selected_candidate_source':audit['selected_candidate_source'],
                'selected_major_iteration':audit['selected_major_iteration'],
                'fallback_original_target_exact':bool(np.array_equal(returned_state,state0)),
                'not_scientific_mapping_acceptance':True},
            'surface_reference_by_part':jsonable(prepared),'direction_reference_by_part':jsonable(normal_reference),
            'direction_reference_coverage':normal_coverage,'palm_rotation_convention_maxabs_error':rotation_error,
            'independent_fixed_URDF_vs_scene_max_link_error_m':scene_error,'actual_source_ranges_audit':ranges,
            'input_sha256':meta['input_sha256'],'original_query_cache_sha256':a.sha(cache),
            'original_full_distance_cache_sha256':a.sha(fullpath),'visible_object_geometry':visual_meta,
            'full_callback_count_including_fresh_final':full_callback_count,'tip_callback_count':tip_callback_count,
            'analytic_jacobian_callback_count':analytic_callback_count,'set_pose_calls':set_pose_calls,'mj_forward_calls_in_set_pose':2*set_pose_calls,
            'pre_solver_tip_callback_count':pre_solver_tip_count,'pre_solver_analytic_jacobian_callback_count':pre_solver_analytic_count,
            'final_state_freshly_recomputed_outside_solver_cache':True,'source_poor_relationships_not_replaced_by_quality_reward':True,
            'time_s':float(d.time),'object_state_and_world_pose_unchanged':True,'runtime_seconds':time.perf_counter()-tick,
            'new_eval_data':False,'new_generator_proposal':False,'physics_steps':0,'training':False,'mapping_acceptance':False}
        for path,h in meta['input_sha256'].items():assert a.sha(path)==h,path
        assert np.array_equal(d.qpos[objadr:objadr+7],initial) and d.time==0
        records.append(record)
        write(partial_path,{'schema':'hug_free_palm_geometry_candidate_cohort_v1','partial':True,
            'protocol_sha256':ph,'cohort_case_indices':indices,'cases':records})
        print(json.dumps({'completed_case':caseid,'selected_candidate':audit['surface_candidate_returned'],
            'selected_iteration':audit.get('selected_major_iteration'),'new_method_termination':audit['termination'],
            'full_surface_evaluations':audit['counters']['full'],'objective_baseline':baseline_fresh['objective'],
            'objective_selected':final_fresh['objective'],'mean_tip_baseline_mm':baseline_fresh['mean_fingertip_error_m']*1000,
            'mean_tip_selected_mm':final_fresh['mean_fingertip_error_m']*1000,'seconds':record['runtime_seconds']}),flush=True)
    for path,h in p['source_sha256'].items():assert a.sha(path)==h,path
    for path,h in p['implementation_sha256'].items():assert a.sha(path)==h,path
    for path,h in protected.items():assert a.sha(path)==h,path
    assert a.sha(pp)==ph and COUNTS['actual']==0
    assert all(getattr(mujoco,api) is forbidden for api in ('mj_step','mj_step1','mj_step2'))
    report={'schema':'hug_free_palm_geometry_candidate_cohort_v1','protocol_sha256':ph,
        'script_sha256':a.sha(__file__),'solver_implementation_sha256':a.sha(OUT/'free_palm_solver.py'),
        'direction_implementation_sha256':a.sha(p['direction_helper_path']),'cohort_case_indices':indices,'cases':records,
        'method_version':2,'method_stage':'free_palm_fine_translation_seed_v2',
        'numba_pool_capacity':numba.config.NUMBA_NUM_THREADS,'numba_threads_per_process':numba.get_num_threads(),
        'runtime_environment':RUNTIME_ENVIRONMENT,
        'runtime_seconds':time.perf_counter()-started,'protected_source_count':len(protected),'protected_hashes_unchanged':True,
        'original_inputs_and_implementations_unchanged':True,'guards':COUNTS,'new_eval_data':False,
        'new_generator_proposal':False,'physics_steps':0,'training':False,'production_changes':False,
        'strict_acceptance_rule_unchanged_sha256':p['strict_acceptance_rule_sha256'],
        'limitations':p['limitations'],'scope':'Versioned static per-grasp translated-palm and joint search, exact original caps and fixed direction guard, six saved development grasps. Independent unchanged scientific audit required.'}
    write(report_path,report)
    print(json.dumps({'completed_cohort':indices,'report':str(report_path),'seconds':report['runtime_seconds']}),flush=True)

if __name__=='__main__':main()
