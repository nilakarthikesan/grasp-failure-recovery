"""Static, predeclared reference comparison of already saved native HUG poses.
No inference, time integration, controller execution or outcome labels.
"""
from pathlib import Path
import contextlib,hashlib,io,itertools,json,sys,time
from collections import defaultdict
import importlib.metadata
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
GUARD={'self_test_calls':0,'actual_attempts':0,'testing':False}
def forbid(*args,**kwargs):
    GUARD['self_test_calls' if GUARD['testing'] else 'actual_attempts']+=1
    raise RuntimeError('Static HUG reference comparison forbids time integration')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbid)
GUARD['testing']=True
for name in ('mj_step','mj_step1','mj_step2'):
    try:getattr(mujoco,name)()
    except RuntimeError:pass
GUARD['testing']=False
from dex_retargeting.constants import HandType,RetargetingType,RobotName,get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter,RetargetedHandPose,camera_landmarks_to_wrist,reference_vectors,reorder_for_mujoco,default_dex_urdf_root
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator,_link_positions,_body_id,TASK_LINK_NAMES
from grasp_failure_prediction.integrations.hug import load_hug_prediction
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
IND=np.array([4,8,12,16,20,2,6,10,14,18])
FINGERS=['thumb','index','middle','ring','little']
PAIRS=list(itertools.combinations(range(5),2))
START=time.perf_counter()
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def ident(m,kind,name):
    v=int(mujoco.mj_name2id(m,kind,name));assert v>=0,(kind,name);return v

def files_in(x):
    result={}
    if isinstance(x,dict):
        if 'path' in x and 'sha256' in x and Path(x['path']).is_file():result[x['path']]=x['sha256']
        for value in x.values():result.update(files_in(value))
    elif isinstance(x,list):
        for value in x:result.update(files_in(value))
    return result

class TraceOpt:
    """Capture NLopt diagnostics without replacing the optimization method."""
    def __init__(self,opt):self.wrapped=opt;self.calls=[]
    def __getattr__(self,key):return getattr(self.wrapped,key)
    def optimize(self,initial):
        tic=time.perf_counter();entry={'initial_qpos_rad':np.asarray(initial).tolist(),'exception':None}
        try:
            value=self.wrapped.optimize(initial);entry['returned_qpos_rad']=np.asarray(value).tolist();return value
        except Exception as exc:
            entry['exception']={'type':type(exc).__name__,'message':str(exc)};raise
        finally:
            for key,fn in [('result_code','last_optimize_result'),('objective_value','last_optimum_value'),('evaluations','get_numevals')]:
                try:v=getattr(self.wrapped,fn)();entry[key]=v if not isinstance(v,float) or np.isfinite(v) else None
                except Exception:entry[key]=None
            entry['seconds']=time.perf_counter()-tic;self.calls.append(entry)

def retarget_current(grasp):
    wrapper=ShadowHandRetargeter();seq=wrapper._retargeting;seq.optimizer.opt=TraceOpt(seq.optimizer.opt)
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):pose=wrapper.retarget(grasp)
    raw=seq.last_qpos.astype(float).copy()
    return tuple(pose.joint_names),pose.qpos.copy(),raw,raw.copy(),{'method':'production ShadowHandRetargeter','config_path':str(wrapper.config_path),'config_sha256':SHA(wrapper.config_path),'scale':seq.optimizer.scaling,'filter':None,'optimizer_call_count':len(seq.optimizer.opt.calls),'solver_calls':seq.optimizer.opt.calls,'solver_stdout':stream.getvalue(),'projected_intertip_pairs':None}

def retarget_dexpilot(grasp,metric=False):
    path=get_default_config_path(RobotName.shadow,RetargetingType.dexpilot,HandType.right)
    RetargetingConfig.set_default_urdf_dir(default_dex_urdf_root().resolve())
    seq=RetargetingConfig.load_from_file(path).build()
    if metric:seq.filter=None;seq.optimizer.scaling=1.0
    seq.optimizer.opt=TraceOpt(seq.optimizer.opt)
    points=camera_landmarks_to_wrist(grasp.landmarks_3d,grasp.T_camera_wrist)@MANO_TO_OPERATOR_RIGHT.T
    vectors=reference_vectors(points,seq.optimizer.target_link_human_indices)
    histories=[];stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        for iteration in range(20):
            filtered=np.asarray(seq.retarget(vectors.astype(np.float32)),float)
            histories.append({'iteration':iteration+1,'raw_qpos_rad':seq.last_qpos.astype(float).tolist(),'filtered_qpos_rad':filtered.tolist(),'projected_intertip_pairs':seq.optimizer.projected.tolist()})
    raw=seq.last_qpos.astype(float).copy()
    q=np.clip(filtered,seq.joint_limits[:,0],seq.joint_limits[:,1])
    info={'method':'Dex Retargeting 0.5.0 DexPilot-style reference'+(' metric-static sensitivity' if metric else ' as-shipped configuration'),
          'config_path':str(path),'config_sha256':SHA(path),'scale':seq.optimizer.scaling,'low_pass_alpha':None if metric else 0.2,
          'optimizer_call_count':len(seq.optimizer.opt.calls),'solver_calls':seq.optimizer.opt.calls,'solver_stdout':stream.getvalue(),
          'target_origin_link_names':seq.optimizer.origin_link_names,'target_task_link_names':seq.optimizer.task_link_names,'target_link_human_indices':seq.optimizer.target_link_human_indices.tolist(),
          'project_dist_m':seq.optimizer.project_dist,'escape_dist_m':seq.optimizer.escape_dist,'eta1':seq.optimizer.eta1,'eta2':seq.optimizer.eta2,
          'iteration_history':histories,'projected_intertip_pairs':seq.optimizer.projected.tolist(),
          'static_iterations20_final_raw_vs_filtered_max_rad':float(np.max(np.abs(raw-filtered)))}
    return tuple(seq.joint_names),q,raw,filtered,info

def main():
    proto_path=OUT/'protocol.json';ph=SHA(proto_path);p=json.loads(proto_path.read_text())
    for f,h in p['code_and_config_sha256'].items():assert SHA(f)==h,f
    ip=Path(p['source_inventory_path']);fp=Path(p['prior_fk_report_path']);assert SHA(ip)==p['source_inventory_sha256'];assert SHA(fp)==p['prior_fk_report_sha256']
    index=json.loads(ip.read_text());prior=json.loads(fp.read_text());pm={c['native_sha256']:c for c in prior['cases']}
    inputs=files_in(index['modern_cases']);inputs[str(ip)]=SHA(ip);inputs[str(fp)]=SHA(fp)
    for f,h in inputs.items():assert SHA(f)==h,f
    validator=ShadowPoseValidator(default_dex_urdf_root());canonical=validator.joint_names;assert len(canonical)==24
    urdf=validator.source_urdf;assert SHA(urdf)==p['production_urdf_sha256']
    # Independent Pinocchio evaluation uses released wrapper but no optimization.
    RetargetingConfig.set_default_urdf_dir(default_dex_urdf_root().resolve())
    refcfg=RetargetingConfig.load_from_file(p['released_config_path']);robot=refcfg.build().optimizer.robot
    rnames=robot.dof_joint_names;rlinks=[robot.get_link_index(s) for s in TASK_LINK_NAMES];rpalm=robot.get_link_index('palm')
    records=[]
    for plan in p['cases']:
        case=index['modern_cases'][plan['source_inventory_index']];assert case['native']['sha256']==plan['native_sha256']
        grasp=load_hug_prediction(case['native']['path']);T_wc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);T_wm=T_wc@grasp.T_camera_wrist
        Rop=T_wm[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
        native_world=grasp.landmarks_3d@T_wc[:3,:3].T+T_wc[:3,3]
        native_operator=camera_landmarks_to_wrist(grasp.landmarks_3d,grasp.T_camera_wrist)@MANO_TO_OPERATOR_RIGHT.T
        model=mujoco.MjModel.from_binary_path(case['scene_mjb']['path']);data=mujoco.MjData(model)
        with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as state:initial=state['qpos'].copy()
        rootj=ident(model,mujoco.mjtObj.mjOBJ_JOINT,'world_joint');rootadr=int(model.jnt_qposadr[rootj]);objj=ident(model,mujoco.mjtObj.mjOBJ_JOINT,'object_free');objadr=int(model.jnt_qposadr[objj]);palm=ident(model,mujoco.mjtObj.mjOBJ_BODY,'palm');obj=ident(model,mujoco.mjtObj.mjOBJ_BODY,'object')
        command=case['stored_shadow_command']
        with np.load(command['path'],allow_pickle=False) as a:
            row=command['row_index'];saved_names=tuple(a['joint_names'].tolist());saved_q=a['nominal_target_rad'][row].copy();saved_root_pos=a['root_target_position_m'][row].copy();saved_root_quat=a['root_target_quaternion_wxyz'][row].copy()
        lower=np.array([model.jnt_range[ident(model,mujoco.mjtObj.mjOBJ_JOINT,n)][0] for n in canonical]);upper=np.array([model.jnt_range[ident(model,mujoco.mjtObj.mjOBJ_JOINT,n)][1] for n in canonical])
        per={'stored_current':(saved_names,saved_q,saved_q.copy(),saved_q.copy(),{'method':'previously recorded original nominal target','optimizer_call_count':0,'stored_shadow_command':command})}
        try:per['current_recomputed']=retarget_current(grasp)
        except Exception as exc:per['current_recomputed']={'status':'rejected','exception':{'type':type(exc).__name__,'message':str(exc)}}
        for label,metric in [('dexpilot_shipped',False),('dexpilot_metric_static',True)]:
            try:per[label]=retarget_dexpilot(grasp,metric)
            except Exception as exc:per[label]={'status':'rejected','exception':{'type':type(exc).__name__,'message':str(exc)}}
        target_vectors=native_operator[IND]-native_operator[0];methods={}
        for label,payload in per.items():
            if isinstance(payload,dict):methods[label]=payload;continue
            names,q,raw,filtered,info=payload
            q=reorder_for_mujoco(names,q,canonical);raw=reorder_for_mujoco(names,raw,canonical);filtered=reorder_for_mujoco(names,filtered,canonical)
            violation=np.maximum(np.maximum(lower-filtered,filtered-upper),0);clip=filtered-q
            record={'status':'computed','joint_names':list(canonical),'qpos_rad':q.tolist(),'qpos_raw20_rad':raw.tolist(),'qpos_filtered20_rad':filtered.tolist(),'solver':info,'limits':{'physical_lower_rad':lower.tolist(),'physical_upper_rad':upper.tolist(),'filter_output_physical_violation_rad':violation.tolist(),'max_physical_violation_rad':float(violation.max()),'physical_clamp_delta_rad':clip.tolist(),'max_physical_clamp_delta_rad':float(np.max(np.abs(clip))),'clamped_joint_names':[n for n,c in zip(canonical,clip) if c!=0],'safety_margin_rad':1.01e-3}}
            if not(np.isfinite(q).all() and np.isfinite(filtered).all() and np.isfinite(raw).all()) or violation.max()>1.01e-3:
                record['status']='rejected';record['rejection_reason']='invalid/nonfinite24-vector or released numerical-limit margin exceeded';methods[label]=record;continue
            pose=RetargetedHandPose(tuple(canonical),q,native_operator,target_vectors,grasp.T_camera_wrist)
            validator.set_pose(pose);assert validator.data.time==0
            fixed_tasks=_link_positions(validator.model,validator.data,urdf);fixed_palm=validator.data.xpos[_body_id(validator.model,'palm')].copy();fixed_palm_R=validator.data.xmat[_body_id(validator.model,'palm')].reshape(3,3).copy()
            fixed_vectors=fixed_tasks-fixed_palm
            # Actual _hug_world_palm_pose and _root_pose_for_palm policies composed explicitly.
            desired_palm_R=Rop@fixed_palm_R
            data.qpos[:]=initial;data.qvel[:]=0;data.qpos[rootadr:rootadr+3]=0;data.qpos[rootadr+3:rootadr+7]=[1,0,0,0]
            for n,val in zip(canonical,q):data.qpos[int(model.jnt_qposadr[ident(model,mujoco.mjtObj.mjOBJ_JOINT,n)])]=val
            mujoco.mj_forward(model,data)
            local_palm_R=data.xmat[palm].reshape(3,3).copy();local_palm=data.xpos[palm].copy()
            rootR=desired_palm_R@local_palm_R.T;rootpos=T_wm[:3,3]-rootR@local_palm;rootquat=np.empty(4);mujoco.mju_mat2Quat(rootquat,rootR.reshape(9))
            data.qpos[rootadr:rootadr+3]=rootpos;data.qpos[rootadr+3:rootadr+7]=rootquat
            if model.nmocap:data.mocap_pos[0]=rootpos;data.mocap_quat[0]=rootquat
            mujoco.mj_forward(model,data);assert data.time==0;assert np.array_equal(data.qpos[objadr:objadr+7],initial[objadr:objadr+7])
            world=_link_positions(model,data,urdf);expected_world=fixed_vectors@Rop.T+T_wm[:3,3]
            objpos=data.xpos[obj].copy();objR=data.xmat[obj].reshape(3,3).copy();shadow_obj=(world-objpos)@objR;native_obj=(native_world-objpos)@objR
            robotq=reorder_for_mujoco(canonical,q,rnames);robot.compute_forward_kinematics(robotq);pin_tasks=np.array([robot.get_link_pose(i)[:3,3] for i in rlinks]);pin_palm=robot.get_link_pose(rpalm)[:3,3]
            pin_vs_mj=np.linalg.norm(pin_tasks-fixed_tasks,axis=1)
            residual=world-native_world[IND];errs=np.linalg.norm(residual,axis=1);objerrs=np.linalg.norm(shadow_obj-native_obj[IND],axis=1);operrs=np.linalg.norm(fixed_vectors-target_vectors,axis=1)
            pairs=[]
            for aa,bb in PAIRS:
                nv=native_world[IND[bb]]-native_world[IND[aa]];sv=world[bb]-world[aa];nl=np.linalg.norm(nv);sl=np.linalg.norm(sv)
                angle=float(np.degrees(np.arccos(np.clip(np.dot(nv,sv)/(nl*sl),-1,1)))) if nl*sl>1e-14 else None
                pairs.append({'fingers':[FINGERS[aa],FINGERS[bb]],'native_vector_world_m':nv.tolist(),'shadow_vector_world_m':sv.tolist(),'vector_error_m':float(np.linalg.norm(sv-nv)),'native_distance_m':float(nl),'shadow_distance_m':float(sl),'length_change_m':float(sl-nl),'angle_deg':angle})
            record.update({'root_target_position_m':rootpos.tolist(),'root_target_quaternion_wxyz':rootquat.tolist(),'target_palm_world_position_m':T_wm[:3,3].tolist(),'target_palm_world_rotation':desired_palm_R.tolist(),'links':[{'link':name,'finger':FINGERS[i%5],'type':'tip' if i<5 else 'middle','native_landmark_index':int(IND[i]),'native_world_m':native_world[IND[i]].tolist(),'shadow_world_m':world[i].tolist(),'native_initial_object_m':native_obj[IND[i]].tolist(),'shadow_initial_object_m':shadow_obj[i].tolist(),'residual_world_m':residual[i].tolist(),'position_error_m':float(errs[i])} for i,name in enumerate(TASK_LINK_NAMES)],
              'tip_mean_error_m':float(errs[:5].mean()),'tip_max_error_m':float(errs[:5].max()),'all10_mean_error_m':float(errs.mean()),'intertip_pairs':pairs,
              'static_invariants':{'time_s':float(data.time),'initial_object_qpos_unchanged':True,'palm_vs_target_origin_error_m':float(np.linalg.norm(data.xpos[palm]-T_wm[:3,3])),'palm_vs_composed_rotation_maxabs':float(np.max(np.abs(data.xmat[palm].reshape(3,3)-desired_palm_R))),'fixed_urdf_vs_captured_scene_max_link_error_m':float(np.linalg.norm(world-expected_world,axis=1).max()),'Pinocchio_vs_fixedURDF_MuJoCo_max_link_error_m':float(pin_vs_mj.max()),'Pinocchio_vs_fixedURDF_MuJoCo_palm_error_m':float(np.linalg.norm(pin_palm-fixed_palm)),'world_vs_object_error_maxdelta_m':float(np.max(np.abs(errs-objerrs))),'world_vs_operator_error_maxdelta_m':float(np.max(np.abs(errs-operrs)))}})
            if label=='stored_current':
                priorworld=np.array([k['shadow_world_m'] for k in pm[case['native']['sha256']]['links']])
                record['stored_baseline_reproduction']={'root_position_maxabs_m':float(np.max(np.abs(rootpos-saved_root_pos))),'root_rotation_maxabs':float(np.max(np.abs(rootR-Rotation.from_quat(saved_root_quat[[1,2,3,0]]).as_matrix()))),'prior_static_FK_link_max_error_m':float(np.linalg.norm(world-priorworld,axis=1).max())}
                assert record['stored_baseline_reproduction']['prior_static_FK_link_max_error_m']<1e-7
            methods[label]=record
        current=methods['stored_current'];cq=np.array(current['qpos_rad']);cw=np.array([x['shadow_world_m'] for x in current['links']])
        for name,method in methods.items():
            if method['status']!='computed':continue
            qw=np.array(method['qpos_rad']);sw=np.array([x['shadow_world_m'] for x in method['links']]);delta=np.linalg.norm(sw-cw,axis=1)
            method['vs_stored_current']={'joint_delta_rad':(qw-cq).tolist(),'max_abs_joint_delta_rad':float(np.max(np.abs(qw-cq))),'target_link_displacements_m':delta.tolist(),'mean_tip_displacement_m':float(delta[:5].mean()),'maximum_tip_displacement_m':float(delta[:5].max())}
        record={'native_sha256':case['native']['sha256'],'native_path':case['native']['path'],'object_id':case['object_id'],'seed':case['native']['seed'],'observation_group_id':case['observation_group_id'],'scene_model_path':case['scene_mjb']['path'],'scene_model_sha256':case['scene_mjb']['sha256'],'geometry_content_hash':case['geometry_content_hash'],'T_world_operator_rotation':Rop.tolist(),'methods':methods}
        records.append(record);write(OUT/f'case_{plan["source_inventory_index"]:02d}.json',record)
        print(json.dumps({'completed_source':plan['source_inventory_index'],'object_id':case['object_id'],'methods':{k:{'status':x['status'],'tip_mean_error_mm':x.get('tip_mean_error_m',0)*1000} for k,x in methods.items()}}),flush=True)
    summary={}
    for label in p['methods']:
        val=[c for c in records if c['methods'][label]['status']=='computed'];rr=[c['methods'][label] for c in val];tips=np.array([[x['position_error_m'] for x in r['links'][:5]] for r in rr]);all10=np.array([[x['position_error_m'] for x in r['links']] for r in rr]);means=tips.mean(1)
        byobj={};byfinger={}
        for objname in sorted({c['object_id'] for c in records}):
            selected=[c['methods'][label] for c in val if c['object_id']==objname];tip=np.array([[x['position_error_m'] for x in r['links'][:5]] for r in selected]);byobj[objname]={'case_count':len(selected),'mean_tip_error_m':float(tip.mean()),'maximum_tip_error_m':float(tip.max()),'percase_mean_tip_range_m':[float(tip.mean(1).min()),float(tip.mean(1).max())]}
        for i,finger in enumerate(FINGERS):byfinger[finger]={'tip_mean_error_m':float(tips[:,i].mean()),'tip_max_error_m':float(tips[:,i].max()),'middle_mean_error_m':float(all10[:,i+5].mean())}
        calls=[x for r in rr for x in r['solver'].get('solver_calls',[])];failed=[x for x in calls if x.get('exception') is not None or (x.get('result_code') is not None and x['result_code']<0)]
        summary[label]={'computed_case_count':len(rr),'rejected_case_count':len(records)-len(rr),'mean_tip_error_m':float(tips.mean()),'maximum_tip_error_m':float(tips.max()),'percase_mean_tip_range_m':[float(means.min()),float(means.max())],'mean_all10_error_m':float(all10.mean()),'mean_intertip_vector_error_m':float(np.mean([x['vector_error_m'] for r in rr for x in r['intertip_pairs']])),'mean_abs_intertip_distance_change_m':float(np.mean([abs(x['length_change_m']) for r in rr for x in r['intertip_pairs']])),'mean_tip_displacement_from_stored_m':float(np.mean([r['vs_stored_current']['mean_tip_displacement_m'] for r in rr])),'max_physical_clamp_delta_rad':max(r['limits']['max_physical_clamp_delta_rad'] for r in rr),'clamped_case_count':sum(bool(r['limits']['clamped_joint_names']) for r in rr),'max_pin_vs_mujoco_link_error_m':max(r['static_invariants']['Pinocchio_vs_fixedURDF_MuJoCo_max_link_error_m'] for r in rr),'max_fixed_vs_scene_link_error_m':max(r['static_invariants']['fixed_urdf_vs_captured_scene_max_link_error_m'] for r in rr),'all_time_zero':all(r['static_invariants']['time_s']==0 for r in rr),'all_object_states_unchanged':all(r['static_invariants']['initial_object_qpos_unchanged'] for r in rr),'solver_call_count':len(calls),'solver_failed_call_count':len(failed),'solver_return_code_counts':{str(code):sum(x.get('result_code')==code for x in calls) for code in sorted({x.get('result_code') for x in calls if x.get('result_code') is not None})},'by_object':byobj,'by_finger':byfinger}
        if label.startswith('dexpilot'):
            summary[label]['projected_pair_case_count']=sum(any(r['solver']['projected_intertip_pairs']) for r in rr);summary[label]['final_projected_pair_count_range']=[min(sum(r['solver']['projected_intertip_pairs']) for r in rr),max(sum(r['solver']['projected_intertip_pairs']) for r in rr)]
    for f,h in inputs.items():assert SHA(f)==h,f
    for f,h in p['code_and_config_sha256'].items():assert SHA(f)==h,f
    assert SHA(proto_path)==ph and GUARD['actual_attempts']==0
    report={'schema':p['schema'],'protocol_sha256':ph,'script_sha256':SHA(__file__),'runtime_seconds':time.perf_counter()-START,'runtime':{'python':sys.version,'numpy':np.__version__,'mujoco':mujoco.__version__,'dex_retargeting':importlib.metadata.version('dex-retargeting'),'pinocchio':importlib.metadata.version('pin')},'scope':p['scope'],'urdf_path':str(urdf),'urdf_sha256':SHA(urdf),'canonical_joint_names':list(canonical),'guards':GUARD,'input_sha256_before_and_after':inputs,'code_and_config_sha256_before_and_after':p['code_and_config_sha256'],'input_inventory_sha256':SHA(ip),'prior_fk_sha256':SHA(fp),'summary':summary,'cases':records,'limitations':p['limitations'],'new_inference':False,'physics_steps':0,'new_eval_data':False,'outcome_labels':False,'production_changes':False,'predictor_training':False}
    write(OUT/'hug_reference_comparison.json',report);write(OUT/'summary.json',summary)
    print(json.dumps({'completed':True,'seconds':report['runtime_seconds'],'report':str(OUT/'hug_reference_comparison.json'),'summary':summary},indent=2),flush=True)
if __name__=='__main__':main()
