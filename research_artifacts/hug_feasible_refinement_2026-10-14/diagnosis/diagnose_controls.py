"""Read-only old-fit diagnosis; time-zero FK, no surface fits or integration."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['NUMBA_NUM_THREADS']='4'
from pathlib import Path
import json,hashlib,pickle,sys,time,warnings
import numpy as np
import mujoco
from scipy.optimize import linprog,minimize
from scipy.optimize._numdiff import approx_derivative
OLD=Path('/private/tmp/hug_constrained_refinement_runtime_v2_1_2026-10-14')
OUT=Path(__file__).parent/'controls'
OUT.mkdir(exist_ok=True)
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sys.path.insert(0,str(OLD))
import run_constrained_refinement as driver
a=driver.a
from grasp_failure_prediction.evaluation.pose_validation import _link_positions,_fixed_tip_offsets,TASK_LINK_NAMES,ShadowPoseValidator
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root,reorder_for_mujoco
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def js(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:js(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [js(v) for v in x]
    return x
def rank(j):return int(np.linalg.matrix_rank(j,tol=1e-9))

def main():
    started=time.perf_counter();p=read(OLD/'protocol.json')
    bound={path:h for path,h in p['source_sha256'].items()}
    bound.update(p['implementation_sha256'])
    bound[str(ROOT/'src/grasp_failure_prediction/evaluation/pose_validation.py')]=sha(ROOT/'src/grasp_failure_prediction/evaluation/pose_validation.py')
    bound[str(ROOT/'research_artifacts/task_outcome_baseline_v1/proprioception_and_commands.npz')]=sha(ROOT/'research_artifacts/task_outcome_baseline_v1/proprioception_and_commands.npz')
    for path,h in bound.items():assert sha(path)==h,path
    for idx in (8,12,16,20,24,28):bound[str(OLD/f'candidate_cohort_{idx}.json')]=sha(OLD/f'candidate_cohort_{idx}.json')
    inv=read(ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json')
    fkmap={c['native_sha256']:c for c in read(ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/static_saved_hug_fk.json')['cases']}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
    weights=np.asarray(asset['weights']);grouped=np.column_stack([weights[:,ids].sum(1) for ids in a.GROUPS.values()])
    validator=ShadowPoseValidator(default_dex_urdf_root())
    driver.COUNTS['testing']=True
    for api in ('mj_step','mj_step1','mj_step2'):
        try:getattr(mujoco,api)(None,None)
        except RuntimeError:pass
        else:raise AssertionError('guard selftest failed')
    driver.COUNTS['testing']=False
    result={'schema':'hug_old_constraints_readonly_diagnosis_v1','scope':'Old traces and time-zero FK only. No new full surface evaluations, solver fits, labels, inference or grasp-eval data.','old_protocol_sha256':sha(OLD/'protocol.json'),'cases':[]}
    for idx in (8,12,16,20,24,28):
        t=time.perf_counter();saved=read(OLD/f'candidate_cohort_{idx}.json')['cases'][0];case=inv['modern_cases'][idx];audit=saved['solver_audit']
        meta,ntri,nlabels,nnorm,original_hand,objects,m,d,initial,objadr,opos,oR=a.load_case(case,grouped,fkmap)
        bound.update(meta['input_sha256'])
        names=saved['joint_names'];q0=np.array(saved['original_qpos_rad']);raw=np.array(saved['raw_candidate_qpos_rad']);limits=np.array(audit['bounded_correction_domain_rad']);low,high=limits.T
        nominal=d.qpos.copy();ids=[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];adr=m.jnt_qposadr[ids];dofs=m.jnt_dofadr[ids]
        palm=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');ob=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'object');rootadr=int(m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')])
        rootpos=np.asarray(meta['root_target_world_position_m']);rootquat=np.asarray(meta['root_target_world_quaternion_wxyz'])
        source=pickle.loads(Path(case['native']['path']).read_bytes());T_wc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
        T_wrist=T_wc@np.asarray(source['T_camera_wrist'],float);Rop=T_wrist[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
        native_world=np.asarray(source['landmarks_3d'])@T_wc[:3,:3].T+T_wc[:3,3];native_tasks=native_world[np.array([4,8,12,16,20,2,6,10,14,18])]
        calls=0;maxpalmer=0.
        def pose(q):
            nonlocal calls,maxpalmer
            calls+=1;q=np.asarray(q)
            d.qpos[:]=nominal;d.qpos[adr]=q;d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat
            mujoco.mj_forward(m,d)
            corrected=rootpos.copy() if np.array_equal(q,q0) else rootpos-(d.xpos[palm]-T_wrist[:3,3])
            d.qpos[rootadr:rootadr+3]=corrected
            if m.nmocap:d.mocap_pos[0]=corrected;d.mocap_quat[0]=rootquat
            mujoco.mj_forward(m,d)
            er=np.linalg.norm(d.xpos[palm]-T_wrist[:3,3]);maxpalmer=max(maxpalmer,float(er))
            assert er<1e-12 and d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
            assert np.array_equal(d.xpos[ob],opos) and np.array_equal(d.xmat[ob].reshape(3,3),oR)
            return _link_positions(m,d,a.URDF)[:5]-native_tasks[:5]
        orig=pose(q0);sq=np.sum(orig**2,axis=1);orig_norm=np.sqrt(sq)
        def cons(q):
            e=pose(q);n=np.linalg.norm(e,axis=1)
            return np.r_[(sq-n**2)/.01**2,(.01-n.mean())/.01]
        offsets=_fixed_tip_offsets(a.URDF)
        def jac(q):
            e=pose(q);anchors=e+native_tasks[:5];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacBody(m,d,jp,jr,palm);pj=jp[:,dofs].copy()
            J=np.zeros((5,3,24))
            for i,n in enumerate(TASK_LINK_NAMES[:5]):
                bid=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,offsets[n][0]);jp.fill(0);jr.fill(0)
                mujoco.mj_jac(m,d,jp,jr,anchors[i],bid);J[i]=jp[:,dofs]-pj
            norms=np.linalg.norm(e,axis=1)
            cj=np.vstack([-2*np.einsum('ij,ijk->ik',e,J)/.01**2,-np.einsum('ij,ijk->k',e/norms[:,None],J)/(5*.01)])
            return J,cj
        def fd(q,h,central):
            q=np.asarray(q);J=np.zeros((6,24));base=cons(q)
            for k in range(24):
                delta=np.zeros(24);delta[k]=h
                if central:J[:,k]=(cons(q+delta)-cons(q-delta))/(2*h)
                else:J[:,k]=(cons(q+delta)-base)/h
            return J
        derivatives={}
        for label,q in [('original',q0),('raw_final',raw)]:
            J,C=jac(q);forward=fd(q,1e-5,False);c6=fd(q,1e-6,True);c7=fd(q,1e-7,True)
            # Full row rank is evidence about local derivatives, not global feasibility.
            sv=np.linalg.svd(C[:5],compute_uv=False)
            derivatives[label]={'tip_coordinate_jacobian_rank':rank(J.reshape(15,24)),'individual_cap_jacobian_rank':rank(C[:5]),'cap_singular_values':sv,
                'forward_1e5_maxabs_difference_from_analytic':np.max(np.abs(forward-C)),
                'central_1e6_maxabs_difference_from_analytic':np.max(np.abs(c6-C)),
                'central_1e7_maxabs_difference_from_analytic':np.max(np.abs(c7-C)),
                'actual_bounded_forward_1e5_maxabs_difference_from_analytic':np.max(np.abs(approx_derivative(cons,q,method='2-point',abs_step=1e-5,bounds=(low,high))-C)),
                'jacobian':C,'tip_coordinate_jacobian':J.reshape(15,24),
                'normalized_constraints':cons(q),
                'physical_or_trust_active_lower_joints':[names[k] for k in np.flatnonzero(q-low<1e-8)],
                'physical_or_trust_active_upper_joints':[names[k] for k in np.flatnonzero(high-q<1e-8)]}
        # Maximize an inward margin simultaneously over all five tip caps, respecting first-order active bounds.
        J,C=jac(q0);direction_bounds=[]
        for lo,hi,qi in zip(low,high,q0,strict=True):direction_bounds.append((0. if qi-lo<1e-8 else -1.,0. if hi-qi<1e-8 else 1.))
        lp=linprog(np.r_[np.zeros(24),-1.],A_ub=np.c_[-C[:5],np.ones(5)],b_ub=np.zeros(5),bounds=direction_bounds+[(None,None)],method='highs')
        inward={'success':lp.success,'message':lp.message}
        if lp.success:
            v=lp.x[:24];v/=max(1.,np.linalg.norm(v));inward.update({'unit_direction':v,'normalized_derivative_min':np.min(C[:5]@v),'unscaled_optimal_margin':lp.x[-1],
                'finite_steps':[{'step_rad':h,'constraints':cons(q0+h*v),'all_exact_caps_nonnegative':bool(np.all(cons(q0+h*v)[:5]>=0))} for h in (1e-5,1e-4,1e-3,1e-2)]})
        # Counterfactual cheap quadratic objective with literal old FD/ftol/eps options.
        control=minimize(lambda q:.5*np.sum((q-raw)**2),q0.copy(),method='SLSQP',bounds=limits,
            constraints={'type':'ineq','fun':cons},options={'maxiter':120,'ftol':1e-9,'eps':1e-5,'disp':False})
        cn=np.linalg.norm(pose(control.x),axis=1)
        old_options_control={'success':control.success,'status':control.status,'message':control.message,'iterations':control.nit,'function_evaluations':control.nfev,
            'qpos_rad':control.x,'max_tip_increase_m':np.max(cn-orig_norm),'individual_tip_increase_m':cn-orig_norm,
            'normalized_constraints':cons(control.x),'allowance_feasible':bool(np.all(cn<=orig_norm+1e-8) and cn.mean()<=.01),
            'distance_original_to_projected_l2_rad':np.linalg.norm(control.x-q0),
            'scope':'Only replace costly surface objective by convex q-to-oldraw squared distance. Constraint callbacks and old SLSQP options unchanged. Not a surface fit or scientific acceptance.'}
        # Cheap restoration only; this does not evaluate or select any surface correction.
        def distance(q):return .5*np.sum((q-raw)**2)
        def dgrad(q):return q-raw
        histories=[]
        def cb(q):histories.append({'min_cap':float(np.min(cons(q)[:5])),'q':np.asarray(q).tolist()})
        pro=minimize(distance,q0.copy(),jac=dgrad,method='SLSQP',bounds=limits,
            constraints={'type':'ineq','fun':cons,'jac':lambda q:jac(q)[1]},callback=cb,
            options={'maxiter':100,'ftol':1e-12,'disp':False})
        pc=cons(pro.x);per=np.linalg.norm(pose(pro.x),axis=1)
        projection={'scope':'FK-only restoration; no surface score or acceptance claim','success':pro.success,'status':pro.status,'message':pro.message,'iterations':pro.nit,
            'qpos_rad':pro.x,'distance_to_raw_l2_rad':np.linalg.norm(pro.x-raw),'distance_original_to_raw_l2_rad':np.linalg.norm(q0-raw),'distance_to_original_l2_rad':np.linalg.norm(pro.x-q0),
            'tip_errors_m':per,'tip_increase_m':per-orig_norm,'constraints':pc,'exact_feasible':bool(np.all(pc>=0)),'allowance_feasible':bool(np.all(per<=orig_norm+1e-8) and per.mean()<=.01),'history':histories}
        trace=audit['major_iteration_assessments'];ass=[r for r in trace if r.get('assessment')]
        increases=np.array([r['assessment']['tip_error_increase_m'] for r in ass]);gaps=np.array([r['assessment']['objective']-audit['baseline']['objective'] for r in ass]);viol=np.max(increases,axis=1)
        imin=int(np.argmin(viol));near=ass[imin]
        trace_summary={'recorded_callbacks':len(trace),'valid_assessments':len(ass),'minimum_max_tip_increase_m':viol[imin],'callback_at_minimum_violation':near['major_iteration'],
            'objective_at_minimum_violation':near['assessment']['objective'],'objective_improved_at_minimum_violation':bool(gaps[imin]<-1e-12),'minimum_per_tip_increase_m':np.min(increases,axis=0),
            'all5_within_10nm_count':int(np.sum(viol<=1e-8)),'any_cost_improving_callbacks':int(np.sum(gaps<0)),
            'largest_max_tip_increase_m':np.max(viol),'each_finger_violation_count':np.sum(increases>1e-8,axis=0),
            'minimum_improving_max_tip_increase_m':float(np.min(viol[gaps<0])) if np.any(gaps<0) else None}
        # Independently restage all recorded callbacks via time-zero FK, compare vector/norm/constraint metadata.
        maxerr=0.;maxcon=0.
        for r in ass:
            er=pose(np.array(r['qpos_rad']));maxerr=max(maxerr,float(np.max(np.abs(er-np.array(r['assessment']['tip_error_vectors_m'])))))
            maxcon=max(maxcon,float(np.max(np.abs(cons(np.array(r['qpos_rad']))-np.array(r['assessment']['normalized_constraint_values'])))))
        # Independent fixed-URDF kinematic orientation chain at original/raw/projected.
        conventions=[]
        for label,q in [('original',q0),('raw_final',raw),('FK_projection',pro.x)]:
            er=pose(q);qfix=np.asarray(reorder_for_mujoco(names,q,validator.joint_names));validator.data.qpos[:]=qfix
            mujoco.mj_forward(validator.model,validator.data);vp=a.ident(validator.model,mujoco.mjtObj.mjOBJ_BODY,'palm')
            R=Rop@validator.data.xmat[vp].reshape(3,3);pos=validator.data.xpos[vp]
            expected=(_link_positions(validator.model,validator.data,a.URDF)-pos)@Rop.T+T_wrist[:3,3]
            conventions.append({'label':label,'palm_rotation_maxabs_error':np.max(np.abs(R-d.xmat[palm].reshape(3,3))),
                'fixed_URDF_vs_scene_all10_max_error_m':np.max(np.linalg.norm(expected-_link_positions(m,d,a.URDF),axis=1))})
        result['cases'].append({'source_inventory_index':idx,'object_id':saved['object_id'],'source_case':saved['source_case'],'native_sha256':saved['native_sha256'],'joint_names':names,
            'old_raw_tip_increases_m':np.linalg.norm(pose(raw),axis=1)-orig_norm,'raw_metadata_max_error_m':float(np.max(np.abs(pose(raw)-np.array(audit['raw_assessment']['tip_error_vectors_m'])))),
            'baseline_metadata_max_error_m':float(np.max(np.abs(orig-np.array(audit['baseline']['tip_error_vectors_m'])))),
            'trace_summary':trace_summary,'replayed_trace_vector_max_error_m':maxerr,'replayed_trace_constraint_max_error':maxcon,'derivatives':derivatives,'inward_direction':inward,'projection':projection,'old_options_control':old_options_control,
            'independent_root_palm_conventions':conventions,'maximum_palm_position_error_m':maxpalmer,'FK_callback_count':calls,'physics_steps':0,'time_s':float(d.time),'runtime_seconds':time.perf_counter()-t})
        print(json.dumps({'completed_index':idx,'trace_min_violation_um':float(viol[imin])*1e6,'rank':derivatives['original']['individual_cap_jacobian_rank'],'inward_margin':inward.get('normalized_derivative_min'),
            'projection_success':pro.success,'projection_allowance_feasible':projection['allowance_feasible'],'projection_delta_rad':projection['distance_to_original_l2_rad'],'seconds':time.perf_counter()-t}),flush=True)
        (OUT/'diagnosis.partial.json').write_text(json.dumps(js(result),indent=2,allow_nan=False)+'\n')
    for path,h in bound.items():assert sha(path)==h,path
    assert driver.COUNTS['actual']==0
    result.update({'source_dependency_sha256':bound,'dependencies_preserved':True,'integration_guards':driver.COUNTS,'physics_steps':0,'new_eval_data':False,'new_full_surface_evaluations':0,'training':False,'runtime_seconds':time.perf_counter()-started,'script_sha256':sha(__file__)})
    (OUT/'diagnosis.json').write_text(json.dumps(js(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'cases':6,'guards':driver.COUNTS,'seconds':time.perf_counter()-started}),flush=True)
if __name__=='__main__':main()
