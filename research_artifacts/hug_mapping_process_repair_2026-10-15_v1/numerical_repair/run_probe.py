"""Five fixed saved-input mappings and one derivative-consistency probe.
No model inference, simulator construction, physics, surface query or eval data.
"""
from pathlib import Path
import hashlib,json,time,signal,sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());protocol_hash=sha(HERE/'protocol.json')
    for p,h in protocol['input_sha256'].items():assert sha(p)==h,p
    assert not (HERE/'results.json').exists()
    def timeout(*a):raise TimeoutError('Frozen 120s wall budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter,camera_landmarks_to_wrist,reference_vectors
    from grasp_failure_prediction.integrations.hug import load_hug_prediction
    from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
    import dex_retargeting.optimizer as module
    assert str(Path(module.__file__).resolve())==protocol['dex_optimizer_path']
    from dex_retargeting.constants import OPERATOR2MANO_RIGHT
    assert np.array_equal(MANO_TO_OPERATOR_RIGHT,OPERATOR2MANO_RIGHT.T)
    grasps={k:load_hug_prediction(v['path']) for k,v in protocol['saved_inputs'].items()}
    records=[];t0=time.perf_counter()
    def run(r,label,source):
        before=r._retargeting.last_qpos.copy();tic=time.perf_counter();pose=r.retarget(grasps[source]);o=r._retargeting.optimizer
        item={'label':label,'source':source,'joint_names':pose.joint_names,'qpos_rad':pose.qpos,'initial_last_qpos_rad':before,'terminal_last_qpos_rad':r._retargeting.last_qpos.copy(),
              'reference_vectors_m':pose.reference_vectors_m,'optimizer_last_code':int(o.opt.last_optimize_result()),'optimizer_last_scalar':float(o.opt.last_optimum_value()),
              'seq_num_retargeting':r._retargeting.num_retargeting,'runtime_seconds':time.perf_counter()-tic,'filter_disabled':r._retargeting.filter is None,'scaling':o.scaling,'norm_delta':o.norm_delta}
        records.append(item);print(label,'finished',round(item['runtime_seconds'],4),flush=True)
    root=ROOT/'third_party/dex-urdf/robots/hands'
    freshA=ShadowHandRetargeter(root);run(freshA,'fresh_A','8')
    freshB=ShadowHandRetargeter(root);run(freshB,'fresh_B','28')
    reuse=ShadowHandRetargeter(root)
    run(reuse,'reused_A1','8');run(reuse,'reused_B','28');run(reuse,'reused_A2','8')
    by={r['label']:r for r in records}
    diffs={label:{'max_absolute_joint_difference_rad':float(np.max(np.abs(by[label]['qpos_rad']-by[reference]['qpos_rad']))),
                  'joint_difference_rad':by[label]['qpos_rad']-by[reference]['qpos_rad'],'bitwise_equal':bool(np.array_equal(by[label]['qpos_rad'],by[reference]['qpos_rad'])),'reference':reference}
           for label,reference in [('reused_A1','fresh_A'),('reused_B','fresh_B'),('reused_A2','fresh_A')]}
    # Fixed interior joint state, not an optimizer output chosen for its discrepancy.
    o=freshA._retargeting.optimizer;limits=freshA.joint_limits
    last=limits.mean(axis=1).astype(np.float32);direction=np.where(np.arange(len(last))%2==0,1.,-1.)
    x=last.astype(float)+direction*.1*(limits[:,1]-limits[:,0])
    assert np.all(x>limits[:,0]) and np.all(x<limits[:,1])
    vectors=np.asarray(by['fresh_A']['reference_vectors_m'],np.float32);fixed=np.array([],np.float32)
    fn=o.get_objective_function(vectors,fixed,last)
    grad=np.zeros_like(x);value=fn(x,grad);fd=np.empty_like(x);h=protocol['gradient_probe']['central_difference_step_rad']
    for i in range(len(x)):
        xp=x.copy();xm=x.copy();xp[i]+=h;xm[i]-=h
        fd[i]=(fn(xp,np.array([]))-fn(xm,np.array([])))/(2*h)
    delta=o.norm_delta;expected=2*delta*(x-last)
    # Disable temporal gradient term in memory at the same fixed probe; no mapping is rerun.
    o.norm_delta=0.;gstatic=np.zeros_like(x);vstatic=fn(x,gstatic);o.norm_delta=delta
    probe={'x_rad':x,'last_qpos_rad':last,'finite_difference_step_rad':h,'returned_scalar':value,'reported_gradient':grad,'finite_difference_gradient':fd,
           'expected_unreturned_regularizer_gradient':expected,'gradient_minus_fd':grad-fd,'gradient_residual_after_subtracting_unreturned_term':grad-fd-expected,
           'maximum_absolute_gradient_mismatch':float(np.max(np.abs(grad-fd))),'maximum_residual_after_subtraction':float(np.max(np.abs(grad-fd-expected))),
           'norm_delta_zero_same_probe_scalar':vstatic,'norm_delta_zero_same_probe_gradient':gstatic,'norm_delta_zero_maximum_gradient_mismatch':float(np.max(np.abs(gstatic-fd))),
           'objective_callback_evaluations':50,'new_optimizer_calls_for_gradient_probe':0}
    result={'schema':'hug-retargeter-state-and-gradient-probe-v1','protocol_sha256':protocol_hash,'mappings':records,'comparisons':diffs,'gradient_probe':probe,
            'native_origin_offset_m':{k:np.asarray(g.landmarks_3d[0])-np.asarray(g.T_camera_wrist[:3,3]) for k,g in grasps.items()},
            'runtime_seconds':time.perf_counter()-t0,'new_model_inference':False,'new_eval_data':False,'physics_steps':0,'surface_queries':0,'production_changed':False}
    signal.alarm(0)
    for p,hsh in protocol['input_sha256'].items():assert sha(p)==hsh,p
    assert sha(HERE/'protocol.json')==protocol_hash
    with (HERE/'results.json').open('x') as f:f.write(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'comparisons':plain(diffs),'gradient_max':probe['maximum_absolute_gradient_mismatch'],'gradient_remainder':probe['maximum_residual_after_subtraction'],'norm_delta_zero_gradient_max':probe['norm_delta_zero_maximum_gradient_mismatch'],'runtime_seconds':result['runtime_seconds']}),flush=True)
if __name__=='__main__':main()
