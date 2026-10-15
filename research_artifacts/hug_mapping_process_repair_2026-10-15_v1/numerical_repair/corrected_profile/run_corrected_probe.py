"""One frozen five-call verification of the explicit independent static profile."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,signal,time
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
    protocol=json.loads((HERE/'protocol.json').read_text());ph=sha(HERE/'protocol.json')
    def verify():
        assert sha(HERE/'protocol.json')==ph
        for p,h in protocol['input_sha256'].items():assert sha(p)==h,p
    verify();assert not (HERE/'results.json').exists()
    def timeout(*a):raise TimeoutError('Frozen120s wall budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter,STATIC_RETARGETING_PROFILE
    from grasp_failure_prediction.integrations.hug import load_hug_prediction
    grasps={k:load_hug_prediction(v['path']) for k,v in protocol['saved_inputs'].items()}
    legacy=json.loads(Path(protocol['legacy_result_path']).read_text());records=[];t0=time.perf_counter();fk_calls=0
    def metrics(robot,qpos,vectors,joint_names):
        nonlocal fk_calls
        assert tuple(robot.dof_joint_names)==tuple(joint_names)
        robot.compute_forward_kinematics(np.asarray(qpos));fk_calls+=1
        palm=robot.get_link_pose(robot.get_link_index('palm'))[:3,3]
        names=['thtip','fftip','mftip','rftip','lftip','thmiddle','ffmiddle','mfmiddle','rfmiddle','lfmiddle']
        achieved=np.stack([robot.get_link_pose(robot.get_link_index(n))[:3,3]-palm for n in names])
        errors=achieved-np.asarray(vectors);norms=np.sqrt(np.einsum('ij,ij->i',errors,errors))
        return {'geometry_reference':'Dex Pinocchio FK of unchanged source URDF, palm-relative10task vectors; no simulator surface or contact test','task_error_vectors_m':errors,
                'task_error_norms_m':norms,'mean_fingertip_error_m':float(norms[:5].mean()),'maximum_fingertip_error_m':float(norms[:5].max()),'mean_all_task_error_m':float(norms.mean())}
    def run(r,label,source):
        before=r._retargeting.last_qpos.copy();tic=time.perf_counter();pose=r.retarget(grasps[source]);o=r._retargeting.optimizer
        item={'label':label,'source':source,'joint_names':pose.joint_names,'qpos_rad':pose.qpos,'pre_reset_last_qpos_rad':before,'terminal_last_qpos_rad':r._retargeting.last_qpos.copy(),
            'reference_vectors_m':pose.reference_vectors_m,'optimizer_last_code':int(o.opt.last_optimize_result()),'optimizer_last_scalar':float(o.opt.last_optimum_value()),
            'seq_num_retargeting':r._retargeting.num_retargeting,'retargeting_provenance':asdict(pose.retargeting_provenance),'runtime_seconds':time.perf_counter()-tic,
            'source_urdf_fk':metrics(o.robot,pose.qpos,pose.reference_vectors_m,pose.joint_names)}
        assert item['seq_num_retargeting']==20 and all(x>0 for x in pose.retargeting_provenance.optimizer_status_codes)
        records.append(item);print(label,'finished',flush=True)
    root=ROOT/'third_party/dex-urdf/robots/hands'
    freshA=ShadowHandRetargeter(root,profile=STATIC_RETARGETING_PROFILE);run(freshA,'fresh_A','8')
    freshB=ShadowHandRetargeter(root,profile=STATIC_RETARGETING_PROFILE);run(freshB,'fresh_B','28')
    reused=ShadowHandRetargeter(root,profile=STATIC_RETARGETING_PROFILE)
    run(reused,'reused_A1','8');run(reused,'reused_B','28');run(reused,'reused_A2','8')
    by={r['label']:r for r in records};comparisons={}
    for label,reference in [('reused_A1','fresh_A'),('reused_B','fresh_B'),('reused_A2','fresh_A')]:
        a=np.asarray(by[label]['qpos_rad']);b=np.asarray(by[reference]['qpos_rad']);comparisons[label]={'reference':reference,'bitwise_equal':bool(np.array_equal(a,b)),'maximum_absolute_joint_difference_rad':float(np.max(np.abs(a-b)))}
    legacy_metrics={r['label']:metrics(freshA._retargeting.optimizer.robot,r['qpos_rad'],r['reference_vectors_m'],r['joint_names']) for r in legacy['mappings']}
    comparisons_to_legacy={}
    for label in ['fresh_A','fresh_B']:
        new=by[label]['source_urdf_fk'];old=legacy_metrics[label];delta=np.asarray(new['task_error_norms_m'][:5])-np.asarray(old['task_error_norms_m'][:5])
        comparisons_to_legacy[label]={'legacy_mean_fingertip_error_m':old['mean_fingertip_error_m'],'corrected_mean_fingertip_error_m':new['mean_fingertip_error_m'],
            'corrected_minus_legacy_fingertip_error_m':delta,'tips_worsened_in_source_urdf_FK':int(np.count_nonzero(delta>0)),
            'source_urdf_original_tip_nonworsening_all':bool(np.all(delta<=0)),'full_mapping_acceptance':'NOT_TESTED'}
    assert fk_calls==10
    result={'schema':'hug-corrected-static-profile-probe-v1','protocol_sha256':ph,'profile_id':STATIC_RETARGETING_PROFILE,'mappings':records,'order_invariance':comparisons,
        'legacy_source_urdf_fk':legacy_metrics,'fresh_profile_comparison':comparisons_to_legacy,'additional_source_urdf_fk_calls':fk_calls,'runtime_seconds':time.perf_counter()-t0,
        'new_model_inference':False,'new_eval_data':False,'physics_steps':0,'surface_queries':0,'new_default_selection':False,'production_default':'legacy_shadow_vector_v1'}
    signal.alarm(0);verify()
    with (HERE/'results.json').open('x') as f:f.write(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(plain({'order_invariance':comparisons,'fresh_comparison':comparisons_to_legacy,'runtime_seconds':result['runtime_seconds']})),flush=True)
if __name__=='__main__':main()
