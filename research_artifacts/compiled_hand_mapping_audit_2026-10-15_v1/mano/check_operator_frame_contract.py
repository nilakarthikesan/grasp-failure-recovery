"""Saved-q check of the vector optimizer coordinate convention, no optimization."""
from pathlib import Path
import hashlib,json,pickle
import mujoco,numpy as np
for n in ['mj_step','mj_step1','mj_step2']:
    setattr(mujoco,n,lambda *a,**k: (_ for _ in ()).throw(RuntimeError('No dynamics')))
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
from grasp_failure_prediction.evaluation.retargeting import RetargetedHandPose
O=Path('/private/tmp/compiled_hand_audit_20261015_v1/mano')
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
P=Path('/Users/nilakarthikesan/grasp-failure-prediction')
A=np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])
IDS=[4,8,12,16,20,2,6,10,14,18]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=ShadowPoseValidator(W/'third_party/dex-urdf/robots/hands');cases=read(O/'target_cases_v1.json')['cases'];rows=[]
for c in cases:
    q=c['target_joint_targets_rad'];names=tuple(q);values=np.array(list(q.values()))
    if c['generator']=='HUG':
        d=pickle.loads(Path(c['native_fk_reference']['path']).read_bytes());T=np.array(d['T_camera_wrist'],dtype=np.float64);L=np.array(d['landmarks_3d'],dtype=np.float64);local=(L-T[:3,3])@T[:3,:3]@A.T
    else:
        d=read(c['target_fk_reference']['path'])['nominal_mapping'];local=np.asarray(d['landmarks_operator_m']);T=np.asarray(d['T_object_wrist'])
    reference=local[IDS]-local[0]
    val=v.validate(RetargetedHandPose(names,values,local,reference,T))
    palm=mujoco.mj_name2id(v.model,mujoco.mjtObj.mjOBJ_BODY,'palm');R=v.data.xmat[palm].reshape(3,3)
    # The hypothetical below changes only target vector axes and therefore is
    # intentionally inconsistent; it is a diagnostic, never a proposed fix.
    wrong=np.linalg.norm(val.achieved_vectors_m@R-reference,axis=1)
    expected=np.linalg.norm(val.achieved_vectors_m-reference,axis=1)
    assert np.array_equal(expected,val.errors_m)
    rows.append({'case_id':c['case_id'],'generator':c['generator'],'WRJ1_rad':q['WRJ1'],'WRJ2_rad':q['WRJ2'],'correct_declared_operator_mean_tip_m':float(expected[:5].mean()),'hypothetical_incorrect_palm_axes_only_mean_tip_m':float(wrong[:5].mean()),'declared_error_recount_max_abs_m':float(np.max(np.abs(expected-val.errors_m))),'time_s':float(v.data.time)})
inputs=[O/'target_cases_v1.json',W/'src/grasp_failure_prediction/evaluation/pose_validation.py',W/'src/grasp_failure_prediction/evaluation/retargeting.py',P/'runs/hug_integration_check/deps/dex_retargeting/configs/teleop/shadow_hand_right.yml',P/'runs/hug_integration_check/deps/dex_retargeting/optimizer.py',Path(__file__)]
report={'schema_version':'mano_operator_frame_contract_static_audit_v1','source_bindings':{str(p):sha(p) for p in inputs},'reason':'Official VectorOptimizer body_pos task-origin vectors are fixed-root FK coordinates. Target ShadowPoseValidator uses that same fixed-base/operator frame. Current wrist angles are part of forward kinematics; rotating achieved into instantaneous palm axes alone would change the contract.','status':'PASS_EXISTING_CONVENTION','cases':rows,'case_count':len(rows),'all_nonzero_wrists':all(abs(r['WRJ1_rad'])+abs(r['WRJ2_rad'])>0 for r in rows),'physics_steps':0,'retarget_optimizer_calls':0,'new_native_inference':0,'target_or_acceptance_criteria_changed':False}
p=O/'operator_frame_contract_v1.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'path':str(p),'sha256':sha(p),'case_count':len(rows),'all_nonzero_wrists':report['all_nonzero_wrists'],'status':report['status']},indent=2))
