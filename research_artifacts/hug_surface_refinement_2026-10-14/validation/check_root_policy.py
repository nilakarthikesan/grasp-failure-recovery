"""Manufactured + original-pose audit of the independent anchor/FK checks."""
from pathlib import Path
import importlib.util,json,pickle,warnings
import numpy as np
import mujoco
BASE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('validation_wrapper',BASE/'validate_candidate.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
a=v.a;old=v.OLD;root=v.ROOT;p=v.read(BASE/'protocol.json');sp=v.read(old/'sampling/protocol.json');op=v.read(old/'orientation/protocol.json');prior=root/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals';inv=v.read(prior/'existing_hug_inventory.json');fk={c['native_sha256']:c for c in v.read(prior/'static_saved_hug_fk.json')['cases']};sourcehash={str(f):v.sha(f) for f in (BASE/'check_root_policy.py',BASE/'validate_candidate.py',BASE/'protocol.json')}
for f,h in p['fixed_files_sha256'].items():assert v.sha(f)==h,f
with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
w=np.asarray(asset['weights']);grouped=np.column_stack([w[:,ids].sum(1) for ids in a.GROUPS.values()]);validator=v.ShadowPoseValidator(a.URDF.parent.parent);rows=[]
for plan in op['cases']:
    case=inv['modern_cases'][plan['source_inventory_index']];meta,tri,labels,norm,hand,objects,m,d,initial,objadr,pos,R=a.load_case(case,grouped,fk)
    names=meta['stored_command']['joint_names'];q=np.array([meta['recovered_original_targets'][n] for n in names]);grasp=v.load_hug_prediction(case['native']['path']);T_wc=np.load(case['frame_files']['T_world_camera.npy']['path']);T_w=T_wc@grasp.T_camera_wrist;Rop=T_w[:3,:3]@v.MANO_TO_OPERATOR_RIGHT.T
    local=v.camera_landmarks_to_wrist(grasp.landmarks_3d,grasp.T_camera_wrist)@v.MANO_TO_OPERATOR_RIGHT.T;pose=v.RetargetedHandPose(tuple(names),q,local,np.zeros((10,3)),grasp.T_camera_wrist);validator.set_pose(pose);palm=v._body_id(validator.model,'palm');fp=validator.data.xpos[palm].copy();fr=validator.data.xmat[palm].reshape(3,3).copy();desired=Rop@fr;expected=(v._link_positions(validator.model,validator.data,validator.source_urdf)-fp)@Rop.T+T_w[:3,3]
    actual=v._link_positions(m,d,validator.source_urdf);pid=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');position=float(np.linalg.norm(d.xpos[pid]-T_w[:3,3]));rotation=float(np.max(np.abs(d.xmat[pid].reshape(3,3)-desired)));error=float(np.linalg.norm(actual-expected,axis=1).max());assert position<1e-7 and rotation<1e-6 and error<1.1e-7
    assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
    rows.append({'case_id':case['object_id']+'_'+Path(case['native']['path']).parent.name,'palm_position_error_m':position,'palm_rotation_maxabs':rotation,'fixed_URDF_vs_captured_scene_max_link_error_m':error,'time_s':float(d.time),'object_qpos_unchanged':True})
for f,h in p['fixed_files_sha256'].items():assert v.sha(f)==h,f
for f,h in sourcehash.items():assert v.sha(f)==h,f
report={'schema':'hug_independent_root_policy_original_pose_preflight_v1','protocol_sha256':v.sha(BASE/'protocol.json'),'script_sha256':v.sha(__file__),'cases':rows,'physics_steps':0,'guards_actual_attempts':v.COUNTS['actual'],'new_eval_data':False,'candidate_not_evaluated':True,'all_preflight_checks_pass':True};v.write(BASE/'root_policy_sanity.json',report);print(json.dumps(report,indent=2))
