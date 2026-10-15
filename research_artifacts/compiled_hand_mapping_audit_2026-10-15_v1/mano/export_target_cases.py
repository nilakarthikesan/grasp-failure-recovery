from pathlib import Path
import hashlib,json
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
for n in ['mj_step','mj_step1','mj_step2']:
    setattr(mujoco,n,lambda *a,**k: (_ for _ in ()).throw(RuntimeError('No dynamics allowed')))
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
O=Path('/private/tmp/compiled_hand_audit_20261015_v1/mano')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def T(model,data,name):
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name);assert bid>=0
    a=np.eye(4);a[:3,:3]=data.xmat[bid].reshape(3,3);a[:3,3]=data.xpos[bid];return a
fixtures=read(O/'saved_target_pose_fixtures_v1.json')['fixtures'];report=read(O/'saved_mano_conversion_audit_v1.json');out=[]
for f in fixtures:
    inputs={f['native']['path']:f['native']['sha256']}
    r={'case_id':f['case_id'],'generator':f['generator'],'variant':'original_saved_nominal_mapping','target_joint_targets_rad':f['joint_targets_rad'],'native_fk_reference':f['native'],'conversion_audit_reference':{'path':str(O/'saved_mano_conversion_audit_v1.json'),'sha256':sha(O/'saved_mano_conversion_audit_v1.json')}}
    if f['generator']=='HUG':
        model=mujoco.MjModel.from_binary_path(f['scene_mjb']['path']);data=mujoco.MjData(model)
        for b in [f['scene_mjb'],f['saved_command'],*f['frame_files'].values()]:assert sha(b['path'])==b['sha256'];inputs[b['path']]=b['sha256']
        with np.load(f['frame_files']['scene_state.npz']['path'],allow_pickle=False) as a:data.qpos[:]=a['qpos']
        with np.load(f['saved_command']['path'],allow_pickle=False) as a:
            row=f['saved_command']['row_index'];rootpos=a['root_target_position_m'][row];rootquat=a['root_target_quaternion_wxyz'][row]
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'world_joint');adr=model.jnt_qposadr[jid]
        data.qpos[adr:adr+3]=rootpos;data.qpos[adr+3:adr+7]=rootquat
        for name,v in f['joint_targets_rad'].items():data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)]]=v
        mujoco.mj_forward(model,data);assert data.time==0
        obj_inv=np.linalg.inv(T(model,data,'object'))
        r['object_from_target_palm']=(obj_inv@T(model,data,'palm')).tolist()
        r['object_from_target_root']=(obj_inv@T(model,data,mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.jnt_bodyid[jid]))).tolist()
        r['target_root_link']=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.jnt_bodyid[jid])
        r['target_fk_reference']=f['scene_mjb'];r['original_status']='EXISTING_SAVED_MAPPED_COMMAND_NO_NEW_ADMISSION'
        r['placement_scope']='Existing stored compiled-scene pose. Native MANO wrist-to-palm convention is verified separately, small compiled-scene serialization effects retained.'
    else:
        nominal=read(f['nominal']['path'])['nominal_mapping'];inputs[f['nominal']['path']]=f['nominal']['sha256']
        t=np.eye(4);t[:3,:3]=Rotation.from_quat(f['object_to_palm']['quaternion_xyzw']).as_matrix();t[:3,3]=f['object_to_palm']['translation_m']
        r['object_from_target_palm']=t.tolist();r['target_fk_reference']=f['nominal'];r['original_status']='ORIGINAL_ENGINEERING_GATE_'+nominal['engineering_gate']
        r['placement_scope']='Original canonical-adapter palm transform. Includes original capsule gate failure; no repaired candidate substitution.'
    for p,h in inputs.items():assert sha(p)==h
    r['input_file_hashes']=inputs;out.append(r)
p=O/'target_cases_v1.json';assert not p.exists();p.write_text(json.dumps({'schema_version':'native_saved_target_cases_v1','source_audit_sha256':sha(O/'saved_mano_conversion_audit_v1.json'),'case_count':len(out),'case_scope':'32 HUG plus 13 ContactGen original saved target poses; no fresh fitting or new grasp outputs; unchanged scientific mapping statuses','cases':out},indent=2)+'\n')
print(json.dumps({'path':str(p),'sha256':sha(p),'cases':len(out),'generator_counts':{'HUG':32,'ContactGen':13}},indent=2))
