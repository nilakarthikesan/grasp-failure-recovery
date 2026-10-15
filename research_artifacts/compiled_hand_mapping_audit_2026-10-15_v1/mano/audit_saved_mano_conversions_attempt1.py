"""Independent saved MANO conversion/compiled FK audit, never optimization/dynamics.

Reads only existing native exports and original mapped command states. The XML
FK implementation below does not import the project's FK/link extraction code.
Produces scalar engineering consistency evidence, not scientific acceptance.
"""
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import hashlib, json, pickle, sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
P=Path('/Users/nilakarthikesan/grasp-failure-prediction')
O=Path('/private/tmp/compiled_hand_audit_20261015_v1/mano')
URDF=W/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'
INV=W/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json'
HIST=W/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/static_saved_hug_fk.json'
A=np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])
TASKS=['thtip','fftip','mftip','rftip','lftip','thmiddle','ffmiddle','mfmiddle','rfmiddle','lfmiddle']
IDS=[4,8,12,16,20,2,6,10,14,18]
TIPIDS=[745,317,444,556,673]
JOINTIDS=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
physics_attempts=[]
def forbidden(*a,**k):
    physics_attempts.append('step');raise RuntimeError('physics forbidden')
for n in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,n,forbidden)
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
from grasp_failure_prediction.evaluation.retargeting import camera_landmarks_to_wrist
from grasp_failure_prediction.proposals.contactgen import load_contactgen_bundle,contactgen_mano_landmarks,contactgen_object_palm_pose

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def emit(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def mabs(v):return float(np.max(np.abs(v),initial=0))
def binding(p):return {'path':str(p),'sha256':sha(p)}
def checkbind(x):assert sha(x['path'])==x['sha256'],x['path']
def matrix(pose):
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(pose['quaternion_xyzw']).as_matrix();T[:3,3]=pose['translation_m'];return T

def xyz(node,name):return np.fromstring(node.get(name,'0 0 0') if node is not None else '0 0 0',sep=' ')
xml=ET.parse(URDF).getroot(); joints=xml.findall('joint')
roots=set(l.get('name') for l in xml.findall('link'))-{j.find('child').get('link') for j in joints}
assert len(roots)==1

def fk(q):
    frames={next(iter(roots)):np.eye(4)};pending=list(joints)
    while pending:
        progress=False
        for j in list(pending):
            par=j.find('parent').get('link');ch=j.find('child').get('link')
            if par not in frames:continue
            T=np.eye(4);orig=j.find('origin');T[:3,3]=xyz(orig,'xyz');T[:3,:3]=Rotation.from_euler('xyz',xyz(orig,'rpy')).as_matrix()
            if j.get('type') in ['revolute','continuous']:
                axis=xyz(j.find('axis'),'xyz');axis=axis/np.linalg.norm(axis)
                T[:3,:3]=T[:3,:3]@Rotation.from_rotvec(axis*q[j.get('name')]).as_matrix()
            else:assert j.get('type')=='fixed'
            frames[ch]=frames[par]@T;pending.remove(j);progress=True
        assert progress
    return frames

def compiled_frames(validator,q):
    data,model=validator.data,validator.model
    for name,v in q.items():
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);assert jid>=0
        data.qpos[model.jnt_qposadr[jid]]=v
    data.qvel[:]=0;mujoco.mj_forward(model,data);assert data.time==0
    source=fk(q);res=[]
    for i in range(1,model.nbody):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i)
        if name not in source:continue
        res.append({'link':name,'position_max_abs_m':mabs(source[name][:3,3]-data.xpos[i]),'rotation_max_abs':mabs(source[name][:3,:3]-data.xmat[i].reshape(3,3))})
    assert max(r['position_max_abs_m'] for r in res)<1e-12
    assert max(r['rotation_max_abs'] for r in res)<1e-12
    return source,res

def runtime_links(model,data):
    results=[]
    for name in TASKS:
        node=next(j for j in joints if j.find('child').get('link')==name)
        if node.get('type')=='fixed':
            pid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,node.find('parent').get('link'));assert pid>=0
            results.append(data.xpos[pid]+data.xmat[pid].reshape(3,3)@xyz(node.find('origin'),'xyz'))
        else:
            bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name);assert bid>=0;results.append(data.xpos[bid].copy())
    return np.asarray(results)

inventory=read(INV);historical={c['native_sha256']:c for c in read(HIST)['cases']}
inputs=[INV,HIST,URDF,Path(__file__)]
cg=[]
s=P/'runs/contactgen_native_run_2026-10-14_v1'
cg.append((s/'download_cn4nexx8e9fxjl/results/native_seed42/contactgen_native_bundle.json',s/'static_mapping_v1/contactgen_nominal_mapping.json','smoke_cube_seed42'))
u=P/'runs/contactgen_unseen_mapping_2026-10-15_v1'
for n in sorted((u/'download/results').glob('*/native/contactgen_native_bundle.json')):
    cg.append((n,u/'static_validation_v1'/n.parent.parent.name/'nominal_mapping.json',n.parent.parent.name))
for n,m,_ in cg:inputs.extend([n,m])
for c in inventory['modern_cases']:
    for b in [c['native'],c['stored_shadow_command'],c['scene_mjb'],*c['frame_files'].values()]:checkbind(b);inputs.append(Path(b['path']))
inputs+=list(W/'src/grasp_failure_prediction'/p for p in ['integrations/hug.py','integrations/hug_frames.py','evaluation/retargeting.py','evaluation/pose_validation.py','evaluation/case_runner.py','proposals/contactgen.py','proposals/translation.py'])
sealed={str(p):sha(p) for p in inputs}
emit(O/'inputs_frozen_v1.json',{'schema':'saved_mano_compiled_conversion_audit_inputs_v1','bindings':sealed,'scientific_threshold_changes':False,'target_changes':False,'scope':'32 original HUG and 13 original ContactGen saved mappings; no generated poses, fits, inference or dynamics','numerical_recount_tolerance_note':'1e-12 XML versus freshly compiled fixed-model float64 consistency; original float32 source rotation error retained descriptively, not a scientific fidelity allowance'})
validator=ShadowPoseValidator(URDF.parent.parent)
report={'schema':'saved_mano_compiled_conversion_audit_v1','inputs_frozen_sha256':sha(O/'inputs_frozen_v1.json'),'runtime':{'python':sys.version,'numpy':np.__version__,'mujoco':mujoco.__version__},'hug':[],'contactgen':[],'new_mapping_proposals':0,'new_native_inference':0,'retarget_optimizer_calls':0,'physics_attempts':physics_attempts,'scientific_acceptance':'NOT_ASSERTED','production_fix_required':False}
fixtures=[]
for c in inventory['modern_cases']:
    native=pickle.loads(Path(c['native']['path']).read_bytes());T=np.asarray(native['T_camera_wrist']);L=np.asarray(native['landmarks_3d']);Tc=np.load(c['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
    Tw=Tc@T
    operator=camera_landmarks_to_wrist(L,T)@A.T
    independent=(L-T[:3,3])@T[:3,:3]@A.T
    assert np.array_equal(operator,independent)
    with np.load(c['stored_shadow_command']['path'],allow_pickle=False) as a:
        row=c['stored_shadow_command']['row_index'];q=dict(zip(a['joint_names'].tolist(),a['nominal_target_rad'][row].tolist(),strict=True));rootpos=a['root_target_position_m'][row].copy();rootquat=a['root_target_quaternion_wxyz'][row].copy()
    F,checks=compiled_frames(validator,q)
    model=mujoco.MjModel.from_binary_path(c['scene_mjb']['path']);data=mujoco.MjData(model)
    with np.load(c['frame_files']['scene_state.npz']['path'],allow_pickle=False) as a:data.qpos[:]=a['qpos']
    initial=data.qpos.copy();oid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'object_free');oa=model.jnt_qposadr[oid]
    rid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'world_joint');ra=model.jnt_qposadr[rid];data.qpos[ra:ra+3]=rootpos;data.qpos[ra+3:ra+7]=rootquat
    for name,v in q.items():data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)]]=v
    data.qvel[:]=0;mujoco.mj_forward(model,data);assert data.time==0 and np.array_equal(data.qpos[oa:oa+7],initial[oa:oa+7])
    shadow=runtime_links(model,data);native_world=L@Tc[:3,:3].T+Tc[:3,3]
    palm=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'palm')
    expected=np.asarray([F[t][:3,3]-F['palm'][:3,3] for t in TASKS])@(Tw[:3,:3]@A.T).T+Tw[:3,3]
    historical_shadow=np.asarray([r['shadow_world_m'] for r in historical[c['native']['sha256']]['links']])
    assert mabs(shadow-historical_shadow)<1e-12
    error=np.linalg.norm(shadow-native_world[IDS],axis=1)
    report['hug'].append({'native':c['native'],'object_id':c['object_id'],'source_wrist_landmark_vs_origin_m':float(np.linalg.norm(L[0]-T[:3,3])),'operator_adapter_vs_independent_max_abs_m':mabs(operator-independent),'source_rotation_orthogonality_max_abs':mabs(T[:3,:3].T@T[:3,:3]-np.eye(3)),'native_roundtrip_max_abs_m':mabs(operator@A@T[:3,:3].T+T[:3,3]-L),'compiled_vs_independent_xml':checks,'saved_scene_vs_original_saved_fk_max_abs_m':mabs(shadow-historical_shadow),'fresh_xml_vs_saved_scene_max_link_position_m':float(np.max(np.linalg.norm(shadow-expected,axis=1))),'actual_palm_vs_native_wrist_m':float(np.linalg.norm(data.xpos[palm]-Tw[:3,3])),'native_to_shadow_mean_tip_error_m':float(error[:5].mean()),'native_to_shadow_max_tip_error_m':float(error[:5].max()),'time_s':float(data.time),'object_qpos_unchanged':True})
    fixtures.append({'generator':'HUG','case_id':c['native']['sha256'],'native':c['native'],'joint_targets_rad':q,'scene_mjb':c['scene_mjb'],'saved_command':c['stored_shadow_command'],'frame_files':c['frame_files']})

for n,m,caseid in cg:
    raw=read(n);bundle=load_contactgen_bundle(n);assert len(bundle.samples)==1;s=bundle.samples[0]
    nominal=read(m)['nominal_mapping'];assert read(m)['native_bundle_sha256']==sha(n)
    q=dict(zip(nominal['joint_names'],nominal['joint_targets_rad'],strict=True))
    F,checks=compiled_frames(validator,q)
    vertices=np.asarray(s.hand_vertices_centered_m);frames=np.asarray(s.hand_frames_centered_m);center=np.asarray(bundle.mesh_vertex_mean_m);Tmesh=matrix(raw['object_input']['object_to_mesh'])
    # Explicit thumb-first source association, independent of adapter constants.
    source_indices=[('j',0),('j',13),('j',14),('j',15),('v',745),('j',1),('j',2),('j',3),('v',317),('j',4),('j',5),('j',6),('v',444),('j',10),('j',11),('j',12),('v',556),('j',7),('j',8),('j',9),('v',673)]
    source=np.stack([frames[i,:3,3] if typ=='j' else vertices[i] for typ,i in source_indices])
    L=(source+center)@Tmesh[:3,:3].T+Tmesh[:3,3]
    Tw=Tmesh@frames[0];Tw[:3,3]+=Tmesh[:3,:3]@center
    adapted=contactgen_mano_landmarks(s,bundle.object_input,bundle.mesh_vertex_mean_m)
    assert mabs(adapted.landmarks_3d-L)<1e-12 and mabs(adapted.T_camera_wrist-Tw)<1e-12
    operator=(L-Tw[:3,3])@Tw[:3,:3]@A.T
    assert mabs(operator-np.asarray(nominal['landmarks_operator_m']))<1e-12
    pose=SimpleNamespace(joint_names=tuple(q),qpos=np.asarray(list(q.values())))
    produced=contactgen_object_palm_pose(adapted,pose,validator).model_dump(mode='json')
    Tp=matrix(produced);stored=matrix(nominal['object_to_palm'])
    assert mabs(Tp-stored)<1e-12
    # Independent XML target chain expressed in the recorded palm axes.
    palm_inv=np.linalg.inv(F['palm']);shadow=np.asarray([(Tp@palm_inv@F[t])[:3,3] for t in TASKS])
    historical=np.asarray([r['shadow_object_m'] for r in nominal['links']])
    assert mabs(shadow-historical)<1e-12
    err=np.linalg.norm(shadow-L[IDS],axis=1)
    report['contactgen'].append({'case_id':caseid,'native':binding(n),'nominal':binding(m),'all21_landmark_adapter_max_abs_m':mabs(adapted.landmarks_3d-L),'all21_landmark_saved_operator_max_abs_m':mabs(operator-np.asarray(nominal['landmarks_operator_m'])),'wrist_transform_adapter_max_abs':mabs(adapted.T_camera_wrist-Tw),'source_root_rotation_orthogonality_max_abs':mabs(Tw[:3,:3].T@Tw[:3,:3]-np.eye(3)),'produced_vs_saved_palm_transform_max_abs':mabs(Tp-stored),'independent_xml_vs_saved_placed_task_max_abs_m':mabs(shadow-historical),'compiled_vs_independent_xml':checks,'native_to_shadow_mean_tip_error_m':float(err[:5].mean()),'native_to_shadow_max_tip_error_m':float(err[:5].max()),'anatomical_palm_equivalence':'NOT_ESTABLISHED','source_hand_geometry':'MANO_right_pca26_nonflat_mean_open_wrist','time_s':float(validator.data.time)})
    fixtures.append({'generator':'ContactGen','case_id':caseid,'native':binding(n),'nominal':binding(m),'joint_targets_rad':q,'object_to_palm':nominal['object_to_palm']})
assert not physics_attempts
assert all(sha(p)==h for p,h in sealed.items())
report['all_input_bytes_unchanged']=True
report['summary']={'HUG_saved_cases':len(report['hug']),'ContactGen_saved_cases':len(report['contactgen']),'compiled_fixed_vs_xml_link_position_max_abs_m':max(c['position_max_abs_m'] for rows in [report['hug'],report['contactgen']] for r in rows for c in r['compiled_vs_independent_xml']),'compiled_fixed_vs_xml_link_rotation_max_abs':max(c['rotation_max_abs'] for rows in [report['hug'],report['contactgen']] for r in rows for c in r['compiled_vs_independent_xml']),'HUG_fresh_xml_vs_saved_scene_max_link_position_m':max(r['fresh_xml_vs_saved_scene_max_link_position_m'] for r in report['hug']),'HUG_native_roundtrip_max_abs_m':max(r['native_roundtrip_max_abs_m'] for r in report['hug']),'HUG_saved_scene_vs_original_saved_fk_max_abs_m':max(r['saved_scene_vs_original_saved_fk_max_abs_m'] for r in report['hug']),'ContactGen_independent_xml_vs_saved_placed_task_max_abs_m':max(r['independent_xml_vs_saved_placed_task_max_abs_m'] for r in report['contactgen']),'status':'PASS_SCOPED_NUMERICAL_RECOUNT_NO_NEW_CONVERSION_DEFECT','native_to_target_surface_fidelity':'REMAINS_PREVIOUSLY_MEASURED_STATUS_UNCHANGED'}
emit(O/'saved_mano_conversion_audit_v1.json',report)
emit(O/'saved_target_pose_fixtures_v1.json',{'schema':'saved_mano_target_pose_bindings_v1','inputs_frozen_sha256':sha(O/'inputs_frozen_v1.json'),'fixtures':fixtures})
print(json.dumps(report['summary'],indent=2))
