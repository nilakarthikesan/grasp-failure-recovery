"""Independent static source32 geometry import review; no inference or dynamics."""
from pathlib import Path
import sys,types,json,pickle,hashlib,warnings
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
RUNTIME=Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check')
CHECKOUT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sys.path.insert(0,str(CHECKOUT/'src'));sys.path.extend([str(RUNTIME/'hug-deps'),str(RUNTIME/'deps')])
import torch
package=types.ModuleType('hug');package.__path__=[str(RUNTIME/'hug/src')];sys.modules['hug']=package
from hug.models.mano import MANO
from hug.utils.transform_utils import six_d_to_rotation_matrix
from manotorch.utils.geometry import rotation_to_axis_angle
from grasp_failure_prediction.evaluation import native_mano as nm
HERE=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_hash=sha(nm.__file__);pp=Path('/private/tmp/hug_native_contract_2026-10-14/protocol.json');p=json.loads(pp.read_text())
counts={'static_forwards':0,'physics_steps':0}
original=mujoco.mj_forward
def forward(*args,**kwargs):counts['static_forwards']+=1;return original(*args,**kwargs)
def forbidden(*args,**kwargs):counts['physics_steps']+=1;raise AssertionError('No physics authorized')
mujoco.mj_forward=forward
for key in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,key,forbidden)
torch.set_num_threads(1)
with warnings.catch_warnings():warnings.simplefilter('ignore');layer=MANO(side='right',use_pca=False,flat_hand_mean=True).float().eval().mano_layer
parents=np.array(layer.kintree_parents,dtype=np.int64);parents[0]=-1;tips=np.array([745,317,444,556,673]);tipj=np.array([15,3,6,12,9]);owner=np.argmax(layer.th_weights.numpy(),axis=1)

def scalar_fk(joints,rootp,rootR,locals):
 out=np.tile(np.eye(4),(16,1,1));out[0,:3,:3]=rootR;out[0,:3,3]=rootp
 for k in range(1,16):
  rel=np.eye(4);rel[:3,:3]=locals[k-1];rel[:3,3]=joints[k]-joints[parents[k]]
  for i in range(4):
   for j in range(4):out[k,i,j]=sum(float(out[parents[k],i,l])*float(rel[l,j]) for l in range(4))
 return out

def check(model,geometry,rootp,rootR,locals):
 model.set_pose(rootp,rootR,locals);read=model.readback();frames=scalar_fk(geometry.neutral_joints,rootp,rootR,locals)
 maxjoint=float(np.linalg.norm(read.joint_positions_world-frames[:,:3,3],axis=1).max())
 rigidtips=np.array([frames[j,:3,:3]@(tip-geometry.neutral_joints[j])+frames[j,:3,3] for tip,j in zip(geometry.neutral_tips,tipj)])
 maxtip=float(np.linalg.norm(read.tip_positions_world-rigidtips,axis=1).max())
 expected=[]
 for j in range(1,16):
  children=np.flatnonzero(parents==j)
  end=geometry.neutral_joints[children[0]] if len(children) else geometry.neutral_tips[np.flatnonzero(tipj==j)[0]]
  expected.append([frames[j,:3,3],frames[j,:3,:3]@(end-geometry.neutral_joints[j])+frames[j,:3,3]])
 endpoint=float(np.linalg.norm(read.capsule_endpoints_world-np.array(expected),axis=2).max())
 palm=geometry.neutral_vertices[owner==0]@rootR.T+rootp
 distances=[cKDTree(read.palm_vertices_world).query(palm)[0].max(),cKDTree(palm).query(read.palm_vertices_world)[0].max()]
 return {'joint_error_m':maxjoint,'rigid_tip_import_error_m':maxtip,'capsule_endpoint_error_m':endpoint,'palm_vertex_set_error_m':float(max(distances)), 'time':float(model.data.time)}
rows=[];models={};last=None
for case in p['cases']:
 path=Path(case['native_path']);assert sha(path)==case['native_sha256'];raw=pickle.loads(path.read_bytes());beta=np.asarray(raw['shape'],dtype=np.float32).reshape(1,10);key=hashlib.sha256(beta.tobytes()).hexdigest()
 if key not in models:
  with torch.no_grad():
   b=torch.from_numpy(beta);neutral=layer(torch.zeros(1,48),b)
   J=(layer.th_J_regressor@(layer.th_v_template+torch.matmul(layer.th_shapedirs,b.T).permute(2,0,1)))[0].numpy().astype(float);J-=J[0].copy()
  V=neutral.verts[0].numpy().astype(float)
  geometry=nm.NativeManoGeometry(J,parents,V,layer.get_mano_closed_faces().numpy(),layer.th_weights.numpy(),V[tips],tipj)
  model=nm.NativeManoStaticModel(geometry);models[key]=(geometry,model)
 geometry,model=models[key]
 rootp=np.asarray(raw['t'],float).reshape(3)
 # Use source stored axis-angle for local rotations. Its conversion to double SO3 is explicit.
 locals=Rotation.from_rotvec(np.asarray(raw['pose'],float).reshape(15,3)).as_matrix()
 rootR=Rotation.from_matrix(np.asarray(raw['T_camera_wrist'],float)[:3,:3]).as_matrix()
 metrics=check(model,geometry,rootp,rootR,locals)
 assert metrics['joint_error_m']<1e-12 and metrics['rigid_tip_import_error_m']<1e-12 and metrics['capsule_endpoint_error_m']<1e-12 and metrics['palm_vertex_set_error_m']<1e-7 and metrics['time']==0
 with torch.no_grad():
  rots=torch.cat([six_d_to_rotation_matrix(torch.from_numpy(np.asarray(raw['R_6d'],dtype=np.float32).reshape(1,6))),six_d_to_rotation_matrix(torch.from_numpy(np.asarray(raw['pose_6d'],dtype=np.float32).reshape(15,6)))])
  source=layer(rotation_to_axis_angle(rots).reshape(1,48),torch.from_numpy(beta))
 read=model.readback();ref=source.transforms_abs[0].numpy()[:,:3,3]+rootp
 rows.append({'index':case['source_inventory_index'],'object_id':case['object_id'],'native_sha256':case['native_sha256'],**metrics,'source_float32_bone_error_m':float(np.linalg.norm(read.joint_positions_world-ref,axis=1).max()),'native_tip_error_m':np.linalg.norm(read.tip_positions_world-(source.verts[0].numpy()[tips]+rootp),axis=1).tolist()})
 assert sha(path)==case['native_sha256'];last=(geometry,model,rootp,rootR,locals)
geometry,model,p0,R0,locals0=last
# A-B-A: reusing a model must not carry state across poses.
model.set_pose(p0,R0,locals0);before=model.data.qpos.copy();frames_before=model.readback()
Q=Rotation.from_rotvec([0.7,-0.4,1.1]).as_matrix();t=np.array([0.13,-0.21,0.87]);check(model,geometry,Q@p0+t,Q@R0,locals0)
cov=model.readback();covariance=float(np.linalg.norm(cov.capsule_endpoints_world-(np.einsum('ij,klj->kli',Q,frames_before.capsule_endpoints_world)+t),axis=2).max())
assert covariance<1e-12
model.set_pose(p0,R0,locals0);assert np.array_equal(before,model.data.qpos)
invalid=[]
for name,argpos,argR,argloc in [('nan_translation',[float('nan'),0,0],R0,locals0),('reflection',p0,np.diag([-1,1,1]),locals0),('scaled_root',p0,2*R0,locals0),('wrong_local_shape',p0,R0,locals0[:14])]:
 try:model.set_pose(argpos,argR,argloc)
 except ValueError:invalid.append(name)
 else:raise AssertionError(name)
 assert np.array_equal(before,model.data.qpos)
model.data.time=1
try:model.readback()
except RuntimeError:invalid.append('advanced_data_readback')
else:raise AssertionError('readback timeguard')
model.data.time=0
assert counts['physics_steps']==0 and sha(nm.__file__)==source_hash
result={'schema':'independent_native_mano_module_static_review_v1','module_path':nm.__file__,'module_sha256':source_hash,'script_sha256':sha(__file__),'native_case_count':len(rows),'distinct_shape_model_count':len(models),'cases':rows,'max_import_joint_error_m':max(r['joint_error_m'] for r in rows),'max_import_tip_error_m':max(r['rigid_tip_import_error_m'] for r in rows),'max_import_capsule_endpoint_error_m':max(r['capsule_endpoint_error_m'] for r in rows),'max_import_palm_vertex_set_error_m':max(r['palm_vertex_set_error_m'] for r in rows),'max_source_float32_bone_error_m':max(r['source_float32_bone_error_m'] for r in rows),'root_rigid_transform_covariance_max_error_m':covariance,'A_B_A_bitwise_qpos_repeat':True,'invalid_inputs_refused_before_mutation':invalid,'counts':counts,'new_eval_data':False,'new_proposals':0,'new_inference':0,'scope':'Static source import and approximation implementation only, not physics or strict surface-preservation acceptance'}
out=HERE/'module_review_results.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
