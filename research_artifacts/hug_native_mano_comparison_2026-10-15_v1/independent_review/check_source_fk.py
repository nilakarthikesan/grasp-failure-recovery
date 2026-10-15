"""Independent scalar/tree FK and native tip rigidity audit on32 existing HUG exports.
No inference model, native proposal generation, simulator, or outcomes.
"""
from pathlib import Path
import sys, types, json, pickle, hashlib, warnings
import numpy as np
# Preserve the installed NumPy, then append the existing licensed MANO dependencies.
ROOT=Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check')
sys.path.extend([str(ROOT/'hug-deps'),str(ROOT/'deps')])
import torch
from scipy.spatial.transform import Rotation
sys.modules['mujoco']=None
pkg=types.ModuleType('hug');pkg.__path__=[str(ROOT/'hug/src')];sys.modules['hug']=pkg
from hug.models.mano import MANO
from hug.utils.transform_utils import six_d_to_rotation_matrix
from manotorch.utils.geometry import rotation_to_axis_angle
HERE=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol_path=Path('/private/tmp/hug_native_contract_2026-10-14/protocol.json')
p=json.loads(protocol_path.read_text())
torch.set_num_threads(1)
with warnings.catch_warnings():
 warnings.simplefilter('ignore');model=MANO(side='right',use_pca=False,flat_hand_mean=True).float().eval()
layer=model.mano_layer
parents=np.array(layer.kintree_parents,dtype=np.int64);parents[0]=-1
assert np.array_equal(parents,[-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14])
tipvertex=np.array([745,317,444,556,673]);tipjoint=np.array([15,3,6,12,9])
rows=[]
for case in p['cases']:
 path=Path(case['native_path']);assert sha(path)==case['native_sha256'];raw=pickle.loads(path.read_bytes())
 shape=torch.from_numpy(np.asarray(raw['shape'],dtype=np.float32).reshape(1,10))
 params=np.concatenate([np.asarray(raw['t']).reshape(3),np.asarray(raw['R_6d']).reshape(6),np.asarray(raw['pose_6d']).reshape(90)]).astype(np.float32)
 with torch.no_grad():
  neutral=layer(torch.zeros((1,48)),shape)
  R=six_d_to_rotation_matrix(torch.from_numpy(params[3:9].reshape(1,6)))
  local=six_d_to_rotation_matrix(torch.from_numpy(params[9:].reshape(15,6)))
  rots=torch.cat([R,local]);aa=rotation_to_axis_angle(rots).reshape(1,48)
  posed=layer(aa,shape)
  effective=layer.rotation_by_axisang(aa)['full_rots'][0].numpy().astype(float)
  J=(layer.th_J_regressor@(layer.th_v_template+torch.matmul(layer.th_shapedirs,shape.T).permute(2,0,1)))[0].numpy().astype(float)
 J=J-J[0]
 G=np.tile(np.eye(4),(16,1,1))
 for k,parent in enumerate(parents):
  relative=np.eye(4);relative[:3,:3]=effective[k];relative[:3,3]=J[k] if parent<0 else J[k]-J[parent]
  if parent<0:G[k]=relative
  else:
   for i in range(4):
    for j in range(4):G[k,i,j]=sum(float(G[parent,i,l])*float(relative[l,j]) for l in range(4))
 ref=posed.transforms_abs[0].numpy().astype(float)
 tip0=neutral.verts[0].numpy().astype(float)[tipvertex]
 tiprigid=np.array([G[k,:3,:3]@(tip-J[k])+G[k,:3,3] for tip,k in zip(tip0,tipjoint)])
 tipnative=posed.verts[0].numpy().astype(float)[tipvertex]
 saved=np.asarray(raw['landmarks_3d']).reshape(21,3).astype(float)
 tipnorm=np.linalg.norm(tiprigid-tipnative,axis=1)
 rows.append({'index':case['source_inventory_index'],'object_id':case['object_id'],'native_sha256':case['native_sha256'],'shape_sha256':hashlib.sha256(shape.numpy().tobytes()).hexdigest(),'shape_norm':float(torch.linalg.norm(shape)),
 'independent_scalar_FK_position_max_error_m':float(np.linalg.norm(G[:,:3,3]-ref[:,:3,3],axis=1).max()),'independent_scalar_FK_rotation_max_entry_error':float(np.max(np.abs(G[:,:3,:3]-ref[:,:3,:3]))),
 'native_float32_effective_rotation_orthogonality_max_error':float(np.max(np.abs(np.einsum('kji,kjl->kil',effective,effective)-np.eye(3)))),
 'rigid_tip_vs_native_skinned_tip_error_m':tipnorm.tolist(),'rigid_tip_vs_native_skinned_tip_mean_m':float(tipnorm.mean()),'rigid_tip_vs_native_skinned_tip_max_m':float(tipnorm.max()),
 'source_neutral_vertices_vs_neutral_raw_shaped_template_max_error_m':float(np.linalg.norm(neutral.verts[0].numpy().astype(float)-(layer.th_v_template+torch.matmul(layer.th_shapedirs,shape.T).permute(2,0,1))[0].numpy().astype(float)+(layer.th_J_regressor@(layer.th_v_template+torch.matmul(layer.th_shapedirs,shape.T).permute(2,0,1)))[0,0].numpy().astype(float),axis=1).max()),
 'native_rebuild_vs_saved_tip_max_error_m':float(np.linalg.norm(tipnative+params[:3]-saved[[4,8,12,16,20]],axis=1).max())})
 assert sha(path)==case['native_sha256']
result={'schema':'independent_source_MANO_FK_and_rigid_tip_diagnostic_v1','script_sha256':sha(__file__),'prior_protocol_sha256':sha(protocol_path),'mano_asset_sha256':sha(ROOT/'hug/assets/mano_models/models/MANO_RIGHT.pkl'),'native_case_count':len(rows),'parents':parents.tolist(),'source_tip_vertex_indices':tipvertex.tolist(),'source_tip_joint_indices':tipjoint.tolist(),'cases':rows,
 'max_scalar_FK_position_error_m':max(r['independent_scalar_FK_position_max_error_m'] for r in rows),'max_scalar_FK_rotation_entry_error':max(r['independent_scalar_FK_rotation_max_entry_error'] for r in rows),'max_rigid_tip_vs_skinned_tip_error_m':max(r['rigid_tip_vs_native_skinned_tip_max_m'] for r in rows),'mean_rigid_tip_vs_skinned_tip_error_m':float(np.mean([r['rigid_tip_vs_native_skinned_tip_mean_m'] for r in rows])),
 'source_shape_count':len(set(r['shape_sha256'] for r in rows)),'new_proposals':0,'physics_steps':0,'simulator_import_blocked':True,'new_eval_data':False}
out=HERE/'source_fk_results.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
