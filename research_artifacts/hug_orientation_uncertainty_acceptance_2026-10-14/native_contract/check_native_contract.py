"""Saved-parameter MANO reconstruction only; no grasp generator or simulator."""
from pathlib import Path
import json,hashlib,pickle,sys,types,warnings
import numpy as np
import torch
BASE=Path(__file__).resolve().parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
HUG=Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((BASE/'protocol.json').read_text());ph=sha(BASE/'protocol.json')
assert sha(__file__)==p['script_sha256']
for path,h in p['source_sha256'].items():assert sha(path)==h,path
# Block simulator imports; expose only the pinned upstream MANO wrapper package.
sys.modules['mujoco']=None
package=types.ModuleType('hug');package.__path__=[str(HUG/'src')];sys.modules['hug']=package
from hug.models.mano import MANO
from hug.utils.data_keys import MANO_MODELS_FOLDER
assert sha(MANO_MODELS_FOLDER/'models/MANO_RIGHT.pkl')==p['mano_asset_sha256']
torch.set_num_threads(2)
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 mano=MANO(side='right',use_pca=False,flat_hand_mean=True).float().eval()
faces=mano.mano_layer.get_mano_closed_faces().detach().numpy()
records=[]
for item in p['cases']:
 path=Path(item['native_path']);assert sha(path)==item['native_sha256'];s=pickle.loads(path.read_bytes())
 params=np.concatenate([np.asarray(s['t']).reshape(3),np.asarray(s['R_6d']).reshape(6),np.asarray(s['pose_6d']).reshape(90)]).astype(np.float32)
 with torch.no_grad():rebuilt=mano(torch.from_numpy(params[None]),betas=torch.from_numpy(np.asarray(s['shape'],dtype=np.float32).reshape(1,10)))
 vertices=rebuilt['vertices'][0].numpy()+params[:3];joints=rebuilt['landmarks_3d'][0].numpy()+params[:3]
 ve=float(np.linalg.norm(vertices-np.asarray(s['mesh_vertices']),axis=1).max());je=float(np.linalg.norm(joints-np.asarray(s['landmarks_3d']),axis=1).max());re=float(np.max(np.abs(rebuilt['R_3x3'][0].numpy()-np.asarray(s['T_camera_wrist'])[:3,:3])))
 face_equal=bool(np.array_equal(faces,np.asarray(s['mesh_faces'])))
 assert vertices.shape==(778,3) and joints.shape==(21,3) and faces.shape==(1552,3)
 assert ve<=p['vertex_landmark_tolerance_m'] and je<=p['vertex_landmark_tolerance_m'] and re<=p['rotation_element_tolerance'] and face_equal
 records.append({**item,'max_vertex_error_m':ve,'max_landmark_error_m':je,'max_rotation_element_error':re,'closed_face_indices_equal':face_equal,'native_vertex_count':778,'native_landmark_count':21,'closed_face_count':1552,'passed':True})
 assert sha(path)==item['native_sha256']
assert sha(BASE/'protocol.json')==ph
for path,h in p['source_sha256'].items():assert sha(path)==h,path
for path,h in json.loads(Path(p['protected_hash_file']).read_text()).items():assert sha(path)==h,path
r={'schema':'hug_saved_native_contract_32_v1','protocol_sha256':ph,'script_sha256':sha(__file__),'cases':records,'distinct_native_count':len(records),'max_vertex_error_m':max(c['max_vertex_error_m'] for c in records),'max_landmark_error_m':max(c['max_landmark_error_m'] for c in records),'max_rotation_element_error':max(c['max_rotation_element_error'] for c in records),'all_face_indices_equal':all(c['closed_face_indices_equal'] for c in records),'protected_hashes_unchanged':True,'new_inference':False,'grasp_generator_loaded':False,'grasp_generator_sampled':False,'physics_steps':0,'simulator_import_blocked':True,'new_eval_data':False,'training':False,'interpretation':'Parametric MANO FK reconstructs already saved parameters; validates export decoding, not new proposals, cross-hand surface fidelity or outcomes. Existing original float32 reconstruction tolerances reused.'}
(BASE/'native_contract.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))
