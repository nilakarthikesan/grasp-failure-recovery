"""Reconstruct an existing MANO export; never load/sample the HUG generator."""
from pathlib import Path
import hashlib, json, os, pickle, sys, types
import numpy as np
import torch

HUG=Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug')
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
NATIVE=ROOT/'runs/hug_grasp_eval_2026-10-13_fresh_cube/output/proposals/hug_cube/sample_0000/native'
OUT=Path('/private/tmp/grasp-hug-confirmation-2026-10-14')
native_path=NATIVE/'proposal.pkl'
expected='573cb003458d9d7cfe85b5b4a7877e414257b3e67d5abda04a09bd6d687fa0c2'
assert hashlib.sha256(native_path.read_bytes()).hexdigest()==expected
# Expose the already verified upstream source package; no model/asset substitution.
package=types.ModuleType('hug'); package.__path__=[str(HUG/'src')]; sys.modules['hug']=package
from hug.models.mano import MANO, project_3d_to_2d
from hug.utils.data_keys import MANO_MODELS_FOLDER
torch.set_num_threads(4)
saved=pickle.loads(native_path.read_bytes())
params=np.concatenate([np.asarray(saved['t']).reshape(3),np.asarray(saved['R_6d']).reshape(6),np.asarray(saved['pose_6d']).reshape(90)]).astype(np.float32)
mano=MANO(side='right',use_pca=False,flat_hand_mean=True).float().eval()
with torch.no_grad():
    rebuilt=mano(torch.from_numpy(params[None]),betas=torch.from_numpy(np.asarray(saved['shape'],dtype=np.float32).reshape(1,10)))
    vertices=rebuilt['vertices'][0].numpy()+params[:3]
    joints=rebuilt['landmarks_3d'][0].numpy()+params[:3]
vertex_error=np.linalg.norm(vertices-np.asarray(saved['mesh_vertices']),axis=1)
joint_error=np.linalg.norm(joints-np.asarray(saved['landmarks_3d']),axis=1)
rotation_error=float(np.max(np.abs(rebuilt['R_3x3'][0].numpy()-np.asarray(saved['T_camera_wrist'])[:3,:3])))
uncapped_faces=mano.mano_layer.th_faces.detach().numpy()
# HUG exports manotorch's closed-wrist table (grasp_model.py:102), not th_faces.
faces=mano.mano_layer.get_mano_closed_faces().detach().numpy()
K=np.load(NATIVE/'intrinsics.npy')
projection=project_3d_to_2d(np.asarray(saved['landmarks_3d'],dtype=np.float64),np.asarray(K,dtype=np.float64))
projection_error=np.linalg.norm(projection-np.asarray(saved['landmarks_2d']),axis=1)
report={
 'date':'2026-10-14','source_code_commit':'ab39670',
 'classification':'parametric_MANO_FK_reconstruction_of_saved_pose_not_new_grasp_model_inference',
 'native_proposal_sha256':expected,
 'mano_asset_sha256':hashlib.sha256((MANO_MODELS_FOLDER/'models/MANO_RIGHT.pkl').read_bytes()).hexdigest(),
 'upstream_mano_source_sha256':hashlib.sha256((HUG/'src/models/mano.py').read_bytes()).hexdigest(),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pythonpath':os.environ.get('PYTHONPATH')},
 'configuration':{'side':'right','use_pca':False,'flat_hand_mean':True,'wrist_center_index':0,'device':'cpu','dtype':'float32'},
 'saved_vertex_count':len(vertices),'saved_landmark_count':len(joints),
 'max_reconstruction_vertex_error_m':float(vertex_error.max()),
 'max_reconstruction_landmark_error_m':float(joint_error.max()),
 'max_wrist_rotation_element_error':rotation_error,
 'face_indices_equal_saved_export':bool(np.array_equal(faces,np.asarray(saved['mesh_faces']))),
 'uncapped_MANO_face_count':len(uncapped_faces),'closed_export_face_count':len(faces),
 'wrist_cap_additional_triangle_count':len(faces)-len(uncapped_faces),
 'uncapped_faces_equal_saved_prefix':bool(np.array_equal(uncapped_faces,np.asarray(saved['mesh_faces'])[:len(uncapped_faces)])),
 'face_table_source':'Official HUG grasp_model.py:102 uses get_mano_closed_faces(); extra14 triangles close wrist',
 'max_projection_error_px':float(projection_error.max()),
 'reconstruction_passed':bool(vertex_error.max()<5e-6 and joint_error.max()<5e-6 and rotation_error<1e-6 and np.array_equal(faces,np.asarray(saved['mesh_faces']))),
 'numerical_reconstruction_tolerance_m':5e-6,
 'numerical_tolerance_scope':'float arithmetic consistency only, not a grasp/contact acceptance tolerance',
 'HUG_generator_loaded':False,'HUG_generator_sampled':False,'physics_steps':0,'new_grasps':False,'licensed_asset_copied':False,
 'scope_limit':'Reproduces native representation; does not validate MANO to Shadow contact preservation.'
}
OUT.mkdir(exist_ok=True)
(OUT/'hug_native_reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert report['reconstruction_passed']
