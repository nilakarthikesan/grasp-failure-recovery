"""Source-only tensor decomposition diagnostic; no models, FK, geometry or physics."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,mujoco,json,hashlib
p=Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/fresh_object_scene_eval_2026-10-11/assets/milk_box/sim_assets/object.xml')
n=ET.parse(p).getroot().find('.//inertial');xx,yy,zz,xy,xz,yz=np.fromstring(n.get('fullinertia'),sep=' ');I=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
e=np.empty(3);v=np.empty(9);q=np.empty(4);steps=int(mujoco.mju_eig3(e,v,q,I.ravel()));r=np.empty(9);mujoco.mju_quat2Mat(r,q);Q=r.reshape(3,3);reconstructed=Q@np.diag(e)@Q.T
ne,nv=np.linalg.eigh(I);N=nv@np.diag(ne)@nv.T
old=json.loads(Path('/private/tmp/hug_current_scene_bridge_2026-10-15_v1/cases/case_16.json').read_text())['current_object_checks']['inertia_tensor_max_error_kg_m2']
x={'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_inertia_tensor':I.tolist(),'mujoco_version':mujoco.__version__,'numpy_version':np.__version__,'mju_eig3_iterations':steps,'mju_eig3_eigenvalues':e.tolist(),'mju_eig3_quaternion':q.tolist(),'mju_eig3_reconstructed_tensor':reconstructed.tolist(),'mju_eig3_max_error_kg_m2':float(np.max(np.abs(reconstructed-I))),'numpy_eigh_max_error_kg_m2':float(np.max(np.abs(N-I))),'recorded_assembled_error_kg_m2':old,'eig3_error_matches_assembled_error_abs_difference':abs(float(np.max(np.abs(reconstructed-I)))-old),'compiled_models':0,'forwards':0,'physics_steps':0,'new_eval_data':False}
Path(__file__).with_name('diagnosis.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
