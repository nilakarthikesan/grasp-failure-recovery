"""Freeze one-case kinematic controls before any actual saved-case call."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=Path('/private/tmp/hug_feasibility_restoration_2026-10-15_v1')
POOL=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/pooled_selection/candidate_targets.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
old=read(BASE/'protocol.json');pool=next(c for c in read(POOL)['cases'] if c['source_inventory_index']==20)
inputs=dict(old['input_sha256'])
for file in [POOL,BASE/'protocol.json']+[BASE/n for n in ['constraint_engine.py','constraint_core.py','free_palm_kinematics.py','exact_direction.py','geometry_helpers.py','surface_component_guard.py']]:inputs[str(file)]=sha(file)
states=[{'label':'original_zero_rotation','state_scaled':[0.]*30},
        {'label':'pooled_zero_rotation','state_scaled':pool['selected_state_scaled']+[0.]*3},
        {'label':'pooled_coupled_world_rotation','state_scaled':pool['selected_state_scaled']+[6.,-4.,5.]}]
out={'schema':'hug_se3_mapping_kinematic_preflight_v1','authorized_scope':'one_saved_case_static_FK_and_bounded_Jacobian_only',
     'authorization':'Parent explicitly authorized manufactured/zero-pose controls and bounded FD Jacobian checks on existing case20; no fit.',
     'source_index':20,'native_sha256':pool['native_sha256'],'pooled_candidate_path':str(POOL),
     'input_sha256':inputs,'implementation_sha256':{str(HERE/n):sha(HERE/n) for n in ['se3_math.py','se3_case.py','test_se3_math.py','check_case20_kinematics.py','prepare_preflight.py']},
     'jacobian_states':states,'fd_step_scaled':1e-4,'jacobian_tolerance_m_per_scaled_state':1e-8,
     'classification':'kinematic numerical test; no mapping accuracy threshold change',
     'rotation_component_limit_degrees':15,'rotation_vector_norm_max_degrees':float(np.sqrt(3.)*15),
     'rotation_scale_rad':.025,'joint_scale_rad':.025,'translation_scale_m':.005,
     'root_composition':'Exp(world rotation vector) left-multiplied onto original floating-root rotation; root translation compensates actual joint-dependent palm offset.',
     'zero_rotation':'exact call to historical27D staging with original quaternion bytes',
     'new_fit':False,'new_eval_data':False,'new_inference':False,'surface_queries':0,'physics_steps':0,'numba_threads':2}
dest=HERE/'preflight_protocol.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
print(sha(dest))
