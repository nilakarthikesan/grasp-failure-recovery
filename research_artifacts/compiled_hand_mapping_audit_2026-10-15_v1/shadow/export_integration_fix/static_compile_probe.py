from pathlib import Path
import sys,importlib.util,json,hashlib
import mujoco,numpy as np
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');R=Path('/private/tmp/compiled_hand_audit_20261015_v1/shadow/export_integration_fix');sys.path.insert(0,str(W/'src'));sys.path.insert(0,str(W/'scripts'))
def forbidden(*a,**kw):raise AssertionError('No integration, output export, render or legacy XML serialization permitted')
for name in ['mj_step','mj_step1','mj_step2','mj_forward','mj_saveLastXML','mj_saveModel']:setattr(mujoco,name,forbidden)
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
import grasp_failure_prediction.evaluation.runner as current
assets=W/'third_party/dex-urdf/robots/hands';proto=load_protocol_registry().resolve('fixed_grasp_lift_v2');before=current.ActuatedShadowRunner(proto,assets)
def load(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
new=load('grasp_failure_prediction.evaluation.runner',R/'after/src/grasp_failure_prediction/evaluation/runner.py');after=new.ActuatedShadowRunner(proto,assets);recompiled=after._scene_spec.copy().compile()
fields=['body_pos','body_quat','body_ipos','body_iquat','body_inertia','body_mass','jnt_pos','jnt_axis','jnt_range','jnt_actfrcrange','geom_pos','geom_quat','geom_size','geom_friction','geom_margin','geom_gap','geom_solref','geom_solimp','geom_contype','geom_conaffinity','mesh_pos','mesh_quat','mesh_vert','mesh_face','actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_ctrlrange','eq_data','eq_solref','qpos0']
for key in fields:
 np.testing.assert_array_equal(getattr(before.model,key),getattr(after.model,key));np.testing.assert_array_equal(getattr(after.model,key),getattr(recompiled,key))
legacy=load('actuated_shadow_pilot_export_fix',R/'after/scripts/actuated_shadow_pilot.py');pilot=legacy.ActuatedShadowRunner(proto,assets);pilot_recompiled=pilot._scene_spec.copy().compile()
for key in fields:np.testing.assert_array_equal(getattr(pilot.model,key),getattr(pilot_recompiled,key))
assert after.data.time==pilot.data.time==0 and after._trace==pilot._trace==[]
report={'status':'PASS','scope':'Static constructor and spec-copy compilation only; no exports or rendering executed.','new_physics_steps':0,'new_forward_calls':0,'new_model_exports':0,'new_evaluation_data':False,'current_vs_retain_spec_arrays_bitwise_equal':fields,'retained_actuated_spec_compile_arrays_bitwise_equal':fields,'legacy_pilot_retained_spec_arrays_bitwise_equal':fields,'actuators':after.model.nu,'welds':after.model.neq,'new_pipeline_xml_serializer_calls':0}
p=R/'static_compile_probe_report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(hashlib.sha256(p.read_bytes()).hexdigest())
