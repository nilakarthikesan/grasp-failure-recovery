from pathlib import Path
import sys,importlib.util,json,hashlib
import numpy as np,mujoco
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');R=Path('/private/tmp/compiled_hand_audit_20261015_v1/shadow/precise_spec_fix');sys.path.insert(0,str(W/'src'))
for p in [R/'runner.py',R/'pose_validation.py']:compile(p.read_text(),str(p),'exec')
def forbidden(*a,**k):raise AssertionError('Forbidden step or lossy XML serializer called')
for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,forbidden)
import grasp_failure_prediction.evaluation.runner as old
from grasp_failure_prediction.evaluation.object_assets import load_object_geometry
root=W/'third_party/dex-urdf/robots/hands'
oldbase=old.build_tabletop_model(root);oldact=old.add_shadow_actuation(oldbase)
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
newpose=load('grasp_failure_prediction.evaluation.pose_validation',R/'pose_validation.py')
new=load('grasp_failure_prediction.evaluation.runner_precise',R/'runner.py')
mujoco.mj_saveLastXML=forbidden
spec=new._build_tabletop_spec(root);base=spec.compile();act=new.add_shadow_actuation(spec);direct=newpose.load_shadow_model(root,floating_root=True)
fields=['body_pos','body_quat','body_ipos','body_iquat','body_inertia','body_mass','jnt_pos','jnt_axis','jnt_range','jnt_actfrcrange','geom_pos','geom_quat','geom_size','geom_friction','geom_margin','geom_gap','geom_solref','geom_solimp','geom_contype','geom_conaffinity','mesh_pos','mesh_quat','mesh_vert','mesh_face']
res={}
for key in fields:
 # body/joint counts append, world geom table inserted so compare per-body geoms.
 vals=getattr(direct,key);actual=getattr(base,key);actvals=getattr(act,key)
 if key.startswith('geom_'):
  actual=actual[1:1+direct.ngeom];actvals=actvals[1:1+direct.ngeom]
 else:actual=actual[:len(vals)];actvals=actvals[:len(vals)]
 res[key]={'base_equal':bool(np.array_equal(vals,actual)),'actuated_equal':bool(np.array_equal(vals,actvals)),'base_max_absolute':float(np.max(np.abs(vals-actual))) if vals.size else 0,'actuated_max_absolute':float(np.max(np.abs(vals-actvals))) if vals.size else 0}
 assert res[key]['base_equal'] and res[key]['actuated_equal'],(key,res[key])
control=['actuator_gainprm','actuator_biasprm','actuator_forcerange','actuator_gear','actuator_dyntype','actuator_gaintype','actuator_biastype','actuator_trntype','actuator_trnid','actuator_ctrllimited','actuator_forcelimited','eq_type','eq_data','eq_solref','eq_solimp','eq_objtype','eq_obj1id','eq_obj2id']
for key in control:assert np.array_equal(getattr(oldact,key),getattr(act,key)),key
assert np.array_equal(act.actuator_ctrlrange,act.jnt_range[1:25])
for name in ['cone','noslip_iterations','integrator','timestep']:assert getattr(oldact.opt,name)==getattr(act.opt,name)
assert np.array_equal(base.body_pos,spec.compile().body_pos)
try:new.add_shadow_actuation(base)
except TypeError:pass
else:raise AssertionError('compiled model must refuse')
# Test native-framed asymmetric mesh import with nontrivial pose and mesh transform.
obj=R/'object_probe';obj.mkdir(exist_ok=True)
(obj/'shape.obj').write_text('''v 0 0 0\nv 0.021 0 0\nv 0 0.033 0\nv 0 0 0.047\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n''')
xml='''<mujoco model="static_precision_object"><compiler angle="radian" meshdir="." strippath="false"/><asset><mesh name="mesh" file="shape.obj" scale="1.123456789012345 0.987654321098765 1.034567890123456" refpos="0.001234567890123 0.002345678901234 0.003456789012345" refquat="0.943456789012345 0.123456789012345 0.234567890123456 0.176543210987654"/></asset><worldbody><body name="native_object" pos="0.012345678901234 -0.023456789012345 0.034567890123456" quat="0.943456789012345 0.123456789012345 0.234567890123456 0.176543210987654"><inertial pos="0.003456789012345 0.004567890123456 0.005678901234567" quat="0.995 0.05 0.06 0.04" mass="0.183456789012345" diaginertia="0.000123456789012 0.000234567890123 0.000323456789012"/><freejoint name="native_free"/><geom type="mesh" mesh="mesh" pos="0.002345678901234 0.004567890123456 0.006789012345678" quat="0.96 0.12 0.15 0.18" friction="0.876543210987654 0.005 0.0001"/></body></worldbody></mujoco>'''
(obj/'object.xml').write_text(xml);geo=load_object_geometry(obj/'object.xml');native=mujoco.MjModel.from_xml_path(str(obj/'object.xml'));ospec=new._build_tabletop_spec(root,object_geometry=geo);obase=ospec.compile();oact=new.add_shadow_actuation(ospec)
objchecks={}
for modelname,model in [('base',obase),('actuated',oact)]:
 b=model.body('object').id;g=model.geom('object_geom').id
 for key in ['body_pos','body_quat','body_ipos','body_iquat','body_mass','body_inertia']:
  a=getattr(native,key)[1];z=getattr(model,key)[b];error=float(np.max(np.abs(a-z)));assert error<1e-14,(modelname,key,error);objchecks[modelname+':'+key]=error
 for key in ['geom_pos','geom_quat','geom_size','geom_friction']:
  a=getattr(native,key)[0];z=getattr(model,key)[g];error=float(np.max(np.abs(a-z)));assert error<1e-14,(modelname,key,error);objchecks[modelname+':'+key]=error
# No step/reset/rollout is called. Compilations only.
report={'status':'PASS','mujoco_version':mujoco.__version__,'physics_steps':0,'model_inference':0,'serialized_hand_roundtrips':0,'direct_hand_arrays':res,'controller_parameter_arrays_bitwise_unchanged':control,'actuator_ctrlrange_matches_original_full_precision_joint_limits':True,'input_spec_unchanged':True,'compiled_model_argument_refused':True,'asymmetric_native_frame_object_checks':objchecks,'implementation_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'pose_validation.py',R/'runner.py']}}
(R/'static_probe_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','direct_fields':len(fields),'control_fields':len(control),'object_max_error':max(objchecks.values()),'sha256':hashlib.sha256((R/'static_probe_report.json').read_bytes()).hexdigest()},indent=2))
# Existing compound fixture: retain 1 visual + 2 collision geoms and native frames.
tests=load('shadow_object_fixture',(W/'tests/test_object_assets.py'))
compound=R/'compound_probe';compound.mkdir(exist_ok=True)
if (compound/'asset').exists():
 import shutil;shutil.rmtree(compound/'asset')
fixture=tests.asset.__wrapped__(compound);geom=load_object_geometry(fixture);native=mujoco.MjModel.from_xml_path(str(fixture))
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
compound_result={}
for kind,pid in [(new.AdroitShadowRunner,'fixed_grasp_lift_v1'),(new.ActuatedShadowRunner,'fixed_grasp_lift_v2')]:
 runner=kind(load_protocol_registry().resolve(pid),root,object_geometry=geom)
 assert len(runner.object_geom_ids)==3 and len(runner.object_collision_geom_ids)==2
 b=runner._object_body_id;ids=sorted(runner.object_geom_ids)
 for field in ['body_mass','body_inertia','body_ipos','body_iquat']:
  np.testing.assert_array_equal(getattr(runner.model,field)[b],getattr(native,field)[1])
 for field in ['geom_pos','geom_quat','geom_friction','geom_contype','geom_conaffinity','geom_group']:
  np.testing.assert_array_equal(getattr(runner.model,field)[ids],getattr(native,field))
 assert runner.data.time==0 and runner._trace==[]
 compound_result[kind.__name__]={'geoms':3,'collision_geoms':2,'frame_and_inertia_arrays_bitwise_equal':True,'steps':0}
report['compound_object_original_fixture']=compound_result
(R/'static_probe_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'compound_status':'PASS','sha256':hashlib.sha256((R/'static_probe_report.json').read_bytes()).hexdigest()},indent=2))
