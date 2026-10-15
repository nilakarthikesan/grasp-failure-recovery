"""Independent repeat/interleave/idempotence checks of actual edited importer."""
from pathlib import Path
import hashlib,json
import mujoco,numpy as np
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');O=Path('/private/tmp/compiled_hand_audit_20261015_v1/mano')
for name in ['mj_step','mj_step1','mj_step2','mj_saveLastXML']:
 setattr(mujoco,name,lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Forbidden integration/serializer')))
from grasp_failure_prediction.evaluation.pose_validation import load_shadow_model
from grasp_failure_prediction.evaluation.runner import _build_tabletop_spec,add_shadow_actuation
root=W/'third_party/dex-urdf/robots/hands'
source=load_shadow_model(root,floating_root=True);spec=_build_tabletop_spec(root);base=spec.compile()
fields=['body_pos','body_quat','body_ipos','body_iquat','body_mass','body_inertia','jnt_pos','jnt_axis','jnt_range','jnt_actfrcrange','geom_pos','geom_quat','geom_size','geom_type','geom_contype','geom_conaffinity','geom_friction','geom_margin','geom_gap','geom_solref','geom_solimp','mesh_pos','mesh_quat','mesh_scale','mesh_vert','mesh_face']
saved={k:getattr(base,k).copy() for k in fields}
rows=[]
for iteration in range(3):
 unrelated=mujoco.MjModel.from_xml_string('<mujoco model="unrelated"><worldbody><body name="UNRELATED"><freejoint/><geom type="sphere" size=".05" mass="2"/></body></worldbody></mujoco>')
 act=add_shadow_actuation(spec)
 assert act.nu==24 and act.neq==1 and act.nmocap==1
 assert mujoco.mj_name2id(act,mujoco.mjtObj.mjOBJ_BODY,'UNRELATED')<0
 # Body and geom/joint IDs of original scene are retained; only arm_target appended.
 for k in fields:
  current=getattr(act,k)
  assert np.array_equal(current[:len(saved[k])],saved[k]),(iteration,k)
 # Caller retains no actuation or arm; repeated compiles must not drop tip masses.
 repeat=spec.compile();assert repeat.nu==0 and repeat.neq==0 and repeat.nmocap==0
 for k in fields:assert np.array_equal(getattr(repeat,k),saved[k]),(iteration,k,'caller_changed')
 rows.append({'iteration':iteration,'all_compiled_scene_field_prefixes_exact':True,'caller_spec_after_actuation_unchanged':True,'unrelated_model_absent':True,'actuator_count':act.nu})
body=[]
for name in ['ffdistal','mfdistal','rfdistal','lfdistal','thdistal']:
 sid=source.body(name).id;bid=base.body(name).id
 assert np.array_equal(source.body_mass[sid],base.body_mass[bid])
 assert np.array_equal(source.body_inertia[sid],base.body_inertia[bid])
 body.append({'name':name,'source_fused_mass_kg':float(source.body_mass[sid]),'scene_fused_mass_kg':float(base.body_mass[bid]),'inertia_exact':True})
inputs=[W/'src/grasp_failure_prediction/evaluation/pose_validation.py',W/'src/grasp_failure_prediction/evaluation/runner.py',Path(__file__)]
r={'schema_version':'independent_live_spec_implementation_review_v1','bindings':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'runtime_mujoco':mujoco.__version__,'status':'PASS_NO_BLOCKING_ISSUE_IN_REVIEWED_SCOPE','repeat_interleave_cases':rows,'fixed_tip_mass_preservation':body,'blocked_apis':['mj_step','mj_step1','mj_step2','mj_saveLastXML'],'physics_steps':0,'source_asset_bytes_changed':False,'acceptance_thresholds_changed':False,'controller_parameter_scope':'Static existing gain/weld/options declarations unchanged; no control performance or dynamics conclusion','review_notes':['Aliases are unique per resolved path and preserve filebytes and existing scale; identical path stays shared.','Compiled fused mass/inertial fields are explicitly persisted before repeated spec compilation. Repeated compile/interleaved unrelated parser calls do not lose masses or mutate caller spec.','actuation takes live spec and copies it; all current source callers updated. Public old MjModel argument now fails clearly rather than silently roundtripping.','Object attachment parses only source object XML and aliases itsassets; hand spec never serialized. Parent object fixture and compiled-cohort checks independently cover object transforms.']}
p=O/'live_spec_implementation_review_v1.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':r['status'],'iterations':len(rows)}))
