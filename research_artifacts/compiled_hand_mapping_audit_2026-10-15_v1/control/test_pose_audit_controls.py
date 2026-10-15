"""Manufactured detector controls; modifies only in-memory scratch model copies."""
import importlib.util,json
from pathlib import Path
import numpy as np
P=Path(__file__).with_name('pose_audit.py')
spec=importlib.util.spec_from_file_location('static_oracle',P);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
tree=a.Tree(a.TARGET);case=a.controls(tree)[0];modelpath=P.parent/'base_before.mjb'
def fresh():return a.model_setup(modelpath,tree)
base=a.case_compare(case,fresh(),tree)
assert base['maximum_link_error']['translation_m']<1e-12
assert base['maximum_link_error']['rotation_rad']<1e-12
# All24 names are the contract; dict insertion order must not become joint order.
shuffled=dict(case);shuffled['target_joint_targets_rad']=dict(reversed(list(case['target_joint_targets_rad'].items())))
reordered=a.case_compare(shuffled,fresh(),tree)
assert reordered['link_errors']==base['link_errors']
# Deliberate collider offset must be detected independently of palm/root fitting.
x=fresh();gid=next(v['gid'] for v in x['mapping'] if v['col']['key']=='palm/collision/0');x['m'].geom_pos[gid,0]+=0.0001
bad_geom=a.case_compare(case,x,tree);detected_geom=bad_geom['collider_errors']['palm/collision/0']['translation_m'];assert abs(detected_geom-0.0001)<1e-12
# Deliberate non-wrist axis corruption cannot be hidden by root compensation.
x=fresh();jid=x['named']['FFJ4'];x['m'].jnt_axis[jid]=[1,0,0]
bad_axis=a.case_compare(case,x,tree);detected_axis=bad_axis['link_errors']['ffdistal']['translation_m'];assert detected_axis>0.001
# Declarative zero-time and entry-point policy remain enforced.
assert not a.BLOCK_ATTEMPTS
try:a.mujoco.mj_step(None,None)
except RuntimeError:pass
assert a.BLOCK_ATTEMPTS==['mj_step']
r={'schema_version':'compiled_static_oracle_manufactured_detector_controls_v1','status':'PASS','script':a.ref(Path(__file__)),'oracle':a.ref(P),'model':a.ref(modelpath),'controls':{'nonzero_wrist_rotated_root_roundtrip':True,'shuffled_named_joint_input_identity':True,'100um_palm_geom_corruption_detected_m':detected_geom,'FFJ4_axis_corruption_detected_distal_translation_m':detected_axis,'physics_entrypoint_refused':True},'all_models_scratch_in_memory_only':True,'asset_edits':0,'physics_steps':0,'solver_calls':0,'inference_calls':0,'fit_calls':0,'evaluation_episodes':0,'training_calls':0}
Path(__file__).with_name('pose_audit_manufactured_controls_v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
