"""Explicit source-closure revision after an authorized shared profile edit."""
from pathlib import Path
import ast,hashlib,json
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
old=read(HERE/'preflight_protocol.json');new=json.loads(json.dumps(old))
path=next(p for p in old['input_sha256'] if p.endswith('/evaluation/retargeting.py'))
original=Path('/private/tmp/hug_retargeter_state_audit_2026-10-15_v1/retargeting.py.before')
assert sha(original)==old['input_sha256'][path]
oldnodes={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
newnodes={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
unchanged=['RetargetingError','_validate_rigid_transform','camera_landmarks_to_wrist','reference_vectors','reorder_for_mujoco','default_dex_urdf_root']
assert all(oldnodes[n]==newnodes[n] for n in unchanged)
new['input_sha256'][path]=sha(path)
for f,h in list(new['input_sha256'].items()):assert sha(f)==h,f
new['implementation_sha256']={f:sha(f) for f in old['implementation_sha256']}
new['implementation_sha256'][str(Path(__file__))]=sha(Path(__file__))
new['schema']='hug_se3_mapping_kinematic_preflight_v2'
new['preflight_revision']={'previous_protocol_sha256':sha(HERE/'preflight_protocol.json'),'prior_attempt':'Stopped at source hash before scene construction; zero FK/surface calls.',
    'source_path':path,'old_sha256':old['input_sha256'][path],'new_sha256':sha(path),'old_source_copy':str(original),
    'reason':'Authorized optional static retargeting profile repair happened concurrently. This FK-only Case control does not instantiate or call either retargeting profile.',
    'unchanged_helper_AST_names':unchanged,'control_script_change':'Added explicit protocol/output arguments to preserve previous protocol/log; same scientific control states and tolerances.'}
out=HERE/'preflight_protocol_v2.json';assert not out.exists();out.write_text(json.dumps(new,indent=2)+'\n');print(sha(out))
