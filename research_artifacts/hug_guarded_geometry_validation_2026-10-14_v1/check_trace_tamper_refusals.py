"""Five copied authentic-record corruptions; no geometry/FK imports or calls."""
from pathlib import Path
from copy import deepcopy
import hashlib,json
import xml.etree.ElementTree as ET
from selection_audit import verify_recorded_selection

OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((OUT/'protocol.json').read_text())
candidate_path=Path(protocol['candidate_path']);candidate_sha256=sha(candidate_path);candidate=json.loads(candidate_path.read_text())
fit=json.loads((candidate_path.parent/'protocol.json').read_text())
assert sha(candidate_path.parent/'protocol.json')==protocol['candidate_protocol_expected_sha256']
urdf=next(k for k in fit['source_sha256'] if k.endswith('shadow_hand_right.urdf'))
assert sha(urdf)==fit['source_sha256'][urdf]
ranges={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))
    for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
original=candidate['cases'][0]
authentic=verify_recorded_selection(original,ranges,protocol)
assert authentic['recorded_guarded_pattern_selection_verified']
def wrong_q_metadata(target):
    target['keyed_direction_evidence']['callbacks'][0]['q_float64_sha256']='0'*64
def changed_native_weights(target):
    ref=next(v for v in target['direction_reference_by_part'].values() if v['available'])
    ref['weights'][0]*=1.5
def fake_accepted_guard(target):
    guard=next(g for g in target['keyed_direction_evidence']['guard_assessments'] if not g['accepted'])
    guard['accepted']=True;guard['refusal_reasons']=[]
def altered_budget(target):
    target['solver_audit']['config']['maximum_FK_evaluations']+=1
def altered_terminal_provenance(target):
    target['terminal_direction_guard']['metadata_callback_sequence']=1
results=[]
for fn in (wrong_q_metadata,changed_native_weights,fake_accepted_guard,altered_budget,altered_terminal_provenance):
    copied=deepcopy(original);fn(copied)
    try:verify_recorded_selection(copied,ranges,protocol)
    except (AssertionError,ValueError,KeyError,TypeError):refused=True
    else:refused=False
    assert refused,fn.__name__
    results.append({'control':fn.__name__,'copied_authentic_record_refused':refused})
assert sha(candidate_path)==candidate_sha256
report={'schema':'hug_guarded_geometry_trace_tamper_refusals_v1','candidate_sha256':sha(candidate_path),
    'validation_protocol_sha256':sha(OUT/'protocol.json'),'selection_audit_sha256':sha(OUT/'selection_audit.py'),
    'script_sha256':sha(__file__),'source_inventory_index':original['source_inventory_index'],
    'authentic_record_verified':True,'control_count':len(results),'refusals':results,'geometry_or_FK_replays':0,
    'physics_steps':0,'new_eval_data':False,'not_scientific_mapping_acceptance':True}
with (OUT/'trace_tamper_refusals.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'authentic_record_verified':True,'control_count':len(results),'all_refused':True}))
