"""Read-only static publication/provenance validation; no runtime imports."""
from pathlib import Path
import argparse,hashlib,json,gzip,ast
base=Path(__file__).resolve().parent
repo=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def manifest(p):
 m=read(p/'manifest.json');files=m['files']
 for name,v in files.items():assert sha(p/name)==v['sha256'],(p,name)
 return len(files)
def inputs(items):
 for name,h in items.items():assert sha(name)==h,name
 return len(items)
parser=argparse.ArgumentParser();parser.add_argument('--skip-parent-manifest',action='store_true');args=parser.parse_args()
counts={name:manifest(base/name) for name in ('sampling','orientation','native_contract')}
if not args.skip_parent_manifest:counts['parent']=manifest(base)
for p in base.rglob('*.py'):ast.parse(p.read_bytes())
s=read(base/'sampling/sampling_uncertainty.json');sp=read(base/'sampling/protocol.json')
oraw=gzip.decompress((base/'orientation/orientation_report_full.json.gz').read_bytes());o=json.loads(oraw);os=read(base/'orientation/summary.json')
n=read(base/'native_contract/native_contract.json');np=read(base/'native_contract/protocol.json')
for report,folder,script in ((s,'sampling','check_sampling.py'),(o,'orientation','check_orientation.py'),(n,'native_contract','check_native_contract.py')):
 assert report['protocol_sha256']==sha(base/folder/'protocol.json')
 assert report['script_sha256']==sha(base/folder/script)
 assert report['physics_steps']==0 and report['new_eval_data'] is False and report['new_inference'] is False and report['protected_hashes_unchanged'] is True
assert hashlib.sha256(oraw).hexdigest()==os['full_uncompressed_report_sha256']
assert len(s['cases'])==len(o['cases'])==6 and n['distinct_native_count']==32
assert n['max_vertex_error_m']==n['max_landmark_error_m']==n['max_rotation_element_error']==0 and n['all_face_indices_equal']
assert s['guards']['actual']==o['guards']['actual']==0
inputs(np['source_sha256']);native_source_count=0
for c in np['cases']:assert sha(c['native_path'])==c['native_sha256'];native_source_count+=1
assert sha('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/assets/mano_models/models/MANO_RIGHT.pkl')==np['mano_asset_sha256']
sourcechecks=0
for c in s['cases']:sourcechecks+=inputs(c['input_sha256'])
om={c['native_sha256']:c for c in o['cases']};bindings=0
for c in s['cases']:
 oc=om[c['native_sha256']];sourcechecks+=inputs(oc['input_sha256']);assert oc['time_s']==0 and oc['qpos_and_object_pose_unchanged']
 for density,res in c['resolutions'].items():
  b=oc['resolutions'][density]['sampling_shared_binding'];assert b['protocol_sha256']==s['protocol_sha256'] and b['query_library_sha256']==sp['previous_query_library_sha256']
  assert b['native_sha256']==c['native_sha256'] and b['original_cache_sha256']==res['cache_sha256'] and b['no_new_eval_data'];bindings+=1
assert bindings==9
rule_hash=sha(base/'acceptance/acceptance_rule.json');assert rule_hash=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4'
w=read(base/'sampling/strict_preservation_witnesses.json');assert w['acceptance_rule_sha256']==rule_hash and w['sampling_report_sha256']==sha(base/'sampling/sampling_uncertainty.json')
assert len(w['witnesses'])==w['witness_count']==178
for x in w['witnesses']:
 tau=x['band_mm']/1000;ni=x['native_cell_distance_interval_m'];mi=x['mapped_cell_distance_interval_m'];assert x['cell_area_mm2']>0 and x['acceptance_rule_sha256']==rule_hash
 assert ((ni[1]<=tau and mi[0]>tau) if x['relationship_change']=='lost_native' else (mi[1]<=tau and ni[0]>tau))
a=read(base/'acceptance/acceptance_application.json');assert a['acceptance_rule_sha256']==rule_hash and a['evaluator_sha256']==sha(base/'acceptance/evaluate_acceptance.py')
assert a['input_reports_sha256']['sampling']==sha(base/'sampling/sampling_uncertainty.json') and a['input_reports_sha256']['orientation_uncompressed']==hashlib.sha256(oraw).hexdigest() and a['input_reports_sha256']['native']==sha(base/'native_contract/native_contract.json')
assert a['strict_fail_resolution_count']==9 and a['conditional_numeric_bounds']
protected=inputs(read(base/'protected_input_sha256.json'))
prior=read(repo/'research_artifacts/hug_palm_object_validation_2026-10-14/protected_input_sha256.json')
for path,h in prior.items():assert sha(repo/path)==h,path
print(json.dumps({'schema':'hug_orientation_uncertainty_publication_verification_v1','manifest_file_counts':counts,'all_protocol_script_hashes_pass':True,'native_source_count':native_source_count,'exact_native_3d_exports':32,'orientation_sampling_bound_resolutions':bindings,'original_case_input_hash_checks':sourcechecks,'witness_inequalities_pass':178,'strict_fail_resolutions_conditional':9,'compressed_orientation_lossless':True,'protected_current_files':protected,'protected_prior_files':len(prior),'actual_physics_steps':0,'new_eval_data':False,'new_generator_inference':False,'production_model_manuscript_unchanged':True,'scope':'Artifact/numerical validation; uniform arithmetic bounds assumed, sampled normals not continuous area certificates, no physical outcome acceptance.'},indent=2))
