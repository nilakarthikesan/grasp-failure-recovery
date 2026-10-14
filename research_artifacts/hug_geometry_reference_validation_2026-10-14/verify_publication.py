"""Read-only artifact/protected-input verification; no runtime or optimization."""
from pathlib import Path
import argparse, ast, gzip, hashlib, json
base=Path(__file__).resolve().parent
repo=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def check_files(folder):
    m=read(folder/'manifest.json');entries=m.get('files_sha256',m.get('files'))
    for name,s in entries.items():
        h=s if isinstance(s,str) else s['sha256']
        assert sha((folder/name).read_bytes())==h,(folder,name)
    return len(entries)
def check_inputs(inputs):
    for path,h in inputs.items():assert sha(Path(path).read_bytes())==h,path
    return len(inputs)
def compressed(folder,full,summary):
    raw=gzip.decompress((folder/full).read_bytes());r=json.loads(raw);s=read(folder/summary)
    expected=s.get('original_report_sha256',s.get('original_full_report_sha256'))
    if expected is not None:assert sha(raw)==expected
    return r,sha(raw)
parser=argparse.ArgumentParser();parser.add_argument('--skip-parent-manifest',action='store_true');args=parser.parse_args()
counts={p:check_files(base/p) for p in ('geometry','reference','reference/corrections','surfaces','signed_profiles')}
if not args.skip_parent_manifest:counts['parent']=check_files(base)
for path in base.rglob('*.py'):ast.parse(path.read_bytes())
g=read(base/'geometry/geometry_cause_report.json')
r,refsha=compressed(base/'reference','hug_reference_comparison_full.json.gz','hug_reference_comparison_compact.json')
s,signedsha=compressed(base/'signed_profiles','signed_hand_visual_profiles_full.json.gz','summary.json')
f=read(base/'surfaces/surface_reference_comparison.json')
for report,folder,protocol,script in ((g,'geometry','protocol.json','investigate_geometry.py'),(r,'reference','protocol.json','compare_reference.py'),(s,'signed_profiles','protocol.json','profile_signed_geometry.py'),(f,'surfaces','protocol_v3.json','compare_surfaces.py')):
    assert report['protocol_sha256']==sha((base/folder/protocol).read_bytes())
    assert report['script_sha256']==sha((base/folder/script).read_bytes())
    assert report['physics_steps']==0 and report['new_eval_data'] is False and report['new_inference'] is False
assert refsha=='0b7a2df411b9763f29c9cda6a5a252417716869dc5728de90369ef5f632738f6'
assert f['reference_report_sha256']==refsha
assert len(r['cases'])==32 and len(f['cases'])==len(s['cases'])==6
assert f['previous_current_surface_regression_max_difference']==0
assert f['guards']['actual']==s['integration_attempts']==g['integration_attempts_in_investigation']==0
inputs={key:check_inputs(report['input_sha256_before_and_after']) for key,report in [('geometry',g),('reference',r),('signed',s)]}
inputs['reference_code_and_config']=check_inputs(r['code_and_config_sha256_before_and_after'])
for c in f['cases']:check_inputs(c['input_sha256'])
protected=read(base/'protected_input_sha256.json');inputs['protected_current_stage']=check_inputs(protected)
previous=read(repo/'research_artifacts/hug_palm_object_validation_2026-10-14/protected_input_sha256.json')
for path,h in previous.items():assert sha((repo/path).read_bytes())==h,path
inputs['protected_previous_stage']=len(previous)
print(json.dumps({'schema':'hug_geometry_reference_publication_verification_v1','subfolder_file_counts':counts,'hash_checks':inputs,'reference_full_uncompressed_sha256':refsha,'signed_full_uncompressed_sha256':signedsha,'all_protocol_script_bindings_pass':True,'compressed_full_reports_lossless':True,'current_surface_regression_max':0,'physics_steps':0,'new_eval_data':False,'new_inference':False,'production_code_model_manuscript_unchanged':True,'case_counts':{'reference':32,'surface':6,'signed':6},'interpretation':'Publication/numerical invariant checks only; not contact or outcome equivalence certification.'},indent=2))
