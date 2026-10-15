"""Seal audit methods and selected candidate bytes before fresh geometry calculations."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re
OUT=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
p=read(OUT/'protocol.DRAFT.json');candidate_path=Path(p['candidate_path']);cp=read(candidate_path);fit_protocol=candidate_path.parent/'protocol.json';fit=read(fit_protocol)
assert cp['protocol_sha256']==sha(fit_protocol)
assert sorted(x['source_inventory_index'] for x in cp['cases'])==p['case_indices']
assert not (OUT/'protocol.json').exists(), 'Never overwrite frozen protocol'
p['script_sha256']=sha(OUT/'validate_candidate.py');p['execution_launcher_sha256']=sha(OUT/'run_validation.py')
p['candidate_schema']=cp['schema'];p['candidate_sha256']=sha(candidate_path);p['candidate_protocol_expected_sha256']=sha(fit_protocol)
bound={str(fit_protocol):sha(fit_protocol)}
# Original per-file scientific and implementation mappings are traversed without modifying values.
def collect(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if isinstance(k,str) and k.startswith('/') and isinstance(v,str) and re.fullmatch('[a-f0-9]{64}',v):
                assert sha(k)==v,k;bound[k]=v
            collect(v)
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str) and Path(value['path']).is_file():
            assert sha(value['path'])==value['sha256'];bound[value['path']]=value['sha256']
    elif isinstance(value,list):
        for v in value:collect(v)
collect(fit);collect(cp)
for name in ('run_constraints.py','constraint_core.py','constraint_engine.py','constraint_solver.py','exact_direction.py','free_palm_kinematics.py','geometry_helpers.py','surface_component_guard.py'):
    path=candidate_path.parent/name
    if path.exists():bound[str(path)]=sha(path)
p['candidate_bound_files_sha256']=bound
ori_path=next(k for k in p['fixed_files_sha256'] if k.endswith('/orientation/protocol.json'));ori=read(ori_path)
for mapping in ('query_cache_sha256','source_file_sha256'):
    for k,v in ori[mapping].items():assert sha(k)==v,k;p['fixed_files_sha256'][k]=v
p['fixed_files_sha256'][str(OUT/'wrapper_parity.json')]=sha(OUT/'wrapper_parity.json')
p['fixed_files_sha256'][str(OUT/'inherited_metric_block.py.txt')]=sha(OUT/'inherited_metric_block.py.txt')
p['fixed_files_sha256'][str(OUT/'summarize_audit.py')]=sha(OUT/'summarize_audit.py')
for name in ('saved_witnesses.json','cached_failure_row_counts.json','witness_extraction_protocol.json','witness_extraction_protocol_v2.json','cached_failure_count_protocol.json','extract_saved_witnesses.py','extract_saved_witnesses_v2.py','count_cached_failure_rows.py','extraction_v1_abort.json','prepare_audit_wrapper.py','prepare_candidate_domain.py','prepare_summary_metadata.py','freeze_protocol.py'):
    p['fixed_files_sha256'][str(OUT/name)]=sha(OUT/name)
p['execution_authorized_by_root_review']=True;p['frozen_utc']=datetime.now(timezone.utc).isoformat()
for k,v in {**p['fixed_files_sha256'],**bound,**read(OUT/'protected_hashes.json')}.items():assert sha(k)==v,k
(OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps({'protocol_sha256':sha(OUT/'protocol.json'),'candidate_sha256':p['candidate_sha256'],'candidate_protocol_expected_sha256':p['candidate_protocol_expected_sha256'],'candidate_bound_files':len(bound),'fixed_files':len(p['fixed_files_sha256']),'script_sha256':p['script_sha256']},indent=2))
