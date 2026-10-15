import ast, hashlib, json
from datetime import datetime,timezone
from pathlib import Path
OUT=Path('/private/tmp/hug_feasible_refinement_2026-10-14/validation')
OLD=Path('/private/tmp/hug_constrained_validation_runtime_v2_1_2026-10-14/validate_candidate.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function_source(path,name):
    source=Path(path).read_text();nodes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1
    return ast.get_source_segment(source,nodes[0])
assert not (OUT/'protocol.json').exists()
assert sha(OUT/'validate_candidate.py')=='e8e9555f841912b01d5d1b0286579ce5ec6751363f51f60098c744fe195df0e9'
for name in ('full_fields','status'):
    assert function_source(OUT/'validate_candidate.py',name)==function_source(OLD,name)
parity=json.loads((OUT/'wrapper_parity.json').read_text())
assert parity['entire_density_query_proximity_cell_orientation_status_measurement_block_literal_bytes_equal']
p=json.loads((OUT/'protocol.DRAFT.json').read_text())
p['candidate_merge_script_path']=str(OUT.parent/'seed_stage/merge_seed_stage.py')
p['fixed_files_sha256'][p['candidate_merge_script_path']]=sha(p['candidate_merge_script_path'])
p['fixed_files_sha256'][str(OUT/'run_validation.py')]=sha(OUT/'run_validation.py')
for path,h in p['fixed_files_sha256'].items():assert sha(path)==h,path
protected=json.loads((OUT/'protected_hashes.json').read_text())
assert sha(OUT/'protected_hashes.json')==p['protected_snapshot_sha256']
assert len(protected)==p['protected_file_count']==89
for path,h in protected.items():assert sha(path)==h,path
assert p['bands_mm']==[1,2,5] and p['individual_fingertip_arithmetic_allowance_m']==0
p['schema']=p['schema'].removesuffix('_DRAFT')
p['frozen_utc']=datetime.now(timezone.utc).isoformat()
p['execution_authorized_by_root_review']=True
p['execution_launcher_sha256']=sha(OUT/'run_validation.py')
p['numba_pool_capacity']=4
p['numba_threads_per_process']=2
p['allowed_full_audit_cohorts']=[[8,12,16],[20,24,28]]
p['maximum_concurrent_audit_processes']=1
p['root_review']='Literal metric/status source and unchanged block parity checked; seed schema/provenance/selection guarded, independent original/final caps from fresh geometry, all-six preflight required, two serial declared audit cohorts with same frozen nine resolutions/rule. No new inference/integration/labels/training.'
(OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps({'frozen_validation_protocol_sha256':sha(OUT/'protocol.json'),'unchanged_scientific_rule':True,'numeric_closure_count':89}))
