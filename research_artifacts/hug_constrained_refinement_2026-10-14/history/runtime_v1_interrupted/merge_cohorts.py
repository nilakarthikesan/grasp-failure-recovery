"""Bind both complete immutable cohorts into the six-case candidate report."""
from pathlib import Path
import hashlib,json
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def main():
    protocol=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json');inputs=[];records=[]
    for cohort in protocol['allowed_cohorts']:
        path=OUT/('candidate_cohort_'+'_'.join(map(str,cohort))+'.json')
        r=read(path)
        assert r['schema']=='hug_global_constrained_surface_candidate_cohort_v1'
        assert r['protocol_sha256']==ph and r['cohort_case_indices']==cohort
        assert len(r['cases'])==3 and [c['source_inventory_index'] for c in r['cases']]==cohort
        assert r['script_sha256']==sha(OUT/'run_constrained_refinement.py')
        assert r['solver_implementation_sha256']==sha(OUT/'constrained_refinement.py')
        assert r['direction_implementation_sha256']==sha(protocol['direction_helper_path'])
        assert r['guards']['actual']==0 and r['guards']['selftests']==3 and r['physics_steps']==0
        assert r['new_eval_data'] is False and r['new_generator_proposal'] is False and r['training'] is False
        assert r['protected_hashes_unchanged'] and r['original_inputs_and_implementations_unchanged']
        inputs.append({'path':str(path),'sha256':sha(path),'cohort_case_indices':cohort,
            'guards':r['guards'],'runtime_seconds':r['runtime_seconds'],'protected_hashes_unchanged':True})
        records.extend(r['cases'])
    planned={c['source_inventory_index']:c['native_sha256'] for c in protocol['cases']}
    assert len(records)==len({c['source_inventory_index'] for c in records})==len({c['native_sha256'] for c in records})==6
    assert {c['source_inventory_index']:c['native_sha256'] for c in records}==planned
    records.sort(key=lambda c:c['source_inventory_index'])
    for path,h in protocol['source_sha256'].items():assert sha(path)==h,path
    for path,h in protocol['implementation_sha256'].items():assert sha(path)==h,path
    for path,h in read(OUT/'protected_hashes.json').items():assert sha(path)==h,path
    result={'schema':'hug_global_constrained_surface_candidate_v1','protocol_sha256':ph,
        'script_sha256':sha(OUT/'run_constrained_refinement.py'),'merge_script_sha256':sha(__file__),
        'solver_implementation_sha256':sha(OUT/'constrained_refinement.py'),
        'direction_implementation_sha256':sha(protocol['direction_helper_path']),
        'cohort_provenance':inputs,'cases':records,'complete_original_six_case_coverage':True,
        'protected_hashes_unchanged':True,'protected_source_count':len(read(OUT/'protected_hashes.json')),
        'original_inputs_and_implementations_unchanged':True,
        'guards':{'actual':0,'selftests':sum(r['guards']['selftests'] for r in inputs)},
        'new_eval_data':False,'new_generator_proposal':False,'physics_steps':0,'training':False,'production_changes':False,
        'strict_acceptance_rule_unchanged_sha256':protocol['strict_acceptance_rule_sha256'],
        'mapping_acceptance_claim':False,'limitations':protocol['limitations']}
    destination=OUT/'candidate_targets.json'
    if destination.exists():raise FileExistsError('Existing merged evidence must not be overwritten')
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete_case_count':6,'candidate_sha256':sha(destination),'cohort_provenance':inputs}))

if __name__=='__main__':main()
