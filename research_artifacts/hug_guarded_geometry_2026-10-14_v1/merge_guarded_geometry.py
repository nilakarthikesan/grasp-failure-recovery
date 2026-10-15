"""Merge exactly the two frozen disjoint cohorts; no fitting or simulation."""
from pathlib import Path
import hashlib,json
OUT=Path(__file__).parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
    pp=OUT/'protocol.json';p=read(pp);ph=sha(pp)
    assert p['execution_authorized_by_root_review']
    assert p['allowed_cohorts']==[[8,16,24],[12,20,28]]
    assert p['implementation_sha256'][str(Path(__file__))]==sha(__file__)
    target=OUT/'candidate_targets.json'
    assert not target.exists() and not (OUT/'fit_summary.json').exists()
    for table in ('source_sha256','implementation_sha256','query_cache_sha256','full_cache_sha256'):
        for path,h in p[table].items():assert sha(path)==h,path
    protected=read(OUT/'protected_hashes.json')
    assert sha(OUT/'protected_hashes.json')==p['protected_snapshot_sha256']
    for path,h in protected.items():assert sha(path)==h,path
    records=[];provenance=[];cohorts=[]
    for indices in p['allowed_cohorts']:
        path=OUT/('candidate_cohort_'+'_'.join(map(str,indices))+'.json');report=read(path)
        assert report['schema']=='hug_guarded_geometry_surface_candidate_cohort_v1'
        assert report['protocol_sha256']==ph
        assert report['script_sha256']==p['implementation_sha256'][str(OUT/'run_guarded_geometry.py')]
        assert report['solver_implementation_sha256']==p['implementation_sha256'][str(OUT/'feasible_surface_refinement.py')]
        assert report['direction_implementation_sha256']==p['implementation_sha256'][str(OUT/'exact_direction.py')]
        assert report['cohort_case_indices']==indices
        assert [r['source_inventory_index'] for r in report['cases']]==indices
        assert report['method_version']==3 and report['method_stage']==p['method_stage']
        assert report['guards']['actual']==0 and report['guards']['selftests']==3
        assert report['physics_steps']==0 and not report['new_eval_data'] and not report['training']
        assert report['protected_hashes_unchanged'] and report['original_inputs_and_implementations_unchanged']
        assert report['strict_acceptance_rule_unchanged_sha256']==p['strict_acceptance_rule_sha256']
        for r in report['cases']:
            assert r['terminal_direction_guard']['accepted']
            assert r['terminal_direction_guard']['metadata_callback_sequence']==r['full_callback_count_including_fresh_final']
            assert r['keyed_direction_evidence']['callback_count']==r['full_callback_count_including_fresh_final']
            assert r['new_eval_data'] is False and r['physics_steps']==0
        records+=report['cases'];cohorts.append(report)
        provenance.append({'path':str(path),'sha256':sha(path),'cohort_case_indices':indices})
    planned={r['source_inventory_index']:r['native_sha256'] for r in p['cases']}
    assert len(records)==len(planned)==6
    assert len({r['source_inventory_index'] for r in records})==len({r['native_sha256'] for r in records})==6
    assert {r['source_inventory_index']:r['native_sha256'] for r in records}==planned
    records.sort(key=lambda r:r['source_inventory_index'])
    result=dict(cohorts[0]);result.update(schema='hug_guarded_geometry_surface_candidate_v1',
        cohort_case_indices=[r['source_inventory_index'] for r in records],cases=records,cohort_provenance=provenance,
        guards={'actual':0,'selftests':sum(r['guards']['selftests'] for r in cohorts),'testing':False},
        merge_script_sha256=sha(__file__),runtime_seconds=sum(r['runtime_seconds'] for r in cohorts),
        summed_process_runtime_seconds=sum(r['runtime_seconds'] for r in cohorts))
    write_new(target,result)
    rows=[]
    for r in records:
        b,f=r['solver_audit']['baseline'],r['solver_audit']['final']
        rows.append({'index':r['source_inventory_index'],'object_id':r['object_id'],'source_case':r['source_case'],
            'original_objective':b['objective'],'returned_objective':f['objective'],
            'objective_reduction_percent':100*(b['objective']-f['objective'])/b['objective'],
            'original_mean_tip_mm':1000*b['mean_fingertip_error_m'],'returned_mean_tip_mm':1000*f['mean_fingertip_error_m'],
            'returned_tip_increases_mm':[1000*v for v in f['tip_error_increase_m']],
            'all_five_exact_caps_pass':all(f['individual_tip_nonworsening_pass']),
            'selected_new_pose':r['solver_audit']['surface_candidate_returned'],
            'terminal_direction_guard_pass':r['terminal_direction_guard']['accepted'],
            'termination':r['solver_audit']['termination'],'full_callbacks':r['solver_audit']['counters']['full'],
            'FK_callbacks':r['solver_audit']['counters']['FK'],
            'guard_refusals':sum(e['kind']=='candidate_guard_assessment' and not e['assessment']['accepted'] for e in r['solver_audit']['events']),
            'independent_fixed_URDF_vs_scene_max_link_error_m':r['independent_fixed_URDF_vs_scene_max_link_error_m']})
    summary={'schema':'hug_guarded_geometry_fit_summary_v1','candidate_sha256':sha(target),'protocol_sha256':ph,
        'six_case_rows':rows,'total_full_surface_evaluations':sum(r['full_callbacks'] for r in rows),
        'summed_process_runtime_seconds':result['summed_process_runtime_seconds'],'cohort_provenance':provenance,
        'protected_sources_unchanged':True,'physics_steps':0,'new_eval_data':False,'training':False,
        'scientific_mapping_acceptance':'INDEPENDENT_AUDIT_PENDING'}
    write_new(OUT/'fit_summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
