"""Bind the one complete predeclared all-six development cohort; no fitting."""
from pathlib import Path
import hashlib, json

OUT=Path(__file__).parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    target=OUT/'candidate_targets.json'
    assert not target.exists()
    cohort_path=OUT/'candidate_cohort_8_12_16_20_24_28.json'
    cohort=json.loads(cohort_path.read_text());p=json.loads((OUT/'protocol.json').read_text())
    indices=[8,12,16,20,24,28]
    assert cohort['schema']=='hug_feasible_seed_stage_surface_candidate_cohort_v2'
    assert cohort['cohort_case_indices']==indices
    assert [r['source_inventory_index'] for r in cohort['cases']]==indices
    assert len({r['native_sha256'] for r in cohort['cases']})==6
    assert cohort['protocol_sha256']==sha(OUT/'protocol.json')
    assert cohort['guards']['actual']==0 and cohort['guards']['selftests']==3
    for table in ('source_sha256','implementation_sha256'):
        for path,h in p[table].items():assert sha(path)==h,path
    result=dict(cohort)
    result['schema']='hug_feasible_seed_stage_surface_candidate_v2'
    result['cohort_provenance']=[{'path':str(cohort_path),'sha256':sha(cohort_path),'cohort_case_indices':indices}]
    result['merge_script_sha256']=sha(__file__)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows=[]
    for r in result['cases']:
        b,f=r['solver_audit']['baseline'],r['solver_audit']['final']
        rows.append({'index':r['source_inventory_index'],'object_id':r['object_id'],'source_case':r['source_case'],
            'original_objective':b['objective'],'returned_objective':f['objective'],
            'objective_reduction_percent':100*(b['objective']-f['objective'])/b['objective'],
            'original_mean_tip_mm':1000*b['mean_fingertip_error_m'],'returned_mean_tip_mm':1000*f['mean_fingertip_error_m'],
            'returned_tip_increases_mm':[1000*v for v in f['tip_error_increase_m']],
            'all_five_exact_caps_pass':all(f['individual_tip_nonworsening_pass']),
            'selected_new_pose':r['solver_audit']['surface_candidate_returned'],
            'full_callbacks':r['solver_audit']['counters']['full'],'FK_callbacks':r['solver_audit']['counters']['FK'],
            'independent_fixed_URDF_vs_scene_max_link_error_m':r['independent_fixed_URDF_vs_scene_max_link_error_m']})
    summary={'schema':'hug_feasible_seed_stage_fit_summary_v2','candidate_sha256':sha(target),
        'protocol_sha256':sha(OUT/'protocol.json'),'six_case_rows':rows,
        'all_six_feasible_improving_poses_selected':all(r['selected_new_pose'] and r['all_five_exact_caps_pass'] for r in rows),
        'total_full_surface_evaluations':sum(r['full_callbacks'] for r in rows),
        'runtime_seconds':cohort['runtime_seconds'],'protected_sources_unchanged':cohort['protected_hashes_unchanged'],
        'physics_steps':0,'new_eval_data':False,'training':False,'scientific_mapping_acceptance':'INDEPENDENT_AUDIT_PENDING'}
    (OUT/'fit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
