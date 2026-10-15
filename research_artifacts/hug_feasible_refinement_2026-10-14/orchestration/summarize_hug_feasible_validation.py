"""Summarize complete independent measurements without changing their rule."""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json
OUT=Path('/private/tmp/hug_feasible_refinement_2026-10-14/validation')
FIT=OUT.parent/'seed_stage'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bound_report(name):
    path=OUT/name;b=read(str(path)+'.binding.json');raw=gzip.decompress(path.read_bytes())
    assert sha(path)==b['gzip_sha256'] and hashlib.sha256(raw).hexdigest()==b['uncompressed_sha256']
    report=json.loads(raw);assert report['candidate_sha256']==sha(FIT/'candidate_targets.json')
    assert report['protocol_sha256']==sha(OUT/'protocol.json')
    assert report['guards']['actual']==0 and report['physics_steps']==0
    assert report['all_source_geometry_and_protected_hashes_unchanged']
    return report
def main():
    assert not (OUT/'summary.json').exists()
    preflight=bound_report('candidate_validation_preflight_only.json.gz')
    reports=[bound_report('candidate_validation_8,12,16.json.gz'),bound_report('candidate_validation_20,24,28.json.gz')]
    cases=[c for report in reports for c in report['cases']]
    assert [c['source_inventory_index'] for c in cases]==[8,12,16,20,24,28]
    fit=read(FIT/'fit_summary.json');fc={r['index']:r for r in fit['six_case_rows']}
    parts=['palm','index','middle','ring','little','thumb']
    statuses=Counter();orientation=Counter();records=[]
    for c in cases:
        assert c['engineering_guard_independent_audit']['source_preflight_status']=='PASS'
        index=c['source_inventory_index'];rows=[]
        for density,r in c['resolutions'].items():
            a=r['frozen_acceptance_application'];statuses[a['strict_relationship_status_conditional']]+=1
            orientation[a['pointwise_orientation_status']]+=1
            bands=[]
            for band in ('1','2','5'):
                pp=r['conditional_sampling']['1e-08']['bands'][band]['independent_parts']
                bp=r['baseline_conditional_sampling']['1e-08']['bands'][band]['independent_parts']
                perpart={part:{'lost_lower_mm2':pp[part]['lost_native_area']['lower_mm2'],
                              'added_lower_mm2':pp[part]['new_mapped_area']['lower_mm2'],
                              'original_lost_lower_mm2':bp[part]['lost_native_area']['lower_mm2'],
                              'original_added_lower_mm2':bp[part]['new_mapped_area']['lower_mm2']} for part in parts}
                o=next(v for v in a['orientation_checks'] if str(v['band_mm'])==band)
                original_point=r['baseline_orientation']['bands'][band]['common_same_label']['pointwise_sign_check']
                bands.append({'band_mm':int(band),'part_area_sums_are_not_unique_union_areas':True,
                    'sum_lost_lower_mm2':sum(v['lost_lower_mm2'] for v in perpart.values()),
                    'sum_added_lower_mm2':sum(v['added_lower_mm2'] for v in perpart.values()),
                    'original_sum_lost_lower_mm2':sum(v['original_lost_lower_mm2'] for v in perpart.values()),
                    'original_sum_added_lower_mm2':sum(v['original_added_lower_mm2'] for v in perpart.values()),
                    'sampled_reliable_sign_reversal_count':o['reliable_sampled_sign_reversal_count'],
                    'original_sampled_reliable_sign_reversal_count':sum(v['query_count'] for v in original_point['flip_denominators'].values()),
                    'pointwise_orientation_status':o['pointwise_status'],'per_part':perpart})
            rows.append({'density_mm':float(density),'query_count':r['query_count'],
                'strict_status':a['strict_relationship_status_conditional'],
                'proximity_status':a['proximity_status_conditional'],'pointwise_orientation_status':a['pointwise_orientation_status'],
                'continuous_contact_equivalence':'UNRESOLVED','bands':bands})
        records.append({'index':index,'case_id':c['case_id'],'fit':fc[index],
            'independent_engineering_and_source_preflight':'PASS','resolutions':rows})
    assert sum(statuses.values())==9
    summary={'schema':'hug_feasible_static_post_audit_summary_v2','candidate_sha256':sha(FIT/'candidate_targets.json'),
        'fit_protocol_sha256':sha(FIT/'protocol.json'),'validation_protocol_sha256':sha(OUT/'protocol.json'),
        'summary_script_sha256':sha(__file__),'case_count':6,'resolution_count':9,
        'strict_status_counts':{s:statuses[s] for s in ('PASS','FAIL','UNRESOLVED')},
        'pointwise_orientation_status_counts':{s:orientation[s] for s in ('PASS','FAIL','UNRESOLVED')},
        'all_six_independent_source_and_engineering_preflights_pass':True,
        'all_six_feasible_improving_poses_selected':fit['all_six_feasible_improving_poses_selected'],
        'new_fit_full_surface_evaluations':fit['total_full_surface_evaluations'],
        'new_fit_runtime_seconds':fit['runtime_seconds'],'protected_numeric_dependency_count':89,
        'all_protected_numeric_hashes_unchanged':True,'physics_steps':0,'new_eval_data':False,'training':False,
        'native_grasp_failure_labels':'NOT_TESTED','heldout_generators_or_objects':'NOT_TESTED',
        'continuous_physical_contact_equivalence':'UNRESOLVED',
        'measurement_files_sha256':{str(OUT/n):sha(OUT/n) for n in ('candidate_validation_preflight_only.json.gz','candidate_validation_8,12,16.json.gz','candidate_validation_20,24,28.json.gz')},
        'records':records}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('records','measurement_files_sha256')},indent=2))
if __name__=='__main__':main()
