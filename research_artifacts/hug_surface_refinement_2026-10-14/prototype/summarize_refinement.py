"""Summarize fixed raw/guarded translation diagnostics, never outcome labels."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    raw=json.loads((OUT/'candidate_targets.json').read_text())
    guarded=json.loads((OUT/'guarded_candidate_targets.json').read_text())
    rm={c['native_sha256']:c for c in raw['cases']}
    records=[]
    for c in guarded['cases']:
        orig=c['engineering_guard']['baseline'];current=c['final_assessment']
        unguarded=c['engineering_guard']['attempted_alphas'][0]['assessment']
        rawc=rm[c['native_sha256']]
        assert c['engineering_guard']['attempted_alphas'][0]['alpha']==1
        assert abs(unguarded['mean_fingertip_error_m']-rawc['landmark_audit']['candidate_mean_fingertip_error_m'])<1e-12
        row={'source_inventory_index':c['source_inventory_index'],'object_id':c['object_id'],'source_case':c['source_case'],
            'native_sha256':c['native_sha256'],'selected_alpha':c['engineering_guard']['selected_alpha'],
            'original_full_objective':orig['objective'],'raw_full_objective':unguarded['objective'],
            'guarded_full_objective':current['objective'],
            'relative_guarded_objective_reduction':1-current['objective']/orig['objective'],
            'original_mean_fingertip_error_mm':orig['mean_fingertip_error_m']*1000,
            'raw_mean_fingertip_error_mm':unguarded['mean_fingertip_error_m']*1000,
            'guarded_mean_fingertip_error_mm':current['mean_fingertip_error_m']*1000,
            'original_mean_middle_error_mm':orig['mean_middle_error_m']*1000,
            'raw_mean_middle_error_mm':unguarded['mean_middle_error_m']*1000,
            'guarded_mean_middle_error_mm':current['mean_middle_error_m']*1000,
            'original_quadrature_unsigned_distance_mae_mm':orig['fixed_quadrature_unsigned_distance_mae_m']*1000,
            'guarded_quadrature_unsigned_distance_mae_mm':current['fixed_quadrature_unsigned_distance_mae_m']*1000,
            'original_quadrature_closest_point_mean_error_mm':orig['fixed_quadrature_closest_point_mean_error_m']*1000,
            'guarded_quadrature_closest_point_mean_error_mm':current['fixed_quadrature_closest_point_mean_error_m']*1000,
            'maximum_guarded_joint_delta_rad':c['maximum_abs_joint_delta_rad'],
            'candidate_fixed_vs_scene_max_link_error_m':c['fixed_URDF_vs_scene_max_link_error_m'],
            'original_targets_preserved':True,'raw_targets_preserved':True,'new_eval_data':False,
            'engineering_guard_passes':current['mean_fingertip_error_m']<=.01,
            'scientific_mapping_acceptance':'INDEPENDENT_FROZEN_RULE_AUDIT_REQUIRED'}
        records.append(row)
    stage=[c['wrist_stage'] for c in raw['cases']]+[d for c in raw['cases'] for d in c['part_diagnostics'].values()]
    result={'schema':'hug_surface_refinement_development_summary_v3','raw_report_sha256':sha(OUT/'candidate_targets.json'),
        'guarded_report_sha256':sha(OUT/'guarded_candidate_targets.json'),'case_count':len(records),
        'raw_stage_count':len(stage),'all_raw_stage_objectives_improved':all(s['accepted_objective_improvement'] for s in stage),
        'solver_success_count':sum(s['solver_success'] for s in stage),'solver_budget_exhaustion_count':sum(s['solver_status']==0 for s in stage),
        'raw_engineering_guard_refusal_count':sum(not c['landmark_audit']['candidate_passes_existing_engineering_guard'] for c in raw['cases']),
        'guarded_existing_engineering_guard_pass_count':sum(c['engineering_guard_passes'] for c in records),
        'original_retained_on_fallback_count':sum(c['selected_alpha']==0 for c in records),
        'minimum_relative_objective_reduction':min(c['relative_guarded_objective_reduction'] for c in records),
        'maximum_relative_objective_reduction':max(c['relative_guarded_objective_reduction'] for c in records),
        'raw_runtime_seconds':raw['runtime_seconds'],'guard_runtime_seconds':guarded['runtime_seconds'],
        'maximum_candidate_fixed_vs_scene_link_error_m':max(c['candidate_fixed_vs_scene_max_link_error_m'] for c in records),
        'guard_actual_physics_attempts':guarded['guards']['actual'],
        'new_eval_data':False,'physics_steps':0,'new_generator_proposal':False,'training':False,
        'source_and_protected_hashes_preserved':raw['protected_hashes_unchanged'] and guarded['protected_hashes_unchanged'],
        'limitations':['These are development objective diagnostics, not scientific acceptance scores.','Original fingertip anatomy becomes less accurate even though frozen geometric objective decreases.','Six existing cases are not independent unseen-object evidence.','Actual exterior normal states and cell-wise proximity must be measured independently.'],
        'cases':records}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))

if __name__=='__main__':main()
