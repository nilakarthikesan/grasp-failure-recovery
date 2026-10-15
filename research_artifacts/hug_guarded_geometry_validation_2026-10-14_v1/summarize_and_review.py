"""Independent read-only closure of bound static audit reports; no FK imports."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import gzip,hashlib,json
import numpy as np

OUT=Path(__file__).parent
inputs={};checks=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    path=Path(path);inputs[str(path)]=sha(path)
    return json.loads(path.read_bytes())
def check(name,value):
    checks.append({'check':name,'pass':bool(value)})
    assert value,name
def report(name):
    path=OUT/name;inputs[str(path)]=sha(path)
    raw=gzip.decompress(path.read_bytes());r=json.loads(raw);b=read(str(path)+'.binding.json')
    check(name+'_compressed_hash',b['gzip_sha256']==inputs[str(path)])
    check(name+'_uncompressed_hash',b['uncompressed_sha256']==hashlib.sha256(raw).hexdigest())
    check(name+'_protocol_candidate',r['protocol_sha256']==protocol_sha==b['protocol_sha256'] and r['candidate_sha256']==candidate_sha==b['candidate_sha256'])
    check(name+'_wrapper_hash',r['script_sha256']==p['script_sha256'])
    check(name+'_zero_actual_steps',r['guards']=={'selftest':3,'actual':0} and r['physics_steps']==0 and not r['new_eval_data'] and not r['new_inference'] and not r['training'])
    check(name+'_protected_scope',r['all_source_geometry_and_protected_hashes_unchanged'] and r['protected_source_count']==89 and not r['skip_orientation_pilot'])
    check(name+'_case_count',len(r['cases'])==b['case_count'])
    return r
p=read(OUT/'protocol.json');protocol_sha=inputs[str(OUT/'protocol.json')]
candidate_path=Path(p['candidate_path']);candidate_sha=sha(candidate_path);inputs[str(candidate_path)]=candidate_sha
fit_summary=read(candidate_path.parent/'fit_summary.json');fit_protocol=read(candidate_path.parent/'protocol.json')
check('fit_candidate_binding',fit_summary['candidate_sha256']==candidate_sha)
check('fit_protocol_binding',fit_summary['protocol_sha256']==p['candidate_protocol_expected_sha256']==inputs[str(candidate_path.parent/'protocol.json')])
check('frozen_wrapper_unchanged',sha(OUT/'validate_candidate.py')==p['script_sha256'])
protected=read(OUT/'protected_hashes.json')
check('protected_snapshot_hash',inputs[str(OUT/'protected_hashes.json')]==p['protected_snapshot_sha256'])
check('all89_protected_unchanged',len(protected)==p['protected_file_count']==89 and all(sha(k)==v for k,v in protected.items()))
check('all_fixed_bindings_unchanged',all(sha(k)==v for k,v in p['fixed_files_sha256'].items()))
parity=read(OUT/'wrapper_parity.json')
check('literal_scientific_parity',parity['entire_new_pose_density_query_proximity_cell_orientation_status_measurement_block_literal_bytes_equal'] and all(v['literal_bytes_equal'] for v in parity['unchanged_functions'].values()))
pre=report('candidate_validation_preflight_only.json.gz')
audits=[report('candidate_validation_8,12,16.json.gz'),report('candidate_validation_20,24,28.json.gz')]
check('preflight_all_six_first',pre['preflight_only'] and [c['source_inventory_index'] for c in pre['cases']]==p['case_indices'])
cases=[c for r in audits for c in r['cases']]
check('full_declared_cohorts_complete',[[c['source_inventory_index'] for c in r['cases']] for r in audits]==p['allowed_full_audit_cohorts'] and [c['source_inventory_index'] for c in cases]==p['case_indices'] and all(not r['preflight_only'] for r in audits))
root=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
old=root/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14'
rule=read(old/'acceptance/acceptance_rule.json');sp=read(old/'sampling/protocol.json');op=read(old/'orientation/protocol.json')
check('unchanged_strict_rule',inputs[str(old/'acceptance/acceptance_rule.json')]=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4')
sampling=read(old/'sampling/sampling_uncertainty.json')
ori_path=old/'orientation/orientation_report_full.json.gz';inputs[str(ori_path)]=sha(ori_path);orientation=json.loads(gzip.decompress(ori_path.read_bytes()))
sm={c['native_sha256']:c for c in sampling['cases']};om={c['native_sha256']:c for c in orientation['cases']}
plans={c['source_inventory_index']:c for c in op['cases']};preflights={c['source_inventory_index']:c for c in pre['cases']}
strict=Counter();prox=Counter();pointwise=Counter();rows=[];witnesses=0
for case in cases:
    index=case['source_inventory_index'];native=case['native_sha256'];eng=case['engineering_guard_independent_audit'];state=case['scene_state_binding']
    label=f'case{index}_'
    check(label+'preflight_record_identical',{k:v for k,v in case.items() if k!='resolutions'}=={k:v for k,v in preflights[index].items() if k!='resolutions'})
    check(label+'numerical_source_PASS',eng['source_preflight_status']=='PASS' and eng['native_landmark_errors_independently_recomputed'] and eng['each_final_fingertip_nonworsening_exact_computed'] and eng['actual_mean_guard_pass'])
    check(label+'exact_all_five_original_caps',np.array_equal(eng['original_tip_caps_sqrt_einsum_m'],eng['candidate_tip_norms_sqrt_einsum_m']) and eng['individual_fingertip_arithmetic_allowance_m']==0)
    check(label+'exact_complete_scene_identity',state['original_qpos_float64_sha256']==state['selected_qpos_float64_sha256'] and state['qpos24_exact_original'] and state['entire_scene_qpos_exact_original_including_root'] and state['time_s']==0 and state['object_qpos_exact_original'] and state['object_world_position_and_rotation_exact_original'])
    selection=eng['recorded_solver_selection']
    check(label+'independent_guard_trace',selection['recorded_guarded_pattern_selection_verified'] and selection['all_guard_part_SSE_validity_and_metadata_bindings_recomputed'] and selection['all_recorded_full_residual_objectives_recomputed'] and selection['all_full_FK_guard_budget_counts_verified'] and selection['full_and_FK_terminal_events_fresh_verified'])
    check(label+'fallback_is_not_geometry_PASS',eng['fallback_original_target_exact'] and selection['selected_candidate_source']=='original_fallback' and selection['original_fallback_does_not_mean_geometric_PASS'] and selection['not_scientific_mapping_acceptance'])
    check(label+'resolution_plan',native==plans[index]['native_sha256'] and list(map(float,case['resolutions']))==list(map(float,plans[index]['densities_mm'])))
    for density,res in case['resolutions'].items():
        label2=label+f'density{density}_'; provenance=res['audit_provenance']
        check(label2+'reused_original_without_new_geometry',provenance['measurement_mode']=='REUSED_EXACT_ORIGINAL_SCENE_STATE_NO_GEOMETRY_RERUN' and not provenance['new_surface_or_orientation_measurement_executed'] and provenance['fresh_source_FK_preflight_this_run'])
        check(label2+'original_data_literal',res['conditional_sampling']==sm[native]['resolutions'][density]['numerical_allowances'] and res['orientation']==om[native]['resolutions'][density])
        check(label2+'provenance_hashes',provenance['original_sampling_report_sha256']==inputs[str(old/'sampling/sampling_uncertainty.json')] and provenance['original_orientation_report_sha256']==inputs[str(ori_path)] and provenance['sampling_protocol_sha256']==inputs[str(old/'sampling/protocol.json')] and provenance['orientation_protocol_sha256']==inputs[str(old/'orientation/protocol.json')] and provenance['acceptance_rule_sha256']==inputs[str(old/'acceptance/acceptance_rule.json')] and provenance['native_sha256']==native and provenance['input_sha256']==case['input_sha256'] and provenance['original_and_selected_scene_qpos_float64_sha256']==state['original_qpos_float64_sha256'])
        cache=next(k for k in op['query_cache_sha256'] if k.endswith(f"{case['case_id']}_density{density}mm_queries.npz"))
        check(label2+'cache_identity',sha(cache)==op['query_cache_sha256'][cache]==res['query_cache_sha256'])
        states=[];normals=[];bands=[]
        for band in rule['bands_mm']:
            bypart=res['conditional_sampling'][str(sp['primary_allowance_m'])]['bands'][str(band)]['independent_parts'];summed={'lost_lower_mm2':0.0,'added_lower_mm2':0.0}
            for part in rule['anatomical_parts']:
                pair=bypart[part];lost=pair['lost_native_area'];added=pair['new_mapped_area']
                current='FAIL' if lost['lower_mm2']>0 or added['lower_mm2']>0 else 'PASS' if lost['upper_mm2']==0 and added['upper_mm2']==0 else 'UNRESOLVED';states.append(current)
                reported=next(c for c in res['frozen_acceptance_application']['proximity_checks'] if c['part']==part and c['band_mm']==band)
                check(label2+f'band{band}_{part}_rule',reported['status']==current and reported['lost_area_lower_mm2']==lost['lower_mm2'] and reported['added_area_lower_mm2']==added['lower_mm2'] and reported['lost_area_upper_mm2']==lost['upper_mm2'] and reported['added_area_upper_mm2']==added['upper_mm2'])
                summed['lost_lower_mm2']+=lost['lower_mm2'];summed['added_lower_mm2']+=added['lower_mm2']
            signs=res['orientation']['bands'][str(band)]['common_same_label']['pointwise_sign_check']
            count=sum(s['query_count'] for s in signs['flip_denominators'].values());normal='FAIL' if count else 'UNRESOLVED';normals.append(normal)
            reported=next(c for c in res['frozen_acceptance_application']['orientation_checks'] if c['band_mm']==band)
            check(label2+f'band{band}_normal_rule',reported['reliable_sampled_sign_reversal_count']==count and reported['pointwise_status']==normal and reported['continuous_orientation_coverage_status']=='UNRESOLVED')
            for direction,examples in signs['witnesses'].items():
                for witness in examples:
                    ns=-float(np.dot(witness['object_outward_normal'],witness['native_outward_normal']));ms=-float(np.dot(witness['object_outward_normal'],witness['mapped_outward_normal']));dead=op['facing_dot_numerical_deadzone']
                    check(label2+f'band{band}_witness{witnesses}_dot',abs(ns-witness['native_facing_compatibility'])<=1e-12 and abs(ms-witness['mapped_facing_compatibility'])<=1e-12 and ((ns>dead and ms < -dead) if direction=='native_facing_mapped_away' else (ns < -dead and ms>dead)))
                    witnesses+=1
            bands.append({'band_mm':band,**summed,'part_area_sums_are_not_unique_union':True,'sampled_reliable_sign_reversal_count':count,'pointwise_orientation_status':normal})
        ps='FAIL' if 'FAIL' in states else 'PASS' if all(s=='PASS' for s in states) else 'UNRESOLVED';ns='FAIL' if 'FAIL' in normals else 'UNRESOLVED';ss='FAIL' if ps=='FAIL' or ns=='FAIL' else 'UNRESOLVED'
        frozen=res['frozen_acceptance_application']
        check(label2+'conjunction',frozen['proximity_status_conditional']==ps and frozen['pointwise_orientation_status']==ns and frozen['strict_relationship_status_conditional']==ss)
        check(label2+'scope_limits',frozen['full_physical_contact_equivalence']=='UNRESOLVED' and frozen['failure_label_preservation']=='NOT_TESTED' and frozen['independent_unseen_object_validation']=='NOT_TESTED')
        strict[ss]+=1;prox[ps]+=1;pointwise[ns]+=1
        rows.append({'source_inventory_index':index,'case_id':case['case_id'],'density_mm':float(density),'strict_status':ss,'proximity_status':ps,'pointwise_orientation_status':ns,'measurement_mode':provenance['measurement_mode'],'new_full_grid_geometry_measurement':False,'bands':bands})
tamper=read(OUT/'trace_tamper_refusals.json')
check('five_meaningful_tamper_refusals',tamper['candidate_sha256']==candidate_sha and tamper['validation_protocol_sha256']==protocol_sha and tamper['control_count']==5 and tamper['authentic_record_verified'] and all(x['copied_authentic_record_refused'] for x in tamper['refusals']) and tamper['geometry_or_FK_replays']==0)
check('all_nine_statuses',len(rows)==9 and strict['FAIL']==9 and prox['FAIL']==9 and pointwise['FAIL']==7 and pointwise['UNRESOLVED']==2)
check('all_six_fit_returns_original',len(fit_summary['six_case_rows'])==6 and all(not r['selected_new_pose'] and r['all_five_exact_caps_pass'] and r['terminal_direction_guard_pass'] for r in fit_summary['six_case_rows']))
check('fit_scope',fit_summary['physics_steps']==0 and not fit_summary['new_eval_data'] and not fit_summary['training'] and fit_summary['protected_sources_unchanged'])
check('parsed_inputs_unchanged',all(sha(k)==v for k,v in inputs.items()))
count=lambda c:{k:c[k] for k in ['PASS','FAIL','UNRESOLVED']}
summary={'schema':'hug_guarded_geometry_independent_static_summary_v1','created_utc':datetime.now(timezone.utc).isoformat(),
    'candidate_sha256':candidate_sha,'fit_protocol_sha256':p['candidate_protocol_expected_sha256'],'validation_protocol_sha256':protocol_sha,
    'summary_script_sha256':sha(__file__),'source_case_count':6,'source_preflight_PASS_count':6,'all_five_tip_caps_PASS_count':30,
    'selected_new_scene_pose_count':0,'unchanged_original_scene_pose_count':6,'reused_original_audit_resolution_count':9,
    'new_full_grid_geometry_measurement_count':0,'new_full_grid_orientation_measurement_count':0,
    'strict_status_counts':count(strict),'proximity_status_counts':count(prox),'pointwise_orientation_status_counts':count(pointwise),
    'continuous_orientation_full_domain':'UNRESOLVED','continuous_physical_contact_equivalence':'UNRESOLVED',
    'failure_label_preservation':'NOT_TESTED','unseen_object_generalization':'NOT_TESTED','physics_steps':0,'new_eval_data':False,
    'new_generator_inference':False,'predictor_training':False,'production_mapper_changed':False,
    'fit_full_surface_callbacks':fit_summary['total_full_surface_evaluations'],'fit_summed_process_runtime_seconds':fit_summary['summed_process_runtime_seconds'],
    'protected_numeric_dependency_count':89,'all_protected_numeric_hashes_unchanged':True,
    'read_only_consistency_check_count':len(checks),'stored_orientation_witness_records_dot_recomputed':witnesses,
    'read_only_input_sha256':inputs,'resolution_records':rows,
    'conclusion':'Numerical/source mapping checks pass in the declared six-case scope. The bounded guard-aware development fit selected no new pose. Nine bound original geometric audits are explicitly reused, not rerun: strict proximity FAIL9 and pointwise orientation FAIL7/UNRESOLVED2. These records do not certify full HUG mapping or native failure equivalence.',
    'limitations':p['limitations']}
for name,obj in [('summary.json',summary),('independent_consistency_review.json',{'schema':'hug_guarded_geometry_read_only_consistency_review_v1','result':'PASS','checks':checks,'input_sha256':inputs,'geometry_or_FK_replays':0,'not_scientific_mapping_acceptance':True})]:
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:summary[k] for k in ['source_preflight_PASS_count','all_five_tip_caps_PASS_count','selected_new_scene_pose_count','reused_original_audit_resolution_count','new_full_grid_geometry_measurement_count','strict_status_counts','proximity_status_counts','pointwise_orientation_status_counts','read_only_consistency_check_count']},indent=2))
