"""Read-only review of bound static reports; no geometry/FK/simulator imports."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import ast, gzip, hashlib, json
import xml.etree.ElementTree as ET
import numpy as np

HERE=Path(__file__).parent
checks=[]; failures=[]; inputs={}; uncompressed={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):
    path=Path(path);data=path.read_bytes();inputs[str(path)]=hashlib.sha256(data).hexdigest()
    return json.loads(data)
def check(name, value):
    checks.append({'check':name,'pass':bool(value)})
    if not value:failures.append(name)
def decompressed(path):
    path=Path(path);data=path.read_bytes();inputs[str(path)]=hashlib.sha256(data).hexdigest()
    raw=gzip.decompress(data);uncompressed[str(path)]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
protocol=read(HERE/'protocol.json'); summary=read(HERE/'summary.json')
protected=read(HERE/'protected_hashes.json'); candidate=read(protocol['candidate_path'])
fit_protocol=read(Path(protocol['candidate_path']).parent/'protocol.json')
fit_summary=read(Path(protocol['candidate_path']).parent/'fit_summary.json')
fixed=protocol['fixed_files_sha256']
rule_path=next(k for k in fixed if k.endswith('acceptance/acceptance_rule.json'))
sampling_path=next(k for k in fixed if k.endswith('sampling/protocol.json'))
orientation_path=next(k for k in fixed if k.endswith('orientation/protocol.json'))
rule=read(rule_path); sampling=read(sampling_path); orientation_protocol=read(orientation_path)
original_sampling=read(next(k for k in fixed if k.endswith('sampling/sampling_uncertainty.json')))
original_orientation=decompressed(next(k for k in fixed if k.endswith('orientation/orientation_report_full.json.gz')))
original_acceptance=read(next(k for k in fixed if k.endswith('acceptance/acceptance_application.json')))
check('unchanged_rule_SHA',inputs[rule_path]=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4')
check('validation_source_binding',sha(HERE/'validate_candidate.py')==protocol['script_sha256'])
check('candidate_binding',inputs[protocol['candidate_path']]==summary['candidate_sha256'])
check('validation_protocol_binding',inputs[str(HERE/'protocol.json')]==summary['validation_protocol_sha256'])
check('fit_protocol_binding',inputs[str(Path(protocol['candidate_path']).parent/'protocol.json')]==summary['fit_protocol_sha256']==protocol['candidate_protocol_expected_sha256'])
check('summary_script_binding',sha('/private/tmp/summarize_hug_feasible_validation.py')==summary['summary_script_sha256'])
check('protected_snapshot_binding',inputs[str(HERE/'protected_hashes.json')]==protocol['protected_snapshot_sha256'])
current_mismatches={path:{'expected':h,'current':sha(path)} for path,h in protected.items() if sha(path)!=h}
fixed_mismatches={path:{'expected':h,'current':sha(path)} for path,h in fixed.items() if sha(path)!=h}
check('all_89_protected_current',len(protected)==protocol['protected_file_count']==summary['protected_numeric_dependency_count']==89 and not current_mismatches)
check('all_fixed_sources_current',not fixed_mismatches)
check('summary_scope_static',summary['physics_steps']==0 and not summary['new_eval_data'] and not summary['training'] and summary['native_grasp_failure_labels']=='NOT_TESTED' and summary['continuous_physical_contact_equivalence']=='UNRESOLVED')

def report(name):
    path=HERE/name;value=decompressed(path);binding=read(str(path)+'.binding.json')
    check(name+'_compressed_binding',inputs[str(path)]==binding['gzip_sha256']==summary['measurement_files_sha256'][str(path)])
    check(name+'_uncompressed_binding',uncompressed[str(path)]==binding['uncompressed_sha256'])
    check(name+'_protocol_candidate_binding',value['protocol_sha256']==binding['protocol_sha256']==inputs[str(HERE/'protocol.json')] and value['candidate_sha256']==binding['candidate_sha256']==inputs[protocol['candidate_path']])
    check(name+'_case_count_binding',binding['case_count']==len(value['cases']))
    check(name+'_script_binding',value['script_sha256']==protocol['script_sha256'])
    check(name+'_all_zero_actual_step_guards',value['physics_steps']==0 and value['guards']['actual']==0 and value['guards']['selftest']==3 and not value['new_eval_data'] and not value['new_inference'] and not value['training'])
    check(name+'_protected_status',value['all_source_geometry_and_protected_hashes_unchanged'] and value['protected_source_count']==89)
    check(name+'_no_pilot_orientation_skip',not value['skip_orientation_pilot'])
    return value

preflight=report('candidate_validation_preflight_only.json.gz')
audits=[report('candidate_validation_8,12,16.json.gz'),report('candidate_validation_20,24,28.json.gz')]
indices=[8,12,16,20,24,28]
check('all_six_preflight_gate',preflight['preflight_only'] and [c['source_inventory_index'] for c in preflight['cases']]==indices and all(c['engineering_guard_independent_audit']['source_preflight_status']=='PASS' and c['resolutions']=={} for c in preflight['cases']))
cases=[c for r in audits for c in r['cases']]
check('declared_cohorts_complete_unique',[[c['source_inventory_index'] for c in r['cases']] for r in audits]==protocol['allowed_full_audit_cohorts'] and [c['source_inventory_index'] for c in cases]==indices)
check('all_reports_nonpilot',all(not r['preflight_only'] for r in audits))
native_sampling={c['native_sha256']:c for c in original_sampling['cases']}
native_orientation={c['native_sha256']:c for c in original_orientation['cases']}
native_acceptance={(r['native_sha256'],str(r['density_mm'])):r for r in original_acceptance['records']}
targets={c['source_inventory_index']:c for c in candidate['cases']}
plans={c['source_inventory_index']:c for c in orientation_protocol['cases']}
preflights={c['source_inventory_index']:c for c in preflight['cases']}
urdf=next(k for k in fit_protocol['source_sha256'] if k.endswith('.urdf'))
physical={j.get('name'):[float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))] for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
strict=Counter();pointwise=Counter();base_orientation=Counter(); rows=[]; witness_count=0; maximum_score_difference=0.
parts_for_sum=['palm','index','middle','ring','little','thumb']
eps=str(sampling['primary_allowance_m'])
for case in cases:
    index=case['source_inventory_index'];native=case['native_sha256'];target=targets[index]
    start=f'case_{index}_'; eng=case['engineering_guard_independent_audit'];root=case['root_policy']
    check(start+'native_identity',native==target['native_sha256']==plans[index]['native_sha256'])
    check(start+'source_preflight_PASS',eng['source_preflight_status']=='PASS' and eng['native_landmark_errors_independently_recomputed'])
    check(start+'preflight_and_full_preflight_identical',{k:v for k,v in case.items() if k!='resolutions'}=={k:v for k,v in preflights[index].items() if k!='resolutions'})
    cap=np.array(eng['original_tip_caps_sqrt_einsum_m']);final=np.array(eng['candidate_tip_norms_sqrt_einsum_m'])
    orig_vectors=np.array(target['baseline_fresh_diagnostics']['task_link_error_vectors_m'])[:5]
    final_vectors=np.array(target['selected_fresh_diagnostics']['task_link_error_vectors_m'])[:5]
    ref_cap=np.sqrt(np.einsum('ij,ij->i',orig_vectors,orig_vectors));ref_final=np.sqrt(np.einsum('ij,ij->i',final_vectors,final_vectors))
    check(start+'caps_against_independent_measurement',np.all(final<=cap) and eng['individual_fingertip_arithmetic_allowance_m']==0 and np.array_equal(cap,ref_cap) and np.array_equal(final,ref_final))
    check(start+'strict_actual_mean',float(final.mean())==eng['actual_mean_fingertip_error_m'] and final.mean()<=.010 and eng['actual_mean_guard_pass'])
    check(start+'native_placement_source_FK_gates',eng['original_source_URDF_vs_scene_max_link_error_m']<1.1e-7 and eng['original_source_URDF_vs_scene_palm_rotation_maxabs']<1e-6 and eng['original_native_wrist_to_palm_position_error_m']<1e-7 and root['native_wrist_to_palm_position_error_m']<1e-7 and root['declared_world_operator_times_base_palm_rotation_maxabs']<1e-6 and root['independent_fixed_URDF_vs_scene_max_link_error_m']<1.1e-7)
    selection=eng['recorded_solver_selection']
    check(start+'new_method_selection_trace',selection['recorded_seed_restoration_selection_verified'] and selection['full_and_FK_terminal_events_fresh_verified'] and selection['exact_computed_all_five_caps_verified_in_recorded_assessments'] and not eng['fallback_original_target_exact'])
    for j,item in enumerate(case['candidate_joint_range_checks']):
        q=target['qpos_rad'][j];q0=target['original_qpos_rad'][j];lo,hi=physical[item['joint']]
        check(start+'range_'+item['joint'],item['joint']==target['joint_names'][j] and item['source_URDF_range_rad']==[lo,hi] and lo<=q<=hi and abs(q-q0)<=.35 and item['delta_from_original_rad']==q-q0 and item['strict_scene_overflow_rad']<=item['scene_representation_difference_rad']+1e-9)
    check(start+'input_sources_current',all(sha(path)==h for path,h in case['input_sha256'].items()))
    check(start+'resolution_coverage',sorted(map(float,case['resolutions']))==sorted(plans[index]['densities_mm']))
    summarized=next(c for c in summary['records'] if c['index']==index)
    check(start+'summary_source_gate',summarized['independent_engineering_and_source_preflight']=='PASS' and summarized['case_id']==case['case_id'])
    for density,res in case['resolutions'].items():
        label=start+'density_'+density+'_'; prox_states=[];normal_states=[];base_normal_states=[]
        record=next(r for r in summarized['resolutions'] if r['density_mm']==float(density))
        cache_path=next(k for k in orientation_protocol['query_cache_sha256'] if k.endswith(f"{case['case_id']}_density{density}mm_queries.npz"))
        check(label+'query_cache_bound',res['query_cache_sha256']==orientation_protocol['query_cache_sha256'][cache_path]==sha(cache_path))
        check(label+'query_count_summary',record['query_count']==res['query_count']==res['orientation']['query_count'])
        check(label+'baseline_sampling_original',res['baseline_conditional_sampling']==native_sampling[native]['resolutions'][density]['numerical_allowances'])
        check(label+'baseline_orientation_original',res['baseline_orientation']==native_orientation[native]['resolutions'][density])
        check(label+'baseline_acceptance_original',res['baseline_acceptance']==native_acceptance[native,density])
        for band in rule['bands_mm']:
            bandkey=str(band);bandlabel=label+f'band_{band}_'
            pp=res['conditional_sampling'][eps]['bands'][bandkey]['independent_parts']
            bp=res['baseline_conditional_sampling'][eps]['bands'][bandkey]['independent_parts']
            summary_band=next(b for b in record['bands'] if b['band_mm']==band)
            for part in rule['anatomical_parts']:
                pair=pp[part];lost=pair['lost_native_area'];added=pair['new_mapped_area']
                state='FAIL' if lost['lower_mm2']>0 or added['lower_mm2']>0 else 'PASS' if lost['upper_mm2']==0 and added['upper_mm2']==0 else 'UNRESOLVED'
                prox_states.append(state)
                frozen=next(c for c in res['frozen_acceptance_application']['proximity_checks'] if c['part']==part and c['band_mm']==band)
                check(bandlabel+part+'_frozen_rule',frozen['status']==state and frozen['lost_area_lower_mm2']==lost['lower_mm2'] and frozen['added_area_lower_mm2']==added['lower_mm2'] and frozen['lost_area_upper_mm2']==lost['upper_mm2'] and frozen['added_area_upper_mm2']==added['upper_mm2'])
                sb=summary_band['per_part'][part]
                check(bandlabel+part+'_summary_values',sb=={'lost_lower_mm2':lost['lower_mm2'],'added_lower_mm2':added['lower_mm2'],'original_lost_lower_mm2':bp[part]['lost_native_area']['lower_mm2'],'original_added_lower_mm2':bp[part]['new_mapped_area']['lower_mm2']})
            for key,field,table in [('sum_lost_lower_mm2','lost_native_area',pp),('sum_added_lower_mm2','new_mapped_area',pp),('original_sum_lost_lower_mm2','lost_native_area',bp),('original_sum_added_lower_mm2','new_mapped_area',bp)]:
                check(bandlabel+key,summary_band[key]==sum(table[part][field]['lower_mm2'] for part in parts_for_sum))
            orientation=res['orientation']['bands'][bandkey]['common_same_label']['pointwise_sign_check']
            baseline=res['baseline_orientation']['bands'][bandkey]['common_same_label']['pointwise_sign_check']
            flips=sum(s['query_count'] for s in orientation['flip_denominators'].values())
            base_flips=sum(s['query_count'] for s in baseline['flip_denominators'].values())
            normal_state='FAIL' if flips else 'UNRESOLVED'; normal_states.append(normal_state);base_normal_states.append('FAIL' if base_flips else 'UNRESOLVED')
            frozen_o=next(c for c in res['frozen_acceptance_application']['orientation_checks'] if c['band_mm']==band)
            check(bandlabel+'orientation_summary_and_frozen',summary_band['sampled_reliable_sign_reversal_count']==flips==frozen_o['reliable_sampled_sign_reversal_count'] and summary_band['original_sampled_reliable_sign_reversal_count']==base_flips and frozen_o['pointwise_status']==normal_state==summary_band['pointwise_orientation_status'])
            for stratum,witnesses in orientation['witnesses'].items():
                for wi,w in enumerate(witnesses):
                    witness_count+=1
                    obj=np.array(w['object_outward_normal']);ns=-float(obj@np.array(w['native_outward_normal']));ms=-float(obj@np.array(w['mapped_outward_normal']))
                    maximum_score_difference=max(maximum_score_difference,abs(ns-w['native_facing_compatibility']),abs(ms-w['mapped_facing_compatibility']))
                    dead=orientation_protocol['facing_dot_numerical_deadzone']
                    signs=(ns>dead and ms < -dead) if stratum=='native_facing_mapped_away' else (ns < -dead and ms>dead)
                    check(bandlabel+f'witness_{stratum}_{wi}',signs and w['native_part']==w['mapped_part'] and w['native_and_mapped_query_exterior'] and w['both_closest_outside_visual'] and w['excluded_tie_buried_degenerate_flags_all_false'] and w['native_unsigned_distance_m']<=band/1000 and w['mapped_unsigned_distance_m']<=band/1000)
        prox='FAIL' if 'FAIL' in prox_states else 'PASS' if all(s=='PASS' for s in prox_states) else 'UNRESOLVED'
        normal='FAIL' if 'FAIL' in normal_states else 'UNRESOLVED'
        strict_state='FAIL' if prox=='FAIL' or normal=='FAIL' else 'UNRESOLVED'
        base_state='FAIL' if 'FAIL' in base_normal_states else 'UNRESOLVED'
        strict[strict_state]+=1;pointwise[normal]+=1;base_orientation[base_state]+=1
        a=res['frozen_acceptance_application']
        check(label+'all_statuses_recomputed',record['strict_status']==a['strict_relationship_status_conditional']==strict_state and record['proximity_status']==a['proximity_status_conditional']==prox and record['pointwise_orientation_status']==a['pointwise_orientation_status']==normal)
        check(label+'continuous_equivalence_not_claimed',record['continuous_contact_equivalence']==a['full_physical_contact_equivalence']=='UNRESOLVED')
        rows.append({'index':index,'density_mm':float(density),'strict_status':strict_state,'pointwise_orientation_status':normal,'original_pointwise_orientation_status':base_state})
check('all_nine_status_counts',len(rows)==9 and {s:strict[s] for s in ['PASS','FAIL','UNRESOLVED']}==summary['strict_status_counts']=={'PASS':0,'FAIL':9,'UNRESOLVED':0})
check('all_nine_pointwise_counts',{s:pointwise[s] for s in ['PASS','FAIL','UNRESOLVED']}==summary['pointwise_orientation_status_counts']=={'PASS':0,'FAIL':9,'UNRESOLVED':0})
check('prior_orientation_7FAIL_2UNRESOLVED',base_orientation['FAIL']==7 and base_orientation['UNRESOLVED']==2 and base_orientation['PASS']==0)
check('witness_dot_score_consistency',maximum_score_difference<=1e-12)
check('summary_fit_values_bound',summary['new_fit_full_surface_evaluations']==fit_summary['total_full_surface_evaluations']==18 and summary['new_fit_runtime_seconds']==fit_summary['runtime_seconds'] and summary['all_six_feasible_improving_poses_selected'])
check('parsed_inputs_still_unchanged',all(sha(path)==h for path,h in inputs.items()))
out={'schema':'hug_feasible_final_read_only_independent_review_v1','review_utc':datetime.now(timezone.utc).isoformat(),
     'result':'PASS' if not failures else 'FAIL','input_sha256':inputs,'uncompressed_sha256':uncompressed,
     'review_script_sha256':sha(__file__),'checks':checks,'failed_checks':failures,'protected_current_mismatches':current_mismatches,'fixed_source_mismatches':fixed_mismatches,
     'resolution_rows':rows,'strict_status_counts':dict(strict),'pointwise_orientation_status_counts':dict(pointwise),'prior_orientation_counts':dict(base_orientation),
     'witness_records_recomputed':witness_count,'maximum_witness_dot_score_difference':maximum_score_difference,
     'scope':'Report bindings, source status, exact cap arithmetic, frozen-rule decisions, baseline/summary values and witness normal-dot signs. No FK or full geometry replay.',
     'scientific_result':'All six numerical/engineering preflights PASS; all nine strict surface and pointwise orientation checks FAIL. No native-contact or label-equivalence certificate.',
     'geometry_replays':0,'physics_steps':0,'new_eval_data':False}
(HERE/'independent_final_read_only_review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
(HERE/'INDEPENDENT_FINAL_READ_ONLY_REVIEW.md').write_text(f'''# Final independent read-only review (2026-10-14)

Review result: **{out['result']}** ({len(checks)} checks; {len(failures)} failures).

All six source/engineering preflights pass. All 30 exact original-relative
fingertip caps and strict 10-mm means are consistent with the independent
native/scene/source-FK measurements and the recorded candidate vectors.
The declared six coarse comparisons and three fine repeats are complete.

Reapplying the unchanged rule gives **0 PASS / 9 FAIL** for strict surface
mapping and **0 PASS / 9 FAIL** for pointwise orientation. The original
pointwise orientation baseline was 7 FAIL / 2 UNRESOLVED. All per-part,
per-band lost/added lower areas and orientation counts in the compact summary
match the bound raw reports and original baseline. {witness_count} stored
orientation witness signs were recomputed from their normal vectors.

Compressed/uncompressed hashes, protocol/candidate/code bindings, all 89
protected numeric dependencies, static guards and zero actual physics-step
counts pass. No case was dropped and no tolerance or acceptance rule changed.
No FK/full-geometry rerun occurred in this review.

The new feasible improving candidates solve the fingertip guard refusal;
they do **not** establish faithful full-surface/contact mapping or failure-label
equivalence. Scientific strict mapping acceptance remains failed.

Details: `independent_final_read_only_review.json`. Candidate SHA256:
`{summary['candidate_sha256']}`; validation protocol SHA256:
`{summary['validation_protocol_sha256']}`.
''')
print(json.dumps({'result':out['result'],'checks':len(checks),'failures':failures,'strict':dict(strict),'pointwise':dict(pointwise),'baseline_orientation':dict(base_orientation),'witnesses':witness_count},indent=2))
