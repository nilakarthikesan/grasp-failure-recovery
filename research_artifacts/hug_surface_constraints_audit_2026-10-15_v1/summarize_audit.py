"""Independent arithmetic read-back of fresh static HUG surface audit reports."""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json,math
OUT=Path(__file__).parent
checks=[];inputs={}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    path=Path(path);inputs[str(path)]=sha(path);return json.loads(path.read_bytes())
def check(label,condition):
    checks.append({'check':label,'pass':bool(condition)})
    if not condition:raise AssertionError(label)
def load_report(name):
    path=OUT/name;inputs[str(path)]=sha(path);raw=gzip.decompress(path.read_bytes());r=json.loads(raw);b=read(str(path)+'.binding.json')
    check(name+'_gzip',inputs[str(path)]==b['gzip_sha256']);check(name+'_json',hashlib.sha256(raw).hexdigest()==b['uncompressed_sha256']);check(name+'_candidate_protocol',r['candidate_sha256']==b['candidate_sha256']==inputs[str(candidate_path)] and r['protocol_sha256']==b['protocol_sha256']==inputs[str(OUT/'protocol.json')]);check(name+'_count',len(r['cases'])==b['case_count']);check(name+'_static_only',r['physics_steps']==0 and r['guards']=={'selftest':3,'actual':0} and not r['new_eval_data'] and not r['new_inference'] and not r['training']);check(name+'_all_bound',r['all_source_geometry_and_protected_hashes_unchanged']);return r
def fitting_row_summary(diag):
    prox=diag['proximity']['passed'];ori=diag['exact_orientation']['passed'];valid=diag['exact_orientation_fields']['valid'];literal=diag['literal_surface_pass']
    check('fitting_row_count_lengths',len(ori)==len(valid) and len(literal)==len(prox)+len(ori))
    check('fitting_row_pass_aggregate',literal==prox+ori and diag['all_witness_constraints_pass']==all(literal))
    return {'proximity_retained_rows':len(prox),'proximity_rows_pass':sum(prox),'proximity_rows_fail':len(prox)-sum(prox),'orientation_retained_rows':len(ori),'orientation_rows_pass':sum(ori),'orientation_rows_fail':len(ori)-sum(ori),'orientation_rows_eligible':sum(valid),'orientation_witness_coverage':diag['orientation_witness_coverage'],'all_retained_rows_pass':diag['all_witness_constraints_pass'],'rank':diag['rank'],'row_coverage_is_full_grid_acceptance':False}
p=read(OUT/'protocol.json');candidate_path=Path(p['candidate_path']);candidate=read(candidate_path)
check('frozen_candidate',inputs[str(candidate_path)]==p['candidate_sha256']);check('frozen_script',sha(OUT/'validate_candidate.py')==p['script_sha256'])
check('frozen_acceptance',next(v for k,v in p['fixed_files_sha256'].items() if k.endswith('/acceptance/acceptance_rule.json'))=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4')
for path,want in {**p['fixed_files_sha256'],**p['candidate_bound_files_sha256'],**read(OUT/'protected_hashes.json')}.items():check('hash_'+path,sha(path)==want)
preflight=load_report('candidate_validation_preflight_only.json.gz');reports=[load_report('candidate_validation_'+','.join(map(str,c))+'.json.gz') for c in p['allowed_full_audit_cohorts']]
check('preflight_all6',[c['source_inventory_index'] for c in preflight['cases']]==p['case_indices'] and preflight['preflight_only'])
records=[c for r in reports for c in r['cases']];check('all_six_exact',[c['source_inventory_index'] for c in records]==p['case_indices']);check('full_reports',all(not r['preflight_only'] and not r['skip_orientation_pilot'] for r in reports))
pre={c['source_inventory_index']:c for c in preflight['cases']};targets={c['source_inventory_index']:c for c in candidate['cases']};summary=[];strict=Counter();prox=Counter();normals=Counter();witnesses=0;unique_witnesses=set();max_dot_difference=0.
for c in records:
    i=c['source_inventory_index']; prefix='case_'+str(i)+'_';eng=c['engineering_guard_independent_audit'];root=c['root_policy'];row={'index':i,'case_id':c['case_id'],'audit_selection_role':c['audit_selection_role'],'palm_displacement_world_mm':[v*1000 for v in root['palm_displacement_world_m']],'palm_displacement_norm_mm':root['palm_displacement_norm_m']*1000,'complete_original_fallback':eng['fallback_original_complete_pose_exact'],'original_mean_tip_error_mm':sum(eng['original_tip_caps_sqrt_einsum_m'])/5*1000,'selected_mean_tip_error_mm':eng['actual_mean_fingertip_error_m']*1000,'resolutions':{}}
    row['exact_original_tip_caps_pass']=True;row['original_tip_caps_mm']=[v*1000 for v in eng['frozen_authoritative_tip_caps_sqrt_einsum_m']];row['selected_tip_errors_mm']=[v*1000 for v in eng['candidate_tip_norms_sqrt_einsum_m']]
    row['fit_retained_witness_diagnostics']={'scope':'Fixed capped saved witness examples, including duplicate nested-band/history rows. Their satisfaction is not a full-grid mapping acceptance claim. Counts are recomputed from supplied per-row truth; separate independent fit review verifies arithmetic and exact geometry.','baseline':fitting_row_summary(targets[i]['baseline_constraint_diagnostics']),'selected':fitting_row_summary(targets[i]['selected_constraint_diagnostics'])}
    dom=c['candidate_domain_audit'];check(prefix+'declared_domain',dom['scaled_state_and_world_translation_within_declared_frozen_bounds'] and dom['domain_endpoint_arithmetic_allowance']==0 and all(b[0]<=v<=b[1] for v,b in zip(dom['selected_state_scaled'],dom['bounds_scaled'])) and all(b[0]<=v<=b[1] for v,b in zip(root['palm_displacement_world_m'],dom['world_displacement_AABB_m'])))
    check(prefix+'role_preserved',c['audit_selection_role']['audited_pose_role']==targets[i]['audited_pose_role'] and c['audit_selection_role']['constraint_selection_accepted']==targets[i]['constraint_selection_accepted'])
    check(prefix+'preflight_matches',{k:v for k,v in c.items() if k!='resolutions'}=={k:v for k,v in pre[i].items() if k!='resolutions'})
    check(prefix+'exact_caps',all(a<=b for a,b in zip(eng['candidate_tip_norms_sqrt_einsum_m'],eng['original_tip_caps_sqrt_einsum_m'])) and eng['individual_fingertip_arithmetic_allowance_m']==0 and all(a<=b for a,b in zip(eng['candidate_tip_norms_sqrt_einsum_m'],eng['frozen_authoritative_tip_caps_sqrt_einsum_m'])))
    check(prefix+'mean',eng['actual_mean_fingertip_error_m']<=.01 and abs(sum(eng['candidate_tip_norms_sqrt_einsum_m'])/5-eng['actual_mean_fingertip_error_m'])<1e-15)
    check(prefix+'root_FK',root['root_orientation_bitwise_original'] and root['translated_anchor_to_palm_position_error_m']<=1e-12 and root['declared_world_operator_times_base_palm_rotation_maxabs']<1e-6 and root['independent_fixed_URDF_vs_scene_max_link_error_m']<1.1e-7)
    check(prefix+'nine_preselected_densities',sorted(c['resolutions'])==(['1','2'] if i in (8,16,24) else ['2']))
    for den,r in c['resolutions'].items():
        pointwise_fail=False;ps=[];out=r['frozen_acceptance_application'];total_flips=0;part_losses={};part_adds={}
        for band in ('1','2','5'):
            parts=r['conditional_sampling']['1e-08']['bands'][band]['independent_parts']
            for part in ('palm','index','middle','ring','little','thumb'):
                x=parts[part];lo=x['lost_native_area']['lower_mm2'];new=x['new_mapped_area']['lower_mm2'];lu=x['lost_native_area']['upper_mm2'];nu=x['new_mapped_area']['upper_mm2'];state='FAIL' if lo>0 or new>0 else 'PASS' if lu==nu==0 else 'UNRESOLVED';ps.append(state)
                declared=next(z for z in out['proximity_checks'] if z['part']==part and str(z['band_mm'])==band)
                check(prefix+den+'_'+part+'_'+band,declared['status']==state and 0<=lo<=lu and 0<=new<=nu)
                part_losses[part+'_'+band]=lo;part_adds[part+'_'+band]=new
            s=r['orientation']['bands'][band]['common_same_label']['pointwise_sign_check'];flips=sum(x['query_count'] for x in s['flip_denominators'].values());total_flips+=flips;pointwise_fail|=flips>0
            declared=next(z for z in out['orientation_checks'] if str(z['band_mm'])==band);check(prefix+den+'_normal_'+band,declared['reliable_sampled_sign_reversal_count']==flips and declared['pointwise_status']==('FAIL' if flips else 'UNRESOLVED'))
            for direction,ws in s['witnesses'].items():
                witnesses+=len(ws)
                for w in ws:
                    unique_witnesses.add((i,den,w['query_index'],direction))
                    native_dot=-sum(a*b for a,b in zip(w['native_outward_normal'],w['object_outward_normal']));mapped_dot=-sum(a*b for a,b in zip(w['mapped_outward_normal'],w['object_outward_normal']))
                    max_dot_difference=max(max_dot_difference,abs(native_dot-w['native_facing_compatibility']),abs(mapped_dot-w['mapped_facing_compatibility']))
                    expected=native_dot>1e-6 and mapped_dot<-1e-6 if direction=='native_facing_mapped_away' else native_dot<-1e-6 and mapped_dot>1e-6
                    check(prefix+den+'_witness_'+band+'_'+direction+'_'+str(w['query_index']),expected and w['native_part']==w['mapped_part'] and w['native_and_mapped_query_exterior'] and w['both_closest_outside_visual'] and w['excluded_tie_buried_degenerate_flags_all_false'] and max_dot_difference<1e-12)
        pstate='FAIL' if 'FAIL' in ps else 'PASS' if set(ps)=={'PASS'} else 'UNRESOLVED';nstate='FAIL' if pointwise_fail else 'UNRESOLVED';sstate='FAIL' if pstate=='FAIL' or nstate=='FAIL' else 'UNRESOLVED'
        check(prefix+den+'_aggregate',out['proximity_status_conditional']==pstate and out['pointwise_orientation_status']==nstate and out['strict_relationship_status_conditional']==sstate)
        check(prefix+den+'_scope',out['full_physical_contact_equivalence']=='UNRESOLVED' and out['failure_label_preservation']=='NOT_TESTED' and out['independent_unseen_object_validation']=='NOT_TESTED')
        strict[sstate]+=1;prox[pstate]+=1;normals[nstate]+=1
        baseline_parts=r['baseline_conditional_sampling']['1e-08']['bands']['2']['independent_parts'];audited_parts=r['conditional_sampling']['1e-08']['bands']['2']['independent_parts'];compare2={}
        for part in ('palm','index','middle','ring','little','thumb'):
            before=baseline_parts[part];after=audited_parts[part]
            compare2[part]={'original_lost_lower_mm2':before['lost_native_area']['lower_mm2'],'audited_lost_lower_mm2':after['lost_native_area']['lower_mm2'],'original_added_lower_mm2':before['new_mapped_area']['lower_mm2'],'audited_added_lower_mm2':after['new_mapped_area']['lower_mm2'],'original_lost_upper_mm2':before['lost_native_area']['upper_mm2'],'audited_lost_upper_mm2':after['lost_native_area']['upper_mm2'],'original_added_upper_mm2':before['new_mapped_area']['upper_mm2'],'audited_added_upper_mm2':after['new_mapped_area']['upper_mm2'],'original_relationship_ambiguous_area_mm2':before['whole_relationship_ambiguous_area_mm2'],'audited_relationship_ambiguous_area_mm2':after['whole_relationship_ambiguous_area_mm2']}
        old_sign=r['baseline_orientation']['bands']['2']['common_same_label']['pointwise_sign_check'];new_sign=r['orientation']['bands']['2']['common_same_label']['pointwise_sign_check'];flips2={'original':{k:v['query_count'] for k,v in old_sign['flip_denominators'].items()},'audited':{k:v['query_count'] for k,v in new_sign['flip_denominators'].items()},'original_eligible_query_count':old_sign['eligible_outside_visual_denominator']['query_count'],'audited_eligible_query_count':new_sign['eligible_outside_visual_denominator']['query_count']}
        row['resolutions'][den]={'query_count':r['query_count'],'same_2mm_band_part_comparison_original_vs_audited':compare2,'same_2mm_band_reliable_orientation_flips':flips2,'comparison_limitations':'Individual anatomical part regions can overlap; not summed. Decreasing a certified lower bound does not alone prove decreased total mismatch area. Orientation query eligibility can change; counts include denominators.','proximity':pstate,'pointwise_orientation':nstate,'strict_relationship':sstate,'sign_reversals_summed_across_nested_bands_not_unique':total_flips,'runtime_seconds':r['runtime_seconds'],'positive_lower_bound_lost_area_by_part_band_mm2':{k:v for k,v in part_losses.items() if v>0},'positive_lower_bound_added_area_by_part_band_mm2':{k:v for k,v in part_adds.items() if v>0}}
    summary.append(row)
check('nine_measurements',sum(strict.values())==9)
report={'schema':'hug_surface_constraints_fresh_surface_audit_summary_v1','candidate_path':str(candidate_path),'candidate_sha256':inputs[str(candidate_path)],'protocol_sha256':inputs[str(OUT/'protocol.json')],'acceptance_rule_sha256':'87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4','fresh_full_grid_measurements':9,'reused_full_pose_reports':0,'native_query_cache_reused_with_fresh_target_fields':True,'strict_status_counts':dict(strict),'proximity_status_counts':dict(prox),'pointwise_orientation_status_counts':dict(normals),'all_six_independent_source_and_tip_gates_pass':True,'records':summary,'saved_witness_records_across_nested_bands':witnesses,'distinct_saved_witnesses_by_case_density_query_direction':len(unique_witnesses),'independently_recomputed_witness_dot_maxabs_difference':max_dot_difference,'all_checks_pass':True,'consistency_checks':len(checks),'checks':checks,'inputs_sha256':inputs,'cohort_runtime_seconds':[r['runtime_seconds'] for r in reports],'preflight_runtime_seconds':preflight['runtime_seconds'],'physics_steps':0,'new_inference':False,'new_eval_data':False,'training':False,'continuous_contact_equivalence':'UNRESOLVED','failure_label_preservation':'NOT_TESTED','independent_unseen_objects':'NOT_TESTED'}
(OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ('fresh_full_grid_measurements','strict_status_counts','proximity_status_counts','pointwise_orientation_status_counts','consistency_checks','cohort_runtime_seconds')},indent=2))
