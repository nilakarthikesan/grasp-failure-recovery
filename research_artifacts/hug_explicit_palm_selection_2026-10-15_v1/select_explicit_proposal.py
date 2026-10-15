"""One post-hoc provenance coverage supplement; no fitting or surface queries."""
from pathlib import Path
import copy, gzip, hashlib, importlib.util, json, time, xml.etree.ElementTree as ET
import numpy as np

OUT = Path(__file__).parent
PRIOR = Path('/private/tmp/hug_free_palm_review_2026-10-15_v1/diagnostic_selection')
spec = importlib.util.spec_from_file_location('frozen_diagnostic_selector', PRIOR/'select_and_verify.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
v=s.v; sha=s.sha; read=s.read; check=s.check; same=s.same; norm=s.norm

def qualifies(record):
    return bool(record['predeclared_translation_seed'] and record['actual_shift_nonzero'] and record['proposal_shift_nonzero'] and record['literal_tip_feasible'] and record['complete_full_callback'])

def select(pool):
    eligible=[record for record in pool if qualifies(record)]
    check(bool(eligible), 'at least one explicit-translation eligible diagnostic')
    return min(eligible, key=lambda record:(record['objective'], record['method_order'], record['callback_sequence'], record['proposal_sequence']))

def main():
    started=time.perf_counter();p=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json')
    check(sha(__file__)==p['script_sha256'],'frozen supplemental script')
    check(not (OUT/'candidate_targets.json').exists(),'never replace existing supplemental candidates')
    for path,expected in p['input_sha256'].items():check(sha(path)==expected,'frozen input '+path)
    with (OUT/'execution_started.json').open('x') as stream:json.dump({'protocol_sha256':ph,'inputs_verified_before_ranking':True,'new_fitting':False},stream)
    methods=[]
    for order,method in enumerate(p['methods']):
        compact=read(method['candidate_path']);protocol=read(method['protocol_path'])
        check(compact['protocol_sha256']==sha(method['protocol_path']),'primary compact/protocol binding')
        cases={}
        for provenance in compact['cohort_provenance']:
            raw=gzip.decompress(Path(provenance['gzip_path']).read_bytes())
            check(sha(provenance['gzip_path'])==provenance['gzip_sha256'] and hashlib.sha256(raw).hexdigest()==provenance['sha256'],'complete cohort byte bindings')
            for row in json.loads(raw)['cases']:
                check(row['source_inventory_index'] not in cases,'unique primary case')
                cases[row['source_inventory_index']]=row
        check(sorted(cases)==[8,12,16,20,24,28],'full six primary cases retained')
        methods.append({'order':order,'method':method,'protocol':protocol,'cases':cases})
    for method in methods[1:]:
        for key in ('surface_configuration','direction_configuration','direction_guard_policy','state_coordinates'):
            check(method['protocol'][key]==methods[0]['protocol'][key],'identical comparable configuration '+key)
        check(method['protocol']['solver_configuration']['f_scale']==methods[0]['protocol']['solver_configuration']['f_scale'],'identical robust scale')
    inventory=read(p['inventory_path'])['modern_cases'];historical={r['source_inventory_index']:r for r in read(p['historical_candidate_path'])['cases']}
    urdf=Path(p['source_urdf']);xml=ET.parse(urdf).getroot();source=v.ShadowPoseValidator(urdf.parent.parent)
    output=[];reports=[]
    for index in p['case_indices']:
        base=methods[0]['cases'][index];caps=np.asarray(base['authoritative_original_tip_caps_sqrt_einsum_m'])
        same(caps,historical[index]['old_original_five_tip_caps_sqrt_einsum_m'],'exact historical caps')
        pool=[];dedup={};trace_reviews=[];mapper_outputs=[];excluded_zero_request=0;excluded_zero_actual=0;excluded_nonseed=0
        for method in methods:
            row=method['cases'][index]
            for key in ('native_sha256','joint_names','original_qpos_rad','original_root_target_position_m','original_root_target_quaternion_wxyz','authoritative_original_tip_caps_sqrt_einsum_m','surface_reference_by_part','direction_reference_by_part','baseline_fresh_diagnostics','domain'):
                check(row[key]==base[key],'identical per-case source basis '+key)
            check(row['solver_audit']['exact_original_fallback'],'supplement observes original fallback without replacing it')
            trace_reviews.append({'method':method['method']['name'],'review':v.trace_review(row,caps,p)})
            mapper_outputs.append({'method':method['method']['name'],'exact_original_fallback':True,'returned_state_scaled':row['selected_state_scaled'],'returned_qpos_rad':row['qpos_rad'],'returned_root_position_m':row['root_target_position_m'],'this_returned_pose_is_not_rewritten':True})
            callbacks={entry['callback_sequence']:entry for entry in row['keyed_direction_evidence']['callbacks']}
            bounds=np.asarray(row['solver_audit']['bounded_correction_domain_scaled'])
            for proposal in row['solver_audit']['proposals']:
                if 'full_assessment' not in proposal:continue
                if proposal['direction']!='predeclared_seed':excluded_nonseed+=1;continue
                requested=np.asarray(proposal['proposal_state_scaled'],float)
                check(requested.shape==(27,) and np.isfinite(requested).all(),'complete finite original proposal')
                seed_index=proposal['seed_index'];seeds=np.asarray(row['solver_audit']['predeclared_seed_state_scaled'])
                check(isinstance(seed_index,int) and 0<=seed_index<len(seeds),'valid frozen seed index')
                same(requested,seeds[seed_index],'requested proposal exactly equals its frozen translation seed')
                same(requested[:24],np.zeros(24),'seed requests original joints before restoration')
                check(proposal['sweep']==0 and proposal['radius_scaled'] is None,'seed-stage provenance')
                requested_shift=.005*requested[24:]
                if not np.any(requested_shift!=0.):excluded_zero_request+=1;continue
                sequence=proposal['candidate_guard']['metadata_callback_sequence'];entry=callbacks[sequence];trace=s.trace_value(entry)
                x=np.asarray(trace['state_scaled'],float);actual_shift=.005*x[24:]
                if not np.any(actual_shift!=0.):excluded_zero_actual+=1;continue
                same(x,proposal['candidate_state_scaled'],'proposal candidate binds complete callback state')
                same(x,entry['state_scaled'],'callback index state binding')
                check(proposal['candidate_guard']['q_float64_sha256']==v.array_sha(x),'guard full state digest binding')
                check(x.shape==(27,) and np.isfinite(x).all() and np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]),'exact original state domain')
                errors=np.asarray(trace['fingertip_error_vectors_m']);tipnorm=norm(errors)
                check(np.all(tipnorm<=caps) and tipnorm.mean()<=.01,'literal historical caps and mean')
                same(errors,proposal['full_assessment']['tip_error_vectors_m'],'proposal full-assessment tip vectors bind trace')
                residual=np.asarray(trace['residuals']);scale=method['protocol']['solver_configuration']['f_scale']
                objective=float(scale*np.sum(np.hypot(scale,residual)-scale));check(np.isfinite(objective),'finite complete objective')
                same(objective,proposal['full_assessment']['objective'],'proposal full objective exactly reproduced')
                guard=s.guard_from_trace(row,trace)
                check(guard['accepted']==proposal['candidate_guard']['accepted'] and guard['refusal_reasons']==proposal['candidate_guard']['refusal_reasons'],'proposal guard provenance independently reproduced')
                origin={'proposal_translation_nonzero_before_restoration':True,'predeclared_translation_seed':True,'exact_frozen_seed_state_equality':True,'proposal_translation_world_m':requested_shift.tolist(),'proposal_state_scaled':requested.tolist(),'proposal_sequence':proposal['sequence'],'proposal_direction':proposal['direction'],'proposal_seed_index':seed_index,'proposal_seed_label':row['solver_audit']['seed_labels'][seed_index],'proposal_sweep':proposal['sweep'],'proposal_radius_scaled':proposal['radius_scaled'],'source_version':method['method']['name'],'callback_sequence':sequence,'restoration':proposal.get('restoration'),'complete_generating_proposal':copy.deepcopy(proposal),'source_protocol_sha256':sha(method['method']['protocol_path']),'source_compact_candidate_sha256':sha(method['method']['candidate_path']),'source_trace_binding':copy.deepcopy(entry)}
                record={'method_order':method['order'],'method_name':method['method']['name'],'callback_sequence':sequence,'proposal_sequence':proposal['sequence'],'state_scaled':x.tolist(),'objective':objective,'predeclared_translation_seed':True,'actual_shift_nonzero':True,'proposal_shift_nonzero':True,'literal_tip_feasible':True,'complete_full_callback':True,'explicit_translation_proposal_origin':origin,'direction_guard':guard,'source_row':row,'trace_tip_errors_m':errors.tolist()}
                key=x.tobytes()
                if key in dedup:
                    old=dedup[key];same(objective,old['objective'],'same exact state/objective across sources');check(guard==old['direction_guard'],'same exact state/guard across sources')
                    old['additional_eligible_origins'].append(origin);continue
                record['additional_eligible_origins']=[];dedup[key]=record;pool.append(record)
        chosen=select(pool)
        field,actual_errors=s.numerical_pose(chosen['source_row'],chosen['state_scaled'],inventory,urdf,xml,source,caps)
        s.near(actual_errors,chosen['trace_tip_errors_m'],1e-12,'independent chosen FK reproduces callback')
        field.update(audited_pose_role='tip_feasible_translated_diagnostic_not_selected',guarded_mapper_selected_fallback=True,direction_guard_pass=chosen['direction_guard']['accepted'],diagnostic_pose_not_promoted=True,extra_direction_guard=chosen['direction_guard'],guarded_mapper_outputs=mapper_outputs,explicit_translation_proposal_origin=chosen['explicit_translation_proposal_origin'],additional_eligible_origins=chosen['additional_eligible_origins'],selection_source={'method_name':chosen['method_name'],'method_order':chosen['method_order'],'callback_sequence':chosen['callback_sequence'],'proposal_sequence':chosen['proposal_sequence'],'objective':chosen['objective'],'source_trace_binding':chosen['explicit_translation_proposal_origin']['source_trace_binding']},scientific_mapping_acceptance='NOT_RUN',supplemental_coverage_not_replacement=True)
        field['selected_fresh_diagnostics']['objective']=chosen['objective'];field['selected_fresh_diagnostics']['objective_from_bound_full_trace']=True
        field['baseline_fresh_diagnostics']['objective']=base['baseline_fresh_diagnostics']['objective'];output.append(field)
        reports.append({'source_inventory_index':index,'eligible_distinct_states':len(pool),'excluded_fully_scored_nonseed_attempts':excluded_nonseed,'excluded_fully_scored_zero_request_attempts':excluded_zero_request,'excluded_zero_actual_shift_attempts':excluded_zero_actual,'chosen_method':chosen['method_name'],'chosen_callback_sequence':chosen['callback_sequence'],'chosen_proposal_sequence':chosen['proposal_sequence'],'chosen_objective':chosen['objective'],'chosen_palm_displacement_world_m':field['palm_displacement_world_m'],'chosen_palm_displacement_norm_m':field['palm_displacement_norm_m'],'original_mean_tip_error_m':float(caps.mean()),'chosen_mean_tip_error_m':float(np.asarray(field['selected_tip_norms_sqrt_einsum_m']).mean()),'maximum_literal_original_cap_excess_m':float(np.max(np.asarray(field['selected_tip_norms_sqrt_einsum_m'])-caps)),'direction_guard':chosen['direction_guard'],'source_FK_preflight':field['source_FK_preflight'],'trace_reviews':trace_reviews,'all_eligible_states':[{k:r[k] for k in ('method_order','method_name','callback_sequence','proposal_sequence','state_scaled','objective')} for r in pool]})
    for path,expected in p['input_sha256'].items():check(sha(path)==expected,'immutable inputs after selection '+path)
    check(v.COUNT['physics_actual']==0,'zero actual integration')
    candidate={'schema':'hug_explicit_palm_proposal_diagnostic_targets_v1','protocol_sha256':ph,'cases':output,'new_eval_data':False,'new_inference':False,'physics_steps':0,'production_changed':False,'guarded_mapper_outputs_changed':False,'original_diagnostic_selection_changed':False,'mapping_acceptance':False,'post_hoc_coverage_supplement':True,'no_more_selection_loops':True}
    (OUT/'candidate_targets.json').write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n')
    result={'schema':'hug_explicit_palm_proposal_selection_review_v1','protocol_sha256':ph,'candidate_sha256':sha(OUT/'candidate_targets.json'),'passed':True,'checks':v.COUNT,'input_count':len(p['input_sha256']),'cases':reports,'runtime_seconds':time.perf_counter()-started,'new_fitting':False,'new_surface_measurement':False,'new_eval_data':False,'physics_steps':0,'scope':'One post-hoc provenance coverage supplement. Original policy-selected diagnostics and primary fallbacks unchanged. Explicit pre-restoration translation origin independently bound to complete callback/guard evidence; selected FK and original caps reconstructed. No scientific surface acceptance claimed.'}
    (OUT/'selection_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'candidate_sha256':sha(OUT/'candidate_targets.json'),'report_sha256':sha(OUT/'selection_report.json'),'checks':v.COUNT,'cases':len(output)}))

if __name__=='__main__':main()
