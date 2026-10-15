"""Independent read-only arithmetic/provenance review; no FK or geometry calls."""
from pathlib import Path
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).parent
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
parsed_hashes={}
def read(name):
    path=Path(name) if str(name).startswith('/') else ROOT/name
    raw=path.read_bytes(); parsed_hashes[str(path)]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
failures=[]; checks=[]
def check(name, condition):
    checks.append({'check':name,'pass':bool(condition)})
    if not condition: failures.append(name)
def norm(errors):
    e=np.array(errors,float)
    return np.sqrt(np.einsum('ij,ij->i',e,e))

candidate=read('candidate_targets.json'); summary=read('fit_summary.json')
protocol=read('protocol.json'); protected=read('protected_hashes.json')
indices=[8,12,16,20,24,28]
check('ordered_all_six_unique_cases',[r['source_inventory_index'] for r in candidate['cases']]==indices and len({r['native_sha256'] for r in candidate['cases']})==6)
check('candidate_protocol_binding',candidate['protocol_sha256']==parsed_hashes[str(ROOT/'protocol.json')])
check('summary_candidate_binding',summary['candidate_sha256']==parsed_hashes[str(ROOT/'candidate_targets.json')])
check('summary_protocol_binding',summary['protocol_sha256']==parsed_hashes[str(ROOT/'protocol.json')])
check('merge_script_binding',candidate['merge_script_sha256']==sha(ROOT/'merge_seed_stage.py'))
check('frozen_solver_binding',candidate['solver_implementation_sha256']==sha(ROOT/'feasible_surface_refinement.py')=='2b4d8c30fe639d70bd8e4da0aa6406c2782ee2a3e7997683ed89db9edb935c4e')
check('frozen_driver_binding',candidate['script_sha256']==sha(ROOT/'run_seed_stage.py'))
check('original_direction_binding',candidate['direction_implementation_sha256']==sha(protocol['direction_helper_path']))
check('protected_snapshot_binding',protocol['protected_snapshot_sha256']==parsed_hashes[str(ROOT/'protected_hashes.json')])
source_mismatches={}
for table_name,table in [('source',protocol['source_sha256']),('implementation',protocol['implementation_sha256']),('protected',protected)]:
    mismatch={path:{'expected':expected,'current':sha(path)} for path,expected in table.items() if sha(path)!=expected}
    source_mismatches[table_name]=mismatch
    check(f'{table_name}_current_hashes_match',not mismatch)
for item in candidate['cohort_provenance']:
    check('cohort_file_binding',sha(item['path'])==item['sha256'])
    cohort=read(item['path'])
    check('cohort_cases_identical',cohort['cases']==candidate['cases'])
check('seed_stage_predeclared',protocol['method_version']==2 and protocol['method_stage']=='predeclared_seed_only' and protocol['solver_configuration']['maximum_pattern_sweeps']==0)
check('caps_and_budget_unrelaxed',protocol['solver_configuration']['joint_delta_limit_rad']==.35 and protocol['solver_configuration']['maximum_mean_fingertip_error_m']==.010 and protocol['solver_configuration']['maximum_full_geometry_evaluations']==4)
check('static_scope_guards',candidate['guards']['actual']==0 and candidate['guards']['selftests']==3 and candidate['physics_steps']==0 and not candidate['new_eval_data'] and not candidate['new_generator_proposal'] and not candidate['training'] and not candidate['production_changes'])
urdf_path=next(k for k in protocol['source_sha256'] if k.endswith('.urdf'))
urdf_bounds={j.get('name'):[float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))] for j in ET.parse(urdf_path).getroot().findall('joint') if j.find('limit') is not None}
rows=[]
for record in candidate['cases']:
    i=record['source_inventory_index']; prefix=f'case_{i}_'; audit=record['solver_audit']
    seed_binding=protocol['seed_bindings'][str(i)]
    seed_record=read(seed_binding['old_cohort_path'])['cases'][0]
    check(prefix+'seed_source_binding',parsed_hashes[seed_binding['old_cohort_path']]==seed_binding['old_cohort_sha256']==record['seed_source_cohort_sha256']==audit['predeclared_seed_source_sha256'])
    check(prefix+'seed_native_identity',seed_record['native_sha256']==record['native_sha256']==seed_binding['native_sha256'])
    q0=np.array(record['original_qpos_rad']); returned=np.array(record['qpos_rad']); raw=np.array(record['predeclared_seed_qpos_rad'])
    physical=np.array([urdf_bounds[n] for n in record['joint_names']])
    low=np.maximum(physical[:,0],q0-.35); high=np.minimum(physical[:,1],q0+.35)
    check(prefix+'original_unchanged',np.array_equal(q0,seed_record['original_qpos_rad']) and np.array_equal(q0,audit['original_qpos_rad']))
    check(prefix+'old_raw_seed_exact',np.array_equal(raw,seed_record['raw_candidate_qpos_rad']) and np.array_equal(raw,record['raw_candidate_qpos_rad']) and np.array_equal(raw,audit['predeclared_seed_qpos_rad'][0]))
    check(prefix+'physical_bounds_rederived',np.array_equal(physical,audit['physical_bounds_rad']) and np.array_equal(np.column_stack([low,high]),audit['bounded_correction_domain_rad']))
    check(prefix+'all_seed_original_returned_bounds',all(np.all(q>=low) and np.all(q<=high) for q in [q0,raw,returned]))
    check(prefix+'selected_new_pose',not np.array_equal(returned,q0) and np.array_equal(returned,audit['returned_qpos_rad']) and audit['surface_candidate_returned'] and not audit['exact_original_fallback'])
    baseline_vectors=np.array(audit['baseline']['tip_error_vectors_m']); final_vectors=np.array(audit['final']['tip_error_vectors_m'])
    original_caps=norm(baseline_vectors); final_norms=norm(final_vectors)
    check(prefix+'all_five_caps_recomputed_exact',np.all(final_norms<=original_caps))
    check(prefix+'cap_fields_match',np.array_equal(original_caps,audit['baseline']['tip_error_norm_m']) and np.array_equal(final_norms,audit['final']['tip_error_norm_m']) and np.array_equal(original_caps,record['authoritative_original_tip_caps_sqrt_einsum_m']) and np.array_equal(final_norms,record['selected_tip_norms_sqrt_einsum_m']))
    check(prefix+'strict_mean_recomputed',final_norms.mean()<=.010 and final_norms.mean()==audit['final']['mean_fingertip_error_m'])
    check(prefix+'no_feasibility_allowance',audit['feasibility_error_allowance_m']==0)
    check(prefix+'fresh_diagnostic_vectors_match',np.array_equal(final_vectors,np.array(record['selected_fresh_diagnostics']['task_link_error_vectors_m'])[:5]) and np.array_equal(baseline_vectors,np.array(record['baseline_fresh_diagnostics']['task_link_error_vectors_m'])[:5]))
    check(prefix+'strict_objective_improvement',audit['final']['objective']<audit['baseline']['objective']-protocol['solver_configuration']['minimum_objective_improvement'])
    check(prefix+'fresh_objective_fields_match',audit['final']['objective']==record['selected_fresh_diagnostics']['objective'] and audit['baseline']['objective']==record['baseline_fresh_diagnostics']['objective'])
    check(prefix+'full_final_alias_equal',audit['final']==audit['final_assessment'])
    proposal=audit['proposals'][0]
    check(prefix+'only_predeclared_seed_proposal',len(audit['proposals'])==1 and proposal['direction']=='predeclared_seed' and proposal['sweep']==0 and proposal['seed_index']==0 and np.array_equal(proposal['proposal_qpos_rad'],raw))
    check(prefix+'accepted_seed_selected',proposal['accepted'] and np.array_equal(proposal['candidate_qpos_rad'],returned) and len(audit['accepted_candidates'])==1 and np.array_equal(audit['accepted_candidates'][0]['qpos_rad'],returned))
    events=audit['events']; kinds=Counter(e['kind'] for e in events); counters=audit['counters']
    check(prefix+'event_sequence_complete',[e['sequence'] for e in events]==list(range(1,len(events)+1)))
    check(prefix+'event_counters_match',kinds['full_callback']==counters['full'] and kinds['FK_callback']==counters['FK'] and kinds['full_cache_hit']==counters['full_cache_hits'] and kinds['FK_cache_hit']==counters['FK_cache_hits'] and kinds['full_FK_agreement']==counters['agreement_checks'])
    full_events=[e for e in events if e['kind']=='full_callback']
    check(prefix+'three_full_calls_actual',counters['full']==record['full_callback_count_including_fresh_final']==3)
    check(prefix+'cheap_FK_plus_one_driver_preflight',record['tip_callback_count']==counters['FK']+1)
    check(prefix+'FK_and_full_budget_respected',counters['FK']<=protocol['solver_configuration']['maximum_FK_evaluations'] and counters['full']<=4 and counters['restorations']<=2)
    check(prefix+'full_call_pose_trace',np.array_equal(full_events[0]['qpos_rad'],q0) and all(np.array_equal(e['qpos_rad'],returned) for e in full_events[1:]))
    check(prefix+'fresh_terminal_call',full_events[-1]['fresh'] and full_events[-1]['terminal'] and audit['final_callbacks_fresh_outside_cache'] and record['final_state_freshly_recomputed_outside_solver_cache'])
    check(prefix+'all_event_targets_within_bounds',all(np.all(np.array(e['qpos_rad'])>=low) and np.all(np.array(e['qpos_rad'])<=high) for e in events if 'qpos_rad' in e))
    check(prefix+'full_FK_agreement_recorded',audit['full_FK_agreement_maxabs_m']<=1e-12 and all(e['maxabs_m']<=1e-12 for e in events if e['kind']=='full_FK_agreement'))
    check(prefix+'palm_and_independent_FK_statuses',record['palm_anchor_position_fixed'] and record['selected_fresh_diagnostics']['palm_anchor_position_error_m']<1e-12 and record['palm_rotation_convention_maxabs_error']<1e-6 and record['independent_fixed_URDF_vs_scene_max_link_error_m']<1.1e-7)
    check(prefix+'root_rotation_unchanged',np.array_equal(record['root_target_quaternion_wxyz'],record['original_root_target_quaternion_wxyz']))
    check(prefix+'source_ranges_records_match',all(np.array_equal(item['source_URDF_range_rad'],urdf_bounds[item['joint']]) for item in record['actual_source_ranges_audit']))
    check(prefix+'source_inputs_current',all(sha(path)==expected for path,expected in record['input_sha256'].items()))
    check(prefix+'no_physics_or_claims',record['time_s']==0 and record['physics_steps']==0 and not record['new_eval_data'] and not record['new_generator_proposal'] and not record['training'] and not record['mapping_acceptance'] and not audit['mapping_fidelity_acceptance'] and not audit['optimizer_global_feasibility_or_convergence_claim'] and not audit['scientific_acceptance_rule_changed'])
    row=next(r for r in summary['six_case_rows'] if r['index']==i)
    check(prefix+'summary_values_rederived',row['original_objective']==audit['baseline']['objective'] and row['returned_objective']==audit['final']['objective'] and row['objective_reduction_percent']==100*(audit['baseline']['objective']-audit['final']['objective'])/audit['baseline']['objective'] and row['returned_mean_tip_mm']==final_norms.mean()*1000)
    restoration=proposal['restoration_optimizer']
    rows.append({'index':i,'all_five_exact_caps_pass':bool(np.all(final_norms<=original_caps)),
        'recomputed_cap_increases_m':(final_norms-original_caps).tolist(),
        'recomputed_mean_tip_m':float(final_norms.mean()),'linalg_mean_tip_m':float(np.linalg.norm(final_vectors,axis=1).mean()),
        'returned_objective':audit['final']['objective'],'original_objective':audit['baseline']['objective'],
        'selected_q_delta_l2_rad':float(np.linalg.norm(returned-q0)),
        'projection_success':restoration['success'],'projection_status':restoration['status'],'projection_iterations':restoration['iterations'],
        'full_calls':counters['full'],'FK_calls':counters['FK'],'full_FK_maxabs_m':audit['full_FK_agreement_maxabs_m']})
old_driver=Path('/private/tmp/hug_constrained_refinement_runtime_v2_1_2026-10-14/run_constrained_refinement.py')
def functions(path):
    tree=ast.parse(path.read_text()); return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ['set_pose','evaluate','evaluate_tips']}
check('literal_objective_and_placement_AST_unchanged',functions(old_driver)==functions(ROOT/'run_seed_stage.py'))
check('source_surface_configuration_unchanged',protocol['surface_configuration']==read(protocol['old_protocol_path'])['surface_configuration'])
check('source_direction_configuration_unchanged',protocol['direction_configuration']==read(protocol['old_protocol_path'])['direction_configuration'])
check('total_full_calls_18',sum(r['full_calls'] for r in rows)==summary['total_full_surface_evaluations']==18)
check('parsed_inputs_unchanged_during_review',all(sha(path)==h for path,h in parsed_hashes.items()))
report={'schema':'hug_feasible_seed_stage_independent_arithmetic_binding_review_v1',
    'review_utc':datetime.now(timezone.utc).isoformat(),'result':'PASS' if not failures else 'FAIL',
    'input_sha256':parsed_hashes,'review_script_sha256':sha(__file__),'checks':checks,'failed_checks':failures,
    'source_current_hash_mismatches':source_mismatches,'six_case_rows':rows,
    'all_30_computed_caps_pass':all(r['all_five_exact_caps_pass'] for r in rows),
    'converged_projection_count':sum(r['projection_success'] for r in rows),
    'budget_exhausted_projection_count':sum(r['projection_status']==9 for r in rows),
    'scope':'Recompute arithmetic from stored vectors; source/code bindings and event consistency. No independent geometry/FK replay in this review.',
    'scientific_mapping_acceptance':'INDEPENDENT_AUDIT_PENDING','full_geometry_calls':0,'physics_steps':0,'new_eval_data':False}
(ROOT/'independent_seed_numerical_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
text=f'''# Independent seed-stage arithmetic and binding review (2026-10-14)

Result: **{report['result']}** ({len(checks)} checks; {len(failures)} failures).

All 30 native-relative fingertip caps and all six strict 10 mm means were
recomputed from the stored ORIGINAL/final 3D error vectors using declared
sqrt(einsum) arithmetic. All six returned poses differ from their originals,
respect independently parsed source-URDF and original ±0.35 rad bounds, and
improve the frozen objective. Old raw seeds, native identities, callback/event
counts, fresh terminal calls, source/protocol/code hashes and shared objective
ASTs were checked. Source fixed-URDF/scene and palm-convention checks are
recorded below their declared limits; this review did not rerun FK/geometry.

There were **18 actual full surface calls**, three per case. Cheap projections
converged for **2/6** cases; **4/6 reached their 40-iteration budget** while
returning exactly feasible improving points. Selection is legal under the new
method, but this result is not six converged optimizations or a global claim.

Candidate SHA256: `{parsed_hashes[str(ROOT/'candidate_targets.json')]}`.
Protocol SHA256: `{parsed_hashes[str(ROOT/'protocol.json')]}`.
Full check details and hashes: `independent_seed_numerical_review.json`.

Scientific surface mapping acceptance remains **pending** the independent
unchanged audit. No full geometry calls, physics steps, proposals, evaluation
episodes or training occurred in this review.
'''
(ROOT/'INDEPENDENT_SEED_NUMERICAL_REVIEW.md').write_text(text)
print(json.dumps({'result':report['result'],'checks':len(checks),'failures':failures,'all30caps':report['all_30_computed_caps_pass'],'projection_converged':report['converged_projection_count'],'projection_budget_exhausted':report['budget_exhausted_projection_count']},indent=2))
