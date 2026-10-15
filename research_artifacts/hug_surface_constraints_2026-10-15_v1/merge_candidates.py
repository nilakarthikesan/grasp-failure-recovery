"""Merge completed frozen cohorts; no candidate ranking or geometry queries."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
p=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json');rows=[];inputs=[]
for indices in p['allowed_cohorts']:
    path=OUT/('candidate_cohort_'+'_'.join(map(str,indices))+'.json');r=read(path)
    assert r['protocol_sha256']==ph and r['cohort']==indices
    rows.extend(r['cases']);inputs.append({'path':str(path),'sha256':sha(path)})
assert sorted(r['source_inventory_index'] for r in rows)==[8,12,16,20,24,28]
summaries=[]
for r in sorted(rows,key=lambda r:r['source_inventory_index']):
    final=r['selected_constraint_diagnostics'];base=r['baseline_constraint_diagnostics']
    assert final['in_domain'] and final['tip_status']['passed']
    assert np.all(np.asarray(r['selected_tip_norms_sqrt_einsum_m'])<=np.asarray(r['authoritative_original_tip_caps_sqrt_einsum_m']))
    def summary(d):return {'proximity_rows_pass':sum(d['proximity']['passed']),'proximity_rows_total':len(d['proximity']['passed']),
        'orientation_rows_pass':sum(d['exact_orientation']['passed']),'orientation_rows_total':len(d['exact_orientation']['passed']),
        'invalid_orientation_rows':sum(not v for v in d['exact_orientation_fields']['valid']),
        'maximum_normalized_violation':d['rank'][2],'all_retained_witness_rows_pass':d['all_witness_constraints_pass'],
        'mean_tip_error_mm':d['tip_status']['mean_m']*1000,'all_exact_original_caps_pass':all(d['tip_status']['individual_pass'])}
    # Top limiting rows are descriptive, selected by existing exact scalar
    # violation after fitting, and never fed back into an optimizer.
    limiting=[]
    for i,source in enumerate(r['constraint_source_rows']['proximity']):
        limiting.append({'type':'proximity','row_index':i,'part':source['row']['anatomical_part'],'band_mm':source['row']['band_mm'],
            'violation_normalized':max(0.,-final['proximity']['margin_m'][i]/.001),'passed':final['proximity']['passed'][i],
            'source_report':source['source_report'],'source_pointer':source['source_json_pointer']})
    for i,source in enumerate(r['constraint_source_rows']['orientation']):
        limiting.append({'type':'orientation','row_index':i,'part':source['row']['native_part'],'band_mm':source['band_mm'],
            'violation_normalized':max(0.,-final['exact_orientation']['sign_margin'][i],-final['exact_orientation']['band_margin_m'][i]/.001,-final['exact_orientation']['query_boundary_margin_m'][i]/.001),
            'passed':final['exact_orientation']['passed'][i],'eligible':final['exact_orientation_fields']['valid'][i],
            'invalid_reasons':[k for k,v in final['exact_orientation_fields']['reason_masks'].items() if v[i]],
            'source_report':source['source_report'],'source_pointer':source['source_json_pointer']})
    limiting.sort(key=lambda z:(z.get('eligible',True),z['passed'],-z['violation_normalized'],z['type'],z['row_index']))
    summaries.append({'source_inventory_index':r['source_inventory_index'],'object_id':r['object_id'],'audited_pose_role':r['audited_pose_role'],
        'baseline':summary(base),'selected':summary(final),'palm_displacement_mm':r['palm_displacement_norm_m']*1000,
        'selected_tip_error_mm':(np.asarray(r['selected_tip_norms_sqrt_einsum_m'])*1000).tolist(),
        'selected_tip_cap_excess_m':(np.asarray(r['selected_tip_norms_sqrt_einsum_m'])-np.asarray(r['authoritative_original_tip_caps_sqrt_einsum_m'])).tolist(),
        'legacy_direction_guard_accepted_diagnostic_only':r['legacy_direction_guard_diagnostic_only']['accepted'],
        'solver_counts':r['solver_audit']['counts'],'termination':r['solver_audit']['termination'],'optimizer_starts':r['solver_audit']['optimizer_starts'],
        'top_limiting_witnesses':limiting[:10],'orientation_coverage':r['orientation_witness_coverage']})
target=OUT/'candidate_targets.json';assert not target.exists()
result={'schema':p['candidate_schema'],'protocol_sha256':ph,'cases':sorted(rows,key=lambda r:r['source_inventory_index']),
    'cohort_bindings':inputs,'new_eval_data':False,'new_inference':False,'physics_steps':0,'production_changed':False,'mapping_acceptance':False}
target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
summary={'schema':'hug_surface_witness_constraint_summary_v1','protocol_sha256':ph,'candidate_sha256':sha(target),'cases':summaries,
    'retained_row_counts_include_duplicates_and_overlapping_parts_bands':True,'total_raw_callbacks':sum(r['solver_audit']['counts']['raw_callback_executions'] for r in rows),
    'total_exact_checks':sum(r['solver_audit']['counts']['exact_checkpoint_executions'] for r in rows),'no_new_eval_data':True,'full_grid_status':'AWAITING_INDEPENDENT_AUDIT'}
(OUT/'fit_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({'candidate':str(target),'sha256':sha(target),'summary':str(OUT/'fit_summary.json')}))
