"""Combine already selected restoration records; never query or re-rank poses."""
from pathlib import Path
import hashlib,json
OUT=Path(__file__).parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json');cases=[];cohorts=[]
for ids in p['allowed_cohorts']:
 path=OUT/('candidate_cohort_'+'_'.join(map(str,ids))+'.json');v=read(path);assert v['protocol_sha256']==ph and v['cohort']==ids
 cases.extend(v['cases']);cohorts.append({'path':str(path),'sha256':sha(path)})
assert sorted(c['source_inventory_index'] for c in cases)==[8,12,16,20,24,28]
cases.sort(key=lambda c:c['source_inventory_index']);summaries=[]
for c in cases:
 s=c['selected_constraint_diagnostics'];b=c['reference_constraint_diagnostics']
 def counts(v):return {'proximity_pass':sum(v['proximity']['passed']),'proximity_rows':len(v['proximity']['passed']),'orientation_pass':sum(v['exact_orientation']['passed']),'orientation_rows':len(v['exact_orientation']['passed']),'orientation_invalid':sum(not x for x in v['exact_orientation_fields']['valid']),'rank':v['rank'],'mean_tip_mm':1000*v['tip_status']['mean_m'],'all_literal_tip_caps_pass':all(v['tip_status']['individual_pass']),'all_retained_rows_pass':v['all_witness_constraints_pass']}
 endpoints=[]
 for e in c['restoration_endpoint_results']:
  endpoints.append({'start_index':e['source_endpoint']['start_index'],'restored_tip_feasible':e['selected'] is not None,'nearest_observed_checkpoint':e['selected'],'optimizer':e['optimizer'],'counts':e['counts'],'actual_projection_mj_forward_calls':e['actual_projection_mj_forward_calls'],'gate':e['improvement_gate'],'exact_surface_summary':None if e['exact_surface_check'] is None else counts(e['exact_surface_check'])})
 summaries.append({'index':c['source_inventory_index'],'object_id':c['object_id'],'coarse_audit_eligible':c['eligible_for_limited_coarse_audit'],'role':c['audited_pose_role'],'selected_endpoint':c['restoration_selected_endpoint'],'palm_shift_mm':1000*c['palm_displacement_norm_m'],'reference':counts(b),'selected':counts(s),'endpoints':endpoints,'exact_surface_checks':c['exact_surface_checks'],'actual_case_mj_forward_calls':c['actual_case_mj_forward_calls'],'new_eval_data':False})
target=OUT/'candidate_targets.json';assert not target.exists();target.write_text(json.dumps({'schema':p['candidate_schema'],'protocol_sha256':ph,'cases':cases,'cohort_bindings':cohorts,'new_eval_data':False,'new_inference':False,'physics_steps':0,'production_changed':False,'mapping_acceptance':False},indent=2)+'\n')
s=OUT/'restoration_summary.json';assert not s.exists();s.write_text(json.dumps({'schema':'hug_endpoint_restoration_summary_v1','candidate_sha256':sha(target),'protocol_sha256':ph,'cases':summaries,'eligible_coarse_audit_indices':[c['source_inventory_index'] for c in cases if c['eligible_for_limited_coarse_audit']],'counts_include_repeated_retained_rows':True,'no_global_projection_optimality_claim':True,'mapping_acceptance':False},indent=2)+'\n');print(json.dumps({'candidate':str(target),'sha256':sha(target),'summary':str(s),'coarse_audit_indices':[c['source_inventory_index'] for c in cases if c['eligible_for_limited_coarse_audit']]}))
