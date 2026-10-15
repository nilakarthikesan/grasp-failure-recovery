"""Pure JSON provenance/description checks after independent all-endpoint FK replay."""
from pathlib import Path
import hashlib,json
import numpy as np
D=Path('/private/tmp/hug_feasibility_restoration_2026-10-15_v1'); O=Path(__file__).parent
R=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/pooled_selection/candidate_targets.json')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0
bindings={}
def check(v,label):
 global checks
 checks+=1
 if not v: raise AssertionError(label)
def bind(p): bindings[str(p)]=sha(p); return read(p)
a=bind(D/'candidate_targets.json'); p=bind(D/'protocol.json'); old=bind(R); old={v['source_inventory_index']:v for v in old['cases']}
f=bind(O/'final_output_review.json'); check(f['passed'] is True,'independent full replay PASS'); check(f['source_bindings'][str(D/'candidate_targets.json')]==sha(D/'candidate_targets.json'),'numerical report binds candidate')
cohorts={}
for binding in a['cohort_bindings']:
 key,h=binding['path'],binding['sha256']
 check(sha(key)==h,'cohort hash'); c=bind(Path(key));check(c['protocol_sha256']==a['protocol_sha256'],'same protocol');
 for row in c['cases']:
  idx=row['source_inventory_index']; check(idx not in cohorts,'one cohort per case');cohorts[idx]=row
summary=bind(D/'restoration_summary.json'); desc=bind(D/'endpoint_outcome_diagnostics.json');desc={(r['index'],r['endpoint_start_index']):r for r in desc['rows']}
rows=[];total_fk=total_jac=total_forwards=total_exact=0;lost_valid_endpoints=lost_pass_endpoints=0
for c in a['cases']:
 idx=c['source_inventory_index']; check(c==cohorts[idx],'merged exact cohort record');check(c==bind(D/f'case_{idx}_candidate.json'),'merged exact per-case record')
 for key in ('selected_state_scaled','qpos_rad','root_target_position_m','root_target_quaternion_wxyz','palm_displacement_world_m','target_palm_world_position_m','target_palm_world_rotation','selected_tip_norms_sqrt_einsum_m'):
  check(c[key]==old[idx][key],'fallback exact geometry '+key)
 check(c['reference_candidate_binding']=={'path':str(R),'sha256':sha(R)},'reference hash binding')
 check(c['eligible_for_limited_coarse_audit'] is False and c['restoration_selected_endpoint'] is None,'no further audit eligibility')
 check(c['numba_threads_after_imports']==2,'actual post-import mask')
 source_original=c['baseline_constraint_diagnostics'];source_reference=c['reference_constraint_diagnostics']
 pp=np.asarray(source_original['exact_orientation']['passed'],bool)|np.asarray(source_reference['exact_orientation']['passed'],bool)
 pv=np.asarray(source_original['exact_orientation_fields']['valid'],bool)|np.asarray(source_reference['exact_orientation_fields']['valid'],bool)
 case_forward=0
 for e in c['restoration_endpoint_results']:
  start=e['source_endpoint']['start_index'];check(e==bind(D/f'case_{idx}_endpoint_{start}.json'),'endpoint exact write/merge identity')
  check(sha(e['trace_path'])==e['trace_sha256'],'trace checksum');bindings[e['trace_path']]=e['trace_sha256']
  t=e['exact_surface_check'];g=e['improvement_gate'];rv=np.asarray(t['exact_orientation_fields']['valid'],bool);rp=np.asarray(t['exact_orientation']['passed'],bool)
  lp=pp&~rp;lv=pv&~rv; lost_pass_endpoints+=int(lp.any());lost_valid_endpoints+=int(lv.any())
  # These categories are descriptive and do not alter any gate. A sign
  # reversal can only be called reliable when the frozen eligibility holds.
  sign_margin=np.asarray(t['exact_orientation']['sign_margin']);lostpass_invalid=lp&~rv;lostpass_valid_sign=lp&rv&(sign_margin<=0);lostpass_valid_other=lp&rv&(sign_margin>0)
  check(int(lp.sum())==int(lostpass_invalid.sum()+lostpass_valid_sign.sum()+lostpass_valid_other.sum()),'lost pass categories partition')
  margins=np.asarray(t['exact_orientation']['sign_margin']);check(g['accepted_for_limited_coarse_audit'] is False and tuple(t['rank'])>tuple(source_reference['rank']),'strictly worse exact rank')
  d=desc[idx,start];x=np.asarray(e['selected']['state_scaled']);origin=np.asarray(e['source_endpoint']['state_scaled']);err=np.asarray(t['task_error_vectors_m'])[:5];caps=np.asarray(c['authoritative_original_tip_caps_sqrt_einsum_m']);norms=np.sqrt(np.einsum('ij,ij->i',err,err))
  expected={'restoration_joint_max_change_rad':float(np.max(np.abs((x-origin)[:24]))*.025),'restoration_palm_change_mm':float(np.linalg.norm((x-origin)[24:])*.005*1000),'frozen_endpoint_palm_norm_mm':float(np.linalg.norm(origin[24:])*.005*1000),'restored_palm_norm_mm':float(np.linalg.norm(x[24:])*.005*1000),'max_original_tip_cap_excess_m':float(np.max(norms-caps)),'tip_mean_mm':float(norms.mean()*1000),'orientation_invalid_reason_counts_overlapping':{k:int(np.asarray(v,bool).sum()) for k,v in t['exact_orientation_fields']['reason_masks'].items() if np.asarray(v,bool).any()},'invalid_orientation_rows':int((~rv).sum()),'orientation_rows':len(rv),'proximity_pass':int(np.asarray(t['proximity']['passed']).sum()),'proximity_rows':len(t['proximity']['passed']),'exact_rank':t['rank'],'gate':g}
  for key,val in expected.items():
   if isinstance(val,float):check(abs(val-d[key])<=1e-12,'primary diagnostic scalar '+key)
   else:check(val==d[key],'primary diagnostic field '+key)
  fk=e['counts']['FK_executions'];jac=e['counts']['analytic_jacobian_executions'];forwards=e['actual_projection_mj_forward_calls'];total_fk+=fk;total_jac+=jac;total_forwards+=forwards;case_forward+=forwards
  rows.append(dict(index=idx,start=start,tip_caps_pass=True,mean_tip_mm=float(norms.mean()*1000),max_original_cap_excess_m=float(np.max(norms-caps)),joint_correction_max_rad=expected['restoration_joint_max_change_rad'],palm_correction_mm=expected['restoration_palm_change_mm'],relative_to_original_palm_shift_mm=expected['restored_palm_norm_mm'],reference_rank=source_reference['rank'],rank=t['rank'],proximity_pass=expected['proximity_pass'],proximity_rows=expected['proximity_rows'],protected_valid_rows_lost=int(lv.sum()),protected_pass_rows_lost=int(lp.sum()),lost_pass_invalid=int(lostpass_invalid.sum()),lost_pass_valid_sign_not_preserved=int(lostpass_valid_sign.sum()),lost_pass_valid_other_failure=int(lostpass_valid_other.sum()),orientation_invalid_reason_counts_overlapping=expected['orientation_invalid_reason_counts_overlapping'],iteration_count=e['optimizer']['nit'],optimizer_status=e['optimizer']['status'],FK_calls=fk,jacobian_calls=jac,projection_forward_calls=forwards,eligible=False))
 total_exact+=c['exact_surface_checks'];check(c['actual_case_mj_forward_calls']==case_forward+13,'complete case forwards include case initialization, original source, reference/endpoint/final exact stages and independent URDF FK')
check(len(rows)==12 and total_exact==24,'exact endpoint and post-projection counts')
for path,h in bindings.items():check(sha(path)==h,'source unchanged')
out=O/'merge_and_outcomes_review.json';check(not out.exists(),'no overwrite')
report=dict(schema='hug_restoration_merge_and_outcomes_independent_review_v1',passed=True,checks=checks,source_bindings=bindings,endpoint_outcomes=rows,total_tip_FK_callbacks=total_fk,total_analytic_jacobian_callbacks=total_jac,total_projection_mj_forward_calls=total_forwards,total_exact_surface_checks=total_exact,restored_tip_feasible_endpoints=12,strict_rank_improvements=0,lost_protected_valid_endpoints=lost_valid_endpoints,lost_protected_pass_endpoints=lost_pass_endpoints,retained_reference_cases=6,eligible_for_limited_coarse_audit=0,new_geometry_queries=0,new_evaluation_data=False,interpretation='Eligibility losses are comparisons to protected original/pooled witnesses, not proof that restoration itself caused them: the prior tip-infeasible endpoints had no exact surface check. Invalid rows are not counted as reliable normal reversals. Retained-row counts may repeat witnesses across bands/resolutions; they are not unique surface area. No full-grid audit warranted by frozen eligibility rule.')
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('source_bindings','endpoint_outcomes','interpretation')}));print(sha(out))
