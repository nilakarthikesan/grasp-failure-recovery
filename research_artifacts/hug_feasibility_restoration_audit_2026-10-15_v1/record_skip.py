"""Read saved JSON only; verify gate arithmetic and record why no coarse audit ran."""
from pathlib import Path
import hashlib,json,math,datetime,collections
OUT=Path(__file__).parent;FIT=Path('/private/tmp/hug_feasibility_restoration_2026-10-15_v1');OLD=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cp=FIT/'candidate_targets.json';pp=FIT/'protocol.json';candidate=read(cp);protocol=read(pp);pooled_path=Path(protocol['pooled_candidate_path']);pooled=read(pooled_path)
assert sha(cp)=='b0a6993006daf7aba9c1ee4d8898892a99dac19a8906f76e7fb1b85295832757'
assert candidate['protocol_sha256']==sha(pp)=='bfe10eb14a41e1e800c7b0477c6b250d36737667ae39fb4cd596731d5b167d6a'
checks=0
def check(c):
 global checks
 assert c;checks+=1
reference={c['source_inventory_index']:c for c in pooled['cases']};records=[];reason_counts=collections.Counter();tiprestored=0
pose_fields=['joint_names','qpos_rad','original_qpos_rad','root_target_position_m','root_target_quaternion_wxyz','palm_displacement_world_m','target_palm_world_position_m','target_palm_world_rotation','selected_state_scaled','bounds_scaled']
for c in candidate['cases']:
 index=c['source_inventory_index'];spec=protocol['cases'][str(index)];ref=reference[index]
 check(c['eligible_for_limited_coarse_audit'] is False);check(c['audited_pose_role']=='pooled_reference_retained');check(c['restoration_selected_endpoint'] is None)
 for field in pose_fields:check(c[field]==ref[field])
 check(c['selected_constraint_diagnostics']['rank']==spec['reference_exact_rank'])
 rows=[]
 for r in c['restoration_endpoint_results']:
  check(r['selected'] is not None);value=r['exact_surface_check'];check(value is not None);gate=r['improvement_gate'];check(gate is not None)
  tiprestored+=int(value['tip_status']['passed']);check(value['tip_status']['passed'] is True)
  rank=value['rank'];prior=spec['reference_exact_rank'];check(len(rank)==len(prior)==5 and all(math.isfinite(x) for x in rank+prior))
  v=value['exact_orientation_fields']['valid'];passed=value['exact_orientation']['passed'];pv=spec['protected_orientation_valid'];ppass=spec['protected_orientation_pass']
  check(len(v)==len(passed)==len(pv)==len(ppass))
  lostvalid=[i for i,(a,b) in enumerate(zip(pv,v)) if a and not b];lostpass=[i for i,(a,b) in enumerate(zip(ppass,passed)) if a and not b]
  reasons=[]
  if not value['in_domain']:reasons.append('original_domain_failed')
  if not value['tip_status']['passed']:reasons.append('exact_original_tip_or_mean_failed')
  if lostvalid:reasons.append('previously_valid_orientation_row_invalidated')
  if lostpass:reasons.append('previously_passed_orientation_row_failed')
  if not tuple(rank)<tuple(prior):reasons.append('no_strict_original_rank_improvement')
  check(gate['refusal_reasons']==reasons);check(gate['lost_orientation_valid_rows']==lostvalid);check(gate['lost_orientation_pass_rows']==lostpass);check(gate['accepted_for_limited_coarse_audit']==(not reasons));check(bool(reasons))
  reason_counts.update(reasons)
  rows.append({'endpoint_start_index':r['source_endpoint']['start_index'],'reported_exact_tip_restoration_pass':value['tip_status']['passed'],'exact_rank':rank,'reference_exact_rank':prior,'strict_rank_improvement':tuple(rank)<tuple(prior),'lost_protected_valid_orientation_row_indices':lostvalid,'lost_protected_pass_orientation_row_indices':lostpass,'refusal_reasons':reasons,'eligible':False})
 check(len(rows)==2)
 records.append({'source_inventory_index':index,'native_sha256':c['native_sha256'],'pose_fields_equal_pooled_reference':pose_fields,'role':c['audited_pose_role'],'fresh_grid_status':'NOT_RUN_NO_ELIGIBLE_CANDIDATE','fresh_orientation_status':'NOT_RUN','endpoints':rows})
check(len(records)==6 and tiprestored==12)
check(not any(c['eligible_for_limited_coarse_audit'] for c in candidate['cases']))
check(not (OUT/'protocol.json').exists())
check(not list(OUT.glob('*_validation.json')) and not list(OUT.glob('case_*_full_qpos.npy')))
summary_path=OLD/'summary.json';prior_summary=read(summary_path)
result={'schema':'hug_restored_candidate_limited_audit_skip_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'NOT_RUN_NO_ELIGIBLE_CANDIDATE','source_bindings':{str(p):sha(p) for p in [cp,FIT/'protocol.json',FIT/'restoration_summary.json',pooled_path,summary_path,OLD/'protocol.json',OUT/'wrapper_parity.json',OUT/'protocol.DRAFT.json',OUT/'prepare_wrapper.py',Path(__file__)]},'independent_scope':'Saved-output eligibility, exact Python rank comparison, protected row Boolean masks and selected pose equality only. No FK or geometric recomputation. Separate numerical reviewer owns source/FK/cap/trust certification.','saved_output_checks_passed':checks,'reported_tip_restored_endpoints':tiprestored,'eligible_candidates':0,'retained_pooled_reference_cases':6,'reason_endpoint_counts_overlapping':dict(reason_counts),'cases':records,'historical_evidence':{'status':'PRIOR_RESULTS_UNCHANGED_NOT_FRESHLY_REMEASURED','source_summary_path':str(summary_path),'source_summary_sha256':sha(summary_path),'strict_failure_comparisons':9,'meaning':'All selected pose/root/domain fields equal the already audited pooled references. Prior nine FAIL results remain historical evidence; this restoration audit performed zero fresh grids. Restored refused endpoints did not receive full-grid status.'},'prepared_wrapper_status':'PREPARED_AND_TEXT_PARITY_CHECKED_ONLY_NOT_EXECUTED_OR_FROZEN','fresh_coarse_grids':0,'fresh_fine_grids':0,'new_distance_queries':0,'new_orientation_queries':0,'new_FK_calls':0,'new_inference':False,'physics_steps':0,'new_eval_data':False,'new_mapping_acceptance':False}
path=OUT/'audit_skip_receipt.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'sha256':sha(path),'checks':checks,'reported_tip_restored_endpoints':tiprestored,'eligible_candidates':0,'reason_counts':dict(reason_counts)},indent=2))
