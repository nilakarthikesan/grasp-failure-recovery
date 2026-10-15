from pathlib import Path
import hashlib,json
OUT=Path(__file__).parent;D=Path('/private/tmp/hug_surface_constraints_2026-10-15_v2')
read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
candidate=D/'candidate_targets.json';merged=read(candidate);checks=0;byindex={};reports=[];bindings={str(candidate):sha(candidate),str(D/'protocol.json'):sha(D/'protocol.json'),str(Path(__file__)):sha(Path(__file__))}
for cohort in ('8_16_24','12_20_28'):
 cp=D/f'candidate_cohort_{cohort}.json';rp=OUT/f'cohort_{cohort}_output_review.json';r=read(rp);c=read(cp)
 assert r['passed'] and r['source_bindings'][str(cp)]==sha(cp);checks+=1
 assert merged['protocol_sha256']==c['protocol_sha256']==sha(D/'protocol.json');checks+=1
 byindex.update({v['source_inventory_index']:v for v in c['cases']});reports.append(r);bindings[str(cp)]=sha(cp);bindings[str(rp)]=sha(rp)
assert sorted(byindex)==[8,12,16,20,24,28];checks+=1
for case in merged['cases']:assert case==byindex[case['source_inventory_index']];checks+=1
assert len(merged['cases'])==6;checks+=1
for extra in [D/'thread_count_deviation.json',Path('/private/tmp/hug_surface_constraints_2026-10-15_v1/thread_count_deviation.json'),OUT/'thread_import_state_review.json',OUT/'pool_policy_review.json',Path('/private/tmp/hug_surface_constraints_root_2026-10-15_v1/cross_version_selection_rule.json')]:bindings[str(extra)]=sha(extra)
report={'schema':'hug_surface_constraints_independent_final_output_review_v2','passed':True,'source_bindings':bindings,'merged_record_identity_checks':checks,'source_and_trace_checks':sum(r['source_and_trace_checks'] for r in reports),'constraint_arithmetic_checks':sum(r['constraint_arithmetic_checks'] for r in reports),'cases':sorted([c for r in reports for c in r['cases']],key=lambda c:c['source_inventory_index']),'scope':'All six V2 candidate records are exactly those independently verified in immutable cohort reviews. Cohort reviews reconstructed original-source FK/caps/domains and replayed all recorded raw/exact witness arithmetic/selection. This aggregation performs only identity/hash checks, with no repeated FK/geometry/optimization.','physics_steps':0,'new_surface_queries':0,'new_eval_data':False,'new_inference':False,'mapping_acceptance':False,'resource_protocol_deviation':{'declared_numba_threads_per_worker':2,'effective_numba_threads_per_worker_after_import':4,'maximum_fit_workers':2,'disclosure':'Nested frozen geometry import overrides earlier thread setting. Numeric review PASS does not certify declared resource-cap compliance. Scientific formulas, caps, predicates and candidate poses were not changed or rerun for this disclosure.'}}
p=OUT/'final_output_review.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'identity_checks':checks,'source_trace_checks':report['source_and_trace_checks'],'arithmetic_checks':report['constraint_arithmetic_checks'],'report_sha256':sha(p)}))
