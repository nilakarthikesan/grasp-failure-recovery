"""Prepare (do not execute) a bounded coarse audit from the frozen previous wrapper."""
from pathlib import Path
import json,hashlib,ast
OUT=Path(__file__).parent;OLD=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1')
source=(OLD/'validate_candidate.py').read_text();s=source
def replace(old,new):
 global s
 assert s.count(old)==1,(old[:100],s.count(old));s=s.replace(old,new)
replace("assert len(targets)==len(candidate['cases'])==6", "assert len(targets)==len(candidate['cases'])==6\n    assert sorted(c['source_inventory_index'] for c in candidate['cases'] if c['eligible_for_limited_coarse_audit'])==protocol['case_indices']")
replace("assert args.case_index is None, 'All six preflight cases must complete together'", "assert args.case_index is None, 'All eligible preflight cases must complete together'")
replace("        if args.case_index is not None and index not in args.case_index:continue", "        if index not in protocol['case_indices']:continue\n        if args.case_index is not None and index not in args.case_index:continue")
start=s.index("        role=target['audited_pose_role']")
end=s.index("        meta,tri,labels,nn,hand,objs,m,d,initial",start)
s=s[:start]+'''        role=target['audited_pose_role'];assert role=='restored_tip_feasible_surface_improvement'
        assert target['eligible_for_limited_coarse_audit'] is True and target['diagnostic_pose_not_promoted'] is True
        assert target['fresh_improvement_gate']['accepted_for_limited_coarse_audit'] is True
        assert target['fresh_improvement_refusal'] is None
        fitcase=candidate_protocol['cases'][str(index)];selected_exact=target['selected_constraint_diagnostics']
        assert tuple(selected_exact['rank'])<tuple(fitcase['reference_exact_rank'])
        protected_valid=np.asarray(fitcase['protected_orientation_valid'],bool);protected_pass=np.asarray(fitcase['protected_orientation_pass'],bool)
        valid=np.asarray(selected_exact['exact_orientation_fields']['valid'],bool);passed=np.asarray(selected_exact['exact_orientation']['passed'],bool)
        assert valid.shape==passed.shape==protected_valid.shape==protected_pass.shape
        assert not np.any(protected_valid&~valid) and not np.any(protected_pass&~passed)
        endpoint_id=target['restoration_selected_endpoint'];assert endpoint_id in (0,1)
        matches=[r for r in target['restoration_endpoint_results'] if r['source_endpoint']['start_index']==endpoint_id];assert len(matches)==1
        endpoint_record=matches[0];assert endpoint_record['improvement_gate']['accepted_for_limited_coarse_audit']
        audit_role={'audited_pose_role':role,'constraint_selection_accepted':target['constraint_selection_accepted'],'diagnostic_pose_not_promoted':True,'rejection_reason':target['rejection_reason'],'selection_source':target['selection_source'],'eligible_for_limited_coarse_audit':True,'retained_witness_improvement_is_not_mapping_acceptance':True}
''' + s[end:]
needle="        assert np.max(np.abs(state[:24]*.025+original_values-qv))<1e-12 and np.max(np.abs(state[24:]*.005-shift))<1e-12"
replace(needle,needle+'''
        endpoint=np.asarray(endpoint_record['source_endpoint']['state_scaled'],float);trust=np.asarray(endpoint_record['trust_bounds_scaled'],float)
        expected_endpoint=np.asarray(next(x for x in fitcase['endpoints'] if x['start_index']==endpoint_id)['state_scaled'],float)
        radius=candidate_protocol['projection_configuration']['trust_half_width_scaled'];assert radius==.25
        expected_trust=np.column_stack((np.maximum(bounds[:,0],endpoint-radius),np.minimum(bounds[:,1],endpoint+radius)))
        assert endpoint.shape==(27,) and np.array_equal(endpoint,expected_endpoint) and np.array_equal(trust,expected_trust)
        assert np.array_equal(state,np.asarray(endpoint_record['selected']['state_scaled']))
        assert np.all(state>=trust[:,0]) and np.all(state<=trust[:,1]), 'Literal endpoint-local restoration trust box'
''')
replace("        if role=='original_fallback':assert np.array_equal(d.qpos,original_q) and np.all(shift==0) and not target['constraint_selection_accepted']", "        assert not np.array_equal(state,np.asarray(target['reference_state_scaled'])), 'Only a genuinely changed improved pose may receive a new coarse audit'")
replace("        for density in selected['densities_mm']:", "        assert 2 in selected['densities_mm']\n        for density in (2,):")
old="""            orientation=None;check=None
            if not args.skip_orientation:
                orientation=ori.inspect_orientation(q,area,objnorm,nd,old['native_inside'],old['native_nearest_part'],tri,labels,nn,newhand,vt,full,p)
                check=status(proximity,orientation,rule,sp['primary_allowance_m'])"""
new="""            orientation=None;check=proximity_counterexample_stop(proximity,rule,sp['primary_allowance_m'])
            stopped_on_proximity=check is not None
            if not stopped_on_proximity:
                orientation=ori.inspect_orientation(q,area,objnorm,nd,old['native_inside'],old['native_nearest_part'],tri,labels,nn,newhand,vt,full,p)
                check=status(proximity,orientation,rule,sp['primary_allowance_m'])"""
replace(old,new)
needle="            assert np.array_equal(d.qpos,nominal) and d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)\n            write(args.out/f'{caseid}_validation.json',rec)"
replace(needle,"""            rec['resolutions'][str(density)]['limited_audit_execution']={'fresh_target_full_fields':True,'early_stop_on_proximity_counterexample':stopped_on_proximity,'orientation_execution':'NOT_RUN' if stopped_on_proximity else 'RUN','orientation_not_run_reason':'definite_frozen_proximity_FAIL_already_refutes_candidate' if stopped_on_proximity else None,'finer_grids_run':False,'further_audit_authorization_required':True}
            assert np.array_equal(d.qpos,nominal) and d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
            write(args.out/f'{caseid}_validation.json',rec)""")
replace("'schema':'hug_surface_constraints_candidate_independent_static_validation_v1'", "'schema':'hug_restored_candidate_limited_coarse_static_validation_v1'")
helper='''def proximity_counterexample_stop(proximity,rule,eps):
    """Exact original per-part distance decision, with no fabricated orientation result."""
    checks=[]
    for band in rule['bands_mm']:
        for part in rule['anatomical_parts']:
            x=proximity[str(eps)]['bands'][str(band)]['independent_parts'][part]
            lo=x['lost_native_area']['lower_mm2'];new=x['new_mapped_area']['lower_mm2'];lu=x['lost_native_area']['upper_mm2'];nu=x['new_mapped_area']['upper_mm2']
            state='FAIL' if lo>0 or new>0 else ('PASS' if lu==0 and nu==0 else 'UNRESOLVED')
            checks.append({'part':part,'band_mm':band,'status':state,'lost_area_lower_mm2':lo,'added_area_lower_mm2':new,'lost_area_upper_mm2':lu,'added_area_upper_mm2':nu})
    if not any(x['status']=='FAIL' for x in checks):return None
    return {'proximity_status_conditional':'FAIL','pointwise_orientation_status':'NOT_RUN','strict_relationship_status_conditional':'FAIL','proximity_checks':checks,'orientation_checks':None,'full_physical_contact_equivalence':'UNRESOLVED','failure_label_preservation':'NOT_TESTED','independent_unseen_object_validation':'NOT_TESTED'}

'''
s=s.replace('def main():',helper+'def main():',1)
# Exact function and metric-block parity; all differences outside the original mathematical definitions are named.
def func(text,name):return ast.get_source_segment(text,next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name==name))
for name in ('full_fields','status'):assert func(s,name)==func(source,name)
metric_start="            cache=Path(p['original_query_cache_root'])"
metric_end="            orientation=None;"
old_metric=source[source.index(metric_start):source.index(metric_end)]
new_metric=s[s.index(metric_start):s.index(metric_end)]
assert new_metric==old_metric
original_ori="                orientation=ori.inspect_orientation(q,area,objnorm,nd,old['native_inside'],old['native_nearest_part'],tri,labels,nn,newhand,vt,full,p)"
assert original_ori in s and original_ori in source
for path,data in [('validate_candidate.py',s),('inherited_coarse_metric_block.py.txt',old_metric)]: (OUT/path).write_text(data)
(OUT/'run_validation.py').write_bytes((OLD/'run_validation.py').read_bytes())
(OUT/'protected_hashes.json').write_bytes((OLD/'protected_hashes.json').read_bytes())
parity={'source_path':str(OLD/'validate_candidate.py'),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'candidate_wrapper_sha256':hashlib.sha256(s.encode()).hexdigest(),'full_fields_byte_identical':True,'status_byte_identical':True,'grid_and_full_proximity_block_byte_identical':True,'orientation_call_and_upstream_function_unchanged':True,'allowed_differences':['restoration eligibility and exact endpoint trust preflight','case subset and coarse density only','short circuit after a definite proximity FAIL, orientation explicitly NOT_RUN','provenance metadata'], 'new_geometry_executed':False}
(OUT/'wrapper_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
p=json.loads((OLD/'protocol.DRAFT.json').read_text())
p.update(schema='hug_feasibility_restoration_limited_coarse_audit_protocol_v1',candidate_path='/private/tmp/hug_feasibility_restoration_2026-10-15_v1/candidate_targets.json',candidate_schema='hug_saved_endpoint_feasibility_restoration_candidates_v1',candidate_protocol_expected_sha256=None,candidate_sha256=None,case_indices=None,allowed_full_audit_cohorts=None,execution_authorized_by_root_review=False,root_review='Pending final candidate eligibility, independent source/FK/cap/trust PASS and explicit root authorization. Preparation only.',scope='Only reviewed genuinely improved restored candidates on unchanged historical coarse grid. Stop on definite proximity FAIL before orientation; otherwise unchanged coarse orientation. No automatic fine grid or all-nine rerun.',amendment='Unchanged full_fields/status/grid/proximity/orientation math. Separate eligible-case/trust preflight and short-circuit execution after definite coarse failure.',candidate_fallback_policy='All six source candidate rows are bound. Retained pooled fallbacks are NOT_RUN by this audit; only eligible changed restored rows may be staged and measured.',candidate_domain_policy='Original physical/trust/source-reach domains unchanged plus exact endpoint-local restoration trust box, no added endpoint allowance.',allowed_audited_pose_roles=['restored_tip_feasible_surface_improvement'],execution_stop_policy='At most one density2mm full distance query per eligible pose. A definite primary-allowance same-part proximity lower bound causes strictFAIL and skips orientation explicitly NOT_RUN. Otherwise original orientation runs once. No finer queries regardless of result without a separately reviewed protocol.',new_eval_data=False,new_inference=False,physics_steps=0)
p.pop('cross_version_selection_scope',None)
(OUT/'protocol.DRAFT.json').write_text(json.dumps(p,indent=2)+'\n')
compile(s,str(OUT/'validate_candidate.py'),'exec')
print(json.dumps({'prepared':str(OUT),'wrapper_sha256':parity['candidate_wrapper_sha256'],'frozen':False,'geometry_executed':False},indent=2))
