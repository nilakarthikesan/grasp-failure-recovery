"""Apply a pre-frozen scoped rule to immutable reports; no simulator/runtime."""
from pathlib import Path
import json,hashlib,gzip
base=Path(__file__).resolve().parent;bundle=base.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
rule=read(base/'acceptance_rule.json');rh=sha(base/'acceptance_rule.json')
assert rh=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4'
sampling=read(bundle/'sampling/sampling_uncertainty.json');sp=read(bundle/'sampling/protocol.json')
orientation=json.loads(gzip.decompress((bundle/'orientation/orientation_report_full.json.gz').read_bytes()))
native=read(bundle/'native_contract/native_contract.json')
assert sampling['protocol_sha256']==sha(bundle/'sampling/protocol.json')
assert orientation['protocol_sha256']==sha(bundle/'orientation/protocol.json')
assert native['protocol_sha256']==sha(bundle/'native_contract/protocol.json')
assert len(sampling['cases'])==len(orientation['cases'])==6
assert native['distinct_native_count']==32 and native['all_face_indices_equal']
assert native['max_vertex_error_m']==native['max_landmark_error_m']==native['max_rotation_element_error']==0
om={c['native_sha256']:c for c in orientation['cases']};records=[]
for c in sampling['cases']:
 oc=om[c['native_sha256']]
 for density,res in c['resolutions'].items():
  bands=res['numerical_allowances'][str(sp['primary_allowance_m'])]['bands'];ors=oc['resolutions'][density];bd=ors['sampling_shared_binding']
  assert bd['protocol_sha256']==sampling['protocol_sha256'] and bd['native_sha256']==c['native_sha256']
  assert bd['original_cache_sha256']==res['cache_sha256'] and bd['no_new_eval_data']
  assert bd['query_library_sha256']==sp['previous_query_library_sha256']
  checks=[];normal=[]
  for band in rule['bands_mm']:
   for part in rule['anatomical_parts']:
    x=bands[str(band)]['independent_parts'][part]
    lo=x['lost_native_area']['lower_mm2'];ne=x['new_mapped_area']['lower_mm2'];lu=x['lost_native_area']['upper_mm2'];nu=x['new_mapped_area']['upper_mm2']
    status='FAIL' if lo>0 or ne>0 else ('PASS' if lu==0 and nu==0 else 'UNRESOLVED')
    checks.append({'part':part,'band_mm':band,'status':status,'lost_area_lower_mm2':lo,'added_area_lower_mm2':ne,'lost_area_upper_mm2':lu,'added_area_upper_mm2':nu})
   signs=ors['bands'][str(band)]['common_same_label']['pointwise_sign_check'];flips=signs['flip_denominators'];n=sum(v['query_count'] for v in flips.values())
   normal.append({'band_mm':band,'pointwise_status':'FAIL' if n else 'UNRESOLVED','reliable_sampled_sign_reversal_count':n,'flip_strata':flips,'continuous_orientation_coverage_status':'UNRESOLVED','reason':'Matching sampled signs cannot certify continuous nearest-feature normals; point witnesses can refute the named query property.'})
  prox='FAIL' if any(x['status']=='FAIL' for x in checks) else ('PASS' if all(x['status']=='PASS' for x in checks) else 'UNRESOLVED')
  point='FAIL' if any(x['pointwise_status']=='FAIL' for x in normal) else 'UNRESOLVED'
  strict='FAIL' if prox=='FAIL' or point=='FAIL' else 'UNRESOLVED'
  records.append({'case_id':c['case_id'],'native_sha256':c['native_sha256'],'density_mm':int(density),'prox_status_conditional':prox,'pointwise_orientation_status':point,'strict_relationship_status_conditional':strict,'proximity_checks':checks,'orientation_checks':normal})
assert len(records)==9 and all(r['strict_relationship_status_conditional']=='FAIL' for r in records)
result={'schema':'hug_scoped_acceptance_application_v1','acceptance_rule_sha256':rh,'evaluator_sha256':sha(__file__),'input_reports_sha256':{'sampling':sha(bundle/'sampling/sampling_uncertainty.json'),'orientation_uncompressed':hashlib.sha256(gzip.decompress((bundle/'orientation/orientation_report_full.json.gz').read_bytes())).hexdigest(),'native':sha(bundle/'native_contract/native_contract.json')},'numerical_conversion_status':'PASS_WITHIN_RECORDED_TESTED_SCOPE','native_decoding_count':32,'exact_native_reconstruction':True,'numerical_scope':'All32savednative3Dexports; previous frame/named-joint/fullsourcebounds/targetFK checks, sourcehash guards and six unchanged current surface poses. Camera projection remains representativecube only. No universal or anatomy claim.','claim':rule['scope'],'primary_uniform_numeric_error_allowance_m':sp['primary_allowance_m'],'conditional_numeric_bounds':True,'formal_global_arithmetic_certificate':False,'case_count':6,'resolution_count':9,'strict_fail_resolution_count':sum(r['strict_relationship_status_conditional']=='FAIL' for r in records),'records':records,'full_physical_contact_equivalence':'UNRESOLVED','failure_label_preservation':'NOT_TESTED','independent_unseen_object_validation':'NOT_TESTED','approximate_predictor_pipeline_adequacy':'NOT_TESTED','new_eval_data':False,'physics_steps':0,'note':'FAIL concerns strict same-part proximity/direction preservation at declared scales on encoded geometry. This is not a task failure label or a rejection of all useful approximate retargeters. All source poor poses remain included; acceptance rule not adjusted to pass current mapping.'}
(base/'acceptance_application.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('records','input_reports_sha256')},indent=2))
