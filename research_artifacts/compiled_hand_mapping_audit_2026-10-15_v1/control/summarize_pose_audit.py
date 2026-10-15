import json,hashlib
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parent

def sha(p):return 'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
def mx(rows,k):return {metric:max(r[k][metric] for r in rows) for metric in rows[0][k]}
B=json.loads((P/'pose_audit_before_v3.json').read_text());A=json.loads((P/'pose_audit_after_v1.json').read_text())
assert B['case_counts_by_generator_variant']==A['case_counts_by_generator_variant']
assert all(B[k]==A[k] for k in ['saved_pose_case_count','manufactured_control_count','case_count_with_controls','links_compared_per_case_excluding_world','collision_geoms_compared_per_case'])
for stage in ['base','tabletop','actuated']:
 assert [c['case_id'] for c in B['results'][stage]['cases']]==[c['case_id'] for c in A['results'][stage]['cases']]
 beforeinputs={r['path']:r['sha256'] for r in B['inputs'] if not r['path'].endswith('.mjb')};afterinputs={r['path']:r['sha256'] for r in A['inputs'] if not r['path'].endswith('.mjb')};assert beforeinputs==afterinputs
pergen={}
for gen in A['case_counts_by_generator']:
 a=[r for r in A['results']['actuated']['cases'] if r['generator']==gen];b=[r for r in B['results']['actuated']['cases'] if r['generator']==gen]
 pergen[gen]={'saved_case_count':len(a),'variants':dict(Counter(r['variant'] for r in a)),'before_target_import_max_link_error':mx(b,'maximum_link_error'),'after_target_import_max_link_error':mx(a,'maximum_link_error'),'before_target_import_max_collider_frame_error':mx(b,'maximum_collider_frame_error'),'after_target_import_max_collider_frame_error':mx(a,'maximum_collider_frame_error'),'target_urdf_domain_valid_count':sum(not r['target_limits_violations_urdf'] for r in a),'original_statuses_preserved':dict(Counter(r['original_status'] for r in a)),'before_compiled_limit_disagreements':sum(not r['limit_classification_same'] for r in b),'after_compiled_limit_disagreements':sum(not r['limit_classification_same'] for r in a),'generator_acceptance_from_this_audit':False,'scope':'Static target-import conversion only; source fidelity and native reconstruction remain separate.'}
alias=[]
for link in ['mfdistal','rfdistal','lfdistal']:
 key=link+'/collision/0';b=next(g for g in B['results']['base']['static']['declared_collider_import'] if g['key']==key);a=next(g for g in A['results']['base']['static']['declared_collider_import'] if g['key']==key)
 alias.append({'key':key,'declared_vertex_count':a['declared_vertex_count'],'before_compiled_vertex_count':b['compiled_vertex_count'],'after_compiled_vertex_count':a['compiled_vertex_count'],'before_vertex_cloud_error_m':b['bidirectional_vertex_cloud_difference_m'],'after_vertex_cloud_error_m':a['bidirectional_vertex_cloud_difference_m'],'vertex_cloud_error_is_not_surface_distance':True})
changed=[]
for a,b in zip(A['results']['actuated']['cases'],B['results']['actuated']['cases']):
 if not b['limit_classification_same']:
  changed.append({'case_id':a['case_id'],'generator':a['generator'],'actual_declared_target_violations':a['target_limits_violations_urdf'],'before_compiled_violations':b['target_limits_violations_compiled'],'after_compiled_violations':a['target_limits_violations_compiled'],'original_scientific_status':a['original_status']})
r={'schema_version':'cross_generator_compiled_target_conversion_comparison_v1','status':'CONVERSION_DEFECTS_CORRECTED_STATIC_DIAGNOSTICS_ONLY','saved_pose_cases':A['saved_pose_case_count'],'manufactured_cases':A['manufactured_control_count'],'distinct_native_output_count_claimed':False,'compiled_configurations':['base','tabletop','actuated'],'body_and_fixed_link_frames_per_case':A['links_compared_per_case_excluding_world'],'collision_geometries_per_case':A['collision_geoms_compared_per_case'],'before':B['summary'],'after':A['summary'],'per_generator':pergen,'wrong_collision_mesh_alias_correction':alias,'saved_case_compiled_legality_corrections':changed,'same_inputs_except_compiled_models':True,'raw_vs_quaternion_rotation_errors_separately_reported':True,'target_asset_and_acceptance_criteria_unchanged':True,'no_source_fidelity_or_generator_acceptance_inferred':True,'mesh_import_precision_limit':'Maximum remaining declared-vs-compiled vertex-cloud residual is 10.192 nm on forearm, associated with finite-precision mesh import; no 1 nm global geometry bound, triangle surface, convex collision exterior or scientific acceptance is certified.','prior_history_preserved':True,'zero_execution':A['zero_execution'],'reports':[ref(P/'pose_audit_before_v3.json'),ref(P/'pose_audit_after_v1.json'),ref(P/'pose_audit_manufactured_controls_v1.json')],'independent_oracle_review':ref(P.parent/'shadow/independent_pose_oracle_review_v1.json'),'summary_script':ref(Path(__file__))}
(P/'compiled_pose_comparison_summary_v1.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'summary_sha256':sha(P/'compiled_pose_comparison_summary_v1.json'),'per_generator':{g:{k:v for k,v in d.items() if k in ['saved_case_count','target_urdf_domain_valid_count','before_compiled_limit_disagreements','after_compiled_limit_disagreements']} for g,d in pergen.items()},'saved_case_corrections':changed},indent=2))
