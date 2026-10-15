"""Read-only numerical receipt; no distance queries or acceptance revisions."""
from pathlib import Path
import hashlib,json
import numpy as np
import audit_raw_surfaces_v1 as r
p=json.loads((r.HERE/'policy_frozen_v1.json').read_text());r.verify(p)
v,s,a,h,b,m=r.modules(p);records=json.loads(Path(p['accounting']).read_text())['records']
with np.load(p['poses'],allow_pickle=False) as z:poses={k:z[k] for k in z.files}
checks=[]
for family in ('visual','palm'):
 root=r.HERE/f'{family}_results_v1';report=json.loads((root/'report.json').read_text())
 assert [c['global_sample_index'] for c in report['cases']]==p['measurement_order']
 for c in report['cases']:
  g=c['global_sample_index'];i=p['source_valid_indices'].index(g)
  for spacing,measure in c['resolutions'].items():
   path=root/measure['query_arrays']['path'];assert r.sha(path)==measure['query_arrays']['sha256']
   with np.load(path,allow_pickle=False) as z:fields={k:z[k] for k in z.files}
   points=fields['cube_query_m'];assert len(points)==measure['query_count'];assert np.isfinite(points).all();assert np.all(np.abs(points)<=.025)
   x={'family':family,'global_sample_index':g,'spacing_mm':float(spacing),'query_count':len(points),'archive_sha256':r.sha(path),'source_and_target':{}}
   if family=='palm':
    for owner in ('source','target'):
     result={k.split('__',1)[1]:value for k,value in fields.items() if k.startswith(owner+'__')};m.verify_query_result(result,len(points))
     delta=points-result['closest_m'];recount=np.sqrt(np.einsum('ij,ij->i',delta,delta));error=float(np.max(np.abs(recount-result['unsigned_distance_m'])));assert error<1e-12
     t=poses['object_from_'+owner+'_palm'][i];local_einsum=np.einsum('ij,jk->ik',points-t[:3,3],t[:3,:3]);local_matmul=(points-t[:3,3])@t[:3,:3];local_error=float(np.max(np.abs(local_einsum-local_matmul)));assert local_error<1e-12
     x['source_and_target'][owner]={'all_required_outputs_finite_and_consistent':True,'max_saved_closest_distance_recount_error_m':error,'max_matmul_vs_einsum_input_transform_difference_m':local_error,'inside_count':int(result['inside_union'].sum()),'outside_count':int(result['outside_union'].sum())}
   else:
    for owner in ('source','target'):
     d=fields[owner+'_distance_by_part_m'];assert d.shape==(len(points),len(s.PARTS));assert not np.isnan(d).any();assert np.all(d>=0)
     x['source_and_target'][owner]={'distance_field_no_NaN':True,'negative_distance_count':int(np.sum(d<0)),'censored_positive_infinity_count':int(np.isposinf(d).sum())}
   checks.append(x)
summary={'status':'PASS_STATIC_SAVED_FIELD_HASH_FINITE_AND_DISTANCE_RECOUNT','policy_sha256':r.sha(r.HERE/'policy_frozen_v1.json'),'checker_sha256':r.sha(__file__),'archive_count':len(checks),'case_count_per_family':11,'all16_original_accounting_retained':True,'raw_target_angles_unchanged':bool(np.array_equal(poses['target_joint_values'],poses['source_hand_qpos'][:,9:])),'checks':checks,'retained_palm_runtime_warnings':str(r.HERE/'palm_stderr_v1.log'),'warning_scope':'Required palm outputs pass finite, unit-normal, partition and independent closest-distance checks. Matmul/einsum input transforms agree. These checks do not create a directed-rounding proof or remove the declared conditional arithmetic limitation.','new_surface_queries':0,'physics_steps':0}
r.write(r.HERE/'saved_field_checks_v1.json',summary)
print(json.dumps({'status':summary['status'],'archive_count':len(checks),'sha256':r.sha(r.HERE/'saved_field_checks_v1.json')}))
