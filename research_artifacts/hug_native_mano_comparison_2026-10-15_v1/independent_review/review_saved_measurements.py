"""Read-only independent aggregation of immutable32 static case reports."""
from pathlib import Path
import json,hashlib,statistics
HERE=Path(__file__).parent;B=Path('/private/tmp/hug_native_mano_2026-10-15_v1/measurement')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pp=B/'protocol.json';p=json.loads(pp.read_text());cases=[]
for path in sorted((B/'cases').glob('case_*.json')):
 r=json.loads(path.read_text());cases.append(r)
assert [r['index'] for r in cases]==list(range(32));assert all(c['physics_steps']==0 and c['object_probes']['unique_points']==512 for c in cases)
result={'schema':'independent_saved_static_comparison_readback_v1','protocol_sha256':sha(pp),'case_count':len(cases),'case_report_sha256':{str(x):sha(x) for x in sorted((B/'cases').glob('case_*.json'))},'source_native_module_sha256':p['bindings'][p['project']+'/src/grasp_failure_prediction/evaluation/native_mano.py'],'all512_object_probes_distinct':True,'all32_source_reconstruction_pass':all(c['native_reconstruction_pass'] for c in cases),'all32_native_bone_import_pass':all(c['bone_numerical_import_pass'] for c in cases),'part_case_count':len(cases)*6,'all192_native_surface_mean_lower_than_Shadow':all(r['native_mesh_to_mano_proxy']['mean_m']<r['native_mesh_to_shadow']['mean_m'] for c in cases for r in c['parts']),'embodiments':{},'new_eval_data':False,'physics_steps':0,'interpretation':'Descriptive finite static comparison, no orientation/continuous-coverage/physical validity acceptance.'}
for key in ('mano','shadow'):
 tips=[x for c in cases for x in c['mano_rigid_tip_errors_m' if key=='mano' else 'shadow_tip_errors_m']]
 field='native_mesh_to_mano_proxy' if key=='mano' else 'native_mesh_to_shadow'
 bands=[]
 for band in [0.001,0.002,0.005]:
  entries=[t for c in cases for r in c['parts'] for t in r['pointwise_band_changes'] if t['band_m']==band]
  bands.append({'band_m':band,'lost':sum(t[key+'_lost'] for t in entries),'added':sum(t[key+'_added'] for t in entries),'native_near':sum(t['native_near'] for t in entries)})
 result['embodiments'][key]={'tip_mean_m':statistics.mean(tips),'tip_max_m':max(tips),'unweighted_mean_of192_area_weighted_part_case_surface_means_m':statistics.mean(r[field]['mean_m'] for c in cases for r in c['parts']),'max_native_triangle_centroid_to_samepart_component_surface_m':max(r[field]['max_m'] for c in cases for r in c['parts']),'unweighted_mean_of192part_case_object_probe_errors_m':statistics.mean(r['object_probe_distance_error_'+key]['mean_m'] for c in cases for r in c['parts']),'pointwise_band_flag_changes':bands}
result['caveats']=['Measurements reuse32 development proposals with one shared MANO shape, not held-out validation.','Rigid neutral tip-site landmarks differ from pose-corrective blended skin vertices even for exact bone kinematics.','Native collision proxy is15capsules and wrist-owned convexpalm, not exact MANO skin or a reproduction of unpublished HUG physics.','Native-mesh trianglecentroid distances are one-directional finite samples, not Hausdorff or continuous surface coverage.','Object probes report unsigned component-boundary distance; may include internal overlapped component boundaries.','Different anatomical embodiment is reduced but native proxy still shows pointwise changes; no original strict criterion is reclassified.']
output=HERE/'saved_measurement_review.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['embodiments'],indent=2))
