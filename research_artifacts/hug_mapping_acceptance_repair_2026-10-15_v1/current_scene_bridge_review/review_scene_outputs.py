"""Read-only saved-array and predicate replay; no simulator imports or geometry."""
import hashlib,json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0
def ck(v,msg):
 global checks
 checks+=1
 if not v:raise AssertionError(msg)
def norm(v):return math.sqrt(math.fsum(x*x for x in v))
def normdiff(a,b):return norm([x-y for x,y in zip(a,b,strict=True)])
def maxdiff(a,b):return max(abs(x-y) for x,y in zip(a,b,strict=True))
def flat(a):return [v for r in a for v in r]
def close(a,b,msg,eps=2e-18):ck(abs(a-b)<=eps,msg)
def quat(q):
 w,x,y,z=q;s=math.sqrt(sum(t*t for t in q));w,x,y,z=[t/s for t in q]
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
def pos(T):return [r[3] for r in T[:3]]
def rot(T):return [r[:3] for r in T[:3]]
root=Path(sys.argv[1]);dest=Path(sys.argv[2]);ck(not dest.exists(),'no overwrite')
p=json.loads((root/'protocol.json').read_text());r=json.loads((root/'results.json').read_text());ph=sha(root/'protocol.json');tol=p['engineering_tolerances'];bindings={str(root/'protocol.json'):ph,str(root/'results.json'):sha(root/'results.json')};rows=[]
ck(r['protocol_sha256']==ph,'protocol');ck(len(r['cases'])==r['case_count']==32,'32 records');ck([x['index'] for x in r['cases']]==list(range(32)),'order'); ck(r['actual_static_forward_calls']==128,'budget')
for flag in ('physics_steps','new_retargeting_solves','new_native_proposals','forbidden_execution_calls'):ck(r[flag]==0,flag)
for flag in ('new_eval_data','native_geometry_equivalence_validated','servo_tracking_validated','historical_surface_failures_reclassified'):ck(r[flag] is False,flag)
for entry in r['cases']:
 cp=Path(entry['path']);ck(sha(cp)==entry['sha256'],'case hash');bindings[str(cp)]=entry['sha256'];c=json.loads(cp.read_text());i=entry['index'];ck(c['source_inventory_index']==i,'index');ck(c['protocol_sha256']==ph,'case protocol');ck(c['literal_command']==p['cases'][i],'fixed command');ck(c['native_sha256']==p['cases'][i]['native_sha256'],'native')
 names=c['literal_command']['joint_names'];q=c['literal_command']['qpos_rad'];ck(len(names)==len(q)==24 and len(set(names))==24,'names')
 limits=[]
 for j,v in zip(c['literal_source_joint_limit_rows'],q,strict=True):
  lo,hi=j['source_limits_rad'];ck(j['value_rad']==v,'limit value');ok=lo<=v<=hi;ck(j['literal_pass']==ok,'limit predicate');close(j['violation_rad'],max(lo-v,v-hi,0),'limit excess');limits.append(ok)
 ck(c['all_literal_source_joint_limits_pass']==all(limits),'all limits')
 for j,n,v in zip(c['current_joint_metadata'],names,q,strict=True):
  ck(j['name']==n and j['value_rad']==v,'metadata names');e=maxdiff(j['compiled_limits_rad'],j['source_limits_rad']);close(j['limit_max_difference_rad'],e,'limit error');ok=e<=tol['source_compiled_joint_limit_rad'];ck(j['source_compiled_ranges_match']==ok and j['source_compiled_limits_match']==(ok and j['limited']),'compiled limits')
 body=c['current_hand_body_virtual_link_checks'];bodyok=[]
 for b in body['compiled_body_rows']+body['fixed_virtual_link_rows']:
  # Both kinds carry their source and realized frames.
  if 'compiled_world_position_m' in b:
   pe=normdiff(b['compiled_world_position_m'],b['xml_world_position_m']);re=maxdiff(flat(b['compiled_world_rotation']),flat(b['xml_world_rotation']));close(pe,b['position_error_m'],'body position');close(re,b['rotation_matrix_max_error'],'body rotation')
  else:pe=b['position_error_m'];re=b['rotation_matrix_max_error']
  ok=pe<=tol['current_position_m'] and re<=tol['current_rotation_matrix'];ck(b['passed']==ok,'body predicate');bodyok.append(ok)
 ck(body['passed']==all(bodyok),'body all')
 ob=c['current_object_checks'];go=[]
 for g in ob['geometry_rows']:
  if g.get('kind')=='legacy_box':go.append(g['passed']);continue
  ck(g['source_compiled_vertex_set_error_m']>=0,'mesh nonnegative')
  if g['collision_active']:ok=g['compiled_support']['graph_contains_all_full_hull_extreme_vertex_indices']
  else:
   v=g['visual_triangle_check'];ck(v['source_triangle_count']==v['compiled_triangle_count'],'triangles');ok=v['passed'];ck(ok==(v['maximum_compiled_vertex_to_source_m']<=tol['mesh_vertex_set_m'] and v['oriented_geometric_triangle_multisets_equal']),'visual predicate')
  ok=ok and g['source_compiled_vertex_set_error_m']<=tol['mesh_vertex_set_m'] and g['masks_margin_gap_group_friction_match'];ck(g['passed']==ok,'object geom predicate');go.append(ok)
 inventory=ob['actual_object_geom_ids']==ob['expected_object_geom_ids'];ck(ob['complete_object_geom_inventory']==inventory,'object inventory');other=ob['position_error_m']<=tol['current_position_m'] and ob['rotation_matrix_max_error']<=tol['current_rotation_matrix'] and ob['mass_error_kg']<=1e-12 and ob['inertial_position_error_m']<=tol['current_position_m'] and inventory and all(go);objok=other and ob['inertia_tensor_max_error_kg_m2']<=1e-12;ck(ob['passed']==objok,'object predicate')
 control=c['current_controller_and_table_checks'];ck(len(control['servo_rows'])==control['servo_count']==24,'servo count')
 for s in control['servo_rows']:ck(s['readback_ctrl_rad']==s['named_target_rad']==dict(zip(names,q))[s['name']],'servo named target')
 co=all(s['passed'] for s in control['servo_rows']) and all(control[k] for k in ('ideal_arm_weld','mocap_input_identity','table_matches_declared_shape_frame','no_extra_pair_exclusion_plugin_flex','complete_scene_geometry_inventory'));ck(control['passed']==co,'controller predicate')
 expected=c['native_frame_reference']['palm_world'];helper=c['actual_helper_palm_world'];close(c['helper_vs_independent_position_error_m'],normdiff(pos(helper),pos(expected)),'native helper pos');close(c['helper_vs_independent_rotation_matrix_max_error'],maxdiff(flat(rot(helper)),flat(rot(expected))),'native helper rot');close(c['staged_palm_position_error_m'],normdiff(pos(c['actual_world_palm']),pos(expected)),'staged palm pos');close(c['staged_palm_rotation_matrix_error'],maxdiff(flat(rot(c['actual_world_palm'])),flat(rot(expected))),'staged palm rot');close(c['source_root_position_error_m'],normdiff(c['actual_root_position_m'],pos(c['independent_root_world'])),'source root pos');close(c['current_root_change_from_historical_saved_m'],normdiff(c['actual_root_position_m'],c['literal_command']['root_position_m']),'root delta')
 passed=c['named_q_and_controller_inputs_exact'] and body['passed'] and c['current_hand_collision_checks']['passed'] and objok and co and c['current_declared_timing_options_match'] and all(j['source_compiled_limits_match'] for j in c['current_joint_metadata']) and max(c[k] for k in ('helper_vs_independent_position_error_m','source_root_position_error_m','staged_palm_position_error_m'))<=tol['current_position_m'] and max(c[k] for k in ('helper_vs_independent_rotation_matrix_max_error','source_root_rotation_matrix_error','staged_palm_rotation_matrix_error'))<=tol['current_rotation_matrix']
 ck(c['current_assembled_static_command_realization_pass']==entry['passed']==passed,'full predicate');ck(c['current_command_within_source_limits_and_realized']==(passed and all(limits)),'limits and realized');ck(c['actual_static_forward_calls']==4 and c['physics_steps']==0 and not c['new_eval_data'],'scope')
 rows.append({'index':i,'object_id':c['object_id'],'passed':passed,'all_literal_source_limits_pass':all(limits),'object_other_than_inertia_pass':other,'object_inertia_error_kg_m2':ob['inertia_tensor_max_error_kg_m2'],'hand_pass':c['current_hand_collision_checks']['passed'],'body_frames_pass':body['passed'],'controller_pass':co,'native_helper_position_error_m':c['helper_vs_independent_position_error_m'],'native_helper_rotation_matrix_error':c['helper_vs_independent_rotation_matrix_max_error'],'historical_root_delta_m':c['current_root_change_from_historical_saved_m']})
ck(r['all_current_assembled_static_command_realizations_pass']==all(x['passed'] for x in rows),'summary all');ck(r['all_literal_source_limits_pass']==all(x['all_literal_source_limits_pass'] for x in rows),'summary limits')
drift=[{'path':path,'expected_sha256':h,'actual_sha256':sha(path) if Path(path).exists() else None} for path,h in p['bindings'].items() if not Path(path).exists() or sha(path)!=h]
report={'passed':True,'scope':'Independent saved-array/hash/boolean replay only; no simulator imports or FK/geometry. Source inertial values are saved observations, not independently remeasured here. PASS means this review is consistent, not that the bridge passed.','check_count':checks,'source_bindings':bindings,'runtime_source_closure_reported_unchanged':r['source_bindings_unchanged'],'current_live_binding_differences':drift,'cases':rows,'passed_cases':sum(x['passed'] for x in rows),'failed_indices':[x['index'] for x in rows if not x['passed']],'static_forward_calls_reported':r['actual_static_forward_calls'],'physics_steps':0,'reviewer_geometry_calls':0,'new_eval_data':False,'replay_script_sha256':sha(__file__)}
dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(dest),'sha256':sha(dest),'checks':checks,'passed_cases':report['passed_cases'],'failed_indices':report['failed_indices'],'live_binding_differences':len(drift)}))
