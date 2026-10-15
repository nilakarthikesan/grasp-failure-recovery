#!/usr/bin/env python3
"""Read sealed native DGA poses, build unchanged-angle palm placement, query static surfaces.
No simulator, generator, optimizer, or training imports. Raw target-invalid poses remain diagnostic.
"""
import argparse, hashlib, importlib.util, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
OLD=Path('/private/tmp/dga_component_constraints_20261015_v1')
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
SCIENCE='sha256:57531e1e79a98f8e6b667bdd9a43e41a5e90ce8bdec35e56a833408baa106375'
JOINTS=('WRJ2','WRJ1','FFJ4','FFJ3','FFJ2','FFJ1','MFJ4','MFJ3','MFJ2','MFJ1','RFJ4','RFJ3','RFJ2','RFJ1','LFJ5','LFJ4','LFJ3','LFJ2','LFJ1','THJ5','THJ4','THJ3','THJ2','THJ1')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return 'sha256:'+h.hexdigest()
def write(path,x):
 with Path(path).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def modules(p):
 sys.path.insert(0,str(OLD/'geometry'))
 v=load('raw_visual_v3',p['visual_validator']);s=load('raw_surface_math',p['surface_primitives']);a=load('raw_independent_fk',p['independent_fk']);h=load('raw_native_helper',p['native_helper']);b=load('raw_triangle_backend',p['triangle_backend']);m=load('raw_palm_math',p['palm_math'])
 s.closest_triangles=lambda point,triangles:b.closest_triangle_pairs(np.repeat(np.asarray(point)[None,:],len(triangles),axis=0),triangles)
 return v,s,a,h,b,m
def verify(p):
 for path,digest in p['dependencies'].items():
  if sha(path)!=digest:raise ValueError('Dependency changed: '+path)
 if p['scientific_protocol_sha256']!=SCIENCE or sha(p['scientific_protocol'])!=SCIENCE:raise ValueError('Science changed')
 if p['code_sha256']!=sha(__file__) or p['spacing_mm_by_global_index']!={str(g):([1.,.5,.25] if g==6 else [1.]) for g in p['source_valid_indices']}:raise ValueError('Implementation or fixed query scope changed')
 if p['bands_mm']!=[.5,1.,2.] or p['distance_allowance_m']!=1e-9 or p['query_fractions']!=[2/5,1/3] or p['execution_eligible'] is not False:raise ValueError('Frozen measurement scope changed')
def select(ledger):
 rows=ledger['records']
 if ledger['sample_count']!=16 or [r['global_sample_index'] for r in rows]!=list(range(16)):raise ValueError('All16 ledger required')
 if any(type(r['native_output_complete_and_finite']) is not bool or not isinstance(r['source_limit_violations'],list) for r in rows):raise ValueError('Explicit integrity and source legality required')
 return [r['global_sample_index'] for r in rows if r['native_output_complete_and_finite'] and not r['source_limit_violations']]
def resolve_child(base,ref):
 child=(Path(base)/ref['path']).resolve()
 if not child.is_relative_to(Path(base).resolve()) or sha(child)!=ref['sha256']:raise ValueError('Child hash/path changed')
 return child
def build(p):
 """No surface queries; preserve raw values and form target root by rigid palm composition."""
 v,s,a,h,b,m=modules(p);ledger=json.loads(Path(p['ledger']).read_text());algebra=json.loads(Path(p['source_frames']).read_text());indices=select(ledger)
 source=a.model(Path(p['source_urdf']));target=a.model(Path(p['target_urdf']))
 if source['root']!='forearm' or target['root']!='world':raise ValueError('Wrong hand roots')
 if [r['global_sample_index'] for r in algebra['records']]!=list(range(16)):raise ValueError('All16 source frames required')
 arr={k:[] for k in ('global_sample_indices','source_hand_qpos','target_joint_values','object_from_source_forearm','object_from_source_palm','object_from_target_palm','source_palm_from_target_palm','object_from_target_root_world','object_from_target_forearm')};records=[];extra={}
 for g,row in enumerate(ledger['records']):
  rec={'global_sample_index':g,'original_mapping_status':row['mapping_status'],'original_source_limit_violations':row['source_limit_violations'],'original_target_limit_violations':row['target_limit_violations'],'original_selected_for_bounded_mapping':row['selected_for_bounded_mapping'],'native_default_reference_status':row['native_default_runtime_reconstruction']['status'],'fp32_reference_status':row['fp32_mapping_reference_reconstruction']['status'],'execution_eligible':False,'surface_disposition':'NOT_TESTED_SOURCE_INVALID' if g not in indices else 'PRIMARY_UNCHANGED_ANGLE_TARGET_LEGAL' if g==6 else 'TARGET_ILLEGAL_DIAGNOSTIC_ONLY'}
  records.append(rec)
  if g not in indices:continue
  if row['native_mapping_input_prerequisites_pass'] is not True or row['fp32_mapping_reference_reconstruction']['status']!='PASS':raise ValueError('Native source/reference prerequisites failed')
  nativepath=Path(row['native_manifest']['path']);extra[str(nativepath)]=row['native_manifest']['sha256'];native=json.loads(nativepath.read_text());rawpath=resolve_child(nativepath.parent,native['output']);extra[str(rawpath)]=sha(rawpath)
  with np.load(rawpath,allow_pickle=False) as z:raw=z['hand_qpos'][row['within_batch_sample_index']]
  if raw.shape!=(33,) or not np.isfinite(raw).all():raise ValueError('Invalid raw33')
  q=dict(zip(JOINTS,raw[9:].tolist(),strict=True))
  if q!=row['joint_targets_rad'] or h.source_limit_violations(source,q):raise ValueError('Source raw joint contract')
  target_violations=h.source_limit_violations(target,q)
  if target_violations!=row['target_limit_violations']:raise ValueError('Target violations changed')
  sf=a.independent_fk(source,q);tf=a.independent_fk(target,q);frames=algebra['records'][g]['transforms'];S=np.array(frames['object_from_palm']);F=np.array(frames['object_from_source_forearm']);T=S.copy();root=T@np.linalg.inv(tf['palm'])
  for x in (S,F,T,root):m.proper(x)
  if not np.allclose(F@sf['palm'],S,atol=1e-9,rtol=0) or not np.allclose(root@tf['palm'],S,atol=1e-9,rtol=0):raise ValueError('Palm composition failed')
  ev=row['mapping_reference'];refpath=Path(ev['path']);extra[str(refpath)]=ev['sha256'];ref=json.loads(refpath.read_text());refnpz=resolve_child(refpath.parent,ref['arrays']);extra[str(refnpz)]=sha(refnpz)
  rec.update(raw33_row_sha256='sha256:'+hashlib.sha256(raw.tobytes()).hexdigest(),raw33_dtype=str(raw.dtype),target_joint_targets_rad=q,object_from_target_palm=T.tolist(),object_from_target_root_world=root.tolist(),raw_native_file=str(rawpath),reference_file=str(refpath),reference_arrays=str(refnpz),within_batch_sample_index=row['within_batch_sample_index'],joint_change_max_abs_rad=0.,palm_compensation_composition_max_abs=float(np.max(np.abs(root@tf['palm']-S))),native_manifest=row['native_manifest'],mapping_reference=ev)
  vals={'global_sample_indices':g,'source_hand_qpos':raw,'target_joint_values':raw[9:].copy(),'object_from_source_forearm':F,'object_from_source_palm':S,'object_from_target_palm':T,'source_palm_from_target_palm':np.eye(4),'object_from_target_root_world':root,'object_from_target_forearm':root@tf['forearm']}
  for k,value in vals.items():arr[k].append(value)
 arr={k:np.asarray(values,dtype=np.int64 if k=='global_sample_indices' else None) for k,values in arr.items()}
 if [r['global_sample_index'] for r in records if r['global_sample_index'] in indices and not r['original_target_limit_violations']]!=[6]:raise ValueError('Expected sole legal raw target row6')
 return arr,records,extra

def prepare():
 old=json.loads((OLD/'validation/native_implementation_frozen_v3.json').read_text());p={k:old[k] for k in ('surface_primitives','independent_fk','native_helper','target_urdf','target_assets_root','triangle_backend','scientific_protocol','scientific_protocol_sha256','native_protocol','native_protocol_sha256')}
 p.update(schema_version='dga_raw_surface_static_policy_v1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),status='FROZEN_BEFORE_NEW_SURFACE_QUERIES',source_urdf='/Users/nilakarthikesan/grasp-failure-prediction/local_assets/dexgrasp_anything/source-86353d710ca71d0f547f0d8b15cc8ca959c78163/assets/urdf/sr_grasp_description/urdf/shadowhand.urdf',ledger=str(OLD/'native_validation/results_v1/native_validation_report.json'),source_frames=str(OLD/'native_validation/results_v1/native_source_frame_report.json'),visual_validator=str(OLD/'validation/validate_component_native_visual_v3.py'),palm_math=str(OLD/'geometry/measure_palm_holdout.py'),code_sha256=sha(__file__),bands_mm=[.5,1.,2.],distance_allowance_m=1e-9,query_fractions=[2/5,1/3],maximum_wall_seconds_per_case_per_family=600,maximum_triangle_tests_per_case=100000000,maximum_polygon_queries_per_case=150000000,maximum_reference_triangles=100000,execution_eligible=False,physics_steps=0,optimization_calls=0,source_reference='Previously verified FP32 geometry; original default TF32 remains failed',original_first_two_mapping_failures_preserved=True,original_preselection=[0,1],claim_scope='New static surfaces on all11 raw source-valid saved native poses. Row6 is sole target-legal case; other10 diagnostics cannot be accepted. Same original cube; no population generalization. Unchanged science; no fitting/clipping, no target/criteria changes.',arithmetic='F64 conditional 1nm allowance, no directed-rounding proof',full_mapping_acceptance='UNRESOLVED_OR_FAIL_NEVER_FROM_THIS_SINGLE_GEOMETRY_TEST')
 arr,records,extra=build(p);p['source_valid_indices']=arr['global_sample_indices'].tolist();p['measurement_order']=[6]+[g for g in p['source_valid_indices'] if g!=6];p['spacing_mm_by_global_index']={str(g):([1.,.5,.25] if g==6 else [1.]) for g in p['source_valid_indices']}
 paths=[Path(v) for k,v in p.items() if isinstance(v,str) and v.startswith('/') and Path(v).is_file()]+[Path(__file__)]
 paths+=list((OLD/'geometry').glob('*.py'))+[OLD/'validation/test_native_visual_validator_v3.py',Path('/private/tmp/dga_surface_preserving_20261015_v1/geometry/palm_box_union.py'),Path('/private/tmp/dga_surface_preserving_20261015_v1/geometry/palm_union_geometry_reference.py')]
 v,s,a,h,b,m=modules(p);target=a.model(Path(p['target_urdf']));paths += [Path(x['path']) for x in h.visual_meshes(target,Path(p['target_assets_root']),a,source_runtime=False).values()]
 p['dependencies']={str(x.resolve()):sha(x) for x in paths};p['dependencies'].update(extra)
 posepath=HERE/'unchanged_angle_poses_v1.npz'
 if posepath.exists():raise FileExistsError('Never overwrite candidates')
 np.savez_compressed(posepath,**arr);account=HERE/'all16_accounting_v1.json';write(account,{'schema_version':'dga_raw_surface_all16_accounting_v1','records':records,'source_valid_indices':p['source_valid_indices'],'surface_queries':0,'physics_steps':0,'original_preselection':[0,1],'original_preselected_failures_preserved':True})
 p['poses']=str(posepath);p['accounting']=str(account)
 for x in (posepath,account):p['dependencies'][str(x)]=sha(x)
 verify(p);write(HERE/'policy_frozen_v1.json',p);print(json.dumps({'policy':str(HERE/'policy_frozen_v1.json'),'sha256':sha(HERE/'policy_frozen_v1.json'),'source_valid_indices':p['source_valid_indices'],'surface_queries':0}))

def run(family):
 p=json.loads((HERE/'policy_frozen_v1.json').read_text());verify(p);v,s,a,h,b,m=modules(p);science=json.loads(Path(p['scientific_protocol']).read_text());records=json.loads(Path(p['accounting']).read_text())['records']
 with np.load(p['poses'],allow_pickle=False) as z:arr={k:z[k] for k in z.files}
 fresh,_,_=build(p)
 if any(arr[k].dtype!=fresh[k].dtype or arr[k].shape!=fresh[k].shape or arr[k].tobytes()!=fresh[k].tobytes() for k in arr):raise ValueError('Frozen raw candidates no longer reproduce')
 out=HERE/f'{family}_results_v1';out.mkdir(exist_ok=False);write(out/'premeasurement_receipt.json',{'policy_sha256':sha(HERE/'policy_frozen_v1.json'),'time_utc':datetime.now(timezone.utc).isoformat(),'family':family,'indices':p['measurement_order'],'surface_queries':0,'physics_steps':0})
 target=a.model(Path(p['target_urdf']));visuals=h.visual_meshes(target,Path(p['target_assets_root']),a,source_runtime=False) if family=='visual' else None
 if family=='palm':
  sourceboxes,source_decl=m.palm_boxes(p['source_urdf'],1);targetboxes,target_decl=m.palm_boxes(p['target_urdf'],8);sp=m.boundary_patches(sourceboxes);tp=m.boundary_patches(targetboxes)
 cases=[]
 for g in p['measurement_order']:
  i=p['source_valid_indices'].index(g);row=records[g];start=time.monotonic();deadline=start+p['maximum_wall_seconds_per_case_per_family'];case={'global_sample_index':g,'surface_disposition':row['surface_disposition'],'target_limit_violations':row['original_target_limit_violations'],'execution_eligible':False,'full_mapping_accepted':False,'resolutions':{}}
  if family=='visual':
   ref=json.loads(Path(row['reference_file']).read_text());j=row['within_batch_sample_index']
   with np.load(row['reference_arrays'],allow_pickle=False) as z:runtime={k:z[k] for k in z.files}
   q=row['target_joint_targets_rad'];frames=a.independent_fk(target,q);root=arr['object_from_target_root_world'][i]
   sm=[v.all_triangle_mesh_reference(s,n,runtime['vertices__'+n][j],runtime['faces__'+n]) for n in ref['vertex_links']];tm=[v.all_triangle_mesh_reference(s,f'{n}__visual{k}',h.object_vertices(x['local'],root@frames[n]),x['faces']) for (n,k),x in visuals.items()]
   if any(sum(len(x['triangles']) for x in meshes)>p['maximum_reference_triangles'] for meshes in (sm,tm)):raise ValueError('Reference size exceeded')
   case.update(exact_geometric_identity_by_part=v.part_identities(sm,tm,s.PARTS[:-1]),source_topology=[{'link':x['name'],**x['metadata']} for x in sm],target_topology=[{'link':x['name'],**x['metadata']} for x in tm],source_cube_profiles=s.hand_cube_profiles(sm),target_cube_profiles=s.hand_cube_profiles(tm));budget=[0,p['maximum_triangle_tests_per_case']]
  else:budget=[0]
  for spacing in p['spacing_mm_by_global_index'][str(g)]:
   points,normals,faces,area,rho=v.offcenter_grid(spacing) if family=='visual' else m.cube_grid(spacing);arrays={'cube_query_m':points,'cube_normals':normals,'cube_face':faces}
   if family=='visual':
    radius=max(p['bands_mm'])/1000+rho+2*s.NUMERICAL_DISTANCE_M;x=s.query_reference(points,sm,radius,budget,deadline);y=s.query_reference(points,tm,radius,budget,deadline);bands={}
    for band in p['bands_mm']:
     groups={}
     for k,name in enumerate((*s.PARTS,'whole_hand_common_runtime_coverage','whole_hand')):
      if name=='forearm_carrier':groups[name]={'status':'NOT_COMPARABLE','reason':'Native runtime omits forearm'};continue
      if name=='whole_hand':sd,td=x['nearest_distance_m'],y['nearest_distance_m']
      elif name=='whole_hand_common_runtime_coverage':sd,td=x['distance_by_part_m'][:,:-1].min(axis=1),y['distance_by_part_m'][:,:-1].min(axis=1)
      else:sd,td=x['distance_by_part_m'][:,k],y['distance_by_part_m'][:,k]
      groups[name]=s.pair_area(sd,td,area,radius,rho,band/1000)
     nr=s.conditional_normals(x,y,normals,band/1000,area);nr['reliable_exterior_witness_area_m2']=None;bands[str(band)]={'groups':groups,'conditional_facet_normals':nr}
    summary={'query_count':len(points),'area_per_cell_m2':area,'covering_radius_m':rho,'search_radius_m':radius,'bands':bands,'source_counters':x['counters'],'target_counters':y['counters']}
    for owner,result in [('source',x),('target',y)]:
     for key in ['distance_by_part_m','closest_by_part_m','normal_by_part','facet_topology_eligible','facet_tie_ambiguous','nearest_part','nearest_part_ambiguous','triangle_index_by_part']:arrays[owner+'_'+key]=result[key]
   else:
    x=m.query_object(points,arr['object_from_source_palm'][i],sourceboxes,sp,deadline,budget,p['maximum_polygon_queries_per_case']);y=m.query_object(points,arr['object_from_target_palm'][i],targetboxes,tp,deadline,budget,p['maximum_polygon_queries_per_case']);summary=m.summarize(x,y,normals,rho,area,p['bands_mm']);summary.update(query_count=len(points),cell_area_m2=area,covering_radius_m=rho)
    for owner,result in [('source',x),('target',y)]:arrays.update({owner+'__'+key:value for key,value in result.items()})
   archive=out/f'global_{g}_grid_{spacing:g}mm.npz';np.savez_compressed(archive,**arrays);summary['query_arrays']={'path':archive.name,'sha256':sha(archive)};case['resolutions'][str(spacing)]=summary;print(json.dumps({'family':family,'global':g,'spacing_mm':spacing,'tests':budget[0]}),flush=True)
  if family=='visual':
   claim=v.strict_case(case,science['native_reference']['matched_parts'],science['acceptance']['bands_mm']);claim['independent_native_scope']='All11 source-valid unchanged saved poses; row6 primary target-legal, 10 target-illegal diagnostic only. Same cube; no population claim.';case['strict_encoded_nearset_acceptance']=claim;case['family_status']=claim['status']
   case['resolution_consistency']={f'{coarse}_to_{fine}':s.resolution_summary(case['resolutions'][str(coarse)]['bands'],case['resolutions'][str(fine)]['bands']) for coarse,fine in zip(p['spacing_mm_by_global_index'][str(g)][:-1],p['spacing_mm_by_global_index'][str(g)][1:])}
   if any(not x['bounds_consistent'] for pairs in case['resolution_consistency'].values() for groups in pairs.values() for metrics in groups.values() for x in metrics.values()):raise ValueError('Inconsistent resolution bounds')
  else:
   near={str(b):m.coverage_status([r['bands'][str(b)]['strict_nearset_preservation_status'] for r in case['resolutions'].values()]) for b in p['bands_mm']};occupancy=m.coverage_status([r['occupancy_and_exclusion']['occupancy_preservation_status'] for r in case['resolutions'].values()]);direction='FAIL' if any(r['bands'][str(b)]['normal_point_facing_sign_change_count'] for r in case['resolutions'].values() for b in p['bands_mm']) else 'UNRESOLVED';case.update(strict_palm_nearset_preservation_by_band=near,strict_palm_occupancy_preservation=occupancy,qualified_palm_union_facing_relationship=direction,strict_declared_palm_proximity_and_occupancy_status=m.combine_status(list(near.values())+[occupancy]),family_status=m.combine_status(list(near.values())+[occupancy,direction]))
  case.update(elapsed_seconds=time.monotonic()-start,exact_formula_tests=budget[0]);write(out/f'global_{g}_report.json',v.rename_query_fields(case));cases.append(case)
 verify(p)
 if {'torch','mujoco','isaacgym','pybullet'} & set(sys.modules):raise ValueError('Forbidden runtime imported')
 report={'schema_version':'dga_raw_surface_family_report_v1','status':'COMPLETE_FROZEN_STATIC_DIAGNOSTIC','family':family,'policy_sha256':sha(HERE/'policy_frozen_v1.json'),'scientific_protocol_sha256':SCIENCE,'cases':cases,'all16_accounting':{'path':p['accounting'],'sha256':sha(p['accounting'])},'original_preselected_failures_preserved':True,'full_mapping_accepted':False,'execution_eligible':False,'limitations':['F64 conditional1nm allowance, not directed rounding','Literal visual triangle distances are not compiled collision or exterior-union contact','Declared palm normals are palm union only; full hand certified exterior directions unresolved','Target-invalid rows remain diagnostic regardless of surface metrics','Single native cube, no population claim; original default TF32 reference failure retained'],'zero_execution':{'physics_steps':0,'native_calls':0,'optimization_calls':0,'evaluation_episodes':0,'training_calls':0}}
 if family=='palm':report['palm_declarations']={'source':source_decl,'target':target_decl,'source_exposed_fragments':len(sp),'target_exposed_fragments':len(tp)}
 write(out/'report.json',v.rename_query_fields(report));print(json.dumps({'report':str(out/'report.json'),'sha256':sha(out/'report.json'),'case_statuses':{str(c['global_sample_index']):c['family_status'] for c in cases}}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','visual','palm']);args=ap.parse_args();prepare() if args.mode=='prepare' else run(args.mode)
