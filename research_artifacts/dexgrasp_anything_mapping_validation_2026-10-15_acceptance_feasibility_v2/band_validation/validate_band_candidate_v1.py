"""Fresh independently offset static visual queries for one immutable band candidate.
Native selection and prior failed cases remain untouched. No surface feedback.
"""
import argparse,importlib.util,json,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path('/private/tmp/dga_mapping_acceptance_20261015_v2');HERE=Path(__file__).resolve().parent;OLD=Path('/private/tmp/dga_component_constraints_20261015_v1')
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'surface'))
import audit_raw_surfaces_v1 as r
FRACTIONS=(13/37,19/41)
def grid(spacing_mm):
 h=spacing_mm/1000;side=int(round(.05/h))
 if not np.isclose(side*h,.05,atol=1e-15,rtol=0):raise ValueError('Grid must partition original cube')
 u=-.025+(np.arange(side)+FRACTIONS[0])*h;v=-.025+(np.arange(side)+FRACTIONS[1])*h;U,V=np.meshgrid(u,v,indexing='ij');points=[];normals=[];labels=[]
 for axis in range(3):
  other=[j for j in range(3) if j!=axis]
  for sign in (-1,1):
   p=np.zeros((side*side,3));p[:,axis]=sign*.025;p[:,other[0]]=U.ravel();p[:,other[1]]=V.ravel();n=np.zeros_like(p);n[:,axis]=sign;points.append(p);normals.append(n);labels.extend([f'{"XYZ"[axis]}{"+" if sign>0 else "-"}']*len(p))
 rho=h*np.sqrt(sum(max(f,1-f)**2 for f in FRACTIONS))
 return np.concatenate(points),np.concatenate(normals),np.array(labels),h*h,rho

def palm_certificate(p):
 with np.load(p['poses'],allow_pickle=False) as z:new={k:z[k] for k in z.files}
 with np.load(p['previous_pose_arrays'],allow_pickle=False) as z:old={k:z[k] for k in z.files}
 i=old['global_sample_indices'].tolist().index(1);same={}
 for k in ['source_hand_qpos','object_from_source_palm','object_from_target_palm']:
  same[k]=new[k][0].dtype==old[k][i].dtype and new[k][0].shape==old[k][i].shape and new[k][0].tobytes()==old[k][i].tobytes()
 if not all(same.values()):raise ValueError('Palm identity certificate failed')
 previous=json.loads(Path(p['previous_palm_report']).read_text());case=next(c for c in previous['cases'] if c['global_sample_index']==1)
 if case['strict_declared_palm_proximity_and_occupancy_status']!='PASS':raise ValueError('Previous palm family did not pass proximity/occupancy')
 for key in ('source_urdf','target_urdf'):
  prior=json.loads(Path(p['previous_palm_policy']).read_text());assert prior[key]==p[key] and prior['dependencies'][p[key]]==r.sha(p[key])
 return {'status':'EXACT_INPUT_IDENTITY_PRESERVES_PREVIOUS_DECLARED_PALM_PROXIMITY_AND_OCCUPANCY_PASS','global_sample_index':1,'exact_array_bytes':same,'source_and_target_palm_declarations_hash_unchanged':True,'previous_report':{'path':p['previous_palm_report'],'sha256':r.sha(p['previous_palm_report'])},'previous_strict_palm_nearset_preservation_by_band':case['strict_palm_nearset_preservation_by_band'],'previous_strict_palm_occupancy_preservation':case['strict_palm_occupancy_preservation'],'palm_directions':'UNRESOLVED','fresh_palm_queries':0,'full_mapping_accepted':False}

def verify(p):
 for path,digest in p['dependencies'].items():
  if r.sha(path)!=digest:raise ValueError('Frozen dependency changed: '+path)
 if p['code_sha256']!=r.sha(__file__) or p['scientific_protocol_sha256']!=r.SCIENCE or r.sha(p['scientific_protocol'])!=r.SCIENCE:raise ValueError('Code/science binding failed')
 if p['query_fractions']!=list(FRACTIONS) or p['spacing_mm']!=[1.,.5,.25] or p['bands_mm']!=[.5,1.,2.] or p['distance_allowance_m']!=1e-9:raise ValueError('Frozen scientific measurement changed')
 if p['maximum_triangle_tests']!=100000000 or p['maximum_wall_seconds']!=600 or p['execution_eligible'] is not False:raise ValueError('Frozen budget/scope changed')

def prepare():
 raw=json.loads((ROOT/'surface/policy_frozen_v1.json').read_text());p={k:raw[k] for k in ('source_urdf','target_urdf','target_assets_root','visual_validator','surface_primitives','independent_fk','native_helper','triangle_backend','palm_math','scientific_protocol','scientific_protocol_sha256')}
 p.update(schema_version='dga_band_single_candidate_visual_policy_v1',status='FROZEN_BEFORE_NEW_CANDIDATE_SURFACE_QUERIES',created_at_utc=datetime.now(timezone.utc).isoformat(),global_sample_index=1,stage='separate_band_proposal_candidate_v1',code_sha256=r.sha(__file__),poses=str(ROOT/'band_results_v1/candidate_pose.npz'),band_report=str(ROOT/'band_results_v1/report.json'),independent_guard_receipt=str(ROOT/'contract_review/band_saved_candidate_independent_recount_v1.json'),raw_accounting=str(ROOT/'surface/all16_accounting_v1.json'),previous_pose_arrays=str(OLD/'native_mapping_results_v1/fit_poses.npz'),previous_palm_report=str(OLD/'native_palm_holdout_results_v1/palm_collider_union_report.json'),previous_palm_policy=str(OLD/'geometry/native_inputs_frozen_v1.json'),query_fractions=list(FRACTIONS),query_fractions_exact=['13/37','19/41'],covering_radius_formula='spacing_m*sqrt((24/37)^2+(22/41)^2)',spacing_mm=[1.,.5,.25],bands_mm=[.5,1.,2.],distance_allowance_m=1e-9,maximum_triangle_tests=100000000,maximum_wall_seconds=600,maximum_reference_triangles=100000,execution_eligible=False,original_preselected_failures_preserved=True,independent_native_validation=False,holdout_query_scope='Fresh offsets independent of fitting and prior grids; already inspected native row1 is development, not independent validation of this new method.',selection='One already frozen selected candidate; no adoption from fitting proxy and no feedback/reoptimization from these results.')
 receipt=json.loads(Path(p['independent_guard_receipt']).read_text());assert receipt['status']=='PASS_ALL_SAVED_GUARDS_RECOUNTED' and receipt['adopted_changed_state'] is True
 for path,digest in receipt['reviewed_files'].items():assert r.sha(path)==digest
 p['dependencies']=dict(raw['dependencies'])
 for path in [__file__,str(HERE/'test_fresh_grid_v1.py'),str(HERE/'grid_controls_v1.log'),str(ROOT/'contract_review/band_implementation_review_v1.json')]+[x for x in p.values() if isinstance(x,str) and x.startswith('/') and Path(x).is_file()]:p['dependencies'][str(Path(path).resolve())]=r.sha(path)
 p['palm_identity_certificate']=palm_certificate(p)
 verify(p);r.write(HERE/'policy_frozen_v1.json',p);print(json.dumps({'policy_sha256':r.sha(HERE/'policy_frozen_v1.json'),'new_surface_queries':0}))

def run():
 p=json.loads((HERE/'policy_frozen_v1.json').read_text());verify(p);v,s,a,h,b,m=r.modules(p);science=json.loads(Path(p['scientific_protocol']).read_text());row=json.loads(Path(p['raw_accounting']).read_text())['records'][1]
 with np.load(p['poses'],allow_pickle=False) as z:pose={k:z[k] for k in z.files}
 if pose['global_sample_indices'].tolist()!=[1]:raise ValueError('Single frozen row required')
 report=json.loads(Path(p['band_report']).read_text());target=a.model(Path(p['target_urdf']));q=dict(zip(r.JOINTS,pose['target_joint_values'][0].tolist(),strict=True))
 if q!=report['selected_state']['q'] or h.source_limit_violations(target,q):raise ValueError('Target changed/invalid')
 assert all(q[k]==row['target_joint_targets_rad'][k] for k in ['WRJ1','WRJ2'])
 with np.load(row['raw_native_file'],allow_pickle=False) as z:raw=z['hand_qpos'][row['within_batch_sample_index']]
 assert raw.dtype==pose['source_hand_qpos'][0].dtype and raw.tobytes()==pose['source_hand_qpos'][0].tobytes()
 frames=a.independent_fk(target,q);root=pose['object_from_target_root_world'][0]
 assert np.max(abs(root@frames['palm']-pose['object_from_target_palm'][0]))<1e-9
 ref=json.loads(Path(row['reference_file']).read_text());j=row['within_batch_sample_index']
 with np.load(row['reference_arrays'],allow_pickle=False) as z:runtime={k:z[k] for k in z.files}
 visuals=h.visual_meshes(target,Path(p['target_assets_root']),a,source_runtime=False);sm=[v.all_triangle_mesh_reference(s,n,runtime['vertices__'+n][j],runtime['faces__'+n]) for n in ref['vertex_links']];tm=[v.all_triangle_mesh_reference(s,f'{n}__visual{k}',h.object_vertices(x['local'],root@frames[n]),x['faces']) for (n,k),x in visuals.items()]
 assert all(sum(len(x['triangles']) for x in meshes)<=p['maximum_reference_triangles'] for meshes in (sm,tm))
 out=HERE/'results_v1';out.mkdir(exist_ok=False);r.write(out/'premeasurement_receipt.json',{'policy_sha256':r.sha(HERE/'policy_frozen_v1.json'),'new_surface_queries':0,'physics_steps':0,'time_utc':datetime.now(timezone.utc).isoformat()});case={'global_sample_index':1,'stage':p['stage'],'resolutions':{},'exact_geometric_identity_by_part':v.part_identities(sm,tm,s.PARTS[:-1]),'full_mapping_accepted':False,'execution_eligible':False};start=time.monotonic();deadline=start+600;budget=[0,100000000]
 for spacing in p['spacing_mm']:
  points,normals,faces,area,rho=grid(spacing);radius=.002+rho+2*s.NUMERICAL_DISTANCE_M;source=s.query_reference(points,sm,radius,budget,deadline);targetfields=s.query_reference(points,tm,radius,budget,deadline);bands={};arrays={'cube_query_m':points,'cube_normals':normals,'cube_face':faces}
  for band in p['bands_mm']:
   groups={}
   for k,name in enumerate((*s.PARTS,'whole_hand_common_runtime_coverage','whole_hand')):
    if name=='forearm_carrier':groups[name]={'status':'NOT_COMPARABLE','reason':'Native runtime omits forearm'};continue
    if name=='whole_hand':sd,td=source['nearest_distance_m'],targetfields['nearest_distance_m']
    elif name=='whole_hand_common_runtime_coverage':sd,td=source['distance_by_part_m'][:,:-1].min(axis=1),targetfields['distance_by_part_m'][:,:-1].min(axis=1)
    else:sd,td=source['distance_by_part_m'][:,k],targetfields['distance_by_part_m'][:,k]
    groups[name]=s.pair_area(sd,td,area,radius,rho,band/1000)
   normals_result=s.conditional_normals(source,targetfields,normals,band/1000,area);normals_result['reliable_exterior_witness_area_m2']=None;bands[str(band)]={'groups':groups,'conditional_facet_normals':normals_result}
  for owner,fields in [('source',source),('target',targetfields)]:
   for key in ['distance_by_part_m','closest_by_part_m','normal_by_part','facet_topology_eligible','facet_tie_ambiguous','nearest_part','nearest_part_ambiguous','triangle_index_by_part']:arrays[owner+'_'+key]=fields[key]
  path=out/f'global_1_grid_{spacing:g}mm.npz';np.savez_compressed(path,**arrays);case['resolutions'][str(spacing)]={'query_count':len(points),'area_per_cell_m2':area,'covering_radius_m':rho,'search_radius_m':radius,'bands':bands,'query_arrays':{'path':path.name,'sha256':r.sha(path)}};print(json.dumps({'grid_mm':spacing,'triangle_tests':budget[0]}),flush=True)
 case['strict_encoded_nearset_acceptance']=v.strict_case(case,science['native_reference']['matched_parts'],science['acceptance']['bands_mm']);case['strict_encoded_nearset_acceptance']['development_cases_only']=True;case['strict_encoded_nearset_acceptance']['independent_native_scope']='None: one frozen proposal on already inspected native row1; this is fresh geometric-query validation only.'
 case['resolution_consistency']={f'{c}_to_{f}':s.resolution_summary(case['resolutions'][str(c)]['bands'],case['resolutions'][str(f)]['bands']) for c,f in [(1.,.5),(.5,.25)]}
 if any(not x['bounds_consistent'] for pairs in case['resolution_consistency'].values() for groups in pairs.values() for metrics in groups.values() for x in metrics.values()):raise ValueError('Inconsistent grid bounds')
 verify(p);assert palm_certificate(p)==p['palm_identity_certificate']
 if {'torch','mujoco','isaacgym','pybullet'}&set(sys.modules):raise ValueError('Forbidden runtime imported')
 final={'schema_version':'dga_band_candidate_fresh_visual_report_v1','status':'COMPLETE_FROZEN_SINGLE_CANDIDATE_GEOMETRY_TEST','policy_sha256':r.sha(HERE/'policy_frozen_v1.json'),'scientific_protocol_sha256':r.SCIENCE,'case':case,'palm_identity_certificate':p['palm_identity_certificate'],'strict_visual_status':case['strict_encoded_nearset_acceptance']['status'],'full_mapping_accepted':False,'execution_eligible':False,'direction_preservation':'UNRESOLVED_NO_COMPLETE_CERTIFIED_EXTERIOR_COVERAGE','original_preselected_failures_preserved':True,'independent_native_validation':False,'triangle_tests':budget[0],'elapsed_seconds':time.monotonic()-start,'limitations':['Conditional F64 1nm numerical allowance, not directed-rounding proof','Literal encoded visual surface, not compiled collision or whole-hand exterior','Already inspected source row1; new method has no independent unseen native outputs here','Unchanged palm identity transfers prior proximity/occupancy pass only; direction remains unresolved'],'zero_execution':{'physics_steps':0,'native_inference_calls':0,'optimization_calls':0,'evaluation_episodes':0,'training_calls':0}}
 r.write(out/'report.json',v.rename_query_fields(final));print(json.dumps({'status':final['strict_visual_status'],'sha256':r.sha(out/'report.json')}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args();prepare() if args.mode=='prepare' else run()
