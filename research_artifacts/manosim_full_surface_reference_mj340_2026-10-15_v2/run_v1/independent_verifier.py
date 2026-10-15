"""Independent numeric readback of saved locked-skin outputs, never engine/field work."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
HERE=ROOT/'research_artifacts/manosim_full_surface_reference_mj340_2026-10-15_v2'
RUN=HERE/'run_v1';PROTOCOL=HERE/'protocol.json'
FREEZE='5d55bb184b6f4078838f02c72793a7baca8ce39a'
PROTOCOL_SHA='00b3ee6d13c3f2c9ed88fe1f07977905ea0f09de3cdea0ccb8327ff9f7246a76'
REPORT_SHA='6c4f180aa2579c16a14f07a29bd046466c0a4cbdfebdea5042ee667424839d06'
FLAGS=('accepted_mapping','native_fidelity_accepted','production_promotion','new_collection_or_training_allowed')
PARTS=('palm','index','middle','ring','little','thumb')

class NoEngine(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('mujoco','torch','manotorch','mano','hug','grasp_failure_prediction','trimesh'):
   raise RuntimeError('Readback prohibits engine/model/geometry imports: '+fullname)
  return None
def require(value,message):
 if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def row(path):return dict(sha256=sha(path),bytes=Path(path).stat().st_size)
def near(a,b,tol,message):
 a=np.asarray(a);b=np.asarray(b);require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),message+':shape/finite')
 error=float(np.max(abs(a-b))) if a.size else 0.;require(error<=tol,message+':'+str(error));return error
def bindings(p):
 require(len(p['bindings'])==3041,'Binding count changed')
 for path,value in p['bindings'].items():require(sha(path)==value,'Bound bytes changed: '+path)
def git(p):
 require(subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==FREEZE,'HEAD changed')
 paths=sorted(set(p['tracked_sources'])|{str(PROTOCOL)})
 for path in paths:
  blob=subprocess.check_output(['git','-C',str(ROOT),'show',FREEZE+':'+str(Path(path).relative_to(ROOT))])
  require(hashlib.sha256(blob).hexdigest()==sha(path),'Git blob changed: '+path)
 return len(paths)
def closed(value):
 for key in FLAGS:require(value.get(key) is False,'Admission flag changed: '+key)
 if 'physical_outcome' in value:require(value['physical_outcome'] is None,'Static result acquired outcome')
def classify(d,r):
 require(d.ndim==2 and d.shape[1]==6 and r.shape==(len(d),) and np.isfinite(d).all() and np.isfinite(r).all() and np.all(d>=0) and np.all(r>=0),'Invalid saved distances/radii')
 e=r[:,None,None]+1e-8;b=np.array([.001,.002,.005])[None,None,:]
 n=d[:,:,None]+e<=b;f=np.maximum(0.,d[:,:,None]-e)>b
 return n,f,~(n|f)
def mm(a,b):return np.einsum('...i,ij->...j',a,b,optimize=False)
def normals(vertices,faces):
 tri=vertices[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1)
 require(np.all(length>0),'Degenerate saved face');return cross/length[:,None],length
def array_inventory(path):
 a=np.load(path,allow_pickle=False)
 return {k:dict(shape=list(a[k].shape),dtype=str(a[k].dtype),sha256=hashlib.sha256(a[k].tobytes()).hexdigest()) for k in a.files}
def skeleton(base,scene,include):
 b=ET.parse(base).getroot();s=ET.parse(scene).getroot()
 bb={x.get('name'):x for x in b.findall('.//body')};sb={x.get('name'):x for x in s.findall('.//body')}
 require(set(bb)==set(sb)=={f'mano_body_{i}' for i in range(16)},'Body inventory changed')
 for name,old in bb.items():
  new=sb[name];require(old.attrib==new.attrib,'Body transform/attributes changed')
  left=[(x.tag,x.attrib) for x in old if x.tag not in ('geom','body')]
  right=[(x.tag,x.attrib) for x in new if x.tag not in ('geom','body')]
  require(left==right,'Joint/inertial parameters changed')
  og=old.findall('./geom');ng=new.findall('./geom');require(len(og)==len(ng)==1,'Original visual geom lost')
  attrs=dict(og[0].attrib);attrs.update(contype='0',conaffinity='0');require(attrs==ng[0].attrib,'Geom changed beyond disabled collision bits')
 require(ET.tostring(b.find('actuator'))==ET.tostring(s.find('actuator')),'Actuators changed')
 bm=b.findall('./asset/mesh');sm=s.findall('./asset/mesh');require(len(bm)==len(sm)==16,'Visual mesh inventory changed')
 for old,new in zip(bm,sm):
  expected=dict(old.attrib);expected['file']=str((Path(base).parent/old.get('file')).resolve());require(expected==new.attrib,'Visual mesh binding changed')
 require([x.attrib for x in s.findall('./include')]==[dict(file=str(Path(include).resolve()))],'New surface include changed')
 return s

def verify():
 guard=NoEngine();sys.meta_path.insert(0,guard);require('mujoco' not in sys.modules,'Engine imported')
 require(sha(PROTOCOL)==PROTOCOL_SHA and sha(RUN/'report.json')==REPORT_SHA,'Actual protocol/report changed')
 p=read(PROTOCOL);report=read(RUN/'report.json');bindings(p);gitcount=git(p)
 require(report['status']=='COMPLETED_LOCKED_NATIVE_SURFACE_PILOT' and report['freeze_commit']==FREEZE and report['protocol_sha256']==PROTOCOL_SHA,'Execution identity differs');closed(report)
 require([(x['index'],x['status'],x['returncode']) for x in report['cases']]==[(20,'COMPLETED',0),(28,'COMPLETED',0)],'Case accounting changed')
 require(p['fixed_articulation'] is True and p['scientific_acceptance_changed'] is False,'Scope changed')
 inventory={str(PROTOCOL):row(PROTOCOL),str(RUN/'report.json'):row(RUN/'report.json')};arrays={};case_receipts=[];all_arrays=set()
 for planned in report['cases']:
  index=planned['index'];scalar=Path(planned['result']);raw=Path(planned['raw_result']);c=read(scalar);m=c['measurement'];sr=c['source_receipt']
  require(scalar.read_bytes()==raw.read_bytes() and sha(scalar)==planned['result_sha256']==planned['raw_result_sha256'],'Tracked/raw scalar mismatch')
  require(sha(planned['log'])==planned['log_sha256'],'Worker log changed')
  require(c['status']=='COMPLETED_LOCKED_SURFACE_STATIC_REFERENCE' and c['failure'] is None and c['partial_artifacts'] is None,'Child incomplete')
  require(c['freeze_commit']==FREEZE and c['protocol_sha256']==PROTOCOL_SHA,'Child freeze differs')
  for value in (c,m,sr,m['compiled_readback'],sr['candidate_receipt']):closed(value)
  require(c['physical_outcome'] is None and m['surface_fidelity_status']=='INCOMPLETE' and m['full_orientation_status']=='NOT_RUN' and m['continuous_surface_equivalence']=='NOT_PROVED','Static results promoted')
  require(all(type(v) is int and v==0 for v in c['forbidden_counts'].values()),'Forbidden operation attempted')
  expected_work=59516096 if index==20 else 141182336
  require(c['guarded_counts']==dict(data=1,forwards=2,full_field_stages=['native_reference','compiled_locked_native_surface'],models=1,triangle_point_tests=expected_work),'Operation quotas differ')
  paths={}
  for artifact in c['artifacts']:
   path=Path(artifact['path']);require(path.is_absolute() and path.resolve().is_relative_to(Path(p['raw_root'])) and not path.is_symlink(),'Artifact escaped')
   require(str(path) not in inventory and row(path)=={k:artifact[k] for k in ('sha256','bytes')},'Artifact changed or duplicate')
   inventory[str(path)]=row(path);paths[path.name]=path
   if path.suffix=='.npz':arrays[str(path)]=array_inventory(path);all_arrays.add(str(path.resolve()))
  require(len(paths)==9,'Expected all9 creation-time artifacts per case')
  for path in (scalar,raw,Path(planned['log'])):inventory[str(path)]=row(path)
  candidate=np.load(paths['candidate_surface.npz'],allow_pickle=False);rb=np.load(paths['compiled_native_surface_readback.npz'],allow_pickle=False)
  v=candidate['native_vertices_world_m'];local=candidate['local_vertices_m'];f=candidate['native_faces'];labels=candidate['native_face_labels'];owner=candidate['saved_owner_world_transform']
  require(v.shape==(778,3) and f.shape==(1552,3) and local.shape==(778,3),'Full skin dimensions changed')
  require(np.array_equal(candidate['native_vertex_ids'],np.arange(778)) and np.array_equal(candidate['native_face_ids'],np.arange(1552)),'Native topology IDs changed')
  decoded=np.load(p['cases'][str(index)]['inputs']['decoded_reference']['path'],allow_pickle=False)
  require(np.array_equal(v,decoded['native_vertices_world_m']) and np.array_equal(f,decoded['faces']),'Candidate changed native full skin')
  shape=np.load(p['cases'][str(index)]['inputs']['shape_asset']['path'],allow_pickle=False)
  require(shape.shape==decoded['shape'].shape==candidate['original_shape_float32'].shape==(1,10) and shape.dtype==np.float32,'Shape layout changed')
  require(np.array_equal(shape,decoded['shape']) and np.array_equal(shape,candidate['original_shape_float32']),'Shape values changed')
  localerr=near(mm(v-owner[:3,3],owner[:3,:3]),local,1e-14,'Inverse saved owner transform')
  near(mm(local,owner[:3,:3].T)+owner[:3,3],v,1e-14,'Local-to-world roundtrip')
  require(np.array_equal(rb['compiled_faces'],f) and np.array_equal(rb['native_face_ids'],np.arange(1552)) and np.array_equal(rb['native_face_labels'],labels),'Compiled face membership changed')
  require(np.array_equal(rb['compiled_vertex_body_ids'],np.ones(778,dtype=int)),'Full surface owner changed')
  near(rb['compiled_local_vertices_m'],local,0.,'Compiled local vertices')
  near(rb['actual_owner_world_transform'],owner,1e-14,'Actual owner transform')
  near(rb['compiled_vertices_world_m'],mm(rb['compiled_local_vertices_m'],rb['actual_owner_world_transform'][:3,:3].T)+rb['actual_owner_world_transform'][:3,3],1e-14,'Independent compiled vertex arithmetic')
  original_vertex=near(rb['compiled_vertices_world_m'],v,1e-12,'Direct original native world vertex readback')
  relative=np.einsum('ij,kj->ik',rb['actual_owner_world_transform'][:3,:3],owner[:3,:3],optimize=False)
  target=np.einsum('ij,nj->ni',relative,v-owner[:3,3],optimize=False)+rb['actual_owner_world_transform'][:3,3]
  near(rb['reference_vertices_world_m'],target,0.,'Exact saved rigid-reexpression reference')
  vertex=near(rb['compiled_vertices_world_m'],target,1e-12,'Full world vertex readback relative reexpressed reference')
  l2=float(max(np.linalg.norm(rb['compiled_vertices_world_m']-target,axis=1)))
  source_norm,areas=normals(v,f);actual_norm,_=normals(rb['compiled_vertices_world_m'],f)
  near(candidate['native_oriented_face_normals_world'],source_norm,1e-13,'Source face normals')
  near(rb['compiled_oriented_face_normals_world'],actual_norm,1e-13,'Compiled face normals')
  target_norm=np.einsum('ij,nj->ni',relative,source_norm,optimize=False)
  near(rb['expected_oriented_face_normals_world'],target_norm,1e-13,'Saved rigid-reexpressed source normals')
  normal=near(actual_norm,target_norm,1e-10,'Corresponding oriented normal readback')
  cr=m['compiled_readback'];require(abs(cr['maximum_vertex_error_m']-vertex)<1e-20 and abs(cr['maximum_vertex_l2_error_m']-l2)<1e-20,'Vertex error scalars differ')
  require(abs(cr['maximum_normal_component_error']-normal)<1e-13 and cr['radius_m']==0. and cr['finger_articulation_allowed'] is False,'Normal/radius/scope scalar changed')
  require(cr['corresponding_triangle_barycentric_discrepancy_bound_m']==cr['maximum_vertex_l2_error_m'],'Corresponding triangle convex-combination bound differs')
  edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);unique,count=np.unique(edges,axis=0,return_counts=True)
  require(len(unique)==2328 and np.all(count==2),'Closed complete topology changed')
  require({name:int(np.sum(labels==i)) for i,name in enumerate(PARTS)}==cr['topology']['face_inventory'],'Anatomical denominator differs')
  prep=read(paths['candidate_receipt.json']);require(prep==sr['candidate_receipt'],'Preparation receipt differs')
  for key,expected in prep['array_hashes'].items():require(hashlib.sha256(candidate[key].dtype.str.encode()+str(candidate[key].shape).encode()+np.ascontiguousarray(candidate[key]).tobytes()).hexdigest()==expected,'Candidate array receipt differs: '+key)
  flex=ET.parse(paths['candidate_flex.xml']).getroot().find('./deformable/flex')
  require(flex is not None and flex.get('body')=='mano_body_0' and float(flex.get('radius'))==0. and flex.get('dim')=='2','Full skin XML owner/radius changed')
  near(np.fromstring(flex.get('vertex'),sep=' ').reshape(778,3),local,0.,'17digit XML vertices');require(np.array_equal(np.fromstring(flex.get('element'),sep=' ',dtype=int).reshape(1552,3),f),'XML faces changed')
  scene=skeleton(p['base_xml'],paths['compiled_scene.xml'],paths['candidate_flex.xml'])
  literal=read(p['cases'][str(index)]['inputs']['literal_pose']['path']);address=sr['named_joint_addresses']
  require(len(address)==19 and rb['literal_qpos'].shape==(67,) and rb['literal_qvel'].shape==(51,) and rb['retained_ctrl'].shape==(51,),'Native51D inventory changed')
  for joint in address:
   values=np.asarray(literal['named_qpos'][joint['name']]).reshape(-1);start=joint['qpos_address'];require(np.array_equal(rb['literal_qpos'][start:start+len(values)],values),'Literal named coordinate changed')
  require(np.array_equal(rb['literal_qvel'],np.zeros(51)),'Static velocities changed')
  kin=np.load(p['cases'][str(index)]['inputs']['kinematic_readback']['path'],allow_pickle=False)
  for key in ('compiled_body_origins_world_m','compiled_world_body_rotations','compiled_joint_anchors_world_m'):near(rb[key],kin[key],0.,'Saved native kinematic frame changed')
  require(np.array_equal(rb['flex_radius'],[0.]) and np.array_equal(rb['flex_margin'],[0.]) and np.array_equal(rb['flex_gap'],[0.]),'Hidden collision inflation')
  for bi in range(16):
   inert=next(body for body in scene.findall('.//body') if body.get('name')==f'mano_body_{bi}').find('./inertial')
   require(rb['body_mass'][bi+1]==float(inert.get('mass')),'Body mass changed')
  # All geometric conversion below is saved point/frame arithmetic, never a query.
  prior_native=read(Path(p['native_run'])/str(index)/'result.json');position=np.array(prior_native['source_receipt']['object_reference_world_position_m']);rotation=np.array(prior_native['source_receipt']['object_reference_world_rotation'])
  ng=np.load(paths['native_reference_geometry.npz'],allow_pickle=False);cg=np.load(paths['compiled_locked_native_surface_geometry.npz'],allow_pickle=False)
  near(ng['triangles_object_m'],mm(v-position,rotation)[f],1e-14,'Native sealed object triangles');near(cg['triangles_object_m'],mm(rb['compiled_vertices_world_m']-position,rotation)[f],1e-14,'Compiled sealed object triangles')
  for geom in (ng,cg):require(np.array_equal(geom['native_face_id'],np.arange(1552)) and np.array_equal(geom['anatomical_part_index'],labels) and np.array_equal(geom['source_vertex_ids'],f),'Sealed query geometry membership changed')
  reference=np.load(paths['native_reference_field.npz'],allow_pickle=False);compiled=np.load(paths['compiled_locked_native_surface_field.npz'],allow_pickle=False)
  old=Path(p['ablation_raw'])/str(index)/'native_reference_field.npz';require(sha(paths['native_reference_field.npz'])==sha(old),'Reference field bytes changed')
  for key in ('query_object_m','cell_radius_m','native_distance_by_part_m'):require(np.array_equal(reference[key],compiled[key]),'Fixed query/radius/native distance changed')
  n=len(reference['query_object_m']);require(n==(19174 if index==20 else 45484) and 2*n*1552==expected_work,'Complete query/triangle denominator changed')
  summaries=[]
  for stage,field in zip(m['stages'],(reference,compiled)):
   closed(stage);nn,nf,na=classify(field['native_distance_by_part_m'],field['cell_radius_m']);mn,mf,ma=classify(field['mapped_distance_by_part_m'],field['cell_radius_m'])
   masks=dict(native_near=nn,native_far=nf,native_ambiguous=na,required=nn|nf,preserved=(nn&mn)|(nf&mf),definite_lost=nn&mf,definite_added=nf&mn,unresolved_required=(nn|nf)&~((nn&mn)|(nf&mf))&~((nn&mf)|(nf&mn)))
   counts={key:int(value.sum()) for key,value in masks.items()}
   for key,value in masks.items():require(np.array_equal(field[key],value) and stage['summary'][key]==counts[key],'Saved whole-cell masks/summary differ: '+key)
   require(counts['required']==counts['preserved']==(341721 if index==20 else 804343) and counts['definite_lost']==counts['definite_added']==counts['unresolved_required']==0,'Relationship denominator changed')
   require(stage['summary']['full_strict_mapping_status']=='INCOMPLETE' and stage['summary']['full_orientation_status']=='NOT_RUN','Field summary promoted')
   summaries.append(counts)
  delta=float(max(abs(compiled['mapped_distance_by_part_m']-reference['mapped_distance_by_part_m']).reshape(-1)))
  require(delta==m['maximum_distance_delta_from_reference_m'],'Distance delta scalar differs')
  require(m['reference_control']['passed'] is True and m['reference_control']['exact_masks'] is True and m['reference_control']['max_distance_error_m']==0.,'Reference control changed')
  case_receipts.append(dict(index=index,status='VERIFIED_LOCKED_NUMERICAL_SURFACE_FIELDS_FULL_GATE_INCOMPLETE',result_sha256=sha(scalar),query_count=n,field_mask_counts=summaries,
   maximum_vertex_error_vs_rigid_reexpressed_reference_m=vertex,maximum_vertex_l2_error_vs_rigid_reexpressed_reference_m=l2,maximum_vertex_error_vs_original_saved_native_m=original_vertex,maximum_corresponding_normal_component_error=normal,maximum_distance_delta_m=delta,
   guarded_counts=c['guarded_counts'],forbidden_counts=c['forbidden_counts'],all_original778vertices1552faces=True,all_prior_witness_rows_in_unchanged_complete_grid=True,
   full_orientation_status='NOT_RUN',surface_fidelity_status='INCOMPLETE',physical_outcome=None))
 require({str(x.resolve()) for x in Path(p['raw_root']).rglob('*.npz')}==all_arrays,'Unindexed raw array')
 bindings(p);require(git(p)==gitcount,'Git source partition changed')
 for path,expected in inventory.items():require(row(path)==expected,'Output changed during readback')
 source=Path(__file__).read_text();sys.meta_path.remove(guard)
 return dict(schema_version='manosim_full_surface_mj340_v2_independent_saved_readback_v1',status='VERIFIED_COMPLETED_LOCKED_NATIVE_SURFACE_FIELDS_FULL_GATE_INCOMPLETE',
  freeze_commit=FREEZE,active_protocol_sha256=PROTOCOL_SHA,expected_bindings=3041,selected_tracked_sources=len(p['tracked_sources']),selected_Git_blob_count=gitcount,
  all_bindings_verified_before=True,all_bindings_verified_after=True,actual_execution_status=report['status'],cases=case_receipts,
  artifact_inventory=inventory,array_inventory=arrays,creation_time_artifact_count=18,verification_model_FK_distance_field_physics_calls=0,
  scope='Two saved native poses; single palm owner; fingers fixed; numerical corresponding-surface and fixed-grid relationship readback only. No continuous equivalence, object-query orientation, articulated dynamics, task outcome or collection admission.',
  physical_outcome=None,accepted_mapping=False,native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False,
  verifier_source_sha256=hashlib.sha256(source.encode()).hexdigest(),verifier_source=source,errors=[])

if __name__=='__main__':
 result=verify()
 with Path(sys.argv[1]).open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status=result['status'],cases=len(result['cases']),Git_blobs=result['selected_Git_blob_count'])))
