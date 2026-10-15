"""Verify failed saved full-surface run without importing an engine or distance code."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
HERE=ROOT/'research_artifacts/manosim_full_surface_reference_mj340_2026-10-15_v1'
RUN=HERE/'run_v1';PROTOCOL=HERE/'protocol.json'
FREEZE='af2ac9cbee332401dc0dbc435130bbcaab2d57ed'
PROTOCOL_SHA='7442bd947c194249e16d992af9930c5581e085e2c8f8e3102f41bc5bc92e5d5c'


class NoEngine(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('mujoco','torch','manotorch','mano','hug','grasp_failure_prediction','trimesh'):
   raise RuntimeError('Saved-only verifier forbids model/geometry imports: '+fullname)
  return None
def require(value,message):
 if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def row(path):return dict(sha256=sha(path),bytes=Path(path).stat().st_size)
def verify_bindings(p):
 require(len(p['bindings'])==3027,'Source binding denominator changed')
 for path,expected in p['bindings'].items():require(sha(path)==expected,'Bound source/input changed: '+path)
def git(p):
 require(subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==FREEZE,'HEAD changed')
 paths=sorted(set(p['tracked_sources'])|{str(PROTOCOL)})
 for path in paths:
  blob=subprocess.check_output(['git','-C',str(ROOT),'show',FREEZE+':'+str(Path(path).relative_to(ROOT))])
  require(hashlib.sha256(blob).hexdigest()==sha(path),'Frozen Git blob changed: '+path)
 return len(paths)
def mask(distance,radius):
 error=radius[:,None,None]+1e-8;bands=np.array([.001,.002,.005])[None,None,:]
 near=distance[:,:,None]+error<=bands
 far=np.maximum(0.,distance[:,:,None]-error)>bands
 return near,far,~(near|far)
def verify():
 guard=NoEngine();sys.meta_path.insert(0,guard)
 require('mujoco' not in sys.modules,'Engine imported before saved-only verifier')
 require(sha(PROTOCOL)==PROTOCOL_SHA,'Active protocol changed')
 p=read(PROTOCOL);r=read(RUN/'report.json');verify_bindings(p);gitcount=git(p)
 require(r['status']=='HARD_STOP' and r['freeze_commit']==FREEZE and r['protocol_sha256']==PROTOCOL_SHA,'Execution identity changed')
 require([(x['index'],x['status']) for x in r['cases']]==[(20,'ERROR'),(28,'NOT_RUN')],'Attempted/unrun denominators changed')
 require(r['physical_outcome'] is None,'Failed study acquired physical label')
 for name in ('accepted_mapping','native_fidelity_accepted','production_promotion','new_collection_or_training_allowed'):
  require(r[name] is False,'Admission remained required false')
 first=r['cases'][0];scalar=Path(first['result']);rawscalar=Path(first['raw_result']);result=read(scalar)
 require(scalar.read_bytes()==rawscalar.read_bytes() and sha(scalar)==first['result_sha256']==first['raw_result_sha256'],'Tracked/raw failure copies differ')
 require(first['returncode']==1 and result['status']=='HARD_STOP' and result['freeze_commit']==FREEZE and result['protocol_sha256']==PROTOCOL_SHA,'Child failure identity differs')
 require(result['failure']==dict(type='ValueError',message='Exact literal native float32 shape identity required'),'Original shape refusal changed')
 require(result['measurement'] is None and result['physical_outcome'] is None and result['artifacts']==[],'Failure acquired completed measurement/label')
 for name in ('accepted_mapping','native_fidelity_accepted','production_promotion','new_collection_or_training_allowed'):
  require(result[name] is False,'Child admission remained required false')
 require(all(type(v) is int and v==0 for v in result['forbidden_counts'].values()),'Forbidden engine/native/training operation attempted')
 expected_counts=dict(data=0,forwards=0,full_field_stages=['native_reference'],models=0,triangle_point_tests=29758048)
 require(result['guarded_counts']==expected_counts,'Pre-refusal operation counts changed')
 partial=result['partial_artifacts'];require(partial['complete_array_inventory'] is False,'Partial run promoted to complete')
 require(partial['recorded_rows']==partial['verified_rows'] and len(partial['recorded_rows'])==2,'Partial creation-time receipts differ')
 require(partial['verification_errors']==[] and partial['unindexed_array_paths']==[],'Partial output integrity failed')
 indexed={};array_inventory={};files=[]
 for artifact in partial['recorded_rows']:
  path=Path(artifact['path']);require(path.is_absolute() and path.resolve().is_relative_to(Path(p['raw_root'])) and not path.is_symlink(),'Partial artifact escaped/aliased')
  require(row(path)=={key:artifact[key] for key in ('sha256','bytes')},'Partial artifact changed')
  indexed[str(path)]=row(path);files.append(path)
 require({str(x.resolve()) for x in Path(p['raw_root']).rglob('*.npz')}==set(indexed),'Undisclosed partial array')
 require(not (RUN/'28').exists() and not (Path(p['raw_root'])/'28').exists(),'Case28 unexpectedly attempted')
 geom=next(path for path in files if path.name=='native_reference_geometry.npz')
 field=next(path for path in files if path.name=='native_reference_field.npz')
 previous=Path(p['ablation_raw'])/'20'
 require(sha(field)==sha(previous/field.name)=='af7cc77b18459b0fb24893574c37ec6406023f33055b94c3ab3513074a621853','Reference field not byte-identical to prior frozen field')
 for path in (geom,field):
  a=np.load(path,allow_pickle=False);old=np.load(previous/path.name,allow_pickle=False)
  require(set(a.files)==set(old.files),'Reference array inventory changed')
  for key in a.files:require(np.array_equal(a[key],old[key]),'Reference array differs: '+key)
  array_inventory[str(path)]={key:dict(shape=list(a[key].shape),dtype=str(a[key].dtype),sha256=hashlib.sha256(a[key].tobytes()).hexdigest()) for key in a.files}
 g=np.load(geom,allow_pickle=False);a=np.load(field,allow_pickle=False)
 require(g['triangles_object_m'].shape==(1552,3,3) and g['source_vertex_ids'].shape==(1552,3),'Reference faces changed')
 require(np.array_equal(g['native_face_id'],np.arange(1552)),'Full native face order changed')
 require(np.isfinite(g['triangles_object_m']).all(),'Nonfinite reference triangles')
 q=a['query_object_m'];radius=a['cell_radius_m'];native=a['native_distance_by_part_m'];mapped=a['mapped_distance_by_part_m']
 require(q.shape==(19174,3) and radius.shape==(19174,) and native.shape==(19174,6),'Field dimensions changed')
 require(np.isfinite(q).all() and np.isfinite(radius).all() and np.all(radius>=0) and np.isfinite(native).all() and np.all(native>=0),'Malformed saved field')
 require(np.array_equal(native,mapped),'Reference distance identity failed')
 nn,nf,na=mask(native,radius);mn,mf,ma=mask(mapped,radius)
 masks=dict(native_near=nn,native_far=nf,native_ambiguous=na,required=nn|nf,
  preserved=(nn&mn)|(nf&mf),definite_lost=nn&mf,definite_added=nf&mn,
  unresolved_required=(nn|nf)&~((nn&mn)|(nf&mf))&~((nn&mf)|(nf&mn)))
 for name,value in masks.items():require(np.array_equal(a[name],value),'Whole-cell predicate mismatch: '+name)
 counts={name:int(value.sum()) for name,value in masks.items()}
 require(counts['required']==341721 and counts['preserved']==341721 and counts['definite_lost']==counts['definite_added']==counts['unresolved_required']==0,'Reference denominator changed')
 require(len(q)*len(g['triangles_object_m'])==29758048,'Triangle-test operation count disagrees with full saved geometry/grid')
 control=result['source_receipt']['reference_control']
 require(control['passed'] is True and control['exact_masks'] is True and control['max_distance_error_m']==0. and control['acceptance_allowance_changed'] is False,'Reference consistency scalar changed')
 shape_path=Path(p['cases']['20']['inputs']['shape_asset']['path']);shape=np.load(shape_path,allow_pickle=False)
 decoded=np.load(p['cases']['20']['inputs']['decoded_reference']['path'],allow_pickle=False)['shape']
 require(sha(shape_path)==p['cases']['20']['expected_shape_sha256'],'Literal shape asset hash changed')
 require(shape.dtype==np.float32 and shape.shape==(1,10) and decoded.dtype==np.float32 and np.array_equal(shape.reshape(-1),decoded.reshape(-1)),'Shape-layout diagnosis not supported by saved values')
 inventory={str(path):row(path) for path in [PROTOCOL,RUN/'report.json',scalar,rawscalar,Path(first['log']),*files]}
 verify_bindings(p);require(git(p)==gitcount,'Selected source inventory changed')
 for path,expected in inventory.items():require(row(path)==expected,'Saved output changed during verification')
 source=Path(__file__).read_text();sys.meta_path.remove(guard)
 return dict(schema_version='manosim_full_surface_mj340_failed_saved_readback_v1',
  status='VERIFIED_HARD_STOP_SHAPE_LAYOUT_BEFORE_ENGINE',freeze_commit=FREEZE,active_protocol_sha256=PROTOCOL_SHA,
  expected_bindings=3027,selected_tracked_sources=len(p['tracked_sources']),selected_Git_blob_count=gitcount,
  all_bindings_verified_before=True,all_bindings_verified_after=True,actual_execution_status='HARD_STOP',
  cases=[dict(index=20,status='VERIFIED_HARD_STOP_AFTER_REFERENCE_BEFORE_MODEL',result_sha256=sha(scalar)),dict(index=28,status='VERIFIED_NOT_RUN')],
  guarded_counts=expected_counts,forbidden_counts=result['forbidden_counts'],partial_artifact_count=2,
  partial_array_inventory_claimed_complete=False,reference_field_byte_identity=True,reference_mask_counts=counts,
  shape_diagnosis=dict(asset_shape=list(shape.shape),asset_dtype=str(shape.dtype),decoded_shape=list(decoded.shape),decoded_dtype=str(decoded.dtype),
   exact_flattened_values=True,source_contract_expected_shape=[10],asset_sha256=sha(shape_path),inference='Strict source layout contract refused an otherwise value-identical literal (1,10) asset; no engine operation reached.'),
  artifact_inventory=inventory,array_inventory=array_inventory,verification_model_FK_field_physics_calls=0,
  physical_outcome=None,accepted_mapping=False,native_fidelity_accepted=False,new_collection_or_training_allowed=False,
  verifier_source_sha256=hashlib.sha256(source.encode()).hexdigest(),verifier_source=source,errors=[])

if __name__=='__main__':
 result=verify()
 with Path(sys.argv[1]).open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status=result['status'],reference_counts=result['reference_mask_counts'],Git_blobs=result['selected_Git_blob_count'])))
