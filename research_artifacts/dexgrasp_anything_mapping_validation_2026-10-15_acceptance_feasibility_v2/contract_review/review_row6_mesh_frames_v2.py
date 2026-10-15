"""Reproduce pinned mesh transforms without importing HandModel or a generator."""
from pathlib import Path
import hashlib,importlib.util,json,sys,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

ROOT=Path('/private/tmp/dga_mapping_acceptance_20261015_v2/contract_review')
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
SRC=Path('/Users/nilakarthikesan/grasp-failure-prediction/local_assets/dexgrasp_anything/source-86353d710ca71d0f547f0d8b15cc8ca959c78163')
def sha(p):return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def xyz(v,key,default='0 0 0'):return np.fromstring(v.get(key,default) if v is not None else default,sep=' ')
def independent_mesh(visual,asset,source):
 m=visual.find('geometry/mesh');vertices=[];faces=[]
 for line in Path(asset).read_text().splitlines():
  fields=line.split()
  if not fields:continue
  if fields[0]=='v':vertices.append([float(x) for x in fields[1:4]])
  if fields[0]=='f':
   assert len(fields)==4, 'This independent check requires actual triangular OBJ faces'
   indices=[int(x.split('/')[0]) for x in fields[1:]];faces.append([i-1 if i>0 else len(vertices)+i for i in indices])
 v=np.asarray(vertices)*xyz(m,'scale','1 1 1')
 if source:v=v[:,[0,2,1]]*np.array([1.,-1.,1.])
 o=visual.find('origin');v=np.einsum('ij,kj->ki',Rotation.from_euler('xyz',xyz(o,'rpy')).as_matrix(),v)+xyz(o,'xyz')
 return v,np.asarray(faces,dtype=np.int64)
def check_triangle_set(v,f,rv,rf):
 # Map independently transformed vertices onto the saved processed runtime
 # geometric vertices before comparing triangles; index order is irrelevant.
 d,idx=cKDTree(rv).query(v,workers=1)
 canonical=lambda a:sorted(set(tuple(sorted(map(int,row))) for row in a))
 return float(d.max()),canonical(idx[f])==canonical(rf)
def main():
 reportpath=Path('/private/tmp/dga_component_constraints_20261015_v1/native_validation/results_v1/native_validation_report.json')
 row=json.loads(reportpath.read_text())['records'][6];refpath=Path(row['mapping_reference']['path']);assert sha(refpath)==row['mapping_reference']['sha256'];meta=json.loads(refpath.read_text());npz=refpath.parent/meta['arrays']['path'];assert sha(npz)==meta['arrays']['sha256']
 with np.load(npz,allow_pickle=False) as z:runtime={k:z[k] for k in z.files}
 a=module('mesh_frame_independent_audit',W/'scripts/audit_dexgrasp_anything_static.py');h=module('mesh_frame_helper',W/'cloud/dexgrasp_anything/validate_native_outputs.py')
 source_urdf=SRC/'assets/urdf/sr_grasp_description/urdf/shadowhand.urdf';target_root=W/'third_party/dex-urdf/robots/hands/shadow_hand';target_urdf=target_root/'shadow_hand_right.urdf'
 source=a.model(source_urdf);target=a.model(target_urdf);source_assets=SRC/'assets/urdf';target_assets=target_root
 s=h.visual_meshes(source,source_assets,a,source_runtime=True);t=h.visual_meshes(target,target_assets,a,source_runtime=False)
 j=row['within_batch_sample_index'];root=runtime['input_from_forearm'][j];q=row['joint_targets_rad'];sf=a.independent_fk(source,q);tf=a.independent_fk(target,q);sp=np.asarray(row['object_from_source_forearm'])@sf['palm'];tr=sp@np.linalg.inv(tf['palm'])
 links=['ffknuckle','ffproximal','ffmiddle','ffdistal','rfknuckle','rfproximal','rfmiddle','rfdistal'];records=[]
 deps={str(p):sha(p) for p in [reportpath,refpath,npz,source_urdf,target_urdf,SRC/'utils/handmodel.py',SRC/'utils/rot6d.py',W/'cloud/dexgrasp_anything/validate_native_outputs.py',W/'scripts/audit_dexgrasp_anything_static.py']}
 for n in links:
  sm,tm=s[n,0],t[n,0];sv,sfcs=independent_mesh(source['links'][n].find('visual'),sm['path'],True);tv,tfcs=independent_mesh(target['links'][n].find('visual'),tm['path'],False)
  source_local_error=max(cKDTree(sm['local']).query(sv)[0].max(),cKDTree(sv).query(sm['local'])[0].max());target_local_error=max(cKDTree(tm['local']).query(tv)[0].max(),cKDTree(tv).query(tm['local'])[0].max())
  source_frame=runtime['forearm_from_link__'+n][j];reconstructed=np.einsum('ij,kj->ki',source_frame,np.c_[sv,np.ones(len(sv))])[:,:3];reconstructed=np.einsum('ij,kj->ki',root[:3,:3],reconstructed)+root[:3,3];assert np.isfinite(reconstructed).all()
  residual,topology=check_triangle_set(reconstructed,sfcs,runtime['vertices__'+n][j],runtime['faces__'+n]);assert residual<1e-12 and topology
  # Different local mesh sets are real asset differences even where nominal
  # link transforms coincide. NN vertex metrics are descriptive, not Hausdorff
  # distance to a continuous surface or an acceptance threshold.
  vdist=max(cKDTree(sv).query(tv)[0].max(),cKDTree(tv).query(sv)[0].max())
  world_s=np.asarray(row['object_from_source_forearm'])@sf[n];world_t=tr@tf[n]
  coords=max(source_local_error,target_local_error);assert coords<1e-12
  source_origin=source['links'][n].find('visual/origin');target_origin=target['links'][n].find('visual/origin')
  records.append({'link':n,'source_asset':str(sm['path']),'target_asset':str(tm['path']),'source_asset_sha256':sha(sm['path']),'target_asset_sha256':sha(tm['path']),'source_origin':dict(source_origin.attrib) if source_origin is not None else {},'target_origin':dict(target_origin.attrib) if target_origin is not None else {},'source_special_rotation':'(x,y,z) -> (x,-z,y) before visual origin','helper_source_local_max_error_m':float(source_local_error),'helper_target_local_max_error_m':float(target_local_error),'runtime_mesh_reconstruction_max_error_m':residual,'runtime_geometric_triangle_set_matches':topology,'source_triangle_count':len(sfcs),'target_triangle_count':len(tfcs),'local_vertex_NN_discrepancy_m':float(vdist),'independent_nominal_link_frame_max_entry':float(np.max(np.abs(world_s-world_t))),'source_local_bounds_m':[sv.min(0).tolist(),sv.max(0).tolist()],'target_local_bounds_m':[tv.min(0).tolist(),tv.max(0).tolist()]})
  for asset in [sm['path'],tm['path']]:deps[str(asset)]=sha(asset)
 result={'schema_version':'dga_row6_index_ring_mesh_frame_review_v1','status':'PASS_NO_COORDINATE_CONVERSION_ERROR_FOUND','global_sample_index':6,'records':records,'previous_matrix_warning_run':{'path':str(ROOT/'row6_index_ring_mesh_frame_review_v1.json'),'sha256':sha(ROOT/'row6_index_ring_mesh_frame_review_v1.json'),'note':'Prior accelerated matmul emitted divide/overflow/invalid RuntimeWarnings although all finite geometric checks passed; this recount uses explicit einsum and checks finiteness.'},'source_reference':'Saved declared FP32 runtime, original default TF32 results remain separate.','conclusion':'Independent scipy xyz-Euler origins, literal scales, and pinned source mesh pre-rotation reproduce both source local geometry and target local geometry; saved source object-space triangle sets reconstruct. Index/ring local mesh geometry differs materially despite coincident nominal link frames. Those differences are not a repeated/missing mesh pre-rotation.','scope':'Eight index/ring visuals only; local vertex-NN distances are descriptive geometry evidence, not a certified surface Hausdorff distance or complete-nearset acceptance proof.','dependency_sha256':deps,'review_code_sha256':sha(__file__),'zero_execution':{'physics_steps':0,'inference_calls':0,'fit_calls':0,'evaluation_episodes':0,'training_calls':0}}
 out=ROOT/'row6_index_ring_mesh_frame_review_v2.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(out),'status':result['status'],'records':[{k:r[k] for k in ['link','runtime_mesh_reconstruction_max_error_m','runtime_geometric_triangle_set_matches','local_vertex_NN_discrepancy_m','independent_nominal_link_frame_max_entry']} for r in records]},indent=2))
if __name__=='__main__':main()
