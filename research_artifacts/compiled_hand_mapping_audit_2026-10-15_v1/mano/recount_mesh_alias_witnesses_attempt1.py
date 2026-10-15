"""Independent closest-triangle and explicit-arithmetic replay of saved alias witnesses."""
from pathlib import Path
import hashlib,json,warnings
from xml.etree import ElementTree as ET
import mujoco,numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
O=Path('/private/tmp/compiled_hand_audit_20261015_v1/mano');R=O/'collision_mesh_alias_independent_v1.json'
r=json.loads(R.read_text());U=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf')
for n in ['mj_step','mj_step1','mj_step2']:
 setattr(mujoco,n,lambda *a,**k: (_ for _ in ()).throw(RuntimeError('No dynamics')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(U)==r['source_urdf_sha256'];assert sha(O/'original_import_without_name_fix.urdf')==r['original_import_urdf_sha256']
model=mujoco.MjModel.from_xml_path(str(O/'original_import_without_name_fix.urdf'));root=ET.parse(U).getroot();checks=[];numerics=[]
def geom_arrays(m,g):
 mid=int(m.geom_dataid[g]);a,n=int(m.mesh_vertadr[mid]),int(m.mesh_vertnum[mid]);b,c=int(m.mesh_faceadr[mid]),int(m.mesh_facenum[mid]);raw=np.asarray(m.mesh_vert[a:a+n],dtype=np.float64);rot=Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).as_matrix()
 out=sum(raw[:,j,None]*rot[None,:,j] for j in range(3))+m.geom_pos[g]
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always');blas=raw@rot.T+m.geom_pos[g]
 assert np.isfinite(out).all() and np.isfinite(blas).all()
 delta=float(np.max(np.abs(out-blas)));assert delta<1e-16
 numerics.append({'mesh':m.mesh(mid).name,'vertices':n,'all_inputs_and_results_finite':True,'explicit_scalar_sum_vs_matmul_max_abs_m':delta,'captured_runtime_warnings':[str(w.message) for w in ws]})
 return out,np.asarray(m.mesh_face[b:b+c],int)
def closest(p,v,f):
 t=v[f];a,b,c=t[:,0],t[:,1],t[:,2]
 ds=[]
 for x,y in [(a,b),(a,c),(b,c)]:
  e=y-x;den=np.einsum('ij,ij->i',e,e);num=np.einsum('ij,ij->i',p-x,e);alpha=np.divide(num,den,out=np.zeros_like(num),where=den>0);q=x+np.clip(alpha,0,1)[:,None]*e;ds.append(np.einsum('ij,ij->i',p-q,p-q))
 e1=b-a;e2=c-a;n=np.cross(e1,e2);n2=np.einsum('ij,ij->i',n,n);ap=p-a;sd=np.einsum('ij,ij->i',ap,n);proj=p-n*np.divide(sd,n2,out=np.zeros_like(sd),where=n2>0)[:,None]
 v0=e1;v1=e2;v2=proj-a;d00=np.einsum('ij,ij->i',v0,v0);d01=np.einsum('ij,ij->i',v0,v1);d11=np.einsum('ij,ij->i',v1,v1);d20=np.einsum('ij,ij->i',v2,v0);d21=np.einsum('ij,ij->i',v2,v1);den=d00*d11-d01*d01;u=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=den>0);w=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=den>0);inside=(den>0)&(u>=0)&(w>=0)&(u+w<=1);plane=np.where(inside,np.einsum('ij,ij->i',p-proj,p-proj),np.inf);ds.append(plane)
 d=np.min(np.stack(ds),axis=0);assert np.isfinite(d).all();return float(np.sqrt(np.min(d)))
for row in r['cases']:
 if row['triangle_mesh_identity']:continue
 link=row['link'];declared=Path(row['declared_collision_path']);assert sha(declared)==row['declared_collision_sha256']
 node=next(l for l in root.findall('link') if l.get('name')==link);col=node.find('collision');mesh=col.find('geometry/mesh');origin=col.find('origin')
 spec=ET.Element('mujoco');ET.SubElement(spec,'compiler',{'angle':'radian'});asset=ET.SubElement(spec,'asset');ET.SubElement(asset,'mesh',{'name':'single','file':str(declared),'scale':mesh.get('scale')});body=ET.SubElement(ET.SubElement(spec,'worldbody'),'body');ET.SubElement(body,'geom',{'type':'mesh','mesh':'single','pos':origin.get('xyz'),'euler':origin.get('rpy')});single=mujoco.MjModel.from_xml_string(ET.tostring(spec,encoding='unicode'))
 gid=int(np.flatnonzero(model.geom_bodyid==model.body(link).id)[0]);actual,af=geom_arrays(model,gid);expected,ef=geom_arrays(single,0)
 ah=ConvexHull(actual).simplices;eh=ConvexHull(expected).simplices
 for label,v,f,source in [('declared_vertices_to_imported_triangles',actual,af,expected),('imported_vertices_to_declared_triangles',expected,ef,actual),('convex_hull_declared_vertices_to_imported_hull',actual,ah,expected),('convex_hull_imported_vertices_to_declared_hull',expected,eh,actual)]:
  d=row[label];p=np.asarray(d['witness_query_local_m']);bind=float(np.min(np.linalg.norm(source-p,axis=1)));assert bind<1e-16
  val=closest(p,v,f);err=abs(val-d['maximum_sampled_vertex_to_surface_distance_m']);assert err<1e-12
  checks.append({'link':link,'direction':label,'witness_on_source_vertex_max_residual_m':bind,'independent_all_triangle_distance_m':val,'saved_distance_delta_m':err})
out={'schema_version':'mesh_alias_independent_witness_recount_v1','input_report':{'path':str(R),'sha256':sha(R)},'script_sha256':sha(__file__),'method':'explicit sum vertex rotations; per-triangle plane projection with barycentric inclusion plus every clamped edge segment; closest over all triangles. No trimesh imports or parent audit helper. Convex hull faces independently reconstructed.','checks':checks,'transform_checks':numerics,'status':'PASS_12_POINT_TO_SURFACE_WITNESSES','physics_steps':0,'fitting_calls':0,'floating_point_scope':'Numerical independent check only; sampled surface witnesses, not an outward-rounded continuous certificate.'}
p=O/'mesh_alias_witness_recount_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),'checks':len(checks),'maximum_delta_m':max(x['saved_distance_delta_m'] for x in checks),'max_matmul_vs_explicit_m':max(x['explicit_scalar_sum_vs_matmul_max_abs_m'] for x in numerics)}))
