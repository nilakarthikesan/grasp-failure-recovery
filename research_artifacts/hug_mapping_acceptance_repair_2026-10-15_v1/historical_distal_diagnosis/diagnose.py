"""Read static historical mesh arrays; no MjData, FK, simulation or native grasps."""
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib,json,sys
import numpy as np
from scipy.spatial import ConvexHull,cKDTree
from scipy.spatial.transform import Rotation
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def transform(T,p):return np.sum(p[:,None,:]*T[None,:3,:3],axis=2)+T[None,:3,3]
def T_from(xyz,rpy):
 T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();T[:3,3]=xyz;return T
def obj(path,scale,T):
 v=np.array([[float(t) for t in s.split()[1:4]] for s in Path(path).read_text().splitlines() if s.startswith('v ')]);v=v*np.array(scale);assert np.isfinite(v).all();return transform(T,v)
def outside_plane_lower(A,B):
 ha=ConvexHull(A);hb=ConvexHull(B);Q=A[ha.vertices];E=hb.equations;norm=np.linalg.norm(E[:,:3],axis=1);S=(Q@E[:,:3].T+E[:,3])/norm;flat=int(np.argmax(S));vi,fi=np.unravel_index(flat,S.shape);value=float(S[vi,fi]);return {'max_signed_outside_support_plane_m':value,'positive_distance_lower_bound_m':max(0.,value),'query_source_vertex_index':int(ha.vertices[vi]),'query_point':Q[vi].tolist(),'target_outward_plane':E[fi].tolist(),'target_plane_normal_norm':float(norm[fi]),'source_hull_vertex_count':len(ha.vertices),'target_hull_vertex_count':len(hb.vertices),'numerical_not_directed_rounding_certificate':True}
def comparison(A,B):
 return {'first_to_second':outside_plane_lower(A,B),'second_to_first':outside_plane_lower(B,A),'symmetric_vertex_set_distance_m':float(max(cKDTree(A).query(B)[0].max(),cKDTree(B).query(A)[0].max()))}
def controls():
 import unittest
 class C(unittest.TestCase):
  def setUp(self):self.c=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])
  def test_identity(self):self.assertLessEqual(outside_plane_lower(self.c,self.c)['positive_distance_lower_bound_m'],1e-14)
  def test_extra_interior_vertices_same_hull(self):
   b=np.vstack([self.c,[0,0,0],[.3,.2,.1]]);r=comparison(self.c,b);self.assertGreater(r['symmetric_vertex_set_distance_m'],1.);self.assertLessEqual(max(r[x]['positive_distance_lower_bound_m'] for x in ['first_to_second','second_to_first']),1e-14)
  def test_translated_cube(self):r=comparison(self.c+.2*np.array([1,0,0]),self.c);self.assertAlmostEqual(r['first_to_second']['positive_distance_lower_bound_m'],.2);self.assertAlmostEqual(r['second_to_first']['positive_distance_lower_bound_m'],.2)
  def test_nested_cubes(self):r=comparison(self.c*.5,self.c);self.assertEqual(r['first_to_second']['positive_distance_lower_bound_m'],0.);self.assertAlmostEqual(r['second_to_first']['positive_distance_lower_bound_m'],.5)
  def test_rigid_rotation_invariance(self):
   T=T_from([.12,-.4,.33],[.2,-.3,.9]);a=self.c+np.array([.2,0,0]);r=comparison(transform(T,a),transform(T,self.c));self.assertAlmostEqual(r['first_to_second']['positive_distance_lower_bound_m'],.2)
  def test_witness_replay(self):r=outside_plane_lower(self.c+.2*np.array([1,0,0]),self.c);plane=np.array(r['target_outward_plane']);self.assertAlmostEqual((np.dot(r['query_point'],plane[:3])+plane[3])/np.linalg.norm(plane[:3]),r['max_signed_outside_support_plane_m'])
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(C);result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful();return result.testsRun

def run():
 p=H/'protocol.json';d=json.loads(p.read_text());ph=sha(p)
 for path,dig in d['bindings'].items():assert sha(path)==dig,path
 assert not (H/'result.json').exists()
 import mujoco
 assert mujoco.__version__==d['mujoco_version']
 for name in ['mj_forward','mj_step','mj_step1','mj_step2']:
  setattr(mujoco,name,lambda *a,**k: (_ for _ in ()).throw(RuntimeError('No FK/physics authorized')))
 urdf=ET.parse(d['source_urdf']).getroot();parts={}
 for name in d['links']:
  node=next(x for x in urdf.findall('link') if x.get('name')==name);parts[name]={}
  for kind in ['collision','visual']:
   item=node.find(kind);m=item.find('geometry/mesh');o=item.find('origin');T=T_from(np.fromstring(o.get('xyz','0 0 0'),sep=' ') if o is not None else np.zeros(3),np.fromstring(o.get('rpy','0 0 0'),sep=' ') if o is not None else np.zeros(3));path=Path(d['source_urdf']).parent/m.get('filename');parts[name][kind]=obj(path,np.fromstring(m.get('scale','1 1 1'),sep=' '),T)
 rows=[];saved_arrays={}
 for case in d['unique_historical_models']:
  model=mujoco.MjModel.from_binary_path(case['path'])
  for name in d['links']:
   bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name);assert bid>=0
   gids=[g for g in range(model.ngeom) if model.geom_bodyid[g]==bid and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH];assert len(gids)==1;g=gids[0];mid=int(model.geom_dataid[g]);va=int(model.mesh_vertadr[mid]);vn=int(model.mesh_vertnum[mid]);raw=np.asarray(model.mesh_vert[va:va+vn],float);T=np.eye(4);q=model.geom_quat[g];T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();T[:3,3]=model.geom_pos[g];local=transform(T,raw);assert np.isfinite(local).all()
   key=f'{case["sha256"][:12]}_{name}';saved_arrays[key]=local
   rows.append({'scene_sha256':case['sha256'],'source_indices':case['source_indices'],'link':name,'compiled_geom_id':g,'compiled_mesh_id':mid,'compiled_mesh_name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_MESH,mid),'compiled_vertex_count':vn,'saved_array_key':key,'historical_to_declared_collision':comparison(local,parts[name]['collision']),'historical_to_source_visual':comparison(local,parts[name]['visual'])})
 for path,dig in d['bindings'].items():assert sha(path)==dig,path
 np.savez_compressed(H/'local_arrays.npz',**saved_arrays)
 report={'schema':'hug_historical_distal_hull_support_diagnosis_v1','protocol_sha256':ph,'rows':rows,'array_file':str(H/'local_arrays.npz'),'array_sha256':sha(H/'local_arrays.npz'),'mesh_encoding_versus_hull_boundary':'Vertex-set distance is an upper bound on convex-hull Hausdorff, never a lower bound. Positive source-vertex target-plane violations are numerical lower witnesses for distance to that target convex solid.','mj_forward_calls':0,'physics_steps':0,'new_grasps':0,'new_eval_data':False,'source_closure_unchanged':True}
 (H/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'rows':len(rows),'result_sha256':sha(H/'result.json'),'max_collision_outside_plane_m':max(max(r['historical_to_declared_collision'][k]['positive_distance_lower_bound_m'] for k in ['first_to_second','second_to_first']) for r in rows),'max_visual_vertex_set_error_m':max(r['historical_to_source_visual']['symmetric_vertex_set_distance_m'] for r in rows)}))
if __name__=='__main__':
 if sys.argv[1]=='controls':controls()
 elif sys.argv[1]=='run':run()
