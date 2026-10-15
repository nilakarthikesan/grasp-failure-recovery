"""Finite numerical convex-solid Hausdorff calculation on frozen source arrays."""
from pathlib import Path
import json,hashlib,unittest,sys
import numpy as np
from scipy.spatial import ConvexHull
H=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def point_triangles(p,triangles):
 a,b,c=triangles[:,0],triangles[:,1],triangles[:,2];v0=b-a;v1=c-a;n=np.cross(v0,v1);nn=np.sum(n*n,axis=1);assert np.all(nn>0)
 h=np.sum((p-a)*n,axis=1)/nn;projection=p-h[:,None]*n
 w=projection-a;d00=np.sum(v0*v0,axis=1);d01=np.sum(v0*v1,axis=1);d11=np.sum(v1*v1,axis=1);d20=np.sum(w*v0,axis=1);d21=np.sum(w*v1,axis=1);den=d00*d11-d01*d01;assert np.all(den>0)
 u=(d11*d20-d01*d21)/den;v=(d00*d21-d01*d20)/den;inside=(u>=0)&(v>=0)&(u+v<=1)
 candidates=[projection];distances=[np.where(inside,np.sum((p-projection)**2,axis=1),np.inf)]
 for A,B in [(a,b),(b,c),(c,a)]:
  edge=B-A;t=np.clip(np.sum((p-A)*edge,axis=1)/np.sum(edge*edge,axis=1),0.,1.);point=A+t[:,None]*edge;candidates.append(point);distances.append(np.sum((p-point)**2,axis=1))
 ds=np.stack(distances);kind,face=np.unravel_index(int(np.argmin(ds)),ds.shape);q=candidates[kind][face];distance=float(np.sqrt(ds[kind,face]));assert np.isfinite(q).all() and np.isfinite(distance);return distance,q,int(face),int(kind)
def directed(A,B):
 ha,hb=ConvexHull(A),ConvexHull(B);tri=B[hb.simplices];rows=[]
 for vi in ha.vertices:
  p=A[vi];s=np.sum(hb.equations[:,:3]*p[None,:],axis=1)+hb.equations[:,3]
  if np.max(s)<=0:d=0.;q=p.copy();face=-1;kind=-1
  else:d,q,face,kind=point_triangles(p,tri)
  rows.append({'vertex_index':int(vi),'distance_m':d,'point':p.tolist(),'closest_point':q.tolist(),'target_triangle_index':face,'closest_feature_code':kind,'inside_target':face==-1})
 return {'maximum_distance_m':max(r['distance_m'] for r in rows),'source_extreme_vertex_count':len(ha.vertices),'target_triangle_count':len(tri),'rows':rows}
def controls():
 class C(unittest.TestCase):
  def setUp(self):self.c=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])
  def test_identity(self):self.assertAlmostEqual(directed(self.c,self.c)['maximum_distance_m'],0.)
  def test_added_interior_points(self):self.assertAlmostEqual(directed(np.vstack([self.c,[0,0,0]]),self.c)['maximum_distance_m'],0.)
  def test_translation_face(self):self.assertAlmostEqual(directed(self.c+[.2,0,0],self.c)['maximum_distance_m'],.2)
  def test_translation_edge(self):self.assertAlmostEqual(directed(self.c+[.2,.3,0],self.c)['maximum_distance_m'],np.hypot(.2,.3))
  def test_translation_corner(self):self.assertAlmostEqual(directed(self.c+[.2,.3,.4],self.c)['maximum_distance_m'],np.linalg.norm([.2,.3,.4]))
  def test_nested(self):self.assertAlmostEqual(directed(self.c*.5,self.c)['maximum_distance_m'],0.);self.assertAlmostEqual(directed(self.c,self.c*.5)['maximum_distance_m'],np.sqrt(3)*.5)
  def test_triangle_face_edge_corner(self):
   t=np.array([[[0.,0,0],[1,0,0],[0,1,0]]]);self.assertAlmostEqual(point_triangles(np.array([.2,.2,.3]),t)[0],.3);self.assertAlmostEqual(point_triangles(np.array([.5,-.2,0]),t)[0],.2);self.assertAlmostEqual(point_triangles(np.array([-1.,-1,0]),t)[0],np.sqrt(2))
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(C));assert result.wasSuccessful()
def run():
 p=H/'hull_distance_protocol.json';proto=json.loads(p.read_text())
 for path,h in proto['bindings'].items():assert sha(path)==h,path
 from xml.etree import ElementTree as ET
 urdf=Path(proto['source_urdf']);xml=ET.parse(urdf).getroot()
 def source(name,kind):
  item=next(e for e in xml.findall('link') if e.get('name')==name).find(kind);o=item.find('origin');xyz=np.zeros(3) if o is None else np.fromstring(o.get('xyz','0 0 0'),sep=' ');rpy=np.zeros(3) if o is None else np.fromstring(o.get('rpy','0 0 0'),sep=' ');assert np.array_equal(rpy,np.zeros(3))
  mesh=item.find('geometry/mesh');scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ');return np.array([[float(t) for t in l.split()[1:4]] for l in (urdf.parent/mesh.get('filename')).read_text().splitlines() if l.startswith('v ')])*scale+xyz
 d=json.loads((H/'result.json').read_text());rows=[r for r in d['rows'] if r['link'] in ['mfdistal','rfdistal','lfdistal']]
 with np.load(H/'local_arrays.npz',allow_pickle=False) as z:
  A=z[rows[0]['saved_array_key']].copy();assert all(np.array_equal(A,z[r['saved_array_key']]) for r in rows)
 B=np.array(source('mfdistal','collision'));assert all(np.array_equal(B,np.array(source(n,'collision'))) for n in ['rfdistal','lfdistal'])
 f=directed(A,B);r=directed(B,A);report={'schema':'hug_historical_distal_convex_solid_hausdorff_v1','protocol_sha256':sha(p),'distinct_old_scene_distal_sets_verified_equal':len(rows),'forward':f,'reverse':r,'symmetric_convex_solid_hausdorff_m':max(f['maximum_distance_m'],r['maximum_distance_m']),'scope':'Numerical global convex-solid Hausdorff: distance to a convex set is convex, so maxima over a convex polytope occur at extreme vertices; every source extreme vertex was compared with every target hull facet triangle if outside, or assigned zero if inside. Floating calculations are not interval-certified. Vertex-set distance and nonconvex visual triangle-surface distance are different quantities.','new_mujoco_loads':0,'new_FK_calls':0,'physics_steps':0,'new_eval_data':False}
 out=H/'hull_distance_result.json';assert not out.exists();out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'sha256':sha(out),'forward':f['maximum_distance_m'],'reverse':r['maximum_distance_m'],'symmetric':report['symmetric_convex_solid_hausdorff_m']}))
if __name__=='__main__':controls() if sys.argv[1]=='controls' else run()
