"""Independent analytic manufactured checks of static comparison distance metrics."""
from pathlib import Path
import sys,importlib.util,unittest
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
import pytest
pytest.importorskip('trimesh')
PATH=Path(__file__).resolve().parents[1]/'scripts/audit_hug_native_mano.py'
spec=importlib.util.spec_from_file_location('audit_module',PATH);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
class Metrics(unittest.TestCase):
 def assertclose(self,x,y):np.testing.assert_allclose(x,y,atol=1e-12,rtol=0)
 def test_triangle_interior_edge_vertex(self):
  triangle=np.array([[[0,0,0],[1,0,0],[0,1,0]]],float)
  queries=np.array([[.2,.2,0],[.2,.2,3],[2,0,0],[-1,-1,0],[1,1,0]])
  self.assertclose(audit.triangle_distance(queries,triangle),[0,3,1,2**.5,.5**.5])
 def test_triangle_multiple_chunk_order(self):
  triangles=np.array([[[0,0,0],[1,0,0],[0,1,0]],[[0,0,2],[1,0,2],[0,1,2]]],float)
  q=np.array([[.2,.2,x] for x in np.linspace(-2,4,107)])
  self.assertclose(audit.triangle_distance(q,triangles),np.minimum(abs(q[:,2]),abs(q[:,2]-2)))
 def test_box_surface_inside_outside(self):
  q=np.array([[0,0,0],[2,0,0],[1,0,0],[2,3,4],[1,2,3]],float)
  self.assertclose(audit.surface_distance(q,[{'kind':'triangles','triangles':audit.box_triangles([1,2,3])}]),[1,1,0,3**.5,0])
 def test_sphere(self):
  q=np.array([[0,0,0],[0,0,3],[0,2,0],[1,0,0]],float)
  g={'kind':'sphere','rotation':np.eye(3),'position':np.zeros(3),'size':[2,0,0]}
  self.assertclose(audit.surface_distance(q,[g]),[2,1,0,1])
 def test_capsule(self):
  q=np.array([[0,0,0],[0,0,4],[0,0,3],[2,0,1],[0,0,1],[3,0,2]],float)
  g={'kind':'capsule','rotation':np.eye(3),'position':np.zeros(3),'size':[1,2,0]}
  self.assertclose(audit.surface_distance(q,[g]),[1,1,0,1,1,2])
 def test_cylinder(self):
  q=np.array([[0,0,0],[0,0,4],[0,0,2],[2,0,1],[0,0,1.5],[3,0,4]],float)
  g={'kind':'cylinder','rotation':np.eye(3),'position':np.zeros(3),'size':[1,2,0]}
  self.assertclose(audit.surface_distance(q,[g]),[1,2,0,1,.5,8**.5])
 def test_rigid_covariance_all_primitives(self):
  q=np.array([[0,0,0],[0,0,4],[0,0,2],[2,0,1],[0,0,1.5],[3,0,4]],float)
  R=Rotation.from_rotvec([.2,.7,-1.1]).as_matrix();t=np.array([3,-7,11])
  for kind in ('sphere','capsule','cylinder'):
   g={'kind':kind,'rotation':np.eye(3),'position':np.zeros(3),'size':[1,2,0]}
   h={**g,'rotation':R,'position':t}
   self.assertclose(audit.surface_distance(q,[g]),audit.surface_distance(q@R.T+t,[h]))
 def test_component_min_not_union_signed_distance(self):
  # Intended unsigned component-surface metric sees internal boundaries, as documented.
  g={'kind':'sphere','rotation':np.eye(3),'position':np.zeros(3),'size':[2,0,0]}
  h={**g,'position':np.array([1.,0,0])}
  self.assertclose(audit.surface_distance(np.array([[-1.,0,0],[3.,0,0]]),[g,h]),[0,0])
 def test_source_geometry_compiled_fromto_and_transform(self):
  xml='<mujoco><worldbody><body pos="1 2 3" euler="10 20 30"><geom type="capsule" fromto="0 0 0 0 0 2" size=".2"/></body></worldbody></mujoco>'
  m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
  g=audit.geometry(m,d,0);p=g['position'];axis=g['rotation'][:,2]
  self.assertclose(audit.surface_distance(np.array([p,p+axis*1.2,p+axis*2]),[g]),[.2,0,.8])
 def test_object_points_distinct_on_box_surface_and_deterministic(self):
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="object"><geom type="box" size="1 2 3"/></body></worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
  points,meta=audit.object_probes(m,d);again,_=audit.object_probes(m,d)
  self.assertEqual(meta['unique_points'],512);self.assertTrue(np.array_equal(points,again))
  self.assertTrue(np.all(np.abs(points)<=np.array([1,2,3])+1e-12))
  self.assertclose(np.min(np.abs(np.abs(points)-np.array([1,2,3])),axis=1),np.zeros(512))
 def test_object_visual_preferred_and_other_bodies_excluded(self):
  xml='<mujoco><worldbody><body name="object"><geom type="box" size="1 1 1" contype="0" conaffinity="0"/><geom type="box" size="2 2 2"/></body><body name="unrelated"><geom type="box" size="100 100 100"/></body></worldbody></mujoco>'
  m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d);points,meta=audit.object_probes(m,d)
  self.assertEqual(meta['reference'],'captured_object_visual');self.assertTrue(np.all(np.abs(points)<=1+1e-12))
 def test_object_probe_missing_object_refused(self):
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" size="1 1 1"/></worldbody></mujoco>');d=mujoco.MjData(m)
  with self.assertRaises(ValueError):audit.object_probes(m,d)
if __name__=='__main__':unittest.main(verbosity=2)
