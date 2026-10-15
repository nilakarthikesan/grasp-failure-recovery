import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from independent_validator import *
from independent_geometry_v5 import MeshUnion,closest_pairs,grid,max_interval,adaptive_compare_meshes,box_boundary,target_part
BASE=Path('/Users/nilakarthikesan/grasp-failure-prediction/local_assets/dexgrasp_anything')
SOURCE=BASE/'source-86353d710ca71d0f547f0d8b15cc8ca959c78163/assets/urdf/sr_grasp_description/urdf/shadowhand.urdf'
TARGET=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf')
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.s=Tree(SOURCE,SOURCE_SHA);cls.t=Tree(TARGET,TARGET_SHA)
 def manufactured(self):
  q={n:(max(self.s.joints[n]['limits'][0],self.t.joints[n]['limits'][0])+min(self.s.joints[n]['limits'][1],self.t.joints[n]['limits'][1]))/2 for n in NAMES}
  R=Rotation.from_euler('xyz',[.31,-.47,.82]).as_matrix();raw=np.r_[[.03,-.02,.07],R.T[:2].reshape(6),[q[n] for n in NAMES]]
  _,sf=decode_source(raw,self.s);root=mm(sf['palm'],inv(self.t.fk(q)['palm']));return raw,q,root
 def test_scipy_five_rotation_vectors(self):
  for a in [0,.3,.71,1.4,2.1]:
   R=Rotation.from_euler('xyz',[a,-a*.4,a*.7]).as_matrix();r=source_rotation(R.T[:2].reshape(6));self.assertLess(np.max(abs(r-R)),1e-12)
 def test_nonorthogonal_symmetric_construction(self):
  r=source_rotation([2,0,0,1,2,0]);theta=math.atan2(2,1);want=Rotation.from_euler('z',(theta-math.pi/2)/2).as_matrix();self.assertLess(np.max(abs(r-want)),1e-12)
 def test_domains_boundaries_and_nextafter(self):
  raw,q,root=self.manufactured()
  for n in NAMES:
   for endpoint in self.t.joints[n]['limits']:
    copy=dict(q);copy[n]=endpoint;self.assertNotIn(n,self.t.violations(copy))
   lo,hi=self.t.joints[n]['limits'];copy=dict(q);copy[n]=np.nextafter(lo,-np.inf);self.assertIn(n,self.t.violations(copy));copy[n]=np.nextafter(hi,np.inf);self.assertIn(n,self.t.violations(copy))
 def test_baseline_identity_and_world_equivariance(self):
  raw,q,root=self.manufactured();a=validate_saved(raw,q,root,self.s,self.t,baseline_joints=q,saved_raw=raw.copy())
  T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[-.6,.2,1.1]).as_matrix();T[:3,3]=[.2,-.1,.3]
  b=validate_saved(raw,q,mm(T,root),self.s,self.t,baseline_joints=q,object_from_input=T,saved_raw=raw.copy())
  self.assertLess(max(abs(a['landmark_errors_m'][n]-b['landmark_errors_m'][n]) for n in a['landmark_sites']),1e-12)
 def test_reject_wrong_palm_root(self):
  raw,q,root=self.manufactured();root[0,3]+=.001
  with self.assertRaisesRegex(ValueError,'palm'):validate_saved(raw,q,root,self.s,self.t)
 def test_reject_changed_wrists(self):
  raw,q,root=self.manufactured();q['WRJ1']+=.001
  with self.assertRaisesRegex(ValueError,'wrists'):validate_saved(raw,q,root,self.s,self.t)
 def test_reject_modified_native(self):
  raw,q,root=self.manufactured();s=raw.copy();s[0]=np.nextafter(s[0],np.inf)
  with self.assertRaisesRegex(ValueError,'bytes'):validate_saved(raw,q,root,self.s,self.t,saved_raw=s)
 def test_reject_reflection_nan_degenerate(self):
  for x in [np.diag([-1,1,1,1]),np.full((4,4),np.nan)]:
   with self.assertRaises(ValueError):rigid(x)
  for r in [np.zeros(6),[1,0,0,1,0,0],[1,0,0,-1,0,0]]:
   with self.assertRaises(ValueError):source_rotation(r)
 def test_distances_triangle_regions_and_degenerate(self):
  triangle=np.array([[0,0,0],[1,0,0],[0,1,0]],float);p=np.array([[.2,.2,1],[0,0,2],[1,1,0],[-1,0,0],[.5,-1,0]],float);expected=[1,2,math.sqrt(.5),1,1]
  self.assertLess(np.max(abs(closest_pairs(p,np.repeat(triangle[None],len(p),axis=0))-expected)),1e-12)
  t=np.zeros((1,3,3));self.assertEqual(closest_pairs(np.array([[0.,3,4]]),t)[0],5)
 def test_uncapped_complete_search_vs_all_pairs(self):
  rng=np.random.default_rng(2026101509);v=rng.normal(size=(90,3));f=np.arange(90).reshape(-1,3);p=rng.normal(size=(37,3))*8;mesh=MeshUnion(v,f);actual,_=mesh.query(p,chunk=7)
  allp=np.repeat(p,len(f),axis=0);allt=np.tile(v[f],(len(p),1,1));expected=closest_pairs(allp,allt).reshape(len(p),len(f)).min(axis=1)
  self.assertLess(np.max(abs(actual-expected)),1e-12);self.assertGreater(actual.max(),.004)
 def test_unused_vertex_does_not_bound_surface_distance(self):
  vertices=np.array([[1.,0,0],[1,1,0],[1,0,1],[0,0,0]])
  mesh=MeshUnion(vertices,np.array([[0,1,2]]));d,_=mesh.query(np.array([[0.,0,0]]));self.assertEqual(d[0],1.)
 def test_box_boundary_exact_unsigned_distance(self):
  v,f=box_boundary([2,4,6]);m=MeshUnion(v,f);p=np.array([[0.,0,0],[3,0,0],[3,4,5],[0,2,0]])
  actual,_=m.query(p);self.assertLess(np.max(abs(actual-[1,2,math.sqrt(12),0])),1e-12)
 def test_all_five_declared_target_part_types_supported(self):
  raw,q,root=self.manufactured();frames={n:mm(root,t)for n,t in self.t.fk(q).items()}
  for part in PARTS:self.assertGreater(len(target_part(self.t,frames,part).faces),0)
 def test_grid_coverage_and_interval_decision(self):
  p,rho=grid(.001);self.assertEqual(p.shape,(15000,3));self.assertTrue(np.all(np.any(abs(p)==.025,axis=1)))
  s=np.zeros(len(p));b=np.full(len(p),.01);c=s.copy();bi=max_interval(s,b,rho);ci=max_interval(s,c,rho);self.assertLess(ci['continuous_max_upper_m'],bi['continuous_max_lower_m'])
  self.assertGreater(rho,math.sqrt(2)*.001/2)
class AdaptiveChecks(unittest.TestCase):
 def test_complete_cover_and_analytic_max_enclosure(self):
  import tempfile
  class Surface:
   def __init__(self,stage):self.stage=stage
   def query(self,p):
    if self.stage==0:return np.zeros(len(p)),0
    return .01+10*p[:,0]**2+20*p[:,1]**2+30*p[:,2]**2+(self.stage==2)*.0001,0
  with tempfile.TemporaryDirectory() as d:
   r=adaptive_compare_meshes(Surface(0),Surface(1),Surface(2),output_dir=d,prefix='analytic')
   for stage,true in [('baseline',.0475),('candidate',.0476)]:self.assertLessEqual(r[stage]['continuous_max_lower_m'],true);self.assertGreaterEqual(r[stage]['continuous_max_upper_m'],true)
   self.assertEqual(len(r['levels']),3);self.assertTrue(r['full_cube_coverage_verified'])
   self.assertLess(sum(x['queried_cell_count'] for x in r['levels']),15000+60000+240000)
if __name__=='__main__':unittest.main(verbosity=2)
