"""Manufactured algebra controls only; no source/native/model inputs."""
import importlib.util,math
from pathlib import Path
import numpy as np
source=Path(__file__).with_name('verify_saved.py');source=source if source.exists() else Path(__file__).with_name('verify_hug_native_generation_v2.py')
s=importlib.util.spec_from_file_location('independent_saved_generation',source);v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
def test_camera_and_world_nonzero():
 R=np.array([[0.,-1,0],[1.,0,0],[0,0,1.]])
 c=np.eye(4);c[:3,:3]=R;c[:3,3]=[.1,-.2,.3]
 got=v.world(R,np.array([[1.,2.,3.]]),c[:3,3]);np.testing.assert_allclose(got,[[-1.9,.8,3.3]],atol=1e-15,rtol=0);v.camera(c)
def test_bad_camera_refused():
 c=np.eye(4);c[0,0]=1.01
 try:v.camera(c)
 except ValueError:return
 raise AssertionError('Nonrigid camera accepted')
def test_rest_compensation_noncommuting():
 a=np.array([.2,-.1,.3]);b=np.array([-.1,.2,.15])
 def q(x):
  n=np.linalg.norm(x);return np.r_[math.cos(n/2),x*math.sin(n/2)/n]
 rest=q(a);local=q(b);joint=v.mul(np.r_[rest[0],-rest[1:]],local)
 np.testing.assert_allclose(v.quat_matrix(v.mul(rest,joint)),v.exp_matrix(b),rtol=0,atol=1e-15)
def test_native_hierarchy_all_branches():
 j=np.arange(48).reshape(16,3)/100.;aa=np.zeros(48,dtype=np.float32);c=np.eye(4);c[:3,3]=[.1,.2,.3];t=np.array([.4,.5,.6],dtype=np.float32)
 a,R,L=v.source_fk(aa,t,c,j)
 np.testing.assert_allclose(a,j-j[0]+t.astype(float)+c[:3,3],rtol=0,atol=8*np.finfo(float).eps);np.testing.assert_array_equal(R,np.tile(np.eye(3),(16,1,1)))
def test_literal_nonfinite_and_dtype_refused():
 for a in (np.array([1.,np.nan,2],dtype=np.float32),np.ones(3,dtype=np.float64)):
  try:v.arr(a,(3,),'float32')
  except ValueError:continue
  raise AssertionError('Bad literal accepted')
def test_geodesic_pi_and_doublecover():
 np.testing.assert_allclose(v.geodesic(np.eye(3),v.exp_matrix(np.array([math.pi,0,0]))),math.pi,rtol=0,atol=1e-15)
 q=np.array([.5,.5,.5,.5]);np.testing.assert_allclose(v.geodesic(v.quat_matrix(q),v.quat_matrix(-q)),0,rtol=0,atol=1e-15)
def test_seed_and_view_descriptions_keep_pair_units():
 def a(dx):
  q=np.zeros(67);q[0]=dx;q[3:]=np.tile([1.,0,0,0],16);return dict(native_qpos=q,rest_quaternions=np.tile([1.,0,0,0],(16,1)),prediction99=np.full(99,dx,dtype=np.float32))
 rows=[(dict(observation='obj/'+str(view),object_path='obj',seed=seed,id=f'{view}_{seed}'),a(view+seed)) for view in (0,1) for seed in (42,43)]
 d=v.pose_diversity(rows);assert d['within_observation_seed42_vs43']['pairs']==2;assert d['across_two_stored_observations_same_object_matched_seed']['pairs']==2;assert d['exact_duplicates']['native_qpos']['exact_duplicate_rows_beyond_first']==1
