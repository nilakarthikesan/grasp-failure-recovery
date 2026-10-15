"""Independent manufactured solver controls; no scene/geometry runtime imports."""
from pathlib import Path
import sys,types,json,gzip,tempfile,unittest
from types import SimpleNamespace
import numpy as np
ROOT=Path('/private/tmp/hug_se3_protected_fit_2026-10-15_v1')
OLD=Path('/private/tmp/hug_se3_mapping_2026-10-15_v1')
sys.path[:0]=[str(ROOT),str(OLD)]
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x) for x in v]
 return v
stub=types.ModuleType('se3_case');stub.base=SimpleNamespace(plain=plain);sys.modules['se3_case']=stub
import protected_solver as s
CONFIG={'maximum_raw_callback_executions_per_case':1000,'maximum_exact_checks_per_case':30,'terminal_raw_and_exact_reserve':2,'maximum_iterations_per_start':25,'surface_difference_step_scaled':1e-4,'raw_cache_size':128,'epigraph_slack_upper':1000.,'tip_inward_target_m':1e-6,'optimizer_ftol':1e-9,'wall_clock_limit_s':180.}
class FakeCase:
 index=999
 def __init__(self):
  self.raw_calls=self.exact_calls=self.tip_jacobian_calls=0
  self.q0=np.zeros(24);self.bounds=np.tile([-1.,1.],(30,1));self.limits=np.tile([-1.,1.],(24,1));self.caps=np.full(5,.006)
  self.domain={'world_displacement_AABB_m':[[-.005,.005]]*3};self.exact_states=[]
 def raw(self,x):
  self.raw_calls+=1;x=np.array(x,float);errors=np.zeros((10,3));errors[:5,0]=.005
  # One failing proximity row wants x>=.5, protected proximity wants x<=.3.
  prox=np.array([x[24]-.5,.3-x[24]]);ori=np.array([1.-1e-8,1.-1e-6,1.-1e-6,1.-1e-6,1.-1e-6])
  norms=np.full(5,.005)
  return {'state_scaled':x,'surface_margin':np.r_[prox-1e-6,ori],'task_error_vectors_m':errors,'tip_status':{'norms_m':norms,'passed':True},'proximity':{'passed':prox>=0}}
 def tip_jacobian(self,x):self.tip_jacobian_calls+=1;return np.zeros((5,3,30))
 def exact(self,x):
  self.exact_calls+=1;v=self.raw(x);self.exact_states.append(np.array(x));valid=np.array([x[25]<=.2]);ori=valid.copy();passed=np.r_[v['proximity']['passed'],ori]
  v.update(in_domain=True,exact_orientation_fields={'valid':valid},exact_orientation={'passed':ori},literal_surface_pass=passed,all_witness_constraints_pass=bool(passed.all()),rank=(int(not passed.all()),int((~valid).sum()),float(max(0.,-v['surface_margin'].min())),int((~passed).sum()),float(np.dot(x,x))))
  return v
class Tests(unittest.TestCase):
 def starts(self):return [{'label':'fixed_pooled','state_scaled':np.r_[np.zeros(24),.2,0.,0.,np.zeros(3)]}]
 def test_complete_30d_key_and_nonfinite_refusal(self):
  x=np.zeros(30);y=x.copy();y[-1]=np.nextafter(0.,1.);self.assertNotEqual(s.state_key(x),s.state_key(y))
  with self.assertRaises(ValueError):s.state_key(np.full(30,np.nan))
 def test_zero_and_nonzero_tip_derivative(self):
  e=np.zeros((5,3));e[1,0]=2.;r=s.norm_jacobian(e,np.ones((5,3,30)));np.testing.assert_array_equal(r[0],np.zeros(30));np.testing.assert_array_equal(r[1],np.ones(30))
 def test_real_solve_cannot_spend_epigraph_on_passed_proximity(self):
  c=FakeCase()
  with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
   x,v,a=s.solve(c,self.starts(),CONFIG,d)
   self.assertLessEqual(x[24],.3+1e-12);self.assertGreater(x[24],.2);self.assertTrue(v['proximity']['passed'][1]);self.assertFalse(v['all_witness_constraints_pass'])
   self.assertLessEqual(a['counts']['raw_callback_executions'],1000);self.assertLessEqual(a['counts']['exact_checkpoint_executions'],30)
   self.assertEqual(a['counts']['raw_callback_executions'],c.raw_calls)
 def test_constraint_and_jacobian_slack_columns(self):
  old=s.minimize
  def fake(fun,z0,**kw):
   g=kw['constraints'][0];a=g['fun'](z0);z=z0.copy();z[-1]+=10.;b=g['fun'](z);J=g['jac'](z0)
   # P2+5O: first failing proximity remains soft; protected proximity and all orientation blocks hard.
   np.testing.assert_allclose(b-a,np.r_[np.zeros(6),10.,np.zeros(6)],atol=1e-12)
   np.testing.assert_array_equal(J[:,-1],np.r_[np.zeros(6),1.,np.zeros(6)])
   self.assertAlmostEqual(J[7,24],-1.,places=10);self.assertAlmostEqual(J[6,24],1.,places=10)
   return SimpleNamespace(status=9,success=False,message='checked actual helper columns',nit=0,nfev=1,njev=1,x=z0.copy())
  s.minimize=fake
  try:
   with tempfile.TemporaryDirectory(dir='/private/tmp') as d:s.solve(FakeCase(),self.starts(),CONFIG,d)
  finally:s.minimize=old
 def test_raw_probe_is_not_selection_candidate(self):
  old=s.minimize
  def fake(fun,z0,**kw):
   z=z0.copy();z[24]=.29;kw['constraints'][0]['fun'](z)
   return SimpleNamespace(status=9,success=False,message='raw only',nit=0,nfev=1,njev=0,x=z0.copy())
  s.minimize=fake
  try:
   c=FakeCase()
   with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
    x,v,a=s.solve(c,self.starts(),CONFIG,d);self.assertEqual(x[24],.2);self.assertFalse(any(y[24]==.29 for y in c.exact_states))
  finally:s.minimize=old
 def test_invalid_orientation_cannot_win(self):
  old=s.minimize
  def fake(fun,z0,**kw):
   z=z0.copy();z[24]=.29;z[25]=.8;kw['callback'](z)
   return SimpleNamespace(status=0,success=True,message='invalid',nit=1,nfev=1,njev=0,x=z)
  s.minimize=fake
  try:
   with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
    x,v,a=s.solve(FakeCase(),self.starts(),CONFIG,d);self.assertEqual(x[24],.2);self.assertEqual(x[25],0.)
  finally:s.minimize=old
 def test_passed_proximity_loss_still_exactly_refused(self):
  old=s.minimize
  def fake(fun,z0,**kw):
   z=z0.copy();z[24]=.5;kw['callback'](z)
   return SimpleNamespace(status=0,success=True,message='violates hard row',nit=1,nfev=1,njev=0,x=z)
  s.minimize=fake
  try:
   with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
    x,v,a=s.solve(FakeCase(),self.starts(),CONFIG,d);self.assertEqual(x[24],.2)
    events=[json.loads(l) for l in gzip.open(a['trace_path'],'rt')];self.assertTrue(any(e['kind']=='protected_reference_refusal' for e in events))
  finally:s.minimize=old
 def test_fresh_terminal_refusal_updates_metadata(self):
  old=s.minimize
  class Changed(FakeCase):
   def exact(self,x):
    v=super().exact(x)
    if self.exact_calls==5:v['rank']=(99,*v['rank'][1:])
    return v
  def fake(fun,z0,**kw):return SimpleNamespace(status=9,success=False,message='unchanged',nit=0,nfev=0,njev=0,x=z0.copy())
  s.minimize=fake
  try:
   with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
    x,v,a=s.solve(Changed(),self.starts(),CONFIG,d);self.assertEqual(a['selected_source_exact_checkpoint']['reason'],'fresh_reference_fallback');self.assertEqual(a['selected_source_exact_checkpoint']['event'],a['actual_returned_fresh_event']);self.assertEqual(x[24],.2)
  finally:s.minimize=old
if __name__=='__main__':unittest.main(verbosity=2)
