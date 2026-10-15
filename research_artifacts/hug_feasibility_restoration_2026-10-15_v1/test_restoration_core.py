"""Manufactured scalar kinematics only; no HUG scene or geometry imports."""
import copy,unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import restoration_core as r
C={'trust_half_width_scaled':.25,'maximum_FK_executions_per_endpoint':400,'maximum_analytic_jacobians_per_endpoint':60,'maximum_iterations_per_endpoint':40,'inward_target_m':1e-9,'optimizer_ftol':1e-12,'cache_size':128}
class Tests(unittest.TestCase):
 def fixture(self):
  x=np.zeros(27);x[0]=.011;bounds=np.tile([-2.,2.],(27,1));caps=np.full(5,.010);events=[]
  def fk(x):
   e=np.zeros((5,3));e[:,0]=x[0];return {'error_vectors_m':e,'in_domain':True}
  def jac(x):
   J=np.zeros((5,3,27));J[:,0,0]=1.;return J
  return x,bounds,caps,events,fk,jac
 def test_linear_endpoint_projection(self):
  x,b,c,events,f,j=self.fixture();v=r.project_endpoint(x,b,c,f,j,C,events.append)
  self.assertIsNotNone(v['selected']);self.assertTrue(v['fresh_FK']['tip_status']['passed']);self.assertLess(v['selected']['distance_squared_scaled'],2e-6);self.assertEqual(v['surface_geometry_calls_during_projection'],0)
 def test_original_cap_and_mean_equality(self):
  e=np.zeros((5,3));e[:,0]=.01;self.assertTrue(r.tip_status(e,np.full(5,.01))['passed']);e[0,0]=np.nextafter(.01,np.inf);self.assertFalse(r.tip_status(e,np.full(5,.01))['passed'])
 def test_zero_norm_jacobian_and_cap(self):
  e=np.zeros((5,3));self.assertTrue(r.tip_status(e,np.zeros(5))['passed']);self.assertTrue(np.array_equal(r.norm_jacobian(e,np.ones((5,3,27))),np.zeros((5,27))))
 def test_full27_key(self):
  x=np.zeros(27);y=x.copy();y[26]=1.;self.assertNotEqual(r.key(x),r.key(y))
 def test_trust_box_cannot_find_distant_feasible_state(self):
  x,b,c,events,f,j=self.fixture();x[0]=1.;v=r.project_endpoint(x,b,c,f,j,C,events.append);self.assertIsNone(v['selected']);self.assertTrue(all(np.max(np.abs(np.asarray(e['value']['state_scaled'])-x))<=.25 for e in events if e['kind']=='FK_callback'))
 def test_raw_constraint_probe_cannot_be_selected(self):
  x,b,c,events,f,j=self.fixture()
  def fake(fun,x0,**kw):
   probe=x0.copy();probe[0]=.001;kw['constraints'][0]['fun'](probe)
   return SimpleNamespace(x=x0,status=9,success=False,message='fake',nit=0,nfev=2,njev=0,fun=fun(x0))
  with patch.object(r,'minimize',fake):v=r.project_endpoint(x,b,c,f,j,C,events.append)
  self.assertIsNone(v['selected']);self.assertEqual(len(v['eligible_checkpoints']),0)
 def test_original_domain_refusal(self):
  x,b,c,events,f,j=self.fixture()
  def invalid(x):v=f(x);v['in_domain']=False;return v
  v=r.project_endpoint(x,b,c,invalid,j,C,events.append);self.assertIsNone(v['selected']);self.assertEqual(v['termination'],'InvalidState')
 def test_fresh_refusal_cannot_return_stale_feasible(self):
  x,b,c,events,f,j=self.fixture();x[0]=.001;calls=0
  def changed(x):
   nonlocal calls
   calls+=1;v=f(x)
   if calls>1:v['error_vectors_m'][:]=1.
   return v
  def fake(fun,x0,**kw):return SimpleNamespace(x=x0,status=0,success=True,message='fake',nit=0,nfev=0,njev=0,fun=fun(x0))
  with patch.object(r,'minimize',fake):v=r.project_endpoint(x,b,c,changed,j,C,events.append)
  self.assertIsNone(v['selected']);self.assertTrue(any(e['kind']=='fresh_selection_refusal' for e in events))
 def test_budget_is_bounded_and_preserved(self):
  x,b,c,events,f,j=self.fixture();cfg=copy.deepcopy(C);cfg['maximum_FK_executions_per_endpoint']=2
  v=r.project_endpoint(x,b,c,f,j,cfg,events.append);self.assertLessEqual(v['counts']['FK_executions'],2);self.assertEqual(v['termination'],'BudgetStop');self.assertIsNone(v['selected'])
 def test_no_orientation_disappearance(self):
  v={'in_domain':True,'tip_status':{'passed':True},'rank':[1,0,1.,1,1.],'exact_orientation_fields':{'valid':[False]},'exact_orientation':{'passed':[False]}}
  g=r.surface_gate(v,[1,0,2.,2,2.],[True],[True]);self.assertFalse(g['accepted_for_limited_coarse_audit']);self.assertEqual(g['lost_orientation_pass_rows'].tolist(),[0]);self.assertEqual(g['lost_orientation_valid_rows'].tolist(),[0])
 def test_strict_rank_and_orientation_gate(self):
  v={'in_domain':True,'tip_status':{'passed':True},'rank':[1,0,1.,1,1.],'exact_orientation_fields':{'valid':[True]},'exact_orientation':{'passed':[True]}}
  self.assertTrue(r.surface_gate(v,[1,0,2.,2,2.],[True],[True])['accepted_for_limited_coarse_audit']);self.assertFalse(r.surface_gate(v,v['rank'],[True],[True])['accepted_for_limited_coarse_audit'])
 def test_empty_orientation_keeps_rank_gate(self):
  v={'in_domain':True,'tip_status':{'passed':True},'rank':[1,0,1.,1,1.],'exact_orientation_fields':{'valid':[]},'exact_orientation':{'passed':[]}}
  self.assertTrue(r.surface_gate(v,[1,0,2.,2,2.],[],[])['accepted_for_limited_coarse_audit']);self.assertFalse(r.surface_gate(v,v['rank'],[],[])['scientific_mapping_acceptance'])
 def test_malformed_rank_refuses(self):
  v={'in_domain':True,'tip_status':{'passed':True},'rank':[0,0,float('nan'),0,0],'exact_orientation_fields':{'valid':[]},'exact_orientation':{'passed':[]}}
  with self.assertRaises(ValueError):r.surface_gate(v,[1,0,2.,2,2.],[],[])
 def test_nonfinite_domain_or_trust_refuses(self):
  x,b,c,events,f,j=self.fixture();b[0,0]=-np.inf
  with self.assertRaises(ValueError):r.project_endpoint(x,b,c,f,j,C,events.append)
  b[0,0]=-2.;cfg=copy.deepcopy(C);cfg['trust_half_width_scaled']=np.inf
  with self.assertRaises(ValueError):r.project_endpoint(x,b,c,f,j,cfg,events.append)
if __name__=='__main__':unittest.main(verbosity=2)
