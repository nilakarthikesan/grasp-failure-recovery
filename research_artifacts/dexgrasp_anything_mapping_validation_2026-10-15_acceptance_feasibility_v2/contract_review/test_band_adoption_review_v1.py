"""Hostile optimizer proposal tests; manufactured fields only, no geometry runs."""
from pathlib import Path
import importlib.util,sys,time
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path('/private/tmp/dga_mapping_acceptance_20261015_v2')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
band=load('review_band',ROOT/'control/band_corrector.py')
core=load('review_core','/private/tmp/dga_component_constraints_20261015_v1/optimizer/component_corrector.py')
SOURCE=np.full((3,8),.004,dtype=np.float64)
INITIAL=np.column_stack([np.full((3,7),.001),np.full(3,.004)])
CAPS={'maximum_site_error_m':.005,'all_site_SSE_m2':28*.005**2,'mean_site_error_m':.005}

def state(x,fields,*,landmark=.001,domain=True):
 return {'variables':np.array(x,dtype=float),'fields':fields.copy(),'component_metrics':core.component_metrics(SOURCE,fields),'strict_domain_valid':domain,'original_wrists_preserved':True,'rigid_frames_valid':True,'independent_fk_valid':True,'landmark_metrics':{'per_site_error_m':[landmark]*28,'maximum_site_error_m':landmark,'mean_site_error_m':float(np.array([landmark]*28).mean()),'all_site_SSE_m2':28*landmark**2}}
class Context:
 fingers=tuple('F'+str(i) for i in range(22));qsource={name:0. for name in fingers}
 def __init__(self,variant):self.variant=variant;self.evaluations=0
 def variable_bounds(self,caps):return np.tile([-1.,1.],(28,1))
 def make_state(self,x,initial,block):
  assert block=='palm'
  f=np.full((3,8),.003);f[:,7]=.004
  if self.variant=='component_regression':f[:,0]=.0005
  return state(x,f,landmark=.01 if self.variant=='landmark_regression' else .001,domain=self.variant!='domain_failure')
 def evaluate_fields(self,s):self.evaluations+=1;return s['fields'].copy()

def exercise(monkeypatch,variant,points,terminal=0,success=False):
 initial=state(np.zeros(28),INITIAL);ctx=Context(variant);before=core.payload_digest(initial);source_before=SOURCE.tobytes()
 def fake_minimize(objective,y0,**kwargs):
  for z in points:
   y=np.zeros(28);y[0]=z
   objective(y);c=kwargs['constraints'][0]['fun'](y);assert c.shape==(48,)
  y=np.zeros(28);y[0]=terminal
  return SimpleNamespace(x=y,success=success,message='Manufactured hostile solver',nit=1)
 monkeypatch.setattr(band,'minimize',fake_minimize)
 selected,trace=band.propose(ctx,core,initial,SOURCE,CAPS,deadline=time.monotonic()+10,max_states=30)
 assert core.payload_digest(initial)==before and SOURCE.tobytes()==source_before
 return initial,selected,trace,ctx

def test_lower_proxy_cannot_adopt_component_regression(monkeypatch):
 initial,selected,trace,_=exercise(monkeypatch,'component_regression',[1.],1.,True)
 assert trace['trials'][0]['proxy_objective']<trace['baseline_proxy']
 assert selected is initial and trace['eligible_improved_count']==0 and not trace['adopted_changed_state']
def test_lower_proxy_cannot_adopt_landmark_regression(monkeypatch):
 initial,selected,trace,_=exercise(monkeypatch,'landmark_regression',[1.],1.,True)
 assert trace['trials'][0]['proxy_objective']<trace['baseline_proxy']
 assert selected is initial and trace['eligible_improved_count']==0
def test_solver_success_does_not_adopt_invalid_domain(monkeypatch):
 initial,selected,trace,ctx=exercise(monkeypatch,'domain_failure',[1.],1.,True)
 assert selected is initial and ctx.evaluations==0
def test_terminal_initial_does_not_discard_verified_earlier_good_state(monkeypatch):
 initial,selected,trace,_=exercise(monkeypatch,'good',[1.],0.,False)
 assert selected is not initial and trace['adopted_changed_state']
 assert core.accepted_feasible(selected,CAPS,initial['component_metrics'])
 assert core.strict_pareto_improvement(selected['component_metrics'],initial['component_metrics'])
 assert trace['scientific_acceptance']=='NOT_TESTED' and trace['execution_eligible'] is False
def test_initial_preserved_when_no_new_state(monkeypatch):
 initial,selected,trace,ctx=exercise(monkeypatch,'good',[],0.,True)
 assert selected is initial and trace['trial_count']==0 and ctx.evaluations==0
