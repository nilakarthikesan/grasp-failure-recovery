"""Synthetic mathematical controls only: no source FK, geometry or evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT=Path('/private/tmp/hug_feasibility_restoration_2026-10-15_v1')
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('review_restoration_core',ROOT/'restoration_core.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
checks=[]
def test(name,fn):
    fn();checks.append(name)
def check(condition):
    assert condition
def raises(cls,fn):
    try:fn()
    except cls:return
    raise AssertionError(f'Expected {cls.__name__}')

caps=np.full(5,.010)
errors=np.zeros((5,3));errors[:,0]=.010
test('inclusive_mean_and_individual_equality',lambda:check(c.tip_status(errors,caps)['passed']))
over=errors.copy();over[0,0]=np.nextafter(.010,np.inf)
test('one_ulp_cap_excess_refused',lambda:check(not c.tip_status(over,caps)['passed']))
test('nonfinite_errors_refused',lambda:raises(ValueError,lambda:c.tip_status(np.full((5,3),np.nan),caps)))
test('negative_cap_refused',lambda:raises(ValueError,lambda:c.tip_status(errors,-caps)))
test('complete_27_key_includes_translation',lambda:check(c.key(np.zeros(27))!=c.key(np.r_[np.zeros(26),.1])))
test('truncated_key_refused',lambda:raises(c.InvalidState,lambda:c.key(np.zeros(24))))

J=np.zeros((5,3,27));J[:,0,:5]=np.eye(5)
test('norm_jacobian_exact_axis_fixture',lambda:check(np.array_equal(c.norm_jacobian(errors,J),J[:,0,:])))
test('zero_norm_uses_finite_zero_subgradient',lambda:check(np.array_equal(c.norm_jacobian(np.zeros((5,3)),J),np.zeros((5,27)))))

cfg=dict(trust_half_width_scaled=.25,maximum_FK_executions_per_endpoint=400,
         maximum_analytic_jacobians_per_endpoint=60,cache_size=512,inward_target_m=1e-9,
         maximum_iterations_per_endpoint=40,optimizer_ftol=1e-12)
bounds=np.tile([-1.,1.],(27,1));proposal=np.zeros(27)
def project(fake,tips,config=None):
    old=c.minimize;c.minimize=fake;events=[]
    try:
        result=c.project_endpoint(proposal,bounds,caps,tips,lambda x:J,cfg if config is None else config,events.append)
    finally:c.minimize=old
    return result,events
def result(x):return SimpleNamespace(x=x,status=9,success=False,message='synthetic terminal',nit=1,nfev=1,njev=0,fun=float(np.dot(x,x)/2))
def constant_tips(x,n=.009):
    e=np.zeros((5,3));e[:,0]=n
    return dict(error_vectors_m=e,in_domain=True)

def raw_not_eligible():
    def fake(fun,x,jac,method,bounds,constraints,callback,options):
        z=x.copy();z[0]=.1;constraints[0]['fun'](z)
        return result(x)
    r,events=project(fake,lambda x:constant_tips(x,.009 if x[0]>0 else .011))
    check(r['selected'] is None and r['eligible_checkpoints']==[])
    check(any(e['kind']=='FK_callback' and e['value']['tip_status']['passed'] for e in events))
test('tip_feasible_raw_probe_never_selected',raw_not_eligible)

def stable_closest():
    def fake(fun,x,jac,method,bounds,constraints,callback,options):
        for val in [.2,.1,-.1]:
            z=x.copy();z[0]=val;callback(z)
        return result(z)
    r,_=project(fake,lambda x:constant_tips(x,.011 if x[0]==0 else .009))
    check(r['selected']['state_scaled'][0]==.1)
    check(r['counts']['FK_executions']==5)
test('closest_endpoint_checkpoint_stable_tie',stable_closest)

def fresh_refusal():
    count=[0]
    def tips(x):count[0]+=1;return constant_tips(x,.009 if count[0]==1 else .011)
    r,events=project(lambda fun,x,**kwargs:result(x),tips)
    check(r['selected'] is None and r['fresh_FK']['tip_status']['passed'] is False)
    check(any(e['kind']=='fresh_selection_refusal' for e in events))
test('fresh_FK_failure_removes_selection',fresh_refusal)

def FK_budget():
    def fake(fun,x,callback,**kwargs):
        for val in [.01,.02,.03,.04]:
            z=x.copy();z[0]=val;callback(z)
        return result(z)
    small=dict(cfg,maximum_FK_executions_per_endpoint=4)
    r,_=project(fake,constant_tips,small)
    check(r['termination']=='BudgetStop' and r['counts']['FK_executions']==4)
    check(r['selected'] is not None and r['fresh_FK']['tip_status']['passed'])
test('FK_budget_includes_reserved_fresh_check',FK_budget)

def jac_budget():
    def fake(fun,x,constraints,**kwargs):
        constraints[0]['jac'](x);constraints[0]['jac'](x)
        return result(x)
    r,_=project(fake,constant_tips,dict(cfg,maximum_analytic_jacobians_per_endpoint=1))
    check(r['termination']=='BudgetStop' and r['counts']['analytic_jacobian_executions']==1)
test('analytic_staging_budget_separate_and_finite',jac_budget)

def trust_refusal():
    def fake(fun,x,callback,**kwargs):
        z=x.copy();z[26]=np.nextafter(.25,np.inf);callback(z)
        return result(z)
    r,events=project(fake,constant_tips)
    check(r['termination']=='InvalidState')
    check(any(e['kind']=='state_refusal' for e in events))
test('trust_endpoint_one_ulp_excess_refused',trust_refusal)

def gate(valid,passed,rank=(1,0,.5,1,.1)):
    return dict(exact_orientation_fields={'valid':valid},exact_orientation={'passed':passed},
                rank=rank,in_domain=True,tip_status={'passed':True})
ref=(1,0,1.,2,0.)
test('rank_improvement_with_preservation_eligible',lambda:check(c.surface_gate(gate([1,1],[1,0]),ref,[1,1],[1,0])['accepted_for_limited_coarse_audit']))
test('bad_but_previously_eligible_row_cannot_be_invalidated',lambda:check(not c.surface_gate(gate([1,0],[1,0]),ref,[1,1],[1,0])['accepted_for_limited_coarse_audit']))
test('previously_passed_row_cannot_fail',lambda:check(not c.surface_gate(gate([1,1],[0,0]),ref,[1,1],[1,0])['accepted_for_limited_coarse_audit']))
test('equal_rank_refused',lambda:check(not c.surface_gate(gate([1,1],[1,0],ref),ref,[1,1],[1,0])['accepted_for_limited_coarse_audit']))

report=dict(passed=True,scope='Independent synthetic mathematics and control-flow only; no source FK, geometry, fitting or eval.',checks_passed=len(checks),checks=checks,
            source_bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'restoration_core.py',Path(__file__)]})
p=OUT/'independent_core_controls.json';p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),output=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
