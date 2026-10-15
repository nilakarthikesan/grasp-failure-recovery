"""Independent synthetic controls; no grasp/source mesh, generator or physics."""
from pathlib import Path
import hashlib, importlib, json, sys
import numpy as np
PRIMARY=Path('/private/tmp/hug_free_palm_fit_2026-10-15_v1')
OUT=Path(__file__).parent
sys.path.insert(0,str(PRIMARY))
from free_palm_solver import FreePalmConfig, solve_free_palm
from keyed_direction_evidence import KeyedDirectionEvidence
from grasp_failure_prediction.evaluation.constrained_surface_refinement import ConstrainedEvaluation,native_direction_residual

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
bound={str(PRIMARY/n):sha(PRIMARY/n) for n in ['free_palm_solver.py','keyed_direction_evidence.py','surface_component_guard.py']}
checks=[]
def check(value,name):
    assert value,name
    checks.append(name)

ref={'palm':{'available':True,'native_score':np.array([-.5,.5]),'weights':np.array([1/12,1/12])}}
def data(scores):
    valid=np.array([True,True]);s=np.asarray(scores)
    residual=native_direction_residual(ref['palm']['native_score'],s,ref['palm']['weights']*6,valid)/np.sqrt(6)
    v=ConstrainedEvaluation(residual,np.zeros((5,3)))
    d={'normal_diagnostics':{'palm':{'available':True,'native_score':[-.5,.5],'candidate_score':s.tolist(),'candidate_valid':valid.tolist(),'residual':residual.tolist()}}}
    return v,d
q0=np.zeros(27);shift=q0.copy();shift[24]=1
keyed=KeyedDirectionEvidence(q0,ref,normal_block_start=0,normal_block_count=2)
v0,d0=data([-.5,.5]);v1,d1=data([.5,-.5]);keyed.record(q0,v0,d0);keyed.record(shift,v1,d1)
a0=keyed.assess(q0,v0);a1=keyed.assess(shift,v1)
check(a0['accepted'] and a0['metadata_callback_sequence']==1,'original lookup unaffected by later translated callback')
check(not a1['accepted'] and a1['metadata_callback_sequence']==2,'same joints different palm shift has separate refused evidence')
check(a0['q_float64_sha256']!=a1['q_float64_sha256'],'translation included in exact27 hash')
for column in (24,25,26):
    unknown=q0.copy();unknown[column]=1e-7
    try:keyed.assess(unknown,v0)
    except ValueError:check(True,f'unrecorded translation coordinate {column} refused')
    else:raise AssertionError('missing translation bytes')
# Native negative score is retained in diagnostic evidence and not rectified.
check(keyed.audit()['callbacks'][0]['normal_diagnostics']['palm']['native_score'][0]==-.5,'negative native direction preserved')

bounds=np.tile([-2.,2.],(27,1));seed=np.zeros((1,27));seed[0,24]=1
cfg=FreePalmConfig(maximum_pattern_sweeps=0,maximum_full_geometry_evaluations=8,maximum_FK_evaluations=200,maximum_restorations=4,initial_pattern_radius_scaled=1.,minimum_pattern_radius_scaled=.125)
def linear_tips(x):
    v=np.zeros((5,3));v[:,0]=.005-.001*x[24];return v
def jac(x):
    v=np.zeros((5,3,27));v[:,0,24]=-.001;return v
def linear_full(x):return ConstrainedEvaluation(np.array([x[24]-1]),linear_tips(x))
out,report=solve_free_palm(q0,bounds,linear_full,evaluate_fingertips=linear_tips,config=cfg,predeclared_seed_state_scaled=seed,evaluate_fingertip_jacobian=jac)
check(np.array_equal(out,seed[0]) and report['surface_candidate_returned'],'solver can select a pure translated palm with unchanged joint coordinates')
check(all(report['final']['individual_tip_nonworsening_pass']),'all original synthetic caps retained for translated solution')
check(report['final_callbacks_fresh_outside_cache'],'translated terminal state freshly checked')

def zero_tips(x):
    v=np.zeros((5,3));v[:,0]=.001*x[24];return v
def zero_jac(x):
    v=np.zeros((5,3,27));v[:,0,24]=.001;return v
def zero_full(x):return ConstrainedEvaluation(np.array([x[24]-1]),zero_tips(x))
zero_cfg=FreePalmConfig(maximum_pattern_sweeps=1,maximum_full_geometry_evaluations=8,maximum_FK_evaluations=300,maximum_restorations=8,initial_pattern_radius_scaled=1.,minimum_pattern_radius_scaled=.125)
out,report=solve_free_palm(q0,bounds,zero_full,evaluate_fingertips=zero_tips,config=zero_cfg,predeclared_seed_state_scaled=seed,evaluate_fingertip_jacobian=zero_jac)
check(np.array_equal(out,q0),'zero original cap never relaxed to accept positive tip error')
check(all(x==0 for x in report['final']['tip_error_norm_m']),'zero cap exact terminal errors retained')
check(any(e['kind']=='analytic_FK_constraint_jacobian' and e['zero_norm_rows']>0 for e in report['events']),'zero norm analytic-row convention exercised')
for path,h in bound.items():check(sha(path)==h,'bound module unchanged '+Path(path).name)
report={'schema':'independent_free_palm_manufactured_preflight_v1','module_sha256':bound,'script_sha256':sha(__file__),'checks':checks,'passed':True,'check_count':len(checks),'actual_grasp_geometry_evaluated':False,'new_eval_data':False,'physics_steps':0}
(OUT/'manufactured_preflight_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'report':str(OUT/'manufactured_preflight_report.json')}))
