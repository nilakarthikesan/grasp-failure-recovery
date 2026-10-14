"""Manufactured verification of recorded major/raw selection, no solver."""
from pathlib import Path
import json,importlib.util,hashlib,copy
import numpy as np
B=Path(__file__).parent;s=importlib.util.spec_from_file_location('constrained_validation',B/'validate_candidate.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
p={'maximum_joint_delta_rad':.35,'individual_fingertip_arithmetic_allowance_m':1e-8,'existing_engineering_mean_tip_guard_m':.010,'minimum_objective_improvement':1e-12};names=[f'joint_{i}' for i in range(24)];bounds={n:(-1.,1.) for n in names};q0=np.zeros(24)
def assessment(cost,increase=0.):
    errors=np.zeros((5,3));errors[:,2]=.005;errors[0,2]+=increase;norm=np.linalg.norm(errors,axis=1)
    return {'objective':cost,'tip_error_vectors_m':errors.tolist(),'tip_error_norm_m':norm.tolist()}
def make(rawcost=8.7,empty=False):
    q1=q0.copy();q1[0]=.01;q2=q0.copy();q2[1]=.02;q3=q0.copy();q3[2]=.03;qr=q0.copy();qr[3]=.04
    trace=[] if empty else [{'major_iteration':1,'qpos_rad':q1.tolist(),'assessment':assessment(9.),'eligible_lower_objective_candidate':True},{'major_iteration':2,'qpos_rad':q2.tolist(),'assessment':assessment(8.,2e-8),'eligible_lower_objective_candidate':False},{'major_iteration':3,'qpos_rad':q3.tolist(),'assessment':assessment(8.5),'eligible_lower_objective_candidate':True}]
    selectq=q0 if empty else (qr if rawcost<8.5 else q3);cost=10. if empty else min(8.5,rawcost);source='original_fallback' if empty else ('raw_final' if rawcost<8.5 else 'major_iteration');iteration=3 if source=='major_iteration' else None
    audit={'baseline':assessment(10.),'major_iteration_assessments':trace,'raw_candidate_qpos_rad':None if empty else qr.tolist(),'raw_assessment':None if empty else assessment(rawcost),'raw_final_refusal_reasons':['optimizer_exception'] if empty else [],'raw_final_eligible_lower_objective_candidate':not empty,'selected_candidate_source':source,'selected_major_iteration':iteration,'returned_qpos_rad':selectq.tolist(),'final_assessment':assessment(cost)}
    return {'joint_names':names,'original_qpos_rad':q0.tolist(),'qpos_rad':selectq.tolist(),'solver_audit':audit,'selected_fresh_diagnostics':{'objective':cost}}
results=[]
for name,target in [('major_minimum_over_better_but_infeasible_iterate',make()),('raw_final_lower_minimum',make(8.)),('raw_cost_tie_keeps_earliest_major',make(8.5)),('no_eligible_exact_original_fallback',make(empty=True))]:
    got=v.verify_recorded_selection(target,bounds,p);results.append({'case':name,'passed':True,'selected_source':got['selected_candidate_source']})
bad=make();bad['solver_audit']['selected_major_iteration']=1
try:v.verify_recorded_selection(bad,bounds,p)
except AssertionError:results.append({'case':'wrong_selected_trace_rejected','passed':True})
else:raise AssertionError('Incorrect selected major iteration accepted')
bad=make();bad['solver_audit']['major_iteration_assessments'][1]['eligible_lower_objective_candidate']=True
try:v.verify_recorded_selection(bad,bounds,p)
except AssertionError:results.append({'case':'infeasible_candidate_false_eligible_flag_rejected','passed':True})
else:raise AssertionError('Incorrect feasible flag accepted')
r={'schema':'hug_constrained_selection_audit_manufactured_v1','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256((B/'validate_candidate.py').read_bytes()).hexdigest(),'tests':results,'physics_steps':0,'new_eval_data':False,'new_solver_runs':False,'objective_trace_truth_not_assumed_geometric_ground_truth':True};(B/'selection_audit_tests.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
