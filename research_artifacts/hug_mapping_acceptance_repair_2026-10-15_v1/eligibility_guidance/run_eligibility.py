"""Frozen static saved-input continuation with exact eligibility guidance."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/private/tmp/hug_se3_protected_fit_2026-10-15_v1')
from launch_protected import preload
preload()
from eligibility_case import Case,base
from feasible_eligibility_solver import solve
from extended_policy import protected_surface_constraint_policy
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal_old_exact(new,old):
 for key in ('rank','literal_surface_pass'):
  assert np.array_equal(new[key],old[key]),key
 assert np.array_equal(new['exact_orientation_fields']['valid'],old['exact_orientation_fields']['valid'])
 assert np.array_equal(new['tip_status']['norms_m'],old['tip_status']['norms_m'])
 assert np.array_equal(new['original_surface_margin'],old['surface_margin'])
 assert new['in_domain'] and new['tip_status']['passed']
def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);args=p.parse_args()
 protocolpath=HERE/'protocol.json';protocol=json.loads(protocolpath.read_text());ph=sha(protocolpath)
 def verify():
  assert sha(protocolpath)==ph
  for path,h in protocol['bindings'].items():assert sha(path)==h,path
 verify();assert protocol['execution_authorized'] and args.index in protocol['case_indices']
 out=HERE/'pilot'/str(args.index);assert not out.exists();out.mkdir(parents=True)
 source=json.loads(Path(protocol['source_cases'][str(args.index)]['path']).read_text())
 segment=json.loads(Path(protocol['pooled_reference_cases'][str(args.index)]['path']).read_text())
 forwards=[0];original=base.mujoco.mj_forward
 def forward(*args,**kw):forwards[0]+=1;return original(*args,**kw)
 base.mujoco.mj_forward=forward;case=Case(args.index)
 assert case.native_hash==source['native_sha256'] and np.array_equal(case.caps,source['original_tip_caps_m'])
 oldref=segment['source_reference_exact'];ref=case.exact(oldref['state_scaled']);equal_old_exact(ref,oldref)
 x0=np.asarray(source['returned_state_scaled']);start=case.exact(x0);equal_old_exact(start,source['returned_exact'])
 policy=protected_surface_constraint_policy(ref)
 hard=policy['hard_mask'];passed=bool(np.all(np.asarray(start['surface_margin'])[hard]>=0))
 preflight={'schema':'hug_added_eligibility_start_preflight_v1','index':args.index,'protocol_sha256':ph,'same_old_exact_fields':True,'pooled_reference':ref,'start_exact':start,'hard_mask':hard,'start_hard_margins_pass':passed,'failed_start_hard_rows':np.flatnonzero(hard&(start['surface_margin']<0)),'static_forward_calls':forwards[0],'physics_steps':0,'new_eval_data':False}
 (out/'start_preflight.json').write_text(json.dumps(base.plain(preflight),indent=2,allow_nan=False)+'\n');verify()
 if not passed:raise ValueError('New stricter search guidance refuses old start; no solve executed')
 tic=time.perf_counter();x,exact,receipt=solve(case,x0,ref,protocol['solver'],out)
 assert base.PHYSICS_CALLS==0 and case.data.time==0;verify()
 result={'schema':'hug_feasible_eligibility_case_v1','index':args.index,'protocol_sha256':ph,'native_sha256':case.native_hash,'original_tip_caps_m':case.caps,'initial_exact':start,'returned_state_scaled':x,'returned_exact':exact,'solver_receipt':receipt,'actual_mj_forward_including_reconstruction_and_bindings':forwards[0],'physics_steps':0,'new_eval_data':False,'new_native_grasps':False,'full_mapping_acceptance':False,'runtime_s':time.perf_counter()-tic}
 target=out/'result.json';assert not target.exists();target.write_text(json.dumps(base.plain(result),indent=2,allow_nan=False)+'\n')
 print(json.dumps({'index':args.index,'result':str(target),'sha256':sha(target),'counts':receipt['counts'],'termination':receipt['termination'],'start_rank':start['rank'],'returned_rank':exact['rank'],'all_witness_constraints_pass':exact['all_witness_constraints_pass'],'runtime_s':result['runtime_s']}),flush=True)
if __name__=='__main__':main()
