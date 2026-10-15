"""Frozen existing-input static feasible path: no inference or integration."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/private/tmp/hug_se3_protected_fit_2026-10-15_v1')
from launch_protected import preload
preload()
from se3_case import Case,base
from feasible_path_solver import solve
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
 protocol_path=HERE/'protocol.json';protocol=json.loads(protocol_path.read_text());ph=sha(protocol_path)
 def verify():
  assert sha(protocol_path)==ph
  for path,h in protocol['bindings'].items():assert sha(path)==h,path
 verify();assert protocol['execution_authorized'] and a.index in protocol['case_indices']
 out=HERE/'pilot'/str(a.index);assert not out.exists()
 source=json.loads(Path(protocol['source_cases'][str(a.index)]['path']).read_text())
 forwards=[0];original=base.mujoco.mj_forward
 def forward(*args,**kw):forwards[0]+=1;return original(*args,**kw)
 base.mujoco.mj_forward=forward
 case=Case(a.index);assert case.native_hash==source['native_sha256']
 assert np.array_equal(case.caps,source['original_tip_caps_m']) and np.array_equal(case.bounds,source['bounds_scaled'])
 x0=np.asarray(source['returned_state_scaled']);old=source['returned_exact'];fresh=case.exact(x0)
 for key in ('rank','literal_surface_pass'):
  assert np.array_equal(fresh[key],old[key]),key
 assert np.array_equal(fresh['exact_orientation_fields']['valid'],old['exact_orientation_fields']['valid'])
 assert np.array_equal(fresh['tip_status']['norms_m'],old['tip_status']['norms_m'])
 assert fresh['in_domain'] and fresh['tip_status']['passed']
 start=time.perf_counter();x,exact,receipt=solve(case,x0,source['source_reference_exact'],protocol['solver'],out)
 assert base.PHYSICS_CALLS==0 and case.data.time==0
 assert np.array_equal(receipt['start_exact']['rank'],old['rank'])
 verify()
 result={'schema':'hug_feasible_path_case_v1','index':a.index,'protocol_sha256':ph,'native_sha256':case.native_hash,'original_tip_caps_m':case.caps,'start_binding_exact_reproduced':True,'initial_exact':fresh,'returned_state_scaled':x,'returned_exact':exact,'solver_receipt':receipt,'actual_mj_forward_including_reconstruction_and_start_binding':forwards[0],'physics_steps':0,'new_eval_data':False,'new_native_grasps':False,'full_mapping_acceptance':False,'runtime_s':time.perf_counter()-start}
 target=out/'result.json';assert not target.exists();target.write_text(json.dumps(base.plain(result),indent=2,allow_nan=False)+'\n')
 print(json.dumps({'index':a.index,'result':str(target),'sha256':sha(target),'counts':receipt['counts'],'termination':receipt['termination'],'start_rank':fresh['rank'],'returned_rank':exact['rank'],'all_witness_constraints_pass':exact['all_witness_constraints_pass'],'runtime_s':result['runtime_s']}),flush=True)
if __name__=='__main__':main()
