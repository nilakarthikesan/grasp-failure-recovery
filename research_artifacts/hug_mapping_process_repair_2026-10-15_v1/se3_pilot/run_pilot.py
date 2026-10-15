from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from se3_case import Case,base
from se3_solver import solve
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
 protocol_path=HERE/'pilot_protocol.json';protocol=json.loads(protocol_path.read_text());frozen=sha(protocol_path)
 assert args.index in protocol['case_indices']
 def verify():
  assert sha(protocol_path)==frozen
  for p,h in protocol['bindings'].items():assert sha(p)==h,p
 verify()
 out=HERE/'pilot'/str(args.index);assert not out.exists(),'Never overwrite a trial'
 forwards=[0];original=base.mujoco.mj_forward
 def forward(*a,**k):forwards[0]+=1;return original(*a,**k)
 base.mujoco.mj_forward=forward
 case=Case(args.index);prior=json.loads(Path(protocol['pooled_candidates']).read_text());source=next(c for c in prior['cases'] if c['source_inventory_index']==args.index)
 assert source['native_sha256']==case.native_hash and np.array_equal(np.asarray(source['authoritative_original_tip_caps_sqrt_einsum_m']),case.caps)
 starts=[{'label':'original_zero_rotation','state_scaled':np.zeros(30)},{'label':'previous_pooled_zero_rotation','state_scaled':np.r_[source['selected_state_scaled'],np.zeros(3)]}]
 start=time.perf_counter();x,exact,receipt=solve(case,starts,protocol['solver'],out)
 assert base.PHYSICS_CALLS==0 and case.data.time==0
 verify()
 final={'schema':'hug_se3_mapping_pilot_case_v1','index':args.index,'native_sha256':case.native_hash,'protocol_sha256':frozen,'state_scaled':x,'selected_exact':exact,'solver_receipt':receipt,'actual_mj_forward_calls_including_reconstruction':forwards[0],'physics_steps':base.PHYSICS_CALLS,'new_eval_data':False,'new_native_grasps':False,'runtime_s':time.perf_counter()-start}
 target=out/'result.json';assert not target.exists();target.write_text(json.dumps(base.plain(final),indent=2,allow_nan=False)+'\n')
 print(json.dumps({'index':args.index,'result':str(target),'sha256':sha(target),'selected_state':x.tolist(),'all_witness_constraints_pass':exact['all_witness_constraints_pass'],'rank':exact['rank'],'tip_pass':exact['tip_status']['passed'],'counts':receipt['counts'],'runtime_s':final['runtime_s'],'physics_steps':0}),flush=True)
if __name__=='__main__':main()
