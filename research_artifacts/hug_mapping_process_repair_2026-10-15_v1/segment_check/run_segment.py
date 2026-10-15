"""Finite saved-endpoint segment check; no optimizer, inference, or integration."""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
INHERITED=Path('/private/tmp/hug_se3_protected_fit_2026-10-15_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact_gate(value,reference):
 valid=np.asarray(value['exact_orientation_fields']['valid'],bool);rv=np.asarray(reference['exact_orientation_fields']['valid'],bool)
 passed=np.asarray(value['literal_surface_pass'],bool);rp=np.asarray(reference['literal_surface_pass'],bool)
 assert valid.shape==rv.shape and passed.shape==rp.shape
 rank=np.asarray(value['rank'],float);rr=np.asarray(reference['rank'],float)
 assert rank.shape==rr.shape==(5,) and np.isfinite(rank).all() and np.isfinite(rr).all()
 lostvalid=np.flatnonzero(rv&~valid);lostpassed=np.flatnonzero(rp&~passed)
 reasons=[]
 if not value['in_domain']:reasons.append('domain')
 if not value['tip_status']['passed']:reasons.append('tips')
 if len(lostvalid):reasons.append('pooled_orientation_eligibility_lost')
 if len(lostpassed):reasons.append('pooled_passed_surface_row_lost')
 if not tuple(rank)<tuple(rr):reasons.append('no_strict_rank_improvement')
 return {'passed':not reasons,'reasons':reasons,'lost_valid_rows':lostvalid,'lost_passed_rows':lostpassed}
def segment_states(reference,terminal):
 reference=np.asarray(reference,float);terminal=np.asarray(terminal,float)
 assert reference.shape==terminal.shape==(30,) and np.isfinite(reference).all() and np.isfinite(terminal).all()
 for k in range(21):yield k,2.**(-k),reference+2.**(-k)*(terminal-reference)
def main():
 protocol_path=HERE/'protocol.json';protocol=json.loads(protocol_path.read_text());protocol_hash=sha(protocol_path)
 def verify():
  assert sha(protocol_path)==protocol_hash
  for path,h in protocol['bindings'].items():assert sha(path)==h,path
 verify();assert protocol['execution_authorized'] is True
 sys.path.insert(0,str(INHERITED))
 from launch_protected import preload
 preload()
 from se3_case import Case,base
 from se3_math import decode_state,ROTATION_COMPONENT_LIMIT_RAD
 assert base.numba.get_num_threads()==2
 counter=[0];original_forward=base.mujoco.mj_forward
 def forward(*args,**kw):counter[0]+=1;return original_forward(*args,**kw)
 base.mujoco.mj_forward=forward
 results=[];start=time.perf_counter()
 for i in (20,28):
  out=HERE/f'case_{i}.json';assert not out.exists()
  source=json.loads(Path(protocol['source_cases'][str(i)]['path']).read_text());sr=source['solver_receipt'];ref=sr['protected_pooled_reference_exact'];x0=np.asarray(sr['frozen_starts'][0]['state_scaled']);terminal=np.asarray(sr['optimizer_starts'][0]['terminal_state_scaled']);case=Case(i)
  assert source['native_sha256']==case.native_hash
  assert np.array_equal(x0,np.asarray(ref['state_scaled']))
  pooled=json.loads(Path(protocol['pooled_candidates']).read_text());p=next(c for c in pooled['cases'] if c['source_inventory_index']==i)
  assert np.array_equal(case.caps,np.asarray(p['authoritative_original_tip_caps_sqrt_einsum_m']))
  assert np.array_equal(x0,np.r_[p['selected_state_scaled'],np.zeros(3)])
  records=[];selected=None;fresh=None;tip_calls=0;exact0=case.exact_calls;raw0=case.raw_calls;forward0=counter[0]
  for k,lam,x in segment_states(x0,terminal):
   q,shift,omega=decode_state(x,case.q0);aabb=np.asarray(case.domain['world_displacement_AABB_m'])
   domain=bool(np.all(x>=case.bounds[:,0]) and np.all(x<=case.bounds[:,1]) and np.all(q>=case.limits[:,0]) and np.all(q<=case.limits[:,1]) and np.all(q>=case.q0-.35) and np.all(q<=case.q0+.35) and np.all(shift>=aabb[:,0]) and np.all(shift<=aabb[:,1]) and np.all(np.abs(omega)<=ROTATION_COMPONENT_LIMIT_RAD))
   row={'k':k,'lambda':lam,'state_scaled':x,'in_domain':domain}
   if domain:
    before=case.raw_calls;case.stage(x);tip_calls+=1;errors=base._link_positions(case.model,case.data,base.a.URDF)[:5]-case.native_tasks[:5];tip=base.exact_tip_status(errors,case.caps);assert case.raw_calls==before
    row.update(error_vectors_m=errors,tip_status=tip)
    if tip['passed']:
     exact=case.exact(x);gate=exact_gate(exact,ref);row.update(exact=exact,gate=gate)
     assert np.array_equal(np.asarray(exact['tip_status']['norms_m']),np.asarray(tip['norms_m']))
     if gate['passed']:selected={'k':k,'lambda':lam,'state_scaled':x,'exact':exact,'gate':gate};records.append(row);break
   records.append(row)
  if selected is not None:
   fresh=case.exact(selected['state_scaled']);gate=exact_gate(fresh,ref)
   same=bool(np.array_equal(fresh['exact_orientation_fields']['valid'],selected['exact']['exact_orientation_fields']['valid']) and np.array_equal(fresh['literal_surface_pass'],selected['exact']['literal_surface_pass']) and tuple(fresh['rank'])==tuple(selected['exact']['rank']))
   selected['fresh_gate']=gate;selected['fresh_predicates_and_rank_identical']=same
   if not gate['passed'] or not same:selected['fresh_refused']=True
  accepted=bool(selected is not None and not selected.get('fresh_refused',False))
  assert tip_calls<=21 and case.exact_calls-exact0<=22 and case.raw_calls-raw0==case.exact_calls-exact0
  assert base.PHYSICS_CALLS==0 and case.data.time==0
  verify()
  result={'schema':'hug_two_endpoint_segment_backtracking_v1','index':i,'protocol_sha256':protocol_hash,'source_case':protocol['source_cases'][str(i)],'native_sha256':case.native_hash,'source_reference_exact':ref,'source_terminal_record':sr['optimizer_starts'][0],'reference_state_scaled':x0,'original_tip_caps_m':case.caps,'bounds_scaled':case.bounds,'domain':case.domain,'records':records,'selected':selected,'selected_fresh_exact':fresh,'strict_improvement_eligible':accepted,'returned_state_scaled':selected['state_scaled'] if accepted else x0,'returned_exact':fresh if accepted else ref,'returned_reference_is_saved_unchanged':not accepted,'counts':{'tip_FK_calls':tip_calls,'exact_checks_including_fresh':case.exact_calls-exact0,'raw_calls_all_from_exact':case.raw_calls-raw0,'actual_mj_forward_after_reconstruction':counter[0]-forward0},'mapping_acceptance':False,'new_eval_data':False,'new_native_grasps':False,'physics_steps':0}
  out.write_text(json.dumps(base.plain(result),indent=2,allow_nan=False)+'\n');results.append({'index':i,'path':str(out),'sha256':sha(out),'eligible':accepted,'selected_lambda':None if selected is None else selected['lambda'],'tip_checks':tip_calls,'exact_checks':case.exact_calls-exact0})
  print(json.dumps(results[-1]),flush=True)
 verify();summary={'schema':'hug_two_endpoint_segment_summary_v1','cases':results,'total_runtime_s':time.perf_counter()-start,'actual_mj_forward_including_reconstruction':counter[0],'physics_steps':0,'new_eval_data':False,'mapping_acceptance':False}
 (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
