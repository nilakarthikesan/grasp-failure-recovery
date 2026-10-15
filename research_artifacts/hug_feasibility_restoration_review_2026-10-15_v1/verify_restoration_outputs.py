"""Independent saved-source FK and all recorded witness arithmetic/selection.
Does not import primary engine/core/solver or query any hand surface.
"""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,pickle,sys,time,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
from constraint_arithmetic import Replay
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT=Path(__file__).parent
OLDREVIEW=Path('/private/tmp/hug_free_palm_review_2026-10-15_v1/verify_outputs.py')
spec=importlib.util.spec_from_file_location('independent_fk_helpers',OLDREVIEW);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
sha=v.sha;read=v.read;check=v.check;same=v.same;near=v.near;norm=v.norm;ident=v.ident

def trace_review(row,saved,op):
 index=row['source_inventory_index'];caps=np.asarray(row['authoritative_original_tip_caps_sqrt_einsum_m']);bounds=np.asarray(row['bounds_scaled'])
 r=Replay(saved['whole_cell_rows'],saved['orientation_rows'],index,op,caps)
 reference=row['reference_constraint_diagnostics'];r.verify(reference,exact=True)
 original=row['baseline_constraint_diagnostics'];ref_rank=tuple(reference['rank'])
 protected_valid=np.asarray(original['exact_orientation_fields']['valid'],bool)|np.asarray(reference['exact_orientation_fields']['valid'],bool)
 protected_pass=np.asarray(original['exact_orientation']['passed'],bool)|np.asarray(reference['exact_orientation']['passed'],bool)
 endpoint_reports=[];coarse_eligible=[]
 def gate(full):
  rv=r.verify(full,exact=True);valid=np.asarray(full['exact_orientation_fields']['valid'],bool);passed=np.asarray(full['exact_orientation']['passed'],bool)
  lostvalid=np.flatnonzero(protected_valid&~valid).tolist();lostpass=np.flatnonzero(protected_pass&~passed).tolist();reasons=[]
  if not full['in_domain']:reasons.append('original_domain_failed')
  if not rv['tip_pass']:reasons.append('exact_original_tip_or_mean_failed')
  if lostvalid:reasons.append('previously_valid_orientation_row_invalidated')
  if lostpass:reasons.append('previously_passed_orientation_row_failed')
  if not tuple(rv['rank'])<ref_rank:reasons.append('no_strict_original_rank_improvement')
  return dict(accepted_for_limited_coarse_audit=not reasons,refusal_reasons=reasons,lost_orientation_valid_rows=lostvalid,lost_orientation_pass_rows=lostpass,exact_rank=rv['rank'],reference_rank=list(ref_rank),scientific_mapping_acceptance=False)
 def tips(value):
  errors=np.asarray(value['error_vectors_m']);check(errors.shape==(5,3) and np.isfinite(errors).all(),'finite original five-tip vectors')
  norms=norm(errors);mean=float(norms.mean());tip=value['tip_status']
  same(norms,tip['norms_m'],'all recorded exact tip norms');same(norms<=caps,tip['individual_pass'],'all recorded old cap truth');same(mean,tip['mean_m'],'all recorded mean arithmetic');same(mean<=.01,tip['mean_pass'],'all recorded inclusive mean gate')
  passed=bool(np.all(norms<=caps) and mean<=.01);check(passed==tip['passed'],'all recorded full tip truth');return passed
 for result in row['restoration_endpoint_results']:
  spec=result['source_endpoint'];start=spec['start_index'];proposal=np.asarray(spec['state_scaled']);path=Path(result['trace_path']);check(sha(path)==result['trace_sha256'],'projection lossless trace binding')
  frozen=FROZEN_PROTOCOL['cases'][str(index)]['endpoints'][start];check(spec==frozen,'full source endpoint binding')
  trust=np.column_stack((np.maximum(bounds[:,0],proposal-.25),np.minimum(bounds[:,1],proposal+.25)));same(trust,result['trust_bounds_scaled'],'exact frozen trust intersection')
  byevent={};bystate={};eligible=[];fresh_event=None;counts={};last=0;terminal=None
  for line in gzip.open(path,'rt'):
   ev=json.loads(line);seq=ev['event'];kind=ev['kind'];check(seq==last+1,'sequential projection trace');last=seq;counts[kind]=counts.get(kind,0)+1
   if kind=='FK_callback':
    full=ev['value'];x=np.asarray(full['state_scaled']);key=x.astype(np.float64).tobytes();check(x.shape==(27,) and np.isfinite(x).all(),'complete finite27 FK state');check(hashlib.sha256(key).hexdigest()==ev['state_sha256'],'complete27 FK checksum')
    check(np.all(x>=trust[:,0]) and np.all(x<=trust[:,1]),'every actual FK state inside trust');tips(full);byevent[seq]=full;bystate[key]=full
   elif kind=='FK_cache_hit':
    source=byevent[ev['source_event']];same(source['state_scaled'],ev['state_scaled'],'cache identity complete27')
   elif kind=='projection_checkpoint':
    x=np.asarray(ev['state_scaled']);full=bystate[x.astype(np.float64).tobytes()];check(ev['tip_status']==full['tip_status'] and ev['in_domain']==full['in_domain'],'checkpoint exact FK binding')
    reason=ev['reason'];check(reason in ('frozen_endpoint','optimizer_terminal') or reason.startswith('major_'),'only authorized checkpoint sources')
    if full['tip_status']['passed'] and full['in_domain']:
     eligible.append(dict(source_event=seq,reason=reason,state_scaled=x.tolist(),distance_squared_scaled=float(np.dot(x-proposal,x-proposal))))
   elif kind=='analytic_jacobian':
    matrix=np.asarray(ev['matrix']);check(matrix.shape==(6,27) and np.isfinite(matrix).all(),'finite six-row original-tip Jacobian');near(matrix[-1],matrix[:5].mean(0),2e-13,'mean Jacobian from same five norm rows')
   elif kind=='optimizer_terminal':terminal={k:v for k,v in ev.items() if k not in ('event','kind')}
   elif kind=='selected_fresh_FK':
    fresh_event=ev;check(ev['value']==byevent[max(byevent)],'fresh selection binds latest actual callback');check(tips(ev['value']),'fresh selected caps')
  same(eligible,result['eligible_checkpoints'],'all and only eligible projection checkpoints')
  expected=min(eligible,key=lambda z:z['distance_squared_scaled']) if eligible else None
  check(result['selected']==expected,'stable nearest-observed feasible checkpoint')
  check(fresh_event is not None and result['fresh_FK']==fresh_event['value'],'final fresh projection source')
  check(result['selected']==fresh_event['source_checkpoint'],'fresh provenance matches chosen checkpoint')
  same(result['selected']['state_scaled'],result['fresh_FK']['state_scaled'],'chosen and fresh state')
  check(terminal==result['optimizer'],'complete optimizer result provenance')
  ct=result['counts'];check(counts.get('FK_callback',0)==ct['FK_executions']<=400,'tip FK executions bounded');check(counts.get('analytic_jacobian',0)==ct['analytic_jacobian_executions']<=60,'analytic staging budget')
  check(counts.get('FK_cache_hit',0)==ct['FK_cache_hits'],'cache count');check(counts.get('projection_checkpoint',0)==ct['selection_checkpoint_count'],'checkpoint count')
  check(result['actual_projection_mj_forward_calls']==2*(ct['FK_executions']+ct['analytic_jacobian_executions'])<=920,'all projection forwards bounded and disclosed')
  check(result['optimizer']['nit']<=40 and sum(ev['reason'].startswith('major_') for ev in eligible)<=40,'bounded iteration count')
  check(result['surface_geometry_calls_during_projection']==0 and result['scientific_mapping_acceptance'] is False,'FK-only projection scope')
  check(last==result['final_event_count'],'complete projection event count')
  x=np.asarray(result['selected']['state_scaled']);full=result['exact_surface_check'];same(x,full['state_scaled'],'only selected projection receives exact retained check')
  g=gate(full);check(g==result['improvement_gate'],'independent exact improvement and protected-orientation gate')
  fk=INDEPENDENT_STATE_FK(x);near(fk['errors'][:5],result['fresh_FK']['error_vectors_m'],0,'independent projected native-world error vectors');near(fk['errors'],full['task_error_vectors_m'],0,'all ten independent projected task vectors')
  actualnorm=norm(fk['errors'][:5]);check(np.all(actualnorm<=caps) and actualnorm.mean()<=.01,'independent actual projected historical caps and mean');check(fk['source_scene_error_m']<1.1e-7,'independent projected source URDF / captured scene FK')
  if g['accepted_for_limited_coarse_audit']:coarse_eligible.append((tuple(g['exact_rank']),start,x))
  endpoint_reports.append(dict(start_index=start,trace_path=str(path),trace_sha256=sha(path),counts=ct,projection_mj_forward_calls=result['actual_projection_mj_forward_calls'],selected_checkpoint=result['selected']['source_event'],max_original_cap_excess_m=float(np.max(actualnorm-caps)),mean_tip_error_m=float(actualnorm.mean()),trust_max_abs_step_scaled=float(np.max(np.abs(x-proposal))),source_scene_error_m=fk['source_scene_error_m'],gate=g))
 check([r['start_index'] for r in endpoint_reports]==[0,1],'exactly two prior fixed endpoints')
 chosen=min(coarse_eligible,key=lambda z:(z[0],z[1])) if coarse_eligible else None
 check(row['restoration_selected_endpoint']==(None if chosen is None else chosen[1]),'fixed across-endpoint selection')
 check(row['eligible_for_limited_coarse_audit']==bool(chosen),'coarse audit eligibility')
 final=r.verify(row['selected_constraint_diagnostics'],exact=True)
 if chosen is None:
  same(row['selected_state_scaled'],row['reference_state_scaled'],'explicit pooled fallback');check(row['fresh_improvement_gate'] is None and row['fresh_improvement_refusal'] is None,'no eligible attempt hidden');check(tuple(row['selected_constraint_diagnostics']['rank'])==ref_rank,'retained reference rank')
 else:
  same(row['selected_state_scaled'],chosen[2],'selected qualifying endpoint');check(row['fresh_improvement_gate']==gate(row['selected_constraint_diagnostics']),'final fresh improvement gate')
 check(row['exact_surface_checks']==2+sum(r['selected'] is not None for r in row['restoration_endpoint_results'])<=5,'reference plus endpoint plus final exact-check count')
 check(row['numba_threads_after_imports']==2 and row['diagnostic_pose_not_promoted'],'execution thread mask and no promotion')
 return dict(arithmetic_checks=r.checks,endpoint_reviews=endpoint_reports,coarse_eligible_count=len(coarse_eligible),rank=row['selected_constraint_diagnostics']['rank'],all_witness_rows_pass=final['all_witness_rows_pass'])


def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--expected-cases',type=int,nargs='+',default=[8,12,16,20,24,28]);p.add_argument('--output',type=Path,default=OUT/'final_output_review.json');args=p.parse_args();start=time.perf_counter()
 candidate=read(args.candidate);primary_path=Path('/private/tmp/hug_feasibility_restoration_2026-10-15_v1/protocol.json');primary=read(primary_path)
 global FROZEN_PROTOCOL;FROZEN_PROTOCOL=primary
 savedpath=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json');saved=read(savedpath)
 oppath=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json';op=read(oppath)
 invpath=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json';inv=read(invpath)['modern_cases']
 oldpath=ROOT/'research_artifacts/hug_canonical_mcp_calibration_2026-10-14_v1/runtime_v3/candidate_targets.json';old={r['source_inventory_index']:r for r in read(oldpath)['cases']}
 urdf=ROOT/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf';xml=ET.parse(urdf).getroot()
 ranges={j.get('name'):np.array([float(j.find('limit').get(k)) for k in ('lower','upper')]) for j in xml.findall('joint') if j.find('limit') is not None}
 bindings={str(path):sha(path) for path in (args.candidate,primary_path,savedpath,oppath,invpath,oldpath,urdf,OLDREVIEW,Path(__file__),OUT/'constraint_arithmetic.py')}
 check(candidate['protocol_sha256']==sha(primary_path),'candidate frozen primary protocol')
 check(candidate['new_eval_data'] is False and candidate['new_inference'] is False and candidate['physics_steps']==0,'static-only output')
 check(sorted(r['source_inventory_index'] for r in candidate['cases'])==sorted(args.expected_cases),'explicit expected original cases')
 for group in ('input_sha256','implementation_sha256','preflight_sha256'):
  for path,h in primary[group].items():check(sha(path)==h,'frozen binding '+path)
 source=ShadowPoseValidator(urdf.parent.parent);spalm=ident(source.model,mujoco.mjtObj.mjOBJ_BODY,'palm');reaches=v.independent_reach(xml)
 records=[];arithmetic_checks=0
 for row in candidate['cases']:
  index=row['source_inventory_index'];case=inv[index]
  check(row['native_sha256']==case['native']['sha256']==old[index]['native_sha256'],'native identity')
  with np.load(case['stored_shadow_command']['path'],allow_pickle=False) as a:
   n=case['stored_shadow_command']['row_index'];names=a['joint_names'].tolist();q0=a['nominal_target_rad'][n];root0=a['root_target_position_m'][n];quat0=a['root_target_quaternion_wxyz'][n]
  check(names==row['joint_names'],'source joint order');same(q0,row['original_qpos_rad'],'originalq preserved');same(root0,row['original_root_target_position_m'],'originalroot preserved');same(quat0,row['root_target_quaternion_wxyz'],'original root quaternion preserved')
  native=pickle.loads(Path(case['native']['path']).read_bytes());T=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);WM=T@np.asarray(native['T_camera_wrist'])
  targets=(np.asarray(native['landmarks_3d'])@T[:3,:3].T+T[:3,3])[[4,8,12,16,20,2,6,10,14,18]];Rop=WM[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
  m=mujoco.MjModel.from_binary_path(case['scene_mjb']['path']);d=mujoco.MjData(m)
  with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as a:d.qpos[:]=a['qpos']
  ja=np.array([m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names]);ra=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);oa=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'object_free')]);palm=ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');obj=ident(m,mujoco.mjtObj.mjOBJ_BODY,'object')
  objq=d.qpos[oa:oa+7].copy();d.qpos[ja]=q0;d.qpos[ra:ra+3]=root0;d.qpos[ra+3:ra+7]=quat0;d.qvel[:]=0
  if m.nmocap:d.mocap_pos[0]=root0;d.mocap_quat[0]=quat0
  mujoco.mj_forward(m,d);opos=d.xpos[obj].copy();oR=d.xmat[obj].copy();baseline=v.task_points(m,d,xml)-targets;caps=norm(baseline[:5])
  same(caps,old[index]['old_original_five_tip_caps_sqrt_einsum_m'],'historical caps exact');same(caps,row['authoritative_original_tip_caps_sqrt_einsum_m'],'no rebase');near(baseline,row['baseline_constraint_diagnostics']['task_error_vectors_m'],0,'baseline source error vectors')
  x=np.asarray(row['selected_state_scaled']);q=q0+.025*x[:24];shift=.005*x[24:];anchor=WM[:3,3]+shift
  same(q,row['qpos_rad'],'decode joints');same(shift,row['palm_displacement_world_m'],'decode world palm shift')
  physical=np.array([ranges[n] for n in names]);check(np.all(q>=physical[:,0]) and np.all(q<=physical[:,1]) and np.all(q>=q0-.35) and np.all(q<=q0+.35),'literal physical andoriginal joint bounds')
  bounds=np.asarray(row['bounds_scaled']);check(np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]),'all27 state bounds')
  rad=reaches+1e-6+caps;lo=np.max(targets[:5]-rad[:,None],axis=0);hi=np.min(targets[:5]+rad[:,None],axis=0)
  near(row['domain']['outer_ball_radii_m'],rad,1e-15,'reach balls');near(row['domain']['palm_world_AABB_m'],np.column_stack([lo,hi]),1e-15,'necessary outerbox');check(np.all(anchor>=lo) and np.all(anchor<=hi),'palm outerbox')
  d.qpos[ja]=q;d.qpos[ra:ra+3]=0;mujoco.mj_forward(m,d);expected_root=anchor-d.xpos[palm];near(row['root_target_position_m'],expected_root,1e-12,'independent q-dependent root compensation')
  d.qpos[ra:ra+3]=row['root_target_position_m']
  if m.nmocap:d.mocap_pos[0]=row['root_target_position_m'];d.mocap_quat[0]=quat0
  mujoco.mj_forward(m,d);points=v.task_points(m,d,xml);errors=points-targets;norms=norm(errors[:5])
  check(np.all(norms<=caps) and norms.mean()<=.010,'all exact oldcaps and original10mmmean');same(norms,row['selected_tip_norms_sqrt_einsum_m'],'selected exact tipnorms');near(errors,row['selected_constraint_diagnostics']['task_error_vectors_m'],0,'selected fresh task vectors')
  near(d.xpos[palm],anchor,1e-12,'actual translated palm');near(d.xpos[palm],row['target_palm_world_position_m'],0,'reported actual palm');near(d.xmat[palm].reshape(3,3),row['target_palm_world_rotation'],0,'reported actual palm rotation')
  qmap=dict(zip(names,q));source.data.qpos[:]=[qmap[n] for n in source.joint_names];source.data.qvel[:]=0;mujoco.mj_forward(source.model,source.data)
  expected=(v.task_points(source.model,source.data,xml)-source.data.xpos[spalm])@Rop.T+anchor;fkerr=float(np.linalg.norm(points-expected,axis=1).max());check(fkerr<1.1e-7,'independent URDF/scene FK')
  same(d.qpos[oa:oa+7],objq,'objectq unchanged');same(d.xpos[obj],opos,'objectpos unchanged');same(d.xmat[obj],oR,'objectrotation unchanged');same(d.qpos[ra+3:ra+7],quat0,'fixedrootquat');check(d.time==0,'no physical time')
  same(row['constraint_source_rows']['proximity'],[r for r in saved['whole_cell_rows'] if r['source_inventory_index']==index],'all fixed proximity rows')
  check(row['constraint_source_rows']['orientation']==[r for r in saved['orientation_rows'] if r['source_inventory_index']==index],'all fixed signed orientation rows')
  def state_fk(x):
   sx=np.asarray(x);sq=q0+.025*sx[:24];sanchor=WM[:3,3]+.005*sx[24:]
   check(np.all(sq>=physical[:,0]) and np.all(sq<=physical[:,1]) and np.all(sq>=q0-.35) and np.all(sq<=q0+.35),'projected physical/correction limits')
   d.qpos[ja]=sq;d.qpos[ra:ra+3]=root0;mujoco.mj_forward(m,d);d.qpos[ra:ra+3]=root0+(sanchor-d.xpos[palm])
   if m.nmocap:d.mocap_pos[0]=d.qpos[ra:ra+3];d.mocap_quat[0]=quat0
   mujoco.mj_forward(m,d);pts=v.task_points(m,d,xml);qmap=dict(zip(names,sq));source.data.qpos[:]=[qmap[n] for n in source.joint_names];mujoco.mj_forward(source.model,source.data)
   expect=(v.task_points(source.model,source.data,xml)-source.data.xpos[spalm])@Rop.T+sanchor
   same(d.qpos[oa:oa+7],objq,'projected object unchanged');same(d.qpos[ra+3:ra+7],quat0,'projected root quaternion unchanged');check(d.time==0,'projected no physical time')
   return {'errors':pts-targets,'source_scene_error_m':float(np.linalg.norm(pts-expect,axis=1).max())}
  global INDEPENDENT_STATE_FK;INDEPENDENT_STATE_FK=state_fk
  tr=trace_review(row,saved,op);arithmetic_checks+=tr['arithmetic_checks']
  records.append({'source_inventory_index':index,'palm_shift_world_m':shift.tolist(),'palm_shift_norm_m':float(np.linalg.norm(shift)),'max_exact_original_cap_excess_m':float(np.max(norms-caps)),'mean_tip_error_m':float(norms.mean()),'source_scene_FK_maxerror_m':fkerr,'exact_original_fallback':bool(np.array_equal(x,np.zeros(27))),'trace_review':tr})
  print(json.dumps({'reviewed_index':index,'checks':v.COUNT['checks'],'arithmetic_checks':arithmetic_checks}),flush=True)
 for path,h in bindings.items():check(sha(path)==h,'inputs unchanged '+path)
 check(v.COUNT['physics_actual']==0,'no integration')
 report={'schema':'hug_saved_endpoint_restoration_independent_output_review_v1','passed':True,'source_and_trace_checks':v.COUNT['checks'],'constraint_arithmetic_checks':arithmetic_checks,'source_bindings':bindings,'cases':records,'runtime_seconds':time.perf_counter()-start,'physics_steps':0,'new_eval_data':False,'new_inference':False,'new_surface_queries':0,'mapping_acceptance':False,'scope':'Independently reconstructed source/native targets, original caps, physical/domain bounds, translated root and source/scene FK; replayed all projection traces, exact retained-row arithmetic and endpoint selection. Independently reconstructed FK/caps for every restored endpoint and selected fallback. No scientific full-grid measurements.'}
 out=args.output;check(not out.exists(),'never overwrite review');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'report_sha256':sha(out)}))
if __name__=='__main__':main()
