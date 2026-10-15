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
 index=row['source_inventory_index'];audit=row['solver_audit'];path=Path(audit['trace_path'])
 check(sha(path)==audit['trace_sha256'],'lossless trace binding')
 r=Replay(saved['whole_cell_rows'],saved['orientation_rows'],index,op,row['authoritative_original_tip_caps_sqrt_einsum_m'])
 raw_events={};exact_events={};counters={};last_event=None;best=None;fresh=None;refused=False
 bounds=np.asarray(row['bounds_scaled']);q0=np.asarray(row['original_qpos_rad'])
 for line in gzip.open(path,'rt'):
  event=json.loads(line);seq=event['event'];kind=event['kind'];counters[kind]=counters.get(kind,0)+1
  check(seq==(0 if last_event is None else last_event)+1,'sequential trace event');last_event=seq
  if kind=='raw_callback':
   full=event['value'];r.verify(full);key=np.asarray(full['state_scaled'],np.float64).tobytes()
   check(hashlib.sha256(key).hexdigest()==event['state_sha256'],'raw complete27 checksum');raw_events[seq]=key
  elif kind=='raw_cache_hit':
   key=np.asarray(event['state_scaled'],np.float64).tobytes();check(event['source_event'] in raw_events,'cache references raw event')
   check(key==raw_events[event['source_event']] and hashlib.sha256(key).hexdigest()==event['state_sha256'],'cache full27 equality')
  elif kind in ('exact_checkpoint','selected_fresh_terminal','fresh_original_fallback'):
   full=event['value'];rv=r.verify(full,exact=True);x=np.asarray(full['state_scaled']);q=q0+.025*x[:24]
   check(rv['tip_pass'] and full['in_domain'] and np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]) and np.all(q>=q0-.35) and np.all(q<=q0+.35),'every selectable exact state originaltips/domain')
   if kind=='exact_checkpoint':
    reason=event['reason'];check(reason=='original_baseline' or reason in ('start_0','start_1','start_0_terminal','start_1_terminal') or reason.startswith(('start_0_major_','start_1_major_')),'checkpoint is authorized source/major/terminal')
    exact_events[seq]=(tuple(rv['rank']),x,rv)
    if best is None or tuple(rv['rank'])<best[0]:best=(tuple(rv['rank']),seq,x)
   else:fresh=event;fresh_review=rv
  elif kind=='fresh_terminal_refusal':refused=True
 counts=audit['counts'];check(last_event==counts['trace_events'],'all trace events accounted')
 expected_exact=counters.get('exact_checkpoint',0)+counters.get('selected_fresh_terminal',0)+counters.get('fresh_original_fallback',0)
 check(expected_exact==counts['exact_checkpoint_executions']<=90,'exact checkpoint budget and executions')
 check(counters.get('raw_callback',0)+expected_exact==counts['raw_callback_executions']<=3000,'raw budget includes raw work in every exact call')
 check(counters.get('constraint_jacobian',0)==counts['analytic_tip_jacobian_executions'],'separate analyticJac executions')
 check(counters.get('raw_cache_hit',0)==counts['raw_cache_hits'],'cache-hit count')
 check(len(exact_events)==counts['selection_eligible_exact_checks'],'exact eligible count')
 check(fresh is not None and fresh['event']==audit['actual_returned_fresh_event'],'actual final provenance')
 check(fresh['value']==row['selected_constraint_diagnostics'],'selected diagnostics bind fresh exact event')
 source=audit['selected_source_exact_checkpoint']
 if refused:
  check(fresh['kind']=='fresh_original_fallback' and source['event']==fresh['event'] and source['reason']=='fresh_original_fallback','terminal fallback provenance repaired')
  same(row['selected_state_scaled'],np.zeros(27),'fallback exact zero state');check(audit['refused_preterminal_selection'] is not None,'refused selected source retained')
 else:
  check(source['event']==best[1],'stable earliest lexicographic best exact state');same(source['state_scaled'],best[2],'selected minimum state')
  check(fresh['selected_source_event']==best[1] and audit['refused_preterminal_selection'] is None,'fresh check original chosen provenance')
 same(fresh['value']['state_scaled'],row['selected_state_scaled'],'returned state matches fresh trace')
 check(len(audit['optimizer_starts'])<=2 and all(s.get('major_iterations_observed',0)<=40 for s in audit['optimizer_starts']),'finite two-start search')
 check(row['legacy_guard_used_for_selection'] is False and row['diagnostic_pose_not_promoted'],'no legacy guard substitution or promotion')
 return {'trace_sha256':sha(path),'arithmetic_checks':r.checks,'trace_event_counts':counters,'selected_event':source['event'],'selected_reason':source['reason'],'fresh_terminal_fallback':refused,'rank':fresh['value']['rank'],**fresh_review}

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--expected-cases',type=int,nargs='+',default=[8,12,16,20,24,28]);p.add_argument('--output',type=Path,default=OUT/'final_output_review.json');args=p.parse_args();start=time.perf_counter()
 candidate=read(args.candidate);primary_path=Path('/private/tmp/hug_surface_constraints_2026-10-15_v2/protocol.json');primary=read(primary_path)
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
  tr=trace_review(row,saved,op);arithmetic_checks+=tr['arithmetic_checks']
  records.append({'source_inventory_index':index,'palm_shift_world_m':shift.tolist(),'palm_shift_norm_m':float(np.linalg.norm(shift)),'max_exact_original_cap_excess_m':float(np.max(norms-caps)),'mean_tip_error_m':float(norms.mean()),'source_scene_FK_maxerror_m':fkerr,'exact_original_fallback':bool(np.array_equal(x,np.zeros(27))),'trace_review':tr})
  print(json.dumps({'reviewed_index':index,'checks':v.COUNT['checks'],'arithmetic_checks':arithmetic_checks}),flush=True)
 for path,h in bindings.items():check(sha(path)==h,'inputs unchanged '+path)
 check(v.COUNT['physics_actual']==0,'no integration')
 report={'schema':'hug_surface_constraints_independent_final_output_review_v2','passed':True,'source_and_trace_checks':v.COUNT['checks'],'constraint_arithmetic_checks':arithmetic_checks,'source_bindings':bindings,'cases':records,'runtime_seconds':time.perf_counter()-start,'physics_steps':0,'new_eval_data':False,'new_inference':False,'new_surface_queries':0,'mapping_acceptance':False,'scope':'Independently reconstructed source/native targets, original caps, physical/domain bounds, translated root and source/scene FK; replayed every raw/exact recorded constraint and exact checkpoint selection. Full scientific surface grid audit remains separate.'}
 out=args.output;check(not out.exists(),'never overwrite review');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'report_sha256':sha(out)}))
if __name__=='__main__':main()
