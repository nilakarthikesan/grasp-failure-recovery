"""All32 existing native-frame + literal nominal-q static assembled-scene bridge.

No inference, retarget solve, controller advance, integration or outcome labels.
"""
from pathlib import Path
import hashlib,json,pickle,time,sys,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from xml_reference import read_urdf,forward_xml,source_limit_rows,obj_vertices,transform,quaternion_matrix_wxyz
from bridge_model import assign_named,compare_body_frames,compare_collisions,joint_metadata_against_source,body_world
from object_reference import read_object
from native_reference import expected_native_palm
from scene_runtime import build_current_runner,reset_captured_scene,stage_named_palm
from scene_checks import object_check,controller_and_table_check
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def main():
 pp=HERE/'protocol.json';p=json.loads(pp.read_text());ph=sha(pp);a=json.loads((HERE/'authorization.json').read_text());assert p['status']=='FROZEN_AUTHORIZED_STATIC_ONLY' and a['authorized'] is True and a['protocol_sha256']==ph
 def verify():
  assert sha(pp)==ph
  for path,h in p['bindings'].items():assert sha(path)==h,path
 verify();assert not (HERE/'results.json').exists() and not (HERE/'cases').exists()
 assert np.__version__==p['runtime']['numpy_version'] and mujoco.__version__==p['runtime']['mujoco_version']
 physics=[0];forwards=[0];forbidden_calls=[0]
 def nostep(*args,**kwargs):physics[0]+=1;raise RuntimeError('No physics integration authorized')
 def noexecution(*args,**kwargs):forbidden_calls[0]+=1;raise RuntimeError('No inference/retargeting/controller execution authorized')
 original=mujoco.mj_forward
 def forward(*args,**kwargs):forwards[0]+=1;return original(*args,**kwargs)
 for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,nostep)
 mujoco.mj_forward=forward
 from grasp_failure_prediction.evaluation import case_runner,runner as rm,retargeting as ret,pose_validation as pv
 from grasp_failure_prediction.integrations.hug import load_hug_prediction
 for cls in (rm.AdroitShadowRunner,rm.ActuatedShadowRunner):
  cls.execute=noexecution;cls._advance=noexecution
 ret.ShadowHandRetargeter.retarget=noexecution;case_runner._run_case_inference=noexecution;case_runner.run_case=noexecution
 assert Path(pv.__file__).resolve()==Path(p['current_pose_validation_source'])
 source=read_urdf(p['source_urdf']);urdfroot=Path(p['source_urdf']).parent.parent
 source_meshes={(c['path'],tuple(c['scale'])):obj_vertices(c['path'],c['scale']) for c in source['collisions'] if c['kind']=='mesh'}
 efforts={j.get('name'):float(j.find('limit').get('effort')) for j in ET.parse(p['source_urdf']).getroot().findall('joint') if j.get('type')=='revolute'}
 inventory=json.loads(Path(p['inventory_path']).read_text())['modern_cases'];tol=p['engineering_tolerances'];runners={};records=[];validator=pv.ShadowPoseValidator(urdfroot);start=time.perf_counter();(HERE/'cases').mkdir()
 for frozen in p['cases']:
  i=frozen['source_inventory_index'];iv=inventory[i];before=forwards[0]
  for path,h in frozen['input_bindings'].items():assert sha(path)==h,path
  resolved_saved=json.loads(Path(iv['resolved_case_path']).read_text());payload=resolved_saved['case'];key=iv['geometry_content_hash']
  if key not in runners:
   rr,resolved,geometry=build_current_runner(payload,p['runtime_asset_root'],urdfroot)
   ref=read_object(Path(p['runtime_asset_root'])/payload['object']['geometry']['mjcf_path']) if payload['object']['geometry'] is not None else None
   runners[key]=(rr,resolved,geometry,ref)
  rr,unusedresolved,geometry,ref=runners[key]
  case=case_runner.EvaluationCase.model_validate(payload);resolved=case_runner.resolve_case(case)
  assert resolved.environment.runner=='grasp_failure_prediction.evaluation.runner:ActuatedShadowRunner'
  assert resolved.environment_config_hash==resolved_saved['environment_config_hash'] and resolved.execution_protocol_config_hash==resolved_saved['execution_protocol_config_hash']
  native=Path(iv['native']['path']);binding=case_runner._source_scene_binding(case,resolved,Path(p['runtime_asset_root']),native,geometry);assert binding['scene_contract_status']=='verified'
  contract=binding['scene_contract'];assert case_runner._geometry_content_hash(geometry)==key==contract['geometry_content_hash']
  op,oq=reset_captured_scene(rr,payload,contract)
  with np.load(frozen['command_path'],allow_pickle=False) as z:
   row=frozen['row_index'];names=z['joint_names'].tolist();q=z['nominal_target_rad'][row].copy();oldp=z['root_target_position_m'][row].copy();oldq=z['root_target_quaternion_wxyz'][row].copy()
  assert names==frozen['joint_names'] and np.array_equal(q,frozen['qpos_rad']) and np.array_equal(oldp,frozen['root_position_m']) and np.array_equal(oldq,frozen['root_quaternion_wxyz'])
  grasp=load_hug_prediction(native)
  # Independently read the immutable native numeric wrist; no generator imported.
  with native.open('rb') as handle:raw=pickle.load(handle)
  wrist=np.asarray(raw['T_camera_wrist'],float).reshape(4,4);assert np.array_equal(grasp.T_camera_wrist,wrist)
  camera=np.load(native.parent/'T_world_camera.npy',allow_pickle=False)
  fixed=forward_xml(source,names,q);expected=expected_native_palm(camera,wrist,fixed['palm'])
  # Context arrays are unused by this static setter. Only saved names/q and
  # actual native transform are supplied; there is no retarget solve.
  pose=ret.RetargetedHandPose(tuple(names),q,np.zeros((21,3)),np.zeros((10,3)),wrist)
  ppalm,qpalm=case_runner._hug_world_palm_pose(grasp,pose,validator,native.parent)
  actualpalm=transform(ppalm,quaternion_matrix_wxyz(qpalm));palm_pe=float(np.linalg.norm(ppalm-expected['palm_world'][:3,3]));palm_re=float(np.max(np.abs(actualpalm[:3,:3]-expected['palm_world'][:3,:3])))
  rp,rq,target=stage_named_palm(rr,names,q,actualpalm)
  # Independent source-tree inversion about the forearm, including fixed source
  # root origin, avoids comparing a method solely against its own output.
  local=np.linalg.inv(fixed['forearm'])@fixed['palm'];wantedR=expected['palm_world'][:3,:3]@local[:3,:3].T;wantedp=expected['palm_world'][:3,3]-wantedR@local[:3,3];wantedroot=transform(wantedp,wantedR);world=forward_xml(source,names,q,wantedroot)
  beforeq=rr.data.qpos.copy();jointrows=assign_named(rr.model,rr.data,names,q);assert np.array_equal(beforeq,rr.data.qpos)
  jm=joint_metadata_against_source(source,jointrows,tol['source_compiled_joint_limit_rad']);limits=source_limit_rows(source,names,q);named_exact=all(rr.data.qpos[r['qpos_address']]==v for r,v in zip(jointrows,q,strict=True))
  bodies=compare_body_frames(source,world,rr.model,rr.data,tol['current_position_m'],tol['current_rotation_matrix']);hand=compare_collisions(source,world,rr.model,rr.data,source_meshes,tol,model_is_hand_only=False)
  obj=object_check(rr.model,rr.data,ref,op,oq,float(contract['mass_kg']),tol);control=controller_and_table_check(rr,source,names,q,rp,rq,efforts)
  currentroot=body_world(rr.model,rr.data,'forearm');currentpalm=body_world(rr.model,rr.data,'palm');rootpe=float(np.linalg.norm(currentroot[:3,3]-wantedp));rootre=float(np.max(np.abs(currentroot[:3,:3]-wantedR)));stagedpe=float(np.linalg.norm(currentpalm[:3,3]-expected['palm_world'][:3,3]));stagedre=float(np.max(np.abs(currentpalm[:3,:3]-expected['palm_world'][:3,:3])))
  input_exact=bool(named_exact and np.array_equal(rr.data.ctrl[rr._actuator_ids],target) and set(rr._hand_joint_names)==set(names));limitsok=all(r['literal_pass'] for r in limits)
  configured=bool(rr.model.opt.timestep==resolved.environment.simulator.physics_timestep_s and rr.physics_timestep_s==resolved.environment.simulator.physics_timestep_s and rr.control_timestep_s==resolved.environment.simulator.control_timestep_s and rr.model.opt.cone==mujoco.mjtCone.mjCONE_ELLIPTIC and rr.model.opt.noslip_iterations==10 and rr.model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
  passed=bool(input_exact and bodies['passed'] and hand['passed'] and obj['passed'] and control['passed'] and configured and all(r['source_compiled_limits_match'] for r in jm) and max(palm_pe,rootpe,stagedpe)<=tol['current_position_m'] and max(palm_re,rootre,stagedre)<=tol['current_rotation_matrix'])
  assert rr.data.time==validator.data.time==0 and rr._trace==[] and rr._step_index==0 and forwards[0]-before==4
  record={'schema':'hug_current_assembled_scene_bridge_case_v2','source_inventory_index':i,'protocol_sha256':ph,'native_sha256':iv['native']['sha256'],'literal_command':frozen,
   'object_id':iv['object_id'],'geometry_content_hash':key,'source_scene_binding':binding,'current_runner_class':resolved.environment.runner,'current_environment_hash':resolved.environment_config_hash,'current_execution_protocol_hash':resolved.execution_protocol_config_hash,
   'native_frame_reference':expected,'actual_helper_palm_world':actualpalm,'helper_vs_independent_position_error_m':palm_pe,'helper_vs_independent_rotation_matrix_max_error':palm_re,
   'independent_root_world':wantedroot,'actual_root_position_m':rp,'actual_root_quaternion_wxyz':rq,'actual_world_palm':currentpalm,'source_root_position_error_m':rootpe,'source_root_rotation_matrix_error':rootre,'staged_palm_position_error_m':stagedpe,'staged_palm_rotation_matrix_error':stagedre,
   'current_root_change_from_historical_saved_m':float(np.linalg.norm(rp-oldp)),'current_root_rotation_change_from_historical_max_entry':float(np.max(np.abs(currentroot[:3,:3]-quaternion_matrix_wxyz(oldq)))),
   'literal_source_joint_limit_rows':limits,'all_literal_source_joint_limits_pass':limitsok,'current_joint_metadata':jm,'named_q_and_controller_inputs_exact':input_exact,'current_hand_body_virtual_link_checks':bodies,'current_hand_collision_checks':hand,'current_object_checks':obj,'current_controller_and_table_checks':control,'current_declared_timing_options_match':configured,
   'current_assembled_static_command_realization_pass':passed,'current_command_within_source_limits_and_realized':bool(passed and limitsok),'actual_static_forward_calls':forwards[0]-before,'physics_steps':0,'new_eval_data':False,'native_geometry_equivalence_validated':False,'servo_tracking_validated':False}
  dest=HERE/'cases'/f'case_{i:02d}.json';dest.write_text(json.dumps(plain(record),indent=2,allow_nan=False)+'\n');records.append({'index':i,'path':str(dest),'sha256':sha(dest),'passed':passed,'literal_source_limits_pass':limitsok,'object_pass':obj['passed'],'hand_pass':hand['passed'],'controls_pass':control['passed'],'helper_position_error_m':palm_pe,'helper_rotation_matrix_error':palm_re,'source_root_position_error_m':rootpe,'historical_root_position_change_m':record['current_root_change_from_historical_saved_m']});print(json.dumps(records[-1]),flush=True)
 verify();assert len(runners)==4 and len(records)==32 and forwards[0]==128 and physics[0]==forbidden_calls[0]==0
 result={'schema':'hug_current_assembled_scene_bridge_summary_v2','protocol_sha256':ph,'scope':p['scope'],'cases':records,'case_count':32,'registered_runner_instances':len(runners),'all_current_assembled_static_command_realizations_pass':all(r['passed'] for r in records),'all_literal_source_limits_pass':all(r['literal_source_limits_pass'] for r in records),'actual_static_forward_calls':forwards[0],'physics_steps':0,'forbidden_execution_calls':0,'new_eval_data':False,'new_retargeting_solves':0,'new_native_proposals':0,'source_bindings_unchanged':True,'runtime_seconds':time.perf_counter()-start,'numpy_version':np.__version__,'mujoco_version':mujoco.__version__,'native_geometry_equivalence_validated':False,'servo_tracking_validated':False,'historical_surface_failures_reclassified':False}
 (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'summary':str(HERE/'results.json'),'sha256':sha(HERE/'results.json'),'passed':result['all_current_assembled_static_command_realizations_pass']}))
if __name__=='__main__':main()
