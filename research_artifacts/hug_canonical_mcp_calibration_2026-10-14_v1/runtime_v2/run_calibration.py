"""Versioned source-only canonical anatomy calibration on six saved HUG cases.
No HUG generator/checkpoint, object-contact objective, physics, or eval data.
"""
from pathlib import Path
import hashlib,importlib.metadata,json,os,pickle,sys,types,xml.etree.ElementTree as ET
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from calibration_math import calibrate,calibrated_vectors,rigid_inverse
B=Path(__file__).parent
STEPS=0
def forbidden(*args,**kwargs):
 global STEPS
 STEPS+=1;raise RuntimeError('Static canonical calibration forbids physics integration')
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden)
from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter,reorder_for_mujoco
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator,_link_positions,_body_id

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write_new(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def norms(e):return np.sqrt(np.einsum('ij,ij->i',e,e))
def transform(R,t):
 T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;return T

def canonical(settings):
 # This function receives ONLY source model/URDF/settings, no case/object/witness data.
 package=types.ModuleType('hug');package.__path__=[str(Path(settings['HUG_root'])/'src')];sys.modules['hug']=package
 from hug.models.mano import MANO
 model=MANO(side='right',use_pca=False,flat_hand_mean=True).float().eval();layer=model.mano_layer
 assert layer.center_idx==0 and not layer.use_pca and layer.flat_hand_mean and layer.side=='right'
 with torch.no_grad():
  output=layer(torch.zeros((1,48),dtype=torch.float32),torch.zeros((1,10),dtype=torch.float32))
  J=output.joints[0].numpy().astype(float)
  raw=(layer.th_J_regressor@layer.th_v_template)[0].numpy().astype(float)
 assert J.shape==(21,3) and np.array_equal(J[0],np.zeros(3))
 assert np.max(np.abs(J[[5,9,13]]-(raw[[1,4,10]]-raw[0])))<settings['neutral_joint_order_agreement_m']
 root=ET.parse(settings['urdf']).getroot();mcp=[]
 for name in settings['Shadow_MCP_joint_names']:
  j=next(x for x in root.findall('joint') if x.get('name')==name)
  assert j.find('parent').get('link')=='palm'
  assert np.array_equal(np.fromstring(j.find('origin').get('rpy','0 0 0'),sep=' '),np.zeros(3))
  mcp.append(np.fromstring(j.find('origin').get('xyz'),sep=' '))
 P=np.asarray(mcp)
 C,FM,FP,dm,dp=calibrate(J[[5,9,13]],J[0],P,minimum_longitudinal_m=settings['minimum_longitudinal_m'],minimum_projected_lateral_m=settings['minimum_projected_lateral_m'])
 validator=ShadowPoseValidator(settings['urdf_root']);validator.data.qpos[:]=0;validator.data.qvel[:]=0;mujoco.mj_forward(validator.model,validator.data)
 palm=_body_id(validator.model,'palm')
 world_joint=next(x for x in root.findall('joint') if x.get('name')=='world_joint')
 assert world_joint.get('type')=='fixed' and world_joint.find('parent').get('link')=='world' and world_joint.find('child').get('link')=='forearm'
 world_origin=world_joint.find('origin');pF=np.fromstring(world_origin.get('xyz','0 0 0'),sep=' ');rpyF=np.fromstring(world_origin.get('rpy','0 0 0'),sep=' ');RF=Rotation.from_euler('xyz',rpyF).as_matrix()
 RP0=validator.data.xmat[palm].reshape(3,3).copy();pP0=validator.data.xpos[palm].copy();assert np.max(np.abs(RP0-RF))<settings['source_vs_compiled_MCP_agreement_m']
 Tchain=transform(RF,pF)
 for joint_name,parent_name,child_name in (('WRJ2','forearm','wrist'),('WRJ1','wrist','palm')):
  jj=next(x for x in root.findall('joint') if x.get('name')==joint_name);assert jj.find('parent').get('link')==parent_name and jj.find('child').get('link')==child_name
  origin=jj.find('origin');xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ');Tchain=Tchain@transform(Rotation.from_euler('xyz',rpy).as_matrix(),xyz)
 neutral_chain_error=float(np.max(np.abs(Tchain-transform(RP0,pP0))));assert neutral_chain_error<settings['source_vs_compiled_MCP_agreement_m']
 compiled=np.array([(validator.data.xpos[_body_id(validator.model,n)]-pP0)@RP0 for n in ('ffknuckle','mfknuckle','rfknuckle')])
 assert np.max(np.abs(compiled-P))<settings['source_vs_compiled_MCP_agreement_m']
 report={'schema':'hug_source_only_canonical_MCP_frame_v1','C_MANO_from_neutral_palm':C.tolist(),'F_MANO_from_MCP_frame':FM.tolist(),'F_palm_from_MCP_frame':FP.tolist(),'neutral_MANO21_joints_m':J.tolist(),'neutral_MANO_MCP_5_9_13_m':J[[5,9,13]].tolist(),'neutral_Shadow_MCP_FFJ4_MFJ4_RFJ4_palm_m':P.tolist(),'neutral_Shadow_compiled_MCP_palm_m':compiled.tolist(),'neutral_MANO_conditioning':dm,'neutral_Shadow_conditioning':dp,'R_B_from_neutral_palm':RP0.tolist(),'p_B_neutral_palm_m':pP0.tolist(),'T_B_from_forearm':transform(RF,pF).tolist(),'T_B_from_neutral_palm_from_source_joint_chain':Tchain.tolist(),'source_joint_chain_vs_compiled_neutral_palm_maxabs':neutral_chain_error,'pointwise_neutral_MCP_residual_vectors_MANO_m':(P@C[:3,:3].T+C[:3,3]-J[[5,9,13]]).tolist(),'settings':settings,'source_only_no_object_or_witness_inputs':True,'calibration_is_new_mapping_not_frame_bugfix':True,'new_neutral_template_not_generator_grasp':True}
 return C,RP0,RF,pF,validator,report

def main():
 assert not (B/'calibration.json').exists() and not (B/'candidate_targets.json').exists(),'No reruns/overwrites'
 p=read(B/'protocol.json');ph=sha(B/'protocol.json');assert p['execution_authorized_by_root_and_independent_review']
 for table in ('source_sha256','implementation_sha256'):
  for path,h in p[table].items():assert sha(path)==h,path
 torch.set_num_threads(1);torch.manual_seed(0)
 C,RP0,RF,pF,validator,cal=canonical(p['canonical_settings']);cal['protocol_sha256']=ph;write_new(B/'calibration.json',cal)
 # Calibration is now fixed and serialized BEFORE reading any case record/landmark.
 inventory=read(p['inventory_path'])['modern_cases'];old=read(p['old_caps_candidate_path']);oldmap={x['source_inventory_index']:x for x in old['cases']}
 urdf=Path(p['canonical_settings']['urdf']);physical={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
 rows=[];taskindices=np.asarray(p['task_human_indices']);expected_names=p['task_link_names']
 for index in p['case_indices']:
  case=inventory[index];prior=oldmap[index];assert case['native']['sha256']==prior['native_sha256']
  source=pickle.loads(Path(case['native']['path']).read_bytes());Tc=np.asarray(source['T_camera_wrist'],float);L=(np.asarray(source['landmarks_3d'])-Tc[:3,3])@Tc[:3,:3]
  Twc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);WM=Twc@Tc
  vp,vb=calibrated_vectors(L,C,RP0);ref=vb[taskindices]
  retargeter=ShadowHandRetargeter(p['canonical_settings']['urdf_root']);seq=retargeter._retargeting
  assert sha(retargeter.config_path)==p['dex_config_sha256']
  assert seq.filter is None and seq.optimizer.scaling==1.0
  assert seq.optimizer.target_link_human_indices.tolist()==[[0]*10,p['task_human_indices']]
  assert seq.optimizer.origin_link_names==['palm']*10 and seq.optimizer.task_link_names==expected_names
  names=list(retargeter.joint_names);assert names==prior['joint_names'];assert p['task_link_names'][:5]==['thtip','fftip','mftip','rftip','lftip'];limits=np.asarray([physical[n] for n in names]);history=[]
  for iteration in range(p['optimizer_repeats']):
   raw=np.asarray(seq.retarget(ref.astype(np.float32)),float);history.append({'repeat':iteration+1,'raw_qpos_rad':raw.tolist(),'nlopt_result_code':int(seq.optimizer.opt.last_optimize_result()),'nlopt_last_value':float(seq.optimizer.opt.last_optimum_value()) if np.isfinite(seq.optimizer.opt.last_optimum_value()) else None})
  rawfinite=raw.shape==(24,) and np.isfinite(raw).all();assert rawfinite
  overflow=np.maximum(np.maximum(limits[:,0]-raw,raw-limits[:,1]),0.);marginpass=bool(np.all(overflow<=p['existing_Dex_optimizer_bound_margin_rad']))
  q=np.clip(raw,limits[:,0],limits[:,1]);physicalpass=bool(np.all(q>=limits[:,0]) and np.all(q<=limits[:,1]))
  validator.data.qpos[:]=reorder_for_mujoco(names,q,validator.joint_names);validator.data.qvel[:]=0;mujoco.mj_forward(validator.model,validator.data)
  ids=validator.model;palm=_body_id(ids,'palm');Btask=_link_positions(ids,validator.data,urdf);Bpalm=validator.data.xpos[palm].copy();BactualR=validator.data.xmat[palm].reshape(3,3).copy()
  RWBraw=WM[:3,:3]@C[:3,:3]@RP0.T;RWB=Rotation.from_matrix(RWBraw).as_matrix();anchor=WM[:3,:3]@C[:3,3]+WM[:3,3];tWB=anchor-RWB@Bpalm
  RWF=RWB@RF;tWF=tWB+RWB@pF;quat=Rotation.from_matrix(RWF).as_quat(scalar_first=True)
  native_world=np.asarray(source['landmarks_3d'])@Twc[:3,:3].T+Twc[:3,3];expected_task_world=native_world[taskindices]
  target_world_from_ref=ref@RWB.T+anchor
  source_reconstruction=float(np.max(np.abs(target_world_from_ref-expected_task_world)))
  m=mujoco.MjModel.from_binary_path(case['scene_mjb']['path']);d=mujoco.MjData(m)
  with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:d.qpos[:]=z['qpos']
  nominal=d.qpos.copy();ja=np.asarray([m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names]);ra=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);oa=int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'object_free')]);objectq=d.qpos[oa:oa+7].copy()
  d.qpos[ja]=np.asarray(prior['original_qpos_rad']);d.qpos[ra:ra+3]=np.asarray(prior['original_root_target_position_m']);d.qpos[ra+3:ra+7]=np.asarray(prior['original_root_target_quaternion_wxyz']);d.qvel[:]=0
  if m.nmocap:d.mocap_pos[0]=d.qpos[ra:ra+3];d.mocap_quat[0]=d.qpos[ra+3:ra+7]
  mujoco.mj_forward(m,d);old_actual_errors=_link_positions(m,d,urdf)-expected_task_world;old_actual_caps=norms(old_actual_errors[:5]);old_saved_caps=np.asarray(prior['authoritative_original_tip_caps_sqrt_einsum_m'])
  assert np.max(np.abs(old_actual_caps-old_saved_caps))<=p['old_cap_reconstruction_allowance_m']
  d.qpos[ja]=q;d.qpos[ra:ra+3]=tWF;d.qpos[ra+3:ra+7]=quat;d.qvel[:]=0
  if m.nmocap:d.mocap_pos[0]=tWF;d.mocap_quat[0]=quat
  mujoco.mj_forward(m,d);scene_palm=_body_id(m,'palm');tWF_source=tWF.copy();tWF=tWF-(d.xpos[scene_palm]-anchor);d.qpos[ra:ra+3]=tWF
  if m.nmocap:d.mocap_pos[0]=tWF
  mujoco.mj_forward(m,d);actual=_link_positions(m,d,urdf);expected_achieved=Btask@RWB.T+tWB
  source_scene_error=float(np.max(np.linalg.norm(actual-expected_achieved,axis=1)));anchor_error=float(np.linalg.norm(d.xpos[scene_palm]-anchor));rotation_error=float(np.max(np.abs(d.xmat[scene_palm].reshape(3,3)-RWB@BactualR)))
  errors=actual-expected_task_world;tipnorm=norms(errors[:5]);oldcaps=np.asarray(prior['authoritative_original_tip_caps_sqrt_einsum_m']);caps=(tipnorm<=oldcaps);meanpass=bool(tipnorm.mean()<=p['maximum_mean_fingertip_error_m'])
  reconstructionpass=source_reconstruction<=p['native_transform_reconstruction_allowance_m'] and source_scene_error<=p['fixed_URDF_vs_scene_link_allowance_m'] and anchor_error<=p['anchor_position_allowance_m'] and rotation_error<=p['rotation_matrix_allowance']
  solverpass=all(h['nlopt_result_code'] in p['supported_NLopt_result_codes'] and h['nlopt_last_value'] is not None for h in history)
  eligible=bool(solverpass and marginpass and physicalpass and caps.all() and meanpass and reconstructionpass)
  assert np.array_equal(d.qpos[oa:oa+7],objectq) and d.time==0 and STEPS==0
  record={'source_inventory_index':index,'native_sha256':case['native']['sha256'],'object_id':case['object_id'],'source_case':Path(case['native']['path']).parent.name,'joint_names':names,'qpos_rad':q.tolist(),'raw_candidate_qpos_rad':raw.tolist(),'original_qpos_rad':prior['original_qpos_rad'],'root_target_position_m':tWF.tolist(),'root_target_quaternion_wxyz':quat.tolist(),'original_root_target_position_m':prior['original_root_target_position_m'],'original_root_target_quaternion_wxyz':prior['original_root_target_quaternion_wxyz'],'T_world_URDF_base_rotation':RWB.tolist(),'T_world_URDF_base_rotation_before_SO3_normalization':RWBraw.tolist(),'T_world_URDF_base_translation_m':tWB.tolist(),'target_palm_world_position_m':anchor.tolist(),'target_palm_world_rotation':(RWB@BactualR).tolist(),'neutral_target_palm_world_transform':(WM@C).tolist(),'T_world_MANO':WM.tolist(),'native_wrist_world_position_m':WM[:3,3].tolist(),'calibrated_anchor_minus_native_wrist_world_m':(anchor-WM[:3,3]).tolist(),'calibrated_anchor_displacement_norm_m':float(np.linalg.norm(anchor-WM[:3,3])),'reference_vectors_neutral_palm_m':vp[taskindices].tolist(),'reference_vectors_URDF_base_m':ref.tolist(),'native_landmarks_wrist_MANO_m':L.tolist(),'native_task_world_m':expected_task_world.tolist(),'achieved_task_world_m':actual.tolist(),'task_link_error_vectors_m':errors.tolist(),'tip_error_norms_sqrt_einsum_m':tipnorm.tolist(),'mean_fingertip_error_m':float(tipnorm.mean()),'old_original_five_tip_caps_sqrt_einsum_m':oldcaps.tolist(),'old_original_tip_caps_never_rebased':True,'old_actual_scene_tip_caps_m':old_actual_caps.tolist(),'old_actual_scene_tip_error_vectors_m':old_actual_errors[:5].tolist(),'root_target_fixed_URDF_prediction_position_m':tWF_source.tolist(),'root_translation_scene_compensation_delta_m':(tWF-tWF_source).tolist(),'old_exact_tip_cap_pass':caps.tolist(),'old_tip_error_increase_m':(tipnorm-oldcaps).tolist(),'old_original_mean_tip_m':float(oldcaps.mean()),'strict_mean_guard_pass':meanpass,'raw_optimizer_margin_pass':marginpass,'raw_joint_overflow_rad':overflow.tolist(),'source_physical_joint_bounds_rad':limits.tolist(),'joint_delta_from_old_mapping_rad':(q-np.asarray(prior['original_qpos_rad'])).tolist(),'SO3_root_rotation_normalization_maxabs':float(np.max(np.abs(RWB-RWBraw))),'SO3_root_rotation_normalization_frobenius':float(np.linalg.norm(RWB-RWBraw)),'achieved_robot_wrist_body_world_m':d.xpos[_body_id(m,'wrist')].tolist(),'robot_wrist_body_minus_native_wrist_world_m':(d.xpos[_body_id(m,'wrist')]-WM[:3,3]).tolist(),'achieved_MCP_index_middle_ring_world_m':[d.xpos[_body_id(m,n)].tolist() for n in ('ffknuckle','mfknuckle','rfknuckle')],'native_MCP_index_middle_ring_world_m':native_world[[5,9,13]].tolist(),'bounded_output_source_physical_limits_pass':physicalpass,'clamping_delta_rad':(q-raw).tolist(),'source_native_transform_reconstruction_maxabs_m':source_reconstruction,'fixed_URDF_vs_scene_max_task_error_m':source_scene_error,'calibrated_anchor_error_m':anchor_error,'calibrated_actual_palm_rotation_maxabs_error':rotation_error,'numeric_reconstruction_pass':bool(reconstructionpass),'supported_solver_status_pass':solverpass,'source_gate_eligible_for_required_radial_control':eligible,'eligible_for_separate_full_surface_audit':False,'required_new_anchor_radial_control':'NOT_RUN','retarget_history':history,'calibration_sha256':sha(B/'calibration.json'),'new_eval_data':False,'physics_steps':0,'training':False,'strict_surface_mapping_acceptance':'NOT_RUN','input_sha256':{k:v for k,v in p['source_sha256'].items() if k in [case['native']['path'],case['scene_mjb']['path'],case['stored_shadow_command']['path']]+[x['path'] for x in case['frame_files'].values()]}}
  rows.append(record);print(json.dumps({'case_index':index,'mean_tip_mm':float(tipnorm.mean())*1000,'old_cap_pass':caps.tolist(),'eligible':eligible}),flush=True)
 for table in ('source_sha256','implementation_sha256'):
  for path,h in p[table].items():assert sha(path)==h,path
 report={'schema':'hug_canonical_MCP_calibration_candidate_v1','protocol_sha256':ph,'calibration_sha256':sha(B/'calibration.json'),'cases':rows,'all_six_source_gate_eligible_for_required_radial_control':all(x['source_gate_eligible_for_required_radial_control'] for x in rows),'all_six_eligible_for_separate_full_surface_audit':False,'required_new_anchor_radial_control':'NOT_RUN','production_changed':False,'new_eval_data':False,'physics_steps':STEPS,'HUG_generator_loaded':False,'HUG_generator_sampled':False,'new_grasp_proposal':False,'new_source_canonical_neutral_template':True,'training':False,'strict_surface_acceptance_rule_unchanged':True,'strict_surface_audit_run':False,'limitations':p['limitations']}
 write_new(B/'candidate_targets.json',report)
if __name__=='__main__':main()
