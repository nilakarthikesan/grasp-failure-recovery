"""Once-frozen author-style capsule fixtures. Import performs no native work."""
from pathlib import Path
import importlib.util,hashlib,json,sys
from xml.etree import ElementTree as ET
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
API=ROOT/'src/grasp_failure_prediction/evaluation/manosim_capsule.py'
SCHEMA='manosim_author_style_capsule_execution_case_v3'
BUDGET=dict(models=6,data=6,static_forwards=18,steps=5250,contact_force_reads=754,maximum_seconds=120,outer_seconds=130)
FORBIDDEN=dict(source_MANO_calls=0,HUG_calls=0,retarget_calls=0,training_calls=0,unregistered_engine_calls=0)

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
math=load('capsule_execution_math',HERE/'execution_math.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scene_xml(base,kind,dt):
 base=Path(base).resolve();root=ET.parse(base).getroot();geoms=root.findall('./worldbody//geom')
 if len(geoms)!=16 or len(root.findall('./default/geom'))!=1 or root.findall('.//include') or root.findall('.//deformable'):raise ValueError('Standalone authored16-geom hand required')
 protected=lambda:[(x.tag,dict(x.attrib)) for x in root.iter() if x.tag in ('body','joint','inertial','position')]
 before=protected();default=ET.tostring(root.find('./default/geom'))
 for a in root.findall('./asset/mesh'):
  p=(base.parent/a.get('file')).resolve()
  if not p.is_file():raise ValueError('Authored mesh asset missing')
  a.set('file',str(p))
 opt=root.find('option')
 if opt is None:opt=ET.SubElement(root,'option')
 opt.set('timestep',str(dt));opt.set('integrator','implicitfast');opt.set('cone','elliptic');opt.set('jacobian','dense');opt.set('gravity','0 0 -9.81' if kind=='authored_gravity_equilibrium' else '0 0 0')
 flag=ET.SubElement(opt,'flag') if opt.find('flag') is None else opt.find('flag');flag.set('contact','enable' if kind=='isolated_capsule_plane' else 'disable');flag.set('energy','enable')
 if kind=='isolated_capsule_plane':
  for g in geoms:g.set('contype','0');g.set('conaffinity','0')
  cap=root.find('./worldbody//body[@name="mano_body_3"]/geom')
  if cap is None or cap.get('type')!='capsule':raise ValueError('Fixed index-distal authored capsule absent')
  cap.set('name','qualification_capsule');cap.set('contype','1');cap.set('conaffinity','1')
 # Exactly one static plane; contact-OFF fixtures keep it noncolliding.
 world=root.find('worldbody');ET.SubElement(world,'geom',dict(name='qualification_plane',type='plane',size='1 1 .01',pos='0 0 -100',contype='1' if kind=='isolated_capsule_plane' else '0',conaffinity='1' if kind=='isolated_capsule_plane' else '0',condim='3',friction='1 .005 .0001'))
 if protected()!=before or ET.tostring(root.find('./default/geom'))!=default:raise ValueError('Authored skeleton/inertials/gains/default compliance changed')
 return ET.tostring(root,encoding='unicode')+'\n'


def frames(data,contract):
 ids=contract.joint_ids;anchors=np.asarray(data.xanchor)[[ids[3],*ids[4:]]].copy();rotations=np.asarray(data.xmat)[list(contract.body_ids)].reshape(16,3,3).copy()
 return math.finite(anchors,(16,3)),math.finite(rotations,(16,3,3))



def place_world_plane(model,plane,position,quaternion):
 """Update only the declared static world plane, including compiled frame cache."""
 if isinstance(plane,(bool,np.bool_)) or int(plane)!=plane or not 0<=plane<len(model.geom_bodyid):raise ValueError('Exact declared plane index required')
 if int(model.geom_bodyid[plane])!=0 or int(model.geom_type[plane])!=0:raise ValueError('Declared static world plane required')
 position=math.finite(position,(3,));quaternion=math.finite(quaternion,(4,))
 if abs(quaternion@quaternion-1)>1e-12:raise ValueError('Proper planned plane quaternion required')
 model.geom_pos[plane]=position;model.geom_quat[plane]=quaternion;model.geom_sameframe[plane]=0 # pinned mjSAMEFRAME_NONE


def contact_operands(model,data,hand,plane,*,pre_time_s,pre_qpos,pre_qvel,reason):
 """Copy live numeric operands only; no force accessor or other engine call."""
 fields=('geom','dim','efc_address','dist','pos','frame','friction','solref','solimp')
 shapes={'geom':2,'pos':3,'frame':9,'friction':5,'solref':2,'solimp':5}
 a={}
 for name in fields:
  rows=[np.asarray(getattr(data.contact[i],name)).copy() for i in range(int(data.ncon))]
  a['contact_'+name]=np.asarray(rows).reshape((int(data.ncon),shapes[name])) if name in shapes else np.asarray(rows)
 a.update(ncon=np.asarray(data.ncon),nefc=np.asarray(data.nefc),hand_geom_id=np.asarray(hand),plane_geom_id=np.asarray(plane),
  actual_geom_ids=np.asarray([hand,plane]),actual_geom_bodyid=np.asarray(model.geom_bodyid[[hand,plane]]).copy(),
  actual_geom_sameframe=np.asarray(model.geom_sameframe[[hand,plane]]).copy(),actual_geom_pos=np.asarray(model.geom_pos[[hand,plane]]).copy(),
  actual_geom_quat=np.asarray(model.geom_quat[[hand,plane]]).copy(),actual_geom_size=np.asarray(model.geom_size[[hand,plane]]).copy(),
  actual_geom_xpos=np.asarray(data.geom_xpos[[hand,plane]]).copy(),actual_geom_xmat=np.asarray(data.geom_xmat[[hand,plane]]).reshape(2,3,3).copy(),
  actual_geom_solref=np.asarray(model.geom_solref[[hand,plane]]).copy(),actual_geom_solimp=np.asarray(model.geom_solimp[[hand,plane]]).copy(),
  actual_geom_friction=np.asarray(model.geom_friction[[hand,plane]]).copy(),actual_geom_priority=np.asarray(model.geom_priority[[hand,plane]]).copy(),
  pre_time_s=np.asarray(pre_time_s),endpoint_time_s=np.asarray(data.time),pre_qpos=np.asarray(pre_qpos).copy(),pre_qvel=np.asarray(pre_qvel).copy(),
  qpos=np.asarray(data.qpos).copy(),qvel=np.asarray(data.qvel).copy(),ctrl=np.asarray(data.ctrl).copy(),qacc=np.asarray(data.qacc).copy(),
  efc_J=np.asarray(data.efc_J).reshape(int(data.nefc),51).copy(),efc_force=np.asarray(data.efc_force).copy(),efc_R=np.asarray(data.efc_R).copy(),efc_aref=np.asarray(data.efc_aref).copy(),
  qfrc_constraint=np.asarray(data.qfrc_constraint).copy(),warning_counts=data.warning.number.copy(),reason=np.asarray(reason),
  snapshot_schema_version=np.asarray('manosim_capsule_actual_contact_operands_v1'))
 return a


def plane_preflight_checks(a,position,quaternion):
 """Existing forward must realize the declared plane and tangent target pair."""
 R=math.finite(a['actual_geom_xmat'],(2,3,3));p=math.finite(a['actual_geom_xpos'],(2,3));desired=math.matrix(quaternion);n=R[1,:,2]
 gap=math.capsule_gap(p[0],R[0],a['actual_geom_size'][0],p[1],n)
 count=int(a['ncon']);one=count==1;absent=count==0 and int(a['nefc'])==0 and not np.any(a['qfrc_constraint'])
 pair=one and set(map(int,a['contact_geom'][0]))=={int(a['hand_geom_id']),int(a['plane_geom_id'])} and int(a['contact_dim'][0])==3
 sign=(1 if int(a['contact_geom'][0,1])==int(a['hand_geom_id']) else -1) if pair else 0
 checks=dict(sameframe_NONE=int(a['actual_geom_sameframe'][1])==0,plane_position=np.max(np.abs(p[1]-position))<=math.LIMITS['contact_gap_m'],
  plane_rotation=np.max(np.abs(R[1]-desired))<=math.LIMITS['contact_normal'],analytic_tangent_gap=abs(gap)<=math.LIMITS['contact_gap_m'],
  contact_inventory=absent or pair,engine_tangent_gap_if_present=bool(absent or pair and abs(float(a['contact_dist'][0]))<=math.LIMITS['contact_gap_m']),
  contact_normal_if_present=bool(absent or pair and np.max(np.abs(a['contact_frame'][0].reshape(3,3)[0]-sign*n))<=math.LIMITS['contact_normal']))
 return {key:bool(value) for key,value in checks.items()}


def contact_sample(model,data,contract,api,engine,pre_v,hand_geom,plane_geom,refusal_witness=None):
 def refuse(message):
  if refusal_witness is not None:refusal_witness(message)
  raise RuntimeError(message)
 out=dict(contact_present=bool(data.ncon),contact_active=False,normal_force_N=0.,contact_J_error=0.,contact_gap_error_m=0.,contact_normal_error=0.,generalized_translation_force_error_N=0.,generalized_angular_force_error_Nm=0.,virtual_power_error_W=0.,virtual_power_scale_W=0.,cone_primal=0.,cone_dual=0.,cone_complementarity=0.,cone_force_scale=1.,cone_gradient_scale=1.,contact_point_world_m=np.zeros(3),contact_frame=np.zeros((3,3)),contact_force=np.zeros(3),engine_contact_J=np.zeros((3,51)),independent_contact_J=np.zeros((3,51)),contact_gradient=np.zeros(3),contact_regularizer=np.zeros(3),contact_reference_acceleration=np.zeros(3),analytic_gap_m=0.,engine_gap_m=0.)
 out.update(capsule_center_world_m=np.asarray(data.geom_xpos[hand_geom]).copy(),capsule_rotation_world=np.asarray(data.geom_xmat[hand_geom]).reshape(3,3).copy(),capsule_size_m=np.asarray(model.geom_size[hand_geom]).copy(),plane_position_world_m=np.asarray(data.geom_xpos[plane_geom]).copy(),plane_rotation_world=np.asarray(data.geom_xmat[plane_geom]).reshape(3,3).copy(),actual_contact_friction=np.zeros(5),actual_contact_solref=np.zeros(2),actual_contact_solimp=np.zeros(5),actual_contact_geom_ids=np.array([-1,-1],dtype=int))
 if data.ncon>1:refuse('Fixed one endpoint-plane contact inventory exceeded')
 if not data.ncon:
  if data.nefc or np.any(data.qfrc_constraint):refuse('Unexpected non-contact constraint')
  return out
 c=data.contact[0]
 if set(map(int,c.geom))!={hand_geom,plane_geom} or int(c.dim)!=3:refuse('Unexpected contact pair/dimension')
 basis=math.finite(np.asarray(c.frame).reshape(3,3),(3,3));point=math.finite(c.pos,(3,));sign=1 if int(c.geom[1])==hand_geom else -1
 R=np.asarray(data.geom_xmat[hand_geom]).reshape(3,3);n=np.asarray(data.geom_xmat[plane_geom]).reshape(3,3)[:,2];gap=math.capsule_gap(data.geom_xpos[hand_geom],R,model.geom_size[hand_geom],data.geom_xpos[plane_geom],n)
 if min(float(c.dist),gap)<-math.LIMITS['maximum_penetration_m']:refuse('Declared5mm maximum contact penetration escaped')
 out.update(actual_contact_friction=np.asarray(c.friction).copy(),actual_contact_solref=np.asarray(c.solref).copy(),actual_contact_solimp=np.asarray(c.solimp).copy(),actual_contact_geom_ids=np.asarray(c.geom).copy(),analytic_gap_m=gap,engine_gap_m=float(c.dist),contact_gap_error_m=abs(gap-float(c.dist)),contact_normal_error=float(np.max(np.abs(basis[0]-sign*n))),contact_point_world_m=point.copy(),contact_frame=basis.copy())
 adr=int(c.efc_address)
 if adr<0:
  if data.nefc or np.any(data.qfrc_constraint):refuse('Excluded contact has unexpected active rows')
  return out
 if adr!=0 or data.nefc!=3:refuse('Exactly three declared elliptic rows required')
 anchors,rotations=frames(data,contract);independent=sign*basis@math.body_point_J(point,3,anchors,rotations)
 J=math.finite(np.asarray(data.efc_J).reshape(3,51),(3,51));ef=math.finite(data.efc_force[:3],(3,));force=engine.force(model,data,0)[:3]
 if not np.allclose(force,ef,rtol=0,atol=1e-10):refuse('Elliptic engine force coordinates differ')
 regularizer=math.finite(data.efc_R[:3],(3,));reference=math.finite(data.efc_aref[:3],(3,));gradient=J@data.qacc+regularizer*ef-reference
 if not np.array_equal(np.asarray(c.solref),[.04,1.]) or not np.array_equal(np.asarray(c.solimp),[.5,.95,.001,.5,2.]) or not np.array_equal(np.asarray(c.friction)[:2],[1.,1.]):refuse('Actual authored capsule contact parameters differ')
 tau=independent.T@force;residual=data.qfrc_constraint-tau;probe=np.sin(np.arange(51));work=max(abs(float(force@(J@pre_v)-tau@pre_v)),abs(float(force@(J@probe)-tau@probe)));scale=max(abs(float(force@(J@pre_v))),abs(float(force@(J@probe))))
 cone=math.cone_residual(ef,gradient,float(c.friction[0]));out.update(contact_active=True,normal_force_N=float(force[0]),contact_J_error=float(np.max(np.abs(J-independent))),generalized_translation_force_error_N=float(np.max(np.abs(residual[:3]))),generalized_angular_force_error_Nm=float(np.max(np.abs(residual[3:]))),virtual_power_error_W=work,virtual_power_scale_W=scale,cone_primal=cone['primal'],cone_dual=cone['dual'],cone_complementarity=cone['complementarity'],cone_force_scale=cone['force_scale'],cone_gradient_scale=cone['gradient_scale'],contact_force=force.copy(),engine_contact_J=J.copy(),independent_contact_J=independent.copy(),contact_gradient=gradient.copy(),contact_regularizer=regularizer.copy(),contact_reference_acceleration=reference.copy())
 return out



def validate_with_snapshot(model,resolve,assets,rest_quaternions,api,save,directory,fixture_xml):
 """Seal raw compiler operands before validation; no Data, FK or engine call."""
 arrays={name:np.asarray(getattr(model,name)).copy() for name in api._COMPILED_ARRAYS}
 arrays.update(named_body_ids=np.array([resolve('body',name) for name in api.BODY_NAMES],dtype=np.int64),
  named_joint_ids=np.array([resolve('joint',name) for name in api.JOINT_NAMES],dtype=np.int64),
  named_actuator_ids=np.array([resolve('actuator',name) for name in api.ACTUATOR_NAMES],dtype=np.int64),
  source_body_names=np.asarray(api.BODY_NAMES),source_joint_names=np.asarray(api.JOINT_NAMES),source_actuator_names=np.asarray(api.ACTUATOR_NAMES),
  source_fullinertia6=np.asarray([[float(x) for x in row['inertial']['fullinertia'].split()] for row in assets['body_records']],dtype=np.float64),
  source_body_mass=np.asarray([row['mass'] for row in assets['body_records']],dtype=np.float64),rest_quaternions=np.asarray(rest_quaternions).copy(),
  model_dimensions=np.asarray([getattr(model,name) for name in ('nq','nv','nu','njnt','nbody','ngeom')],dtype=np.int64),
  snapshot_schema_version=np.asarray('manosim_capsule_prevalidation_compiled_operands_v1'),validation_performed=np.asarray(False),
  fixture_xml_sha256=np.asarray(sha(fixture_xml)),authored_capsule_xml_sha256=np.asarray(assets['capsule_xml_sha256']),
  asset_closure_sha256=np.asarray(assets['closure_sha256']),mapping_controller_source_sha256=np.asarray(sha(API)))
 save(Path(directory)/'prevalidation_compiled_model.npz','PREVALIDATION_UNINTERPRETED_COMPILED_MODEL_OPERANDS',**arrays)
 return api.validate_compiled_contract(model,resolve,assets,rest_quaternions)


def prepare_case(index,spec,protocol,output,engine,notify):
 import mujoco
 api=load('frozen_capsule_command',API);output=Path(output);artifacts=[];prepared=[]
 def seal(path,label):
  row=dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,label=label);artifacts.append(row);notify(dict(event='artifact',artifact=row));return row
 def save(path,label,**arrays):
  with path.open('xb') as f:np.savez_compressed(f,**arrays)
  return seal(path,label)
 with np.load(spec['literal_buffers']['path'],allow_pickle=False) as file:b={k:file[k].copy() for k in file.files}
 with np.load(spec['source_readbacks']['path'],allow_pickle=False) as file:src={k:file[k].copy() for k in file.files}
 target=api.target_from_source(b['original_postAA32'],b['original_camera_translation32'],b['original_cameraWorld64'],b['rest_quaternions'],shape_float32=b['shape'])
 if not np.array_equal(target.qpos,b['source_initialized_q0']) or not np.array_equal(target.qpos,src['actual_named_qpos'][0]):raise ValueError('Exact qualified source target byte identity required')
 assets=api.inspect_authored_assets(spec['capsule_xml'],spec['shape_asset'],spec['builder_report'],expected_shape_sha256=spec['expected_shape_sha256'],native_shape_float32=b['shape'])
 for fixture in [c for c in math.fixture_plan() if c['index']==index]:
  key=fixture['kind']+'_'+str(round(1/fixture['dt_s']));directory=output/key;directory.mkdir();xp=directory/'fixture.xml';xp.write_text(scene_xml(spec['capsule_xml'],fixture['kind'],fixture['dt_s']));seal(xp,'explicit_new_capsule_fixture_XML')
  model=engine.model(xp);resolve=lambda kind,name:int(mujoco.mj_name2id(model,{'body':mujoco.mjtObj.mjOBJ_BODY,'joint':mujoco.mjtObj.mjOBJ_JOINT,'actuator':mujoco.mjtObj.mjOBJ_ACTUATOR,'mesh':mujoco.mjtObj.mjOBJ_MESH}[kind],name))
  contract=validate_with_snapshot(model,resolve,assets,b['rest_quaternions'],api,save,directory,xp);data=engine.data(model);goal=np.asarray(target.qpos).copy();command=api.command_from_qpos(goal,contract)
  data.qpos[:]=goal;data.qvel[:]=0.;data.ctrl[:]=command.ctrl;engine.forward(model,data)
  anchors,R=frames(data,contract);static=math.static_checks(data.qpos,goal,anchors,R,src['source_joint_anchors_world'][0],src['source_body_rotations_world'][0],data.actuator_length,command.ctrl,data.actuator_force)
  save(directory/'static_command.npz','source_pose_actual_named_transmission_and_zero_servo_force',qpos=data.qpos.copy(),ctrl=data.ctrl.copy(),actuator_length=data.actuator_length.copy(),actuator_force=data.actuator_force.copy(),anchors=anchors,rotations=R,source_anchors=src['source_joint_anchors_world'][0],source_rotations=src['source_body_rotations_world'][0])
  total_mass=float(model.body_mass.sum());wrist_kp=300.
  plane=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'qualification_plane'));hand=-1;normal=np.zeros(3);plane_pos=np.zeros(3)
  start=goal.copy()
  if fixture['kind']=='zero_gravity_tracking':
   start[:3]+=[.01,-.01,.01];start[3:7]=math.quat_product(goal[3:7],math.quat_exp(np.array([.1,-.05,.08])));start[7:]=np.tile([1.,0,0,0],15)
  elif fixture['kind']=='isolated_capsule_plane':
   hand=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'qualification_capsule'));normal=math.plane_normal(data.geom_xmat[hand].reshape(3,3));size=model.geom_size[hand];plane_pos=data.geom_xpos[hand]-normal*(size[0]+size[1]/np.sqrt(2));place_world_plane(model,plane,plane_pos,math.plane_quaternion(normal))
   contract=api.validate_compiled_contract(model,resolve,assets,b['rest_quaternions'])
  data.qpos[:]=start;data.qvel[:]=0.;data.ctrl[:]=api.command_from_qpos(start,contract).ctrl;data.qacc_warmstart[:]=0.;engine.forward(model,data)
  plane_checks=None
  if fixture['kind']=='isolated_capsule_plane':
   operand=contact_operands(model,data,hand,plane,pre_time_s=data.time,pre_qpos=data.qpos,pre_qvel=data.qvel,reason='SECOND_EXISTING_FORWARD_PLANE_PREFLIGHT')
   save(directory/'plane_preflight.npz','actual_plane_frame_and_tangent_contact_before_dynamics',**operand,planned_plane_position=plane_pos,planned_plane_quaternion=math.plane_quaternion(normal),planned_plane_normal=normal)
   plane_checks=plane_preflight_checks(operand,plane_pos,math.plane_quaternion(normal))
   if not all(plane_checks.values()):raise ValueError('Actual world plane/tangent contact preflight failed before any dynamics: '+str(plane_checks))
  save(directory/'fixture_model.npz','actual_unchanged_native_inertials_controller_compliance_and_isolated_plane',body_mass=model.body_mass.copy(),body_inertia=model.body_inertia.copy(),body_ipos=model.body_ipos.copy(),body_iquat=model.body_iquat.copy(),body_parentid=model.body_parentid.copy(),dof_damping=model.dof_damping.copy(),dof_armature=model.dof_armature.copy(),geom_size=model.geom_size.copy(),geom_bodyid=model.geom_bodyid.copy(),geom_type=model.geom_type.copy(),joint_pos=model.jnt_pos.copy(),body_pos=model.body_pos.copy(),body_quat=model.body_quat.copy(),option_disableflags=np.asarray(model.opt.disableflags),option_enableflags=np.asarray(model.opt.enableflags),gravity=model.opt.gravity.copy(),actuator_gainprm=model.actuator_gainprm.copy(),actuator_biasprm=model.actuator_biasprm.copy(),actuator_gear=model.actuator_gear.copy(),geom_solref=model.geom_solref.copy(),geom_solimp=model.geom_solimp.copy(),geom_priority=model.geom_priority.copy(),geom_sameframe=model.geom_sameframe.copy(),geom_contype=model.geom_contype.copy(),geom_conaffinity=model.geom_conaffinity.copy(),geom_pos=model.geom_pos.copy(),geom_quat=model.geom_quat.copy(),native_target=goal,initial_state=start,plane_normal=normal,plane_position=plane_pos)
  if not all(static.values()):raise ValueError('Static source mapping/transmission gate failed before any dynamics')
  samples=np.stack([reference_qpos(fixture,start,goal,normal,k*fixture['dt_s']) for k in range(fixture['steps'])]+[reference_qpos(fixture,start,goal,normal,fixture['duration_s'])])
  branches=api.require_continuous_principal_targets(samples,contract)
  branch_receipt=dict(samples=len(branches),minimum_pi_margin_rad=min(float(x.pi_margin_rad.min()) for x in branches),ctrl_sequence_sha256=hashlib.sha256(np.stack([x.ctrl for x in branches]).tobytes()).hexdigest(),finite_sampled_check_only=True)
  metadata=directory/'command_contract.json'
  metadata.write_text(json.dumps(dict(target=target.metadata(),compiled=contract.metadata(),planned_branch_check=branch_receipt,command=command.metadata(),static_checks=static,plane_preflight_checks=plane_checks),sort_keys=True,indent=2,allow_nan=False)+'\n');seal(metadata,'pre_dynamics_all_sampled_command_chart_and_static_mapping')
  prepared.append(dict(fixture=fixture,directory=directory,model=model,data=data,contract=contract,goal=goal,start=start,normal=normal,hand=hand,plane=plane,static=static,plane_preflight_checks=plane_checks,total_mass=total_mass,wrist_kp=wrist_kp))
  notify(dict(event='static_fixture_preflight_completed',fixture=fixture,steps=0))
 return dict(index=index,api=api,source=src,artifacts=artifacts,prepared=prepared,save=save)


def reference_qpos(fixture,start,goal,normal,time_s):
 reference=goal.copy()
 if fixture['kind']=='zero_gravity_tracking':reference=math.trajectory(start,goal,math.smooth_fraction(time_s,.5))
 elif fixture['kind']=='isolated_capsule_plane':reference[:3]-=.0005*normal*math.smooth_fraction(time_s,.1)
 return reference


def execute_case(context,engine,notify):
 index=context['index'];api=context['api'];src=context['source'];save=context['save'];artifacts=context['artifacts'];measurements=[];context['measurements']=measurements
 for session in context['prepared']:
  context['current_session']=session
  fixture,directory,model,data,contract,goal,start,normal,hand,plane,static,total_mass,wrist_kp=(session[k] for k in ('fixture','directory','model','data','contract','goal','start','normal','hand','plane','static','total_mass','wrist_kp'))
  context['current_fixture']=fixture;rows=[];context['partial_rows']=rows
  for step in range(fixture['steps']):
   t=step*fixture['dt_s'];reference=reference_qpos(fixture,start,goal,normal,t)
   requested=api.command_from_qpos(reference,contract);data.ctrl[:]=requested.ctrl;pre_q=data.qpos.copy();pre_v=data.qvel.copy();pre_time=float(data.time);engine.step(model,data)
   anchors,R=frames(data,contract)
   if fixture['kind']=='isolated_capsule_plane':
    def witness(reason):
     path=directory/'failed_contact_operands.npz'
     if not path.exists():save(path,'FAILED_PARTIAL_UNINTERPRETED_actual_contact_before_refusal',**contact_operands(model,data,hand,plane,pre_time_s=pre_time,pre_qpos=pre_q,pre_qvel=pre_v,reason=reason))
    try:contact=contact_sample(model,data,contract,api,engine,pre_v,hand,plane,refusal_witness=witness)
    except BaseException as error:
     witness(type(error).__name__+': '+str(error));raise
   else:
    if data.ncon or data.nefc or np.any(data.qfrc_constraint):raise RuntimeError('Contact-OFF fixture generated contact/constraint force')
    contact={}
   row=dict(pre_time_s=pre_time,endpoint_time_s=float(data.time),pre_qpos=pre_q,pre_qvel=pre_v,endpoint_qpos=data.qpos.copy(),endpoint_qvel=data.qvel.copy(),requested_ctrl=requested.ctrl.copy(),effective_ctrl=data.ctrl.copy(),reference_qpos=reference,phase_code=int(t>=(.1 if fixture['kind']=='isolated_capsule_plane' else .5)),body_rotations_start=R,anchors_start=anchors,bone_angle_error_rad=math.angle_errors(R,src['source_body_rotations_world'][0]),warning_counts=data.warning.number.copy(),solver_niter=np.asarray(data.solver_niter).copy(),start_qacc=data.qacc.copy(),start_qfrc_actuator=data.qfrc_actuator.copy(),start_qfrc_passive=data.qfrc_passive.copy(),start_qfrc_bias=data.qfrc_bias.copy(),start_qfrc_constraint=data.qfrc_constraint.copy(),start_actuator_force=data.actuator_force.copy(),start_actuator_length=data.actuator_length.copy(),start_actuator_velocity=data.actuator_velocity.copy(),actuator_power_W=float(data.qfrc_actuator@pre_v),passive_power_W=float(data.qfrc_passive@pre_v),contact_power_W=float(data.qfrc_constraint@pre_v),engine_energy_start_J=data.energy.copy(),**contact)
   if not all(np.isfinite(np.asarray(v)).all() for v in row.values()):raise RuntimeError('Nonfinite recorded native fixture channel')
   rows.append(row)
  engine.forward(model,data);api.verify_compiled_contract(model,contract);a={k:np.asarray([row[k] for row in rows]) for k in rows[0]};save(directory/'trajectory.npz','aligned_actual_native_state_phase_command_and_force_clocks',**a)
  assessment=math.assess(fixture,a,goal,src['source_body_rotations_world'][0],total_mass,wrist_kp);assessment['static_source_and_transmission_checks']=static;assessment['plane_preflight_checks']=session['plane_preflight_checks']
  if not all(static.values()):assessment['status']='FAIL'
  measurements.append(dict(fixture=fixture,assessment=assessment,final_qpos=data.qpos.tolist(),final_rotations=frames(data,contract)[1].tolist(),phase_clock='reference and forces at pre_time_s; state endpoint at endpoint_time_s; body frames at pre_time_s',command_clipping=False,**math.FLAGS));notify(dict(event='fixture_completed',fixture=fixture,status=assessment['status']));context['current_fixture']=None;context['partial_rows']=[]
 convergence=[]
 for kind in math.KINDS:
  pair=[r for r in measurements if r['fixture']['kind']==kind];dx=float(np.linalg.norm(np.asarray(pair[0]['final_qpos'])[:3]-np.asarray(pair[1]['final_qpos'])[:3]));dr=float(math.angle_errors(pair[0]['final_rotations'],pair[1]['final_rotations']).max());convergence.append(dict(kind=kind,wrist_difference_m=dx,bone_difference_rad=dr,passed=dx<=math.LIMITS['convergence_wrist_m'] and dr<=math.LIMITS['convergence_bone_rad']))
 return dict(schema_version=SCHEMA,index=index,status='COMPLETED',measurements=measurements,convergence=convergence,qualification='PASS' if all(m['assessment']['status']=='PASS' for m in measurements) and all(c['passed'] for c in convergence) else 'NOT_QUALIFIED',artifacts=artifacts,forbidden_counts=FORBIDDEN,physical_outcome=None,**math.FLAGS)
