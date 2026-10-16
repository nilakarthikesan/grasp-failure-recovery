"""Finite prospective task collection. Engine operations only through injected guard.

The authored capsule approximation is a separate domain from native skin/hardware.
Privileged physics arrays provide labels and never enter observable features.
"""
from pathlib import Path
from types import SimpleNamespace
import functools,hashlib,importlib.util,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PROFILE=ROOT/'research_artifacts/manosim_native_physics_profile_2026-10-16_v2'
SCHEMA='manosim_native_object_pilot_v2'
COUNTS=dict(models=45,data=45,static_forwards=405,resets=270,steps=412500,contact_force_reads=412500*256,alignment_triangle_point_tests=60*32*10000)
FORBIDDEN=dict(MANO_calls=0,HUG_calls=0,retarget_calls=0,predictor_fits=0,unregistered_engine_calls=0)
QUARANTINE_CODES=frozenset({'RESET_SUPPORT_STABILITY','PRECLOSE_TRACKING','CONTACT_PENETRATION','ACTUAL_PRINCIPAL_CHART'})
class QuarantineError(RuntimeError):
 def __init__(self,code,message):
  if code not in QUARANTINE_CODES:raise ValueError('Unknown prospective quarantine reason')
  self.code=code;super().__init__(message)

FLAGS=dict(full_skin_parity_claimed=False,hardware_calibrated=False,scientific_training_allowed=False,native_task_qualified=False)


def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
@functools.cache
def modules():
 sys.path.insert(0,str(ROOT/'src'))
 from grasp_failure_prediction.evaluation import manosim_recording as rec,native_task_labels as labels
 for module in (rec,labels):
  if not Path(module.__file__).resolve().is_relative_to(ROOT/'src'):raise RuntimeError('Canonical project-work recording/label module required')
 return SimpleNamespace(m=load('native_object_pilot_math',HERE/'pilot_math.py'),u=load('native_object_pilot_interpolation',ROOT/'research_artifacts/manosim_capsule_execution_2026-10-15_v3/execution_math.py'),api=load('native_object_pilot_capsule',ROOT/'src/grasp_failure_prediction/evaluation/manosim_capsule.py'),hand=load('native_object_pilot_hand',PROFILE/'native_hand.py'),profile=load('native_object_pilot_gravity',PROFILE/'profile_math.py'),rec=rec,labels=labels)
def digest(value):return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def npz(path):
 with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def arrays_from_rows(rows):return {k:np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}
def write_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


class Artifacts:
 def __init__(self,root):self.root=Path(root).resolve();self.rows=[]
 def add(self,path,label):
  path=Path(path).resolve()
  if not path.is_relative_to(self.root) or path.is_symlink() or not path.is_file():raise ValueError('Raw artifact escaped allowed root')
  row=dict(path=str(path),sha256=file_sha(path),bytes=path.stat().st_size,label=label)
  if any(x['path']==row['path'] for x in self.rows):raise ValueError('Artifact already published')
  self.rows.append(row)
  journal=self.root.parent/'artifact_index.jsonl'
  with journal.open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush()
  return row
 def json(self,path,value,label):write_json(path,value);return self.add(path,label)
 def arrays(self,path,value,label):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():raise ValueError('Array path already exists')
  with path.open('xb') as f:np.savez_compressed(f,**value)
  return self.add(path,label)
 def relocate(self,old,new):
  old=Path(old).resolve();new=Path(new).resolve()
  if not old.is_relative_to(self.root) or not new.is_relative_to(self.root) or new.exists():raise ValueError('Invalid quarantine relocation')
  old.rename(new)
  for row in self.rows:
   path=Path(row['path'])
   if path.is_relative_to(old):
    original=str(path);destination=str(new/path.relative_to(old));row.update(original_publication_path=original,path=destination)
    with (self.root.parent/'artifact_index.jsonl').open('a') as f:f.write(json.dumps(dict(event='artifact_relocation',original_path=original,path=destination,sha256=row['sha256'],bytes=row['bytes']),sort_keys=True)+'\n')
  files=[new] if new.is_file() else sorted(new.rglob('*'))
  for path in files:
   if path.is_file() and str(path.resolve()) not in {a['path'] for a in self.rows}:self.add(path,'quarantined_partial_publication')
 def verify(self):
  for row in self.rows:
   p=Path(row['path'])
   if p.is_symlink() or file_sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Creation-time artifact changed')
  actual={str(p.resolve()) for p in self.root.rglob('*') if p.is_file()}
  if any(p.is_symlink() for p in self.root.rglob('*')) or actual!={r['path'] for r in self.rows}:raise ValueError('Unindexed raw output files')


def frames(data,c):
 ids=list(c.body_ids);joints=list(c.joint_ids)[3:]
 return np.asarray(data.xanchor)[joints].copy(),np.asarray(data.xmat)[ids].reshape(16,3,3).copy()


def contact_sums(model,data,c,engine,pre_qvel,raw,attempt_id,witnessed,pre_qpos=None):
 """Returned step contact/frame/J are preceding-interval start diagnostics."""
 if not 0<=int(data.ncon)<=256:raise RuntimeError('Declared256 contact ceiling exceeded')
 sums=dict(hand_object_normal_force_N=0.,hand_table_normal_force_N=0.,object_table_normal_force_N=0.,maximum_penetration_m=0.,contact_count=int(data.ncon))
 sums.update(pair_peak_force6=np.zeros((3,6)),pair_peak_present=np.zeros(3,bool),pair_peak_geom_ids=np.full((3,2),-1,int),pair_peak_frame=np.zeros((3,3,3)),pair_peak_position=np.zeros((3,3)),pair_peak_distance_m=np.zeros(3),pair_peak_friction=np.zeros((3,5)),pair_peak_solref=np.zeros((3,2)),pair_peak_solimp=np.zeros((3,5)))
 hand=set(c.hand.body_ids);obj=c.object_body_id;table=int(engine.resolve(model,'geom','pilot_table'));J=np.asarray(data.efc_J).reshape(int(data.nefc),57)
 for i in range(int(data.ncon)):
  contact=data.contact[i];g0,g1=map(int,contact.geom);b0,b1=map(int,model.geom_bodyid[[g0,g1]])
  kind='hand_object' if (b0 in hand and b1==obj) or (b1 in hand and b0==obj) else 'hand_table' if (b0 in hand and g1==table) or (b1 in hand and g0==table) else 'object_table' if (b0==obj and g1==table) or (b1==obj and g0==table) else None
  if kind is None:raise RuntimeError('Unplanned collision pair; hand self-collision must be disabled')
  force=engine.force(model,data,i)
  if not np.isfinite(force).all() or force[0]<-1e-10:raise RuntimeError('Invalid nonnegative normal contact force')
  sums[kind+'_normal_force_N']+=max(0.,float(force[0]));sums['maximum_penetration_m']=max(sums['maximum_penetration_m'],max(0.,-float(contact.dist)))
  slot=('hand_object','hand_table','object_table').index(kind)
  if not sums['pair_peak_present'][slot] or force[0]>sums['pair_peak_force6'][slot,0]:
   sums['pair_peak_present'][slot]=True
   for key,value in dict(pair_peak_force6=force,pair_peak_geom_ids=[g0,g1],pair_peak_frame=np.asarray(contact.frame).reshape(3,3),pair_peak_position=contact.pos,pair_peak_distance_m=contact.dist,pair_peak_friction=contact.friction,pair_peak_solref=contact.solref,pair_peak_solimp=contact.solimp).items():sums[key][slot]=value
  key=(attempt_id,kind)
  penetration=float(contact.dist)<-.005
  if (key not in witnessed and force[0]>.001) or penetration:
   address=int(contact.efc_address);dim=int(contact.dim);gids=[g0,g1];row=dict(force_time_s=np.asarray(float(data.time)-float(model.opt.timestep)),contact_index=np.asarray(i),geom_ids=np.asarray(gids),body_ids=np.asarray([b0,b1]),contact_position=np.asarray(contact.pos).copy(),contact_frame=np.asarray(contact.frame).reshape(3,3).copy(),contact_distance_m=np.asarray(contact.dist),contact_force6=force.copy(),contact_friction=np.asarray(contact.friction).copy(),contact_solref=np.asarray(contact.solref).copy(),contact_solimp=np.asarray(contact.solimp).copy(),efc_address=np.asarray(address),contact_dim=np.asarray(dim),start_qvel=np.asarray(pre_qvel).copy(),start_qacc=np.asarray(data.qacc).copy(),geom_positions=np.asarray(data.geom_xpos)[gids].copy(),geom_rotations=np.asarray(data.geom_xmat)[gids].reshape(2,3,3).copy(),geom_size=np.asarray(model.geom_size)[gids].copy(),geom_type=np.asarray(model.geom_type)[gids].copy())
   if pre_qpos is not None:row['start_qpos']=np.asarray(pre_qpos).copy()
   row['start_hand_anchors'],row['start_hand_rotations']=frames(data,c.hand)
   row.update(start_object_body_origin=np.asarray(data.xpos)[obj].copy(),start_object_body_rotation=np.asarray(data.xmat)[obj].reshape(3,3).copy())
   if address>=0:row.update(efc_J=J[address:address+dim].copy(),efc_force=np.asarray(data.efc_force)[address:address+dim].copy(),efc_R=np.asarray(data.efc_R)[address:address+dim].copy(),efc_aref=np.asarray(data.efc_aref)[address:address+dim].copy())
   name=kind+'_penetration_failure_witness.npz' if penetration else kind+'_first_positive_witness.npz'
   raw.arrays(raw.root/attempt_id/name,row,'penetration_failure_contact_operands' if penetration else 'first_positive_contact_operands');witnessed.add(key)
  if penetration:raise QuarantineError('CONTACT_PENETRATION','Declared5mm physical contact penetration bound escaped')
 return sums


class Pilot:
 def __init__(self,p,raw,engine):self.p=p;self.raw=raw;self.e=engine;self.x=modules();self.models={};self.prepared={};self.results=[];self.alignment={};self.witnessed=set();self.active=None
 def align(self):
  for observation_id,spec in sorted(self.p['observations'].items()):
   a=self.x.m.alignment_inputs(self.x.m.observation(spec['path']));checks=dict(support=bool(a['passed_support']))
   if a['passed_support']:
    v,f=self.x.m.obj_mesh(spec['visual_mesh']);
    if len(f)>10000:raise RuntimeError('Predeclared visual triangle ceiling exceeded')
    self.e.charge('alignment_triangle_point_tests',len(f)*32);tri=v[f]+np.asarray(spec['world_position']);ids,nearest,d=self.x.m.nearest_surface(a['points_world'],tri)
    a.update(nearest_triangle_ids=ids,nearest_triangles=tri[ids],nearest_points=nearest,distances_m=d,source_mesh_sha256=np.asarray(file_sha(spec['visual_mesh'])))
    checks['within_depth_pixel_allowance']=bool(np.all(d<=a['allowances_m']))
   row=self.raw.arrays(self.raw.root/'alignment'/(observation_id+'.npz'),a,'depth_visual_alignment_operands')
   self.alignment[observation_id]=dict(passed=all(checks.values()),checks=checks,artifact=row,scope='engineering depth/pixel alignment; no physical fidelity claim')
 def build_models(self):
  for key,spec in self.p['models'].items():
   props=read(spec['properties']);collision=[self.x.m.obj_mesh(path)[0] for path in spec['collision_meshes']];table=float(min(v[:,2].min() for v in collision)+props['world_position'][2])
   xml=self.x.m.scene_xml(self.p['hand']['capsule_xml'],spec['object_xml'],props['world_position'],table,spec['dt_s'],self.x.profile);folder=self.raw.root/'models'/key
   self.raw.json(folder/'scene.json',dict(xml_profile='authored capsule + hand gravcomp1; uncompensated original object; explicit imported stock compliance',table_z_m=table,properties=props,original_object_xml_sha256=file_sha(spec['object_xml']),scene_xml_sha256=hashlib.sha256(xml.encode()).hexdigest()),'scene_source_contract')
   path=folder/'scene.xml';path.write_text(xml);self.raw.add(path,'compiled_scene_xml');model=self.e.model(xml)
   # Capture compiled operands before any validation or MjData construction.
   operands={name:np.asarray(getattr(model,name)).copy() for name in self.x.api._COMPILED_ARRAYS};operands.update(object_body_ipos=np.asarray(model.body_ipos).copy(),object_body_mass=np.asarray(model.body_mass).copy(),global_qpos0=np.asarray(model.qpos0).copy(),gravity=np.asarray(model.opt.gravity).copy(),timestep_s=np.asarray(model.opt.timestep),integrator=np.asarray(int(model.opt.integrator)),cone=np.asarray(int(model.opt.cone)),solver=np.asarray(int(model.opt.solver)),solver_iterations=np.asarray(int(model.opt.iterations)),solver_tolerance=np.asarray(model.opt.tolerance))
   model_art=self.raw.arrays(folder/'compiled_model.npz',operands,'compiled_before_validation')
   c=self.x.hand.validate_hand_scene(model,lambda kind,name:self.e.resolve(model,kind,name),self.p['hand']['assets'],np.asarray(self.p['hand']['rest_quaternions']),object_joint_name='pilot_object_free',api=self.x.api);profile=self.x.profile.compensated_model_receipt(model,c.hand);object_receipt=self.x.m.object_model_receipt(model,c.object_body_id,spec['object_xml'],props['world_position'],self.x.api);data=self.e.data(model)
   if not np.allclose(model.qpos0[67:70]+self.x.m.rotation(model.qpos0[70:74])@model.body_ipos[c.object_body_id],props['world_position'],rtol=0,atol=1e-12):raise ValueError('Compiled original object COM/placement mismatch')
   rc=self.x.rec.NativeRecordingContract(assets_sha256='sha256:'+self.p['hand']['assets']['closure_sha256'],shape_sha256='sha256:'+self.p['hand']['expected_shape_sha256'],collider_sha256=self.p['collider_sha256'],controller_sha256=self.p['controller_sha256'],compiled_model_sha256='sha256:'+model_art['sha256'],controller_protocol_id='manosim_native_object_pilot_minjerk_capsule_gravcomp_v1',sampling_protocol_id='native_task_control10ms_physics'+str(spec['dt_s'])+'_v1',control_interval_s=.01)
   self.models[key]=(model,data,c,rc)
   self.raw.json(folder/'model_contract.json',dict(scene=c.metadata(),recording=rc.model_dump(mode='json'),gravity_profile=profile,object_source_receipt=object_receipt,compiled_artifact=model_art),'model_contract')
 def source_checks(self):
  for row in self.p['schedule']:
   self.active=row['attempt_id'];model,data,c,rc=self.models[row['model_id']];a=npz(row['proposal']['path']);goal=self.x.api.target_from_source(a['postAA32'],a['translation32'],a['cameraWorld64'],a['rest_quaternions'],shape_float32=a['shape']).qpos.copy()
   if not np.array_equal(goal,a['native_qpos']):raise ValueError('Saved proposal/native coordinate bytes differ')
   if a['shape'].tobytes()!=np.load(self.p['hand']['shape_asset'],allow_pickle=False).tobytes() or not np.array_equal(a['rest_quaternions'],self.p['hand']['rest_quaternions']):raise ValueError('Proposal literal shape/rest differs from compiled asset cohort')
   aa=a['postAA32'].copy();aa[3:]=0;aa[39:42]=[1,1,0];opened=self.x.api.target_from_source(aa,a['translation32'],a['cameraWorld64'],a['rest_quaternions'],shape_float32=a['shape']).qpos.copy();start=opened.copy();start[:3]+=a['source_rotations_world'][0]@np.array([.03,.03,0.])
   self.e.reset(model,data);data.qpos[:67]=goal;cmd=self.x.hand.command_from_hand_qpos(goal,c,api=self.x.api);self.x.hand.write_hand_controls(data,cmd,c);self.e.forward(model,data);anchors,R=frames(data,c.hand)
   checks=self.x.u.static_checks(data.qpos[:67],goal,anchors,R,a['source_anchors_world'],a['source_rotations_world'],data.actuator_length,cmd.ctrl,data.actuator_force)
   wp,wr,oc=self.x.m.endpoint_pose(data.qpos,a['rest_quaternions'][0],model.body_ipos[c.object_body_id]);checks['endpoint_formula']=bool(np.max(np.abs(wp-np.asarray(data.xpos)[c.hand.body_ids[0]]))<=1e-12 and np.max(np.abs(wr-R[0]))<=1e-12 and np.max(np.abs(oc-np.asarray(data.xipos)[c.object_body_id]))<=1e-12)
   knots=np.stack([self.x.m.target(t,goal,opened,start,self.x.u) for t in np.arange(551)*.01])
   try:self.x.api.require_continuous_principal_targets(knots,c.hand);checks['all_control_chart_knots']=True
   except ValueError:checks['all_control_chart_knots']=False
   checks['depth_visual_alignment']=self.alignment[row['observation_id']]['passed']
   captured=self.x.m.observation(self.p['observations'][row['observation_id']]['path']);camera_operands=self.x.m.source_cast_camera_operands(a['K'],a['cameraWorld64'],captured['camera']['K'],captured['T_world_camera']);checks['capture_camera_and_K32_cast']=bool(camera_operands['capture_camera_and_K32_cast_pass'])
   self.raw.arrays(self.raw.root/self.active/'source_chart_gate.npz',dict(goal=goal,opened=opened,start=start,command_knots=knots,actual_qpos=data.qpos.copy(),actual_anchors=anchors,actual_rotations=R,expected_anchors=a['source_anchors_world'],expected_rotations=a['source_rotations_world'],actual_actuator_length=data.actuator_length.copy(),actual_actuator_force=data.actuator_force.copy(),command_ctrl=cmd.ctrl,**camera_operands),'static_source_chart_gate')
   self.prepared[self.active]=dict(goal=goal,opened=opened,start=start,rest=a['rest_quaternions'],checks=checks,passed=all(checks.values()))
  self.active=None
 def metadata(self,row,rc,*,execution='not_executed',outcome=None,refusal=None):
  h=self.x.rec.canonical_sha256;provenance=dict(code=h(self.p['source_hashes']),configuration=h(self.p['task_configuration']),inputs=h(row),native_pose='sha256:'+row['proposal']['sha256'],label_rule=h(self.x.labels.label_rule_contract(self.x.labels.NativeTaskLabelSpec(physics_interval_s=row['dt_s']))),source_profile=self.p['profile_sha256'],hand_version=self.p['hand_version_sha256'],capture=row['capture_sha256'],scenario=row['scenario_sha256'],seeds=h(row['seeds']))
  mapping=dict(status='ok') if refusal is None else dict(status='refused_guard',code=refusal,message=refusal)
  return self.x.rec.NativeAttemptMetadata(attempt_id=row['attempt_id'],contract_sha256=rc.contract_sha256,object_id=row['object_id'],geometry_group_id=row['geometry_group_id'],observation_group_id=row['observation_id'],proposal_group_id=row['proposal_group_id'],mapping=mapping,execution_status=execution,outcome=outcome,provenance_hashes=provenance)
 def execute(self,row):
  self.active=row['attempt_id'];model,data,c,rc=self.models[row['model_id']];g=self.prepared[self.active];rec=self.x.rec.NativeRecorder(rc);folder=self.raw.root/self.active;rows=[];forces=[];began=time.monotonic()
  if not g['passed']:
   rec.write(folder/'recording',self.metadata(row,rc,refusal='STATIC_SOURCE_CHART_OR_DEPTH_ALIGNMENT_REFUSAL'));self.index_recording(folder/'recording');result=dict(attempt_id=self.active,status='MAPPING_REFUSAL',outcome=None,static_checks=g['checks']);self.results.append(result);self.raw.json(folder/'result.json',result,'attempt_result');return
  try:
   self.e.reset(model,data);data.qpos[:67]=g['start'];cmd=self.x.hand.command_from_hand_qpos(g['start'],c,api=self.x.api);self.x.hand.write_hand_controls(data,cmd,c);self.e.forward(model,data)
   def state():
    wrist,R,obj=self.x.m.endpoint_pose(data.qpos,g['rest'][0],model.body_ipos[c.object_body_id]);return dict(state_time_s=float(data.time),qpos=data.qpos.copy(),qvel=data.qvel.copy(),object_com_world_m=obj,wrist_position_world_m=wrist,wrist_rotation_world=R)
   rows.append(state());initial_object_rotation=self.x.m.rotation(data.qpos[70:74]);dt=row['dt_s'];substeps=int(round(.01/dt));steps=int(round(5.5/dt))
   for k in range(steps):
    if time.monotonic()-began>60:raise RuntimeError('Per-rollout60s hard ceiling')
    t=k*dt
    if k%substeps==0:
     target=self.x.m.target(t,g['goal'],g['opened'],g['start'],self.x.u);cmd=self.x.hand.command_from_hand_qpos(target,c,api=self.x.api);self.x.hand.write_hand_controls(data,cmd,c);requested=cmd.ctrl.copy();control_start=float(data.time);control_phase=self.x.m.phase(t)
    pre_v=data.qvel.copy();pre_q=data.qpos.copy();pre_time=float(data.time);self.e.step(model,data);force=contact_sums(model,data,c,self.e,pre_v,self.raw,self.active,self.witnessed,pre_qpos=pre_q)
    forces.append(dict(force_time_s=pre_time,command_tick_time_s=t,**force,start_qvel=pre_v,start_qacc=data.qacc.copy(),start_qfrc_bias=data.qfrc_bias.copy(),start_qfrc_passive=data.qfrc_passive.copy(),start_qfrc_actuator=data.qfrc_actuator.copy(),start_qfrc_constraint=data.qfrc_constraint.copy(),start_energy=data.energy.copy(),actual_ctrl=data.ctrl.copy()));rows.append(state())
    if k+1==int(round(.25/dt)) and max(f['hand_object_normal_force_N'] for f in forces)<=.001:
     delta=float(np.linalg.norm(rows[-1]['object_com_world_m']-rows[0]['object_com_world_m']));relative=initial_object_rotation.T@self.x.m.rotation(data.qpos[70:74]);angle=float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
     if delta>.001 or angle>np.deg2rad(2):raise QuarantineError('RESET_SUPPORT_STABILITY','Reset support stability guard refused: no-hand-contact object drift/rotation')
    if k+1==int(round(1./dt)):
     wrist_error=float(np.linalg.norm(data.qpos[:3]-g['opened'][:3]));angles=self.x.u.angle_errors(self.x.m.bone_rotations(data.qpos[:67],g['rest']),self.x.m.bone_rotations(g['opened'],g['rest']))
     if wrist_error>.002 or angles.max()>np.deg2rad(2):raise QuarantineError('PRECLOSE_TRACKING','Preclose actual2mm/2deg tracking guard refused')
    if (k+1)%substeps==0:
     end=float(data.time);rec.append(time_s=end,action_interval_start_s=control_start,phase=control_phase,qpos_by_name=dict(zip(self.x.rec.QPOS_NAMES,data.qpos[:67],strict=True)),qvel_by_name=dict(zip(self.x.rec.QVEL_NAMES,data.qvel[:51],strict=True)),requested_ctrl_by_name=dict(zip(self.x.rec.CTRL_NAMES,requested[list(c.hand.actuator_ids)],strict=True)),effective_ctrl_by_name=dict(zip(self.x.rec.CTRL_NAMES,data.ctrl[list(c.hand.actuator_ids)],strict=True)))
   self.e.forward(model,data);self.x.hand.verify_hand_scene(model,c,api=self.x.api)
   st=arrays_from_rows(rows);ft=arrays_from_rows(forces);trace={k:st[k] for k in ('state_time_s','object_com_world_m','wrist_position_world_m','wrist_rotation_world')};trace.update({k:ft[k] for k in ('force_time_s','hand_object_normal_force_N','hand_table_normal_force_N','object_table_normal_force_N')});score=self.x.labels.score_native_task(trace,spec=self.x.labels.NativeTaskLabelSpec(physics_interval_s=dt))
   if score['outcome'] is None:raise RuntimeError('Completed task trace received null scorer outcome')
   self.raw.arrays(folder/'physics_state.npz',st,'endpoint_physics_state');self.raw.arrays(folder/'physics_forces.npz',ft,'preceding_interval_forces');self.raw.json(folder/'label_report.json',score,'privileged_supervision_report')
   rec.write(folder/'recording',self.metadata(row,rc,execution='completed',outcome=score['outcome']));self.index_recording(folder/'recording')
   result=dict(attempt_id=self.active,status='COMPLETED',outcome=score['outcome'],label_rule_sha256=score['label_rule_sha256'],static_checks=g['checks'],final_object_COM_m=rows[-1]['object_com_world_m'].tolist(),final_wrist_position_m=rows[-1]['wrist_position_world_m'].tolist(),final_bone_rotations=self.x.m.bone_rotations(data.qpos[:67],g['rest']).tolist(),actual_steps=steps,elapsed_s=time.monotonic()-began,**FLAGS);self.raw.json(folder/'result.json',result,'attempt_result');self.results.append(result)
  except Exception as error:
   # Keep all available physics rows before propagating the hard stop. A null
   # runtime recording is the only exportable marker for this attempted slot.
   if rows and float(data.time)>rows[-1]['state_time_s'] and np.isfinite(data.qpos).all():rows.append(state())
   if rows:self.raw.arrays(folder/'partial_physics_state.npz',arrays_from_rows(rows),'failed_partial_endpoint_state')
   if forces:self.raw.arrays(folder/'partial_physics_forces.npz',arrays_from_rows(forces),'failed_partial_start_forces')
   recording=folder/'recording'
   if recording.exists():self.raw.relocate(recording,folder/'quarantined_recording')
   for name in ('result.json','label_report.json'):
    if (folder/name).exists():self.raw.relocate(folder/name,folder/('quarantined_'+name))
   failure_rec=self.x.rec.NativeRecorder(rc)
   for saved in rec._rows:failure_rec._rows.append(saved)
   failure_rec.write(recording,self.metadata(row,rc,execution='runtime_error'));self.index_recording(recording)
   if isinstance(error,QuarantineError):
    self.x.hand.verify_hand_scene(model,c,api=self.x.api)
    result=dict(attempt_id=self.active,status='RUNTIME_QUARANTINE',outcome=None,reason_code=error.code,reason=str(error),static_checks=g['checks'],actual_steps=len(rows)-1,partial_control_samples=rec.sample_count,**FLAGS);self.raw.json(folder/'result.json',result,'quarantined_attempt_null_result');self.results.append(result);return
   raise
 def index_recording(self,folder):
  for path in sorted(folder.iterdir()):self.raw.add(path,'native_recording')
 def run(self):
  self.align();self.build_models();self.source_checks()
  if len(self.prepared)!=135 or self.e.counts['steps']!=0:raise RuntimeError('All135 static checks before dynamics required')
  self.e.static_gate=True;self.e.note('ALL135_STATIC_SOURCE_CHART_CHECKS_COMPLETED_BEFORE_ANY_STEP')
  # Prospective identities and contracts become immutable before task outcomes.
  schedule=dict(schema_version='manosim_native_task_schedule_v1',domain=self.x.rec.NATIVE_DOMAIN,source_profile_sha256=self.p['profile_sha256'],hand_version_sha256=self.p['hand_version_sha256'],label_rule_id='manosim_native_grasp_lift_task_labels_v1',label_rule_sha256=digest(self.x.labels.label_rule_contract()),contracts=list({c.contract_sha256:c.model_dump(mode='json') for _,_,_,c in self.models.values() if c.sampling_protocol_id=='native_task_control10ms_physics0.002_v1'}.values()),attempts=[])
  for row in self.p['schedule']:
   if row['cohort']=='primary':schedule['attempts'].append({**row['training_identity'],'contract_sha256':self.models[row['model_id']][3].contract_sha256})
  self.raw.json(self.raw.root/'primary_schedule_before_task.json',schedule,'prospective_primary_training_identity_only')
  categories=sorted({a['category_id'] for a in schedule['attempts']},key=lambda x:hashlib.sha256(('native_category_order_v1:'+x).encode()).hexdigest());folds=[]
  for i,category in enumerate(categories):folds.append(dict(fold_id='held_category_'+category,test_category=category,validation_category=categories[(i+1)%len(categories)],assignments={a['attempt_id']:'test' if a['category_id']==category else 'validation' if a['category_id']==categories[(i+1)%len(categories)] else 'train' for a in schedule['attempts']}))
  self.raw.json(self.raw.root/'category_split_manifest_before_task.json',dict(schema_version='manosim_native_category_splits_v1',schedule=dict(path=str(self.raw.root/'primary_schedule_before_task.json'),sha256='sha256:'+file_sha(self.raw.root/'primary_schedule_before_task.json')),selection_uses_outcomes=False,folds=folds),'prospective_category_roles')
  self.raw.json(self.raw.root/'catalog_seven_fold_manifest_before_task.json',self.x.m.catalog_seven_fold_manifest(self.p['schedule'],self.p['catalog_folds_spec'],self.p['catalog_binding'],self.p['catalog_folds_binding']),'prospective_catalog_object_and_category_roles')
  for row in self.p['schedule']:
   self.execute(row);self.e.note('ATTEMPT '+row['attempt_id']+' '+self.results[-1]['status'])
  byid={r['attempt_id']:r for r in self.results};matched=[]
  for row in self.p['schedule']:
   if row['cohort']!='matched_1ms':continue
   a=byid[row['matched_primary_id']];b=byid[row['attempt_id']];checks={};metrics={}
   if a['status']==b['status']=='COMPLETED':
    metrics=dict(object_COM_m=float(np.linalg.norm(np.asarray(a['final_object_COM_m'])-b['final_object_COM_m'])),wrist_position_m=float(np.linalg.norm(np.asarray(a['final_wrist_position_m'])-b['final_wrist_position_m'])),bone_angle_rad=float(self.x.u.angle_errors(np.asarray(a['final_bone_rotations']),np.asarray(b['final_bone_rotations'])).max()))
    checks=dict(label_exact=(a['outcome']['success'],a['outcome']['failure_type'])==(b['outcome']['success'],b['outcome']['failure_type']),**{k:metrics[k]<=self.x.m.MATCH_LIMITS[k] for k in metrics})
    if not a['outcome']['success'] and checks['label_exact']:metrics['failure_time_s']=abs(a['outcome']['first_failure_time_s']-b['outcome']['first_failure_time_s']);checks['failure_time_s']=metrics['failure_time_s']<=.01
   matched.append(dict(primary=a['attempt_id'],matched=b['attempt_id'],passed=bool(checks and all(checks.values())),checks=checks,metrics=metrics))
  self.raw.json(self.raw.root/'primary_collection_unqualified.json',dict(schema_version='manosim_native_collection_v1',schedule=dict(path=str(self.raw.root/'primary_schedule_before_task.json'),sha256='sha256:'+file_sha(self.raw.root/'primary_schedule_before_task.json')),attempts={r['attempt_id']:dict(path=str(self.raw.root/r['attempt_id']/'recording/attempt.json'),sha256='sha256:'+file_sha(self.raw.root/r['attempt_id']/'recording/attempt.json')) for r in self.p['schedule'] if r['cohort']=='primary'},qualifications={}),'unqualified_primary_collection_no_training_admission')
  self.active=None;return dict(schema_version=SCHEMA+'_measurement',status='COMPLETED',all135_static_before_task=True,attempts=self.results,alignment=self.alignment,matched=matched,matched_numerical_readiness_passed=len(matched)==15 and all(x['passed'] for x in matched),physical_outcome=None,**FLAGS)
