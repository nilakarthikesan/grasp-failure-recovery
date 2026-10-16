"""Manufactured operands only. Never compile or query actual native/object assets."""
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import importlib.util,io,pickle,sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load('test_object_pilot_math',HERE/'pilot_math.py');c=load('test_object_pilot_case',HERE/'pilot_case.py');r=load('test_object_pilot_runner',HERE/'run_pilot.py')
u=load('test_object_pilot_interpolation',HERE.parents[1]/'research_artifacts/manosim_capsule_execution_2026-10-15_v3/execution_math.py')

@pytest.fixture(autouse=True)
def no_actual_engine(monkeypatch):
 import mujoco
 def no(*a,**kw):raise AssertionError('Manufactured controls prohibit actual model/FK/geometry/physics')
 for name in ('MjModel','MjData','MjSpec','mj_forward','mj_step','mj_step1','mj_step2','mj_kinematics','mj_collision','mj_contactForce','mj_integratePos'):monkeypatch.setattr(mujoco,name,no)


def neutral():return np.r_[np.zeros(3),np.tile([1.,0,0,0],16)]

def test_fixed_counts_and_closed_native_admission():
 assert c.COUNTS['models']==c.COUNTS['data']==45 and c.COUNTS['static_forwards']==405 and c.COUNTS['resets']==270
 assert c.COUNTS['steps']==120*2750+15*5500==412500
 assert c.COUNTS['alignment_triangle_point_tests']==60*32*10000
 assert not any(c.FORBIDDEN.values()) and not any(c.FLAGS.values())


def test_source_open_approach_settle_close_lift_are_separate():
 goal=neutral();opened=goal.copy();opened[7:11]=u.quat_exp(np.array([.5,0,0]));start=opened.copy();start[:3]=[.03,.03,0]
 assert np.array_equal(m.target(.25,goal,opened,start,u),start)
 assert np.array_equal(m.target(.75,goal,opened,start,u),opened)
 assert np.array_equal(m.target(.999,goal,opened,start,u),opened)
 assert np.array_equal(m.target(1.,goal,opened,start,u),opened)
 assert np.allclose(m.target(1.5,goal,opened,start,u),goal,atol=1e-15)
 assert m.target(2.5,goal,opened,start,u)[2]==.15 and m.phase(.75)=='approach' and m.phase(.999)=='approach' and m.phase(1.)=='close_fingers' and m.phase(1.49)=='close_fingers' and m.phase(1.5)=='lift'


def test_endpoint_uses_raw_qpos_instead_of_stale_body_transforms():
 q=np.r_[neutral(),[2.,3,4],u.quat_exp(np.array([0,0,np.pi/2]))];q[:3]=[1,2,3];q[3:7]=u.quat_exp(np.array([0,np.pi/3,0]));rest=u.quat_exp(np.array([np.pi/5,0,0]));ipos=np.array([.1,0,0])
 p,R,obj=m.endpoint_pose(q,rest,ipos)
 assert np.array_equal(p,[1,2,3]) and np.allclose(R,u.matrix(u.quat_product(rest,q[3:7])))
 assert np.allclose(obj,[2,3.1,4])
 with pytest.raises(ValueError):m.endpoint_pose(q*2,rest,ipos)


def test_all16_rotation_chain_uses_rest_and_actual_extra_quaternions():
 q=neutral();rest=np.tile([1.,0,0,0],(16,1));q[3:7]=u.quat_exp(np.array([0,0,.3]));rest[1]=u.quat_exp(np.array([.2,0,0]));q[11:15]=u.quat_exp(np.array([0,.4,0]));R=m.bone_rotations(q,rest)
 assert np.allclose(R[2],u.matrix(q[3:7])@u.matrix(rest[1])@u.matrix(q[11:15])) and np.allclose(R[4],u.matrix(q[3:7]))


def obs(depth=1000,mask=True):
 from PIL import Image
 a=np.full((20,20),depth,np.uint16);b=np.full((20,20),255 if mask else 0,np.uint8)
 def png(array):
  f=io.BytesIO();Image.fromarray(array).save(f,format='PNG');return f.getvalue()
 return dict(depth=png(a),object_mask=png(b),camera=dict(width=20,height=20,K=np.array([[100.,0,10],[0,200,10],[0,0,1]])),T_world_camera=np.eye(4),grasp=None)


def test_fixed32_pixel_queries_full_ray_depth_bin_and_half_pixel_diagonal():
 a=m.alignment_inputs(obs());assert a['passed_support'] and len(a['pixel_ids'])==32 and len(set(a['pixel_ids']))==32
 assert a['valid_mask'].sum()==16*16 and np.array_equal(a['pixel_ids'],np.flatnonzero(a['valid_mask'])[np.linspace(0,255,32,dtype=int)])
 assert np.allclose(a['points_camera'],a['camera_rays'])
 expected=.001*np.linalg.norm(a['camera_rays'],axis=1)+.5*np.sqrt(100.**-2+200.**-2)
 assert np.array_equal(a['allowances_m'],expected)


@pytest.mark.parametrize('kind',['no_mask','sentinel','zero','bad_camera','bad_K'])
def test_depth_contract_refuses_or_preserves_support_refusal(kind):
 data=obs(mask=kind!='no_mask',depth=65535 if kind=='sentinel' else 0 if kind=='zero' else 1000)
 if kind=='bad_camera':data['T_world_camera'][0,0]=2
 if kind=='bad_K':data['camera']['K'][0,1]=1
 if kind in ('bad_camera','bad_K'):
  with pytest.raises(ValueError):m.alignment_inputs(data)
 else:assert m.alignment_inputs(data)['passed_support'] is False


def test_independent_point_triangle_query_handles_interior_edges_and_degenerate():
 tri=np.array([[[0.,0,0],[1,0,0],[0,1,0]],[[3,0,0],[3,0,0],[3,0,0]]]);points=np.array([[.2,.2,1],[2,0,0],[3,0,2]])
 ids,p,d=m.nearest_surface(points,tri);assert ids.tolist()==[0,0,1] and np.allclose(p,[[.2,.2,0],[1,0,0],[3,0,0]]) and np.allclose(d,[1,1,2])
 with pytest.raises(ValueError):m.nearest_surface(points*np.nan,tri)


def xmls(tmp_path):
 hand=ET.Element('mujoco');ET.SubElement(hand,'compiler',angle='radian');default=ET.SubElement(hand,'default');ET.SubElement(default,'geom',solref='.04 1',solimp='.5 .95 .001');assets=ET.SubElement(hand,'asset');ET.SubElement(assets,'mesh',name='handmesh',file='hand.obj');world=ET.SubElement(hand,'worldbody')
 for i in range(16):
  b=ET.SubElement(world,'body',name=f'mano_body_{i}');ET.SubElement(b,'inertial',mass='.025',pos='0 0 0',diaginertia='.001 .001 .001');ET.SubElement(b,'joint',name=f'joint{i}')
  if i==0:
   for j in range(3):ET.SubElement(b,'joint',name=f'slide{j}')
  ET.SubElement(b,'geom',name=f'handgeom{i}',type='mesh',mesh='handmesh')
 act=ET.SubElement(hand,'actuator');ET.SubElement(act,'position',kp='300',joint='slide0')
 obj=ET.fromstring('<mujoco><asset><mesh name="visual" file="visual.obj"/><mesh name="collision" file="collision.obj"/></asset><worldbody><body name="source" pos="0 0 0"><inertial pos="0 0 0" mass=".2" fullinertia=".001 .002 .003 0 0 0"/><freejoint/><geom type="mesh" mesh="visual" contype="0" conaffinity="0"/><geom type="mesh" mesh="collision" friction=".8 .005 .0001"/></body></worldbody></mujoco>')
 hp=tmp_path/'hand.xml';op=tmp_path/'object.xml';hp.write_text(ET.tostring(hand,encoding='unicode'));op.write_text(ET.tostring(obj,encoding='unicode'));return hp,op


def test_xml_preserves_inertia_and_gains_and_explicitly_separates_masks_and_compliance(tmp_path):
 hp,op=xmls(tmp_path);profile=c.load('mock_pilot_profile',HERE.parents[1]/'research_artifacts/manosim_native_physics_profile_2026-10-16_v2/profile_math.py');root=ET.fromstring(m.scene_xml(hp,op,[1.,2,3],2.9,.002,profile))
 assert root.find('./default/geom').get('solref')=='.04 1'
 assert root.find('./worldbody/body[@name="pilot_object"]/inertial').get('fullinertia')=='.001 .002 .003 0 0 0'
 assert root.find('./actuator/position').get('kp')=='300'
 hand=root.find('./worldbody/body[@name="mano_body_0"]');obj=root.find('./worldbody/body[@name="pilot_object"]');g=obj.findall('geom')[1]
 assert hand.get('gravcomp')=='1' and obj.get('gravcomp')=='0' and obj.get('pos')=='1.0 2.0 3.0'
 assert (g.get('contype'),g.get('conaffinity'))==('1','6') and g.get('friction')=='.8 .005 .0001' and g.get('solref')=='.02 1' and g.get('solimp')=='.9 .95 .001 .5 2'
 assert root.find('./option').get('cone')=='elliptic' and root.find('./option/flag').get('energy')=='enable'
 assert Path(root.find('./asset/mesh[@name="pilot_object__collision"]').get('file')).is_absolute()


@pytest.mark.parametrize('mutation',['nested','body_rotation','geom_rotation','mesh_scale','inertial_offset'])
def test_no_silent_erasure_of_source_object_frames(tmp_path,mutation):
 hp,op=xmls(tmp_path);root=ET.parse(op).getroot();body=root.find('./worldbody/body')
 if mutation=='nested':ET.SubElement(body,'body')
 elif mutation=='body_rotation':body.set('quat','.5 .5 .5 .5')
 elif mutation=='geom_rotation':body.find('geom').set('quat','.5 .5 .5 .5')
 elif mutation=='mesh_scale':root.find('./asset/mesh').set('scale','2 1 1')
 else:body.find('inertial').set('pos','.1 0 0')
 op.write_text(ET.tostring(root,encoding='unicode'));profile=SimpleNamespace(gravity_profile_xml=lambda text:text)
 with pytest.raises(ValueError):m.scene_xml(hp,op,[0,0,0],0,.002,profile)


def test_unpickler_rejects_executable_globals_and_labeled_observation(tmp_path):
 path=tmp_path/'p.pkl';path.write_bytes(pickle.dumps({'grasp':{'fake':True}}))
 with pytest.raises(ValueError):m.observation(path)
 path.write_bytes(pickle.dumps(Path('/tmp')))
 with pytest.raises(ValueError):m.observation(path)


def test_creation_time_arrays_once_only_and_unindexed_or_tampered_output(tmp_path):
 a=c.Artifacts(tmp_path/'raw');a.arrays(a.root/'x.npz',{'x':np.ones(2)},'manufactured');a.verify()
 with pytest.raises(ValueError):a.arrays(a.root/'x.npz',{'x':np.ones(2)},'again')
 (a.root/'unexpected').write_text('bad')
 with pytest.raises(ValueError):a.verify()
 (a.root/'unexpected').unlink();(a.root/'x.npz').write_bytes(b'changed')
 with pytest.raises(ValueError):a.verify()


@pytest.mark.parametrize('text',['{"a":1e400}','{"a":NaN}','{"a":1,"a":2}'])
def test_nonfinite_and_duplicate_json_fails_closed(tmp_path,text):
 path=tmp_path/'a.json';path.write_text(text)
 with pytest.raises(ValueError):r.read(path)


def fake_engine(tmp_path,fault=None):
 q=np.r_[neutral(),[0,0,0,1,0,0,0]];d=SimpleNamespace(qpos=q,qvel=np.zeros(57),qacc=np.zeros(57),ctrl=np.zeros(51),qfrc_bias=np.zeros(57),qfrc_passive=np.zeros(57),qfrc_actuator=np.zeros(57),qfrc_constraint=np.zeros(57),qfrc_applied=np.zeros(57),xfrc_applied=np.zeros((18,6)),time=0.,warning=SimpleNamespace(number=np.zeros(8,int)))
 model=SimpleNamespace(opt=SimpleNamespace(timestep=.002));calls=[]
 def step(m,data):
  calls.append('step');data.time+=.002
  if fault=='warning':data.warning.number[0]+=1
  elif fault=='NaN':data.qacc[0]=np.nan
  elif fault=='clock':data.time+=.001
  elif fault=='ctrl':data.ctrl[0]=1
  elif fault=='load':data.qfrc_applied[0]=1
  elif fault=='quat':data.qpos[3:7]*=2
 mj=SimpleNamespace(MjModel=SimpleNamespace(from_xml_string=lambda x:model),MjData=lambda m:d,mj_step=step,mj_forward=lambda *a:None,mj_resetData=lambda *a:None,mj_contactForce=lambda m,d,i,out:out.fill(0))
 e=r.Engine(mj,{'budget':{**c.COUNTS,**r.BUDGET}},c.Artifacts(tmp_path/'raw'));return e,model,d,calls


def test_global_static_gate_and_quota_reject_before_fake_integration(tmp_path):
 e,model,d,calls=fake_engine(tmp_path)
 with pytest.raises(RuntimeError,match='135'):e.step(model,d)
 assert not calls and e.counts['steps']==0
 e.static_gate=True;e.counts['steps']=c.COUNTS['steps']
 with pytest.raises(RuntimeError,match='quota'):e.step(model,d)
 assert not calls


@pytest.mark.parametrize('fault',['warning','NaN','clock','ctrl','load','quat'])
def test_per_actual_step_guards_and_failure_operand_preservation(tmp_path,fault):
 e,model,d,calls=fake_engine(tmp_path,fault);e.static_gate=True
 with pytest.raises((ValueError,RuntimeError)):e.step(model,d)
 assert calls==['step'] and e.counts['steps']==1 and len(e.raw.rows)==1 and Path(e.raw.rows[0]['path']).is_file()


def test_blocked_model_inference_and_training_imports():
 for name in ('torch','manotorch.manolayer','smplx','hug.inference','transformers'):
  with pytest.raises(RuntimeError):r.ImportGuard().find_spec(name)
 assert r.ImportGuard().find_spec('numpy') is None


def test_failed_physics_receipt_never_permits_object_pilot(tmp_path,monkeypatch):
 p=tmp_path/'p.json';report=tmp_path/'r.json';receipt=tmp_path/'i.json'
 r.write(p,{});r.write(report,dict(schema_version=r.PHYSICS_REPORT_SCHEMA,status='COMPLETED',qualification='NOT_QUALIFIED'));r.write(receipt,{})
 with pytest.raises(ValueError):r.physical_evidence(p,report,receipt,r.sha(receipt))


def test_null_metadata_retains_contract_and_never_promotes_refusal(tmp_path):
 x=c.modules();sha='sha256:'+'a'*64;contract=x.rec.NativeRecordingContract(assets_sha256=sha,shape_sha256=sha,collider_sha256=sha,controller_sha256=sha,compiled_model_sha256=sha,controller_protocol_id='testv1',sampling_protocol_id='testv1',control_interval_s=.01)
 row=dict(attempt_id='a',object_id='o',geometry_group_id='g',observation_id='c',proposal_group_id='p',proposal={'sha256':'a'*64},capture_sha256=sha,scenario_sha256=sha,seeds={'generator':42,'physics':0,'sensor':0},dt_s=.002)
 p=dict(source_hashes={},task_configuration={},profile_sha256=sha,hand_version_sha256=sha);pilot=c.Pilot(p,c.Artifacts(tmp_path/'raw'),SimpleNamespace())
 meta=pilot.metadata(row,contract,refusal='DEPTH_ALIGNMENT_FAIL');assert meta.outcome is None and meta.execution_status=='not_executed' and meta.mapping.message
 rec=x.rec.NativeRecorder(contract);out=tmp_path/'recording';rec.write(out,meta);loaded=x.rec.load_native_attempt(out)
 assert loaded[3] is None and loaded[0].outcome is None and not loaded[2]['native_admission']['admitted']


def manufactured_plan(tmp_path,monkeypatch):
 asset=tmp_path/'assets';generation=tmp_path/'generation';generation.mkdir();monkeypatch.setattr(r,'ASSET_ROOT',asset);monkeypatch.setattr(r,'GENERATION',generation/'report.json');objects=[];rows=[];i=0
 for cat in range(5):
  for size in ('small','medium','large'):
   for replica in ('A','B'):
    object_id=f'test/{size}/{cat}{replica}';directory=asset/object_id/'sim_assets';(directory/'collision').mkdir(parents=True);(directory/'collision/a.obj').write_text('no query');(directory/'object.xml').write_text('source object '+object_id);r.write(directory/'properties.json',{'world_position':[1.,2,3]})
    objects.append(dict(object_id=object_id,official_geometry_category=str(cat),official_size_bin=size,cell_replica=replica,geometry_asset_files=[{'dataset_path':object_id+'/collision.obj'}]))
    for capture in ('00000','00001'):
     path=asset/object_id/(capture+'.pkl');path.write_bytes(b'synthetic PNG observation '+capture.encode())
     for seed in (42,43):
      output=generation/(str(i)+'.npz');output.write_bytes(b'manufactured proposal '+str(i).encode());rows.append(dict(object_path=object_id,observation=str(path.relative_to(asset)),seed=seed,path=str(output),sha256=r.sha(output),input_sha256=r.sha(path),id=str(i),grasp_label_available=False));i+=1
 return rows,{'objects':objects}


def test_exact120_primary_15_prospective_matched_and_no_outcome_selection(tmp_path,monkeypatch):
 rows,catalog=manufactured_plan(tmp_path,monkeypatch);schedule,models,observations=r.plan(rows,catalog,{})
 assert len(schedule)==135 and len(models)==45 and len(observations)==60
 matched=[x for x in schedule if x['cohort']=='matched_1ms'];assert len(matched)==15
 for row in matched:
  source=next(x for x in schedule if x['attempt_id']==row['matched_primary_id'])
  assert row['replica']=='A' and row['seeds']['generator']==42 and row['proposal']==source['proposal'] and row['dt_s']==.001 and source['dt_s']==.002
  assert observations[row['observation_id']]['path'].endswith('00000.pkl') and row['scenario_sha256']!=source['scenario_sha256']
 rows[0]['seed']=44
 with pytest.raises(ValueError):r.plan(rows,catalog,{})


def test_real_native_refusal_writer_to_training_readiness_preserves_all_denominators(tmp_path,monkeypatch):
 """Software-only plan→135 refusals→120primary loader; no fit or physics."""
 rows,catalog=manufactured_plan(tmp_path,monkeypatch);schedule,models,observations=r.plan(rows,catalog,{})
 x=c.modules();h='sha256:'+'a'*64
 folds=manufactured_folds(schedule)
 p=dict(catalog_folds_spec=folds,catalog_binding={'sha256':'a'*64},catalog_folds_binding={'sha256':'b'*64},schedule=schedule,models=models,observations=observations,profile_sha256=h,hand_version_sha256=h,source_hashes={},task_configuration={})
 e=SimpleNamespace(counts={'steps':0},static_gate=False,note=lambda *a:None)
 class SyntheticRefusals(c.Pilot):
  def align(self):self.events=['alignment']
  def build_models(self):
   self.events.append('models')
   for key,spec in self.p['models'].items():
    contract=x.rec.NativeRecordingContract(assets_sha256=h,shape_sha256=h,collider_sha256=h,controller_sha256=h,compiled_model_sha256=c.digest(key),controller_protocol_id='synthetic_only',sampling_protocol_id='native_task_control10ms_physics'+str(spec['dt_s'])+'_v1',control_interval_s=.01)
    self.models[key]=(None,None,None,contract)
  def source_checks(self):self.events.append('all135_static');self.prepared={row['attempt_id']:{'passed':False,'checks':{'manufactured_static_refusal':False}} for row in schedule}
  def execute(self,row):
   assert self.events==['alignment','models','all135_static'] and self.e.static_gate and self.e.counts['steps']==0
   return super().execute(row)
 pilot=SyntheticRefusals(p,c.Artifacts(tmp_path/'raw'),e);result=pilot.run();pilot.raw.verify()
 assert len(result['attempts'])==135 and not result['matched_numerical_readiness_passed'] and all(row['outcome'] is None for row in result['attempts'])
 from grasp_failure_prediction.prediction import manosim_training as training
 collection=pilot.raw.root/'primary_collection_unqualified.json';data=training.load_data(collection)
 assert data.readiness['scheduled']==120 and data.readiness['excluded']==120 and data.readiness['eligible']==0 and not data.readiness['scientific_training_allowed']
 split=pilot.raw.root/'category_split_manifest_before_task.json';assert r.read(split)==training.freeze_splits(pilot.raw.root/'primary_schedule_before_task.json')
 assert all(fold['status'].startswith('SKIPPED') for fold in training.split_indices(data,split))


@pytest.mark.parametrize('value',[{'exit_code':1,'integration_calls':0},{'exit_code':0,'integration_calls':1},{}])
def test_failed_or_nonzero_integration_controls_cannot_prepare(tmp_path,value):
 path=tmp_path/'controls.json';r.write(path,value)
 with pytest.raises(ValueError):r.check_controls(path)


def test_full_native_API_and_callback_bypass_guards(tmp_path):
 e,model,data,calls=fake_engine(tmp_path)
 for name in ('mj_step','mj_forward','MjModel','MjData'):
  with pytest.raises(RuntimeError):getattr(e.mj,name)(None,None)
 assert e.forbidden['unregistered_engine_calls']==4 and not calls


def test_actual_principal_chart_jump_is_explicit_quarantine_with_raw_state(tmp_path):
 e,model,d,calls=fake_engine(tmp_path);e.static_gate=True;d.qpos[3:7]=u.quat_exp(np.array([np.pi-1e-4,0,0]))
 original=e.original['mj_step']
 def jump(m,d):original(m,d);d.qpos[3:7]=u.quat_exp(np.array([np.pi+1e-4,0,0]))
 e.original['mj_step']=jump
 with pytest.raises(r.core().QuarantineError) as error:e.step(model,d)
 assert error.value.code=='ACTUAL_PRINCIPAL_CHART' and len(e.raw.rows)==1
 a=c.npz(e.raw.rows[0]['path']);assert np.array_equal(a['preceding_qpos'][3:7],u.quat_exp(np.array([np.pi-1e-4,0,0])))


def test_penetration_witness_precedes_explicit_runtime_quarantine(tmp_path):
 contact=SimpleNamespace(geom=np.array([0,1]),dist=-.006,efc_address=0,dim=3,pos=np.array([.1,0,0]),frame=np.eye(3).ravel(),friction=np.ones(5),solref=np.array([.02,1]),solimp=np.array([.9,.95,.001,.5,2]))
 model=SimpleNamespace(geom_bodyid=np.array([1,17,0]),geom_size=np.ones((3,3)),geom_type=np.array([3,7,0]),opt=SimpleNamespace(timestep=.002))
 data=SimpleNamespace(ncon=1,contact=[contact],nefc=3,efc_J=np.zeros((3,57)),qacc=np.zeros(57),geom_xpos=np.zeros((3,3)),geom_xmat=np.tile(np.eye(3).ravel(),(3,1)),xpos=np.zeros((18,3)),xmat=np.tile(np.eye(3).ravel(),(18,1)),xanchor=np.zeros((20,3)),efc_force=np.ones(3),efc_R=np.ones(3),efc_aref=np.zeros(3),time=.002)
 contract=SimpleNamespace(hand=SimpleNamespace(body_ids=tuple(range(1,17)),joint_ids=tuple(range(19))),object_body_id=17)
 engine=SimpleNamespace(resolve=lambda *a:2,force=lambda *a:np.array([2.,0,0,0,0,0]));raw=c.Artifacts(tmp_path/'raw');q=np.r_[neutral(),[0,0,0,1,0,0,0]]
 with pytest.raises(c.QuarantineError) as error:c.contact_sums(model,data,contract,engine,np.zeros(57),raw,'a',set(),pre_qpos=q)
 assert error.value.code=='CONTACT_PENETRATION' and len(raw.rows)==1
 a=c.npz(raw.rows[0]['path']);assert a['contact_distance_m']==-.006 and a['start_hand_anchors'].shape==(16,3) and a['start_object_body_rotation'].shape==(3,3) and np.array_equal(a['start_qpos'],q) and a['efc_J'].shape==(3,57)


def test_expected_guard_continues_next_independent_reset_unexpected_bug_aborts(tmp_path):
 x=c.modules();h='sha256:'+'a'*64;contract=x.rec.NativeRecordingContract(assets_sha256=h,shape_sha256=h,collider_sha256=h,controller_sha256=h,compiled_model_sha256=h,controller_protocol_id='synthetic_only',sampling_protocol_id='native_task_control10ms_physics0.002_v1',control_interval_s=.01)
 q=np.r_[neutral(),[0.,0,0,1,0,0,0]];data=SimpleNamespace(qpos=q,qvel=np.zeros(57),ctrl=np.zeros(51),time=0.);model=SimpleNamespace(body_ipos=np.zeros((18,3)));hand=SimpleNamespace(actuator_ids=tuple(range(51)),model_signature='mock');hc=SimpleNamespace(hand=hand,object_body_id=17);resets=[];stepcalls=[]
 def reset(m,d):resets.append(True);d.time=0;d.qpos[:]=q;d.qvel[:]=0
 def step(m,d):
  stepcalls.append(True)
  if len(stepcalls)==1:raise c.QuarantineError('PRECLOSE_TRACKING','manufactured expected tracking quarantine')
  raise RuntimeError('manufactured unexpected numerical bug')
 engine=SimpleNamespace(reset=reset,forward=lambda *a:None,step=step)
 p=dict(source_hashes={},task_configuration={},profile_sha256=h,hand_version_sha256=h);pilot=c.Pilot(p,c.Artifacts(tmp_path/'raw'),engine)
 fake_hand=SimpleNamespace(command_from_hand_qpos=lambda *a,**kw:SimpleNamespace(ctrl=np.zeros(51)),write_hand_controls=lambda d,cmd,c:d.ctrl.fill(0),verify_hand_scene=lambda *a,**kw:True)
 pilot.x=SimpleNamespace(**{**x.__dict__,'hand':fake_hand});pilot.models={'model':(model,data,hc,contract)}
 base=dict(object_id='o',geometry_group_id='g',observation_id='obs',proposal_group_id='proposal',proposal={'sha256':'a'*64},capture_sha256=h,scenario_sha256=h,seeds={'generator':42,'physics':0,'sensor':0},dt_s=.002,model_id='model')
 for aid in ('a','b'):pilot.prepared[aid]=dict(passed=True,checks={'manufactured':True},goal=neutral(),opened=neutral(),start=neutral(),rest=np.tile([1.,0,0,0],(16,1)))
 pilot.execute(dict(base,attempt_id='a'));assert pilot.results[0]['status']=='RUNTIME_QUARANTINE' and pilot.results[0]['outcome'] is None
 with pytest.raises(RuntimeError,match='unexpected'):pilot.execute(dict(base,attempt_id='b'))
 assert len(resets)==2 and len(stepcalls)==2
 for aid in ('a','b'):
  meta,_,payload,arrays=x.rec.load_native_attempt(pilot.raw.root/aid/'recording');assert meta.execution_status=='runtime_error' and meta.outcome is None and arrays is None


def test_late_completed_publication_failure_withdraws_label_and_native_markers(tmp_path,monkeypatch):
 x=c.modules();h='sha256:'+'a'*64;contract=x.rec.NativeRecordingContract(assets_sha256=h,shape_sha256=h,collider_sha256=h,controller_sha256=h,compiled_model_sha256=h,controller_protocol_id='synthetic_only',sampling_protocol_id='native_task_control10ms_physics0.002_v1',control_interval_s=.01)
 q=np.r_[neutral(),[0.,0,0,1,0,0,0]];d=SimpleNamespace(qpos=q.copy(),qvel=np.zeros(57),ctrl=np.zeros(51),qacc=np.zeros(57),qfrc_bias=np.zeros(57),qfrc_passive=np.zeros(57),qfrc_actuator=np.zeros(57),qfrc_constraint=np.zeros(57),energy=np.zeros(2),time=0.);model=SimpleNamespace(body_ipos=np.zeros((18,3)));hc=SimpleNamespace(hand=SimpleNamespace(actuator_ids=tuple(range(51)),model_signature='mock'),object_body_id=17)
 def reset(m,d):d.time=0;d.qpos[:]=q;d.qvel[:]=0
 def step(m,d):d.time+=.002
 engine=SimpleNamespace(reset=reset,forward=lambda *a:None,step=step)
 p=dict(source_hashes={},task_configuration={},profile_sha256=h,hand_version_sha256=h);pilot=c.Pilot(p,c.Artifacts(tmp_path/'raw'),engine)
 fake_hand=SimpleNamespace(command_from_hand_qpos=lambda *a,**kw:SimpleNamespace(ctrl=np.zeros(51)),write_hand_controls=lambda d,cmd,c:d.ctrl.fill(0),verify_hand_scene=lambda *a,**kw:True)
 pilot.x=SimpleNamespace(**{**x.__dict__,'hand':fake_hand});pilot.models={'model':(model,d,hc,contract)};pilot.prepared={'a':dict(passed=True,checks={'manufactured':True},goal=neutral(),opened=neutral(),start=neutral(),rest=np.tile([1.,0,0,0],(16,1)))}
 row=dict(attempt_id='a',object_id='o',geometry_group_id='g',observation_id='obs',proposal_group_id='proposal',proposal={'sha256':'a'*64},capture_sha256=h,scenario_sha256=h,seeds={'generator':42,'physics':0,'sensor':0},dt_s=.002,model_id='model')
 monkeypatch.setattr(c,'contact_sums',lambda *a,**kw:dict(hand_object_normal_force_N=0.,hand_table_normal_force_N=0.,object_table_normal_force_N=0.,maximum_penetration_m=0.,contact_count=0))
 original=x.rec.NativeRecorder.write
 def fail_after_write(self,path,metadata):
  out=original(self,path,metadata)
  if metadata.execution_status=='completed':raise OSError('manufactured late publication error')
  return out
 monkeypatch.setattr(x.rec.NativeRecorder,'write',fail_after_write)
 with pytest.raises(OSError,match='late publication'):pilot.execute(row)
 folder=pilot.raw.root/'a';meta,_,_,arrays=x.rec.load_native_attempt(folder/'recording')
 assert meta.execution_status=='runtime_error' and meta.outcome is None and arrays is not None
 assert not (folder/'label_report.json').exists() and (folder/'quarantined_label_report.json').is_file() and (folder/'quarantined_recording/attempt.json').is_file() and not (folder/'result.json').exists()
 pilot.raw.verify()


def manufactured_folds(schedule):
 objects=sorted({r['object_id'] for r in schedule});folds=[]
 for i in range(7):
  ordered=objects[i:]+objects[:i];folds.append(dict(fold_id='manufactured_'+str(i),train_object_ids=ordered[:10],validation_object_ids=ordered[10:15],test_object_ids=ordered[15:],generalization_claim='SOFTWARE_ONLY'))
 return dict(schema_version='hug_native_small_pilot_object_category_fold_spec_v1',catalog_sha256='a'*64,training_performed=False,no_labels_loaded=True,folds=folds)


def test_catalog_seven_folds_join_physical_object_and_exclude_matched_rows(tmp_path,monkeypatch):
 rows,catalog=manufactured_plan(tmp_path,monkeypatch);schedule,_,_=r.plan(rows,catalog,{});spec=manufactured_folds(schedule);out=m.catalog_seven_fold_manifest(schedule,spec,{'sha256':'a'*64},{'sha256':'b'*64})
 assert len(out['folds'])==7 and out['trainer_five_category_manifest_unchanged'] and not out['seven_fold_training_API_implemented']
 for fold in out['folds']:
  assert len(fold['assignments'])==120 and all(not key.startswith('matched') for key in fold['assignments'])
  for obj in {r['object_id'] for r in schedule}:assert len({fold['assignments'][row['attempt_id']] for row in schedule if row['cohort']=='primary' and row['object_id']==obj})==1
 spec['folds'][0]['test_object_ids'][0]=spec['folds'][0]['train_object_ids'][0]
 with pytest.raises(ValueError):m.catalog_seven_fold_manifest(schedule,spec,{'sha256':'a'*64},{'sha256':'b'*64})
