"""Manufactured controls only: no native assets, models, FK or stepping."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,json
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load('manufactured_capsule_math',HERE/'execution_math.py');c=load('manufactured_capsule_case',HERE/'capsule_case.py');r=load('manufactured_capsule_runner',HERE/'run_capsule.py')

@pytest.fixture(autouse=True)
def prohibit_engine(monkeypatch):
 import mujoco
 def no(*a,**kw):raise AssertionError('Manufactured controls prohibit actual native work')
 for k in ('MjModel','MjData','MjSpec','mj_forward','mj_step','mj_kinematics','mj_collision','mj_integratePos','mj_contactForce'):monkeypatch.setattr(mujoco,k,no)

def neutral():return np.r_[np.zeros(3),np.tile([1.,0,0,0],16)]
def test_exact_plan_and_budget_are_prospective():
 p=m.fixture_plan();assert len(p)==12 and sum(x['steps'] for x in p)==10500 and set(x['index'] for x in p)=={20,28}
 assert len({(x['index'],x['kind'],x['dt_s']) for x in p})==12
 assert r.BUDGET['models']==12 and r.BUDGET['static_forwards']==36 and r.BUDGET['steps']==10500
 assert r.method()['all12_static_checks_before_first_step'] is True

def test_quaternion_trajectory_endpoints_and_antipodal_equivalence():
 a=neutral();b=a.copy();a[:3]=[.01,-.01,.01];b[7:11]=m.quat_exp(np.array([.2,-.1,.05]))
 np.testing.assert_allclose(m.trajectory(a,b,0),a,atol=1e-15);np.testing.assert_allclose(m.trajectory(a,b,1),b,atol=1e-15)
 for t in np.linspace(0,1,101):assert np.max(np.abs(np.linalg.norm(m.trajectory(a,b,t)[3:].reshape(16,4),axis=1)-1))<1e-14
 assert m.smooth_fraction(-1,.5)==0 and m.smooth_fraction(1,.5)==1
 assert np.array_equal(m.quat_log(np.array([-1.,0,0,0])),np.zeros(3))

def test_ancestral_named_point_J_all51_columns_against_independent_rigid_motion():
 anchors=np.arange(48).reshape(16,3)*.001;R=np.stack([m.matrix(m.quat_exp(np.array([.03*i,-.01*i,.02*i]))) for i in range(16)]);p=np.array([.1,-.02,.04]);J=m.body_point_J(p,3,anchors,R);h=1e-6
 for k in range(51):
  if k<3:expected=np.eye(3)[:,k]
  else:
   body,axis=divmod(k-3,3)
   if body in (0,1,2,3):
    w=R[body][:,axis];lever=p-anchors[body]
    def rot(t):return lever*np.cos(t)+np.cross(w,lever)*np.sin(t)+w*(w@lever)*(1-np.cos(t))
    expected=(rot(h)-rot(-h))/(2*h)
   else:expected=np.zeros(3)
  np.testing.assert_allclose(J[:,k],expected,rtol=0,atol=1e-10)


def test_analytic_capsule_plane_gap_and_declared_endpoint_normal():
 R=m.matrix(m.quat_exp(np.array([.2,-.1,.3])));n=m.plane_normal(R);size=np.array([.01,.02,0]);center=np.array([.1,.2,.3]);plane=center-n*(.01+.02/np.sqrt(2))
 assert abs(m.capsule_gap(center,R,size,plane,n))<1e-16
 assert abs(m.capsule_gap(center-.0005*n,R,size,plane,n)+.0005)<1e-16
 np.testing.assert_allclose(m.matrix(m.plane_quaternion(n))[:,2],n,atol=1e-14)
 assert abs(n@R[:,2]-1/np.sqrt(2))<1e-14


def tracking_arrays(case,goal,mass=.4):
 n=case['steps'];q=np.repeat(goal[None],n,axis=0);bias=np.tile([0,0,mass*9.81],(n,1));act=np.zeros((n,51));act[:,:3]=bias
 if case['kind']=='authored_gravity_equilibrium':q[:,2]-=mass*9.81/300
 return dict(endpoint_time_s=np.arange(1,n+1)*case['dt_s'],endpoint_qpos=q,bone_angle_error_rad=np.zeros((n,16)),start_qfrc_actuator=act,start_qfrc_passive=np.zeros((n,51)),start_qfrc_bias=np.pad(bias,((0,0),(0,48))))

def test_authored_gravity_sag_is_analytic_not_a_frame_failure():
 case=next(x for x in m.fixture_plan() if x['kind']=='authored_gravity_equilibrium');goal=neutral();a=tracking_arrays(case,goal);result=m.assess(case,a,goal,np.tile(np.eye(3),(16,1,1)),.4,300)
 assert result['status']=='PASS' and abs(result['metrics']['expected_wrist_world_m'][2]+.01308)<1e-15
 a['endpoint_qpos'][:,2]=0
 assert m.assess(case,a,goal,np.tile(np.eye(3),(16,1,1)),.4,300)['status']=='FAIL'

@pytest.mark.parametrize('fault',['wrist','one_bone','gravity_force'])
def test_declared_dynamic_gates_are_not_static5um(fault):
 case=next(x for x in m.fixture_plan() if x['kind']=='authored_gravity_equilibrium');goal=neutral();a=tracking_arrays(case,goal)
 if fault=='wrist':a['endpoint_qpos'][:,0]+=.0021
 elif fault=='one_bone':a['bone_angle_error_rad'][:,15]=np.deg2rad(2.1)
 else:a['start_qfrc_actuator'][:,2]-=.02*.4*9.81
 assert m.assess(case,a,goal,np.tile(np.eye(3),(16,1,1)),.4,300)['status']=='FAIL'


def test_static_source_gate_keeps5um_and_control_semantics():
 q=neutral();anchors=np.zeros((16,3));R=np.tile(np.eye(3),(16,1,1));ctrl=np.arange(51,dtype=float)
 assert all(m.static_checks(q,q,anchors,R,anchors,R,ctrl,ctrl,np.zeros(51)).values())
 moved=anchors.copy();moved[15,0]=6e-6
 assert not m.static_checks(q,q,moved,R,anchors,R,ctrl,ctrl,np.zeros(51))['anchors']
 assert not m.static_checks(q,q,anchors,R,anchors,R,ctrl+1e-9,ctrl,np.zeros(51))['transmission']


def test_fixture_XML_preserves_default_template_native_gains_and_inertials(tmp_path):
 from xml.etree import ElementTree as ET
 assets=''.join(f'<mesh name="mano_bone_{i}" file="m{i}.obj"/>' for i in range(16));bodies=''.join(f'<body name="mano_body_{i}"><joint name="j{i}" type="ball"/><inertial mass=".02" diaginertia=".001 .002 .003"/><geom type="'+('mesh' if i==0 else 'capsule')+'" fromto="0 0 0 0 0 .02" size=".01"/></body>' for i in range(16))
 text=f'<mujoco><default><geom solref=".04 1" solimp=".5 .95 .001"/></default><asset>{assets}</asset><worldbody>{bodies}</worldbody><actuator><position joint="j1" kp="30"/></actuator></mujoco>';p=tmp_path/'base.xml';p.write_text(text)
 for i in range(16):(tmp_path/f'm{i}.obj').write_text('manufactured no compile')
 out=ET.fromstring(c.scene_xml(p,'isolated_capsule_plane',.002));before=ET.fromstring(text)
 assert len(out.findall('./worldbody//geom'))==17 and len(out.findall('.//geom'))==18
 for tag in ('body','joint','inertial','position'):
  assert [a.attrib for a in before.iter(tag)]==[a.attrib for a in out.iter(tag)]
 assert ET.tostring(out.find('./default/geom'))==ET.tostring(before.find('./default/geom'))
 assert out.find('option').get('cone')=='elliptic' and out.find('option/flag').get('energy')=='enable'
 assert sum(g.get('contype')=='1' for g in out.findall('./worldbody//geom'))==2

@pytest.mark.parametrize('reverse',[False,True])
def test_contact_pair_order_analytic_J_force_power_and_clock_operands(monkeypatch,reverse):
 p=np.array([.1,.2,0]);anchors=np.zeros((16,3));R=np.tile(np.eye(3),(16,1,1));normal=np.array([0.,0,1]);B=np.array([[0.,0,1],[1,0,0],[0,1,0]])
 if reverse:B=np.array([[0.,0,-1],[1,0,0],[0,-1,0]])
 sign=-1 if reverse else 1;J=sign*B@m.body_point_J(p,3,anchors,R);f=np.array([.2,.03,-.02]);force=SimpleNamespace(force=lambda *a:np.r_[f,np.zeros(3)])
 contact=SimpleNamespace(geom=np.array([1,0] if reverse else [0,1]),dim=3,frame=B.ravel(),pos=p,dist=0.,efc_address=0,solref=np.array([.04,1]),solimp=np.array([.5,.95,.001,.5,2]),friction=np.array([1.,1.,.005,.005,.0001]))
 model=SimpleNamespace(geom_size=np.array([[1.,1.,.01],[.01,.02,0]]));data=SimpleNamespace(ncon=1,nefc=3,contact=[contact],geom_xmat=np.tile(np.eye(3).ravel(),(2,1)),geom_xpos=np.stack((p,p+[0,0,.03])),efc_J=J,efc_force=f,efc_R=np.ones(3),efc_aref=f,qacc=np.zeros(51),qfrc_constraint=J.T@f)
 monkeypatch.setattr(c,'frames',lambda *a:(anchors,R));v=np.linspace(-.1,.2,51);result=c.contact_sample(model,data,None,None,force,v,1,0)
 assert result['contact_active'] and result['normal_force_N']==.2
 for k in ('contact_J_error','contact_gap_error_m','contact_normal_error','generalized_translation_force_error_N','generalized_angular_force_error_Nm','virtual_power_error_W'):assert result[k]<1e-14


def fake_engine(tmp_path,change=None):
 q=neutral();data=SimpleNamespace(qpos=q,qvel=np.zeros(51),qacc=np.zeros(51),ctrl=np.zeros(51),qfrc_applied=np.zeros(51),xfrc_applied=np.zeros((17,6)),warning=SimpleNamespace(number=np.zeros(8,int)),time=0.,ncon=1)
 model=SimpleNamespace(opt=SimpleNamespace(timestep=.002));calls=[]
 def op(m,d):
  calls.append('engine')
  if change=='warning':d.warning.number[0]+=1
  elif change=='nonfinite':d.qacc[0]=np.nan
  elif change=='ctrl':d.ctrl[0]+=1
  elif change=='load':d.qfrc_applied[0]+=1
  elif change=='clock':d.time+=.003
  else:d.time+=.002
 mj=SimpleNamespace(MjModel=SimpleNamespace(from_xml_path=lambda p:model),MjData=lambda m:data,mj_step=op,mj_forward=lambda m,d:None,mj_collision=lambda *a:None,mj_name2id=lambda *a:1,mj_contactForce=lambda m,d,i,out:out.fill(0))
 raw=tmp_path/'raw';directory=raw/'20'/'manufactured';directory.mkdir(parents=True);path=directory/'fixture.xml';path.write_text('never compiled')
 p=dict(raw_root=str(raw),fixture_xml_sha256={'20/manufactured':r.sha(path)})
 return mj,p,path,calls


def test_all_static_before_any_integration_and_registered_API_only(tmp_path):
 mj,p,path,calls=fake_engine(tmp_path)
 with r.Engine(mj,p,lambda v:None) as engine:
  engine.index=20;model=engine.model(path);data=engine.data(model)
  with pytest.raises(RuntimeError,match='static'):engine.step(model,data)
  assert engine.counts['steps']==0 and not calls
  engine.static_gate=True;engine.step(model,data);assert engine.counts['steps']==1 and calls==['engine']
  with pytest.raises(RuntimeError):mj.mj_collision(model,data)

@pytest.mark.parametrize('fault',['warning','nonfinite','ctrl','load','clock'])
def test_every_actual_step_numerical_guard_fault(tmp_path,fault):
 mj,p,path,calls=fake_engine(tmp_path,fault)
 with r.Engine(mj,p,lambda v:None) as engine:
  engine.index=20;model=engine.model(path);data=engine.data(model);engine.static_gate=True
  with pytest.raises(RuntimeError):engine.step(model,data)
 assert engine.counts['steps']==1 and calls==['engine']


def test_once_creationtime_output_and_unindexed_partial_evidence(tmp_path):
 p=tmp_path/'a.npz';p.write_bytes(b'manufactured');row=r.binding(p);r.verify_outputs([row],tmp_path,True)
 extra=tmp_path/'unexpected.xml';extra.write_text('partial')
 assert r.verify_outputs([row],tmp_path,False)==[r.binding(extra)]
 with pytest.raises(ValueError):r.verify_outputs([row],tmp_path,True)
 extra.unlink();p.write_bytes(b'changed')
 with pytest.raises(ValueError):r.verify_outputs([row],tmp_path,False)

@pytest.mark.parametrize('payload',['{"x":NaN}','{"x":1e400}','{"x":1,"x":2}'])
def test_nonfinite_or_duplicate_manifest_refused(tmp_path,payload):
 p=tmp_path/'m.json';p.write_text(payload)
 with pytest.raises(ValueError):r.read(p)


def test_no_models_reference_generator_training_imports():
 for n in ('torch','manotorch.manolayer','smplx','hug','grasp_failure_prediction.prediction'):
  with pytest.raises(RuntimeError):r.ImportGuard().find_spec(n)
 assert r.ImportGuard().find_spec('numpy') is None

@pytest.mark.parametrize('second_static_fails',[False,True])
def test_both_cases_all12_static_before_any_mocked_execution(tmp_path,monkeypatch,second_static_fails):
 raw=tmp_path/'raw';protocol=tmp_path/'protocol.json';protocol.write_text('{}');p=dict(raw_root=str(raw),cases={'20':{},'28':{}});events=[]
 monkeypatch.setattr(r,'validate',lambda *a:p)
 class E:
  def __init__(self,*a):self.counts={k:0 for k in ('models','data','static_forwards','steps','contact_force_reads')};self.per_case={i:self.counts.copy() for i in (20,28)};self.forbidden=[];self.index=None;self.static_gate=False
  def __enter__(self):return self
  def __exit__(self,*a):pass
 def prepare(index,spec,p,directory,engine,notify):
  assert engine.counts['steps']==0;events.append(('static',index))
  if index==28 and second_static_fails:raise ValueError('manufactured second-case static refusal')
  for k,v in dict(models=6,data=6,static_forwards=12).items():engine.counts[k]+=v;engine.per_case[index][k]+=v
  return dict(index=index,prepared=[None]*6)
 def execute(context,engine,notify):
  assert engine.static_gate and events[:2]==[('static',20),('static',28)];index=context['index'];events.append(('dynamic',index))
  for k,v in dict(static_forwards=6,steps=5250).items():engine.counts[k]+=v;engine.per_case[index][k]+=v
  return dict(index=index,measurements=[dict(fixture=f) for f in m.fixture_plan() if f['index']==index])
 fake=SimpleNamespace(prepare_case=prepare,execute_case=execute,math=m)
 monkeypatch.setattr(r,'Engine',E);monkeypatch.setattr(r,'core',lambda:fake)
 code=r.worker(protocol,'f'*40);result=r.read(raw/'result.json')
 if second_static_fails:
  assert code==1 and all(e[0]!='dynamic' for e in events) and result['actual_counts']['steps']==0 and result['remaining_case_indices_NOT_RUN']==[20,28]
  assert result['failure']['message']=='manufactured second-case static refusal'
 else:
  assert code==0 and events==[('static',20),('static',28),('dynamic',20),('dynamic',28)] and result['actual_counts']['steps']==10500


def test_full_worker_identity_inventory_not_empty_completion(tmp_path,monkeypatch):
 case=r.core();p={};v=dict(schema_version=r.SCHEMA+'_worker',protocol_sha256='a'*64,freeze_commit='b'*40,physical_outcome=None,**m.FLAGS,status='COMPLETED',artifacts=[],actual_counts={k:r.BUDGET[k] for k in ('models','data','static_forwards','steps','contact_force_reads')},all12_static_checks_before_dynamics=True,cases=[],fixture_schedule=[])
 with pytest.raises(ValueError,match='2case'):r.check_worker(v,p,'a'*64,'b'*40,0,tmp_path)
 v['cases']=[dict(index=20),dict(index=28)];v['fixture_schedule']=[dict(fixture=f,status='COMPLETED_ASSESSED') for f in m.fixture_plan()]
 with pytest.raises(ValueError,match='76'):r.check_worker(v,p,'a'*64,'b'*40,0,tmp_path)
 v.update(status='HARD_STOP',failure={'message':'retained'})
 r.check_worker(v,p,'a'*64,'b'*40,1,tmp_path)
 with pytest.raises(ValueError):r.check_worker(v,p,'a'*64,'b'*40,0,tmp_path)


def test_prevalidation_compiler_snapshot_sealed_before_validator_refusal(tmp_path,monkeypatch):
 names=tuple(f'mano_body_{i}' for i in range(16));joints=tuple(f'j{i}' for i in range(19));acts=tuple(f'a{i}' for i in range(51));events=[]
 model=SimpleNamespace(nq=67,nv=51,nu=51,njnt=19,nbody=17,ngeom=17,body_mass=np.arange(17,dtype=float),body_inertia=np.ones((17,3)),body_iquat=np.tile([1.,0,0,0],(17,1)),body_ipos=np.zeros((17,3)),body_pos=np.zeros((17,3)),body_quat=np.tile([1.,0,0,0],(17,1)),body_parentid=np.arange(17))
 keys=tuple(k for k in vars(model) if k.startswith('body_'));assets=dict(body_records=[dict(mass=.025,inertial={'fullinertia':'.001 .002 .003 .0001 .0002 .0003'}) for _ in names],capsule_xml_sha256='a'*64,closure_sha256='b'*64)
 def refuse(*a):
  events.append('validation')
  assert events[0]=='sealed'
  raise ValueError('manufactured inertia validation refusal')
 api=SimpleNamespace(_COMPILED_ARRAYS=keys,BODY_NAMES=names,JOINT_NAMES=joints,ACTUATOR_NAMES=acts,validate_compiled_contract=refuse)
 xml=tmp_path/'fixture.xml';xml.write_text('manufactured XML never compiled');lib=tmp_path/'api.py';lib.write_text('manufactured no engine');monkeypatch.setattr(c,'API',lib)
 def save(path,label,**a):
  events.append('sealed');assert label=='PREVALIDATION_UNINTERPRETED_COMPILED_MODEL_OPERANDS'
  with path.open('xb') as f:np.savez_compressed(f,**a)
 with pytest.raises(ValueError,match='inertia validation'):c.validate_with_snapshot(model,lambda kind,name:({'body':names,'joint':joints,'actuator':acts}[kind].index(name)+1),assets,np.tile([1.,0,0,0],(16,1)),api,save,tmp_path,xml)
 with np.load(tmp_path/'prevalidation_compiled_model.npz',allow_pickle=False) as a:
  np.testing.assert_array_equal(a['named_body_ids'],np.arange(1,17));np.testing.assert_array_equal(a['body_inertia'],model.body_inertia)
  np.testing.assert_array_equal(a['model_dimensions'],[67,51,51,19,17,17]);np.testing.assert_array_equal(a['source_fullinertia6'],np.tile([.001,.002,.003,.0001,.0002,.0003],(16,1)))
  assert not a['validation_performed'].item() and a['fixture_xml_sha256'].item()==c.sha(xml) and a['mapping_controller_source_sha256'].item()==c.sha(lib)
 assert events==['sealed','validation']


def test_archived_source_uses_frozen_blob_not_repaired_live_API(tmp_path,monkeypatch):
 monkeypatch.setattr(r,'ROOT',tmp_path);source=tmp_path/'api.py';source.write_text('new repaired API');old=b'frozen failed API';data=tmp_path/'runtime.dat';data.write_text('unchanged old runtime')
 h=__import__('hashlib').sha256(old).hexdigest();p=dict(active_bindings={str(source):h,str(data):r.sha(data)},selected_git_sources={'api.py':h});calls=[]
 def git(*a):calls.append(a);return old
 monkeypatch.setattr(r,'git',git)
 assert r.archived_closure(p,'oldfreeze')=={str(data):r.sha(data)} and calls==[('show','oldfreeze:api.py')]
 data.write_text('tampered old runtime')
 with pytest.raises(ValueError,match='Active required input changed'):r.archived_closure(p,'oldfreeze')


def test_archived_source_blob_or_partition_tampering_refused(tmp_path,monkeypatch):
 monkeypatch.setattr(r,'ROOT',tmp_path);source=tmp_path/'old.py';source.write_text('old');h=r.sha(source);p=dict(active_bindings={str(source):h},selected_git_sources={'old.py':h});monkeypatch.setattr(r,'git',lambda *a:b'wrong blob')
 with pytest.raises(ValueError,match='Archived selected source'):r.archived_closure(p,'oldfreeze')
 p['selected_git_sources']['old.py']='f'*64
 with pytest.raises(ValueError,match='Archived selected source'):r.archived_closure(p,'oldfreeze')


def manufactured_plane(cache=0):
 normal=np.array([1.,0,1.])/np.sqrt(2);rotation=m.matrix(m.plane_quaternion(normal));center=np.array([.1,.2,.3]);position=center-normal*(.01+.02/np.sqrt(2));size=np.array([[1.,1.,.01],[.01,.02,0]])
 model=SimpleNamespace(geom_bodyid=np.array([0,4]),geom_type=np.array([0,3]),geom_sameframe=np.array([cache,0]),geom_pos=np.stack((position,center)),geom_quat=np.stack((m.plane_quaternion(normal),[1.,0,0,0])),geom_size=size,geom_solref=np.tile([.04,1.],(2,1)),geom_solimp=np.tile([.5,.95,.001,.5,2.],(2,1)),geom_friction=np.tile([1.,.005,.0001],(2,1)),geom_priority=np.zeros(2,int))
 basis=np.stack((normal,[0.,1,0],np.cross(normal,[0.,1,0])));contact=SimpleNamespace(geom=np.array([0,1]),dim=3,efc_address=-1,dist=0.,pos=position.copy(),frame=basis.ravel(),friction=np.array([1.,1.,.005,.005,.0001]),solref=np.array([.04,1]),solimp=np.array([.5,.95,.001,.5,2]))
 data=SimpleNamespace(ncon=1,nefc=0,contact=[contact],geom_xpos=np.stack((position,center)),geom_xmat=np.stack((rotation.ravel(),np.eye(3).ravel())),qpos=neutral(),qvel=np.zeros(51),ctrl=np.zeros(51),qacc=np.zeros(51),efc_J=np.empty((0,51)),efc_force=np.empty(0),efc_R=np.empty(0),efc_aref=np.empty(0),qfrc_constraint=np.zeros(51),warning=SimpleNamespace(number=np.zeros(8,int)),time=0.)
 return model,data,position,m.plane_quaternion(normal)


def test_plane_setter_clears_compiled_cache_and_changes_only_world_plane():
 model,data,p,q=manufactured_plane(cache=3);protected={k:np.asarray(v).copy() for k,v in vars(model).items()};target=p+np.array([.001,0,0])
 c.place_world_plane(model,0,target,q)
 assert model.geom_sameframe[0]==0
 np.testing.assert_array_equal(model.geom_pos[0],target);np.testing.assert_array_equal(model.geom_quat[0],q)
 for key,before in protected.items():
  after=getattr(model,key)
  if key in ('geom_pos','geom_quat','geom_sameframe'):np.testing.assert_array_equal(after[1:],before[1:])
  else:np.testing.assert_array_equal(after,before)
 with pytest.raises(ValueError,match='world plane'):c.place_world_plane(model,1,p,q)
 with pytest.raises(ValueError,match='plane index'):c.place_world_plane(model,-1,p,q)


def plane_operands(model,data):return c.contact_operands(model,data,1,0,pre_time_s=0.,pre_qpos=data.qpos,pre_qvel=data.qvel,reason='MANUFACTURED_STATIC_READBACK')


def test_actual_plane_preflight_requires_realized_frame_and_tangent_pair():
 model,data,p,q=manufactured_plane();a=plane_operands(model,data)
 checks=c.plane_preflight_checks(a,p,q);assert all(checks.values()) and all(type(v) is bool for v in checks.values())
 assert a['actual_geom_sameframe'][1]==0 and a['efc_J'].shape==(0,51) and a['contact_geom'].shape==(1,2)
 assert a['snapshot_schema_version'].item()=='manosim_capsule_actual_contact_operands_v1'
 data.geom_xpos[0,0]+=1
 assert a['actual_geom_xpos'][1,0]!=data.geom_xpos[0,0] # copied operands, never live views


@pytest.mark.parametrize('fault',['BODYROT_cache','ignored_rotation','wrong_position','extra_pair','wrong_pair','engine_gap','normal'])
def test_plane_preflight_rejects_old_cache_and_actual_contact_mismatches(fault):
 model,data,p,q=manufactured_plane()
 if fault=='BODYROT_cache':model.geom_sameframe[0]=3
 elif fault=='ignored_rotation':data.geom_xmat[0]=np.eye(3).ravel()
 elif fault=='wrong_position':data.geom_xpos[0,0]+=.0001
 elif fault=='extra_pair':data.ncon=2;data.contact.append(data.contact[0])
 elif fault=='wrong_pair':data.contact[0].geom[:]=[1,5]
 elif fault=='engine_gap':data.contact[0].dist=-2e-9
 else:data.contact[0].frame=np.eye(3).ravel()
 assert not all(c.plane_preflight_checks(plane_operands(model,data),p,q).values())


def test_penetration_refusal_seals_live_operands_before_throw_without_force_call(tmp_path):
 model,data,p,q=manufactured_plane();normal=m.matrix(q)[:,2];data.geom_xpos[1]-=.006*normal;data.contact[0].dist=-.006;events=[]
 class Force:
  def force(self,*a):raise AssertionError('Penetration witness must not read force API')
 def save(reason):
  events.append('sealed')
  with (tmp_path/'failed_contact_operands.npz').open('xb') as f:np.savez_compressed(f,**c.contact_operands(model,data,1,0,pre_time_s=0.,pre_qpos=data.qpos,pre_qvel=data.qvel,reason=reason))
 with pytest.raises(RuntimeError,match='5mm'):c.contact_sample(model,data,None,None,Force(),np.zeros(51),1,0,refusal_witness=save)
 assert events==['sealed']
 with np.load(tmp_path/'failed_contact_operands.npz',allow_pickle=False) as a:
  assert a['contact_dist'][0]==-.006 and a['ncon']==1 and a['actual_geom_sameframe'][1]==0
  assert a['reason'].item()=='Declared5mm maximum contact penetration escaped'


def test_zero_contact_at_exact_tangent_boundary_is_not_a_physical_refusal():
 model,data,p,q=manufactured_plane();data.ncon=0
 a=plane_operands(model,data);checks=c.plane_preflight_checks(a,p,q)
 assert all(checks.values()) and a['contact_geom'].shape==(0,2)
 assert checks['engine_tangent_gap_if_present'] and checks['contact_normal_if_present']
 data.geom_xpos[1]-=np.array([1.,0,1.])/np.sqrt(2)*2e-9
 assert not c.plane_preflight_checks(plane_operands(model,data),p,q)['analytic_tangent_gap']
