"""Source/transaction mocked controls. No real native model/engine operations."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
c=load('compensated_mock_case',HERE/'physics_case.py');m=c.math;r=load('compensated_mock_runner',HERE/'run_profile.py')
@pytest.fixture(autouse=True)
def prohibit_engine(monkeypatch):
 import mujoco
 def no(*a,**kw):raise AssertionError('Manufactured controls prohibit real native operations')
 for k in ('MjModel','MjData','MjSpec','mj_forward','mj_step','mj_kinematics','mj_collision','mj_integratePos','mj_contactForce'):monkeypatch.setattr(mujoco,k,no)


def test_fixed_budget_includes_all51_approach_knots_before_any_steps():
 plan=c.fixture_plan();assert len(plan)==12 and sum(x['steps'] for x in plan)==9750
 assert r.BUDGET==dict(models=12,data=12,static_forwards=240,steps=9750,contact_force_reads=4500,maximum_seconds=600,outer_seconds=610)
 assert r.method()['approach']['reference_swept_bound'] and 'no continuous' in r.method()['approach']['actual_trajectory_scope']
 assert r.method()['gravity_compensation']['qfrc_bias_feedforward'] is False
 assert not any(r.method()['flags'].values())

def neutral():return np.r_[np.zeros(3),np.tile([1.,0,0,0],16)]


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
  for k,v in dict(models=6,data=6,static_forwards=114).items():engine.counts[k]+=v;engine.per_case[index][k]+=v
  return dict(index=index,prepared=[None]*6)
 def execute(context,engine,notify):
  assert engine.static_gate and events[:2]==[('static',20),('static',28)];index=context['index'];events.append(('dynamic',index))
  for k,v in dict(static_forwards=6,steps=4875).items():engine.counts[k]+=v;engine.per_case[index][k]+=v
  return dict(index=index,measurements=[dict(fixture=f) for f in m.fixture_plan() if f['index']==index])
 fake=SimpleNamespace(prepare_case=prepare,execute_case=execute,math=m)
 monkeypatch.setattr(r,'Engine',E);monkeypatch.setattr(r,'core',lambda:fake)
 code=r.worker(protocol,'f'*40);result=r.read(raw/'result.json')
 if second_static_fails:
  assert code==1 and all(e[0]!='dynamic' for e in events) and result['actual_counts']['steps']==0 and result['remaining_case_indices_NOT_RUN']==[20,28]
  assert result['failure']['message']=='manufactured second-case static refusal'
 else:
  assert code==0 and events==[('static',20),('static',28),('dynamic',20),('dynamic',28)] and result['actual_counts']['steps']==9750


def test_full_worker_identity_inventory_not_empty_completion(tmp_path,monkeypatch):
 case=r.core();p={};v=dict(schema_version=r.SCHEMA+'_worker',protocol_sha256='a'*64,freeze_commit='b'*40,physical_outcome=None,**m.FLAGS,status='COMPLETED',artifacts=[],actual_counts={k:r.BUDGET[k] for k in ('models','data','static_forwards','steps','contact_force_reads')},all12_static_checks_before_dynamics=True,cases=[],fixture_schedule=[])
 with pytest.raises(ValueError,match='2case'):r.check_worker(v,p,'a'*64,'b'*40,0,tmp_path)
 v['cases']=[dict(index=20),dict(index=28)];v['fixture_schedule']=[dict(fixture=f,status='COMPLETED_ASSESSED') for f in m.fixture_plan()]
 with pytest.raises(ValueError,match='80'):r.check_worker(v,p,'a'*64,'b'*40,0,tmp_path)
 v.update(status='HARD_STOP',failure={'message':'retained'})
 r.check_worker(v,p,'a'*64,'b'*40,1,tmp_path)
 with pytest.raises(ValueError):r.check_worker(v,p,'a'*64,'b'*40,0,tmp_path)


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


def test_quotas_refuse_before_fake_engine_or_force_accessor(tmp_path):
 mj,p,path,calls=fake_engine(tmp_path)
 with r.Engine(mj,p,lambda v:None) as engine:
  engine.index=20;model=engine.model(path);data=engine.data(model);engine.static_gate=True
  engine.per_case[20]['steps']=c.BUDGET['steps']
  with pytest.raises(RuntimeError,match='quota_steps'):engine.step(model,data)
  assert not calls and engine.counts['steps']==0
  data.ncon=2;engine.per_case[20]['contact_force_reads']=c.BUDGET['contact_force_reads']
  with pytest.raises(RuntimeError,match='quota_contact_force_reads'):engine.force(model,data,1)
  assert engine.counts['contact_force_reads']==0
  with pytest.raises(RuntimeError,match='contact_force'):engine.force(model,data,2)
