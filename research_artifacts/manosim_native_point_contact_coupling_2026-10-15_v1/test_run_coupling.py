"""Mocked source/freeze/output/API controls; no native model construction."""
import importlib.util,sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('manufactured_coupling_runner',HERE/'run_coupling.py');r=importlib.util.module_from_spec(s);sys.modules[s.name]=r;s.loader.exec_module(r)

@pytest.fixture(autouse=True)
def no_engine(monkeypatch):
 import mujoco
 def no(*a,**kw):raise AssertionError('Real engine is prohibited by controls')
 for name in ('MjModel','MjData','mj_forward','mj_step','mj_fullM','mj_contactForce'):monkeypatch.setattr(mujoco,name,no)


def fake_engine(tmp_path,change=None):
 model=SimpleNamespace();data=SimpleNamespace(qpos=np.zeros(67),qvel=np.zeros(51),ctrl=np.zeros(51),qacc=np.zeros(51),qfrc_applied=np.zeros(51),xfrc_applied=np.zeros((17,6)),warning=SimpleNamespace(number=np.zeros(8,int)),time=0.)
 calls=[]
 def forward(m,d):
  calls.append('forward')
  if change=='warning':d.warning.number[0]=1
  elif change=='nonfinite':d.qacc[0]=np.nan
  elif change=='clock':d.time=.1
  elif change=='state':d.qpos[0]+=1
  elif change=='force':d.qfrc_applied[0]+=1
 def fullM(m,dst,packed):calls.append('fullM');dst[:]=np.eye(51)
 mj=SimpleNamespace(MjModel=SimpleNamespace(from_xml_path=lambda p:(calls.append('compile') or model)),MjData=lambda m:(calls.append('data') or data),mj_forward=forward,mj_fullM=fullM,mj_step=lambda *a:pytest.fail('No step'),mj_collision=lambda *a:pytest.fail('No query'),mj_name2id=lambda *a:1)
 x=tmp_path/'fixture.xml';x.write_text('<mujoco/>');p=dict(fixture_xml_sha256={'20':r.sha(x)},budget=dict(model_compilations=1,data_constructions=1,static_forward_calls=24,full_mass_matrix_calls=12))
 return mj,p,x,calls


def test_actual_public_API_guard_calls_only_registered_four_paths(tmp_path):
 mj,p,x,calls=fake_engine(tmp_path)
 with r.EngineGuard(mj,p,20,lambda v:None) as g:
  with pytest.raises(RuntimeError):mj.mj_collision(None,None)
  m=mj.MjModel.from_xml_path(str(x));d=mj.MjData(m);mj.mj_forward(m,d);out=np.zeros((51,51));mj.mj_fullM(m,out,np.zeros(100))
  assert np.array_equal(out,np.eye(51))
  with pytest.raises(RuntimeError):mj.MjData(m)
 assert calls==['compile','data','forward','fullM'] and g.counts==dict(model_compilations=1,data_constructions=1,static_forward_calls=1,full_mass_matrix_calls=1)

@pytest.mark.parametrize('fault',['warning','nonfinite','clock','state','force'])
def test_every_paired_static_forward_guard_fault_refused(tmp_path,fault):
 mj,p,x,calls=fake_engine(tmp_path,fault)
 with r.EngineGuard(mj,p,20,lambda v:None) as g:
  m=mj.MjModel.from_xml_path(str(x));d=mj.MjData(m)
  with pytest.raises(RuntimeError):mj.mj_forward(m,d)
 assert g.counts['static_forward_calls']==1


def test_engine_quota_denies_before_actual_work(tmp_path):
 mj,p,x,calls=fake_engine(tmp_path)
 with r.EngineGuard(mj,p,20,lambda v:None) as g:
  m=mj.MjModel.from_xml_path(str(x))
  with pytest.raises(RuntimeError):mj.MjModel.from_xml_path(str(x))
 assert calls==['compile'] and g.counts['model_compilations']==1


def test_output_creationtime_hash_and_unindexed_numeric_refusal(tmp_path):
 x=tmp_path/'a.npz';np.savez_compressed(x,a=np.arange(3));a=r.binding(x);r.verify_outputs([a],tmp_path)
 np.savez_compressed(tmp_path/'extra.npz',b=[1])
 with pytest.raises(ValueError):r.verify_outputs([a],tmp_path)
 (tmp_path/'extra.npz').unlink();x.write_bytes(b'changed')
 with pytest.raises(ValueError):r.verify_outputs([a],tmp_path)


def test_source_and_training_imports_blocked():
 for name in ('manotorch.manolayer','hug','transformers','grasp_failure_prediction.prediction'):
  with pytest.raises(RuntimeError):r.ImportGuard().find_spec(name)
 assert r.ImportGuard().find_spec('torch.autograd') is None


@pytest.fixture
def frozen(tmp_path,monkeypatch):
 root=tmp_path/'root';here=root/'research_artifacts/coupling';here.mkdir(parents=True);skin=root/'skin';skin.mkdir()
 monkeypatch.setattr(r,'ROOT',root);monkeypatch.setattr(r,'HERE',here);monkeypatch.setattr(r,'SKIN',skin)
 for name in ('run_coupling.py','contact_kinematics.py','coupling_case.py','README.md'):(here/name).write_text('manufactured')
 (here/'controls.json').write_text(json.dumps(dict(exit_code=0,integration_calls=0)))
 for name in ('skin_kinematics.py','contact_law.py'):(skin/name).write_text('old frozen source')
 raw=root/'runs/old.npz';raw.parent.mkdir();raw.write_bytes(b'old raw');bindings={str(raw):r.sha(raw),**{str(x):r.sha(x) for x in skin.iterdir()}}
 spec={'20':dict(base_xml='manufactured'),'28':dict(base_xml='manufactured')};p_old=dict()
 monkeypatch.setattr(r,'prior',lambda *a:(p_old,dict(bindings),{},spec))
 case=SimpleNamespace(BUDGET=dict(physics_steps=0),GATES={'a':1},PARAMETERS={'k':20000},FLAGS={'accepted_mapping':False},scene_xml=lambda p:('<mujoco/>','same'))
 monkeypatch.setattr(r,'case_module',lambda:case)
 pp=here/'protocol.json';p=r.prepare(pp,'a'*64,'b'*64);commit='a'*40
 files={rel:(root/rel).read_bytes() for rel in p['tracked_sources']};files[str(pp.relative_to(root))]=pp.read_bytes()
 monkeypatch.setattr(r,'git',lambda *args:(commit+'\n').encode() if args==('rev-parse','HEAD') else files[args[1].split(':',1)[1]])
 return pp,commit,raw


def test_manufactured_prepare_does_not_git_classify_raw_inputs(frozen):
 path,commit,raw=frozen;p=r.validate(path,commit);assert str(raw) in p['bindings'] and not any('/runs/' in x for x in p['tracked_sources'])
 with pytest.raises(ValueError):r.prepare(path,'a'*64,'b'*64)

@pytest.mark.parametrize('fault',['raw','gates','parameters','domain','profile','admission','root','partition'])
def test_frozen_changes_rejected(frozen,fault):
 path,commit,raw=frozen
 if fault=='raw':raw.write_bytes(b'changed')
 else:
  p=r.read(path)
  if fault=='gates':p['gates']['a']=2
  elif fault=='parameters':p['parameters']['k']=1
  elif fault=='domain':p['state_domain']='old GS q0'
  elif fault=='profile':p['physical_profile']='source32 plus oldJ'
  elif fault=='admission':p['accepted_mapping']=True
  elif fault=='root':p['raw_root']='/outside'
  else:p['tracked_sources']={}
  path.write_text(json.dumps(p))
 with pytest.raises(ValueError):r.validate(path,commit)


def completed_child(tmp_path,monkeypatch):
 case=r.case_module();raw=tmp_path/'raw';raw.mkdir();names=['physical_profile.json','surface_tangent_and_alignment.npz','native_contact_disabled.xml','original_model_and_named_layout.npz']+[f'condition_{i:02d}.npz' for i in range(12)]
 artifacts=[]
 for name in names:
  path=raw/name;path.write_bytes(b'manufactured never compiled/evaluated');artifacts.append(r.binding(path))
 points=[dict(part=str(i)) for i in range(6)];layout=[dict(name='manufactured')]
 p=dict(budget=case.BUDGET,cases={'20':dict(material_points=points,saved_named_joint_addresses=layout)})
 conditions=[dict(condition=i,part=str(i//2),mode=('normal','slide')[i%2],passed=True,root_rigid_wrench_equivalence_claimed=False) for i in range(12)]
 m=dict(status='COMPLETED',forbidden_counts=case.FORBIDDEN,physical_outcome=None,**case.FLAGS,source32_derivative_qualified=False,root_rigid_wrench_equivalence_claimed=False,static_point_coupling_status='PASS',source32_alignment_status='FAIL',material_points=points,named_layout=layout,conditions=conditions,profile_hash='a'*64,state_hash='b'*64)
 v=dict(schema_version=r.SCHEMA+'_case',index=20,protocol_sha256='c'*64,freeze_commit='d'*40,physical_outcome=None,**case.FLAGS,status='COMPLETED',artifacts=artifacts,counts={k:v for k,v in case.BUDGET.items() if k!='maximum_seconds_per_case'},actual_engine_counts={k:case.BUDGET[k] for k in ('model_compilations','data_constructions','static_forward_calls','full_mass_matrix_calls')},forbidden_attempts=[],failure=None,unindexed_partial_files=[],measurement=m)
 return v,p,raw


def test_parent_preserves_finite_scientific_fail_separate_from_transaction(tmp_path,monkeypatch):
 v,p,raw=completed_child(tmp_path,monkeypatch);r.check_result(v,p,'c'*64,'d'*40,20,raw,0)
 v['measurement']['static_point_coupling_status']='FAIL';r.check_result(v,p,'c'*64,'d'*40,20,raw,0)

@pytest.mark.parametrize('fault',['counts','forbidden','identity','null','admission','points','missing_array','unexpected_file','measurement'])
def test_parent_refuses_child_method_identity_and_publication_errors(tmp_path,monkeypatch,fault):
 v,p,raw=completed_child(tmp_path,monkeypatch)
 if fault=='counts':v['actual_engine_counts']['static_forward_calls']=23
 elif fault=='forbidden':v['measurement']['forbidden_counts']={}
 elif fault=='identity':v['index']=28
 elif fault=='null':v['physical_outcome']='success'
 elif fault=='admission':v['measurement']['accepted_mapping']=True
 elif fault=='points':v['measurement']['material_points']=[]
 elif fault=='missing_array':v['artifacts']=v['artifacts'][:-1]
 elif fault=='unexpected_file':(raw/'unindexed.xml').write_text('failed late artifact')
 else:v['measurement']['status']='partial'
 with pytest.raises(ValueError):r.check_result(v,p,'c'*64,'d'*40,20,raw,0)


def test_actual_source_independent_receipt_schema_is_explicit():
 assert r.SOURCE_READBACK_SCHEMA=='manosim_native_source_pipeline_independent_saved_readback_v1'
 assert r.SOURCE_READBACK_STATUS=='VERIFIED_COMPLETED_SOURCE_CAMERA_PIPELINE_GEOMETRY_ONLY'
