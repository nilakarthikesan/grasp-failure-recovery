"""Manufactured algebra and mocked engines only; no actual fixture execution."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('manufactured_compliant_contact',HERE/'run_contact_dynamics.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

@pytest.fixture(autouse=True)
def no_real_engine(monkeypatch):
 import mujoco
 def no(*a,**k):raise AssertionError('No actual model/FK/contact/physics in controls')
 for name in ('MjModel','MjData','mj_forward','mj_step','mj_step1','mj_step2','mj_kinematics','mj_contactForce'):monkeypatch.setattr(mujoco,name,no)
 for name in ('get_mjcb_control','get_mjcb_passive'):
  if hasattr(mujoco,name):monkeypatch.setattr(mujoco,name,lambda:None)

def test_exact_four_fixtures_eight_conditions_27000_steps_and_parameters():
 assert len(d.plan())==8 and sum(c['steps'] for c in d.plan())==27000
 assert {c['kind'] for c in d.plan()}=={'freefall','support','friction_low','friction_high'}
 assert d.BUDGET==dict(models=8,data=8,initial_forwards=8,steps=27000,law_evaluations=27000,maximum_seconds=120,outer_seconds=130)
 assert d.PARAMETERS==dict(stiffness_n_per_m=20000.,damping_ns_per_m=2*np.sqrt(20000.*.2),friction=.5,friction_speed_m_per_s=.001)
 assert all(v is False for v in d.FLAGS.values())
 for c in d.plan():
  assert c['dt_s'] in (.00025,.000125)
  assert c['steps']*c['dt_s']==dict(freefall=.25,support=1.,friction_low=.5,friction_high=.5)[c['kind']]

def test_only_three_named_translations_and_collision_disabled_XML():
 for c in d.plan():
  x=ET.fromstring(d.xml(c));assert not x.findall('.//asset') and not x.findall('.//include') and x.find('actuator') is None
  assert x.find('./option/flag').get('contact')=='disable'
  assert [(j.get('name'),j.get('type'),j.get('axis')) for j in x.findall('.//joint')]==[('sx','slide','1 0 0'),('sy','slide','0 1 0'),('sz','slide','0 0 1')]
  g=x.find('.//geom');assert g.get('contype')==g.get('conaffinity')=='0'
  assert x.find('deformable') is None and not any('file' in node.attrib for node in x.iter())
 with pytest.raises(ValueError):d.xml(d.plan()[0],model_id='standard_mujoco_rigid_contact_mj340_v1')

def test_analytic_references_and_all_original_limits_unchanged():
 assert d.checks.MASS*d.checks.GRAVITY_M_S2/d.PARAMETERS['stiffness_n_per_m']==pytest.approx(98.1e-6)
 assert .001*.25/np.sqrt(1-.25**2)==pytest.approx(.0002581988897471611)
 assert .5*.5*9.81==2.4525
 original=d.ROOT/'research_artifacts/manosim_contact_dynamics_2026-10-15_v1/analytic_checks.py'
 assert (HERE/'analytic_checks.py').read_bytes()==original.read_bytes()

def one_sample(tmp_path,kind='friction_low'):
 c=next(c for c in d.plan() if c['kind']==kind);p=np.array([.13,-.2,.01-.00012]);v=np.array([.0003,-.0002,-.001]);b=d.Budget(tmp_path)
 r=d.evaluate_law(c,p,v,b);force=r['hand_force_world_n']+d.external_force(c);acc=force/.2+[0,0,c['gravity']];v1=v+c['dt_s']*acc;p1=p+c['dt_s']*v1
 data=SimpleNamespace(qpos=p1.copy(),qvel=v1,qacc=acc,qfrc_applied=force,qfrc_constraint=np.zeros(3),time=c['dt_s'],ncon=0,nefc=0,warning=SimpleNamespace(number=np.zeros(8,dtype=int)),solver_niter=np.zeros(20,dtype=int),energy=np.zeros(2))
 return c,d.sample(c,p,v,data,np.zeros(3),r),r,b

def test_frozen_strict51_kernel_force_reaction_power_and_discrete_energy(tmp_path):
 c,a,r,b=one_sample(tmp_path)
 assert r['hand_generalized_load'].shape==(51,) and not r['hand_generalized_load'][3:].any()
 assert r['object_generalized_load'].shape==(1,) and not r['object_generalized_load'].any()
 for key in ('force_balance_N','discrete_impulse_work_error_J','integration_energy_identity_error_J','point_power_error_W','linear_reaction_error_N','angular_reaction_error_N_m'):assert a[key]<1e-12
 assert r['normal_dissipation_power_w']<=0 and r['friction_power_w']<=0
 assert a['law_common_point_world_m'][2]==pytest.approx(a['gap_start_m'])
 assert a['measured_integration_defect_J']!=0 # continuous dissipation does not imply zero discrete defect
 assert a['start_time_s']==0 and a['endpoint_time_s']==c['dt_s']
 assert b.counts['law_evaluations']==1


def test_law_budget_rejects_before_kernel_call(tmp_path,monkeypatch):
 b=d.Budget(tmp_path);b.counts['law_evaluations']=27000
 monkeypatch.setattr(d.law,'evaluate_contact',lambda **k:pytest.fail('Kernel must not be called'))
 with pytest.raises(RuntimeError):d.evaluate_law(d.plan()[0],np.array([0.,0.,.5]),np.zeros(3),b)
 assert b.counts['law_evaluations']==27000


def fake_engine(tmp_path,step=None):
 model=SimpleNamespace(opt=SimpleNamespace(timestep=.001));data=SimpleNamespace(qpos=np.zeros(3),qvel=np.zeros(3),qacc=np.zeros(3),ctrl=np.zeros(0),qfrc_applied=np.zeros(3),qfrc_constraint=np.zeros(3),warning=SimpleNamespace(number=np.zeros(8,dtype=int)),time=0.,ncon=0,nefc=0);calls=[]
 def default_step(m,v):calls.append('step');v.time+=.001
 mod=SimpleNamespace(MjModel=SimpleNamespace(from_xml_string=lambda t:(calls.append('compile') or model)),MjData=lambda m:data,mj_forward=lambda m,v:calls.append('forward'),mj_step=step or default_step,mj_contactForce=lambda *a:pytest.fail('No engine contacts'))
 return mod,model,data,calls,d.Budget(tmp_path)

def test_engine_load_is_explicit_and_only_registered_call_path(tmp_path):
 mod,_,_,calls,b=fake_engine(tmp_path)
 with d.Engine(mod,b) as e:
  with pytest.raises(RuntimeError):e.compile('<mujoco/>')
  m=e.compile(d.xml(d.plan()[0]));v=e.data(m);e.call('initial_forwards',m,v);e.apply_force(m,v,np.array([.1,.2,.3]));e.call('steps',m,v)
  with pytest.raises(RuntimeError):mod.mj_contactForce(m,v,0,np.zeros(6))
 assert calls==['compile','forward','step']

@pytest.mark.parametrize('fault',['warning','nonfinite','time','load','contacts','constraints','stock_force'])
def test_per_step_guard_rejects_every_unexpected_source(tmp_path,fault):
 def step(m,v):
  v.time+=.001
  if fault=='warning':v.warning.number[0]=1
  elif fault=='nonfinite':v.qacc[0]=np.nan
  elif fault=='time':v.time+=.0001
  elif fault=='load':v.qfrc_applied[0]+=1
  elif fault=='contacts':v.ncon=1
  elif fault=='constraints':v.nefc=1
  else:v.qfrc_constraint[0]=1
 mod,_,_,_,b=fake_engine(tmp_path,step)
 with d.Engine(mod,b) as e:
  m=e.compile(d.xml(d.plan()[0]));v=e.data(m);e.call('initial_forwards',m,v);e.apply_force(m,v,np.zeros(3))
  with pytest.raises(RuntimeError):e.call('steps',m,v)
 assert b.counts['steps']==1


def manufactured_arrays(case):
 n=case['steps'];t=np.arange(1,n+1)*case['dt_s'];a={k:np.zeros(n) for k in ('force_balance_N','discrete_impulse_work_error_J','euler_position_error_m','integration_energy_identity_error_J','linear_reaction_error_N','angular_reaction_error_N_m','point_power_error_W','law_power_identity_residual_w','engine_ncon','engine_nefc','law_normal_dissipation_power_w','law_friction_power_w','measured_integration_defect_J','continuous_dissipation_step_J')}
 a.update(endpoint_time_s=t,law_normal_force_n=np.full(n,1.962),law_hand_force_world_n=np.tile([0.,0.,1.962],(n,1)),engine_qfrc_constraint=np.zeros((n,3)),position_endpoint_m=np.zeros((n,3)),velocity_endpoint_m_s=np.zeros((n,3)),gap_start_m=np.full(n,-98.1e-6),mechanical_energy_endpoint_J=np.zeros(n),start_qacc=np.zeros((n,3)))
 if case['kind']=='freefall':a['position_endpoint_m'][-1,2],a['velocity_endpoint_m_s'][-1,2]=d.checks.euler_freefall(.5,0.,case['dt_s'],n)
 if case['kind']=='friction_low':a['velocity_endpoint_m_s'][:,0]=.001*.25/np.sqrt(1-.25**2)
 if case['kind']=='friction_high':a['start_qacc'][:,0]=2.4525
 return a

@pytest.mark.parametrize('kind',['freefall','support','friction_low','friction_high'])
def test_fixed_assessment_pass_failure_and_null_admission(kind):
 c=next(c for c in d.plan() if c['kind']==kind);a=manufactured_arrays(c);assert d.assess(c,a)['status']=='PASS'
 a['integration_energy_identity_error_J'][0]=1.;r=d.assess(c,a);assert r['status']=='FAIL' and r['failures']==['integration_energy_identity_error_J'];assert r['physical_outcome'] is None and all(r[k] is False for k in d.FLAGS)


def test_no_gate_can_replace_soft_friction_reference_with_zero_sticking():
 c=next(c for c in d.plan() if c['kind']=='friction_low');a=manufactured_arrays(c);a['velocity_endpoint_m_s'][:]=0
 assert d.assess(c,a)['failures']==['regularized_creep_reference']

@pytest.fixture
def frozen(tmp_path,monkeypatch):
 root=tmp_path/'root';here=root/'research_artifacts/contact';here.mkdir(parents=True);monkeypatch.setattr(d,'ROOT',root);monkeypatch.setattr(d,'HERE',here)
 old=root/'old_source.py';old.write_text('old source');dep=tmp_path/'runtime.bin';dep.write_bytes(b'runtime');prior=root/'runs/prior.json';prior.parent.mkdir();prior.write_text('{}')
 module=SimpleNamespace(runtime=lambda:dict(numpy='2.2.6',mujoco='3.4.0'),runtime_files=lambda:[dep]);oldp=dict(bindings={str(old):d.sha(old)},tracked_sources=[str(old)]);evidence=dict(files={str(prior):d.sha(prior)})
 monkeypatch.setattr(d,'prior',lambda:(module,oldp,evidence))
 for p in d.sources():p.write_text('manufactured')
 (here/'controls.json').write_text(json.dumps(dict(exit_code=0,integration_calls=0)));path=here/'protocol.json';p=d.prepare(path);commit='a'*40
 blobs={str(Path(n).relative_to(root)):Path(n).read_bytes() for n in p['tracked_sources']+[str(path)]}
 monkeypatch.setattr(d.subprocess,'check_output',lambda cmd,**k:commit+'\n' if cmd[-1]=='HEAD' or cmd[-1].endswith('^{commit}') else blobs[cmd[-1].split(':',1)[1]])
 return path,commit,dep,prior


def test_source_only_protocol_and_raw_hash_partition(frozen):
 p,commit,_,raw=frozen;data=d.validate(p,commit)[0];assert str(raw) in data['bindings'] and str(raw) not in data['tracked_sources']
 with pytest.raises(ValueError):d.prepare(p)

@pytest.mark.parametrize('fault',['source','runtime','prior','parameters','dtpair','solver','limits','admission','rawroot','partition','stability'])
def test_changed_frozen_contract_refused(frozen,fault):
 p,commit,dep,raw=frozen
 if fault=='source':d.sources()[0].write_text('changed')
 elif fault=='runtime':dep.write_bytes(b'changed')
 elif fault=='prior':raw.write_text('changed')
 else:
  a=d.read(p)
  if fault=='parameters':a['parameters']['stiffness_n_per_m']=1.
  elif fault=='dtpair':a['timestep_pair_s']=[.0001]
  elif fault=='solver':a['solver']['integrator']='RK4'
  elif fault=='limits':a['limits']['support_force_relative']=1.
  elif fault=='admission':a['collection_or_training_allowed']=True
  elif fault=='rawroot':a['raw_root']='/outside'
  elif fault=='stability':a['local_resting_friction_stability']['rate_per_s']=1.
  else:a['tracked_sources']=[]
  p.write_text(json.dumps(a))
 with pytest.raises(ValueError):d.validate(p,commit)


def test_unexpected_failure_retains_current_partial_and_rest_not_run(tmp_path,monkeypatch):
 output=tmp_path/'tracked';raw=tmp_path/'raw';output.mkdir();raw.mkdir();p=tmp_path/'p.json';p.write_text('{}');monkeypatch.setattr(d,'validate',lambda *a:(dict(output_root=str(output),raw_root=str(raw)),'frozen'))
 class Failed(d.Measurement):
  def run(self,*a):self.save('partial.npz',dict(x=np.arange(3.)),'original');raise RuntimeError('manufactured')
 monkeypatch.setattr(d,'Measurement',Failed);r=d.worker(p,'frozen');assert r['status']=='HARD_STOP' and r['numerical_compliant_contact_qualification']=='NOT_QUALIFIED';assert r['cases'][0]['status']=='HARD_STOPPED_PARTIAL' and all(x['status']=='NOT_RUN' for x in r['cases'][1:]);assert all(v==0 for v in r['operation_counts'].values())


def test_prospective_resting_friction_step_bound_is_limited_and_satisfied():
    damping=.5*.2*9.81/.001;rate=damping/.2;hmax=2/rate
    assert damping==pytest.approx(981.) and rate==pytest.approx(4905.)
    assert .001>hmax and .0005>hmax
    assert max(d.DT_PAIR)<hmax
    assert d.DT_PAIR==(.00025,.000125)
