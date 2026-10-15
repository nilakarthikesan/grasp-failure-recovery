"""Manufactured mechanics and mocked transactions only; never real engine work."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('manufactured_contact_dynamics',HERE/'run_contact_dynamics.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
a=d.checks


@pytest.fixture(autouse=True)
def prohibit_real_engine(monkeypatch):
    import mujoco
    def blocked(*args,**kwargs):raise AssertionError('Manufactured controls prohibit real model/FK/physics/contact APIs')
    for name in ('MjModel','MjData','mj_forward','mj_step','mj_step1','mj_step2','mj_kinematics','mj_contactForce'):
        monkeypatch.setattr(mujoco,name,blocked)
    for name in ('get_mjcb_control','get_mjcb_passive'):
        if hasattr(mujoco,name):monkeypatch.setattr(mujoco,name,lambda:None)


def test_exact_eight_fixture_fourteen_model_10650_step_plan():
    plan=d.plan()
    assert len(plan)==14 and sum(c['steps'] for c in plan)==10650
    assert len({(c['kind'],c['geometry']) for c in plan})==8
    assert len([c for c in plan if not c['steps']])==2
    assert sum(c['kind']=='pair_impact' for c in plan)==2
    assert d.BUDGET==dict(models=14,data=14,initial_forwards=14,steps=10650,contact_force_reads=10652,maximum_seconds=120,outer_seconds=130)
    assert all(not value for value in d.FLAGS.values())


def test_all_exact_XML_has_known_solids_no_assets_or_native_geometry():
    for case in d.plan():
        root=ET.fromstring(d.xml(case));assert root.find('asset') is None and root.find('actuator') is None
        assert not root.findall('.//include') and not any('file' in x.attrib for x in root.iter())
        opt=root.find('option');assert opt.get('integrator')=='Euler' and opt.get('cone')=='elliptic'
        assert opt.get('solver')=='Newton'
        assert opt.get('impratio')=='1' and opt.get('noslip_iterations')=='0'
        if case['geometry']=='flex':
            flex=root.find('./deformable/flex');assert flex.get('radius')=='0' and flex.get('body')=='floor'
            assert flex.get('element')=='0 1 2' and flex.find('contact').get('condim')=='3'
        assert root.find('.//geom').get('solref')=='-10000 -200'
    with pytest.raises(ValueError):d.xml(dict(d.plan()[0],steps=1))
    with pytest.raises(ValueError,match='new protocol'):d.xml(d.plan()[0],model_id='native_exact_skin_penalty_law_v1')


def test_source_derived_normal_and_soft_friction_references():
    assert a.rest_penetration()==pytest.approx(98.1e-6,abs=1e-18)
    force=.25*a.MU*a.MASS*a.GRAVITY_M_S2
    assert a.regularized_creep(force)==pytest.approx(.000613125,abs=1e-18)
    assert a.regularized_creep(force)>0 # no exact-stick claim
    z,v=a.euler_freefall(.5,0.,.001,250)
    assert z==pytest.approx(.5-.5*9.81*.001**2*250*251)
    assert v==pytest.approx(-2.4525)


@pytest.mark.parametrize('kind',['normal','creep','freefall','quat'])
def test_nonfinite_or_invalid_analytic_parameters_refused(kind):
    with pytest.raises(ValueError):
        if kind=='normal':a.rest_penetration(stiffness=np.nan)
        elif kind=='creep':a.regularized_creep(np.inf)
        elif kind=='freefall':a.euler_freefall(0.,0.,.001,True)
        else:a.quaternion_matrix([.9,0.,0.,0.])


def test_all_three_contact_axes_and_rigid_rotational_levers():
    r,t,_=d.frame();point=np.array([.2,.3,.4]);origins=[t,np.array([-.1,.2,.15])]
    j=a.contact_jacobian(r,origins,[r,np.eye(3)],[0,6],point,12)
    v=np.linspace(-.2,.3,12)
    w0=r@v[3:6];w1=v[9:12]
    expected=r@(v[6:9]+np.cross(w1,point-origins[1])-v[:3]-np.cross(w0,point-origins[0]))
    np.testing.assert_allclose(j@v,expected,rtol=0,atol=1e-15)
    p0,p1=a.virtual_work(j,v,[.7,.1,-.2]);assert abs(p0-p1)<1e-15
    # Deliberate tangent error changes independent virtual work.
    wrong=j.copy();wrong[1,0]+=.1
    assert abs(np.array([.7,.1,-.2])@(wrong@v)-p1)>1e-6


def test_shared_world_contact_point_balances_force_and_angular_wrench():
    force=np.array([.3,-.7,1.1]);point=np.array([.4,.2,-.1]);origins=[np.array([1.,0,0]),np.array([0.,1,0])]
    linear,angular=a.action_reaction(force,point,origins)
    assert linear==0 and angular<1e-15
    wrong=np.cross(point-origins[0],-force)+np.cross(point+[.1,0,0]-origins[1],force)+np.cross(origins[0],-force)+np.cross(origins[1],force)
    assert np.linalg.norm(wrong)>.01


@pytest.mark.parametrize('force,gradient,valid',[
    ([1.,.1,.2],[0.,0.,0.],True),([1.,.5,0.],[.5,-1.,0.],True),
    ([1.,.6,0.],[0.,0.,0.],False),([1.,0.,0.],[-1.,0.,0.],False),
    ([1.,0.,0.],[1.,0.,0.],False),([0.],[1.],True)])
def test_primal_dual_and_complementarity_are_independent_checks(force,gradient,valid):
    r=a.cone_residual(force,gradient,.5 if len(force)==3 else 0.)
    passed=max(r['primal_N'],r['dual_m_s2'],r['complementarity_N_m_s2'])<1e-15
    assert passed==valid


def test_discrete_impulse_work_has_analytic_constant_mass_reference():
    before=np.array([.2,-.1,.3]);force=np.array([.4,0.,-1.962]);h=.001
    after=before+h*force/a.MASS
    f,w=a.discrete_force_work(a.MASS,before,after,force,h)
    assert f<1e-13 and w<1e-16
    f,_=a.discrete_force_work(a.MASS,before,after+[.001,0,0],force,h)
    assert f>.1


def fake_data(nv=3):
    return SimpleNamespace(qpos=np.zeros(3),qvel=np.zeros(nv),qacc=np.zeros(nv),ctrl=np.zeros(0),qfrc_applied=np.zeros(nv),
        warning=SimpleNamespace(number=np.zeros(8,dtype=int)),time=0.)


def fake_engine(tmp_path,forward=None,step=None):
    data=fake_data();model=SimpleNamespace(opt=SimpleNamespace(timestep=.001));calls=[]
    def default_step(m,v):calls.append('step');v.time+=.001
    def default_forward(m,v):calls.append('forward')
    module=SimpleNamespace(MjModel=SimpleNamespace(from_xml_string=lambda x:(calls.append('compile') or model)),
        MjData=lambda m:data,mj_forward=forward or default_forward,mj_step=step or default_step,
        mj_contactForce=lambda m,v,i,out:None,mj_step1=lambda *a:calls.append('bad'))
    return module,model,data,calls,d.Budget(tmp_path)


def test_exact_XML_and_api_guard_precedes_any_native_call(tmp_path):
    module,model,data,calls,b=fake_engine(tmp_path)
    with d.Engine(module,b) as engine:
        with pytest.raises(RuntimeError):engine.compile('<mujoco/>')
        assert not calls and b.counts['models']==0
        m=engine.compile(d.xml(d.plan()[0]));v=engine.data(m);engine.call('initial_forwards',m,v)
        with pytest.raises(RuntimeError):module.mj_step(m,v)
        engine.call('steps',m,v)
    assert calls==['compile','forward','step'] and b.counts['steps']==1


@pytest.mark.parametrize('fault',['warning','nonfinite','clock','applied','python_warning'])
def test_per_step_warning_finite_time_and_force_guards(tmp_path,fault):
    def step(m,v):
        v.time+=.001
        if fault=='warning':v.warning.number[0]+=1
        elif fault=='nonfinite':v.qacc[0]=np.nan
        elif fault=='clock':v.time+=.0001
        elif fault=='applied':v.qfrc_applied[0]=1.
        else:__import__('warnings').warn('manufactured')
    module,_,_,_,b=fake_engine(tmp_path,step=step)
    with d.Engine(module,b) as engine:
        m=engine.compile(d.xml(d.plan()[0]));v=engine.data(m)
        engine.call('initial_forwards',m,v)
        with pytest.raises(RuntimeError):engine.call('steps',m,v)
    assert b.counts['steps']==1


def test_quota_refuses_before_extra_work_and_native_imports_refused(tmp_path):
    b=d.Budget(tmp_path);b.counts['steps']=d.BUDGET['steps']
    with pytest.raises(RuntimeError):b.charge('steps')
    assert b.counts['steps']==10650
    finder=d.Imports()
    for name in ('torch.nn','manotorch','hug','grasp_failure_prediction.evaluation'):
        with pytest.raises(RuntimeError):finder.find_spec(name)
    assert finder.find_spec('xml.etree') is None


def manufactured_arrays(case):
    n=case['steps'];t=np.arange(1,n+1)*case['dt_s'];z,v=a.euler_freefall(.5,0.,case['dt_s'],n)
    arrays={key:np.zeros(n) for key in ('force_balance_N','discrete_impulse_work_error_J','action_reaction_N','angular_action_reaction_N_m','cone_primal_N','cone_dual_m_s2','cone_complementarity_N_m_s2')}
    arrays.update(endpoint_time_s=t,cone_force_scale_N=np.ones(n),cone_gradient_scale_m_s2=np.ones(n),
        positions_world_m=np.zeros((n,1,3)),velocities_world_m_s=np.zeros((n,1,3)),start_qacc=np.zeros((n,3)),
        contact_gap_m=np.full(n,-a.rest_penetration()),contact_force_world_N=np.tile([0.,0.,a.MASS*a.GRAVITY_M_S2],(n,1)),
        independent_energy_J=np.zeros(n),linear_momentum_kg_m_s=np.zeros((n,3)),total_angular_momentum_kg_m2_s=np.zeros((n,3)))
    if case['kind']=='freefall':arrays['positions_world_m'][-1,0,2]=z;arrays['velocities_world_m_s'][-1,0,2]=v
    if case['kind']=='friction_low':arrays['velocities_world_m_s'][:,0,0]=a.regularized_creep(.25*a.MU*a.MASS*a.GRAVITY_M_S2)
    if case['kind']=='friction_high':arrays['start_qacc'][:,0]=.5*a.MU*a.GRAVITY_M_S2
    return arrays


@pytest.mark.parametrize('kind',['freefall','support_flex','support_solid','friction_low','friction_high','pair_impact'])
def test_manufactured_analytic_gate_pass_and_failure_retained(kind):
    case=next(c for c in d.plan() if c['kind']==kind);arrays=manufactured_arrays(case)
    assert d.assess(case,arrays)['status']=='PASS'
    arrays['force_balance_N'][0]=1.
    result=d.assess(case,arrays);assert result['status']=='FAIL' and result['failures']==['force_balance_N']
    assert result['physical_outcome'] is None and all(result[key] is False for key in d.FLAGS)


def test_low_friction_does_not_define_exact_stick_and_high_sliding_reference_is_fixed():
    case=next(c for c in d.plan() if c['kind']=='friction_low');values=manufactured_arrays(case)
    values['velocities_world_m_s'][:]=0.
    assert d.assess(case,values)['status']=='FAIL'
    case=next(c for c in d.plan() if c['kind']=='friction_high');values=manufactured_arrays(case);values['start_qacc'][:]=0.
    assert d.assess(case,values)['status']=='FAIL'


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    root=tmp_path/'root';here=root/'research_artifacts'/'contact';here.mkdir(parents=True)
    monkeypatch.setattr(d,'ROOT',root);monkeypatch.setattr(d,'HERE',here)
    old=here.parent/'old_source.py';old.write_text('unchanged old source')
    dep=tmp_path/'runtime.bin';dep.write_bytes(b'unchanged runtime')
    evidence_file=tmp_path/'prior.json';evidence_file.write_text('{}')
    oldp=dict(bindings={str(old):d.sha(old)},tracked_sources=[str(old)])
    evidence=dict(files={str(evidence_file):d.sha(evidence_file)},radius_policy=dict(rigid_only=True),prior_protocol_sha256='a'*64)
    module=SimpleNamespace(runtime=lambda:dict(numpy='2.2.6',mujoco='3.4.0',executable='manufactured'),runtime_files=lambda:[dep])
    monkeypatch.setattr(d,'prior',lambda:(module,oldp,evidence))
    for f in d.sources():f.write_text('manufactured source')
    (here/'controls.json').write_text(json.dumps(dict(exit_code=0,integration_calls=0)))
    path=here/'protocol.json';p=d.prepare(path);commit='a'*40
    contents={str(Path(name).relative_to(root)):Path(name).read_bytes() for name in p['tracked_sources']+[str(path)]}
    def git(command,text=False):
        if command[-1]=='HEAD' or command[-1].endswith('^{commit}'):return commit+'\n'
        return contents[command[-1].split(':',1)[1]]
    monkeypatch.setattr(d.subprocess,'check_output',git)
    return path,commit,dep,evidence_file


def test_fresh_prepare_and_committed_protocol_source_validation_only(frozen):
    path,commit,_,_=frozen
    assert d.validate(path,commit)[1]==commit
    with pytest.raises(ValueError):d.prepare(path)


@pytest.mark.parametrize('fault',['source','runtime','prior','budget','solver','limits','admission','HEAD','xml','rawroot','partition'])
def test_frozen_input_or_method_changes_fail_closed(frozen,monkeypatch,fault):
    path,commit,dep,prior=frozen
    if fault=='source':d.sources()[0].write_text('changed')
    elif fault=='runtime':dep.write_bytes(b'changed')
    elif fault=='prior':prior.write_text('{"changed":true}')
    elif fault=='HEAD':
        original=d.subprocess.check_output
        monkeypatch.setattr(d.subprocess,'check_output',lambda command,**kw:'b'*40+'\n' if command[-1]=='HEAD' else original(command,**kw))
    else:
        p=d.read(path)
        if fault=='budget':p['budget']['steps']+=1
        elif fault=='solver':p['solver']['iterations']=1
        elif fault=='limits':p['limits']['force_balance_N']=1e5
        elif fault=='admission':p['collection_or_training_allowed']=True
        elif fault=='xml':p['xml_sha256'][d.plan()[0]['id']]='0'*64
        elif fault=='rawroot':p['raw_root']='/outside/root'
        else:p['tracked_sources']=[]
        path.write_text(json.dumps(p))
    with pytest.raises(ValueError):d.validate(path,commit)


def test_output_hash_and_unindexed_array_refusal(tmp_path):
    measurement=d.Measurement(d.plan()[0],tmp_path)
    measurement.save('numeric.npz',dict(x=np.arange(3.)),'manufactured')
    d.verify_artifacts(measurement.artifacts,tmp_path)
    np.savez_compressed(tmp_path/'extra.npz',x=[1.])
    with pytest.raises(ValueError):d.verify_artifacts(measurement.artifacts,tmp_path)
    (tmp_path/'extra.npz').unlink();(tmp_path/'numeric.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):d.verify_artifacts(measurement.artifacts,tmp_path)


@pytest.mark.parametrize('token',['NaN','1e999'])
def test_nonfinite_JSON_refused(tmp_path,token):
    path=tmp_path/'bad.json';path.write_text('{"x":'+token+'}')
    with pytest.raises(ValueError):d.read(path)


def test_matching_solid_reference_is_separate_from_timestep_convergence():
    results=[]
    for case in d.plan():
        if case['kind'] not in ('support_flex','support_solid'):continue
        arrays=manufactured_arrays(case);measurement=dict(assessment=d.assess(case,arrays),final_position_world_m=[[0.,0.,.01]],final_velocity_world_m_s=[[0.,0.,0.]])
        results.append(dict(case=case,status='MEASURED',measurement=measurement))
    assert all(r['status']=='PASS' for r in d.matched_solid_reference(results))
    results[0]['measurement']['final_position_world_m'][0][2]+=.001
    assert d.matched_solid_reference(results)[0]['status']=='FAIL'


def test_worker_unexpected_failure_preserves_current_partial_and_remaining_not_run(tmp_path,monkeypatch):
    output=tmp_path/'tracked';raw=tmp_path/'raw';output.mkdir();raw.mkdir();protocol=tmp_path/'p.json';protocol.write_text('{}')
    monkeypatch.setattr(d,'validate',lambda *a:(dict(output_root=str(output),raw_root=str(raw)),'frozen'))
    class Failed(d.Measurement):
        def run(self,*a):
            self.save('pre_failure.npz',dict(x=np.arange(3.)),'original_write_time')
            raise RuntimeError('manufactured unexpected failure before any engine call')
    monkeypatch.setattr(d,'Measurement',Failed)
    report=d.worker(protocol,'frozen')
    assert report['status']=='HARD_STOP' and report['numerical_standard_contact_qualification']=='NOT_QUALIFIED'
    assert report['cases'][0]['status']=='HARD_STOPPED_PARTIAL'
    assert all(row['status']=='NOT_RUN' for row in report['cases'][1:])
    assert all(value==0 for value in report['operation_counts'].values())
    assert report['artifacts'][0]['label']=='original_write_time' and not report['unindexed_array_paths']
