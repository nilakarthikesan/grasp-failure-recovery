"""Manufactured-only controls; never construct a real MuJoCo model or call FK."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('primitive_flex_controls', Path(__file__).with_name('run_qualification.py'))
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)


def manufactured_result(case):
    r, t, _ = q.frame(case['frame']); rows = []
    nv = 12 if case['attachment'] == 'single_body' else 24
    for probe in q.probes(case):
        contacts = []
        if probe['placement'] == 'penetrating':
            pos = r@(np.array(probe['surface_center_local']) + [0., 0., probe['side']*(case['radius_m']+probe['expected_gap_m']/2)])+t
            contacts = [dict(distance_m=probe['expected_gap_m'], position_world=pos.tolist(),
                normal_world=(-probe['side']*r[:, 2]).tolist(), engine_normal_jacobian=[0.]*nv,
                expected_vertex_normal_jacobian=[0.]*nv, translation_columns=[0,1,2,6,7,8],
                barycentric_weights=[1/3]*3, efc_address=0, constraint_active=True)]
        rows.append(dict(probe=probe, contact_count=len(contacts), contacts=contacts))
    return dict(case=case, status='COMPILED_AND_MEASURED', probes=rows, vertex_max_error_m=0.,
                exact_faces=True, exact_owner_bindings=True, nv=nv)


def test_finite_twelve_case_eightyfour_forward_plan_and_no_native_assets():
    assert len(q.cases()) == 12 and len({x['case_id'] for x in q.cases()}) == 12
    assert sum(1+len(q.probes(x)) for x in q.cases()) == 84
    assert q.BUDGET['model_compile_attempts'] == 12 and q.BUDGET['physics_steps'] == 0
    for case in q.cases():
        xml = q.validate_xml(case, q.make_xml(case)); root = ET.fromstring(xml)
        assert root.find('asset') is None and root.find('actuator') is None
        expected_free = 2 if case['attachment'] == 'single_body' else 4
        assert len(root.findall('.//freejoint')) == expected_free


def test_nonconvex_gap_is_in_projected_hull_but_clear_of_both_triangles():
    case = next(c for c in q.cases() if c['geometry'] == 'disconnected')
    vertices, faces = q.geometry(case); gap = vertices.mean(axis=0)
    assert vertices[:, 0].min() < gap[0] < vertices[:, 0].max()
    assert min(abs(vertices[:, 0])) > q.SPHERE_RADIUS + case['radius_m']
    assert faces.tolist() == [[0,1,2],[3,4,5]]
    assert [p['placement'] for p in q.probes(case)].count('gap_clearance') == 2


@pytest.mark.parametrize('change', ['radius','asset','face','joint'])
def test_xml_byte_binding_refuses_mutation(change):
    case = q.cases()[0]; xml = q.make_xml(case)
    substitutions = dict(radius=('radius="0"','radius=".1"'), asset=('<worldbody>','<asset/><worldbody>'),
                         face=('element="0 1 2"','element="0 2 1"'), joint=('freejoint','joint'))
    old, new = substitutions[change]
    with pytest.raises(ValueError): q.validate_xml(case, xml.replace(old,new))


def test_rigid_transform_and_barycentric_arithmetic_only():
    r,t,quat = q.frame('rigid_nonzero')
    assert np.max(abs(r.T@r-np.eye(3))) < 1e-14 and abs(np.linalg.det(r)-1) < 1e-14
    v=q.VERTICES@r.T+t
    np.testing.assert_allclose(q.barycentric(v.mean(axis=0),v),[1/3]*3,atol=1e-14,rtol=0)
    assert abs(np.linalg.norm(quat)-1) < 1e-14


def test_every_manufactured_result_and_global_frame_comparison():
    results=[manufactured_result(case) for case in q.cases()]
    assert all(q.check_case(row)['status']=='PASS' for row in results)
    assessments=q.summarize(results)
    assert all(row['status']=='BOUNDED_PRIMITIVE_CONTROLS_PASS' and row['compiled_cases']==6 for row in assessments)


@pytest.mark.parametrize('kind',['gap_contact','missing_penetration','normal','distance','vertices','jacobian','missing_probe','nonfinite'])
def test_bad_manufactured_measurement_never_qualifies(kind):
    result=manufactured_result(next(c for c in q.cases() if c['geometry']=='disconnected'))
    if kind=='gap_contact':result['probes'][2]['contacts']=copy.deepcopy(result['probes'][0]['contacts']);result['probes'][2]['contact_count']=1
    elif kind=='missing_penetration':result['probes'][0]['contacts']=[];result['probes'][0]['contact_count']=0
    elif kind=='normal':result['probes'][0]['contacts'][0]['normal_world']=[0,0,1]
    elif kind=='distance':result['probes'][0]['contacts'][0]['distance_m']+=1e-8
    elif kind=='vertices':result['vertex_max_error_m']=1.1e-14
    elif kind=='jacobian':result['probes'][0]['contacts'][0]['engine_normal_jacobian'][0]=1e-6
    elif kind=='missing_probe':result['probes'].pop()
    else:result['probes'][0]['contacts'][0]['distance_m']=float('nan')
    try: assert q.check_case(result)['status']=='FAIL'
    except ValueError: pass


def test_multibody_rotation_mismatch_is_separate_from_translation_support():
    result=manufactured_result(next(c for c in q.cases() if c['attachment']=='three_bodies'))
    result['probes'][0]['contacts'][0]['engine_normal_jacobian'][3]=.02
    check=q.check_case(result)
    assert check['status']=='PASS' and check['max_vertex_kinematic_rotation_jacobian_error']==.02
    assert check['three_owner_rotation_is_diagnostic_only']


def test_touching_excluded_contact_has_no_required_constraint_but_penetration_does():
    result=manufactured_result(q.cases()[0]);touch=result['probes'][1]
    contact=copy.deepcopy(result['probes'][0]['contacts'][0]);contact['constraint_active']=False;contact['efc_address']=-1
    contact['distance_m']=0.;contact['position_world']=[.04/3,.04/3,0.]
    touch['contacts']=[contact];touch['contact_count']=1
    assert q.check_case(result)['status']=='PASS'
    result['probes'][0]['contacts'][0]['constraint_active']=False
    assert q.check_case(result)['status']=='FAIL'


def test_compile_rejection_is_retained_as_not_qualified():
    results=[manufactured_result(c) for c in q.cases()]
    results[0]=dict(case=q.cases()[0],status='COMPILE_REJECTED',probes=[])
    assert q.summarize(results)[0]['status']=='NOT_QUALIFIED'
    with pytest.raises(ValueError):q.summarize(results[:-1])


def test_before_work_budget_counts_attempts_and_refuses_extra(tmp_path):
    b=q.EngineBudget(tmp_path)
    for _ in range(12):b.charge('model_compile_attempts')
    with pytest.raises(RuntimeError):b.charge('model_compile_attempts')
    assert b.counts['model_compile_attempts']==12 and b.forbidden_attempts
    with pytest.raises(RuntimeError):b.charge('physics_steps')
    assert b.counts['physics_steps']==0


def fake_data():
    return SimpleNamespace(qpos=np.zeros(7),qvel=np.zeros(6),qacc=np.zeros(6),ctrl=np.zeros(0),
                           time=0.,warning=[SimpleNamespace(number=0)])


@pytest.mark.parametrize('fault',['none','warning','qacc','time'])
def test_guarded_forward_with_fake_engine_only(tmp_path,fault):
    calls=[]; model=object(); data=fake_data()
    def forward(m,d):
        calls.append('forward')
        if fault=='warning':d.warning[0].number+=1
        elif fault=='qacc':d.qacc[0]=np.nan
        elif fault=='time':d.time=.01
    module=SimpleNamespace(MjModel=SimpleNamespace(from_xml_string=lambda xml:model),MjData=lambda m:data,
                           mj_forward=forward,mj_step=lambda *a:calls.append('bad_step'))
    b=q.EngineBudget(tmp_path)
    with q.engine_scope(module,b,['synthetic_xml']):
        m=module.MjModel.from_xml_string('synthetic_xml');d=module.MjData(m)
        if fault=='none':module.mj_forward(m,d)
        else:
            with pytest.raises(RuntimeError):module.mj_forward(m,d)
        with pytest.raises(RuntimeError):module.mj_step(m,d)
    assert calls==['forward'] and b.counts['explicit_forward_attempts']==1


def test_undeclared_xml_refused_before_compile(tmp_path):
    calls=[];module=SimpleNamespace(MjModel=SimpleNamespace(from_xml_string=lambda xml:calls.append(xml)),
        MjData=lambda m:None,mj_forward=lambda *a:None)
    b=q.EngineBudget(tmp_path)
    with q.engine_scope(module,b,['allowed']):
        with pytest.raises(RuntimeError):module.MjModel.from_xml_string('native_or_other')
    assert not calls and b.counts['model_compile_attempts']==0


def test_native_imports_refused():
    finder=q.ForbiddenImports()
    for name in ('manotorch','torch.nn','grasp_failure_prediction.anything','hug'):
        with pytest.raises(RuntimeError):finder.find_spec(name)
    assert finder.find_spec('xml.etree') is None


def test_artifact_hash_tamper_and_unindexed_arrays(tmp_path):
    file=tmp_path/'x.npz';np.savez_compressed(file,x=[1.]);row=q.artifact(file,'synthetic')
    q.verify_artifacts([row],tmp_path)
    with pytest.raises(ValueError):q.verify_artifacts([row,row],tmp_path)
    np.savez_compressed(tmp_path/'extra.npz',x=[2.])
    with pytest.raises(ValueError):q.verify_artifacts([row],tmp_path)
    file.write_bytes(b'changed')
    with pytest.raises(ValueError):q.verify_artifacts([row],tmp_path)


@pytest.mark.parametrize('token',['NaN','1e999'])
def test_json_nonfinite_refused(tmp_path,token):
    path=tmp_path/'bad.json';path.write_text('{"value":'+token+'}')
    with pytest.raises(ValueError):q.read(path)


def test_prepare_never_overwrites_or_uses_other_path(tmp_path):
    path=tmp_path/'existing.json';path.write_text('{}')
    with pytest.raises(FileExistsError):q.prepare(path)
    with pytest.raises(ValueError):q.prepare(tmp_path/'new.json')


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    root=tmp_path/'project';here=root/'research_artifacts'/'primitive';here.mkdir(parents=True)
    monkeypatch.setattr(q,'ROOT',root);monkeypatch.setattr(q,'HERE',here)
    runtime=dict(python='manufactured',executable='/manufactured/python',numpy='manufactured',mujoco='3.3.7')
    monkeypatch.setattr(q,'runtime',lambda:runtime)
    dependency=tmp_path/'runtime.bin';dependency.write_bytes(b'manufactured package bytes')
    monkeypatch.setattr(q,'runtime_files',lambda:[dependency])
    for path in q.source_files():path.write_text('manufactured source')
    (here/q.CONTROLS_NAME).write_text(json.dumps(dict(exit_code=0,integration_calls=0)))
    protocol=here/q.PROTOCOL_NAME;q.prepare(protocol)
    frozen_bytes={str(path.relative_to(root)):path.read_bytes() for path in [*q.source_files(),protocol]}
    commit='a'*40
    def git(command,text=False):
        if command[-1]=='HEAD' or command[-1].endswith('^{commit}'):return commit+'\n'
        return frozen_bytes[command[-1].split(':',1)[1]]
    monkeypatch.setattr(q.subprocess,'check_output',git)
    return protocol,dependency,commit


def test_manufactured_prepare_and_exact_git_blob_validation(frozen,monkeypatch):
    protocol,dependency,commit=frozen
    assert q.validate(protocol,commit)[1]==commit
    original=q.subprocess.check_output
    monkeypatch.setattr(q.subprocess,'check_output',lambda command,**kwargs:('b'*40+'\n')
        if command[-1]=='HEAD' else original(command,**kwargs))
    with pytest.raises(ValueError):q.validate(protocol,commit)


@pytest.mark.parametrize('kind',['source','runtime','budget','xml','admission'])
def test_frozen_protocol_or_input_tamper_refused(frozen,kind):
    protocol,dependency,commit=frozen
    if kind=='source':q.source_files()[0].write_bytes(b'mutated source')
    elif kind=='runtime':dependency.write_bytes(b'mutated runtime')
    else:
        p=q.read(protocol)
        if kind=='budget':p['budgets']['physics_steps']=1
        elif kind=='xml':p['xml_sha256'][q.cases()[0]['case_id']]='0'*64
        else:p['collection_or_training_allowed']=True
        protocol.write_text(json.dumps(p))
    with pytest.raises(ValueError):q.validate(protocol,commit)


def test_v1_unused_declaration_is_preserved_and_new_preparation_is_explicit(frozen):
    protocol,_,_=frozen
    old=q.HERE/'protocol.json';before=old.read_bytes()
    assert protocol.name=='protocol_v2.json' and q.CONTROLS_NAME=='controls_v2.json'
    assert q.read(protocol)['prior_unused_declaration']['sha256']==q.sha(old)
    assert q.read(protocol)['prior_unused_declaration']['execution_status']=='NEVER_EXECUTED'
    assert old.read_bytes()==before
    with pytest.raises(FileExistsError):q.prepare(old)
