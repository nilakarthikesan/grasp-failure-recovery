"""Only manufactured metadata/arrays; no real MANO/engine operations."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('manufactured_skin_case',HERE/'skin_case.py')
case=importlib.util.module_from_spec(spec);spec.loader.exec_module(case)
skin=case.load_skin()


def fake_contract():
    names=['wrist_tx','wrist_ty','wrist_tz','wrist_rot']+[f'mano_joint_{j}' for j in range(1,16)]
    body_names=[f'mano_body_{j}' for j in range(16)]
    rows=[dict(name=name,qpos_address=k if k<3 else 3+4*(k-3),
        dof_address=k if k<3 else 3+3*(k-3),type=2 if k<3 else 1) for k,name in enumerate(names)]
    model=SimpleNamespace(nq=67,nv=51,nu=51,njnt=19,
        jnt_qposadr=np.array([r['qpos_address'] for r in rows]),
        jnt_dofadr=np.array([r['dof_address'] for r in rows]),
        jnt_type=np.array([r['type'] for r in rows]),
        jnt_bodyid=np.array([1,1,1,1]+list(range(2,17))),
        body_parentid=np.array([0,0]+[1+skin.PARENTS[j] for j in range(1,16)]),
        body_quat=np.tile([1.,0.,0.,0.],(17,1)),body_pos=np.zeros((17,3)),
        jnt_axis=np.vstack((np.eye(3),np.zeros((16,3)))),jnt_pos=np.zeros((19,3)))
    def name2id(model,kind,name):
        return names.index(name) if kind=='joint' else body_names.index(name)+1
    mj=SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY='body',mjOBJ_JOINT='joint'),mj_name2id=name2id)
    return model,mj,rows


def test_named_state_requires_exact_actual_layout_without_mjdata_or_fk():
    model,mj,rows=fake_contract(); result=case.named_model_contract(model,mj,rows)
    assert result['nq']==67 and result['nv']==51 and result['nu']==51
    assert result['named_joint_addresses']==rows
    assert result['geometric_simulator_articulation_status']=='NOT_RUN'
    assert result['physical_contact_status']=='NOT_RUN'


@pytest.mark.parametrize('defect',['address','type','tree','root','slide','anchor','nan_rest','saved_layout'])
def test_wrong_actual_tangent_metadata_refuses_without_normalizing(defect):
    model,mj,rows=fake_contract()
    if defect=='address':model.jnt_dofadr[4]+=1
    elif defect=='type':model.jnt_type[5]=2
    elif defect=='tree':model.body_parentid[3]=1
    elif defect=='root':model.body_pos[1,0]=.01
    elif defect=='slide':model.jnt_axis[0]=[0.,1.,0.]
    elif defect=='anchor':model.jnt_pos[0,0]=.01
    elif defect=='nan_rest':model.body_quat[5,0]=np.nan
    else:rows[4]=dict(rows[4],dof_address=999)
    with pytest.raises(ValueError):case.named_model_contract(model,mj,rows)


def test_body_rest_rotation_composes_before_right_local_joint_increment():
    q=np.zeros(67);q[:3]=[.1,-.2,.3]
    q[3:]=skin.rotation_increment(torch.linspace(-.4,.6,48,dtype=torch.float64).reshape(16,3)).numpy().reshape(-1)
    rest=skin.rotation_increment(torch.linspace(.2,-.1,48,dtype=torch.float64).reshape(16,3)).numpy()
    saved=q.copy();delta=np.linspace(-.01,.02,51)
    integrated=case.expected_integrated_qpos(q,delta,skin)
    t,bodies=case.body_inputs(integrated,rest,skin)
    expected=skin.quaternion_product(torch.from_numpy(rest),skin.quaternion_product(
        torch.from_numpy(q[3:].reshape(16,4)),skin.rotation_increment(torch.from_numpy(delta[3:].reshape(16,3)))))
    torch.testing.assert_close(bodies,expected,rtol=0,atol=2e-16)
    np.testing.assert_array_equal(q,saved)
    np.testing.assert_array_equal(t.numpy(),q[:3]+delta[:3])
    # Known noncommuting90degree x/y rotations distinguish multiplication
    # order analytically, without an invented lower bound on random angles.
    half=torch.sqrt(torch.tensor(.5,dtype=torch.float64))
    qx=torch.stack((half,half,half*0,half*0));qy=torch.stack((half,half*0,half,half*0))
    expected_xy=torch.tensor([.5,.5,.5,.5],dtype=torch.float64)
    expected_yx=torch.tensor([.5,.5,.5,-.5],dtype=torch.float64)
    eps=torch.finfo(torch.float64).eps
    torch.testing.assert_close(skin.quaternion_product(qx,qy),expected_xy,rtol=0,atol=2*eps)
    torch.testing.assert_close(skin.quaternion_product(qy,qx),expected_yx,rtol=0,atol=2*eps)


def test_zero_delta_retains_literal_saved_named_state_exactly():
    q=np.zeros(67);q[:3]=[.2,.1,-.3];q[3:]=np.tile([1.,0.,0.,0.],16)
    np.testing.assert_array_equal(case.expected_integrated_qpos(q,np.zeros(51),skin),q)
    with pytest.raises(ValueError):case.expected_integrated_qpos(q[:66],np.zeros(51),skin)
    with pytest.raises(ValueError):case.expected_integrated_qpos(q,np.full(51,np.nan),skin)


def test_forward_jacobian_callback_can_refuse_before_any_skin_evaluation(monkeypatch):
    calls=[]
    def forbidden(*a,**kw):calls.append('skin');raise AssertionError('must not evaluate')
    monkeypatch.setattr(skin,'tangent_vertices',forbidden)
    def quota():raise RuntimeError('budget exhausted')
    with pytest.raises(RuntimeError,match='budget exhausted'):
        skin.skin_jacobian(torch.zeros(3,dtype=torch.float64),torch.zeros((16,4)),None,None,evaluation_callback=quota)
    assert calls==[]


def test_named_state_baseline_cannot_pass_displaced_from_independent_saved_skin():
    target=np.zeros((778,3));same=target.copy()
    checks,metrics=case.baseline_checks(same,same,target)
    assert all(checks.values()) and all(value==0 for value in metrics.values())
    displaced=same.copy();displaced[25,2]=6e-6
    checks,_=case.baseline_checks(displaced,same,target)
    assert checks['source32_named_state_baseline_matches_saved'] is False
    assert checks['continuous64_named_state_baseline_matches_saved'] is True
    checks,_=case.baseline_checks(same,displaced,target)
    assert checks['continuous64_named_state_baseline_matches_saved'] is False
    displaced[0,0]=np.nan
    with pytest.raises(ValueError):case.baseline_checks(same,displaced,target)


def test_declaration_is_metadata_only_and_has_fixed_operation_inventory(tmp_path,monkeypatch):
    prior=tmp_path/'prior';prior.mkdir()
    inputs={key:dict(path=str(tmp_path/key),sha256='a'*64) for key in ('decoded_reference','shape_asset')}
    p=dict(cases={str(i):dict(inputs=inputs,expected_shape_sha256='a'*64) for i in (20,28)},base_xml=str(tmp_path/'hand.xml'))
    (prior/'protocol.json').write_text(json.dumps(p))
    for i in (20,28):
        out=prior/f'run_v1/{i}';out.mkdir(parents=True)
        (out/'result.json').write_text(json.dumps(dict(source_receipt=dict(
            compiled_readback_artifact=dict(path=str(tmp_path/'saved.npz'),sha256='b'*64),named_joint_addresses=[]))))
    monkeypatch.setattr(case,'PRIOR',prior)
    monkeypatch.setattr(np,'load',lambda *a,**k: (_ for _ in ()).throw(AssertionError('no array preparation')))
    d=case.declaration()
    assert set(d['cases'])=={'20','28'} and len(d['motion_grid']['poses'])==109
    assert d['budget']['mujoco_integratePos_calls']==109
    assert d['budget']['jacobian_forward_evaluations']==51
    assert d['budget']['finite_difference_evaluations']==102
    assert d['budget']['data_constructions']==d['budget']['mujoco_forward_calls']==d['budget']['physics_steps']==0
    assert d['gates']['numerical_limits_are_contact_allowances'] is False
    assert d['runtime_skin']['torch_threads']==1
