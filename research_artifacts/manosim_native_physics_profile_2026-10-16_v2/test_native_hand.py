"""Manufactured scene-index/control contracts; no model or FK execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('new_native_hand',Path(__file__).with_name('native_hand.py'));h=importlib.util.module_from_spec(s);sys.modules[s.name]=h;s.loader.exec_module(h)


def manufactured():
    model=SimpleNamespace(nq=74,nv=57,nu=51,njnt=20,jnt_type=np.r_[np.full(3,2),np.ones(16),0],jnt_qposadr=np.r_[np.arange(3),3+4*np.arange(16),67],jnt_dofadr=np.r_[np.arange(3),3+3*np.arange(16),51],jnt_bodyid=np.r_[np.ones(4),np.arange(2,17),17],body_parentid=np.r_[0,0,np.ones(15),0],body_gravcomp=np.r_[0,np.ones(16),0])
    class API:
        BODY_NAMES=tuple(f'mano_body_{i}' for i in range(16))
        @staticmethod
        def validate_compiled_contract(view,resolve,assets,rest):
            assert (view.nq,view.nv,view.nu,view.njnt)==(67,51,51,19)
            assert view.jnt_qposadr[-1]==67 # Extra object arrays remain part of full fingerprint.
            return SimpleNamespace(metadata=lambda:{'manufactured':True},model_signature='manufactured_scene')
        @staticmethod
        def verify_compiled_contract(model,c):return True
    resolve=lambda kind,name:19 if name=='object_free' else API.BODY_NAMES.index(name)+1
    return model,API,resolve


def test_hand_prefix_with_free_object_and_only_control_write():
    model,api,res=manufactured();c=h.validate_hand_scene(model,res,{},None,object_joint_name='object_free',api=api)
    d=SimpleNamespace(qpos=np.arange(74,dtype=float),qvel=np.arange(57,dtype=float),ctrl=np.zeros(51),qfrc_applied=np.zeros(57));before={k:v.copy() for k,v in vars(d).items()}
    q,v=h.read_hand_state(d,c);q[:]=0;v[:]=0
    np.testing.assert_array_equal(d.qpos,before['qpos']);np.testing.assert_array_equal(d.qvel,before['qvel'])
    h.write_hand_controls(d,SimpleNamespace(ctrl=np.ones(51),model_signature='manufactured_scene'),c)
    np.testing.assert_array_equal(d.qpos[67:],before['qpos'][67:]);np.testing.assert_array_equal(d.qvel,before['qvel']);np.testing.assert_array_equal(d.qfrc_applied,before['qfrc_applied']);assert (d.ctrl==1).all()
    assert c.metadata()['object_state_mutated_by_controller'] is False and c.metadata()['new_collection_or_training_allowed'] is False


@pytest.mark.parametrize('fault',['address','dof','type','compensation','dimensions'])
def test_unsupported_free_object_layout_refuses_instead_of_silent_remap(fault):
    m,a,r=manufactured()
    if fault=='address':m.jnt_qposadr[19]=0
    elif fault=='dof':m.jnt_dofadr[19]=0
    elif fault=='type':m.jnt_type[19]=1
    elif fault=='compensation':m.body_gravcomp[17]=1
    else:m.nv+=1
    with pytest.raises(ValueError):h.validate_hand_scene(m,r,{},None,object_joint_name='object_free',api=a)


def test_foreign_scene_command_cannot_write_control():
    model,api,res=manufactured();c=h.validate_hand_scene(model,res,{},None,object_joint_name='object_free',api=api)
    d=SimpleNamespace(ctrl=np.zeros(51))
    with pytest.raises(ValueError,match='identity'):h.write_hand_controls(d,SimpleNamespace(ctrl=np.ones(51),model_signature='other'),c)
    assert not np.any(d.ctrl)
