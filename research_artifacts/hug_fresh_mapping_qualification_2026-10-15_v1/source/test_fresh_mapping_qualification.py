"""Manufactured observation/gate controls; no optimization or physics."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_failure_prediction.evaluation.fresh_mapping_qualification import (
    capture_solve, compare_commands, independent_local_operator_landmarks, settings_gate,
)


@dataclass
class Provenance:
    profile_id: str='profile'
    config_sha256: str='a'*64
    vector_scaling: float=1.
    low_pass_filter_enabled: bool=False
    temporal_norm_delta: float=0.
    reset_before_proposal: bool=True
    refinement_calls: int=2


class Sequence:
    def __init__(self):
        self.optimizer=SimpleNamespace(opt=SimpleNamespace(last_optimize_result=lambda:4))
        self.calls=0
    def retarget(self, value):
        self.calls+=1
        return np.asarray(value)+self.calls*.01


class Retargeter:
    def __init__(self, refuse=False):
        self._retargeting=Sequence()
        self.refuse=refuse
    def retarget(self, value):
        for _ in range(2):
            raw=self._retargeting.retarget(value)
        if self.refuse:
            raise ValueError('manufactured refusal')
        return SimpleNamespace(qpos=np.clip(raw,-1,1),joint_names=('a','b'),retargeting_provenance=Provenance())


def test_capture_does_not_change_solver_state_or_returned_commands():
    plain, observed=Retargeter(),Retargeter()
    expected=plain.retarget(np.array([2.,0.]))
    original=observed._retargeting.retarget
    pose,record=capture_solve(observed,np.array([2.,0.]))
    np.testing.assert_array_equal(pose.qpos,expected.qpos)
    assert record['raw_refinement_qpos_rad']==[[2.01,.01],[2.02,.02]]
    assert record['optimizer_status_codes']==[4,4]
    assert record['max_final_clipping_rad']==1.02
    assert observed._retargeting.retarget==original
    assert plain._retargeting.calls==observed._retargeting.calls==2
    assert settings_gate(record,Provenance().__dict__)


def test_capture_retains_refusal_and_restores_method():
    target=Retargeter(refuse=True)
    original=target._retargeting.retarget
    pose,record=capture_solve(target,np.zeros(2))
    assert pose is None and record['status']=='refused'
    assert record['message']=='manufactured refusal'
    assert len(record['raw_refinement_qpos_rad'])==2
    assert target._retargeting.retarget==original


@pytest.mark.parametrize('change', ['profile_id','config_sha256','temporal_norm_delta','reset_before_proposal','refinement_calls'])
def test_settings_gate_rejects_changed_settings(change):
    _,record=capture_solve(Retargeter(),np.zeros(2))
    expected=Provenance().__dict__.copy()
    expected[change]='changed'
    assert not settings_gate(record,expected)


@pytest.mark.parametrize('statuses', [[4,-1],[4,0],[4,None],[4,True],[4]])
def test_settings_gate_requires_all_observed_success_codes(statuses):
    _,record=capture_solve(Retargeter(),np.zeros(2))
    record['optimizer_status_codes']=statuses
    assert not settings_gate(record,Provenance().__dict__)


def test_order_comparison_is_named_and_tolerance_is_explicit():
    base=dict(status='returned',joint_names=['a','b'],qpos_rad=[.1,.2])
    tiny=dict(base,qpos_rad=[.1,.2+5e-11])
    assert compare_commands(base,tiny,1e-10)['passed']
    assert not compare_commands(base,tiny,1e-12)['passed']
    assert not compare_commands(base,tiny,1e-10)['bit_exact']
    assert not compare_commands(base,dict(base,joint_names=['b','a']),1e-10)['passed']
    assert not compare_commands(base,dict(status='refused'),1e-10)['passed']


def test_independent_frame_with_nonidentity_rotation_and_translation():
    angle=.73
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    translation=np.array([.3,-.4,.7])
    local=np.arange(63).reshape(21,3)/1000
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=translation
    grasp=SimpleNamespace(landmarks_3d=local@rotation.T+translation,T_camera_wrist=transform)
    basis=np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])
    np.testing.assert_allclose(independent_local_operator_landmarks(grasp),local@basis.T,atol=1e-15,rtol=0)
