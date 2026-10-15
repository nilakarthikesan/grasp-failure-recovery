"""Static/contract controls; no real integration or task outcomes."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation import approach_execution as approach
from grasp_failure_prediction.evaluation.recorded_runner import RecordedActuatedShadowRunner
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import RetargetedHandPose,default_dex_urdf_root
from grasp_failure_prediction.evaluation.schema import canonical_sha256
from grasp_failure_prediction.evaluation.table_clearance import TableClearanceConfiguration,plan_table_clearance


@pytest.fixture(autouse=True)
def no_physics(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('approach development controls forbid integration')
    for name in ('mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,name,forbidden)


def test_named_protocol_retains_base_task_and_hashes_configuration():
    base=load_protocol_registry().resolve('fixed_grasp_lift_v2');before=canonical_sha256(base)
    protocol=approach.grasp_approach_protocol(base)
    assert protocol.execution_protocol_id=='hug_grasp_approach_v1'
    assert protocol.parameters==base.parameters and protocol.success==base.success and protocol.states==base.states
    assert protocol.base_protocol_config_hash==before and canonical_sha256(base)==before
    changed=approach.grasp_approach_protocol(base,approach=approach.ApproachConfiguration(path_samples=65))
    assert canonical_sha256(changed)!=canonical_sha256(protocol)
    with pytest.raises(ValueError):approach.grasp_approach_protocol(load_protocol_registry().resolve('fixed_grasp_lift_v1'))
    with pytest.raises(ValueError):approach.grasp_approach_protocol(base,clearance=TableClearanceConfiguration(check_closing_targets=True))


@pytest.mark.parametrize('value',[True,2,258,3.5])
def test_bad_sample_count_refused(value):
    with pytest.raises(ValueError):approach.ApproachConfiguration(path_samples=value)


def test_candidates_keep_signed_denominator_and_use_world_grasp_axes():
    rows=approach.approach_candidates([1.,0,0,0],[0,0,0],[0,1,-1])
    assert [r['candidate_id'] for r in rows]==['palm_plus_y','palm_minus_y','finger_plus_z','finger_minus_z']
    assert [r['orientation_eligible'] for r in rows]==[True,False,False,True]
    assert all('outcome' not in r for r in rows)
    quat=np.array([np.sqrt(.5),0,0,np.sqrt(.5)])
    rotated=approach.approach_candidates(quat,[0,0,0],[-1,0,-1])
    np.testing.assert_allclose(rotated[0]['direction_world'],[-1,0,0],atol=1e-15)


def test_vertical_fallback_is_explicit_and_duplicate_is_retained_as_refusal():
    rows=approach.approach_candidates([1.,0,0,0],[0,0,0],[0,1,-1],configuration=approach.ApproachConfiguration(allow_vertical_fallback=True))
    assert len(rows)==5 and rows[-1]['candidate_id']=='vertical_negative_z'
    assert rows[-1]['rejection_reason']=='duplicate_direction'
    assert len(approach.approach_candidates([1.,0,0,0],[0,0,0],[0,1,-1]))==4


def test_ambiguous_objectward_direction_is_refused_without_a_label():
    rows=approach.approach_candidates([1.,0,0,0],[0,0,0],[1,0,0])
    assert all(not r['orientation_eligible'] for r in rows)


@pytest.fixture
def scene():
    root=default_dex_urdf_root()
    if not (root/'shadow_hand/shadow_hand_right.urdf').is_file():pytest.skip('optional Shadow assets absent')
    fixture=json.loads((Path(__file__).parent/'fixtures/shadow_hug_cube_target.json').read_text())
    protocol=approach.grasp_approach_protocol(load_protocol_registry().resolve('fixed_grasp_lift_v2'))
    runner=approach.GraspApproachRecordedRunner(protocol,root)
    runner.reset(seed=1,object_mass_kg=.18,object_position_m=np.array([1.,1.,1.]),object_orientation_wxyz=np.array([1.,0,0,0]))
    pose=RetargetedHandPose(tuple(fixture['joint_names']),np.asarray(fixture['joints']),np.zeros((21,3)),np.zeros((10,3)),np.eye(4))
    position=np.array([0.,0.,.13]);quat=np.array([np.cos(np.pi/6),0,np.sin(np.pi/6),0])
    return runner,pose,position,quat


def test_sampling_is_static_nonmutating_and_reports_spacing(scene):
    runner,pose,pos,quat=scene
    wrist=plan_table_clearance(runner,pose,pos,quat)
    assert wrist.feasible
    before=runner.data.qpos.copy();mass=runner.model.body_mass.copy()
    report=approach.sampled_approach_clearance(runner,wrist.adjusted_pose,wrist.root_position_m,wrist.root_quaternion_wxyz,[0.,0.,-1.],samples=33)
    assert len(report['samples'])==33
    assert report['samples'][0]['alpha']==0 and report['samples'][-1]['alpha']==1
    assert report['maximum_sample_spacing_m']==.0025
    assert report['minimum_table_clearance_m']>=.010
    assert report['minimum_proximal_object_distance_m']>=.001
    np.testing.assert_array_equal(runner.data.qpos,before);np.testing.assert_array_equal(runner.model.body_mass,mass)
    assert runner.data.time==0 and runner.prediction_recorder.sample_count==0


def test_fixed_order_choice_and_all_candidate_refusals_are_recorded(scene,monkeypatch):
    runner,pose,pos,quat=scene;calls=[]
    def measure(*args,**kwargs):
        calls.append(np.asarray(args[4]).copy())
        return {'minimum_table_clearance_m':.02,'minimum_proximal_object_distance_m':.02,'samples':[]}
    monkeypatch.setattr(approach,'sampled_approach_clearance',measure)
    plan=approach.plan_grasp_approach(runner,pose,pos,quat)
    feasible=[r for r in plan.candidates if r['status']=='FEASIBLE']
    assert plan.selected_candidate_id==feasible[0]['candidate_id']
    assert len(calls)==len(feasible)
    record=plan.as_record();assert len(record['candidates'])==4 and record['outcome'] is None
    assert record['continuous_swept_collision_certificate'] is False
    for i,name in enumerate(pose.joint_names):
        if name not in {'WRJ1','WRJ2'}:assert plan.adjusted_pose.qpos[i]==pose.qpos[i]


def test_motion_receives_adjusted_wrist_but_original_world_palm(scene,monkeypatch):
    runner,pose,pos,quat=scene;forwarded={}
    monkeypatch.setattr(approach,'sampled_approach_clearance',lambda *a,**kw:dict(minimum_table_clearance_m=.02,minimum_proximal_object_distance_m=.02,samples=[]))
    def execute(self,adjusted,**kwargs):forwarded.update(pose=adjusted,**kwargs);return 'no dynamics'
    monkeypatch.setattr(RecordedActuatedShadowRunner,'execute',execute)
    assert runner.execute(pose,grasp_palm_position_m=pos,grasp_palm_quaternion_wxyz=quat)=='no dynamics'
    np.testing.assert_array_equal(forwarded['grasp_palm_position_m'],pos)
    np.testing.assert_array_equal(forwarded['grasp_palm_quaternion_wxyz'],quat)
    assert forwarded['pose'] is runner.approach_plan.adjusted_pose
    assert runner.data.time==0


def test_all_paths_refused_preserves_unexecuted_refusal(scene,monkeypatch):
    runner,pose,pos,quat=scene
    monkeypatch.setattr(approach,'sampled_approach_clearance',lambda *a,**kw:dict(minimum_table_clearance_m=-.001,minimum_proximal_object_distance_m=.02,samples=[]))
    monkeypatch.setattr(RecordedActuatedShadowRunner,'execute',lambda *a,**kw:pytest.fail('refusal must precede execution'))
    with pytest.raises(approach.ApproachPlanningRejected) as error:
        runner.execute(pose,grasp_palm_position_m=pos,grasp_palm_quaternion_wxyz=quat)
    assert error.value.outcome is None and error.value.mapping_status=='unreachable_pregrasp'
    assert runner.prediction_recorder.sample_count==0 and runner.data.time==0
    assert all(row['status']=='REFUSED' for row in runner.approach_plan.candidates)


def test_override_is_rejected_before_planning(scene,monkeypatch):
    runner,pose,pos,quat=scene
    monkeypatch.setattr(approach,'plan_grasp_approach',lambda *a,**kw:pytest.fail('do not plan unversioned overrides'))
    with pytest.raises(ValueError,match='override'):
        runner.execute(pose,grasp_palm_position_m=pos,grasp_palm_quaternion_wxyz=quat,approach_direction_world=[0,0,-1])


def test_unversioned_runner_refused_before_model_creation(monkeypatch):
    monkeypatch.setattr(RecordedActuatedShadowRunner,'__init__',lambda *a,**kw:pytest.fail('do not instantiate'))
    with pytest.raises(TypeError,match='version'):
        approach.GraspApproachRecordedRunner(load_protocol_registry().resolve('fixed_grasp_lift_v2'),default_dex_urdf_root())
