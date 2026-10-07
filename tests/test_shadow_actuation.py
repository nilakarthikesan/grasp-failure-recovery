"""Optional integration invariants for the partner's Shadow assets/runtime."""
import importlib.util
from pathlib import Path
import sys
import json

import numpy as np
import pytest

pytest.importorskip('grasp_failure_prediction.evaluation')
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.runner import build_tabletop_model


def test_fixed_closure_preserves_target_and_nonflexion_commands():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    try:
        from actuated_shadow_pilot import force_close_targets
        names = ('WRJ1', 'THJ5', 'FFJ2', 'THJ1')
        target = np.array([.1, .2, .9, .3])
        limits = np.array([[-1, 1], [-1, 1], [0, 1], [0, 1]])
        command = force_close_targets(names, target, limits, .2)
        np.testing.assert_allclose(command, [.1, .2, 1., .5])
        np.testing.assert_array_equal(target, [.1, .2, .9, .3])
        np.testing.assert_array_equal(force_close_targets(names, target, limits, 0), target)
        with pytest.raises(ValueError):
            force_close_targets(names, target, limits, float('nan'))
    finally:
        sys.path.pop(0)


def test_actuation_preserves_collision_geometry_and_has_finite_force_limits():
    root=Path(__file__).resolve().parents[1]
    assets=root/'runs/hug_integration_check/dex-urdf/robots/hands'
    if not assets.is_dir():pytest.skip('Optional downloaded Shadow asset checkout is absent')
    sys.path.insert(0,str(root/'scripts'))
    try:
        from actuated_shadow_pilot import add_actuation
        base=build_tabletop_model(assets)
        actuated=add_actuation(base)
    finally:
        sys.path.pop(0)
    assert actuated.nu==24
    assert actuated.neq==1  # ideal arm, never the object
    assert actuated.ngeom==base.ngeom
    np.testing.assert_array_equal(actuated.geom_type,base.geom_type)
    np.testing.assert_allclose(actuated.geom_size,base.geom_size)
    np.testing.assert_allclose(actuated.geom_pos,base.geom_pos)
    np.testing.assert_allclose(actuated.body_mass[:base.nbody],base.body_mass)
    assert np.all(actuated.actuator_forcelimited)
    assert np.isfinite(actuated.actuator_forcerange).all()
    assert np.all(actuated.actuator_forcerange[:,1]>0)
    object_body=base.body('object').id
    assert object_body not in actuated.eq_obj1id
    assert object_body not in actuated.eq_obj2id


def test_engineered_shadow_pickup_and_open_thumb_negative_control():
    root=Path(__file__).resolve().parents[1]
    assets=root/'runs/hug_integration_check/dex-urdf/robots/hands'
    if not assets.is_dir():pytest.skip('Optional downloaded Shadow asset checkout is absent')
    fixture=json.loads((root/'tests/fixtures/shadow_cube_baseline.json').read_text())
    sys.path.insert(0,str(root/'scripts'))
    try:
        from actuated_shadow_pilot import ActuatedShadowRunner, run_candidate
        from grasp_failure_prediction.evaluation.retargeting import RetargetedHandPose
        from dataclasses import replace
        protocol=load_protocol_registry().resolve('fixed_grasp_lift_v1')
        protocol=protocol.model_copy(update={'parameters':protocol.parameters.model_copy(update={'grip_command':1.})})
        runner=ActuatedShadowRunner(protocol,assets)
        snapshot={'qpos':runner.model.qpos0.copy(),'qvel':np.zeros(runner.model.nv)}
        address=runner._object_qpos_address
        snapshot['qpos'][address:address+3]=[0,0,.024784489]
        pose=RetargetedHandPose(tuple(fixture['joint_names']),np.array(fixture['joints']),
                               np.zeros((21,3)),np.zeros((10,3)),np.eye(4))
        report,_=run_candidate(runner,pose,np.array(fixture['palm_position_m']),
                               np.array(fixture['quaternion_wxyz']),snapshot)
        assert report['score']['success']
        assert report['opposition_time_s']['hold']>=2
        assert report['peak_force_fraction_of_configured_limit']<=1.000001
        open_joints=pose.qpos.copy()
        for i,name in enumerate(pose.joint_names):
            if name.startswith('TH'):open_joints[i]=0
        report,_=run_candidate(runner,replace(pose,qpos=open_joints),np.array(fixture['palm_position_m']),
                               np.array(fixture['quaternion_wxyz']),snapshot)
        assert not report['score']['success']
    finally:
        sys.path.pop(0)


def test_hug_target_needs_closing_pressure_without_object_aware_fit():
    root = Path(__file__).resolve().parents[1]
    assets = root / 'runs/hug_integration_check/dex-urdf/robots/hands'
    if not assets.is_dir():
        pytest.skip('Optional downloaded Shadow asset checkout is absent')
    fixture = json.loads((root / 'tests/fixtures/shadow_hug_cube_target.json').read_text())
    sys.path.insert(0, str(root / 'scripts'))
    try:
        from actuated_shadow_pilot import ActuatedShadowRunner, run_candidate
        from grasp_failure_prediction.evaluation.retargeting import RetargetedHandPose
        pose = RetargetedHandPose(tuple(fixture['joint_names']), np.array(fixture['joints']),
                                 np.zeros((21, 3)), np.zeros((10, 3)), np.eye(4))
        protocol = load_protocol_registry().resolve('fixed_grasp_lift_v1')
        outcomes = []
        for delta in (0., .1):
            runner = ActuatedShadowRunner(protocol, assets, force_close_delta_rad=delta)
            snapshot = {'qpos': runner.model.qpos0.copy(), 'qvel': np.zeros(runner.model.nv)}
            address = runner._object_qpos_address
            snapshot['qpos'][address:address + 3] = [0, 0, .024784489]
            report, _ = run_candidate(runner, pose, np.array(fixture['palm_position_m']),
                                      np.array(fixture['quaternion_wxyz']), snapshot)
            outcomes.append(report)
        assert not outcomes[0]['score']['success']
        assert outcomes[0]['opposition_time_s']['hold'] == 0
        assert outcomes[1]['score']['success']
        assert outcomes[1]['score']['hold_duration_s'] == 2.
        assert not outcomes[1]['score']['approach_collision']
        assert outcomes[1]['opposition_time_s']['hold'] >= 2.
        assert outcomes[1]['peak_force_fraction_of_configured_limit'] <= 1.000001
        np.testing.assert_array_equal(pose.qpos, fixture['joints'])
    finally:
        sys.path.pop(0)
