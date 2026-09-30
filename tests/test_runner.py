from __future__ import annotations

import numpy as np

from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import (
    ShadowHandRetargeter,
    default_dex_urdf_root,
)
from grasp_failure_prediction.evaluation.runner import (
    AdroitShadowRunner,
    ExecutionState,
)
from test_retargeting import synthetic_hug_prediction


def make_runner() -> AdroitShadowRunner:
    protocol = load_protocol_registry().resolve("fixed_grasp_lift_v1")
    return AdroitShadowRunner(protocol, default_dex_urdf_root())


def execute_synthetic(runner: AdroitShadowRunner):
    grasp, _ = synthetic_hug_prediction()
    pose = ShadowHandRetargeter().retarget(grasp)
    runner.reset(
        seed=9000,
        object_mass_kg=0.18,
        object_position_m=np.array([0.0, 0.0, 0.03]),
        object_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    return runner.execute(
        pose,
        grasp_palm_position_m=np.array([0.0, 0.0, 0.09]),
        grasp_palm_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_fixed_protocol_executes_reviewed_state_order() -> None:
    trace = execute_synthetic(make_runner())
    assert trace.states == tuple(ExecutionState)


def test_fixed_protocol_reaches_configured_lift_height() -> None:
    runner = make_runner()
    trace = execute_synthetic(runner)
    approach_end = [
        step for step in trace.steps if step.state is ExecutionState.APPROACH
    ][-1]
    hold_end = [step for step in trace.steps if step.state is ExecutionState.HOLD][-1]
    expected = runner.protocol.parameters.lift_height_m
    assert np.isclose(hold_end.palm_position_m[2] - approach_end.palm_position_m[2], expected)


def test_fixed_protocol_is_deterministic() -> None:
    first = execute_synthetic(make_runner())
    second = execute_synthetic(make_runner())
    first_object = np.stack([step.object_position_m for step in first.steps])
    second_object = np.stack([step.object_position_m for step in second.steps])
    np.testing.assert_allclose(first_object, second_object, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first.steps[-1].hand_qpos, second.steps[-1].hand_qpos)
