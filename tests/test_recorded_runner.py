"""Static bridge tests: never instantiate a model or advance physics."""

from types import SimpleNamespace

import numpy as np
import pytest

from grasp_failure_prediction.evaluation.prediction_recording import PredictionRecorder
from grasp_failure_prediction.evaluation.recorded_runner import RecordedActuatedShadowRunner
from grasp_failure_prediction.evaluation.runner import ExecutionState


def stub_runner():
    runner = object.__new__(RecordedActuatedShadowRunner)
    runner._hand_joint_names = ("J1", "J2")
    runner._hand_qpos_addresses = np.array([2, 3])
    runner._recorded_dof_addresses = np.array([1, 2])
    runner._actuator_ids = np.array([0, 1])
    runner._recorded_target_limits = np.array([[-1., 1.], [-.5, .5]])
    runner.physics_timestep_s = .002
    runner.prediction_recorder = PredictionRecorder(runner._hand_joint_names, runner._recorded_target_limits)
    runner.data = SimpleNamespace(time=.04, ctrl=np.array([2., -.75]),
                                  qpos=np.array([100., 200., .2, .3]),
                                  qvel=np.array([100., .4, .5]))
    runner._trace = [SimpleNamespace(
        time_s=.04, state=ExecutionState.CLOSE_FINGERS, contact_count=2,
        hand_object_contact=True, hand_table_contact=False, object_table_contact=True,
        maximum_contact_normal_force_n=10.,
    )]
    return runner


def test_bridge_uses_actual_controls_and_measured_qpos_qvel():
    runner = stub_runner()
    runner._capture_prediction_step(ExecutionState.CLOSE_FINGERS, np.array([2., -.75]), 0.)
    values = runner.prediction_recorder.arrays()
    np.testing.assert_array_equal(values["hand_qpos"], [[.2, .3]])
    np.testing.assert_array_equal(values["hand_qvel"], [[.4, .5]])
    np.testing.assert_array_equal(values["requested_target_rad"], [[2., -.75]])
    np.testing.assert_array_equal(values["applied_target_rad"], [[1., -.5]])
    assert "full_qpos" not in values and "contact_count" not in values


def test_bridge_rejects_trace_time_mismatch():
    runner = stub_runner()
    runner.data.time = .08
    with pytest.raises(ValueError, match="not aligned"):
        runner._capture_prediction_step(ExecutionState.CLOSE_FINGERS, np.array([2., -.75]), 0.)
    assert runner.prediction_recorder.sample_count == 0


def test_bridge_rejects_requested_actual_control_mismatch():
    runner = stub_runner()
    with pytest.raises(ValueError, match="servo controls differ"):
        runner._capture_prediction_step(ExecutionState.CLOSE_FINGERS, np.zeros(2), 0.)
    assert runner.prediction_recorder.sample_count == 0
