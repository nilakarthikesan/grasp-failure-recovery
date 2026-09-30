from __future__ import annotations

import numpy as np

from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.runner import (
    ExecutionState,
    ExecutionTrace,
    RunnerStep,
)
from grasp_failure_prediction.evaluation.scoring import FailureType, score_trace


def make_step(
    index: int,
    state: ExecutionState,
    height: float,
    *,
    hand_object: bool = False,
    hand_table: bool = False,
) -> RunnerStep:
    return RunnerStep(
        index=index,
        time_s=index * 0.04,
        state=state,
        palm_position_m=np.zeros(3),
        hand_qpos=np.zeros(24),
        object_position_m=np.array([0.0, 0.0, height]),
        contact_count=int(hand_object or hand_table),
        hand_object_contact=hand_object,
        hand_table_contact=hand_table,
        object_table_contact=False,
    )


def protocol():
    return load_protocol_registry().resolve("fixed_grasp_lift_v1")


def test_success_requires_contact_lift_and_full_hold() -> None:
    steps = [make_step(0, ExecutionState.RESET, 0.03)]
    steps += [
        make_step(index, ExecutionState.LIFT, 0.18, hand_object=True)
        for index in range(1, 3)
    ]
    steps += [
        make_step(index, ExecutionState.HOLD, 0.18, hand_object=True)
        for index in range(3, 53)
    ]
    result = score_trace(ExecutionTrace(tuple(steps)), protocol(), control_timestep_s=0.04)
    assert result.success
    assert result.failure_type is None
    assert result.hold_duration_s == 2.0


def test_no_contact_is_distinct_from_failed_lift() -> None:
    steps = (
        make_step(0, ExecutionState.RESET, 0.03),
        make_step(1, ExecutionState.CLOSE_FINGERS, 0.03),
        make_step(2, ExecutionState.SCORE, 0.03),
    )
    result = score_trace(ExecutionTrace(steps), protocol(), control_timestep_s=0.04)
    assert result.failure_type is FailureType.NO_CONTACT


def test_drop_after_reaching_height_is_labeled_drop() -> None:
    steps = (
        make_step(0, ExecutionState.RESET, 0.03),
        make_step(1, ExecutionState.LIFT, 0.18, hand_object=True),
        make_step(2, ExecutionState.HOLD, 0.04, hand_object=False),
        make_step(3, ExecutionState.SCORE, 0.03),
    )
    result = score_trace(ExecutionTrace(steps), protocol(), control_timestep_s=0.04)
    assert result.failure_type is FailureType.DROP
