"""Outcome scoring and failure taxonomy for fixed grasp/lift traces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .registry import ExecutionProtocolSpec
from .runner import ExecutionState, ExecutionTrace


class FailureType(str, Enum):
    INVALID_RETARGET = "invalid_retarget"
    UNREACHABLE_PREGRASP = "unreachable_pregrasp"
    APPROACH_COLLISION = "approach_collision"
    NO_CONTACT = "no_contact"
    FAILED_ACQUISITION = "failed_acquisition"
    SLIP_DURING_HOLD = "slip_during_hold"
    DROP = "drop"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"


@dataclass(frozen=True)
class ScoredOutcome:
    success: bool
    failure_type: FailureType | None
    acquired_object: bool
    maximum_lift_m: float
    hold_duration_s: float
    final_object_height_m: float
    approach_collision: bool


def _longest_true_duration(values: list[bool], timestep_s: float) -> float:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * timestep_s


def score_trace(
    trace: ExecutionTrace,
    protocol: ExecutionProtocolSpec,
    *,
    control_timestep_s: float,
) -> ScoredOutcome:
    if not trace.steps:
        raise ValueError("cannot score an empty execution trace")
    heights = np.asarray([step.object_position_m[2] for step in trace.steps])
    initial_height = float(heights[0])
    maximum_lift = max(0.0, float(np.max(heights) - initial_height))
    final_height = float(heights[-1])
    manipulation_states = {
        ExecutionState.CLOSE_FINGERS,
        ExecutionState.LIFT,
        ExecutionState.HOLD,
    }
    acquired = any(
        step.hand_object_contact and step.state in manipulation_states
        for step in trace.steps
    )
    approach_collision = any(
        step.hand_table_contact
        and step.state
        in {ExecutionState.MOVE_TO_PREGRASP, ExecutionState.APPROACH}
        for step in trace.steps
    )
    hold_steps = [step for step in trace.steps if step.state is ExecutionState.HOLD]
    stable_threshold = (
        initial_height
        + protocol.success.minimum_lift_m
        - protocol.success.maximum_object_drop_m
    )
    stable_hold = [step.object_position_m[2] >= stable_threshold for step in hold_steps]
    hold_duration = _longest_true_duration(stable_hold, control_timestep_s)
    reached_lift = maximum_lift >= protocol.success.minimum_lift_m
    final_drop = float(np.max(heights) - final_height)
    timed_out = trace.steps[-1].time_s > protocol.parameters.total_timeout_s + 1e-9
    success = (
        acquired
        and reached_lift
        and hold_duration >= protocol.success.required_hold_s
        and final_drop <= protocol.success.maximum_object_drop_m
        and not approach_collision
        and not timed_out
    )

    failure: FailureType | None = None
    if not success:
        if timed_out:
            failure = FailureType.TIMEOUT
        elif approach_collision:
            failure = FailureType.APPROACH_COLLISION
        elif not acquired:
            failure = FailureType.NO_CONTACT
        elif not reached_lift:
            failure = FailureType.FAILED_ACQUISITION
        elif final_drop > protocol.success.maximum_object_drop_m:
            failure = FailureType.DROP
        else:
            failure = FailureType.SLIP_DURING_HOLD

    return ScoredOutcome(
        success=success,
        failure_type=failure,
        acquired_object=acquired,
        maximum_lift_m=maximum_lift,
        hold_duration_s=hold_duration,
        final_object_height_m=final_height,
        approach_collision=approach_collision,
    )
