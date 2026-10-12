from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.rollout_events import annotate_rollout_events
from grasp_failure_prediction.evaluation.runner import ExecutionState, ExecutionTrace, RunnerStep
from grasp_failure_prediction.evaluation.scoring import FailureType, ScoredOutcome, score_trace


DT = 0.04


def step(index, state, height, *, contact=True, table=False, time=None):
    return RunnerStep(
        index=index, time_s=index * DT if time is None else time, state=state,
        palm_position_m=np.zeros(3), hand_qpos=np.zeros(24),
        object_position_m=np.array([0.0, 0.0, height]), contact_count=int(contact),
        hand_object_contact=contact, hand_table_contact=False, object_table_contact=table,
    )


def protocol():
    return load_protocol_registry().resolve("fixed_grasp_lift_v2")


def annotate(steps, outcome=None):
    trace = ExecutionTrace(tuple(steps))
    if outcome is None:
        outcome = score_trace(trace, protocol(), control_timestep_s=DT)
    return annotate_rollout_events(trace, protocol(), control_timestep_s=DT, outcome=outcome)


def initial():
    return step(0, ExecutionState.RESET, 0.03, contact=False, table=True)


def successful_steps():
    return [initial(), step(1, ExecutionState.LIFT, 0.18)] + [
        step(index, ExecutionState.HOLD, 0.18) for index in range(2, 52)
    ] + [step(52, ExecutionState.SCORE, 0.18)]


def test_contact_and_small_bump_are_not_lift_or_retention_proxies():
    result = annotate([
        initial(), step(1, ExecutionState.CLOSE_FINGERS, 0.03),
        step(2, ExecutionState.LIFT, 0.08), step(3, ExecutionState.HOLD, 0.03),
        step(4, ExecutionState.SCORE, 0.03),
    ])
    assert result["binary_outcome"] == {"label": 0, "failure_type": "failed_acquisition"}
    assert result["lifted_contact_proxy"]["status"] == "not_observed"
    assert result["retained_lift_proxy"]["status"] == "not_observed"
    assert result["observed_height_loss"]["status"] == "unsupported"
    deadline = result["acquisition_deadline"]
    assert deadline["status"] == "observed"
    assert deadline["interval_start_exclusive_s"] == 0.08
    assert deadline["interval_end_inclusive_s"] == 0.12
    assert deadline["final_failed_acquisition"] is True
    assert deadline["proxy_observed_by_deadline"] is False


@pytest.mark.parametrize(("contact", "table"), [(False, False), (True, True)])
def test_height_alone_or_supported_object_is_not_lifted_contact_proxy(contact, table):
    result = annotate([
        initial(), step(1, ExecutionState.LIFT, 0.18, contact=contact, table=table),
        step(2, ExecutionState.HOLD, 0.18, contact=contact, table=table),
        step(3, ExecutionState.SCORE, 0.18, contact=contact, table=table),
    ])
    assert result["lifted_contact_proxy"]["status"] == "not_observed"
    assert result["observed_height_loss"]["interval_end_inclusive_s"] is None


def test_protocol_hold_requirement_is_explicit_and_success_unchanged():
    result = annotate(successful_steps())
    assert result["binary_outcome"] == {"label": 1, "failure_type": None}
    assert result["lifted_contact_proxy"]["time_s"] == 0.04
    assert result["retained_lift_proxy"]["status"] == "observed"
    assert result["retained_lift_proxy"]["time_s"] == 2.04
    assert result["retained_lift_proxy"]["sampled_hold_duration_s"] == 2.0
    assert result["observed_height_loss"]["status"] == "not_observed"
    assert result["criteria"]["required_hold_s"] == 2.0
    assert result["criteria"]["stable_lift_height_m"] == pytest.approx(0.15)
    assert result["acquisition_deadline"]["final_failed_acquisition"] is False
    json.dumps(result, allow_nan=False)


def test_observed_height_loss_is_an_interval_without_claiming_slip_mechanism():
    result = annotate([
        initial(), step(1, ExecutionState.LIFT, 0.18),
        step(2, ExecutionState.HOLD, 0.16), step(3, ExecutionState.HOLD, 0.03, contact=False),
        step(4, ExecutionState.SCORE, 0.03, contact=False),
    ])
    assert result["binary_outcome"]["failure_type"] == "drop"
    event = result["observed_height_loss"]
    assert event["status"] == "observed"
    assert event["interval_start_exclusive_s"] == 0.08
    assert event["interval_end_inclusive_s"] == event["observed_time_s"] == 0.12
    assert event["mechanism"] == "unknown"
    assert result["retained_lift_proxy"]["status"] == "not_observed"


def test_transient_loss_does_not_rewrite_eventual_episode_success():
    steps = [initial(), step(1, ExecutionState.LIFT, 0.18),
             step(2, ExecutionState.LIFT, 0.10), step(3, ExecutionState.LIFT, 0.18)]
    steps += [step(index, ExecutionState.HOLD, 0.18) for index in range(4, 54)]
    steps += [step(54, ExecutionState.SCORE, 0.18)]
    result = annotate(steps)
    assert result["binary_outcome"]["label"] == 1
    assert result["observed_height_loss"]["status"] == "observed"
    assert result["observed_height_loss"]["observed_time_s"] == 0.08


def test_short_trace_is_censored_and_has_no_invented_future_deadline():
    result = annotate([initial(), step(1, ExecutionState.LIFT, 0.18)])
    assert result["observations"]["end_time_s"] == 0.04
    assert result["acquisition_deadline"]["status"] == "unknown"
    assert result["acquisition_deadline"]["interval_end_inclusive_s"] is None
    assert result["retained_lift_proxy"]["status"] == "censored"
    assert result["observed_height_loss"]["status"] == "censored"


def test_empty_trace_is_unknown_while_existing_binary_outcome_is_preserved():
    outcome = ScoredOutcome(False, FailureType.FAILED_ACQUISITION, False,
                            0.0, 0.0, 0.03, False)
    result = annotate([], outcome)
    assert result["observations"]["start_time_s"] is None
    assert result["binary_outcome"]["label"] == 0
    for key in ("lifted_contact_proxy", "retained_lift_proxy", "acquisition_deadline", "observed_height_loss"):
        assert result[key]["status"] == "unknown"


def test_missing_initial_reference_is_unknown_instead_of_assuming_a_new_table_height():
    outcome = ScoredOutcome(False, FailureType.FAILED_ACQUISITION, True,
                            0.0, 0.0, 0.18, False)
    result = annotate([step(1, ExecutionState.LIFT, 0.18),
                       step(2, ExecutionState.HOLD, 0.18),
                       step(3, ExecutionState.SCORE, 0.18)], outcome)
    assert result["observations"]["initial_reference_available"] is False
    assert result["criteria"]["reference_height_m"] is None
    for key in ("lifted_contact_proxy", "retained_lift_proxy", "acquisition_deadline", "observed_height_loss"):
        assert result[key]["status"] == "unknown"
        assert result[key]["reason"] == "missing_reset_height_reference"


def test_gapped_transition_cannot_invent_acquisition_deadline_or_hold_coverage():
    result = annotate([
        initial(), step(1, ExecutionState.LIFT, 0.18),
        step(3, ExecutionState.HOLD, 0.18), step(4, ExecutionState.SCORE, 0.18),
    ])
    assert result["observations"]["sampling_complete"] is False
    assert result["acquisition_deadline"]["status"] == "unknown"
    assert result["acquisition_deadline"]["reason"] == "gapped_lift_to_hold_transition"
    assert result["retained_lift_proxy"]["status"] == "censored"
    assert result["observed_height_loss"]["status"] == "censored"


def test_height_loss_across_gap_keeps_wider_observed_bounds():
    result = annotate([
        initial(), step(1, ExecutionState.LIFT, 0.18),
        step(6, ExecutionState.HOLD, 0.03, contact=False),
        step(7, ExecutionState.SCORE, 0.03, contact=False),
    ])
    event = result["observed_height_loss"]
    assert event["status"] == "observed"
    assert event["interval_start_exclusive_s"] == 0.04
    assert event["interval_end_inclusive_s"] == 0.24


def test_final_scoring_drop_without_proxy_crossing_is_unknown():
    result = annotate([
        initial(), step(1, ExecutionState.LIFT, 0.23),
        step(2, ExecutionState.HOLD, 0.19), step(3, ExecutionState.SCORE, 0.19),
    ])
    assert result["binary_outcome"]["failure_type"] == "drop"
    assert result["observed_height_loss"]["status"] == "unknown"
    assert result["observed_height_loss"]["reason"] == "final_failure_not_explained_by_height_proxy"


@pytest.mark.parametrize("failure", [FailureType.APPROACH_COLLISION, FailureType.TIMEOUT,
                                     FailureType.RUNTIME_ERROR, FailureType.INVALID_RETARGET])
def test_other_failure_causes_are_explicitly_unsupported(failure):
    outcome = ScoredOutcome(False, failure, True, 0.15, 0.0, 0.03, False)
    result = annotate([
        initial(), step(1, ExecutionState.LIFT, 0.18),
        step(2, ExecutionState.HOLD, 0.03), step(3, ExecutionState.SCORE, 0.03),
    ], outcome)
    assert result["failure_category_supported"] is False
    assert result["observed_height_loss"]["status"] == "unsupported"
    assert result["acquisition_deadline"]["final_failed_acquisition"] is None
    assert result["binary_outcome"]["failure_type"] == failure.value


@pytest.mark.parametrize("bad_dt", [0.0, -0.04, float("nan"), float("inf")])
def test_invalid_control_timestep_is_rejected(bad_dt):
    steps = successful_steps()
    outcome = score_trace(ExecutionTrace(tuple(steps)), protocol(), control_timestep_s=DT)
    with pytest.raises(ValueError, match="control_timestep_s"):
        annotate_rollout_events(ExecutionTrace(tuple(steps)), protocol(),
                                control_timestep_s=bad_dt, outcome=outcome)


@pytest.mark.parametrize("bad_step", [
    step(1, ExecutionState.LIFT, float("nan")),
    step(1, ExecutionState.LIFT, 0.18, time=float("inf")),
    step(1, ExecutionState.LIFT, 0.18, time=0.0),
    step(0, ExecutionState.LIFT, 0.18, time=0.04),
    step(-1, ExecutionState.LIFT, 0.18, time=0.04),
    step(1.5, ExecutionState.LIFT, 0.18, time=0.04),
    step(1, ExecutionState.LIFT, 0.18, time=-0.04),
])
def test_invalid_trace_values_are_rejected(bad_step):
    outcome = ScoredOutcome(False, FailureType.FAILED_ACQUISITION, False,
                            0.0, 0.0, 0.03, False)
    with pytest.raises(ValueError):
        annotate([initial(), bad_step], outcome)


def test_wrong_object_position_shape_and_backwards_phases_are_rejected():
    outcome = ScoredOutcome(False, FailureType.FAILED_ACQUISITION, False,
                            0.0, 0.0, 0.03, False)
    with pytest.raises(ValueError, match="three-element"):
        annotate([replace(initial(), object_position_m=np.zeros(2))], outcome)
    with pytest.raises(ValueError, match="phases"):
        annotate([initial(), step(1, ExecutionState.HOLD, 0.18),
                  step(2, ExecutionState.LIFT, 0.18)], outcome)
    with pytest.raises(ValueError, match="unsupported execution state"):
        annotate([replace(initial(), state="reset")], outcome)


def test_inconsistent_outcome_is_rejected_without_changing_it():
    outcome = ScoredOutcome(True, FailureType.DROP, True, 0.15, 2.0, 0.18, False)
    with pytest.raises(ValueError, match="successful outcome"):
        annotate(successful_steps(), outcome)
