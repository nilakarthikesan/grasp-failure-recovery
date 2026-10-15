"""Conservative, sampled event annotations for fixed grasp/lift traces.

These annotations are supervision and analysis metadata, never predictor inputs.
They do not change :func:`score_trace` or the final binary episode outcome. A
height threshold crossing is an observed motion proxy, not a physical slip
onset, proof of a secured grip, or necessarily a terminal task failure.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from .registry import ExecutionProtocolSpec
from .runner import ExecutionState, ExecutionTrace
from .scoring import FailureType, ScoredOutcome


SCHEMA_VERSION = "rollout_events_v1"

_LIFT_STATES = {ExecutionState.LIFT, ExecutionState.HOLD}
_LOSS_STATES = _LIFT_STATES | {ExecutionState.SCORE}
_UNSUPPORTED_FAILURES = {
    FailureType.INVALID_RETARGET,
    FailureType.UNREACHABLE_PREGRASP,
    FailureType.APPROACH_COLLISION,
    FailureType.TIMEOUT,
    FailureType.RUNTIME_ERROR,
}


def _validate_trace(trace: ExecutionTrace, control_timestep_s: float) -> None:
    if not math.isfinite(control_timestep_s) or control_timestep_s <= 0:
        raise ValueError("control_timestep_s must be finite and positive")
    state_order = {state: index for index, state in enumerate(ExecutionState)}
    previous = None
    for step in trace.steps:
        if not isinstance(step.index, Integral) or isinstance(step.index, bool) or step.index < 0:
            raise ValueError("trace indices must be nonnegative integers")
        if not math.isfinite(step.time_s) or step.time_s < 0:
            raise ValueError("trace times must be finite and nonnegative")
        try:
            position = np.asarray(step.object_position_m, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("object positions must be finite three-element vectors") from exc
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("object positions must be finite three-element vectors")
        if not isinstance(step.state, ExecutionState):
            raise ValueError("trace contains an unsupported execution state")
        if previous is not None:
            if step.time_s <= previous.time_s or step.index <= previous.index:
                raise ValueError("trace times and indices must be strictly increasing")
            if state_order[step.state] < state_order[previous.state]:
                raise ValueError("trace phases must follow the fixed protocol order")
        previous = step


def _consecutive(previous, step, control_timestep_s: float) -> bool:
    return (
        step.index == previous.index + 1
        and math.isclose(
            step.time_s - previous.time_s,
            control_timestep_s,
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
    )


def annotate_rollout_events(
    trace: ExecutionTrace,
    protocol: ExecutionProtocolSpec,
    *,
    control_timestep_s: float,
    outcome: ScoredOutcome,
) -> dict:
    """Return JSON-safe sampled proxies and a phase-based acquisition deadline.

    ``lifted_contact_proxy`` requires the protocol minimum lift, hand/object
    contact, and no object/table contact during LIFT or HOLD. Contact alone never
    qualifies. ``retained_lift_proxy`` additionally requires a consecutive run
    of qualifying HOLD observations totaling ``required_hold_s``. Its sampled
    duration follows the scorer's count-times-control-timestep convention;
    between-sample continuous stability is not established.

    ``observed_height_loss`` is the first sampled crossing below
    ``minimum_lift_m - maximum_object_drop_m`` after the lift proxy. The crossing
    is bounded by two observed endpoints, not assigned an exact physical onset.
    It can occur in an eventually successful trajectory. Failed acquisition is
    instead assessed at the observed LIFT-to-HOLD control interval. Missing
    transitions, unsupported failure causes, and unobserved follow-up are
    represented explicitly; no future-horizon targets are created here.
    """

    _validate_trace(trace, control_timestep_s)
    if outcome.success and outcome.failure_type is not None:
        raise ValueError("a successful outcome cannot have a failure_type")
    if not outcome.success and outcome.failure_type is None:
        raise ValueError("an unsuccessful outcome must have a failure_type")

    steps = trace.steps
    failure = outcome.failure_type
    supported = failure not in _UNSUPPORTED_FAILURES
    complete = bool(steps and steps[-1].state is ExecutionState.SCORE)
    sampling_complete = bool(steps) and all(
        _consecutive(previous, step, control_timestep_s)
        for previous, step in zip(steps, steps[1:])
    )
    initial_reference_available = bool(steps and steps[0].state is ExecutionState.RESET)
    reference = float(steps[0].object_position_m[2]) if initial_reference_available else None
    lift_threshold = (
        reference + protocol.success.minimum_lift_m if reference is not None else None
    )
    stable_threshold = (
        lift_threshold - protocol.success.maximum_object_drop_m
        if lift_threshold is not None else None
    )
    unknown_reason = ("empty_trace" if not steps else
                      "missing_reset_height_reference" if reference is None else "not_observed")
    lifted = {"status": "unknown" if reference is None else "not_observed",
              "time_s": None, "step_index": None, "reason": unknown_reason}
    retained = {"status": "unknown" if reference is None else "not_observed",
                "time_s": None, "step_index": None,
                "sampled_hold_duration_s": 0.0, "reason": unknown_reason}
    deadline = {
        "status": "unknown", "interval_start_exclusive_s": None,
        "interval_end_inclusive_s": None, "last_lift_step_index": None,
        "first_hold_step_index": None, "proxy_observed_by_deadline": None,
        "final_failed_acquisition": None,
        "reason": unknown_reason if reference is None else "missing_lift_to_hold_transition",
    }
    loss = {
        "status": "unknown", "interval_start_exclusive_s": None,
        "interval_end_inclusive_s": None, "observed_time_s": None,
        "step_index": None, "mechanism": "unknown",
        "reason": unknown_reason if reference is None else "lift_proxy_not_observed",
    }

    lift_index = None
    for index, step in enumerate(steps if reference is not None else ()):
        if (step.state in _LIFT_STATES
                and float(step.object_position_m[2]) >= lift_threshold
                and step.hand_object_contact and not step.object_table_contact):
            lift_index = index
            lifted = {"status": "observed", "time_s": float(step.time_s),
                      "step_index": int(step.index), "reason": None}
            break

    # Count only uninterrupted qualifying HOLD endpoints, as in final scoring.
    # Dropped samples break a run instead of inventing intervening observations.
    run = longest = 0
    for index, step in enumerate(steps):
        qualifies = (
            lift_index is not None and index >= lift_index
            and step.state is ExecutionState.HOLD
            and float(step.object_position_m[2]) >= stable_threshold
            and step.hand_object_contact and not step.object_table_contact
        )
        if not qualifies:
            run = 0
            continue
        if index and not _consecutive(steps[index - 1], step, control_timestep_s):
            run = 0
        run += 1
        longest = max(longest, run)
        if (retained["status"] != "observed"
                and run * control_timestep_s + 1e-9 >= protocol.success.required_hold_s):
            retained.update(status="observed", time_s=float(step.time_s),
                            step_index=int(step.index), reason=None)
    retained["sampled_hold_duration_s"] = float(longest * control_timestep_s)
    if (reference is not None and retained["status"] != "observed"
            and (not complete or not sampling_complete)):
        retained.update(status="censored", reason="incomplete_or_gapped_follow_up")

    for previous, step in zip(steps if reference is not None else (), steps[1:]):
        if previous.state is ExecutionState.LIFT and step.state is ExecutionState.HOLD:
            if not _consecutive(previous, step, control_timestep_s):
                deadline["reason"] = "gapped_lift_to_hold_transition"
                break
            observed_by_deadline = (
                lifted["time_s"] is not None and lifted["time_s"] <= previous.time_s
            )
            deadline.update(
                status="observed",
                interval_start_exclusive_s=float(previous.time_s),
                interval_end_inclusive_s=float(step.time_s),
                last_lift_step_index=int(previous.index), first_hold_step_index=int(step.index),
                proxy_observed_by_deadline=bool(observed_by_deadline),
                final_failed_acquisition=bool(
                    complete and supported and not outcome.success
                    and failure in {FailureType.NO_CONTACT, FailureType.FAILED_ACQUISITION}
                    and not observed_by_deadline
                ),
                reason=None,
            )
            # An incomplete episode cannot establish the final deadline label.
            if not complete or not supported:
                deadline["final_failed_acquisition"] = None
            break

    if lift_index is not None:
        for index in range(lift_index + 1, len(steps)):
            previous, step = steps[index - 1], steps[index]
            if (step.state in _LOSS_STATES
                    and float(previous.object_position_m[2]) >= stable_threshold
                    and float(step.object_position_m[2]) < stable_threshold):
                loss.update(
                    status="observed", interval_start_exclusive_s=float(previous.time_s),
                    interval_end_inclusive_s=float(step.time_s),
                    observed_time_s=float(step.time_s), step_index=int(step.index), reason=None,
                )
                break
        if loss["status"] != "observed":
            if not complete or not sampling_complete:
                loss.update(status="censored", reason="incomplete_or_gapped_follow_up")
            elif failure in {FailureType.DROP, FailureType.SLIP_DURING_HOLD}:
                loss.update(status="unknown", reason="final_failure_not_explained_by_height_proxy")
            else:
                loss.update(status="not_observed", reason="no_sampled_height_threshold_crossing")
    elif reference is not None:
        loss.update(status="unsupported" if complete else "censored",
                    reason="lift_proxy_not_observed")

    if not supported:
        loss.update(status="unsupported", reason="unsupported_final_failure_category")

    return {
        "schema_version": SCHEMA_VERSION,
        "binary_outcome": {"label": int(outcome.success),
                           "failure_type": failure.value if failure else None},
        "criteria": {
            "reference_height_m": reference,
            "reference_source": "first_recorded_object_height_as_in_score_trace",
            "minimum_lift_m": float(protocol.success.minimum_lift_m),
            "lifted_contact_height_m": lift_threshold,
            "stable_lift_height_m": stable_threshold,
            "required_hold_s": float(protocol.success.required_hold_s),
            "lifted_contact_proxy": "LIFT/HOLD: minimum lift plus hand/object contact and no object/table contact",
            "retained_lift_proxy": "consecutive qualifying HOLD endpoints times control timestep; sampled retention only",
            "observed_height_loss": "first sampled crossing below stable lift height after lifted/contact proxy; not physical slip onset",
        },
        "observations": {
            "start_time_s": float(steps[0].time_s) if steps else None,
            "end_time_s": float(steps[-1].time_s) if steps else None,
            "control_timestep_s": float(control_timestep_s),
            "initial_reference_available": initial_reference_available,
            "completed_score": complete, "sampling_complete": sampling_complete,
            "timing": "trace timestamps are observed control-interval endpoints; events use observed bounds only",
        },
        "lifted_contact_proxy": lifted,
        "retained_lift_proxy": retained,
        "acquisition_deadline": deadline,
        "observed_height_loss": loss,
        "failure_category_supported": supported,
        "annotation_policy": "supervision and analysis only; events, future observations and final outcomes are never predictor features",
        "limitations": [
            "proxies do not establish a secured grip or an exact physical slip mechanism",
            "height-loss events may recover and do not overwrite binary episode success",
            "acquisition deadline is a task-phase control interval, not physical loss onset",
            "no future-horizon labels or training windows are created",
            "missing observations are not evidence of event-free follow-up",
        ],
    }
