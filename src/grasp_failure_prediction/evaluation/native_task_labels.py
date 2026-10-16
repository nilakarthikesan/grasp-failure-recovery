"""Pure native task labels from explicitly clocked privileged diagnostics.

No simulator/controller/reference model is invoked. These arrays are supervision
only, never predictor inputs. Runtime, refused mapping and invalid traces yield
null physical outcomes; a malformed run cannot become a negative example.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from .manosim_recording import canonical_sha256
from .prediction_recording import MappingStatus, PhysicalOutcome

SCHEMA = "manosim_native_task_labels_v1"
TRACE_KEYS = frozenset({"state_time_s", "force_time_s", "object_com_world_m", "wrist_position_world_m", "wrist_rotation_world",
                        "hand_object_normal_force_N", "hand_table_normal_force_N", "object_table_normal_force_N"})
PRIORITY = ("approach_collision", "no_contact", "failed_acquisition", "drop", "slip_during_hold")


class NativeTaskLabelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label_rule_id: Literal["manosim_native_grasp_lift_task_labels_v1"] = "manosim_native_grasp_lift_task_labels_v1"
    physics_interval_s: Literal[.002,.001] = .002
    control_interval_s: Literal[.01] = .01
    close_start_s: Literal[1.] = 1.
    lift_start_s: Literal[1.5] = 1.5
    hold_start_s: Literal[2.5] = 2.5
    task_end_s: Literal[5.5] = 5.5
    acquired_lift_m: Literal[.10] = .10
    minimum_hold_lift_m: Literal[.08] = .08
    maximum_relative_shift_m: Literal[.02] = .02
    contact_threshold_N: Literal[.001] = .001
    clock_atol_s: Literal[1e-9] = 1e-9
    proper_rotation_atol: Literal[1e-8] = 1e-8


def label_rule_contract(spec=NativeTaskLabelSpec()):
    spec = NativeTaskLabelSpec.model_validate(spec.model_dump())
    return dict(schema_version=SCHEMA + "_contract", spec=spec.model_dump(),
                source_sha256="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                state_clock=f"0..{spec.task_end_s}s inclusive at{spec.physics_interval_s}s; COM/wrist position/rotation all at state_time_s",
                force_clock=f"0..{spec.task_end_s-spec.physics_interval_s}s at{spec.physics_interval_s}s; nonnegative aggregate normal force at force_time_s (preceding interval start)",
                force_policy="qualified hand-object/hand-table/object-table pair sums, strict >1mN; no control-rate subsampling",
                table_collision_taxonomy="hand-table at any phase is approach_collision; actual phase retained",
                priority_for_equal_first_failure_time=list(PRIORITY), time_policy="canonical validated physics tick; observed clock retained per event, ties by tick", slip_reference="object COM in wrist frame at first hold sample",
                height_comparison="world COM z against initial COM z plus declared threshold; no fitted tolerance",
                hold_rule="acquired>=10cm and no >1mN object-table support at2.5s; >=8cm and no sampled >1mN object-table support throughout3s hold;2cm drop margin", predictor_feature=False)


def _base(spec):
    contract = label_rule_contract(spec)
    return dict(schema_version=SCHEMA + "_report", label_rule_contract=contract, label_rule_sha256=canonical_sha256(contract),
                outcome=None, failure_events=[], metrics={}, scientific_training_admitted=False)


def _array(value, shape, name):
    a = np.asarray(value)
    if a.dtype.kind not in "fiu" or a.shape != shape: raise ValueError("invalid numeric diagnostic layout: " + name)
    if not np.isfinite(a).all(): raise ValueError("nonfinite diagnostic: " + name)
    return a.astype(np.float64, copy=True)


def score_native_task(trace, *, mapping_status="ok", execution_status="completed", spec=NativeTaskLabelSpec()):
    """Return earliest causal failure, resolving only exact-time ties by priority.

    State and force clocks have separate explicit lengths. No force timestamp is
    inferred from an endpoint clock. The producer must establish actual pair
    signs, clock semantics and complete2ms coverage under its frozen protocol.
    """
    spec = NativeTaskLabelSpec.model_validate(spec.model_dump()); report = _base(spec)
    mapping = MappingStatus(mapping_status)
    if execution_status not in ("completed", "runtime_error", "not_executed"): raise ValueError("unknown native execution status")
    if mapping is not MappingStatus.OK:
        if execution_status != "not_executed": raise ValueError("mapping refusal cannot claim physical execution")
        report.update(status="NULL_MAPPING_OR_SETUP_REFUSAL", reason=mapping.value); return report
    if execution_status != "completed":
        report.update(status="NULL_RUNTIME_OR_UNEXECUTED", reason=execution_status); return report
    if not isinstance(trace, dict) or set(trace) != TRACE_KEYS:
        report.update(status="NULL_INVALID_TRACE", reason="exact named diagnostic fields required"); return report
    try:
        steps = int(round(spec.task_end_s / spec.physics_interval_s)); N, M = steps + 1, steps
        st = _array(trace["state_time_s"], (N,), "state_time_s"); ft = _array(trace["force_time_s"], (M,), "force_time_s")
        expected = np.arange(N, dtype=np.float64) * spec.physics_interval_s
        if (not np.allclose(st, expected, rtol=0, atol=spec.clock_atol_s) or not np.allclose(ft, expected[:-1], rtol=0, atol=spec.clock_atol_s)
                or np.any(np.diff(st)<=0) or np.any(np.diff(ft)<=0)): raise ValueError("incomplete or misaligned independent state/force clocks")
        obj = _array(trace["object_com_world_m"], (N,3), "object COM"); wrist = _array(trace["wrist_position_world_m"], (N,3), "wrist position")
        R = _array(trace["wrist_rotation_world"], (N,3,3), "wrist rotation")
        if (np.max(np.abs(np.einsum('nji,njk->nik',R,R,optimize=False)-np.eye(3)))>spec.proper_rotation_atol
                or np.max(np.abs(np.linalg.det(R)-1))>spec.proper_rotation_atol): raise ValueError("improper measured wrist rotation")
        ho = _array(trace["hand_object_normal_force_N"], (M,), "hand-object normal force"); ht = _array(trace["hand_table_normal_force_N"], (M,), "hand-table normal force"); ot = _array(trace["object_table_normal_force_N"], (M,), "object-table normal force")
        if np.any(ho<0) or np.any(ht<0) or np.any(ot<0): raise ValueError("normal-force magnitudes cannot be signed/negative")
    except (ValueError, TypeError, OverflowError) as error:
        report.update(status="NULL_INVALID_OR_NONFINITE_TRACE", reason=str(error)); return report
    close_index = int(round(spec.close_start_s / spec.physics_interval_s)); lift_index = int(round(spec.lift_start_s / spec.physics_interval_s)); hold_index = int(round(spec.hold_start_s / spec.physics_interval_s))
    # Time masks use the frozen indices after clock validation, avoiding rounding
    # a boundary sample into the preceding phase.
    preclose = np.arange(M) < close_index; acquisition = (np.arange(M)>=close_index)&(np.arange(M)<=lift_index)
    contact = ho > spec.contact_threshold_N; table = ht > spec.contact_threshold_N
    initial_z = float(obj[0,2]); acquired_height = initial_z + spec.acquired_lift_m; hold_height = initial_z + spec.minimum_hold_lift_m
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            rise = obj[:,2] - initial_z
            relative = np.einsum('nji,nj->ni', R, obj-wrist, optimize=False)
            shifts = np.linalg.norm(relative[hold_index:]-relative[hold_index],axis=1)
        if not np.isfinite(relative).all() or not np.isfinite(rise).all() or not np.isfinite(shifts).all(): raise ValueError("nonfinite derived pose metric")
    except (ValueError,FloatingPointError,OverflowError) as error:
        report.update(status="NULL_INVALID_OR_NONFINITE_TRACE",reason=str(error));return report
    raised = np.flatnonzero(obj[:,2] >= acquired_height); events = []
    def phase(index):
        time=index*spec.physics_interval_s
        return 'reset_pregrasp' if time<.25 else 'approach' if time<spec.close_start_s else 'close_fingers' if time<spec.lift_start_s else 'lift' if time<spec.hold_start_s else 'hold'
    def event(kind, index, clock, reason):
        events.append(dict(failure_type=kind,time_s=float(index*spec.physics_interval_s),observed_time_s=float((st if clock=='state' else ft)[index]),physics_tick=int(index),actual_phase=phase(index),clock=clock,reason=reason))
    approach = np.flatnonzero((preclose&contact)|table)
    if len(approach): event("approach_collision",int(approach[0]),"force","hand-object before close or hand-table at any time")
    if not np.any(contact&acquisition):
        events.append(dict(failure_type="no_contact",time_s=spec.lift_start_s,observed_time_s=float(st[lift_index]),physics_tick=lift_index,actual_phase=phase(lift_index),clock="state_phase_boundary",reason="no >1mN hand-object normal force by acquisition end"))
    if obj[hold_index,2] < acquired_height: event("failed_acquisition",hold_index,"state","COM below10cm at lift end")
    support=ot>spec.contact_threshold_N
    if support[hold_index]:event("failed_acquisition",hold_index,"force","object-table support >1mN at first hold interval")
    later_support=np.flatnonzero((np.arange(M)>hold_index)&support)
    if len(later_support):event("drop",int(later_support[0]),"force","object-table support >1mN after first hold interval")
    if len(raised):
        drops=np.flatnonzero((np.arange(N)>=raised[0])&(obj[:,2]<hold_height))
        if len(drops): event("drop",int(drops[0]),"state","COM below8cm after first10cm raise")
    slips=np.flatnonzero(shifts>spec.maximum_relative_shift_m)
    if len(slips): event("slip_during_hold",int(hold_index+slips[0]),"state","object COM shifts >2cm in measured wrist frame")
    events.sort(key=lambda row:(row['physics_tick'],PRIORITY.index(row['failure_type'])))
    success = not events
    outcome = PhysicalOutcome(success=success, failure_type=None if success else events[0]['failure_type'],label_rule_id=spec.label_rule_id,
                              first_failure_time_s=None if success else events[0]['time_s'])
    valid_contacts=np.flatnonzero(contact&acquisition)
    report.update(status="SCORED",outcome=outcome.model_dump(mode="json"),failure_events=events,
                  metrics=dict(state_samples=N,force_intervals=M,initial_COM_height_m=initial_z,final_COM_height_m=float(obj[-1,2]),COM_rise_at_lift_end_m=float(rise[hold_index]),
                               minimum_hold_COM_rise_m=float(rise[hold_index:].min()),maximum_hold_relative_COM_shift_m=float(shifts.max()),first_raised_time_s=float(st[raised[0]]) if len(raised) else None,
                               first_acquisition_contact_time_s=float(ft[valid_contacts[0]]) if len(valid_contacts) else None,preclose_hand_object_force_max_N=float(ho[preclose].max()),
                               hand_table_force_max_N=float(ht.max()),hold_duration_s=spec.task_end_s-spec.hold_start_s,acquisition_met=bool(obj[hold_index,2]>=acquired_height and not support[hold_index]),object_table_force_max_hold_N=float(ot[hold_index:].max()),hold_table_support_absent=bool(not np.any(support[hold_index:])),object_table_support_scope="sampled preceding-interval force at every frozen physics tick",
                               hold_height_met=bool(np.all(obj[hold_index:,2]>=hold_height)),hold_slip_met=bool(not len(slips)),table_collision=bool(np.any(table))))
    return report
