"""Manufactured diagnostic labels only: no engine, native model or task data."""
import numpy as np
import pytest
from grasp_failure_prediction.evaluation import native_task_labels as m


def trace():
    spec=m.NativeTaskLabelSpec();st=np.arange(2751)*.002;ft=st[:-1];height=np.zeros(len(st));height[750:1251]=np.linspace(0,.1,501);height[1251:]=.1
    obj=np.zeros((len(st),3));obj[:,2]=height;wrist=obj.copy();ho=np.zeros(len(ft));ho[500:]=.01
    return dict(state_time_s=st,force_time_s=ft,object_com_world_m=obj,wrist_position_world_m=wrist,wrist_rotation_world=np.tile(np.eye(3),(len(st),1,1)),hand_object_normal_force_N=ho,hand_table_normal_force_N=np.zeros(len(ft)),object_table_normal_force_N=np.zeros(len(ft)))


@pytest.fixture(autouse=True)
def no_native_operations(monkeypatch):
    import mujoco
    def blocked(*a,**kw):raise AssertionError('pure native scorer controls cannot invoke native operations')
    for name in ('MjModel','MjData','mj_forward','mj_kinematics','mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,name,blocked)


def test_manufactured_success_complete_hold_clock_and_exact_boundaries():
    r=m.score_native_task(trace());assert r['status']=='SCORED' and r['outcome']['success'] and r['outcome']['first_failure_time_s'] is None
    assert r['metrics']['hold_duration_s']==3 and r['metrics']['first_acquisition_contact_time_s']==1.
    assert not r['scientific_training_admitted'] and r['metrics']['state_samples']==2751 and r['metrics']['force_intervals']==2750


def test_ten_cm_acquisition_then_eight_cm_hold_has_explicit_drop_margin():
    a=trace();a['object_com_world_m'][1300:,2]=.085;a['wrist_position_world_m'][1300:,2]=.085
    r=m.score_native_task(a);assert r['outcome']['success'] and r['metrics']['hold_height_met']


@pytest.mark.parametrize('kind,time',[('precontact',.1),('table',4.),('no_contact',1.5),('failed_acquisition',2.5),('drop',3.),('slip',3.)])
def test_every_physical_cause_uses_earliest_real_clock(kind,time):
    a=trace()
    if kind=='precontact':a['hand_object_normal_force_N'][50]=.002
    elif kind=='table':a['hand_table_normal_force_N'][2000]=.002
    elif kind=='no_contact':a['hand_object_normal_force_N'][:751]=0
    elif kind=='failed_acquisition':a['object_com_world_m'][:,2]*=.99;a['wrist_position_world_m'][:,2]*=.99
    elif kind=='drop':a['object_com_world_m'][1500:,2]=.079;a['wrist_position_world_m'][1500:,2]=.079
    else:a['object_com_world_m'][1500:,0]=.020001
    expected={'precontact':'approach_collision','table':'approach_collision','slip':'slip_during_hold'}.get(kind,kind)
    r=m.score_native_task(a);assert not r['outcome']['success'] and r['outcome']['failure_type']==expected and r['outcome']['first_failure_time_s']==time


def test_thresholds_are_strict_and_close_start_is_not_precontact():
    a=trace();a['hand_object_normal_force_N'][:500]=.001;a['hand_table_normal_force_N'][:]=.001
    assert m.score_native_task(a)['outcome']['success']
    a['hand_object_normal_force_N'][499]=.0010001
    assert m.score_native_task(a)['outcome']['first_failure_time_s']==.998


def test_earliest_cause_precedes_priority_except_for_exact_time_ties():
    a=trace();a['object_com_world_m'][1500:,0]=.03;a['hand_table_normal_force_N'][2000]=.002
    r=m.score_native_task(a);assert r['outcome']['failure_type']=='slip_during_hold' and r['outcome']['first_failure_time_s']==3.
    a['hand_table_normal_force_N'][1500]=.002
    r=m.score_native_task(a);assert r['outcome']['failure_type']=='approach_collision' and r['outcome']['first_failure_time_s']==3.


def test_wrist_frame_relative_shift_uses_actual_rotation_not_world_translation():
    a=trace();a['object_com_world_m'][:,0]=.1;a['wrist_position_world_m'][:,0]=0
    a['wrist_rotation_world'][1500:]=np.diag([-1.,-1.,1.])
    r=m.score_native_task(a);assert r['outcome']['failure_type']=='slip_during_hold' and r['metrics']['maximum_hold_relative_COM_shift_m']>.19


@pytest.mark.parametrize('fault',['nonfinite','clock_endpoint_for_force','incomplete','negative_force','improper_rotation','unknown_field'])
def test_unknown_or_invalid_diagnostics_yield_null_not_negative_label(fault):
    a=trace()
    if fault=='nonfinite':a['object_com_world_m'][0,2]=np.nan
    elif fault=='clock_endpoint_for_force':a['force_time_s']=a['force_time_s']+.002
    elif fault=='incomplete':a['state_time_s']=a['state_time_s'][:-1]
    elif fault=='negative_force':a['hand_object_normal_force_N'][400]=-.01
    elif fault=='improper_rotation':a['wrist_rotation_world'][100,0,0]=2
    else:a['caller_success']=True
    r=m.score_native_task(a);assert r['status'].startswith('NULL_') and r['outcome'] is None and not r['failure_events']


def test_runtime_and_mapping_refusals_do_not_need_fake_samples():
    assert m.score_native_task(None,execution_status='runtime_error')['outcome'] is None
    r=m.score_native_task(None,mapping_status='setup_error',execution_status='not_executed');assert r['status']=='NULL_MAPPING_OR_SETUP_REFUSAL' and r['outcome'] is None
    with pytest.raises(ValueError):m.score_native_task(None,mapping_status='setup_error',execution_status='completed')


def test_rule_identity_binds_source_and_fixed_profile_no_fitted_thresholds():
    c=m.label_rule_contract();r=m.score_native_task(trace());assert r['label_rule_sha256']==m.canonical_sha256(c) and c['predictor_feature'] is False
    with pytest.raises(Exception):m.NativeTaskLabelSpec(minimum_hold_lift_m=.07)


def test_simultaneous_failure_priority_uses_validated_tick_not_subtick_roundoff():
 a=trace();a['object_com_world_m'][1500:,0]=.03;a['hand_table_normal_force_N'][1500]=.002;a['force_time_s']=a['force_time_s']+1e-13
 r=m.score_native_task(a);assert r['outcome']['failure_type']=='approach_collision' and r['outcome']['first_failure_time_s']==3.
 assert r['failure_events'][0]['observed_time_s']>3. and r['failure_events'][0]['physics_tick']==1500


def test_derived_numeric_overflow_returns_null_not_a_slip_or_success_label():
 a=trace();a['object_com_world_m'][:,0]=1e308;a['wrist_position_world_m'][:,0]=-1e308
 r=m.score_native_task(a);assert r['outcome'] is None and r['status'].startswith('NULL_')


def test_one_ms_matched_rule_has_distinct_hash_same_phases_and_thresholds():
 a=trace();spec=m.NativeTaskLabelSpec(physics_interval_s=.001);st=np.arange(5501)*.001;ft=st[:-1];height=np.zeros(len(st));height[1500:2501]=np.linspace(0,.1,1001);height[2501:]=.1
 for k in ('object_com_world_m','wrist_position_world_m'):
  a[k]=np.zeros((len(st),3));a[k][:,2]=height
 a['state_time_s']=st;a['force_time_s']=ft;a['wrist_rotation_world']=np.tile(np.eye(3),(len(st),1,1));a['hand_object_normal_force_N']=np.where(ft>=1.,.01,0.);a['hand_table_normal_force_N']=np.zeros(len(ft));a['object_table_normal_force_N']=np.zeros(len(ft))
 r=m.score_native_task(a,spec=spec);assert r['outcome']['success'] and r['metrics']['state_samples']==5501 and r['metrics']['force_intervals']==5500
 assert r['label_rule_sha256']!=m.score_native_task(trace())['label_rule_sha256']
 a['hand_table_normal_force_N'][4000]=.002;r=m.score_native_task(a,spec=spec)
 assert r['outcome']['failure_type']=='approach_collision' and r['failure_events'][0]['actual_phase']=='hold'


def test_large_tilted_object_height_alone_does_not_certify_clear_acquisition():
 a=trace();a['object_table_normal_force_N'][1250:]=.02
 r=m.score_native_task(a);assert not r['outcome']['success'] and r['outcome']['failure_type']=='failed_acquisition' and r['outcome']['first_failure_time_s']==2.5
 assert not r['metrics']['acquisition_met'] and not r['metrics']['hold_table_support_absent']
 assert r['failure_events'][0]['actual_phase']=='hold' and r['failure_events'][0]['clock']=='force'


def test_later_table_support_is_first_causal_drop_even_when_COM_stays_high():
 a=trace();a['object_table_normal_force_N'][1500]=.001001
 r=m.score_native_task(a);assert r['outcome']['failure_type']=='drop' and r['outcome']['first_failure_time_s']==3.
 a['object_table_normal_force_N'][1500]=.001;assert m.score_native_task(a)['outcome']['success']
 a['object_table_normal_force_N'][1500]=-.01;assert m.score_native_task(a)['outcome'] is None
 a['object_table_normal_force_N'][1500]=np.nan;assert m.score_native_task(a)['outcome'] is None
