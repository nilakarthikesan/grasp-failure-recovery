"""Manufactured software controls only, no real pilot reads or native calls."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest

s=importlib.util.spec_from_file_location("independent_cohort_math",Path(__file__).with_name("verify_cohort.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def trace():
    dt=.002;t=np.arange(2751)*dt;obj=np.zeros((len(t),3));obj[:,2]=np.clip((t-1.5)/1,0,1)*.15
    st=dict(state_time_s=t,object_com_world_m=obj,wrist_position_world_m=obj.copy(),wrist_rotation_world=np.tile(np.eye(3),(len(t),1,1)))
    ho=np.zeros(2750);ho[500:]=.1
    ft=dict(force_time_s=t[:-1],hand_object_normal_force_N=ho,hand_table_normal_force_N=np.zeros(2750),object_table_normal_force_N=np.zeros(2750))
    return st,ft

def controls():
    t=np.arange(1,551)*.01;s=t-.01;phase=np.where(s<1,"approach",np.where(s<1.5,"close_fingers",np.where(s<2.5,"lift","hold")))
    q=np.zeros((550,67));q[:,3::4]=1
    return dict(time_s=t,action_interval_start_s=s,phase=phase,hand_qpos=q,hand_qvel=np.zeros((550,51)),effective_ctrl=np.zeros((550,51)))

def test_manufactured_success():
    st,ft=trace();o,events,_=m.label_from_saved(st,ft,.002);assert o['success'] and not events

def test_earliest_force_failure_not_later_drop():
    st,ft=trace();ft['hand_object_normal_force_N'][100]=.002;st['object_com_world_m'][-1,2]=0
    o,events,_=m.label_from_saved(st,ft,.002);assert o['failure_type']=='approach_collision' and o['first_failure_time_s']==.2

def test_no_contact_tie_precedes_failed_acquisition():
    st,ft=trace();ft['hand_object_normal_force_N'][:]=0;st['object_com_world_m'][:]=0
    o,events,_=m.label_from_saved(st,ft,.002);assert o['failure_type']=='no_contact' and o['first_failure_time_s']==1.5

def test_null_refusal_never_accesses_missing_observations():
    reason,x=m.feature_prefix(None,None,'refused_guard','not_executed');assert reason=='mapping_or_setup_refusal' and x is None

def test_runtime_not_negative():
    reason,x=m.feature_prefix(None,None,'ok','runtime_error');assert reason=='runtime_or_unexecuted' and x is None

def test_prefix_at149_and_future_failure_only():
    a=controls();o=dict(success=False,failure_type='drop',first_failure_time_s=3.)
    reason,x=m.feature_prefix(a,o,'ok','completed');assert reason is None and abs(x['cutoff_s']-1.49)<1e-12 and x['shape']==[20,169]
    o['first_failure_time_s']=1.49;reason,x=m.feature_prefix(a,o,'ok','completed');assert reason=='already_failed_at_or_before_cutoff' and x is None

def test_post_lift_close_history_refused():
    a=controls();a['phase'][-1]='close_fingers'
    with pytest.raises(ValueError,match='Invalid phase'):m.feature_prefix(a,dict(success=True,failure_type=None,first_failure_time_s=None),'ok','completed')

def test_fold_empty_counts_preserve_nulls():
    rows=[dict(attempt_id=str(i),object_id=str(i),geometry_group_id='g'+str(i),observation_id='o'+str(i),proposal_group_id='p'+str(i)) for i in range(3)]
    values={str(i):dict(status='MAPPING_REFUSAL',success=None,prefix=None,exclusion='mapping_or_setup_refusal') for i in range(3)}
    f=[dict(fold_id='manufactured',assignments={str(i):r for i,r in enumerate(m.ROLES)})]
    result=m.fold_readiness(f,rows,values)[0];assert result['data_status']=='EMPTY_OR_SINGLE_CLASS_ROLE'
    assert all(x['null_outcomes']==1 and x['eligible']==0 for x in result['roles'].values())

def test_fold_group_leak_hard_refuses():
    rows=[dict(attempt_id=str(i),object_id=str(i),geometry_group_id='same',observation_id='o'+str(i),proposal_group_id='p'+str(i)) for i in range(3)]
    f=[dict(fold_id='manufactured',assignments={str(i):r for i,r in enumerate(m.ROLES)})]
    with pytest.raises(ValueError,match='Cross-role'):m.fold_readiness(f,rows,{})
