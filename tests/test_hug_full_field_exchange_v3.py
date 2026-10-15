"""Manufactured v3 joint-restoration/counterexample controls, no real solves."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'research_artifacts/hug_full_field_mapping_repair_2026-10-15_v3'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


core=module('manufactured_exchange_v3',HERE/'exchange_core.py')
helpers=module('manufactured_v2_helpers',ROOT/'tests/test_hug_full_field_exchange.py')
complete=helpers.complete
state=helpers.state
proposal=helpers.proposal


@pytest.fixture(autouse=True)
def no_real_work(monkeypatch):
    def forbidden(*a,**k):pytest.fail('No real geometry, physics or optimizer in v3 controls')
    for name in ('mj_forward','mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,name,forbidden)
    monkeypatch.setattr(core,'minimize',forbidden)


class ToyCase(helpers.ToyCase):
    def relationship_error_m(self,active):return np.full(len(active),1e-8)


def test_restoration_activation_all_preserved_rows_cutoff_inclusive_no_truncation():
    baseline=complete(np.zeros(50));baseline['margin_m'][:]=core.RESTORE_ACTIVATION_MARGIN_M
    baseline['margin_m'][0,0,0]=np.nextafter(core.RESTORE_ACTIVATION_MARGIN_M,np.inf)
    expected=int(baseline['fields']['preserved'].sum())-1
    selected=core.restoration_rows(baseline)
    assert len(selected)==expected==899 and selected==tuple(sorted(selected))
    baseline['fields']['preserved'][1,0,0]=False
    assert (1,0,0) not in core.restoration_rows(baseline)
    baseline['margin_m'][1,0,0]=np.nan
    with pytest.raises(ValueError,match='Complete finite'):core.restoration_rows(baseline)


def test_every_lost_and_new_definite_counterexample_is_retained_without_sampling():
    baseline=complete(np.zeros(50));candidate=complete(np.full(50,.020))
    rows=core.counterexample_rows(baseline,baseline,candidate)
    assert len(rows)==900 and rows==tuple(sorted(rows))
    baseline=complete([.001]);candidate=complete([.002])
    assert core.counterexample_rows(baseline,baseline,candidate)


def test_previously_unresolved_near_guard_hardens_only_no_new_definite_boundary():
    current=complete([.001]);active=((0,0,0),);e=1e-8
    policy=core.active_policy(current,active,active,np.array([e]))
    assert policy['hard_mask'].tolist()==[True]
    assert policy['no_new_definite_guard_mask'].tolist()==[True]
    assert not policy['full_preservation_hard_mask'].any()
    assert core.active_hard_pass(dict(surface_margin=np.array([-2*e/.001])),policy)
    assert not core.active_hard_pass(dict(surface_margin=np.array([-2*e/.001-1e-12])),policy)
    assert policy['slack_coefficients'].tolist()==[0.]


def test_unresolved_far_guard_has_strict_inward_search_buffer():
    current=complete([.001],native=np.full((1,6),.020));active=((0,0,0),);e=1e-8
    policy=core.active_policy(current,active,active,np.array([e]))
    assert policy['no_new_definite_guard_mask'].all()
    assert not core.active_hard_pass(dict(surface_margin=np.array([-2*e/.001])),policy)
    assert core.active_hard_pass(dict(surface_margin=np.array([(-2*e+1.000001e-9)/.001])),policy)


def test_preserved_guard_is_never_weakened_and_existing_defect_remains_soft():
    active=((0,0,0),)
    policy=core.active_policy(complete([0.]),active,active,np.array([1e-8]))
    assert policy['full_preservation_hard_mask'].all() and policy['offset'][0]==0
    assert not core.active_hard_pass(dict(surface_margin=np.array([-1e-9])),policy)
    policy=core.active_policy(complete([.020]),active,active,np.array([1e-8]))
    assert not policy['hard_mask'].any() and policy['slack_coefficients'][0]==1


def test_unresolved_guard_requires_original_finite_cell_error():
    with pytest.raises(ValueError,match='Fixed radius'):
        core.active_policy(complete([.001]),((0,0,0),),((0,0,0),),np.array([0.]))


class RestoreCase(ToyCase):
    def __init__(self):super().__init__();self.raw_rows=[];self.raw_calls=0
    def cheap(self,x):
        value=state(x);good=bool(x[0]!=0)
        value.literal_tip_caps_pass[:]=good
        value.tip_error_norms_m[:]=.009 if good else .010000000001
        value.tip_errors_world_m[:,0]=value.tip_error_norms_m
        return value
    def raw(self,x,active):
        self.raw_rows.append(tuple(active));self.raw_calls+=1
        return dict(state=self.cheap(x),task_error_vectors_m=self.cheap(x).tip_errors_world_m,
                    surface_margin=np.full(len(active),1e-5)-x[0])


def near_bound_baseline():
    value=complete([0.]);value['state']=state(np.zeros(30));value['margin_m'][:]=1.
    value['margin_m'][0,0,0]=5e-8
    return value


def test_joint_restoration_includes_actual_protected_surface_and_fixed_fd(monkeypatch):
    case=RestoreCase();baseline=near_bound_baseline();events=[]
    def manufactured_optimizer(fun,x0,**kw):
        assert kw['options']['maxiter']==8
        c=kw['constraints'][0]
        assert c['fun'](x0).shape==(7,)
        jac=c['jac'](x0)
        assert jac.shape==(7,30) and jac[-1,0]==pytest.approx(-1.)
        assert case.raw_rows and all(x==((0,0,0),) for x in case.raw_rows)
        assert len(case.raw_rows)==31 # one shared baseline, thirty FD probes
        x=x0.copy();x[0]=.005
        return SimpleNamespace(x=x,status=0,success=True,message='manufactured',nit=1)
    monkeypatch.setattr(core,'minimize',manufactured_optimizer)
    x,_=core.restore_literal_caps(case,baseline,lambda kind,**row:events.append((kind,row)))
    assert x is None
    assert events[0][1]['selected_count']==1 and not events[0][1]['acceptance_allowance']
    # Optimizer success and tip feasibility cannot admit a selected surface
    # violation; unselected relationships are still checked in the full scan.
    assert events[-1][1]['restoration_callbacks']==34
    assert events[-1][1]['literal_caps_pass']
    assert not events[-1][1]['all_selected_original_relationships_pass']


def test_restoration_callback_budget_refuses_without_retry_or_extra_final_callback(monkeypatch):
    case=RestoreCase();events=[]
    def consume(fun,x0,**kw):
        c=kw['constraints'][0]
        for i in range(500):
            x=x0.copy();x[1]=i*1e-9;c['fun'](x)
        pytest.fail('The 400 callback guard must stop this manufactured optimizer')
    monkeypatch.setattr(core,'minimize',consume)
    x,_=core.restore_literal_caps(case,near_bound_baseline(),lambda kind,**row:events.append((kind,row)))
    assert x is None and events[-1][0]=='restoration_budget_refusal'
    assert events[-1][1]['restoration_callbacks']==400 and case.raw_calls==399


@pytest.mark.parametrize('far,margin,passed',[(False,0.,True),(True,0.,False),(True,.5e-9,True)])
def test_restoration_actual_selected_predicates_keep_inclusive_near_strict_far(monkeypatch,far,margin,passed):
    class BoundaryCase(RestoreCase):
        def raw(self,x,active):
            return dict(state=self.cheap(x),surface_margin=np.full(len(active),margin/.001))
    case=BoundaryCase();baseline=near_bound_baseline()
    if far:
        baseline=complete([.020],native=np.full((1,6),.020));baseline['margin_m'][:]=1.
        baseline['margin_m'][0,0,0]=5e-8
    def manufactured(*a,**k):
        x=np.zeros(30);x[0]=.005
        return SimpleNamespace(x=x,status=9,success=False,message='manufactured',nit=8)
    monkeypatch.setattr(core,'minimize',manufactured)
    events=[];x,_=core.restore_literal_caps(case,baseline,lambda kind,**row:events.append((kind,row)))
    assert (x is not None)==passed
    assert events[-1][1]['all_selected_original_relationships_pass']==passed
    if far and passed:assert not events[-1][1]['selected_inward_search_target_pass'].all()


def test_guard_counterexample_enters_next_solve_hard_and_persists():
    class CounterCase(ToyCase):
        def _full(self,x):
            # Cell1 is initially unresolved at1mm. Any real move makes it
            # opposite-definite; the first checkpoint must add its guard.
            value=complete([max(0.,.010-.006*x[0]),.001 if x[0]==0 else .002])
            value['state']=state(x);return value
    policies=[]
    def local(*args):policies.append((args[7],tuple(args[7]['retained_counterexample_mask'])));return proposal()
    result=core.solve(CounterCase(),local,{},lambda *a,**k:None)
    assert len(result['rounds'])==3
    assert result['retained_guard_counterexamples']
    assert not policies[0][0]['retained_counterexample_mask'].any()
    assert policies[1][0]['no_new_definite_guard_mask'].any()
    assert policies[2][0]['no_new_definite_guard_mask'].any()
    assert set(result['rounds'][0]['checkpoint_guard_counterexamples']).issubset(result['rounds'][1]['activated_rows'])
    assert not result['accepted_mapping'] and result['repaired_relationship_count']==0


def test_three_rounds_six_scan_budget_and_explicit_same_part_receipt():
    case=ToyCase();result=core.solve(case,proposal,{},lambda *a,**k:None)
    assert len(result['rounds'])==3 and result['accepted_relationship_repair']
    assert len(case.full_calls)==6
    receipt=result['same_part_admission_receipt']
    assert receipt['relationship_repair_credit'] and receipt['newly_preserved_relationships']==18
    assert receipt['all_baseline_preserved'] and receipt['new_definite_violations']==0
    assert receipt['policy']['minimum_newly_preserved_relationships']==1 and not receipt['accepted_mapping']


def test_final_readback_still_refuses_equal_rank_root_mutation():
    with pytest.raises(ValueError,match='Final current-scene'):
        core.solve(ToyCase(final_mutation=True),proposal,{},lambda *a,**k:None)


def test_early_refusal_keeps_same_part_and_guard_receipts(monkeypatch):
    case=ToyCase();monkeypatch.setattr(core,'restore_literal_caps',lambda *a:(None,state(np.zeros(30))))
    result=core.solve(case,proposal,{},lambda *a,**k:None)
    assert result['status']=='REFUSED_NO_FEASIBLE_START'
    assert result['retained_guard_counterexamples']==[]
    assert not result['same_part_admission_receipt']['relationship_repair_credit']


def test_bridge_records_every_checkpoint_at_write_time_and_refuses_seventh(tmp_path):
    bridge=module('manufactured_bridge_v3',HERE/'repair_case.py')
    case=bridge.RepairCase.__new__(bridge.RepairCase);case.v1=helpers.field
    case.counts=dict(full_field_checkpoints=0);case.checkpoint_artifacts=[];case.output=tmp_path
    case.query=np.zeros((1,3));case.radius=np.zeros(1);case.area=np.ones(1)
    case.native_distance=np.zeros((1,6));case.grid=dict(required_relationships=18)
    case._hand=lambda:[];case._part_distances=lambda hand,points,part:np.full(len(points),.010)
    case._orientation_current=lambda hand:complete([0.])['orientation']
    for i in range(6):case._full_current(state(np.zeros(30)),str(i))
    assert len(case.checkpoint_artifacts)==6
    for receipt in case.checkpoint_artifacts:
        assert receipt['sha256']==helpers.field.sha(receipt['path'])
        assert receipt['bytes']==Path(receipt['path']).stat().st_size
    with pytest.raises(RuntimeError,match='Six full-field'):case._full_current(state(np.zeros(30)),'seventh')
    assert len(case.checkpoint_artifacts)==6


def test_bridge_frozen_error_and_kernel_bytes_reuse_without_geometry_change():
    bridge=module('manufactured_error_bridge_v3',HERE/'repair_case.py')
    case=bridge.RepairCase.__new__(bridge.RepairCase);case.radius=np.array([.001,.002]);case.v1=helpers.field
    np.testing.assert_array_equal(case.relationship_error_m(((1,0,0),(0,5,2))),case.radius[[1,0]]+1e-8)
    for name in ('geometry_kernel.py','geometry_patch.json'):
        assert (HERE/name).read_bytes()==(HERE.parent/'hug_full_field_mapping_repair_2026-10-15_v2'/name).read_bytes()


def test_joint_restoration_capacity_refuses_complete_inventory_before_optimizer():
    case=RestoreCase();baseline=complete(np.zeros(50));baseline['margin_m'][:]=1e-8
    events=[];x,_=core.restore_literal_caps(case,baseline,lambda kind,**row:events.append((kind,row)))
    assert x is None and case.raw_calls==0
    receipt=events[-1][1]
    assert events[-1][0]=='restoration_capacity_refusal' and receipt['local_constraint_rows']==906
    assert len(receipt['complete_selected_rows'])==900 and not receipt['truncated']


def test_exchange_capacity_retains_complete_inventory_and_never_calls_local_optimizer(monkeypatch):
    all_rows=tuple((c,p,b) for c in range(30) for p in range(6) for b in range(3))
    monkeypatch.setattr(core,'activate_worst',lambda *a:all_rows)
    def forbidden(*a):pytest.fail('Over-capacity local optimizer must never run')
    result=core.solve(ToyCase(),forbidden,{},lambda *a,**k:None)
    row=result['rounds'][0]
    assert row['capacity_refusal'] and not row['truncated'] and row['local_constraint_rows']==546
    assert result['activated_rows']==all_rows and not result['accepted_mapping']
