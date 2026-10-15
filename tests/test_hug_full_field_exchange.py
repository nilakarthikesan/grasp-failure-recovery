"""Manufactured constraint exchange/geometry controls, no native cases or fit."""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import mujoco
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'research_artifacts/hug_full_field_mapping_repair_2026-10-15_v2'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


core=module('manufactured_exchange_core',HERE/'exchange_core.py')
field=module('manufactured_exchange_field',HERE.parent/'hug_full_field_mapping_repair_2026-10-15_v1/audit_full_field.py')


@pytest.fixture(autouse=True)
def no_real_work(monkeypatch):
    def forbidden(*a,**kw):pytest.fail('No real model, physics or optimizer in manufactured controls')
    for name in ('mj_forward','mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,name,forbidden)
    monkeypatch.setattr(core,'minimize',forbidden)


def complete(distance,*,native=None,radius=None):
    d=np.asarray(distance,float)
    if d.ndim==1:d=np.repeat(d[:,None],6,axis=1)
    n=np.zeros_like(d) if native is None else np.asarray(native,float)
    r=np.zeros(len(d)) if radius is None else np.asarray(radius,float)
    fields,_=field.field_status(n,d,r)
    error=r[:,None,None]+1e-8;b=np.asarray([1.,2.,5.])[None,None,:]/1000.
    margin=np.where(fields['native_near'],b-error-d[:,:,None],d[:,:,None]-error-b)
    return dict(fields=fields,margin_m=margin,distance_m=d,
        orientation=dict(valid=np.array([True,False]),passed=np.array([True,False]),
                         score=np.array([-.6,.2]),distance_m=np.array([.001,.002])))


def test_full_field_protection_refuses_unactivated_relationship_loss_and_new_violation():
    baseline=complete([.010,0.]);candidate=complete([0.,.020])
    gate=core.full_protection(baseline,candidate)
    assert not gate['passed'] and gate['lost_baseline_passing']==18 and gate['new_definite_violations']==18


def test_new_definite_violation_from_prior_uncertainty_is_not_permitted():
    baseline=complete([.001]);candidate=complete([.002])
    gate=core.full_protection(baseline,candidate)
    assert gate['new_definite_violations']>0 and not gate['passed']


def test_repaired_relationships_cannot_be_lost_on_later_step():
    baseline=complete([.010]);current=complete([.004]);regressed=complete([.006])
    assert core.full_protection(baseline,regressed)['passed']
    assert not core.full_protection(current,regressed)['passed']


def test_count_to_ambiguity_is_not_relationship_repair_rank_primary():
    baseline=complete([.0011]);ambiguous=complete([.001]);repaired=complete([0.])
    a,b,c=map(core.full_rank,(baseline,ambiguous,repaired))
    assert a[0]==b[0] and b[1]<a[1]
    assert c[0]<b[0]
    assert (ambiguous['fields']['preserved']&~baseline['fields']['preserved']).sum()==0


def test_activation_uses_complete_fields_stable_ties_and_retains_old_rows():
    full=complete([.010,.010,0.]);first=core.activate_worst(full)
    assert len(first)==18 and all(row[0]==0 for row in first)
    second=complete([0.,.020,0.]);combined=core.activate_worst(second,first)
    assert set(first).issubset(combined) and len(combined)==36
    assert tuple(sorted(combined))==combined
    assert combined==core.activate_worst(second,first)


def test_active_newly_preserved_rows_are_hard_without_cap_or_band_allowance():
    baseline=complete([.010]);current=complete([.004]);active=core.activate_worst(baseline)
    old=core.active_policy(baseline,active);new=core.active_policy(current,active)
    assert not old['hard_mask'].any() and new['hard_mask'].sum()==6
    assert not new['slack_coefficients'][new['hard_mask']].any()
    assert np.array_equal(new['offset'],np.zeros(18)) # native-near is inclusive


@pytest.mark.parametrize('change',['valid','passed'])
def test_exact_orientation_eligibility_and_native_away_pass_are_protected(change):
    baseline=complete([.010]);candidate=complete([0.])
    candidate['orientation'][change][0]=False
    assert not core.full_protection(baseline,candidate)['passed']


def state(x):
    return SimpleNamespace(literal_tip_caps_pass=np.ones(5,bool),mean_tip_guard_pass=True,
        physical_joint_limits_pass=True,engineering_domain_pass=True,
        tip_error_norms_m=np.zeros(5),tip_errors_world_m=np.zeros((5,3)),
        joint_qpos_rad=.025*np.asarray(x)[:24],root_position_m=np.zeros(3),root_quaternion_wxyz=np.array([1.,0.,0.,0.]))


class ToyCase:
    def __init__(self,regress=False,final_mutation=False):
        self.bounds=np.tile([-14.,14.],(30,1));self.caps=np.full(5,.01)
        self.regress=regress;self.final_mutation=final_mutation;self.full_calls=[]
    def cheap(self,x):return state(x)
    def tip_jacobian(self,x):return np.zeros((5,3,30))
    def orientation(self,x):return complete([0.])['orientation']
    def _full(self,x):
        value=complete([max(0.,.010-.006*x[0]),.020 if self.regress and x[0]!=0. else 0.])
        value['state']=state(x);return value
    def original_full(self):
        self.full_calls.append('original');return self._full(np.zeros(30))
    def full(self,x,label):
        self.full_calls.append(label);value=self._full(x)
        if label=='fresh_final' and self.final_mutation:
            # Keep scalar rank equal, but violate deterministic serialized-state
            # readback. A rank-only check would overlook this mutation.
            value['state'].root_position_m[0]=1e-9
        return value
    def raw(self,x,active):
        value=self._full(x);cell,part,band=np.asarray(active,int).T
        return dict(surface_margin=value['margin_m'][cell,part,band]/.001,
                    task_error_vectors_m=value['state'].tip_errors_world_m,state=value['state'])


def proposal(*args):
    step=np.zeros(30);step[0]=1.
    return dict(step_scaled=step,usable_finite_trust_step=True,manufactured_no_optimizer=True)


def test_two_rounds_keep_all_new_repairs_hard_and_five_full_scans():
    case=ToyCase();events=[];policies=[]
    def local(*args):policies.append(args[7]);return proposal()
    result=core.solve(case,local,{},lambda kind,**kw:events.append((kind,kw)))
    assert result['status']=='IMPROVED_RELATIONSHIPS_DIAGNOSTIC'
    assert result['accepted_relationship_repair'] and result['repaired_relationship_count']==18
    assert not result['accepted_mapping'] and not result['full_strict_mapping_accepted']
    assert len(result['rounds'])==2 and len(case.full_calls)==5
    assert not policies[0]['hard_mask'].any() and policies[1]['hard_mask'].sum()==6
    assert set(result['rounds'][0]['activated_rows']).issubset(result['rounds'][1]['activated_rows'])
    assert result['final_protection']['passed']


def test_full_field_regression_refuses_candidate_never_replaces_current():
    case=ToyCase(regress=True);result=core.solve(case,proposal,{},lambda *a,**k:None)
    assert result['repaired_relationship_count']==0 and not result['accepted_relationship_repair']
    assert all(not r['selected_feasible_search_step'] for r in result['rounds'])
    assert not result['returned_state_scaled'].any()
    assert len(case.full_calls)<=5


def test_equal_rank_but_changed_final_serialized_root_is_hard_error():
    case=ToyCase(final_mutation=True)
    with pytest.raises(ValueError,match='Final current-scene'):
        core.solve(case,proposal,{},lambda *a,**k:None)


def test_literal_restore_refusal_keeps_explicit_nonacceptance_schema(monkeypatch):
    case=ToyCase()
    monkeypatch.setattr(core,'restore_literal_caps',lambda *a:(None,state(np.zeros(30))))
    result=core.solve(case,proposal,{},lambda *a,**k:None)
    assert result['status']=='REFUSED_NO_FEASIBLE_START'
    assert not result['accepted_relationship_repair'] and not result['accepted_mapping']
    assert result['repaired_relationship_count']==0 and not result['feasible_search_progress_only']
    assert case.full_calls==['original']


@pytest.mark.parametrize('optimizer_success,actual_pass',[(True,False),(False,True)])
def test_restoration_uses_actual_literal_caps_not_optimizer_status(monkeypatch,optimizer_success,actual_pass):
    class RestoreCase(ToyCase):
        def cheap(self,x):
            value=state(x);good=bool(x[0]!=0 and actual_pass)
            value.literal_tip_caps_pass[:]=good
            value.tip_error_norms_m[:]=.009 if good else .010000000001
            value.tip_errors_world_m[:,0]=value.tip_error_norms_m
            return value
    case=RestoreCase();events=[]
    def manufactured_optimizer(fun,x0,**kw):
        assert kw['options']['maxiter']==25 and kw['method']=='SLSQP'
        assert all(lo>=-.01 and hi<=.01 for lo,hi in kw['bounds'])
        assert kw['constraints'][0]['fun'](x0).shape==(6,)
        assert kw['constraints'][0]['jac'](x0).shape==(6,30)
        x=x0.copy();x[0]=.005
        return SimpleNamespace(x=x,status=0 if optimizer_success else 9,success=optimizer_success,message='manufactured',nit=1)
    monkeypatch.setattr(core,'minimize',manufactured_optimizer)
    x,_=core.restore_literal_caps(case,lambda kind,**row:events.append((kind,row)))
    assert (x is not None)==actual_pass
    assert events[-1][1]['literal_caps_pass']==actual_pass
    assert events[-1][1]['optimizer_success']==optimizer_success


def test_ambiguity_only_progress_is_never_called_relationship_repair():
    class AmbiguityCase(ToyCase):
        def _full(self,x):
            value=complete([.0011 if x[0]==0 else .001]);value['state']=state(x);return value
    result=core.solve(AmbiguityCase(),proposal,{},lambda *a,**k:None)
    assert result['status']=='FEASIBLE_SEARCH_PROGRESS_ONLY'
    assert result['feasible_search_progress_only'] and result['repaired_relationship_count']==0
    assert not result['accepted_relationship_repair'] and not result['accepted_mapping']
    assert all(not row['accepted_relationship_repair'] for row in result['rounds'])


def test_restoration_rejects_optimizer_return_outside_declared_narrow_box(monkeypatch):
    case=ToyCase();original=case.cheap
    def failed_initial(x):
        value=original(x)
        if not np.any(x):value.literal_tip_caps_pass[:]=False
        return value
    case.cheap=failed_initial
    def outside(*a,**k):
        x=np.zeros(30);x[0]=.010000000000000002
        return SimpleNamespace(x=x,status=0,success=True,message='manufactured outside narrow box',nit=1)
    monkeypatch.setattr(core,'minimize',outside)
    with pytest.raises(ValueError,match='literal declared search box'):
        core.restore_literal_caps(case,lambda *a,**k:None)


def test_patch_has_exact_three_declared_substitutions_and_only_install_append():
    patch=json.loads((HERE/'geometry_patch.json').read_text());source=Path(patch['source_path']).read_text()
    for change in patch['substitutions']:
        assert source.count(change['before'])==change['count']==1
        source=source.replace(change['before'],change['after'])
    assert (HERE/'geometry_kernel.py').read_text()==source+patch['append']


def test_diagnosed_kernel_matches_original_manufactured_mesh_distance_and_inside():
    original=module('manufactured_geometry_original',HERE.parent/'hug_full_field_mapping_repair_2026-10-15_v1/source_snapshots/audit_other_surfaces.py')
    clean=module('manufactured_geometry_clean',HERE/'geometry_kernel.py')
    vertices=np.array([[x,y,z] for x in (-1.,1.) for y in (-.5,.5) for z in (-.25,.25)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        tri,eq,_=original.hull(vertices);ctri,ceq,_=clean.hull(vertices)
    np.testing.assert_array_equal(ctri,tri);np.testing.assert_array_equal(ceq,eq)
    points=np.array([[0.,0.,0.],[1.,.1,0.],[2.,0.,0.],[-2.,.7,.4],[.1,.2,.3]])
    geom=dict(kind=int(mujoco.mjtGeom.mjGEOM_MESH),size=np.ones(3),pos=np.zeros(3),R=np.eye(3),tri=tri,equations=eq)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        old=original.geom_query(geom,points);new=clean.geom_query(geom,points)
    for a,b in zip(old,new):np.testing.assert_array_equal(a,b)
    assert old[0][0]<0 and old[0][2]>0
    clean.install(original)
    assert original.geom_query is clean.geom_query and original.hull is clean.hull


def test_bridge_complete_domain_rows_not_capped_and_literal_final_replay(tmp_path):
    bridge=module('manufactured_repair_bridge',HERE/'repair_case.py')
    case=bridge.RepairCase.__new__(bridge.RepairCase)
    case.v1=field;case.counts=dict(full_field_checkpoints=0);case.output=tmp_path
    case.query=np.zeros((3,3));case.radius=np.zeros(3);case.area=np.ones(3)
    case.native_distance=np.zeros((3,6));case.grid=dict(required_relationships=54)
    case._hand=lambda:[]
    case._part_distances=lambda hand,points,part:np.full(len(points),.010)
    case._orientation_current=lambda hand:complete([0.])['orientation']
    full=case._full_current(state(np.zeros(30)),'manufactured_full')
    assert full['summary']['required']==54 and full['summary']['definite_lost']==54
    with np.load(full['array_path'],allow_pickle=False) as saved:
        assert saved['distance_m'].shape==(3,6)
        assert saved['required'].sum()==54
    case.counts['full_field_checkpoints']=5
    with pytest.raises(RuntimeError,match='Five full-field'):case._full_current(state(np.zeros(30)),'excess')


def test_bridge_original_reference_requires_exact_all_masks_and_binds_numeric_comparison(tmp_path):
    bridge=module('manufactured_reference_bridge',HERE/'repair_case.py')
    case=bridge.RepairCase.__new__(bridge.RepairCase);case.index=20;case.v1=field
    case.query=np.zeros((1,3));case.radius=np.zeros(1);case.native_distance=np.zeros((1,6))
    full=complete([.010]);directory=tmp_path/'run_v1/20';directory.mkdir(parents=True)
    path=directory/'original_20_density2mm.npz'
    def save(fields,distance):
        with path.open('wb') as f:np.savez(f,query_object_m=case.query,cell_radius_m=case.radius,
            native_distance_by_part_m=case.native_distance,current_distance_by_part_m=distance,**fields)
        case.p=dict(v1_protocol=str(tmp_path/'protocol.json'),bindings={str(path.resolve()):field.sha(path)})
    save(full['fields'],full['distance_m']+5e-14)
    receipt=case._compare_original_full_reference(full)
    assert receipt['all_classification_arrays_exact'] and receipt['max_distance_discrepancy_m']>0
    assert not receipt['scientific_predicate_or_tip_cap_allowance']
    changed={k:v.copy() for k,v in full['fields'].items()};changed['required'][0,0,0]=False
    save(changed,full['distance_m'])
    with pytest.raises(ValueError,match='classification changed required'):case._compare_original_full_reference(full)
    save(full['fields'],full['distance_m']+2e-12)
    with pytest.raises(ValueError,match='numerical distance comparison'):case._compare_original_full_reference(full)
    case.p['bindings']={}
    with pytest.raises(ValueError,match='explicitly source-bound'):case._compare_original_full_reference(full)


def test_bridge_orientation_count_covers_checkpoint_queries_and_finite_invalid_rows():
    bridge=module('manufactured_orientation_bridge',HERE/'repair_case.py')
    case=bridge.RepairCase.__new__(bridge.RepairCase);case.counts={'orientation_checks':0}
    case.oq=np.zeros((2,3));case.on=np.zeros((2,3));case.op=np.zeros(2,int);case.visual_tri=np.empty((0,3,3))
    case.osign=np.array([-1.,1.]);case.oband=np.array([.002,.002]);case.ori_rows=[{},{}]
    case.orientation_protocol=dict(facing_dot_numerical_deadzone=1e-8,query_boundary_tolerance_m=1e-8)
    value=dict(valid=np.array([True,False]),score=np.array([-.6,.2]),distance_m=np.array([.001,.002]),reason_masks={})
    case.direction=SimpleNamespace(mapped_direction_fields=lambda *a,**k:value)
    case.core=SimpleNamespace(orientation_constraints=lambda *a:dict(passed=np.array([True,False])))
    out=case._orientation_current([])
    assert case.counts['orientation_checks']==1 and not out['valid'][1]
    value['score'][1]=np.nan
    with pytest.raises(ValueError,match='Finite exact orientation'):case._orientation_current([])
    assert case.counts['orientation_checks']==2
