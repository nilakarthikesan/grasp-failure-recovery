"""Manufactured arrays only: no scenes, FK, optimization or physics."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / 'research_artifacts/hug_full_field_mapping_repair_2026-10-15_v1/audit_full_field.py'
spec = importlib.util.spec_from_file_location('hug_full_field_audit', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def distances(values):
    return np.repeat(np.asarray(values, float)[:, None], 6, axis=1)


@pytest.fixture(autouse=True)
def no_real_work(monkeypatch):
    import mujoco
    def fail(*a, **kw): raise AssertionError('No physics or static model construction in manufactured checks')
    for name in ('mj_step','mj_step1','mj_step2','mj_forward'):
        monkeypatch.setattr(mujoco,name,fail)


def test_literal_near_inclusive_far_strict_and_allowance_unchanged():
    # A whole-cell upper bound equal to the band is near. A lower bound equal
    # to the band is ambiguous, not far. Use zero d/r to make equality exact.
    d = distances([.001-audit.ALLOWANCE, .001+audit.ALLOWANCE])
    c = audit.classifications(d, np.zeros(2))
    assert c['near'][0,:,0].all()
    assert not c['far'][1,:,0].any()
    assert c['ambiguous'][1,:,0].all()
    for kwargs in ({'bands':(1,2,6)},{'allowance':1e-7}):
        with pytest.raises(ValueError,match='Frozen'):audit.classifications(d,np.zeros(2),**kwargs)


def test_all_native_definite_rows_are_required_not_only_historical_failures():
    native=distances([0.,.020]);r=np.zeros(2)
    ledger=audit.constraint_rows(native,r)
    assert len(ledger['cell_index'])==2*6*3
    assert ledger['required_near'][:18].all() and not ledger['required_near'][18:].any()
    current=native.copy();current[0,0]=.020;current[1,1]=0.
    arrays,summary=audit.field_status(native,current,r,historical_mapped=native)
    assert summary['historical_definite_mismatch']==0
    assert summary['required']==36
    assert summary['definite_lost']==3 and summary['definite_added']==3
    assert summary['preserved']==30
    assert summary['full_strict_mapping_status']=='FAIL'
    assert not summary['native_definite_relationships_pass']


def test_candidate_regressions_include_ambiguity_and_new_definite_changes():
    native=distances([0.,.020]);r=np.zeros(2);baseline=native.copy();candidate=native.copy()
    candidate[0,0]=.001 # ambiguous at1mm, still preserved at2/5mm
    candidate[1,1]=0. # new added-near atallbands
    arrays,summary=audit.field_status(native,candidate,r,baseline_mapped=baseline)
    assert summary['lost_protected_passing']==4
    assert summary['new_definite_violation']==3
    assert summary['unresolved_required']==1
    assert not arrays['new_definite_violation'][0,0,0]


def test_original_mismatch_inventory_partition_is_complete():
    native=distances([0.,.020]);old=native[::-1].copy();current=native.copy()
    current[0,0]=.001 # repair not certified at1mm
    _,s=audit.field_status(native,current,np.zeros(2),historical_mapped=old)
    assert s['historical_definite_mismatch']==36
    assert s['historical_mismatch_now_preserved']==35
    assert s['historical_mismatch_now_unresolved']==1
    assert s['historical_mismatch_still_definite']==0
    assert s['full_strict_mapping_status']=='INCOMPLETE'
    assert s['full_orientation_status']=='NOT_RUN'
    assert s['no_mapping_pass_issued']


def test_identity_still_does_not_issue_orientation_or_continuous_fidelity_pass():
    native=distances([0.,.020,.001]);r=np.zeros(3)
    _,s=audit.field_status(native,native,r)
    assert s['native_definite_relationships_pass']
    assert s['native_ambiguous']==6
    assert s['full_strict_mapping_status']=='INCOMPLETE'
    assert s['continuous_coverage_status']=='NOT_PROVED'


@pytest.mark.parametrize('distance,radius',[
    (np.zeros((2,5)),np.zeros(2)),(np.zeros((2,6)),np.zeros(1)),
    (np.full((2,6),np.nan),np.zeros(2)),(np.full((2,6),np.inf),np.zeros(2)),
    (-np.ones((2,6)),np.zeros(2)),(np.zeros((2,6)),-np.ones(2))])
def test_unavailable_or_invalid_fields_refuse(distance,radius):
    with pytest.raises(ValueError):audit.classifications(distance,radius)


def test_complete_constraint_order_is_stable_and_indexes_literal_native_field():
    native=distances([0.,.001,.002,.005,.020]);r=np.zeros(5)
    a=audit.constraint_rows(native,r);b=audit.constraint_rows(native,r)
    for name in a:np.testing.assert_array_equal(a[name],b[name])
    rows=list(zip(a['cell_index'],a['part_index'],a['band_index']))
    assert rows==sorted(set(rows))
    c=audit.classifications(native,r)
    assert np.array_equal(a['required_near'],c['near'][a['cell_index'],a['part_index'],a['band_index']])


def test_full_bounds_reproduce_unchanged_inherited_interval_arithmetic():
    inherited=PATH.parents[1]/'hug_orientation_uncertainty_acceptance_2026-10-14/sampling/cell_intervals.py'
    spec=importlib.util.spec_from_file_location('inherited_cell_intervals',inherited)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=distances([.0001,.002,.004,.012]);mapped=native[::-1].copy();r=np.array([0.,.0001,.0002,0.])
    fields,_=audit.field_status(native,mapped,r)
    for j in range(6):
        for k,band in enumerate(audit.BANDS):
            nD,nP=module.masks(native[:,j],r,band/1000.,audit.ALLOWANCE)
            mD,mP=module.masks(mapped[:,j],r,band/1000.,audit.ALLOWANCE)
            np.testing.assert_array_equal(fields['definite_lost'][:,j,k],nD&~mP)
            np.testing.assert_array_equal(fields['definite_added'][:,j,k],mD&~nP)
            np.testing.assert_array_equal(fields['required'][:,j,k],nD|~nP)


def test_write_and_hash_bindings_refuse_overwrite_and_changed_input(tmp_path):
    path=tmp_path/'receipt.json';audit.write(path,dict(a=1))
    bound={str(path):audit.sha(path)};audit.verify(bound)
    with pytest.raises(FileExistsError):audit.write(path,dict(a=2))
    path.write_text('{}')
    with pytest.raises(ValueError,match='Bound bytes changed'):audit.verify(bound)


def case_plan():
    poses=[dict(source_inventory_index=20,pose_id=f'{role}_20') for role in ('original','experimental')]
    prefix='manufactured_density2mm'
    grid=dict(density_mm=2,native_cache=f'/private/tmp/hug-other-surfaces-2026-10-14/visual_reference_surfaces/{prefix}_queries.npz',
        original_cache=f'/private/tmp/hug_sampling_checks_2026-10-14/ephemeral/{prefix}_full_current.npz',
        cache_binding=f'/private/tmp/hug_sampling_checks_2026-10-14/ephemeral/{prefix}_full_current.binding.json',
        query_count=3,original_definite_mismatch_tuples=2)
    p=dict(cases=[dict(source_inventory_index=20,native_sha256='native',poses=poses,grids=[grid])],
           bindings={grid[k]:'unused_mock_hash' for k in ('native_cache','original_cache','cache_binding')})
    old=dict(poses=deepcopy(poses))
    sampling=dict(cases=[dict(source_inventory_index=20,native_sha256='native',case_id='manufactured',densities_mm=[2])])
    counts=dict(cases=[dict(source_inventory_index=20,densities={'2':dict(query_count=3,failure_rows=2)})])
    return p,old,sampling,counts


@pytest.mark.parametrize('change',['pose_order','drop_grid','query_count','domain_path','unbound_input','native_identity'])
def test_plan_refuses_reordering_trimming_or_unbound_domain(change):
    p,old,sampling,counts=case_plan();audit.validate_case_plan(p,old,sampling,counts)
    case=p['cases'][0]
    if change=='pose_order':case['poses']=case['poses'][::-1]
    elif change=='drop_grid':case['grids']=[]
    elif change=='query_count':case['grids'][0]['query_count']=2
    elif change=='domain_path':case['grids'][0]['native_cache']='/other/cache.npz'
    elif change=='unbound_input':p['bindings'].pop(case['grids'][0]['cache_binding'])
    else:case['native_sha256']='other_native'
    with pytest.raises(ValueError):audit.validate_case_plan(p,old,sampling,counts)
